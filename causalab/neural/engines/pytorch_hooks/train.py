"""The reference engine's train runner (spec §2.11, §8): the forwards of a
fit, under the shared loop.

What a ``train`` section means — seed, objective, schedules, eval, early
stop, checkpoints — is :mod:`causalab.neural.shared.training`'s, and so is
the update loop (``fit_loop``), which advances each fit's ``FitState``
against its ``FitSpec`` and never sees an executor. This module is how
*this* engine runs a step's forwards and an eval pass — the loop's ``step``
and ``evaluate`` callbacks — and what it keeps per fit beside the loop's
state to do so (:class:`_Fit`).

**Cohorts** (§4). The loop fits several points **together**: the points of
one cohort (:func:`~causalab.protocol.plan.fit_cohorts` — one realization,
one row set, one frame) step in lockstep, and each step's forward is one
model call over the concatenation of every member's minibatch, each member's
writes on its own rows (``cohort.py``). Every member keeps its own seed,
stages, optimizer, minibatch order, anneal, eval cadence and early-stop
state; a member whose budget is spent or whose patience ran out drops out of
the batch while the rest go on. The summed loss backpropagates into disjoint
parameters, so each member's gradient is its own. ``fit_rows`` bounds the rows
of one grad forward (a member's minibatch is never split); the no-grad eval
passes batch the same way under the engine's ``batch_rows``.

Beside the cohort forward: the row budget and its out-of-memory retry
(``budget.py``), the captured paths (``cuda_graphs.py``, ``graph_cohort.py``,
``graph_reuse.py``), the fit-constant groups served from the campaign
``ForwardCache`` and their per-member tallies, and the eval executor with its
graph pool.
"""

from __future__ import annotations

import contextlib
import dataclasses
import copy
import logging
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from causalab.neural.engines.pytorch_hooks.budget import Meter, RowBudget, cuda_meter
from causalab.neural.engines.pytorch_hooks.cohort import (
    cohort_entries,
    groups_read_by,
    run_groups,
)
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.execution import TrainOutcome
from causalab.neural.shared.executor_base import ForwardCache
from causalab.neural.shared.featurizers import Stage, featurizer_cache
from causalab.neural.shared.mechanisms import operand_names
from causalab.neural.shared.metrics import column_token_ids
from causalab.neural.shared.services import input_roles

from causalab.neural.shared.training.draw import Drawn
from causalab.neural.shared.training.executors import (
    device_scored,
    eval_reads as _eval_reads,
    fit_spec,
    fresh_for_eval as _fresh_for_eval,
    minibatch_executors,
    score_executor as _score,
    seeded_stages,
)
from causalab.neural.shared.training.loop import fit_loop
from causalab.neural.shared.training.objective import metric_tensor, step_loss
from causalab.neural.shared.training.objective import regularizer as _regularizer
from causalab.neural.shared.training.spec import FitSpec
from causalab.neural.shared.training.state import FitState, build_fit_state
from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import (
    Document,
    concrete_int,
    concrete_str,
)

from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    GraphPool,
    TrainingGraphs,
    copy_executor_stages,
    make_executor,
)
from causalab.neural.engines.pytorch_hooks.graph_cohort import (
    CohortGraphs,
    EvaluationGraphs,
    Member,
    WindowItem,
    cohort_graph_reason,
)
from causalab.neural.engines.pytorch_hooks.graph_reuse import FitGraphCache

__all__ = ["run_cohort_training", "run_training"]


def _inner_interning(executor: PointExecutor, rows: tuple[int, ...] | str) -> Any:
    """The store handle a fit's inner executor gets. A solo graph fit runs on
    the graph executor's own frozen cache instead of the store (its constants
    must sit in storage the captures own); a captured cohort keeps the store
    (``GraphExecutor.keep_store``) and copies what its graph needs out of it."""
    if executor.cuda_graphs and not getattr(executor, "keep_store", False):
        return None
    return executor.inner_interning(rows)


def _slot_rows(doc: Document, executor: PointExecutor) -> int:
    """The largest minibatch this dataset can produce, before inner executors
    are built and their graph eligibility is chosen."""
    assert doc.train is not None
    return min(
        concrete_int(doc.train.batch["pairs"], "train.batch.pairs"),
        len(executor.rows_for_metrics()),
    )


def _inner_executor(
    doc: Document,
    executor: PointExecutor,
    *,
    role_rows: Mapping[str, list[dict[str, Any]]],
    grad_enabled: bool,
    rows: tuple[int, ...] | str,
    batches: Mapping[str, EncodedBatch] | None = None,
    drawn: bool = False,
    stage_cache: dict[str, Stage] | None = None,
) -> PointExecutor:
    """This engine's ``ExecutorFactory`` (``shared/training/executors.py``): a fit's
    minibatch or eval executor over the point's ``executor``.

    A minibatch's rows are a *slice* of the campaign's, so its captures live
    under ``(digest, indices)`` in the shared ForwardCache — never under the
    whole role's key — and only the groups this fit cannot change are ever
    read from or written to it (``inner_interning``, §4 "Fits"). For the
    shipped methods that is the source forward: run once per slice, then
    served on every step, epoch and point that shares the digest. The eval
    split is not the campaign's rows, so its captures get their own key,
    ``(digest, split)``. A ``drawn`` minibatch consults no store at all and
    runs eager: it is the per-epoch rebuild of the minibatch executors, not
    the draw, that a captured graph cannot follow.

    No ``batch_rows`` on a minibatch: ``train.batch.pairs`` is the document's
    own batching knob for grad forwards, so the execution bound applies to
    the no-grad passes — the eval executor carries the point's, an eval pass
    being a no-grad forward over the whole split, exactly what microbatching
    is for. ``stage_cache`` replaces the point's shared cache (one stage per
    name) for an eval executor whose captures outlive the fit."""
    return make_executor(
        doc,
        executor.bundle,
        cuda_graphs=(
            False if drawn else executor.cuda_graphs and executor.fit_cuda_graphs
        ),
        role_rows=role_rows,
        role_fields=executor.role_fields,
        load_tensors=executor.load_tensors,
        load_table=executor.load_table,
        stage_cache=executor.stage_cache if stage_cache is None else stage_cache,
        grad_enabled=grad_enabled,
        coords=executor.coords,
        batch_rows=None if grad_enabled else executor.batch_rows,
        interning=None if drawn else _inner_interning(executor, rows),
        batches=batches,
    )


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(eq=False)
class _Fit:
    """One point's fit on this engine: the loop's ``spec`` and ``state``, and
    beside them everything that runs a forward — the point's executor, one
    executor per minibatch, the drawn roles, what the captured paths keep per
    fit, and the store's tallies. The loop sees none of it; this module's
    callbacks reach it by the member's index."""

    doc: Document
    executor: PointExecutor
    spec: FitSpec
    state: FitState
    minibatch_executors: list[PointExecutor]
    #: §2.2 ``draw``: the fit's drawn roles (``None`` when no role draws) —
    #: each new epoch rebuilds the minibatch executors over a fresh draw
    drawn: Drawn | None = None
    #: the pool the fit's held-out inference replays capture into — the
    #: cohort's, or the solo bank's — handed to the eval executor when it is
    #: built; None when the fit runs eagerly
    graph_pool: GraphPool | None = None
    #: the captured objective of each minibatch executor
    graph_objectives: "dict[PointExecutor, TrainingObjective]" = dataclasses.field(
        default_factory=dict
    )
    #: what this fit's inner passes paid for constant groups and were served
    run: int = 0
    served: int = 0
    #: the block each forward this fit took part in resumed at
    resumed: list[int] = dataclasses.field(default_factory=list)

    @property
    def stages(self) -> dict[str, Stage]:
        return self.state.stages

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.state.optimizer

    @property
    def store(self) -> ForwardCache | None:
        return self.executor.interning.cache if self.executor.interning else None

    @property
    def indices(self) -> tuple[int, ...]:
        """The rows of the minibatch the fit stands at."""
        return self.spec.batches[self.state.order[self.state.position]]

    @property
    def minibatch(self) -> PointExecutor:
        """The executor of the minibatch the fit stands at."""
        return self.minibatch_executors[self.state.order[self.state.position]]

    def redraw(self) -> None:
        """§2.2 ``draw``: a new member per row for a new epoch — the
        minibatch executors rebuilt over it, and the captured objectives
        (keyed by executor) dropped with the ones they replaced."""
        assert self.drawn is not None
        self.minibatch_executors = self.drawn.minibatches()  # type: ignore[assignment]  # this engine's factory built them
        self.graph_objectives.clear()


def _prepare_fit(doc: Document, executor: PointExecutor) -> _Fit:
    """One point's fit, ready to step, after the two routings a captured fit
    depends on are checked: ``cuda_graphs.unsupported_reason`` sends a drawn
    document and a constrained one down the eager path, as it does
    ``control`` and ``phases``. The cohort and single-fit graphs map
    optimizer parameters onto worker stages by identity, and a dual is no
    stage's parameter; a drawn point's minibatch executors are rebuilt every
    epoch. The first is asserted as the invariant it is, the second refused
    as a backstop.

    Then, per member and in one breath: the fit's plain data, its stages —
    seeded and built on the point's own cache — its state, and its minibatch
    executors (this engine's :func:`_inner_executor`)."""
    train = doc.train
    assert train is not None
    if executor.cuda_graphs and executor.fit_cuda_graphs:
        assert not any(
            spec.draw is not None and role in executor.role_rows
            for role, spec in input_roles(doc).items()
        ), "unsupported_reason routes a drawn point to the eager executor"
        constrained = [term for term in train.objective if term.constraint is not None]
        if constrained:
            raise ProtocolError(
                "P4",
                f"train.objective.{constrained[0].name}.constraint: the Lagrangian "
                "duals step on the eager loop — this run captures fit graphs "
                "(fit_cuda_graphs); run it eager",
            )
    spec = fit_spec(doc, executor)
    state = build_fit_state(
        spec, stages=seeded_stages(spec, executor), device=executor.bundle.device
    )
    minibatches, drawn = minibatch_executors(
        doc,
        executor,
        spec,
        _inner_executor,  # type: ignore[arg-type]  # PointExecutor in, as built
    )
    return _Fit(
        doc=doc,
        executor=executor,
        spec=spec,
        state=state,
        minibatch_executors=minibatches,  # type: ignore[arg-type]  # as built
        drawn=drawn,
    )


def _loss(fit: _Fit, minibatch: PointExecutor) -> torch.Tensor:
    """One member's objective over ``minibatch``'s reads (``step_loss``). In
    a step that is the minibatch the fit stands at; any other of the fit's
    minibatch executors is scored over its own rows of the partition."""
    state = fit.state
    rows = None  # the state's own
    if state.position >= len(state.order) or minibatch is not fit.minibatch:
        rows = next(
            fit.spec.batches[i]
            for i, candidate in enumerate(fit.minibatch_executors)
            if candidate is minibatch
        )
    return step_loss(state, fit.spec, minibatch.dense_value, rows=rows)


class TrainingObjective:
    """The normal objective, with token-column resolution outside replay."""

    def __init__(
        self,
        executor: PointExecutor,
        stages: dict[str, Stage],
        *,
        weight: torch.Tensor | None = None,
    ) -> None:
        self.executor = executor
        self.stages = stages
        #: per-row loss weights in place of the plain mean over the executor's
        #: rows — a captured cohort slot's padding rows carry zero
        #: (``graph_cohort.py``); ``None`` is the mean
        self.weight = weight
        self.labels: dict[str, dict[str, torch.Tensor]] = {}
        train = executor.doc.train
        assert train is not None
        for term in train.objective:
            target = term.metric
            if not isinstance(target, str):
                continue
            metric = executor.doc.metrics[target]
            fields = {"cross_entropy": ("target",), "logit_diff": ("a", "b")}.get(
                str(metric.kind), ()
            )
            self.labels[target] = {
                field: torch.tensor(
                    column_token_ids(
                        executor.bundle.tokenizer,
                        [
                            row[
                                concrete_str(
                                    metric.fields[field], f"metric field {field}"
                                )
                            ]
                            if metric.token_form == "id"
                            else str(
                                row[
                                    concrete_str(
                                        metric.fields[field], f"metric field {field}"
                                    )
                                ]
                            )
                            for row in executor.rows_for_metrics()
                        ],
                        token_form=str(metric.token_form),
                        where=f"metric {metric.kind}.{field}",
                    ),
                    dtype=torch.long,
                    device=executor.bundle.device,
                )
                for field in fields
            }

    def for_executor(self, executor: PointExecutor) -> TrainingObjective:
        return TrainingObjective(
            executor, {name: executor.stage(name) for name in self.stages}
        )

    def copy_labels(self, other: TrainingObjective) -> None:
        for name, fields in self.labels.items():
            for field, value in fields.items():
                value.copy_(other.labels[name][field])

    def __call__(self) -> torch.Tensor:
        mb = self.executor
        train = mb.doc.train
        assert train is not None
        loss = torch.zeros((), device=mb.bundle.device)
        for term_spec in train.objective:
            if term_spec.constraint is not None:
                # refused before capture in `_prepare_fit`; said again here so a
                # graph objective can never silently drop the duals
                raise ProtocolError(
                    "P4",
                    f"{term_spec.path(0)}: a constraint term's duals step on the "
                    "eager loop, not inside a captured graph",
                )
            weight, target = (
                term_spec.weight,
                term_spec.metric
                if term_spec.metric is not None
                else term_spec.regularizer,
            )
            w = float(weight) if isinstance(weight, (int, float)) else 1.0
            if isinstance(target, str):
                metric = mb.doc.metrics[target]
                of_value = mb.dense_value(str(metric.of))
                target_value = (
                    mb.dense_value(str(metric.fields["target"]))
                    if metric.kind == "kl"
                    else None
                )
                per_row = metric_tensor(
                    metric,
                    of_value,
                    mb.rows_for_metrics(),
                    mb.bundle.tokenizer,
                    target_value=target_value,
                    token_ids=self.labels[target],
                )
                if self.weight is None:
                    term = per_row.mean()
                else:
                    term = (per_row * self.weight).sum() / self.weight.sum()
            elif isinstance(target, tuple):
                reg_kind, reg_target = target
                term = _regularizer(
                    reg_kind,
                    reg_target,
                    self.stages,
                    term_spec.reduce or "mean",
                    term_spec.costs,
                )
            else:
                raise ProtocolError("P2", f"unresolvable objective term {target!r}")
            loss = loss + w * term
        return loss


def run_training(
    doc: Document,
    executor: PointExecutor,
    request: ExecutionRequest,
    *,
    graph_cache: FitGraphCache | None = None,
) -> TrainOutcome:
    """Fit one point: :func:`run_cohort_training` over a cohort of one."""
    return run_cohort_training([doc], [executor], request, graph_cache=graph_cache)[0]


def run_cohort_training(
    docs: Sequence[Document],
    executors: Sequence[PointExecutor],
    request: ExecutionRequest,
    *,
    fit_rows: int | None = None,
    meter: Meter | None = None,
    graph_cache: FitGraphCache | None = None,
) -> list[TrainOutcome]:
    """Fit the points of one cohort together (module docstring); one outcome
    per point, in order.

    Each ``executor`` is its point's full-data executor — its stage cache is
    shared with every training minibatch, so the stages it later evaluates
    are the fitted ones. Every point's outcome carries the trained stages by
    featurizer name (for the save manifest) **and** the ``train.eval`` score.

    The eval score is returned rather than dropped because it used to be
    computed and then consumed only inside the ``early_stop`` branch, so a fit
    document's saved metric table was the **train** score under a name a reader
    took for the eval one. Spec §2.12 says every metric a document declares is
    saved; step 4 of the causal protocol asks for train and eval together. Both
    were unsatisfiable. :class:`~causalab.neural.shared.execution.TrainOutcome`
    carries it to the run tree as a sibling record — never as a column in the
    metric table, whose rows are a different split.

    **Which weights the returned stages hold.** With ``train.early_stop`` the
    loop selects a fit by its eval score, so the returned stages are the
    *best-scoring* ones, restored from a snapshot taken at each improvement.
    Without ``early_stop`` there is nothing selecting, and they are the last
    ones. ``TrainOutcome.eval_score.selected`` says which, and its ``metrics``
    always describe the weights actually returned.

    ``fit_rows`` bounds the rows of one grad forward: members are packed into
    forwards under it in cohort order, a member's minibatch never split.
    ``None`` measures the bound on the cohort's first step (``budget.py``: a
    one-member probe under peak-memory tracking, unbounded off CUDA), and any
    window that still runs out of memory is retried at half the rows.
    The batched eval passes pack under the engine's ``batch_rows`` when it is
    authored and otherwise under the fit's own bound, in a budget of their own
    (:func:`_advance_eval_budget`): a measured bound's eval window that runs
    out of memory shrinks and retries without touching the grad bound, and
    the shrink is kept for every later pass; an authored bound is fixed for
    eval windows too and re-raises. The outcome reports the smaller of the two
    budgets' bounds as ``fit_rows`` — the number every window of the fit ran
    under, so pinning it is safe — and both budgets' shrinks as one
    ``fit_rows_shrinks``. A grad shrink lowers the grad bound in place, so a
    re-run pinned at the reported number packs exactly as this run did; only
    an eval shrink (``batch_rows`` unauthored) leaves the report below the
    grad bound, and a re-run pinned there packs its grad windows smaller —
    a different rounding, not a different fit. ``meter`` is the device the
    probes read — the tests' seam for a simulated one; the engine leaves it
    to be found from the model (``cuda_meter``).
    """
    if len(docs) != len(executors):
        raise ValueError(
            f"{len(docs)} documents but {len(executors)} executors — one per point"
        )
    if len(docs) > 1 and graph_cache is not None:
        # A solo bank cannot serve this layout; release its pool before either
        # a captured or eager cohort allocates its working set.
        graph_cache.close()
    # A cohort of graph-eligible members is captured whole, on a fixed slot
    # layout (graph_cohort.py). If eligibility or the row bound refuses that
    # layout, keep the eager cohort's batching and shared constants.
    cohort_reason: str | None = None
    slot_rows = [_slot_rows(doc, ex) for doc, ex in zip(docs, executors)]
    if (
        any(isinstance(executor, GraphExecutor) for executor in executors)
        and len(docs) > 1
    ):
        cohort_reason = cohort_graph_reason(executors, fit_rows, slot_rows)
    if cohort_reason is not None:
        logging.getLogger(__name__).info("cohort runs eagerly: %s", cohort_reason)
    captured_cohort = (
        cohort_reason is None
        and len(docs) > 1
        and all(isinstance(executor, GraphExecutor) for executor in executors)
    )
    for executor in executors:
        executor.fit_cuda_graphs = cohort_reason is None
        if isinstance(executor, GraphExecutor):
            # a captured cohort's minibatch and eval executors keep the store
            # (shared sources, prefix resume); decided before they are built
            executor.keep_store = len(docs) > 1
    fits = [_prepare_fit(doc, executor) for doc, executor in zip(docs, executors)]
    if meter is None and executors:
        meter = cuda_meter(executors[0].bundle.model)
    budget = RowBudget.of(fit_rows, meter)
    batch_rows = executors[0].batch_rows if executors else None
    eval_budget: RowBudget | None = None
    # the floor of a measured bound: the smallest minibatch any member will
    # ever step — known before the loop, so it does not depend on which
    # minibatch the shuffle draws first (an epoch's last one is a remainder)
    unit = min((len(batch) for fit in fits for batch in fit.spec.batches), default=None)
    cohort_graphs: CohortGraphs | None = None
    evaluation_graphs: EvaluationGraphs | None = None
    # one allocator pool for every graph the fit captures (GraphPool): a
    # cohort's step and evaluation graphs (graph_cohort.py, "Memory") and its
    # members' inference replays here; a solo fit's buckets and inference
    # replays on its bank's, below. The loop owns the pools it opens and
    # releases them in the finally, after every graph holder; a cached bank's
    # pool belongs to the FitGraphCache.
    cohort_pool: GraphPool | None = None
    solo_pool: GraphPool | None = None
    if captured_cohort:
        cohort_pool = GraphPool()
        evaluation_graphs = EvaluationGraphs(pool=cohort_pool)
        for fit in fits:
            fit.graph_pool = cohort_pool
        cohort_graphs = CohortGraphs(
            [
                Member(
                    key=id(fit),
                    executor=fit.executor,
                    stages=fit.stages,
                    parameters=[
                        p
                        for group in fit.optimizer.param_groups
                        for p in group["params"]
                    ],
                    objective_reads=fit.spec.objective_reads,
                    pairs=pairs,
                )
                for fit, pairs in zip(fits, slot_rows, strict=True)
            ],
            make_objective=TrainingObjective,
            pool=cohort_pool,
        )
    for fit in fits:
        # the held-out capture cache is a solo-fit affair: the members of a
        # captured cohort evaluate together, each on its own eval executor
        fit.executor.graph_cache = (
            graph_cache
            if isinstance(fit.executor, GraphExecutor) and len(fits) == 1
            else None
        )
    graphs = None
    success = False
    if len(fits) == 1 and isinstance(fits[0].executor, GraphExecutor):
        parameters = [
            p for group in fits[0].optimizer.param_groups for p in group["params"]
        ]
        if graph_cache:
            graphs = graph_cache.begin(fits[0].executor, parameters)
        else:
            solo_pool = GraphPool()
            graphs = TrainingGraphs(parameters, pool=solo_pool)
        fits[0].graph_pool = graphs.pool
    try:

        def step(members: Sequence[int]) -> None:
            current = [(fits[i], fits[i].minibatch) for i in members]
            for _fit, minibatch in current:
                minibatch.reset_reads()
            _run_step_windows(
                current, budget, unit, graphs=graphs, cohort_graphs=cohort_graphs
            )

        def evaluate(members: Sequence[int]) -> list[dict[str, float]]:
            nonlocal eval_budget
            eval_budget = _advance_eval_budget(batch_rows, budget, eval_budget)
            return _evaluate(
                [fits[i] for i in members],
                request,
                eval_budget,
                graphs=evaluation_graphs,
            )

        def on_epoch(members: Sequence[int]) -> None:
            for fit in (fits[i] for i in members):
                if fit.drawn is not None:
                    fit.redraw()

        finished = fit_loop(
            [fit.state for fit in fits],
            [fit.spec for fit in fits],
            step=step,
            evaluate=evaluate,
            on_epoch=on_epoch,
        )
        # the bound reported is the one every window of the fit ran under — the
        # grad budget's, or the eval budget's once an eval window shrank below it
        # — so the number an author pins is one no window of the run refused; an
        # authored `batch_rows` governs the eval windows on its own and says
        # nothing about `fit_rows`, and a grad budget that never resolved (off
        # CUDA) stays `None`, since `null` means every member in one forward and
        # an eval shrink there changed no grad window. The shrinks of both budgets
        # are one count: the bound was too loose by that many, whichever kind of
        # window found out (an eval budget under an authored `batch_rows` is fixed
        # and never shrinks). A grad shrink lowered the grad bound in place, so
        # a re-run pinned at the report packs as this run did; only an eval shrink
        # leaves the report below the grad bound, and a re-run pinned there packs
        # its grad windows smaller
        reported = budget.bound
        if (
            reported is not None
            and batch_rows is None
            and eval_budget is not None
            and eval_budget.bound is not None
        ):
            reported = min(reported, eval_budget.bound)
        shrinks = budget.shrinks + (
            eval_budget.shrinks if eval_budget is not None else 0
        )
        # what only this engine knows about the fit goes onto the loop's
        # outcome here: the store's tallies, a drawn role's members, the bound
        outcomes = [
            dataclasses.replace(
                outcome,
                fit_forwards=(
                    {"run": fit.run, "served": fit.served}
                    if fit.store is not None
                    else None
                ),
                resumed=tuple(fit.resumed),
                draws=fit.drawn.record() if fit.drawn is not None else {},
                fit_rows=reported,
                fit_rows_shrinks=shrinks,
            )
            for fit, outcome in zip(fits, finished)
        ]

        success = True
        return outcomes
    finally:
        if evaluation_graphs is not None:
            evaluation_graphs.close()
        if cohort_graphs is not None:
            cohort_graphs.close()
        if graph_cache is not None:
            graph_cache.finish(success=success)
        elif graphs is not None:
            graphs.close()
        for fit in fits:
            evaluation = fit.executor.eval_executor
            if isinstance(evaluation, GraphExecutor) and (
                graph_cache is None or evaluation is not graph_cache.eval_executor
            ):
                evaluation.close()
        # every graph holder of the fit is closed: release the pool the loop
        # opened. A cached bank's pool stays with the bank (FitGraphCache).
        if cohort_pool is not None:
            cohort_pool.close()
        elif solo_pool is not None:
            solo_pool.close()


def _advance_eval_budget(
    batch_rows: int | None, budget: RowBudget, previous: RowBudget | None
) -> RowBudget:
    """The budget the batched eval passes pack under, built on the first
    pass and advanced on every later one — one object for the whole fit, so
    what an eval window learns by running out of memory is kept for every
    later pass rather than re-learnt each time.

    An authored ``batch_rows`` bounds every no-grad forward of the run, so it
    is the fixed bound here. Otherwise the eval passes pack under the fit's
    own bound, so one number (``fit_rows_resolved``) pins both kinds of
    window: a budget seeded from the grad bound as it stands and re-seeded
    **down** to it on every pass (the grad bound only ever shrinks), never
    up past what an eval window already failed at. It is fixed exactly when
    the grad bound is — an authored ``fit_rows`` is never shrunk for eval
    windows either (§8) — and shrinks on its own otherwise: the grad bound
    never moves on an eval window's account, while ``fit_rows_shrinks`` counts
    the windows of both kinds that packed under the fit's bound and did not
    fit.
    """
    if batch_rows is not None:
        return (
            previous if previous is not None else RowBudget.of(batch_rows, meter=None)
        )
    if previous is None:
        return RowBudget(
            bound=budget.bound, fixed=budget.fixed, meter=None, resolved=True
        )
    if budget.bound is not None and (
        previous.bound is None or budget.bound < previous.bound
    ):
        previous.bound = budget.bound
    return previous


def _run_step_windows(
    current: Sequence[tuple[_Fit, PointExecutor]],
    budget: RowBudget,
    unit: int | None,
    *,
    graphs: TrainingGraphs | None = None,
    cohort_graphs: CohortGraphs | None = None,
) -> None:
    """One optimizer step's grad forwards: the members packed into windows
    under the budget (``budget.py``), each window one forward, one summed
    loss and one backward. A window the device cannot hold is retried at a
    smaller bound: its members have not stepped, so their gradients are
    zeroed and the window re-packed (:func:`_run_windows`).

    A captured cohort (``cohort_graphs``) takes the whole step first — every
    stepping member in its slot, one replay — and the budget is not consulted;
    a step it declines runs as the eager cohort below. Captured work also
    bypasses the eager per-member tally brackets, so receipt forward/reuse
    counts are not comparable to eager counts; the graph benchmark reports
    captures and replays separately.

    Each eager window runs under one :func:`featurizer_cache` scope: a member's
    rotation or mask is evaluated once and serves its read, every write's
    featurize and inverse and — for a subspace, whose penalty reads the
    rotation through ``slot_params`` — the regularizer, with the gradient
    reaching the parameter through that one evaluation. A gate's ``l1``
    penalty reads ``soft_mask()`` itself, one evaluation beside the shared
    table: under ``hard_concrete`` the table is the sampled mask and the
    penalty the deterministic one, and a ``head`` / ``site`` group's penalty
    wants the mask before its expansion over the group's coordinates. The
    scope closes before the optimizer moves the parameter."""
    if cohort_graphs is not None and not cohort_graphs.disabled:
        items: list[WindowItem] = []
        for fit, minibatch in current:
            if minibatch not in fit.graph_objectives:
                fit.graph_objectives[minibatch] = TrainingObjective(
                    minibatch, fit.stages
                )
            items.append(
                WindowItem(
                    key=id(fit),
                    indices=fit.indices,
                    minibatch=minibatch,
                    objective=fit.graph_objectives[minibatch],
                )
            )
        if cohort_graphs.backward(items):
            return

    def body(window: Sequence[tuple[_Fit, PointExecutor]]) -> None:
        if graphs is not None and not graphs.disabled and len(window) == 1:
            # no scope of ours around the graph path: every captured pass
            # opens its own, isolated one (`cuda_graphs.captured_pass`), which
            # is what keeps the map inside the graph rather than baked in
            # stale — so sitting outside the eager scope below is a choice,
            # not a necessity; the eval round's `graphs.forward` runs inside
            # one (`_evaluate`) on the same guarantee
            fit, minibatch = window[0]
            if minibatch not in fit.graph_objectives:
                fit.graph_objectives[minibatch] = TrainingObjective(
                    minibatch, fit.stages
                )
            objective = fit.graph_objectives[minibatch]
            if graphs.backward(minibatch, objective):
                return
        with featurizer_cache():
            _run_batched(
                [
                    (fit, minibatch, fit.spec.objective_reads)
                    for fit, minibatch in window
                ]
            )
            loss = torch.zeros(())
            for fit, minibatch in window:
                with _tally(fit):
                    loss = loss + _loss(fit, minibatch)
            loss.backward()

    def abandon(window: Sequence[tuple[_Fit, PointExecutor]]) -> None:
        for fit, minibatch in window:
            fit.state.optimizer.zero_grad()
            minibatch.reset_reads()

    _run_windows(current, budget, body, abandon, unit)


def _run_windows(
    items: Sequence[tuple[_Fit, PointExecutor]],
    budget: RowBudget,
    body: Callable[[Sequence[tuple[_Fit, PointExecutor]]], None],
    abandon: Callable[[Sequence[tuple[_Fit, PointExecutor]]], None],
    unit: int | None,
) -> None:
    """Pack ``items`` into windows under ``budget`` and run ``body`` on each.
    A window that raises the allocator's out-of-memory error is **retried**
    at a smaller bound: ``abandon`` undoes what the failed attempt left on its
    members, the allocator's cache is released, the bound halved (never below
    one member) and the window re-packed; a single member that does not fit
    is re-raised, since nothing smaller exists.

    The release happens *after* the ``except`` block, deliberately: while the
    handler runs, the exception's traceback still holds the failed body's
    frames — the loss at the root of the window's autograd graph, the capture
    dict — so every activation the forward allocated is still referenced, and
    ``empty_cache`` inside the handler would return nothing. Leaving the
    handler drops the traceback first.

    ``unit`` is the floor of a measured bound (``RowBudget.run``): the
    smallest window any member will bring, decided by the caller over the
    whole fit rather than over this step's items. Only a budget still probing
    reads it; a resolved or fixed one (every eval budget) ignores it.

    A failed attempt's tallies (``fit_forwards``, ``prefix_reuse``) stay: its
    forwards did run. A fit whose windows shrank — of either kind, under the
    fit's bound — says so in its receipt (``fit_rows_shrinks``).
    """
    pending = list(items)
    while pending:
        window, rest = budget.take(pending, _window_rows)
        rows = sum(_window_rows(item) for item in window)
        failed: tuple[int, int] | None = None
        try:
            budget.run(rows, lambda: body(window), unit)
        except torch.OutOfMemoryError:
            largest = max(_window_rows(item) for item in window)
            if not budget.can_shrink(rows, largest):
                raise
            failed = (rows, largest)
        if failed is not None:
            abandon(window)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            budget.shrink(*failed)
            continue
        pending = rest


def _window_rows(item: tuple[_Fit, PointExecutor]) -> int:
    """The rows one member brings to a packed window: its executor's."""
    return len(item[1].rows_for_metrics())


@contextlib.contextmanager
def _tally(fit: _Fit) -> Iterator[None]:
    """Attribute the passes run inside to ``fit``: the constant groups the
    store ran and served, and the forwards that resumed from a prefix, as a
    difference over the shared tallies — per member, since a cohort's passes
    interleave several points'. The one forward several members share (a
    cohort forward) is credited to each of them by :func:`_run_batched`
    instead, outside any member's bracket."""
    store = fit.store
    if store is None:
        yield
        return
    run_before, served_before = len(store.inner_executed), len(store.inner_served)
    resumed_before = len(store.resumed)
    try:
        yield
    finally:
        fit.run += len(store.inner_executed) - run_before
        fit.served += len(store.inner_served) - served_before
        fit.resumed.extend(store.resumed[resumed_before:])


def _run_batched(
    members: Sequence[tuple[_Fit, PointExecutor, Sequence[str]]],
    *,
    frames: Mapping[str, EncodedBatch] | None = None,
) -> None:
    """Run the groups ``reads`` need on every member's executor, batched
    across members where the group admits it (``cohort.batchable``): one
    forward per input role over the members' rows together. Each member's
    operand reads — the source groups the store serves — are run first, on
    the member's own tally; a group left out runs on its executor's own path
    when the read is asked for. ``frames`` is the members' concatenated batch
    per role when the caller holds a prepared one (an eval layout's,
    ``EvaluationGraphs.frames_for``); by default it is built per forward."""
    for fit, executor, reads in members:
        with _tally(fit):
            for model, _role in groups_read_by(executor.doc, reads):
                if model == "original":
                    continue
                im = executor.doc.intervened_models[model]
                for ename in im.writes if isinstance(im.writes, tuple) else ():
                    for operand in operand_names(executor.doc.writes[ename].do.payload):
                        if operand in executor.doc.reads:
                            executor.read_value(operand)
    by_role = cohort_entries([(executor, reads) for _, executor, reads in members])
    owner = {id(executor): fit for fit, executor, _ in members}
    for role, entries in by_role.items():
        store = members[0][0].store
        before = len(store.resumed) if store is not None else 0
        frame = frames.get(role) if frames else None
        if frame is None:
            run_groups(entries)
        else:
            run_groups(entries, frame=frame)
        if store is not None:
            resumed = store.resumed[before:]
            for entry in entries:
                owner[id(entry.executor)].resumed.extend(resumed)


def _evaluate(
    due: Sequence[_Fit],
    request: ExecutionRequest,
    budget: RowBudget,
    *,
    graphs: EvaluationGraphs | None = None,
) -> list[dict[str, float]]:
    """One eval pass for every fit in ``due`` (§2.11): each fit's scores, in
    order — the loop records them. One fit runs its own pass (:func:`_run_eval`);
    several run the trained groups batched across the fits whose split
    agrees, packed under ``budget`` (:func:`_advance_eval_budget`), and are
    scored on the result.

    Under a captured cohort, ``graphs`` serves the pass: the fits due on one
    split that the budget would run as one window are one replay of the
    cohort's eval layout (``EvaluationGraphs``) — the layout's first pass
    runs eagerly on its prepared frame, and a single fit still due is served
    from its slot rather than a pass of its own.
    """
    scores: dict[int, dict[str, float]] = {}
    if len(due) == 1 and (graphs is None or graphs.bank is None):
        fit = due[0]
        split = _eval_split(fit)
        with _tally(fit):
            if fit.graph_pool is not None:
                # build the fit's eval executor on the fit's pool first; the
                # pass finds it kept on the point executor (`eval_executor`)
                _eval_executor(
                    fit.doc, fit.executor, request, split, pool=fit.graph_pool
                )
            scores[id(fit)] = _run_eval(fit.doc, fit.executor, request, split)
    else:
        prepared: list[tuple[_Fit, PointExecutor]] = []
        by_split: dict[str, list[tuple[_Fit, PointExecutor]]] = {}
        for fit in due:
            split = _eval_split(fit)
            eval_executor = _eval_executor(
                fit.doc, fit.executor, request, split, pool=fit.graph_pool
            )
            _fresh_for_eval(eval_executor)
            prepared.append((fit, eval_executor))
            by_split.setdefault(split, []).append((fit, eval_executor))

        def members_of(
            window: Sequence[tuple[_Fit, PointExecutor]],
        ) -> list[tuple[PointExecutor, Sequence[str]]]:
            return [(executor, _eval_reads(fit.doc)) for fit, executor in window]

        def body(window: Sequence[tuple[_Fit, PointExecutor]]) -> None:
            _run_batched(
                [
                    (fit, eval_executor, _eval_reads(fit.doc))
                    for fit, eval_executor in window
                ],
                frames=graphs.frames_for(members_of(window))
                if graphs is not None
                else None,
            )

        def abandon(window: Sequence[tuple[_Fit, PointExecutor]]) -> None:
            for _fit, eval_executor in window:
                eval_executor.reset_reads()

        # the replays first, outside any featurizer scope: a capture's
        # warm-up and recording passes run under their own isolated scope
        # (`cuda_graphs.captured_pass`), so every stage is evaluated inside
        # the graph and nothing shared in from eager code is baked into it.
        # The replay is the whole layout's frame, so it stands in for the
        # pass only where the budget would run the group as one window
        replayed: set[str] = set()
        if graphs is not None:
            for split, group in by_split.items():
                _whole, rest = budget.take(group, _window_rows)
                if not rest:
                    if graphs.forward(members_of(group)):
                        replayed.add(split)
                elif graphs.holds(members_of(group)):
                    # the bound fell below this group. It only falls because
                    # the device ran short (`RowBudget.shrink`,
                    # `_advance_eval_budget`; never back up), and the
                    # layout's replay is the fit's widest eval frame — so the
                    # capture is given back now rather than held against a
                    # later due set small enough to fit again (members stop),
                    # and not recaptured at that smaller size: the recapture
                    # churn `CohortGraphs` declines too
                    graphs.invalidate()
        # one scope for the eager remainder of the round: the stages are in
        # eval mode and do not move until the next optimizer step, so every
        # window's forwards and the scoring reads share one evaluation of
        # each stage
        try:
            with featurizer_cache():
                for split, group in by_split.items():
                    if split not in replayed:
                        _run_windows(group, budget, body, abandon, None)
                for fit, eval_executor in prepared:
                    with _tally(fit):
                        scores[id(fit)] = _score(fit.doc, eval_executor)
        finally:
            # scored (or not), released — as `_run_eval` releases its own:
            # device storage nothing needs until the next pass, and a
            # replay's values alias the graph pool the next training replay
            # overwrites, which nothing may still hold by then
            # (cuda_graphs.Replay, ``pool``)
            for _fit, eval_executor in prepared:
                eval_executor.reset_reads()
    return [scores[id(fit)] for fit in due]


def _eval_split(fit: _Fit) -> str:
    assert fit.spec.eval_split is not None
    return fit.spec.eval_split


def _run_eval(
    doc: Document,
    executor: PointExecutor,
    request: ExecutionRequest,
    split: str,
) -> dict[str, float]:
    """One eval pass over ``split`` for one fit: the declared eval metrics,
    hard-gate eval mode. ``executor`` is the point's full-data executor; the
    pass runs on its :func:`_eval_executor` (built beforehand on the fit's
    pool when the fit captures graphs)."""
    eval_executor = _eval_executor(doc, executor, request, split)
    copy_executor_stages(eval_executor, executor)
    _fresh_for_eval(eval_executor)
    try:
        # after the copy and the mode switch, so the shared evaluation is of
        # the stages this pass scores
        with featurizer_cache():
            return _score(doc, eval_executor)
    finally:
        eval_executor.reset_reads()


def _eval_executor(
    doc: Document,
    executor: PointExecutor,
    request: ExecutionRequest,
    split: str,
    *,
    pool: GraphPool | None = None,
) -> PointExecutor:
    """The executor every eval pass of this fit runs on — built on the first
    pass and kept on the point executor (``eval_executor``) for the rest.

    One per fit, not one per pass: the split's rows are read and tokenized
    once, and its captures live under ``(digest, split)`` in the shared store
    — the eval split is not the campaign's rows, so it gets its own key —
    with only the fit-constant groups eligible (``inner_interning``). The
    trained model's group is re-run on every pass, as it must be.
    The row bound (``batch_rows``) carries over from the point executor: an
    eval pass is a no-grad forward over the whole split, exactly what
    microbatching is for.

    Built lazily, on the first pass, rather than before the epoch loop: an
    ``updates`` budget shorter than one epoch never evaluates, and the eval
    split is only guaranteed readable through the eval path."""
    if executor.eval_executor is not None:
        return executor.eval_executor
    cache = executor.graph_cache

    def build() -> PointExecutor:
        split_rows = request.env.datasets.rows(split)
        # Retained captures must not mutate a prior fit's saved stages.
        stages = (
            copy.deepcopy(executor.stage_cache)
            if cache is not None
            else executor.stage_cache
        )
        return _inner_executor(
            doc,
            executor,
            role_rows={role: split_rows for role in executor.role_rows},
            grad_enabled=False,
            rows=split,
            stage_cache=stages,
        )

    built = cache.evaluation(split, build) if cache is not None else build()
    # its reads stay on the device when a metric selects from them there
    built.device_reads = device_scored(doc)
    if isinstance(built, GraphExecutor):
        # its inference replays join the fit's pool (GraphPool); the point
        # executor itself never captures during a fit and carries no pool
        built.graph_pool = pool
    executor.eval_executor = built
    return built
