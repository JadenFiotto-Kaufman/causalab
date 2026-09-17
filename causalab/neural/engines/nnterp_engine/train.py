"""This engine's train runner (spec §2.11): **plan → run → load**.

What a ``train`` section means, and the update loop itself, are
:mod:`causalab.neural.shared.training`'s; the fit's body — the stages, the
optimizer, the forwards of a step as traces, the backward between them — is
:mod:`.fit`'s, and runs where the model is. What is here is the client's two
ends of it:

* **plan** (:func:`plan_fit`): the fit as data
  (:class:`~causalab.neural.engines.nnterp_engine.fit.TrainPlan`). The
  ``FitSpec``, with a recipe for every stage the fit's forwards name; the
  client's own stages, built on the point executor's cache in the spec's
  order, their saved starts recorded for the plan; the step's template
  programs, planned once on a gradient-enabled executor over the point's
  whole frame; the eval split's programs and metrics, planned up front when
  the fit will evaluate.
* **run** (:func:`~causalab.neural.engines.nnterp_engine.fit.run_fit`): in
  this process, or as one NDIF job — the same body either way.
* **load** (:func:`_load`): the returned state into the client's **own**
  stage objects — the point executor's ``stage_cache``, which the finish
  phase evaluates — and the returned records into a ``TrainOutcome``. The
  digest of each stage as the fit built it is compared with the client's; a
  difference (another BLAS's last bit in a QR) warns, and the returned state
  stands either way.

Every fit is a cohort of one, and one job.

A §2.2 ``draw`` re-plans the step's programs each epoch from a fresh draw,
which only the client can do: it runs locally (``run_fit(redraw=)``) and is
refused for a remote fit.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Any, Mapping, Sequence

from causalab.neural.engines.nnterp_engine.executor import NnterpExecutor
from causalab.neural.engines.nnterp_engine.program import GroupProgram
from causalab.neural.engines.nnterp_engine.fit import (
    Artifacts,
    TrainPlan,
    run_fit,
    stage_digest,
)
from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.execution import Checkpoint, TrainEvalScore, TrainOutcome
from causalab.neural.shared.executor_base import ExecutorBase
from causalab.neural.shared.featurizers import Gate
from causalab.neural.shared.services import input_roles
from causalab.neural.shared.training.draw import Drawn
from causalab.neural.shared.training.executors import (
    eval_executor,
    fit_spec,
    score_spec,
    seeded_stages,
)
from causalab.neural.shared.training.spec import FitSpec
from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import Document

__all__ = ["plan_fit", "run_training"]


def _inner_executor(
    doc: Document,
    executor: ExecutorBase,
    *,
    role_rows: Mapping[str, list[dict[str, Any]]],
    grad_enabled: bool,
    rows: tuple[int, ...] | str,
    batches: Mapping[str, EncodedBatch] | None = None,
    drawn: bool = False,
) -> NnterpExecutor:
    """This engine's ``ExecutorFactory`` (``shared/training/executors.py``):
    the executor a fit's programs are planned on — over the point's rows with
    gradients, or over the eval split's without — sharing the point's stage
    cache. ``rows`` and ``drawn`` key a forward store, which this engine does
    not keep."""
    del rows, drawn
    assert isinstance(executor, NnterpExecutor)
    return NnterpExecutor(
        doc,
        executor.bundle,
        role_rows=role_rows,
        role_fields=executor.role_fields,
        load_tensors=executor.load_tensors,
        load_table=executor.load_table,
        stage_cache=executor.stage_cache,  # shared: one stage per name
        grad_enabled=grad_enabled,
        coords=executor.coords,
        batches=batches,
        remote=executor.remote,
    )


def _with_every_stage(
    spec: FitSpec, doc: Document, executor: NnterpExecutor
) -> FitSpec:
    """``spec`` with a recipe for every featurizer a read or a write of the
    document names, after the trained ones: a fit's programs carry stacks by
    name, so the fit builds every stage they resolve to, in this order."""
    names = [recipe.name for recipe in spec.recipes]
    for entry in (*executor.doc.reads.values(), *executor.doc.writes.values()):
        chain = entry.featurizer
        if isinstance(chain, str):
            chain = (chain,)
        for name in chain if isinstance(chain, (list, tuple)) else ():
            if name not in names:
                names.append(str(name))
    extra = names[len(spec.recipes) :]
    return dataclasses.replace(
        spec,
        recipes=spec.recipes + tuple(executor.stage_recipe(name) for name in extra),
        featurizers={**spec.featurizers, **{n: doc.featurizers[n] for n in extra}},
    )


@dataclasses.dataclass(eq=False)
class _Planned:
    """One point's fit, planned: what ships, and what stays here to finish
    it — the digest of each client stage as built, and the drawn roles."""

    plan: TrainPlan
    executor: NnterpExecutor
    init_digest: dict[str, str]
    drawn: Drawn | None


def plan_fit(
    doc: Document, executor: NnterpExecutor, request: ExecutionRequest
) -> _Planned:
    """Plan one point's fit on its full-data ``executor`` (module docstring):
    the client's stages are built here, on the executor's own cache."""
    drawing = sorted(
        role for role, data in input_roles(doc).items() if data.draw is not None
    )
    if drawing and executor.remote:
        raise ProtocolError(
            "P4",
            "this fit draws a counterfactual member per row each epoch "
            f"(data.{drawing[0]}.draw), which re-plans the step's forwards from "
            "the drawn rows every epoch — planning is the client's, and a "
            "remote fit is one job whose programs are planned once, before it "
            "is submitted. Fit a drawn role against a locally loaded bundle "
            "(remote=False).",
        )
    spec = _with_every_stage(fit_spec(doc, executor), doc, executor)
    drawn = Drawn.of(doc, executor, spec.seed, _inner_executor)

    # the client's stages, their saved starts recorded for the plan
    artifacts = Artifacts()
    loaders = executor.load_tensors, executor.load_table
    executor.load_tensors, executor.load_table = artifacts.recording(*loaders)
    try:
        seeded_stages(spec, executor)
    finally:
        executor.load_tensors, executor.load_table = loaders
    init_digest = {
        recipe.name: stage_digest(executor.stage_cache[recipe.name])
        for recipe in spec.recipes
    }

    frames = {role: executor.frame(role) for role in executor.role_rows}
    every_row = list(range(len(executor.rows_for_metrics())))
    if drawn is not None:
        # a draw is a selection of one expanded frame; one "minibatch" of
        # every row is the whole drawn frame the step's templates are over
        drawn.bind([every_row], frames)
        (planner,) = drawn.minibatches()
    else:
        planner = _inner_executor(
            doc,
            executor,
            role_rows=executor.role_rows,
            grad_enabled=True,
            rows=tuple(every_row),
            batches=frames,
        )
    assert isinstance(planner, NnterpExecutor)
    train = planner.fit_programs(spec.objective_reads)

    evaluation: tuple[GroupProgram, ...] = ()
    if spec.eval_every_epochs is not None and spec.epochs >= spec.eval_every_epochs:
        evaluator = eval_executor(doc, executor, request, _inner_executor)
        assert isinstance(evaluator, NnterpExecutor)
        # the eval pass travels with the fit: its metrics are resolved here,
        # where the tokenizer is, and scored where the fit runs
        spec = dataclasses.replace(
            spec,
            score=score_spec(
                doc, evaluator.rows_for_metrics(), evaluator.bundle.tokenizer
            ),
        )
        assert spec.score is not None
        evaluation = evaluator.fit_programs(spec.score.reads)

    plan = TrainPlan(
        label=f"{executor.bundle.key}{dict(executor.coords) or ''}",
        spec=spec,
        train=train,
        eval=evaluation,
        artifacts=artifacts,
    )
    return _Planned(plan, executor, init_digest, drawn)


def _redraw(drawn: Drawn, spec: FitSpec) -> tuple[GroupProgram, ...]:
    """A new epoch's draw (§2.2), and the step's programs planned over it."""
    (planner,) = drawn.minibatches()
    assert isinstance(planner, NnterpExecutor)
    return planner.fit_programs(spec.objective_reads)


def _load(planned: _Planned, result: Mapping[str, Any]) -> TrainOutcome:
    """The fit's result into the client: each stage's state and plain
    attributes onto the point executor's own stage object — identity kept, so
    the finish phase evaluates the fitted featurizer — left as ``finish``
    leaves a stage (eval mode, no pinned mask); then the outcome."""
    cache = planned.executor.stage_cache
    moved = sorted(
        name
        for name, digest in result["init_digest"].items()
        if planned.init_digest.get(name) != digest
    )
    if moved:
        warnings.warn(
            f"fit {planned.plan.label}: the stages {moved} were built where the "
            "fit ran with other bits than here (a differing BLAS moves the last "
            "bit of a QR-completed rotation). The returned state is the fit's "
            "and is what is loaded; a local re-run would start elsewhere.",
            stacklevel=2,
        )
    for name, state in result["state"].items():
        stage = cache[name]
        stage.load_state_dict(dict(state))
        for attr, value in result["attrs"][name].items():
            if attr == "pool":
                for pool_attr, pool_value in value.items():
                    setattr(stage.pool, pool_attr, pool_value)
            else:
                setattr(stage, attr, value)
        stage.eval()
        if isinstance(stage, Gate):
            stage.frozen_mask = None
    spec = planned.plan.spec
    return TrainOutcome(
        stages={name: cache[name] for name in spec.trained_names},
        eval_score=(
            None
            if result["eval_score"] is None
            else TrainEvalScore(**result["eval_score"])
        ),
        diagnostics=result["diagnostics"],
        controls=result["controls"],
        control_trace=result["control_trace"],
        constraints=result["constraints"],
        constraint_trace=result["constraint_trace"],
        anneals=result["anneals"],
        phases=tuple(result["phases"]),
        checkpoints=tuple(Checkpoint(**c) for c in result["checkpoints"]),
        draws=planned.drawn.record() if planned.drawn is not None else {},
    )


def run_training(
    docs: Sequence[Document],
    executors: Sequence[NnterpExecutor],
    request: ExecutionRequest,
) -> list[TrainOutcome]:
    """Fit each point; one outcome per point, in order.

    Each ``executor`` is its point's full-data executor: the fitted state is
    loaded into the stages of its own cache, which the finish phase reads, and
    the outcome's ``stages`` are those same objects. With
    ``train.early_stop`` they hold the best-scoring weights
    (``eval_score.selected``), otherwise the last. ``executor.remote`` is
    where the fit runs — here, or as one NDIF job."""
    if len(docs) != len(executors):
        raise ValueError(
            f"{len(docs)} documents but {len(executors)} executors — one per point"
        )
    outcomes: list[TrainOutcome] = []
    for doc, executor in zip(docs, executors):
        planned = plan_fit(doc, executor, request)
        with executor._kernel_path():  # pyright: ignore[reportPrivateUsage]
            result = run_fit(
                executor.bundle.model,
                planned.plan,
                remote=executor.remote,
                redraw=(
                    None
                    if planned.drawn is None
                    else functools.partial(_redraw, planned.drawn, planned.plan.spec)
                ),
                # a job's log lines are the only sign of life a client waiting
                # on one gets; a fit in this process prints its own
                progress=bool(executor.remote) and executor.remote != "local",
            )
        outcomes.append(_load(planned, result))
    return outcomes
