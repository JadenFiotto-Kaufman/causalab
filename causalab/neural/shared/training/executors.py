"""The executor side of a fit — what an engine does *around* the loop, shared
because every engine does it the same way: make the fit's plain data from a
document and its point executor (:func:`fit_spec`, :func:`score_spec`), build
the trained stages on the executor's own cache (:func:`seeded_stages`), cut
the minibatch executors (:func:`minibatch_executors`, through the engine's
:class:`ExecutorFactory`), and score an eval executor's reads
(:func:`eval_pass`).

Nothing the loop imports lives here: :mod:`.loop`, :mod:`.state`,
:mod:`.spec` and :mod:`.objective` never see an executor.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

import torch

from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.executor_base import ExecutorBase, document_seed
from causalab.neural.shared.featurizers import Stage, StageRecipe, featurizer_cache
from causalab.neural.shared.metrics import (
    GATHERED_KINDS,
    excluded_rows,
    metric_in_ids,
)
from causalab.neural.shared.training.draw import Drawn, slice_rows
from causalab.neural.shared.training.objective import score
from causalab.neural.shared.training.spec import (
    EarlyStop,
    FitSpec,
    ResolvedMetric,
    ScoreSpec,
)
from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import (
    READ_TARGET_METRIC_KINDS,
    Document,
    concrete_int,
    concrete_str,
    metric_column_fields,
    metric_reads_vocabulary,
)

__all__ = [
    "ExecutorFactory",
    "checkpoint_steps",
    "device_scored",
    "eval_executor",
    "eval_pass",
    "eval_reads",
    "fit_spec",
    "fresh_for_eval",
    "minibatch_executors",
    "resolve_metric",
    "score_executor",
    "score_spec",
    "seeded_stages",
]


class ExecutorFactory(Protocol):
    """How an engine builds a fit's inner executors over the point's
    ``executor`` — same bundle, role fields, loaders and coordinates, and the
    point's **own stage cache**, so every inner executor evaluates the stages
    the optimizer steps.

    ``grad_enabled=True`` is a training minibatch: ``role_rows`` and
    ``batches`` are the same row selection of the point's rows and encoded
    frames. ``grad_enabled=False`` is the eval executor: ``role_rows`` are the
    eval split's, encoded by the executor itself (``batches=None``). ``rows``
    names the rows for an engine that keys a forward store by them — the
    minibatch's indices, or the split's name. ``drawn`` marks a minibatch over
    a freshly drawn role (§2.2 ``draw``), whose forwards are constant for no
    two epochs."""

    def __call__(
        self,
        doc: Document,
        executor: ExecutorBase,
        *,
        role_rows: Mapping[str, list[dict[str, Any]]],
        grad_enabled: bool,
        rows: tuple[int, ...] | str,
        batches: Mapping[str, EncodedBatch] | None = None,
        drawn: bool = False,
    ) -> ExecutorBase: ...


#: The answer columns an objective kind gathers at, one id per row — what
#: ``objective.metric_tensor`` reads as ``token_ids``.
_OBJECTIVE_ID_FIELDS: Mapping[str, tuple[str, ...]] = {
    "cross_entropy": ("target",),
    "logit_diff": ("a", "b"),
    "soft_accuracy": ("a", "b"),
}


def resolve_metric(
    doc: Document,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    objective: bool,
) -> ResolvedMetric:
    """Metric ``name`` over ``rows`` with its answers resolved to token ids
    (``metrics.metric_in_ids``) — the last place the tokenizer is asked
    anything. ``objective`` resolves every row and keeps the answer columns
    as tensors a minibatch indexes; otherwise the rows with no answer stay
    excluded, and a gathered kind's ids are listed once for every pass."""
    metric = doc.metrics[name]
    resolved, id_rows = metric_in_ids(
        metric, rows, tokenizer, eligible_only=not objective
    )
    kind = str(metric.kind)
    vocabulary = len(tokenizer)
    token_ids: dict[str, torch.Tensor] = {}
    gathered_ids: dict[str, tuple[int, ...]] | None = None
    if objective:
        columns = metric_column_fields(resolved)
        for field in _OBJECTIVE_ID_FIELDS.get(kind, ()):
            token_ids[field] = torch.tensor(
                [row[columns[field]] for row in id_rows], dtype=torch.long
            )
    elif kind in GATHERED_KINDS:
        # `metrics.metric_token_ids`' lists — the ids of the rows that carry
        # answers, in row order — read off the rows just resolved
        excluded = excluded_rows(resolved, id_rows, kind)
        columns = metric_column_fields(resolved)
        gathered_ids = {
            field: tuple(
                row[column] for i, row in enumerate(id_rows) if i not in excluded
            )
            for field, column in columns.items()
        }
    return ResolvedMetric(
        name=name,
        metric=resolved,
        rows=tuple(id_rows),
        vocabulary=vocabulary,
        vocab_axis=metric_reads_vocabulary(doc, metric),
        token_ids=token_ids,
        gathered_ids=gathered_ids,
    )


def fit_spec(doc: Document, executor: ExecutorBase) -> FitSpec:
    """One point's :class:`~causalab.neural.shared.training.spec.FitSpec`:
    the ``train`` section resolved against the point's rows — the minibatch
    partition, the update budget, the objective's metrics in token ids, the
    eval cadence — and a recipe per trained stage, read off ``executor``
    (its point's full-data executor) without building anything."""
    train = doc.train
    assert train is not None
    for pname in train.params:
        if pname.partition(".")[0] not in doc.featurizers:
            # The compile refuses this under §5 rule 30 (this engine declares
            # no 'train_free_params'); here for a document that arrived
            # unvalidated, so routing should not have sent it here.
            raise ProtocolError(
                "P4",
                f"free params entries ({pname!r}) are not trainable in this "
                "engine — featurizer slots only; validation refuses this at "
                "load (rule 30), so this document arrived unvalidated",
            )
    rows = executor.rows_for_metrics()
    pairs = concrete_int(train.batch["pairs"], "train.batch.pairs")
    batches = tuple(
        tuple(range(start, min(start + pairs, len(rows))))
        for start in range(0, len(rows), pairs)
    )
    if "epochs" in train.steps:
        epochs = concrete_int(train.steps["epochs"], "train.steps.epochs")
        total_steps = epochs * len(batches)
    else:
        total_steps = concrete_int(train.steps["updates"], "train.steps.updates")
        epochs = -(-total_steps // len(batches))

    eval_every_epochs = None
    if train.eval is not None:
        if "epochs" not in train.eval["every"]:
            # The loop only reaches an eval on an epoch boundary. Accepting an
            # `updates` counter here would run *no* eval at all and still save
            # the fit — a silent wrong number — so refuse instead of
            # pretending. The compile refuses this under §5 rule 30 (no engine
            # here declares 'train_eval_updates'); here for a document that
            # arrived unvalidated.
            raise ProtocolError(
                "P4",
                "train.eval.every must count epochs in this engine — "
                f"got {sorted(train.eval['every'])}; an update counter would "
                "silently never evaluate",
            )
        eval_every_epochs = concrete_int(
            train.eval["every"]["epochs"], "train.eval.every.epochs"
        )

    tokenizer = executor.bundle.tokenizer
    metrics: dict[str, ResolvedMetric] = {}
    objective_reads: list[str] = []
    for term in train.objective:
        if term.metric is None:
            continue
        if term.metric not in metrics:
            metrics[term.metric] = resolve_metric(
                doc, term.metric, rows, tokenizer, objective=True
            )
        objective_reads.extend(metrics[term.metric].reads)

    # a stage per trained featurizer in `train.params` order, then every
    # member of a budget pool: a pool is built whole, whichever member is
    # asked for first (`featurizers.link_budget_pools`)
    names = list(dict.fromkeys(pname.partition(".")[0] for pname in train.params))
    names += [
        name
        for name in sorted(doc.featurizers)
        if name not in names
        and doc.featurizers[name].kind == "gate"
        and isinstance(doc.featurizers[name].pool, str)
    ]
    recipes: list[StageRecipe] = [executor.stage_recipe(name) for name in names]

    return FitSpec(
        # one reader of train.seed, shared with the featurizer inits the
        # executor builds — the loop and the init cannot disagree about it
        seed=document_seed(doc),
        params=tuple(train.params),
        trained_names=tuple(sorted({p.split(".", 1)[0] for p in train.params})),
        optimizer=dict(train.optimizer),
        objective=tuple(train.objective),
        metrics=metrics,
        objective_reads=tuple(objective_reads),
        batches=batches,
        epochs=epochs,
        total_steps=total_steps,
        eval_split=(
            concrete_str(train.eval["split"], "train.eval.split")
            if train.eval is not None
            else None
        ),
        eval_metrics=tuple(train.eval["metrics"]) if train.eval is not None else (),
        eval_every_epochs=eval_every_epochs,
        early_stop=(
            EarlyStop(
                metric=str(train.early_stop["metric"]),
                mode=str(train.early_stop["mode"]),
                patience=concrete_int(train.early_stop["patience"], "patience"),
            )
            if train.early_stop is not None
            else None
        ),
        anneal=train.anneal,
        control=train.control,
        phases=train.phases,
        # §2.12 `trajectory`: the updates after which the fit is photographed
        checkpoint_steps=frozenset(checkpoint_steps(doc, total_steps, len(batches))),
        recipes=tuple(recipes),
        featurizers={name: doc.featurizers[name] for name in names},
        model_info=executor.bundle.info,
        coords=dict(executor.coords),
    )


def checkpoint_steps(
    doc: Document, total_steps: int, batches_per_epoch: int
) -> set[int]:
    """The updates after which a ``trajectory`` entry (§2.12) photographs
    the fit, or the empty set. ``count: n`` spaces n checkpoints equally over
    the run — ``floor(i · total / n)`` for ``i = 1..n``, so the last is the
    final update and a run shorter than ``n`` updates yields as many distinct
    checkpoints as it has updates; ``updates: n`` / ``epochs: n`` photograph
    every n of them, and the final update always."""
    entry = next((e for e in doc.save if e.kind == "trajectory"), None)
    if entry is None or entry.every is None or total_steps <= 0:
        return set()
    ((unit, n),) = entry.every.items()
    if unit == "count":
        return {max(1, (i * total_steps) // n) for i in range(1, n + 1)}
    stride = n if unit == "updates" else n * batches_per_epoch
    return {step for step in range(stride, total_steps + 1, stride)} | {total_steps}


def seeded_stages(spec: FitSpec, executor: ExecutorBase) -> dict[str, Stage]:
    """The fit's trained stages, built on ``executor``'s own stage cache —
    shared with every inner executor and with the finish phase after the fit,
    so what they evaluate are the fitted objects.

    Built in ``train.params`` order, then every other recipe — the discipline
    ``state.build_stages`` repeats, so a stage built from the spec alone
    starts bit-identical to the one built here. Every draw is the stage's own
    seeded generator's, so a member's init is a function of its own document
    whatever fitted beside it, and no global RNG moves."""
    trained = {
        fname: executor.stage(fname)
        for fname in dict.fromkeys(pname.partition(".")[0] for pname in spec.params)
    }
    for recipe in spec.recipes:
        executor.stage(recipe.name)  # `build_stages`' order: every other recipe
    return trained


def minibatch_executors(
    doc: Document,
    executor: ExecutorBase,
    spec: FitSpec,
    executor_factory: ExecutorFactory,
) -> tuple[list[ExecutorBase], Drawn | None]:
    """One executor per minibatch of ``spec.batches``, built by the engine's
    factory, and the fit's drawn roles (``None`` when no role draws).

    A minibatch's rows are a *slice* of the campaign's, so its captures live
    under ``(digest, indices)`` in the shared ForwardCache — never under the
    whole role's key — and only the groups this fit cannot change are ever
    read from or written to it (``inner_interning``, §4 "Fits"). The
    minibatch is a row *selection* of the point's frame, not a fresh encode
    of its rows: every minibatch of every point in a cohort is then in one
    padded frame, which is what lets their forwards concatenate. No
    ``batch_rows=``: ``train.batch.pairs`` is the document's own batching
    knob for grad forwards, so the execution bound applies to the no-grad
    passes and to ``train.eval``, not to a minibatch."""
    # the draw's refusals first: they read the rows, not the frames, and a
    # refused document should not pay for an encode
    drawn = Drawn.of(doc, executor, spec.seed, executor_factory)
    frames = {role: executor.frame(role) for role in executor.role_rows}
    batches = [list(indices) for indices in spec.batches]
    if drawn is not None:
        # §2.2 `draw`: this fit's minibatches read a freshly drawn member per
        # row each epoch, encoded as selections of one expanded frame; the
        # inner store is not consulted for them (a source forward over a
        # drawn role changes every epoch, so nothing about it is constant)
        drawn.bind(batches, frames)
        return drawn.minibatches(), drawn
    return [
        executor_factory(
            doc,
            executor,
            role_rows=slice_rows(executor.role_rows, indices),
            grad_enabled=True,
            rows=tuple(indices),
            batches={role: frame.select(indices) for role, frame in frames.items()},
        )
        for indices in batches
    ], None


def score_spec(
    doc: Document, rows: Sequence[Mapping[str, Any]], tokenizer: Any
) -> ScoreSpec:
    """The document's ``train.eval`` metrics resolved over ``rows`` — the
    eval split's, as the eval executor pairs them."""
    assert doc.train is not None and doc.train.eval is not None
    return ScoreSpec(
        split=concrete_str(doc.train.eval["split"], "train.eval.split"),
        metrics=tuple(
            resolve_metric(doc, name, rows, tokenizer, objective=False)
            for name in doc.train.eval["metrics"]
        ),
    )


def eval_reads(doc: Document) -> tuple[str, ...]:
    """The reads the document's ``train.eval`` metrics reduce, in order."""
    assert doc.train is not None and doc.train.eval is not None
    reads: list[str] = []
    for name in doc.train.eval["metrics"]:
        metric = doc.metrics[name]
        reads.append(str(metric.of))
        if metric.kind in READ_TARGET_METRIC_KINDS:
            reads.append(str(metric.fields["target"]))
    return tuple(reads)


def device_scored(doc: Document) -> bool:
    """Whether a fit's eval executor keeps its reads on the device
    (``ExecutorBase.device_reads``): some eval metric selects from them there,
    and ``objective.score`` gathers the answer columns and copies those, not
    the vocabulary. A softmax-class eval keeps the one host copy
    ``_finalize_read`` makes."""
    train = doc.train
    eval_metrics = (
        train.eval["metrics"] if train is not None and train.eval is not None else ()
    )
    return any(str(doc.metrics[name].kind) in GATHERED_KINDS for name in eval_metrics)


def eval_executor(
    doc: Document,
    executor: ExecutorBase,
    request: ExecutionRequest,
    executor_factory: ExecutorFactory,
) -> ExecutorBase:
    """The executor a fit's eval passes run on, over its ``train.eval``
    split: the split's rows under every role, no gradients, its reads kept
    where the metrics want them (:func:`device_scored`). Built by the caller
    on the first pass rather than before the loop: an ``updates`` budget
    shorter than one epoch never evaluates, and the eval split is only
    guaranteed readable through the eval path."""
    assert doc.train is not None and doc.train.eval is not None
    split = concrete_str(doc.train.eval["split"], "train.eval.split")
    split_rows = request.env.datasets.rows(split)
    built = executor_factory(
        doc,
        executor,
        role_rows={role: split_rows for role in executor.role_rows},
        grad_enabled=False,
        rows=split,
    )
    built.device_reads = device_scored(doc)
    return built


def fresh_for_eval(evaluator: ExecutorBase) -> None:
    for stage in evaluator.stage_cache.values():
        stage.eval()
    # the stages moved since the last pass, so every read is stale; the
    # encoded split and the interned constant captures are not (reset_reads)
    evaluator.reset_reads()


def score_executor(doc: Document, evaluator: ExecutorBase) -> dict[str, float]:
    """The declared eval metrics over ``evaluator``'s reads — run on its own
    path unless a batched pass or a replay already filled them. The metrics
    are resolved over its rows once (``ExecutorBase.score_spec``: the rows
    never change), and every pass is ``objective.score`` over its
    ``dense_value``."""
    if evaluator.score_spec is None:
        evaluator.score_spec = score_spec(
            doc, evaluator.rows_for_metrics(), evaluator.bundle.tokenizer
        )
    return score(evaluator.score_spec, evaluator.dense_value)


def eval_pass(doc: Document, evaluator: ExecutorBase) -> dict[str, float]:
    """One eval pass on ``evaluator``: eval mode, one :func:`featurizer_cache`
    scope over the pass — opened after the mode switch, so the shared
    evaluation is of the stages this pass scores — and the reads released
    afterwards (device storage nothing needs until the next pass)."""
    fresh_for_eval(evaluator)
    try:
        with featurizer_cache():
            return score_executor(doc, evaluator)
    finally:
        evaluator.reset_reads()
