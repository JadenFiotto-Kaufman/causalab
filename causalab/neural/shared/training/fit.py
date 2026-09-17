"""One point's fit as the loop holds it (:class:`Fit`), how it is prepared
from the document (:func:`prepare_fit`: seed, optimizer groups, minibatch
partition, schedules), and the seam an engine builds a fit's inner executors
through (:class:`ExecutorFactory`)."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Protocol, Sequence, TypeVar

import torch

from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.execution import Checkpoint
from causalab.neural.shared.executor_base import (
    ExecutorBase,
    ForwardCache,
    document_seed,
)
from causalab.neural.shared.featurizers import Stage
from causalab.neural.shared.training.draw import Drawn, slice_rows
from causalab.neural.shared.training.schedules import (
    DUAL_GROUP,
    Control,
    Phase,
    build_controls,
    build_phases,
    parse_anneals,
)
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import (
    PER_PARAMS_OPTIMIZER_FIELDS,
    READ_TARGET_METRIC_KINDS,
    AnnealSchedule,
    ConstraintSpec,
    Document,
    TrainSpec,
    concrete_int,
)

__all__ = [
    "ExecutorFactory",
    "Fit",
    "add_dual_groups",
    "build_optimizer",
    "checkpoint_steps",
    "prepare_fit",
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


F = TypeVar("F", bound="Fit")


@dataclasses.dataclass
class Fit:
    """One point's fit, as the loop advances it: what the document declared,
    what the loop built for it, and where it stands. An engine whose forward
    keeps state per fit subclasses it (``prepare_fit(fit_type=…)``)."""

    doc: Document
    executor: ExecutorBase
    seed: int
    stages: dict[str, Stage]
    trained_names: tuple[str, ...]
    optimizer: torch.optim.Optimizer
    batches: list[list[int]]
    minibatch_executors: list[ExecutorBase]
    epochs: int
    total_steps: int
    anneals: dict[str, AnnealSchedule]
    eval_every_epochs: int | None
    order_rng: torch.Generator
    #: the generator a ``hard_concrete`` gate's per-step mask draw comes from
    #: — this member's own, so a cohort cannot mix members' samples
    mask_rng: torch.Generator
    #: this fit's reads the objective needs, in order — the groups a step runs
    objective_reads: tuple[str, ...]
    #: how this fit's inner executors are built — the minibatches at prepare
    #: and on a redraw, the eval executor on the first pass
    executor_factory: "ExecutorFactory"
    step: int = 0
    epoch: int = 0
    #: the current epoch's minibatch order and the next position in it
    order: list[int] = dataclasses.field(default_factory=list)
    position: int = 0
    active: bool = True
    best: float | None = None
    stale: int = 0
    eval_passes: int = 0
    last_score: dict[str, float] | None = None
    #: the executor :func:`~causalab.neural.shared.training.loop.evaluate_fits`
    #: runs this fit's eval passes on, built on the first pass
    eval_executor: ExecutorBase | None = None
    # `early_stop` selects a fit by its eval score, so the fit it selected is
    # the one that must be saved. Without a snapshot the loop returned the
    # *last* stages — after `patience` non-improving evals, the worst of the
    # tail — and nothing in the saved bundle said which you had.
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_score: dict[str, float] | None = None
    #: what this fit's inner passes paid for constant groups and were served
    run: int = 0
    served: int = 0
    #: the block each forward this fit took part in resumed at
    resumed: list[int] = dataclasses.field(default_factory=list)
    #: §2.11 ``control``: the live value of every named term's weight (the
    #: authored weight is a controller's start), the controllers bound to
    #: this fit's stages, and each controller's per-update trace
    live_weights: dict[str, float] = dataclasses.field(default_factory=dict)
    controls: dict[str, Control] = dataclasses.field(default_factory=dict)
    #: §2.11 ``phases``: the windows in update terms, the optimizer group
    #: each ``train.params`` entry owns (its base ``lr`` / ``weight_decay``
    #: kept so a phase can restore them), and which window the fit is in —
    #: ``-1`` before the first update
    phases: tuple[Phase, ...] = ()
    groups_by_entry: dict[str, int] = dataclasses.field(default_factory=dict)
    phase_index: int = -1
    control_trace: dict[str, list[dict[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: §2.11 ``constraint``: per constraint term, its dual pair ``(λ₁, λ₂)``
    #: — a parameter in the fit's optimizer under a ``maximize`` group — and
    #: the per-update trace of the duals after their ascent
    duals: dict[str, torch.nn.Parameter] = dataclasses.field(default_factory=dict)
    #: the authored constraint per term name — the target and the initial
    #: duals are read from here, not recovered from the trace
    constraint_specs: dict[str, ConstraintSpec] = dataclasses.field(
        default_factory=dict
    )
    constraint_trace: dict[str, list[dict[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: §2.2 ``draw``: the fit's drawn roles (``None`` when no role draws) —
    #: the redraw that rebuilds the minibatch executors with one fresh member
    #: per row each epoch, and per drawn role the member each row took
    drawn: Drawn | None = None
    #: §2.12 ``trajectory``: the updates after which the fit is photographed,
    #: and the photographs
    checkpoint_steps: frozenset[int] = frozenset()
    checkpoints: list[Checkpoint] = dataclasses.field(default_factory=list)
    #: the last update's loss and its terms (``term.<name>``, ``weight.<name>``),
    #: what a checkpoint taken after that update records — kept as the
    #: detached device scalars (a weight is the float the update used) and
    #: read to the host together by :meth:`loss_record` only when a
    #: checkpoint asks, so an update pays no round trip for a record it
    #: does not take
    last_loss: torch.Tensor | None = None
    term_values: dict[str, torch.Tensor | float] = dataclasses.field(
        default_factory=dict
    )

    def loss_record(self) -> tuple[float, dict[str, float]]:
        """The last update's ``(loss, term_values)`` as floats: every tensor
        of the record in one host read, the values ``float(tensor)`` would
        give — ``stack`` promotes to the widest dtype among them, and a float
        widened is the same number; ``nan`` and no terms before the first
        update."""
        if self.last_loss is None:
            return float("nan"), {}
        tensors = [self.last_loss] + [
            value
            for value in self.term_values.values()
            if isinstance(value, torch.Tensor)
        ]
        read = iter(
            torch.stack(
                [t.reshape(()).to(self.last_loss.device) for t in tensors]
            ).tolist()
        )
        loss = float(next(read))
        return loss, {
            key: float(next(read)) if isinstance(value, torch.Tensor) else value
            for key, value in self.term_values.items()
        }

    def minibatches_rebuilt(self) -> None:
        """A redraw replaced ``minibatch_executors`` (§2.2 ``draw``): the
        seam for an engine that keys state by minibatch executor."""

    @property
    def train(self) -> Any:
        assert self.doc.train is not None
        return self.doc.train

    @property
    def store(self) -> ForwardCache | None:
        return self.executor.interning.cache if self.executor.interning else None

    @property
    def exhausted(self) -> bool:
        return self.step >= self.total_steps or self.epoch >= self.epochs


def prepare_fit(
    doc: Document,
    executor: ExecutorBase,
    *,
    executor_factory: ExecutorFactory,
    fit_type: type[F] = Fit,  # type: ignore[assignment]  # the default is the bound
) -> F:
    """One point's :class:`Fit`, ready to step: the seed, the trained stages
    and their optimizer groups, the minibatch partition and one executor per
    minibatch (built by ``executor_factory`` as row selections of the point's
    frame), the schedules. ``executor`` is the point's full-data executor —
    its stage cache is shared with every inner executor, so the stages it
    later evaluates are the fitted ones."""
    train = doc.train
    assert train is not None
    # one reader of train.seed, shared with the featurizer inits the executor
    # builds below — the loop and the init cannot disagree about the seed.
    # Seeded per member, right before its stages are built: a member's init
    # is a function of its own document, whatever fitted beside it
    seed = document_seed(doc)
    torch.manual_seed(seed)

    trained_names = tuple(sorted({p.split(".", 1)[0] for p in train.params}))
    stages: dict[str, Stage] = {}
    parameters: list[torch.nn.Parameter] = []
    # one optimizer parameter group per `train.params` entry, so a per-parameter
    # `lr` / `weight_decay` (§2.11, a mapping keyed by those entries) lands on
    # exactly the tensors its entry names; with scalar settings the groups
    # share every hyperparameter and the optimizer's arithmetic is that of
    # one flat list (Adam's state is per tensor either way)
    groups: list[dict[str, Any]] = []
    for pname in train.params:
        fname, _, slot = pname.partition(".")
        if fname not in doc.featurizers:
            # The compile refuses this under §5 rule 30 (this engine declares
            # no 'train_free_params'); here for a document that arrived
            # unvalidated, so routing should not have sent it here.
            raise ProtocolError(
                "P4",
                f"free params entries ({pname!r}) are not trainable in this "
                "engine — featurizer slots only; validation refuses this at "
                "load (rule 30), so this document arrived unvalidated",
            )
        stage = executor.stage(fname)
        stages[fname] = stage
        if slot:
            group_params = [stage.slot_params()[slot]]
        else:
            group_params = [p for p in stage.parameters() if p.requires_grad]
        parameters.extend(group_params)  # type: ignore[arg-type]
        group: dict[str, Any] = {"params": group_params}
        for field in PER_PARAMS_OPTIMIZER_FIELDS:
            value = train.optimizer.get(field)
            if isinstance(value, Mapping):
                # the schema checked every entry of train.params is a key
                group[field] = float(value[pname])
        groups.append(group)
    if not parameters:
        raise ProtocolError("P2", "train.params resolved to no trainable tensors")

    optimizer = build_optimizer(train.optimizer, groups)
    groups_by_entry = {pname: i for i, pname in enumerate(train.params)}
    duals = add_dual_groups(train, optimizer, executor)
    constraint_specs = {
        term.name: term.constraint
        for term in train.objective
        if term.constraint is not None and term.name is not None
    }

    n_examples = len(executor.rows_for_metrics())
    pairs = concrete_int(train.batch["pairs"], "train.batch.pairs")
    batches = [
        list(range(start, min(start + pairs, n_examples)))
        for start in range(0, n_examples, pairs)
    ]
    if "epochs" in train.steps:
        epochs = concrete_int(train.steps["epochs"], "train.steps.epochs")
        total_steps = epochs * len(batches)
    else:
        total_steps = concrete_int(train.steps["updates"], "train.steps.updates")
        epochs = -(-total_steps // len(batches))

    eval_every_epochs = None
    if train.eval is not None:
        if "epochs" not in train.eval["every"]:
            # This loop only reaches an eval on an epoch boundary. Accepting an
            # `updates` counter here would run *no* eval at all and still save
            # the fit, which is the silent-wrong-number this commit exists to
            # remove — so refuse instead of pretending. The compile refuses
            # this under §5 rule 30 (this engine declares no
            # 'train_eval_updates'); here for a document that arrived
            # unvalidated.
            raise ProtocolError(
                "P4",
                "train.eval.every must count epochs in this engine — "
                f"got {sorted(train.eval['every'])}; an update counter would "
                "silently never evaluate",
            )
        eval_every_epochs = concrete_int(
            train.eval["every"]["epochs"], "train.eval.every.epochs"
        )

    # A minibatch's rows are a *slice* of the campaign's, so its captures live
    # under `(digest, indices)` in the shared ForwardCache — never under the
    # whole role's key — and only the groups this fit cannot change are ever
    # read from or written to it (`inner_interning`, §4 "Fits"). For the
    # shipped methods that is the source forward: run once per slice here,
    # then served on every step, epoch and point that shares the digest.
    # The minibatch is a row *selection* of the point's frame, not a fresh
    # encode of its rows: every minibatch of every point in a cohort is then
    # in one padded frame, which is what lets their forwards concatenate.
    # No `batch_rows=`: `train.batch.pairs` is the document's own batching
    # knob for grad forwards, so the execution bound applies to the no-grad
    # passes and to `train.eval` (below), not to a minibatch.
    # the draw's refusals first: they read the rows, not the frames, and a
    # refused document should not pay for an encode
    drawn = Drawn.of(doc, executor, seed, executor_factory)
    frames = {role: executor.frame(role) for role in executor.role_rows}
    if drawn is not None:
        # §2.2 `draw`: this fit's minibatches read a freshly drawn member per
        # row each epoch, encoded as selections of one expanded frame; the
        # inner store is not consulted for them (a source forward over a
        # drawn role changes every epoch, so nothing about it is constant)
        drawn.bind(batches, frames)
        minibatch_executors = drawn.minibatches()
    else:
        minibatch_executors = [
            executor_factory(
                doc,
                executor,
                role_rows=slice_rows(executor.role_rows, indices),
                grad_enabled=True,
                rows=tuple(indices),
                batches={role: frame.select(indices) for role, frame in frames.items()},
            )
            for indices in batches
        ]
    objective_reads: list[str] = []
    for term in train.objective:
        if term.metric is not None:
            metric = doc.metrics[term.metric]
            objective_reads.append(str(metric.of))
            if metric.kind in READ_TARGET_METRIC_KINDS:
                objective_reads.append(str(metric.fields["target"]))
    # §2.11 `control`: the closed-loop schedules, and the live value of every
    # named term's weight (the authored weight is a controller's start)
    live_weights: dict[str, float] = {
        term.name: float(term.weight)
        for term in train.objective
        if term.name is not None and isinstance(term.weight, (int, float))
    }
    controls = build_controls(train.control, stages, live_weights)
    anneals = parse_anneals(train.anneal, stages, live_weights)
    phases = build_phases(train.phases, total_steps, stages, live_weights)
    fit = fit_type(
        doc=doc,
        executor=executor,
        seed=seed,
        stages=stages,
        trained_names=trained_names,
        optimizer=optimizer,
        batches=batches,
        minibatch_executors=minibatch_executors,
        epochs=epochs,
        total_steps=total_steps,
        anneals=anneals,
        eval_every_epochs=eval_every_epochs,
        order_rng=torch.Generator().manual_seed(seed),
        # its own generator object at the same seed, as the batch order and
        # the subspace init have theirs: a local stream, never the global one
        mask_rng=torch.Generator().manual_seed(seed),
        objective_reads=tuple(objective_reads),
        executor_factory=executor_factory,
        live_weights=live_weights,
        controls=controls,
        control_trace={target: [] for target in controls},
        duals=duals,
        constraint_specs=constraint_specs,
        constraint_trace={name: [] for name in duals},
        phases=phases,
        drawn=drawn,
        groups_by_entry=groups_by_entry,
        # §2.12 `trajectory`: the updates after which the fit is photographed
        checkpoint_steps=frozenset(checkpoint_steps(doc, total_steps, len(batches))),
    )
    return fit


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


def add_dual_groups(
    train: TrainSpec, optimizer: torch.optim.Optimizer, executor: ExecutorBase
) -> dict[str, torch.nn.Parameter]:
    """§2.11 ``constraint``: one ``(λ₁, λ₂)`` parameter per constraint term,
    appended to the fit's optimizer as its own group — ``maximize`` so the
    step is an ascent on ``λ₁(s − t) + λ₂(s − t)²``, the authored ``dual.lr``,
    and none of what the group would otherwise inherit from the constructor:
    no weight decay and no momentum, so under ``sgd`` the step is exactly
    ``lr · gradient`` whatever the document's ``momentum`` (under ``adamw``
    it is Adam's, as §2.11 says). The duals share the optimizer's
    arithmetic and nothing else: not its schedule, not its phases. A dual
    is no stage's parameter, so an engine that maps optimizer parameters
    onto stages by identity refuses a constraint before it prepares the fit."""
    constrained = [term for term in train.objective if term.constraint is not None]
    duals: dict[str, torch.nn.Parameter] = {}
    for term in constrained:
        assert term.constraint is not None and term.name is not None
        lam = torch.nn.Parameter(
            torch.tensor(
                list(term.constraint.init),
                dtype=torch.float32,
                device=executor.bundle.device,
            )
        )
        optimizer.add_param_group(
            {
                "params": [lam],
                "lr": float(term.constraint.dual_lr),
                "weight_decay": 0.0,
                "momentum": 0.0,
                "maximize": True,
                DUAL_GROUP: term.name,
            }
        )
        duals[term.name] = lam
    return duals


def _optimizer_default(spec: Mapping[str, Any], field: str, fallback: float) -> float:
    """The constructor-level value of ``lr`` / ``weight_decay``: the scalar the
    document gave, or — when the field is a per-parameter mapping (§2.11) —
    the largest of its values. Torch needs one default even when every
    parameter group overrides it; every group *does* override it here
    (:func:`prepare_fit` writes the field on each group), so the default is
    never the value any tensor is stepped with."""
    value = spec.get(field, fallback)
    if isinstance(value, Mapping):
        return max(float(v) for v in value.values())
    return float(value)


def build_optimizer(
    spec: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]
) -> torch.optim.Optimizer:
    """The optimizer over ``groups`` — one ``{"params": [...], lr?,
    weight_decay?}`` per ``train.params`` entry; a group's own ``lr`` /
    ``weight_decay`` override the constructor's, which is how a rotation and a
    gate step at different rates inside one fit."""
    name = str(spec["name"])
    lr = _optimizer_default(spec, "lr", 0.0)
    weight_decay = _optimizer_default(spec, "weight_decay", 0.0)
    params = [dict(g) for g in groups]
    if name in ("adamw", "adam"):
        raw_betas = spec.get("betas", (0.9, 0.999))
        betas = (float(raw_betas[0]), float(raw_betas[1]))
        eps = float(spec.get("eps", 1e-8))
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(spec.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    raise ProtocolError("P4", f"unknown optimizer {name!r}")
