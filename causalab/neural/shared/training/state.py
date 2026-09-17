"""One point's fit as the loop advances it (:class:`FitState`) and how it is
built from its :class:`~causalab.neural.shared.training.spec.FitSpec`
(:func:`build_fit_state`: the trained stages, the optimizer's groups, a
constraint's duals, the two seeded generators, the schedules bound to the
stages).

A :class:`FitState` reaches no executor, document or model: torch modules,
an optimizer, generators and numbers. It pickles with :mod:`pickle`, stages
included, and a copy steps on exactly as the original would.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import torch

from causalab.neural.shared.execution import Checkpoint
from causalab.neural.shared.featurizers import (
    Stage,
    build_recipe,
    link_budget_pools,
)
from causalab.neural.shared.training.schedules import (
    DUAL_GROUP,
    Control,
    Phase,
    build_controls,
    build_phases,
    parse_anneals,
)
from causalab.neural.shared.training.spec import FitSpec
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import (
    PER_PARAMS_OPTIMIZER_FIELDS,
    AnnealSchedule,
    ObjectiveTerm,
)

__all__ = [
    "FitState",
    "add_dual_groups",
    "build_fit_state",
    "build_optimizer",
    "build_stages",
]


@dataclasses.dataclass(eq=False)
class FitState:
    """Everything one fit's updates move, and where the fit stands.

    ``eq=False``: a state is compared and hashed by identity — two fits of
    one document are two fits."""

    stages: dict[str, Stage]
    trained_names: tuple[str, ...]
    optimizer: torch.optim.Optimizer
    #: the minibatch order's generator, and the one a ``hard_concrete`` gate's
    #: per-step mask draw comes from — this member's own, so a cohort cannot
    #: mix members' samples
    order_rng: torch.Generator
    mask_rng: torch.Generator
    step: int = 0
    epoch: int = 0
    #: the current epoch's minibatch order and the next position in it
    order: list[int] = dataclasses.field(default_factory=list)
    position: int = 0
    active: bool = True
    #: ``early_stop``: the best eval value so far, the passes since, and the
    #: stages and scores of the best pass — the fit ``early_stop`` selects is
    #: the one the outcome returns, so it is kept, not recovered
    best: float | None = None
    stale: int = 0
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_score: dict[str, float] | None = None
    eval_passes: int = 0
    last_score: dict[str, float] | None = None
    #: §2.11 ``control``: the live value of every named term's weight (the
    #: authored weight is a controller's start), the controllers bound to
    #: this fit's stages, and each controller's per-update trace
    live_weights: dict[str, float] = dataclasses.field(default_factory=dict)
    controls: dict[str, Control] = dataclasses.field(default_factory=dict)
    control_trace: dict[str, list[dict[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: §2.11 ``anneal``: the open-loop schedules, by what they move
    anneals: dict[str, AnnealSchedule] = dataclasses.field(default_factory=dict)
    #: §2.11 ``phases``: the windows in update terms, the optimizer group
    #: each ``train.params`` entry owns (its base ``lr`` / ``weight_decay``
    #: kept so a phase can restore them), and which window the fit is in —
    #: ``-1`` before the first update
    phases: tuple[Phase, ...] = ()
    groups_by_entry: dict[str, int] = dataclasses.field(default_factory=dict)
    phase_index: int = -1
    #: §2.11 ``constraint``: per constraint term, its dual pair ``(λ₁, λ₂)``
    #: — a parameter in the fit's optimizer under a ``maximize`` group — and
    #: the per-update trace of the duals after their ascent
    duals: dict[str, torch.nn.Parameter] = dataclasses.field(default_factory=dict)
    constraint_trace: dict[str, list[dict[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: §2.12 ``trajectory``: the photographs taken so far
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

    def exhausted(self, spec: FitSpec) -> bool:
        return self.step >= spec.total_steps or self.epoch >= spec.epochs


def build_stages(
    spec: FitSpec,
    *,
    device: str | torch.device = "cpu",
    load_tensors: Any = None,
    load_table: Any = None,
) -> dict[str, Stage]:
    """The fit's stages built from the spec alone, as an executor builds them
    (``ExecutorBase.stage``) — so a spec-built stage starts bit-identical to
    the one the point's executor would have built.

    The discipline that makes it so: ``torch.manual_seed(seed)`` first — the
    one deliberate use of the global RNG, which torch's ``orthogonal``
    parametrization completes a ``matrix_exp`` / ``stiefel`` base from —
    then the trained featurizers in ``train.params`` order, each through
    :func:`~causalab.neural.shared.featurizers.build_recipe` with the seed as
    its explicit init seed, and after each one every budget pool linked
    (:func:`~causalab.neural.shared.featurizers.link_budget_pools`), which
    builds a pool's other members in name order, then every other recipe the
    spec carries, in recipe order. The returned map holds every stage built,
    pool co-members included.

    ``load_tensors`` / ``load_table`` open the bundles a featurizer's
    ``init`` or ``file_path`` names; a spec that names none needs neither,
    and one that does is refused without them rather than built from
    nothing."""

    def no_loader(path: Any) -> Any:
        raise ProtocolError(
            "P4",
            f"a featurizer of this fit starts from a saved bundle ({path!r}); "
            "building its stages from the spec needs the loader that opens it "
            "(build_stages(load_tensors=))",
        )

    load_tensors = load_tensors or no_loader  # a missing table loader is
    # `build_stack`'s own refusal
    torch.manual_seed(spec.seed)
    recipes = {recipe.name: recipe for recipe in spec.recipes}
    cache: dict[str, Stage] = {}

    def build(name: str) -> Stage:
        if name not in cache:
            if name not in recipes:
                raise ProtocolError(
                    "P2", f"the fit's spec carries no recipe for featurizer {name!r}"
                )
            build_recipe(
                recipes[name],
                spec.featurizers,
                stage_cache=cache,
                load_tensors=load_tensors,
                load_table=load_table,
                device=device,
                seed=spec.seed,
                coords=spec.coords,
                model_info=spec.model_info,
            )
            link_budget_pools(spec.featurizers, cache, build)
        return cache[name]

    for pname in spec.params:
        build(pname.partition(".")[0])
    for recipe in spec.recipes:
        # what the spec carries beyond the trained stages and their pools —
        # an engine that builds every stage its forwards name from the spec
        # lists them after those, and they are built in that order
        build(recipe.name)
    return cache


def build_fit_state(
    spec: FitSpec,
    *,
    stages: Mapping[str, Stage] | None = None,
    device: str | torch.device | None = None,
    load_tensors: Any = None,
    load_table: Any = None,
) -> FitState:
    """One fit, ready to step.

    ``stages`` are the stages to train, by featurizer name — an engine hands
    over its point executor's own (``executors.seeded_stages``), so every
    forward it runs and the finish phase after the fit see the trained
    objects by identity. Without them the stages are built here from the
    spec (:func:`build_stages`, with its loaders). ``device`` is where a
    constraint's duals live (and spec-built stages): the stages' own device
    when unnamed."""
    if stages is None:
        stages = build_stages(
            spec,
            device=device or "cpu",
            load_tensors=load_tensors,
            load_table=load_table,
        )
    trained: dict[str, Stage] = {}
    parameters: list[torch.nn.Parameter] = []
    # one optimizer parameter group per `train.params` entry, so a per-parameter
    # `lr` / `weight_decay` (§2.11, a mapping keyed by those entries) lands on
    # exactly the tensors its entry names; with scalar settings the groups
    # share every hyperparameter and the optimizer's arithmetic is that of
    # one flat list (Adam's state is per tensor either way)
    groups: list[dict[str, Any]] = []
    for pname in spec.params:
        fname, _, slot = pname.partition(".")
        stage = stages[fname]
        trained[fname] = stage
        if slot:
            group_params = [stage.slot_params()[slot]]
        else:
            group_params = [p for p in stage.parameters() if p.requires_grad]
        parameters.extend(group_params)  # type: ignore[arg-type]
        group: dict[str, Any] = {"params": group_params}
        for field in PER_PARAMS_OPTIMIZER_FIELDS:
            value = spec.optimizer.get(field)
            if isinstance(value, Mapping):
                # the schema checked every entry of train.params is a key
                group[field] = float(value[pname])
        groups.append(group)
    if not parameters:
        raise ProtocolError("P2", "train.params resolved to no trainable tensors")
    if device is None:
        device = parameters[0].device

    optimizer = build_optimizer(spec.optimizer, groups)
    duals = add_dual_groups(spec.objective, optimizer, device)
    # §2.11 `control`: the closed-loop schedules, and the live value of every
    # named term's weight (the authored weight is a controller's start)
    live_weights: dict[str, float] = {
        term.name: float(term.weight)
        for term in spec.objective
        if term.name is not None and isinstance(term.weight, (int, float))
    }
    controls = build_controls(spec.control, trained, live_weights)
    return FitState(
        stages=trained,
        trained_names=spec.trained_names,
        optimizer=optimizer,
        order_rng=torch.Generator().manual_seed(spec.seed),
        # its own generator object at the same seed, as the batch order and
        # the subspace init have theirs: a local stream, never the global one
        mask_rng=torch.Generator().manual_seed(spec.seed),
        live_weights=live_weights,
        controls=controls,
        control_trace={target: [] for target in controls},
        anneals=parse_anneals(spec.anneal, trained, live_weights),
        phases=build_phases(spec.phases, spec.total_steps, trained, live_weights),
        groups_by_entry={pname: i for i, pname in enumerate(spec.params)},
        duals=duals,
        constraint_trace={name: [] for name in duals},
    )


def add_dual_groups(
    objective: Sequence[ObjectiveTerm],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
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
    duals: dict[str, torch.nn.Parameter] = {}
    for term in objective:
        if term.constraint is None:
            continue
        assert term.name is not None
        lam = torch.nn.Parameter(
            torch.tensor(list(term.constraint.init), dtype=torch.float32, device=device)
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
    (:func:`build_fit_state` writes the field on each group), so the default
    is never the value any tensor is stepped with."""
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
