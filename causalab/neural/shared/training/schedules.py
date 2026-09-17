"""A fit's schedules (spec §2.11), each bound to what it moves: ``control``
(closed-loop, :mod:`.control`'s PID over a gate's kept-unit count),
``anneal`` (open-loop ramps), ``phases`` (windows of the run with their own
params, rates, schedules and pinned masks) and ``optimizer.schedule`` (the
linear warm-up and decay of the learning rate)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from causalab.neural.shared.featurizers import Gate, Stage
from causalab.neural.shared.training.control import PidController, build_controller
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import (
    OBJECTIVE_WEIGHT_PREFIX,
    PER_PARAMS_OPTIMIZER_FIELDS,
    AnnealSchedule,
    PhaseSpec,
)

if TYPE_CHECKING:
    from causalab.neural.shared.training.fit import Fit

__all__ = [
    "DUAL_GROUP",
    "Control",
    "Phase",
    "advance_phase",
    "apply_lr_schedule",
    "build_controls",
    "build_phases",
    "lr_factor",
    "parse_anneals",
    "set_anneal",
]


@dataclasses.dataclass
class Control:
    """One §2.11 ``control`` entry, bound to the loop: the controller, the
    gate or gates whose kept-unit count it observes (a list of gates is one
    signal — their counts summed — as a list-valued ``l1`` is one penalty),
    and how the controlled value is written back — into a named term's live
    weight, or onto a stage attribute the way an anneal writes one."""

    controller: PidController
    ramp: tuple[float, float, float]
    initial: float
    signal_stages: Sequence[Stage]
    term: str | None = None
    hyper: tuple[str, str] | None = None  # (featurizer, attribute)
    signal: str = "hard_mask_size"

    def signal_counts(self) -> list[tuple[torch.Tensor, int]]:
        """Per observed gate, its kept-unit count through the hard mask as
        the device scalar it is, and its unit count — what
        :func:`~causalab.neural.shared.training.loop.read_signals` brings to the host for every controller of a
        step in one read."""
        counts: list[tuple[torch.Tensor, int]] = []
        for stage in self.signal_stages:
            assert isinstance(stage, Gate)
            hard = stage.hard_mask()
            counts.append((hard.sum(), int(hard.numel())))
        return counts

    def read_signal(self, kept_counts: Sequence[float], units: Sequence[int]) -> float:
        # kept-unit counts through the hard mask, summed over the named gates
        # (`hard_mask_size`), or that sum over the gates' total unit count
        # (`hard_mask_fraction`) — CONTROL_SIGNALS; ``kept_counts`` are the
        # gates' counts as floats, in :meth:`signal_counts` order
        kept = 0.0
        for count in kept_counts:
            kept += count
        total = sum(units)
        if self.signal == "hard_mask_fraction":
            return kept / total if total else 0.0
        return kept

    def apply(
        self, value: float, stages: Mapping[str, Stage], live_weights: dict[str, float]
    ) -> None:
        if self.term is not None:
            live_weights[self.term] = value
        else:
            assert self.hyper is not None
            setattr(stages[self.hyper[0]], self.hyper[1], value)


def build_controls(
    control: Mapping[str, Mapping[str, Any]] | None,
    stages: Mapping[str, Stage],
    live_weights: Mapping[str, float],
) -> dict[str, Control]:
    """The loop's controllers, one per ``train.control`` target (§2.11).
    Validation has already resolved every target and signal; what is decided
    here is the binding — which live weight or which attribute the value
    lands on — and the initial value it starts from."""
    if not control:
        return {}
    out: dict[str, Control] = {}
    for target, spec in control.items():
        ((signal, signal_target),) = spec["signal"].items()
        names = (
            [signal_target] if isinstance(signal_target, str) else list(signal_target)
        )
        signal_stages = [stages[str(name)] for name in names]
        ramp = tuple(float(v) for v in spec["setpoint"]["ramp"])
        if target.startswith(OBJECTIVE_WEIGHT_PREFIX):
            name = target[len(OBJECTIVE_WEIGHT_PREFIX) :].rpartition(".")[0]
            initial = live_weights[name]
            out[target] = Control(
                controller=build_controller(spec, initial=initial),
                ramp=ramp,  # type: ignore[arg-type]
                initial=initial,
                signal_stages=signal_stages,
                term=name,
                signal=str(signal),
            )
        else:
            fname, _, tail = target.partition(".")
            hyper = tail.rsplit(".", 1)[-1]
            stage = stages[fname]
            if not hasattr(stage, hyper):
                raise ProtocolError("P2", f"{fname!r} has no controllable {hyper!r}")
            initial = float(getattr(stage, hyper))
            out[target] = Control(
                controller=build_controller(spec, initial=initial),
                ramp=ramp,  # type: ignore[arg-type]
                initial=initial,
                signal_stages=signal_stages,
                hyper=(fname, hyper),
                signal=str(signal),
            )
    return out


def lr_factor(step: int, total_steps: int, warmup_frac: float) -> float:
    """HF's ``get_linear_schedule_with_warmup`` at update ``step`` (0-based):
    ``step / warmup`` while warming up, then ``(total − step) / (total − warmup)``
    down to 0 at the last update. ``warmup = warmup_frac · total`` (a float, as
    HF takes it). The first update runs at lr 0 when there is any warm-up, as
    in the reference implementation."""
    warmup = warmup_frac * total_steps
    if step < warmup:
        return step / max(1.0, warmup)
    return max(0.0, (total_steps - step) / max(1.0, total_steps - warmup))


#: The lr a param group was built with, kept beside the group so the schedule
#: multiplies the authored value and never its own previous output.
_BASE_LR = "_schedule_base_lr"


def apply_lr_schedule(fit: "Fit") -> None:
    """Set every group's lr for this update under §2.11 ``optimizer.schedule``:
    a no-op under ``constant``. Per-entry lrs are each scaled by the same factor.
    Refused beside ``phases`` at load (rule 4), so no other writer of ``lr``
    runs in the same fit."""
    spec = fit.doc.train.optimizer if fit.doc.train is not None else {}
    schedule = str(spec.get("schedule", "constant"))
    if schedule == "constant":
        return
    warmup_frac = float(spec.get("warmup_frac", 0.1))
    factor = lr_factor(fit.step, fit.total_steps, warmup_frac)
    for group in fit.optimizer.param_groups:
        if DUAL_GROUP in group:
            continue  # a constraint's duals ascend at their own authored rate
        if _BASE_LR not in group:
            group[_BASE_LR] = group["lr"]
        group["lr"] = group[_BASE_LR] * factor


#: The key marking a constraint term's dual-pair parameter group (§2.11), by
#: the term's name — what the lr schedule and the phase machinery skip.
DUAL_GROUP = "_constraint_duals_of"


@dataclasses.dataclass(frozen=True)
class Phase:
    """One §2.11 ``phases`` entry in update terms: the half-open window
    ``[start, end)``, the ``train.params`` entries that step in it, the
    per-entry ``lr`` / ``weight_decay`` it overrides (absent = the top-level
    group's), its own schedules, and the gates whose hard mask it pins."""

    start: int
    end: int
    params: frozenset[str]
    optimizer: Mapping[str, Mapping[str, float]]
    anneals: dict[str, AnnealSchedule]
    freeze_masks: tuple[str, ...]


def build_phases(
    phases: Sequence[PhaseSpec] | None,
    total_steps: int,
    stages: Mapping[str, Stage],
    live_weights: Mapping[str, float],
) -> tuple[Phase, ...]:
    """Resolve ``train.phases`` against this fit's update count (§2.11). A
    ``frac`` end is ``round(frac · total_steps)``, so consecutive phases meet
    without a gap; an ``updates`` end is taken as written and the last one
    must be the run's length — the parser could not know it, so a document
    whose phases do not partition the run is refused here, before any step.
    A phase that resolves to zero updates (a frac too small for the run) is
    refused rather than silently skipped: its params would never train."""
    if not phases:
        return ()
    out: list[Phase] = []
    start = 0
    for i, phase in enumerate(phases):
        ((unit, value),) = phase.until.items()
        end = int(round(float(value) * total_steps)) if unit == "frac" else int(value)
        if i == len(phases) - 1 and unit == "frac":
            end = total_steps  # rounding never leaves the tail unowned
        if end <= start:
            raise ProtocolError(
                "P2",
                f"train.phases[{i}] spans no update: it would end at {end} of "
                f"{total_steps} after the previous phase ended at {start}",
            )
        if end > total_steps:
            raise ProtocolError(
                "P2",
                f"train.phases[{i}] ends at update {end}, past the run's {total_steps}",
            )
        if i == len(phases) - 1 and end != total_steps:
            raise ProtocolError(
                "P2",
                f"train.phases partition the run: the last phase ends at update "
                f"{end}, the run at {total_steps}",
            )
        optimizer: dict[str, dict[str, float]] = {}
        for field, setting in (phase.optimizer or {}).items():
            if isinstance(setting, Mapping):
                optimizer[field] = {k: float(v) for k, v in setting.items()}
            else:
                optimizer[field] = {p: float(setting) for p in phase.params}
        out.append(
            Phase(
                start=start,
                end=end,
                params=frozenset(phase.params),
                optimizer=optimizer,
                anneals=parse_anneals(phase.anneal, stages, live_weights),
                freeze_masks=tuple(phase.freeze_masks),
            )
        )
        start = end
    return tuple(out)


def advance_phase(fit: "Fit") -> None:
    """Before an update: enter the phase that owns ``fit.step``, if the fit is
    not in it yet. Entering sets, per ``train.params`` entry, whether its
    tensors take gradients and its optimizer group's ``lr`` /
    ``weight_decay`` — the phase's override, the top-level value, or ``0``
    for an entry the phase leaves out (the group stays, so Adam's moments
    survive the boundary and a later phase resumes rather than restarts) —
    and pins each ``freeze_masks`` gate's hard mask as of this step, clearing
    any pin a previous phase set on a gate this one does not name."""
    if not fit.phases:
        return
    index = fit.phase_index
    while index + 1 < len(fit.phases) and fit.step >= fit.phases[index + 1].start:
        index += 1
    if index == fit.phase_index:
        return
    fit.phase_index = index
    phase = fit.phases[index]
    for entry, gi in fit.groups_by_entry.items():
        group = fit.optimizer.param_groups[gi]
        active = entry in phase.params
        for param in group["params"]:
            param.requires_grad_(active)
        for field in PER_PARAMS_OPTIMIZER_FIELDS:
            base_key = f"_phase_base_{field}"
            if base_key not in group:
                group[base_key] = group.get(field, 0.0)  # the top-level value
            if not active:
                group[field] = 0.0
            else:
                group[field] = phase.optimizer.get(field, {}).get(
                    entry, group[base_key]
                )
    pinned = set(phase.freeze_masks)
    for name, stage in fit.stages.items():
        if not isinstance(stage, Gate):
            continue
        if name in pinned:
            with torch.no_grad():
                stage.frozen_mask = stage.hard_mask().detach().clone()
        else:
            stage.frozen_mask = None


def parse_anneals(
    anneal: Mapping[str, AnnealSchedule] | None,
    stages: Mapping[str, Stage],
    live_weights: Mapping[str, float],
) -> dict[str, AnnealSchedule]:
    """The loop's open-loop schedules (§2.11), each bound to what it moves: a
    trained featurizer's hyperparameter, or a named objective term's live
    weight — the same dict a controller writes and
    :func:`~causalab.neural.shared.training.loop.step_loss` reads, so an
    annealed weight and a controlled one move through one path. Validation
    resolved both target forms at load; this re-checks for a document that
    arrived unvalidated."""
    if anneal is None:
        return {}
    out: dict[str, AnnealSchedule] = {}
    for dotted, schedule in anneal.items():
        if dotted.startswith(OBJECTIVE_WEIGHT_PREFIX):
            name = dotted[len(OBJECTIVE_WEIGHT_PREFIX) :].rpartition(".")[0]
            if name not in live_weights:
                raise ProtocolError(
                    "P2",
                    f"anneal target {dotted!r} is not a named objective term's weight",
                )
        elif dotted.split(".", 1)[0] not in stages:
            raise ProtocolError("P2", f"anneal target {dotted!r} is not being trained")
        if not isinstance(schedule, AnnealSchedule):
            raise ProtocolError(
                "P2", f"anneal schedule for {dotted!r} is [start, end, frac]"
            )
        out[dotted] = schedule
    return out


def set_anneal(
    fit: "Fit",
    dotted: str,
    schedule: AnnealSchedule,
    *,
    step: int | None = None,
    total_steps: int | None = None,
) -> None:
    """Write this update's scheduled value where ``dotted`` points: into the
    named term's live weight, or onto the stage attribute the path names.
    ``step`` / ``total_steps`` default to the run's; a phase passes its own
    window so its schedule spans the phase (§2.11)."""
    value = schedule.value_at(
        fit.step if step is None else step,
        fit.total_steps if total_steps is None else total_steps,
    )
    if dotted.startswith(OBJECTIVE_WEIGHT_PREFIX):
        # `step_loss` reads the live weight of a named term (§2.11), so the
        # schedule lands there and the checkpoint's `weight.<name>` is the
        # value the update used
        name = dotted[len(OBJECTIVE_WEIGHT_PREFIX) :].rpartition(".")[0]
        fit.live_weights[name] = value
        return
    stages = fit.stages
    fname, _, tail = dotted.partition(".")
    hyper = tail.rsplit(".", 1)[-1]
    stage = stages[fname]
    if not hasattr(stage, hyper):
        raise ProtocolError("P2", f"{fname!r} has no annealable {hyper!r}")
    if (
        isinstance(stage, Gate)
        and stage.parametrization in ("clamp", "budget")
        and hyper == "temperature"
    ):
        # validation refuses this at load (rule 4); a document that reached
        # the loop unvalidated must not have its schedule silently ignored
        raise ProtocolError(
            "P2",
            f"{fname!r} is a {stage.parametrization} gate: its mask has no "
            "temperature to anneal — θ itself under clamp, σ(θ + c_k) with the "
            "shift solved per step under budget (§2.5)",
        )
    setattr(stage, hyper, value)
