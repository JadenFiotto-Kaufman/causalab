"""The update loop every training engine steps (:func:`fit_loop`), and what
it is made of: the ``early_stop`` bookkeeping (:func:`record_eval`), the
controllers and checkpoints after an update (:func:`read_signals`,
:func:`after_update`) and the outcome (:func:`finish`).

The loop owns everything that is not a forward, and touches nothing that
runs one: it advances :class:`~causalab.neural.shared.training.state.
FitState` objects against their :class:`~causalab.neural.shared.training.
spec.FitSpec` and calls back for the rest. A member is named to a callback
by its **index** into ``states``, so an engine keeps whatever it needs per
member — executors, graph pools, tallies — in a list of its own beside them.

* ``step(members)`` — run this update's grad forwards for those members and
  leave their gradients accumulated: reads reset, the forwards, each
  member's :func:`~causalab.neural.shared.training.objective.step_loss` over
  the minibatch its state stands at, the backward;
* ``evaluate(members)`` — one eval pass for each, returning its scores
  (:func:`~causalab.neural.shared.training.objective.score`), in order;
* ``on_epoch(members)`` — optional: those members begin a new epoch (not
  their first) with this update — where an engine redraws a §2.2 ``draw``.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch

from causalab.neural.shared.execution import TrainEvalScore, TrainOutcome
from causalab.neural.shared.featurizers import Gate
from causalab.neural.shared.training.control import ramp_setpoint
from causalab.neural.shared.training.diagnostics import (
    checkpoint,
    fit_diagnostics,
    restore,
    snapshot,
)
from causalab.neural.shared.training.schedules import (
    Control,
    advance_phase,
    apply_lr_schedule,
    set_anneal,
)
from causalab.neural.shared.training.spec import FitSpec
from causalab.neural.shared.training.state import FitState

__all__ = [
    "EpochFn",
    "EvalFn",
    "StepFn",
    "after_update",
    "begin_epoch",
    "begin_update",
    "finish",
    "fit_loop",
    "read_signals",
    "record_eval",
]

#: One optimizer step's grad forwards, for the members (indices into the
#: loop's ``states``) stepping now — their stages in train mode, schedules
#: set, gradients zeroed: reset the reads, run the forwards, build each
#: member's ``step_loss`` and backpropagate it. The loop steps the optimizers
#: afterwards.
StepFn = Callable[[Sequence[int]], None]

#: One eval pass for every member that owes one: its ``train.eval`` metrics
#: over its split, one score mapping per member, in order.
EvalFn = Callable[[Sequence[int]], Sequence[Mapping[str, float]]]

#: The members beginning a new epoch — not their first — with this update.
EpochFn = Callable[[Sequence[int]], None]


def fit_loop(
    states: Sequence[FitState],
    specs: Sequence[FitSpec],
    *,
    step: StepFn,
    evaluate: EvalFn,
    on_epoch: EpochFn | None = None,
) -> list[TrainOutcome]:
    """Step ``states`` in lockstep until each is spent or stopped; one
    outcome per member (:func:`finish`), in order.

    Per update and per active member: a new epoch draws its minibatch order
    from the member's own generator (``on_epoch`` hears of every epoch after
    the first); the stages go to train mode and a sampled gate draws its
    mask; phases and anneals are set for this update; the gradients are
    zeroed (:func:`begin_update`). ``step`` then runs the forwards and the
    backward for all of them. After it each member applies its lr schedule,
    steps its optimizer, projects its stages back onto their feasible sets,
    and lets its controllers and checkpoints observe the update; a member
    whose epoch ended on an eval boundary is handed to ``evaluate`` and its
    scores recorded (:func:`record_eval`)."""
    if len(states) != len(specs):
        raise ValueError(f"{len(states)} states but {len(specs)} specs — one each")
    while True:
        current = [i for i, state in enumerate(states) if state.active]
        if not current:
            break
        turning: list[int] = []
        for i in current:
            if begin_epoch(states[i], specs[i]):
                turning.append(i)
        if turning and on_epoch is not None:
            on_epoch(turning)
        for i in current:
            begin_update(states[i], specs[i])
        step(current)
        for i in current:
            state, spec = states[i], specs[i]
            apply_lr_schedule(state, spec)
            state.optimizer.step()
            for stage in state.stages.values():
                stage.project()  # back onto the feasible set (a clamp gate)
            state.step += 1
            state.position += 1
        # the members' controllers observe their gates in one host read,
        # after every member has stepped (the members are independent, so
        # the order of stepping and observing across them changes nothing)
        signals = read_signals([states[i] for i in current])
        due: list[int] = []
        for i in current:
            state, spec = states[i], specs[i]
            after_update(state, spec, signals)
            if state.position >= len(state.order) or state.step >= spec.total_steps:
                # an epoch ended — complete, or cut short by an `updates`
                # budget; either way the eval it owes runs now
                state.epoch += 1
                if (
                    spec.eval_every_epochs is not None
                    and state.epoch % spec.eval_every_epochs == 0
                ):
                    due.append(i)
        if due:
            scores = evaluate(due)
            if len(scores) != len(due):
                raise ValueError(
                    f"the eval pass returned {len(scores)} scores for {len(due)} fits"
                )
            for i, member_scores in zip(due, scores):
                record_eval(states[i], specs[i], dict(member_scores))
        for i in current:
            if states[i].exhausted(specs[i]):
                states[i].active = False
    return [finish(state, spec) for state, spec in zip(states, specs)]


def begin_epoch(state: FitState, spec: FitSpec) -> bool:
    """Draw a new epoch's minibatch order when the last one is spent; whether
    this is a *later* epoch's start — the first epoch's is where the fit was
    prepared, so nothing about it is new."""
    if state.position < len(state.order):
        return False
    state.order = torch.randperm(len(spec.batches), generator=state.order_rng).tolist()
    state.position = 0
    return state.step > 0


def begin_update(state: FitState, spec: FitSpec) -> None:
    """Set one member up for the update it is about to take: train mode, a
    sampled gate's one draw for the step, the phase and the schedules'
    values at this update, the gradients zeroed."""
    pools_drawn: set[int] = set()
    for stage in state.stages.values():
        stage.train(True)
        if isinstance(stage, Gate) and stage.samples_per_step:
            # One draw per step (concrete noise or budget k), shared
            # by the read and write, from this member's generator.
            # A budget pool draws once for all its gates (§2.5).
            if stage.pool is not None:
                if id(stage.pool) not in pools_drawn:
                    stage.pool.resample(state.mask_rng)
                    pools_drawn.add(id(stage.pool))
                continue
            stage.resample(state.mask_rng)
    advance_phase(state)
    for dotted, schedule in state.anneals.items():
        set_anneal(
            state, dotted, schedule, step=state.step, total_steps=spec.total_steps
        )
    if state.phase_index >= 0:
        phase = state.phases[state.phase_index]
        for dotted, schedule in phase.anneals.items():
            # a phase's schedule runs over the phase's own steps
            set_anneal(
                state,
                dotted,
                schedule,
                step=state.step - phase.start,
                total_steps=phase.end - phase.start,
            )
    state.optimizer.zero_grad()


def record_eval(state: FitState, spec: FitSpec, scores: dict[str, float]) -> None:
    """One eval pass's scores onto the fit, and its ``early_stop``
    bookkeeping (§2.11): an improvement snapshots the stages — the fit
    ``early_stop`` selects is the one :func:`finish` restores — and more than
    ``patience`` passes without one stop the fit."""
    state.eval_passes += 1
    state.last_score = scores
    if spec.early_stop is None:
        return
    value = scores[spec.early_stop.metric]
    mode = spec.early_stop.mode
    improved = (
        state.best is None
        or (mode == "max" and value > state.best)
        or (mode == "min" and value < state.best)
    )
    if improved:
        state.best, state.stale = value, 0
        state.best_state = snapshot(state.stages)
        state.best_score = dict(scores)
    else:
        state.stale += 1
        if state.stale > spec.early_stop.patience:
            state.active = False


def finish(state: FitState, spec: FitSpec) -> TrainOutcome:
    """The fit's outcome. With ``train.early_stop`` the stages are restored to
    the best-scoring snapshot first, so the returned stages — and the
    ``eval_score.metrics`` beside them — are the selected fit's
    (``selected: "early_stop.best"``); without it they are the last ones.

    What only the engine that ran the forwards knows — the store's tallies
    (``fit_forwards``, ``resumed``), the row bound, a drawn role's members —
    it writes onto the outcome itself."""
    selected = "last"
    last_score = state.last_score
    if state.best_state is not None:
        restore(state.stages, state.best_state)
        last_score = state.best_score
        selected = "early_stop.best"
    for stage in state.stages.values():
        stage.eval()
        if isinstance(stage, Gate):
            # a pinned mask is a phase's, not the bundle's: the saved gate is
            # θ, and an apply reads its own hard split from it (§2.5)
            stage.frozen_mask = None
    eval_score = None
    if spec.eval_split is not None and last_score is not None:
        eval_score = TrainEvalScore(
            split=spec.eval_split,
            metrics=dict(last_score),
            passes=state.eval_passes,
            featurizers=state.trained_names,
            selected=selected,
        )
    trained = {name: state.stages[name] for name in state.trained_names}
    constraints = spec.constraints
    return TrainOutcome(
        stages=trained,
        eval_score=eval_score,
        diagnostics=fit_diagnostics(trained),
        controls={
            target: {
                "initial": state.controls[target].initial,
                "final": state.controls[target].controller.value,
                "signal_final": trace[-1]["signal"] if trace else float("nan"),
                "setpoint_final": trace[-1]["setpoint"] if trace else float("nan"),
                "updates": float(len(trace)),
            }
            for target, trace in state.control_trace.items()
        },
        control_trace=state.control_trace,
        constraints={
            name: {
                "target": constraints[name].target,
                # the duals the first update stepped with are the authored
                # init by construction: nothing steps them before it
                "lambda1_initial": constraints[name].init[0],
                "lambda2_initial": constraints[name].init[1],
                "lambda1_final": float(state.duals[name][0].detach()),
                "lambda2_final": float(state.duals[name][1].detach()),
                "value_final": trace[-1]["value"] if trace else float("nan"),
                "updates": float(len(trace)),
            }
            for name, trace in state.constraint_trace.items()
        },
        constraint_trace=state.constraint_trace,
        anneals={
            target: {
                "start": schedule.start,
                "end": schedule.end,
                "final": schedule.value_at(state.step, spec.total_steps),
                "shape": schedule.shape,
            }
            for target, schedule in state.anneals.items()
        },
        phases=tuple(
            {
                "start": phase.start,
                "end": phase.end,
                "params": list(phase.params),
                "freeze_masks": list(phase.freeze_masks),
            }
            for phase in state.phases
        ),
        checkpoints=tuple(state.checkpoints),
    )


def read_signals(fits: Sequence[FitState]) -> dict[tuple[int, str], float]:
    """Every controller's signal after this update, for every member of the
    step, keyed by ``(id(fit), target)`` — the gates' kept-unit counts
    brought to the host together, one read for the step rather than one per
    gate per controller per member (``stack`` promotes to the widest dtype
    among them, and a float widened is the same number)."""
    pending: list[tuple[int, str, Control, list[tuple[torch.Tensor, int]]]] = []
    tensors: list[torch.Tensor] = []
    for fit in fits:
        for target, control in fit.controls.items():
            counts = control.signal_counts()
            pending.append((id(fit), target, control, counts))
            tensors.extend(count for count, _ in counts)
    if not tensors:
        return {}
    device = tensors[0].device
    values = iter(torch.stack([t.reshape(()).to(device) for t in tensors]).tolist())
    return {
        (key, target): control.read_signal(
            [float(next(values)) for _ in counts], [units for _, units in counts]
        )
        for key, target, control, counts in pending
    }


def after_update(
    fit: FitState, spec: FitSpec, signals: Mapping[tuple[int, str], float]
) -> None:
    """What follows one member's optimizer step (``fit.step`` already counts
    it): every controller observes the fit — its signal read by
    :func:`read_signals` — and moves its target for the *next* update
    (§2.11), and a scheduled ``trajectory`` checkpoint (§2.12) photographs
    the slots — so a checkpoint's ``weight.<term>`` is the weight the update
    used and its ``control.<target>`` the value set after it."""
    for target, control in fit.controls.items():
        signal = signals[(id(fit), target)]
        setpoint = ramp_setpoint(*control.ramp, fit.step, spec.total_steps)
        value = control.controller.step(signal, setpoint)
        control.apply(value, fit.stages, fit.live_weights)
        fit.control_trace[target].append(
            {
                "step": float(fit.step),
                "value": value,
                "signal": signal,
                "setpoint": setpoint,
            }
        )
    constraints = spec.constraints
    for name, lam in fit.duals.items():
        # the density this update saw and where the ascent left the duals
        # for the next one (the duals it stepped *with* are the previous
        # row's, or the authored init on the first)
        fit.constraint_trace[name].append(
            {
                "step": float(fit.step),
                "value": fit.term_values.get(f"term.{name}", float("nan")),
                "target": constraints[name].target,
                "lambda1": float(lam[0].detach()),
                "lambda2": float(lam[1].detach()),
            }
        )
    if fit.step in spec.checkpoint_steps:
        loss, term_values = fit.loss_record()
        fit.checkpoints.append(
            checkpoint(
                fit.step,
                fit.epoch,
                {name: fit.stages[name] for name in fit.trained_names},
                loss=loss,
                term_values=term_values,
                controls={
                    target: fit.control_trace[target][-1]["value"]
                    for target in fit.controls
                },
                phase=fit.phase_index if fit.phases else None,
            )
        )
