"""The update loop every training engine steps (:func:`fit_loop`), and what
it is made of: the step's objective (:func:`step_loss`), the eval pass
(:func:`score`, :func:`evaluate_fits`), the ``early_stop`` bookkeeping
(:func:`record_eval`), the controllers and checkpoints after an update
(:func:`read_signals`, :func:`after_update`) and the outcome
(:func:`finish`).

The loop owns everything that is not a forward. An engine plugs in two
things: ``step_forward`` — run this step's grad forwards for the members
handed to it and leave their gradients accumulated — and ``evaluate`` — run
an eval pass for the fits that owe one and :func:`record_eval` each score.
:func:`evaluate_fits` is the eval of an engine with nothing to add: one pass
per fit on an executor its :class:`~causalab.neural.shared.training.fit.
ExecutorFactory` built."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch

from causalab.neural.shared.execution import TrainEvalScore, TrainOutcome
from causalab.neural.shared.executor_base import ExecutorBase
from causalab.neural.shared.featurizers import Gate, featurizer_cache
from causalab.neural.shared.metrics import (
    GATHERED_KINDS,
    compute_metric,
    gathered_metric,
    metric_token_ids,
)
from causalab.neural.shared.training.control import ramp_setpoint
from causalab.neural.shared.training.diagnostics import (
    checkpoint,
    fit_diagnostics,
    restore,
    snapshot,
)
from causalab.neural.shared.training.fit import Fit
from causalab.neural.shared.training.objective import metric_tensor, regularizer
from causalab.neural.shared.training.schedules import (
    Control,
    advance_phase,
    apply_lr_schedule,
    set_anneal,
)
from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.schema import (
    READ_TARGET_METRIC_KINDS,
    Document,
    concrete_int,
    concrete_str,
    metric_reads_vocabulary,
)

__all__ = [
    "Evaluate",
    "StepForward",
    "after_update",
    "eval_reads",
    "evaluate_fits",
    "device_scored",
    "finish",
    "fit_loop",
    "fresh_for_eval",
    "read_signals",
    "record_eval",
    "score",
    "step_loss",
]

#: One optimizer step's grad forwards: for every ``(fit, minibatch executor)``
#: stepping now — reads reset, gradients zeroed, stages in train mode — run
#: the forwards, build each member's :func:`step_loss` and backpropagate it.
#: The loop steps the optimizers afterwards.
StepForward = Callable[[Sequence[tuple[Fit, ExecutorBase]]], None]

#: One eval pass for every fit that owes one: score each on its
#: ``train.eval`` split and hand the scores to :func:`record_eval`.
Evaluate = Callable[[Sequence[Fit]], None]


def fit_loop(
    fits: Sequence[Fit], *, step_forward: StepForward, evaluate: Evaluate
) -> None:
    """Step ``fits`` in lockstep until each is spent or stopped.

    Per step and per active fit: a new epoch draws its minibatch order from
    the fit's own generator (and its drawn roles anew, §2.2); the stages go
    to train mode and a sampled gate draws its mask; phases and anneals are
    set for this update; the minibatch's reads are reset and the gradients
    zeroed. ``step_forward`` then runs the forwards and the backward for all
    of them. After it each fit applies its lr schedule, steps its optimizer,
    projects its stages back onto their feasible sets, and lets its
    controllers and checkpoints observe the update; a fit whose epoch ended
    on an eval boundary is handed to ``evaluate``."""
    while True:
        stepping = [fit for fit in fits if fit.active]
        if not stepping:
            break
        current: list[tuple[Fit, ExecutorBase]] = []
        for fit in stepping:
            if fit.position >= len(fit.order):
                fit.order = torch.randperm(
                    len(fit.batches), generator=fit.order_rng
                ).tolist()
                fit.position = 0
                if fit.drawn is not None and fit.step > 0:
                    # §2.2 `draw`: a new member per row for the new epoch
                    # (the first epoch's draw was taken at prepare)
                    fit.drawn.redraw(fit)
            pools_drawn: set[int] = set()
            for stage in fit.stages.values():
                stage.train(True)
                if isinstance(stage, Gate) and stage.samples_per_step:
                    # One draw per step (concrete noise or budget k), shared
                    # by the read and write, from this member's generator.
                    # A budget pool draws once for all its gates (§2.5).
                    if stage.pool is not None:
                        if id(stage.pool) not in pools_drawn:
                            stage.pool.resample(fit.mask_rng)
                            pools_drawn.add(id(stage.pool))
                        continue
                    stage.resample(fit.mask_rng)
            advance_phase(fit)
            for dotted, schedule in fit.anneals.items():
                set_anneal(fit, dotted, schedule)
            if fit.phase_index >= 0:
                phase = fit.phases[fit.phase_index]
                for dotted, schedule in phase.anneals.items():
                    # a phase's schedule runs over the phase's own steps
                    set_anneal(
                        fit,
                        dotted,
                        schedule,
                        step=fit.step - phase.start,
                        total_steps=phase.end - phase.start,
                    )
            minibatch = fit.minibatch_executors[fit.order[fit.position]]
            minibatch.reset_reads()
            fit.optimizer.zero_grad()
            current.append((fit, minibatch))
        step_forward(current)
        due: list[Fit] = []
        for fit, _ in current:
            apply_lr_schedule(fit)
            fit.optimizer.step()
            for stage in fit.stages.values():
                stage.project()  # back onto the feasible set (a clamp gate)
            fit.step += 1
            fit.position += 1
        # the members' controllers observe their gates in one host read,
        # after every member has stepped (the members are independent, so
        # the order of stepping and observing across them changes nothing)
        signals = read_signals([fit for fit, _ in current])
        for fit, _ in current:
            after_update(fit, signals)
            if fit.position >= len(fit.order) or fit.step >= fit.total_steps:
                # an epoch ended — complete, or cut short by an `updates`
                # budget; either way the eval it owes runs now
                fit.epoch += 1
                if (
                    fit.eval_every_epochs is not None
                    and fit.epoch % fit.eval_every_epochs == 0
                ):
                    due.append(fit)
        if due:
            evaluate(due)
        for fit, _ in current:
            if fit.exhausted:
                fit.active = False


def step_loss(fit: Fit, minibatch: ExecutorBase) -> torch.Tensor:
    """This update's objective for one member, and — on the member — the
    record of it (``last_loss``, ``term_values``) a checkpoint taken after
    the update carries. A named term's weight is its *live* value: the
    authored one until a controller moves it (§2.11)."""
    loss = torch.zeros(())
    term_values: dict[str, torch.Tensor | float] = {}
    for index, term in enumerate(fit.train.objective):
        w = float(term.weight) if isinstance(term.weight, (int, float)) else 1.0
        if term.name is not None:
            w = fit.live_weights.get(term.name, w)  # a controlled weight moves
        if term.metric is not None:
            metric = fit.doc.metrics[term.metric]
            of_value = minibatch.dense_value(str(metric.of))
            target_value = (
                minibatch.dense_value(str(metric.fields["target"]))
                if metric.kind in READ_TARGET_METRIC_KINDS
                else None
            )
            value = metric_tensor(
                metric,
                of_value,
                minibatch.rows_for_metrics(),
                minibatch.bundle.tokenizer,
                target_value=target_value,
            ).mean()
        else:
            assert term.regularizer is not None
            kind, targets = term.regularizer
            value = regularizer(
                kind, targets, fit.stages, term.reduce or "mean", term.costs
            )
        if term.constraint is not None:
            # §2.11 `constraint`: λ₁(s − t) + λ₂(s − t)², the duals ascended by
            # their own optimizer group — the term has no weight
            assert term.name is not None
            lam = fit.duals[term.name]
            gap = value - term.constraint.target
            loss = loss + lam[0] * gap + lam[1] * gap * gap
            term_values[f"term.{term.name}"] = float(value.detach())
            term_values[f"lambda1.{term.name}"] = float(lam[0].detach())
            term_values[f"lambda2.{term.name}"] = float(lam[1].detach())
            continue
        loss = loss + w * value
        term_values[f"term.{term.name or index}"] = value.detach()
        term_values[f"weight.{term.name or index}"] = w
    fit.last_loss = loss.detach()
    fit.term_values = term_values
    return loss


def record_eval(fit: Fit, scores: dict[str, float]) -> None:
    """One eval pass's scores onto the fit, and its ``early_stop``
    bookkeeping (§2.11): an improvement snapshots the stages — the fit
    ``early_stop`` selects is the one :func:`finish` restores — and more than
    ``patience`` passes without one stop the fit."""
    fit.eval_passes += 1
    fit.last_score = scores
    if fit.train.early_stop is None:
        return
    metric_name = str(fit.train.early_stop["metric"])
    mode = str(fit.train.early_stop["mode"])
    value = scores[metric_name]
    improved = (
        fit.best is None
        or (mode == "max" and value > fit.best)
        or (mode == "min" and value < fit.best)
    )
    if improved:
        fit.best, fit.stale = value, 0
        fit.best_state = snapshot(fit.stages)
        fit.best_score = dict(scores)
    else:
        fit.stale += 1
        if fit.stale > concrete_int(fit.train.early_stop["patience"], "patience"):
            fit.active = False


def fresh_for_eval(eval_executor: ExecutorBase) -> None:
    for stage in eval_executor.stage_cache.values():
        stage.eval()
    # the stages moved since the last pass, so every read is stale; the
    # encoded split and the interned constant captures are not (reset_reads)
    eval_executor.reset_reads()


def score(doc: Document, eval_executor: ExecutorBase) -> dict[str, float]:
    """The declared eval metrics over ``eval_executor``'s reads — run on its
    own path unless a batched pass or a replay already filled them.

    A kind that only selects entries of the projection at the answer ids
    (``metrics.GATHERED_KINDS`` — the presets' ``iia``) gathers them where
    the read sits — the device, on an executor whose ``device_reads`` is on
    — and copies the one or two columns, never the vocabulary, with the ids
    resolved once per executor (its rows never change); every other kind
    reduces a CPU copy of the whole value in float, as it always did, one
    copy per read however many metrics read it. Either way the numbers are
    the ones the whole-vocabulary CPU path computes, to the bit
    (``metrics.gathered_metric``). A replay's values are graph-owned
    storage: they are consumed here and released by the caller before the
    next replay."""
    assert doc.train is not None and doc.train.eval is not None
    scores: dict[str, float] = {}
    rows = eval_executor.rows_for_metrics()
    tokenizer = eval_executor.bundle.tokenizer
    host: dict[str, torch.Tensor] = {}

    def on_host(read: str) -> torch.Tensor:
        if read not in host:
            host[read] = eval_executor.dense_value(read).detach().cpu()
        return host[read]

    for name in doc.train.eval["metrics"]:
        metric = doc.metrics[name]
        if str(metric.kind) in GATHERED_KINDS:
            ids = eval_executor.metric_token_ids.get(name)
            if ids is None:
                ids = metric_token_ids(metric, rows, tokenizer)
                eval_executor.metric_token_ids[name] = ids
            values = gathered_metric(
                metric,
                eval_executor.dense_value(str(metric.of)),
                rows,
                tokenizer,
                token_ids=ids,
            )
        else:
            values = compute_metric(
                metric,
                on_host(str(metric.of)),
                rows,
                tokenizer,
                target_value=on_host(str(metric.fields["target"]))
                if metric.kind in READ_TARGET_METRIC_KINDS
                else None,
                vocab_axis=metric_reads_vocabulary(doc, metric),
            )
        numeric = [v for v in values if isinstance(v, (int, float))]
        scores[name] = sum(numeric) / len(numeric) if numeric else 0.0
    return scores


def device_scored(doc: Document) -> bool:
    """Whether a fit's eval executor keeps its reads on the device
    (``ExecutorBase.device_reads``): some eval metric selects from them there,
    and :func:`score` gathers the answer columns and copies those, not the
    vocabulary. A softmax-class eval keeps the one host copy
    ``_finalize_read`` makes."""
    train = doc.train
    eval_metrics = (
        train.eval["metrics"] if train is not None and train.eval is not None else ()
    )
    return any(str(doc.metrics[name].kind) in GATHERED_KINDS for name in eval_metrics)


def evaluate_fits(due: Sequence[Fit], request: ExecutionRequest) -> None:
    """One eval pass per fit in ``due`` (§2.11), each on its own executor over
    its ``train.eval`` split — built by the fit's factory on the first pass
    and kept (``Fit.eval_executor``), so the split is read and encoded once —
    in eval mode under one :func:`featurizer_cache` scope, then the fit's
    :func:`record_eval`. Built lazily rather than before the loop: an
    ``updates`` budget shorter than one epoch never evaluates, and the eval
    split is only guaranteed readable through the eval path."""
    for fit in due:
        assert fit.train.eval is not None
        eval_executor = fit.eval_executor
        if eval_executor is None:
            split = concrete_str(fit.train.eval["split"], "train.eval.split")
            split_rows = request.env.datasets.rows(split)
            eval_executor = fit.executor_factory(
                fit.doc,
                fit.executor,
                role_rows={role: split_rows for role in fit.executor.role_rows},
                grad_enabled=False,
                rows=split,
            )
            eval_executor.device_reads = device_scored(fit.doc)
            fit.eval_executor = eval_executor
        fresh_for_eval(eval_executor)
        try:
            # after the mode switch, so the shared evaluation is of the
            # stages this pass scores
            with featurizer_cache():
                scores = score(fit.doc, eval_executor)
        finally:
            # device storage nothing needs until the next pass
            eval_executor.reset_reads()
        record_eval(fit, scores)


def eval_reads(fit: Fit) -> tuple[str, ...]:
    assert fit.train.eval is not None
    reads: list[str] = []
    for name in fit.train.eval["metrics"]:
        metric = fit.doc.metrics[name]
        reads.append(str(metric.of))
        if metric.kind in READ_TARGET_METRIC_KINDS:
            reads.append(str(metric.fields["target"]))
    return tuple(reads)


def finish(fit: Fit) -> TrainOutcome:
    """The fit's outcome. With ``train.early_stop`` the stages are restored to
    the best-scoring snapshot first, so the returned stages — and the
    ``eval_score.metrics`` beside them — are the selected fit's
    (``selected: "early_stop.best"``); without it they are the last ones."""
    selected = "last"
    last_score = fit.last_score
    if fit.best_state is not None:
        restore(fit.stages, fit.best_state)
        last_score = fit.best_score
        selected = "early_stop.best"
    for stage in fit.stages.values():
        stage.eval()
        if isinstance(stage, Gate):
            # a pinned mask is a phase's, not the bundle's: the saved gate is
            # θ, and an apply reads its own hard split from it (§2.5)
            stage.frozen_mask = None
    eval_score = None
    if fit.train.eval is not None and last_score is not None:
        eval_score = TrainEvalScore(
            split=concrete_str(fit.train.eval["split"], "train.eval.split"),
            metrics=dict(last_score),
            passes=fit.eval_passes,
            featurizers=fit.trained_names,
            selected=selected,
        )
    trained = {name: fit.stages[name] for name in fit.trained_names}
    return TrainOutcome(
        stages=trained,
        eval_score=eval_score,
        diagnostics=fit_diagnostics(trained),
        fit_forwards=(
            {"run": fit.run, "served": fit.served} if fit.store is not None else None
        ),
        resumed=tuple(fit.resumed),
        controls={
            target: {
                "initial": fit.controls[target].initial,
                "final": fit.controls[target].controller.value,
                "signal_final": trace[-1]["signal"] if trace else float("nan"),
                "setpoint_final": trace[-1]["setpoint"] if trace else float("nan"),
                "updates": float(len(trace)),
            }
            for target, trace in fit.control_trace.items()
        },
        control_trace=fit.control_trace,
        constraints={
            name: {
                "target": fit.constraint_specs[name].target,
                # the duals the first update stepped with are the authored
                # init by construction: nothing steps them before it
                "lambda1_initial": fit.constraint_specs[name].init[0],
                "lambda2_initial": fit.constraint_specs[name].init[1],
                "lambda1_final": float(fit.duals[name][0].detach()),
                "lambda2_final": float(fit.duals[name][1].detach()),
                "value_final": trace[-1]["value"] if trace else float("nan"),
                "updates": float(len(trace)),
            }
            for name, trace in fit.constraint_trace.items()
        },
        constraint_trace=fit.constraint_trace,
        draws=fit.drawn.record() if fit.drawn is not None else {},
        anneals={
            target: {
                "start": schedule.start,
                "end": schedule.end,
                "final": schedule.value_at(fit.step, fit.total_steps),
                "shape": schedule.shape,
            }
            for target, schedule in fit.anneals.items()
        },
        phases=tuple(
            {
                "start": phase.start,
                "end": phase.end,
                "params": list(phase.params),
                "freeze_masks": list(phase.freeze_masks),
            }
            for phase in fit.phases
        ),
        checkpoints=tuple(fit.checkpoints),
    )


def read_signals(fits: Sequence[Fit]) -> dict[tuple[int, str], float]:
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


def after_update(fit: Fit, signals: Mapping[tuple[int, str], float]) -> None:
    """What follows one member's optimizer step (``fit.step`` already counts
    it): every controller observes the fit — its signal read by
    :func:`read_signals` — and moves its target for the *next* update
    (§2.11), and a scheduled ``trajectory`` checkpoint (§2.12) photographs
    the slots — so a checkpoint's ``weight.<term>`` is the weight the update
    used and its ``control.<target>`` the value set after it."""
    for target, control in fit.controls.items():
        signal = signals[(id(fit), target)]
        setpoint = ramp_setpoint(*control.ramp, fit.step, fit.total_steps)
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
    for name, lam in fit.duals.items():
        # the density this update saw and where the ascent left the duals
        # for the next one (the duals it stepped *with* are the previous
        # row's, or the authored init on the first)
        fit.constraint_trace[name].append(
            {
                "step": float(fit.step),
                "value": fit.term_values.get(f"term.{name}", float("nan")),
                "target": fit.constraint_specs[name].target,
                "lambda1": float(lam[0].detach()),
                "lambda2": float(lam[1].detach()),
            }
        )
    if fit.step in fit.checkpoint_steps:
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
