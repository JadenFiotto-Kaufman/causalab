"""What a fit records about itself: :func:`fit_diagnostics` (saved beside
the bundle), the trajectory photograph (:func:`checkpoint`, §2.12) and the
state an ``early_stop`` selection is restored from (:func:`snapshot` /
:func:`restore`)."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from causalab.neural.shared.execution import MASK_DECISIVE_MARGIN, Checkpoint
from causalab.neural.shared.featurizers import (
    ORTHONORMAL_TOLERANCE,
    Gate,
    Stage,
    Subspace,
    orthonormality_deviation,
)

__all__ = ["checkpoint", "fit_diagnostics", "restore", "snapshot"]


def fit_diagnostics(stages: Mapping[str, Stage]) -> dict[str, dict[str, Any]]:
    """What each fit can say about *itself*, saved beside the bundle.

    The case this exists for: a DBM fit whose θ never separates — **no**
    dimension outside [0.1, 0.9] — can still score **1.000**. The mechanism is
    that :meth:`Gate._mask` returns a *hard* ``θ > 0`` mask in eval mode, and
    ``run_training`` puts the stages in eval mode before returning — so the
    1.000 is the hard mask, and with θ never separated ``θ > 0`` is a coin
    flip on gradient noise. Roughly half the dimensions swap, which at a
    readout layer scores 1.000.

    Such a fit produces a **meaningless mask and a perfect number**, and
    nothing in the run's saved outputs used to say so. These two numbers do:

    ``decisive_fraction``
        The fraction of dimensions where σ(θ) is outside
        [0.5 − :data:`MASK_DECISIVE_MARGIN`, 0.5 + …]. Near 0 means the gate
        never committed and the score below it describes noise, whatever it
        says.
    ``hard_mask_size``
        How many dimensions ``θ > 0`` keeps — the mask the eval-mode score was
        actually computed through, which is the number a localization claim is
        about.
    ``frozen_units`` / ``reawakened_units`` (with ``dead`` when authored)
        The dead-unit bookkeeping (§2.5 ``dead``): how many units a
        ``freeze_after`` rule froze, and how many units were hard-off after
        some step yet are kept by the final hard mask — the count a ``leak``
        exists to raise, and exactly 0 on a frozen gate. Recorded for every
        gate, rule or none, so the two fits are comparable.

    Under a ``clamp`` gate (§2.5 ``parametrization``) the soft mask is ``θ``
    itself and the hard split ``θ > ½``, so both numbers are read through
    :meth:`Gate.soft_mask` / :meth:`Gate.hard_mask` rather than spelled here;
    ``parametrization`` is recorded so the record says which split it counted.
    Under ``hard_concrete`` the soft mask is the *deterministic* stretched and
    clipped ``σ(θ)`` — the mean of the sampled training mask — and ``stretch``
    is recorded beside it because the hard split depends on it. Two
    consequences for ``decisive_fraction`` there: the stretch scales the
    distance from ½ by ``ζ − γ`` (1.2 at the default), so a unit is decisive
    at a smaller ``|θ|`` than a ``sigmoid`` gate's and *saturates* at exactly
    0 or 1 once ``|θ| ≳ logit((1 − γ)/(ζ − γ))``; and the temperature does not
    enter the deterministic mask, so an anneal of β leaves the number alone
    where a ``sigmoid`` anneal of ``T → 0`` drives it to 1. The number is the
    right one for the gate it describes, but it is not comparable across maps
    at one :data:`MASK_DECISIVE_MARGIN`.

    Both count the gate's *units* — one ``theta`` entry each. On a grouped gate
    (§2.5 ``group``) a unit is a head, or one ``(expert, neuron)`` of the
    expert table, so ``hard_mask_size`` is a head or expert-neuron count and
    ``groups`` records how many units there were; ``width`` stays the site's
    coordinate width. On a position gate (§2.5 ``axis``) a unit is an
    addressed token position, so ``width`` is the window's length rather
    than the site's coordinate width — which is recorded nowhere — and
    ``axis`` is recorded to say so.

    This is the half of the DBM finding that needs no GPU. Retuning `l1` and
    the anneal so θ *does* separate needs a run to validate and is deliberately
    not asserted here.

    A fitted ``subspace`` reports ``orthonormality_deviation`` — ``max|QᵀQ − I|``
    of the rotation the bundle saves — and ``within_tolerance``, its verdict
    against :data:`~causalab.neural.shared.featurizers.ORTHONORMAL_TOLERANCE`
    (``1.0``/``0.0``). The verdict is recorded beside the value because the
    value is raw roundoff, whose last digits are accumulation order and differ
    across BLAS and device, while the verdict is what a reader acts on: a
    ``0.0`` is a rotation no later document can name as an ``init``. The
    ``cayley`` map's fp32 error is ~1e-6 where fits live but grows quadratically
    in ``‖X‖`` once ``X⊥`` goes rank-deficient
    (:class:`~causalab.neural.shared.featurizers.Cayley`, *Conditioning*), and
    nothing else checks orthonormality at save time — the next check is
    ``_init_basis`` refusing the rotation as a start in a later run, a
    diagnostic that would otherwise arrive one run away from its cause.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, stage in stages.items():
        if isinstance(stage, Subspace):
            with torch.no_grad():
                deviation = orthonormality_deviation(stage.weight)
            out[name] = {
                "k": float(stage.k),
                "orthonormality_deviation": deviation,
                "within_tolerance": float(deviation <= ORTHONORMAL_TOLERANCE),
            }
            continue
        theta = getattr(stage, "theta", None)
        if not isinstance(stage, Gate) or theta is None:
            continue
        with torch.no_grad():
            soft = stage.soft_mask().detach().float()
            decisive = (soft - 0.5).abs() > MASK_DECISIVE_MARGIN
            # a budget's last drawn k: the gate's own, or its pool's
            budget_k = stage._k if stage.pool is None else stage.pool.k
            out[name] = {
                "width": float(stage.width),
                **({"groups": float(theta.numel())} if stage.groups else {}),
                # §2.5 `axis`: `width` is then the window length and every count
                # above is over positions — the record says so, as `groups` does
                **({"axis": stage.axis} if stage.axis is not None else {}),
                "decisive_fraction": float(decisive.float().mean()),
                "hard_mask_size": float(stage.hard_mask().sum()),
                "temperature": float(stage.temperature),
                "parametrization": stage.parametrization,
                # §2.5 the mapping form: with it, `decisive_fraction` reports the
                # backward map's confidence (the forward is 0/1 by construction)
                # and `hard_mask_size` is the eval split, which the forward's
                # count above ½ equals under sigmoid and clamp only — so the
                # record says which regime its numbers were taken in
                **(
                    {"forward": stage.forward_mask}
                    if stage.forward_mask is not None
                    else {}
                ),
                **(
                    {"stretch": list(stage.stretch)}
                    if stage.stretch is not None
                    else {}
                ),
                **(
                    {"init_fill": stage.init_fill}
                    if stage.init_fill is not None
                    else {}
                ),
                **(
                    {"init_from_scores": stage.init_scores}
                    if stage.init_scores is not None
                    else {}
                ),
                # §2.5 `dead`: the rule as authored, `frozen_units`, and
                # `reawakened_units` — the latter under every rule and none,
                # since it is the observable a leak exists to move
                **stage.dead_diagnostics(),
                # a loaded gate read out at a count (§2.5 `top_k`): the split
                # the numbers above were counted through is a cut, not the
                # map's threshold, and the record says so
                **({"top_k": float(stage.top_k)} if stage.top_k is not None else {}),
                # a budget gate (§2.5): the schedule, the cut `hard_mask_size`
                # is read at, and the budget of the last step — the number a
                # trajectory checkpoint's `k` column carries
                **(
                    {
                        "k_schedule": dict(stage.k_schedule),  # carries `of`
                        "eval_k": float(stage.eval_k()),
                        **({"k": float(budget_k)} if budget_k is not None else {}),
                        "stop_grad_shift": float(stage.stop_grad_shift),
                        # a pooled gate (§2.5 `pool`): the pool, its unit
                        # count, and how many of THIS gate's units the pooled
                        # cut keeps — `hard_mask_size` above is that number,
                        # `eval_k` the pool's
                        **(
                            {
                                "pool": stage.pool.name,
                                "pool_units": float(stage.pool.units),
                            }
                            if stage.pool is not None
                            else {}
                        ),
                    }
                    if stage.parametrization == "budget"
                    and stage.k_schedule is not None
                    else {}
                ),
            }
    return out


def snapshot(stages: Mapping[str, Stage]) -> dict[str, dict[str, torch.Tensor]]:
    """Detached copies of every trained stage's parameters.

    ``state_dict`` rather than ``slot_params`` on purpose: a ``subspace``
    stage's ``weight`` is *computed* by an orthogonal parametrization, so the
    tensor the optimizer actually steps is
    ``parametrizations.weight.original``. Restoring the materialized weight
    would restore nothing.
    """
    return {
        name: {key: value.detach().clone() for key, value in stage.state_dict().items()}
        for name, stage in stages.items()
    }


def restore(
    stages: Mapping[str, Stage], snapshot: Mapping[str, Mapping[str, torch.Tensor]]
) -> None:
    """Put the snapshotted parameters back, in place."""
    for name, stage in stages.items():
        state = snapshot.get(name)
        if state is not None:
            stage.load_state_dict(dict(state))


def checkpoint(
    step: int,
    epoch: int,
    stages: Mapping[str, Stage],
    *,
    loss: float,
    term_values: Mapping[str, float],
    controls: Mapping[str, float],
    phase: int | None = None,
) -> Checkpoint:
    """One photograph of the fit after ``step`` updates: every trained slot,
    detached to the CPU, and the record a reader wants beside it — the loss
    and its terms, each gate's mask numbers, each controlled value."""
    slots = {
        name: {
            slot: param.detach().to("cpu").clone()
            for slot, param in stage.slot_params().items()
        }
        for name, stage in stages.items()
    }
    record: dict[str, Any] = {"step": step, "epoch": epoch, "loss": loss, **term_values}
    for name, values in fit_diagnostics(stages).items():
        for key in ("hard_mask_size", "decisive_fraction", "k"):
            if key in values:
                record[f"{name}.{key}"] = values[key]
    for target, value in controls.items():
        record[f"control.{target}"] = value
    if phase is not None:
        record["phase"] = phase  # §2.11 `phases`: the window this update ran in
    return Checkpoint(step=step, epoch=epoch, slots=slots, record=record)
