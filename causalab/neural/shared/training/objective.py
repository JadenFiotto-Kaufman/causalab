"""The differentiable side of a ``train.objective`` (spec §2.11): metric
tensors with a gradient (:func:`metric_tensor`, the objective-side twin of
``metrics.compute_metric``) and the penalties over trained featurizers
(:func:`regularizer`). Pure torch over the shared featurizer stages — every
engine's loss is built from these, so an objective means one thing whichever
engine steps it."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from causalab.neural.shared.featurizers import Gate, Stage
from causalab.neural.shared.metrics import (
    column_token_ids,
    js_divergence,
    restrict_token_ids,
)
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import MetricSpec, concrete_str

__all__ = ["metric_tensor", "regularizer"]


def metric_tensor(
    metric: MetricSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None = None,
    token_ids: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """A differentiable per-example metric (the objective-side twin of
    ``metrics.compute_metric``). Only the reduction kinds with a gradient
    are usable in an objective."""
    logits = of_value[:, 0, :] if of_value.dim() == 3 else of_value
    logits = logits.float()
    kind = str(metric.kind)

    form = str(metric.token_form)  # §2.10; `auto` is the historical default

    def ids(field: str) -> torch.Tensor:
        if token_ids is not None:
            return token_ids[field]
        column = concrete_str(metric.fields[field], f"metric field {field}")
        return torch.tensor(
            column_token_ids(
                tokenizer,
                [row[column] if form == "id" else str(row[column]) for row in rows],
                token_form=form,
                where=f"metric {kind}.{field}",
            ),
            dtype=torch.long,
            device=logits.device,
        )

    if kind == "cross_entropy":
        log_probs = torch.log_softmax(logits, dim=-1)
        return -log_probs.gather(1, ids("target").unsqueeze(1)).squeeze(1)
    if kind == "logit_diff":
        return logits.gather(1, ids("a").unsqueeze(1)).squeeze(1) - logits.gather(
            1, ids("b").unsqueeze(1)
        ).squeeze(1)
    if kind == "soft_accuracy":
        # the same margin through a sigmoid (metrics.py's twin): its gradient
        # vanishes once a row is decided, so the fit spends its updates on the
        # rows still near the boundary — what a soft-accuracy objective is for
        margin = logits.gather(1, ids("a").unsqueeze(1)).squeeze(1) - logits.gather(
            1, ids("b").unsqueeze(1)
        ).squeeze(1)
        return torch.sigmoid(margin)
    if kind == "kl":
        if target_value is None:
            raise ProtocolError("P2", "kl needs its target read's value")
        target = target_value[:, 0, :] if target_value.dim() == 3 else target_value
        p = torch.log_softmax(logits, dim=-1)
        q = torch.log_softmax(target.float(), dim=-1)
        return (p.exp() * (p - q)).sum(dim=-1)
    if kind == "js":
        if target_value is None:
            raise ProtocolError("P2", "js needs its target read's value")
        target = target_value[:, 0, :] if target_value.dim() == 3 else target_value
        # the arithmetic the saved metric table reports (metrics.js_divergence),
        # so the objective and the record cannot disagree; differentiable in
        # `logits`, and the target is a constant model's read
        return js_divergence(
            logits, target.float(), restrict_token_ids(metric, rows, tokenizer)
        )
    raise ProtocolError(
        "P2",
        f"metric kind {kind!r} has no gradient — objectives compose from "
        "cross_entropy / logit_diff / soft_accuracy / kl / js (§2.11)",
    )


def _cost(
    piece: torch.Tensor, target: str, costs: Mapping[str, float] | str | None
) -> torch.Tensor:
    """§2.11 ``costs``: one target's penalized quantities, scaled before the
    concatenation. A table costs the target its entry (1 when unlisted);
    ``parameter_count`` divides by the target's own element count — ``theta``
    as stored, so on a grouped gate its *units* (heads, experts; positions
    on a position gate), not the
    coordinates they span: the same per-unit reading :func:`regularizer`
    gives the quantity — so under ``reduce: sum`` the term is a sum of
    per-featurizer means — NeuroSurgeon's λ scaled with the parameter count,
    one weight meaning one thing across featurizers of different sizes
    without the author doing the division. Under the default ``reduce: mean``
    the same word gives ``(1/N) Σ_f S_f/n_f``: the per-featurizer equality
    kept, the whole term scaled by the total unit count ``N`` — the width
    coupling the word exists to remove — so ``parameter_count`` pairs with
    ``sum`` (the other pairing is legal, and is that coupling)."""
    if costs is None:
        return piece
    if costs == "parameter_count":
        # never empty: a gate has theta, a param target matched a slot
        return piece / piece.numel()
    assert isinstance(costs, Mapping)
    return piece * float(costs.get(target, 1.0))


def regularizer(
    kind: str,
    targets: Sequence[str],
    stages: Mapping[str, Stage],
    reduce: str = "mean",
    costs: Mapping[str, float] | str | None = None,
) -> torch.Tensor:
    """One penalty over everything ``targets`` name (§2.11): the ``reduce``
    — the mean when unauthored, or the sum — over the **concatenation** of
    their penalized quantities, not a mean of per-featurizer means — so under
    the mean a gate with more units weighs more, and one weight over forty
    layers' gates counts selected units across all of them; under the sum a
    kept unit costs the term's weight whatever the unit count, NeuroSurgeon's
    ``λ · Σ`` convention, which is what lets one weight mean one thing across
    gates of different sizes.

    For a ``gate`` under ``l1`` the quantity is the SOFT mask ``σ(θ/T)``, not
    ``|θ|`` — DBM sparsity pushes mask mass toward zero features,
    temperature-annealed — over ``theta`` as stored, so on a grouped gate it
    is per unit: selected heads, not the coordinates they span; the whole
    (expert, neuron) table, not the slots a token happens to fill; on a
    position gate (§2.5 ``axis``) the addressed positions. Under
    ``l0`` it is the gate's **expected kept fraction** per unit
    (:meth:`Gate.expected_l0`): Louizos et al.'s closed form for a
    ``hard_concrete`` gate, whose sampled training mask makes the soft mask
    the wrong surrogate.
    For every other featurizer (or a gate under ``l2``) it is ``|p|`` or ``p²``
    over its params; a dotted target restricts to that one slot. The pairing
    is by map, not by kind: ``l0`` on a deterministic gate would be ``l1``
    under a second name, and ``l1`` on a ``hard_concrete`` gate would penalize
    a mask its training forward never uses — both, and ``l0`` on a featurizer
    with no mask, are refused at validation (rule 4) and again here.

    ``costs`` scales each target's quantities before the concatenation
    (:func:`_cost`), so a table ``{"gate_15": 0.25}`` makes a kept unit of
    that gate a quarter as expensive as one elsewhere, and
    ``"parameter_count"`` makes every target weigh its mean.
    """
    pieces: list[torch.Tensor] = []
    for target in targets:
        fname, _, slot_name = target.partition(".")
        stage = stages[fname]
        if isinstance(stage, Gate) and stage.parametrization == "budget":
            raise ProtocolError(
                "P2",
                f"regularizer target {target!r}: a budget gate's mask sums to the "
                "step's budget by construction — it takes no sparsity penalty (§2.5)",
            )
        if isinstance(stage, Gate) and kind in ("l1", "l0"):
            sampled = stage.parametrization == "hard_concrete"
            if sampled != (kind == "l0"):
                raise ProtocolError(
                    "P2",
                    f"regularizer target {target!r}: {kind!r} does not pair with a "
                    f"{stage.parametrization!r} gate — 'l0' is the expected kept "
                    "fraction of a sampled (hard_concrete) mask, 'l1' the mean of a "
                    "deterministic one (§2.11); validation refuses this at load",
                )
            # the gate's own relaxed mask — σ(θ/T), or θ itself under `clamp`
            # — or, under `hard_concrete`, its expected L0
            quantity = stage.soft_mask() if kind == "l1" else stage.expected_l0()
            pieces.append(_cost(quantity.flatten(), target, costs))
            continue
        if kind == "l0":
            raise ProtocolError(
                "P2",
                f"regularizer target {target!r}: 'l0' is the expected kept fraction "
                "of a gate's mask; this featurizer has no mask",
            )
        own: list[torch.Tensor] = []
        for slot, param in stage.slot_params().items():
            if slot_name and slot != slot_name:
                continue
            own.append((param.abs() if kind == "l1" else param.pow(2)).flatten())
        if not own:
            raise ProtocolError("P2", f"regularizer target {target!r} matches no slot")
        # one target, one cost: `parameter_count` counts every slot it matched
        pieces.append(_cost(torch.cat(own), target, costs))
    cat = torch.cat(pieces)
    return cat.sum() if reduce == "sum" else cat.mean()
