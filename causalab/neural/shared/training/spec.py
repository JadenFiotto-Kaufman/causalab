"""A fit as **plain data**: :class:`FitSpec` — everything the update loop
needs to know about one point's fit that does not change while it runs — and
:class:`ScoreSpec`, an eval pass's metrics over one row set.

Both are frozen, hold no executor, document, tokenizer or model, and pickle
with :mod:`pickle`. That is the property the loop is built on: a fit can be
built (:func:`~causalab.neural.shared.training.state.build_fit_state`) and
stepped (:func:`~causalab.neural.shared.training.loop.fit_loop`) wherever the
spec is, given something that answers ``read(name)``.

What makes them tokenizer-free is :func:`~causalab.neural.shared.metrics.
metric_in_ids`: every metric here is carried with its answers already
resolved to token ids (:class:`ResolvedMetric`), over rows cut down to the
columns the metric names.

This module imports nothing that builds or runs a forward; the specs are
made from a document and its executor in :mod:`.executors`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import torch

from causalab.neural.shared.featurizers import StageRecipe
from causalab.protocol.registry import ModelInfo
from causalab.protocol.schema import (
    READ_TARGET_METRIC_KINDS,
    AnnealSchedule,
    ConstraintSpec,
    FeaturizerSpec,
    MetricSpec,
    ObjectiveTerm,
    PhaseSpec,
)

__all__ = ["EarlyStop", "FitSpec", "ResolvedMetric", "ScoreSpec"]


@dataclasses.dataclass(frozen=True)
class ResolvedMetric:
    """One declared metric over one fixed row set, with nothing left to look
    up: ``metric`` and ``rows`` are :func:`~causalab.neural.shared.metrics.
    metric_in_ids`' pair (answers as token ids, rows cut to the metric's
    columns), ``vocabulary`` is the tokenizer's length the ids were checked
    against, and ``vocab_axis`` is ``metric_reads_vocabulary`` for the
    document the metric came from."""

    name: str
    metric: MetricSpec
    rows: tuple[Mapping[str, Any], ...]
    vocabulary: int
    vocab_axis: bool = True
    #: an objective metric's answer columns as one ``long`` tensor per field
    #: over ``rows`` (``cross_entropy``: ``target``; ``logit_diff`` /
    #: ``soft_accuracy``: ``a``, ``b``) — a minibatch indexes them
    token_ids: Mapping[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    #: an eval metric of a gathered kind (``metrics.GATHERED_KINDS``): its
    #: answer ids over the rows that carry answers, as
    #: ``metrics.metric_token_ids`` lists them; ``None`` for every other kind
    gathered_ids: Mapping[str, tuple[int, ...]] | None = None

    @property
    def reads(self) -> tuple[str, ...]:
        """The reads the metric reduces: ``of``, and the ``target`` read of a
        kind that compares two reads (``kl``, ``js``)."""
        if self.metric.kind in READ_TARGET_METRIC_KINDS:
            return (str(self.metric.of), str(self.metric.fields["target"]))
        return (str(self.metric.of),)


@dataclasses.dataclass(frozen=True)
class ScoreSpec:
    """One eval pass, as data: the split's name and its declared metrics
    resolved over the split's rows, in ``train.eval.metrics`` order — what
    :func:`~causalab.neural.shared.training.objective.score` reduces a pass's
    reads with."""

    split: str
    metrics: tuple[ResolvedMetric, ...]

    @property
    def reads(self) -> tuple[str, ...]:
        """Every read the pass needs, in metric order."""
        return tuple(read for metric in self.metrics for read in metric.reads)

    @property
    def device_scored(self) -> bool:
        """Whether some metric selects answer entries from its read where the
        read sits — the pass then keeps its reads on the device and copies
        the gathered columns, never the vocabulary."""
        return any(metric.gathered_ids is not None for metric in self.metrics)


@dataclasses.dataclass(frozen=True)
class EarlyStop:
    """§2.11 ``early_stop``: the eval metric a fit is selected by, which way
    is better, and how many passes without an improvement stop it."""

    metric: str
    mode: str
    patience: int


@dataclasses.dataclass(frozen=True)
class FitSpec:
    """One point's fit, declared and resolved (§2.11): what
    :func:`~causalab.neural.shared.training.state.build_fit_state` builds a
    :class:`~causalab.neural.shared.training.state.FitState` from, and what
    the loop reads beside it. Nothing here moves during the fit."""

    #: ``train.seed`` — the featurizer inits, the minibatch order, the masks
    seed: int
    #: ``train.params`` as authored, one optimizer group each, and the
    #: featurizers they name
    params: tuple[str, ...]
    trained_names: tuple[str, ...]
    #: ``train.optimizer`` as authored: the name, the scalar or per-entry
    #: ``lr`` / ``weight_decay``, betas, momentum, the lr schedule
    optimizer: Mapping[str, Any]
    #: ``train.objective`` in authored order, and the metrics its terms name,
    #: resolved over the fit's rows
    objective: tuple[ObjectiveTerm, ...]
    metrics: Mapping[str, ResolvedMetric]
    #: the reads the objective needs, in term order — the groups a step runs
    objective_reads: tuple[str, ...]
    #: the minibatch partition: row indices into the fit's rows, in order
    batches: tuple[tuple[int, ...], ...]
    epochs: int
    total_steps: int
    #: ``train.eval``: the split, the metric names and the cadence in epochs;
    #: ``None`` / ``()`` without an eval
    eval_split: str | None = None
    eval_metrics: tuple[str, ...] = ()
    eval_every_epochs: int | None = None
    early_stop: EarlyStop | None = None
    #: the eval pass itself, for an engine that resolves it when it plans the
    #: fit rather than at the first pass — a fit that runs where the model is
    #: carries its metrics with it (:func:`.executors.score_spec`). The
    #: reference engine resolves the same pass on its eval executor, lazily
    #: (:func:`.executors.eval_pass`), and leaves this ``None``
    score: ScoreSpec | None = None
    #: ``train.anneal`` / ``train.control`` / ``train.phases`` as authored;
    #: :func:`build_fit_state` binds them to the stages
    anneal: Mapping[str, AnnealSchedule] | None = None
    control: Mapping[str, Mapping[str, Any]] | None = None
    phases: tuple[PhaseSpec, ...] | None = None
    #: §2.12 ``trajectory``: the updates after which the fit is photographed
    checkpoint_steps: frozenset[int] = frozenset()
    #: how the trained stages are built when nobody hands them over: a recipe
    #: per stage in build order (the trained featurizers in ``params`` order,
    #: and every member of a budget pool — a pool is built whole), the
    #: featurizer specs they are built from, the model's static facts a
    #: grouped gate derives its map from, and the point's sweep coordinates
    recipes: tuple[StageRecipe, ...] = ()
    featurizers: Mapping[str, FeaturizerSpec] = dataclasses.field(default_factory=dict)
    model_info: ModelInfo | None = None
    coords: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def constraints(self) -> dict[str, ConstraintSpec]:
        """§2.11 ``constraint``: the authored constraint per term name."""
        return {
            term.name: term.constraint
            for term in self.objective
            if term.constraint is not None and term.name is not None
        }
