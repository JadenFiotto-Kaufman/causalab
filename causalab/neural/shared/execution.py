"""The engine-neutral half of ``Engine.execute``: run every point, lower its
metrics, and fill the output tables.

Everything here consumes the *executor surface* — ``read_value`` /
``dense_value`` / ``windowed_value`` / ``run_all`` / ``rows_for_metrics`` /
``is_generated`` / ``addressed_steps`` / ``generated_ids`` / ``bundle`` — and
nothing in it knows whether a hook or a trace produced the tensors. An engine
supplies its executor factory (and, if it trains, its train runner) and keeps
only its own identity stamp.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from causalab.causal.scoring import ScoringCheck
from causalab.neural.shared.executor_base import ForwardCache, Interning, PrefixPlan
from causalab.neural.shared.fires import group_label
from causalab.neural.shared.location_ledger import (
    ledger_identity,
    ledger_records,
    point_ledger,
)
from causalab.neural.shared.metrics import compute_metric
from causalab.neural.shared.outputs import (
    MetricTable,
    TensorFile,
    write_outputs,
)
from causalab.neural.shared.head import capture_spec
from causalab.neural.shared.services import input_roles, site_identity, spec_identity
from causalab.protocol.canonical import canonical_model
from causalab.protocol.engine import ExecutionRequest, RunResult
from causalab.protocol.errors import ProtocolError
from causalab.protocol.examples import example_labels
from causalab.protocol.resolve import build_artifact_identity
from causalab.protocol.fit_splits import check_fit_splits
from causalab.protocol.resolution import (
    Eligibility,
    Resolution,
    Unavailable,
    available,
    cell_key,
    cell_record,
    unavailable,
)
from causalab.protocol.plan import (
    PointPlan,
    fit_cohorts,
    generated_budget,
    interned_groups,
    lower_bands,
    plan_point,
)
from causalab.protocol.run import (
    RAGGED_KEY,
    RUN_RECORD_NAME,
    record_fires,
    record_measured_bounds,
    record_ragged_geometry,
)
from causalab.provenance import runtime_identity
from causalab.protocol.estimand import metric_record_identity
from causalab.protocol.schema import (
    READ_TARGET_METRIC_KINDS,
    WHOLE_WINDOW_METRIC_KINDS,
    Document,
    SiteSpec,
    metric_reads_vocabulary,
    parse_document,
)

__all__ = [
    "ExecutorSurface",
    "MASK_DECISIVE_MARGIN",
    "TrainEvalScore",
    "TrainOutcome",
    "TrainRunner",
    "campaign_cache",
    "campaign_plans",
    "execute_request",
    "featurizer_identity",
]


class ExecutorSurface(Protocol):
    """What :func:`execute_request` needs from a point executor."""

    bundle: Any

    def run_all(self) -> None: ...
    def read_value(self, name: str) -> Any: ...
    def resolution(self, name: str) -> Resolution: ...
    def row_resolutions(self, name: str) -> list[Unavailable | None]: ...
    def dense_value(self, name: str) -> Any: ...
    def dense_rows(self, name: str, rows: Sequence[int]) -> Any: ...
    def windowed_value(self, name: str) -> list[Any]: ...
    def generated_metric(self, metric: Any) -> list[list[Any]]: ...
    def is_generated(self, name: str) -> bool: ...
    def addressed_steps(self, name: str) -> list[list[int]]: ...
    def generated_ids(self, name: str) -> list[list[int]]: ...
    def rows_for_metrics(self) -> list[dict[str, Any]]: ...
    def check_scoring(self) -> ScoringCheck: ...


#: The run receipt's block for the scoring identity (spec §2.2): per base
#: dataset ref, what :meth:`ExecutorSurface.check_scoring` found before the
#: point's first forward — ``{"digest", "string_mode", "result"}``, the result
#: one of ``causalab.causal.scoring.SCORING_RESULTS``. A sibling of the
#: ``execution`` block: recorded, never gated, because the refusal has already
#: happened by the time anything is written.
SCORING_KEY = "scoring"


def record_scoring(output_dir: Path, ref: str, check: ScoringCheck) -> Path | None:
    """Write ``check`` into the run receipt (``protocol.json`` under
    ``output_dir``) as ``scoring.<ref>``, merged per ref, and return the
    receipt's path.

    The receipt is written before execution by
    :func:`~causalab.protocol.run.write_run_record`; this adds to it what the
    pre-forward check found. A caller that wrote no receipt (a workflow step
    records its run in ``_step.json``; an engine test drives
    ``execute_request`` directly) has nothing to amend: ``None``.
    """
    receipt = output_dir / RUN_RECORD_NAME
    if not receipt.is_file():
        return None
    record = json.loads(receipt.read_text())
    block = dict(record.get(SCORING_KEY) or {})
    block[ref] = check.as_record()
    record[SCORING_KEY] = block
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return receipt


@dataclasses.dataclass(frozen=True)
class TrainEvalScore:
    """One point's ``train.eval`` result: the split it was measured on, the
    mean per eval metric, and how many eval passes ran.

    ``passes`` is there so a reader can tell "evaluated once at the end" from
    "evaluated every epoch and early-stopped", which decides whether the score
    describes the returned weights or merely the last pass over them.
    """

    split: str
    metrics: Mapping[str, float]
    passes: int
    #: The trained featurizers this score describes — the join back to the
    #: saved bundle, which stamps the same point digest.
    featurizers: tuple[str, ...] = ()
    #: How the returned weights were chosen: ``"early_stop.best"`` when the
    #: loop restored the best-scoring snapshot, ``"last"`` when nothing
    #: selected. Without it a reader cannot tell whether this score describes
    #: the saved weights or merely the final pass over them.
    selected: str = "last"

    def as_record(self, *, point: str, coords: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "point": point,
            "coords": dict(coords),
            "split": self.split,
            "metrics": dict(self.metrics),
            "passes": self.passes,
            "featurizers": list(self.featurizers),
            "selected": self.selected,
        }


#: A soft mask is "decisive" at a dimension when σ(θ) is outside
#: ``[0.5 - MASK_DECISIVE_MARGIN, 0.5 + MASK_DECISIVE_MARGIN]`` — i.e. outside
#: [0.1, 0.9] at the default — wide enough that a gate whose σ never left the
#: relaxed band reports a decisive fraction of 0 rather than crediting noise.
MASK_DECISIVE_MARGIN = 0.4


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """One photograph of a fit (§2.12 ``trajectory``): the trained
    featurizers' slots after ``step`` updates, detached and on the CPU, and
    what the fit says about itself there — ``step``, ``epoch``, the loss and
    every objective term's value at that update, every trained gate's
    ``hard_mask_size`` / ``decisive_fraction``, every controlled
    hyperparameter's live value. The record rides on the bundle entry as
    numbers, so a reader has the scalar trace from the header alone."""

    step: int
    epoch: int
    slots: Mapping[str, Mapping[str, Any]]
    record: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class TrainOutcome:
    """What an engine's train loop produced.

    ``stages`` is the fitted stage per trained featurizer name — what the save
    manifest writes. ``eval_score`` is the held-out score from ``train.eval``,
    and it is deliberately *not* a metric-table column: it is measured on a
    different split, i.e. a different population from the point's metric rows,
    and folding it in under the same metric name is exactly the confusion that
    made a fit document's ``iia.json`` read as an eval score when it was the
    train score (spec §2.12).
    """

    stages: Mapping[str, Any]
    eval_score: TrainEvalScore | None = None
    #: Per trained featurizer, whatever the fit can say about *itself* — see
    #: :func:`~causalab.neural.engines.pytorch_hooks.train.fit_diagnostics`.
    #: Written beside the bundle, because a fit that produced a meaningless
    #: parameter and a perfect score is otherwise indistinguishable from a
    #: good one.
    diagnostics: Mapping[str, Mapping[str, float]] = dataclasses.field(
        default_factory=dict
    )
    #: What the fit's inner passes paid for its constant groups (``run``) and
    #: what the campaign store handed them instead (``served``) — the saving
    #: §4 "Fits" promises, tallied per point by the loop, since a cohort's
    #: passes interleave several points' (§4 "Cohorts"). ``None`` when the
    #: loop ran against no store.
    fit_forwards: Mapping[str, int] | None = None
    #: The block each of this fit's forwards that resumed started at (§4
    #: "Resume"): one entry per forward the point took part in, cohort
    #: forwards included, in the shape of ``ForwardCache.resumed``.
    resumed: tuple[int, ...] = ()
    #: The rows-per-forward bound every window of the fit's cohort ran under
    #: (§8 ``fit_rows``): the authored bound, or the one the engine measured
    #: when none was authored — after any shrink, of a grad window or of an
    #: eval window packing under it, so pinning it is safe (a run that shrank
    #: says so in ``fit_rows_shrinks``; only when the shrink was an eval
    #: window's does the pinned re-run pack its grad windows smaller than
    #: this one did); ``None`` when the cohort ran unbounded.
    fit_rows: int | None = None
    #: How many windows packed under the fit's bound — grad forwards, and the
    #: batched eval passes when ``batch_rows`` is unauthored — were re-packed
    #: after running out of memory: the measured bound was too loose by that
    #: many. A cohort's count, reported on each of its members.
    fit_rows_shrinks: int = 0
    #: Per ``train.control`` target (§2.11): where the controlled value
    #: started and ended, the last signal and setpoint, the update count —
    #: written into ``fit_diagnostics.json`` beside the featurizer records.
    controls: Mapping[str, Mapping[str, float]] = dataclasses.field(
        default_factory=dict
    )
    #: The same, per update: ``{"step", "value", "signal", "setpoint"}`` rows
    #: — what a checkpoint records about the controller at its step.
    control_trace: Mapping[str, Sequence[Mapping[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: Per ``constraint`` term (§2.11), by name: the target density, the dual
    #: pair where it started and ended (``lambda1_initial`` … ``lambda2_final``),
    #: the density the last update saw (``value_final``) and the update count
    #: — written into ``fit_diagnostics.json`` under ``constraints``.
    constraints: Mapping[str, Mapping[str, float]] = dataclasses.field(
        default_factory=dict
    )
    #: The same, per update: ``{"step", "value", "target", "lambda1",
    #: "lambda2"}`` rows — the duals *after* that update's ascent.
    constraint_trace: Mapping[str, Sequence[Mapping[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: Per drawn counterfactual role (§2.2 ``draw``): the kind, the ``eval``
    #: member the non-training forwards read, and ``members`` — one list per
    #: epoch of the member index each row took — written into
    #: ``fit_diagnostics.json`` under ``draws``, so a reader sees which member
    #: each row trained against (under ``train.steps.updates`` a partial
    #: final epoch lists a member for every row, including rows whose
    #: minibatch never ran).
    draws: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    #: Per ``train.anneal`` target (§2.11): the schedule's endpoints, shape
    #: and the value the last update used — beside ``controls`` in
    #: ``fit_diagnostics.json``, so a reader sees what moved the weight it
    #: finds in the trajectory without re-deriving the schedule.
    anneals: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    #: Per ``train.phases`` entry (§2.11), in order: the update window it
    #: covered (``start``, ``end``), what trained in it (``params``), which
    #: masks it pinned (``freeze_masks``) — written under ``phases`` in
    #: ``fit_diagnostics.json``; empty for a one-phase fit.
    phases: Sequence[Mapping[str, Any]] = ()
    #: The fit's checkpoints, when the document saves a ``trajectory``
    #: (§2.12); empty otherwise. The last one, when present, is the fit the
    #: returned ``stages`` hold — unless ``early_stop`` restored an earlier
    #: best, which the record's ``step`` lets a reader see.
    checkpoints: Sequence[Checkpoint] = ()


#: An engine's train loop: fit these points **together** where it can (§4
#: "Cohorts") and return one outcome per point, in order. The points are one
#: cohort — same realization, same rows, same frame (``plan.fit_cohorts``) —
#: so the loop may run their optimizer steps as one forward each.
TrainRunner = Callable[
    [Sequence[Document], Sequence[Any], ExecutionRequest], Sequence[TrainOutcome]
]


@dataclasses.dataclass(frozen=True)
class _Windowed:
    """One continuation metric's per-example results, plus what the rows need
    to stay legible: the steps each value scored, and whether the example
    addressed anything at all."""

    values: list[list[Any]]
    steps: list[list[int]] | None
    matched: list[bool]


def campaign_plans(
    docs: Sequence[Document], canonical: Sequence[Mapping[str, Any]]
) -> tuple[PointPlan, ...]:
    """The per-point plans a campaign executes from.

    Public because the interning claim is checkable arithmetic:
    :func:`~causalab.protocol.plan.interned_groups` over these plans is how
    many forward groups a run *owes*, and
    :attr:`~causalab.protocol.engine.RunResult.forwards` is what it paid. One
    derivation, so the number a caller verifies against is the number
    execution keyed on.

    ``canonical`` is the points' canonical forms, in lockstep with ``docs``
    (:attr:`~causalab.protocol.engine.ExecutionRequest.canonical`): the data
    half of a group's identity is read from there, never recomputed.
    """
    if len(docs) != len(canonical):
        raise ProtocolError(
            "P2",
            f"{len(docs)} point documents but {len(canonical)} canonical forms "
            "— the two are in lockstep per point",
        )
    return tuple(
        plan_point(doc, data_identity=_data_identity(doc, form))
        for doc, form in zip(docs, canonical)
    )


def _data_identity(doc: Document, canonical: Mapping[str, Any]) -> dict[str, str]:
    """Input role → the identity of the rows that role will be encoded from.

    Folded into every forward-group digest so two points reading *different*
    data on the same role never intern together. What determines a role's batch
    is the **content** of the rows the ref selects plus the one field the
    executor tokenizes out of them (``DataRole.resolved_field`` —
    ``<column>[eval]`` for a drawn role, §2.2 — the spelling
    :func:`~causalab.neural.shared.services.resolve_roles` hands the engine),
    so the identity is ``"<content digest>#<field>"`` where the digest is the
    one the canonical form already stamped for that role (§2.2, §7: sha256 over
    the selected rows, not the file). The ref's *name* is deliberately absent:
    two tables under one name must never intern, and one table under two names
    must — which a name-keyed identity got backwards on both counts.

    The role names mirror ``resolve_roles`` (``counterfactual[0]`` for a
    tuple-valued role) so the keys line up with the plan's ``input``.
    """
    digests = _role_digests(doc, canonical)
    return {
        role_name: f"{digests[role_name]}#{role_spec.resolved_field}"
        for role_name, role_spec in input_roles(doc).items()
    }


def _role_digests(doc: Document, canonical: Mapping[str, Any]) -> dict[str, str]:
    """Input role → the content digest of the rows it reads, read off the
    point's canonical form (``data.<role>.digest``; a tuple-valued role is a
    list there, indexed in step with ``resolve_roles``).

    Read, not recomputed: the canonical form's digest is the one the point
    digest committed to, so there is one content digest per table in the
    system and the interning identity cannot drift from the provenance one.
    A role without a stamped digest is refused rather than named — nothing
    executable lacks one (rows resolve through the same ref the stamp did),
    so this only fires on a canonical form that is not this point's.
    """
    stamped = canonical.get("data")
    if not isinstance(stamped, Mapping):
        raise ProtocolError(
            "P2", "canonical form carries no 'data' section to read digests from"
        )
    digests: dict[str, str] = {}
    for role, value in doc.data.items():
        entries = value if isinstance(value, tuple) else (value,)
        forms_raw = stamped.get(role)
        # the shapes must agree, never broadcast: a tuple-valued role is a
        # list in the canonical form and a single role is one mapping, so a
        # form of the other shape is not this point's and is refused
        if isinstance(value, tuple) != isinstance(forms_raw, (list, tuple)):
            raise ProtocolError(
                "P2",
                f"data role {role!r} is {'tuple' if isinstance(value, tuple) else 'single'}"
                "-valued but its canonical form is not — the two are the same point's",
            )
        forms = list(forms_raw) if isinstance(forms_raw, (list, tuple)) else [forms_raw]
        if len(forms) != len(entries):
            raise ProtocolError(
                "P2",
                f"data role {role!r} has {len(entries)} entries but its "
                f"canonical form has {len(forms)}",
            )
        for j, form in enumerate(forms):
            role_name = role if not isinstance(value, tuple) else f"{role}[{j}]"
            digest = form.get("digest") if isinstance(form, Mapping) else None
            if not isinstance(digest, str) or not digest:
                raise ProtocolError(
                    "P2",
                    f"data role {role_name!r} has no content digest in its "
                    "canonical form — the interning identity is the digest of "
                    "the rows a role reads (§2.2), never the ref's name",
                )
            digests[role_name] = digest
    return digests


def _tap_union(
    docs: Sequence[Document], plans: Sequence[PointPlan]
) -> dict[str, tuple[SiteSpec, ...]]:
    """Forward-group digest → every site the campaign taps in that group.

    The union *is* the interning. Taps are deliberately absent from a group's
    digest, so the single pass a shared digest earns has to capture every
    address any point will ask of it — for a 32-layer scan that is one
    counterfactual forward with 32 taps instead of 32 forwards with one each.

    Continuation reads are excluded: those are served by the decode's
    per-step accumulation, not by the prefill capture this store holds, so a
    decoding group contributes only its prompt-frame taps (and can therefore
    still hand its prefill to a non-decoding point that shares the digest).

    A site enters the union as the read **captures** it
    (:func:`~causalab.neural.shared.head.capture_spec`): an ``lm_head`` read
    at named positions is served from ``ln_final``, so that is what the
    shared pass stores for it — ``[rows, seq, d_model]``, not the whole
    vocabulary — and what a later point's lookup asks for.
    """
    union: dict[str, dict[str, SiteSpec]] = {}
    for doc, plan in zip(docs, plans):
        for group in plan.groups:
            wanted = union.setdefault(group.digest, {})
            for tap in group.taps:
                read = doc.reads[tap.read]
                if generated_budget(doc, read.pos) is not None:
                    continue
                spec = capture_spec(doc, group.model, group.input, tap.read)
                wanted[json.dumps(spec_identity(spec), sort_keys=True)] = spec
    return {digest: tuple(specs.values()) for digest, specs in union.items()}


def campaign_cache(
    docs: Sequence[Document], plans: Sequence[PointPlan]
) -> ForwardCache:
    """The one :class:`ForwardCache` a campaign runs against: the tap union
    per digest (§3) and the prefix plans per digest (§4 "Resume").

    The prefix arithmetic is read off the **interned** groups, whose taps are
    the union over every sharer: the block a shared pass may start at is the
    shallowest any of them taps, exactly as the block it may stop after is
    the deepest. ``wanted_prefix_depths`` inverts that map by prefix
    identity, so the first pass over an input's rows — whichever model runs
    it — knows every depth a later intervened model will want.
    ``prefix_owed`` is that prefix's lifetime, in the shape of ``owed``: per
    (identity, depth), how many group instances across the points may still
    start from it — every one whose interned ``resume_at`` reaches the depth.
    """
    prefix_plans: dict[str, PrefixPlan] = {}
    wanted: dict[str, set[int]] = {}
    for group in interned_groups(plans):
        depth = group.resume_at
        prefix_plans[group.digest] = PrefixPlan(
            base_digest=group.base_digest,
            resume_at=depth,
            write_depth=group.write_depth,
        )
        if depth > 0:
            wanted.setdefault(group.base_digest, set()).add(depth)
    prefix_owed: Counter[tuple[str, int]] = Counter()
    for plan in plans:
        for group in plan.groups:
            reach = prefix_plans[group.digest].resume_at
            for depth in wanted.get(group.base_digest, ()):
                if depth <= reach:
                    prefix_owed[(group.base_digest, depth)] += 1
    return ForwardCache(
        wanted=_tap_union(docs, plans),
        prefix_plans=prefix_plans,
        wanted_prefix_depths={base: frozenset(v) for base, v in wanted.items()},
        prefix_owed=dict(prefix_owed),
        owed=dict(Counter(group.digest for plan in plans for group in plan.groups)),
    )


def execute_request(
    request: ExecutionRequest,
    *,
    engine_name: str,
    executor_factory: Callable[
        [Document, ExecutionRequest, Mapping[str, Any], "Interning | None"],
        ExecutorSurface,
    ],
    train_runner: TrainRunner | None = None,
    intern_forwards: bool = False,
) -> RunResult:
    """Run one :class:`ExecutionRequest` through one engine's executors.

    ``train_runner`` is the engine's train loop, cohort-shaped
    (:data:`TrainRunner`); an engine without one (its ``grad`` capability
    absent, so routing never sends it a ``train`` document) refuses loudly if
    a train document reaches it anyway.

    ``intern_forwards`` says this engine's executor consults the shared
    :class:`~causalab.neural.shared.executor_base.ForwardCache` (§3), so
    :attr:`~causalab.protocol.engine.RunResult.forwards` reports what the run
    paid. An engine that has not claimed the interning leaves it False and
    reports 0 — "not measured" rather than a number it did not count.

    **Order** (§4 "Cohorts"). The campaign runs in two phases. First, cohort
    by cohort (:func:`~causalab.protocol.plan.fit_cohorts`), every point of
    the cohort is prepared — its executor built, its ledger checked — and the
    cohort's fits run together through ``train_runner``. Then every point is
    finished in **point order**: its own whole-role passes, metrics and saved
    entries. Nothing a point writes depends on which cohort it fitted in, so
    the outputs are the per-point loop's; only the fits' inner passes are
    shared.
    """
    # The whole campaign is planned before anything runs, because §3's
    # interning is a property of the point *set*: a forward group can only be
    # shared once you know which other points share it, and the union of taps
    # it must capture only exists across all of them.
    # Planned and run in the execution form: a band site (§2.4 ``layers``) is
    # lowered to its per-layer members here, once, so the plans' taps, the
    # campaign's tap union and every executor name the same reads and sites
    # (`lower_bands` is idempotent; the executor lowers again for callers that
    # build one directly). What a band read has no member for — a save, a
    # metric — is refused here, before any forward.
    docs = tuple(lower_bands(parse_document(point_raw)) for point_raw in request.points)
    plans = campaign_plans(docs, request.canonical) if intern_forwards else ()
    # `owed` (inside `campaign_cache`) is the same count `interned_groups`
    # merges: how many point groups key into each digest. It bounds a
    # capture's lifetime to its sharers.
    cache = campaign_cache(docs, plans) if intern_forwards else ForwardCache()
    # which points may fit together: the same rows on the same realization
    # (the data identity the group digests carry); without a planned
    # campaign there is no such identity and every point stands alone
    cohorts = (
        fit_cohorts(
            docs,
            [_data_identity(doc, form) for doc, form in zip(docs, request.canonical)],
        )
        if intern_forwards
        else tuple((i,) for i in range(len(docs)))
    )

    prepared: list[_Prepared | None] = [None] * len(docs)
    for cohort in cohorts:
        if all(docs[i].train is None for i in cohort):
            # nothing to fit: built when its turn comes, in point order below,
            # so the store it starts against is what the points before it left
            continue
        members = [
            _prepare_point(
                docs[i],
                request,
                coords=request.coords[i],
                point_digest=request.digests[i],
                canonical=request.canonical[i],
                executor_factory=executor_factory,
                interning=(
                    Interning(
                        digests={
                            (group.model, group.input): group.digest
                            for group in plans[i].groups
                        },
                        cache=cache,
                    )
                    if intern_forwards
                    else None
                ),
            )
            for i in cohort
        ]
        fits = [member for member in members if member.doc.train is not None]
        if fits:
            if train_runner is None:
                raise ProtocolError(
                    "P4",
                    f"this document declares a train section, which the "
                    f"{engine_name!r} engine does not implement — its 'grad' "
                    "capability is absent, so routing should not have sent it "
                    "here",
                )
            outcomes = train_runner(
                [member.doc for member in fits],
                [member.executor for member in fits],
                request,
            )
            if len(outcomes) != len(fits):
                raise AssertionError(
                    f"the train loop returned {len(outcomes)} outcomes for a "
                    f"cohort of {len(fits)} points"
                )
            for member, outcome in zip(fits, outcomes):
                member.outcome = outcome
        for i, member in zip(cohort, members):
            prepared[i] = member

    tensor_files: dict[str, TensorFile] = {}
    metric_files: dict[str, MetricTable] = {}
    train_evals: list[Mapping[str, Any]] = []
    fit_diagnostics: list[Mapping[str, Any]] = []
    routing_mismatch: list[Mapping[str, Any]] = []
    summaries: list[Mapping[str, Any]] = []
    cells: list[Resolution] = []
    for i in range(len(prepared)):
        member = prepared[i]
        if member is None:  # a point that fits nothing
            member = _prepare_point(
                docs[i],
                request,
                coords=request.coords[i],
                point_digest=request.digests[i],
                canonical=request.canonical[i],
                executor_factory=executor_factory,
                interning=(
                    Interning(
                        digests={
                            (group.model, group.input): group.digest
                            for group in plans[i].groups
                        },
                        cache=cache,
                    )
                    if intern_forwards
                    else None
                ),
            )
        summaries.append(
            _execute_point(
                member,
                request,
                tensor_files=tensor_files,
                metric_files=metric_files,
                train_evals=train_evals,
                fit_diagnostics=fit_diagnostics,
                routing_mismatch=routing_mismatch,
                cells=cells,
            )
        )
        # the point is finished: its executor — the eval executor, the read
        # values, the frames it holds — dies here as it did in the per-point
        # loop, not at the end of the request
        prepared[i] = None
    first_doc = docs[0]
    first_realization = canonical_model(first_doc.raw["model"])
    attention_backends = {doc.model.attn_implementation for doc in docs}
    # implementation requirements the points' addresses imposed (§7.3, e.g.
    # "attn_eager") — execution metadata beside the engine name, never
    # canonical form: the documents and their digests are implementation-blind
    applied = sorted(
        {
            requirement
            for summary in summaries
            for requirement in summary.get("implementations", ())
        }
    )
    identity_base = {
        "produced_by": request.document_digest,
        "model_key": str(first_doc.model.key),
        "model_revision": str(first_doc.model.revision),
        "model_dtype": str(first_realization["dtype"]),
        "model_quantization": first_realization.get("quantization"),
        # A backend sweep has no single file-level selection. Each tensor
        # entry carries its own choice, just as fitted entries do below.
        "model_attn_implementation": (
            first_realization.get("attn_implementation")
            if len(attention_backends) == 1
            else None
        ),
        "engine": engine_name,
        **({"implementations": ",".join(applied)} if applied else {}),
        # `runtime_identity().short_revision`, not a `git rev-parse` here: the
        # field keeps its shape — a short hex string — and loses its ability to
        # say "unknown". An install that records no revision still has a tree
        # digest, so the value always identifies content.
        "commit": runtime_identity().short_revision,
    }
    files = write_outputs(
        request.output_dir,
        tensor_files,
        metric_files,
        identity_base=identity_base,
        train_evals=train_evals,
        fit_diagnostics=fit_diagnostics,
        routing_mismatch=routing_mismatch,
    )
    # every write member of every point fired the count its kind declares
    # (§4 "Fires") — the executor refused the run otherwise, before this
    # line; the counts go into the receipt once, after the whole campaign,
    # so a refused run's receipt records no subset
    record_fires(
        request.output_dir,
        {
            str(summary["point"]): summary["fires"]
            for summary in summaries
            if summary.get("fires")
        },
    )
    # the bounds the run measured when none was authored (§8 `fit_rows`,
    # `batch_rows`): into the receipt's `execution` block, as the numbers to pin
    record_measured_bounds(request.output_dir, summaries)
    # the ragged-write geometry the points landed under (§5 rule 19):
    # into the same block, only when some write ran under a non-`refuse`
    # policy — a receipt that authors none is byte-identical to before
    record_ragged_geometry(request.output_dir, summaries)
    return RunResult(
        files=files,
        summaries=tuple(summaries),
        forwards=len(cache.executed),
        cells=tuple(cells),
    )


@dataclasses.dataclass
class _Prepared:
    """One point between its two phases (see :func:`execute_request`): built
    and, if it trains, fitted — its own passes not yet run."""

    doc: Document
    coords: Mapping[str, Any]
    point_digest: str
    canonical: Mapping[str, Any]
    executor: ExecutorSurface
    interning: "Interning | None"
    #: the location ledger (§6), only when the document saves one
    ledger: Any
    #: the attention backend the loaded model runs — observed at load, stamped
    #: on what the point saves beside the authored one
    runtime_attention: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    outcome: TrainOutcome | None = None


def _prepare_point(
    doc: Document,
    request: ExecutionRequest,
    *,
    coords: Mapping[str, Any],
    point_digest: str,
    canonical: Mapping[str, Any],
    executor_factory: Callable[
        [Document, ExecutionRequest, Mapping[str, Any], "Interning | None"],
        ExecutorSurface,
    ],
    interning: "Interning | None",
) -> _Prepared:
    if doc.train is not None:
        # §5 rule 22's cross-table refusal, before an executor exists: a fit
        # whose training rows and `train.eval.split` rows share a prompt is
        # refused here, so no engine's forward — and no minibatch — runs on it
        check_fit_splits(doc, request.env.datasets)
    executor = executor_factory(doc, request, coords, interning)
    config = getattr(
        getattr(getattr(executor, "bundle", None), "model", None), "config", None
    )
    loaded_backend = getattr(config, "_attn_implementation", None)
    # Runtime observation is inherited, but compatibility remains authored.
    runtime_attention = build_artifact_identity(
        loaded_attn_implementation=loaded_backend
    )
    # the base table's scoring identity against the document's `match` modes
    # (§2.2, §2.10): refused before any forward, recorded in the run receipt
    base_ref = doc.data["base"].dataset
    if isinstance(base_ref, str):
        record_scoring(request.output_dir, base_ref, executor.check_scoring())
    # the location ledger (§6), only when the document saves one — resolved
    # here, before any forward, from the encoded batch (`protocol/ledger.py`)
    ledger = point_ledger(executor, doc)
    return _Prepared(
        doc=doc,
        coords=coords,
        point_digest=point_digest,
        canonical=canonical,
        executor=executor,
        interning=interning,
        ledger=ledger,
        runtime_attention=runtime_attention,
    )


def rank_records(
    stages: Mapping[str, Any], point_digest: str, coords: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The rows a ``rank`` save (§2.12) carries for one point: one per unit of
    every gate the point built — trained or loaded — with the unit's ``theta``,
    its position in the gate's ranking (``0`` = kept first), whether the
    eval-mode split keeps it, the map that split is read through, and the
    ``top_k`` the cut was made at (``null`` under the map's own threshold).
    Units are flat indices into ``theta``, so on a grouped gate a row is a head
    or an ``(expert, neuron)`` entry, never a coordinate; on a position gate
    (§2.5 ``axis``) a row is an addressed token position, and ``axis``
    records which (``null`` for a feature gate). A gate in a budget
    pool (§2.5 ``pool``) also carries the pool's name and the unit's position in
    the **pooled** ranking (``pool_rank``), the order the pooled cut is made in;
    both are ``null`` otherwise. The point's provenance and coordinates ride
    along as on every table."""
    import torch

    from causalab.neural.shared.featurizers import Gate

    rows: list[dict[str, Any]] = []
    for name in sorted(stages):
        stage = stages[name]
        if not isinstance(stage, Gate):
            continue
        with torch.no_grad():
            theta = stage.theta.detach().float().view(-1).tolist()
            rank = stage.rank().view(-1).tolist()
            hard = stage.hard_mask().view(-1).tolist()
            pooled = (
                stage.pool.member_rank(stage).view(-1).tolist()
                if stage.pool is not None
                else [None] * len(theta)
            )
        for unit, (value, position, kept, pool_rank) in enumerate(
            zip(theta, rank, hard, pooled)
        ):
            rows.append(
                {
                    "featurizer": name,
                    "unit": unit,
                    "theta": value,
                    "rank": int(position),
                    "hard": bool(kept),
                    "parametrization": stage.parametrization,
                    # §2.5 axis: which axis `unit` indexes — as `pool` says what
                    # `pool_rank` ranks against
                    "axis": stage.axis,
                    "top_k": stage.top_k,
                    "pool": stage.pool.name if stage.pool is not None else None,
                    "pool_rank": int(pool_rank) if pool_rank is not None else None,
                    "point": point_digest,
                    "coords": dict(coords),
                }
            )
    return rows


def _execute_point(
    member: _Prepared,
    request: ExecutionRequest,
    *,
    tensor_files: dict[str, TensorFile],
    metric_files: dict[str, MetricTable],
    train_evals: list[Mapping[str, Any]],
    fit_diagnostics: list[Mapping[str, Any]],
    routing_mismatch: list[Mapping[str, Any]],
    cells: list[Resolution],
) -> Mapping[str, Any]:
    doc, executor, interning, ledger = (
        member.doc,
        member.executor,
        member.interning,
        member.ledger,
    )
    coords, point_digest, canonical = (
        member.coords,
        member.point_digest,
        member.canonical,
    )
    runtime_attention = member.runtime_attention
    # one result cell per save entry (spec §4.1): what this point's run
    # resolved and what it could not. Appended to the campaign's list here,
    # and the unavailable ones repeated in this point's summary.
    point_cells: list[Resolution] = []
    trained_stages: dict[str, Any] = {}
    fit_forwards: Mapping[str, int] | None = None
    # what this point's passes resumed from a cached prefix instead of
    # recomputing (§4 "Resume"): the fit's, tallied per point by the loop,
    # plus its own, a difference over the campaign store
    fit_resumed: tuple[int, ...] = ()
    fit_rows: int | None = None
    fit_rows_shrinks = 0
    resumed_before = len(interning.cache.resumed) if interning is not None else 0
    if doc.train is not None:
        outcome = member.outcome
        if outcome is None:
            raise AssertionError("a train point reached its own passes unfitted")
        # what the fit's inner passes paid for its constant groups, and what
        # the store handed them instead — the saving §4 "Fits" promises, per
        # point. Reported in this point's summary, which `RunResult` returns
        # to the caller and nothing writes to disk.
        fit_forwards = outcome.fit_forwards
        fit_resumed = outcome.resumed
        fit_rows = outcome.fit_rows
        fit_rows_shrinks = outcome.fit_rows_shrinks
        trained_stages = dict(outcome.stages)
        if outcome.eval_score is not None:
            train_evals.append(
                outcome.eval_score.as_record(point=point_digest, coords=coords)
            )
        if (
            outcome.diagnostics
            or outcome.controls
            or outcome.anneals
            or outcome.phases
            or outcome.constraints
            or outcome.draws
        ):
            fit_diagnostics.append(
                {
                    "point": point_digest,
                    "coords": dict(coords),
                    "featurizers": {
                        name: dict(values)
                        for name, values in outcome.diagnostics.items()
                    },
                    **(
                        {
                            "controls": {
                                target: dict(values)
                                for target, values in outcome.controls.items()
                            }
                        }
                        if outcome.controls
                        else {}
                    ),
                    **(
                        {
                            "anneals": {
                                target: dict(values)
                                for target, values in outcome.anneals.items()
                            }
                        }
                        if outcome.anneals
                        else {}
                    ),
                    **(
                        {"phases": [dict(phase) for phase in outcome.phases]}
                        if outcome.phases
                        else {}
                    ),
                    **(
                        {
                            "constraints": {
                                name: dict(values)
                                for name, values in outcome.constraints.items()
                            }
                        }
                        if outcome.constraints
                        else {}
                    ),
                    **(
                        {
                            "draws": {
                                role: dict(values)
                                for role, values in outcome.draws.items()
                            }
                        }
                        if outcome.draws
                        else {}
                    ),
                }
            )
    # Order is load-bearing. The point executor is counted and grad-free, so
    # `_may_intern` lets it read from and publish to the store for every
    # model, the trained one included; that is safe only because no forward
    # of it runs until the fit above has finished and the stages are final.
    executor.run_all()
    # what every write through an expert-keyed gate found about the pair's
    # routing (executor_base._align_by_expert): the base slots whose expert
    # the operand's side never activated, per layer and example. Filled by
    # the writes the full-data pass above landed, so it describes the rows
    # the metric tables describe
    for (write, layer, example), (mismatched, slots) in sorted(
        (getattr(executor, "routing_mismatch", None) or {}).items()
    ):
        routing_mismatch.append(
            {
                "point": point_digest,
                "coords": dict(coords),
                "write": write,
                "layer": layer,
                "example": example,
                "mismatched": mismatched,
                "slots": slots,
            }
        )
    metric_values: dict[str, list[Any]] = {}
    windowed: dict[str, _Windowed] = {}
    #: metrics whose read is an unavailable cell, and so are they (§4.1)
    inherited: dict[str, Unavailable] = {}
    for qname, metric in doc.metrics.items():
        of_name = str(metric.of)
        target_name = (
            str(metric.fields["target"])
            if metric.kind in READ_TARGET_METRIC_KINDS
            else None
        )
        if executor.is_generated(of_name):
            # a continuation read addresses as many positions as the row
            # generated, so its metric reduces per step and reports which
            # steps it saw (§2.3, §2.10)
            windowed[qname] = _Windowed(
                values=executor.generated_metric(metric),
                steps=(
                    None
                    if str(metric.kind) in WHOLE_WINDOW_METRIC_KINDS
                    else executor.addressed_steps(of_name)
                ),
                matched=[bool(steps) for steps in executor.addressed_steps(of_name)],
            )
            continue
        resolved_of = executor.resolution(of_name)
        key = cell_key(qname, coords)
        # the rows of the read(s) this metric reduces that aligned on nothing
        # (§4.1): each is an excluded measurement of *this* metric, under the
        # read's reason, keyed under the metric's cell (§2.10 "Eligibility")
        per_row = _row_exclusions(executor, qname, of_name, target_name, key)
        eligible = [i for i, cell in enumerate(per_row) if cell is None]
        if isinstance(resolved_of, Unavailable) and (not any(per_row) or not eligible):
            # the whole cell is unavailable — for a reason that is not any
            # row's (an `expert:` face the router sent no token), or because
            # every row failed to align: the metric inherits the cell — same
            # reason, the read's detail — rather than reducing a gather with
            # empty rows into a number
            inherited[qname] = unavailable(
                resolved_of.reason,
                f"metric {qname!r} reduces read {of_name!r}, which is unavailable: "
                + resolved_of.detail,
                key,
            )
            metric_values[qname] = []
            continue
        rows = executor.rows_for_metrics()
        if any(per_row):
            # some rows aligned and some did not: score the rows that did,
            # over exactly their positions, and put the typed `unavailable`
            # in each excluded row's place — the value is computed over the
            # eligible rows only, and the excluded ones are never averaged in
            of_dense = executor.dense_rows(of_name, eligible)
            target_dense = (
                executor.dense_rows(target_name, eligible)
                if target_name is not None
                else None
            )
            rows = [rows[i] for i in eligible]
        else:
            of_dense = executor.dense_value(of_name)
            target_dense = (
                executor.dense_value(target_name) if target_name is not None else None
            )
        values = compute_metric(
            metric,
            of_dense,
            rows,
            executor.bundle.tokenizer,
            target_value=target_dense,
            vocab_axis=metric_reads_vocabulary(doc, metric),
            denominator_key=key,
        )
        if any(per_row):
            scored = iter(values)
            values = [cell if cell is not None else next(scored) for cell in per_row]
        metric_values[qname] = values
    # every metric cell's eligibility record (§2.10): how many rows its
    # decision rule was evaluated over, of how many considered, the excluded
    # ones by reason — derived from the rows, and repeated on the cell and in
    # this point's summary so no consumer has to count the table again
    eligibility: dict[str, Eligibility] = {
        **{
            qname: Eligibility.of(values)
            for qname, values in metric_values.items()
            if qname not in inherited
        },
        **{qname: _windowed_eligibility(window) for qname, window in windowed.items()},
    }
    # every metric row is a base row (§2.2), so its label is the base row's
    labels = example_labels(executor.rows_for_metrics())
    for entry in doc.save:
        key = cell_key(entry.value, coords)
        if entry.kind == "trajectory":  # §2.12: the fit's checkpoints, one bundle
            point_cells.append(
                available({"file_path": entry.file_path, "kind": entry.kind}, key)
            )
            trajectory_file = tensor_files.setdefault(entry.file_path, TensorFile())
            for checkpoint in outcome.checkpoints:
                for fname, slots in checkpoint.slots.items():
                    # the same identity the featurizer's own bundle carries, so
                    # a checkpoint reloads through the same checks. `featurizer`
                    # and `step` are coordinates of the entry, not of the
                    # document (not in the digest): a fit that trains several
                    # featurizers photographs each of them at every step, and
                    # the name is what keeps `gate_3`'s theta from overwriting
                    # `gate_7`'s under one `theta[step=n]` key — a consumer
                    # names its own (`entry: {"featurizer": "gate_3", "step":
                    # 40}`); a one-featurizer consumer's `{"step": 40}` is
                    # already unique
                    stage = trained_stages[fname]
                    identity = {
                        **featurizer_identity(
                            doc,
                            fname,
                            _featurizer_site(doc, fname),
                            point_digest,
                            stage=stage,
                            group_map=getattr(stage, "groups", None),
                            canonical=canonical,
                        ),
                        **ledger_identity(ledger),
                    }
                    for slot, tensor in slots.items():
                        trajectory_file.add(
                            slot,
                            tensor,
                            {**coords, "featurizer": fname, "step": checkpoint.step},
                            label_entry=fname,
                            identity=identity,
                            record=checkpoint.record,
                        )
                    trajectory_file.record_common(identity)
            continue
        if entry.kind == "rank":  # §2.12: every gate's units ordered by theta
            point_cells.append(
                available({"file_path": entry.file_path, "kind": entry.kind}, key)
            )
            table = metric_files.setdefault(entry.file_path, MetricTable())
            table.rows.extend(
                rank_records(
                    {**executor.stage_cache, **trained_stages}, point_digest, coords
                )
            )
            continue
        if entry.kind is not None:  # `location_ledger` (§2.12): a JSON table
            assert ledger is not None
            point_cells.append(
                available({"file_path": entry.file_path, "kind": entry.kind}, key)
            )
            table = metric_files.setdefault(entry.file_path, MetricTable())
            table.rows.extend(ledger_records(ledger, point_digest, coords))
            continue
        if entry.value in doc.metrics:
            point_cells.append(
                inherited.get(entry.value)
                or _metric_cell(
                    entry.value,
                    entry.file_path,
                    eligibility[entry.value],
                    metric_values.get(entry.value, []),
                    key,
                )
            )
            table = metric_files.setdefault(entry.file_path, MetricTable())
            spec = doc.metrics[entry.value]
            # the record's identity (§2.10): authored on the metric, or the
            # kind's own — repeated on every row, never a digest field
            identity = metric_record_identity(
                str(spec.kind), unit=spec.unit, estimand_version=spec.estimand_version
            )
            if entry.value in windowed:
                window = windowed[entry.value]
                table.add_windowed(
                    entry.value,
                    window.values,
                    coords,
                    point_digest,
                    identity=identity,
                    steps=window.steps,
                    matched=window.matched,
                    labels=labels,
                )
            else:
                table.add(
                    entry.value,
                    metric_values[entry.value],
                    coords,
                    point_digest,
                    identity=identity,
                    labels=labels,
                )
        elif entry.value in doc.reads:
            # the site and the data go on the entry too: a harvested
            # activation is bound to where it was read and to what was read,
            # and a consumer (a script step fitting a basis on it, then a
            # document starting a fit from that basis) has no other way to
            # prove the site agrees, or to record which data the basis saw
            read = doc.reads[entry.value]
            read_site = site_identity(doc, str(read.site))
            dataset = str(input_roles(doc)[str(read.input)].dataset)
            resolved = executor.resolution(entry.value)
            cell: Resolution = (
                resolved
                if isinstance(resolved, Unavailable)
                else available({"file_path": entry.file_path, "key": key}, key)
            )
            point_cells.append(cell)
            tensor_files.setdefault(entry.file_path, TensorFile()).add(
                entry.value,
                executor.read_value(entry.value),
                coords,
                reduce=entry.reduce,
                identity={
                    "produced_by": point_digest,
                    **runtime_attention,
                    **build_artifact_identity(
                        model_attn_implementation=doc.model.attn_implementation,
                    ),
                    "trained_on": dataset,
                    "trained_on_digest": request.env.datasets.digest(dataset),
                    **(
                        {"site": json.dumps(read_site, sort_keys=True)}
                        if read_site
                        else {}
                    ),
                    # nothing for an available cell; the four status fields
                    # for an unavailable one — so a result written before
                    # the value existed is byte-identical (spec §4.1)
                    **cell_record(cell),
                    # the ledger's digest, only when one was emitted (§8)
                    **ledger_identity(ledger),
                },
            )
        else:  # a trained featurizer bundle
            stage = trained_stages.get(entry.value)
            if stage is None:
                raise ProtocolError(
                    "P2", f"featurizer {entry.value!r} was not trained this run"
                )
            point_cells.append(
                available(
                    {"file_path": entry.file_path, "featurizer": entry.value}, key
                )
            )
            bundle_file = tensor_files.setdefault(entry.file_path, TensorFile())
            identity = {
                **featurizer_identity(
                    doc,
                    entry.value,
                    entry.site,
                    point_digest,
                    stage=stage,
                    group_map=getattr(stage, "groups", None),
                    canonical=canonical,
                ),
                **runtime_attention,
                **ledger_identity(ledger),  # the ledger stamp, recorded (§8)
            }
            for slot, param in stage.slot_params().items():
                # per entry, not per file: a swept fit writes one file from
                # many points, and only the entry table can say which point
                # produced which rotation (§8)
                bundle_file.add(
                    slot,
                    param.detach(),
                    coords,
                    label_entry=entry.value,
                    identity=identity,
                )
            bundle_file.record_common(identity)
    # the windows this point's passes resumed from a cached prefix (§4
    # "Resume"), by the block they started at: the fit's forwards this point
    # took part in, then its own
    resumed = [
        *fit_resumed,
        *(interning.cache.resumed[resumed_before:] if interning is not None else []),
    ]
    # per forward group with writes, each member's fire count (§4 "Fires") —
    # what this point's own pass counted, or the counts of the pass that
    # produced the captures it was served; an engine whose executor keeps no
    # tally records nothing rather than zeroes
    fires = {
        group_label(model, input_role): dict(counts)
        for (model, input_role), counts in sorted(
            (getattr(executor, "fires", None) or {}).items()
        )
        if counts
    }
    ragged = {
        f"{model}/{write}": dict(geometry)
        for (model, write), geometry in sorted(
            (getattr(executor, "ragged_geometry", None) or {}).items()
        )
    }
    cells.extend(point_cells)
    excluded = {
        cell.denominator_key: cell.record()
        for cell in point_cells
        if isinstance(cell, Unavailable)
    }
    return {
        "point": point_digest,
        **runtime_attention,
        "coords": dict(coords),
        "metrics": {
            name: _summary_stat(values) for name, values in metric_values.items()
        },
        # every metric cell's `n_eligible` / `n_considered` (§2.10), the
        # excluded rows by reason when there were any — the aggregate cell's
        # denominator, beside the aggregate
        **(
            {"eligibility": {name: e.as_record() for name, e in eligibility.items()}}
            if eligibility
            else {}
        ),
        **({"fit_forwards": fit_forwards} if fit_forwards is not None else {}),
        # the rows-per-grad-forward bound the fit ran under (§8 `fit_rows`):
        # what an author pins to reproduce a measured one
        **({"fit_rows": fit_rows} if fit_rows is not None else {}),
        **({"fit_rows_shrinks": fit_rows_shrinks} if fit_rows_shrinks else {}),
        **(
            {"prefix_reuse": {"resumed": len(resumed), "blocks_skipped": sum(resumed)}}
            if resumed
            else {}
        ),
        **({"fires": fires} if fires else {}),
        # per write that landed ragged under a declared policy (§5 rule 19):
        # the policy, the per-row widths and the width buckets the executor
        # recorded before any forward — the receipt's `execution.ragged`
        **({RAGGED_KEY: ragged} if ragged else {}),
        **(
            {"implementations": sorted(getattr(executor, "applied_requirements", ()))}
            if getattr(executor, "applied_requirements", None)
            else {}
        ),
        # only when something was excluded: a point whose every cell measured
        # summarizes exactly as before
        **({"unavailable": excluded} if excluded else {}),
    }


def _summary_stat(values: list[Any]) -> Any:
    """The aggregate: the mean over the **eligible** numeric rows. An excluded
    row is an ``Unavailable``, not a number, so it is never in the
    denominator here (§2.10 "Eligibility")."""
    numeric = [v for v in values if isinstance(v, (int, float))]
    if numeric:
        return sum(numeric) / len(numeric)
    return f"{len(values)} rows"


def _row_exclusions(
    executor: ExecutorSurface,
    qname: str,
    of_name: str,
    target_name: str | None,
    key: str,
) -> list[Unavailable | None]:
    """Per base row, the ``unavailable`` a metric's row is when the read it
    reduces (or, for ``kl``, the read it compares against) aligned on nothing
    for that row (§4.1) — re-keyed under the metric's own cell and saying
    which read — else ``None``. The row-level form of "a metric over an
    unavailable read inherits the cell"."""
    per_row = list(executor.row_resolutions(of_name))
    if target_name is not None:
        per_row = [
            of_cell or target_cell
            for of_cell, target_cell in zip(
                per_row, executor.row_resolutions(target_name), strict=True
            )
        ]
    return [
        None
        if cell is None
        else unavailable(
            cell.reason,
            f"metric {qname!r} reduces a read unavailable on this row: {cell.detail}",
            key,
        )
        for cell in per_row
    ]


def _windowed_eligibility(window: _Windowed) -> Eligibility:
    """A continuation metric's eligibility, per example: a row that addressed
    nothing (``matched: false`` — the anchor's value occurred nowhere in what
    it generated) is excluded under ``alignment_missing``; a row whose every
    position scored is eligible; a row with an excluded position is counted
    under that position's reason."""
    per_example: list[Any] = []
    for values, matched in zip(window.values, window.matched, strict=True):
        if not matched:
            per_example.append(
                unavailable("alignment_missing", "the row addressed nothing", "")
            )
            continue
        excluded = next((v for v in values if isinstance(v, Unavailable)), None)
        per_example.append(excluded if excluded is not None else values)
    return Eligibility.of(per_example)


def _metric_cell(
    name: str,
    file_path: str,
    counts: Eligibility,
    values: list[Any],
    key: str,
) -> Resolution:
    """The result cell of one metric at one point (§4.1): available — with
    its ``n_eligible`` / ``n_considered`` in the mapping — when at least one
    row was eligible, else the ``unavailable`` its rows all are, under the
    first excluded row's reason. A metric over zero rows is available: nothing
    was excluded."""
    first = next((v for v in values if isinstance(v, Unavailable)), None)
    if counts.n_eligible == 0 and first is not None:
        return unavailable(
            first.reason,
            f"metric {name!r}: all {counts.n_considered} rows are excluded "
            f"measurements — {first.detail}",
            key,
        )
    return available(
        {"file_path": file_path, "metric": name, **counts.as_record()}, key
    )


def _featurizer_site(doc: Document, name: str) -> str | None:
    """The site a trained featurizer's own ``save`` entry restates (§2.12) —
    what its bundle's identity is stamped with, and so what a checkpoint of
    it is stamped with too. Every trained featurizer has one (rule 10)."""
    for entry in doc.save:
        if entry.kind is None and entry.value == name and entry.site is not None:
            return entry.site
    return None


def featurizer_identity(
    doc: Document,
    name: str,
    site_name: str | None,
    point_digest: str,
    *,
    stage: Any = None,
    group_map: tuple[int, int] | None = None,
    canonical: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """The ArtifactIdentity a trained featurizer bundle stamps (§8): what the
    document implies about the fit, plus whatever the fitted ``stage`` knows
    that the document only pointed at — a ``subspace`` seeded from a saved
    basis records that basis (``Stage.identity_fields``).

    ``group_map`` is the ``(groups, group_width)`` a grouped gate was built
    over — read off the trained stage, since the document authors only the
    group *kind* and the map is derived from the site (§2.5).

    ``group_map`` is the ``(groups, group_width)`` a grouped gate was built
    over — read off the trained stage, since the document authors only the
    group *kind* and the map is derived from the site (§2.5).

    ``trained_on`` is the ref the fit read on ``base`` — a human-readable
    name. ``trained_on_digest``, stamped when the point's ``canonical`` form
    is given, is what that name resolved to: the sha256 of the ``table_bytes``
    of the rows the ref selected, as canonicalized (§2.2) — the same digest
    the point digest and the forward-group identity carry, so a reader can
    tell two fits on same-named, different tables apart from the header
    alone. It is a record, not a load-time expectation: an apply document
    legitimately reads a different split than the fit trained on.
    """
    spec = doc.featurizers[name]
    site = site_identity(doc, site_name)
    base = doc.data["base"]
    trained_on = base.dataset if not isinstance(base, tuple) else base[0].dataset
    trained_on_digest = (
        _role_digests(doc, canonical)[
            "base" if not isinstance(base, tuple) else "base[0]"
        ]
        if canonical is not None
        else None
    )
    realization = canonical_model(doc.raw["model"])
    return build_artifact_identity(
        **(stage.identity_fields() if stage is not None else {}),
        produced_by=point_digest,
        model_key=str(doc.model.key),
        model_revision=str(doc.model.revision),
        model_dtype=str(realization["dtype"]),
        model_quantization=realization.get("quantization"),
        model_attn_implementation=realization.get("attn_implementation"),
        site=site,
        k=spec.k if isinstance(spec.k, int) else None,
        # a gate stamps its effective map, `sigmoid` when unauthored: the hard
        # split a replay must reproduce depends on it (§2.5), and a bundle
        # fitted before the field existed is read as sigmoid by both checks
        parametrization=spec.parametrization
        if isinstance(spec.parametrization, str)
        else ("sigmoid" if spec.kind == "gate" else None),
        group=spec.group if isinstance(spec.group, str) else None,
        group_map=list(group_map) if group_map is not None else None,
        dtype=spec.dtype if isinstance(spec.dtype, str) else "fp32",
        trained_on=str(trained_on),
        trained_on_digest=trained_on_digest,
    )
