"""``causalab dry-run`` — everything a run decides before weights load,
resolved and reported (spec §9). The report answers a twelve-item pre-flight
checklist (listed on :class:`DryRunReport`) plus a per-site report.

A run decides most of what it does before it touches a model: what the
document composes to, which tables it reads, how many points the sweep
expands to and how many forwards they owe, what the engine must be able to
do, and — from the registry's static entry alone — whether every site it
names exists on the model, what shape and width it has and which engines
read or write it. Until now those answers were spread over ``validate``
(which refuses or says ``OK``), ``explain`` (which prints the plan) and the
run itself (which loads weights first and refuses afterwards). This module
collects them into one report, produced with no accelerator, no network and
no model cached: the compile is torch-free, and the model facts come from
:mod:`causalab.protocol.registry`.

**Re-raise versus report.** A document that does not *compile* — a parse
error, a §5 refusal, an unavailable site (``V4`` with the reason
``component_unavailable``) — is re-raised: there is nothing to report about
a document that is not valid, and the CLI prints ``refused: …`` and exits
``1`` exactly as ``validate`` does, plus the refusal's reason-coded record. A
document that compiles but whose *engine* falls short is **reported**, not
raised: :func:`dry_run` asks :func:`~causalab.protocol.compile.check_engine`
once per candidate engine and turns each refusal into a
``capability_shortfall`` diagnostic — the diagnostic kind the compiler
reserved for exactly this, and this module is its one producer.

**Undecided is a status, never a green.** The pure layer holds no
tokenizer, so per-row window counts and token widths, the answer tokens a
metric matches, and the counterfactual pair's validity are decided when the
run encodes its inputs; a dense entry without ``layer_types`` cannot say
which mixer a layer carries; controls live on the workflow. The report
names each of these under :attr:`DryRunReport.undecided` rather than
omitting it, so a tokenizer-time refusal is never mistaken for a dry run
that passed (spec §5).

Torch-free: imports only ``causalab.protocol``. Engines reach it as
constructed objects — constructing an engine loads no weights
(``PytorchHooksEngine.execute`` is where ``load_model`` is called) — and
:func:`dry_run` calls neither ``execute`` nor a loader.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, get_args

from causalab.protocol.compile import (
    CompiledProtocol,
    Diagnostic,
    check_engine,
    compile_protocol,
)
from causalab.protocol.engine import Engine, choose_engine
from causalab.protocol.errors import ProtocolError, ValidationError
from causalab.protocol.loader import check_data_columns, check_row_roles
from causalab.protocol.plan import interned_groups, plan_point
from causalab.protocol.registry import (
    COMPONENT_STREAMS,
    INTERIOR_ROWS,
    Inventory,
    ModelInfo,
    capability,
    component_shape,
    component_width,
    inventory,
    predicate_holds,
    unavailable_at_load,
)
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.schema import (
    LAYERLESS_COMPONENTS,
    MODEL_DTYPE_DEFAULT,
    Document,
    PositionSpec,
)

__all__ = [
    "SITE_STATUSES",
    "UNDECIDED_TOPICS",
    "CompositionReport",
    "DataReport",
    "DryRunReport",
    "EngineReport",
    "ForwardsReport",
    "ModelReport",
    "OutputReport",
    "PointsReport",
    "ReadoutReport",
    "Refusal",
    "ShardsReport",
    "SiteReport",
    "SiteStatus",
    "Undecided",
    "UndecidedTopic",
    "dry_run",
    "shard_count",
    "site_report",
]


# --------------------------------------------------------------------------- #
# the two closed vocabularies of the report (tabulated in spec §9)
# --------------------------------------------------------------------------- #

#: What the registry entry can say about one site the document names.
#: ``available`` — the tensor exists, its shape and width are known and the
#: capability row says who reads and writes it; ``undecided`` — the entry
#: cannot decide a fact the run will (which mixer the layer carries, a
#: module-tree predicate, a measured address); ``refused`` — the row or the
#: entry says there is no such tensor (``V4``, ``component_unavailable``) or
#: the site's ``head`` names an axis the component has none of.
SiteStatus = Literal["available", "undecided", "refused"]
SITE_STATUSES: tuple[SiteStatus, ...] = get_args(SiteStatus)

#: The facts a dry run leaves to the run, by name — each is a line of the
#: report, never an omission. ``tokenization``: window counts, token widths
#: and answer tokens (sec. 2.3, sec. 2.10); ``pair_validity``: the
#: counterfactual pair's checks over rows and tokenizer (sec. 2.2);
#: ``controls``: declared one layer up, on the workflow; ``stream_at_layer``:
#: which mixer a layer carries on an entry without ``layer_types``;
#: ``module_tree``: a predicate or an address only the loaded modules decide;
#: ``inventory``: the per-layer inventory needs ``layer_types``; ``engines``:
#: which engine serves, when none was handed in; ``model``: the model key is
#: swept and the site facts are the first point's.
UndecidedTopic = Literal[
    "tokenization",
    "pair_validity",
    "controls",
    "stream_at_layer",
    "module_tree",
    "inventory",
    "engines",
    "model",
]
UNDECIDED_TOPICS: tuple[UndecidedTopic, ...] = get_args(UndecidedTopic)


# --------------------------------------------------------------------------- #
# the records
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Refusal:
    """One refusal as a record: the code, the §5 rule (number and slug) when
    it is one, the field, the :data:`~causalab.protocol.errors.REASON_CODES`
    entry when the refusal carries one, and the rendered text."""

    code: str
    rule: int | None
    rule_id: str | None
    path: str | None
    reason: str | None
    message: str

    @classmethod
    def from_error(cls, err: ProtocolError) -> Refusal:
        return cls(
            code=err.code,
            rule=getattr(err, "rule", None),
            rule_id=getattr(err, "rule_id", None),
            path=err.path,
            reason=err.reason,
            message=str(err),
        )

    def render(self) -> str:
        """The record on one line, after the rendered text: what the CLI
        prints so the reason code is visible, not only the rule."""
        where = f" at {self.path}" if self.path else ""
        rule = f" ({self.rule_id})" if self.rule_id else ""
        reason = f", reason {self.reason}" if self.reason else ""
        return f"code {self.code}{rule}{where}{reason}"


@dataclasses.dataclass(frozen=True)
class CompositionReport:
    """Checklist item 1 — method/application composition: the campaign
    digest, the title, and the overrides that were applied on the way
    (``--set`` / a step's ``set``)."""

    digest: str
    title: str | None
    overrides: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class DataReport:
    """Checklist item 2 — one resolved dataset ref (a split is a ref): the
    roles that read it, its content digest and its columns."""

    ref: str
    roles: tuple[str, ...]
    digest: str
    columns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ModelReport:
    """Checklist item 5 — the model configuration, from the registry entry
    (no config fetched): the realization the document names and the entry's
    static widths. ``layer_pattern`` is the entry's ``layer_types`` when it
    declares one — what decides the stream at a layer offline."""

    key: str
    revision: str
    dtype: str
    quantization: str | None
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    family: str | None
    layer_pattern: tuple[str, ...] | None


@dataclasses.dataclass(frozen=True)
class PointsReport:
    """Checklist item 7 — sweep expansion: the point count and the axes
    (id, number of values) it came from."""

    n: int
    axes: tuple[tuple[str, int], ...]


@dataclasses.dataclass(frozen=True)
class ForwardsReport:
    """Checklist item 12, the forward half — what one point plans
    (``per_point``, the first point's plan) and what the campaign owes once
    §3's content dedup interns shared forwards (``campaign``)."""

    per_point: int
    campaign: int


@dataclasses.dataclass(frozen=True)
class ShardsReport:
    """Checklist item 12, the shard half — ``count = ceil(n_points /
    shard_size)`` for a caller-given ``shard_size``, ``None`` when none was
    given. A new derivation over a caller input: the run shards on
    ``--points START:STOP`` and computes no count of its own."""

    n_points: int
    shard_size: int | None
    count: int | None


@dataclasses.dataclass(frozen=True)
class EngineReport:
    """Checklist item 11, per candidate engine — whether it serves the
    campaign, what it lacks, and :func:`~causalab.protocol.compile.check_engine`'s
    refusal as a ``capability_shortfall`` diagnostic (reported, not raised)."""

    name: str
    serves: bool
    lacks: tuple[str, ...]
    shortfall: Diagnostic | None
    refusal: Refusal | None


@dataclasses.dataclass(frozen=True)
class SiteReport:
    """The per-site report — one site the document names, resolved from the registry
    entry alone: availability, tensor shape, width, head space, and who reads
    and writes it. ``layers`` is every layer the site takes across the points
    (a swept ``layer`` lists them all; a layer-less component lists none).
    ``writes`` is the mechanisms a write may use, ``None`` for a read-only
    row, with the row's ``why``."""

    name: str
    component: str
    layers: tuple[int, ...]
    head: int | None
    expert: Any
    stream: str | None
    shape: str | None
    width: int | None
    head_space: int | None
    reads: tuple[str, ...]
    writes: tuple[str, ...] | None
    why: str
    status: SiteStatus
    undecided: tuple[str, ...]
    refusal: Refusal | None

    def __post_init__(self) -> None:
        if self.status not in SITE_STATUSES:
            raise AssertionError(
                f"unknown site status {self.status!r}; expected one of {SITE_STATUSES}"
            )


@dataclasses.dataclass(frozen=True)
class ReadoutReport:
    """Checklist item 9 — one read: where it taps, and the metrics that
    reduce it."""

    name: str
    site: str
    model: str
    input: str
    metrics: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class OutputReport:
    """Checklist item 10 — one ``save`` entry: the value, its binding, the
    file it produces and, for a non-value entry, its kind."""

    value: str
    binding: str
    file_path: str
    kind: str | None


@dataclasses.dataclass(frozen=True)
class Undecided:
    """One fact the dry run leaves to the run, by topic, with the reason."""

    topic: UndecidedTopic
    detail: str

    def __post_init__(self) -> None:
        if self.topic not in UNDECIDED_TOPICS:
            raise AssertionError(
                f"unknown undecided topic {self.topic!r}; expected one of "
                f"{UNDECIDED_TOPICS}"
            )


@dataclasses.dataclass(frozen=True)
class DryRunReport:
    """Everything a run decides before weights load — the pre-flight
    checklist and the per-site report, as one frozen record.

    The twelve checklist items, and where each is answered:

    1. method/application composition — ``composition``;
    2. datasets and splits — ``data``;
    3. counterfactual invariants — ``undecided`` (``pair_validity``): the
       pair's checks need rows and a tokenizer; the ``--data`` pass checks
       column existence and row roles here (rules 4, 20, 25), a refusal of
       which is a ``refusals`` entry;
    4. tokenization and semantic positions — ``undecided``
       (``tokenization``), never a green;
    5. model configuration — ``model``;
    6. site names and expected tensor shapes — ``sites``, ``inventory``;
    7. sweep expansion — ``points``;
    8. controls — ``undecided`` (``controls``): declared on the workflow;
    9. readouts — ``readouts``;
    10. output schema — ``outputs``;
    11. required engine capabilities — ``capabilities``, ``engines``;
    12. estimated point and shard counts — ``points``, ``forwards``,
        ``shards``.

    ``refusals`` are the refusals the dry run *reports* rather than raises
    (no candidate engine serves; the ``--data`` pass); ``diagnostics`` are
    the compile's own; ``ok`` is "no refusals". A document that does not
    compile never becomes a report — :func:`dry_run` re-raises.
    """

    composition: CompositionReport
    data: tuple[DataReport, ...]
    model: ModelReport
    points: PointsReport
    forwards: ForwardsReport
    shards: ShardsReport
    capabilities: tuple[str, ...]
    engines: tuple[EngineReport, ...]
    sites: tuple[SiteReport, ...]
    inventory: Inventory | None
    readouts: tuple[ReadoutReport, ...]
    outputs: tuple[OutputReport, ...]
    refusals: tuple[Refusal, ...]
    undecided: tuple[Undecided, ...]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def ok(self) -> bool:
        return not self.refusals

    @property
    def undecided_topics(self) -> tuple[UndecidedTopic, ...]:
        """The topics left to the run, each once, in report order."""
        seen: list[UndecidedTopic] = []
        for item in self.undecided:
            if item.topic not in seen:
                seen.append(item.topic)
        return tuple(seen)


# --------------------------------------------------------------------------- #
# the derivations
# --------------------------------------------------------------------------- #


def shard_count(n_points: int, shard_size: int | None) -> int | None:
    """``ceil(n_points / shard_size)`` — the number of ``--points`` shards of
    at most ``shard_size`` points a campaign needs; ``None`` without a size.
    The arithmetic a scheduler script would otherwise do by hand."""
    if shard_size is None:
        return None
    if shard_size < 1:
        raise ValueError(f"shard_size must be a positive point count, got {shard_size}")
    return math.ceil(n_points / shard_size)


def site_report(
    name: str,
    component: str,
    info: ModelInfo,
    *,
    layers: Sequence[int] = (),
    head: int | None = None,
    expert: Any = None,
    stream: str | None = None,
) -> SiteReport:
    """The per-site report for one site, from the registry entry alone.

    Availability is what the row's predicates and the entry decide
    (:func:`~causalab.protocol.registry.unavailable_at_load`); shape, width
    and head space are :func:`~causalab.protocol.registry.component_shape`'s
    (a ``head`` narrows the width to one head's slice, or refuses on a
    component without a head axis); read and write support is the row's.
    What the entry cannot decide is named under ``undecided``: the stream at
    a layer for a stream-bound component on an entry without
    ``layer_types``, a predicate only the module tree answers, an
    attention-interior address the per-family tap table has not measured.
    A compiled document's sites are all ``available`` or ``undecided`` —
    the compile refused the rest — so ``refused`` is what a caller sees who
    asks about a site the compile has not yet seen.
    """
    row = capability(component)
    undecided: list[str] = []

    def refused(err: ProtocolError) -> SiteReport:
        return SiteReport(
            name=name,
            component=component,
            layers=tuple(layers),
            head=head,
            expert=expert,
            stream=stream,
            shape=None,
            width=None,
            head_space=None,
            reads=tuple(sorted(row.reads)),
            writes=None if row.writes is None else tuple(sorted(row.writes)),
            why=row.why,
            status="refused",
            undecided=(),
            refusal=Refusal.from_error(err),
        )

    unavailable = unavailable_at_load(info, component)
    if unavailable is not None:
        return refused(
            ValidationError(
                4,
                f"site {name!r}: {unavailable}",
                path=f"sites.{name}.component",
                reason="component_unavailable",
            )
        )
    try:
        shape = component_shape(info, component)
        width = shape.width
        if head is not None:
            width = component_width(info, component, head=head)
    except ValidationError as err:
        return refused(err)

    if stream is None:
        bound = COMPONENT_STREAMS.get(component)
        if bound is not None:
            stream = bound
        elif info.layer_types is not None and layers:
            at = {info.layer_types[layer] for layer in layers}
            stream = next(iter(at)) if len(at) == 1 else None
    if (
        component in COMPONENT_STREAMS
        and component not in LAYERLESS_COMPONENTS
        and info.layer_types is None
    ):
        undecided.append(
            f"exists only on a {COMPONENT_STREAMS[component]!r} mixer, and "
            f"model {info.key!r} declares no layer pattern (layer_types): which "
            "mixer each layer carries is decided against the loaded module"
        )
    open_predicates = [p for p in row.requires if predicate_holds(info, p) is None]
    if open_predicates:
        undecided.append(
            f"requires {sorted(open_predicates)}, which the entry cannot decide — "
            "the module tree does, at load"
        )
    if component in INTERIOR_ROWS and row.address_on(info) is None:
        undecided.append(
            "an attention-interior address the per-family tap table has not "
            f"measured for family {info.family!r}: served by measurement at "
            "load where that is unambiguous, refused where it is not"
        )
    return SiteReport(
        name=name,
        component=component,
        layers=tuple(layers),
        head=head,
        expert=expert,
        stream=stream,
        shape=shape.describe(),
        width=width,
        head_space=shape.head_space,
        reads=tuple(sorted(row.reads)),
        writes=None if row.writes is None else tuple(sorted(row.writes)),
        why=row.why,
        status="undecided" if undecided else "available",
        undecided=tuple(undecided),
        refusal=None,
    )


def _sites(docs: Sequence[Document], info: ModelInfo) -> tuple[SiteReport, ...]:
    """One report per distinct site the points name — a swept ``layer``
    collects into one report's ``layers``."""
    keyed: dict[tuple[str, str, Any, Any, Any], set[int]] = {}
    for doc in docs:
        for name, site in doc.sites.items():
            key = (name, str(site.component), site.head, site.expert, site.stream)
            layers = keyed.setdefault(key, set())
            if isinstance(site.layers, tuple):
                layers.update(site.layers)
    return tuple(
        site_report(
            name,
            component,
            info,
            layers=sorted(layers),
            head=head if isinstance(head, int) else None,
            expert=expert,
            stream=stream if isinstance(stream, str) else None,
        )
        for (name, component, head, expert, stream), layers in keyed.items()
    )


def _mentions_window(value: Any) -> bool:
    """Whether a position spelling has a tokenizer-decided window (a
    ``variable`` or ``column`` anchor — bare, or as the ``scope`` of an index
    — or ``all``) anywhere in it. A parsed :class:`PositionSpec` carries a
    scoped anchor as ``scope`` plus ``anchor_source``; a raw spelling nests
    it as ``{"scope": {"variable": …}}``."""
    if isinstance(value, PositionSpec):
        return (
            value.variable is not None
            or value.column is not None
            or bool(value.all)
            or (
                value.scope is not None
                and value.anchor_source in ("variable", "column")
            )
        )
    if isinstance(value, Mapping):
        return any(
            (key in ("variable", "column", "all") and item is not None)
            or _mentions_window(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_mentions_window(item) for item in value)
    return False


def _windowed(doc: Document) -> tuple[str, ...]:
    """The reads and writes whose position the tokenizer decides."""
    out: list[str] = []
    for kind, table in (("reads", doc.reads), ("writes", doc.writes)):
        for name, entry in table.items():
            pos = entry.pos
            if isinstance(pos, str) and pos in doc.positions:
                pos = doc.positions[pos]
            if _mentions_window(pos):
                out.append(f"{kind}.{name}")
    return tuple(out)


def _engines(
    compiled: CompiledProtocol, engines: Sequence[Engine]
) -> tuple[EngineReport, ...]:
    """Per candidate, what :func:`check_engine` would refuse — as a
    ``capability_shortfall`` diagnostic, never a raise."""
    out: list[EngineReport] = []
    for engine in engines:
        offered = engine.effective_capabilities
        lacks = tuple(sorted(compiled.capabilities - offered))
        try:
            check_engine(compiled, offered)
        except ValidationError as err:
            out.append(
                EngineReport(
                    name=engine.name,
                    serves=False,
                    lacks=lacks,
                    shortfall=Diagnostic(
                        "capability_shortfall", f"{engine.name}: {err}", path=err.path
                    ),
                    refusal=Refusal.from_error(err),
                )
            )
        else:
            out.append(EngineReport(engine.name, True, lacks, None, None))
    return tuple(out)


def _routing_refusal(
    compiled: CompiledProtocol,
    engines: Sequence[Engine],
    reports: Sequence[EngineReport],
) -> Refusal:
    """No candidate serves. As :func:`~causalab.protocol.run.route_engine`
    orders it: when the shortfall is a fact the *document* authored (a
    training field rule 30 decides), the refusal names the field — the first
    candidate's ``check_engine`` refusal under rule 30 — and routing's own
    generated text (rule 13, naming every candidate's shortfall) is the
    answer only when none has a narrower one."""
    refusals = [report.refusal for report in reports if report.refusal is not None]
    for refusal in refusals:
        if refusal.rule == 30:
            return refusal
    try:
        choose_engine(list(compiled.point_documents), list(engines))
    except ValidationError as err:
        return Refusal.from_error(err)
    return refusals[0]  # routing would pick one check_engine refuses per point


def dry_run(
    document: CompiledProtocol | Path | Mapping[str, Any],
    env: ResolutionEnv,
    *,
    engines: Sequence[Engine] = (),
    shard_size: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    check_data: bool = False,
) -> DryRunReport:
    """Resolve everything a run decides before weights load, and report it.

    ``document`` is a compiled result, a path or a tree; a path or a tree is
    compiled here through :func:`~causalab.protocol.compile.compile_protocol`
    with ``overrides`` applied, and a compiled result is taken as is (then
    ``overrides`` only says what the caller applied). ``engines`` are the
    candidates to ask :func:`~causalab.protocol.compile.check_engine` about —
    constructed objects, which load no weights; none means the engine
    question is left ``undecided``. ``shard_size`` plans ``--points`` shards
    of at most that many points. ``check_data`` runs the ``validate --data``
    pass (column and prompt-variable existence at every point, rule 25's
    row roles) and reports its refusal instead of raising it.

    Never calls ``load_model``, ``execute`` or a config fetch: the model
    facts are the registry entry's, and an unregistered ``model.key`` is
    the compile's ``V4`` refusal, re-raised.

    Raises what the compile raises (:class:`~causalab.protocol.errors.ParseError`,
    :class:`~causalab.protocol.errors.ValidationError`); returns a
    :class:`DryRunReport` for every document that compiles, ``ok`` when
    nothing it reports is a refusal.
    """
    applied: Mapping[str, Any] = dict(overrides or {})
    if isinstance(document, CompiledProtocol):
        compiled = document
    else:
        base_directory = document.parent if isinstance(document, Path) else None
        compiled = compile_protocol(
            document,
            base_directory,
            applied or None,
            env.datasets,
            env.artifacts,
            None,
            model_info=env.model_info,
        )
    docs = compiled.point_documents
    first = docs[0]
    undecided: list[Undecided] = []
    refusals: list[Refusal] = []

    # (5) the model, from the entry the compile already resolved
    info = env.model_info(str(first.model.key))
    keys = sorted({str(doc.model.key) for doc in docs})
    if len(keys) > 1:
        undecided.append(
            Undecided(
                "model",
                f"model.key is swept over {keys}; the model and site facts below "
                f"are the first point's ({info.key!r})",
            )
        )
    quantization = first.model.quantization
    model = ModelReport(
        key=info.key,
        revision=str(first.model.revision),
        dtype=str(first.model.dtype or MODEL_DTYPE_DEFAULT),
        quantization=(
            None
            if quantization is None
            else f"{quantization.scheme} ({quantization.method})"
        ),
        hidden_size=info.hidden_size,
        num_layers=info.num_layers,
        num_heads=info.num_heads,
        num_kv_heads=info.num_kv_heads,
        head_dim=info.head_dim,
        vocab_size=info.vocab_size,
        family=info.family,
        layer_pattern=None if info.layer_types is None else tuple(info.layer_types),
    )

    # (1) composition
    composition = CompositionReport(
        digest=compiled.digests.document,
        title=first.title,
        overrides=applied,
    )

    # (2) datasets and splits — every ref the points name, with its roles
    roles_of: dict[str, list[str]] = {}
    for doc in docs:
        for role, spec in doc.data.items():
            members = spec if isinstance(spec, tuple) else (spec,)
            for member in members:
                roles = roles_of.setdefault(str(member.dataset), [])
                if role not in roles:
                    roles.append(role)
    data = tuple(
        DataReport(
            ref=ref,
            roles=tuple(roles_of.get(ref, ())),
            digest=identity.digest,
            columns=tuple(identity.columns),
        )
        for ref, identity in sorted(compiled.data.items())
    )

    # (7), (12) points, forwards, shards
    points = PointsReport(
        n=len(compiled.points.points),
        axes=tuple((axis.id, len(axis.values)) for axis in compiled.points.axes),
    )
    plans = [plan_point(doc) for doc in docs]
    forwards = ForwardsReport(
        per_point=plans[0].num_forwards, campaign=len(interned_groups(plans))
    )
    shards = ShardsReport(
        n_points=points.n,
        shard_size=shard_size,
        count=shard_count(points.n, shard_size),
    )

    # (11) capabilities and, per candidate engine, the shortfall
    engine_reports = _engines(compiled, engines)
    if not engines:
        undecided.append(
            Undecided(
                "engines",
                "which engine serves the campaign: no candidate was handed in "
                "(pass --engine to route; the shipped engines' capability sets "
                "live in the engine modules)",
            )
        )
    elif not any(report.serves for report in engine_reports):
        refusals.append(_routing_refusal(compiled, engines, engine_reports))

    # (6) sites and the inventory
    sites = _sites(docs, info)
    for site in sites:
        for why in site.undecided:
            topic: UndecidedTopic = (
                "stream_at_layer" if "layer pattern" in why else "module_tree"
            )
            undecided.append(Undecided(topic, f"site {site.name!r}: {why}"))
    try:
        inv: Inventory | None = inventory(info)
    except ValidationError as err:
        inv = None
        undecided.append(Undecided("inventory", str(err)))

    # (9), (10) readouts and outputs
    readouts = tuple(
        ReadoutReport(
            name=name,
            site=str(read.site),
            model=str(read.model),
            input=str(read.input),
            metrics=tuple(
                mname
                for mname, metric in first.metrics.items()
                if str(metric.of) == name
                or any(str(value) == name for value in metric.fields.values())
            ),
        )
        for name, read in first.reads.items()
    )
    outputs = tuple(
        OutputReport(
            value=entry.value,
            binding=(
                f"site={entry.site}"
                if entry.site is not None
                else f"model={entry.model}, input={entry.input}"
            ),
            file_path=entry.file_path,
            kind=entry.kind,
        )
        for entry in first.save
    )

    # (3), (4), (8) — the run's, named here
    windowed = sorted({name for doc in docs for name in _windowed(doc)})
    tokenizer = (
        f"the windows of {windowed} (a variable, column or all position) "
        if windowed
        else "every position's token index "
    )
    undecided.append(
        Undecided(
            "tokenization",
            tokenizer + "and the rows' token widths are decided when the run "
            "encodes its inputs (sec. 2.3), as are the answer tokens a metric "
            "matches (token_form, sec. 2.10)",
        )
    )
    if check_data:
        try:
            check_data_columns(compiled, env)
            for doc in docs:
                check_row_roles(doc, env)
        except ProtocolError as err:
            refusals.append(Refusal.from_error(err))
        pair = (
            "column and prompt-variable existence and the declared row roles "
            "were checked at every point (the --data pass); "
        )
    else:
        pair = (
            "column and prompt-variable existence and the declared row roles "
            "are checked under --data; "
        )
    undecided.append(
        Undecided(
            "pair_validity",
            pair + "the counterfactual pair's validity (answer change, intended "
            "edit, no unintended edit, tokenizer stability) is checked over the "
            "rows and the tokenizer at run (sec. 2.2)",
        )
    )
    undecided.append(
        Undecided(
            "controls",
            "controls are declared one layer up, on the workflow that applies "
            "this document; a document dry run decides none",
        )
    )

    return DryRunReport(
        composition=composition,
        data=data,
        model=model,
        points=points,
        forwards=forwards,
        shards=shards,
        capabilities=tuple(sorted(compiled.capabilities)),
        engines=engine_reports,
        sites=sites,
        inventory=inv,
        readouts=readouts,
        outputs=outputs,
        refusals=tuple(refusals),
        undecided=tuple(undecided),
        diagnostics=compiled.diagnostics,
    )
