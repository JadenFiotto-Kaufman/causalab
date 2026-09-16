"""The one compiler (spec §9).

An intervention specification reaches an engine through four doors — the
``causalab`` CLI, :func:`~causalab.protocol.run.run_protocol`, a workflow's
protocol step, and workflow validation — and until this module each door
composed the pipeline for itself: three re-implemented the ``set`` overrides
before calling the loader, one could not pass overrides or a base directory
at all. Identical *today* is not a property; it is a coincidence maintained by
discipline, and the documents that break it first are easy to name:
overrides, base directories, deferred artifacts. A document could validate
under one resolution context and execute under another.

:func:`compile_protocol` is the one path. Its six inputs are the whole of the
context a compile depends on — the authored document, the directory relative
references resolve from, the overrides, the dataset and artifact resolvers,
and what the engine can do — and its eight outputs are everything a verb, a
runner or a dry run reads afterwards. Every door calls it; nothing else
composes the sequence. Standalone validation, standalone execution, workflow
validation and workflow execution therefore cannot disagree about what a
document *is*: only the resolvers differ, and the compiled result says what
those deferred.

**The order is data.** :data:`STAGES` is the compile in the order it runs —
read, override, resolve, families, paths, axes, gate, expand, validate,
canonicalize, digest, identify, route — and ``test_compile_protocol.py`` holds
it to the table spec §9 prints. Authoring sugar (``--set``, artifact-valued
fields, ``at_once`` families, path blocks, named axes, sweeps) resolves to the
explicit form *before* validation and hashing, once, so no engine ever needs a
second sweep implementation. It is also the extension seam: a later compiler
stage is one more entry in this tuple with one more row in the table, inserted
where the order says it belongs — ``families`` (§3.1), ``paths`` (path patching
as a compile into the existing nouns, §3.2) and ``axes`` (correlated
rows and dependent axes, §3.2) are the three inserted so far, all before the
gate, which is what makes each of them sugar. ``paths`` runs before ``axes``:
the axes stage keeps the tree it parsed and ``expand`` builds every point from
that tree, so everything that rewrites the tree must already have run; the
path compiler copies ``pos`` verbatim into every emitted entry, so an
``{"axis": …}`` wrapper there is lowered by the stage after it.

**Byte-identical to the loader it replaces.** Every canonical form and every
digest this module produces is the value ``loader.load`` computed before it
existed — the corpus, golden, workflow and demo pins are the proof, and none
moved. The one behavioural change is a *wider report*, never a different
refusal: the loader stopped at the first point that failed the checklist,
while points are independent documents, so the compiler validates every point
and raises the distinct violations together as a
:class:`~causalab.protocol.errors.ValidationErrors`. A lone violation, and a
sweep whose points all fail identically, still raise the single
``ValidationError`` with its text unchanged.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Collection, Literal, Mapping, cast, get_args

from causalab.protocol import canonical as _canonical
from causalab.protocol.axes import (
    Axes,
    canonical_axes,
    expand_axes,
    has_axes,
    lower_axes,
    parse_axes,
)
from causalab.protocol.engine import requires_campaign
from causalab.protocol.errors import ParseError, ValidationError, ValidationErrors
from causalab.protocol.families import expand_families
from causalab.protocol.loader import (
    apply_overrides,
    check_json_values,
    check_loaded_featurizers,
    load_text,
)
from causalab.protocol.paths import describe_paths, expand_paths, has_path_block
from causalab.protocol.registry import ModelInfo, get_model_info
from causalab.protocol.resolve import (
    ArtifactStore,
    DatasetResolver,
    ResolutionEnv,
    resolve_artifact_fields,
)
from causalab.protocol.schema import (
    Document,
    check_protocol_version,
    dotted_path,
    parse_document,
)
from causalab.protocol.sweep import DEFAULT_POINT_CAP, Expansion, Point, expand
from causalab.protocol.validate import check_engine_support, validate_document

__all__ = [
    "DIAGNOSTIC_KINDS",
    "STAGES",
    "Authored",
    "CompiledPoint",
    "CompiledPoints",
    "CompiledProtocol",
    "DataIdentity",
    "Diagnostic",
    "DiagnosticKind",
    "Digests",
    "ResolvedArtifact",
    "Stage",
    "check_engine",
    "compile_protocol",
    "read_document",
]


# --------------------------------------------------------------------------- #
# the eight outputs
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class CompiledPoint(Point):
    """One compiled intervention: the sweep point's coordinates and concrete
    tree (:class:`~causalab.protocol.sweep.Point`), plus its validated parse
    and its canonical form — the bytes its provenance digest is over (§7)."""

    document: Document
    canonical: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class CompiledPoints(Expansion):
    """The expanded points with their coordinates (§3), as the
    :class:`~causalab.protocol.sweep.Expansion` every consumer already reads,
    each point a :class:`CompiledPoint`; plus the **explicit document** they
    were expanded from — the composition with overrides applied and every
    artifact-valued field resolved, sweep wrappers intact — and its parse.
    The explicit form is what canonicalization and hashing consume, so it is
    reported beside the points rather than recomputed by anyone."""

    points: tuple[CompiledPoint, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    explicit: Mapping[str, Any]
    document: Document

    @property
    def documents(self) -> tuple[Document, ...]:
        return tuple(point.document for point in self.points)

    @property
    def canonical(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(point.canonical for point in self.points)

    @property
    def coords(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(point.coords for point in self.points)


@dataclasses.dataclass(frozen=True)
class DataIdentity:
    """One resolved dataset ref: the content digest stamped into the canonical
    form (§2.2) and the table's columns — the schema ``validate --data``
    checks references against."""

    digest: str
    columns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ResolvedArtifact:
    """One reference the document makes outside itself, as the compile resolved
    it.

    A *value* reference (``{"artifact": ref, "key": k}``, §1) names ``key``;
    its value is in the explicit document. A *file* reference (a featurizer's
    or a param's ``file_path``, §2.5/§2.6) names none, and carries the
    stamped ArtifactIdentity the compile read from the file's header (§8) —
    or ``None`` when the store deferred the check, or the file is unstamped.
    ``deferred`` is the store's word: a workflow validating a step-dependent
    document answers with declared representatives and defers every file
    check to run time (workflow spec §2.3), and the compiled result says so
    instead of looking identical to a real resolution."""

    path: str
    reference: str
    key: str | None
    deferred: bool
    identity: Mapping[str, Any] | None


@dataclasses.dataclass(frozen=True)
class Digests:
    """The document digest (the campaign) and every point's digest (the
    provenance units, §7) — the two identities ``--resume`` compares: a
    workflow's protocol step is reused by its inner document digest, and a
    tensor is stamped with the point digest that produced it."""

    document: str
    points: tuple[str, ...]


#: What a compile can report without refusing. Closed, and tabulated in spec
#: §9 (``test_compile_protocol.py`` holds the two together):
#:
#: * ``deferred_check`` — the artifact resolver deferred a file's existence
#:   and identity check to run time (workflow validation of a step-dependent
#:   document). The base did this silently; the compiled result now says it;
#: * ``capability_shortfall`` — the document requires a capability an engine
#:   lacks. A compile handed ``engine_capabilities`` *refuses* on it (rule
#:   13, the ``route`` stage), as does :func:`check_engine` for the routed
#:   engine; the kind is produced by :func:`~causalab.protocol.dry_run.dry_run`,
#:   per candidate engine, from what ``check_engine`` would refuse — without
#:   refusing.
DiagnosticKind = Literal["deferred_check", "capability_shortfall"]
DIAGNOSTIC_KINDS: tuple[DiagnosticKind, ...] = get_args(DiagnosticKind)


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    """One non-refusing finding of a compile. Violations are not diagnostics:
    they are raised, one as itself and several as
    :class:`~causalab.protocol.errors.ValidationErrors`, so a returned
    :class:`CompiledProtocol` is always a valid one."""

    kind: DiagnosticKind
    message: str
    path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in DIAGNOSTIC_KINDS:
            raise AssertionError(
                f"unknown diagnostic kind {self.kind!r}; expected one of "
                f"{DIAGNOSTIC_KINDS}"
            )


@dataclasses.dataclass(frozen=True)
class CompiledProtocol:
    """What one compile produced — the seven outputs every door reads, plus
    the derived record of what the compile lowered, and nothing else.

    * ``canonical`` — the canonical document (§7), wrappers intact: the
      campaign;
    * ``points`` — the expanded points with their coordinates, each parsed,
      validated and canonicalized, and the explicit document they came from;
    * ``data`` — every dataset ref the points name, with its content digest
      and columns;
    * ``artifacts`` — every artifact reference, with what was read and what
      the store deferred;
    * ``capabilities`` — the engine capabilities the campaign requires (§8),
      derived from the registry rows through :func:`~causalab.protocol.engine.requires`
      — never a second table;
    * ``digests`` — the document digest and the point digests;
    * ``diagnostics`` — what the compile found and did not refuse on;
    * ``lowered`` — the derived record (§6) of what the ``paths`` stage
      lowered: ``{}`` for every document without a path block, else the block
      as written, its policy, the receiver order, the restorer boundary and
      the names of every emitted entry (:func:`~causalab.protocol.paths.describe_paths`).
      Bound to the run's identity by reference — every emitted name is a key of
      ``canonical["method"]``, and ``digests`` are over that canonical form —
      and written into the run receipt as ``derived``. **Last**, and the §9.1
      outputs table's last row: ``test_compile_protocol.py`` holds the two in
      order.
    """

    canonical: Mapping[str, Any]
    points: CompiledPoints
    data: Mapping[str, DataIdentity]
    artifacts: tuple[ResolvedArtifact, ...]
    capabilities: frozenset[str]
    digests: Digests
    diagnostics: tuple[Diagnostic, ...]
    lowered: Mapping[str, Any]

    @property
    def point_documents(self) -> tuple[Document, ...]:
        """The validated parse of every point — what routing, planning and
        the ``--data`` pass read."""
        return self.points.documents


@dataclasses.dataclass(frozen=True)
class Authored:
    """The read prefix of a compile, on its own: the authored document with
    overrides applied and nothing yet resolved.

    A workflow needs this *before* it can compile a step-dependent inner
    document — the authored tree is what it walks for step references, and
    those decide which store the compile resolves against (workflow spec
    §2.3). :func:`read_document` produces it with the compiler's own first two
    stages, and :func:`compile_protocol` accepts it back as the authored
    document, so the workflow's reading and the compiler's are one
    implementation and not two that agree."""

    raw: Mapping[str, Any]


# --------------------------------------------------------------------------- #
# the stages, in order
# --------------------------------------------------------------------------- #

#: The compile, in the order it runs. Each name is a stage function below;
#: :func:`compile_protocol` walks this tuple and nothing else decides the
#: order. Spec §9 prints the same table, and ``test_compile_protocol.py``
#: holds the two together — a new stage is one entry here and one row there.
Stage = Literal[
    "read",
    "override",
    "resolve",
    "families",
    "paths",
    "axes",
    "gate",
    "expand",
    "validate",
    "canonicalize",
    "digest",
    "identify",
    "route",
]
STAGES: tuple[Stage, ...] = get_args(Stage)


@dataclasses.dataclass
class _Build:
    """The compile in progress: the six inputs, then what each stage adds."""

    source: Path | Mapping[str, Any] | Authored
    base_directory: Path | None
    overrides: Mapping[str, Any] | None
    env: ResolutionEnv
    engine_capabilities: frozenset[str] | None
    point_cap: int | None
    # read
    authored: dict[str, Any] = dataclasses.field(default_factory=dict)
    # resolve + families + paths + axes + gate
    explicit: dict[str, Any] = dataclasses.field(default_factory=dict)
    lowered: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    axes: Axes | None = None
    document: Document | None = None
    # expand + validate
    expansion: Expansion | None = None
    documents: tuple[Document, ...] = ()
    # canonicalize + digest
    canonical: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    canonical_points: tuple[Mapping[str, Any], ...] = ()
    digests: Digests | None = None
    # identify + route
    data: dict[str, DataIdentity] = dataclasses.field(default_factory=dict)
    artifacts: list[ResolvedArtifact] = dataclasses.field(default_factory=list)
    capabilities: frozenset[str] = frozenset()
    diagnostics: list[Diagnostic] = dataclasses.field(default_factory=list)


def _read(build: _Build) -> None:
    """Read the authored source and check it is an intervention specification
    of the version this compiler reads (§1).

    The version check comes first so that a workflow document, or a v1
    document, is refused as that — before the override stage addresses the
    tree by path and would refuse it as a path that does not exist. An
    :class:`Authored` source has been through this stage already and is taken
    as is."""
    source = build.source
    if isinstance(source, Authored):
        build.authored = dict(source.raw)
        return
    raw = dict(load_text(source)) if isinstance(source, Path) else dict(source)
    check_protocol_version(raw)
    build.authored = raw


def _override(build: _Build) -> None:
    """``--set`` / a step's ``set`` (§9), on the authored tree, by
    section-rooted path (§1) — and the digest is the overridden document's."""
    if build.overrides:
        build.authored = apply_overrides(build.authored, build.overrides)


def _resolve(build: _Build) -> None:
    """Artifact-valued fields resolve first (§1: legal anywhere a value is),
    then the tree is held to the JSON object model whatever surface produced
    it."""
    build.explicit = resolve_artifact_fields(build.authored, build.env)
    check_json_values(build.explicit)


def _families(build: _Build) -> None:
    """``at_once`` families materialize into the entries they denote (§3.1,
    rule 26).

    Here, and not later, is the whole of why they are *sugar*: every stage
    after this one — the shape gate, the checklist, sweep expansion, the
    canonical form, the digest — sees exactly the document the author would
    have written out by hand. A no-op on any specification that declares no
    family, which is every one written before §3.1. Families live in the
    ``method`` group (§1), which is where the expansion looks; a tree without
    one is the shape gate's refusal to make, next."""
    build.explicit = expand_families(build.explicit)


def _paths(build: _Build) -> None:
    """A ``method.path_patching`` block lowers to the sites, reads, writes and
    intervened models it denotes (§3.2) — after families, so a wrapper inside
    the block is refused here rather than expanded, and before the gate, so
    the parser, the checklist, sweep expansion, the canonical form and the
    digest see exactly the hand-written document. A no-op on any document
    without a block, which is every one written before §3.2.

    The lowering is also the compile's eighth output: what the block stood
    for — its policy, the receiver order, the restorer boundary, the emitted
    names — is kept as a *derived* record (§6), never as a canonical section,
    which is what keeps the block digest-neutral and the shipped document's
    pins in place."""
    if not has_path_block(build.explicit):
        return
    authored = build.explicit
    build.explicit = expand_paths(authored)
    build.lowered = describe_paths(authored, build.explicit)


def _axes(build: _Build) -> None:
    """Named axes — correlated row tuples and dependent axes (§3.2) — parse,
    and the document lowers to its **display form**: every ``{"axis": …}``
    wrapper becomes the ``{"sweep": [column]}`` it stands for and the ``axes``
    group is removed, so the gate and ``CompiledPoints.explicit`` see a
    document any swept one could be. After
    families and paths, so a wrapper a family entry carries has been copied to
    every member and a path block's emitted entries are in the tree before the
    references are found; before the gate, which knows no fifth group.

    The parsed axes are kept: ``expand`` walks *them* — the correlated rows
    as the rows they are, never as the display form's cross product — and
    ``canonicalize`` writes the block into the campaign's canonical form. A
    no-op on any document without the group, which is every one written
    before §3.2."""
    if not has_axes(build.explicit):
        return
    build.axes = parse_axes(build.explicit, build.env.model_info)
    build.explicit = lower_axes(build.explicit, build.axes)


def _gate(build: _Build) -> None:
    """The strict parse of the explicit form, sweep wrappers intact (rules 1
    and 2): a shape gate on what the sweep is about to expand."""
    build.document = parse_document(build.explicit)


def _expand(build: _Build) -> None:
    """Sweeps expand into compiled interventions (§3, rule 14) — over the
    named axes when the document declared any (§3.2: each entry's concrete
    tree, then its ordinary sweep axes), else exactly as before."""
    if build.axes is None:
        build.expansion = expand(build.explicit, point_cap=build.point_cap)
    else:
        build.expansion = expand_axes(build.axes, point_cap=build.point_cap)


def _validate(build: _Build) -> None:
    """Every point is parsed and held to the checklist (§5) — a point is exactly
    as valid as the same document written by hand — and every ``file_path``
    load is checked against the document that names it (§2.5/§8).

    **Every point, not the first failing one.** Points are independent
    documents, so a violation in point 3 is knowable when point 0's is
    reported. Distinct violations across points are raised together as a
    :class:`~causalab.protocol.errors.ValidationErrors`, in point order;
    identical ones (a sweep whose every point breaks the same rule the same
    way) collapse to the one the loader always reported, so a single-rule
    refusal keeps its text. A point that fails to *parse* is not a checklist
    violation and stops the pass where it stands — unless an earlier point had
    already been refused, in which case that refusal is the report, as it was.

    Rules 13 and 30 need to know the engine (§2.8, §2.11): they are evaluated
    exactly when ``engine_capabilities`` were given — 13 from whether they
    include ``pytorch_fn_local``, 30 from the training verbs. No production
    door knows the engine here; :func:`check_engine` runs the same two rules
    for the engine routing chose. Rule 29 reads the operands' widths from
    the model's static config (``env.model_info``), never from weights."""
    if build.expansion is None:
        raise AssertionError("validate runs after expand")
    caps = build.engine_capabilities
    documents: list[Document] = []
    refused: list[ValidationError] = []
    for point in build.expansion.points:
        try:
            pdoc = parse_document(point.raw)
        except ParseError:
            if refused:
                break  # the earlier refusal is the report, as it always was
            raise
        try:
            validate_document(
                pdoc, engine_capabilities=caps, model_info=build.env.model_info
            )
            check_loaded_featurizers(pdoc, build.env)
        except ValidationError as err:
            refused.append(err)
        documents.append(pdoc)
    _raise_distinct(refused)
    build.documents = tuple(documents)


def _raise_distinct(refused: list[ValidationError]) -> None:
    """Report what several points refused on, once.

    One failing point raises exactly what it raised. Several raise their
    *distinct* violations together — a sweep whose every point breaks one rule
    the same way is one violation, and is raised as the first point's, so a
    single-rule refusal keeps the text the loader always gave it."""
    if not refused:
        return
    if len(refused) == 1:
        raise refused[0]
    distinct: dict[tuple[str, str | None, str], ValidationError] = {}
    for err in refused:
        each = err.errors if isinstance(err, ValidationErrors) else (err,)
        for violation in each:
            distinct.setdefault(
                (violation.code, violation.path, violation.message), violation
            )
    if len(distinct) == 1:
        raise next(iter(distinct.values()))
    raise ValidationErrors(tuple(distinct.values()))


def _canonicalize(build: _Build) -> None:
    """The canonical document (wrappers intact — the campaign; the ``axes``
    block beside them when one was authored, §3.2) and every point's
    canonical form (fully materialized — the provenance units), §7.

    Canonicalization refuses too — a layer outside the model, a component on
    the wrong stream (rule 4) — and it decides per point, from the model's
    static metadata. Those refusals are collected across points exactly as the
    checklist's are: distinct ones together, identical ones once."""
    if build.expansion is None:
        raise AssertionError("canonicalize runs after expand")
    build.canonical = _canonical.canonicalize(
        build.explicit,
        build.env,
        axes=None if build.axes is None else canonical_axes(build.axes),
    )
    canonical_points: list[Mapping[str, Any]] = []
    refused: list[ValidationError] = []
    for point in build.expansion.points:
        try:
            canonical_points.append(_canonical.canonicalize(point.raw, build.env))
        except ValidationError as err:
            refused.append(err)
    _raise_distinct(refused)
    build.canonical_points = tuple(canonical_points)


def _digest(build: _Build) -> None:
    """``sha256`` of the canonical bytes, document and points (§7)."""
    build.digests = Digests(
        document=_canonical.digest(build.canonical),
        points=tuple(_canonical.digest(c) for c in build.canonical_points),
    )


def _identify(build: _Build) -> None:
    """What the document reached outside itself, as resolved: every dataset
    ref's identity and schema, every artifact reference and whether the store
    deferred its check."""
    env = build.env
    for doc in build.documents:
        for role in _data_roles(doc):
            ref = role.dataset
            if isinstance(ref, str) and ref not in build.data:
                build.data[ref] = DataIdentity(
                    digest=env.datasets.digest(ref),
                    columns=tuple(env.datasets.columns(ref)),
                )
    defers: Callable[[str], bool] | None = getattr(env.artifacts, "defers", None)
    seen: set[tuple[str, str]] = set()
    for path, reference, key in _value_references(build.authored):
        if (path, reference) in seen:
            continue
        seen.add((path, reference))
        build.artifacts.append(
            ResolvedArtifact(
                path=path,
                reference=reference,
                key=key,
                deferred=defers is not None and defers(reference),
                identity=None,
            )
        )
    for doc in build.documents:
        for path, file_path in _file_references(doc):
            if (path, file_path) in seen:
                continue
            seen.add((path, file_path))
            deferred = defers is not None and defers(file_path)
            build.artifacts.append(
                ResolvedArtifact(
                    path=path,
                    reference=file_path,
                    key=None,
                    deferred=deferred,
                    identity=None
                    if deferred
                    else env.artifacts.read_identity(file_path),
                )
            )
            if deferred:
                build.diagnostics.append(
                    Diagnostic(
                        "deferred_check",
                        f"{file_path!r} is loaded from a run tree: its existence "
                        "and identity are checked at run time, not here",
                        path=path,
                    )
                )


def _route(build: _Build) -> None:
    """The capabilities the whole campaign requires (§8) — the union over the
    points, derived from the registry rows — and, when the caller said what
    the engine offers, the refusal of a shortfall against them (rule 13),
    here rather than at run: a document an engine cannot execute is refused
    before any weights load, with the same generated text routing gives."""
    build.capabilities = requires_campaign(build.documents)
    if build.engine_capabilities is None:
        return
    _refuse_shortfall(build.capabilities, build.engine_capabilities)


def _refuse_shortfall(required: frozenset[str], offered: frozenset[str]) -> None:
    """Rule 13's routing refusal (§8), generated from the missing entries —
    never hand-written per case. The wording differs from
    :func:`~causalab.protocol.engine.choose_engine`'s, which names each
    candidate's shortfall, because here the engine is already chosen."""
    missing = required - offered
    if missing:
        raise ValidationError(
            13,
            f"the engine does not support this document: it requires "
            f"{sorted(required)} and lacks {sorted(missing)} (sec. 8)",
        )


_STAGE: Mapping[Stage, Callable[[_Build], None]] = {
    "read": _read,
    "override": _override,
    "resolve": _resolve,
    "families": _families,
    "paths": _paths,
    "axes": _axes,
    "gate": _gate,
    "expand": _expand,
    "validate": _validate,
    "canonicalize": _canonicalize,
    "digest": _digest,
    "identify": _identify,
    "route": _route,
}


def _data_roles(doc: Document) -> list[Any]:
    """Every data role of a point, base first (§2.2)."""
    out: list[Any] = []
    for value in doc.data.values():
        out.extend(value if isinstance(value, tuple) else (value,))
    return out


def _value_references(node: Any, *, _path: str = "") -> list[tuple[str, str, str]]:
    """``(path, artifact, key)`` for every ``{"artifact": …, "key": …}`` node
    of the authored document — the references :func:`resolve_artifact_fields`
    replaced by their values."""
    found: list[tuple[str, str, str]] = []
    if isinstance(node, Mapping):
        artifact = node.get("artifact")
        if isinstance(artifact, str) and isinstance(node.get("key"), str):
            found.append((dotted_path(_path.split(".")), artifact, str(node["key"])))
            return found
        for key, value in node.items():
            found.extend(
                _value_references(value, _path=f"{_path}.{key}" if _path else key)
            )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_value_references(item, _path=f"{_path}[{index}]"))
    return found


def _file_references(doc: Document) -> list[tuple[str, str]]:
    """``(path, file_path)`` for the two places a document *loads* a file:
    featurizer bundles and params (§2.5, §2.6). ``save`` also carries
    ``file_path`` keys, but those are outputs, never loads."""
    found: list[tuple[str, str]] = []
    for name, spec in doc.featurizers.items():
        if isinstance(spec.file_path, str):
            found.append((f"featurizers.{name}.file_path", spec.file_path))
    for name, pspec in doc.params.items():
        if isinstance(pspec.file_path, str):
            found.append((f"params.{name}.file_path", pspec.file_path))
    return found


# --------------------------------------------------------------------------- #
# the entry points
# --------------------------------------------------------------------------- #


def read_document(
    authored_document: Path | Mapping[str, Any],
    base_directory: Path | None,
    overrides: Mapping[str, Any] | None,
) -> Authored:
    """The read prefix — :data:`STAGES` up to and including ``override`` — on
    its own.

    For the caller that has to see the authored tree before it can choose a
    resolver (a workflow, deciding whether an inner document depends on a
    step), and for the ``--register-from-hf`` pre-pass, which needs the
    overridden ``model.key`` before anything resolves. The result goes back
    into :func:`compile_protocol` as the authored document, so reading here
    and compiling there is one implementation run in two halves."""
    build = _Build(
        source=authored_document,
        base_directory=base_directory,
        overrides=overrides,
        env=ResolutionEnv(
            datasets=cast(DatasetResolver, _NoResolver()),
            artifacts=cast(ArtifactStore, _NoResolver()),
        ),
        engine_capabilities=None,
        point_cap=None,
    )
    _STAGE["read"](build)
    _STAGE["override"](build)
    return Authored(raw=build.authored)


class _NoResolver:
    """Stands where a resolver would in :func:`read_document`, which never
    resolves. Reaching it is a bug in the stage order, and says so."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"read_document() resolves nothing (asked for {name!r})")


def compile_protocol(
    authored_document: Path | Mapping[str, Any] | Authored,
    base_directory: Path | None,
    overrides: Mapping[str, Any] | None,
    dataset_resolver: DatasetResolver,
    artifact_resolver: ArtifactStore,
    engine_capabilities: Collection[str] | None,
    *,
    point_cap: int | None = DEFAULT_POINT_CAP,
    model_info: Callable[[str], ModelInfo] = get_model_info,
) -> CompiledProtocol:
    """Compile one intervention specification: the six inputs in, the eight
    outputs out, :data:`STAGES` in between.

    ``authored_document`` is a path, the document as a tree, or an
    :class:`Authored` prefix. ``base_directory`` is where the document's
    relative references resolve from; a path defaults to its own directory, a
    tree has none unless one is given. ``overrides`` are ``--set`` / a step's
    ``set`` (section-rooted dotted paths, §1, §9). ``dataset_resolver`` and
    ``artifact_resolver`` are the two services a load resolves against (a
    workflow validating a step-dependent document passes a deferring store,
    and the result says what was deferred). ``engine_capabilities`` is what
    the engine can do, when known: rules 13 and 30 are decided from it, and a
    shortfall against what the document requires is refused (rule 13). No
    door knows the engine before it has compiled — routing needs the points
    — so each compiles with ``None`` and calls :func:`check_engine` for the
    engine it chose, before any weights load.

    Two further knobs are keyword-only:
    ``point_cap`` (``--max-points`` / a step's ``max_points``, rule 14) and
    ``model_info`` (the third resolution service — static model metadata,
    the registry's by default).

    Refuses as the loader did — a :class:`~causalab.protocol.errors.ParseError`
    or :class:`~causalab.protocol.errors.ValidationError`, several independent
    violations as a :class:`~causalab.protocol.errors.ValidationErrors` — so a
    returned :class:`CompiledProtocol` is a valid document.
    """
    build = _Build(
        source=authored_document,
        base_directory=base_directory,
        overrides=overrides,
        env=ResolutionEnv(
            datasets=dataset_resolver,
            artifacts=artifact_resolver,
            model_info=model_info,
        ),
        engine_capabilities=(
            None if engine_capabilities is None else frozenset(engine_capabilities)
        ),
        point_cap=point_cap,
    )
    for stage in STAGES:
        _STAGE[stage](build)
    if build.expansion is None or build.document is None or build.digests is None:
        raise AssertionError("the stage list did not run to completion")
    points = tuple(
        CompiledPoint(
            coords=point.coords,
            raw=point.raw,
            document=document,
            canonical=canonical,
        )
        for point, document, canonical in zip(
            build.expansion.points, build.documents, build.canonical_points
        )
    )
    return CompiledProtocol(
        canonical=build.canonical,
        points=CompiledPoints(
            axes=build.expansion.axes,
            points=points,
            explicit=build.explicit,
            document=build.document,
        ),
        data=dict(build.data),
        artifacts=tuple(build.artifacts),
        capabilities=build.capabilities,
        digests=build.digests,
        diagnostics=tuple(build.diagnostics),
        lowered=build.lowered,
    )


def check_engine(
    compiled: CompiledProtocol, engine_capabilities: Collection[str]
) -> None:
    """The engine-aware rules of a compile, for the engine routing chose —
    the seam the invariant of spec §5 hangs on: **a configuration accepted
    by preflight must either execute or fail with a narrower runtime
    condition that preflight could not know.**

    A compile handed ``engine_capabilities`` decides rule 13 (``pytorch_fn``
    needs a local engine), rule 30 (the fit as authored needs the engine's
    training verbs) and the routing shortfall (rule 13) itself. No production
    door has them at compile time — routing needs the compiled points — so
    :func:`~causalab.protocol.run.run_protocol` and the workflow runner
    compile with ``None``, choose an engine, and call this with
    ``engine.effective_capabilities`` **before** the engine loads a model. It
    re-enters the compiler's own rule functions
    (:func:`~causalab.protocol.validate.check_engine_support`, the ``route``
    stage's refusal) per point, collecting distinct violations as the
    ``validate`` stage does; it is not a second pipeline. A dry run
    calls it once per candidate engine and reports instead of raising.

    Raises :class:`~causalab.protocol.errors.ValidationError` (several as
    :class:`~causalab.protocol.errors.ValidationErrors`); returns ``None`` when
    the engine can execute every point as authored.
    """
    offered = frozenset(engine_capabilities)
    refused: list[ValidationError] = []
    for pdoc in compiled.point_documents:
        try:
            check_engine_support(pdoc, offered)
        except ValidationError as err:
            refused.append(err)
    _raise_distinct(refused)
    _refuse_shortfall(compiled.capabilities, offered)
