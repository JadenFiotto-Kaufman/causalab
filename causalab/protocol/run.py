"""Running an intervention specification from Python (spec §9).

The CLI was the only way to run a document. Everything else the protocol layer
does — load, validate, expand, canonicalize, plan, digest — has been a Python
function all along, so "run" was the one verb a notebook, a script step or a
test could not reach without shelling out and parsing stdout.

This module is that verb, and the CLI is now a wrapper over it: argument
parsing, printing and exit codes there, the run itself here. Which way round
that dependency goes is the point: a CLI over a library can be
replaced, scripted around and tested in-process, while a library that only
exists inside a CLI cannot.

Two things are deliberately *not* parameters.

``overrides`` and the point cap belong to
:func:`causalab.protocol.compile.compile_protocol`, which this function will
call for you when handed a path or a tree. A caller who needs either compiles
the document itself and passes the
:class:`~causalab.protocol.compile.CompiledProtocol` — which is also what the
CLI does, since ``--set`` and ``--max-points`` are compile-time concerns and a
run should not re-decide them. There is one compiler and every entry point
calls it, so what this function executes is byte for byte what
``causalab validate`` accepted.

``resume`` is a **workflow** concept, not a protocol one: ``--resume`` skips a
step whose outputs are already on disk with a matching stamped digest, and a
protocol run has no step boundaries to resume at. Sharding a campaign is
``points`` (below), which an external scheduler can fan out and recombine by
digest. A ``resume`` parameter here would be a promise this layer does not
keep.

This module imports no engine and no torch: :mod:`causalab.protocol.engine` is
the contract, and the engines that satisfy it are the caller's to supply, so
the pure verbs stay importable on a machine with no accelerator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from causalab.protocol.compile import CompiledProtocol, check_engine, compile_protocol
from causalab.protocol.engine import Engine, ExecutionRequest, RunResult, choose_engine
from causalab.protocol.errors import ProtocolError, ValidationError
from causalab.protocol.loader import LoadedProtocol, check_row_roles
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.validate import check_engine_support

__all__ = [
    "FIRES_KEY",
    "MODEL_SOURCES",
    "RUN_RECORD_NAME",
    "execution_record",
    "resolved_fit_rows",
    "FIT_ROWS_RESOLVED_KEY",
    "FIT_ROWS_SHRINKS_KEY",
    "RAGGED_KEY",
    "ragged_geometry",
    "record_ragged_geometry",
    "MEASURED_BOUNDS",
    "bound_shrinks",
    "fit_rows_shrinks",
    "measured_bounds",
    "record_measured_bounds",
    "resolved_bound",
    "parse_points",
    "record_fires",
    "run_protocol",
    "write_run_record",
]

#: The run receipt's filename inside the output directory. The exported
#: name keeps `record`: it is public API (``causalab.protocol.__all__``),
#: and §11.1 does not spend a compatibility break on a word.
RUN_RECORD_NAME = "protocol.json"

#: The receipt's fire-count block (§4 "Fires"): per point digest, per forward
#: group (``<model> on <input>``), how many times each write member fired per
#: forward — a state write, at how many distinct steps. An **observation of
#: the run**, so it sits beside the ``execution`` block rather than in it:
#: that block is the execution parameters the run was declared with, and its
#: contract closes it to anything else (:func:`execution_record`).
FIRES_KEY = "fires"

#: The closed vocabulary of ``execution.model_source`` (§8): ``loaded`` when
#: the engine loaded the document's model itself, ``caller`` when it ran a
#: caller-owned bundle handed to its constructor (§9, the ownership contract).
MODEL_SOURCES = frozenset({"loaded", "caller"})


def parse_points(spec: str, n_points: int) -> range:
    """The ``--points`` shard selector: a half-open ``[start, stop)`` range.

    Refused rather than clamped when it falls outside ``[0, n_points]`` or
    selects nothing — a shard that silently became a different shard is worse
    than one that failed, because its artifacts would still stamp as members
    of the campaign.
    """
    try:
        start_text, stop_text = spec.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except ValueError:
        raise ProtocolError("P4", f"--points {spec!r} is not START:STOP") from None
    if not (0 <= start < stop <= n_points):
        raise ProtocolError(
            "P4",
            f"--points {spec!r} is outside the campaign's {n_points} points "
            "or selects none",
        )
    return range(start, stop)


def execution_record(engine: Any, request: Any = None) -> dict[str, Any]:
    """The ``execution`` block of a run receipt: the batch geometry the chosen
    engine will run under, and where its model came from (§8, execution
    scale; §9, the ownership contract).

    Execution provenance has exactly one recorder, and this is it.
    ``batch_rows`` is the engine's microbatch bound — ``None`` when the engine
    runs every forward group whole, and for an engine that has no such bound
    at all (the nnsight engine). ``fit_rows`` is its rows-per-grad-forward
    bound for a fit — the members of a fit cohort are packed into forwards
    under it — ``None`` when the engine measures it, and for an engine with
    no grad path. With a ``request``, an engine that reads the request's
    ``execution`` block (a workflow step's overrides) reports the value it
    will run under through its ``effective_<bound>`` methods; an engine that
    never reads the block keeps its own value, so the record never claims an
    override that was not applied.
    ``model_source`` is ``"caller"`` when the
    engine was built around a caller-owned bundle and ``"loaded"`` when it
    loads the document's model itself (:data:`MODEL_SOURCES`; an engine that
    declares nothing loads). Both are read off the engine rather than declared
    by the document because they are execution, not identity: they enter no
    canonical document, no digest and no artifact stamp, so two layouts of one
    document — or the same weights loaded and handed in — differ in their
    receipts here and nowhere else. **Recorded, not gated**: a
    layout-dependent flip in a top-1 token is something a reader of two
    receipts can see and attribute, not something a run refuses over. A later
    execution parameter adds its own key beside these; nothing else belongs
    in the block — what the run *observed* (the ``fires`` block,
    :func:`record_fires`; the ``scoring`` block) is recorded beside it,
    not in it.
    """
    model_source = getattr(engine, "model_source", "loaded")
    if model_source not in MODEL_SOURCES:
        raise AssertionError(
            f"engine {getattr(engine, 'name', engine)!r} reports model_source "
            f"{model_source!r}; expected one of {sorted(MODEL_SOURCES)}"
        )

    def bound(name: str) -> Any:
        # an engine that reads a request's `execution` block says what it
        # will run under (`effective_<name>`); one that never reads it keeps
        # its own value, so the receipt never claims an override nobody applied
        effective = getattr(engine, f"effective_{name}", None)
        if request is not None and callable(effective):
            return effective(request)
        return getattr(engine, name, None)

    return {
        "batch_rows": bound("batch_rows"),
        "fit_rows": bound("fit_rows"),
        "model_source": model_source,
    }


#: The receipt's keys for a bound the run measured rather than authored (§8):
#: beside the ``null`` request, the number to pin as the bound to reproduce the
#: run — the bound every window ran under, after any shrink — and, only when
#: non-zero, the most windows any one cohort re-packed after running out of
#: memory. The pin is still safe: a grad shrink lowered the bound in place,
#: so a re-run at it packs as this run did; only an eval shrink leaves the
#: report below the grad bound, and a re-run pinned there packs its grad
#: windows smaller, which the drift tier's author should know.
FIT_ROWS_RESOLVED_KEY = "fit_rows_resolved"
FIT_ROWS_SHRINKS_KEY = "fit_rows_shrinks"

#: The measured bounds, one row each: the requested key (also the summaries'
#: key for the bound a fit ran under), the summaries' shrink key, the
#: receipt's resolved key and the receipt's shrink key. One row today —
#: ``fit_rows`` bounds a cohort's grad forwards and, unless ``batch_rows`` is
#: authored, its batched eval passes too.
MEASURED_BOUNDS: tuple[tuple[str, str, str, str], ...] = (
    ("fit_rows", "fit_rows_shrinks", FIT_ROWS_RESOLVED_KEY, FIT_ROWS_SHRINKS_KEY),
)


def resolved_bound(summaries: Sequence[Mapping[str, Any]], key: str) -> int | None:
    """The bound ``key`` (``fit_rows`` or ``batch_rows``) a run's fits
    reported, or ``None`` when none did: the **smallest** over the points. A
    run's cohorts measure independently (different frames, different
    parametrizations) and an authored bound is never shrunk, so the number
    safe to pin is the one every cohort of the run ran at or above."""
    bounds = [
        int(summary[key]) for summary in summaries if isinstance(summary.get(key), int)
    ]
    return min(bounds) if bounds else None


def resolved_fit_rows(summaries: Sequence[Mapping[str, Any]]) -> int | None:
    """:func:`resolved_bound` for ``fit_rows`` — what a receipt records as
    ``execution.fit_rows_resolved`` when the run authored no bound."""
    return resolved_bound(summaries, "fit_rows")


def bound_shrinks(summaries: Sequence[Mapping[str, Any]], key: str) -> int:
    """The most windows any one cohort of the run re-packed after running out
    of memory under the bound whose shrink key is ``key`` — a run that shrank
    is one whose measured bound was too loose, which the author reading the
    resolved number should know. The **largest** over the points, not a sum:
    a cohort's count is reported on each of its members, so a sum would
    multiply it by the cohort's size, and what the number answers is whether,
    and how badly, the measurement missed."""
    return max(
        (
            int(summary[key])
            for summary in summaries
            if isinstance(summary.get(key), int)
        ),
        default=0,
    )


def fit_rows_shrinks(summaries: Sequence[Mapping[str, Any]]) -> int:
    """:func:`bound_shrinks` for ``fit_rows`` — the grad windows and the eval
    windows packed under it."""
    return bound_shrinks(summaries, "fit_rows_shrinks")


def measured_bounds(
    execution: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """The keys a receipt's ``execution`` block gains from what the run's
    fits measured (:data:`MEASURED_BOUNDS`): for each requested bound that is
    ``null`` and that some fit reported, the resolved number, and its shrink
    count when non-zero. An authored bound is already the receipt's; a run
    with no fit, or off CUDA where a bound stays unbounded, adds nothing."""
    out: dict[str, int] = {}
    for requested, summary_shrinks, resolved_key, shrinks_key in MEASURED_BOUNDS:
        if execution.get(requested) is not None:
            continue
        resolved = resolved_bound(summaries, requested)
        if resolved is None:
            continue
        out[resolved_key] = resolved
        shrinks = bound_shrinks(summaries, summary_shrinks)
        if shrinks:
            out[shrinks_key] = shrinks
    return out


def record_measured_bounds(
    output_directory: Path, summaries: Sequence[Mapping[str, Any]]
) -> Path | None:
    """Write the bounds the run measured (:func:`measured_bounds`) into the
    receipt's ``execution`` block and return the receipt's path — the same
    amend-after-execution as :func:`record_fires`. A caller that wrote no
    receipt has nothing to amend: ``None``."""
    receipt = output_directory / RUN_RECORD_NAME
    if not receipt.is_file():
        return None
    record = json.loads(receipt.read_text())
    execution = record.get("execution")
    if not isinstance(execution, dict):
        return receipt
    added = measured_bounds(execution, summaries)
    if not added:
        return receipt
    execution.update(added)
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return receipt


#: The receipt's ``execution`` key for the ragged-write geometry a run landed
#: under (intervention protocol spec §5 rule 19, §8): present **only**
#: when some write ran under a non-``refuse`` ``ragged`` policy, so a receipt
#: of a document that authors none is byte-identical to before the key.
RAGGED_KEY = "ragged"


def ragged_geometry(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The ``execution.ragged`` block from the points' summaries: per
    ``"<intervened_model>/<write>"`` that landed ragged, the policy, the
    per-row widths and the ``[width, rows]`` buckets — the geometry the
    executor recorded at its pre-forward width check (``check_write_widths``).

    Points of one document address the same rows, so a write's geometry is
    ordinarily one value across them; a sweep over the write's position can
    make it differ per point, and then ``widths`` / ``buckets`` are ``None``
    at the top and each point's own are recorded under ``by_point``, keyed by
    point digest — the block keeps one shape either way. Empty when no point
    landed a ragged write."""
    seen: dict[str, dict[str, Any]] = {}
    by_point: dict[str, dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        entries = summary.get(RAGGED_KEY)
        if not isinstance(entries, Mapping):
            continue
        point = str(summary.get("point"))
        for key, geometry in entries.items():
            by_point.setdefault(key, {})[point] = dict(geometry)
            seen.setdefault(key, dict(geometry))
    out: dict[str, Any] = {}
    for key, geometry in seen.items():
        variants = by_point[key]
        shapes = {json.dumps(g, sort_keys=True) for g in variants.values()}
        if len(shapes) == 1:
            out[key] = geometry
        else:
            out[key] = {
                "policy": geometry["policy"],
                "widths": None,
                "buckets": None,
                "by_point": {
                    point: {"widths": g["widths"], "buckets": g["buckets"]}
                    for point, g in sorted(variants.items())
                },
            }
    return out


def record_ragged_geometry(
    output_directory: Path, summaries: Sequence[Mapping[str, Any]]
) -> Path | None:
    """Write the ragged-write geometry the run landed under
    (:func:`ragged_geometry`) into the receipt's ``execution`` block as
    :data:`RAGGED_KEY` and return the receipt's path — the same
    amend-after-execution as :func:`record_measured_bounds`, and like it a
    fact **recorded, not gated**. Nothing is written, and no key appears, when
    no write landed ragged; a caller that wrote no receipt has nothing to
    amend: ``None``."""
    receipt = output_directory / RUN_RECORD_NAME
    if not receipt.is_file():
        return None
    geometry = ragged_geometry(summaries)
    if not geometry:
        return receipt
    record = json.loads(receipt.read_text())
    execution = record.get("execution")
    if not isinstance(execution, dict):
        return receipt
    execution[RAGGED_KEY] = geometry
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return receipt


def write_run_record(
    compiled: CompiledProtocol,
    output_directory: Path,
    selected: range,
    *,
    engine: Any = None,
) -> Path:
    """``<output_directory>/protocol.json`` — the record of what ran.

    The saved tables say what the numbers are; this says what produced them:
    the canonical document (every default materialized, dtype and quantization
    included), its digest, the per-point provenance digests, the digest of
    the method group alone, and — under ``execution`` — the batch
    geometry the chosen ``engine`` runs under (:func:`execution_record`). It is
    what someone reproducing the run reads first, and it is written **before**
    execution so a crashed run still says what it was, which is why the
    geometry is taken from the engine's bound rather than from anything the
    run observed.
    """
    record: dict[str, Any] = {
        "document_digest": compiled.digests.document,
        "canonical": compiled.canonical,
        "execution": execution_record(engine),
        "points": [
            {
                "index": index,
                "digest": compiled.digests.points[index],
                "coords": dict(compiled.points.points[index].coords),
            }
            for index in selected
        ],
    }
    # what the compile lowered (§3.2, §6): a path block's policy, receiver
    # order, restorer boundary and emitted names — derived, never canonical,
    # and bound to the digests above by reference (every emitted name is a key
    # of ``canonical.method``). Absent, not empty, when nothing was lowered.
    if compiled.lowered:
        record["derived"] = json.loads(json.dumps(compiled.lowered))
    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory / RUN_RECORD_NAME
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return target


def record_fires(
    output_directory: Path, fires: Mapping[str, Mapping[str, Mapping[str, int]]]
) -> Path | None:
    """Write the run's fire counts into the receipt as its ``fires`` block
    (:data:`FIRES_KEY`) and return the receipt's path.

    ``fires`` is ``{point digest: {"<model> on <input>": {write: count}}}``
    — per forward group with writes, how many times each member fired per
    forward (§4 "Fires"; ``neural/shared/fires.py``). The engine writes it
    **once the whole campaign has run**, never per point: a point whose member
    fired other than its declared count refuses the run before this is
    called, so a refused run's receipt carries no ``fires`` key at all rather
    than the counts of the points before it — the same all-or-nothing the
    tables keep. A caller that wrote no receipt (a workflow step records its
    run in ``_step.json``; an engine test drives ``execute_request``
    directly) has nothing to amend: ``None``.

    The counts are layout-invariant — the same under ``--batch-rows`` as
    whole — so two layouts of one document still differ in their receipts at
    ``execution.batch_rows`` and nowhere else (§8).
    """
    receipt = output_directory / RUN_RECORD_NAME
    if not receipt.is_file():
        return None
    record = json.loads(receipt.read_text())
    record[FIRES_KEY] = {
        point: {group: dict(counts) for group, counts in groups.items()}
        for point, groups in fires.items()
    }
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return receipt


def run_protocol(
    document: CompiledProtocol | LoadedProtocol | Path | Mapping[str, Any],
    environment: ResolutionEnv,
    engines: Sequence[Any],
    output_directory: Path,
    *,
    points: str | None = None,
    sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> RunResult:
    """Execute one intervention specification and return what it produced.

    ``document`` is a compiled document, a path, or the document itself as a
    tree; a path or a tree is compiled here against ``environment`` (a path's
    relative references resolve from its own directory). Pass a
    :class:`~causalab.protocol.compile.CompiledProtocol` — or the
    :class:`~causalab.protocol.loader.LoadedProtocol` view of one — when the
    compile needs options (``overrides``, a non-default ``point_cap``), or
    when the caller has already validated it and does not want a second parse.

    ``engines`` are candidate implementations of the
    :class:`~causalab.protocol.engine.Engine` contract;
    :func:`~causalab.protocol.engine.choose_engine` routes the document's
    requirements to one of them, or refuses (§8). Supplying them is the
    caller's job precisely so this module imports none.

    ``points`` shards the campaign: ``"START:STOP"`` over the expanded points.
    The slice applies to every per-point tuple in lockstep and the **campaign
    digest is untouched**, so a shard's artifacts still stamp and dedup as
    members of the whole campaign — which is what lets an external scheduler
    fan shards out and recombine them by digest.

    The run receipt is written before the first forward pass; see
    :func:`write_run_record`.

    Beside it the run appends its **event stream**, ``events.jsonl``
    (:mod:`causalab.io.events`; workflow spec §4.3): ``phase_started`` before
    the engine runs, then — once it has returned — one ``progress`` line per
    point, one ``metric`` line per summarized metric, ``result_committed``
    naming the files, ``phase_completed`` and ``campaign_terminal``. A run
    that raises leaves a stream without a terminal line. ``sink`` is the
    optional adapter handed each line after its local write; its failure
    becomes a ``warning`` line and changes nothing else. The stream is a
    sidecar: it enters no receipt, no digest and no stamp.

    Rule 25 is checked here rather than only in ``validate --data``: a
    declared row convention that does not describe the resolved batch is
    refused before the engine is chosen, so no weights load (§2.8.1). The
    engine is chosen and held to the rules that needed to know it by
    :func:`route_engine`, likewise before any weights: what the document
    decides is refused at load, and the engine is left only the conditions
    the load could not know (§5).
    """
    compiled = _compiled(document, environment)
    for point in compiled.point_documents:
        check_row_roles(point, environment)  # §5 rule 25, before any weights
    chosen = route_engine(compiled, engines)
    n_points = len(compiled.points.points)
    selected = parse_points(points, n_points) if points is not None else range(n_points)
    write_run_record(compiled, output_directory, selected, engine=chosen)
    request = ExecutionRequest(
        points=tuple(compiled.points.points[index].raw for index in selected),
        canonical=tuple(compiled.points.points[index].canonical for index in selected),
        digests=tuple(compiled.digests.points[index] for index in selected),
        coords=tuple(compiled.points.points[index].coords for index in selected),
        document_digest=compiled.digests.document,
        env=environment,
        output_dir=output_directory,
    )
    # function-local: `protocol/` keeps no module-level edge to `io/`
    from causalab.io.events import EVENTS_FILE, EventLog

    log = EventLog(
        output_directory / EVENTS_FILE,
        identity={
            "document_digest": compiled.digests.document,
            "points": [selected.start, selected.stop],
        },
        sink=sink,
    )
    log.emit("phase_started", {"phase": "execute", "n_points": len(selected)})
    result = chosen.execute(request)
    _emit_run_events(log, request, result)
    return result


def route_engine(compiled: CompiledProtocol, engines: Sequence[Any]) -> Engine:
    """Choose the engine for a compiled document and hold it to the rules
    that needed to know it — §5 rules 13 and 30 and the §8 capability
    shortfall, through :func:`~causalab.protocol.compile.check_engine` —
    before it loads a model. The one routing step both doors share: this
    module's :func:`run_protocol` and the workflow runner's protocol step.

    Routing's own refusal is generated from the missing capabilities, which
    names a verb. When the shortfall is a fact the *document* authored — a
    training field rule 30 decides — the refusal should name the field
    instead, so the engine-aware rules run per candidate first and the
    routing text is the answer only when none of them has a narrower one.
    """
    try:
        chosen = choose_engine(list(compiled.point_documents), list(engines))
    except ValidationError as shortfall:
        for engine in engines:
            for point in compiled.point_documents:
                check_engine_support(point, engine.effective_capabilities)
        raise shortfall
    check_engine(compiled, chosen.effective_capabilities)
    return chosen


def _emit_run_events(log: Any, request: ExecutionRequest, result: RunResult) -> None:
    """The lines a finished execution appends, from what the engine returned
    (§4.3). The engine runs the campaign whole, so per-point ``progress`` is
    reported when it has: one line per point in run order, each naming its
    ``point_digest``, then a ``metric`` line per summarized metric — the same
    ``_summary_stat`` value ``explain`` prints, never a fact the run receipt
    does not carry — and ``result_committed`` naming the files written."""
    total = len(request.digests)
    for index, digest in enumerate(request.digests):
        log.emit(
            "progress",
            {
                "point_digest": digest,
                "index": index,
                "completed": index + 1,
                "total": total,
            },
        )
    for summary in result.summaries:
        for name, value in sorted(summary.get("metrics", {}).items()):
            log.emit(
                "metric",
                {"point_digest": summary.get("point"), "name": name, "value": value},
            )
    log.emit("result_committed", {"files": sorted(result.files)})
    log.emit("phase_completed", {"phase": "execute", "forwards": result.forwards})
    log.emit("campaign_terminal", {"outcome": "completed"})


def _compiled(
    document: CompiledProtocol | LoadedProtocol | Path | Mapping[str, Any],
    environment: ResolutionEnv,
) -> CompiledProtocol:
    """The compile a run executes — the one handed in, or the one path."""
    if isinstance(document, CompiledProtocol):
        return document
    if isinstance(document, LoadedProtocol):
        if document.compiled is None:
            raise AssertionError(
                "a LoadedProtocol is built by load(); this one was not"
            )
        return document.compiled
    return compile_protocol(
        document,
        document.parent if isinstance(document, Path) else None,
        None,
        environment.datasets,
        environment.artifacts,
        None,
        model_info=environment.model_info,
    )
