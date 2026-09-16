"""The workflow runner (docs/workflow_protocol.md §8).

Executes a loaded workflow: steps in topological order, each step's outputs
under ``<out-root>/<output_dir>/<step>/``, protocol steps through the standard engine
routing against the run-tree/external artifact overlay, and script steps by
resolving their inputs, calling ``main(inputs, outputs)``, then verifying and
stamping what they wrote.

There is no engine choice at the workflow level — engines are chosen per
protocol step from the list the caller supplies (v2 ships one).

**The run tree is the publication.** There is no `save` section and no copy
step: a step's declared outputs land in its own directory and stay there
(§0). What the runner adds beside them is a record — ``_step.json`` per step,
``workflow.json`` for the run.

**A step is attempted, verified, then published** (§8). Its writes go to an
attempt directory (``.attempts/<step>/<id>/``, §1.1); every declared output is
verified against its format and content-digested into the record; one rename
publishes the attempt as ``<step>/``. A published step directory is therefore a
complete unit or absent, ``--resume`` reuses a unit only when the digests it
recorded still match the bytes on disk **and the package that wrote it is the
package running** (§7: the record's ``implementation.tree_digest`` is
``runtime_identity()``'s), and ``workflow.json`` is written in a ``finally`` so
even an interrupted run classifies every step — unless the derived status and
the runner's memory disagree or the stream cannot be read (§4.3); then no
manifest is written rather than a wrong one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.compile import compile_protocol
from causalab.protocol.engine import Engine, ExecutionRequest, choose_engine
from causalab.protocol.run import execution_record, measured_bounds, route_engine
from causalab.protocol.errors import ProtocolError, ProtocolWarning, ValidationError
from causalab.protocol.resolve import ArtifactStore, ResolutionEnv
from causalab.protocol.schema import tree_path
from causalab.protocol.sweep import DEFAULT_POINT_CAP, short_coords
from causalab.io.events import EVENTS_FILE, EventLog, EventSink, read_events
from causalab.io.step_record import SIDECAR, write_sidecar
from causalab.protocol.tables import TABLE_SUFFIX, read_table
from causalab.provenance import runtime_identity
from causalab.workflow import behavioral, conditional, fan_out, nested
from causalab.workflow.derived import derive_statuses
from causalab.workflow.document import (
    CONTROL_INPUT,
    CONTROLS_FILE,
    DEFAULT_STOP_AFTER_FAILURE_RATE,
    BehavioralStep,
    ConditionalStep,
    DecisionStep,
    LoadedWorkflow,
    OutputDecl,
    ProtocolStep,
    Reference,
    ScriptStep,
    certifier_subject,
    resolve_script,
)
from causalab.workflow.manifest import (
    ATTEMPTS_DIR,
    RETAINED_FAILED_ATTEMPTS,
    STDERR_TAIL_BYTES,
    AttemptFailure,
    classify_unreached,
    displaced_attempt,
    new_attempt_dir,
    prune_attempts,
    publish_attempt,
    retain_superseded,
    superseded_units,
    remove_if_empty,
    restore_displaced,
    write_attempt_record,
    write_manifest,
)
from causalab.workflow.reduction import REDUCTION_INPUT

__all__ = [
    "CERTIFIER_STATUSES",
    "ControlFailure",
    "IMPLEMENTATION_FIELDS",
    "QUALIFICATION_IDENTITY_FIELDS",
    "INSTRUMENT_FAILURE",
    "OverlayArtifacts",
    "SIDECAR",
    "WRITE_BOUNDARIES",
    "WorkflowRunResult",
    "coords_token",
    "run_workflow",
]

#: What identity a script-written tensor is stamped as coming from.
SCRIPT_ENGINE = "script"

#: The fields of a step record's ``implementation`` block (§8): what
#: :func:`causalab.provenance.runtime_identity` said about the ``causalab``
#: package that ran the step. Closed — the spec's §8 ``stamping`` row lists
#: exactly these and ``tests/workflow/test_resume_implementation.py`` holds
#: the row, this tuple and the written record together.
#:
#: Only ``tree_digest`` is *compared* on ``--resume`` (§7): it is the digest
#: of every byte that executed, so it is "the same code" and nothing else is.
#: The other three are *recorded* so a reader can see which revision produced
#: a unit that is no longer reusable: ``resolved_revision`` localizes a digest
#: mismatch to a commit, ``dirty`` says whether that revision alone names the
#: bytes, and ``version`` is the package version those bytes claimed. None of
#: them refuses — a dirty tree is legitimate work, and it already has a
#: different tree digest from the clean one when its bytes differ.
IMPLEMENTATION_FIELDS: tuple[str, ...] = (
    "tree_digest",
    "version",
    "resolved_revision",
    "dirty",
)

#: The runner's write boundaries — the instants at which a crash leaves the
#: run tree in a state ``--resume`` has to cope with (§8). Closed: each name
#: is a ``_boundary(...)`` site (``outputs_partial`` is raised from inside a
#: step's own writes, which the runner cannot interpose), and the interruption
#: test enumerates them so a new boundary cannot appear without a recovery
#: case.
WRITE_BOUNDARIES: tuple[str, ...] = (
    "attempt_created",  # the attempt dir exists, nothing written yet
    "outputs_partial",  # some declared outputs written, not all
    "outputs_written",  # every output written, none verified
    "verified",  # outputs verified and digested, no step record yet
    "recorded",  # step record written into the attempt, not published
    "displaced",  # a stale published unit moved aside, new one not yet in place
    "committed",  # the step is published on disk, not yet narrated on the stream: the manifest is refused
    "published",  # the step is published and narrated, the manifest not yet written
    "superseded",  # a displaced prior unit is about to be marked and retained as superseded (§8)
    "manifest",  # the manifest is written to its temp file, not renamed
)


class ScriptFailure(ProtocolError):
    """An isolated script exited non-zero: its stderr travels with the error so
    the failed attempt can retain a bounded tail of it (§8)."""

    def __init__(self, message: str, *, stderr: str) -> None:
        super().__init__("P2", message)
        self.stderr = stderr


class ControlFailure(ProtocolError):
    """A certified control's failure rate exceeded its declared bound, or its
    certifier's rows left a point of the control without a row (§8) — a
    control that did not run on a point cannot certify it. Either way the
    certifying step is ``failed``, so every step downstream of it is
    ``blocked`` by the manifest's own rule. Points that failed certification
    are on the stream as ``warning`` lines with ``reason: instrument_failure``;
    an uncertified point is named in the message and nowhere else."""

    def __init__(self, message: str) -> None:
        super().__init__("P2", message)


#: The ``warning`` reason a control point that failed certification carries on
#: the stream (§4.3). Not a status: ``derive_statuses`` reads only
#: ``attempt_failed`` warnings, so these lines move no step word.
INSTRUMENT_FAILURE = "instrument_failure"

#: What a certifying script's rows may say about a point. The other two words
#: of the status vocabulary (``waived``, ``not_run``) are the layer's, never a
#: certifier's.
CERTIFIER_STATUSES: tuple[str, ...] = ("passed", "failed")

#: The identity of a qualification as a dependent's record carries it (§8,
#: ``controls.identity.<control>``): the control's fully resolved document
#: digest (which document qualified), the ``implementation.tree_digest`` of the
#: code that ran it (the same digest ``--resume`` compares,
#: so a control record from another tree is re-run, never reused) and the
#: engine. Closed on purpose: nothing from the ``execution`` block (``batch_rows``,
#: ``model_source``), no device and no install path — facts a run observes,
#: which legitimately vary between two runs of one campaign — can enter it.
QUALIFICATION_IDENTITY_FIELDS: tuple[str, ...] = (
    "document_digest",
    "tree_digest",
    "engine",
)

_MISSING = object()


def coords_token(coords: Mapping[str, Any], *, entry: str | None = None) -> str:
    """One point's coordinates spelled the way a saved bundle's header spells
    them (``causalab.neural.shared.outputs.TensorFile.add``): the short axis
    names against ``entry`` — an axis on the saved read's own entity drops the
    entity prefix (``pos``, not ``recv_original.pos``) — and every non-scalar
    value as its sorted JSON text; the whole dict as sorted JSON, so two
    spellings compare as strings.

    The one function the ledger tokenizes with, on both sides of the join
    (§8): a certifying script copies its rows' ``coords`` from a header, so
    the control's points must be spelled the same way or valid work — a
    control swept on an axis of its own saved read, or on a non-scalar
    coordinate — is refused as matching no point."""
    return json.dumps(
        {
            short: _plain(value)
            for short, value in short_coords(coords, entry=entry).items()
        },
        sort_keys=True,
    )


def _plain(value: Any) -> Any:
    """The header writer's value spelling, mirrored byte for byte: a scalar
    as it is, anything else as its sorted JSON text. Mirrored rather than
    imported — the writer lives beside torch, and this module must load for
    ``validate`` without it."""
    if isinstance(value, (int, float, str, bool)):
        return value
    return json.dumps(value, sort_keys=True)


@dataclasses.dataclass
class _ControlLedger:
    """What one run knows about each control step's points (§8): the
    declaration, the control's points (digest, coordinates) with each point's
    status once known, the explicit document a coordinate is looked up in
    when the control did not sweep it, and the certifying step if any.

    ``matched_random``, ``shuffled_source`` and ``full_component`` points are
    ``passed`` when they ran — the pairing (the permutation, the swap) was
    checked at load and the comparison against the target is a reduction, not
    a status. ``self_swap`` points are ``None`` until their certifier publishes
    ``controls.json`` (``not_run`` on the control's own record, which carries
    every point's coordinates so a ``--resume`` re-seats them whether or not
    the certifier is reused)."""

    entries: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    def declare(
        self,
        name: str,
        step: ProtocolStep,
        loaded: LoadedWorkflow,
        digests: Sequence[str],
        coords: Sequence[Mapping[str, Any]],
        *,
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record a control step's points as it runs; returns the record block.

        ``identity`` is the qualification's identity every dependent inherits
        (§8): :data:`QUALIFICATION_IDENTITY_FIELDS` — the control's resolved
        document digest, the ``tree_digest`` of the code that ran it and the
        engine. Nothing the run merely *observed* (``execution``: the row
        bound, the model source) is in it, so a dependent can never be keyed
        to a fact that legitimately varies between two identical runs."""
        control = dict(step.control or {})
        kind = str(control.get("kind"))
        # a draw that ran, a permutation that ran, or a full-component swap
        # that ran, is `passed`: the comparison against the target is a
        # reduction, not a status (§2.2)
        status = (
            "passed"
            if kind in ("matched_random", "shuffled_source", "full_component")
            else None
        )
        points = [
            {"digest": digest, "coords": dict(point), "status": status}
            for digest, point in zip(digests, coords)
        ]
        self.entries[name] = {
            **control,
            "points": points,
            "raw": loaded.inner[name].raw,
            "certifier": _certifier_of(loaded, name),
            "identity": {
                field: identity[field] for field in QUALIFICATION_IDENTITY_FIELDS
            },
        }
        block: dict[str, Any] = dict(control)
        # every kind records its points with their coordinates (§8) — the
        # `--resume` path re-seats a swept control from here; a `self_swap`
        # point is `not_run` on its own record until its certifier says
        block["by_point"] = {
            p["digest"]: {"coords": p["coords"], "status": status or "not_run"}
            for p in points
        }
        block["n_points"] = len(points)
        # the load-time site-equivalence verdict against the target (rule 16,
        # §8): derived, so recorded here and never canonical
        if name in loaded.equivalence:
            block["equivalence"] = dict(loaded.equivalence[name])
        if status is not None:
            block["n_failed"] = 0
        return block

    def restore(
        self, name: str, record: Mapping[str, Any], loaded: LoadedWorkflow
    ) -> None:
        """Re-seat what a reused step's record says (``--resume``), so a
        dependent that runs again inherits the same statuses."""
        step = loaded.document.steps.get(name)
        control = record.get("control")
        if isinstance(step, ProtocolStep) and isinstance(control, Mapping):
            by_point = control.get("by_point")
            # a record written before every kind carried `by_point` names its
            # points by digest only: their coordinates are the loader's, when
            # its compile expanded the same digests
            compiled = loaded.inner[name].compiled
            known = (
                dict(
                    zip(
                        loaded.inner[name].point_digests,
                        (dict(p.coords) for p in compiled.points.points),
                    )
                )
                if compiled is not None
                else {}
            )
            points = [
                {
                    "digest": digest,
                    "coords": dict(entry.get("coords", {})),
                    "status": None
                    if entry.get("status") == "not_run"
                    else entry.get("status"),
                }
                for digest, entry in (by_point or {}).items()
            ] or [
                {
                    "digest": digest,
                    "coords": dict(known.get(digest, {})),
                    "status": None,
                }
                for digest in record.get("point_digests", [])
            ]
            implementation = record.get("implementation")
            self.entries[name] = {
                **{
                    k: v
                    for k, v in control.items()
                    if k not in ("by_point", "n_points", "n_failed")
                },
                "points": points,
                "raw": loaded.inner[name].raw,
                "certifier": _certifier_of(loaded, name),
                # the reused record's own identity (§8): `--resume` already
                # held its tree digest to the running package's
                "identity": {
                    "document_digest": record.get("document_digest"),
                    "tree_digest": implementation.get("tree_digest")
                    if isinstance(implementation, Mapping)
                    else None,
                    "engine": record.get("engine"),
                },
            }
        certifies = record.get("certifies")
        if isinstance(certifies, Mapping):
            subject = str(certifies.get("control"))
            entry = self.entries.get(subject)
            if entry is not None:
                for point in entry["points"]:
                    seen = certifies.get("by_point", {}).get(point["digest"])
                    if isinstance(seen, Mapping):
                        point["status"] = seen.get("status")

    def certify(
        self,
        certifier: str,
        subject: str,
        rows: Sequence[Mapping[str, Any]],
        bound: float,
        emit: Any,
    ) -> dict[str, Any]:
        """Join a certifier's rows onto the control's points by coordinates,
        narrate every failed point, and hold the rate to ``bound``."""
        entry = self.entries.get(subject)
        if entry is None:
            raise ProtocolError(
                "P2",
                f"step {certifier!r} certifies {subject!r}, which has not run in "
                "this process — a control is certified after it publishes",
            )
        if not rows:
            raise ProtocolError(
                "P2",
                f"step {certifier!r} wrote an empty {CONTROLS_FILE} — a certification "
                "with no points decides nothing",
            )
        by_token = self._spellings(entry)
        certified: dict[str, dict[str, Any]] = {}
        for row in rows:
            status = row.get("status")
            if status not in CERTIFIER_STATUSES:
                raise ProtocolError(
                    "P2",
                    f"step {certifier!r}: {CONTROLS_FILE} row says status {status!r}; "
                    f"a certifier says one of {list(CERTIFIER_STATUSES)}",
                )
            coords = row.get("coords")
            token = json.dumps(
                coords if isinstance(coords, Mapping) else {}, sort_keys=True
            )
            point = by_token.get(token)
            if point is None:
                raise ProtocolError(
                    "P2",
                    f"step {certifier!r}: {CONTROLS_FILE} row at coordinates {coords} "
                    f"matches no point of control {subject!r} (has "
                    f"{[json.loads(coords_token(p['coords'])) for p in entry['points']]})",
                )
            if point["digest"] in certified:
                raise ProtocolError(
                    "P2",
                    f"step {certifier!r}: {CONTROLS_FILE} holds two rows at "
                    f"coordinates {coords} — one point, one row",
                )
            certified[point["digest"]] = point
            point["status"] = str(status)
        # every point the control expanded needs a row: a control that did not
        # run on a point cannot certify it, and filling the point in would
        # only dilute the bounded rate (§8)
        uncovered = [p for p in entry["points"] if p["digest"] not in certified]
        if uncovered:
            first = uncovered[0]
            raise ControlFailure(
                f"step {certifier!r}: {CONTROLS_FILE} has no row for "
                f"{len(uncovered)} of {len(entry['points'])} points of control "
                f"{subject!r} — first {first['digest']} at coordinates "
                f"{first['coords']}; a control that did not run on a point cannot "
                f"certify it, so every step downstream of {certifier!r} is blocked"
            )
        failed = [p for p in entry["points"] if p["status"] == "failed"]
        for point in failed:
            emit(
                "warning",
                {
                    "step": subject,
                    "reason": INSTRUMENT_FAILURE,
                    "point": point["digest"],
                    "coords": point["coords"],
                    "certified_by": certifier,
                },
            )
        n_points = len(entry["points"])
        rate = len(failed) / n_points
        block = {
            "control": subject,
            "of": entry.get("of"),
            "kind": entry.get("kind"),
            "by_point": {
                p["digest"]: {"coords": p["coords"], "status": p["status"]}
                for p in entry["points"]
            },
            "n_failed": len(failed),
            "n_points": n_points,
            "stop_after_failure_rate": bound,
        }
        if rate > bound:
            raise ControlFailure(
                f"control {subject!r} ({entry.get('kind')} of {entry.get('of')!r}): "
                f"{len(failed)} of {n_points} points failed certification — rate "
                f"{rate:.3g} exceeds stop_after_failure_rate {bound:g}; the failing "
                "points are instrument_failure on the stream and every step "
                f"downstream of {certifier!r} is blocked"
            )
        return block

    @staticmethod
    def _spellings(entry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Every header spelling of every point of a control, to the point:
        :func:`coords_token` against no entity and against each value the
        control's document saves — a certifier copies its rows' ``coords``
        from one of those bundles' headers. Two points of one expansion
        differ in some axis value, and a spelling keeps every value, so no
        token names two points."""
        raw = entry.get("raw") or {}
        saves = (
            raw.get("method", {}).get("save", []) if isinstance(raw, Mapping) else []
        )
        entities: list[str | None] = [None]
        for save in saves:
            value = save.get("value") if isinstance(save, Mapping) else None
            if isinstance(value, str) and value not in entities:
                entities.append(value)
        return {
            coords_token(point["coords"], entry=entity): point
            for point in entry["points"]
            for entity in entities
        }

    def inherit(
        self,
        name: str,
        loaded: LoadedWorkflow,
        digests: Sequence[str],
        coords: Sequence[Mapping[str, Any]],
        raw: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """The ``controls`` block of a dependent step (§8): for every point,
        the status of each upstream control whose points agree with it —
        ``instrument_invalid`` where a control ``failed``, ``not_run`` where
        no control covers the point or one covering it has not been
        certified, ``passed`` otherwise — or ``None`` when no control is
        upstream of the step. Beside the statuses, ``identity`` names the
        qualification each status came from, per control (§8): the same
        triple for every point of the fanout, by construction."""
        upstream = _ancestors(loaded.dependencies, name)
        controls = sorted(
            c
            for c, entry in self.entries.items()
            if c in upstream or entry.get("certifier") in upstream
        )
        if not controls:
            return None
        by_point: dict[str, Any] = {}
        n_invalid = 0
        for digest, point in zip(digests, coords):
            per: dict[str, str] = {}
            for control in controls:
                entry = self.entries[control]
                matched = [
                    q
                    for q in entry["points"]
                    if _agree(point, raw, q["coords"], entry["raw"])
                ]
                if not matched:
                    continue  # pinned elsewhere: this control says nothing here
                if any(q["status"] in (None, "not_run") for q in matched):
                    per[control] = "not_run"
                elif any(q["status"] == "failed" for q in matched):
                    per[control] = "failed"
                else:
                    per[control] = "passed"
            if "failed" in per.values():
                status = "instrument_invalid"
                n_invalid += 1
            elif not per or "not_run" in per.values():
                status = "not_run"  # no control covers the point, or none has run
            else:
                status = "passed"
            by_point[digest] = {
                "coords": dict(point),
                "controls": per,
                "status": status,
            }
        return {
            "inherited_from": controls,
            "identity": {
                control: dict(self.entries[control].get("identity", {}))
                for control in controls
            },
            "by_point": by_point,
            "n_invalid": n_invalid,
            "n_points": len(by_point),
        }


def _certifier_of(loaded: LoadedWorkflow, control: str) -> str | None:
    steps = loaded.document.steps
    for name in steps:
        if certifier_subject(steps, name) == control:
            return name
    return None


def _ancestors(dependencies: Mapping[str, tuple[str, ...]], name: str) -> set[str]:
    """Every step upstream of ``name``, transitively."""
    seen: set[str] = set()
    pending = list(dependencies.get(name, ()))
    while pending:
        upstream = pending.pop()
        if upstream in seen:
            continue
        seen.add(upstream)
        pending.extend(dependencies.get(upstream, ()))
    return seen


def _agree(
    coords_a: Mapping[str, Any],
    raw_a: Mapping[str, Any],
    coords_b: Mapping[str, Any],
    raw_b: Mapping[str, Any],
) -> bool:
    """Two points of two documents agree when, on every axis either sweeps,
    the other point has the same value — as its own coordinate, or as the
    value its document authored there (a control pinned to one layer by
    ``set`` agrees with the dependent's point at that layer). Values are
    compared as canonical values (:func:`_same`): ``layers: 18`` and
    ``layers: [18]`` are one. An axis the other document does not have
    constrains nothing."""
    for axis, value in coords_a.items():
        other = coords_b[axis] if axis in coords_b else _lookup(raw_b, axis)
        if other is not _MISSING and not _same(other, value):
            return False
    for axis, value in coords_b.items():
        other = coords_a[axis] if axis in coords_a else _lookup(raw_a, axis)
        if other is not _MISSING and not _same(other, value):
            return False
    return True


def _same(a: Any, b: Any) -> bool:
    """Equal as canonical values. A bare index is the one-layer band ``[n]``
    (IM spec §2.4) — the fold ``schema._band`` and ``canonical._canon_site``
    make, so a sweep value or ``set`` override ``layers: 18`` and a document's
    ``layers: [18]`` are one value here as they are one digest there; a
    longer band stays a list (``18`` and ``[18, 19]`` differ). Mirrored, not
    imported: ``_band`` refuses whatever is not a band, and an axis here may
    carry any value."""
    return _as_band(a) == _as_band(b)


def _as_band(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    return value


def _lookup(raw: Mapping[str, Any], axis: str) -> Any:
    """The value an explicit document authors at a section-rooted dotted
    path, or :data:`_MISSING`; a sweep wrapper there is missing too (the
    value would be a coordinate)."""
    node: Any = raw
    for part in tree_path(axis):
        key, index = part, None
        if part.endswith("]") and "[" in part:
            key, rest = part[:-1].split("[", 1)
            index = int(rest) if rest.isdigit() else None
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
        if index is not None:
            if not isinstance(node, list) or index >= len(node):
                return _MISSING
            node = node[index]
    if isinstance(node, Mapping) and "sweep" in node:
        return _MISSING
    return node


def _boundary(name: str, step: str | None) -> None:
    """Fault-injection seam. A no-op in production; the interruption test
    replaces it with one that raises at a chosen ``(name, step)``."""
    assert name in WRITE_BOUNDARIES, name


@dataclasses.dataclass(frozen=True)
class OverlayArtifacts:
    """The §3 overlay: step outputs in the run tree shadow the external
    artifacts root. Every check is real here — this is the run-time store the
    load-time :class:`~causalab.workflow.document.DeferredArtifacts` defers
    to."""

    run_root: Path
    outer: ArtifactStore
    step_names: frozenset[str]

    def _local(self) -> Any:
        from causalab.protocol.resolve import FileArtifacts

        return FileArtifacts(root=self.run_root)

    def _in_run_tree(self, ref: str) -> bool:
        # STEP NAMES shadow the external root (§3) — exactly the names, never
        # mere directory existence: a rerun into a used output dir, or an
        # external ref matching a stray directory, must not change resolution
        return ref.split("/", 1)[0] in self.step_names

    def read_value(self, artifact: str, key: str) -> Any:
        if self._in_run_tree(artifact):
            return self._local().read_value(artifact, key)
        return self.outer.read_value(artifact, key)

    def file_digest(self, file_path: str) -> str:
        if self._in_run_tree(file_path):
            return self._local().file_digest(file_path)
        return self.outer.file_digest(file_path)

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None:
        if self._in_run_tree(file_path):
            return self._local().read_identity(file_path)
        return self.outer.read_identity(file_path)

    def resolve_path(self, file_path: str) -> Path:
        if self._in_run_tree(file_path):
            return self.run_root / file_path
        outer_resolve = getattr(self.outer, "resolve_path", None)
        if outer_resolve is None:
            raise ProtocolError(
                "P2", f"outer artifact store cannot resolve {file_path!r} to a file"
            )
        return outer_resolve(file_path)


@dataclasses.dataclass(frozen=True)
class WorkflowRunResult:
    """What one workflow run produced: the manifest (also on disk as
    ``workflow.json``) and the run tree it landed in."""

    manifest: Mapping[str, Any]
    run_root: Path


def run_workflow(
    loaded: LoadedWorkflow,
    env: ResolutionEnv,
    out_root: Path,
    engines: Sequence[Engine],
    *,
    resume: bool = False,
    reuse_nondeterministic: bool = False,
    sink: EventSink | None = None,
) -> WorkflowRunResult:
    """Execute one loaded workflow into ``<out_root>/<output_dir>/``.

    Each step is attempted in its own directory, verified, then published by
    one rename (§8). The manifest is written in the ``finally`` — a failed or
    interrupted run still classifies every step, unless the derived status
    and the runner's memory disagree or the stream cannot be read (§4.3), then
    no manifest is written rather than a wrong one — and if writing it fails
    while a step failure is in flight, the step failure is what propagates.

    Beside the manifest the run appends its **event stream**, ``events.jsonl``
    (§4.3; :mod:`causalab.io.events`): ``phase_started`` as each step's turn
    begins, ``result_committed`` at the publish moment, ``phase_completed``
    when a step is published or reused, ``warning`` for a retained failed
    attempt, and ``campaign_terminal`` once the manifest is written by a run
    that ran to its end (every step completed, or a step failed) — so a run
    that never got there, or one interrupted (``KeyboardInterrupt``,
    ``SystemExit``: the manifest is written, no terminal line), leaves a
    stream without one (:func:`_terminal`). ``sink`` is the optional
    adapter handed each line after its local write; its failure becomes a
    ``warning`` line and changes nothing else. The stream is a sidecar: it is
    an input to no reuse decision and no identity, and ``workflow.json`` is
    byte for byte what a run with no sink writes.

    **The stream is the authority for status** (§4.3, §8). Each step's
    ``status`` in ``workflow.json`` is :func:`~causalab.workflow.derived.derive_statuses`
    over the lines this run appended, not the runner's memory of what it did;
    the memory supplies the other fields (files, digests, ``error``,
    ``blocked_by``) and must agree on the word — if it does not (an emitter
    bug, or an interrupt that landed between a memory assignment and its
    emit), no manifest is written: a ``ProtocolError`` naming the step and
    both words is raised *before* the write, or the interrupt propagates as
    itself beside a ``ProtocolWarning``. A stream this run cannot read back
    at write time likewise leaves no manifest and the failure in flight as
    it was — the manifest is derived from the stream, never from memory in
    its place (:func:`_derived_statuses`). A stream this run cannot open
    (a torn ``events.jsonl`` in the run tree, a ``seq`` gap) is a
    ``ProtocolError`` here, before any step, chaining the read error that
    names ``path:line``; nothing is written — move the sidecar aside to start
    a fresh stream, or restore it byte-for-byte."""
    # Once per run, before anything is written: the identity every record of
    # this run carries and every reuse decision compares (§7). A run that
    # cannot say what code it is (`ProvenanceError`) leaves no tree claiming
    # it can.
    implementation = _implementation()
    run_root = out_root / loaded.document.output_dir
    run_root.mkdir(parents=True, exist_ok=True)
    overlay = OverlayArtifacts(
        run_root=run_root,
        outer=env.artifacts,
        step_names=frozenset(loaded.document.steps),
    )
    run_env = ResolutionEnv(
        datasets=env.datasets, artifacts=overlay, model_info=env.model_info
    )
    step_manifest: dict[str, Any] = {}
    failure: BaseException | None = None
    manifest: dict[str, Any] | None = None
    # the controls layer's memory for this run (§8): what each control step's
    # points are, and their statuses once a certifier says
    ledger = _ControlLedger()
    # beside the manifest, never inside a step directory (§4.3); a step name
    # has no dot (§5 rule 3), so the two cannot collide. Opening reads the
    # existing stream back to continue its `seq`: one it cannot read (a torn
    # tail, a foreign line, a gap) refuses the run before anything is written
    stream = run_root / EVENTS_FILE
    try:
        log = EventLog(stream, identity={}, sink=sink)
    except (ValueError, OSError) as err:
        raise ProtocolError(
            "P2",
            f"{stream} cannot be read ({err}); the run was not started — move "
            "the sidecar aside to start a fresh stream or restore it "
            "byte-for-byte",
        ) from err

    # the conditional layer's memory for this run (§2.8): every step a verdict
    # took out of the run, directly or through a step it depends on, with the
    # decision that did it — filled as each conditional runs or is reused,
    # read when the skipped step's turn comes
    skipped_by: dict[str, dict[str, Any]] = {}
    # the joins a skipped child does not skip (§2.9, `require: selected`)
    selective = fan_out.selective_joins(loaded.document.steps)
    # a nested workflow's steps (§2.10) execute rooted at `<run_root>/<step>/`,
    # against their own document's table and an overlay over that sub-root —
    # the inner document's `artifact: "<inner>/file"` strings resolve there —
    # while the stream, the manifest and `skipped_by` carry the flattened
    # names; one overlay per sub-root, built as its first step comes up — the
    # sub-root IS the key (it is `step_root`), so one document nested twice
    # gets two overlays whichever objects the loader hands back
    envs: dict[str, ResolutionEnv] = {"": run_env}
    try:
        for name in loaded.order:
            owner, local, rel = nested.locate(loaded, name)
            step = owner.document.steps[local]
            step_root = run_root / rel if rel else run_root
            step_dir = step_root / local
            if rel not in envs:
                envs[rel] = ResolutionEnv(
                    datasets=env.datasets,
                    artifacts=OverlayArtifacts(
                        run_root=step_root,
                        outer=env.artifacts,
                        step_names=frozenset(owner.document.steps),
                    ),
                    model_info=env.model_info,
                )
            step_env = envs[rel]
            restore_displaced(step_root, local, step_dir)
            log.emit("phase_started", {"step": name, "type": step.type})
            if name in skipped_by:
                # the third outcome (§2.8, §8): no attempt, no directory, no
                # `result_committed`; memory before the emit, as for a reuse
                skipped = conditional.skipped_entry(step.type, skipped_by[name])
                step_manifest[name] = skipped
                log.emit(
                    "phase_completed",
                    {
                        "step": name,
                        "status": "skipped",
                        "skipped_by": skipped["skipped_by"],
                    },
                )
                if step_dir.exists():
                    # a rerun's skip supersedes the step's earlier published
                    # unit (§8): retained and marked, never left where a
                    # reader beside its files would take it for accepted
                    _boundary("superseded", name)
                    retain_superseded(
                        step_dir,
                        step_root / ATTEMPTS_DIR / local,
                        superseded_by={
                            "attempt": None,
                            "identity": None,
                            "published": None,
                            "skipped_by": skipped["skipped_by"],
                        },
                    )
                _attach_superseded(step_root, local, skipped)
                continue
            reused = _reusable(
                owner,
                local,
                step,
                step_dir,
                resume,
                reuse_nondeterministic,
                implementation,
                engines,
            )
            if reused is not None:
                step_manifest[name] = reused
                ledger.restore(local, reused, owner)
                if isinstance(step, ConditionalStep):
                    # a reused conditional re-seats its verdict from its record
                    conditional.fold_skips(
                        name, _flattened_skips(loaded, rel, reused), loaded, skipped_by
                    )
                log.emit("phase_completed", _completed_payload(name, reused))
                # the retained prior units are the tree's, not the record's:
                # a reused entry lists them as a fresh run's does (§8)
                _attach_superseded(step_root, local, reused)
                continue
            try:
                # §2.8: a required receipt is checked before the step is
                # scheduled — before an attempt directory, before any engine
                # is chosen (`route_engine` is inside `_attempt_step`), before
                # any device. A refusal is a failed attempt like any other,
                # and it changes no file under `step_dir`: an earlier published
                # unit stays as published (`disposition: accepted`) while the
                # manifest's `failed` is the authority (§2.8) — the skip path
                # above retains and marks because a skip is a decision about
                # the run; a refusal is an attempt that produced nothing
                for _, container, at in nested.containers(loaded, name):
                    # a receipt on the `workflow` step itself (§2.10): checked
                    # before any of its steps is allocated. The rebased step
                    # against the run root — a flattened producer name is the
                    # path to its receipt, so both names in the refusal are
                    # the ones the manifest and the stream carry (`a/b`
                    # requires `a/gate_k`; never `b`, never `gate_k`)
                    flat = nested.qualified(at, container)
                    conditional.check_receipt(
                        flat, loaded.document.steps[flat], run_root
                    )
                conditional.check_receipt(name, loaded.document.steps[name], run_root)
                entry, displaced = _attempt_step(
                    local,
                    step,
                    owner,
                    step_env,
                    step_root,
                    engines,
                    implementation,
                    ledger=ledger,
                    emit=log.emit
                    if owner is loaded
                    else _emit_as(log.emit, local, name),
                    skipped=nested.local_skips(rel, skipped_by),
                )
            except BaseException as err:
                step_manifest[name] = {
                    "type": step.type,
                    "status": "failed",
                    "error": {"type": type(err).__name__, "message": str(err)},
                }
                # the attempt is retained under `.attempts/` (§8); the stream
                # says so, and then the failure propagates as before
                log.emit(
                    "warning",
                    {
                        "step": name,
                        "reason": "attempt_failed",
                        "error": step_manifest[name]["error"],
                    },
                )
                raise
            step_manifest[name] = entry
            if isinstance(step, ConditionalStep):
                conditional.fold_skips(
                    name, _flattened_skips(loaded, rel, entry), loaded, skipped_by
                )
            # the publish moment: the verified attempt is `<step>/` now (§8),
            # and memory says so before the stream does. A run dying at the
            # `committed` seam leaves the two disagreeing (`completed` vs
            # `pending`), so the `finally` refuses the manifest and `--resume`
            # finds the unit. Past the two emits the stream and the memory
            # agree, so a failure at `published` — after the publish, before
            # the manifest — gets a manifest whose `completed` is derived
            _boundary("committed", name)
            log.emit("result_committed", {"step": name, "files": list(entry["files"])})
            log.emit("phase_completed", _completed_payload(name, entry))
            _boundary("published", name)
            if displaced is not None:
                # supersession preserves (§8): the unit this publish displaced
                # is marked and retained, never deleted — after the publish is
                # narrated, so a death here leaves a published step and a
                # displaced unit the next run's `restore_displaced` retains
                _boundary("superseded", name)
                retain_superseded(
                    displaced,
                    step_root / ATTEMPTS_DIR / local,
                    superseded_by={
                        "attempt": displaced_attempt(displaced),
                        "identity": entry["identity"],
                        "published": name,
                    },
                )
            _attach_superseded(step_root, local, entry)
    except BaseException as err:
        failure = err
        raise
    finally:
        entries: dict[str, Any] = {
            **step_manifest,
            **classify_unreached(
                loaded.order, loaded.dependencies, step_manifest, selective=selective
            ),
        }
        # §4.3: the stream is the authority for status. `derived` is the word
        # each step carries — or None when no manifest may be written and a
        # failure is propagating out of this `finally` as itself (a
        # `ProtocolWarning` has said why); on a clean run the same conditions
        # are a `ProtocolError` instead
        derived = _derived_statuses(log, loaded, entries, failure, selective)
        if derived is not None:
            manifest = {
                "output_dir": loaded.document.output_dir,
                "steps": {
                    name: {**entry, "status": derived[name]}
                    for name, entry in entries.items()
                },
            }
            if loaded.nondeterministic:
                manifest["nondeterministic"] = list(loaded.nondeterministic)
            if loaded.nested:
                # record-only (§2.10, §8), like `nondeterministic`: which steps
                # each nested workflow contributed and the inner digest its
                # entry carries — never canonical, never compared on --resume
                manifest["nested"] = nested.manifest_block(loaded)
            written = False
            try:
                write_manifest(
                    run_root, manifest, between=lambda: _boundary("manifest", None)
                )
                written = True
            except BaseException as manifest_err:
                if failure is None:
                    raise
                # the step failure is the finding; a manifest that could not be
                # written is reported beside it, never in its place
                warnings.warn(
                    f"workflow.json could not be written ({manifest_err!r}); "
                    f"the step failure {failure!r} is re-raised",
                    ProtocolWarning,
                    stacklevel=2,
                )
            if written:
                # the manifest is on disk, so the stream may say the run ran
                # to its end — if it did: an interrupt writes no terminal line
                # (`_terminal`). Only after the manifest (§4.3) — a stream
                # with a terminal line and no manifest would be a lie.
                _terminal(log, manifest, failure)
    for sub_root in nested.sub_roots(loaded):
        # a nested sub-root (§2.10) a clean run left nothing in: its
        # `.attempts/`, then the directory itself when every step was skipped
        remove_if_empty(run_root / sub_root / ATTEMPTS_DIR)
        remove_if_empty(run_root / sub_root)
    remove_if_empty(run_root / ATTEMPTS_DIR)
    if manifest is None:
        # unreachable by construction — the manifest is withheld only while a
        # failure propagates out of the `finally` — but the returned
        # scientific record is guarded by a refusal, not by an `assert` that
        # `python -O` drops
        raise ProtocolError(
            "P2",
            "workflow.json was not written and no failure is in flight; the "
            "run has no manifest to return",
        )
    return WorkflowRunResult(manifest=manifest, run_root=run_root)


def _emit_as(emit: Any, local: str, name: str) -> Any:
    """``_attempt_step`` narrates under the name it was handed — an inner
    step's local one — while the stream carries flattened names (§2.10,
    §4.3): a payload whose ``step`` is *this* step is rewritten to ``name``;
    one naming another step (a control's subject) is left for its own owner
    to spell, never renamed to the caller."""

    def wrapped(event: str, payload: Mapping[str, Any]) -> Any:
        return emit(
            event,
            {**payload, "step": name} if payload.get("step") == local else payload,
        )

    return wrapped


def _flattened_skips(
    loaded: LoadedWorkflow, rel: str, entry: Mapping[str, Any]
) -> dict[str, Any]:
    """A conditional's record as ``fold_skips`` reads it (§2.10): its
    ``skipped`` names and its evidence's ``step`` — the producer, which
    ``fold_skips`` copies into every ``skipped_by`` block as ``decision_step``
    — spelled under the conditional's sub-root, so the block's two names are
    both manifest keys; and every ``workflow`` step among the skipped
    replaced by its steps — ``on_false: ["tail"]`` skips every ``tail/*``,
    each its own entry (§8). The record itself keeps what the conditional
    wrote."""
    skipped = nested.prefix_skips(rel, entry.get("skipped") or ())
    evidence = dict(entry.get("evidence") or {})
    if rel and evidence.get("step") is not None:
        evidence["step"] = nested.qualified(rel, str(evidence["step"]))
    return {
        **entry,
        "evidence": evidence,
        "skipped": nested.expand_skips(loaded.document.steps, skipped),
    }


def _completed_payload(name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """The ``phase_completed`` payload (§4.3): the step, its word and — for a
    protocol step — ``forwards``, the forward groups its engine ran, so a
    reader of the stream can see a qualification ran once for its target's
    whole fanout without opening the step record."""
    payload: dict[str, Any] = {"step": name, "status": entry["status"]}
    if "forwards" in entry:
        payload["forwards"] = entry["forwards"]
    return payload


def _derived_statuses(
    log: EventLog,
    loaded: LoadedWorkflow,
    entries: Mapping[str, Mapping[str, Any]],
    failure: BaseException | None,
    selective: frozenset[str] = frozenset(),
) -> dict[str, str] | None:
    """§4.3: the word each step carries in ``workflow.json``, derived from the
    lines this run appended to the stream — or ``None`` when no manifest may
    be written and the ``failure`` in flight is left to propagate as itself.

    The in-memory ``entries`` supply every other field and must agree on the
    word. Two things stop the stream from deciding:

    * **it cannot be read back** (a torn tail, a foreign line — a sidecar IO
      problem, not a status). The read of a sidecar is held to the rule its
      writes follow (:func:`write_manifest`, :func:`_terminal`): it may not
      mask a step failure in flight. With one propagating this warns and
      withholds the manifest; on a clean run it is a ``ProtocolError`` chaining
      the read error. The manifest is derived from the stream and is not
      written without it — never from memory in its place, which would be the
      memory-sourced manifest §4.3 retires.
    * **it disagrees with memory** — an emitter bug, or an interrupt that
      landed between a memory assignment and its emit. An emitter bug is
      refused: a ``ProtocolError`` naming the step and both words, chaining an
      ``Exception`` in flight so neither finding is lost. An interrupt
      (``KeyboardInterrupt``, ``SystemExit`` — a ``BaseException`` that is not
      an ``Exception``) stays an interrupt: this warns, withholds the manifest
      and lets it propagate unchanged, exit code included.
    """
    try:
        derived = derive_statuses(
            (r for r in read_events(log.path) if r["seq"] >= log.opened_at),
            order=loaded.order,
            dependencies=loaded.dependencies,
            selective=selective,
        )
    except (ValueError, KeyError, TypeError, OSError) as read_err:
        if failure is None:
            raise ProtocolError(
                "P2",
                "workflow.json was not written: events.jsonl could not be read "
                f"({read_err}); the manifest is derived from the stream and is "
                "not written without it",
            ) from read_err
        warnings.warn(
            "workflow.json was not written: events.jsonl could not be read "
            f"({read_err!r}); the step failure {failure!r} is re-raised",
            ProtocolWarning,
            stacklevel=3,
        )
        return None
    for name, entry in entries.items():
        if entry["status"] == derived[name]:
            continue
        if failure is not None and not isinstance(failure, Exception):
            warnings.warn(
                f"workflow.json was not written: step {name!r} is "
                f"{entry['status']!r} in memory and {derived[name]!r} on the "
                f"stream because {failure!r} landed between the two",
                ProtocolWarning,
                stacklevel=3,
            )
            return None
        raise ProtocolError(
            "P2",
            f"workflow.json would say step {name!r} is {entry['status']!r} "
            f"but events.jsonl records {derived[name]!r}; the manifest is "
            "derived from the stream and is not written disagreeing with it",
        ) from failure
    return derived


def _terminal(
    log: EventLog, manifest: Mapping[str, Any], failure: BaseException | None
) -> None:
    """The stream's last line, for a run that ran to its end: every step
    completed, or a step failed and the runner still finished (the manifest is
    written). An interrupt — ``KeyboardInterrupt``, ``SystemExit``: a
    ``BaseException`` that is not an ``Exception`` — also gets its manifest
    but is not an end the run reached: no line is appended, so ``terminal()``
    reads False exactly as it does for a run that died before its ``finally``
    (§4.3: the absence is what "did not finish" reads as). A sidecar write may
    not mask a step failure in flight — the same rule the manifest write
    follows — so while one is propagating, a failure here is a warning beside
    it; on a clean run it is raised like any other write."""
    if failure is not None and not isinstance(failure, Exception):
        return
    payload = {
        "outcome": "failed" if failure is not None else "completed",
        "steps": {name: entry["status"] for name, entry in manifest["steps"].items()},
    }
    try:
        log.emit("campaign_terminal", payload)
    except BaseException as stream_err:
        if failure is None:
            raise
        warnings.warn(
            f"events.jsonl could not be appended ({stream_err!r}); "
            f"the step failure {failure!r} is re-raised",
            ProtocolWarning,
            stacklevel=3,
        )


def _implementation() -> dict[str, Any]:
    """The running package's identity as a step record carries it (§7, §8):
    the :data:`IMPLEMENTATION_FIELDS` of ``runtime_identity()``.

    Asked once per run, not per step. ``runtime_identity()`` hashes every file
    of the installed package the first time it is called (a fraction of a
    second for the few hundred files the package ships) and is cached for the
    process,
    so every step of one run carries the same answer by construction. Imported
    as a module attribute so a test can stand in a different identity.
    ``location`` — an absolute path — is deliberately not recorded: a record
    that carried it would change when the tree moved, and moving a run tree
    alone must not bust reuse."""
    identity = runtime_identity()
    return {
        "tree_digest": identity.tree_digest,
        "version": identity.version,
        "resolved_revision": identity.resolved_revision,
        "dirty": identity.dirty,
    }


def _attempt_step(
    name: str,
    step: Any,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    run_root: Path,
    engines: Sequence[Engine],
    implementation: Mapping[str, Any],
    *,
    ledger: _ControlLedger | None = None,
    emit: Any = None,
    skipped: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Attempt → verify → publish for one step (§8): the step's record, and
    the prior unit this publish displaced (or ``None``), which the caller
    retains as superseded once the publish is narrated.

    A declared fan-out (§2.9) runs through the same path: a child (``shard``
    set) is its parent's document over a point selection; the parent
    (``fan_out`` set) is the join, an arm here and nothing in the run loop —
    ``skipped`` is the run's ``skipped_by`` map, which a ``selected`` join
    reads to name the children a verdict left out.

    Every write goes to a fresh attempt directory. On any failure —
    ``KeyboardInterrupt`` included — the attempt keeps bounded metadata
    (``attempt.json``) and the exception propagates; the published step
    directory, if any, is untouched.

    The controls layer sits inside the attempt (§8): a control step's points
    are recorded as it runs; a certifying step's ``controls.json`` is joined
    onto its control's points **before** the step is verified and published,
    so a failure rate over the declared bound is a failed attempt like any
    other; and a dependent protocol step's record carries the statuses it
    inherits, written by its own run into its own directory."""
    step_dir = run_root / name
    prune_attempts(run_root, name, keep=RETAINED_FAILED_ATTEMPTS - 1)
    attempt_dir = new_attempt_dir(run_root, name)
    started = time.time()
    ledger = ledger if ledger is not None else _ControlLedger()
    emit = emit if emit is not None else (lambda event, payload: None)
    try:
        _boundary("attempt_created", name)
        shard = getattr(step, "shard", None)
        if (
            isinstance(step, (ProtocolStep, BehavioralStep))
            and step.fan_out is not None
        ):
            # the join (§2.9): the children's records and tables re-assembled
            # by point digest into one receipt; no engine runs here
            entry = fan_out.run_join_step(
                name,
                step,
                loaded,
                run_root,
                attempt_dir,
                implementation,
                skipped=skipped,
            )
        elif isinstance(step, ProtocolStep):
            entry = _run_protocol_step(
                name,
                step,
                loaded,
                run_env,
                attempt_dir,
                engines,
                implementation,
                ledger=ledger,
                selection=None if shard is None else shard["points"],
            )
        elif isinstance(step, BehavioralStep):
            # the declarative behavioral runner (§2.7): the same attempt →
            # verify → publish path, its record built in behavioral.py
            entry = behavioral.run_behavioral_step(
                name,
                step,
                loaded,
                run_env,
                attempt_dir,
                engines,
                implementation,
                selection=None if shard is None else shard["points"],
            )
        elif isinstance(step, DecisionStep):
            # a typed decision over a values object (§2.8): decision.json
            # written into the attempt, verified and published like any file
            entry = conditional.run_decision_step(
                name, step, loaded, run_root, attempt_dir, implementation
            )
        elif isinstance(step, ConditionalStep) and name in loaded.children:
            # a per-child conditional's parent (§2.9): one verdict per child,
            # read from the children's records — the children folded the skips
            entry = fan_out.run_conditional_join(
                name, step, loaded, run_root, implementation
            )
        elif isinstance(step, ConditionalStep):
            # the verdict over a producer's decision.json (§2.8): a record and
            # no data file; the skips it names are folded in by the caller
            entry = conditional.run_conditional_step(
                name, step, loaded, run_root, implementation
            )
        else:
            entry = _run_script_step(
                name, step, loaded, run_root, attempt_dir, implementation
            )
            subject = certifier_subject(loaded.document.steps, name)
            if subject is not None:
                control_step = loaded.document.steps[subject]
                bound = (
                    control_step.stop_after_failure_rate
                    if isinstance(control_step, ProtocolStep)
                    and control_step.stop_after_failure_rate is not None
                    else DEFAULT_STOP_AFTER_FAILURE_RATE
                )
                entry["certifies"] = ledger.certify(
                    name,
                    subject,
                    read_table(attempt_dir / CONTROLS_FILE),
                    bound,
                    emit,
                )
        if shard is not None:
            # a child's record (§2.9): its own identity — the parent's entry
            # digest plus its selection, what `--resume` compares — and the
            # shard it ran, beside the sliced `points` / `point_digests`
            entry["identity"] = _step_identity(loaded, name, step)
            entry["shard"] = json.loads(json.dumps(dict(shard)))
        _boundary("outputs_written", name)
        entry["digests"], entry["checks"] = _verify_outputs(
            name, step, attempt_dir, entry["files"]
        )
        _boundary("verified", name)
        # the record's disposition (§8, a closed vocabulary): `candidate`
        # while it sits in the attempt, `accepted` as it is published — a
        # reader never infers acceptance from absence
        entry["disposition"] = "candidate"
        write_sidecar(attempt_dir, entry)
        _boundary("recorded", name)
        entry["disposition"] = "accepted"
        write_sidecar(attempt_dir, entry)
        displaced = publish_attempt(
            attempt_dir, step_dir, between=lambda: _boundary("displaced", name)
        )
    except BaseException as err:
        stderr = err.stderr if isinstance(err, ScriptFailure) else None
        write_attempt_record(
            attempt_dir,
            step=name,
            started=started,
            failure=AttemptFailure.from_exception(err, stderr),
            declared=_declared_files(step),
        )
        raise
    remove_if_empty(attempt_dir.parent)
    return entry, displaced


def _attach_superseded(run_root: Path, name: str, entry: dict[str, Any]) -> None:
    """The step's retained prior units on its manifest entry (§8) — read from
    the run tree, the same on the skip, the reuse and the publish path, so one
    tree lists one `superseded` whichever path a run took to the step."""
    retained = superseded_units(run_root, name)
    if retained:
        entry["superseded"] = retained


def _declared_files(step: Any) -> dict[str, str]:
    """``{relative file: slot}`` for a script step, the decision file for a
    decision step; a protocol step's files are known only after its engine
    ran."""
    if isinstance(step, ScriptStep):
        return {decl.file: slot for slot, decl in step.outputs.items()}
    if isinstance(step, DecisionStep):
        return {behavioral.DECISION_FILE: "decision"}
    return {}


def _reusable(
    loaded: LoadedWorkflow,
    name: str,
    step: Any,
    step_dir: Path,
    resume: bool,
    reuse_nondeterministic: bool,
    implementation: Mapping[str, Any],
    engines: Sequence[Engine],
) -> dict[str, Any] | None:
    """The prior run's record for ``name`` if ``--resume`` may reuse it (§8).

    The digest comparison is what makes this correct: a script step's digest
    carries its script's content hash, so editing the script busts the reuse.
    A step that declared itself non-deterministic is never reused silently —
    replaying it is exactly what it said it cannot guarantee.

    The **implementation** must be the same code too (§7): the record's
    ``implementation.tree_digest`` — the bytes of the ``causalab`` package
    that ran the step — must equal the running package's, and a record with
    no such block (one written before it was recorded) is not trusted, like a
    record without digests. Refusal here is silent re-execution, exactly as a
    digest mismatch is. ``dirty`` is never consulted: a dirty tree whose bytes
    differ already has a different tree digest, and equal digests are the
    same code whatever the flag says — a dirty tree is legitimate work.

    A protocol step's record carries its **engine** — the third member of the
    qualification identity (:data:`QUALIFICATION_IDENTITY_FIELDS`) — and it
    must be the engine the step would run under now (:func:`_engine_for`): a
    same-tree run under another engine re-runs the step rather than
    inheriting the old engine's identity into its dependents, and a record
    without the key is not trusted, as one without an ``implementation``
    block is not. When no configured engine covers the step on this host
    (:func:`_engine_for` is ``None``) there is no engine to compare against,
    and a record that carries its own is reused: a content-digest-verified
    record of work done is not invalidated by a host that could not redo it
    (fail-closed on a valid ``--resume`` under ``auto``, where the install may
    have changed). A script step records no engine (its outputs' stamps carry
    :data:`SCRIPT_ENGINE`), so nothing is compared for it — and neither is
    for a fanned-out parent (§2.9): its record is the join, which names no
    engine because no engine ran it; its children's records carry theirs,
    and :func:`fan_out.evidence_holds` below binds the join to them.

    And the record's **content digests** must match the published bytes.
    Existence is never enough: a truncated output, a file overwritten in
    place, or a record from before digests were recorded all mean the unit is
    not reusable, and the step runs again.

    And for a **conditional or a decision** the evidence must hold (§2.8): a
    conditional's record names the ``evidence_identity`` it read;
    if the producer's current ``decision.json`` carries another, the numbers
    behind the verdict moved and the conditional is re-evaluated. A decision
    whose values file or producer identity moved is re-made the same way.
    ``evidence_identity`` is a run-time value, so this clause — not the
    load-time digest — is where it binds.

    And a step of **any kind** that declares ``requires_receipt`` is reusable
    only while the producer's current ``decision.json`` still carries the
    required outcome (§2.8): a receipt that flipped since the step ran is
    never reused — the step goes back through ``check_receipt``, which
    refuses."""
    if not resume:
        return None
    record_path = step_dir / SIDECAR
    if not record_path.is_file():
        return None
    try:
        with record_path.open() as handle:
            record = json.load(handle)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    want = _step_identity(loaded, name, step)
    if record.get("identity") != want:
        return None
    if isinstance(step, ProtocolStep) and step.fan_out is None:
        want_engine = _engine_for(loaded, name, engines)
        recorded_engine = record.get("engine")
        if recorded_engine is None:
            return None
        if want_engine is not None and recorded_engine != want_engine:
            return None
    recorded = record.get("implementation")
    if not isinstance(recorded, dict):
        return None
    if recorded.get("tree_digest") != implementation["tree_digest"]:
        return None
    if isinstance(step, ScriptStep) and not step.is_deterministic:
        if not reuse_nondeterministic:
            return None
    files = record.get("files")
    digests = record.get("digests")
    if not isinstance(files, list) or not isinstance(digests, dict):
        return None
    if set(digests) != {str(rel) for rel in files}:
        return None
    for rel, want_digest in digests.items():
        target = step_dir / str(rel)
        if not target.is_file() or _sha256(target) != want_digest:
            return None
    if not conditional.evidence_holds(step, step_dir, record):
        return None
    if not fan_out.evidence_holds(step, step_dir, record):
        return None
    return {**record, "status": "reused"}


def _step_identity(loaded: LoadedWorkflow, name: str, step: Any) -> str:
    """What `--resume` compares: the step's own digest — a behavioral step's
    canonical entry carries its decoding, checker, split and decision, so a
    changed seed or split is a step that runs again (§2.7, §7); a decision's
    and a conditional's carry the rule, the predicate and the gated sides
    (§2.8). A protocol step's is its inner document's digest, as before — except
    a fanned-out one and a child (§2.9): the join's identity is its entry
    digest (the `fan_out` block is in it) and a child's is its parent's plus
    its shard, so a changed width is a step that runs again."""
    if isinstance(step, (ScriptStep, BehavioralStep, DecisionStep, ConditionalStep)):
        return loaded.step_digests[name]
    if isinstance(step, ProtocolStep) and (
        step.fan_out is not None or step.shard is not None
    ):
        return loaded.step_digests[name]
    return loaded.inner_digests[name]


def _engine_for(
    loaded: LoadedWorkflow, name: str, engines: Sequence[Engine]
) -> str | None:
    """The ``engine`` a protocol step's record carries when it runs now — what
    a reused record must match (§8): the name of the engine
    :func:`route_engine` chooses for it, which is :func:`choose_engine`'s
    capability match over the point documents — computed here over the
    load-time compile, whose verbs are the run-time compile's (a document's
    needs are authored, never resolved from an artifact). ``None`` when no
    configured engine covers the step: nothing is compared, a record carrying
    its own ``engine`` is reused, and a step that does run has its own routing
    refuse with the real message."""
    try:
        return choose_engine(
            list(loaded.inner[name].point_documents), list(engines)
        ).name
    except ValidationError:
        return None


# --------------------------------------------------------------------------- #
# protocol steps
# --------------------------------------------------------------------------- #


def _run_protocol_step(
    name: str,
    step: ProtocolStep,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    step_dir: Path,
    engines: Sequence[Engine],
    implementation: Mapping[str, Any],
    *,
    ledger: _ControlLedger | None = None,
    selection: Sequence[int] | None = None,
) -> dict[str, Any]:
    """One protocol step's attempt (§8). ``selection`` — a fanned-out child's
    point indices (§2.9) — slices every per-point tuple of the request in
    lockstep, as ``run_protocol``'s ``--points`` does; the document digest is
    untouched, so a child's artifacts stamp as members of the whole campaign.
    ``None`` runs every point."""
    doc_path = (loaded.workflow_dir / step.document).resolve()
    # The one compiler (IM spec §9), with the step's inputs: the document, its
    # own directory for relative artifact paths, the step's `set` as the
    # overrides (section-rooted, IM spec §1), and the run-tree overlay
    # as the artifact store — real resolution now, since earlier steps' outputs
    # exist. Validation compiled the same document through the same function
    # against the deferring store, so the two cannot resolve it differently.
    inner = compile_protocol(
        doc_path,
        doc_path.parent,
        step.set,
        run_env.datasets,
        run_env.artifacts,
        None,
        point_cap=step.max_points if step.max_points is not None else DEFAULT_POINT_CAP,
        model_info=run_env.model_info,
    )
    # routing, and the rules that needed the engine (IM spec §5 rules 13, 30)
    # against the one it chose — before it loads a model, as `run_protocol`
    engine = route_engine(inner, engines)
    indices = range(len(inner.points.points)) if selection is None else tuple(selection)
    if any(i < 0 or i >= len(inner.points.points) for i in indices):
        raise ProtocolError(
            "P2",
            f"step {name!r}: its shard selects point indices {list(indices)} of a "
            f"document that compiled {len(inner.points.points)} point(s) — the "
            "run-time compile and the load-time expansion disagree",
        )
    request = ExecutionRequest(
        points=tuple(inner.points.points[i].raw for i in indices),
        canonical=tuple(inner.points.points[i].canonical for i in indices),
        digests=tuple(inner.digests.points[i] for i in indices),
        coords=tuple(inner.points.points[i].coords for i in indices),
        document_digest=inner.digests.document,
        env=run_env,
        output_dir=step_dir,
        # the step's own row bounds (§2.2) — execution, so they ride on the
        # request and never on the compiled document
        execution=step.execution,
    )
    result = engine.execute(request)
    record: dict[str, Any] = {
        "type": "intervention_protocol",
        "status": "completed",
        "identity": inner.digests.document,
        "implementation": dict(implementation),  # the code that ran it (§7)
        "document": step.document,
        "engine": engine.name,
        "document_digest": inner.digests.document,  # fully resolved (§7)
        "points": len(request.points),
        "point_digests": list(request.digests),  # the provenance units (§7)
        # the sweep axes a downstream script groups by (§6)
        "axes": [axis.id for axis in inner.points.axes],
        "files": sorted(result.files),
        # the row bounds this step ran under — the engine's, overridden by the
        # step's own `execution` block; declared before execution, `null` when
        # unbounded (IM spec §8) — and the one recorder of them, the same block
        # `protocol.json` carries for a document run; execution, so it enters
        # no digest and no stamp
        "execution": execution_record(engine, request),
        # the forward groups the engine actually ran for this step (§4.3,
        # §8) — a record field, never an identity field: `--resume` compares
        # `identity`, `implementation` and the content digests, not this
        "forwards": result.forwards,
    }
    # the bounds the step measured rather than authored (IM spec §8): the
    # numbers to pin in the step's `execution` block to reproduce the step
    record["execution"].update(measured_bounds(record["execution"], result.summaries))
    # the controls layer (§2.2, §8) — like `reduction`, recorded only when
    # authored: the step's own declaration and its waivers, and the statuses
    # it inherits from every control upstream of it, joined by coordinates
    if step.waive is not None:
        record["waive"] = {kind: dict(w) for kind, w in step.waive.items()}
    if ledger is not None:
        digests = list(request.digests)
        coords = [dict(point) for point in request.coords]
        if step.control is not None:
            record["control"] = ledger.declare(
                name,
                step,
                loaded,
                digests,
                coords,
                identity={
                    "document_digest": inner.digests.document,
                    "tree_digest": implementation["tree_digest"],
                    "engine": engine.name,
                },
            )
        inherited = ledger.inherit(name, loaded, digests, coords, inner.points.explicit)
        if inherited is not None:
            record["controls"] = inherited
    return record


# --------------------------------------------------------------------------- #
# script steps
# --------------------------------------------------------------------------- #


def _run_script_step(
    name: str,
    step: ScriptStep,
    loaded: LoadedWorkflow,
    run_root: Path,
    step_dir: Path,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve inputs, run the script, verify and stamp its outputs (§4)."""
    from causalab.io import step_io

    resolved, tensor_identities = _resolve_inputs(
        name, step, run_root, loaded.workflow_dir
    )
    if step.reduction is not None:
        # the authored reduction contract travels the channel inputs already
        # travel (§2.6): the script reads it as `inputs["reduction"]`, in
        # process or across the isolation boundary alike. Rule 12 has refused
        # any authored input of the same name, so nothing is shadowed here.
        resolved[REDUCTION_INPUT] = json.loads(json.dumps(dict(step.reduction)))
    subject = certifier_subject(loaded.document.steps, name)
    if subject is not None:
        # a certifying step gets the declaration it certifies the same way
        # (§2.2); rule 14 has refused an authored input of this name
        control_step = loaded.document.steps[subject]
        declaration = (
            dict(control_step.control or {})
            if isinstance(control_step, ProtocolStep)
            else {}
        )
        resolved[CONTROL_INPUT] = json.loads(
            json.dumps({"step": subject, **declaration})
        )
    outputs = {slot: step_dir / decl.file for slot, decl in step.outputs.items()}

    if step.runtime and step.runtime.get("isolate"):
        _run_isolated(name, step, loaded, resolved, outputs)
    else:
        _run_in_process(name, step, loaded, resolved, outputs)

    identity = step_io.inherited_identity(tensor_identities)
    identity["produced_by"] = loaded.step_digests[name]
    identity["engine"] = SCRIPT_ENGINE

    # stamping rewrites a bundle, so it happens before the outputs are
    # verified and digested (`_verify_outputs`) — the digest is of the bytes
    # that get published
    for slot, decl in step.outputs.items():
        target = outputs[slot]
        what = f"step {name!r}: output {slot!r} ({decl.file})"
        if not target.is_file():
            raise ProtocolError(
                "P2",
                f"{what} was not written — a script step must create every "
                "output it declares",
            )
        if decl.suffix == ".safetensors":
            step_io.stamp_tensor(target, identity, what=what)

    return {
        "type": "script",
        "status": "completed",
        "identity": loaded.step_digests[name],
        "implementation": dict(implementation),  # the code that ran it (§7)
        "script": step.script,
        "script_sha256": step.script_sha256,
        # which sibling files the identity hashed beside a `{"path": …}` script
        # (§4.2, §7) — the keys the canonical entry carries, when it does
        **(
            {"closure": dict(step.closure), "closure_sha256": step.closure_sha256}
            if step.closure
            else {}
        ),
        "digest": loaded.step_digests[name],
        "is_deterministic": step.is_deterministic,
        "inputs": {
            key: (value.target if isinstance(value, Reference) else value)
            for key, value in step.inputs.items()
        },
        "axes": [],  # a script step carries no sweep coordinates of its own
        "files": sorted(decl.file for decl in step.outputs.values()),
        **({"runtime": dict(step.runtime)} if step.runtime else {}),
        # the declaration a review reads from the tree (§2.6) — like
        # `runtime`, recorded only when authored
        **({"reduction": dict(step.reduction)} if step.reduction else {}),
    }


def _resolve_inputs(
    name: str, step: ScriptStep, run_root: Path, workflow_dir: Path
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    """The §3 grammar, resolved: a locator becomes a path, a selector reads
    through it. Also returns the identity of every tensor input, which is what
    a safetensors output inherits (§4). A relative ``path`` resolves against
    ``workflow_dir``, exactly as rule 4 checked it at load."""
    from causalab.io import step_io

    resolved: dict[str, Any] = {}
    identities: list[Mapping[str, Any]] = []
    for slot, value in step.inputs.items():
        if not isinstance(value, Reference):
            resolved[slot] = value
            continue
        what = f"step {name!r}: input {slot!r} ({value.target})"
        if value.step is not None:
            target = run_root / value.step / str(value.file)
        else:
            candidate = Path(str(value.path))
            target = (
                candidate
                if candidate.is_absolute()
                else (workflow_dir / candidate).resolve()
            )
        if not target.is_file():
            raise ProtocolError("P2", f"{what} does not exist at {str(target)!r}")
        if value.key is not None:
            values = step_io.read_values(target)
            if value.key not in values:
                raise ProtocolError(
                    "P2",
                    f"{what}: no key {value.key!r} in {target.name} "
                    f"(has {sorted(values)})",
                )
            resolved[slot] = values[value.key]
            continue
        if value.entry is not None or value.slot is not None:
            tensor, identity = step_io.read_tensor_with_identity(
                target, slot=value.slot, entry=value.entry, what=what
            )
            resolved[slot] = tensor
            identities.append(identity)
            continue
        resolved[slot] = target
        if target.suffix == ".safetensors":
            # an unselected bundle is handed over as a path, but its identity
            # still flows: a fit over one harvest is bound to that harvest
            try:
                _, identity = step_io.read_tensor_with_identity(target, what=what)
                identities.append(identity)
            except ProtocolError:
                pass  # multi-slot bundle: nothing unambiguous to inherit
    return resolved, identities


def _verify_outputs(
    name: str, step: Any, attempt_dir: Path, files: Sequence[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Every file the step's record lists, verified for its format and
    content-digested (§8): ``({file: sha256}, {file: check})``.

    For a script step the files are its declared outputs, each checked against
    its declaration; a protocol step's files are what its engine reported
    having written, checked by format alone."""
    declared = (
        {decl.file: decl for decl in step.outputs.values()}
        if isinstance(step, ScriptStep)
        else {}
    )
    digests: dict[str, str] = {}
    checks: dict[str, str] = {}
    for rel in sorted(files):
        target = attempt_dir / rel
        what = f"step {name!r}: output {rel!r}"
        checks[rel] = _verify_output(target, declared.get(rel), what)
        digests[rel] = _sha256(target)
    return digests, checks


def _verify_output(target: Path, decl: OutputDecl | None, what: str) -> str:
    """One output, checked against its format before it may be published
    (§2.5, §8); returns the name of the check that passed.

    Existence is where v1's load-time column check moved to, and it is later
    than a load error — but it is against the real file rather than a
    declaration believed on faith, and the declaration is what a consuming
    step was validated against. Every format at least exists and is non-empty;
    the record formats parse; the two figure formats with a signature carry
    it. What cannot be verified structurally (``.html``) is recorded as such,
    so the check's weakness is visible in the step record rather than implied."""
    if not target.is_file():
        raise ProtocolError(
            "P2", f"{what} was not written — every declared output must exist"
        )
    if target.stat().st_size == 0:
        raise ProtocolError("P2", f"{what} is empty (0 bytes)")
    suffix = target.suffix
    if suffix == TABLE_SUFFIX:
        return _verify_json_output(target, decl, what)
    if suffix == ".safetensors":
        _verify_safetensors(target, what)
        return "safetensors-header"
    if suffix in _SIGNATURES:
        with target.open("rb") as handle:
            head = handle.read(len(_SIGNATURES[suffix]))
        if head != _SIGNATURES[suffix]:
            raise ProtocolError(
                "P2", f"{what} does not start with the {suffix} signature"
            )
        return f"{suffix[1:]}-signature"
    return "non-empty"  # .html and anything else: no structure to check


#: Magic bytes of the figure formats that have them (§2.5).
_SIGNATURES: dict[str, bytes] = {".png": b"\x89PNG\r\n\x1a\n", ".pdf": b"%PDF"}


def _verify_json_output(target: Path, decl: OutputDecl | None, what: str) -> str:
    """A JSON output parses, and matches its declared shape when it has one
    (§2.3): declared ``keys`` against the values object written, declared
    ``columns`` against the rows written."""
    try:
        with target.open() as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ProtocolError("P2", f"{what} is not valid JSON: {err}") from err
    if decl is None or (decl.keys is None and decl.columns is None):
        return "json"
    if decl.keys is not None:
        if not isinstance(payload, dict):
            raise ProtocolError(
                "P2",
                f"{what} declares keys, so it must be a values object (a JSON "
                "mapping of name to value), not a "
                f"{type(payload).__name__}",
            )
        missing = sorted(set(decl.keys) - set(payload))
        if missing:
            raise ProtocolError(
                "P2",
                f"{what}: declares keys {sorted(decl.keys)} but wrote "
                f"{sorted(payload)} — missing {missing}",
            )
        return "json-values"
    rows = read_table(target)  # a JSON array of row objects, or a P2
    if not rows:
        return "json-table"  # an empty table satisfies any column declaration
    present = set(rows[0])
    missing = sorted(set(decl.columns or {}) - present)
    if missing:
        raise ProtocolError(
            "P2",
            f"{what}: declares columns {sorted(decl.columns or {})} but wrote "
            f"{sorted(present)} — missing {missing}",
        )
    return "json-table"


def _verify_safetensors(target: Path, what: str) -> None:
    """The bundle's header parses and its data section is exactly as long as
    the header says — a truncated or padded file fails here, with no tensor
    library involved (format: https://github.com/huggingface/safetensors)."""
    size = target.stat().st_size
    with target.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ProtocolError("P2", f"{what} is not a safetensors file (no header)")
        (header_len,) = struct.unpack("<Q", prefix)
        raw = handle.read(header_len)
    if len(raw) != header_len:
        raise ProtocolError("P2", f"{what} is truncated inside its safetensors header")
    try:
        header = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ProtocolError(
            "P2", f"{what} has an unreadable safetensors header: {err}"
        ) from err
    if not isinstance(header, dict):
        raise ProtocolError("P2", f"{what} has a malformed safetensors header")
    data_end = 0
    for key, spec in header.items():
        if key == "__metadata__" or not isinstance(spec, dict):
            continue
        offsets = spec.get("data_offsets")
        if isinstance(offsets, list) and len(offsets) == 2:
            data_end = max(data_end, int(offsets[1]))
    if size != 8 + header_len + data_end:
        raise ProtocolError(
            "P2",
            f"{what}: safetensors header promises {8 + header_len + data_end} "
            f"bytes, the file has {size}",
        )


def _sha256(target: Path) -> str:
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_in_process(
    name: str,
    step: ScriptStep,
    loaded: LoadedWorkflow,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> None:
    """Import the script and call ``main`` (§4).

    The import happens *here*, not at load: ``validate``/``digest`` must not
    pull a script's dependencies in (§4.2). By the time we are running, the
    process is already committed to executing."""
    import importlib.util

    target = resolve_script(step, loaded.workflow_dir, f"steps.{name}.script")
    spec = importlib.util.spec_from_file_location(f"_causalab_step_{name}", target)
    if spec is None or spec.loader is None:
        raise ProtocolError("P2", f"step {name!r}: cannot import {str(target)!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise ProtocolError(
            "P2", f"step {name!r}: {step.script!r} has no callable 'main'"
        )
    main(inputs, dict(outputs))


def _run_isolated(
    name: str,
    step: ScriptStep,
    loaded: LoadedWorkflow,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> None:
    """Run the script in a subprocess with its own dependency set (§4.1).

    Tensor-valued inputs cannot cross the process boundary, so an isolated step
    takes its tensors as *paths* — which means it must not use an ``entry``
    selector. Refused here rather than silently pickling a tensor into JSON."""
    runtime = dict(step.runtime or {})
    payload: dict[str, Any] = {}
    for slot, value in inputs.items():
        if isinstance(value, Path):
            payload[slot] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
            payload[slot] = value
        else:
            raise ProtocolError(
                "P2",
                f"step {name!r}: input {slot!r} is a {type(value).__name__}, "
                "which cannot cross a process boundary — an isolated step takes "
                "tensors as paths, so drop the 'entry'/'slot' selector and read "
                "the bundle inside the script",
            )
    target = resolve_script(step, loaded.workflow_dir, f"steps.{name}.script")
    request = {
        "script": str(target),
        "inputs": payload,
        "outputs": {slot: str(path) for slot, path in outputs.items()},
    }
    # The runner's own environment with `deps` layered on top: `--python` names
    # the interpreter this process runs under, `--no-project` keeps uv from
    # re-syncing whatever project the working directory happens to be in, and
    # `--with` builds an ephemeral overlay whose site-packages precede the
    # interpreter's — the declared deps win, the runner's environment is not
    # modified, and `causalab` is imported from the same bytes that are running
    # here. That holds for an editable checkout and an installed wheel alike;
    # the former `uv run` from the package's parent directory needed a
    # pyproject there, which only a checkout has.
    command = ["uv", "run", "--no-project", "--python", _python()]
    for dep in runtime.get("deps", ()):
        command += ["--with", str(dep)]
    command += ["python", "-m", "causalab.workflow.isolate"]

    environ = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "VIRTUAL_ENV")
        if key in os.environ
    }
    for passthrough in runtime.get("env", ()):
        if str(passthrough) in os.environ:
            environ[str(passthrough)] = os.environ[str(passthrough)]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            env=environ,
        )
    except FileNotFoundError as err:
        raise ProtocolError(
            "P2",
            f"step {name!r}: isolation needs 'uv' on PATH ({err})",
        ) from err
    if completed.returncode != 0:
        raise ScriptFailure(
            f"step {name!r}: isolated script failed (exit "
            f"{completed.returncode})\n{completed.stderr.strip()[-STDERR_TAIL_BYTES:]}",
            stderr=completed.stderr,
        )


def _python() -> str:
    return sys.executable
