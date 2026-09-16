"""Engine-neutral §8 services: tensor-bundle loading, input-role resolution,
and site identity records.

These were the reference engine's private helpers, moved here because a
second engine needs them verbatim: nothing in them touches a
hook, a trace, or a loaded model — they read the document and the resolution
environment.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import random
from pathlib import Path
from typing import Any, Mapping

import torch

from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.errors import ProtocolError
from causalab.protocol.examples import example_id_defect
from causalab.protocol.schema import DataRole, Document, SiteSpec

__all__ = [
    "BundlePoint",
    "TensorBundle",
    "check_caller_bundle",
    "input_roles",
    "load_table",
    "load_tensors",
    "resolve_roles",
    "shuffle_order",
    "site_identity",
]


@dataclasses.dataclass(frozen=True)
class BundlePoint:
    """One producing point's slice of a bundle: the tensors sharing a
    coordinate suffix, plus that entry's stamped record.

    Slicing by suffix rather than selecting each slot on its own is what
    keeps a multi-slot bundle coherent — an SAE's ``enc`` and ``dec`` must
    come from the same fit, not from whichever entries each lookup found.
    """

    tensors: dict[str, torch.Tensor]
    suffix: str
    record: dict[str, Any]
    what: str
    #: the entry's ArtifactIdentity (§8): the file-level stamp, overridden by
    #: whatever the ``entries`` record says for this entry
    #: (:func:`causalab.protocol.resolve.entry_identity`)
    identity: dict[str, Any] = dataclasses.field(default_factory=dict)

    def tensor(self, slot: str) -> torch.Tensor:
        key = f"{slot}{self.suffix}"
        if key not in self.tensors:
            raise ProtocolError(
                "P2",
                f"{self.what}: the bundle has no {key!r} — an entry's slots "
                f"must be complete (has {sorted(self.tensors)})",
            )
        return self.tensors[key]


@dataclasses.dataclass(frozen=True)
class TensorBundle:
    """One loaded ``.safetensors`` file: its tensors plus the ``entries``
    table from the header (§8, :mod:`causalab.protocol.bundles`).

    :meth:`point` is the only way in. A bundle written by a swept document
    holds one entry per point per slot, so asking for a bare slot name would
    either ``KeyError`` or — worse — silently take whichever entry a plain
    dict lookup happened to find.
    """

    tensors: dict[str, torch.Tensor]
    entry_coords: dict[str, Any]
    #: the header's ``__metadata__`` table as written — the file-level
    #: ArtifactIdentity plus the serialized ``entries``; a hand-built bundle
    #: carries none
    header: dict[str, Any] = dataclasses.field(default_factory=dict)

    def point(
        self,
        slot: str,
        want: Any,
        *,
        what: str,
        implicit: bool = False,
    ) -> BundlePoint:
        """The entry for ``slot`` selected by ``want`` (a coordinate
        mapping; ``implicit`` when derived from the consuming point rather
        than authored), as a coherent slice of the bundle."""
        from causalab.protocol.bundles import select_entry
        from causalab.protocol.resolve import entry_identity

        key = select_entry(
            self.tensors.keys(),
            slot,
            want,
            what=what,
            coords_by_key=self.entry_coords or None,
            implicit=implicit,
        )
        record = self.entry_coords.get(key, {})
        record = record if isinstance(record, dict) else {}
        return BundlePoint(
            tensors=self.tensors,
            suffix=key[len(slot) :],
            record=record,
            what=what,
            identity=entry_identity(self.header, key),
        )


def check_caller_bundle(
    bundle: Any, realization: Mapping[str, Any], *, device: str
) -> None:
    """Refuse a caller-owned bundle that does not realize the document's model.

    An engine built with ``bundle=`` (spec §9, the ownership contract) runs
    that bundle instead of loading — so before any forward, the document's
    canonical ``model`` block (:func:`~causalab.protocol.canonical.canonical_model`:
    ``key``, ``revision``, ``dtype``, the materialized ``quantization``, and
    any explicit ``attn_implementation``) is
    compared field by field with what the bundle says it is, and the engine's
    ``device`` with the bundle's requested one. A disagreement refuses, naming
    both sides: the run receipt and every ``ArtifactIdentity`` stamp would
    otherwise describe a model that did not run. The comparison is by value,
    the way the loader's cache key is (a materialized ``quantization`` block
    is a mapping, order-free).
    """
    disagreements = [
        f"{what}: the document says {theirs!r}, the bundle {ours!r}"
        for what, theirs, ours in (
            ("model.key", str(realization["key"]), bundle.key),
            ("model.revision", str(realization["revision"]), bundle.revision),
            ("model.dtype", str(realization["dtype"]), bundle.dtype),
            (
                "model.quantization",
                realization.get("quantization"),
                bundle.quantization,
            ),
            ("device", device, bundle.device),
        )
        if theirs != ours
    ]
    if "attn_implementation" in realization:
        wanted = realization["attn_implementation"]
        actual = getattr(bundle.model.config, "_attn_implementation", None)
        if wanted != actual:
            disagreements.append(
                f"model.attn_implementation: the document says {wanted!r}, "
                f"the bundle {actual!r}"
            )
    if disagreements:
        raise ProtocolError(
            "P4",
            "the caller-owned bundle does not realize this document's model, "
            "so the run receipt and every artifact stamp would describe a "
            "model that did not run — " + "; ".join(disagreements) + ". Hand "
            "the engine a bundle built for this document (or edit the "
            "document to say what actually runs)",
        )


def site_identity(doc: Document, site_name: str | None) -> dict[str, Any] | None:
    """One site as the ArtifactIdentity records it — the non-null address
    fields only, the shape ``loader.py`` builds its expectation in."""
    if site_name is None or site_name not in doc.sites:
        return None
    return spec_identity(doc.sites[site_name])


def spec_identity(record: SiteSpec) -> dict[str, Any]:
    """:func:`site_identity` of a site record itself — for a site the
    executor addresses without the document naming it (the ``ln_final``
    capture of a projecting ``lm_head`` read, ``execution._tap_union``)."""
    # the band as a JSON list — the stamp is serialized, and the loader's
    # expectation (`loader._featurizer_expectation`) is spelled the same way
    layers = list(record.layers) if isinstance(record.layers, tuple) else record.layers
    return {
        key: value
        for key, value in {
            "component": record.component,
            "layers": layers,
            "head": record.head,
            "expert": record.expert,
            "stream": record.stream,
        }.items()
        if value is not None
    }


def input_roles(doc: Document) -> dict[str, DataRole]:
    """The document's input roles under the names a read's ``input`` uses:
    ``base``, ``counterfactual``, or ``counterfactual[j]`` for a list-valued
    role (§2.2). The one place the naming rule lives, so the rows an executor
    batches, the data identity a forward group is keyed on and the stamp a
    harvested read carries all name a role the same way."""
    roles: dict[str, DataRole] = {}
    for role, value in doc.data.items():
        if isinstance(value, tuple):
            roles.update({f"{role}[{j}]": spec for j, spec in enumerate(value)})
        else:
            roles[role] = value
    return roles


def resolve_roles(
    doc: Document, request: ExecutionRequest
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Dataset rows + field selector per input role, rows paired by index.

    A counterfactual role that authors ``shuffle: {seed}`` (§2.2) has its rows
    permuted by :func:`shuffle_order` before the pairing — the same rows, met
    by different base rows — so the ``shuffled_source`` control (workflow spec
    §2.2) is a document and not a serialized permuted table. The base role is
    never permuted (the parser refuses ``shuffle`` there), and the row-count
    check below runs on the permuted list, whose length is unchanged.

    ``rows`` is part of the :class:`~causalab.protocol.resolve.DatasetResolver`
    contract, so this reads it directly — a resolver without it is a typing
    error at construction, not a surprise at run time."""
    rows_of = request.env.datasets.rows
    role_rows: dict[str, list[dict[str, Any]]] = {}
    role_fields: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for role_name, role_spec in input_roles(doc).items():
        rows = rows_of(str(role_spec.dataset))
        # the run's own check of what `validate --data` refuses (§2.2): a
        # label column that cannot label its rows
        defect = example_id_defect(rows)
        if defect is not None:
            raise ProtocolError(
                "P2", f"data.{role_name} dataset {role_spec.dataset!r}: {defect}"
            )
        if role_spec.shuffle is not None:
            order = shuffle_order(int(role_spec.shuffle["seed"]), len(rows))
            rows = [rows[i] for i in order]
        role_rows[role_name] = rows
        # §2.2 `draw`: outside a fit's updates a drawn role reads its fixed
        # `eval` member (`resolved_field`); the fit redraws per epoch from
        # these same rows
        role_fields[role_name] = role_spec.resolved_field
        lengths[role_name] = len(role_rows[role_name])
    if len(set(lengths.values())) > 1:
        raise ProtocolError(
            "P2",
            f"input roles have unequal row counts {lengths} — rows are paired "
            "by index (§2.2)",
        )
    return role_rows, role_fields


def shuffle_order(seed: int, n: int) -> list[int]:
    """The permutation a ``shuffle: {seed}`` role applies (§2.2): the indices
    ``0..n-1`` shuffled by ``random.Random(seed).shuffle`` — stdlib only,
    torch-free, a pure function of ``(seed, n)``, so two runs of one document
    pair the same rows and two seeds give two pairings. The permuted role's
    row ``i`` is the authored row ``order[i]``."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


@functools.lru_cache(maxsize=32)
def _read_bundle(path: str, _stamp: tuple[int, int]) -> TensorBundle:
    """One bundle, read once. The cache matters: a write operand resolves
    its ``params`` tensor on every application, so an uncached read would
    re-open the same file for every batch of every point.

    ``_stamp`` is the file's (mtime, size), so a path rewritten in the same
    process — a step re-run into an existing run tree — is a cache miss
    rather than a stale tensor."""
    from causalab.io.tensor_files import load_file

    from causalab.protocol.resolve import read_safetensors_metadata

    meta = read_safetensors_metadata(Path(path)) or {}
    raw_entries = meta.get("entries")
    entry_coords: dict[str, Any] = {}
    if isinstance(raw_entries, str):
        try:
            decoded = json.loads(raw_entries)
        except json.JSONDecodeError as err:
            raise ProtocolError(
                "P2", f"{path}: unreadable 'entries' table in the header — {err}"
            ) from err
        if isinstance(decoded, dict):
            entry_coords = decoded
    return TensorBundle(
        tensors=load_file(path), entry_coords=entry_coords, header=dict(meta)
    )


def load_table(
    request: ExecutionRequest, file_path: str
) -> tuple[list[dict[str, Any]], bytes]:
    """A saved metric table referenced by a gate's ``init.from_scores``
    (§2.5), resolved through the artifact store exactly as :func:`load_tensors`
    resolves a bundle, as ``(rows, bytes)`` — the rows to read the start off,
    the bytes to stamp its digest with."""
    from causalab.protocol.tables import read_table

    artifacts = request.env.artifacts
    resolve = getattr(artifacts, "resolve_path", None)
    if resolve is not None:
        target = Path(resolve(file_path))
    else:
        root = getattr(artifacts, "root", None)
        if root is None:
            raise ProtocolError("P2", "artifact store exposes no filesystem root")
        target = Path(root) / file_path
    return read_table(target), target.read_bytes()


def load_tensors(request: ExecutionRequest, file_path: str) -> TensorBundle:
    """Load a tensor bundle referenced by a featurizer/params file_path,
    resolved through the artifact store (which owns the run-tree/external
    overlay inside a workflow)."""
    artifacts = request.env.artifacts
    resolve = getattr(artifacts, "resolve_path", None)
    if resolve is not None:
        target = Path(resolve(file_path))
    else:
        root = getattr(artifacts, "root", None)
        if root is None:
            raise ProtocolError("P2", "artifact store exposes no filesystem root")
        target = Path(root) / file_path
    stat = target.stat()
    return _read_bundle(str(target), (stat.st_mtime_ns, stat.st_size))
