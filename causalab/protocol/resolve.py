"""Resolution: the services a document is loaded *against*.

An intervention specification references things outside itself — datasets by ref,
prior runs' artifacts, a model's static config. The spec makes resolving
them part of loading (§1 artifact-valued fields, §2.2 dataset digests,
§5.15), but *what* they resolve against is an environment, not a global:
tests resolve against fixture files, production against task-generated
datasets and real run directories. :class:`ResolutionEnv` bundles the three
services; everything here is stdlib-only.

* **Artifacts** — ``{"artifact": "<ref>", "key": "<field>"}`` reads one
  value from a prior run at load. A ref names a JSON value table:
  ``<root>/<ref>.json`` or ``<root>/<ref>/values.json`` (first hit wins).
  Missing artifact or key = load error, never a default (§5.15).
* **Datasets** — a ref is a local path (relative to the data root) holding
  a serialized table; the resolver reports its content digest (stamped into
  the canonical form, §2.2), its columns (checked by ``validate --data``)
  and its rows (what a run consumes). The repo's task datasets are
  *generated*, but they are generated **ahead of the load**, by
  :mod:`causalab.tasks.serialize`, and enter here as ordinary tables — so
  resolution stays stdlib-only and a document's digest never depends on
  importing task code or a tokenizer.
* **Models** — static config metadata via
  :mod:`causalab.protocol.registry`.

Featurizer ``file_path`` artifacts get their existence checked here and
their ``ArtifactIdentity`` (§8) read from the safetensors header — the
header is a JSON prefix, so no tensor library is involved at load.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from causalab.protocol.errors import ValidationError
from causalab.protocol.registry import ModelInfo, get_model_info
from causalab.tables import SPLIT_COLUMN, table_bytes

__all__ = [
    "ArtifactStore",
    "DatasetResolver",
    "FileArtifacts",
    "FileDatasets",
    "ResolutionEnv",
    "endpoints",
    "entry_identity",
    "entry_table",
    "read_safetensors_metadata",
    "resolve_artifact_fields",
    "split_dataset_ref",
]


class ArtifactStore(Protocol):
    """Where prior runs' outputs are found."""

    def read_value(self, artifact: str, key: str) -> Any: ...

    def file_digest(self, file_path: str) -> str: ...

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None: ...


class DatasetResolver(Protocol):
    """Where dataset refs resolve. ``digest`` is the content digest stamped
    into canonical forms; ``columns`` backs ``validate --data``; ``rows`` is
    the table content a run consumes.

    All three are one contract on purpose. The pure verbs
    (``validate``/``explain``/``digest``) only need the first two, but a
    resolver that cannot produce rows cannot back a ``run`` — so the
    requirement is declared here instead of being discovered by a ``getattr``
    probe deep inside an engine."""

    def digest(self, ref: str) -> str: ...

    def columns(self, ref: str) -> tuple[str, ...]: ...

    def rows(self, ref: str) -> list[dict[str, Any]]: ...


@dataclasses.dataclass(frozen=True)
class ResolutionEnv:
    """The three resolution services a load runs against."""

    datasets: DatasetResolver
    artifacts: ArtifactStore
    model_info: Callable[[str], ModelInfo] = get_model_info


# --------------------------------------------------------------------------- #
# artifact-valued fields (§1, §5.15)
# --------------------------------------------------------------------------- #


def resolve_artifact_fields(
    raw: Any,
    env: ResolutionEnv,
    *,
    _path: str = "",
    _seen: frozenset[tuple[str, str]] = frozenset(),
) -> Any:
    """Replace every ``{"artifact": …, "key": …}`` node in a raw tree with
    the value it reads — recursively, so an artifact may itself store a
    reference (a cycle is a load error, not a hang). Runs before the parse
    gate, so a ref is legal anywhere a value is (§1); a mapping that
    *looks* like a ref but is malformed refuses rather than loading as a
    literal dict."""
    if isinstance(raw, Mapping):
        if isinstance(raw.get("artifact"), str):
            if set(raw) != {"artifact", "key"} or not isinstance(raw.get("key"), str):
                raise ValidationError(
                    15,
                    f"malformed artifact reference {dict(raw)!r} — the shape is "
                    '{"artifact": "<ref>", "key": "<field>"} exactly (§1)',
                    path=_path,
                )
            pair = (str(raw["artifact"]), str(raw["key"]))
            if pair in _seen:
                raise ValidationError(
                    15, f"artifact reference cycle through {pair!r}", path=_path
                )
            try:
                value = env.artifacts.read_value(*pair)
            except (FileNotFoundError, KeyError) as err:
                raise ValidationError(
                    15,
                    f"artifact-valued field did not resolve: {err} — a missing "
                    "artifact is a load error, never a default (§1)",
                    path=_path,
                ) from err
            return resolve_artifact_fields(
                value, env, _path=_path, _seen=_seen | {pair}
            )
        return {
            key: resolve_artifact_fields(
                value, env, _path=f"{_path}.{key}" if _path else key, _seen=_seen
            )
            for key, value in raw.items()
        }
    if isinstance(raw, list):
        return [
            resolve_artifact_fields(item, env, _path=_path, _seen=_seen) for item in raw
        ]
    return raw


# --------------------------------------------------------------------------- #
# file-backed implementations
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FileArtifacts:
    """Artifacts under one root directory (a run-output tree)."""

    root: Path

    def _value_table(self, artifact: str) -> Mapping[str, Any]:
        for candidate in (
            self.root / f"{artifact}.json",
            self.root / artifact / "values.json",
        ):
            if candidate.is_file():
                table = json.loads(candidate.read_text())
                if not isinstance(table, dict):
                    raise ValidationError(
                        15, f"artifact {artifact!r} is not a JSON object of values"
                    )
                return table
        raise FileNotFoundError(f"no artifact {artifact!r} under {self.root}")

    def read_value(self, artifact: str, key: str) -> Any:
        table = self._value_table(artifact)
        if key not in table:
            raise KeyError(
                f"artifact {artifact!r} has no key {key!r} (has {sorted(table)})"
            )
        return table[key]

    def file_digest(self, file_path: str) -> str:
        target = self.root / file_path
        if not target.is_file():
            raise ValidationError(
                15, f"artifact file {file_path!r} not found under {self.root} (§5.15)"
            )
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None:
        target = self.root / file_path
        if not target.is_file():
            raise ValidationError(
                15, f"artifact file {file_path!r} not found under {self.root} (§5.15)"
            )
        return read_safetensors_metadata(target)

    def resolve_path(self, file_path: str) -> Path:
        return self.root / file_path


def read_safetensors_metadata(path: Path) -> Mapping[str, Any] | None:
    """The ``__metadata__`` table of a safetensors file — a pure header read
    (8-byte little-endian header length, then a JSON object), no tensor
    library involved. Returns ``None`` when the file carries no metadata.

    Format reference: https://github.com/huggingface/safetensors#format.
    """
    with path.open("rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ValidationError(
                15, f"{path} is not a safetensors file (truncated header)"
            )
        (header_len,) = struct.unpack("<Q", prefix)
        header = json.loads(fh.read(header_len))
    meta = header.get("__metadata__")
    return meta if isinstance(meta, Mapping) else None


# --------------------------------------------------------------------------- #
# ArtifactIdentity (§8)
# --------------------------------------------------------------------------- #

#: The stamped-identity schema for featurizer bundles: these keys live in the
#: safetensors ``__metadata__`` table (string-valued, per the format). The
#: engine stamps them at save; the loader refuses a ``file_path`` load whose
#: stamped values contradict the document (§2.5).
#:
#: ⚠️ **Migration.** ``model_dtype`` and ``model_quantization`` joined this
#: schema when precision entered the record (§2.1). A bundle fitted before
#: that carries neither, so it no longer matches a document that names them
#: and ``_check_loaded_featurizers`` refuses it. That is intended and not a
#: bug to route around — a rotation fitted in bf16 is not the same artifact as
#: one fitted in fp32, and pretending otherwise is what the stamp exists to
#: prevent. Locally kept fitted artifacts must be re-fitted once; nothing in
#: the repo ships one.
ARTIFACT_IDENTITY_KEYS: tuple[str, ...] = (
    "produced_by",
    "model_key",
    "model_revision",
    "model_dtype",
    "model_quantization",
    "model_attn_implementation",
    "tokenizer",
    "site",
    "k",
    "parametrization",
    # a gate's grouping (§2.5): the unit kind and its derived
    # ``[groups, group_width]`` map, so a mask fitted over 16 heads of 256 is
    # refused against a site whose heads are laid out any other way
    "group",
    "group_map",
    # a hard-concrete gate's stretch ``[γ, ζ]`` (§2.5 ``parametrization``):
    # the hard split is ``θ > logit((½−γ)/(ζ−γ))``, so a reader of the bundle
    # (``analysis.random_mask``) needs it to match the fit's own threshold, and
    # the loader compares it — an authored stretch is expected of the bundle,
    # a non-default stamp refused by a document authoring none (``loader.py``).
    # Registering a key here also admits it per entry (``entry_identity``) and
    # carries it forward on a re-stamp (``step_io.inherited_identity``)
    "stretch",
    # a budget gate's pool (§2.5 ``pool``): a pooled member's θ is a ranking
    # only relative to its co-members, so the pool's name is compared by the
    # loader in both directions (like ``group``) and ``pool_units`` — the
    # pool's unit count — rides along as provenance a reader sizes a cut by
    "pool",
    "pool_units",
    # a position gate's ``axis`` (§2.5): θ is one entry per token position,
    # not per coordinate. Compared both ways at compile (``loader.py``, the
    # ``group`` shape) and per entry at the build
    # (``featurizers._check_entry_identity``); registering it here also
    # carries it forward on a re-stamp (``step_io.inherited_identity``), so a
    # script step's output inherits the axis its inputs were fitted over
    "axis",
    # a straight-through fit's ``forward`` (§2.5, the mapping form of
    # ``parametrization``): provenance only — the loader's expectation never
    # carries it and ``check_artifact_identity`` compares only the keys the
    # expectation has, so it gates no reload; registered so the stamp path
    # (``build_artifact_identity``, which refuses unknown keys) admits it.
    # Unlike ``axis`` above, deliberately no load-time clause: ``axis``
    # changes what θ's entries *index* (W positions or W coordinates),
    # ``forward`` only which loss produced them — the readout is the same
    # map's hard split either way
    "forward",
    "dtype",
    "trained_on",
    "trained_on_digest",
    "engine",
    # Which optional engine implementations a run actually applied — e.g.
    # `attn_eager` when the nnterp engine forces eager attention to reach the
    # pattern interior. Runtime provenance of the same kind as `engine`,
    # stamped by `neural/shared/execution.py` on every tensor file such a run
    # writes.
    "implementations",
    "loaded_attn_implementation",  # observed backend, inherited as runtime provenance
    "commit",
    # a ``subspace`` fit initialised from a saved basis (§2.5 ``init``): the
    # basis's own provenance, which columns of it seeded the fit, and a digest
    # of the seeding matrix — so the record says where the fit *started*, not
    # only where it ended
    "init_produced_by",
    "init_trained_on",
    "init_components",
    "init_digest",
    # the digest of the run's location ledger (§6, `protocol/ledger.py`) —
    # stamped only when the run emitted one, and recorded, never compared:
    # it names the ledger table the fit was made under, so a reader can find
    # the tokens the parameter was trained on. A loading run that saves its
    # own ledger records the tokens it selected on its rows; no document
    # implies this key, so `check_artifact_identity` never asks for it
    "location_ledger_sha256",
)


def build_artifact_identity(**fields: Any) -> dict[str, str]:
    """Stringify identity fields for a safetensors ``__metadata__`` table
    (the format only carries ``str -> str``). Unknown keys are refused so
    the schema stays closed; absent fields are simply not stamped."""
    unknown = set(fields) - set(ARTIFACT_IDENTITY_KEYS)
    if unknown:
        raise AssertionError(f"unknown ArtifactIdentity fields {sorted(unknown)}")
    return {
        key: value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        for key, value in fields.items()
        if value is not None
    }


#: Extra guidance for the mismatches whose *cause* is not where the reader
#: looks first, as ``str.format`` templates over ``want`` (what the document
#: implies) and ``got`` (what the bundle was stamped with).
#:
#: ``model_dtype`` is the one that has cost real time twice. A ``model`` block
#: with no ``dtype`` **implies fp32**, so an apply document that simply omits
#: the field is refused against a bf16 fit — and the fix is in the document,
#: not on the command line: ``--dtype`` exists only on a *document* run
#: (it is a ``--set model.dtype=…`` shorthand), and a workflow run has no such
#: flag at all, so a chained fit → apply can only be repaired in the file.
_MISMATCH_HINTS: dict[str, str] = {
    "model_attn_implementation": (
        '. Write "attn_implementation": "{got}" inside the document\'s "model" '
        'object, or set "model.attn_implementation" in the workflow step\'s "set" object'
    ),
    "model_dtype": (
        '. Precision is a document fact: write "dtype": "{got}" into this '
        "document's 'model' block, next to the fit's. A 'model' with no "
        "'dtype' implies 'fp32', which is what an apply document usually gets "
        "wrong. The --dtype flag is not the fix: it is a --set shorthand on a "
        "document run, and a workflow run does not accept it at all"
    ),
}


def check_artifact_identity(
    stamped: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    *,
    what: str,
) -> None:
    """Refuse a loaded bundle whose stamped identity contradicts the
    document (§2.5). A bundle with no identity at all is refused too — an
    unverifiable artifact is a provenance hole, not a pass."""
    if stamped is None:
        raise ValidationError(
            15,
            f"{what}: the artifact carries no ArtifactIdentity metadata — "
            "nothing to check, so the load refuses (§2.5)",
        )
    normalized_expected = build_artifact_identity(**expected)
    for key, want in normalized_expected.items():
        got = stamped.get(key)
        if got is not None and str(got) != want:
            raise ValidationError(
                15,
                f"{what}: ArtifactIdentity mismatch on {key!r} — the document "
                f"implies {want!r} but the bundle was stamped {got!r} (§2.5)"
                + _MISMATCH_HINTS.get(key, "").format(want=want, got=got),
            )
        if got is None:
            raise ValidationError(
                15,
                f"{what}: ArtifactIdentity is missing {key!r} — the bundle "
                "cannot prove it matches the document (§2.5)",
            )


def split_dataset_ref(ref: str) -> tuple[str, str | None]:
    """``"weekdays#train"`` → ``("weekdays", "train")``; a bare ref → ``(ref, None)``.

    The fragment names one split *inside* a table (§2.2): a dataset is one
    table, every row declares its split in the :data:`~causalab.tables.
    SPLIT_COLUMN` column, and a document selects one by fragment rather than by
    pointing at a second file. Borrowed from URL fragment syntax, which means
    exactly this — a named part of one resource.

    Split on the **last** ``#`` so a data root containing a ``#`` in a directory
    name still resolves. An empty fragment is refused rather than silently
    meaning "the whole table": ``weekdays#`` is a typo, not a selection.
    """
    base, sep, fragment = ref.rpartition("#")
    if not sep:
        return ref, None
    if not fragment:
        raise ValidationError(
            22,
            f"dataset ref {ref!r} ends in an empty '#' fragment — name a split "
            f"({base!r}#train) or drop the '#' to take the whole table",
            path="data",
        )
    return base, fragment


def _declared_splits(base: str, rows: list[dict[str, Any]]) -> set[str]:
    """The split values a table declares — refusing one that declares none [V22].

    Every table declares, and the requirement is the point rather than a
    formality: a split that is optional is a split that can be omitted, and an
    omitted one is exactly the state the column exists to abolish. A single
    undivided pool says so with one uniform value, which costs a word and buys
    the guarantee that no table is silently unlabelled.
    """
    missing = [i for i, row in enumerate(rows) if SPLIT_COLUMN not in row]
    if missing:
        where = (
            "no row declares one"
            if len(missing) == len(rows)
            else f"{len(missing)} of {len(rows)} rows do not (first: row {missing[0]})"
        )
        raise ValidationError(
            22,
            f"dataset {base!r} has no {SPLIT_COLUMN!r} column — {where}. Every "
            f"table declares which split its rows are (§2.2); rebuild it with "
            f"scripts/build_task_dataset.py --split all for one undivided pool, "
            f"or as a partitioned table whose every row names its split",
        )
    return {str(row[SPLIT_COLUMN]) for row in rows}


def endpoints(row: Mapping[str, Any]) -> set[str]:
    """The prompts a row puts in front of the model, both sides of the pair."""
    out: set[str] = set()
    value = row.get("input")
    if isinstance(value, str):
        out.add(value)
    counterfactuals = row.get("counterfactual_inputs")
    if isinstance(counterfactuals, list):
        out.update(str(v) for v in counterfactuals if isinstance(v, str))
    return out


def _check_splits_are_disjoint(base: str, rows: list[dict[str, Any]]) -> None:
    """Splits of one table share no prompt, at either endpoint [V22].

    Row-level disjointness is free — a row declares one split — but that is the
    weaker half. The leak that matters is a *prompt* appearing as a training
    base and again as a test counterfactual, which reports a training score
    under a held-out name. Because both splits live in one table, that is now a
    question about the bytes in front of us, so it is answered here rather than
    asserted by whoever ran the builder.

    Checked for every table with more than one split, at the one place a ref
    becomes rows, so no verb can skip it. A deliberate train-equals-test
    ablation is spelled by naming *one* split twice in the document, where it is
    visible, instead of by two splits that quietly coincide.
    """
    declared = {str(row[SPLIT_COLUMN]) for row in rows if SPLIT_COLUMN in row}
    if len(declared) < 2:
        return
    seen: dict[str, str] = {}
    for row in rows:
        split = str(row[SPLIT_COLUMN])
        for endpoint in endpoints(row):
            other = seen.setdefault(endpoint, split)
            if other != split:
                raise ValidationError(
                    22,
                    f"dataset {base!r} leaks across splits: the prompt "
                    f"{endpoint!r} appears in both {other!r} and {split!r}. "
                    f"Splits of one table must be endpoint-disjoint (§2.2) — "
                    f"rebuild the table so its inputs are partitioned into "
                    f"groups before they are paired",
                )


def entry_table(metadata: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The per-entry provenance table a producer writes into a bundle's header
    (``outputs.TensorFile``): key → ``{"slot", "coords", …identity}``. Empty
    for a bundle written without one, in which case the keys themselves
    (:func:`causalab.protocol.bundles.parse_entry_key`) are all there is."""
    if not metadata:
        return {}
    raw = metadata.get("entries")
    if not isinstance(raw, str):
        return {}
    try:
        table = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return table if isinstance(table, dict) else {}


def entry_identity(metadata: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    """The identity of one bundle entry (§8): the file-level stamp, overridden
    by whatever the ``entries`` table records for that key — the per-entry
    fields a swept producer could not stamp file-wide. One rule for every
    reader of a header: a script inheriting provenance and an engine loading
    a featurizer see the same identity for the same entry."""
    if not metadata:
        return {}
    identity = {
        field: value
        for field, value in metadata.items()
        if field in ARTIFACT_IDENTITY_KEYS
    }
    entry = entry_table(metadata).get(key, {})
    identity.update(
        {
            field: value
            for field, value in entry.items()
            if field in ARTIFACT_IDENTITY_KEYS
        }
    )
    return identity


@dataclasses.dataclass(frozen=True)
class FileDatasets:
    """Dataset refs as serialized JSON tables under one data root.

    A ref ``weekdays/data`` resolves to ``<root>/weekdays/data.json`` — a JSON
    array of row objects. Task-generated tables are written by
    :mod:`causalab.tasks.serialize` into this same layout (deterministically,
    so the digest is reproducible), and nothing beside them: a table is
    exactly the bytes a document names (§2.2). A workflow that names the
    table pins its content digest in its own ``pins`` section (workflow spec
    §7), never in a file beside the table.

    **A ref may name one split of that table**: ``weekdays/data#train`` keeps
    only the rows whose ``split`` column is ``"train"`` (§2.2,
    :func:`split_dataset_ref`). Selection happens here, at the one place a ref
    becomes rows, so nothing downstream — schema, canonical form, engines —
    needs to know a split exists.

    The content digest is the sha256 of the **selected rows'** canonical bytes,
    not of the file. That is what makes the fragment safe: two splits of one
    table carry two distinct digests, so run identity is preserved; and a run's
    digest depends only on the rows it consumed, so adding a split to a table
    does not invalidate a run over the splits already in it. For a whole-table
    ref the two agree by construction, because
    :func:`~causalab.tasks.serialize.write_dataset_table` writes exactly
    :func:`~causalab.tables.table_bytes`.
    """

    root: Path
    #: Searched in order after ``root`` misses. The CLI puts the shipped task
    #: tables (``causalab/tasks/``) here, so a document can mix a private table
    #: under ``--data-root`` with ``<task>/data/<variant>`` — and a private
    #: table of the same ref shadows the shipped one, visibly, root first.
    fallback_roots: tuple[Path, ...] = ()

    @property
    def roots(self) -> tuple[Path, ...]:
        seen: list[Path] = []
        for root in (self.root, *self.fallback_roots):
            if root not in seen:
                seen.append(root)
        return tuple(seen)

    def _file(self, ref: str) -> Path:
        for root in self.roots:
            for candidate in (root / f"{ref}.json", root / ref):
                if candidate.is_file():
                    return candidate
        where = " nor ".join(str(root) for root in self.roots)
        raise ValidationError(
            4, f"dataset {ref!r} not found under {where}", path="data"
        )

    def digest(self, ref: str) -> str:
        return hashlib.sha256(table_bytes(self.rows(ref))).hexdigest()

    def table_digest(self, ref: str) -> str:
        """The sha256 of the **whole** table's canonical bytes, every split
        included (fragment ignored: one table, one digest) — what a workflow's
        ``pins.datasets`` records for the table a document reads, and what
        :func:`~causalab.tasks.serialize.write_dataset_table` returns for the
        bytes it wrote. :meth:`digest` is the *selected rows'* digest, which
        for a split ref is a different quantity — and refuses a bare ref over
        a partitioned table, which a pin over the table must not."""
        base, _fragment = split_dataset_ref(ref)
        return hashlib.sha256(table_bytes(self._table(base))).hexdigest()

    def columns(self, ref: str) -> tuple[str, ...]:
        rows = self.rows(ref)
        cols: set[str] = set()
        for row in rows:
            cols.update(row)
        return tuple(sorted(cols))

    def rows(self, ref: str) -> list[dict[str, Any]]:
        base, fragment = split_dataset_ref(ref)
        rows = self._table(base)
        declared = _declared_splits(base, rows)
        if fragment is None:
            if len(declared) > 1:
                raise ValidationError(
                    22,
                    f"dataset {base!r} carries {len(declared)} splits "
                    f"({sorted(declared)}) and the ref names none of them — a "
                    f"bare ref would consume every split at once. Select one: "
                    f"{base}#{sorted(declared)[0]}",
                )
            return rows
        selected = [row for row in rows if str(row.get(SPLIT_COLUMN)) == fragment]
        if not selected:
            raise ValidationError(22, self._no_such_split(base, fragment, rows))
        return selected

    def _table(self, base: str) -> list[dict[str, Any]]:
        rows = json.loads(self._file(base).read_text())
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise ValidationError(
                4, f"dataset {base!r} is not a JSON array of row objects"
            )
        _check_splits_are_disjoint(base, rows)
        return rows

    @staticmethod
    def _no_such_split(base: str, fragment: str, rows: list[dict[str, Any]]) -> str:
        """Why a fragment selected nothing — a missing column and a mistyped
        value are different mistakes and get different messages."""
        if not any(SPLIT_COLUMN in row for row in rows):
            return (
                f"dataset {base!r} declares no {SPLIT_COLUMN!r} column, so it has "
                f"no split {fragment!r} to select — rebuild the table with a "
                f"{SPLIT_COLUMN!r} column, or drop the '#{fragment}' fragment"
            )
        available = sorted({str(row.get(SPLIT_COLUMN)) for row in rows})
        return (
            f"dataset {base!r} has no rows in split {fragment!r} "
            f"(it declares {available})"
        )
