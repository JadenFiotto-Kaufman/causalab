"""Run outputs: the save manifest written to disk, stamped (spec §2.12, §8).

Tensors (reads, featurizer bundles) land in ``.safetensors`` with the
point's ``ArtifactIdentity`` in the header; per-example metric tables land
in ``.json`` as an array of row objects (IM spec §2.12 — JSON and safetensors
are the only two formats). In swept documents the authored ``file_path`` is
unchanged: axis coordinates become columns of the metric tables and key
suffixes on tensor entries (``rot[k=8,seed=0]``).

**Per-entry provenance.** A swept document writes one file from many points,
so file-level identity can only carry what *every* point agrees on: the
fields that vary (``k``, the point digest, a swept site) live in an
``entries`` table in the safetensors ``__metadata__``, one record per tensor
key. That table is what makes an entry both selectable
(:mod:`causalab.protocol.bundles`) and provable — before it, whichever point
executed last silently stamped the whole file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.bundles import RAGGED_SUFFIX, entry_key
from causalab.protocol.errors import ProtocolError
from causalab.protocol.estimand import IDENTITY_COLUMNS
from causalab.protocol.examples import EXAMPLE_ID_COLUMN
from causalab.protocol.resolution import Unavailable
from causalab.protocol.resolve import build_artifact_identity
from causalab.protocol.sweep import coordinate_label, short_coords
from causalab.protocol.tables import write_table

__all__ = [
    "ELIGIBLE_COLUMN",
    "REASON_CODE_COLUMN",
    "MetricTable",
    "TensorFile",
    "write_outputs",
]

#: The eligibility record on every metric row (spec §2.10 "Eligibility"):
#: ``eligible`` is ``true`` on a row the metric's decision rule was evaluated
#: over and ``false`` on an excluded measurement, which alone also carries
#: ``reason_code`` — the :data:`~causalab.protocol.errors.ReasonCode` of the
#: ``unavailable`` the row became. Both derived, never authored (§6).
ELIGIBLE_COLUMN = "eligible"
REASON_CODE_COLUMN = "reason_code"


class TensorFile:
    """Accumulates tensor entries for one save file across points."""

    def __init__(self) -> None:
        self.entries: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, str] = {}
        #: key -> {"slot", "coords", …identity}; serialized as ``entries``
        self.entry_meta: dict[str, dict[str, Any]] = {}
        self._common_seen = False

    def add(
        self,
        name: str,
        value: Any,
        coords: Mapping[str, Any],
        *,
        label_entry: str | None = None,
        reduce: str | None = None,
        identity: Mapping[str, Any] | None = None,
        record: Mapping[str, Any] | None = None,
    ) -> None:
        """Add one point's value under ``name``.

        ``label_entry`` is the *declared* entity the coordinates belong to,
        which for a featurizer bundle is the featurizer, not the slot: the
        axis ``featurizers.rot.k`` shortens to ``k`` against ``rot`` and to
        ``rot.k`` against ``weight``, and only the former is a name a
        consuming document can write in an ``entry`` selector.

        ``identity`` fields are stamped as strings (the ArtifactIdentity
        contract); ``record`` fields ride on the entry **as JSON values** —
        what a ``trajectory`` checkpoint says about itself (its step, the
        controlled weight, the kept count), which a reader wants as numbers."""
        entity = label_entry or name
        key = entry_key(name, coordinate_label(coords, entry=entity) if coords else "")
        self.entry_meta[key] = {
            "slot": name,
            "coords": {
                short: _plain(coord)
                for short, coord in short_coords(coords, entry=entity).items()
            },
            **{k: str(v) for k, v in (identity or {}).items()},
            **{k: _plain(v) for k, v in (record or {}).items()},
        }
        from causalab.neural.shared.executor_base import RaggedValue

        if reduce is not None:
            self.entries[key] = _reduce_rows(value, reduce)
            return
        if isinstance(value, RaggedValue):
            # ragged reads persist as the flat gather + per-row widths
            self.entries[key] = value.flat.detach().to("cpu").contiguous()
            self.entries[f"{key}{RAGGED_SUFFIX}"] = torch.tensor(
                value.widths, dtype=torch.long
            )
            return
        self.entries[key] = value.detach().to("cpu").contiguous()

    def record_common(self, identity: Mapping[str, Any]) -> None:
        """Fold one point's identity into the file-level stamp, keeping only
        the fields every point so far agrees on.

        A single-point document therefore stamps exactly what it always did;
        a swept one drops the fields that differ (``k``, the point digest)
        rather than letting the last point speak for the file. The dropped
        fields are still provable per entry via the ``entries`` table."""
        stamped = {key: str(value) for key, value in identity.items()}
        if not self._common_seen:
            self.metadata.update(stamped)
            self._common_seen = True
            return
        for key in list(self.metadata):
            if self.metadata[key] != stamped.get(key):
                del self.metadata[key]


def _reduce_rows(value: Any, reduce: str) -> torch.Tensor:
    """§2.12 ``reduce``: a statistic over a read's gathered rows instead of
    the rows themselves — ``(…, width)`` collapses to ``(width,)``, the
    broadcast form a write operand takes.

    One branch per verb in :data:`~causalab.protocol.schema.SAVE_REDUCTIONS`,
    and the vocabulary is closed: a new verb is a PR that adds a branch here,
    a §2.12 row, and a test. The docs↔code guard
    (``tests/protocol/test_vocabulary_census.py``) fails if the two drift.

    Reducing here rather than downstream is the point: the un-reduced
    harvest never reaches disk, which for an ablation grid is the difference
    between gigabytes of activations and kilobytes of means. The
    accumulation is fp32 regardless of the run's dtype — a bf16 sum over
    thousands of rows loses the low bits it is meant to average.
    """
    from causalab.neural.shared.executor_base import RaggedValue

    rows = value.flat if isinstance(value, RaggedValue) else value
    flat = (
        rows.detach().to(device="cpu", dtype=torch.float32).reshape(-1, rows.shape[-1])
    )
    if flat.shape[0] == 0 and reduce in ("mean", "std", "median"):
        # an unavailable cell (a scoped slice that selected no rows, spec
        # §4.1): the statistic of no observations is undefined, and NaN is the
        # honest `(width,)` answer — torch already says so for `mean` and
        # `std`, but `median` raises on an empty axis. `sum` (0) and `count`
        # (0) fall through: both are right, and together they compose.
        return torch.full((flat.shape[-1],), float("nan"), dtype=torch.float32)
    if reduce == "mean":
        out = flat.mean(dim=0)
    elif reduce == "sum":
        # the numerator half of a weighted mean across points or shards: a
        # mean of means is wrong whenever the point row counts differ
        out = flat.sum(dim=0)
    elif reduce == "std":
        # the *sample* standard deviation (torch's default correction=1): the
        # rows are a sample of examples drawn from a table, not the population.
        # One row therefore gives NaN, which is the honest answer — the spread
        # of a single observation is undefined, and a 0.0 would read as "no
        # variation".
        out = flat.std(dim=0)
    elif reduce == "median":
        # the lower of the two middle values at even row counts, which is what
        # torch.median does; no interpolation, so the saved value is one that
        # a row actually held
        out = flat.median(dim=0).values
    elif reduce == "count":
        # how many rows were reduced, as a width-vector so every reduction has
        # the one shape §2.12 promises. It is the denominator that makes `sum`
        # composable across points, and it records a truncated or ragged
        # harvest that a `mean` alone would hide.
        out = torch.full((flat.shape[-1],), float(flat.shape[0]), dtype=torch.float32)
    else:
        raise ProtocolError("P2", f"unknown save reduction {reduce!r}")
    return out.contiguous()


class MetricTable:
    """Accumulates per-example metric rows for one save file across points.

    A value may be an :class:`~causalab.protocol.resolution.Unavailable` —
    the row is a structurally unobservable measurement (its address aligned
    on nothing, its answer column is empty; spec §4.1) — and is then written
    with a ``null`` value, ``eligible: false`` and its ``reason_code``, so an
    excluded row and a row that scored ``null`` for another reason never look
    alike after a group-by (§2.10 "Eligibility")."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        values: list[Any],
        coords: Mapping[str, Any],
        point_digest: str,
        *,
        identity: Mapping[str, Any],
        labels: Sequence[str] | None = None,
    ) -> None:
        """One row per value. ``labels`` are the rows' ``example_id``s
        (``protocol/examples.py``, the base role's); ``None`` labels each row
        by its index, which is what a table without the column resolves to."""
        for label, value in zip(_labels(labels, len(values)), values):
            self.rows.append(
                self._row(name, label, value, coords, point_digest, identity=identity)
            )

    def add_windowed(
        self,
        name: str,
        values: list[list[Any]],
        coords: Mapping[str, Any],
        point_digest: str,
        *,
        identity: Mapping[str, Any],
        steps: list[list[int]] | None,
        matched: list[bool],
        labels: Sequence[str] | None = None,
    ) -> None:
        """Rows for a metric over a read that addresses several positions.

        One row per (example, position), carrying the ``step`` it scored and
        whether the example addressed anything at all. An example that
        addressed **nothing** — a row that stopped generating, or never said
        the value a ``variable`` anchor looks for — still gets exactly one
        row, with a null value and ``matched=false``: "the model never said
        it" has to be distinguishable from "it said it and scored 0", and a
        missing row would make the two look identical after a group-by.

        ``steps`` is ``None`` for a kind that reduces the whole window to
        one value (``decode``): there is no single step such a value belongs
        to, so the column stays null rather than lying about one.
        """
        row_labels = _labels(labels, len(values))
        for example, row_values in enumerate(values):
            if not row_values:
                self.rows.append(
                    self._row(
                        name,
                        row_labels[example],
                        None,
                        coords,
                        point_digest,
                        identity=identity,
                        step=None,
                        matched=matched[example],
                    )
                )
                continue
            for offset, value in enumerate(row_values):
                self.rows.append(
                    self._row(
                        name,
                        row_labels[example],
                        value,
                        coords,
                        point_digest,
                        identity=identity,
                        step=steps[example][offset] if steps is not None else None,
                        matched=matched[example],
                    )
                )

    def _row(
        self,
        name: str,
        label: str,
        value: Any,
        coords: Mapping[str, Any],
        point_digest: str,
        *,
        identity: Mapping[str, Any],
        step: int | None = None,
        matched: bool | None = None,
    ) -> dict[str, Any]:
        """One metric row: ``{example_id, metric, value, [step, matched],
        …coords, unit, estimand_version, eligible, [reason_code],
        produced_by}``. ``example_id`` is the base row's label (spec §2.2,
        ``protocol/examples.py``): the author's, or the row index as a string
        for a table without the column. ``identity`` is the record's ``unit`` /
        ``estimand_version`` (spec §2.10) — authored on the metric or derived
        from its kind (``estimand.metric_record_identity``) — repeated on
        every row, because a table has no envelope to carry it once
        (``tables.py``). ``null`` for a kind with no scalar value.

        ``eligible`` is the row's eligibility record (§2.10 "Eligibility"):
        ``false`` — with the ``reason_code`` of the ``Unavailable`` the value
        is, and a ``null`` value — for an excluded measurement; ``false`` too,
        under ``alignment_missing``, for a continuation row that addressed
        nothing (``matched: false`` — the anchor's value occurred nowhere in
        what the row generated); ``true`` otherwise, with no ``reason_code``
        column, as an available cell records nothing (§4.1)."""
        row: dict[str, Any] = {EXAMPLE_ID_COLUMN: label, "metric": name}
        excluded: Unavailable | None = value if isinstance(value, Unavailable) else None
        if excluded is not None:
            row["value"] = None
        elif isinstance(value, dict):
            import json

            row["value"] = json.dumps(value, sort_keys=True)
        else:
            row["value"] = value
        if matched is not None:
            row["step"] = step
            row["matched"] = matched
        row.update({axis: _plain(coord) for axis, coord in coords.items()})
        row.update({column: identity[column] for column in IDENTITY_COLUMNS})
        if excluded is not None:
            row[ELIGIBLE_COLUMN] = False
            row[REASON_CODE_COLUMN] = excluded.reason
        elif matched is False:
            row[ELIGIBLE_COLUMN] = False
            row[REASON_CODE_COLUMN] = "alignment_missing"
        else:
            row[ELIGIBLE_COLUMN] = True
        row["produced_by"] = point_digest
        return row


def _labels(labels: Sequence[str] | None, count: int) -> list[str]:
    if labels is None:
        return [str(index) for index in range(count)]
    if len(labels) != count:
        raise ValueError(f"{len(labels)} labels for {count} rows")
    return list(labels)


def _plain(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)):
        return value
    import json

    return json.dumps(value, sort_keys=True)


#: Where a run writes its ``train.eval`` scores. A sibling of the save
#: manifest's own files, never a column inside one: the eval score is measured
#: on a different split, so it is a different population from the metric rows
#: and does not belong in the same table (spec §2.12).
TRAIN_EVAL_FILE = "train_eval.json"

#: Where a run writes what each fit can say about *itself*. A separate file
#: from the trained bundle because the bundle's metadata is a closed identity
#: schema, and separate from the metric table because these are properties of
#: a parameter, not of an example.
FIT_DIAGNOSTICS_FILE = "fit_diagnostics.json"

#: The routing-mismatch table of every write through an
#: expert-keyed gate (spec §2.5 ``expert_neuron``): per point, write, layer and
#: example, how many of the base slots held an expert the operand's side never
#: activated — and so kept their base value — out of the slots addressed. A
#: property of the (base, counterfactual) pair's routing, not of a parameter,
#: so it sits beside :data:`FIT_DIAGNOSTICS_FILE` rather than inside it.
ROUTING_MISMATCH_FILE = "routing_mismatch.json"


def write_outputs(
    output_dir: Path,
    tensor_files: Mapping[str, TensorFile],
    metric_files: Mapping[str, MetricTable],
    *,
    identity_base: Mapping[str, Any],
    train_evals: Sequence[Mapping[str, Any]] = (),
    fit_diagnostics: Sequence[Mapping[str, Any]] = (),
    routing_mismatch: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Path]:
    """Write every accumulated save file under ``output_dir``; returns
    manifest path → absolute path.

    ``train_evals`` is one record per point that declared ``train.eval`` — the
    held-out score the fit was selected by. It is written to
    :data:`TRAIN_EVAL_FILE` only when there is something to write, so a run
    with no fit produces no empty file; ``fit_diagnostics`` and
    ``routing_mismatch`` follow the same rule.
    """
    from causalab.io.tensor_files import save_file

    written: dict[str, Path] = {}
    for rel, tensors in tensor_files.items():
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = build_artifact_identity(**identity_base)
        metadata.update(tensors.metadata)
        metadata["entries"] = json.dumps(tensors.entry_meta, sort_keys=True)
        save_file(tensors.entries, str(target), metadata=metadata)
        written[rel] = target
    for rel, table in metric_files.items():
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        write_table(target, table.rows)
        written[rel] = target
    for rel, records in (
        (TRAIN_EVAL_FILE, train_evals),
        (FIT_DIAGNOSTICS_FILE, fit_diagnostics),
        (ROUTING_MISMATCH_FILE, routing_mismatch),
    ):
        if not records:
            continue
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(list(records), indent=2) + "\n")
        written[rel] = target
    return written
