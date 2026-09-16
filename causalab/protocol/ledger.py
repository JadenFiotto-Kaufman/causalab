"""The location ledger (spec §6, §7) — resolved token indices as
an auditable **derived output**, and its digest as a recorded provenance key.

A document never contains resolved token indices (§7); the engine derives
them when it encodes its inputs. The ledger is the record of that derivation:
**one row per (example, edit group, constituent, side, token index, token id,
decoded token)** — every token every position of the run addressed, on every
input, with the id and the decoded piece that sat there. It is the table an
author would otherwise build by hand to prove resolution was right, made a ``save`` kind
(``{"kind": "location_ledger", "file_path": …}``, §2.12): **opt-in per
document, no default output** — a campaign addressing hundreds of occurrences
has a large ledger,
and a run that did not ask for one writes nothing.

**The digest.** :func:`ledger_digest` is ``sha256`` over the canonical rows:
each row serialized as JSON with sorted keys and minimal separators
(``json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``),
the serialized rows sorted as strings, joined by ``"\\n"``, UTF-8. Only the
seven columns enter it — never the point digest or coordinates a saved row
also carries — so a reader can recompute it from the saved table alone.

**The stamp.** A run that emitted a ledger stamps its digest as
``location_ledger_sha256`` on every artifact it writes
(:data:`~causalab.protocol.resolve.ARTIFACT_IDENTITY_KEYS`). It is
**recorded, never compared**, like ``engine``: it names the ledger table the
fit was made under, so a reader can find the tokens the parameter was trained
on and set them beside the tokens a later run selects. A run that loads the
artifact and saves its own ledger gets the tokens it resolved on *its* rows,
whatever the fit's were — applying a fitted parameter to other data is the
ordinary use, and the ledger is how the researcher inspects the selection.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Iterable, Mapping

__all__ = [
    "LEDGER_COLUMNS",
    "LEDGER_IDENTITY_KEY",
    "LedgerRow",
    "LocationLedger",
    "ledger_digest",
]

#: The seven columns of one ledger row (§6), in this order.
LEDGER_COLUMNS: tuple[str, ...] = (
    "example",
    "edit_group",
    "constituent",
    "side",
    "token_index",
    "token_id",
    "decoded_token",
)
#: The ``ArtifactIdentity`` key the ledger's digest is stamped under (§8).
LEDGER_IDENTITY_KEY: str = "location_ledger_sha256"


@dataclasses.dataclass(frozen=True)
class LedgerRow:
    """One addressed token: which example, in which forward group (``model``
    on ``input``), by which constituent (a position's name, or ``name[k]`` for
    the k-th member of a non-atomic set), on which side (the input role), at
    which index of the row's own token sequence (0 = the row's first real
    token, chat prefix included — padding never enters), with what token id
    and decoded piece."""

    example: int
    edit_group: str
    constituent: str
    side: str
    token_index: int
    token_id: int
    decoded_token: str

    def as_record(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in LEDGER_COLUMNS}


class LocationLedger:
    """The rows one point's resolution produced, keyed so that resolving the
    same address twice (the width pre-flight, then the write) records it once.
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[int, str, str, str, int], LedgerRow] = {}

    def add(self, row: LedgerRow) -> None:
        key = (row.example, row.edit_group, row.constituent, row.side, row.token_index)
        previous = self._rows.get(key)
        if previous is not None and previous != row:
            raise AssertionError(
                f"ledger row {key} recorded twice with different tokens: "
                f"{previous} vs {row}"
            )
        self._rows[key] = row

    def rows(self) -> tuple[LedgerRow, ...]:
        """Every row, by (example, edit group, constituent, side, token
        index) — a reading order; the digest sorts its own lines."""
        return tuple(self._rows[key] for key in sorted(self._rows))

    def records(self) -> list[dict[str, Any]]:
        return [row.as_record() for row in self.rows()]

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def digest(self) -> str:
        return ledger_digest(self.records())


def _canonical(record: Mapping[str, Any]) -> str:
    return json.dumps(
        {column: record[column] for column in LEDGER_COLUMNS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def ledger_digest(records: Iterable[Mapping[str, Any]]) -> str:
    """``sha256`` over the canonical rows (module docstring): sorted-key
    minimal JSON per row, rows sorted, ``\\n``-joined, UTF-8. Only the seven
    ledger columns of each record are read."""
    lines = sorted(_canonical(record) for record in records)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
