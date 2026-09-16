"""The location ledger at the engine seam (spec §6, §2.12, §8): build it from
an executor's resolved positions when — and only when — the document asks,
and stamp its digest on what the point writes.

The ledger itself — rows and digest — is torch-free ``protocol/ledger.py``;
the rows come from ``ExecutorBase.location_ledger``, which resolves every
prompt-frame position of every read and write through the same ``_positions``
the gathers use. This module is the glue ``execution.py`` calls: three
functions, so the point loop gains three lines.
"""

from __future__ import annotations

from typing import Any, Mapping

from causalab.protocol.ledger import LEDGER_IDENTITY_KEY, LocationLedger
from causalab.protocol.schema import Document

__all__ = ["ledger_identity", "ledger_records", "point_ledger", "wants_ledger"]


def wants_ledger(doc: Document) -> bool:
    """Whether the document opts into the ledger: a ``save`` entry of kind
    ``location_ledger`` (§2.12). Nothing else makes a run write one."""
    return any(entry.kind == "location_ledger" for entry in doc.save)


def point_ledger(executor: Any, doc: Document) -> LocationLedger | None:
    """This point's ledger when the document asks for one, else ``None``.
    Built before any forward: positions are a fact of the encoded batch. A
    loaded artifact's own stamped ledger digest is provenance to read, not a
    condition to meet — the ledger lists the tokens this run selected on its rows."""
    if not wants_ledger(doc):
        return None
    return executor.location_ledger()


def ledger_identity(ledger: LocationLedger | None) -> dict[str, str]:
    """The identity fields a ledger adds to every artifact the point writes:
    its digest under :data:`~causalab.protocol.ledger.LEDGER_IDENTITY_KEY`,
    or nothing when no ledger was emitted (absent equals absent)."""
    return {LEDGER_IDENTITY_KEY: ledger.digest} if ledger is not None else {}


def ledger_records(
    ledger: LocationLedger, point_digest: str, coords: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The rows the saved table carries for one point: the seven ledger
    columns plus the point's provenance (``point``) and coordinates
    (``coords``), which a swept document needs to tell its points apart and
    which the digest deliberately excludes."""
    return [
        {**record, "point": point_digest, "coords": dict(coords)}
        for record in ledger.records()
    ]
