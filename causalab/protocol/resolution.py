"""``available`` / ``unavailable`` / ``invalid`` as values — the resolution
triple (spec §4.1).

A run resolves many things: a component to a tensor, a selector to rows, a
bundle key to an entry. Each resolution ends one of three ways, and the
protocol keeps the three apart by *type* rather than by prose:

* :class:`Available` — it resolved; ``mapping`` says to what.
* :class:`Unavailable` — the document is legal, but a **structural fact of
  the data or the model that the document could not know** left nothing to
  measure: the router sent an expert no token at the addressed positions, so
  the ``expert:`` face of that component has width zero here. This is a
  *result*: it appears in the output with its reason code and it counts in
  the denominator, so "155 of 157 cells" is one line read from the run and
  not campaign-side bookkeeping.
* :class:`Invalid` — a **document-decidable defect**: a selector that names
  no entry, an unknown component, a width that cannot be. This is what a
  validator *raises from* (the §5 checklist's :class:`~causalab.protocol
  .errors.ValidationError` with its existing code); it never travels in a
  result, and :func:`cell_record` refuses it.

The rule: **``unavailable`` belongs in ordinary results and denominators,
``invalid`` stops validation.** A cell that is unavailable because of a
registry fact cites the capability row (:func:`~causalab.protocol.registry
.capability`'s ``reason`` / ``why``) rather than restating it; a cell that is
unavailable because of a data fact says which fact, in ``detail``.

``reason`` is drawn from the closed :data:`~causalab.protocol.errors
.REASON_CODES` (spec §2.4's table) — the same vocabulary a refusal carries in
:attr:`~causalab.protocol.errors.ProtocolError.reason`, so a caller branches
on one set of names whether the fact surfaced as a value or as an exception.

**The denominator is data.** ``denominator_key`` is the string an aggregator
counts a cell under: :func:`cell_key` — the saved value's name plus the
point's coordinate label, which is also the tensor entry's key in a bundle
(``r[layer=3,pos=2]``, or bare ``r`` for an un-swept document). Defined once
here so the result file, the per-point summary and the CLI's ``cells`` line
all count the same thing. :class:`Denominator` is the reduction: ``eligible``
of ``total``, with the excluded cells grouped by reason.

This module imports no torch and no engine: the triple is part of the pure
layer, so a consumer that only reads results (a summary script, a dry run)
can name the states without an accelerator.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping

from causalab.protocol.bundles import entry_key
from causalab.protocol.errors import REASON_CODES, ReasonCode
from causalab.protocol.sweep import coordinate_label

__all__ = [
    "STATUS_KEY",
    "UNAVAILABLE",
    "Available",
    "Denominator",
    "Eligibility",
    "Invalid",
    "Resolution",
    "Unavailable",
    "available",
    "cell_key",
    "cell_record",
    "invalid",
    "unavailable",
]

#: The result-cell field that marks an unavailable cell. An available cell
#: records **nothing** — no ``status: "available"`` — so every result written
#: before this value existed is byte-identical to one written after.
STATUS_KEY = "status"
#: The one value :data:`STATUS_KEY` takes.
UNAVAILABLE = "unavailable"


@dataclasses.dataclass(frozen=True)
class Available:
    """A resolution that produced something: ``mapping`` names what (the
    saved key, the rows, the address — whatever the resolving site knows)."""

    mapping: Mapping[str, Any]
    denominator_key: str


@dataclasses.dataclass(frozen=True)
class Unavailable:
    """A legal cell with nothing to measure, and why.

    ``reason`` is a :data:`~causalab.protocol.errors.ReasonCode`; ``detail`` is
    the fact in prose (which expert, at which positions); ``denominator_key``
    is what the aggregator counts this cell under (:func:`cell_key`).
    """

    reason: ReasonCode
    detail: str
    denominator_key: str

    def __post_init__(self) -> None:
        if self.reason not in REASON_CODES:
            raise AssertionError(
                f"unknown reason code {self.reason!r}; expected one of {REASON_CODES}"
            )

    def record(self) -> dict[str, Any]:
        """The four result-cell fields (spec §4.1)."""
        return {
            STATUS_KEY: UNAVAILABLE,
            "reason": self.reason,
            "detail": self.detail,
            "denominator_key": self.denominator_key,
        }


@dataclasses.dataclass(frozen=True)
class Invalid:
    """A document-decidable defect: what a validator raises from.

    ``error_code`` is the existing rule or parse code (``V15``, ``P4``) — the
    triple adds no rule numbers. An ``Invalid`` never enters a result; it is
    not a member of :data:`Resolution`, and :func:`cell_record` refuses it.
    """

    error_code: str
    detail: str


#: The value-carrying pair. ``Invalid`` is deliberately not in it.
Resolution = Available | Unavailable


def available(mapping: Mapping[str, Any], denominator_key: str) -> Available:
    return Available(mapping=dict(mapping), denominator_key=denominator_key)


def unavailable(reason: ReasonCode, detail: str, denominator_key: str) -> Unavailable:
    return Unavailable(reason=reason, detail=detail, denominator_key=denominator_key)


def invalid(error_code: str, detail: str) -> Invalid:
    return Invalid(error_code=error_code, detail=detail)


def cell_key(value: str, coords: Mapping[str, Any]) -> str:
    """The denominator key of one saved value at one point.

    The value's name plus the point's coordinate label — exactly the key the
    value's tensor entry takes in a saved bundle (``entry_key``), so a
    reader can go from the ``cells`` line to the entry it names.
    """
    return entry_key(value, coordinate_label(coords, entry=value))


def cell_record(cell: Resolution | Invalid) -> dict[str, Any]:
    """What a result cell records about its resolution: nothing when
    available, the four fields when unavailable. Refuses an ``Invalid`` —
    a defect stops validation and never becomes a cell — which is why the
    signature admits one: so the refusal is the contract, not a type hint."""
    if isinstance(cell, Available):
        return {}
    if isinstance(cell, Unavailable):
        return cell.record()
    raise TypeError(
        f"{type(cell).__name__} is not a result value: `invalid` stops "
        "validation and never enters a result"
    )


@dataclasses.dataclass(frozen=True)
class Denominator:
    """``eligible`` of ``total`` cells, with the excluded ones by reason.

    The numbers a summary reads instead of keeping its own books: an
    aggregator that means over ``eligible`` cells and names the excluded
    ones is honest about both.
    """

    total: int
    eligible: int
    #: reason → the excluded cells' denominator keys, sorted
    unavailable: Mapping[ReasonCode, tuple[str, ...]]

    @classmethod
    def of(cls, cells: Iterable[Resolution]) -> "Denominator":
        total = eligible = 0
        excluded: dict[ReasonCode, list[str]] = {}
        for cell in cells:
            cell_record(cell)  # refuses an Invalid before it is counted
            total += 1
            if isinstance(cell, Available):
                eligible += 1
            else:
                excluded.setdefault(cell.reason, []).append(cell.denominator_key)
        return cls(
            total=total,
            eligible=eligible,
            unavailable={
                reason: tuple(sorted(keys)) for reason, keys in sorted(excluded.items())
            },
        )

    @property
    def excluded(self) -> int:
        return self.total - self.eligible

    def as_record(self) -> dict[str, Any]:
        """The denominator as numbers: ``eligible``, ``total``, and per reason
        the count and the excluded keys."""
        return {
            "eligible": self.eligible,
            "total": self.total,
            "unavailable": {
                reason: {"count": len(keys), "cells": list(keys)}
                for reason, keys in self.unavailable.items()
            },
        }

    def render(self) -> str:
        """One line: ``155 / 157 eligible; 2 excluded: empty_selector ×2``."""
        head = f"{self.eligible} / {self.total} eligible"
        if not self.excluded:
            return head
        by_reason = ", ".join(
            f"{reason} ×{len(keys)}" for reason, keys in self.unavailable.items()
        )
        return f"{head}; {self.excluded} excluded: {by_reason}"


@dataclasses.dataclass(frozen=True)
class Eligibility:
    """How many of a metric cell's rows its decision rule was evaluated over
    (spec §2.10 "Eligibility"): ``n_eligible`` of ``n_considered``, with the
    excluded rows counted by reason.

    The row-level twin of :class:`Denominator`, and a **third** denominator
    named apart from the other two on purpose: ``n_eligible`` counts the rows
    a metric's *decision rule* saw; ``save.reduce: "count"`` (§2.12) counts
    the rows a saved read's reduction collapsed; a workflow reduction's
    ``unit`` (workflow §2.6) is the statistical unit a table is later reduced
    over. Derived from the rows, never authored (§6): ``of`` reads a metric's
    per-example values, where an excluded row is an :class:`Unavailable`.
    """

    n_eligible: int
    n_considered: int
    #: reason → how many rows it excluded, sorted by reason
    excluded: Mapping[ReasonCode, int]

    @classmethod
    def of(cls, values: Iterable[Any]) -> "Eligibility":
        considered = eligible = 0
        by_reason: dict[ReasonCode, int] = {}
        for value in values:
            considered += 1
            if isinstance(value, Unavailable):
                by_reason[value.reason] = by_reason.get(value.reason, 0) + 1
            else:
                eligible += 1
        return cls(
            n_eligible=eligible,
            n_considered=considered,
            excluded=dict(sorted(by_reason.items())),
        )

    def as_record(self) -> dict[str, Any]:
        """The aggregate cell's two counts, plus ``excluded`` by reason only
        when a row was excluded — a cell whose every row was eligible records
        exactly the two numbers."""
        record: dict[str, Any] = {
            "n_eligible": self.n_eligible,
            "n_considered": self.n_considered,
        }
        if self.excluded:
            record["excluded"] = dict(self.excluded)
        return record
