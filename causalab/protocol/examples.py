"""The row label of an example — ``example_id`` (spec §2.2).

A dataset may carry an ``example_id`` column: the author's name for the row.
When it does, every row must carry one, non-empty and unique within the table.
When it does not, a row's label is its zero-based index as a string. Either way
one label names one example, and the base row's label names the pair — rows
are paired by index and the base role is never permuted, so the label a metric
row carries is the base row's.

Torch-free: the loader checks the column on ``validate --data``, the executor
labels metric rows, the engine labels continuations, and the workflow's
behavioral step joins the two — all through these two functions.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: The optional dataset column, and the column every per-example table writes.
EXAMPLE_ID_COLUMN = "example_id"


def example_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """One label per row: its ``example_id`` as a string, or its index.

    Assumes :func:`example_id_defect` returned ``None`` for ``rows``.
    """
    if not any(EXAMPLE_ID_COLUMN in row for row in rows):
        return [str(index) for index in range(len(rows))]
    return [str(row[EXAMPLE_ID_COLUMN]) for row in rows]


def example_id_defect(rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Why the table's ``example_id`` column cannot label its rows, or ``None``.

    A table without the column has no defect. One that has it must carry it
    on every row, non-empty, and no two rows may share a label.
    """
    carrying = [index for index, row in enumerate(rows) if EXAMPLE_ID_COLUMN in row]
    if not carrying:
        return None
    if len(carrying) != len(rows):
        missing = [i for i in range(len(rows)) if i not in set(carrying)]
        return (
            f"rows {missing[:3]}{'…' if len(missing) > 3 else ''} carry no "
            f"{EXAMPLE_ID_COLUMN} while other rows do"
        )
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        value = row[EXAMPLE_ID_COLUMN]
        label = "" if value is None else str(value)
        if not label.strip():
            return f"row {index} has an empty {EXAMPLE_ID_COLUMN}"
        if label in seen:
            return (
                f"{EXAMPLE_ID_COLUMN} {label!r} labels rows {seen[label]} and {index}"
            )
        seen[label] = index
    return None
