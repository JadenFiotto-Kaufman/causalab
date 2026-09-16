"""CPU-only batch planning for behavioral datasets, before model allocation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BehavioralBatch:
    """Original row indices and identities, with one explicit scientific split."""

    split: str
    indices: tuple[int, ...]
    example_ids: tuple[str, ...]
    prefix_condition: str | None
    output_length: int | None


def plan_batches(
    rows: Sequence[Mapping[str, Any]], *, batch_rows: int
) -> tuple[BehavioralBatch, ...]:
    """Group before chunking; never confuse a compute shard with a split.

    All rows need nonempty example_id and split. Prepared target rows must
    declare both prefix_condition and output_length. Parent/donor identities
    cannot cross splits. Returned indices select the original rows unchanged.
    """
    if type(batch_rows) is not int or batch_rows < 1:
        raise ValueError("batch_rows must be a positive integer")
    if not rows:
        raise ValueError("a fresh behavioral plan needs at least one row")
    groups: dict[tuple[str, str | None, int | None], list[int]] = defaultdict(list)
    seen: set[str] = set()
    parents: dict[str, str] = {}
    for index, row in enumerate(rows):
        identity, split = row.get("example_id"), row.get("split")
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(split, str)
            or not split
        ):
            raise ValueError("every row needs a nonempty example_id and split")
        if identity in seen:
            raise ValueError(f"duplicate example_id {identity!r}")
        seen.add(identity)
        for parent in (
            row.get("parent_example_id", identity),
            row.get("counterfactual_example_id"),
        ):
            if parent is not None:
                if not isinstance(parent, str) or not parent:
                    raise ValueError(
                        "parent and donor identities must be nonempty strings"
                    )
                if parents.setdefault(parent, split) != split:
                    raise ValueError(f"parent {parent!r} crosses scientific splits")
        prefix, length = row.get("prefix_condition"), row.get("output_length")
        if prefix is not None or length is not None:
            if (
                not isinstance(prefix, str)
                or not prefix
                or type(length) is not int
                or length < 0
            ):
                raise ValueError(
                    "target rows need a prefix_condition and nonnegative output_length"
                )
        groups[(split, prefix, length)].append(index)
    batches = []
    for (split, prefix, length), indices in groups.items():
        for start in range(0, len(indices), batch_rows):
            selected = tuple(indices[start : start + batch_rows])
            batches.append(
                BehavioralBatch(
                    split,
                    selected,
                    tuple(str(rows[i]["example_id"]) for i in selected),
                    prefix,
                    length,
                )
            )
    return tuple(batches)
