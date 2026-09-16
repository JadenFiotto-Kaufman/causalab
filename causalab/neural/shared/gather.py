"""Position gathers whose backward matches what the index can hold.

The executor reads and writes a tap's tensor at per-row positions through
one advanced index, ``tensor[row_ids, idx]`` (``row_ids (rows, 1)`` against
``idx (rows, width)``, or two flat index vectors for a ragged landing).
Autograd's backward for that gather is ``index_put_(..., accumulate=True)``:
on CUDA a sort of the flattened indices and a segmented accumulate
(``indexing_backward_kernel``), because two entries of the index *could* name
the same element and their gradients would have to sum. 📐 Profiled
(``fullprof0910``, das optimizer step 5): 97 launches × 36 µs per step,
ALU-bound on the sort — for a gather that never repeats an element.

The executor builds the index from Python position lists, so it knows without
a device round-trip whether two entries name the same ``(row, position)``
element. When none do, every gathered element has exactly one source and the
backward is a plain scatter — the non-accumulating ``index_put_``, one
elementwise kernel and no sort — with the same gradient to the bit: each
destination receives its one value, every other element is zero, nothing is
summed. That fact travels *with* the index: :func:`dense_index` and
:func:`flat_index` decide it from the lists they are built from
(:func:`rows_are_distinct`, over the pairs — a table whose rows are distinct
but which names one batch row twice is not distinct) and hand it to
:func:`gather_positions` as :attr:`PositionIndex.distinct`, so no caller can
pair a repeating index with the sort-free backward. A gather that may repeat
an element (the ``padded_masked`` landing pads a row with a duplicate of its
last real position) keeps autograd's accumulating one.

Two facts about the paths this picks, for the record: ``index_put_`` with
``accumulate=False`` is one of the CUDA ops ``torch.use_deterministic_algorithms``
refuses (the accumulating one is on its deterministic list) — nothing in this
repository enables that mode, and on a distinct index the scatter has no race
to be nondeterministic about, but the sort-free path is what a future
deterministic-mode run would have to fall back from. And the index tensors
are cached per distinct table (:data:`_INDEX_CACHE_SIZE`): the reuse is a
property of repeated forwards over one fixed frame — an eval, a
``--cuda-graphs`` replay — where every hook call resolves the same table; a
fit's minibatches are each a different table, so a training step keys the
cache and misses, and its dead entries are evicted by count.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Sequence

import torch

__all__ = [
    "PositionIndex",
    "dense_index",
    "flat_index",
    "gather_positions",
    "rows_are_distinct",
    "splice_features",
]

#: How many distinct position tables' index tensors stay resident per
#: process. A table is a few hundred bytes at eval width (a fit's tables are
#: larger and churn — module docstring); an eval workflow addresses a few
#: dozen.
_INDEX_CACHE_SIZE = 1024


@dataclasses.dataclass(frozen=True, eq=False)
class PositionIndex:
    """The index tensors a position table gathers with, and what that table
    can hold: ``tensor[row_ids, idx]`` is the gather, ``distinct`` says no two
    entries of it name the same ``(row, position)`` element — decided on the
    host, from the lists, when the index was built. Compared by identity
    (``eq=False``): one object per cached table, and a field-wise ``__eq__``
    over tensors would raise on its own truth value."""

    row_ids: torch.Tensor
    idx: torch.Tensor
    distinct: bool

    @property
    def pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The advanced index itself — ``tensor[index.pair]``."""
        return self.row_ids, self.idx


def rows_are_distinct(
    per_row: Sequence[Sequence[int]], rows: Sequence[int] | None = None
) -> bool:
    """Whether no two entries of a position table name the same ``(row,
    position)`` element — the condition under which a gather at those
    positions has one source per element. ``rows`` names the batch row each
    table row belongs to (the table's own order by default); a batch row
    named twice makes overlapping positions collide even when every table
    row is distinct on its own."""
    row_names = range(len(per_row)) if rows is None else rows
    pairs = [(r, p) for r, row in zip(row_names, per_row) for p in row]
    return len(set(pairs)) == len(pairs)


@functools.lru_cache(maxsize=_INDEX_CACHE_SIZE)
def _dense_index(
    rows: tuple[int, ...], per_row: tuple[tuple[int, ...], ...], device: str
) -> PositionIndex:
    return PositionIndex(
        row_ids=torch.tensor(rows, dtype=torch.long, device=device).unsqueeze(1),
        idx=torch.tensor(per_row, dtype=torch.long, device=device),
        distinct=rows_are_distinct(per_row, rows),
    )


@functools.lru_cache(maxsize=_INDEX_CACHE_SIZE)
def _flat_index(
    rows: tuple[int, ...], per_row: tuple[tuple[int, ...], ...], device: str
) -> PositionIndex:
    return PositionIndex(
        row_ids=torch.tensor(
            [r for r, row in zip(rows, per_row) for _ in row],
            dtype=torch.long,
            device=device,
        ),
        idx=torch.tensor(
            [p for row in per_row for p in row], dtype=torch.long, device=device
        ),
        distinct=rows_are_distinct(per_row, rows),
    )


def dense_index(
    per_row: Sequence[Sequence[int]],
    device: torch.device | str,
    *,
    rows: Sequence[int] | None = None,
) -> PositionIndex:
    """The index for a position table whose rows all have one width:
    ``row_ids (rows, 1)`` against ``idx (rows, width)``. ``rows`` names the
    batch rows the table's entries belong to (the table's own order by
    default).

    Built once per distinct ``(rows, table, device)`` and reused: the
    executor resolves the same positions for the same batch on every forward
    (each read and write, every hook call), and each ``torch.tensor(list,
    device="cuda")`` is a host→device copy the launch queue waits on. The
    tensors are only ever read from, so sharing them is safe.
    """
    rows_key = tuple(range(len(per_row))) if rows is None else tuple(rows)
    return _dense_index(rows_key, tuple(tuple(r) for r in per_row), str(device))


def flat_index(
    per_row: Sequence[Sequence[int]],
    device: torch.device | str,
    *,
    rows: Sequence[int] | None = None,
) -> PositionIndex:
    """The index as two flat vectors, one entry per position of every row —
    the ragged form of :func:`dense_index`, the same caching."""
    rows_key = tuple(range(len(per_row))) if rows is None else tuple(rows)
    return _flat_index(rows_key, tuple(tuple(r) for r in per_row), str(device))


class _DistinctGather(torch.autograd.Function):
    """``tensor[row_ids, idx]`` for an index with one source per element:
    the backward scatters the gradient without accumulating."""

    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, row_ids: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(row_ids, idx)
        ctx.source_shape = tensor.shape
        return tensor[row_ids, idx]

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        row_ids, idx = ctx.saved_tensors
        out = grad.new_zeros(ctx.source_shape)
        out.index_put_((row_ids, idx), grad, accumulate=False)
        return out, None, None


def gather_positions(tensor: torch.Tensor, index: PositionIndex) -> torch.Tensor:
    """``tensor[index.pair]``, with the sort-free backward when the index
    repeats no element (:attr:`PositionIndex.distinct`).

    The forward is the plain advanced index either way, so the value is the
    same object autograd would produce; only the backward differs, and only
    when a gradient can flow (a tensor that needs none, or a no-grad forward,
    takes the plain index and pays nothing).
    """
    if not index.distinct or not torch.is_grad_enabled() or not tensor.requires_grad:
        return tensor[index.pair]
    return _DistinctGather.apply(tensor, index.row_ids, index.idx)


def splice_features(
    landed: torch.Tensor, fslice: slice, value: torch.Tensor
) -> torch.Tensor:
    """``landed`` with ``value`` in its feature slice ``fslice`` — what a
    landing writes back over the positions it gathered ``landed`` from.

    When the slice is the whole feature axis the written value *is* the
    whole slice, and ``value`` is returned as it is: no copy, and no second
    gather of the positions to splice into (the landing used to re-gather
    them, paying the sorted backward a second time).
    """
    if fslice == slice(None):
        return value
    out = landed.clone()
    out[..., fslice] = value
    return out
