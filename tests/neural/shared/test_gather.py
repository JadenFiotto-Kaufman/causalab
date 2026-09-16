"""The position gather with a sort-free backward (``neural/shared/gather.py``).

The executor's ``tensor[row_ids, idx]`` gathers repeat no element when no two
entries of the position table name the same ``(row, position)``, and then
autograd's sorted, accumulating scatter is a plain scatter. This pins the
claim that buys the kernel: the value is the plain index's, and the gradient
is the plain index's **to the bit** — on a dense ``(rows, 1) × (rows,
width)`` index, on two flat vectors (the ragged landing), on a
feature-sliced use, and with a downstream that mixes positions so every
gathered element carries a gradient. A table with a repeated element is
handed to autograd's own backward unchanged, and the tell — the sort-free
backward on such a table would drop a contribution — is why the verdict is
decided where the index is built (``PositionIndex.distinct``, over the
pairs: a batch row named twice by two distinct table rows is a repeat too)
rather than asserted by a caller.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.shared.gather import (
    PositionIndex,
    dense_index,
    flat_index,
    gather_positions,
    rows_are_distinct,
)


def _reference(
    tensor: torch.Tensor, index: PositionIndex
) -> tuple[torch.Tensor, torch.Tensor]:
    """The plain advanced index and its autograd gradient under a
    position-mixing downstream."""
    source = tensor.detach().clone().requires_grad_(True)
    out = source[index.pair]
    _downstream(out).backward()
    assert source.grad is not None
    return out.detach(), source.grad.detach()


def _downstream(out: torch.Tensor) -> torch.Tensor:
    # every element weighted differently in fp32, so a dropped or doubled
    # contribution changes the gradient somewhere. On a bf16 `out` the cast's
    # backward rounds the upstream gradient to bf16 (exact integers only to
    # 256), so neighbouring weights can coincide there; detection on that leg
    # rests on the element-wise `torch.equal` over the whole source gradient
    weights = torch.arange(1, out.numel() + 1, dtype=torch.float32).reshape(out.shape)
    return (out.float() * weights).sum() + (out.float().sum(dim=-2) ** 2).sum()


def _check(tensor: torch.Tensor, index: PositionIndex) -> None:
    out = gather_positions(tensor, index)
    _downstream(out).backward()
    ref_out, ref_grad = _reference(tensor, index)
    assert torch.equal(out.detach(), ref_out)
    assert tensor.grad is not None and torch.equal(tensor.grad, ref_grad)


class TestNumerics:
    pytestmark = pytest.mark.numerical_unit

    def test_the_dense_gather_matches_the_plain_index_and_its_gradient(self) -> None:
        torch.manual_seed(0)
        tensor = torch.randn(3, 6, 5, requires_grad=True)
        index = dense_index([[5, 1, 3], [0, 2, 4], [1, 5, 0]], "cpu")
        assert index.distinct
        _check(tensor, index)
        # elements the index never named carry no gradient
        untouched = torch.ones(3, 6, dtype=torch.bool)
        untouched[index.pair] = False
        assert tensor.grad is not None
        assert torch.equal(
            tensor.grad[untouched], torch.zeros_like(tensor.grad[untouched])
        )

    def test_the_flat_gather_matches_on_a_ragged_table(self) -> None:
        torch.manual_seed(1)
        tensor = torch.randn(3, 7, 4, requires_grad=True)
        index = flat_index([[6, 2], [0], [1, 3, 5]], "cpu")
        assert index.distinct
        _check(tensor, index)

    def test_a_feature_sliced_use_keeps_the_gradient(self) -> None:
        """The landing slices features off the gathered value; the gradient of
        the unsliced features is zero and the sliced ones match."""
        torch.manual_seed(2)
        tensor = torch.randn(2, 5, 8, requires_grad=True)
        index = dense_index([[4, 0], [2, 3]], "cpu")
        out = gather_positions(tensor, index)[..., 2:6]
        _downstream(out).backward()
        source = tensor.detach().clone().requires_grad_(True)
        _downstream(source[index.pair][..., 2:6]).backward()
        assert source.grad is not None and tensor.grad is not None
        assert torch.equal(tensor.grad, source.grad)

    def test_bf16_matches_too(self) -> None:
        torch.manual_seed(3)
        tensor = torch.randn(4, 9, 16, dtype=torch.bfloat16, requires_grad=True)
        index = dense_index(
            [[8, 1, 4, 6], [0, 2, 3, 5], [7, 8, 1, 0], [3, 4, 5, 6]], "cpu"
        )
        assert index.distinct
        _check(tensor, index)

    def test_a_repeated_position_takes_the_accumulating_backward(self) -> None:
        """The ``padded_masked`` landing pads a row with its last real position:
        the gradients of the two slots sum. The index built from that table
        says so itself, and the gather keeps autograd's backward."""
        torch.manual_seed(4)
        tensor = torch.randn(2, 5, 3, requires_grad=True)
        index = dense_index([[1, 4, 4], [0, 2, 3]], "cpu")
        assert not index.distinct
        _check(tensor, index)

    def test_a_batch_row_named_twice_takes_the_accumulating_backward(self) -> None:
        """Two distinct table rows on the same batch row with overlapping
        positions name one element twice: the bucket landing's ``rows=`` is
        part of the verdict, not only the positions."""
        torch.manual_seed(5)
        tensor = torch.randn(3, 5, 3, requires_grad=True)
        index = dense_index([[0, 1], [1, 2]], "cpu", rows=[2, 2])
        assert not index.distinct
        _check(tensor, index)

    def test_mutation_the_sort_free_backward_is_wrong_on_a_repeated_element(
        self,
    ) -> None:
        """Why the verdict is decided with the index: on a table with a
        duplicate, the non-accumulating scatter keeps one of the two
        contributions and the gradient is off. Reachable only by asserting
        ``distinct`` past the builder."""
        torch.manual_seed(6)
        tensor = torch.randn(2, 5, 3, requires_grad=True)
        built = dense_index([[1, 4, 4], [0, 2, 3]], "cpu")
        forced = PositionIndex(built.row_ids, built.idx, distinct=True)
        out = gather_positions(tensor, forced)
        _downstream(out).backward()
        _, ref_grad = _reference(tensor, built)
        assert tensor.grad is not None
        assert not torch.equal(tensor.grad, ref_grad)


class TestProperties:
    pytestmark = pytest.mark.property

    @settings(max_examples=60, deadline=None)
    @given(
        rows=st.integers(min_value=1, max_value=4),
        seq=st.integers(min_value=1, max_value=8),
        width=st.integers(min_value=1, max_value=8),
        features=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    def test_any_distinct_table_reproduces_the_plain_gradient(
        self, rows: int, seq: int, width: int, features: int, seed: int
    ) -> None:
        width = min(width, seq)
        generator = torch.Generator().manual_seed(seed)
        per_row = [
            torch.randperm(seq, generator=generator)[:width].tolist()
            for _ in range(rows)
        ]
        index = dense_index(per_row, "cpu")
        assert index.distinct
        tensor = torch.randn(
            rows, seq, features, generator=generator, requires_grad=True
        )
        _check(tensor, index)

    @settings(max_examples=60, deadline=None)
    @given(
        rows=st.integers(min_value=1, max_value=4),
        seq=st.integers(min_value=1, max_value=6),
        width=st.integers(min_value=1, max_value=6),
        features=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    def test_any_table_reproduces_the_plain_gradient(
        self, rows: int, seq: int, width: int, features: int, seed: int
    ) -> None:
        """Repeats allowed, in positions and in batch rows: whichever
        backward the index picks, the gradient is autograd's."""
        generator = torch.Generator().manual_seed(seed)
        per_row = torch.randint(0, seq, (rows, width), generator=generator).tolist()
        batch_rows = torch.randint(0, rows, (rows,), generator=generator).tolist()
        index = dense_index(per_row, "cpu", rows=batch_rows)
        assert index.distinct == rows_are_distinct(per_row, batch_rows)
        tensor = torch.randn(
            rows, seq, features, generator=generator, requires_grad=True
        )
        _check(tensor, index)

    def test_without_a_gradient_the_plain_index_is_returned(self) -> None:
        tensor = torch.randn(2, 4, 3)
        index = dense_index([[3, 0], [1, 2]], "cpu")
        out = gather_positions(tensor, index)
        assert not out.requires_grad
        assert torch.equal(out, tensor[index.pair])
        with torch.no_grad():
            leaf = tensor.clone().requires_grad_(True)
            out = gather_positions(leaf, index)
        assert not out.requires_grad

    def test_rows_are_distinct_reads_the_pairs(self) -> None:
        assert rows_are_distinct([[0, 1], [1, 0]])
        assert rows_are_distinct([])
        assert rows_are_distinct([[], [3]])
        assert not rows_are_distinct([[0, 1], [2, 2]])
        # the same batch row twice: distinct rows, but overlapping elements
        assert not rows_are_distinct([[0, 1], [1, 2]], rows=[4, 4])
        assert rows_are_distinct([[0, 1], [2, 3]], rows=[4, 4])


class TestIndexTensors:
    """The index tensors a position table gathers with are built once per
    distinct table and device, and are the plain construction's."""

    pytestmark = pytest.mark.property

    def test_dense_is_the_plain_construction_and_is_reused(self) -> None:
        per_row = [[5, 1, 3], [0, 2, 4]]
        index = dense_index(per_row, "cpu")
        assert torch.equal(index.row_ids, torch.arange(2).unsqueeze(1))
        assert torch.equal(index.idx, torch.tensor(per_row))
        again = dense_index([list(r) for r in per_row], torch.device("cpu"))
        assert again is index
        other = dense_index([[5, 1, 3], [0, 2, 5]], "cpu")
        assert other.idx is not index.idx

    def test_rows_name_which_batch_rows_the_table_covers(self) -> None:
        index = dense_index([[1, 2], [3, 4]], "cpu", rows=[4, 1])
        assert index.row_ids.tolist() == [[4], [1]]
        assert index.idx.tolist() == [[1, 2], [3, 4]]
        default = dense_index([[1, 2], [3, 4]], "cpu")
        assert default is not index  # a different table

    def test_flat_is_one_entry_per_position(self) -> None:
        per_row = [[6, 2], [], [1, 3, 5]]
        index = flat_index(per_row, "cpu")
        assert index.row_ids.tolist() == [0, 0, 2, 2, 2]
        assert index.idx.tolist() == [6, 2, 1, 3, 5]
        assert flat_index(per_row, "cpu") is index
        tensor = torch.randn(3, 7, 2)
        assert torch.equal(tensor[index.pair], tensor[[0, 0, 2, 2, 2], [6, 2, 1, 3, 5]])
