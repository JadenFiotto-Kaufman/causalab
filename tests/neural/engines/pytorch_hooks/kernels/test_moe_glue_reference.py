"""The ATen semantics of the grouped-experts glue
(``pytorch_hooks/kernels/moe_glue_reference.py``), pinned on the CPU.

The references are plain torch in fp32, so wherever the CPU kernel takes the
same order as the CUDA one they are held to ``torch.equal`` against the
library ops; where the CPU order differs (the two reductions) they are held
to the order written in the docstring, with a mutation showing the order is
load-bearing, and to the true sum within float tolerance. Anything downstream
of a transcendental (the gate's ``exp``) is held within a couple of ulps,
since the CPU's ``exp`` is vendor-specific. The CUDA orders and the gate
kernel's bits are pinned on the device by ``tests/golden/test_moe_glue_kernels.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.kernels import moe_glue_reference as ref

WIDTHS = (1, 3, 8, 16, 33, 64, 136)


@st.composite
def routing_tables(
    draw: st.DrawFn, max_tokens: int = 24
) -> tuple[int, int, torch.Tensor]:
    """``(num_experts, top_k, top_k_index)`` with skewed and empty experts:
    the ids are drawn from a random subset of the experts."""
    num_experts = draw(st.integers(1, 48))
    top_k = draw(st.integers(1, 8))
    tokens = draw(st.integers(1, max_tokens))
    active = draw(
        st.lists(
            st.integers(0, num_experts - 1),
            min_size=1,
            max_size=num_experts,
            unique=True,
        )
    )
    ids = draw(
        st.lists(
            st.sampled_from(active), min_size=tokens * top_k, max_size=tokens * top_k
        )
    )
    return num_experts, top_k, torch.tensor(ids, dtype=torch.long).view(tokens, top_k)


def _perm(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.sort(index.reshape(-1), stable=True).indices
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    return perm, inv


class TestStableCountingSort:
    pytestmark = pytest.mark.property

    @given(routing_tables())
    @settings(max_examples=80, deadline=None)
    def test_the_stable_sort_its_inverse_and_the_histogram_offsets(
        self, table: tuple[int, int, torch.Tensor]
    ) -> None:
        num_experts, _, index = table
        ids = index.reshape(-1)
        perm, inv_perm, offsets = ref.stable_counting_sort(ids, num_experts)
        assert torch.equal(perm, torch.sort(ids, stable=True).indices)
        assert torch.equal(inv_perm[perm], torch.arange(ids.numel()))
        assert torch.equal(perm[inv_perm], torch.arange(ids.numel()))
        counts = torch.histc(ids.float(), bins=num_experts, min=0, max=num_experts - 1)
        assert torch.equal(offsets, torch.cumsum(counts, 0, dtype=torch.int32))
        assert offsets.dtype is torch.int32 and int(offsets[-1]) == ids.numel()

    @given(routing_tables())
    @settings(max_examples=40, deadline=None)
    def test_within_an_expert_rows_keep_ascending_original_index(
        self, table: tuple[int, int, torch.Tensor]
    ) -> None:
        num_experts, _, index = table
        ids = index.reshape(-1)
        perm, _, offsets = ref.stable_counting_sort(ids, num_experts)
        start = 0
        for expert in range(num_experts):
            end = int(offsets[expert])
            rows = perm[start:end]
            assert (ids[rows] == expert).all()
            assert torch.equal(rows, rows.sort().values)
            start = end


class TestGather:
    pytestmark = pytest.mark.numerical_unit

    @given(
        routing_tables(max_tokens=12),
        st.sampled_from(WIDTHS),
        st.sampled_from([torch.bfloat16, torch.float32]),
    )
    @settings(max_examples=60, deadline=None)
    def test_the_backward_fold_is_the_cpu_index_put_s(
        self, table: tuple[int, int, torch.Tensor], width: int, dtype: torch.dtype
    ) -> None:
        """The CPU ``index_put_(accumulate=True)`` folds duplicates in array
        order with a rounding after each addition — the same fold as CUDA's
        wide kernel, so the ``per_step`` reference is pinned here exactly."""
        _, top_k, index = table
        tokens = index.shape[0]
        perm, inv_perm = _perm(index)
        torch.manual_seed(width)
        hidden = (torch.randn(tokens, width) * 8).to(dtype).requires_grad_(True)
        out = ref.gather_rows(hidden, perm, top_k)
        assert torch.equal(out, hidden.detach()[perm // top_k])
        grad = (torch.randn_like(out) * 8).to(dtype)
        (autograd,) = torch.autograd.grad(out, hidden, grad)
        ours = ref.gather_rows_backward(grad, inv_perm, top_k, "per_step")
        assert ours.dtype is dtype and torch.equal(ours, autograd)

    def test_the_two_roundings_differ_where_bf16_drops_a_unit(self) -> None:
        """Why the width decides: eight rows of ``256, 1, 1, …`` sum to 263
        in fp32 (rounded once → 264 in bf16) but stay at 256 when every
        partial sum is rounded to bf16."""
        top_k, width = 8, 4
        grad = torch.ones(top_k, width, dtype=torch.bfloat16)
        grad[0] = 256.0
        inv_perm = torch.arange(top_k)
        once = ref.gather_rows_backward(grad, inv_perm, top_k, "once")
        step = ref.gather_rows_backward(grad, inv_perm, top_k, "per_step")
        assert torch.equal(once, torch.full((1, width), 264.0, dtype=torch.bfloat16))
        assert torch.equal(step, torch.full((1, width), 256.0, dtype=torch.bfloat16))

    def test_the_fold_order_is_ascending_array_position(self) -> None:
        """A token whose slots sit at array positions 5, 0, 3 folds them
        as rows 0, 3, 5 — where they appear in ``perm``, not by slot."""
        top_k = 3
        grad = torch.zeros(6, 2, dtype=torch.bfloat16)
        grad[0] = 256.0
        grad[3] = 1.0
        grad[5] = 1.0
        # token 0 owns positions (5, 0, 3); token 1 the rest
        inv_perm = torch.tensor([5, 0, 3, 1, 2, 4])
        out = ref.gather_rows_backward(grad, inv_perm, top_k, "per_step")
        assert torch.equal(out[0], torch.tensor([256.0, 256.0], dtype=torch.bfloat16))
        grad[0], grad[5] = (
            1.0,
            256.0,
        )  # now 1 + 1 + 256: 258 rounds to 258 → 258 in bf16
        out = ref.gather_rows_backward(grad, inv_perm, top_k, "per_step")
        assert torch.equal(out[0], torch.tensor([258.0, 258.0], dtype=torch.bfloat16))

    def test_the_rounding_is_chosen_by_width(self) -> None:
        assert ref.index_backward_rounding(8) == "once"
        assert ref.index_backward_rounding(32) == "once"
        assert ref.index_backward_rounding(33) == "per_step"
        assert ref.index_backward_rounding(2048) == "per_step"


class TestSlotSum:
    pytestmark = pytest.mark.numerical_unit

    def test_eight_slots_land_in_four_interleaved_accumulators(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(5, 8, 7) * 1e3
        want = (((x[:, 0] + x[:, 4]) + (x[:, 1] + x[:, 5])) + (x[:, 2] + x[:, 6])) + (
            x[:, 3] + x[:, 7]
        )
        assert torch.equal(ref.slot_sum_cuda_order(x), want)

    def test_mutation_the_sequential_order_is_a_different_number(self) -> None:
        x = torch.tensor([[[1e8], [1.0], [1.0], [1.0], [-1e8], [1.0], [1.0], [1.0]]])
        sequential = torch.zeros(1, 1)
        for slot in range(8):
            sequential = sequential + x[:, slot]
        assert not torch.equal(ref.slot_sum_cuda_order(x), sequential)
        assert torch.equal(ref.slot_sum_cuda_order(x), torch.tensor([[6.0]]))

    @given(st.integers(1, 12), st.sampled_from([torch.bfloat16, torch.float32]))
    @settings(max_examples=30, deadline=None)
    def test_it_is_the_sum_and_rounds_once(
        self, top_k: int, dtype: torch.dtype
    ) -> None:
        torch.manual_seed(top_k)
        x = torch.randn(9, top_k, 16).to(dtype)
        out = ref.slot_sum_cuda_order(x)
        assert out.dtype is dtype
        assert torch.allclose(out.double(), x.double().sum(1), rtol=1e-2, atol=1e-2)
        if dtype is torch.float32:
            assert torch.allclose(out.double(), x.double().sum(1), rtol=1e-5, atol=1e-4)


class TestRowSum:
    pytestmark = pytest.mark.numerical_unit

    def test_the_workflow_s_launch_is_32_vectorized_lanes(self) -> None:
        for rows in (16, 70, 4368, 9984, 93600):
            assert ref.row_sum_config(rows, 2048) == ref.RowSumConfig(lanes=32, vec=4)

    def test_narrow_rows_are_unvectorized_and_short_batches_widen_the_block(
        self,
    ) -> None:
        assert ref.row_sum_config(70, 8) == ref.RowSumConfig(lanes=8, vec=1)
        assert ref.row_sum_config(70, 128) == ref.RowSumConfig(lanes=32, vec=1)
        assert ref.row_sum_config(1, 2048) == ref.RowSumConfig(lanes=512, vec=4)
        assert ref.row_sum_config(70, 136) == ref.RowSumConfig(lanes=32, vec=4)

    def test_what_is_not_modelled_is_refused_by_name(self) -> None:
        with pytest.raises(ref.UnsupportedReduction, match="vector tail"):
            ref.row_sum_config(70, 130)
        with pytest.raises(
            ref.UnsupportedReduction, match="split across warps"
        ) as info:
            ref.row_sum_config(4368, 8192)
        assert (info.value.rows, info.value.width) == (4368, 8192)

    def test_the_order_written_out_for_one_wide_row(self) -> None:
        """Width 256 over 16 rows: 32 lanes × 4-vectors × 2 steps. Lane
        ``l`` holds vectors ``l`` and ``l + 32``; its four accumulators are
        combined left to right; the lanes pair up (0,1), (2,3), … five
        times."""
        torch.manual_seed(1)
        x = torch.randn(16, 256) * 1e3
        assert ref.row_sum_config(16, 256) == ref.RowSumConfig(lanes=32, vec=4)
        v = x.view(16, 64, 4)
        acc = v[:, :32] + v[:, 32:]  # (16, 32 lanes, 4): step 0 then step 1
        lane = ((acc[..., 0] + acc[..., 1]) + acc[..., 2]) + acc[..., 3]
        while lane.shape[1] > 1:
            lane = lane[:, 0::2] + lane[:, 1::2]
        assert torch.equal(ref.row_sum_cuda_order(x), lane[:, 0])

    def test_a_short_batch_widens_the_block_and_changes_the_order(self) -> None:
        """Three rows: the block has 64 lanes, one vector each, halved
        through shared memory before the warp tree — the vectors are
        combined before lanes ``l`` and ``l + 32`` meet, so the number is
        not the 32-lane one."""
        torch.manual_seed(1)
        x = torch.randn(3, 256) * 1e3
        assert ref.row_sum_config(3, 256) == ref.RowSumConfig(lanes=64, vec=4)
        v = x.view(3, 64, 4)
        lane = ((v[..., 0] + v[..., 1]) + v[..., 2]) + v[..., 3]  # (3, 64)
        lane = lane[:, :32] + lane[:, 32:]
        while lane.shape[1] > 1:
            lane = lane[:, 0::2] + lane[:, 1::2]
        assert torch.equal(ref.row_sum_cuda_order(x), lane[:, 0])
        assert not torch.equal(
            ref.row_sum_cuda_order(x), ref.row_sum_cuda_order(x.repeat(6, 1))[:3]
        )

    def test_mutation_the_sequential_order_is_a_different_number(self) -> None:
        x = torch.zeros(16, 256)  # 16 rows: the 32-lane block
        x[:, 0], x[:, 1], x[:, 128] = 1e8, 1.0, -1e8
        exact = x.sum(-1, dtype=torch.float64).float()
        out = ref.row_sum_cuda_order(x)
        # lane 0 holds 1e8 (step 0) and −1e8 (step 1) in accumulator 0: they
        # cancel exactly, then the 1.0 in accumulator 1 survives
        assert torch.equal(out, torch.ones(16))
        assert torch.equal(exact, torch.ones(16))
        naive = torch.zeros(16)
        for column in range(256):
            naive = naive + x[:, column]
        assert torch.equal(naive, torch.zeros(16))  # 1e8 + 1 lost the unit first
        assert not torch.equal(naive, out)

    @given(st.sampled_from([8, 16, 64, 128, 136, 256, 2048]), st.integers(1, 40))
    @settings(max_examples=30, deadline=None)
    def test_it_is_the_sum(self, width: int, rows: int) -> None:
        torch.manual_seed(width + rows)
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(rows, width).to(dtype)
            out = ref.row_sum_cuda_order(x)
            assert out.dtype is dtype and out.shape == (rows,)
            tolerance = 1e-5 if dtype is torch.float32 else 2e-2
            assert torch.allclose(
                out.double(), x.double().sum(-1), rtol=tolerance, atol=tolerance
            )


def _library_epilogue(
    proj: torch.Tensor,
    weights: torch.Tensor,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """The library's lines, ops for ops."""
    weighted = proj * weights[perm].unsqueeze(-1)
    weighted = weighted[inv_perm]
    tokens = perm.numel() // top_k
    return weighted.view(tokens, top_k, proj.shape[-1]).sum(dim=1).to(proj.dtype)


class TestEpilogue:
    pytestmark = pytest.mark.numerical_unit

    @given(
        routing_tables(max_tokens=12), st.sampled_from([torch.bfloat16, torch.float32])
    )
    @settings(max_examples=40, deadline=None)
    def test_forward_and_backward_against_the_library_lines(
        self, table: tuple[int, int, torch.Tensor], dtype: torch.dtype
    ) -> None:
        """The products and ``d proj_out`` are single roundings and match to
        the bit on any device; the two reductions match within tolerance
        here (the CPU sums in another order) and to the bit on CUDA."""
        _, top_k, index = table
        tokens = index.shape[0]
        width = 256
        perm, inv_perm = _perm(index)
        torch.manual_seed(tokens)
        proj = torch.randn(tokens * top_k, width).to(dtype).requires_grad_(True)
        weights = torch.rand(tokens * top_k).to(dtype).requires_grad_(True)
        theirs = _library_epilogue(proj, weights, perm, inv_perm, top_k)
        ours = ref.epilogue_forward(
            proj.detach(), weights.detach(), inv_perm, top_k, dtype
        )
        assert ours.dtype is dtype and ours.shape == theirs.shape
        assert torch.allclose(ours.float(), theirs.float(), rtol=2e-2, atol=2e-2)
        grad = torch.randn_like(theirs)
        d_proj, d_weights = torch.autograd.grad(theirs, (proj, weights), grad)
        ours_proj, ours_weights = ref.epilogue_backward(
            grad, proj.detach(), weights.detach(), perm, top_k
        )
        assert torch.equal(ours_proj, d_proj)
        assert ours_weights.dtype is dtype
        assert torch.allclose(
            ours_weights.float(), d_weights.float(), rtol=2e-2, atol=2e-2
        )

    def test_the_forward_is_the_slot_sum_of_rounded_products(self) -> None:
        torch.manual_seed(3)
        tokens, top_k, width = 5, 8, 256
        index = torch.randint(0, 16, (tokens, top_k))
        perm, inv_perm = _perm(index)
        proj = torch.randn(tokens * top_k, width).to(torch.bfloat16)
        weights = torch.rand(tokens * top_k).to(torch.bfloat16)
        products = (proj[inv_perm].float() * weights.float().unsqueeze(-1)).to(
            torch.bfloat16
        )
        want = ref.slot_sum_cuda_order(products.view(tokens, top_k, width))
        assert torch.equal(
            ref.epilogue_forward(proj, weights, inv_perm, top_k, torch.bfloat16), want
        )

    def test_the_weight_gradient_is_the_row_sum_of_rounded_products(self) -> None:
        torch.manual_seed(4)
        tokens, top_k, width = 5, 8, 256
        index = torch.randint(0, 16, (tokens, top_k))
        perm, inv_perm = _perm(index)
        proj = torch.randn(tokens * top_k, width).to(torch.bfloat16)
        weights = torch.rand(tokens * top_k).to(torch.bfloat16)
        grad = torch.randn(tokens, width).to(torch.bfloat16)
        _, d_weights = ref.epilogue_backward(grad, proj, weights, perm, top_k)
        products = (grad.float()[perm // top_k] * proj.float()).to(torch.bfloat16)
        want = torch.empty_like(weights)
        want[perm] = ref.row_sum_cuda_order(products)
        assert torch.equal(d_weights, want)


class TestSiluMul:
    """The pure-torch gate formulas against the CPU kernels, within a couple
    of ulps: everything downstream of ``exp`` is vendor-specific on the CPU
    (Sleef-vectorized on x86, another libm on arm64), so even the saved
    activation — and with it the ``d up`` product — can differ from ATen's
    vectorized ``silu`` by an ulp before any rounding here; on CUDA ATen's
    ``silu_backward`` is moreover the fma-contracted form this reference
    cannot write. Bit-level parity of the gate is established for the
    **Triton kernel** by ``tests/golden/test_moe_glue_kernels.py`` on the
    H100, not by this formula; ``torch.equal`` is reserved for the pure
    add/mul ordering claims above (slot sum, row sum, index fold)."""

    pytestmark = pytest.mark.numerical_unit

    #: a couple of ulps of the dtype, relative and absolute
    TOLERANCE = {torch.bfloat16: (2e-2, 1e-2), torch.float32: (1e-5, 1e-5)}

    def test_forward_matches_the_library_gate(self) -> None:
        torch.manual_seed(5)
        for dtype, (rtol, atol) in self.TOLERANCE.items():
            gate_up = (torch.randn(64, 96, device="cpu") * 4).to(dtype)
            gate, up = gate_up.chunk(2, dim=-1)
            theirs = F.silu(gate) * up
            ours = ref.silu_mul_forward(gate_up)
            assert ours.dtype is dtype and ours.shape == theirs.shape
            assert torch.allclose(ours.float(), theirs.float(), rtol=rtol, atol=atol)

    def test_backward_matches_autograd(self) -> None:
        torch.manual_seed(6)
        for dtype, (rtol, atol) in self.TOLERANCE.items():
            gate_up = (
                (torch.randn(64, 96, device="cpu") * 4).to(dtype).requires_grad_(True)
            )
            gate, up = gate_up.chunk(2, dim=-1)
            out = F.silu(gate) * up
            grad = torch.randn_like(out)
            (theirs,) = torch.autograd.grad(out, gate_up, grad)
            ours = ref.silu_mul_backward(grad, gate_up.detach())
            assert ours.shape == gate_up.shape and ours.dtype is dtype
            assert torch.allclose(ours.float(), theirs.float(), rtol=rtol, atol=atol)
