"""The fused MoE glue kernels on the device, held to the bit
(``pytorch_hooks/kernels/``): the accelerator half of
``tests/neural/engines/pytorch_hooks/kernels/``.

Three layers, each ``torch.equal``:

1. **the ATen orders hold on this device** — the pure-torch references of
   ``moe_glue_reference.py`` (the stable sort, the index backward's fold and
   rounding, the slot sum's four accumulators, the vectorized row sum's
   lanes and tree) against the library ops they describe, on the workflow's
   shapes and small odd ones, in bf16 / fp16 / fp32;
2. **each kernel is its reference** — forward and backward, same shapes;
3. **the fused experts path is the library function** — output, ``d
   hidden_states``, ``d top_k_weights``, ``d gate_up_proj``, ``d down_proj``
   — with the default kernel set, and with the gate kernel on; and once
   more captured in a CUDA graph and replayed.

A failure prints the count of differing elements and the largest gap; the
gate test also reports whether the other multiply-add form would have
matched, so one run settles ``GATE_FMA``.
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch
import transformers.integrations.moe as moe

from causalab.neural.engines.pytorch_hooks.experts_path import lean_grouped_mm_forward
from causalab.neural.engines.pytorch_hooks.kernels import moe_glue
from causalab.neural.engines.pytorch_hooks.kernels import moe_glue_reference as ref
from causalab.neural.engines.pytorch_hooks.kernels import moe_glue_triton as kernels
from causalab.neural.shared.kernel_options import ENV_MOE_GLUE

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not kernels.available(), reason="requires Triton"),
]

DEVICE = "cuda"
DTYPES = (torch.bfloat16, torch.float16, torch.float32)

#: (tokens, top_k, hidden, intermediate, experts): the workflow's training
#: batches (S = 9984 and 4368), its eval batch (S = 93 600), and small odd
#: ones — the tiny fixture's dims (H = 8: the small-stride index backward,
#: an 8-lane row sum the epilogue leaves to ATen), a 136-wide hidden with
#: top_k 3 (a vector count not divisible by the lanes), and 20 pairs (below
#: the bitonic-sort bound, so the sort stays with ``torch.sort``).
SHAPES: tuple[tuple[int, int, int, int, int], ...] = (
    (1248, 8, 2048, 512, 256),
    (546, 8, 2048, 512, 256),
    (11700, 8, 2048, 512, 256),
    (7, 10, 8, 32, 128),
    (37, 3, 136, 64, 17),
    (5, 4, 256, 96, 6),
)
SMALL = SHAPES[3:]


def _assert_equal(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> None:
    assert ours.shape == theirs.shape, f"{name}: shape {ours.shape} vs {theirs.shape}"
    assert ours.dtype == theirs.dtype, f"{name}: dtype {ours.dtype} vs {theirs.dtype}"
    if torch.equal(ours, theirs):
        return
    diff = (ours.float() - theirs.float()).abs()
    where = torch.nonzero(ours != theirs)
    raise AssertionError(
        f"{name}: {int((ours != theirs).sum())} of {ours.numel()} elements differ, "
        f"max |Δ| = {float(diff.max()):.3e}, first at {where[0].tolist()}: "
        f"{ours[tuple(where[0])].item()!r} vs {theirs[tuple(where[0])].item()!r}"
    )


def routing(
    tokens: int, top_k: int, num_experts: int, seed: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k ids and normalized weights the way the router shapes them,
    with a skewed expert distribution."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    logits = torch.randn(tokens, num_experts, device=DEVICE, generator=generator)
    logits[:, : max(1, num_experts // 8)] += 2.0  # a few hot experts
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    values, index = torch.topk(probs, top_k, dim=-1)
    values = values / values.sum(dim=-1, keepdim=True)
    return index, values.to(dtype)


def synthetic_experts(
    hidden: int, intermediate: int, num_experts: int, dtype: torch.dtype, seed: int = 0
) -> Any:
    """An experts module as transformers' decorator shapes it: fused
    ``[gate | up]`` and ``down`` weights, the default silu gate, no bias."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    module = torch.nn.Module()
    module.num_experts = num_experts
    module.hidden_dim = hidden
    module.intermediate_dim = intermediate
    module.has_gate = True
    module.has_bias = False
    module.is_transposed = False
    module.act_fn = torch.nn.SiLU()
    module.config = SimpleNamespace()
    scale = hidden**-0.5
    module.gate_up_proj = torch.nn.Parameter(
        (
            torch.randn(
                num_experts,
                2 * intermediate,
                hidden,
                device=DEVICE,
                generator=generator,
            )
            * scale
        ).to(dtype)
    )
    module.down_proj = torch.nn.Parameter(
        (
            torch.randn(
                num_experts, hidden, intermediate, device=DEVICE, generator=generator
            )
            * scale
        ).to(dtype)
    )
    module._apply_gate = types.MethodType(moe._default_apply_gate, module)
    return module


def _loss(out: torch.Tensor) -> torch.Tensor:
    """A downstream that weights every element differently — the same one
    for the eager reference and the captured step."""
    scale = torch.linspace(
        0.5, 1.5, out.numel(), device=DEVICE, dtype=torch.float32
    ).view(out.shape)
    return (out.float() * scale).sum()


def run_experts(
    fn: Any,
    experts: Any,
    hidden: torch.Tensor,
    index: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Output and every gradient: hidden states, routing weights, both
    expert weights — under a downstream that weights each element
    differently."""
    source = hidden.clone().requires_grad_(True)
    w = weights.clone().requires_grad_(True)
    out = fn(experts, source, index, w)
    grads = torch.autograd.grad(
        _loss(out), (source, w, experts.gate_up_proj, experts.down_proj)
    )
    names = ("output", "d_hidden", "d_top_k_weights", "d_gate_up_proj", "d_down_proj")
    return dict(zip(names, (out.detach(), *(g.detach() for g in grads))))


@pytest.fixture(autouse=True)
def _default_options(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(ENV_MOE_GLUE, raising=False)
    yield


class TestTheAtenOrdersHoldOnThisDevice:
    @pytest.mark.parametrize("num_pairs", [33, 70, 111, 129, 4096, 4368, 9984, 93600])
    def test_torch_sort_is_stable_above_32_keys(self, num_pairs: int) -> None:
        ids = torch.randint(0, 256, (num_pairs,), device=DEVICE)
        _, perm = torch.sort(ids)
        _assert_equal("perm", perm, torch.sort(ids, stable=True).indices)
        ours, inv, offsets = ref.stable_counting_sort(ids, 256)
        _assert_equal("counting perm", ours, perm)
        counts = torch.histc(ids.int(), bins=256, min=0, max=255)
        _assert_equal("offsets", offsets, torch.cumsum(counts, 0, dtype=torch.int32))

    def test_torch_sort_is_not_stable_at_32_keys(self) -> None:
        """The bound the sort plan rests on: below it the plan keeps
        ``torch.sort``, whatever order it takes."""
        torch.manual_seed(0)
        unstable = 0
        for _ in range(50):
            ids = torch.randint(0, 4, (32,), device=DEVICE)
            unstable += int(
                not torch.equal(torch.sort(ids)[1], torch.sort(ids, stable=True)[1])
            )
        assert unstable > 0, (
            "the bitonic sort happened to be stable 50 times; revisit the bound"
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_the_index_backward_fold(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        tokens, top_k, hidden, _, num_experts = shape
        index, _ = routing(tokens, top_k, num_experts, 1, dtype)
        perm = torch.sort(index.reshape(-1), stable=True).indices
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=DEVICE)
        source = (
            (torch.randn(tokens, hidden, device=DEVICE) * 8)
            .to(dtype)
            .requires_grad_(True)
        )
        out = source[perm // top_k]
        grad = (torch.randn_like(out) * 8).to(dtype)
        (theirs,) = torch.autograd.grad(out, source, grad)
        rounding = ref.index_backward_rounding(hidden)
        _assert_equal(
            f"index backward ({rounding})",
            ref.gather_rows_backward(grad, inv, top_k, rounding),
            theirs,
        )
        other = "once" if rounding == "per_step" else "per_step"
        mismatch = ref.gather_rows_backward(grad, inv, top_k, other)
        if dtype is not torch.float32 and theirs.numel() >= 4096:
            assert not torch.equal(mismatch, theirs), (
                f"both roundings match at width {hidden}"
            )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_the_slot_sum_order(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        tokens, top_k, hidden, _, _ = shape
        x = (torch.randn(tokens, top_k, hidden, device=DEVICE) * 4).to(dtype)
        _assert_equal("slot sum", ref.slot_sum_cuda_order(x), x.sum(dim=1))

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_the_row_sum_order(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        tokens, top_k, hidden, _, _ = shape
        rows = tokens * top_k
        x = (torch.randn(rows, hidden, device=DEVICE) * 4).to(dtype)
        try:
            ours = ref.row_sum_cuda_order(x)
        except ref.UnsupportedReduction as refused:
            pytest.skip(str(refused))
        _assert_equal("row sum", ours, x.sum(dim=-1))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_the_gate_formulas(self, dtype: torch.dtype) -> None:
        """The reference formulas against ATen's silu on the device: the
        forward and the ``d up`` half exactly; ``d gate`` exactly in bf16 and
        within an ulp otherwise — nvcc contracts ``1 + x · (1 − s)`` into an
        fma plain torch cannot write (the kernel does, and its test below
        carries the bit claim)."""
        gate_up = (
            (torch.randn(4096, 1024, device=DEVICE) * 4).to(dtype).requires_grad_(True)
        )
        gate, up = gate_up.chunk(2, dim=-1)
        out = torch.nn.functional.silu(gate) * up
        _assert_equal("silu·up", ref.silu_mul_forward(gate_up.detach()), out.detach())
        grad = torch.randn_like(out)
        (theirs,) = torch.autograd.grad(out, gate_up, grad)
        ours = ref.silu_mul_backward(grad, gate_up.detach())
        inner = out.shape[-1]
        _assert_equal("d up", ours[:, inner:], theirs[:, inner:])
        if dtype is torch.bfloat16:
            _assert_equal("d gate", ours[:, :inner], theirs[:, :inner])
        else:
            assert torch.allclose(
                ours[:, :inner].float(), theirs[:, :inner].float(), rtol=1e-3, atol=1e-4
            )


class TestEachKernelIsItsReference:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_counting_sort(self, shape: tuple[int, ...]) -> None:
        tokens, top_k, _, _, num_experts = shape
        index, _ = routing(tokens, top_k, num_experts, 2, torch.bfloat16)
        ids = index.reshape(-1)
        want = ref.stable_counting_sort(ids, num_experts)
        got = kernels.counting_sort(ids, num_experts)
        for name, a, b in zip(("perm", "inv_perm", "offsets"), got, want):
            _assert_equal(name, a, b)
        got32 = kernels.counting_sort(ids.int(), num_experts)
        for name, a, b in zip(("perm", "inv_perm", "offsets"), got32, want):
            _assert_equal(f"{name} (int32 ids)", a, b)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_gather_forward_and_backward(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        tokens, top_k, hidden, _, num_experts = shape
        index, _ = routing(tokens, top_k, num_experts, 3, dtype)
        perm, inv, _ = ref.stable_counting_sort(index.reshape(-1), num_experts)
        source = (torch.randn(tokens, hidden, device=DEVICE) * 8).to(dtype)
        _assert_equal(
            "gather",
            kernels.gather_rows(source, perm, top_k),
            ref.gather_rows(source, perm, top_k),
        )
        grad = (torch.randn(tokens * top_k, hidden, device=DEVICE) * 8).to(dtype)
        for rounding in ("per_step", "once"):
            _assert_equal(
                f"gather backward ({rounding})",
                kernels.gather_rows_backward(grad, inv, top_k, rounding),
                ref.gather_rows_backward(grad, inv, top_k, rounding),
            )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_epilogue_forward_and_backward(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        tokens, top_k, hidden, _, num_experts = shape
        if not moe_glue._epilogue_admits(tokens * top_k, hidden):
            pytest.skip("the epilogue leaves this row sum to ATen")
        index, weights = routing(tokens, top_k, num_experts, 4, dtype)
        perm, inv, _ = ref.stable_counting_sort(index.reshape(-1), num_experts)
        proj = (torch.randn(tokens * top_k, hidden, device=DEVICE) * 4).to(dtype)
        w = weights.reshape(-1)
        _assert_equal(
            "epilogue",
            kernels.epilogue_forward(proj, w, inv, top_k, dtype),
            ref.epilogue_forward(proj, w, inv, top_k, dtype),
        )
        grad = (torch.randn(tokens, hidden, device=DEVICE) * 4).to(dtype)
        d_proj, d_w = kernels.epilogue_backward(grad, proj, w, perm, top_k)
        want_proj, want_w = ref.epilogue_backward(grad, proj, w, perm, top_k)
        _assert_equal("d proj_out", d_proj, want_proj)
        _assert_equal("d weights", d_w, want_w)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_gate_forward_and_backward(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        tokens, top_k, _, intermediate, _ = shape
        gate_up = (
            (torch.randn(tokens * top_k, 2 * intermediate, device=DEVICE) * 4)
            .to(dtype)
            .requires_grad_(True)
        )
        gate, up = gate_up.chunk(2, dim=-1)
        out = torch.nn.functional.silu(gate) * up
        _assert_equal(
            "silu·up", kernels.silu_mul_forward(gate_up.detach()), out.detach()
        )
        grad = torch.randn_like(out)
        (theirs,) = torch.autograd.grad(out, gate_up, grad)
        report = {}
        for fma in (True, False):
            ours = kernels.silu_mul_backward(grad, gate_up.detach(), fma)
            report[fma] = int((ours != theirs).sum())
        chosen = kernels.silu_mul_backward(grad, gate_up.detach(), moe_glue.GATE_FMA)
        assert torch.equal(chosen, theirs), (
            f"d gate_up with GATE_FMA={moe_glue.GATE_FMA}: mismatches by form {report} "
            f"of {theirs.numel()} elements"
        )


def _library(
    experts: Any, hidden: torch.Tensor, index: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return moe.grouped_mm_experts_forward(experts, hidden, index, weights)


class TestTheFusedPathIsTheLibrary:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_with_the_default_kernels(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        tokens, top_k, hidden, intermediate, num_experts = shape
        experts = synthetic_experts(hidden, intermediate, num_experts, dtype)
        index, weights = routing(tokens, top_k, num_experts, 5, dtype)
        source = torch.randn(tokens, hidden, device=DEVICE).to(dtype)
        theirs = run_experts(_library, experts, source, index, weights)
        ours = run_experts(lean_grouped_mm_forward, experts, source, index, weights)
        for name in theirs:
            _assert_equal(name, ours[name], theirs[name])
            assert torch.isfinite(ours[name]).all(), name

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_with_the_gate_kernel_too(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ENV_MOE_GLUE, "all")
        tokens, top_k, hidden, intermediate, num_experts = shape
        experts = synthetic_experts(hidden, intermediate, num_experts, dtype)
        index, weights = routing(tokens, top_k, num_experts, 6, dtype)
        source = torch.randn(tokens, hidden, device=DEVICE).to(dtype)
        theirs = run_experts(_library, experts, source, index, weights)
        ours = run_experts(lean_grouped_mm_forward, experts, source, index, weights)
        for name in theirs:
            _assert_equal(name, ours[name], theirs[name])

    def test_the_plan_on_the_workflow_shape_runs_every_default_kernel(self) -> None:
        tokens, top_k, hidden, intermediate, num_experts = SHAPES[0]
        experts = synthetic_experts(hidden, intermediate, num_experts, torch.bfloat16)
        index, weights = routing(tokens, top_k, num_experts, 7, torch.bfloat16)
        source = torch.randn(tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
        seen: list[moe_glue.GluePlan] = []
        real = moe_glue.plan_moe_glue

        def spy(**kwargs: Any) -> moe_glue.GluePlan:
            plan = real(**kwargs)
            seen.append(plan)
            return plan

        import causalab.neural.engines.pytorch_hooks.experts_path as experts_path

        original = experts_path.moe_glue.plan_moe_glue
        experts_path.moe_glue.plan_moe_glue = spy
        try:
            with torch.no_grad():
                lean_grouped_mm_forward(experts, source, index, weights)
        finally:
            experts_path.moe_glue.plan_moe_glue = original
        assert seen == [
            moe_glue.GluePlan(sort=True, gather=True, epilogue=True, gate=True)
        ]

    def test_captured_and_replayed(self) -> None:
        """The fused path inside a CUDA graph: warmed up on a side stream,
        captured, replayed twice with new inputs — the replay's numbers are
        the eager ones (no host sync, no stale value)."""
        tokens, top_k, hidden, intermediate, num_experts = SHAPES[1]
        dtype = torch.bfloat16
        experts = synthetic_experts(hidden, intermediate, num_experts, dtype)
        index, weights = routing(tokens, top_k, num_experts, 8, dtype)
        static_hidden = torch.randn(
            tokens, hidden, device=DEVICE, dtype=dtype, requires_grad=True
        )
        static_weights = weights.clone().requires_grad_(True)
        static_index = index.clone()

        def step() -> torch.Tensor:
            out = lean_grouped_mm_forward(
                experts, static_hidden, static_index, static_weights
            )
            _loss(out).backward()
            return out

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                step()
                experts.gate_up_proj.grad = None
                experts.down_proj.grad = None
                static_hidden.grad = None
                static_weights.grad = None
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        for seed in (21, 22):
            new_index, new_weights = routing(tokens, top_k, num_experts, seed, dtype)
            new_hidden = torch.randn(tokens, hidden, device=DEVICE, dtype=dtype)
            static_index.copy_(new_index)
            static_weights.data.copy_(new_weights)
            static_hidden.data.copy_(new_hidden)
            graph.replay()
            torch.cuda.synchronize()
            replayed = {
                "output": captured.detach().clone(),
                "d_hidden": static_hidden.grad.clone(),
                "d_top_k_weights": static_weights.grad.clone(),
                "d_gate_up_proj": experts.gate_up_proj.grad.clone(),
                "d_down_proj": experts.down_proj.grad.clone(),
            }
            eager = run_experts(_library, experts, new_hidden, new_index, new_weights)
            for name in eager:
                _assert_equal(f"replay {seed}: {name}", replayed[name], eager[name])
