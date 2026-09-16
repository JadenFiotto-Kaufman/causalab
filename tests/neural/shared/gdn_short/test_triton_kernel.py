"""The Triton single-chunk kernel against the float32 references — on a CUDA
device; skipped cleanly without one (``uv run pytest
tests/neural/shared/gdn_short -m cuda`` on a GPU node).

Tolerances, and why:

* **float32 inputs** — the kernel is float32 throughout (3×TF32 dots) and
  rounds only on store, so against ``single_chunk_gated_delta_rule_torch`` (same math,
  cuBLAS order) it is float32 ulps apart: ``atol=1e-4, rtol=1e-4`` on
  values, ``rtol=1e-3`` on gradients (as ``test_reference.py`` justifies).
* **bf16 inputs** — the kernel's only bf16 rounding is the final store of
  ``o`` and of the gradients, so against the float32 oracle *on the same
  bf16-valued inputs* the error is one bf16 rounding of the result:
  ``|Δ| ≤ 2^-8 · max|o|`` ≈ 0.4 %; the bound is 2 %. FLA's own error is
  reported beside it when FLA is importable, and the kernel is required to
  be no farther from the oracle than FLA is — FLA rounds its normalized
  q̂/k̂, ``A``, ``u`` and the chunk state to bf16 mid-way.
* **gradcheck** — finite differences in float32 on a tiny problem, with the
  step and tolerances loosened to what float32 differences can resolve
  (``eps=1e-2`` central differences: truncation ~1e-4, rounding ~1e-4).
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch

from causalab.neural.shared.gdn_short.binding import short_seq_dispatcher
from causalab.neural.shared.gdn_short.reference import (
    recurrent_gated_delta_rule_reference,
    single_chunk_gated_delta_rule_torch,
)
from causalab.neural.shared.gdn_short.options import ShortSeqKernelOptions
from causalab.neural.shared.gdn_short.triton_kernel import (
    ShortSeqUnsupported,
    single_chunk_gated_delta_rule,
)

pytestmark = [
    pytest.mark.numerical_unit,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

VALUE_TOL = dict(atol=1e-4, rtol=1e-4)
GRAD_TOL = dict(atol=1e-4, rtol=1e-3)
BF16_REL = 0.02

NAMES = ("q", "k", "v", "g", "beta")


def _inputs(
    seed: int,
    b: int,
    t: int,
    h: int,
    hv: int,
    dk: int,
    dv: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, h, dk, generator=gen).to(dtype)
    k = torch.randn(b, t, h, dk, generator=gen).to(dtype)
    v = torch.randn(b, t, hv, dv, generator=gen).to(dtype)
    # the mixer's g is float32 whatever the model dtype; beta is model dtype
    g = -torch.nn.functional.softplus(torch.randn(b, t, hv, generator=gen))
    beta = torch.rand(b, t, hv, generator=gen).to(dtype)
    return tuple(x.cuda() for x in (q, k, v, g, beta))


def _grads(
    fn: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    w_o: torch.Tensor,
    w_s: torch.Tensor | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    leaves = [x.clone().requires_grad_(True) for x in inputs]
    o, s = fn(*leaves, output_final_state=w_s is not None)
    loss = (o.float() * w_o).sum()
    if w_s is not None:
        loss = loss + (s.float() * w_s).sum()
    return o, torch.autograd.grad(loss, leaves)


def _kernel(*inputs: torch.Tensor, output_final_state: bool = False) -> Any:
    return single_chunk_gated_delta_rule(
        *inputs, output_final_state=output_final_state, use_qk_l2norm_in_kernel=True
    )


class TestFloat32Parity:
    @pytest.mark.parametrize("t", [1, 5, 13, 16, 17, 32])
    @pytest.mark.parametrize("heads", [(4, 4), (2, 4)], ids=["repeated", "grouped"])
    @pytest.mark.parametrize("dims", [(16, 16), (128, 128), (64, 128)])
    def test_forward_state_and_every_gradient(
        self, t: int, heads: tuple[int, int], dims: tuple[int, int]
    ) -> None:
        h, hv = heads
        dk, dv = dims
        inputs = _inputs(t * 7 + h, 3, t, h, hv, dk, dv)
        w_o = torch.randn(3, t, hv, dv, device="cuda")
        w_s = torch.randn(3, hv, dk, dv, device="cuda")
        o_ref, g_ref = _grads(single_chunk_gated_delta_rule_torch, inputs, w_o, w_s)
        o, grads = _grads(_kernel, inputs, w_o, w_s)
        assert o.dtype == torch.float32
        torch.testing.assert_close(o, o_ref, **VALUE_TOL)
        _, s = _kernel(*inputs, output_final_state=True)
        _, s_ref = single_chunk_gated_delta_rule_torch(*inputs, output_final_state=True)
        torch.testing.assert_close(s, s_ref, **VALUE_TOL)
        for name, a, b in zip(NAMES, grads, g_ref):
            assert a.shape == b.shape and a.dtype == b.dtype, name
            torch.testing.assert_close(a, b, msg=lambda m: f"d{name}: {m}", **GRAD_TOL)

    def test_the_output_alone_gives_the_same_gradients(self) -> None:
        """Without a final state the kernel's ``u`` output is ``None`` and no
        zero cotangent is materialized: the gradients are the closed form's."""
        inputs = _inputs(5, 2, 13, 4, 4, 128, 128)
        w_o = torch.randn(2, 13, 4, 128, device="cuda")
        _, g_ref = _grads(single_chunk_gated_delta_rule_torch, inputs, w_o, None)
        _, grads = _grads(_kernel, inputs, w_o, None)
        for name, a, b in zip(NAMES, grads, g_ref):
            torch.testing.assert_close(a, b, msg=lambda m: f"d{name}: {m}", **GRAD_TOL)

    def test_gradcheck_on_a_tiny_problem(self) -> None:
        inputs = _inputs(9, 1, 4, 1, 2, 16, 16)
        leaves = [x.clone().double().float().requires_grad_(True) for x in inputs]

        def fn(*xs: torch.Tensor) -> torch.Tensor:
            o, s = _kernel(*xs, output_final_state=True)
            return (o.sum(-1) + s.sum((-1, -2)).unsqueeze(1)).sum()

        assert torch.autograd.gradcheck(
            fn, leaves, eps=1e-2, atol=5e-2, rtol=5e-2, fast_mode=True
        )


class TestBf16Numerics:
    """The documented numerics of the swap: bf16 inputs as the model has
    them, the float32 oracle on the same values as the yardstick."""

    @pytest.mark.parametrize("t", [13, 16])
    def test_within_one_bf16_rounding_of_the_oracle(self, t: int) -> None:
        inputs = _inputs(21 + t, 4, t, 8, 8, 128, 128, dtype=torch.bfloat16)
        w_o = torch.randn(4, t, 8, 128, device="cuda")
        o_ref, g_ref = _grads(recurrent_gated_delta_rule_reference, inputs, w_o, None)
        o, grads = _grads(_kernel, inputs, w_o, None)
        assert o.dtype == torch.bfloat16
        errors = {"o": (o.float() - o_ref).abs().max() / o_ref.abs().max()}
        for name, a, b in zip(NAMES, grads, g_ref):
            assert a.dtype == inputs[NAMES.index(name)].dtype, name
            errors[f"d{name}"] = (a.float() - b).abs().max() / b.abs().max()
        print({k: f"{v.item():.2e}" for k, v in errors.items()})
        for name, error in errors.items():
            assert error.item() <= BF16_REL, (name, error.item())

    def test_no_farther_from_the_oracle_than_fla(self) -> None:
        fla = pytest.importorskip("fla.ops.gated_delta_rule")
        inputs = _inputs(77, 4, 13, 8, 8, 128, 128, dtype=torch.bfloat16)
        w_o = torch.randn(4, 13, 8, 128, device="cuda")
        _, g_ref = _grads(recurrent_gated_delta_rule_reference, inputs, w_o, None)
        o_ref, _ = recurrent_gated_delta_rule_reference(*inputs)

        def fla_call(*xs: torch.Tensor, output_final_state: bool = False) -> Any:
            return fla.chunk_gated_delta_rule(
                *xs, output_final_state=output_final_state, use_qk_l2norm_in_kernel=True
            )

        o_fla, g_fla = _grads(fla_call, inputs, w_o, None)
        o_ours, g_ours = _grads(_kernel, inputs, w_o, None)
        report: dict[str, tuple[float, float]] = {}
        for name, ref, a, b in zip(
            ("o", *[f"d{n}" for n in NAMES]),
            (o_ref, *g_ref),
            (o_ours, *g_ours),
            (o_fla, *g_fla),
        ):
            ours = (a.float() - ref).abs().max().item()
            theirs = (b.float() - ref).abs().max().item()
            report[name] = (ours, theirs)
        print("max abs error vs float32 oracle (ours, FLA):", report)
        for name, (ours, theirs) in report.items():
            assert ours <= theirs + 1e-3 * o_ref.abs().max().item(), (
                name,
                ours,
                theirs,
            )


class TestCaptureAndBinding:
    def test_replays_inside_a_cuda_graph(self) -> None:
        """Warm-up on a side stream, capture forward + backward, replay on new
        inputs copied into the captured storage: the replay equals eager."""
        static = [
            x.clone().requires_grad_(i < 5)
            for i, x in enumerate(_inputs(1, 2, 13, 4, 4, 128, 128))
        ]
        w_o = torch.randn(2, 13, 4, 128, device="cuda")

        def step() -> torch.Tensor:
            o, _ = _kernel(*static)
            grads = torch.autograd.grad((o * w_o).sum(), static)
            return torch.cat([o.flatten(), *[g.flatten() for g in grads]])

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        for seed in (2, 3):
            fresh = _inputs(seed, 2, 13, 4, 4, 128, 128)
            with torch.no_grad():
                for dst, src in zip(static, fresh):
                    dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, step(), **VALUE_TOL)

    def test_the_dispatcher_routes_real_cuda_tensors(self) -> None:
        options = ShortSeqKernelOptions(16)
        seen: list[str] = []

        def bound(
            q: Any, k: Any, v: Any, g: Any = None, beta: Any = None, **kw: Any
        ) -> Any:
            seen.append("bound")
            return single_chunk_gated_delta_rule_torch(q, k, v, g, beta)

        def short(q: Any, k: Any, v: Any, g: Any, beta: Any, **kw: Any) -> Any:
            seen.append("short")
            return single_chunk_gated_delta_rule(q, k, v, g, beta, **kw)

        dispatch = short_seq_dispatcher(bound, options, short)
        q, k, v, g, beta = _inputs(4, 2, 13, 4, 4, 128, 128)
        o, _ = dispatch(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        o_ref, _ = single_chunk_gated_delta_rule_torch(q, k, v, g, beta)
        torch.testing.assert_close(o, o_ref, **VALUE_TOL)
        q, k, v, g, beta = _inputs(4, 2, 17, 4, 4, 128, 128)
        dispatch(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        assert seen == ["short", "bound"]

    def test_direct_calls_outside_the_scope_are_refused_by_name(self) -> None:
        q, k, v, g, beta = _inputs(4, 1, 13, 4, 4, 128, 128)
        with pytest.raises(ShortSeqUnsupported, match="zero state"):
            single_chunk_gated_delta_rule(
                q, k, v, g, beta, initial_state=torch.zeros(1)
            )
        with pytest.raises(ShortSeqUnsupported, match="T <= 32"):
            long = _inputs(4, 1, 33, 4, 4, 128, 128)
            single_chunk_gated_delta_rule(*long)
        with pytest.raises(ShortSeqUnsupported, match="power-of-two K"):
            odd = _inputs(4, 1, 13, 4, 4, 96, 128)
            single_chunk_gated_delta_rule(*odd)
        with pytest.raises(ShortSeqUnsupported, match="CUDA"):
            single_chunk_gated_delta_rule(*(x.cpu() for x in (q, k, v, g, beta)))
