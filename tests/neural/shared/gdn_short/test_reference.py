"""The two float32 references of ``gdn_short/reference.py`` agree with each
other and with transformers' own torch chunk function — forward, final state
and every gradient — over random short sequences.

Tolerances: everything here is float32 on values of order one (``q̂``, ``k̂``
unit vectors, ``v`` standard normal, ``beta`` in ``(0, 1)``, ``g`` in
``(-3, 0]``). The three forms differ only in the order of a few hundred
float32 operations per output element, so their disagreement is a few
float32 ulps — measured at ≤ 4e-7 on outputs and ≤ 3e-6 on gradients of
magnitude up to 10 (``uv run python`` over the seeds below). The bounds
are 100× that: ``atol=1e-4`` with ``rtol=1e-4`` on values and ``1e-3`` on
gradients, which a wrong formula (a dropped term, a transposed factor, a
missing decay) exceeds by orders of magnitude.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.shared.gdn_short.reference import (
    expand_key_heads,
    recurrent_gated_delta_rule_reference,
    single_chunk_gated_delta_rule_torch,
)
from causalab.neural.shared.kernels import torch_implementation

pytestmark = pytest.mark.property

VALUE_TOL = dict(atol=1e-4, rtol=1e-4)
GRAD_TOL = dict(atol=1e-4, rtol=1e-3)


def _transformers_chunk() -> Any:
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

    return torch_implementation(modeling.torch_chunk_gated_delta_rule)


def _inputs(
    seed: int, b: int, t: int, h: int, group: int, dk: int, dv: int
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)
    hv = h * group
    q = torch.randn(b, t, h, dk, generator=gen)
    k = torch.randn(b, t, h, dk, generator=gen)
    v = torch.randn(b, t, hv, dv, generator=gen)
    g = -torch.nn.functional.softplus(torch.randn(b, t, hv, generator=gen))
    beta = torch.rand(b, t, hv, generator=gen)
    return q, k, v, g, beta


shapes = st.tuples(
    st.integers(0, 2**31 - 1),  # seed
    st.integers(1, 3),  # batch
    st.integers(1, 16),  # T
    st.sampled_from([1, 2]),  # key heads
    st.sampled_from([1, 2]),  # value heads per key head
    st.sampled_from([8, 16, 32]),  # K
    st.sampled_from([8, 16, 32]),  # V
)


class TestOracleAgainstTransformers:
    @settings(max_examples=40, deadline=None)
    @given(shapes, st.sampled_from([4, 8, 64]))
    def test_the_recurrence_matches_the_torch_chunk_function(
        self, shape: tuple[int, ...], chunk_size: int
    ) -> None:
        """The oracle is the recurrence; transformers' function is the chunked
        algorithm at any chunk size (several chunks at 4 and 8, one at 64) —
        they must compute the same function, state included."""
        q, k, v, g, beta = _inputs(*shape)
        hv = v.shape[2]
        o_ref, s_ref = _transformers_chunk()(
            expand_key_heads(q, hv),
            expand_key_heads(k, hv),
            v,
            g,
            beta,
            chunk_size=chunk_size,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        o, s = recurrent_gated_delta_rule_reference(
            q, k, v, g, beta, output_final_state=True
        )
        torch.testing.assert_close(o, o_ref, **VALUE_TOL)
        torch.testing.assert_close(s, s_ref, **VALUE_TOL)


class TestClosedFormAgainstOracle:
    @settings(max_examples=60, deadline=None)
    @given(shapes)
    def test_forward_and_final_state(self, shape: tuple[int, ...]) -> None:
        q, k, v, g, beta = _inputs(*shape)
        o, s = recurrent_gated_delta_rule_reference(
            q, k, v, g, beta, output_final_state=True
        )
        o2, s2 = single_chunk_gated_delta_rule_torch(
            q, k, v, g, beta, output_final_state=True
        )
        torch.testing.assert_close(o2, o, **VALUE_TOL)
        torch.testing.assert_close(s2, s, **VALUE_TOL)

    @settings(max_examples=40, deadline=None)
    @given(shapes)
    def test_every_gradient(self, shape: tuple[int, ...]) -> None:
        """The same random cotangents on the output and the final state, so
        the gradient through the state is checked too; all five inputs."""
        q, k, v, g, beta = _inputs(*shape)
        gen = torch.Generator().manual_seed(shape[0] ^ 0x5EED)
        b, t, hv, dv = v.shape
        w_o = torch.randn(b, t, hv, dv, generator=gen)
        w_s = torch.randn(b, hv, k.shape[-1], dv, generator=gen)

        def grads(fn: Any) -> tuple[torch.Tensor, ...]:
            leaves = [x.clone().requires_grad_(True) for x in (q, k, v, g, beta)]
            o, s = fn(*leaves, output_final_state=True)
            return torch.autograd.grad((o * w_o).sum() + (s * w_s).sum(), leaves)

        for name, a, b_ in zip(
            ("q", "k", "v", "g", "beta"),
            grads(recurrent_gated_delta_rule_reference),
            grads(single_chunk_gated_delta_rule_torch),
        ):
            torch.testing.assert_close(b_, a, msg=lambda m: f"d{name}: {m}", **GRAD_TOL)

    def test_no_l2norm_and_an_explicit_scale_are_honoured(self) -> None:
        q, k, v, g, beta = _inputs(3, 2, 7, 1, 2, 16, 16)
        kwargs: dict[str, Any] = dict(scale=0.25, use_qk_l2norm=False)
        o, _ = recurrent_gated_delta_rule_reference(q, k, v, g, beta, **kwargs)
        o2, _ = single_chunk_gated_delta_rule_torch(q, k, v, g, beta, **kwargs)
        torch.testing.assert_close(o2, o, **VALUE_TOL)
        o3, _ = single_chunk_gated_delta_rule_torch(q, k, v, g, beta, scale=0.25)
        assert not torch.allclose(o3, o2)  # the l2norm flag does something

    def test_grouped_heads_read_the_mixers_repeat_interleave(self) -> None:
        """Value head ``h`` reads key head ``h // group`` — the mixer's
        ``repeat_interleave`` — so the grouped call equals the repeated one."""
        q, k, v, g, beta = _inputs(11, 2, 9, 2, 2, 16, 16)
        o_grouped, _ = single_chunk_gated_delta_rule_torch(q, k, v, g, beta)
        o_repeated, _ = single_chunk_gated_delta_rule_torch(
            q.repeat_interleave(2, dim=2), k.repeat_interleave(2, dim=2), v, g, beta
        )
        torch.testing.assert_close(o_grouped, o_repeated, **VALUE_TOL)
        with pytest.raises(ValueError, match="not a multiple"):
            expand_key_heads(q, 3)
