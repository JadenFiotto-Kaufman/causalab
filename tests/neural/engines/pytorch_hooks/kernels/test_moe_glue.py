"""The fused glue kernels' plan and fallback (``pytorch_hooks/kernels/moe_glue.py``)
on a machine without CUDA: which kernel each condition admits, that the plan
is empty off CUDA whatever the options say, and that the experts path with
every kernel enabled is still the library's numbers to the bit here. The
kernels themselves run only under ``tests/golden/test_moe_glue_kernels.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers.integrations.moe as moe

from causalab.neural.engines.pytorch_hooks import experts_path
from causalab.neural.engines.pytorch_hooks.experts_path import (
    has_default_silu_gate,
    lean_grouped_mm_forward,
)
from causalab.neural.engines.pytorch_hooks.kernels import moe_glue
from causalab.neural.engines.pytorch_hooks.kernels import moe_glue_triton as kernels
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.kernel_options import ENV_MOE_GLUE, MoeGlueOptions

from ..test_experts_path import _routing, _run

ALL = MoeGlueOptions(kernels=frozenset({"sort", "gather", "epilogue", "gate"}))


def _cuda_shaped(tokens: int, width: int, dtype: torch.dtype = torch.bfloat16) -> Any:
    """What the plan reads off a hidden-state tensor, as if on CUDA."""
    return SimpleNamespace(
        device=torch.device("cuda"), dtype=dtype, shape=(tokens, width)
    )


def _plan(
    *,
    tokens: int = 1248,
    top_k: int = 8,
    width: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    weights_dtype: torch.dtype | None = None,
    expert_dtype: torch.dtype | None = None,
    default_silu_gate: bool = True,
    options: MoeGlueOptions = ALL,
) -> moe_glue.GluePlan:
    return moe_glue.plan_moe_glue(
        hidden_states=_cuda_shaped(tokens, width, dtype),
        top_k_weights=SimpleNamespace(dtype=weights_dtype or dtype),
        expert_dtype=expert_dtype or dtype,
        num_pairs=tokens * top_k,
        top_k=top_k,
        default_silu_gate=default_silu_gate,
        options=options,
    )


class TestPlan:
    pytestmark = pytest.mark.unit

    @pytest.fixture(autouse=True)
    def _triton_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kernels, "available", lambda: True)

    def test_off_cuda_the_plan_is_empty_whatever_the_options(self) -> None:
        plan = moe_glue.plan_moe_glue(
            hidden_states=torch.zeros(4, 2048, dtype=torch.bfloat16),
            top_k_weights=torch.zeros(4, 8, dtype=torch.bfloat16),
            expert_dtype=torch.bfloat16,
            num_pairs=32 * 8,
            top_k=8,
            default_silu_gate=True,
            options=ALL,
        )
        assert plan == moe_glue.GluePlan() and not plan.any

    def test_without_triton_the_plan_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(kernels, "available", lambda: False)
        assert not _plan().any

    def test_the_workflow_s_training_shapes_take_every_kernel(self) -> None:
        for tokens in (546, 1248):
            plan = _plan(tokens=tokens, options=MoeGlueOptions())
            assert plan == moe_glue.GluePlan(
                sort=True, gather=True, epilogue=True, gate=True
            )

    def test_the_eval_batch_keeps_torch_sort(self) -> None:
        """S = 93 600: the counting sort is linear in pairs × experts and
        loses to cub's radix sort there (``MAX_COUNTING_SORT_PAIRS``)."""
        plan = _plan(tokens=11700, options=MoeGlueOptions())
        assert plan == moe_glue.GluePlan(
            sort=False, gather=True, epilogue=True, gate=True
        )
        bound = moe_glue.MAX_COUNTING_SORT_PAIRS
        assert _plan(tokens=bound // 8).sort
        assert not _plan(tokens=bound // 8 + 1).sort

    def test_the_options_are_the_ceiling(self) -> None:
        assert not _plan(options=MoeGlueOptions(kernels=frozenset())).any
        only_sort = _plan(options=MoeGlueOptions(kernels=frozenset({"sort"})))
        assert only_sort == moe_glue.GluePlan(sort=True)

    def test_thirty_two_pairs_or_fewer_keep_torch_sort(self) -> None:
        """CUDA sorts 32 or fewer keys with an unstable bitonic sort the
        stable counting sort would not reproduce."""
        assert not _plan(tokens=4, top_k=8, width=256).sort
        assert _plan(tokens=11, top_k=3, width=256).sort

    def test_the_epilogue_needs_one_dtype_and_a_modelled_row_sum(self) -> None:
        assert not _plan(weights_dtype=torch.float32).epilogue
        assert not _plan(expert_dtype=torch.float16).epilogue
        assert not _plan(width=8).epilogue, "eight lanes, not the 32-lane block"
        assert not _plan(width=8192).epilogue, "a warp split"
        assert not _plan(width=130, tokens=100).epilogue, "a vector tail"
        assert not _plan(tokens=1, top_k=8, width=2048).epilogue, "a 512-lane block"
        assert _plan(tokens=37, top_k=3, width=136).epilogue
        assert (
            _plan(dtype=torch.float32).epilogue and _plan(dtype=torch.float16).epilogue
        )

    def test_the_gather_and_gate_admissions(self) -> None:
        assert _plan(width=8).gather and _plan(width=3).gather
        assert not _plan(default_silu_gate=False).gate
        assert _plan(options=MoeGlueOptions()).gate


class TestWithoutTriton:
    pytestmark = pytest.mark.unit

    @pytest.mark.skipif(kernels.available(), reason="Triton is importable here")
    def test_a_kernel_call_is_refused_by_name(self) -> None:
        with pytest.raises(RuntimeError, match="need Triton"):
            kernels.counting_sort(torch.zeros(4, dtype=torch.long), 2)
        with pytest.raises(RuntimeError, match="need Triton"):
            kernels.silu_mul_forward(torch.zeros(2, 4))


def _experts(bundle: ModelBundle) -> Any:
    from ..test_sites_round3_moe_interior import MOE_LAYER

    return bundle.model.model.layers[MOE_LAYER].mlp.experts


class TestExpertsPathOnTheCpu:
    pytestmark = pytest.mark.smoke

    def test_the_fixture_s_gate_is_the_default_silu_one(
        self, qwen35moe_bundle: ModelBundle
    ) -> None:
        experts = _experts(qwen35moe_bundle)
        assert has_default_silu_gate(experts)
        assert not has_default_silu_gate(SimpleNamespace(has_gate=False))
        custom = SimpleNamespace(has_gate=True, act_fn=torch.nn.SiLU())
        custom._apply_gate = lambda x: x
        assert not has_default_silu_gate(custom)
        plain = SimpleNamespace(has_gate=True, act_fn=torch.nn.SiLU())
        plain._apply_gate = moe._default_apply_gate.__get__(plain)
        assert has_default_silu_gate(plain)
        other = SimpleNamespace(has_gate=True, act_fn=torch.nn.GELU())
        other._apply_gate = moe._default_apply_gate.__get__(other)
        assert not has_default_silu_gate(other)

    @pytest.mark.parametrize(
        "register", ["register_forward_hook", "register_forward_pre_hook"]
    )
    def test_activation_hooks_keep_the_module_gate(
        self, qwen35moe_bundle: ModelBundle, register: str
    ) -> None:
        experts = _experts(qwen35moe_bundle)
        assert has_default_silu_gate(experts)
        handle = getattr(experts.act_fn, register)(lambda *args: None)
        try:
            assert not has_default_silu_gate(experts)
        finally:
            handle.remove()
        assert has_default_silu_gate(experts)

    def test_every_kernel_enabled_changes_nothing_off_cuda(
        self, qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experts = _experts(qwen35moe_bundle)
        params = (experts.gate_up_proj, experts.down_proj)
        flags = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(True)
        calls: list[str] = []

        def refuse(*args: Any, **kwargs: Any) -> None:
            calls.append("sort")
            raise AssertionError("a fused kernel must not run off CUDA")

        for name in ("counting_sort", "fused_gather", "fused_epilogue", "fused_gate"):
            monkeypatch.setattr(moe_glue, name, refuse)
        monkeypatch.setenv(ENV_MOE_GLUE, "all")
        try:
            hidden, index, weights = _routing(experts, tokens=7, seed=11)
            ours = _run(lean_grouped_mm_forward, experts, hidden, index, weights)
            theirs = _run(
                moe.grouped_mm_experts_forward, experts, hidden, index, weights
            )
        finally:
            for p, flag in zip(params, flags):
                p.requires_grad_(flag)
        assert calls == []
        for name, a, b in zip(
            ("output", "d_hidden", "d_gate_up", "d_down"), ours, theirs
        ):
            assert torch.equal(a, b), name

    def test_the_plan_is_read_from_the_environment_per_call(
        self, qwen35moe_bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[MoeGlueOptions] = []
        real = moe_glue.plan_moe_glue

        def spy(**kwargs: Any) -> moe_glue.GluePlan:
            seen.append(kwargs["options"])
            return real(**kwargs)

        monkeypatch.setattr(experts_path.moe_glue, "plan_moe_glue", spy)
        experts = _experts(qwen35moe_bundle)
        hidden, index, weights = _routing(experts, tokens=3, seed=1)
        monkeypatch.setenv(ENV_MOE_GLUE, "off")
        with torch.no_grad():
            lean_grouped_mm_forward(experts, hidden, index, weights)
        monkeypatch.delenv(ENV_MOE_GLUE)
        with torch.no_grad():
            lean_grouped_mm_forward(experts, hidden, index, weights)
        assert [o.kernels for o in seen] == [frozenset(), MoeGlueOptions().kernels]
