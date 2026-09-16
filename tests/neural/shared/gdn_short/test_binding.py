"""The routing decision and the knob of ``gdn_short/binding.py`` and
``gdn_short/options.py`` — the engine-free half. The
dispatcher is exercised with a stand-in kernel and CPU tensors whose device
type the predicate is told is CUDA (there is none here); the real kernel's
own tests are ``test_triton_kernel.py``, on a CUDA device.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.shared.gdn_short import binding
from causalab.neural.shared.gdn_short.binding import (
    selects_single_chunk,
    short_seq_dispatcher,
)
from causalab.neural.shared.gdn_short.options import (
    DEFAULT_SHORT_SEQ,
    ENV_SHORT_SEQ,
    KernelOptionError,
    ShortSeqKernelOptions,
)
from causalab.neural.shared.gdn_short.triton_kernel import MAX_SEQ_LEN, tile_rows

pytestmark = pytest.mark.unit


def _facts(**overrides: Any) -> dict[str, Any]:
    """The workflow's call: ``[96, 13, 32, 128]`` on CUDA from a zero state."""
    facts: dict[str, Any] = dict(
        seq_len=13,
        key_heads=32,
        value_heads=32,
        key_dim=128,
        value_dim=128,
        device_type="cuda",
        threshold=16,
        initial_state=False,
        varlen=False,
    )
    facts.update(overrides)
    return facts


class TestSelectsSingleChunk:
    def test_the_workflows_call_is_selected(self) -> None:
        assert selects_single_chunk(**_facts())
        assert selects_single_chunk(**_facts(key_heads=16))  # the un-repeated layout
        assert selects_single_chunk(**_facts(seq_len=1))
        assert selects_single_chunk(**_facts(seq_len=16))

    @pytest.mark.parametrize(
        "facts",
        [
            _facts(seq_len=17),  # above the threshold: FLA
            _facts(seq_len=0),
            _facts(device_type="cpu"),  # the torch path stays
            _facts(initial_state=True),  # a cached prefill
            _facts(varlen=True),
            _facts(key_heads=3),  # 32 % 3 != 0
            _facts(key_dim=96),
            _facts(value_dim=512),
            _facts(threshold=0),  # disabled
        ],
        ids=[
            "long",
            "empty",
            "cpu",
            "initial_state",
            "varlen",
            "heads",
            "key_dim",
            "value_dim",
            "disabled",
        ],
    )
    def test_everything_else_keeps_the_bound_kernel(
        self, facts: dict[str, Any]
    ) -> None:
        assert not selects_single_chunk(**facts)

    def test_the_threshold_is_capped_at_the_kernels_longest_tile(self) -> None:
        assert selects_single_chunk(**_facts(seq_len=32, threshold=32))
        assert not selects_single_chunk(**_facts(seq_len=33, threshold=99))
        assert tile_rows(13) == 16 and tile_rows(17) == 32 and tile_rows(32) == 32
        with pytest.raises(ValueError, match=f"T <= {MAX_SEQ_LEN}"):
            tile_rows(MAX_SEQ_LEN + 1)


class TestDispatcher:
    @staticmethod
    def _tensors(t: int) -> tuple[torch.Tensor, ...]:
        q = torch.zeros(2, t, 4, 16)
        v = torch.zeros(2, t, 4, 16)
        g = torch.zeros(2, t, 4)
        return q, q.clone(), v, g, g.clone()

    def test_routes_by_length_and_hands_the_call_through_whole(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        def bound(
            q: Any, k: Any, v: Any, g: Any = None, beta: Any = None, **kw: Any
        ) -> str:
            calls.append(("bound", kw))
            return "bound"

        def short(q: Any, k: Any, v: Any, g: Any, beta: Any, **kw: Any) -> str:
            calls.append(("short", kw))
            return "short"

        # CPU tensors here: tell the predicate they are on CUDA
        monkeypatch.setattr(
            binding,
            "selects_single_chunk",
            lambda **facts: selects_single_chunk(**{**facts, "device_type": "cuda"}),
        )
        dispatch = short_seq_dispatcher(bound, ShortSeqKernelOptions(16), short)
        assert dispatch.__wrapped__ is bound  # type: ignore[attr-defined]
        extra = dict(
            output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=None
        )
        q, k, v, g, beta = self._tensors(13)
        assert dispatch(q, k, v, g=g, beta=beta, initial_state=None, **extra) == "short"
        q, k, v, g, beta = self._tensors(17)
        assert dispatch(q, k, v, g=g, beta=beta, initial_state=None, **extra) == "bound"
        q, k, v, g, beta = self._tensors(13)
        state = torch.zeros(2, 4, 16, 16)
        assert (
            dispatch(q, k, v, g=g, beta=beta, initial_state=state, **extra) == "bound"
        )
        assert (
            dispatch(q, k, v, g=None, beta=beta, **extra) == "bound"
        )  # no gate: FLA's error
        assert [name for name, _ in calls] == ["short", "bound", "bound", "bound"]
        # every keyword the mixer said reaches whichever kernel runs
        assert calls[0][1] == dict(initial_state=None, **extra)
        assert calls[1][1] == dict(initial_state=None, **extra)

    def test_on_cpu_tensors_nothing_is_routed(self) -> None:
        dispatch = short_seq_dispatcher(
            lambda *a, **k: "bound",
            ShortSeqKernelOptions(16),
            lambda *a, **k: pytest.fail("the single-chunk kernel needs CUDA"),
        )
        q, k, v, g, beta = self._tensors(13)
        assert dispatch(q, k, v, g=g, beta=beta) == "bound"


class TestOptions:
    def test_unset_means_the_default_where_triton_is_importable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from causalab.neural.shared.gdn_short import triton_kernel

        monkeypatch.setattr(triton_kernel, "triton_available", lambda: True)
        assert ShortSeqKernelOptions.from_env({}).threshold == DEFAULT_SHORT_SEQ
        monkeypatch.setattr(triton_kernel, "triton_available", lambda: False)
        options = ShortSeqKernelOptions.from_env({ENV_SHORT_SEQ: " "})
        assert options.threshold == 0 and not options.enabled

    def test_the_variable_sets_the_threshold(self) -> None:
        assert ShortSeqKernelOptions.from_env({ENV_SHORT_SEQ: "32"}).threshold == 32
        assert not ShortSeqKernelOptions.from_env({ENV_SHORT_SEQ: "0"}).enabled
        assert ShortSeqKernelOptions(DEFAULT_SHORT_SEQ).enabled

    def test_refusals_are_structured_and_name_the_option(self) -> None:
        with pytest.raises(KernelOptionError, match=r"short_seq=33.*up to 32"):
            ShortSeqKernelOptions(MAX_SEQ_LEN + 1)
        with pytest.raises(KernelOptionError, match="short_seq=-1"):
            ShortSeqKernelOptions(-1)
        with pytest.raises(KernelOptionError, match=ENV_SHORT_SEQ) as info:
            ShortSeqKernelOptions.from_env({ENV_SHORT_SEQ: "sixteen"})
        assert (info.value.option, info.value.value) == ("short_seq", "sixteen")

    def test_a_disabled_option_installs_nothing(self) -> None:
        model = torch.nn.Linear(2, 2)
        with binding.short_seq_kernel_path(model, ShortSeqKernelOptions(0)):
            pass  # no modeling module, nothing to rebind: a no-op either way
