"""The fused MoE glue kernels' on/off set (``neural/shared/kernel_options.py``,
``MoeGlueOptions``) — the engine-free half: what the environment can say and
what is refused.
"""

from __future__ import annotations

import pytest

from causalab.neural.shared.kernel_options import (
    DEFAULT_MOE_GLUE_KERNELS,
    ENV_MOE_GLUE,
    MOE_GLUE_KERNELS,
    KernelOptionError,
    MoeGlueOptions,
)

pytestmark = pytest.mark.unit


class TestMoeGlueOptions:
    def test_unset_or_empty_is_every_kernel(self) -> None:
        for environ in ({}, {ENV_MOE_GLUE: ""}, {ENV_MOE_GLUE: "  "}):
            options = MoeGlueOptions.from_env(environ)
            assert options.is_default and options.kernels == DEFAULT_MOE_GLUE_KERNELS
        assert DEFAULT_MOE_GLUE_KERNELS == frozenset(MOE_GLUE_KERNELS)

    def test_off_all_and_a_subset(self) -> None:
        assert MoeGlueOptions.from_env({ENV_MOE_GLUE: "off"}).kernels == frozenset()
        assert MoeGlueOptions.from_env({ENV_MOE_GLUE: "ALL"}).kernels == frozenset(
            MOE_GLUE_KERNELS
        )
        subset = MoeGlueOptions.from_env({ENV_MOE_GLUE: "sort, gate"})
        assert subset.kernels == frozenset({"sort", "gate"})
        assert subset.enabled("gate") and not subset.enabled("gather")
        assert not subset.is_default

    def test_an_unknown_kernel_is_refused_by_name(self) -> None:
        with pytest.raises(
            KernelOptionError, match=r"moe_glue='histc'.*no such kernel"
        ) as info:
            MoeGlueOptions.from_env({ENV_MOE_GLUE: "sort,histc"})
        assert (info.value.option, info.value.value) == ("moe_glue", "histc")

    def test_the_process_environment_is_the_default_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_MOE_GLUE, "off")
        assert MoeGlueOptions.from_env().kernels == frozenset()
        monkeypatch.delenv(ENV_MOE_GLUE)
        assert MoeGlueOptions.from_env().is_default
