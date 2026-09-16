"""Caller-side kernel options read from the environment — the reference
engine's switch for the fused Triton kernels around the grouped-experts GEMMs
(``pytorch_hooks/kernels/moe_glue.py``).

:class:`MoeGlueOptions` from ``CAUSALAB_MOE_GLUE`` — unset means every kernel
(:data:`DEFAULT_MOE_GLUE_KERNELS`), ``off`` none, ``all`` every kernel, a
comma list a subset. Each kernel reproduces the ATen path it replaces bit
for bit, so the knob exists to switch a kernel off for a bisection; it never
changes numbers. A kernel named here still runs only where its plan admits
it (a CUDA tensor, Triton importable, a shape whose ATen order the kernel
models); a kernel not named never runs.

The DeltaNet kernels' knobs need no code here: FLA reads its tuning options
from its own environment at dispatch time (``docs/attention_backends.md``
lists them), and the chunk size is not exposed — a chunking other than the
default is mathematically the same function but not bit-identical.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Mapping

__all__ = [
    "DEFAULT_MOE_GLUE_KERNELS",
    "ENV_MOE_GLUE",
    "KernelOptionError",
    "MOE_GLUE_KERNELS",
    "MoeGlueOptions",
]

#: The environment variable :meth:`MoeGlueOptions.from_env` reads.
ENV_MOE_GLUE = "CAUSALAB_MOE_GLUE"

#: The fused glue kernels of the grouped-experts path, by name: the stable
#: counting sort, the row gather with its exact index backward, the
#: weight·un-sort·slot-sum epilogue, and ``silu(gate) * up``.
MOE_GLUE_KERNELS: tuple[str, ...] = ("sort", "gather", "epilogue", "gate")

#: The kernels on when nothing is set: every one, each proven bit-identical
#: to the ATen path on the H100 by ``tests/golden/test_moe_glue_kernels.py``
#: (📐 one H100, 2026-09-15: output and all four gradients equal on the
#: workflow's shapes in bf16 / fp16 / fp32; the gate kernel's libdevice
#: ``exp`` / ``div_rn`` / ``fma`` match nvcc's compiled ``silu`` there).
DEFAULT_MOE_GLUE_KERNELS: frozenset[str] = frozenset(MOE_GLUE_KERNELS)


class KernelOptionError(ValueError):
    """A kernel option that cannot be honoured: a value outside the option's
    accepted set."""

    def __init__(self, option: str, value: Any, reason: str) -> None:
        self.option = option
        self.value = value
        self.reason = reason
        super().__init__(f"kernel option {option}={value!r}: {reason}")


@dataclasses.dataclass(frozen=True)
class MoeGlueOptions:
    """Which fused glue kernels of the grouped-experts path may run. A
    kernel named here still runs only where its plan admits it (a CUDA
    tensor, Triton importable, a shape whose ATen order the kernel models);
    a kernel not named never runs."""

    kernels: frozenset[str] = DEFAULT_MOE_GLUE_KERNELS

    def __post_init__(self) -> None:
        unknown = sorted(set(self.kernels) - set(MOE_GLUE_KERNELS))
        if unknown:
            raise KernelOptionError(
                "moe_glue",
                unknown[0],
                f"no such kernel; the kernels are {list(MOE_GLUE_KERNELS)}",
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "MoeGlueOptions":
        """``CAUSALAB_MOE_GLUE``: unset or empty is the default set, ``off``
        none, ``all`` every kernel, otherwise a comma-separated subset of
        :data:`MOE_GLUE_KERNELS`."""
        env = os.environ if environ is None else environ
        raw = env.get(ENV_MOE_GLUE, "").strip().lower()
        if not raw:
            return cls()
        if raw == "off":
            return cls(kernels=frozenset())
        if raw == "all":
            return cls(kernels=frozenset(MOE_GLUE_KERNELS))
        names = [name.strip() for name in raw.split(",") if name.strip()]
        return cls(kernels=frozenset(names))

    @property
    def is_default(self) -> bool:
        return self.kernels == DEFAULT_MOE_GLUE_KERNELS

    def enabled(self, kernel: str) -> bool:
        return kernel in self.kernels
