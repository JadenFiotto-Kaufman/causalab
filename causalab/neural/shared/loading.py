"""What every engine's loader shares: the precision names' torch dtypes and
the module behind an nnsight envoy."""

from __future__ import annotations

from typing import Any, Mapping
from types import MappingProxyType

import torch

__all__ = ["TORCH_DTYPES", "torch_module"]

#: The document's precision names (``schema.PRECISION_DTYPES``) as torch dtypes.
TORCH_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
)


def torch_module(envoy: Any) -> torch.nn.Module:
    """The torch module an nnsight envoy wraps (``Envoy._module``) — what the
    kernel-path binding inspects and the family predicates walk."""
    module = getattr(envoy, "_module", None)
    if not isinstance(module, torch.nn.Module):
        raise AssertionError("an nnsight envoy wraps a torch module")
    return module
