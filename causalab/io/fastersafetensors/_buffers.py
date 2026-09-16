"""Byte views over tensor memory, for callers who want the bytes."""

from __future__ import annotations

import torch


def as_memoryview(tensor: torch.Tensor) -> memoryview:
    """A read-only byte view over a contiguous CPU tensor, sharing its memory
    and keeping it alive."""
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("as_memoryview needs a contiguous CPU tensor")
    flat = tensor.reshape(-1)
    # ``ndarray.data`` is already a memoryview; wrapping the array itself trips
    # the 3.10 typeshed, whose ``Buffer`` protocol numpy's stubs do not declare
    return flat.view(torch.uint8).numpy().data.toreadonly()
