"""Hypothesis strategies and comparison helpers shared by the tests."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod

import safetensors.torch as reference
import torch
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from causalab.io.fastersafetensors._dtypes import TORCH_TO_HEADER

#: The settings every property test here runs under, applied per test rather
#: than through a process-global profile (which would reach every Hypothesis
#: test in the session): a generated tensor dict can take longer than the
#: default deadline on a loaded machine, and the blob makes a failure
#: reproducible.
SETTINGS = settings(
    deadline=None, suppress_health_check=[HealthCheck.too_slow], print_blob=True
)

DTYPES: list[torch.dtype] = list(TORCH_TO_HEADER)
"""Every dtype the format and the installed torch both know."""


def _reference_serializes(dtype: torch.dtype) -> bool:
    try:
        reference.save({"t": torch.empty(0, dtype=dtype)})
    except (KeyError, TypeError, ValueError):
        # its dtype table has no row for this dtype (0.6.2 lacks the fnuz
        # float8 pair and complex64 that 0.8.0 has)
        return False
    return True


REFERENCE_DTYPES: list[torch.dtype] = [d for d in DTYPES if _reference_serializes(d)]
"""The subset the installed reference library can serialize. Tests that
compare bytes or tensors with the reference draw from this list; the rest of
``DTYPES`` is covered by the tests that round-trip through this library
alone."""


def tensor_from_bytes(
    raw: bytes, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    if not raw:
        return torch.empty(shape, dtype=dtype)
    return (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(shape)
    )


@st.composite
def tensors(
    draw: st.DrawFn,
    dtypes: list[torch.dtype] | None = None,
    max_dims: int = 3,
    max_extent: int = 4,
) -> torch.Tensor:
    """A tensor of random bit patterns: any dtype, scalars and 0-size shapes included."""
    dtype = draw(st.sampled_from(dtypes or DTYPES))
    shape = tuple(draw(st.lists(st.integers(0, max_extent), max_size=max_dims)))
    nbytes = prod(shape) * dtype.itemsize
    raw = draw(st.binary(min_size=nbytes, max_size=nbytes))
    return tensor_from_bytes(raw, dtype, shape)


plain_names = st.from_regex(r"[a-z][a-z0-9_./]{0,12}", fullmatch=True)
unicode_names = st.text(min_size=1, max_size=8).filter(lambda s: s != "__metadata__")
names = st.one_of(plain_names, unicode_names)


def tensor_dicts(
    min_size: int = 0, max_size: int = 8, dtypes: list[torch.dtype] | None = None
) -> st.SearchStrategy[dict[str, torch.Tensor]]:
    return st.dictionaries(
        names, tensors(dtypes=dtypes), min_size=min_size, max_size=max_size
    )


metadata = st.one_of(
    st.none(),
    st.just({}),
    st.dictionaries(
        st.text(min_size=1, max_size=6), st.text(max_size=10), min_size=1, max_size=4
    ),
)


def same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal dtype, shape, device and bits (NaNs compare equal, float8 works)."""
    if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
        return False
    return torch.equal(_bytes(a), _bytes(b))


def _bytes(t: torch.Tensor) -> torch.Tensor:
    # an empty tensor counts as contiguous whatever its strides, and a
    # non-unit stride makes the uint8 view refuse; there are no bytes to compare
    if t.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    flat = t.contiguous().reshape(-1)
    # A singleton can be "contiguous" with a non-unit stride too. Torch's
    # byte view still requires stride 1, including for a stepped singleton.
    if flat.stride(0) != 1:
        flat = flat.clone(memory_format=torch.contiguous_format)
    return flat.view(torch.uint8).cpu()


def same_dict(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(same_tensor(a[k], b[k]) for k in a)
