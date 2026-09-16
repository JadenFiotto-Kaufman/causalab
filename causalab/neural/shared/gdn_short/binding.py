"""Route short sequences to the single-chunk kernel, per forward.

The mixer calls the module global ``torch_chunk_gated_delta_rule`` (whatever
``kernels.torch_kernel_path`` has bound it to — FLA on CUDA, transformers'
torch function elsewhere).
:func:`short_seq_kernel_path` rebinds that global, in every modeling module
the model reaches, to a dispatcher that captures the current binding and
decides per call: a call :func:`selects_single_chunk` accepts runs
:func:`.triton_kernel.single_chunk_gated_delta_rule`; every other call runs
the captured binding unchanged. Restored on exit, like the guard it
composes with; the reference engine enters it after that guard and before the
DeltaNet taps, so a kernel-boundary tap wraps the dispatcher and sees the
same arguments and returns whichever kernel runs.

The decision is a pure function of the call's shape and flags — no tensor
is read, no device is synchronized — so it is the same at a CUDA graph's
warm-up pass and at its capture, and the graph replays the kernel it
captured. A sequence longer than the threshold, a call with an initial
state (a cached decode's prefill), a variable-length batch, or a model off
CUDA all keep the bound kernel.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator

import torch

from causalab.neural.shared.gdn_short.options import (
    CHUNK_KERNEL_GLOBAL,
    ShortSeqKernelOptions,
)
from causalab.neural.shared.gdn_short.triton_kernel import (
    MAX_SEQ_LEN,
    single_chunk_gated_delta_rule,
)
from causalab.neural.shared.kernels import _kernel_modules

__all__ = [
    "ShortSeqKernelOptions",
    "selects_single_chunk",
    "short_seq_dispatcher",
    "short_seq_kernel_path",
]


def _power_of_two_in_range(size: int) -> bool:
    return 16 <= size <= 256 and not size & (size - 1)


def selects_single_chunk(
    *,
    seq_len: int,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    device_type: str,
    threshold: int,
    initial_state: bool,
    varlen: bool,
) -> bool:
    """Whether a chunk-kernel call with these facts runs the single-chunk
    kernel: on CUDA, ``1 <= T <= threshold`` (``threshold <=``
    :data:`MAX_SEQ_LEN`), from a zero state, equal-length sequences, key
    heads dividing value heads, power-of-two head dimensions in ``[16,
    256]``."""
    return (
        device_type == "cuda"
        and 1 <= seq_len <= min(threshold, MAX_SEQ_LEN)
        and not initial_state
        and not varlen
        and value_heads % key_heads == 0
        and _power_of_two_in_range(key_dim)
        and _power_of_two_in_range(value_dim)
    )


def short_seq_dispatcher(
    bound: Callable[..., Any],
    options: ShortSeqKernelOptions,
    short: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """The dispatcher over ``bound`` (the global as it was): the mixer's call
    shape — ``(q, k, v, g=, beta=, **kwargs)`` — with the routing decision
    from :func:`selects_single_chunk`. ``short`` is the single-chunk kernel,
    this module's :func:`single_chunk_gated_delta_rule` unless a test hands
    in a stand-in."""
    if short is None:
        short = single_chunk_gated_delta_rule

    def dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if (
            g is not None
            and beta is not None
            and selects_single_chunk(
                seq_len=int(query.shape[1]),
                key_heads=int(key.shape[2]),
                value_heads=int(value.shape[2]),
                key_dim=int(key.shape[-1]),
                value_dim=int(value.shape[-1]),
                device_type=query.device.type,
                threshold=options.threshold,
                initial_state=kwargs.get("initial_state") is not None,
                varlen=kwargs.get("cu_seqlens") is not None,
            )
        ):
            return short(query, key, value, g, beta, **kwargs)
        return bound(query, key, value, g=g, beta=beta, **kwargs)

    dispatch.__wrapped__ = bound  # type: ignore[attr-defined]
    return dispatch


@contextlib.contextmanager
def short_seq_kernel_path(
    model: torch.nn.Module, options: ShortSeqKernelOptions | None = None
) -> Iterator[None]:
    """While active, every chunk-kernel call of ``model``'s DeltaNet mixers
    goes through :func:`short_seq_dispatcher` (module docstring). ``None``
    reads the options from the environment; a disabled option installs
    nothing. Restored on exit either way."""
    if options is None:
        options = ShortSeqKernelOptions.from_env()
    if not options.enabled:
        yield
        return
    rebound: list[tuple[Any, Any]] = []
    try:
        for modeling in _kernel_modules(model):
            current = getattr(modeling, CHUNK_KERNEL_GLOBAL)
            rebound.append((modeling, current))
            setattr(
                modeling, CHUNK_KERNEL_GLOBAL, short_seq_dispatcher(current, options)
            )
        yield
    finally:
        for modeling, current in reversed(rebound):
            setattr(modeling, CHUNK_KERNEL_GLOBAL, current)
