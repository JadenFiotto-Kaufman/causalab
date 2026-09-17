"""Bind a non-CUDA model to transformers' own kernel implementations.

The Gated DeltaNet mixer (``modeling_qwen3_5_moe.py`` and its relatives)
calls four **module-global** functions — ``causal_conv1d_fn``,
``causal_conv1d_update``, ``torch_chunk_gated_delta_rule``,
``torch_recurrent_gated_delta_rule`` — each decorated with transformers'
``use_kernel_func_from_hub_with_fallback``. That decorator resolves the
implementation **once, at import time**: the optional package's CUDA kernel
(``causal_conv1d``, ``fla``) when it is importable, transformers' torch
function otherwise. There is no device check. With the
``flash-linear-attention`` extra installed, a model on the CPU (every unit
test on ``tiny-random/qwen3.5-moe``; a caller-owned CPU model) therefore
dispatches CUDA kernels on CPU tensors and dies inside them
(``Expected x.is_cuda() to be true``).

The torch implementation is still there: the decorator wraps it with
``functools.wraps``, so it is the innermost ``__wrapped__`` of the module
global. :func:`torch_kernel_path` rebinds the four globals — for the duration
of one forward, when and only when the model's weights are not on a CUDA
device — to transformers' **own** wrapper over that torch function, built by
the same decorator with a package that cannot import, and restores them on
exit. The wrapper rather than the bare function, because the module global's
*shape* is part of the nnterp engine's address table: its ``.source``
interiors peel the hub wrapper's ``implementation_0`` call before descending
into the torch body (``nnsight_tracing/addresses.py``), and a bare function
has nothing to peel. A CUDA model is untouched, so an installed kernel keeps
serving it; a machine without the extras is untouched too, because its module
globals already dispatch to the torch functions (checked through the
wrapper's closure, not assumed). Both engines wrap their model calls in it, so
where a model runs decides which kernel runs, not which packages happen to
be installed.

Two entry points. :func:`torch_kernel_path` is the per-forward guard both
engines wrap their model calls in — restore-on-exit, so a process holding a
CPU model and a CUDA model of one family serves each correctly. It covers
nothing that calls the model **outside** an engine: a test's reference
forward, a caller's own ``bundle.model(...)``, a direct call of the kernel.
For those, :func:`bind_kernel_path` binds a family's globals for the device a
model was **loaded** on, and stays: the reference engine's loader calls it, so
on a machine with the extras a CPU load leaves the family on the torch path and
a later CUDA load puts the installed kernels back — the most recent load's
device decides what a bare forward of that family runs. The nnsight loader
calls it with the device it was asked for (its weights are placed on first
trace, so at load nothing can be read off them); its executor's forwards go
through the guard as well.

Restore-on-exit for the guard rather than rebinding at load: module globals are process
state, and a process may hold a CPU model and a CUDA model of the same family
at once (the engine's test session does). Composed with the DeltaNet taps
(``pytorch_hooks/delta_interface.py``), which capture the globals at *their*
entry and call through to them: entered first, this hands them the torch
path to wrap.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import weakref
from typing import Any, Callable, Iterator

import torch

__all__ = [
    "KERNEL_GLOBALS",
    "bind_kernel_path",
    "torch_kernel_path",
    "torch_implementation",
]

#: The DeltaNet mixer's kernel-boundary globals, per modeling module — the
#: same four ``delta_interface`` swaps for its taps — each with the hub-kernel
#: name transformers decorates it under (the decorator's first argument).
_HUB_NAMES: dict[str, str] = {
    "causal_conv1d_fn": "causal_conv1d_fn",
    "causal_conv1d_update": "causal_conv1d_update",
    "torch_chunk_gated_delta_rule": "chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule": "fused_recurrent_gated_delta_rule",
}
KERNEL_GLOBALS: tuple[str, ...] = tuple(_HUB_NAMES)

#: A package name nothing provides: handed to transformers' decorator so its
#: import fails and the wrapper it builds dispatches to the torch function.
_NO_PACKAGE = "causalab_torch_kernel_path"

#: torch function -> the hub-shaped wrapper over it, built once so the module
#: global is the same object on every forward
_TORCH_PATHS: dict[int, Callable[..., Any]] = {}

#: (modeling module name, global) -> the binding the environment installed —
#: what :func:`bind_kernel_path` puts back for a CUDA load after a CPU load
_INSTALLED: dict[tuple[str, str], Callable[..., Any]] = {}


def torch_implementation(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The innermost function under ``fn``'s ``functools.wraps`` chain — for a
    transformers kernel global, its torch implementation; for an undecorated
    function, ``fn`` itself."""
    seen: set[int] = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        inner = getattr(fn, "__wrapped__", None)
        if inner is None:
            break
        fn = inner
    return fn


#: model -> its kernel modeling modules, scanned once: the guard runs on every
#: forward of a model off CUDA, and a model's module tree does not move. Keyed
#: weakly on the module itself, never on ``id()``: an address is unique only
#: among live objects, and a collected model's entry must not answer for the
#: next model allocated where it was
_KERNEL_MODULES: "weakref.WeakKeyDictionary[torch.nn.Module, list[Any]]" = (
    weakref.WeakKeyDictionary()
)


def _kernel_modules(model: torch.nn.Module) -> list[Any]:
    """Every modeling module some submodule of ``model`` is defined in that
    exports all of :data:`KERNEL_GLOBALS` — the files whose kernel dispatch
    this model's forward reaches. Scanned once per model object."""
    cached = _KERNEL_MODULES.get(model)
    if cached is not None:
        return cached
    out: list[Any] = []
    seen: set[str] = set()
    for module in model.modules():
        name = type(module).__module__
        if name in seen:
            continue
        seen.add(name)
        try:
            modeling = importlib.import_module(name)
        except ImportError:  # a class defined in a namespace importlib cannot reach
            continue
        if all(hasattr(modeling, attr) for attr in KERNEL_GLOBALS):
            out.append(modeling)
    _KERNEL_MODULES[model] = out
    return out


def _dispatches_to(wrapper: Callable[..., Any], torch_fn: Callable[..., Any]) -> bool:
    """Whether ``wrapper`` — a transformers kernel global — already calls
    ``torch_fn``: its closure's ``implementation`` is the torch function. A
    function whose closure the inspection cannot read is treated as not."""
    try:
        nonlocals = inspect.getclosurevars(wrapper).nonlocals
    except (TypeError, ValueError):
        return False
    return nonlocals.get("implementation") is torch_fn


def _torch_path(name: str, torch_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Transformers' own wrapper over ``torch_fn`` with no package to prefer:
    the same object the module global is on a machine without the extras."""
    cached = _TORCH_PATHS.get(id(torch_fn))
    if cached is None:
        from transformers.integrations.hub_kernels import (
            use_kernel_func_from_hub_with_fallback,
        )

        cached = use_kernel_func_from_hub_with_fallback(_HUB_NAMES[name], _NO_PACKAGE)(
            torch_fn
        )
        _TORCH_PATHS[id(torch_fn)] = cached
    return cached


def _on_cuda(model: torch.nn.Module) -> bool:
    """Whether the model's weights are on CUDA, read off its first parameter.

    A single-device placement is assumed — the reference engine's ``.to(device)``
    and the nnsight bundle's requested device are both one device. A model
    straddling devices (``device_map`` offload) is not handled: its DeltaNet
    blocks would run whatever the first parameter's device selects.
    """
    for parameter in model.parameters():
        return parameter.device.type == "cuda"
    return False


def bind_kernel_path(model: torch.nn.Module, *, on_cuda: bool | None = None) -> None:
    """Bind the DeltaNet kernel globals of every modeling module ``model``
    reaches for the device its weights are on, and leave them bound (module
    docstring): the torch path off CUDA, the installed kernels on CUDA. A
    model with no such module, or a machine without the extras, is left as it
    is.

    ``on_cuda`` overrides the inspection of the weights — for a loader that
    knows the device it will place the model on before the weights are there
    (nnsight dispatches on first trace, so at load they are still on
    ``meta``)."""
    if on_cuda is None:
        on_cuda = _on_cuda(model)
    for modeling in _kernel_modules(model):
        for name in KERNEL_GLOBALS:
            current = getattr(modeling, name)
            torch_fn = torch_implementation(current)
            key = (modeling.__name__, name)
            already_torch = torch_fn is current or _dispatches_to(current, torch_fn)
            if not already_torch:
                # whatever dispatches elsewhere is the environment's binding
                _INSTALLED[key] = current
            if on_cuda:
                installed = _INSTALLED.get(key)
                if installed is not None and current is not installed:
                    setattr(modeling, name, installed)
            elif not already_torch:
                setattr(modeling, name, _torch_path(name, torch_fn))


@contextlib.contextmanager
def torch_kernel_path(model: torch.nn.Module) -> Iterator[None]:
    """While active, a model whose weights are not on CUDA runs transformers'
    torch implementations of the DeltaNet kernel globals (module docstring);
    a CUDA model is untouched. Every rebinding is restored on exit."""
    if _on_cuda(model):
        yield
        return
    rebound: list[tuple[Any, str, Any]] = []
    try:
        for modeling in _kernel_modules(model):
            for name in KERNEL_GLOBALS:
                current = getattr(modeling, name)
                torch_fn = torch_implementation(current)
                if torch_fn is current or _dispatches_to(current, torch_fn):
                    continue
                rebound.append((modeling, name, current))
                setattr(modeling, name, _torch_path(name, torch_fn))
        yield
    finally:
        for modeling, name, current in reversed(rebound):
            setattr(modeling, name, current)
