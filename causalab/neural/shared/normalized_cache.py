"""An LRU cache keyed on a call's *bound* arguments, not on its spelling.

``functools.lru_cache`` hashes the call exactly as it was written: ``f("k")``,
``f(key="k")`` and ``f("k", revision="main")`` are three keys even when the
function binds all three to the same arguments. For a function whose result is
expensive and identity-bearing — a model loader, whose bundle is held by every
executor of a run — that means two spellings of one realization occupy two
slots of a small cache and construct the model twice.

:func:`normalized_cache` binds each call against the function's signature,
applies the defaults, and keys the cache on the resulting argument values in
signature order. A parameter whose value is not hashable, or whose equal values
are not equal as Python objects, names a ``keys`` function that maps it to the
canonical hashable form used *for the key only*; the function itself receives
the arguments exactly as the caller passed them. The decorated object keeps
:func:`functools.lru_cache`'s public surface — ``cache_info()``,
``cache_clear()``, ``__wrapped__`` — and adds :meth:`NormalizedCache.renewed`,
an empty cache over the same function for a test that wants isolation.

One function of the spelling problem is enough: the key is built once per call
from the signature, so a caller cannot reintroduce a second key for one
realization by rewriting the call. A ``**kwargs`` parameter is refused at
decoration time — it binds to a ``dict``, which is not hashable and has no
canonical order — and so is a ``keys`` entry that names no parameter. The
cache is a plain callable, not a descriptor: it decorates module-level
functions, not methods (``self`` would never bind).
"""

from __future__ import annotations

import functools
import inspect
import threading
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Mapping,
    NamedTuple,
    ParamSpec,
    TypeVar,
)

__all__ = ["CacheInfo", "NormalizedCache", "normalized_cache"]

P = ParamSpec("P")
R = TypeVar("R")

KeyFunction = Callable[[Any], Hashable]


class CacheInfo(NamedTuple):
    """The counters :func:`functools.lru_cache` reports, in its order."""

    hits: int
    misses: int
    maxsize: int
    currsize: int


def _identity(value: Any) -> Hashable:
    return value


class NormalizedCache(Generic[P, R]):
    """The cached callable :func:`normalized_cache` builds.

    Least-recently-used eviction at ``maxsize`` entries. Two threads that miss
    on the same key at once both call the function, as with
    :func:`functools.lru_cache`; the later result replaces the earlier one.
    """

    #: what :func:`functools.update_wrapper` carries over from the function
    __wrapped__: Callable[P, R]
    __name__: str
    __qualname__: str
    __doc__: str | None

    def __init__(
        self,
        fn: Callable[P, R],
        *,
        maxsize: int,
        keys: Mapping[str, KeyFunction],
    ) -> None:
        if maxsize < 1:
            raise TypeError(f"maxsize must be at least 1, got {maxsize}")
        signature = inspect.signature(fn)
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                raise TypeError(
                    f"{fn.__qualname__} takes **{parameter.name}: a keyword-variadic "
                    "parameter binds to a dict, which has no hashable canonical form"
                )
        unknown = sorted(set(keys) - set(signature.parameters))
        if unknown:
            raise TypeError(
                f"{fn.__qualname__} has no parameter named {unknown} to key on"
            )
        functools.update_wrapper(self, fn)
        self._signature = signature
        self._keys: dict[str, KeyFunction] = dict(keys)
        self._maxsize = maxsize
        self._entries: OrderedDict[tuple[Hashable, ...], R] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def key_of(self, *args: P.args, **kwargs: P.kwargs) -> tuple[Hashable, ...]:
        """The cache key a call binds to: every parameter's value, defaults
        applied, in signature order, each passed through its ``keys`` function
        when one is declared."""
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = []
        for name in self._signature.parameters:
            part = self._keys.get(name, _identity)(bound.arguments[name])
            try:
                hash(part)
            except TypeError as error:
                raise TypeError(
                    f"{self.__qualname__}: argument {name}={part!r} is not hashable; "
                    f"pass keys={{{name!r}: <canonical form>}} to normalized_cache"
                ) from error
            key.append(part)
        return tuple(key)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        key = self.key_of(*args, **kwargs)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self._hits += 1
                return self._entries[key]
            self._misses += 1
        value = self.__wrapped__(*args, **kwargs)
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
        return value

    def cache_info(self) -> CacheInfo:
        with self._lock:
            return CacheInfo(
                self._hits, self._misses, self._maxsize, len(self._entries)
            )

    def cache_clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def renewed(self) -> "NormalizedCache[P, R]":
        """An empty cache over the same function, same size, same key functions
        — what a test binds in place of the module's to keep its loads apart
        from the session's without touching the session's entries."""
        return NormalizedCache(self.__wrapped__, maxsize=self._maxsize, keys=self._keys)


def normalized_cache(
    *,
    maxsize: int,
    keys: Mapping[str, KeyFunction] | None = None,
) -> Callable[[Callable[P, R]], NormalizedCache[P, R]]:
    """Decorate ``fn`` with a :class:`NormalizedCache` of ``maxsize`` entries.

    ``keys`` maps a parameter name to the function that turns its value into
    the hashable canonical form the key uses; parameters not named are keyed
    on their value as bound.
    """

    def decorate(fn: Callable[P, R]) -> NormalizedCache[P, R]:
        return NormalizedCache(fn, maxsize=maxsize, keys=keys or {})

    return decorate
