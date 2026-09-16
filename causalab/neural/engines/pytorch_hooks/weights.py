"""Weight materialization for the reference engine's loader (``loading.py``).

The stock path — ``from_pretrained`` on the CPU, then ``Module.to(device)`` —
moves every byte through one thread. This loader reads the wanted tensors of a
safetensors checkpoint straight onto the device with several shards in flight:
one planned Rust read of the whole checkpoint with the GIL released
(:class:`FastersafetensorsReader` over :mod:`causalab.io.fastersafetensors`,
what :func:`default_reader` returns), or sixteen ``safe_open(device=…)``
threads when a caller passes :class:`SafetensorsReader` explicitly.

:func:`load_pretrained` hands the tensors to transformers through its public
``state_dict=`` entry, so every checkpoint rename and fusion in its conversion
mapping still runs. Which checkpoint keys the model wants is decided with
transformers' own renamer against a meta-device instance — a multimodal
checkpoint's vision tower and MTP head are never read — and a key the model
wanted and did not get is refused, never silently initialized.

Memory is the same as the stock path's: a tensor is allocated once, on the
device, in the reading thread, and becomes the parameter without a copy.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import torch

from causalab.protocol.errors import ProtocolError

__all__ = [
    "CheckpointReader",
    "FastersafetensorsReader",
    "Prefetch",
    "SafetensorsReader",
    "Shard",
    "ShardReader",
    "TensorHeader",
    "checkpoint_files",
    "default_reader",
    "load_pretrained",
    "read_header",
    "shard_plan",
    "wanted_keys",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The checkpoint on disk
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TensorHeader:
    """One tensor's entry in a safetensors header: the strings the file
    carries, so a lazy stand-in can answer ``get_dtype`` / ``get_shape``
    without touching the data section."""

    dtype: str
    shape: tuple[int, ...]


def read_header(path: Path) -> dict[str, TensorHeader]:
    """The tensor table of a safetensors file — a pure header read (8-byte
    little-endian length, then JSON; ``__metadata__`` is not a tensor).

    Format reference: https://github.com/huggingface/safetensors#format.
    """
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ProtocolError(
                "P2", f"{path} is not a safetensors file (truncated header)"
            )
        (length,) = struct.unpack("<Q", prefix)
        table = json.loads(fh.read(length))
    return {
        name: TensorHeader(dtype=str(entry["dtype"]), shape=tuple(entry["shape"]))
        for name, entry in table.items()
        if name != "__metadata__"
    }


def checkpoint_files(key: str, revision: str) -> tuple[Path, ...] | None:
    """The safetensors shards of a checkpoint, resolved through transformers'
    own file resolver (``cached_file`` / ``get_checkpoint_shard_files``): a
    local directory as is, a Hub id through the cache (``HF_HUB_OFFLINE``
    honoured), the index's ``weight_map`` naming the shards when there is one,
    ``model.safetensors`` otherwise. The same route the stock loader takes —
    ``huggingface_hub.snapshot_download`` is not it: for Xet-backed repos it
    asks the Hub for a read token that anonymous callers are refused (CI has
    no token), while transformers' route downloads them.

    ``None`` when the checkpoint ships no safetensors — the caller then takes
    the stock loader, which knows the other formats.
    """
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
    from transformers.utils.hub import cached_file, get_checkpoint_shard_files

    index = cached_file(
        key,
        SAFE_WEIGHTS_INDEX_NAME,
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    if index is not None:
        shards, _metadata = get_checkpoint_shard_files(key, index, revision=revision)
        # transformers types the list as optional; an index that names no shard
        # is not a safetensors checkpoint the fast path can read
        return tuple(Path(shard) for shard in shards) if shards else None
    single = cached_file(
        key,
        SAFE_WEIGHTS_NAME,
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    return (Path(single),) if single is not None else None


# ---------------------------------------------------------------------------
# What the model wants, and where it is
# ---------------------------------------------------------------------------


def wanted_keys(meta_model: Any, keys: Iterable[str]) -> frozenset[str]:
    """The checkpoint keys ``meta_model`` will consume — decided with
    transformers' own renamer, so the answer is the one its loading loop
    reaches (``convert_and_load_state_dict_in_model``, transformers 5.16):
    every ``WeightRenaming`` in turn, at most one ``WeightConverter``, the
    ``base_model_prefix`` added or stripped, then membership in the meta
    state dict; a key that already names a parameter is kept as is.

    A multimodal checkpoint's vision tower and MTP head fall out here: neither
    renames onto a text-model parameter, and the loader never reads them.
    """
    from transformers.core_model_loading import (
        WeightConverter,
        WeightRenaming,
        rename_source_key,
    )
    from transformers.conversion_mapping import get_model_conversion_mapping

    mapping = get_model_conversion_mapping(meta_model, None, None)
    renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]
    meta_state = meta_model.state_dict()
    prefix = meta_model.base_model_prefix
    wanted: set[str] = set()
    for key in keys:
        renamed, _ = rename_source_key(key, renamings, converters, prefix, meta_state)
        if renamed not in meta_state and key in meta_state:
            renamed, _ = rename_source_key(key, [], [], prefix, meta_state)
        if renamed in meta_state:
            wanted.add(key)
    return frozenset(wanted)


@dataclasses.dataclass(frozen=True)
class Shard:
    """One file and the keys the model wants out of it, in header order."""

    path: Path
    keys: tuple[str, ...]
    headers: Mapping[str, TensorHeader]


def shard_plan(
    tables: Sequence[tuple[Path, Mapping[str, TensorHeader]]],
    wanted: frozenset[str],
) -> tuple[tuple[Shard, ...], frozenset[str]]:
    """Partition ``wanted`` over the files whose headers ``tables`` are: the
    shards that carry at least one wanted key (a file the model wants nothing
    from is never opened again), and the wanted keys no file carries — the
    caller's refusal, by name.

    A key in two files is refused: the format promises uniqueness across a
    sharded checkpoint, and a loader that picked one silently would load
    whichever file sorted first.
    """
    seen: dict[str, Path] = {}
    shards: list[Shard] = []
    for path, headers in tables:
        keys = tuple(name for name in headers if name in wanted)
        for name in keys:
            if name in seen:
                raise ProtocolError(
                    "P2",
                    f"tensor {name!r} appears in both {seen[name]} and {path}; "
                    "a sharded checkpoint carries each tensor once",
                )
            seen[name] = path
        if keys:
            shards.append(Shard(path=path, keys=keys, headers=headers))
    return tuple(shards), wanted - frozenset(seen)


# ---------------------------------------------------------------------------
# Readers: one shard's wanted tensors onto the device
# ---------------------------------------------------------------------------


@runtime_checkable
class ShardReader(Protocol):
    """Read the named tensors of one shard onto ``device``, each as a tensor
    the caller owns outright (torch-allocated, nothing to release later).

    ``concurrency`` is how many shards the loader may hand a reader at once —
    the reader's own memory story decides it.
    """

    @property
    def concurrency(self) -> int: ...

    def read(
        self, path: Path, keys: Sequence[str], device: torch.device
    ) -> dict[str, torch.Tensor]: ...


@runtime_checkable
class CheckpointReader(Protocol):
    """Read every wanted tensor of every shard in one call — for a reader that
    plans its own concurrency across files (:class:`FastersafetensorsReader`)."""

    def read_all(
        self, shards: Sequence["Shard"], device: torch.device
    ) -> dict[str, torch.Tensor]: ...


@dataclasses.dataclass(frozen=True)
class SafetensorsReader:
    """``safe_open(device=…).get_tensor`` per key: one device allocation per
    tensor, the copy outside the GIL, so sixteen shards in flight cost no
    memory beyond the tensors themselves. The reference reader, taken only when
    a caller passes it; the default is :class:`FastersafetensorsReader`."""

    concurrency: int = 16

    def read(
        self, path: Path, keys: Sequence[str], device: torch.device
    ) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device=str(device)) as handle:
            return {key: handle.get_tensor(key) for key in keys}


@dataclasses.dataclass(frozen=True)
class FastersafetensorsReader:
    """:func:`causalab.io.fastersafetensors.torch.load_files` over the whole
    checkpoint: one call, the wanted keys only, files in flight and the
    transport decided by its planner for this machine, the reads in Rust with
    the GIL released. Same memory story as :class:`SafetensorsReader` — every
    tensor is allocated by torch on the device and filled in place. What
    :func:`default_reader` returns."""

    def read_all(
        self, shards: Sequence["Shard"], device: torch.device
    ) -> dict[str, torch.Tensor]:
        from causalab.io.fastersafetensors.torch import load_files

        return load_files(
            [shard.path for shard in shards],
            device=str(device),
            keys=[key for shard in shards for key in shard.keys],
        )


def default_reader() -> ShardReader | CheckpointReader:
    """The reader :func:`load_pretrained` takes when none is passed: the
    planned Rust read. :class:`SafetensorsReader` is the explicit alternative."""
    return FastersafetensorsReader()


# ---------------------------------------------------------------------------
# The lazy stand-ins transformers materializes
# ---------------------------------------------------------------------------


class Prefetch:
    """Every shard's read submitted at once (per shard, bounded by a
    :class:`ShardReader`'s ``concurrency``; or as one whole-checkpoint read
    for a :class:`CheckpointReader`), each tensor handed out exactly once.

    The reads start on the first :meth:`take`, not on construction: transformers
    warms its caching allocator with one allocation the size of the model right
    before it materializes anything, and reads that started earlier would fill
    the device before that allocation.
    """

    def __init__(
        self,
        shards: Sequence[Shard],
        reader: ShardReader | CheckpointReader,
        device: torch.device,
    ) -> None:
        self._shards = tuple(shards)
        self._reader = reader
        self._device = device
        self._shard_of = {key: shard for shard in shards for key in shard.keys}
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._futures: dict[Path, Future[dict[str, torch.Tensor]]] = {}

    def __enter__(self) -> "Prefetch":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _start(self) -> None:
        with self._lock:
            if self._pool is not None:
                return
            reader = self._reader
            if isinstance(reader, CheckpointReader):
                # one read for the whole checkpoint; every shard's tensors
                # arrive together, so every path shares the one future
                self._pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="causalab-weights"
                )
                whole = self._pool.submit(reader.read_all, self._shards, self._device)
                for shard in self._shards:
                    self._futures[shard.path] = whole
                return
            self._pool = ThreadPoolExecutor(
                max_workers=max(1, reader.concurrency),
                thread_name_prefix="causalab-weights",
            )
            for shard in self._shards:
                self._futures[shard.path] = self._pool.submit(
                    reader.read, shard.path, shard.keys, self._device
                )

    def take(self, key: str) -> torch.Tensor:
        """The tensor for ``key``, ownership included: the reference leaves
        this object, so a fully taken shard holds nothing."""
        self._start()
        shard = self._shard_of[key]
        tensors = self._futures[shard.path].result()
        with self._lock:
            return tensors.pop(key)

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


class _LazyWeight:
    """What transformers sees in the ``state_dict``: the slice surface it
    materializes through (``[...]``, ``get_shape``, ``get_dtype``), backed by
    a :class:`Prefetch`. ``__getitem__`` is the one call that touches data."""

    __slots__ = ("_key", "_header", "_prefetch")

    def __init__(self, key: str, header: TensorHeader, prefetch: Prefetch) -> None:
        self._key = key
        self._header = header
        self._prefetch = prefetch

    def get_shape(self) -> list[int]:
        return list(self._header.shape)

    def get_dtype(self) -> str:
        return self._header.dtype

    def __getitem__(self, index: Any) -> torch.Tensor:
        return self._prefetch.take(self._key)[index]


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def load_pretrained(
    key: str,
    revision: str,
    *,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str | None,
    reader: ShardReader | CheckpointReader | None = None,
) -> Any:
    """``AutoModelForCausalLM.from_pretrained(key, revision, dtype,
    attn_implementation)`` with its weights on ``device`` — the stock result,
    read as the module docstring describes.

    The stock loader takes over when the checkpoint ships no safetensors.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    target = torch.device(device)
    # ``None`` leaves the attention backend to transformers' default
    attention = (
        {"attn_implementation": attn_implementation}
        if attn_implementation is not None
        else {}
    )
    files = checkpoint_files(key, revision)
    if files is None:
        logger.info(
            "weights: stock loader (no safetensors) key=%s revision=%s", key, revision
        )
        return AutoModelForCausalLM.from_pretrained(
            key,
            revision=revision,
            dtype=dtype,
            **attention,
            device_map={"": device},
        )

    config = AutoConfig.from_pretrained(key, revision=revision)
    with torch.device("meta"):
        meta = AutoModelForCausalLM.from_config(config, dtype=dtype, **attention)
    tables = [(path, read_header(path)) for path in files]
    wanted = wanted_keys(meta, (name for _, table in tables for name in table))
    shards, absent = shard_plan(tables, wanted)
    if absent:
        raise ProtocolError(
            "P2",
            f"{key}@{revision}: the model wants {len(absent)} tensor(s) no shard "
            f"carries, first {sorted(absent)[0]!r}",
        )
    chosen = reader if reader is not None else default_reader()
    logger.info(
        "weights: reader=%s shards=%d tensors=%d device=%s key=%s",
        type(chosen).__name__,
        len(shards),
        len(wanted),
        target,
        key,
    )
    with Prefetch(shards, chosen, target) as prefetch:
        state = {
            name: _LazyWeight(name, shard.headers[name], prefetch)
            for shard in shards
            for name in shard.keys
        }
        model, info = type(meta).from_pretrained(
            None,
            config=meta.config,
            state_dict=state,
            dtype=dtype,
            **attention,
            device_map={"": device},
            output_loading_info=True,
        )
    _check_complete(info, key=key, revision=revision)
    # what the stock loader records; nothing hashed reads it, the receipt's
    # identity is the bundle's ``key``, but the two paths should not differ
    model.config.name_or_path = key
    return model


def _check_complete(info: Mapping[str, Any], *, key: str, revision: str) -> None:
    """A fast path that left a parameter at its random initialization would
    be the worst kind of wrong — numbers that run. transformers raises on a
    shape mismatch itself; what it only *reports* is a parameter it never saw
    (``missing``) and a tensor it had no parameter for (``unexpected``, which
    :func:`wanted_keys` should have excluded). Both are refused here, naming
    the first key."""
    missing = sorted(info.get("missing_keys", ()))
    unexpected = sorted(info.get("unexpected_keys", ()))
    problems = [
        (label, entries)
        for label, entries in (("missing", missing), ("unexpected", unexpected))
        if entries
    ]
    if problems:
        detail = "; ".join(
            f"{len(entries)} {label} (first {entries[0]!r})"
            for label, entries in problems
        )
        raise ProtocolError(
            "P2",
            f"{key}@{revision}: the weight reader and the model disagree about "
            f"the checkpoint — {detail}. Nothing was loaded.",
        )
