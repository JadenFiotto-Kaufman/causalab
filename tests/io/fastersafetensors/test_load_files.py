"""``load_files``: a sharded checkpoint as one call."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import safetensors.torch as reference
import torch

import causalab.io.fastersafetensors.torch as fst
from causalab.io.fastersafetensors import _core
from causalab.io.fastersafetensors.errors import FormatError
from tests.io.fastersafetensors.strategies import same_dict

pytestmark = pytest.mark.unit

DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.int64,
    torch.uint8,
    torch.bool,
    torch.float16,
]


@pytest.fixture(scope="module")
def shards(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[Path], dict[str, torch.Tensor]]:
    root = tmp_path_factory.mktemp("ckpt")
    gen = torch.Generator().manual_seed(7)
    paths: list[Path] = []
    everything: dict[str, torch.Tensor] = {}
    counts = [3, 40, 12, 1, 25]
    for shard, count in enumerate(counts):
        tensors: dict[str, torch.Tensor] = {}
        for i in range(count):
            dtype = DTYPES[(shard + i) % len(DTYPES)]
            shape = [
                int(torch.randint(0, 40, (1,), generator=gen)) for _ in range(i % 3)
            ]
            raw = torch.randint(
                0,
                256,
                (
                    int(torch.tensor(shape).prod()) * dtype.itemsize
                    if shape
                    else dtype.itemsize,
                ),
                dtype=torch.uint8,
                generator=gen,
            )
            tensors[f"model.layers.{shard}.block.{i}.weight"] = raw.view(dtype).reshape(
                shape
            )
        p = root / f"model-{shard:05d}-of-{len(counts):05d}.safetensors"
        reference.save_file(tensors, str(p), {"format": "pt"})
        paths.append(p)
        everything.update(tensors)
    return paths, everything


def test_union_of_files(shards: tuple[list[Path], dict[str, torch.Tensor]]) -> None:
    paths, everything = shards
    got = fst.load_files(paths)
    assert same_dict(got, everything)
    union: dict[str, torch.Tensor] = {}
    for p in paths:
        union.update(fst.load_file(p))
    assert same_dict(got, union)
    assert list(got) == sorted(everything)


def test_keys_subset_is_one_job_over_exactly_the_wanted_bytes(
    shards: tuple[list[Path], dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, everything = shards
    names = sorted(everything)[::7]
    calls: list[list[_core.FileJob]] = []
    original = _core.read_job

    def recording(
        files: list[_core.FileJob], *args: Any, **kwargs: Any
    ) -> _core.ReadReport:
        calls.append(files)
        return original(files, *args, **kwargs)

    monkeypatch.setattr(_core, "read_job", recording)
    got = fst.load_files(paths, keys=names)
    assert list(got) == names
    assert same_dict(got, {k: everything[k] for k in names})
    assert len(calls) == 1
    (files,) = calls
    assert [Path(p) for p, _ in files] == [
        p for p in paths if any(k in fst.load_file(p) for k in names)
    ]
    wanted = sum(everything[k].numel() * everything[k].element_size() for k in names)
    assert (
        sum(nbytes for _, transfers in files for _, nbytes, *_ in transfers) == wanted
    )
    assert all(dst is None for _, transfers in files for _, _, _, dst, _ in transfers)


def test_whole_checkpoint_is_one_job(
    shards: tuple[list[Path], dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, everything = shards
    calls = 0
    original = _core.read_job

    def counting(*args: Any, **kwargs: Any) -> _core.ReadReport:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_core, "read_job", counting)
    got = fst.load_files(paths)
    assert calls == 1
    assert same_dict(got, everything)


def test_duplicate_names_across_files_are_refused(tmp_path: Path) -> None:
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    fst.save_file({"w": torch.zeros(2), "x": torch.ones(1)}, a)
    fst.save_file({"w": torch.zeros(2)}, b)
    with pytest.raises(FormatError, match="'w' appears in both"):
        fst.load_files([a, b])


def test_missing_key_and_empty_request(
    shards: tuple[list[Path], dict[str, torch.Tensor]],
) -> None:
    paths, _ = shards
    with pytest.raises(KeyError, match=r"not\.a\.tensor"):
        fst.load_files(paths, keys=["not.a.tensor"])
    assert fst.load_files(paths, keys=[]) == {}
    assert fst.load_files([]) == {}


@pytest.mark.cuda
def test_load_files_to_cuda(shards: tuple[list[Path], dict[str, torch.Tensor]]) -> None:
    paths, everything = shards
    got = fst.load_files(paths, device="cuda")
    assert all(v.device.type == "cuda" for v in got.values())
    assert same_dict({k: v.cpu() for k, v in got.items()}, everything)
    assert same_dict({k: v.cpu() for k, v in got.items()}, fst.load_files(paths))


@pytest.mark.cuda
def test_load_files_to_cuda_allocates_only_destinations(
    shards: tuple[list[Path], dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = torch.empty
    allocations = []

    def empty(size, **kwargs):
        assert not isinstance(size, int), "a whole-batch reservation is unnecessary"
        allocations.append(tuple(size))
        return original(size, **kwargs)

    paths, everything = shards
    # Patched for the load only: the comparison helpers allocate too.
    with monkeypatch.context() as patched:
        patched.setattr(torch, "empty", empty)
        got = fst.load_files(paths, device="cuda:0")
    assert same_dict({k: v.cpu() for k, v in got.items()}, everything)
    assert sorted(allocations) == sorted(tuple(t.shape) for t in everything.values())


def test_available_bytes_counts_torch_reserved_cache() -> None:
    """After a caller warms torch's allocator the driver shows little free;
    the cache it reserved is still available to this load."""
    from causalab.io.fastersafetensors._files import available_bytes

    assert available_bytes(free=15, reserved=70, allocated=1) == 84
    assert available_bytes(free=15, reserved=0, allocated=0) == 15
    # reserved never below allocated in torch's accounting; clamp anyway
    assert available_bytes(free=15, reserved=5, allocated=9) == 15


def test_device_headroom_reads_all_three_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from causalab.io.fastersafetensors import _files

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (15, 80))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 70)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: 1)
    assert _files.device_headroom(0) == 84


def test_load_allocates_tensors_without_a_contiguous_batch_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model a fragmented CUDA cache: tensor blocks fit; a batch block does not."""
    from dataclasses import replace

    from causalab.io.fastersafetensors import _files

    path = tmp_path / "batch.safetensors"
    tensors = {"a": torch.arange(32), "b": torch.arange(16)}
    fst.save_file(tensors, path)
    original_empty = torch.empty
    original_execute = _files.Plan.execute
    allocations = []

    def fragmented_empty(size, *, dtype, device):
        if isinstance(size, int):
            raise torch.OutOfMemoryError("no contiguous batch block")
        allocations.append(tuple(size))
        return original_empty(size, dtype=dtype, device="cpu")

    def host_execute(self, files, device):
        return original_execute(replace(self, staging=None), files, torch.device("cpu"))

    monkeypatch.setattr(torch, "empty", fragmented_empty)
    monkeypatch.setattr(_files, "device_headroom", lambda index: 32 << 30)
    monkeypatch.setattr(_files, "sync_torch", lambda device: None)
    monkeypatch.setattr(_files.Plan, "execute", host_execute)
    got = fst.load_files([path], device="cuda:0")
    assert same_dict(got, tensors)
    assert sorted(allocations) == [(16,), (32,)]
