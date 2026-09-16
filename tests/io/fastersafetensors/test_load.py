"""``load``/``load_file`` agree with the reference in both directions."""

from __future__ import annotations

from pathlib import Path

import pytest
import safetensors.torch as reference
import torch
from hypothesis import given

import causalab.io.fastersafetensors.torch as fst
from causalab.io.fastersafetensors.errors import FormatError, StorageError
from tests.io.fastersafetensors.strategies import (
    SETTINGS,
    REFERENCE_DTYPES,
    metadata,
    same_dict,
    tensor_dicts,
)

pytestmark = pytest.mark.property


@SETTINGS
@given(tensor_dicts(), metadata)
def test_load_inverts_save(
    tensors: dict[str, torch.Tensor], meta: dict[str, str] | None
) -> None:
    loaded = fst.load(fst.save(tensors, meta))
    assert same_dict(loaded, tensors)
    assert list(loaded) == sorted(tensors)


@SETTINGS
@given(tensor_dicts(min_size=1, dtypes=REFERENCE_DTYPES))
def test_load_file_both_directions(
    tmp_path_factory: pytest.TempPathFactory, tensors: dict[str, torch.Tensor]
) -> None:
    d = tmp_path_factory.mktemp("both")
    ours, theirs = d / "ours.safetensors", d / "theirs.safetensors"
    fst.save_file(tensors, ours, {"format": "pt"})
    reference.save_file(tensors, str(theirs), {"format": "pt"})
    assert ours.read_bytes() == theirs.read_bytes()
    assert same_dict(fst.load_file(theirs), reference.load_file(str(theirs)))
    assert same_dict(reference.load_file(str(ours)), fst.load_file(ours))
    assert same_dict(fst.load_file(ours), tensors)


def test_truncated_file_is_a_format_error(tmp_path: Path) -> None:
    p = tmp_path / "t.safetensors"
    fst.save_file({"a": torch.arange(100.0)}, p)
    whole = p.read_bytes()
    for cut in (3, 40, len(whole) - 1):
        p.write_bytes(whole[:cut])
        with pytest.raises(FormatError):
            fst.load_file(p)
        with pytest.raises(FormatError):
            fst.load(whole[:cut])
        with pytest.raises(FormatError):
            fst.safe_open(p)


def test_missing_file_is_a_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError) as info:
        fst.load_file(tmp_path / "missing.safetensors")
    assert isinstance(info.value, OSError)
    with pytest.raises(StorageError):
        fst.safe_open(tmp_path / "missing.safetensors")


def test_garbage_header_is_a_format_error() -> None:
    with pytest.raises(FormatError):
        fst.load(b"\x08\x00\x00\x00\x00\x00\x00\x00[]      ")
    with pytest.raises(FormatError):
        fst.load(b'\x10\x00\x00\x00\x00\x00\x00\x00{"a":{"dtype":"Q"}}')


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_load_file_to_mps(tmp_path: Path) -> None:
    t = {"a": torch.randn(8, 8), "b": torch.arange(5, dtype=torch.int32)}
    p = tmp_path / "m.safetensors"
    fst.save_file(t, p)
    out = fst.load_file(p, device="mps")
    assert all(v.device.type == "mps" for v in out.values())
    assert same_dict({k: v.cpu() for k, v in out.items()}, t)


@pytest.mark.cuda
def test_load_file_to_cuda(tmp_path: Path) -> None:
    t = {"a": torch.randn(256, 256), "b": torch.arange(1000), "e": torch.empty(0, 4)}
    p = tmp_path / "c.safetensors"
    reference.save_file(t, str(p))
    out = fst.load_file(p, device="cuda")
    assert all(v.device.type == "cuda" for v in out.values())
    assert same_dict({k: v.cpu() for k, v in out.items()}, t)
    assert same_dict(
        fst.load_file(p, device=0), reference.load_file(str(p), device="cuda:0")
    )


@pytest.mark.cuda
def test_load_file_to_cuda_equals_cpu_load(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(3)
    t = {
        "big": torch.randn(
            3000, 3000, generator=gen
        ),  # 36 MB: more than one staging buffer
        "odd": torch.randint(0, 255, (777,), dtype=torch.uint8, generator=gen),
        "bf": torch.randn(64, 64, generator=gen).to(torch.bfloat16),
        "e": torch.empty(0, 4),
    }
    p = tmp_path / "c.safetensors"
    fst.save_file(t, p)
    on_cpu = fst.load_file(p)
    on_gpu = fst.load_file(p, device="cuda:0")
    assert all(v.device == torch.device("cuda", 0) for v in on_gpu.values())
    assert same_dict({k: v.cpu() for k, v in on_gpu.items()}, on_cpu)
