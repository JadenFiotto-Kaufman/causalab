"""``save`` produces the reference library's bytes."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
import safetensors.torch as reference
import torch
from hypothesis import given, settings

import causalab.io.fastersafetensors.torch as fst
from causalab.io.fastersafetensors.errors import FormatError
from tests.io.fastersafetensors.strategies import (
    SETTINGS,
    DTYPES,
    REFERENCE_DTYPES,
    metadata,
    same_dict,
    tensor_dicts,
    tensors,
)

pytestmark = pytest.mark.property


def _split(obj: bytes) -> tuple[dict[str, object], bytes]:
    (n,) = struct.unpack("<Q", obj[:8])
    return json.loads(obj[8 : 8 + n]), obj[8 + n :]


@settings(parent=SETTINGS, max_examples=300)
@given(tensor_dicts(dtypes=REFERENCE_DTYPES), metadata)
def test_save_matches_reference(
    tensors_: dict[str, torch.Tensor], meta: dict[str, str] | None
) -> None:
    ours = fst.save(tensors_, meta)
    if not tensors_ and meta == {}:
        # safetensors 0.8.0 writes `{},"__metadata__":{}}` here: invalid JSON
        # it cannot read back itself. We write `{"__metadata__":{}}`.
        assert fst.load(ours) == {}
        return
    theirs = reference.save(tensors_, meta)
    if meta is None or len(meta) <= 1:
        assert ours == theirs
    else:
        # the reference serializes metadata from a HashMap, so its key order
        # varies per process: compare the header structurally, the data exactly
        assert _split(ours) == _split(theirs)
        assert len(ours) == len(theirs)


@pytest.mark.parametrize(("meta", "key_present"), [(None, False), ({}, True)])
def test_metadata_none_versus_empty(
    meta: dict[str, str] | None, key_present: bool
) -> None:
    t = {"a": torch.zeros(1, dtype=torch.uint8)}
    out = fst.save(t, meta)
    assert (b'"__metadata__":{}' in out) is key_present
    assert out == reference.save(t, meta)
    assert (
        fst.save({}, None)
        == reference.save({}, None)
        == b"\x08" + bytes(7) + b"{}      "
    )


def test_metadata_keeps_insertion_order() -> None:
    header, _ = _split(fst.save({}, {"z": "1", "a": "2", "m": "3"}))
    meta = header["__metadata__"]
    assert isinstance(meta, dict)
    assert list(meta) == ["z", "a", "m"]


@SETTINGS
@given(tensors(dtypes=REFERENCE_DTYPES))
def test_every_supported_dtype_round_trips_through_reference(t: torch.Tensor) -> None:
    assert same_dict(reference.load(fst.save({"t": t})), {"t": t})


@pytest.mark.parametrize("dtype", sorted(set(DTYPES) - set(REFERENCE_DTYPES), key=str))
def test_dtypes_beyond_the_reference_round_trip_here(dtype: torch.dtype) -> None:
    """A dtype the installed reference cannot serialize still has its header
    name and round-trips through this library."""
    t = torch.arange(6, dtype=torch.uint8).view(torch.uint8)
    t = t[: 6 // dtype.itemsize * dtype.itemsize].view(dtype)
    assert same_dict(fst.load(fst.save({"t": t})), {"t": t})


def test_unsupported_dtype_is_a_format_error() -> None:
    unsupported = getattr(torch, "float8_e8m0fnu", None)
    if unsupported is None:
        pytest.skip("this torch has no dtype the format lacks")
    with pytest.raises(FormatError, match="float8_e8m0fnu"):
        fst.save({"x": torch.zeros(2, dtype=unsupported)})


def test_reserved_name_and_non_tensor_are_format_errors() -> None:
    with pytest.raises(FormatError, match="__metadata__"):
        fst.save({"__metadata__": torch.zeros(1)})
    not_a_tensor: dict[str, Any] = {"x": [1, 2, 3]}
    with pytest.raises(FormatError, match=r"not a torch\.Tensor"):
        fst.save(not_a_tensor)


def test_non_contiguous_tensors_are_packed() -> None:
    t = torch.arange(6.0).reshape(2, 3).t()
    assert not t.is_contiguous()
    assert fst.save({"a": t}) == reference.save({"a": t.contiguous()})


def test_serialize_payload_shape() -> None:
    t = {
        "a": torch.arange(6, dtype=torch.int32),
        "b": torch.ones(2, dtype=torch.float64),
    }
    payload = fst.serialize(t, {"k": "v"})
    assert payload.header[:8] == struct.pack("<Q", len(payload.header) - 8)
    parts = payload.parts()
    assert [p.nbytes for p in parts] == [16, 24]  # F64 before I32: descending dtype
    assert all(p.readonly for p in parts)
    assert payload.nbytes == len(payload.header) + 40
    assert payload.to_bytes() == fst.save(t, {"k": "v"})


def test_save_file_atomic_leaves_no_temp_file(tmp_path: Path) -> None:
    t = {"a": torch.arange(100.0), "b": torch.ones(3, dtype=torch.int8)}
    p = tmp_path / "a.safetensors"
    fst.save_file(t, p, {"format": "pt"}, atomic=True)
    assert [x.name for x in tmp_path.iterdir()] == ["a.safetensors"]
    assert p.read_bytes() == reference.save(t, {"format": "pt"})
    # an atomic write over an existing file replaces it whole
    fst.save_file({"c": torch.zeros(2)}, p, atomic=True)
    assert [x.name for x in tmp_path.iterdir()] == ["a.safetensors"]
    assert same_dict(fst.load_file(p), {"c": torch.zeros(2)})


def test_save_file_durable_round_trips(tmp_path: Path) -> None:
    t = {"a": torch.randn(16, 16), "b": torch.arange(5)}
    p = tmp_path / "d.safetensors"
    fst.save_file(t, p, durable=True)
    assert same_dict(fst.load_file(p), t)
    fst.save_file(t, p, durable=True, atomic=False)
    assert p.read_bytes() == reference.save(t)
    assert [x.name for x in tmp_path.iterdir()] == ["d.safetensors"]


def test_save_file_into_missing_directory_is_a_storage_error(tmp_path: Path) -> None:
    from causalab.io.fastersafetensors.errors import StorageError

    with pytest.raises(StorageError):
        fst.save_file({"a": torch.zeros(1)}, tmp_path / "no" / "such" / "f.safetensors")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cuda
def test_save_from_cuda_matches_cpu() -> None:
    t = {"a": torch.randn(64, 32), "b": torch.arange(10)}
    on_gpu = {k: v.cuda() for k, v in t.items()}
    assert fst.save(on_gpu) == reference.save(t)


@pytest.mark.cuda
def test_save_file_from_cuda_is_byte_identical_to_reference(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(11)
    t = {
        "big": torch.randn(4096, 4096, generator=gen),  # 64 MB: several staging chunks
        "small": torch.arange(7, dtype=torch.int16),
        "bf": torch.randn(33, 65, generator=gen).to(torch.bfloat16),
        "e": torch.empty(0, 2),
        "strided": torch.arange(24.0).reshape(4, 6).t(),
    }
    on_gpu = {k: v.cuda() for k, v in t.items()}
    ours, theirs = tmp_path / "ours.safetensors", tmp_path / "theirs.safetensors"
    fst.save_file(on_gpu, ours, {"format": "pt"})
    reference.save_file(
        {k: v.contiguous() for k, v in t.items()}, str(theirs), {"format": "pt"}
    )
    assert ours.read_bytes() == theirs.read_bytes()
    assert [x.name for x in tmp_path.iterdir()] == sorted(
        ["ours.safetensors", "theirs.safetensors"]
    )
    mixed = {"gpu": on_gpu["big"], "cpu": t["small"]}
    fst.save_file(mixed, ours, durable=True)
    assert same_dict(fst.load_file(ours), {"gpu": t["big"], "cpu": t["small"]})


def test_tensors_sharing_storage_are_each_written_whole(tmp_path: Path) -> None:
    """A deliberate divergence: the reference refuses tensors that share
    storage; here each is written whole and comes back as its own tensor."""
    base = torch.arange(6, dtype=torch.float32)
    tensors = {"a": base, "b": base.view(2, 3)}
    with pytest.raises(RuntimeError, match="share"):
        reference.save_file(tensors, str(tmp_path / "ref.safetensors"))
    path = tmp_path / "out.safetensors"
    fst.save_file(tensors, path)
    loaded = fst.load_file(path)
    assert same_dict(loaded, tensors)
    assert (
        loaded["a"].untyped_storage().data_ptr()
        != loaded["b"].untyped_storage().data_ptr()
    )
