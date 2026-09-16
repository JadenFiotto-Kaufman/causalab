"""The extension's functions, directly."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from causalab.io.fastersafetensors import _core
from causalab.io.fastersafetensors.errors import (
    CudaError,
    FormatError,
    ReadError,
    StorageError,
    WriteError,
)

pytestmark = pytest.mark.unit


def test_build_then_parse() -> None:
    header, order = _core.build_header(
        [("b", "F32", [2], 8), ("a", "I64", [1], 8), ("c", "F32", [0, 3], 0)],
        [("k", "v")],
    )
    assert order == [1, 0, 2]  # I64 first, then F32 by name
    assert len(header) % 8 == 0
    header_len, tensors, metadata = _core.parse_header(header + bytes(16))
    assert header_len == len(header) - 8
    assert metadata == [("k", "v")]
    assert [
        (n, d, s, st - header_len - 8, en - header_len - 8)
        for n, d, s, st, en in tensors
    ] == [
        ("a", "I64", [1], 0, 8),
        ("b", "F32", [2], 8, 16),
        ("c", "F32", [0, 3], 16, 16),
    ]
    assert _core.parse_header(_core.build_header([], None)[0])[2] is None
    assert _core.parse_header(_core.build_header([], [])[0])[2] == []


def test_build_header_errors() -> None:
    with pytest.raises(FormatError, match="unknown dtype"):
        _core.build_header([("a", "Q7", [1], 1)])
    with pytest.raises(FormatError, match="needs 8"):
        _core.build_header([("a", "F32", [2], 7)])
    with pytest.raises(FormatError, match="twice"):
        _core.build_header([("a", "F32", [1], 4), ("a", "F32", [1], 4)])


def test_parse_file_header_reads_only_the_header(tmp_path: Path) -> None:
    header, _ = _core.build_header([("a", "U8", [4], 4)])
    p = tmp_path / "f"
    p.write_bytes(header + b"wxyz")
    assert _core.parse_file_header(str(p)) == _core.parse_header(header + b"wxyz")
    p.write_bytes(header + b"wxy")
    with pytest.raises(FormatError, match="promises 4"):
        _core.parse_file_header(str(p))
    p.write_bytes(header[:5])
    with pytest.raises(FormatError):
        _core.parse_file_header(str(p))
    with pytest.raises(StorageError):
        _core.parse_file_header(str(tmp_path / "missing"))


def _read(
    files: list[_core.FileJob], transport: str = "pread", device: int | None = None
) -> _core.ReadReport:
    """``read_job`` under a fixed host plan: 2 files in flight, 512-byte
    pieces, 2 readers per file, no staging."""
    return _core.read_job(files, 2, 512, 2, transport, None, device)


def test_read_job_fills_host_tensors(tmp_path: Path) -> None:
    p = tmp_path / "data"
    data = bytes(range(256)) * 16
    p.write_bytes(data)
    out = torch.empty(4096, dtype=torch.uint8)
    base = out.data_ptr()
    transfers = [(i * 1024, 1024, base + i * 1024, None, None) for i in range(4)]
    report = _read([(str(p), transfers)])
    assert report["bytes_read"] == 4096
    assert report["pieces"] == 8  # 4 transfers split at 512
    assert report["files_opened"] == 1
    assert bytes(out.numpy()) == data


def test_read_job_out_of_range_names_the_range(tmp_path: Path) -> None:
    p = tmp_path / "data"
    p.write_bytes(bytes(4096))
    out = torch.empty(512, dtype=torch.uint8)
    with pytest.raises(StorageError, match=r"4000\.\.4512") as info:
        _read([(str(p), [(4000, 512, out.data_ptr(), None, None)])])
    assert "outside" in str(info.value)


def test_read_job_argument_errors(tmp_path: Path) -> None:
    p = tmp_path / "data"
    p.write_bytes(bytes(1024))
    out = torch.empty(1024, dtype=torch.uint8)
    base = out.data_ptr()
    with pytest.raises(ValueError, match="overlap"):
        _read(
            [(str(p), [(0, 512, base, None, None), (512, 512, base + 100, None, None)])]
        )
    with pytest.raises(ValueError, match="null"):
        _read([(str(p), [(0, 512, 0, None, None)])])
    with pytest.raises(ValueError, match="without a device"):
        _read([(str(p), [(0, 512, base, 0, None)])])
    with pytest.raises(ValueError, match="transport"):
        _read([(str(p), [])], transport="carrier pigeon")
    with pytest.raises(StorageError):
        _read([(str(tmp_path / "missing"), [])])
    # a device destination with a plan that stages nothing: the runtime is
    # loaded first (CudaError where there is none), then the engine refuses.
    # Without a device the CudaError is the library missing, or — where the
    # CUDA torch wheels put a libcudart on a host with no driver, as on CI —
    # cudaSetDevice failing; both are the runtime refusing before any I/O.
    if torch.cuda.is_available():
        with pytest.raises(ReadError, match="staging"):
            _read([(str(p), [(0, 512, base, 0, None)])], device=0)
    else:
        with pytest.raises(CudaError, match="libcudart is not available|cudaSetDevice"):
            _read([(str(p), [(0, 512, base, 0, None)])], device=0)


def test_write_object_streams_header_then_parts(tmp_path: Path) -> None:
    p = tmp_path / "out"
    a = torch.arange(4, dtype=torch.uint8)
    b = torch.tensor([9, 8], dtype=torch.uint8)
    specs = [("a", "U8", [4], 4), ("e", "U8", [0], 0), ("b", "U8", [2], 2)]
    parts = [(a.data_ptr(), 4, False), (0, 0, False), (b.data_ptr(), 2, False)]
    header, _ = _core.build_header(specs, [("k", "v")])
    written = _core.write_object(str(p), specs, parts, [("k", "v")], True, True)
    assert written == len(header) + 6
    assert p.read_bytes() == header + bytes([0, 1, 2, 3, 9, 8])
    assert sorted(x.name for x in tmp_path.iterdir()) == ["out"]  # no temp left behind
    with pytest.raises(ValueError, match="null"):
        _core.write_object(str(p), specs[:1], [(0, 4, False)])
    with pytest.raises(WriteError, match="holds 2 bytes"):
        _core.write_object(str(p), specs[:1], [(b.data_ptr(), 2, False)])
    with pytest.raises(WriteError, match="parts given"):
        _core.write_object(str(p), specs, parts[:1])
    with pytest.raises(FormatError, match="unknown dtype"):
        _core.write_object(str(p), [("a", "Q7", [4], 4)], parts[:1])
    with pytest.raises(StorageError):
        _core.write_object(str(tmp_path / "no" / "dir"), [], [])
    with pytest.raises(ValueError, match="without a device"):
        _core.write_object(str(p), specs[:1], [(a.data_ptr(), 4, True)])


def test_probe_cuda_shape() -> None:
    cuda = _core.probe_cuda()
    assert set(cuda) == {"cudart", "cufile", "devices"}
    for library in ("cudart", "cufile"):
        available, detail = cuda[library]
        assert isinstance(available, bool) and detail
    assert cuda["devices"] >= 0
    assert (cuda["devices"] > 0) == torch.cuda.is_available()
