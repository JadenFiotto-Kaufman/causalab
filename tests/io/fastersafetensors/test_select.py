"""Selections: ``load_files(select=...)``, ``get_sharded``, and what they read."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
import safetensors.torch as reference
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import causalab.io.fastersafetensors.torch as fst
from causalab.io.fastersafetensors import _core, _files
from causalab.io.fastersafetensors._dtypes import header_dtype
from causalab.io.fastersafetensors._select import (
    Selection,
    SelectItem,
    Shard,
    select,
    select_shards,
)
from causalab.io.fastersafetensors.errors import FormatError, ReadError, SelectError
from tests.io.fastersafetensors.strategies import SETTINGS, same_tensor, tensors

pytestmark = pytest.mark.property

CUT_DTYPES = [torch.float32, torch.int64, torch.uint8, torch.bfloat16, torch.int8]
Index = SelectItem | tuple[SelectItem, ...]


@st.composite
def cuts(draw: st.DrawFn) -> tuple[torch.Tensor, Index]:
    """A tensor and an index of ints, step-1 slices and at most one Ellipsis,
    every dimension covered by one of the three."""
    t = draw(tensors(dtypes=CUT_DTYPES, max_dims=4, max_extent=5))
    items: list[SelectItem] = []
    for size in t.shape:
        kinds = ["int", "slice", "full"] if size else ["slice", "full"]
        kind = draw(st.sampled_from(kinds))
        if kind == "int":
            items.append(draw(st.integers(-size, size - 1)))
        elif kind == "slice":
            bound = st.none() | st.integers(-size - 1, size + 1)
            items.append(slice(draw(bound), draw(bound)))
        else:
            items.append(slice(None))
    if draw(st.booleans()):
        at = draw(st.integers(0, len(items)))
        run = 0
        while at + run < len(items) and items[at + run] == slice(None):
            run += 1
        k = draw(st.integers(0, run))
        items[at : at + k] = [Ellipsis]
    if len(items) == 1 and draw(st.booleans()):
        return t, items[0]
    return t, tuple(items)


def packed(t: torch.Tensor) -> torch.Tensor:
    """``t`` in fresh contiguous memory. ``.contiguous()`` hands back an empty
    view unchanged, strides and all, which ``same_tensor`` cannot re-view."""
    return t.clone(memory_format=torch.contiguous_format)


def box_of(t: torch.Tensor, sel: Selection) -> torch.Tensor:
    """``t`` cut to the selection's box, by plain slicing."""
    return t[tuple(slice(lo, hi) for lo, hi in sel.ranges)]


def land(t: torch.Tensor, sel: Selection, reads: list[_core.ReadRow]) -> torch.Tensor:
    """What the engine would land for ``reads`` of ``t``: each read's bytes
    from the tensor's flat bytes, its placements (or the whole range) copied
    to ``dst``; then the view."""
    flat = t.contiguous().reshape(-1).view(torch.uint8)
    nbytes = box_of(t, sel).numel() * t.element_size()
    dest = torch.empty(nbytes, dtype=torch.uint8)
    for offset, length, dst, placements in reads:
        span = flat[offset : offset + length]
        for src, at, n in placements or [(0, 0, length)]:
            dest[dst + at : dst + at + n] = span[src : src + n]
    if nbytes:
        block = dest.view(t.dtype).reshape(sel.box_shape)
    else:
        block = torch.empty(sel.box_shape, dtype=t.dtype)
    return block[sel.view]


def reads_for(
    t: torch.Tensor, sel: Selection, path: str = "/tmp/x"
) -> tuple[list[_core.ReadRow], _core.CoalesceDict]:
    (reads,), summary = _core.select_reads(
        [(path, "t", list(t.shape), sel.ranges, header_dtype(t.dtype))]
    )
    return reads, summary


@SETTINGS
@given(cuts())
def test_selection_box_and_view_reproduce_the_index(
    cut: tuple[torch.Tensor, Index],
) -> None:
    t, index = cut
    sel = select(tuple(t.shape), index)
    expected = t[index]
    assert sel.result_shape == tuple(expected.shape)
    assert sel.wanted_bytes(t.element_size()) == expected.numel() * t.element_size()
    assert sel.is_plain
    assert same_tensor(packed(box_of(t, sel)[sel.view]), packed(expected))


@SETTINGS
@given(cuts())
def test_core_reads_land_exactly_the_selection(cut: tuple[torch.Tensor, Index]) -> None:
    t, index = cut
    sel = select(tuple(t.shape), index)
    reads, summary = reads_for(t, sel)
    assert same_tensor(packed(land(t, sel, reads)), packed(t[index]))
    box_bytes = box_of(t, sel).numel() * t.element_size()
    assert summary["wanted_bytes"] == box_bytes
    assert box_bytes <= summary["read_bytes"] <= t.numel() * t.element_size()
    assert summary["reads"] == len(reads) <= summary["runs"]
    # reads are in file order and disjoint; placements are a packed gather
    assert all(a[0] + a[1] <= b[0] for a, b in pairwise(reads))
    for _, length, _, placements in reads:
        if placements is None:
            continue
        assert placements[0][1] == 0 and len(placements) > 1
        for (src, dst, n), (src2, dst2, _) in pairwise(placements):
            assert src + n <= src2 and dst + n == dst2
        assert placements[-1][0] + placements[-1][2] == length


def test_inner_cut_coalesces_under_the_gap_policy() -> None:
    t = torch.zeros(8, 6, 4)
    reads, summary = reads_for(t, select(tuple(t.shape), (slice(None), 1, slice(1, 3))))
    # eight 8-byte runs 96 bytes apart: one read, gaps read through
    assert len(reads) == 1
    offset, length, dst, placements = reads[0]
    assert (offset, dst) == ((1 * 4 + 1) * 4, 0)  # element [0, 1, 1]
    assert placements is not None and len(placements) == 8
    assert summary == {
        "runs": 8,
        "reads": 1,
        "wanted_bytes": 64,
        "read_bytes": length,
        "amplification": length / 64,
    }
    # a full box and an outer cut are one plain read each
    for index in (Ellipsis, (slice(2, 5),), (3,)):
        reads, summary = reads_for(t, select(tuple(t.shape), index))
        assert len(reads) == 1 and reads[0][3] is None
        assert summary["read_bytes"] == summary["wanted_bytes"]


def test_select_reads_refusals() -> None:
    with pytest.raises(SelectError, match="outside its extent"):
        _core.select_reads([("/tmp/x", "t", [4, 4], [(0, 4), (2, 5)], "F32")])
    with pytest.raises(SelectError, match="ranges for a"):
        _core.select_reads([("/tmp/x", "t", [4, 4], [(0, 4)], "F32")])
    with pytest.raises(FormatError, match="unknown dtype"):
        _core.select_reads([("/tmp/x", "t", [4], [(0, 4)], "Q7")])
    assert _core.shard_ranges([4, 9], 1, 1, 3) == [(0, 4), (3, 6)]
    with pytest.raises(SelectError, match="does not divide"):
        _core.shard_ranges([4, 9], 1, 0, 2)


@pytest.fixture(scope="module")
def cut_file(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, torch.Tensor]]:
    p = tmp_path_factory.mktemp("select") / "f.safetensors"
    gen = torch.Generator().manual_seed(11)
    t = {
        "w": torch.randn(8, 6, 4, generator=gen),
        "v": torch.randint(-100, 100, (12, 5), dtype=torch.int16, generator=gen),
        "u": torch.randn(6, generator=gen).to(torch.bfloat16),
        "s": torch.tensor(3.0, dtype=torch.float64),
    }
    reference.save_file(t, str(p))
    return p, t


class Recorder:
    """Records every ``_core.read_job`` call's files and reports, then runs it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[list[_core.FileJob]] = []
        self.reports: list[_core.ReadReport] = []
        original = _core.read_job

        def recording(
            files: list[_core.FileJob], *args: Any, **kwargs: Any
        ) -> _core.ReadReport:
            self.calls.append(files)
            report = original(files, *args, **kwargs)
            self.reports.append(report)
            return report

        monkeypatch.setattr(_core, "read_job", recording)

    @property
    def transfers(self) -> list[_core.Transfer]:
        return [t for files in self.calls for _, ts in files for t in ts]

    @property
    def bytes_read(self) -> int:
        return sum(nbytes for _, nbytes, *_ in self.transfers)


@settings(parent=SETTINGS, max_examples=60)
@given(cuts())
def test_load_files_select_matches_reference(
    tmp_path_factory: pytest.TempPathFactory, cut: tuple[torch.Tensor, Index]
) -> None:
    t, index = cut
    p = tmp_path_factory.mktemp("cut") / "c.safetensors"
    reference.save_file({"t": t, "other": torch.arange(3.0)}, str(p))
    got = fst.load_files([p], select={"t": index})
    assert set(got) == {"other", "t"}
    assert same_tensor(got["t"], packed(reference.load_file(str(p))["t"][index]))
    assert got["t"].is_contiguous()


def test_outer_cut_reads_exactly_the_wanted_bytes(
    cut_file: tuple[Path, dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    p, t = cut_file
    rec = Recorder(monkeypatch)
    got = fst.load_files([p], keys=["w"], select={"w": (slice(2, 5),)})
    assert same_tensor(got["w"], t["w"][2:5])
    assert len(rec.calls) == 1
    assert rec.bytes_read == 3 * 6 * 4 * 4
    assert all(placements is None for *_, placements in rec.transfers)


def test_inner_cut_reads_no_more_than_the_tensor(
    cut_file: tuple[Path, dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    p, t = cut_file
    rec = Recorder(monkeypatch)
    got = fst.load_files(
        [p],
        keys=["w", "v"],
        select={"w": (slice(None), 1, slice(1, 3)), "v": (Ellipsis, slice(0, 2))},
    )
    assert same_tensor(got["w"], t["w"][:, 1, 1:3])
    assert same_tensor(got["v"], t["v"][..., 0:2])
    wanted = got["w"].numel() * 4 + got["v"].numel() * 2
    full = t["w"].numel() * 4 + t["v"].numel() * 2
    assert wanted < rec.bytes_read <= full
    # one coalesced read per tensor, the runs as placements
    assert [len(placements or []) for *_, placements in rec.transfers] == [8, 12]
    (report,) = rec.reports
    assert report["placed_pieces"] == 2
    assert report["gap_bytes"] == rec.bytes_read - wanted
    assert report["bytes_read"] == rec.bytes_read


def test_stepped_slices_read_their_box(
    cut_file: tuple[Path, dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    p, t = cut_file
    rec = Recorder(monkeypatch)
    got = fst.load_files(
        [p], keys=["v"], select={"v": (slice(1, None, 3), slice(None, None, 2))}
    )
    assert same_tensor(got["v"], t["v"][1::3, ::2])
    assert got["v"].is_contiguous()
    assert rec.bytes_read == 10 * 5 * 2  # rows 1..10 whole: one plain read
    assert rec.transfers[0][4] is None


@pytest.mark.parametrize(
    ("dim", "world"), [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (-1, 2), (2, 4)]
)
def test_get_sharded_equals_chunk(
    cut_file: tuple[Path, dict[str, torch.Tensor]], dim: int, world: int
) -> None:
    p, t = cut_file
    with fst.safe_open(p) as f:
        for rank in range(world):
            got = f.get_sharded("w", dim, rank, world)
            assert same_tensor(got, torch.chunk(t["w"], world, dim)[rank].contiguous())
            assert got.is_contiguous()


def test_get_sharded_refusals(cut_file: tuple[Path, dict[str, torch.Tensor]]) -> None:
    p, _ = cut_file
    with fst.safe_open(p) as f:
        with pytest.raises(SelectError, match="does not divide"):
            f.get_sharded("w", 0, 0, 3)
        with pytest.raises(SelectError, match="rank 2"):
            f.get_sharded("w", 0, 2, 2)
        with pytest.raises(SelectError, match="out of range"):
            f.get_sharded("w", 3, 0, 2)
        with pytest.raises(SelectError, match="out of range"):
            f.get_sharded("w", -4, 0, 2)
        with pytest.raises(SelectError, match="world of 0"):
            f.get_sharded("w", 0, 0, 0)
        with pytest.raises(SelectError, match="rank -1"):
            f.get_sharded("w", 0, -1, 2)
        with pytest.raises(SelectError, match="out of range"):
            f.get_sharded("s", 0, 0, 1)


def test_select_shards_through_load_files(
    cut_file: tuple[Path, dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    p, t = cut_file
    rec = Recorder(monkeypatch)
    for rank in range(2):
        got = fst.load_files([p], select=select_shards(["w", "v"], 0, rank, 2))
        assert set(got) == set(t)
        assert same_tensor(got["w"], torch.chunk(t["w"], 2, 0)[rank])
        assert same_tensor(got["v"], torch.chunk(t["v"], 2, 0)[rank])
        assert same_tensor(got["u"], t["u"])
    nbytes = {k: v.numel() * v.element_size() for k, v in t.items()}
    assert rec.bytes_read == nbytes["w"] + nbytes["v"] + 2 * (nbytes["u"] + nbytes["s"])
    assert select_shards(["a"], -1, 1, 4) == {"a": Shard(-1, 1, 4)}
    with pytest.raises(SelectError, match="does not divide"):
        fst.load_files([p], select=select_shards(["u"], 0, 0, 4))


def test_selection_errors(cut_file: tuple[Path, dict[str, torch.Tensor]]) -> None:
    p, _ = cut_file
    with pytest.raises(KeyError, match="nope"):
        fst.load_files([p], select={"nope": (0,)})
    with pytest.raises(SelectError, match="not in keys"):
        fst.load_files([p], keys=["w"], select={"v": (0,)})
    with pytest.raises(SelectError, match="out of bounds"):
        fst.load_files([p], select={"w": (9,)})
    with pytest.raises(SelectError, match="too many indices"):
        fst.load_files([p], select={"u": (0, 0)})
    with pytest.raises(SelectError, match="step"):
        fst.load_files([p], select={"u": (slice(None, None, -1),)})
    bad: dict[str, Any] = {"w": ("x",)}
    with pytest.raises(SelectError, match="'w'"):
        fst.load_files([p], select=bad)


def test_explain_shows_wanted_against_read(
    cut_file: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    p, _ = cut_file
    text = fst.explain(p, select={"w": (slice(None), slice(1, 3))})
    assert "selections:" in text
    assert (
        "w: (8, 6, 4) -> (8, 2, 4), 256 bytes wanted, 704 read in 1 read(s) over 8 run(s)"
        in text
    )
    assert "GB to read for" in text
    # the plan's line summarises the request: w's 8 runs plus one per whole tensor
    assert "selections: 11 runs coalesced into 4 reads, amplification x" in text
    assert "gaps up to" in text and "io_cost_us" in text
    plain = fst.explain(p)
    assert (
        "selections:" not in plain
        and "wanted" not in plain
        and "coalesced" not in plain
    )
    sharded = fst.explain(p, keys=["w"], select=select_shards(["w"], 0, 1, 2))
    assert (
        "w: (8, 6, 4) -> (4, 6, 4), 384 bytes wanted, 384 read in 1 read(s) over 1 run(s)"
        in sharded
    )
    assert "coalesced" not in sharded


def test_get_slice_inner_dims_read_their_runs_not_the_block(
    cut_file: tuple[Path, dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    p, t = cut_file
    rec = Recorder(monkeypatch)
    with fst.safe_open(p) as f:
        got = f.get_slice("w")[:, 1]
        stepped = f.get_slice("w")[1::2, 2:4, ::3]
    assert same_tensor(got, t["w"][:, 1])
    assert same_tensor(stepped, t["w"][1::2, 2:4, ::3])
    first, second = rec.transfers
    assert (
        first[4] is not None and len(first[4]) == 8
    )  # one coalesced read, eight placements
    assert first[1] < t["w"].numel() * 4
    assert (
        second[4] is not None and len(second[4]) == 7
    )  # the box is rows 1..7: seven runs


def test_read_job_host_placements(tmp_path: Path) -> None:
    p = tmp_path / "data"
    p.write_bytes(bytes(range(256)))
    out = torch.full((8,), 255, dtype=torch.uint8)
    placements = [(r * 8, r * 2, 2) for r in range(4)]
    job: list[_core.FileJob] = [(str(p), [(2, 26, out.data_ptr(), None, placements)])]
    report = _core.read_job(job, 1, 0, 1, "pread", None)
    assert report["bytes_read"] == 26 and report["gap_bytes"] == 18
    assert report["placed_pieces"] == 1 and report["scatter_copies"] == 0
    assert out.tolist() == [2, 3, 10, 11, 18, 19, 26, 27]
    for bad in ([(25, 0, 2)], [(0, 0, 2), (8, 4, 2)]):
        with pytest.raises(ReadError, match="placement"):
            _core.read_job(
                [(str(p), [(2, 26, out.data_ptr(), None, bad)])], 1, 0, 1, "pread", None
            )
    overlapping: list[_core.Transfer] = [
        (2, 26, out.data_ptr(), None, placements),
        (0, 4, out.data_ptr() + 6, None, None),
    ]
    with pytest.raises(ValueError, match="overlap"):
        _core.read_job([(str(p), overlapping)], 1, 0, 1, "pread", None)


def _cuda_plan(path: Path, nbytes: int) -> _files.Plan:
    return _files.Plan.for_read(
        [str(path)], [nbytes], torch.device("cuda", 0), check_fit=False
    )


@pytest.mark.cuda
def test_load_files_select_to_cuda(
    cut_file: tuple[Path, dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    p, t = cut_file
    rec = Recorder(monkeypatch)
    select = {"w": (slice(None), 1, slice(1, 3)), "v": (slice(2, 7),), "s": ()}
    got = fst.load_files([p], device="cuda", select=select)
    assert all(v.device.type == "cuda" for v in got.values())
    assert same_tensor(got["w"].cpu(), t["w"][:, 1, 1:3])
    assert same_tensor(got["v"].cpu(), t["v"][2:7])
    assert same_tensor(got["s"].cpu(), t["s"])
    assert same_tensor(got["u"].cpu(), t["u"])
    # w is the strided device path: eight equal placements at constant
    # strides, landed by one 2-D copy, not eight
    (report,) = rec.reports
    assert report["placed_pieces"] == 1 and report["scatter_copies"] == 0
    on_cpu = fst.load_files([p], select=select)
    assert all(same_tensor(got[k].cpu(), on_cpu[k]) for k in got)


@pytest.mark.cuda
def test_get_sharded_and_stepped_on_cuda(
    cut_file: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    p, t = cut_file
    with fst.safe_open(p, device="cuda:0") as f:
        for rank in range(4):
            got = f.get_sharded("w", 0, rank, 4)
            assert got.device == torch.device("cuda", 0)
            assert same_tensor(got.cpu(), torch.chunk(t["w"], 4, 0)[rank])
        assert same_tensor(f.get_slice("w")[:, 1:3].cpu(), t["w"][:, 1:3])
        assert same_tensor(
            f.get_slice("w")[1::2, 2:4, ::3].cpu(), t["w"][1::2, 2:4, ::3]
        )
    got = fst.load_files(
        [p],
        device="cuda",
        select={"w": (slice(None), slice(1, 3)), "v": (slice(1, None, 3), 2)},
    )
    assert same_tensor(got["w"].cpu(), t["w"][:, 1:3])
    assert same_tensor(got["v"].cpu(), t["v"][1::3, 2])


@pytest.mark.cuda
def test_read_job_device_placements(tmp_path: Path) -> None:
    p = tmp_path / "data"
    p.write_bytes(bytes(range(256)))
    device = torch.device("cuda", 0)
    out = torch.full((8,), 255, dtype=torch.uint8, device=device)
    regular = [(r * 8, r * 2, 2) for r in range(4)]
    torch.cuda.synchronize(0)
    report = _cuda_plan(p, 26).execute(
        [(str(p), [(2, 26, out.data_ptr(), 0, regular)])], device
    )
    assert out.cpu().tolist() == [2, 3, 10, 11, 18, 19, 26, 27]
    assert report["bytes_read"] == 26 and report["gap_bytes"] == 18
    assert (
        report["placed_pieces"] == 1 and report["scatter_copies"] == 0
    )  # one 2-D copy
    # irregular lengths: one copy per placement
    out = torch.full((6,), 255, dtype=torch.uint8, device=device)
    irregular = [(0, 0, 1), (8, 1, 2), (16, 3, 3)]
    torch.cuda.synchronize(0)
    report = _cuda_plan(p, 19).execute(
        [(str(p), [(2, 19, out.data_ptr(), 0, irregular)])], device
    )
    assert out.cpu().tolist() == [2, 10, 11, 18, 19, 20]
    assert report["placed_pieces"] == 1 and report["scatter_copies"] == 3


def test_shard_ranges() -> None:
    assert Shard(1, 1, 3).ranges((4, 9)) == [(0, 4), (3, 6)]
    assert Shard(-2, 0, 2).ranges((4, 9)) == [(0, 2), (0, 9)]
    assert Selection.from_ranges(
        (4, 9), Shard(1, 2, 3).ranges((4, 9))
    ).result_shape == (4, 3)
