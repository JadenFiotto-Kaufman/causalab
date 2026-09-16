"""Tensor-parallel narrowing in coordinated loads: a tensor with a ``shards``
entry is delivered to each rank as its own shard, read directly (one run when
the shard is an outer cut) or through a row-block read and an all-to-all
(an inner cut), never broadcast, and counted at shard size."""

from __future__ import annotations

import json
from datetime import timedelta
from itertools import pairwise
from math import prod
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hypothesis import given, settings
from hypothesis import strategies as st

import causalab.io.fastersafetensors.torch as fst
from causalab.io.fastersafetensors import _distributed, _files
from causalab.io.fastersafetensors._distributed import (
    Broadcast,
    Cooperative,
    Direct,
    Exchange,
    ShardLayout,
    exchange_input,
    plan_items,
    row_blocks,
)
from causalab.io.fastersafetensors._header import Layout, TensorEntry
from causalab.io.fastersafetensors._select import Selection, Shard, select
from causalab.io.fastersafetensors.errors import PlanError, ReadError, SelectError
from tests.io.fastersafetensors.strategies import same_tensor

# --- pure planning -----------------------------------------------------------


@st.composite
def shardable(draw: st.DrawFn) -> tuple[tuple[int, ...], int, int, int]:
    """``(shape, dim, world, itemsize)`` with ``shape[dim]`` a multiple of ``world``."""
    ndim = draw(st.integers(1, 4))
    dim = draw(st.integers(0, ndim - 1))
    world = draw(st.integers(1, 6))
    shape = [draw(st.integers(0, 5)) for _ in range(ndim)]
    shape[dim] = world * draw(st.integers(0, 4))
    return tuple(shape), dim, world, draw(st.sampled_from([1, 2, 4, 8]))


@given(shardable(), st.data())
@settings(max_examples=150)
def test_shard_layout_is_rows_of_world_pieces(
    case: tuple[tuple[int, ...], int, int, int], data: st.DataObject
) -> None:
    shape, dim, world, itemsize = case
    rank = data.draw(st.integers(0, world - 1))
    signed_dim = data.draw(st.sampled_from([dim, dim - len(shape)]))
    layout = ShardLayout.of(shape, itemsize, Shard(signed_dim, rank, world))
    assert layout.rows == prod(shape[:dim])
    assert layout.cols == prod(shape[dim:])
    assert layout.piece * world == layout.cols
    assert layout.rows * layout.cols * itemsize == prod(shape) * itemsize
    chunk = torch.empty(shape, dtype=torch.uint8).new_empty(shape)
    expected = (
        torch.chunk(chunk, world, dim)[rank].numel() * itemsize if chunk.numel() else 0
    )
    assert layout.nbytes == expected
    # one contiguous run exactly when no index precedes the cut
    assert layout.direct == (layout.rows <= 1)


pytestmark = pytest.mark.unit


def test_shard_layout_refusals() -> None:
    with pytest.raises(SelectError, match="does not divide"):
        ShardLayout.of((6, 8), 4, Shard(0, 0, 4))
    with pytest.raises(SelectError, match="out of range"):
        ShardLayout.of((6, 8), 4, Shard(2, 0, 2))
    with pytest.raises(SelectError, match="rank 2"):
        Shard(0, 2, 2)
    with pytest.raises(SelectError, match="world of 0"):
        Shard(0, 0, 0)
    with pytest.raises(SelectError, match="rank -1"):
        Shard(0, -1, 2)


@given(st.integers(0, 40), st.integers(1, 9))
def test_row_blocks_partition_the_rows(rows: int, world: int) -> None:
    blocks = row_blocks(rows, world)
    assert len(blocks) == world
    assert blocks[0][0] == 0 and blocks[-1][1] == rows
    assert all(a[1] == b[0] for a, b in pairwise(blocks))
    sizes = [hi - lo for lo, hi in blocks]
    assert max(sizes) - min(sizes) <= 1
    assert sizes == sorted(sizes, reverse=True)


@st.composite
def exchange_cases(draw: st.DrawFn) -> tuple[torch.Tensor, int, int, int, list[int]]:
    """A tensor, the cut, the group size and the shard index each group rank wants."""
    shape, dim, world, _ = draw(shardable().filter(lambda c: c[1] > 0))
    group = draw(st.integers(1, 5))
    wanted = [draw(st.integers(0, world - 1)) for _ in range(group)]
    dtype = draw(
        st.sampled_from([torch.uint8, torch.int16, torch.float32, torch.int64])
    )
    numel = prod(shape)
    tensor = torch.arange(numel, dtype=torch.int64).to(dtype).reshape(shape)
    return tensor, dim, world, group, wanted


@given(exchange_cases())
@settings(max_examples=150)
def test_exchange_delivers_every_rank_its_shard(
    case: tuple[torch.Tensor, int, int, int, list[int]],
) -> None:
    tensor, dim, world, group, wanted = case
    itemsize = tensor.element_size()
    layout = ShardLayout.of(tuple(tensor.shape), itemsize, Shard(dim, wanted[0], world))
    blocks = row_blocks(layout.rows, group)
    flat = (
        tensor.reshape(-1).view(torch.uint8)
        if tensor.numel()
        else torch.empty(0, dtype=torch.uint8)
    )
    # what each source rank sends: for every destination, its wanted piece of each row
    sent = []
    for lo, hi in blocks:
        block = flat[lo * layout.row_bytes : hi * layout.row_bytes]
        sent.append(exchange_input(block, layout, wanted))
    # the all-to-all: destination i receives segment i of every source, in source order
    for s, (lo, hi) in zip(sent, blocks, strict=True):
        assert s.numel() == group * (hi - lo) * layout.piece_bytes
    for i, index in enumerate(wanted):
        received = [
            s[
                i * (hi - lo) * layout.piece_bytes : (i + 1)
                * (hi - lo)
                * layout.piece_bytes
            ]
            for s, (lo, hi) in zip(sent, blocks, strict=True)
        ]
        got = torch.cat(received)
        expected = torch.chunk(tensor, world, dim)[index].contiguous()
        assert got.numel() == expected.numel() * itemsize
        if expected.numel():
            assert torch.equal(got.view(tensor.dtype).reshape(expected.shape), expected)


def _entry(name: str, shape: tuple[int, ...], start: int) -> TensorEntry:
    return TensorEntry(name, "F32", shape, start, start + prod(shape) * 4)


def _picked(prepared: _distributed._Prepared) -> list[tuple[str, tuple[int, ...]]]:
    """``(name, shape)`` of every pick in the rank's chunk jobs."""
    return [
        (e.name, s.result_shape)
        for job in prepared.owned_chunks
        for _, ps in job
        for e, s in ps
    ]


def test_plan_items_broadcasts_a_narrowed_tensor_above_the_cooperative_size() -> None:
    # A cooperative read lands pieces of the file by offset from the tensor's
    # start, so only a whole, plain selection may take it; the narrowed and the
    # stepped ones here deliver more than ``cooperative_bytes`` yet broadcast.
    huge = _entry("huge", (100, 8), 0)  # 3200 B
    whole = Selection.full(huge.shape)
    reshaped = select(huge.shape, (None, Ellipsis))  # the same bytes, a new axis
    narrowed = select(huge.shape, (slice(10, 100),))  # 2880 B from row 10
    stepped = select(huge.shape, (slice(None, None, 2),))  # 1600 B, every other row
    wanted = [("f", [(huge, s) for s in (whole, reshaped, narrowed, stepped)])]
    items = plan_items(wanted, {}, {}, 2, cooperative_bytes=1024)
    assert [type(item.delivery) for item in items] == [
        Cooperative,
        Cooperative,
        Broadcast,
        Broadcast,
    ]


def test_plan_items_counts_shards_at_shard_size_and_agrees_across_ranks() -> None:
    world = 4
    big = _entry("big", (64, 8), 0)  # 2048 B, sharded on dim 0: 512 B per rank
    inner = _entry("inner", (8, 16), 2048)  # 512 B, sharded on dim 1: exchange
    rep = _entry("rep", (4, 4), 2560)  # 64 B replicated
    huge = _entry("huge", (100, 8), 2624)  # 3200 B replicated: cooperative
    wanted = [("f", [(e, Selection.full(e.shape)) for e in (big, inner, rep, huge)])]
    shards = {"big": Shard(0, 0, world), "inner": Shard(1, 0, world)}
    exchanged = {"inner": (3, 2, 1, 0)}
    prepared = [
        _distributed._Prepared(
            plan_items(wanted, shards, exchanged, world, cooperative_bytes=1024),
            torch.device("cpu"),
            rank,
            world,
        )
        for rank in range(world)
    ]
    kinds = [type(item.delivery) for item in prepared[0].items]
    assert kinds == [Direct, Exchange, Broadcast, Cooperative]
    # delivered sizes: shards at shard size
    assert prepared[0].sizes == [512, 128, 64, 3200]
    assert prepared[0].largest == 3200 and prepared[0].broadcast_largest == 64
    # the stream order and the rounds are the same on every rank
    assert len({tuple(p.chunk_of) for p in prepared}) == 1
    assert len({tuple(map(tuple, p.rounds)) for p in prepared}) == 1
    # every rank reads its shard of `big`, a row block of `inner`, and the
    # owner of `rep` reads it; nobody reads `huge` in a chunk job
    owner = prepared[0].items[2].delivery
    assert isinstance(owner, Broadcast)
    for rank, p in enumerate(prepared):
        assert ("big", (16, 8)) in _picked(p)
        assert ("inner", (2, 16)) in _picked(p)
        assert ("huge", (100, 8)) not in _picked(p)
        assert sum(p.chunk_bytes) == 512 + 128 + (64 if rank == owner.owner else 0)
    assert sum(("rep", (4, 4)) in _picked(p) for p in prepared) == 1


def test_fit_and_budget_use_the_largest_delivered_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # two 3 GiB shards and a 5 GiB replicated tensor: the consumer-held and
    # arriving terms are 5 GiB, the in-flight window is sized by the broadcast
    # tensors alone
    gib = 1 << 30
    monkeypatch.setattr(_distributed, "RESIDENT_OVERRIDE_BYTES", None)
    assert _distributed._in_flight_bytes(2 * gib) == max(
        2 * gib, _distributed.IN_FLIGHT_BYTES
    )
    assert _distributed.resident_budget(40 * gib, 5 * gib, 2 * gib, 64 << 20) == max(
        _distributed.COOPERATIVE_BYTES,
        min(
            _distributed.RESIDENT_MAX_BYTES,
            (40 * gib - 10 * gib - _distributed._in_flight_bytes(2 * gib) - (64 << 20))
            // 2,
        ),
    )
    # FASTERSAFETENSORS_RESIDENT_GIB, parsed at import with the other knobs, wins,
    # on CPU too, where there is no headroom to derive a budget from.
    monkeypatch.setattr(_distributed, "RESIDENT_OVERRIDE_BYTES", 3 * gib)
    assert _distributed.resident_budget(40 * gib, 5 * gib, 2 * gib, 64 << 20) == 3 * gib
    cpu = _distributed._Prepared([], torch.device("cpu"), 0, 1)
    assert _distributed._budget_for(cpu) == 3 * gib
    monkeypatch.setattr(_distributed, "RESIDENT_OVERRIDE_BYTES", None)
    assert _distributed._budget_for(cpu) == _distributed.RESIDENT_MAX_BYTES


def test_fit_check_reserves_the_resident_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicit budget bypasses the derived clamp, so the fit check must
    # reserve it: a too-large override is a PlanError, not an OOM mid-load.
    gib = 1 << 30
    monkeypatch.setattr(_distributed._files, "cuda_index", lambda device: 0)
    monkeypatch.setattr(_distributed._files, "device_headroom", lambda index: 10 * gib)
    monkeypatch.setattr(_distributed, "COOPERATIVE_BYTES", 4 * gib)
    monkeypatch.setattr(_distributed, "IN_FLIGHT_BYTES", 4 * gib)
    prepared = _distributed._Prepared([], torch.device("cpu"), 0, 1)
    monkeypatch.setattr(_distributed, "RESIDENT_OVERRIDE_BYTES", None)
    _distributed._check_fit(prepared)  # 4 GiB resident floor + 4 GiB in flight fit
    monkeypatch.setattr(_distributed, "RESIDENT_OVERRIDE_BYTES", 8 * gib)
    with pytest.raises(
        PlanError, match=r"FASTERSAFETENSORS_RESIDENT_GIB read-ahead budget \(8 GiB\)"
    ):
        _distributed._check_fit(prepared)
    # A small override reserves only what the scheduler will hold: this load
    # fails the derived check (4 + 4 GiB + two 2 GiB tensors) and passes with it.
    monkeypatch.setattr(_distributed, "RESIDENT_OVERRIDE_BYTES", None)
    prepared.largest = 2 * gib
    with pytest.raises(PlanError, match=r"resident chunks \(4 GiB\)"):
        _distributed._check_fit(prepared)
    monkeypatch.setattr(_distributed, "RESIDENT_OVERRIDE_BYTES", 1 * gib)
    _distributed._check_fit(prepared)


# --- real collectives ---------------------------------------------------------

WORLD = 3
"""Processes spawned; group rank ``g`` is global rank ``g + 1`` (rank 0 sits out)."""


def _checkpoint(root: Path) -> dict[str, torch.Tensor]:
    tensors = {
        "w": torch.arange(48).reshape(6, 8),  # int64, 384 B: outer cut, direct
        "v": torch.arange(48, dtype=torch.float32).reshape(6, 8),  # inner cut, exchange
        "u": torch.arange(24, dtype=torch.int16).reshape(2, 3, 4),  # 3-d inner cut
        "x": torch.arange(64, dtype=torch.uint8).reshape(
            4, 16
        ),  # world 4 on a group of 2
        "k": torch.arange(16, dtype=torch.int32).reshape(
            4, 4
        ),  # both ranks want shard 0
        "t": torch.arange(6, dtype=torch.float32).reshape(
            1, 6
        ),  # inner cut, one row: direct
        "e": torch.empty(0, 8),  # inner cut of nothing
        "z": torch.arange(18, dtype=torch.int16).reshape(3, 6),  # rows do not divide
        "r": torch.arange(5),  # 40 B replicated, packed
        "mid": torch.arange(
            32, dtype=torch.float32
        ),  # 128 B replicated, direct broadcast
        "big": torch.arange(100, dtype=torch.int32),  # 400 B replicated, cooperative
        "q": torch.arange(18).reshape(2, 9),  # 144 B; fewer rows than the 3-rank group
    }
    fst.save_file({k: tensors[k] for k in ("w", "v", "r", "q")}, root / "0.safetensors")
    fst.save_file(
        {k: tensors[k] for k in ("u", "x", "k", "big")}, root / "1.safetensors"
    )
    fst.save_file(
        {k: tensors[k] for k in ("t", "e", "z", "mid")}, root / "2.safetensors"
    )
    return tensors


def _shards(g: int) -> dict[str, Shard]:
    return {
        "w": Shard(0, g, 2),
        "v": Shard(1, g, 2),
        "u": Shard(2, g, 2),
        "x": Shard(1, 2 * g, 4),
        "k": Shard(0, 0, 2),
        "t": Shard(-1, g, 2),
        "e": Shard(1, g, 2),
        "z": Shard(1, g, 2),
    }


def _expected(tensors: dict[str, torch.Tensor], g: int) -> dict[str, torch.Tensor]:
    out = {}
    for name, tensor in tensors.items():
        shard = _shards(g).get(name)
        out[name] = (
            tensor
            if shard is None
            else torch.chunk(tensor, shard.world, shard.dim)[shard.rank]
        ).contiguous()
    return out


def worker(rank: int, root: str) -> None:
    directory = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=60),
    )
    group = dist.new_group([1, 2], backend="gloo")
    try:
        _distributed.BROADCAST_BYTES = 64
        _distributed.DIRECT_BYTES = 100
        _distributed.COOPERATIVE_BYTES = 200
        _distributed.IN_FLIGHT_BYTES = 200
        paths = [directory / f"{i}.safetensors" for i in range(3)]
        tensors = fst.load_files(paths)  # the reference, independently
        # Over the whole world: an inner cut of a tensor with fewer rows than
        # ranks leaves the last rank an empty row block to exchange.
        world = dist.group.WORLD
        assert isinstance(world, dist.ProcessGroup)
        whole = fst.load_files(paths[:1], shards={"q": Shard(1, rank, 3)}, group=world)
        assert same_tensor(
            whole["q"], torch.chunk(tensors["q"], 3, 1)[rank].contiguous()
        )
        if rank == 0:
            return
        assert isinstance(group, dist.ProcessGroup)
        g = rank - 1
        expected = _expected(tensors, g)

        reads: list[tuple[str, int, int]] = []
        execute = _files.Plan.execute

        def record(self, files, device):
            reads.extend(
                (path, offset, nbytes)
                for path, ts in files
                for offset, nbytes, *_ in ts
            )
            return execute(self, files, device)

        _files.Plan.execute = record
        broadcast = dist.broadcast
        broadcast_bytes = 0

        def counting(tensor, *args, **kwargs):
            nonlocal broadcast_bytes
            broadcast_bytes += tensor.numel() * tensor.element_size()
            return broadcast(tensor, *args, **kwargs)

        _distributed.dist.broadcast = counting

        got = fst.load_files(paths, device="cpu", shards=_shards(g), group=group)
        assert set(got) == set(expected)
        for name in expected:
            assert same_tensor(got[name], expected[name]), name
        assert all(
            t.untyped_storage().nbytes() == t.numel() * t.element_size()
            for t in got.values()
        )
        # only the replicated tensors below the cooperative threshold travel by broadcast
        assert broadcast_bytes == 40 + 128 + 144
        # this rank read exactly its half of `w` from the file
        entry = Layout.from_file(paths[0]).tensors["w"]
        assert (str(paths[0]), entry.start + g * 192, 192) in reads
        (directory / f"rank{rank}.json").write_text(
            json.dumps({"reads": reads, "broadcast": broadcast_bytes})
        )

        # the same through stream_files, over two requests, plus a name in no request
        streamed = list(
            fst.stream_files(
                [(paths[:2], None), (paths[2:], ["t", "e", "z", "mid"])],
                device="cpu",
                group=group,
                shards=_shards(g),
            )
        )
        assert sorted(name for name, _ in streamed) == sorted(expected)
        for name, tensor in streamed:
            assert same_tensor(tensor, expected[name]), name
        with pytest.raises(SelectError, match="nope"):
            list(
                fst.stream_files(
                    [(paths[:1], None)],
                    device="cpu",
                    group=group,
                    shards={"nope": Shard(0, 0, 2)},
                )
            )

        # refusals cross the group before any byte moves
        with pytest.raises(ReadError, match=r"preparation.*does not divide"):
            fst.load_files(paths, shards={"w": Shard(0, g, 4)}, group=group)
        with pytest.raises(ReadError, match=r"preparation.*not in keys"):
            fst.load_files(paths, keys=["r"], shards={"w": Shard(0, g, 2)}, group=group)
        with pytest.raises(ReadError, match=r"preparation.*both"):
            fst.load_files(
                paths,
                select={"w": (slice(0, 2),)},
                shards={"w": Shard(0, g, 2)},
                group=group,
            )
        with pytest.raises(ReadError, match=r"preparation.*nope"):
            fst.load_files(paths, shards={"nope": Shard(0, g, 2)}, group=group)
        with pytest.raises(PlanError, match="identical"):
            disagreeing = Shard(0, 0, 2) if g == 0 else Shard(0, 1, 3)
            fst.load_files(paths, shards={"w": disagreeing}, group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="needs Gloo")
def test_sharded_delivery_over_a_group(tmp_path: Path) -> None:
    tensors = _checkpoint(tmp_path)
    mp.spawn(worker, args=(str(tmp_path),), nprocs=WORLD, join=True)
    records = [
        json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (1, 2)
    ]
    nbytes = {k: v.numel() * v.element_size() for k, v in tensors.items()}
    # across the group: every replicated byte once, every exchanged tensor once
    # (row blocks), a direct shard per rank that wants it
    expected = (
        nbytes["r"]
        + nbytes["mid"]
        + nbytes["big"]
        + nbytes["q"]
        + nbytes["v"]
        + nbytes["u"]
        + nbytes["x"]
        + nbytes["z"]
        + nbytes["e"]
        + nbytes["w"]
        + nbytes["t"]  # halves, one each
        + nbytes["k"]  # shard 0 twice
    )
    assert sum(n for rec in records for _, _, n in rec["reads"]) == expected


def test_shards_without_a_group_are_selections(tmp_path: Path) -> None:
    tensors = _checkpoint(tmp_path)
    paths = [tmp_path / f"{i}.safetensors" for i in range(3)]
    for g in range(2):
        got = fst.load_files(paths, shards=_shards(g))
        expected = _expected(tensors, g)
        assert set(got) == set(expected)
        for name in expected:
            assert same_tensor(got[name], expected[name]), name
    with pytest.raises(SelectError, match="both"):
        fst.load_files(
            paths, select={"w": (slice(0, 2),)}, shards={"w": Shard(0, 0, 2)}
        )
