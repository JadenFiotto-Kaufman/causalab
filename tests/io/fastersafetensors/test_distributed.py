"""Real multiprocess collectives, with storage ownership recorded per rank."""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import causalab.io.fastersafetensors.torch as fst
from causalab.io.fastersafetensors import _distributed, _files
from causalab.io.fastersafetensors.errors import PlanError, ReadError


pytestmark = pytest.mark.unit


def worker(rank: int, root: str, backend: str = "gloo") -> None:
    directory = Path(root)
    device = "cpu" if backend == "gloo" else f"cuda:{max(0, rank - 1)}"
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    # Global rank zero is not a group member: exercise subgroup source mapping.
    group = dist.new_group([1, 2], backend=backend)
    try:
        if rank == 0:
            return
        assert isinstance(group, dist.ProcessGroup)
        # Pack boundaries across dtypes inside a 64-byte buffer; the requested
        # half of ``a`` (192 B) goes direct and stays below the resident budget.
        _distributed.BROADCAST_BYTES = 64
        _distributed.DIRECT_BYTES = 100
        _distributed.COOPERATIVE_BYTES = 200
        _distributed.IN_FLIGHT_BYTES = 200
        paths = [directory / f"{i}.safetensors" for i in range(3)]
        opened = []
        execute = _files.Plan.execute

        def record(self, files, device):
            opened.extend(path for path, _ in files)
            return execute(self, files, device)

        _files.Plan.execute = record
        select = {"a": fst.Shard(1, 1, 2), "b": (slice(None, None, 2),)}
        got = fst.load_files(paths, device=device, select=select, group=group)
        got = {name: tensor.cpu() for name, tensor in got.items()}
        assert torch.equal(got["a"], torch.arange(48).reshape(6, 8)[:, 4:])
        assert torch.equal(got["b"], torch.arange(11, dtype=torch.float32)[::2])
        assert got["empty"].shape == (0,)
        assert got["scalar"].item() == 19
        assert got["bytes"].tolist() == list(range(61))
        # Independent per-tensor allocations, not views holding a shard alive.
        assert all(
            t.untyped_storage().nbytes() == t.numel() * t.element_size()
            for t in got.values()
        )
        (directory / f"rank{rank}.json").write_text(json.dumps(opened))
        # Whole ``a`` (384 B) is above the resident budget: read cooperatively,
        # every rank lands its own piece of the file and the all-gather
        # completes it.
        before = len(opened)
        whole = fst.load_files(paths[:1], device=device, group=group)
        assert torch.equal(whole["a"].cpu(), torch.arange(48).reshape(6, 8))
        assert opened[before:] == [str(paths[0])]
        # Rows 1-5 of ``a`` (320 B) are also above the cooperative size, but
        # not the file's bytes from the tensor's start: the owner reads the
        # selection and broadcasts it rather than every rank landing a piece.
        narrowed = fst.load_files(
            paths[:1], device=device, select={"a": (slice(1, 6),)}, group=group
        )
        assert torch.equal(narrowed["a"].cpu(), torch.arange(48).reshape(6, 8)[1:6])
        # Streamed requests give the same tensors as one call each, in request order.
        streamed = list(
            fst.stream_files(
                [
                    (paths[:1], None),
                    (paths[1:], ["b", "scalar"]),
                    (paths[2:], ["empty"]),
                ],
                device=device,
                group=group,
            )
        )
        assert [name for name, _ in streamed] == ["a", "b", "scalar", "empty"]
        tensors = dict(streamed)
        assert torch.equal(tensors["a"].cpu(), torch.arange(48).reshape(6, 8))
        assert tensors["scalar"].item() == 19 and tensors["b"].shape == (11,)
        with pytest.raises(PlanError, match="identical"):
            fst.load_files(
                paths, device=device, keys=["a"] if rank == 1 else ["b"], group=group
            )
        with pytest.raises(ReadError, match=r"preparation.*rank 1"):
            fst.load_files(
                paths,
                device=device,
                keys=["a"] if rank == 1 else ["missing"],
                group=group,
            )

        def fail(self, files, device):
            if rank == 2:
                raise OSError("simulated reader failure")
            return execute(self, files, device)

        _files.Plan.execute = fail
        with pytest.raises(
            ReadError, match=r"read \(chunk \d+\).*simulated reader failure"
        ):
            fst.load_files(paths, device=device, group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "gloo",
            marks=pytest.mark.skipif(not dist.is_gloo_available(), reason="needs Gloo"),
        ),
        pytest.param(
            "nccl",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(
                    torch.cuda.device_count() < 2, reason="needs two CUDA devices"
                ),
            ],
        ),
    ],
)
def test_replicated_load_subgroup(tmp_path: Path, backend: str) -> None:
    fst.save_file({"a": torch.arange(48).reshape(6, 8)}, tmp_path / "0.safetensors")
    fst.save_file(
        {"b": torch.arange(11, dtype=torch.float32)}, tmp_path / "1.safetensors"
    )
    fst.save_file(
        {
            "bytes": torch.arange(61, dtype=torch.uint8),
            "scalar": torch.tensor(19),
            "empty": torch.empty(0),
        },
        tmp_path / "2.safetensors",
    )
    mp.spawn(worker, args=(str(tmp_path), backend), nprocs=3, join=True)
    # A rank reads its slice as several chunk jobs, so a file may be opened
    # more than once; which files each rank touched is the point.
    opened = [
        sorted(set(json.loads((tmp_path / f"rank{rank}.json").read_text())))
        for rank in (1, 2)
    ]
    # Contiguous slices of the request's bytes: the half of ``a`` that was
    # asked for (0.safetensors) is group rank 0's; ``b`` and 2.safetensors'
    # tensors fall in rank 1's.
    assert opened == [
        [str(tmp_path / "0.safetensors")],
        [str(tmp_path / "1.safetensors"), str(tmp_path / "2.safetensors")],
    ]


def test_reads_raises_a_scheduling_failure_instead_of_stalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The first job is submitted from the streaming thread; its done callback
    # pumps the queue on a worker thread, where ``submit`` fails once. The
    # executor would drop that exception, so ``take`` for the second chunk must
    # raise the recorded failure, even though its own pump then succeeds.
    monkeypatch.setattr(_distributed, "READ_WORKERS", 1)
    gate = threading.Event()

    class Pool:
        def __init__(self, executor: ThreadPoolExecutor) -> None:
            self.executor = executor
            self.calls = 0

        def submit(self, fn: object, *args: object) -> Future:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("cannot schedule")
            return self.executor.submit(gate.wait)

    class Chunks:
        n_rounds = 2
        chunk_bytes = (1, 1)

    prepared = Chunks()
    with ThreadPoolExecutor(max_workers=1) as executor:
        reads = _distributed._Reads(  # pyright: ignore[reportPrivateUsage]
            Pool(executor),  # pyright: ignore[reportArgumentType]
            budget=10,
        )
        try:
            reads.add(prepared)  # pyright: ignore[reportArgumentType]
            assert reads.running == 1 and len(reads.queue) == 1
            first = reads.take(prepared, 0)  # pyright: ignore[reportArgumentType]
        finally:
            gate.set()  # never leave the executor's shutdown waiting on the job
        assert first.result() is True
        with reads.changed:
            assert reads.changed.wait_for(lambda: reads.failure is not None, timeout=10)
        # The failed submit left the queue and the counters as they were.
        assert reads.running == 0 and reads.resident == 1 and len(reads.queue) == 1
        outcome: list[BaseException] = []

        def second() -> None:
            try:
                reads.take(prepared, 1)  # pyright: ignore[reportArgumentType]
            except BaseException as exc:
                outcome.append(exc)

        taker = threading.Thread(target=second, daemon=True)
        taker.start()
        taker.join(timeout=10)
        assert not taker.is_alive(), (
            "take() stalled on a chunk that cannot be scheduled"
        )
    # The recorded failure, not one raised by the taker's own pump (that one
    # scheduled the chunk: the pool's third call succeeds).
    assert len(outcome) == 1 and outcome[0] is reads.failure
    assert str(outcome[0]) == "cannot schedule"


def test_assign_chunks_isolates_oversized_items() -> None:
    # 40 bytes over 4 chunks: target 10; the 25-byte item stands alone.
    assert _distributed.assign_chunks([5, 5, 25, 3, 2], chunks=4) == [0, 0, 1, 2, 2]
    assert _distributed.assign_chunks([], chunks=4) == []
    assert _distributed.assign_chunks([7], chunks=4) == [0]
    # Even items: about `chunks` groups.
    assert _distributed.assign_chunks([1] * 8, chunks=4) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_assign_owners_cuts_contiguous_slices() -> None:
    # 26 bytes over 2 ranks: midpoints 5, 10.5, 14.5, 21.5, 25.5 against the cut at 13.
    assert _distributed.assign_owners([10, 1, 7, 7, 1], world=2) == [0, 0, 1, 1, 1]
    assert _distributed.assign_owners([], world=3) == []
    # Fewer items than ranks: consecutive items land on distinct, increasing ranks.
    assert _distributed.assign_owners([5, 5, 5], world=16) == [2, 8, 13]
    # All-empty items still get valid ranks.
    assert _distributed.assign_owners([0, 0], world=4) == [0, 1]
