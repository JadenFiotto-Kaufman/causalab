"""Type stubs for the Rust extension module ``causalab.io.fastersafetensors._core``.

Errors cross as the classes of :mod:`causalab.io.fastersafetensors.errors`: the format
grammar raises ``FormatError``, storage ``StorageError``, the planner
``PlanError``, a selection ``SelectError``, the CUDA runtime ``CudaError``,
and a job the engines cannot run ``ReadError`` / ``WriteError``. Argument
validation (bad pointers, mismatched lengths) raises ``ValueError``.
"""

from typing import TypedDict

TensorRow = tuple[str, str, list[int], int, int]
"""``(name, dtype, shape, absolute start, absolute end)``."""
ParsedHeader = tuple[int, list[TensorRow], list[tuple[str, str]] | None]
"""``(header_len, tensors, metadata)``; ``metadata`` is ``None`` when the
header has no ``__metadata__`` key."""
SpecRow = tuple[str, str, list[int], int]
"""``(name, dtype, shape, nbytes)`` of a tensor to write."""
Placement = tuple[int, int, int]
"""``(offset within the read, offset within the destination, nbytes)``: one
piece of a coalesced read and where it lands (``fst_core::select::Placement``)."""
Transfer = tuple[int, int, int, int | None, list[Placement] | None]
"""``(file offset, nbytes, address, dst_offset, placements)``: ``dst_offset``
``None`` lands the bytes at ``address`` on the host; an int names a device
buffer at ``address`` and the offset into it. ``placements`` ``None`` lands
the whole range contiguously there; a list lands only those pieces, packed
from that origin, and must be a gather of the range (sorted, disjoint,
packed) or the engine refuses before any I/O."""
ReadRow = tuple[int, int, int, list[Placement] | None]
"""``(offset within the tensor, nbytes, dst within the selection's
destination, placements)``: one ``fst_core::select::Read``."""
SelectItem = tuple[str, str, list[int], list[tuple[int, int]], str]
"""``(path, tensor name, shape, ranges, header dtype)``: one box to resolve."""
FileJob = tuple[str, list[Transfer]]
"""One file and the transfers out of it."""
PartRow = tuple[int, int, bool]
"""``(address, nbytes, on_device)`` of one part to write."""

class EnvDict(TypedDict):
    mounts: list[tuple[str, str]]
    nvidia_fs_loaded: bool
    cufile_library: str | None
    cpus: int

class PlanDict(TypedDict):
    files_in_flight: int
    split_bytes: int
    readers_per_file: int
    transport: str
    staging: tuple[int, int] | None
    reasons: list[str]

class CudaDict(TypedDict):
    cudart: tuple[bool, str]
    cufile: tuple[bool, str]
    devices: int

class ReadReport(TypedDict):
    bytes_read: int
    pieces: int
    files_opened: int
    gap_bytes: int
    placed_pieces: int
    scatter_copies: int
    elapsed: float

class CoalesceDict(TypedDict):
    runs: int
    reads: int
    wanted_bytes: int
    read_bytes: int
    amplification: float

def parse_header(bytes: bytes) -> ParsedHeader: ...
def parse_file_header(path: str) -> ParsedHeader: ...
def build_header(
    specs: list[SpecRow], metadata: list[tuple[str, str]] | None = None
) -> tuple[bytes, list[int]]: ...
def read_job(
    files: list[FileJob],
    files_in_flight: int,
    split_bytes: int,
    readers_per_file: int,
    transport: str,
    staging: tuple[int, int] | None,
    device: int | None = None,
) -> ReadReport:
    """Run one read job through the read engine as the plan fields say, GIL
    released; returns once every byte has landed. ``bytes_read`` counts what
    storage delivered, gaps between placements included (``gap_bytes``);
    ``scatter_copies`` counts per-placement device copies for placed pieces
    whose pattern was not one regular 2-D copy. Trusts the addresses
    (contiguous, exactly sized, alive, torch's stream synchronized); only
    ``causalab.io.fastersafetensors._files`` calls it."""

def write_object(
    path: str,
    specs: list[SpecRow],
    parts: list[PartRow],
    metadata: list[tuple[str, str]] | None = None,
    durable: bool = False,
    atomic: bool = True,
    device: int | None = None,
) -> int:
    """Write the header for ``specs`` then ``parts`` (``parts[i]`` is
    ``specs[i]``'s bytes) through the write engine; device parts drain
    through pinned staging on ``device``. Returns the bytes written. Trusts
    the addresses; only ``causalab.io.fastersafetensors._serialize`` calls it."""

def probe_env() -> EnvDict: ...
def probe_cuda() -> CudaDict: ...
def storage_classes(paths: list[str]) -> list[str]: ...
def plan_read(
    paths: list[str],
    wanted_bytes: list[int],
    device: str | None = None,
    free_bytes: int | None = None,
    coalesced: tuple[int, int, int, int] | None = None,
    allocation_bytes: int | None = None,
) -> PlanDict:
    """``coalesced`` is ``(runs, reads, wanted_bytes, read_bytes)`` from
    ``select_reads``; when it merged anything the reasons say so."""

def select_reads(items: list[SelectItem]) -> tuple[list[list[ReadRow]], CoalesceDict]:
    """Resolve every box to the core's reads, coalesced under the profile's
    policy for its file's storage class; the machine is probed once. A box
    outside its shape or a sub-byte run off a byte boundary is a
    ``SelectError``, an unknown dtype a ``FormatError``."""

def shard_ranges(
    shape: list[int], dim: int, rank: int, world: int
) -> list[tuple[int, int]]:
    """``(start, end)`` per dimension of shard ``rank`` of ``world`` along
    ``dim``; ``SelectError`` when it does not divide or is out of range."""
