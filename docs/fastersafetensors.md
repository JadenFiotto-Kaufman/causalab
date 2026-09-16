# fastersafetensors — `causalab.io.fastersafetensors`

`safetensors`, read and written the way the machine underneath can go fastest.
Same public API as `safetensors.torch`; same bytes on disk; a different path for
the bytes, chosen per machine from what it actually has — storage class per
path, `nvidia_fs` and `libcufile`, device memory headroom, CPUs — and printed
on request by `explain()`.

**Where it lives.** The Rust core is the Cargo workspace at the repository
root (`crates/fst-core`, `crates/fst-cuda`, `crates/fst-py`); the Python
surface is `causalab/io/fastersafetensors/`; the extension module
`causalab.io.fastersafetensors._core` is built by maturin, causalab's build
backend, on every `uv sync` and `uv build`. The Python tests are
`tests/io/fastersafetensors/`, the Rust tests live beside the code. The
library was developed as a standalone package and inlined here; the
measurements this document cites are reproduced in it (§Measurements).

## Group loading and sharded delivery

- `load_files(..., group=group)` coordinates CPU/Gloo or CUDA/NCCL ranks:
  replicated tensors are read once per group and distributed in bounded batches.
- `shards={name: Shard(dim, rank, world)}` delivers only each rank's tensor
  slice; inner-dimension shards use row-block reads and all-to-all exchange.
  See [tensor-parallel delivery](tp-narrowed-delivery.md).
- `stream_files(requests, group=group)` pipelines requests and yields tensors
  in request order, with read-ahead bounded by device headroom.
- Replicated tensors selected whole above `COOPERATIVE_BYTES` are read
  cooperatively; a narrowed or stepped selection above that size is read by
  its owner and broadcast, and counts in the fit check. Pinned host buffers
  are pooled for reuse. `FASTERSAFETENSORS_TRACE=1` prints read diagnostics.
- A rank's slice of a request is cut into read jobs of about
  `FASTERSAFETENSORS_CHUNK_GIB` (4 GiB) each, clamped to 4..16
  (`chunk_count`); `FASTERSAFETENSORS_CHUNKS` forces a count. Staging slots
  recycle on per-slot CUDA fences instead of draining the copy stream.
- A failure while scheduling a rank's read jobs is raised through the round's
  error collective on every rank instead of stalling the group.

All group members must make matching collective calls, with identical
files, keys and replicated selections, and the same values of the
environment knobs that shape the collectives (`FASTERSAFETENSORS_CHUNK_GIB`,
`FASTERSAFETENSORS_CHUNKS`, `FASTERSAFETENSORS_COOPERATIVE_GIB`): they are
part of the request signature, so a per-node difference is a `PlanError`
rather than a hang. The other knobs (`FASTERSAFETENSORS_INFLIGHT`,
`FASTERSAFETENSORS_INFLIGHT_GIB`, `FASTERSAFETENSORS_READ_WORKERS`,
`FASTERSAFETENSORS_COORDINATED_READERS`, `FASTERSAFETENSORS_RESIDENT_GIB`)
are local to a rank and need not match: they decide when a rank waits or
how much it reads ahead, never which collective it issues. Every knob that
is not an integer or is below its minimum is a `PlanError` naming the
variable when the module is imported. Per-rank
shards may differ in rank; their dimension and world size must agree. CUDA
callers must first set the current device. Existing calls without a group
retain independent loading.

## Rules that are not obvious from the code

- **Torch owns device memory.** Rust fills pointers the Python layer hands it;
  it never allocates on a device. A tensor's lifetime must never depend on a
  Rust object.
- **The planner is pure.** `plan_read` reads an `Env` and a `Profile`, never
  the machine. The numbers live in the profile (`docs/profiles/`, schema in
  §Calibration profile), not in constants; add a rule only with the
  measurement it comes from, in the rule's comment and in this document.
- **Orchestration is tested against the simulators**, `storage::sim` and
  `fst_cuda::sim`, for exact operation logs. Real backends get round-trip
  tests. The format gets reference fixtures (`crates/fst-core/src/fixtures.rs`,
  bytes written by `safetensors` 0.8.0) and properties.
- **Byte-identical output.** `build_header` must produce the reference
  library's bytes for the same tensors. Layout order is descending `Dtype`
  (the enum's declaration order is the reference's) then ascending name.
- **Errors are typed per domain** and map one-to-one onto
  `causalab.io.fastersafetensors.errors`. No bare strings, no bare `Exception`.
- **Question the measurement before the code.** If a rule looks wrong on a
  new machine, measure first; the rules are the measurements.

## Why

Loading Qwen3.6-35B-A3B (26 shards, 69 GB of parameters) on an H100
node took 36 s cold / 17 s warm through transformers' stock path. The bytes can
move in 8 s cold / 3 s warm: the stock path is one thread end to end, and the
NFS mount caps a single file near 3 GB/s however that file is split. Neither
`safetensors` (mmap slices, materialized under the GIL) nor `fastsafetensors`
(whole-shard device buffers that do not fit beside a 69 GB model on an 80 GB
card; GDS absent without the kernel module) had the shape to get there. A
thin orchestration over `safe_open(device=…)` with sixteen shards in flight
did — 16 s cold / 12 s warm end to end, in causalab's `weights.py`.

On the write side, an activation writer that streams chunks to disk builds
the header itself and writes the parts as memoryviews with no join, because
the reference `save_file` copies everything once more and holds the GIL while
it does.

Both are the same idea: the format is fixed and simple; the win is in *how*
the bytes move, and that depends on the machine. This library is that idea as a
package with its own core, so neither trick has to be re-derived per project.

## Measurements the rules rest on

One H100 80 GB node, 16 cores, 2 TB RAM, weights on an NFSv3 mount. Cold = `posix_fadvise(DONTNEED)` on every
shard, verified against `/proc/meminfo`; warm = the run right after.

| raw bytes, 26 shards, 71.9 GB | cold GB/s | warm GB/s |
|---|---|---|
| safetensors sequential, `device=cuda` | 2.6 | 7.2 |
| safetensors, 4 files in flight | 7.4 | 20.0 |
| safetensors, 16 files in flight | 8.3 | 25.9 |
| fastsafetensors nogds, 16 threads, all files | 8.1 | 24.0 |
| fastsafetensors nogds, 32 threads, all files | 9.7 | 17.7 |
| fastsafetensors, one file at a time, 16 threads, 64–256 MB blocks | 2.9–3.1 | 23 |
| fastsafetensors, two files at a time | 3.6–3.8 | 24 |
| transformers-style per-key mmap slices, 4 or 16 workers | 2.4–2.6 | 7.1 |

Facts the planner encodes:

1. **One NFS file caps near 3 GB/s** regardless of thread count or block size.
   Throughput comes from files in flight; 16 is enough, 32 no better.
2. **Warm is memcpy-bound** near 25 GB/s from page cache to device across 16
   threads; a single stream is 7 GB/s.
3. **mmap slices materialized in Python threads serialize** (the copy runs
   under the GIL); `safe_open(device="cuda")` releases it.
4. **fastsafetensors' no-GDS path equals threaded plain reads.** Its advantage
   is GDS, which needs the `nvidia_fs` module (absent on the nodes measured)
   and a GDS-capable mount (NFSv3/TCP is not one).
5. **Whole-file device buffers double memory per file in flight.** A reader
   must allocate per tensor to keep many files in flight beside a large model.

Nodes probed: the H100 node (no `nvidia_fs`, `libcufile` 1.15.1 present, an
NFS-mounted home, a local xfs volume at `/tmp`) and an 8× B200 183 GB node
with the same software picture.

Reproduced on the same node with the library's raw-read and write benchmarks,
every arm in a fresh process, two repeats, both NFS caches, plus write
throughput from CPU and GPU tensors to local xfs and NFS. Everything above reproduces within 20% except the threaded `safetensors`
**warm** rows: 13.4–13.7 GB/s (not 20.0) with 4 files and 11.3 GB/s (not
25.9) with 16, on torch 2.9 and 2.14 alike. The table's numbers were second
runs in one process; a fresh process pays ~3 s to first-allocate 1045 device
tensors, which whole-file buffers (fastsafetensors, 24.7 GB/s) avoid. A
per-tensor reader must grow the allocator once, not `cudaMalloc` per tensor.

**Kimi K3 across eight ranks (one 8× B200 183 GB node, K3's 96 shards /
1.56 TB on an NFSv3 mount; the multi-rank benchmark):**
eight processes, rank `r` on `cuda:r`, each `load_files(shards, keys=<its
tensors>)` — experts by `E % 8`, everything else replicated. A full rank is
295 GB (181 GB experts + 114 GB replicated) and does not fit, nor does a
16-way virtual rank (205 GB), so the load is the first 52 layers: 167 GB per
rank (100 GB experts in 34 272 tensors + 67 GB replicated), 1.34 TB landed,
869 GB over the wire. Idle node, two repeats: **cold 92.6 s wall / 73.4 s for
the slowest rank's own `load_files`, warm 41.1 s / 23.9 s.** Cold moves 11.8
GB/s over the wire against a measured aggregate ceiling of 12.3 GB/s (16
whole files in flight) to 14.6 GB/s (128 files) — 80–95% of the link;
sixteen readers per file (`--readers-per-file 16`, the `split_helps: true`
plan) reach the 14.6 and cut cold to 60 s but cost 17% warm. Warm lands
~56 GB/s node-wide however the ranks split the bytes: one process alone
reads page cache to its GPU at 22 GB/s, eight together get 7–8 GB/s each.
Replication is the expensive part: eight ranks reading the same 114 GB cold
get 2.2 GB/s each and 2.2 GB/s over the wire (page-fault convoy on shared
pages), where one rank alone reads its own subset at 9.4 GB/s. One rank
alone also fetches 222 GB for its 167 GB (NFS readahead pulls the
neighbouring ranks' experts); eight ranks together fetch exactly what they
want. The mount, for the profile (`docs/profiles/b200-nfs-2026-09-09.json`):
one file 2.0 GB/s cold, 9.26 at 8 files, 12.34 at 16, 29.2 warm to a GPU;
a single cold 4 KiB / 64 KiB / 1 MiB `pread` costs 1117 / 1262 / 2538 µs
(`/tmp` xfs: 116 / 293 / 1673), `open` 364 µs; H2D pinned 55.6 GB/s.

## Architecture

```
crates/fst-core    format · storage (posix, mmap, sim) · env · plan      pure Rust, no CUDA, no Python
crates/fst-cuda    dlopen libcudart / libcufile · pinned staging · sim   pure Rust, builds without CUDA
crates/fst-py      PyO3 module `fastersafetensors._core`                 releases the GIL around I/O
python/fastersafetensors   the torch-facing API, mirrors safetensors.torch
```

**Ownership rule.** Device memory is allocated by torch, never by Rust. The
Python layer allocates the destination tensor (host or device) and hands its
pointer down; Rust fills it. Accounting stays with torch's allocator and no
tensor's lifetime depends on a Rust object.

**Decision rule.** `plan::plan_read(env, profile, request)` is a pure
function. It never touches the machine; `env::Env::probe()` does, once. The
numbers its rules rest on come from the `profile::Profile` it is handed (see
§Calibration profile), not from constants in the code; every reason line names
the profile's `source` and the entry it read, and the plan carries its reasons.

**Test rule.** The I/O boundary (`storage::Storage`, `fst_cuda::DeviceCopier`,
`fst_cuda::DirectStorage`) has a simulated implementation that logs every call
and fails on schedule. Orchestration is tested against those for exact
behaviour. Real backends are tested for round trips. The format is tested
against fixtures written by the reference library and by property: build then
parse is identity, layout order is the reference's.

**Write engine** (`fst-core::write`). A `Payload` is a `format::BuiltHeader`
plus the caller's parts reordered into data-section order; a `Part` is
`Host(&[u8])` or `Device { ptr: fst_cuda::DevicePtr, nbytes }`. `write_object`
hands the header and then each part to one `PartWriter` as separate
`write_parts` calls — nothing is joined in memory. Device parts drain through
a `StagingRing` of `count` pinned buffers of `bytes` each: the copies of the
next chunks are enqueued before the current chunk is written, and because
`DeviceCopier` offers one stream-wide `synchronize` (no per-copy fences) the
ring synchronizes once per wrap; per-buffer events are a later refinement.
`WriteOptions { durable, atomic, staging }`: `durable` is `fsync` of the file
and its directory before returning; `atomic` writes to a sibling
`.name.<pid>.<n>.tmp` and renames it into place, removing the temp on any
failure. `serialize_to_vec` is the same header and ordering for `save() ->
bytes`. To support this, the storage traits grew three methods:
`PartWriter::finish(self: Box<Self>, durable)` (the object is complete only
after `finish`; `write_parts` may be called repeatedly), `Storage::rename(from,
to, durable)` and `Storage::remove(path)`; `sim` logs them as `Op::Finish`,
`Op::Rename`, `Op::Remove` and can fault them, `mmap` returns `Unsupported`.
**Read engine** (`fst-core::read`). `execute(plan, storage, copier, direct,
job)` runs a `ReadPlan`. The job is `ReadJob { files: Vec<FileJob> }`, `FileJob { path,
transfers: Vec<Transfer> }`, `Transfer { range: ReadRange, dest: Dest }`, with
`Dest::Host(&mut [u8])` — a caller-owned slice exactly the range's length,
filled directly — or `Dest::Device { ptr: DevicePtr, offset }` — a
caller-owned device buffer, filled from `offset` through the plan's pinned
staging ring and the `DeviceCopier` the caller passes. `Pread` and `Mmap`
both read through the `Storage` given (the caller picks the backend that
matches the transport); `CuFile` reads device pieces through the
`DirectStorage` passed as `direct`, with no staging, and host pieces through
the `Storage` (GDS only targets device memory). cuFile can fail per file at
run time, so under a `CuFile` plan a piece that fails with
`CudaError::Register` is re-read through pread and staging and the file stays
on pread for the rest of the job; `CudaError::Unavailable` switches the whole
job; a `CuFile` plan with `direct = None` runs on pread throughout. Each
switch is one `Fallback { path: Option<PathBuf>, reason }` in the report
(`path: None` for the job-wide ones); any other cuFile error is
`ReadError::DirectRead` naming path and range and fails the job. The fallback
ring is the plan's staging if it has one, else the planner's pread geometry,
and is allocated only when a piece first stages — a job read entirely through
cuFile pins nothing. Concurrency is the plan's: `files_in_flight` file workers,
one `RangeReader` each, `readers_per_file` piece workers over pieces of
`split_bytes` (device pieces also never exceed one staging buffer). A staging
slot is reused only after a `synchronize()`; the ring frees dirty slots in
batches because the trait offers nothing finer (per-buffer fences are a later
refinement). The first failure is recorded before anything else happens, no
piece starts after it, and the error (`ReadError`) names the path and range.
Success returns a `ReadReport`: `bytes_read` split into `cufile_bytes`,
`staged_bytes` and `host_bytes`, pieces, files opened through the `Storage`,
the `fallbacks`, wall time — and, for placed transfers (§Selections and
sharded reads), `gap_bytes`, `placed_pieces` and `scatter_copies`.
### Selections and sharded reads

Tensor-parallel loading reads a slice of a tensor per rank. Along the
outermost dimension a slice is one contiguous byte range; along an inner
dimension it is many short runs with gaps — a row-parallel `Linear` weight
`[out, in]` sharded on `in` over 8 ranks in bf16 is `out` runs of `in / 8 × 2`
bytes (Kimi K3's dense layers: 7168 runs of 4608 bytes at a 36 KiB stride).
On NFS every `pread` pays a round trip however small it is, so the runs must
be merged into larger reads and the wanted bytes then placed into a
contiguous destination. K3 is 1.56 TB, mostly per-expert tensors that are
filtered whole per rank; the strided case is the dense few percent, so
correctness and a predictable cost matter more than the last GB/s.

**Runs** (`fst-core::select`). A `Selection` is an N-d box over a tensor's
shape — one half-open range per dimension, validated against the shape at
construction (`Selection::new`, `::full`, `::shard(shape, dim, rank, world)`
with a typed `SelectError::DoesNotDivide` when the dimension does not divide
evenly). `Selection::runs(dtype)` yields the box's contiguous byte runs in
row-major order, relative to the tensor's first byte, merging dimensions that
are fully selected from the inside out: a full box is one run; a box full on
every inner dimension is one run per outer index; a box partial on two
dimensions is one run per (outer, middle) index. Concatenated, the runs are
exactly `torch.narrow`'s bytes. Sub-byte dtypes (`F4`, `F6_*`) are refused
unless every run starts and ends on a byte boundary
(`SelectError::SubByteMisaligned`); nothing is rounded.

**Coalescing rule** (`select::coalesce(runs, policy) -> Vec<Read>`). Runs are
taken in order; a run joins the read before it when the gap between them is
at most `max_gap_bytes` and the read stays within `max_read_bytes`, else it
starts a new read. Each `Read { offset, len, dst, placements }` carries its
`Placement { src, dst, len }`s: sorted by source, disjoint, landing back to
back from `dst == 0` (a gather — the read's wanted bytes are one contiguous
piece of the destination from `Read::dst`). The policy comes from the profile
through `plan::coalesce_policy(profile, storage_class)` /
`coalesce_policy_for(entry)`:

```
max_gap_bytes  = clamp(io_cost_us × single_file_gbps × 1000, 64 KiB, 8 MiB)
max_read_bytes = STAGING_BUFFER_BYTES (16 MiB)
```

`io_cost_us × single_file_gbps` is the bytes the mount moves in the fixed
cost of one read call, so reading through a gap that size is never dearer
than a second call. The floor, 64 KiB, is 21 µs at 3 GB/s — less than any
read call costs on any storage — and is what an entry with `io_cost_us`
unmeasured gets. The ceiling, 8 MiB, is half a staging buffer, so a slow
mount's large `io_cost_us` cannot turn a sparse selection into reading the
whole tensor. `max_gap_bytes = 0` disables coalescing (one read per run).
Bytes read exceed bytes wanted by at most one gap per run: amplification
`≤ 1 + max_gap / min_run`. For the K3 row-parallel case above the 32 KiB gaps
are under the floor, so the whole 264 MB tensor is read in 16 MiB pieces and
one eighth kept — 8× the bytes, but 7168 round trips (≈ 1.8 s at 250 µs each,
serialized per file) become 17 reads at 3 GB/s (≈ 90 ms). The planner prints
the numbers: when a `ReadRequest` carries the `CoalesceSummary` of its
selections, `explain()` adds `selections: N runs coalesced into M reads,
amplification x8.00 (gaps up to 585 KiB from profile … Nfs.io_cost_us 200 µs ×
single_file_gbps 3 GB/s; reads up to 16 MiB)`.

**Placement in the read engine.** A `Transfer` may carry `placements`
(empty: the range lands whole, as before). Validation checks the gather
invariants (`ReadError::InvalidPlacement` naming the placement and its
fault; `PlacedDestinationMismatch` when the destination is not exactly the
placements' total). A placed transfer is split into pieces along its
placements: a piece takes whole placements while they fit `split_bytes` (and
one staging buffer for a device destination), stops at the end of the last
that does, and the next piece starts at the next placement — so no piece
begins or ends inside a gap, a regular pattern stays regular piece by piece,
and only a placement longer than the block on its own is cut. A single
placement covering the whole range is normalised to an ordinary transfer, so
a whole-tensor read is unchanged. Host pieces are read into a per-worker
scratch buffer and copied out placement by placement. Device pieces always
stage (cuFile lands a range whole; a placed piece under a `CuFile` plan takes
the staging route by design, not as a fallback). A piece whose placements
are a regular 2-D pattern — equal lengths, constant source and destination
strides, the strided-slice case — lands with one
`DeviceCopier::copy_to_device_2d(src, dst, Copy2D { src_offset, src_pitch,
width, height, dst_offset, dst_pitch })`, `cudaMemcpy2DAsync` in `fst-cuda`
(`Call::ToDevice2D` in the simulator); an irregular piece lands with one
height-1 copy per placement. The report counts `gap_bytes` (read and
dropped; `bytes_read - gap_bytes` is what landed), `placed_pieces`, and
`scatter_copies` (per-placement copies for irregular pieces).

**Still slow.** Sub-byte dtypes whose runs are not byte-aligned are refused
rather than handled (a caller shards those on an outer dimension or pads).
Irregular scatters — a box partial on two or more dimensions, which arises
from selecting a sub-range on more than one axis, not from tensor-parallel
sharding — issue one device copy per placement; correct, counted, and a
candidate for a batched copy if it ever shows up in a profile. `io_cost_us`
is an **unmeasured placeholder** for NFS (200 µs) and absent elsewhere; the
multi-rank bench measures it.

### CUDA runtime

`fst-cuda` opens `libcudart` and `libcufile` with `dlopen` on first use and
resolves each symbol by name; nothing links against CUDA. Search order is the
soname (so `LD_LIBRARY_PATH` and `ldconfig` win) then the toolkit prefix:
`libcudart.so.13`, `.so.12`, `.so`, `/usr/local/cuda/lib64/libcudart.so.{13,12,}`;
`libcufile.so.0`, `.so`, `/usr/local/cuda/lib64/libcufile.so{,.0}`,
`/usr/lib/x86_64-linux-gnu/libcufile.so{.0,}` — the same paths `env::Env::probe`
checks. A missing library is `CudaError::Unavailable` naming it and every
candidate's `dlopen` message; `fst_cuda::probe()` prints the picture.

The copier holds one non-blocking stream per device. Copies on it do not
order against the legacy default stream, so a caller that has work in
flight on torch's default stream must synchronise before handing a pointer
down. `record()` returns a fence (`cudaEventRecord` + `cudaEventSynchronize`,
blocking-sync, no timing) so a thread can wait for its own copies while
others keep enqueueing.

Observed on the H100 node (CUDA 13.0, cuFile 1.15.1, no `nvidia_fs`,
`allow_compat_mode: true`):

- `cudaHostAlloc` of 16 MiB (zeroed): 6.5 ms; of 1 GiB: 386 ms. Pinned
  staging is worth pooling, not allocating per copy.
- H2D `cudaMemcpyAsync` of 1 GiB from pinned: 55 GB/s (PCIe Gen5). D2H the
  same order.
- cuFile opens the driver in compat mode. On the NFS home (`vers=3`),
  `O_DIRECT` opens are accepted, the handle registers,
  `cuFileBufRegister` succeeds for a 256 MiB destination, and `cuFileRead`
  lands 256 MiB at 2.5 GB/s through cuFile's POSIX bounce buffers — below
  the 3 GB/s a plain read of one NFS file gets, with nothing to gain until
  a GDS-capable mount exists.
- On `/tmp` (xfs over a software-RAID `md` device) `cuFileHandleRegister`
  fails with `CU_FILE_HANDLE_NOT_REGISTERED` (5027): libcufile asks udev for
  `ID_FS_USAGE` on the md device, finds none, and cannot build its file object —
  with or without `force_compat_mode`. The library surfaces it as
  `CudaError::Register` naming the file. So on this node cuFile cannot read
  local storage at all; the planner's `Transport::CuFile` stays gated on
  `nvidia_fs` and a block device libcufile recognises.
### python api

`fst-py` is a thin face over `fst-core` and `fst-cuda`: `parse_header` /
`parse_file_header` / `build_header` (format), `probe_env` / `probe_cuda` /
`storage_classes` / `plan_read` (env, plan), and two byte movers that run
the engines with the GIL released:

- `read_job(files, files_in_flight, split_bytes, readers_per_file,
  transport, staging, device)` builds a `fst_core::read::ReadJob` and calls
  `execute` with the `ReadPlan` those fields describe (the same plan
  `plan_read` returned; the Python layer passes it back rather than
  re-planning). `files` is `[(path, [(offset, nbytes, address,
  dst_offset, placements)])]`: `dst_offset=None` is `Dest::Host` over the
  address, an int is `Dest::Device { ptr: DevicePtr { address, nbytes:
  dst_offset + landed, device }, offset }`. `placements` is `None` when the
  range lands whole, else `[(src, dst, len)]` — a `select::Read`'s
  placements verbatim, put on the engine's `Transfer`; the landing (what the
  host slice must be exactly, what the device buffer must hold from its
  offset) is their total, and the engine refuses placements that are not a
  sorted, disjoint, packed gather of the range before any I/O. Returns the
  `ReadReport` (`bytes_read`, `pieces`, `files_opened`, `gap_bytes`,
  `placed_pieces`, `scatter_copies`, `elapsed`).
- `select_reads(items)` resolves `[(path, name, shape, ranges, dtype)]` boxes
  to the core's reads — `Selection::new`, `runs(dtype)`, `coalesce` under
  `plan::coalesce_policy_for(profile.lookup(env, path).entry)`, the machine
  probed once per call — as `[(offset, nbytes, dst, placements)]` per item
  plus the `CoalesceSummary` (`runs`, `reads`, `wanted_bytes`,
  `read_bytes`, `amplification`); `plan_read(..., coalesced=(runs, reads,
  wanted, read))` hands the summary back so the reasons carry the
  amplification line. `shard_ranges(shape, dim, rank, world)` is
  `Selection::shard`. `SelectError` maps to `errors.SelectError`.
- `write_object(path, specs, parts, metadata, durable, atomic, device)`
  builds a `fst_core::write::Payload` from `(name, dtype, shape, nbytes)`
  specs and `(address, nbytes, on_device)` parts — `Part::Host` over host
  memory, `Part::Device` for CUDA memory — and calls `write_object` with
  `WriteOptions { durable, atomic, staging: StagingRing::default() }` (four
  16 MiB pinned buffers) and the same cached runtime.

Those two are the crate's only `unsafe` (`io::host_slice`, `io::host_bytes`)
and the only pointer-trusting code; their callers are
`fastersafetensors._files` and `fastersafetensors._serialize`, which
allocate every destination and source as a contiguous tensor with torch,
pass `data_ptr()` with the exact byte length and device, keep the tensor
alive across the call, and `torch.cuda.synchronize(device)` before a device
pointer goes down (the runtime's copies run on a non-blocking stream that
does not order against torch's default stream; after the engine returns the
copies are complete, it synchronizes). Addresses are still checked for what
can be checked without trusting them: non-null, no wrap, no two destinations
overlapping, a device destination naming its device. Addresses rather than
buffer-protocol objects because the stable ABI (`abi3-py310`) only exposes
the buffer protocol from Python 3.11.

Errors: `FormatError`, `StorageError`, `PlanError`, `CudaError` map to the
Python class of the same name. `ReadError` / `WriteError` are composites: a
variant wrapping a domain error maps to that domain's class (a missing file
is a `StorageError` whichever engine met it, a failed copy a `CudaError`, a
`CuFile` plan on this build a `PlanError`), and the variants describing a
job the engine cannot run (`DestinationMismatch`, `StagingRequired`, a part
whose length disagrees with its spec, ...) map to `errors.ReadError` /
`errors.WriteError`. Every match is exhaustive; the `Display` text is the
message.

The Python layer (`_files.py`): headers read concurrently (16 at a time),
`plan_read` once for the request, every destination allocated with
`torch.empty(..., device=device)`, one `read_job` for the whole request.
The fit check counts result and scratch allocations, independently of read
amplification, against driver-free memory plus reusable torch allocator cache.
It does not guarantee that a contiguous allocation fits. The whole-load
allocator warm-up has been removed.
`safe_open` makes one plan per file for the open device (fit check off:
each read is a subset) and reads each tensor or slice block straight onto
the device. Non-CUDA accelerators (mps) read to a host tensor and `.to()`
afterwards, because the engine's device path is the CUDA runtime.
**Selections** (`_select.py`). `load_files(..., select={name: index |
Shard})` and `safe_open(...).get_sharded(name, dim, rank, world)` cut a
tensor before it is read. Python parses the index (ints, slices, `Ellipsis`,
`None`, as `t[index]` takes them) into a `Selection`: per dimension the
`[lo, hi)` range the box covers and the view that turns the box into the
result; a `Shard(dim, rank, world)` is `_core.shard_ranges`
(`Selection::shard`; `select_shards(names, dim, rank, world)` builds the
mapping, a dimension that does not divide is a `SelectError`). That is all
Python decides. `_files.Resolved` hands every pick's `(path, name, shape,
ranges, dtype)` to `_core.select_reads` in one call and gets the core's reads
back — runs coalesced under the profile's policy for each file's storage —
and `stage` turns each read into a `read_job` row: the read shifted by the
tensor's start, its placements verbatim, landing in the result tensor when
the result is the box's bytes (whole tensors, shards, inner cuts: ints and
`None` only reshape) or in a scratch box of the box's shape that torch's
`copy_` narrows when the index had a step. The `CoalesceSummary` goes to
`plan_read`, so `explain(..., select=...)` prints the amplification line
among the reasons and, per selection, `name: shape -> result, N bytes
wanted, M read in R read(s) over K run(s)`. `get_slice(name)[index]` goes
through the same `Selection`, so an inner-dimension slice reads its runs
(one coalesced read when the gaps are under the policy's) instead of the
outer covering block it read before.

`_serialize.py`: `save_file(tensors, path, metadata, *, durable=False,
atomic=True)` prepares each tensor where it lives (contiguous; CUDA tensors
stay on the device, one ordinal per write — parts on a second CUDA device
are copied to the host first; other accelerators are copied to the host),
synchronizes torch and calls `write_object`. `save() -> bytes` and
`serialize() -> Payload` copy device tensors to the host first: a payload is
bytes for a caller who owns the write, and there is no file to stream to.

## Calibration profile

The planner's rules are shaped by measurements, and the measurements differ per
machine. Rather than have the library calibrate itself, the figures arrive as a
JSON document — a `profile::Profile` — produced up front by whatever measured
the machine (the same figures a simulation of it would use). The library ships
one, `Profile::default_measured()`, which is the table in §Measurements and is
kept serialized at `docs/profiles/h100-nfs-2026-09-08.json` (a test
asserts the file equals `default_measured()`). A tool that emits this schema
plugs its numbers straight into `plan_read`.

Set `FASTERSAFETENSORS_PROFILE` to a calibration JSON path to merge an
explicit override into the built-in profile. Planning and selection coalescing
use the same override; invalid files raise `ProfileError`. `explain()` names
the selected profile and states that its applicability is not verified.
There is no hardware-based automatic profile selection.

### Schema (`schema_version` 1)

Top level:

| field | type | meaning |
|---|---|---|
| `schema_version` | integer | `1`. Anything else is refused. |
| `source` | string | Who produced the numbers: tool, host, date. Printed in every reason line. |
| `storage` | object | One entry per storage class, keyed `LocalBlock`, `Nfs`, `Fuse`, `Ram`, `OtherNetwork`, `Other` (the `env::StorageClass` names). Any subset; a class without an entry is planned with the conservative shape below. |
| `mounts` | object | Entries keyed by mount point (`"/nfs/home"`), overriding the class entry for every path under them. The longest mount point that is a prefix of the path wins — the rule `Env::storage_class` uses. |
| `device` | object or `null` | The host-to-device path. |

A storage entry (the value under a class or mount key):

| field | type | meaning |
|---|---|---|
| `single_file_gbps` | number > 0 | One file read alone, cold, GB/s (decimal) — whatever the readers or block size. The ceiling the mount puts on a file. Names the "one file caps near N GB/s" reason. |
| `aggregate_gbps` | array of `{files_in_flight, gbps}` | Aggregate cold throughput against files in flight; non-empty, strictly increasing in `files_in_flight`. The planner keeps in flight the smallest count within 15% of the table's maximum (`profile::PLATEAU_TOLERANCE`), capped at the files requested. |
| `split_helps` | boolean | Whether splitting one file across concurrent readers raises its throughput. True: `min(cpus, 16)` readers per file over 64 MiB pieces. False: each file read whole by one reader. |
| `page_cache_gbps` | number > 0 or `null` | Aggregate warm (page cache) throughput, for the record. |
| `open_cost_ms` | number ≥ 0 or `null` | `open(2)` cost on the mount, for the record. |
| `io_cost_us` | number ≥ 0 or `null` | Fixed cost of one read call beyond its bytes — the round trip a `pread` pays however small it is — in microseconds. With `single_file_gbps` it sets the coalescing gap for strided selections (§Selections and sharded reads): `clamp(io_cost_us × single_file_gbps × 1000 B, 64 KiB, 8 MiB)`. `null` is unmeasured: the floor applies. Optional with a default, so the schema stays at version 1. The built-in NFS entry carries **200 µs, an unmeasured placeholder** (a small NFSv3 read over TCP is a request, a reply, and the server's lookup: a few hundred microseconds on a datacenter LAN); the multi-rank bench measures it. |
| `gds` | object or `null` | `{ "registers": bool, "read_gbps": number or null }`. `registers` is whether `cuFileHandleRegister` succeeds on this storage; `read_gbps` is measured `cuFileRead` throughput. `null` means unmeasured and is treated as `registers: false`. The `CuFile` transport is chosen only when the environment has `nvidia_fs` and `libcufile` **and** every entry the request touches says `registers: true`. |

`device`:

| field | type | meaning |
|---|---|---|
| `h2d_pinned_gbps` | number > 0 | `cudaMemcpyAsync` host-to-device from pinned memory, GB/s. |
| `h2d_pageable_gbps` | number > 0 | The same from pageable memory. |
| `pinned_alloc_ms_per_mib` | number ≥ 0 | `cudaHostAlloc` cost per MiB. |

Validation on load (`ProfileError`, typed): unknown storage class key,
non-positive rate, aggregate table empty or not strictly increasing,
unsupported `schema_version`, unknown field anywhere (`deny_unknown_fields`).

### Example

```json
{
  "schema_version": 1,
  "source": "calib-tool 0.3 on an H100 node, 2026-09-08",
  "storage": {
    "Nfs": {
      "single_file_gbps": 3.0,
      "aggregate_gbps": [
        { "files_in_flight": 1, "gbps": 2.6 },
        { "files_in_flight": 4, "gbps": 7.4 },
        { "files_in_flight": 16, "gbps": 8.3 },
        { "files_in_flight": 32, "gbps": 9.7 }
      ],
      "split_helps": false,
      "page_cache_gbps": 25.9,
      "open_cost_ms": null,
      "io_cost_us": 200.0,
      "gds": { "registers": false, "read_gbps": 2.5 }
    },
    "LocalBlock": {
      "single_file_gbps": 3.0,
      "aggregate_gbps": [
        { "files_in_flight": 1, "gbps": 2.6 },
        { "files_in_flight": 4, "gbps": 7.4 }
      ],
      "split_helps": true,
      "page_cache_gbps": null,
      "open_cost_ms": null,
      "gds": { "registers": true, "read_gbps": null }
    }
  },
  "mounts": {
    "/tmp": {
      "single_file_gbps": 3.0,
      "aggregate_gbps": [{ "files_in_flight": 4, "gbps": 7.4 }],
      "split_helps": true,
      "gds": { "registers": false }
    }
  },
  "device": {
    "h2d_pinned_gbps": 55.0,
    "h2d_pageable_gbps": 7.2,
    "pinned_alloc_ms_per_mib": 0.4
  }
}
```

`page_cache_gbps`, `open_cost_ms`, `io_cost_us`, `gds` and `gds.read_gbps` may
be omitted (they default to `null`); `storage`, `mounts` and `device` may be
omitted too.
The `/tmp` override above is the measured H100 node: an xfs volume over an md
device that libcufile cannot register, so the class's `registers: true` is overruled for that mount.

### Merge rule

`Profile::merge(base, over)` combines two profiles section by section: a
storage class or mount point present in `over` replaces the base's entry for
it and the rest of the base's entries stay; `device` is `over`'s when it has
one, else the base's; `source` becomes `"<over.source> over <base.source>"`.
The intended use is `merge(&Profile::default_measured(), &node_profile)`: a
node's tool only has to describe what it measured.

### Where the defaults come from

`default_measured()` is §Measurements plus the CUDA runtime notes:

- `Nfs`: single file 3.0 GB/s (fastsafetensors one file at a time, 16
  threads, 64–256 MB blocks: 2.9–3.1); aggregate cold 2.6 / 7.4 / 8.3 / 9.7 at
  1 / 4 / 16 / 32 files (the 32 row is fastsafetensors nogds, 32 threads);
  page cache 25.9 at 16; `split_helps: false`; `gds.registers: false`,
  `read_gbps: 2.5` (cuFile compat mode through bounce buffers on NFSv3);
  `io_cost_us: 200` is an **unmeasured placeholder**, marked so in `source`,
  until a small-read round trip on the mount is measured.
- `LocalBlock`: **unmeasured placeholder**, marked so in `source`. The table
  is the NFS table cut at 4 files, which reproduces the M0 rule of four local
  files in flight, each split across readers; `gds.registers: true` is the M0
  assumption that a block device is GDS-capable when `nvidia_fs` is loaded.
  Replace with figures measured on a local mount.
- `Ram`: the warm page-cache figures stand in for memcpy reads: 7.2 GB/s one
  stream, 20.0 at 4, 25.9 at 16; `split_helps: true`; no cuFile.
- `device`: 55 GB/s pinned H2D (1 GiB, PCIe Gen5); `cudaHostAlloc` 386 ms per
  GiB ≈ 0.4 ms/MiB; pageable H2D unmeasured, set to the 7.2 GB/s single-stream
  figure as a floor.
- `Fuse`, `OtherNetwork`, `Other`: no entry. They get
  `StorageProfile::conservative()` — the `Nfs` entry — and the reason line
  says `no entry for …: using conservative NFS-shaped defaults`.

The plateau tolerance is 15%, not 10%: at 10% the measured NFS table would pick
32 files in flight (8.3 is 85.6% of 9.7), against the finding that 16 is
enough and 32 no better warm (17.7 against 25.9 GB/s).

## Public API (`fastersafetensors.torch`)

Mirror of `safetensors.torch`, same signatures, same results:

- `save_file(tensors, filename, metadata=None)`, `save(tensors, metadata=None) -> bytes`
- `load_file(filename, device="cpu") -> dict[str, Tensor]`, `load(data) -> dict`
- `safe_open(filename, framework="pt", device="cpu")` with `keys()`,
  `metadata()`, `get_tensor(name)`, `get_slice(name)`

Additions, all optional:

- `load_files(filenames, device, keys=None, select=None)`: a sharded
  checkpoint as one call, many files in flight. What causalab's loader will
  call. `select` maps names to an index (`(slice(None), slice(0, 64))`) or a
  `Shard(dim, rank, world)`; the tensor comes back at the selected shape and
  only its runs are read. `select_shards(names, dim, rank, world)` builds
  the mapping for a tensor-parallel rank; an undividable dimension is a
  `SelectError`.
- `safe_open(...).get_sharded(name, dim, rank, world)`: `torch.chunk(t,
  world, dim)[rank]` read without the rest of the tensor.
- `serialize(tensors, metadata) -> Payload` with `.header`, `.parts()`,
  `.nbytes`: the zero-copy form, for callers that own the write.
- `explain(filename_or_filenames, device, keys=None, select=None) -> str`:
  the plan, in words; per selection, bytes wanted against bytes read.
- GPU-resident tensors accepted by `save_file`/`save`: `save_file` drains
  them through the write engine's pinned ring, D2H overlapped with the
  write; `save` copies them to the host first.
- `save_file(..., durable=False, atomic=True)`: `fsync` before returning;
  temp-then-rename so the name never holds a half-written object.

## Milestones

**M0 — scaffold.** Workspace, crate skeletons, the format
module complete and tested against reference fixtures, storage traits with
posix / mmap / sim backends, env probe, planner v1 with the measured rules,
CUDA traits with a simulated runtime, PyO3 module stub, Python package stub.

**M1 — parallel workstreams**:

| workstream | delivers | tests |
|---|---|---|
| `feat/read-engine` | `fst-core::read`: execute a `ReadPlan` over `Storage` — thread pool, files in flight, per-file splits, one `RangeReader` per file, destination buffers filled exactly once; a `Destination` that is host slices or staged device pointers via `DeviceCopier` | sim storage + sim CUDA: every byte lands once, ops match the plan, faults propagate with the failing range named; proptest over plans × requests |
| `feat/write-engine` | `fst-core::write`: `Payload` (built header + caller parts) through `PartWriter`; device-resident parts drained through pinned staging with D2H overlapped against writes; optional `fsync`; atomic temp-then-rename | sim: parts written in order, staging never exceeds its ring, faults leave no half-named file; real posix round trip byte-identical to the reference |
| `feat/cuda-runtime` | `fst-cuda`: `dlopen` implementations of `DeviceCopier` (cudart: `cudaHostAlloc`, `cudaMemcpyAsync`, streams, `cudaMemGetInfo`) and `DirectStorage` (cufile: driver open, handle/buffer register, `cuFileRead`). All `unsafe` in one `ffi` module | unit tests for symbol resolution and error mapping without CUDA; integration tests (behind `FST_CUDA_TESTS=1`) run on a CUDA node |
| `feat/python-api` | `fst-py` + `fastersafetensors.torch`: the mirror API above; buffer protocol in, caller-owned pointers out; GIL released around I/O; torch dtype ↔ `Dtype` table | hypothesis: `save` bytes equal `safetensors.torch.save`; `load_file`/`safe_open` equal the reference on files the reference wrote and vice versa; `load_files` on a synthetic sharded checkpoint; `explain` names the rules |
| `feat/bench` | the benchmarks: the raw-bytes and end-to-end weight-loading arms, plus a write bench against `safetensors.save_file`; JSON lines out | run on the H100 node; the results are the tables in this document |

Shared vocabulary is fixed in M0: `format::{Dtype, TensorInfo, Layout,
TensorSpec, BuiltHeader}`, `storage::{Storage, RangeReader, PartWriter,
ReadRange}`, `env::Env`, `plan::{ReadRequest, ReadPlan, Transport, Staging}`,
`fst_cuda::{DevicePtr, DeviceCopier, DirectStorage, PinnedBuffer}`. A
workstream that needs a change to these says so when it proposes it.

**M2 — integrate and measure.** `load_files` on A3B on the H100 node against
`safetensors` threaded and `fastsafetensors`; `save_file` from GPU against the
reference; causalab's `weights.py` switched to `fastersafetensors.torch.load_files`.

**M3 — GDS.** The engine side is in place: `Transport::CuFile` executes
through `fst_cuda::CuFile`, and a file cuFile cannot register (as every file
on the H100 node's `/tmp`, an xfs volume over an md device) or a library that will not load
falls back to pread and staging at run time, recorded in the report — a
wrong planning decision costs a slow path, never a failed load. The whole
path is exercised against `fst_cuda::sim_direct::SimDirect` (per-path
`Register`, `Unavailable`, arbitrary cuFile errors on the n-th call). What
remains needs hardware: a node with `nvidia_fs` loaded and a mount whose
block device libcufile can resolve (`ID_FS_USAGE` via udev — md and dm
devices are what fail today). On such a node the calibration profile says
which mounts register; `Env::probe` then lets the planner choose `CuFile`
for them, and the benchmark measures cuFile against pread-plus-staging before the
rule is written down.
### M2 end to end

`load_files` of Qwen3.6-35B-A3B (26 shards, 1045 tensors, 71.9 GB) to
`cuda:0` on the H100 node (H100 80 GB, 16 cores, shards on NFS): one
`read_job` over all 26
files, plan (from `explain`) `16 files in flight, 1 reader per file, unsplit,
transport pread, staging 32 x 16 MiB pinned buffers`; cudart 13.0 via
`libcudart.so.13`, cuFile 1.15.0 present, no `nvidia_fs`. Every run is
a fresh process; cold evicts every shard with `posix_fadvise(DONTNEED)`
first (`/proc/meminfo` `Cached` grew by 71.9 GB during each cold run, by 0
during each warm one); **warm is the first run in the process** right after
a cold one. The timer covers `load_files` plus a final
`torch.cuda.synchronize`; CUDA context creation (0.3–0.5 s) is before it.
Node otherwise idle (GPU 0 MiB used; load average 1.4 before the first run,
6.3 after the last — these runs).

| run | allocator warm-up | seconds | GB/s | peak `max_memory_allocated` |
|---|---|---|---|---|
| cold 1 | yes | 8.03 | 8.96 | 71.9 GB |
| warm 1 | yes | 2.19 | 32.8 | 71.9 GB |
| cold 2 | yes | 8.44 | 8.52 | 71.9 GB |
| warm 2 | yes | 2.22 | 32.4 | 71.9 GB |
| cold | no | 7.79 | 9.23 | 71.9 GB |
| warm | no | 2.20 | 32.7 | 71.9 GB |

Against the references above: `safetensors` with 16 files in flight moved
the raw bytes at 8.3 GB/s cold / 25.9 warm; causalab's loader took 16 s
cold / 12 s warm end to end. Cold is the NFS mount's ceiling (16 files in
flight, ~3 GB/s per file); warm is above the earlier memcpy-bound figure
because the engine's pieces go pinned buffer → `cudaMemcpyAsync` with
sixteen file readers sharing thirty-two staging slots, no Python between.
Peak device memory is exactly the model: destinations are allocated per
tensor and nothing else lives on the device.

The former allocator warm-up (`_files.warm_allocator`: one `torch.empty` of the
total, freed at once, before the per-tensor allocations) made no measurable
difference here — with all 1045 destinations allocated up front and the
reads in one engine call afterwards, the `cudaMalloc` growth is not
interleaved with I/O and costs well under 0.1 s either way. It was removed in the upstream update: aggregate headroom does not
guarantee that a whole-load contiguous allocation fits.

**M3 — GDS**, when a node with `nvidia_fs` and a local NVMe staging path
exists. Until then the `CuFile` transport is planned but only unit-tested.

## Conventions

- `uv sync` builds the extension (maturin; `rustup` on PATH, the Rust version
  pinned in `rust-toolchain.toml`). `cargo fmt --all --check`,
  `uv run cargo clippy --workspace --all-targets -- -D warnings` (every crate,
  `fst-py` included — under `uv run` so pyo3's build script sees a Python ≥ 3.10)
  and `cargo test` (the pure crates); `uv run pytest
  tests/io/fastersafetensors` covers the Python surface and, through it,
  `fst-py`.
- Dependencies pinned; releases at least a week old at the time of pinning.
- No `unwrap`/`expect`/`panic` outside tests (workspace lints deny them).
  `unsafe` only in `fst-core::storage::mmap` (the map) and `fst-cuda::ffi`.
- Errors are enums per domain (`FormatError`, `StorageError`, `PlanError`,
  `CudaError`), mapped one-to-one onto the Python exception classes in
  `causalab.io.fastersafetensors.errors`.

**Integrated re-run (after every workstream merged; idle node, load 0.5; the
raw-read benchmark, two repeats, shared NFS cache):**

| arm | cold s | cold GB/s | warm s | warm GB/s | peak GB |
|---|---|---|---|---|---|
| fastersafetensors `load_files` | 7.95 | 9.0 | 2.18 | 32.9 | 71.9 |
| fastsafetensors nogds, 16 threads | 8.09 | 8.9 | 2.91 | 24.7 | 71.9 |
| safetensors, 16 files in flight | 10.22 | 7.0 | 6.19 | 11.6 | 71.9 |

All three arms produce the same checksum. Warm is the first run in a fresh
process. The stock transformers loader on the same checkpoint was 36 s cold and
17 s warm; causalab's threaded reader 16 s and 12 s.
