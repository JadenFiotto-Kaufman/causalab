# CUDA graphs

Enable supported CUDA replay with `causalab run … --device cuda --cuda-graphs`
or `PytorchHooksEngine(device="cuda", cuda_graphs=True)`. Eager execution is the
default. This execution option also applies to workflow runs; it does not alter
authored documents or their canonical digests. It does not enable `torch.compile`.

The supported path is a frozen, unquantized Qwen3 or Qwen3.6 text model with
eager attention, last-token block-output or logits reads, and at most one swap
write using an identity, gate, or Cayley subspace featurizer. Qwen3.6 requires
the grouped expert backend. Unsupported workloads use eager execution; INFO
logging explains the fallback.

Training keeps optimizer updates, annealing, metric scoring, and early stopping in
the ordinary Python loop. A cohort whose members are all graph-eligible is
captured as **one graph on a fixed slot layout**: each member owns
`min(train.batch.pairs, training row count)` rows of the captured frame, a shorter minibatch (an
epoch's remainder) is padded by repeating its last row with a loss weight of
zero, and one capture serves every step; a member that stops early keeps its
slot and is no longer stepped. A padded step is therefore not the eager
cohort's arithmetic — the padding rows change the MoE expert group sizes, so
values move at bf16 rounding — while a step whose slots are all full and active is the
eager cohort bit for bit. The members keep the campaign store: source
forwards are shared across them and the captured forward resumes below the
shallowest write from a prefix buffer of its own, filled from the store (a
prefix the store lacks is computed over the same frame, stopped at the resume
block). Prepared per-row masks, positions, and padded prefixes are cached per
minibatch; repeated steps copy them into captured storage without rebuilding
masks or padding. A cohort with an ineligible member, or an authored `fit_rows`
below its slot rows, keeps eager cohort batching and shared sources. A capture
that runs out of memory or encounters an incompatible layout releases its buffers
and hands the remaining passes to the eager cohort. Training capture requires
one input role with exactly one forward group per member; other layouts fall back
eagerly. Prefixes already stored from padded frames remain available after fallback,
so their bf16 rounding can differ from a fresh eager run.

Repeated cohort evaluation uses a separate graph, captured on the second use
of the same member and row layout. It keeps the ordinary row budget, shared
sources, prefix resume, and each member's evaluation mode. Read results stay on
the device after replay and are scored there — the answer columns a metric
selects copied to the host, not the vocabulary — and released before the next
replay, so metric scoring and early-stop decisions stay unchanged.
The graph reads the fit's existing parameter storage directly; only gate
temperatures need to be staged for evaluation.
Only one evaluation layout is captured per fit: a changed window or membership
releases it and continues eagerly. A one-off evaluation never pays capture cost.
Distinct evaluation splits also count as distinct layouts; no bank is kept per split.
Controllers, trajectory saves, phased training, a `constraint` term, a drawn
counterfactual role (its minibatch executors are rebuilt each epoch),
stochastic and budget gates, dead-unit rules, soft-accuracy and JS objectives,
objective-weight annealing, and continuation decoding use eager execution.
Inference batches requiring splitting under `batch_rows` also use eager execution.
The low-rank Cayley map is shared with eager execution; capture uses an unchecked
inverse to avoid host synchronization.

Captures belong to an execution request and are released on completion or
failure. Compatible fits can reuse captures automatically. No captures persist
to disk. A fit's graphs share one memory pool (below); a training allocation
OOM, in a capture or a replay, releases the fit's captures and retries eagerly.
Other CUDA capture errors still surface; recovery cannot be assumed after a
capture-invalidated error. Evaluation batches can still exceed the available
GPU memory.

## One pool per fit

A captured graph allocates from a memory pool, and a graph captured without
one is given a private pool that holds its whole working set — every
intermediate of the forward and backward — for as long as the graph lives. A
fit captures several graphs: a solo fit one training graph per padded shape
and attention mask its minibatches take (a *bucket*), plus its held-out
inference replays; a cohort its step graph, its evaluation graph, and its
members' inference replays. All of a fit's graphs are captured into **one
pool** (`GraphPool` in `cuda_graphs.py`; SGLang captures its batch-size
buckets into one pool the same way), so the fit's footprint is one working
set plus each graph's live outputs, not one working set per graph. The pool
comes with one capture stream: the allocator caches blocks per stream, so
graphs captured on streams of their own would share nothing even on one pool
(measured on an H100: three graphs reserve 3× on three streams, 1× on one).
There is no cap on the number of buckets and no memory guard before a
capture. The pool is released with the fit's graphs; its segments return to
the allocator when the pool itself is released, not once its last graph is
gone. A solo fit's pool belongs to its training bank, so a seed sweep's
compatible fits keep it along with their captures; a cohort's pool belongs to
the cohort run. The pool is per fit rather than per request because a fit is
the unit that releases everything at once: its buckets, its evaluation graph
and its held-out replays share one working set and die together, while the
next fit's shapes are its own; the fit cache already spans the compatible fits
of a seed sweep, which is where a per-request pool would have helped.

The pool is opened with `use_on_oom=True`, so an eager allocation that would
otherwise fail may take the pool's free blocks. When a capture or replay runs
out of memory, the holder releases its graphs and, if no other graph of the
fit holds the pool, the pool itself, so the eager remainder of the fit gets
the working set back, as it did when every graph had a private pool. That
release only counts graphs already captured: a training OOM before the fit's
held-out replays are captured closes the pool, and those replays then keep
private pools, as they did before. A pool closed while a graph is still alive
is a closing-order bug: it is logged as a warning and the graph is reset,
after which a replay of it raises.

Graphs on one pool hand each other the same blocks: a later capture can place
its outputs where an earlier graph's intermediates were, so replaying the
earlier graph overwrites them. Two invariants make this safe, and every
replay site keeps them:

1. **Graphs on one pool never replay concurrently.** Every replay is issued
   on the current stream, one at a time.
2. **Every value a replay produces that is read afterwards is consumed
   before another graph on the pool replays**, or lives outside the pool.
   A training or cohort step replay's gradients are read by the optimizer
   step that follows it in the same loop iteration, before any evaluation or
   inference replay; its loss is not read. An inference replay's captures
   and routing are cloned immediately after the replay. The evaluation
   graph's reads are scored — what the metric selects copied to the host —
   and released before the next replay on the pool. Everything a
   replay reads that the graph does not produce — tokens, masks, positions,
   labels, padding weights, staged operands, frozen source captures, the
   cohort's frames and prefix buffer — is allocated outside capture and so
   never sits in the pool.

## Compilation caches

A captured CUDA graph cannot be saved: every process captures its own, and the
graph flag neither compiles anything nor configures a compiler cache. What a
run *does* compile, and can share, are kernels: on a Gated DeltaNet model with
the `flash-linear-attention` extra, Triton builds the delta-rule kernels and
TileLang the Hopper backward kernels before the first forward (20–30 s on a
fresh cache for the six-step Qwen3.6-35B-A3B workflow), and a compile runner
that goes through Inductor adds Inductor's artifacts. Each compiler caches
under the user's home directory by default, so a fresh node, container or user
compiles again.

### One root, opted into

The cache is opt-in. `CAUSALAB_COMPILE_CACHE=<path>` names the root every
compiler a run uses is pointed at; with the variable unset or empty, each
compiler keeps its own default cache and nothing here is configured:

```bash
export CAUSALAB_COMPILE_CACHE=/shared/causalab-compile-cache  # one root for every compiler
causalab run benchmarks/cuda_graphs/standard.json --device cuda --out out/standard

CAUSALAB_COMPILE_CACHE= causalab run benchmarks/cuda_graphs/standard.json \
  --device cuda --out out/standard                          # no root: each compiler's own cache
```

Both engines' loaders point the compilers at the root as a model lands on a
CUDA device (`causalab/neural/shared/compile_cache.py`), before the first
kernel is built. Artifacts land under a **toolchain signature** — a hash of
the versions of torch, its CUDA runtime, triton, tilelang,
flash-linear-attention and transformers, the CPython ABI tag, and the GPU's
name and compute capability — so jobs share a directory exactly when their
artifacts are interchangeable, and incompatible environments never see each
other's files. (The ABI is there because Triton's cache also holds a compiled
launcher module keyed on its source and the platform, not the interpreter.)

```
<root>/<signature>/toolchain.json         what the signature stands for
<root>/<signature>/triton/                TRITON_CACHE_DIR
<root>/<signature>/tilelang/              TILELANG_CACHE_DIR
<root>/<signature>/inductor[-<policy>]/   TORCHINDUCTOR_CACHE_DIR
```

Within one directory the compilers are safe for concurrent writers on their
own: each publishes an artifact by writing a temporary file and renaming it
into place (Triton's `FileCacheManager.put`, TileLang's
`KernelCache._atomic_write`, Inductor's `write_atomic`), which POSIX
filesystems, NFS included, do atomically per directory, and reads a key only
once its files exist. No locking is added and none is needed. NFS attribute
caching can make a reader briefly miss a file another node just published; a
miss recompiles the same bytes. The manifest is written once, atomically; a
truncated one (an unclean shutdown between the write and the writeback) is
replaced on the next run.

**A shared root is shared across users.** A root whose mode is group-writable
— an operator's `mkdir -m 2770 <path>`, made once — is treated as one several
users write into: the directories `configure` creates there are `2770` and in
the root's group (a root without the setgid bit would otherwise hand them to
their creator's primary group), and the process umask is widened once to allow
group writes (logged at INFO). The second half is needed because the compilers
create their own per-kernel directories under the umask; with the default
`022`, a second user could read the first user's kernels but not complete a
kernel directory a killed job left partial, nor compile the same missing
kernel at the same time, and both end in `EACCES` inside the compiler. The
umask change is process-wide and deliberate: files the job writes afterwards
are group-writable too, which on a volume a team shares is the point. Any
other root is personal and follows the umask; a directory another job created
first keeps that job's bits.

**Who is on the other side of the root.** The artifacts are cubins and shared
objects that every participating job's compiler loads into its process, and
none of the three caches is a trust boundary: anyone who can write to the root
can run code in every job that reads it. A shared root is therefore a statement
that everyone in its group may run code in everyone else's jobs. Keep a root
writable by that group only and never world-writable: a world-writable root is
warned about on every path, and on a shared root that warning matters most —
whoever creates `<root>/<signature>` first supplies the artifacts every job
with that toolchain loads.

While the root is set it is the one setting: an explicit `TRITON_CACHE_DIR`,
`TILELANG_CACHE_DIR` or `TORCHINDUCTOR_CACHE_DIR` is overridden with a warning
(the point of the root is that all three compilers' artifacts sit in one signed
namespace; an operator who wants one compiler elsewhere leaves the root unset).
A root that cannot be created or written is a warning too, not a failed model
load: the compilers keep their own defaults, as they do with no root at all.

A compile runner that measures compilation through Inductor uses the same
layout, with its retained-operator policy naming the Inductor directory
(`inductor-aten-cumsum-exp-sum-<hash>`): custom lowering registrations are not
part of Inductor's own cache key, so a policy must not share a directory with
plain compilation even when every version matches.

Not shared, on purpose: CUDA graph captures (not serializable), FLA's autotune
"cache" (`FLA_CACHE_MODE` reads configuration files FLA ships per GPU; nothing a
run generates) and model weights (the Hugging Face hub cache).

### Verify reuse and measure startup

The loaders log the effective directory (`compile cache: <root>/<signature>`),
and `<root>/<signature>/toolchain.json` records what the signature stands for.
A cache hit does not eliminate artifact loading, specialization checks or CUDA
graph capture. Record whether compiler artifacts existed before the run, and
include model loading, capture, evaluation and saving in end-to-end timings.
Filesystem and page-cache state can affect loading time even when the compiler
reports hits.

For a fresh-compilation control, point `CAUSALAB_COMPILE_CACHE` at a new empty
root. Do not clear a root while other jobs use it. Revalidate numerical behavior
when changing the compile policy; successful cache loading is not evidence of
scientific equivalence.

## Verification

Verify on a CUDA host with cached model weights:

```bash
uv run pytest tests/golden/test_cuda_graphs.py tests/golden/test_graph_cohort.py tests/golden/test_graph_reuse.py tests/golden/test_cayley_capture.py -q
```
