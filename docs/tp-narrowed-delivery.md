# Tensor-parallel narrowed delivery

Coordinated loads (`load_files(group=...)`, `stream_files`) replicate every
tensor: one rank reads it, everyone receives it whole, and the consumer cuts
its tensor-parallel shard afterwards. Per rank that costs

    parameters + 2 x largest whole tensor + read-ahead + in-flight window

because the consumer's loop still references the previous whole tensor while
the next one lands. Nemotron-Ultra-253B at TP16 on 80 GB (30 GB of
parameters, 14 tensors of 13-14 GB) left 15 GB of slack; a larger tensor or
more parameters per rank does not load with any loader that honours that
contract.

`shards` removes the whole-tensor terms: the caller names, per tensor, the
shard each rank wants, and the rank receives only that shard. The largest
delivered tensor becomes the largest shard, and the bytes a sharded tensor
sends between nodes drop from `world x nbytes` to `nbytes`.

## API

```python
from causalab.io.fastersafetensors.torch import Shard, load_files, stream_files

shards = {name: Shard(dim, rank, world) for name, (dim, rank, world) in cuts}

load_files(files, device, keys, select, group=group, shards=shards)
stream_files(requests, device, group=group, shards=shards)
```

- `Shard(dim, rank, world)` is `torch.chunk(t, world, dim)[rank]`; `dim` may
  be negative, `shape[dim]` must divide by `world` (`SelectError` otherwise,
  before any byte moves), and `0 <= rank < world` is checked on construction.
- `rank` may differ per group member; `dim` and `world` must agree across the
  group (the request signature covers them; disagreement is a `PlanError`).
  `world` need not equal the group size, and two members may want the same
  shard (key/value heads replicated across a tensor-parallel group).
- A name in both `select` and `shards` is a `SelectError`. A name in
  `shards` that the request does not load is a `KeyError` for `load_files`
  and, for `stream_files`, a `SelectError` after the last request (names are
  matched request by request).
- Without `group`, `shards` behaves as `select` entries: the same tensors
  come back, read independently.

Tensors not named in `shards` are replicated exactly as before: owner
broadcast below `COOPERATIVE_BYTES` and for any narrowed or stepped
selection, cooperative read for a tensor selected whole above it. A narrowed
selection above the size therefore enters the in-flight window and counts in
the fit check.

## How a shard is read

A shard's bytes depend on where the cut falls. With `rows` the product of
the dimensions before `dim` and `cols` the product from `dim` on, the tensor
is `rows` rows of `cols` elements, each row `world` pieces, and the shard is
piece `rank` of every row (`_distributed.ShardLayout`).

- `rows <= 1` (a cut along the outer dimension, or leading dimensions of
  extent one): the shard is one contiguous run. The rank reads it straight
  into the delivered tensor as part of its ordinary chunk jobs (`Direct`),
  so many shards of one file are one long engine job, which is what network
  storage needs.
- `rows > 1` (a cut along an inner dimension): the shard is `rows` runs of
  `cols / world` elements. Reading them as runs is one short read per row,
  and reading through the gaps makes every rank read the whole tensor
  (the 8-process convoy on identical uncached pages that took Kimi-K3 to
  2.5 hours). Instead group rank `q` reads rows block `q` of the tensor
  (`torch.tensor_split(rows, group)`, one contiguous run, `1/group` of the
  bytes) in its chunk jobs, and at the tensor's turn one
  `all_to_all_single` sends each rank the pieces it wants of every row
  (`Exchange`). The pieces arrive in row order because the blocks are, so
  the receive buffer is the shard.

Cost model per sharded tensor of `N` bytes on a group of `G`:

| cut | storage bytes per rank | collective bytes per rank | transient memory per rank |
| --- | --- | --- | --- |
| outer (`Direct`) | `N / world` | none | none beyond the shard |
| inner (`Exchange`) | `N / G` | send `N / G`, receive `N / world` | row block `N / G` + its packed copy |

Replicated, for comparison: `N / G` from storage per rank (owner slice or
cooperative piece), `N` received by every rank, and `N` held by every rank.

## Memory accounting

Everything that bounds memory counts a sharded tensor at what the rank
holds: `_Prepared.sizes` are delivered bytes (the shard), `chunk_bytes` are
the bytes the rank's job reads (shard or row block), `largest` is the largest
delivered tensor, `broadcast_largest` sizes the in-flight window (sharded and
cooperative tensors never enter it), and `exchange_bytes` (the largest row
block, for the packed copy beside it) joins the fit check and the read-ahead
budget. `_check_fit` needs

    resident chunks + exchange copy + in-flight window + 2 x largest + pack buffer

within device headroom, and `resident_budget` halves what those terms leave.
The read-ahead scheduler and the chunk rounds are unchanged; sharded tensors
form one more chunk sequence, identical on every rank, so round `c` is the
same set of tensors everywhere and the collectives stay in one order.

## What the sglang side has to do

`fastersafetensors_weights_iterator` in `model_loader/weight_utils.py` marks
the seam. To use `shards` it must, before the read:

1. Build `shards` from the model. For every checkpoint name that lands in a
   parameter with `output_dim` (`ColumnParallelLinear`, `VocabParallelEmbedding`
   when the vocabulary divides) the shard is `Shard(output_dim, tp_rank,
   tp_size)`; with `input_dim` (`RowParallelLinear`) `Shard(input_dim,
   tp_rank, tp_size)`; for `QKVParallelLinear` key/value projections whose
   heads are replicated, `Shard(output_dim, tp_rank // replicas,
   num_kv_heads)`. The mapping from checkpoint name to parameter and shard
   id is what `load_weights` computes today (`stacked_params_mapping`); it
   has to run once ahead of the read, or the model has to expose it.
2. Tell `weight_loader` the tensor is already narrowed. `ColumnParallelLinear`,
   `MergedColumnParallelLinear` and `RowParallelLinear` already carry
   `use_presharded_weights`, which skips the `narrow`; the flag is per layer
   and set at construction, so either the loader constructs layers with it
   when it will shard, or the loader checks the incoming shape against the
   parameter shard and skips the narrow when they match. The second is the
   smaller change and keeps replicated tensors (those the loader did not
   shard) working through the same path.
3. Shard only tensors worth it. A shard smaller than a few MiB costs a read
   round trip and saves little; the linear weights and embeddings are the
   bytes that matter.

Until then, the `del loaded_weight` at the end of the model's
`for name, loaded_weight in weights` loop is the cheap interim: it removes
the consumer-held whole tensor and roughly doubles the slack.

## Not covered

- Fused checkpoint tensors that sglang narrows twice (a `gate_up_proj` stored
  fused in the checkpoint is split into shards, each narrowed by
  tensor-parallel rank): not one `Shard`. The loader has to leave such names
  replicated, or the library would need a list of cuts per tensor.
- Quantization scales with block layouts, Marlin repacking, bitsandbytes
  4-bit (which already reads its own portion), and any `weight_loader` that
  reshapes before narrowing.
- Expert tensors under `ep_size=1`: replicated per rank by design, and
  expert-parallel loads use `keys` without a group.
- The Direct path issues one read per shard; a request whose sharded tensors
  are many and small produces many short reads per rank. Batching several
  shards' runs into one engine job already happens (one job per chunk), but
  the runs themselves are as short as the shards.
