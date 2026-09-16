# Optional attention backends

The base install needs neither FlashAttention nor Flash Linear Attention (FLA).
The `pytorch_hooks` reference engine loads eager full attention; `nnsight`
keeps the model's Transformers default (normally SDPA). For supported linear
attention models such as Qwen3.5/3.6, Transformers supplies PyTorch delta-rule
and causal-convolution fallbacks when the optional packages are absent.

## Install

From a checkout on Linux with a supported NVIDIA GPU and CUDA-enabled PyTorch:

```bash
uv sync --extra flash-attn
uv sync --extra flash-linear-attention
# To keep both installed, select both in the same sync:
uv sync --extra flash-attn --extra flash-linear-attention
```

`flash-attn` installs FlashAttention 2. `flash-linear-attention` installs FLA's
CUDA dependencies and `causal-conv1d`, accelerating both the delta-rule kernel
and the short convolution used by Qwen's linear-attention layers. These kernels
serve models on a CUDA device only: Transformers binds them at import time
without checking where a tensor lives, so both engines bind a model whose
weights are not on CUDA (a CPU test model, a caller-owned CPU model) to
Transformers' torch implementations for the duration of each forward
(`causalab/neural/shared/kernels.py`). Installing the extra therefore leaves the
CPU test tiers and CPU runs unchanged. The extras
are independent of `nnsight`; a runtime install using that engine also needs
`--extra nnsight` (the dev group already includes it).

These extras are guarded by Linux package markers; selecting them on macOS or
Windows installs no attention kernels. CPU-only Linux users should omit them.
FlashAttention 2 needs supported GPU hardware and fp16/bf16 inputs; source
builds need a compatible CUDA toolkit, including `nvcc`, and a C++ compiler.
The upstream [FlashAttention installation guide](https://github.com/Dao-AILab/flash-attention#installation-and-features)
and [FLA installation guide](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)
describe hardware support. Set `MAX_JOBS` to limit compilation memory if needed.

Training through FLA on Hopper GPUs (including H100) also needs a working
backward kernel. FLA 0.5.2 rejects gated-delta backward with Triton versions
3.4.0 through 3.7.0 because of incorrect results. With that combination,
install `tilelang` in the same environment (FLA selects it automatically), or
use a compatible stack with Triton 3.7.1 or newer.

FLA reads its tuning knobs from its own environment at dispatch time, so none
of them needs causalab code (FLA 0.5.2, from its source): `FLA_TILELANG` (`1`
forces the tilelang `chunk_bwd_dqkwg` backward, `0` the Triton one — which FLA
refuses on Hopper with Triton 3.4–3.7.0, fla-org #640), `FLA_FLASH_QLA` (`0`
disables the FlashQLA backend, Qwen's fused tilelang forward and backward for
K = V = 128 on SM90/SM100, used automatically when the `flash_qla` package is
importable), `FLA_CACHE_MODE` with `FLA_CONFIG_DIR` (pinned `num_warps` /
`num_stages` / `BV` per Triton kernel from `<dir>/<kernel_name>.json` in place
of autotuning; the tilelang backward has no such table), `FLA_USE_TMA`,
`FLA_TRIL_PRECISION` (`ieee` / `tf32` / `tf32x3` in the triangular solve),
`FLA_USE_FAST_OPS`, `FLA_DISABLE_BACKEND_DISPATCH`. The one knob that is a
call argument rather than an environment variable, `chunk_size` ∈ {16, 32,
64} (default 64), cannot be set from outside the modeling code: Transformers'
hub wrapper filters keyword arguments to the implementation's named
parameters, and FLA takes it through `**kwargs`. Measured on one H100 with
Qwen3.6-35B-A3B (2026-09-11) no other value helps anyway — 16 fails inside
tilelang ("No valid warp partition for T.gemm: M=16"), 32 runs slower and
changes the fit — so the default is the only viable path on that stack.

Both compilers build their kernels before a model's first forward and cache
them on disk, under the home directory by default. To share those artifacts
across jobs on a common filesystem, set `CAUSALAB_COMPILE_CACHE` to a directory
there; see [compilation caches](cuda_graphs.md#compilation-caches).

The checkout configures isolated extension builds against the runtime PyTorch
version, following [uv's build dependency guidance](https://docs.astral.sh/uv/concepts/projects/config/#augmenting-build-dependencies).
Static metadata for the pinned extensions lets `uv lock` resolve on a machine
without CUDA. A source build still needs the toolkit; metadata does not supply
compiled kernels.

For pip from a checkout, install the base and build tools first, then the extras
without build isolation so they compile against that environment's PyTorch:

```bash
pip install -e .
pip install setuptools wheel packaging ninja
pip install --no-build-isolation -e '.[flash-attn,flash-linear-attention]'
```

## Select a backend

Choose the backend in the campaign or application JSON, alongside the model
and precision:

```json
{
  "model": {
    "key": "Qwen/Qwen3.6-35B-A3B",
    "revision": "main",
    "dtype": "bf16",
    "attn_implementation": "flash_attention_2"
  }
}
```

`model.attn_implementation` accepts `"eager"`, `"sdpa"`, or
`"flash_attention_2"`. Both engines read this field. The model and hardware
must support the requested backend; a missing or unusable backend raises
Transformers' error. Installing an extra provides the dependency; it does not
select the full-attention backend.

A workflow can set the same field for a protocol step using its existing
`set` object:

```json
{
  "set": {
    "model.attn_implementation": "flash_attention_2",
    "model.dtype": "bf16"
  }
}
```

The field supports ordinary sweep and bind wrappers, so backend comparisons
can be defined in the campaign itself. Each backend occupies a separate model
cache entry and can retain a full copy of the weights. For models close to GPU
memory capacity, compare backends in separate processes rather than one sweep. Explicit choices enter campaign,
forward-group and step digests and are stamped as
`model_attn_implementation` in tensor and fitted-featurizer artifacts. Applying
a fitted artifact checks this field when the application declares it.
Omission makes no backend compatibility assertion: explicitly select the same
backend in fit and apply documents when this check is required. Point receipts
and tensor entries also record `loaded_attn_implementation`, including when
the document omits the choice; `implementations` records eager requirements. A
caller-owned model must match the document's declared backend; the engine
refuses a mismatch before running it.

Omitting the field preserves the existing engine defaults: eager for hooks,
and the Transformers model default for nnsight. An explicit eager choice is
distinct from omission, since omission leaves that choice to the engine. The
engine constructor has no backend option: the JSON is the source of truth.

The nnsight executor temporarily switches to eager attention for traces that
need attention scores or probabilities, and restores the selected backend
afterward. The hooks executor similarly switches a forward to eager when it
reads or writes `attention_query`, `attention_key`, `attention_scores`,
`attention_probs`, or `attention_z`. Module-boundary interventions, including
residuals, MLPs, attention outputs, and value projections, keep the selected
backend. Prefill and decode use one backend for the whole continuation, including
when only a generated-token read needs attention interiors. The selection is
restored even when a forward raises. Prefix caches separate
activations by backend so an eager forward never resumes from an accelerated
prefix. A hooks run that switches to eager records `attn_eager` in its applied
implementation metadata. These temporary switches apply to caller-owned models too; wrapping
one in a bundle changes nothing.

FLA selection is separate from `attn_implementation`: Transformers' Qwen
modeling functions automatically dispatch to importable FLA and
`causal-conv1d` implementations. Start a fresh Python process after installing
or removing them, since Transformers resolves these functions during import.
This accelerates existing linear-attention layers; it does not turn ordinary
softmax attention into linear attention. Causalab's delta boundary taps wrap
the functions Transformers selected. Fused kernels need not expose the same
interior tensors.

## Return to the fallbacks

```bash
uv sync
```

An exact sync without the GPU extras removes their packages (include any other
extras you want to retain). Restart Python afterward. With pip, remove
`flash-attn`, `flash-linear-attention`, `fla-core`, and `causal-conv1d` from the
environment to restore the package-free paths. Setting eager full attention
alone does not disable FLA in linear-attention layers.

Neither GPU extra is in the base `requirements.lock.txt` export or the default
dev group. Ordinary installs and CPU tests therefore retain their existing
dependency footprint.

## Reproducibility and batching variance

Accelerated kernels introduce numerical nondeterminism through batching
variance: the same example can produce different floating-point results when
batch size, batch composition, padding, or row chunking changes. A fixed random
seed does not eliminate this variance. Differences can affect logits, generated
tokens, intervention metrics, and fitted parameters.

For reproducible comparisons, keep the backend, precision, batching, hardware,
and package versions fixed, and compare numerical results with tolerances.
Selecting eager full attention alone does not remove variance from FLA;
remove the linear-attention extras as described above to use the Transformers
fallbacks. The fallbacks also do not guarantee bitwise equality across hardware
or batching changes.
