"""The single-chunk gated delta rule as an autograd function over the Triton
programs of ``_triton_kernels.py``, behind FLA's calling convention.

Why a separate kernel for short sequences
-----------------------------------------

FLA's ``chunk_gated_delta_rule`` (0.5.2) has one tile size, ``BT = chunk_size
∈ {16, 32, 64}``, 64 unless the caller says otherwise — and transformers'
hub wrapper drops the keyword, so the model always runs 64. Nothing in the
chunked path shortcuts ``T < BT``: ``NT = cdiv(T, BT) = 1`` and every kernel
runs its full tile with the rows past ``T`` masked. At the workflow's
``T = 13`` that is 80 % padding in every tile, and the inter-chunk half of
the algorithm — ``chunk_gated_delta_rule_fwd_h`` (the state carried to the
next chunk), ``chunk_gated_delta_rule_bwd_dhu`` (its gradient), and the
``h``/``dh`` terms of ``chunk_bwd_dqkwg`` — is computed for a single chunk
started from a zero state, where it is identically zero work. (``chunk_size
= 16`` takes the unfused ``kkt`` + ``solve_tril`` intra-chunk path and a
tilelang backward whose 16-row tiles it refuses to build; ``32`` runs but
slower; ``FLA_TILELANG=0`` is refused outright on Hopper with Triton
3.4–3.7.0, fla-org #640; ``fused_recurrent_gated_delta_rule`` has no
backward at all. So there was no configuration to switch to.)

With ``initial_state=None`` and ``T ≤ BT`` the rule has a closed form in
``[T, T]`` and ``[T, K]`` tiles (``reference.single_chunk_gated_delta_rule_torch``)::

    gc = cumsum(g);  D_ij = exp(gc_i - gc_j) [i >= j]
    A  = (I + strict_lower((beta k̂) k̂^T ⊙ D))^-1
    u  = A (beta v);   o = scale * (q̂ k̂^T ⊙ D) u

One Triton program per (sequence, value head) computes it — forward, or the
backward with the forward's tiles recomputed — in registers, in float32
(the dots on tensor cores as three TF32 passes, :data:`DOT_PRECISION`),
reading each input once and writing each output once. There is no state tile, no scratch, no host synchronization and no
autotuning: the launch is CUDA-graph capture-safe and compiles in the warm-up
pass of a capture. The final state, when a caller asks for one, is
``Σ_t exp(gc_T - gc_t) k̂_t u_t^T`` — a batched matmul in torch over the
kernel's ``u``, differentiable through autograd like anything else.

Numerics (documented change). FLA rounds the normalized q̂/k̂ to bf16
before its tiles, keeps ``A``, ``w``, ``u`` and the chunk state ``h`` in
bf16, and runs its triangular solve at tf32. This kernel keeps everything in
float32 (its 3×TF32 dots measured identical to IEEE float32 to four digits)
and rounds only ``o`` and the gradients on store — so it sits
closer to the float32 oracle than the path it replaces, and differs from
FLA at bf16 rounding level (the measured deltas are in the parity tests).
Inputs may be bf16, fp16 or fp32.

Scope: ``T ≤`` :data:`MAX_SEQ_LEN`, ``initial_state=None``, no
``cu_seqlens``, ``K`` and ``V`` powers of two in ``[16, 256]``, ``H | HV``.
The binding (``binding.py``) checks all of it before routing a call here;
a direct call with anything else is refused by name.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
from typing import Any

import torch

from causalab.neural.shared.gdn_short.reference import (
    L2NORM_EPS,
    expand_key_heads,
    l2norm,
)

__all__ = [
    "MAX_SEQ_LEN",
    "ShortSeqUnsupported",
    "single_chunk_gated_delta_rule",
    "tile_rows",
    "triton_available",
]

#: The longest sequence the closed form is tiled for: two tile sizes, 16 and
#: 32 rows. Larger tiles hold ``[BT, 128]`` float32 operands a program cannot
#: keep in registers; longer sequences are FLA's.
MAX_SEQ_LEN = 32

#: Warps per program. Measured on one H100 (2026-09-15, the
#: ``_gpu`` matrix, ``[96, 13, 32, 128]`` bf16): forward 4 warps 164 µs vs
#: 2 warps 192 / 8 warps 245; backward 4 warps 423 µs vs 8 warps 650 (the
#: dozen ``[16, 128]`` float32 tiles spill less at 4). Module attributes so a
#: sweep can rebind them.
FWD_NUM_WARPS = 4
BWD_NUM_WARPS = 4

#: ``tl.dot`` input precision for the float32 tiles: ``"tf32x3"`` (tensor
#: cores, three TF32 passes — float32-accurate), ``"ieee"`` (FMA, exact
#: float32) or ``"tf32"`` (tensor cores, 10-bit mantissa). Same H100 matrix,
#: at 4/4 warps: ``tf32x3`` 499 µs fwd+bwd vs ``ieee`` 587 at B=96 (377 vs
#: 409 at B=42) with the output and every gradient identical to ``ieee`` to
#: four digits against the float32 oracle; ``tf32`` was faster still but
#: moved the output error from 2.3e-4 to 4.1e-4 (FLA: 5.8e-4). The numerics
#: statement is for the default; ``ieee`` is the exact fallback.
DOT_PRECISION = "tf32x3"


class ShortSeqUnsupported(ValueError):
    """A call the single-chunk kernel does not cover — the binding never
    routes one here, so this names a direct caller's mistake."""


@functools.cache
def triton_available() -> bool:
    """Whether ``triton`` is importable — the kernel's only requirement
    beyond a CUDA tensor. Looked up once: the binding asks on every forward."""
    return importlib.util.find_spec("triton") is not None


def tile_rows(seq_len: int) -> int:
    """The tile the kernel pads ``seq_len`` to: 16 or 32 rows."""
    if seq_len < 1 or seq_len > MAX_SEQ_LEN:
        raise ShortSeqUnsupported(
            f"the single-chunk kernel covers 1 <= T <= {MAX_SEQ_LEN}, got T={seq_len}"
        )
    return 16 if seq_len <= 16 else 32


def _check_head_dim(name: str, size: int) -> None:
    if size < 16 or size > 256 or size & (size - 1):
        raise ShortSeqUnsupported(
            f"the single-chunk kernel takes a power-of-two {name} in [16, 256], got {size}"
        )


def _kernels() -> Any:
    return importlib.import_module("causalab.neural.shared.gdn_short._triton_kernels")


class _SingleChunkGatedDeltaRule(torch.autograd.Function):
    """``(o, u) = f(q, k, v, g, beta)``; ``u`` is returned only when the
    caller needs it for the final state (``None`` otherwise, so autograd
    never materializes a zero cotangent for it)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        use_l2norm: bool,
        want_u: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t, h, dk = k.shape
        hv, dv = v.shape[2], v.shape[3]
        bt = tile_rows(t)
        o = torch.empty(b, t, hv, dv, dtype=q.dtype, device=q.device)
        u = (
            torch.empty(b, t, hv, dv, dtype=torch.float32, device=q.device)
            if want_u
            else None
        )
        _kernels().single_chunk_fwd_kernel[(b * hv,)](
            q,
            k,
            v,
            g,
            beta,
            o,
            u if u is not None else o,
            t,
            scale,
            L2NORM_EPS,
            H=h,
            HV=hv,
            K=dk,
            V=dv,
            BT=bt,
            L2NORM=use_l2norm,
            STORE_U=u is not None,
            PREC=DOT_PRECISION,
            num_warps=FWD_NUM_WARPS,
        )
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.scale = scale
        ctx.use_l2norm = use_l2norm
        ctx.set_materialize_grads(False)
        return o, u

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, do: torch.Tensor | None, du: torch.Tensor | None
    ) -> tuple[Any, ...]:
        q, k, v, g, beta = ctx.saved_tensors
        b, t, h, dk = k.shape
        hv, dv = v.shape[2], v.shape[3]
        if do is None:
            do = torch.zeros(b, t, hv, dv, dtype=v.dtype, device=v.device)
        do = do.contiguous()
        du = du.contiguous() if du is not None else None
        # dq/dk per value head; summed over the group below when H < HV, so
        # they are float32 there and land in q's dtype directly otherwise
        grad_dtype = q.dtype if h == hv else torch.float32
        dq = torch.empty(b, t, hv, dk, dtype=grad_dtype, device=q.device)
        dk_ = torch.empty(b, t, hv, dk, dtype=grad_dtype, device=q.device)
        dv_ = torch.empty_like(v)
        dg = torch.empty_like(g, dtype=torch.float32)
        dbeta = torch.empty_like(beta)
        _kernels().single_chunk_bwd_kernel[(b * hv,)](
            q,
            k,
            v,
            g,
            beta,
            do,
            du if du is not None else do,
            dq,
            dk_,
            dv_,
            dg,
            dbeta,
            t,
            ctx.scale,
            L2NORM_EPS,
            H=h,
            HV=hv,
            K=dk,
            V=dv,
            BT=tile_rows(t),
            L2NORM=ctx.use_l2norm,
            HAS_DU=du is not None,
            PREC=DOT_PRECISION,
            num_warps=BWD_NUM_WARPS,
        )
        if h != hv:
            dq = dq.view(b, t, h, hv // h, dk).sum(3).to(q.dtype)
            dk_ = dk_.view(b, t, h, hv // h, dk).sum(3).to(k.dtype)
        return dq, dk_, dv_, dg.to(g.dtype), dbeta, None, None, None


def single_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FLA's ``chunk_gated_delta_rule`` signature, for one chunk from a zero
    state. ``q, k [B, T, H, K]``, ``v [B, T, HV, V]``, ``g, beta [B, T, HV]``;
    returns ``(o [B, T, HV, V] in q's dtype, final state [B, HV, K, V]
    float32 or None)``. Extra keywords (FLA's tuning knobs) are ignored, as
    FLA ignores the ones it does not know."""
    del kwargs
    if initial_state is not None:
        raise ShortSeqUnsupported("the single-chunk kernel starts from a zero state")
    if cu_seqlens is not None:
        raise ShortSeqUnsupported(
            "the single-chunk kernel takes equal-length sequences"
        )
    if q.shape != k.shape or q.shape[:2] != v.shape[:2] or g.shape != beta.shape:
        raise ShortSeqUnsupported(
            f"shapes q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}, "
            f"g {tuple(g.shape)}, beta {tuple(beta.shape)} do not agree"
        )
    h, hv = q.shape[2], v.shape[2]
    if hv % h:
        raise ShortSeqUnsupported(
            f"value heads {hv} are not a multiple of key heads {h}"
        )
    _check_head_dim("K", q.shape[-1])
    _check_head_dim("V", v.shape[-1])
    if not q.is_cuda:
        raise ShortSeqUnsupported("the single-chunk kernel runs on CUDA tensors")
    tile_rows(q.shape[1])
    if scale is None:
        scale = float(k.shape[-1]) ** -0.5
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, beta = g.contiguous(), beta.contiguous()
    o, u = _SingleChunkGatedDeltaRule.apply(  # type: ignore[no-untyped-call]
        q, k, v, g, beta, scale, use_qk_l2norm_in_kernel, output_final_state
    )
    if not output_final_state:
        return o, None
    assert u is not None
    # S_T = Σ_t exp(gc_T - gc_t) k̂_t u_t^T, in float32 as FLA returns it
    kf = expand_key_heads(k.to(torch.float32), hv)
    if use_qk_l2norm_in_kernel:
        kf = l2norm(kf)
    gc = g.to(torch.float32).cumsum(dim=1)  # [B, T, HV]
    weights = (gc[:, -1:] - gc).exp()
    state = torch.einsum("bthk,bthv->bhkv", kf * weights.unsqueeze(-1), u)
    return o, state
