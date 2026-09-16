"""Reference forms of the gated delta rule, in plain float32 torch.

Two functions, both differentiable by autograd and both the numerics oracle
for the single-chunk kernel beside them (``triton_kernel.py``):

* :func:`recurrent_gated_delta_rule_reference` — the recurrence itself, one
  sequential step per token on the ``[K, V]`` state, exactly as transformers'
  ``torch_recurrent_gated_delta_rule`` spells it (``modeling_qwen3_5_moe.py``)
  but for a whole sequence and with the grouped-head layout FLA takes. Slow,
  obviously right, and the oracle every other form is measured against.
* :func:`single_chunk_gated_delta_rule_torch` — the closed form the kernel
  computes, for a sequence that fits one chunk with no initial state. It is
  the chunked algorithm with its inter-chunk half removed (module docstring
  of ``triton_kernel.py``), written as a handful of batched matmuls.

Layout is FLA's: ``q, k [B, T, H, K]``, ``v [B, T, HV, V]``, ``g, beta
[B, T, HV]`` with ``H | HV`` (a value head ``h`` reads key head
``h // (HV // H)``). transformers' mixer repeats q/k to ``HV`` heads before
the call, so in the model ``H == HV``; the grouped layout is kept so the
kernel can take the un-repeated tensors one day. ``g`` is the log-space
decay (non-positive), ``beta`` the post-sigmoid write gate. Outputs are
float32 whatever the inputs were: an oracle rounds nothing away.
"""

from __future__ import annotations

import torch

__all__ = [
    "l2norm",
    "expand_key_heads",
    "recurrent_gated_delta_rule_reference",
    "single_chunk_gated_delta_rule_torch",
]

#: transformers' and FLA's shared epsilon: ``x * rsqrt(sum(x*x) + eps)``.
L2NORM_EPS = 1e-6


def l2norm(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    """transformers' ``l2norm`` (``modeling_qwen3_5_moe.l2norm``), which FLA's
    ``l2norm_fwd`` matches: the epsilon sits inside the root."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def expand_key_heads(x: torch.Tensor, hv: int) -> torch.Tensor:
    """``[B, T, H, K] -> [B, T, HV, K]``: value head ``h`` reads key head
    ``h // (HV // H)`` — ``repeat_interleave``, as the mixer does it."""
    h = x.shape[2]
    if h == hv:
        return x
    if hv % h != 0:
        raise ValueError(f"value heads {hv} are not a multiple of key heads {h}")
    return x.repeat_interleave(hv // h, dim=2)


def _prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    hv = v.shape[2]
    qf = expand_key_heads(q.to(torch.float32), hv)
    kf = expand_key_heads(k.to(torch.float32), hv)
    if use_qk_l2norm:
        qf = l2norm(qf)
        kf = l2norm(kf)
    if scale is None:
        scale = float(k.shape[-1]) ** -0.5
    return (
        qf,
        kf,
        v.to(torch.float32),
        g.to(torch.float32),
        beta.to(torch.float32),
        scale,
    )


def recurrent_gated_delta_rule_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The recurrence, one token at a time in float32::

        S_t = exp(g_t) S_{t-1};  u_t = beta_t (v_t - S_t^T k_t)
        S_t = S_t + k_t u_t^T;   o_t = scale * S_t^T q_t

    Returns ``(o [B, T, HV, V], S_T [B, HV, K, V] or None)``, float32.
    """
    qf, kf, vf, gf, bf, scale = _prepare(q, k, v, g, beta, scale, use_qk_l2norm)
    b, t, hv, dk = kf.shape
    dv = vf.shape[-1]
    state = (
        torch.zeros(b, hv, dk, dv, dtype=torch.float32, device=vf.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    outs: list[torch.Tensor] = []
    for i in range(t):
        state = state * gf[:, i].exp()[..., None, None]
        kv_mem = (state * kf[:, i].unsqueeze(-1)).sum(dim=-2)
        delta = (vf[:, i] - kv_mem) * bf[:, i].unsqueeze(-1)
        state = state + kf[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * qf[:, i].unsqueeze(-1)).sum(dim=-2) * scale)
    o = torch.stack(outs, dim=1)
    return o, (state if output_final_state else None)


def single_chunk_gated_delta_rule_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The closed form for one chunk from a zero state (the kernel's math)::

        gc = cumsum(g);  D_ij = exp(gc_i - gc_j) for i >= j, else 0
        L  = strict_lower((beta k̂) k̂^T ⊙ D);  A = (I + L)^{-1}
        u  = A (beta v)
        o  = scale * (q̂ k̂^T ⊙ D) u
        S_T = Σ_t exp(gc_T - gc_t) k̂_t u_t^T          (final state, on request)

    Any ``T`` is accepted here (the matmuls do not care); the kernel pads to
    its tile. Returns ``(o [B, T, HV, V], S_T or None)``, float32.
    """
    qf, kf, vf, gf, bf, scale = _prepare(q, k, v, g, beta, scale, use_qk_l2norm)
    b, t, hv, _ = kf.shape
    # [B, HV, T, ·] for the batched matmuls
    qh, kh, vh = (x.transpose(1, 2) for x in (qf, kf, vf))
    gh, bh = gf.transpose(1, 2), bf.transpose(1, 2)
    gc = gh.cumsum(dim=-1)  # [B, HV, T]
    causal = torch.ones(t, t, dtype=torch.bool, device=vf.device).tril()
    decay = torch.where(causal, (gc.unsqueeze(-1) - gc.unsqueeze(-2)).exp(), 0.0)
    kb = kh * bh.unsqueeze(-1)
    lower = torch.where(causal.tril(-1), (kb @ kh.transpose(-1, -2)) * decay, 0.0)
    eye = torch.eye(t, dtype=torch.float32, device=vf.device)
    a_inv = torch.linalg.solve_triangular(
        eye + lower, eye.expand_as(lower), upper=False, unitriangular=True
    )
    u = a_inv @ (vh * bh.unsqueeze(-1))
    attn = (qh @ kh.transpose(-1, -2)) * decay * scale
    o = (attn @ u).transpose(1, 2)
    state = None
    if output_final_state:
        weights = (gc[..., -1:] - gc).exp()  # exp(gc_T - gc_t)
        state = (kh * weights.unsqueeze(-1)).transpose(-1, -2) @ u  # [B, HV, K, V]
    return o, state
