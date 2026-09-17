"""The grouped-experts forward this engine runs: transformers' ``grouped_mm``
function without the two kernels a single-process model pays for nothing.

``transformers.integrations.moe.grouped_mm_experts_forward`` (5.16) sorts the
``S = tokens · top_k`` (token, slot) pairs by expert, gathers the hidden
states into that order, runs the two grouped projections, weights, un-sorts
and reduces. Two of its lines exist for **expert parallelism**: under EP the
router marks a slot routed to another rank with the sentinel expert id
``num_local_experts``, the sort pushes those rows to the tail, ``histc``
drops them from the offsets so ``grouped_mm`` skips them and leaves their
output rows uninitialised — and a pre-mask on the gathered input and a
post-mask on the weighted output (``masked_fill_(sentinel_mask, 0.0)``) zero
what would otherwise be NaN. In one process there is no sentinel: every id is
below ``num_experts``, both masks are all-``False``, both forward fills are
no-ops — and both **backwards** still run, ``grad.masked_fill(mask, 0)`` over
the full ``(S, hidden)`` gradient each. 📐 Profiled (``fullprof0910``, das
step 5): 56 launches × 64 µs per optimizer step, ALU-bound, 5 % of the run's
kernel time, on the autograd thread with no Python frame — the backward of
those two lines.

:func:`lean_grouped_mm_forward` is the same function with the masks left out
when the module cannot see a sentinel (:func:`may_route_to_sentinels`: the
model was not loaded with ``distributed_config.enable_expert_parallel``), and
one more change to the backward alone: the un-sort ``weighted_out[inv_perm]``
is a permutation, so its gradient is the gather ``grad[perm]`` —
:class:`_PermuteRows` says so, where autograd's generic index backward would
sort the indices and accumulate (``indexing_backward_kernel``) as if two rows
could coincide. Every number the forward produces, and every gradient, is
bit-identical to the library's: a mask that is all-``False`` changes nothing,
a permutation's transpose is its inverse, and the test suite pins both
against the library function on the tiny MoE (output, input gradient, both
expert-weight gradients, ``torch.equal``). A model under EP is handed to the
library function unchanged.

Containment mirrors :mod:`.experts_interface`: :func:`lean_experts_path`
installs the function as the ``"grouped_mm"`` entry of ``ALL_EXPERTS_FUNCTIONS``
for the duration of one engine forward and restores the previous entry on
exit — so the nnterp engine, whose ``.source`` address table descends into
the library function's own body, never sees it. The executor enters it
**before** the experts-interface taps, which capture whatever ``"grouped_mm"``
dispatches to at their entry and wrap it: the tapped interior is then this
function's, and its two ``_grouped_linear`` calls — reached through the module
attribute, as the library's are — are the two the taps count.

On CUDA with Triton importable, the glue around the two grouped linears runs
as the fused kernels of :mod:`.kernels.moe_glue` where each one's plan admits
it — a stable counting sort in place of ``torch.sort`` + ``histc`` +
``cumsum`` + the inverse scatter, a row gather whose backward folds each
token's slots in the index backward's own order and rounding, one kernel
each way for the weight multiply, un-sort, slot sum and cast, and
``silu(gate) * up``. Each reproduces the ATen order it replaces bit for bit
(:mod:`.kernels.moe_glue_reference` states those orders with their ATen
sources); the two ``_grouped_linear`` calls stay as they are. The plan is
decided before any tensor op and is empty off CUDA, so this tier runs the
eager lines below unchanged; ``CAUSALAB_MOE_GLUE`` (``shared/kernel_options.py``)
switches kernels off.

The library function is the source of truth for everything else here: the
suite's drift canary reads its source and fails when the lines this module
mirrors (the two masks, the two grouped linears, the un-sort) are no longer
where this docstring says, so a transformers bump re-examines this copy
rather than silently diverging from it.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch

from causalab.neural.engines.pytorch_hooks.kernels import moe_glue
from causalab.neural.shared.kernel_options import MoeGlueOptions

__all__ = [
    "has_default_silu_gate",
    "lean_experts_path",
    "lean_grouped_mm_forward",
    "may_route_to_sentinels",
]


def has_default_silu_gate(module: Any) -> bool:
    """Whether the experts module's gate is the library's default
    ``silu(gate) * up`` — ``_apply_gate`` is ``_default_apply_gate`` and
    ``act_fn`` is silu (``torch.nn.SiLU``, or transformers' own
    ``SiLUActivation`` that ``ACT2FN["silu"]`` builds, whose forward is
    ``F.silu``) — the one the fused gate kernel reproduces; a custom gate or
    another activation keeps the module's. Forward hooks on ``act_fn`` also
    keep the module call so intervention taps can read and replace its output.
    """
    import transformers.activations as activations
    import transformers.integrations.moe as moe

    if not getattr(module, "has_gate", False):
        return False
    apply_gate = getattr(module, "_apply_gate", None)
    default = getattr(apply_gate, "__func__", apply_gate) is moe._default_apply_gate
    silu = (torch.nn.SiLU, activations.SiLUActivation)
    activation = getattr(module, "act_fn", None)
    return (
        default
        and isinstance(activation, silu)
        and not activation._forward_hooks
        and not activation._forward_pre_hooks
    )


def may_route_to_sentinels(module: Any) -> bool:
    """Whether the experts module's routing table can hold the EP sentinel
    id — ``num_local_experts``, the id the router writes into a slot owned
    by another rank, one past the module's local expert range. Decided
    positively, so a premise that cannot be read takes the library path
    (whose masks make a sentinel harmless) rather than this module's (which
    would leave the sentinel rows uninitialised):

    * the model was loaded with ``distributed_config.enable_expert_parallel``
      (read from the config the experts module carries — ``self.config``, set
      by transformers' ``use_experts_implementation`` decorator); or
    * the module's id space is not its weights' whole expert axis:
      transformers' ``MoEParamShard`` rewrites ``module.num_experts`` to the
      per-rank count while the parameter keeps its global expert dim, so
      ``num_experts != weight.shape[0]`` is a sharded module however it was
      asked for — and a module on which neither can be read counts as one.
    """
    config = getattr(module, "config", None)
    distributed = getattr(config, "distributed_config", None)
    if getattr(distributed, "enable_expert_parallel", False):
        return True
    weight_name = "gate_up_proj" if getattr(module, "has_gate", True) else "up_proj"
    weight = getattr(module, weight_name, None)
    num_experts = getattr(module, "num_experts", None)
    if weight is None or num_experts is None:
        return True
    return int(weight.shape[0]) != int(num_experts)


class _PermuteRows(torch.autograd.Function):
    """``rows[order]`` for a permutation ``order`` whose inverse is known:
    the backward is the gather by the inverse, not a sorted scatter-add."""

    @staticmethod
    def forward(
        ctx: Any, rows: torch.Tensor, order: torch.Tensor, inverse: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(inverse)
        return rows[order]

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (inverse,) = ctx.saved_tensors
        return grad[inverse], None, None


def lean_grouped_mm_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """``grouped_mm_experts_forward`` (module docstring) without the sentinel
    masks, for a module that cannot route to a sentinel; the library function
    itself for one that can."""
    import transformers.integrations.moe as moe

    if may_route_to_sentinels(self):
        return moe.grouped_mm_experts_forward(
            self, hidden_states, top_k_index, top_k_weights
        )

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    # which glue runs fused (kernels/moe_glue.py): decided before any tensor
    # op, empty off CUDA or without Triton
    plan = moe_glue.plan_moe_glue(
        hidden_states=hidden_states,
        top_k_weights=top_k_weights,
        expert_dtype=self.down_proj.dtype,
        num_pairs=expert_ids.numel(),
        top_k=num_top_k,
        default_silu_gate=has_default_silu_gate(self),
        options=MoeGlueOptions.from_env(),
    )

    # S = tokens · top_k (token, slot) pairs, sorted by expert; inv_perm is
    # the un-sort; offsets the per-expert inclusive counts grouped_mm takes
    if plan.sort:
        perm, inv_perm, offsets = moe_glue.counting_sort(expert_ids, self.num_experts)
        expert_ids_g = expert_ids[perm] if self.has_bias else None
    else:
        expert_ids_g, perm = torch.sort(expert_ids)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)
        # histc rather than bincount, as the library does (CUDA-graph safe);
        # CPU/MPS histc wants a float input
        histc_input = (
            expert_ids_g.float()
            if device.type in ("cpu", "mps")
            else expert_ids_g.int()
        )
        tokens_per_expert = torch.histc(
            histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    # no sentinel: every id is below num_experts, so offsets[-1] == S and the
    # kernel writes every row — nothing to clamp, nothing to mask

    if plan.gather:
        selected_hidden_states_g = moe_glue.fused_gather(
            hidden_states, perm, inv_perm, num_top_k
        )
    else:
        selected_hidden_states_g = hidden_states[perm // num_top_k]

    if self.has_gate:
        selected_weights = self.gate_up_proj
        selected_biases = (
            self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
        )
    else:
        selected_weights = self.up_proj
        selected_biases = self.up_proj_bias[expert_ids_g] if self.has_bias else None

    # the fused [gate | up] (or plain up) projection, per expert
    proj_out = moe._grouped_linear(
        selected_hidden_states_g,
        selected_weights,
        offsets,
        bias=selected_biases,
        is_transposed=self.is_transposed,
    )
    if plan.gate:
        proj_out = moe_glue.fused_gate(proj_out)
    elif self.has_gate:
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # the down-projection, per expert
    selected_biases = self.down_proj_bias[expert_ids_g] if self.has_bias else None
    proj_out = moe._grouped_linear(
        proj_out,
        self.down_proj,
        offsets,
        bias=selected_biases,
        is_transposed=self.is_transposed,
    )

    if plan.epilogue:
        # weight, un-sort, slot sum and cast in one launch each way; the
        # routing weights are read in token order, so no gather of them
        return moe_glue.fused_epilogue(
            proj_out, sample_weights, inv_perm, perm, num_top_k, hidden_states.dtype
        )

    sample_weights_g = sample_weights[perm]
    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)

    # un-sort: a permutation, whose backward is the gather by `perm`
    weighted_out = _PermuteRows.apply(weighted_out, inv_perm, perm)

    # the library's deterministic reshape+sum over the slots (fp32 accumulate)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(
        dim=1
    )
    return final_hidden_states.to(hidden_states.dtype)


@contextlib.contextmanager
def lean_experts_path() -> Iterator[None]:
    """While active, ``"grouped_mm"`` dispatches to
    :func:`lean_grouped_mm_forward`; the entry that was there is put back on
    exit (restore-not-delete, as :func:`.experts_interface_taps`)."""
    import transformers.integrations.moe as moe

    previous = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
    moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] = lean_grouped_mm_forward
    try:
        yield
    finally:
        moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] = previous
