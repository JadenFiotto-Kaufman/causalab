"""Interior addresses over nnsight ``.source`` — the upstreamable half.

The module-boundary vocabulary needs no table: envoys mirror the module tree,
so the shared site map addresses them directly. The *interiors* — tensors
``transformers`` computes inside one function call — are reached through
nnsight's ``.source``, which names every call and assignment in a forward.
This module is the table of those names and the matcher that resolves them,
and deliberately nothing else:

* **the table is pure data** — one :class:`SourceAddress` per
  ``(family tree, component)``: the op path from the anchor's ``.source``,
  which handle carries the value, how often it fires, and the row
  bookkeeping around it;
* **the matcher is pure string logic** — substring match, call-op
  disambiguation, exactly-one-hit or a refusal carrying the full op
  inventory;
* **the navigation lives in the executor** — recursive ``.source`` drilling
  only works inside a trace, so the lines that walk a resolved address stay
  in :mod:`causalab.neural.engines.nnsight_nnterp.executor`.

**This module imports nothing from ``causalab``** (enforced by a test): it is
exactly the hybrid/interior accessor layer nnterp's issue #18 asks for, and
keeping it protocol-free is what makes "move it upstream later" a file move
rather than a rewrite. Protocol lowering — SiteSpec resolution, layouts,
write math, refusal policy — stays in ``neural/shared/`` and never comes here.

Why substring patterns, not exact names
---------------------------------------

``.source`` names ops after the variable or symbol plus a positional suffix
(``attn_weights_1``), and the suffix moves when a transformers release adds or
removes a line — that is how transformers 5 broke nnterp's GPT-2 dropout
address. A pattern matches by substring and *refuses* on zero or multiple
hits, so a drifted forward fails loudly with the real inventory (the CI
canary's failure mode) instead of silently reading a neighbouring tensor.
The one systematic ambiguity — a variable that is first assigned and then
called, so both ops carry its name — is resolved structurally: the *call* op
is the hit whose own source line invokes the matched symbol.

📐 The measured facts the table encodes (2026-09-16, transformers 5.16.1,
nnsight 0.8.1.dev125, verified on ``tiny-random/qwen3.5-moe``, the tiny Llama
and the tiny GPT-2; the qwen inventory matches the real Qwen3.6-35B-A3B's):
the attention function's call is ``attention_interface_1`` on all three trees
(``_0`` binds the implementation; GPT-2 has no ``apply_rotary_pos_emb`` op,
which is why q/k are read off the call's arguments rather than off RoPE);
inside the eager attention function ``attn_weights_1`` is the post-mask
softmax input and ``attn_weights_2`` the softmax output
(``softmax(attn_weights_1) == attn_weights_2`` exactly); the call's
``output[0]`` is z, already transposed back to ``(b, s, H, d)``; the chunked
delta kernel needs an ``implementation_0`` peel (the hub-fallback wrapper);
and inside the grouped experts kernel the assignment ``inv_perm[perm] =
arange`` (``inv_perm_1``) reports the *right-hand side* as its output, so the
permutation is derived from the sort itself.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Callable, Iterable, Literal, Mapping

__all__ = [
    "ADDRESSES",
    "GENERATED_ADDRESSES",
    "AddressResolutionError",
    "SourceAddress",
    "components_addressed",
    "match_op",
]


class AddressResolutionError(ValueError):
    """A pattern did not resolve to exactly one op.

    A plain ``ValueError`` on purpose: this module knows nothing of the
    protocol's error vocabulary. The executor wraps it with the component,
    layer and library version before it reaches a document author.
    """


@dataclasses.dataclass(frozen=True)
class SourceAddress:
    """One interior tensor, addressed through ``.source`` from its anchor —
    the envoy the shared site map resolves the component to (the mixer, or
    the experts module)."""

    #: Op patterns from ``anchor.source`` inward, one ``.source`` drill
    #: between consecutive elements: ``("attention_interface", "attn_weights_1")``
    #: is ``anchor.source.<attention_interface call>.source.<attn_weights_1>``.
    #: Each element is a substring matched against that level's ``names`` —
    #: NEVER a hardcoded ``_n`` suffix for a symbol that appears once (the
    #: suffix is what drifts); a suffix is spelled only where two live ops
    #: share the symbol (``attn_weights_1`` / ``_2``, ``proj_out_0`` / ``_3``,
    #: ``last_recurrent_state_1``, ``range_1``).
    path: tuple[str, ...]
    #: Which handle of the last op carries the value: its return, or its
    #: ``(args, kwargs)`` — how a kernel's arguments are reached.
    handle: Literal["output", "inputs"] = "output"
    #: Indices into the handle's value, applied in order: ``(0,)`` is element
    #: 0 of a tuple return, ``(0, 1)`` is ``inputs[0][1]`` (positional
    #: argument 1), ``(1, "g")`` is keyword argument ``g``.
    select: tuple[int | str, ...] = ()
    #: How often the op fires per forward. ``"per_chunk"`` needs
    #: ``tracer.iter`` loop machinery and a ``trip``; ``"per_step"`` is a
    #: decode-only op firing once per generated token, whose value is a
    #: state with no position axis of its own.
    fires: Literal["once", "per_chunk", "per_step"] = "once"
    #: For a per-fire address: the path (same form as ``path``) of the loop's
    #: own ``range(...)`` — the length of its output is the fire count. 📐
    #: Read off the loop itself, never off config (the kernel pads to a chunk
    #: multiple, so the count is the kernel's fact, not the sequence
    #: length's).
    trip: tuple[str, ...] | None = None
    #: The path (same form as ``path``) of the ``torch.sort`` whose
    #: ``output[1]`` maps sorted rows → token-major rows. When set, the
    #: value's rows are in the kernel's expert-sorted order and the executor
    #: un-sorts reads / re-sorts writes through it — the sorted layout is
    #: grouped_mm bookkeeping, never the component's meaning. The sort is
    #: unstable, so the permutation is read off the very sort the kernel
    #: ran, requested first.
    align: tuple[str, ...] | None = None
    #: The value's rows are expert rows — ``(batch·position·top_k, …)`` — and
    #: the executor re-packs them token-major to the declared 2-D native
    #: shape ``(batch·position, top_k·…)``. Pure row bookkeeping; the
    #: declared ``FeatureShape`` stays the semantic description.
    expert_rows: bool = False
    #: A value computed from the handle rather than read off it:
    #: ``"argsort_perm"`` — the inverse of the sort permutation the handle
    #: selects, i.e. each token-major row's index in sorted order.
    derive: str | None = None
    #: Implementation switches the address is only valid under —
    #: ``{"attn_eager"}``: the fused attention kernels never materialize the
    #: tensor; ``{"experts_grouped"}``: the grouped experts kernel is where
    #: the per-expert interior's ops live.
    requires: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        # The executor memoizes each drilled prefix once per trace and asks
        # for the trip and the sort before the value, so both must live in
        # the value's own subtree: the trip beside it (the loop's range is in
        # the loop's source), the sort at its level or an enclosing one.
        assert self.trip is None or self.trip[:-1] == self.path[:-1], self
        assert (
            self.align is None or self.path[: len(self.align) - 1] == self.align[:-1]
        ), self


_OP_SUFFIX = re.compile(r"_\d+$")


def match_op(
    pattern: str,
    names: Iterable[str],
    line_of: Callable[[str], str] | None = None,
) -> str:
    """The one op ``pattern`` names, or a refusal carrying the inventory.

    Substring match over ``names``. When several ops match — the systematic
    case is a variable assigned and then called, both ops named after it —
    the hits whose own source line *calls* the matched symbol (the op's name
    minus its positional suffix, immediately followed by ``(``) are preferred,
    which ``line_of`` makes possible; anything still ambiguous refuses rather
    than guessing.
    """
    all_names = list(names)
    hits = [n for n in all_names if pattern in n]
    if len(hits) > 1 and line_of is not None:
        calls = [n for n in hits if f"{_OP_SUFFIX.sub('', n)}(" in line_of(n)]
        if calls:
            hits = calls
    if len(hits) == 1:
        return hits[0]
    what = "no op matches" if not hits else f"{len(hits)} ops match ({hits})"
    raise AddressResolutionError(
        f"pattern {pattern!r}: {what}. The installed library's forward names "
        f"these ops: {all_names}. A missing or ambiguous pattern usually means "
        "a transformers release moved this forward's code — re-verify the "
        "address table against the new source."
    )


# --------------------------------------------------------------------------- #
# the table, keyed by (family tree, component)
# --------------------------------------------------------------------------- #

#: The full-attention mixer's interior, anchored on ``self_attn``. All five
#: are the ``attention_interface(...)`` call or live inside it: q and k are
#: the call's arguments 1 and 2 — the post-RoPE query and the key *before*
#: ``repeat_kv``, exactly the tensors the reference engine's attention
#: interface hands its slots, and family-agnostic (GPT-2 has no RoPE op) —
#: z is the call's own return (``output[0]``, already ``(b, s, H, d)``; the
#: drilled ``attn_output_0`` is the pre-transpose tensor, a different box),
#: and the two pattern-shaped tensors are assignments inside the eager
#: function. Only the softmax's neighbourhood needs eager: q, k and z exist
#: under every implementation.
_ATTENTION: dict[str, SourceAddress] = {
    "attention_query": SourceAddress(("attention_interface",), "inputs", (0, 1)),
    "attention_key": SourceAddress(("attention_interface",), "inputs", (0, 2)),
    "attention_scores": SourceAddress(
        # ⚠️ `_1`, the post-mask softmax input — not `_0` (pre-mask). The
        # component is *defined* as the softmax's input (softmax(scores) ==
        # pattern, pinned exact); the pre-mask tensor is a different box.
        ("attention_interface", "attn_weights_1"),
        requires=frozenset({"attn_eager"}),
    ),
    "attention_probs": SourceAddress(
        # the softmax's output, read AND written here: a write is consumed by
        # the value multiply downstream, where a write to the mixer's
        # returned attn_weights would reach nothing.
        ("attention_interface", "attn_weights_2"),
        requires=frozenset({"attn_eager"}),
    ),
    "attention_z": SourceAddress(("attention_interface",), "output", (0,)),
}

#: The Gated DeltaNet interior, anchored on ``linear_attn`` — 30 of
#: Qwen3.6-35B-A3B's 40 layers: none of these tensors crosses a module
#: boundary.
#:
#: 📐 The mixer projects ``mixed_qkv`` and the gate ``z`` first, runs the
#: causal conv (channels-first), splits into q/k/v (pre ``repeat_interleave``,
#: so ``deltanet_query``/``key`` are in *key-head* space), computes
#: ``beta = σ(b)`` and the decay ``g``, tiles q/k to the value-head count and
#: hands everything to the chunked delta kernel — whose arguments 0/1 are the
#: tiled q/k (``delta_query``/``delta_key``: the tensors the reference engine's
#: kernel swap sees) and whose own ``.source`` needs the ``implementation_0``
#: peel (the hub-kernel-with-fallback wrapper). In prefill the kernel advances
#: the recurrent state once per 64-token chunk (``last_recurrent_state_1``;
#: ``_0`` is the zero init, ``_2`` the ``None`` rebind), so the state fires
#: ``per_chunk`` with the trip count read off the loop's own ``range_1``
#: (``range_0`` is the intra-chunk loop). The *recurrent* kernel — per-token
#: states — runs only at ``seq_len == 1`` under a cache: decode-only by the
#: modeling code's own dispatch (modeling_qwen3_5_moe.py:507), so the
#: per-token faces (``delta_kv_mem``, ``delta_state_update``, ``delta_state``)
#: have no prefill address and are refused by name.
_KERNEL = "chunk_gated_delta_rule"
_DELTANET: dict[str, SourceAddress] = {
    # ⚠️ channels-first (b, width, s) — the declared shape carries it
    "delta_conv": SourceAddress(("causal_conv1d_fn",)),
    "deltanet_query": SourceAddress(("query_reshape",)),
    "deltanet_key": SourceAddress(("key_reshape",)),
    "delta_value": SourceAddress(("value_reshape",)),
    "delta_beta": SourceAddress(("b_sigmoid",)),
    # the kernel's arguments — one `inputs` request serves all three. An
    # op's inputs must be requested before anything drills into its source
    # (measured: OutOfOrderError otherwise), which the rank table's order
    # guarantees.
    "delta_query": SourceAddress((_KERNEL,), "inputs", (0, 0)),
    "delta_key": SourceAddress((_KERNEL,), "inputs", (0, 1)),
    "delta_decay": SourceAddress((_KERNEL,), "inputs", (1, "g")),
    "deltanet_state": SourceAddress(
        (_KERNEL, "implementation_0", "last_recurrent_state_1"),
        fires="per_chunk",
        trip=(_KERNEL, "implementation_0", "range_1"),
    ),
    "delta_kernel_output": SourceAddress((_KERNEL,), "output", (0,)),
}

#: The per-expert MoE interior, anchored on ``mlp.experts``. All six live
#: inside the grouped experts kernel (``experts_forward`` is the dispatch's
#: call — the same assigned-then-called ambiguity as ``attention_interface``,
#: resolved the same way). 📐 The kernel sorts the ``(token, slot)`` rows by
#: expert (``torch_sort``), runs the fused gate_up projection (``proj_out_0``;
#: ``_apply_gate`` splits and gates it), down-projects (``proj_out_3``),
#: un-sorts and weights. The sorted layout is bookkeeping, so every
#: sorted-space value carries ``align`` and is presented token-major. The
#: top-level ``self_act_fn_0`` beside ``self__apply_gate_0`` is the dead
#: ``has_gate=False`` branch — the activation is the call *inside*
#: ``_apply_gate``.
_EXPERTS = "experts_forward"
_SORT = (_EXPERTS, "torch_sort")
_GROUPED = frozenset({"experts_grouped"})
_MOE: dict[str, SourceAddress] = {
    # The two halves of the fused [gate_e | up_e] projection are ONE capture —
    # the first _grouped_linear's return (`proj_out_0`, pre-chunk) — with two
    # addresses through the declared fused axis. The `_0` suffix is
    # load-bearing: `proj_out` is reassigned down the forward (`proj_out_3`
    # is the down-projection).
    "expert_gate_proj": SourceAddress(
        (_EXPERTS, "proj_out_0"),
        align=_SORT,
        expert_rows=True,
        requires=_GROUPED,
    ),
    "expert_up_proj": SourceAddress(
        (_EXPERTS, "proj_out_0"),
        align=_SORT,
        expert_rows=True,
        requires=_GROUPED,
    ),
    # the act call INSIDE _apply_gate: act_fn(gate) alone, before the `· up`
    # multiply — the same tensor `mlp_activation` names on the llama family
    "expert_activation": SourceAddress(
        (_EXPERTS, "self__apply_gate", "self_act_fn"),
        align=_SORT,
        expert_rows=True,
        requires=_GROUPED,
    ),
    "expert_neuron_output": SourceAddress(
        (_EXPERTS, "self__apply_gate"),
        align=_SORT,
        expert_rows=True,
        requires=_GROUPED,
    ),
    # the down-projection's return, BEFORE the routing weight — the registry
    # identity: routed_output == the slot-sum of expert_output · router_scores.
    # Still expert-sorted at this line (the weighting and un-sort happen
    # downstream), hence the align.
    "expert_output": SourceAddress(
        (_EXPERTS, "proj_out_3"),
        align=_SORT,
        expert_rows=True,
        requires=_GROUPED,
    ),
    # each token-major (token, slot) row's index in the kernel's sorted
    # order: the inverse of the sort's own permutation. Derived rather than
    # read because the kernel's `inv_perm_1` assignment reports its
    # right-hand side (the `arange`) as its output.
    "expert_permutation": SourceAddress(
        _SORT,
        "output",
        (1,),
        derive="argsort_perm",
        expert_rows=True,
        requires=_GROUPED,
    ),
}

#: Every address, keyed ``(family tree, component)`` — the executor's single
#: lookup point. The trees are the registry families' names; the GPT-2 tree
#: shares the attention rows (its ``attn`` forward makes the same interface
#: call), and its dense MLP has none of the other interiors.
ADDRESSES: Mapping[tuple[str, str], SourceAddress] = {
    **{("llama_tree", c): a for c, a in {**_ATTENTION, **_DELTANET, **_MOE}.items()},
    **{("gpt2_tree", c): a for c, a in _ATTENTION.items()},
}


#: What decode dispatches differently, keyed like :data:`ADDRESSES`. At
#: ``seq_len == 1`` under a cache the DeltaNet mixer calls the *recurrent*
#: kernel (``torch_recurrent_gated_delta_rule``), so the chunked kernel's
#: addresses are dead there and the state is one tensor per step:
#: ``last_recurrent_state_2``, the state the kernel returns after the token
#: (``_0`` is the incoming state, ``_1`` the in-loop rebind). 📐 Verified on
#: ``tiny-random/qwen3.5-moe`` under ``tracer.iter``: the op has no prefill
#: occurrence, so a step body must be pinned to its forward by a location
#: that fires every forward before this op is requested. The attention
#: function is the same call in decode, so the ``interface`` rows of
#: :data:`ADDRESSES` serve there unchanged and need no entry here.
GENERATED_ADDRESSES: Mapping[tuple[str, str], SourceAddress] = {
    ("llama_tree", "deltanet_state"): SourceAddress(
        ("recurrent_gated_delta_rule", "implementation_0", "last_recurrent_state_2"),
        fires="per_step",
    ),
}


def components_addressed() -> frozenset[str]:
    """Every component some tree addresses."""
    return frozenset(component for _, component in ADDRESSES)
