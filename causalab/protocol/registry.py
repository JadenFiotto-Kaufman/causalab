"""Static model metadata — the widths canonicalization derives from.

Deriving featurizer widths and param shapes (spec §6) needs the model's
*static configuration* (hidden size, depth, head counts), never its weights.
This registry keeps that metadata deterministic and offline: entries for the
models the repo actually uses are declared here as data, tests register
their tiny-random models, and an HF config can be adapted explicitly with
:func:`model_info_from_hf_config` when a caller opts in — the protocol layer
itself never touches the network.

Shapes per component (the ``(model, site) → d`` rule of §2.5, and more).
:func:`component_shape` answers with a
:class:`~causalab.protocol.shapes.FeatureShape` rather than an integer, because
four questions turn on the same fact and used to be answered in four places:
how wide the feature axis is, whether there is one at all, how many heads
``head`` may name, and how the module's native tensor relates to the executor's
``(batch, position, feature)`` contract.

===================================  =======================================
component                            shape
===================================  =======================================
``embeddings``, ``block_input``,     ``(batch, position, hidden)`` — the
``block_output``, ``attention_output``,   residual stream. The three norm taps
``mlp_input``, ``mlp_output``,       are here because an RMSNorm maps the
``ln_final``, ``attention_input_norm``,   residual stream to itself, so both
``block_mid``, ``mlp_input_norm``    its sides are hidden-wide
``mlp_activation``                   ``(batch, position, intermediate)`` (the
                                     family caveat of *which* tensor this
                                     names lives in the backend, not here)
``mlp_neuron_output``                 ``(batch, position, intermediate)``; the
                                     complete down-projection input
``attention_premix``                  ``(batch, position, heads·head_dim)``,
                                     head-major and already flattened — the
                                     o-projection's input, query-head space
``lm_head``                          ``(batch, position, vocab)``
``routed_output``,                   ``(batch·position, hidden)`` — hidden-wide,
``shared_expert_output``             but flattened like the rest of the MoE
                                     interior
``router_logits``                    ``(batch·position, num_experts)``
``router_scores``                    ``(batch·position, top_k)``, **ranking**:
                                     column *k* is the *k*-th ranked expert, a
                                     different expert for different tokens, so
                                     a basis fitted across positions is fitted
                                     across a shuffled basis. Basis-fitting
                                     featurizers are refused; per-column ones
                                     are not
``expert_idx``                       ``(batch·position, top_k)``, **integral**:
                                     a routing table of integer expert ids —
                                     no featurizer, no gradient
``expert_permutation``               ``(batch·position, top_k)``, **integral**:
                                     the serving kernel's row bookkeeping —
                                     for each (token, slot), the row index in
                                     expert-sorted order
``expert_gate_proj``,                ``(batch·position, top_k·moe_inner)`` —
``expert_up_proj``,                  one vector per routed expert slot,
``expert_activation``,               token-major and **ranking**: slot *k* is
``expert_neuron_output``              the *k*-th ranked expert (the two proj
                                     halves share one fused capture)
``expert_output``                    ``(batch·position, top_k·hidden)``,
                                     token-major, ranking; the value is
                                     **pre-routing-weight**: summing
                                     ``expert_output · router_scores`` over
                                     the top-k axis gives ``routed_output``
``shared_expert_gate_proj``,         ``(batch·position, shared_inner)``
``shared_expert_up_proj``,
``shared_expert_activation``
``shared_expert_gate``               ``(batch·position, 1)`` — one mixing
                                     scalar per token
``input_ids``                        ``(batch, position)``, **integral**: no
                                     feature axis at all, so not a feature
                                     space in any sense
``attention_probs``                  ``(batch, head, position[query],
                                     key_position[key])`` — **two position
                                     axes**, so no contract form. Every
                                     refusal the executor makes about it is
                                     derived from that
``deltanet_query``, ``deltanet_key``,  the three DeltaNet faces only the nnsight
``deltanet_state``                   engine serves, from the same
                                     four ``linear_*`` dimensions — see
                                     :func:`_deltanet_shape`. q/k are
                                     pre-GVA-tiling (key-head space) where
                                     ``delta_query``/``delta_key`` are post;
                                     ``deltanet_state`` is ``(batch,
                                     position[chunk], head, k_dim·v_dim)``,
                                     its position axis the kernel's 64-token
                                     chunk index where ``delta_state`` is per
                                     step (:data:`BACKEND_PAIRS`). The other
                                     eight ``deltanet_*`` spellings are aliases
                                     of ``delta_*``
===================================  =======================================
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, Literal, Mapping, get_args

from causalab.protocol import shapes
from causalab.protocol.errors import ProtocolError, ReasonCode, ValidationError
from causalab.protocol.schema import (
    COMPONENTS,
    DEPRECATED_COMPONENTS,
    GATE_GROUP_AXES,
    GATE_GROUPS,
    DEPRECATED_IN,
    LAYERLESS_COMPONENTS,
    MECHANISMS,
    STREAMS,
    Stream,
)
from causalab.protocol.shapes import FeatureShape

__all__ = [
    "BACKEND_PAIRS",
    "BLOCK_TAPS",
    "CAPABILITIES",
    "COMPONENT_STREAMS",
    "DOCS_TABLE_MODEL",
    "ENGINES",
    "FAMILIES",
    "GPT2_TREE",
    "HOOK_KINDS",
    "INTERIOR_ROWS",
    "LLAMA_TREE",
    "OVERRIDE_KEYS",
    "PACKINGS",
    "PREDICATES",
    "RELATIONS",
    "TAP_KINDS",
    "TAP_SCOPES",
    "BackendPair",
    "Capability",
    "FamilyAdapter",
    "HookKind",
    "Identity",
    "LayerInventory",
    "ModelInfo",
    "OverrideKey",
    "Packing",
    "Predicate",
    "Relation",
    "Tap",
    "TapKind",
    "TapScope",
    "TreeAddress",
    "alias_would_rebind",
    "backend_pair",
    "capability",
    "component_shape",
    "component_width",
    "components_served_by",
    "engine_component_summary",
    "expert_axis_refusal",
    "expert_neuron_group_map",
    "families_in_table",
    "family",
    "family_for",
    "family_in_table",
    "gate_group_map",
    "gate_param_shape",
    "get_model_info",
    "GROUP_SITE_SELECTORS",
    "head_group_map",
    "head_space_refusal",
    "identities_for",
    "identity",
    "inventory",
    "mixer_children",
    "model_info_from_hf_config",
    "native_shape",
    "predicate_holds",
    "register_family",
    "register_model",
    "render_component_tables",
    "render_family_table",
    "site_group_map",
    "unavailable_at_load",
    "write_capabilities",
    "write_policy_refusal",
]


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """The static facts canonicalization needs about one model."""

    key: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    #: The dense MLP's inner width — ``None`` on a tower whose every block is a
    #: sparse-MoE block (Qwen3.6-35B-A3B): there is no dense MLP, so
    #: ``mlp_activation`` names no tensor and :func:`component_shape` refuses
    #: it at load, the same refusal the run makes from the module tree. 🐞 The
    #: adapter used to fall back to ``4 · hidden`` there, so ``validate`` sized
    #: an ``mlp_activation`` featurizer at 8192 for a tensor that does not
    #: exist and the run refused with a ``NotImplementedError``.
    intermediate_size: int | None
    vocab_size: int
    native_dtype: str = "fp32"
    #: The HF ``model_type`` of the config the entry describes, exactly as
    #: :func:`model_info_from_hf_config` reads it off the *text* config
    #: (``llama``, ``gpt2``, ``qwen3_5_moe_text`` — 📐 measured on the tiny
    #: fixtures and on the A3B's own config: the Qwen3.5-MoE *wrapper* config
    #: says ``qwen3_5_moe``, its text config ``qwen3_5_moe_text``, and the
    #: adapter reads the text config, so that is the key). It is the key of
    #: the per-family tap table (:attr:`Capability.overrides`): the address of
    #: each attention-interior component on this family, and what lets
    #: :func:`predicate_holds` decide ``split_qkv`` / ``gated_attention``
    #: offline. Not in any canonical form. An entry that leaves it unset is a
    #: family the table has not met: the run serves it by measurement and
    #: ``validate`` decides nothing about its mixer interior.
    family: str | None = None
    num_experts: int | None = None
    #: top-k: how many of ``num_experts`` each token is routed to. The width of
    #: ``router_scores``, whose axis is that top-k list.
    num_experts_per_tok: int | None = None
    #: The shared expert's inner width. Deliberately separate from
    #: ``intermediate_size``: a MoE checkpoint can carry three different inner
    #: widths (dense ``intermediate_size``, ``moe_intermediate_size`` per routed
    #: expert, and this one), and reading the wrong one is silent.
    shared_expert_intermediate_size: int | None = None
    #: The *routed* experts' inner width (``moe_intermediate_size``) — the third
    #: of those three inner widths, and the feature width of the per-expert
    #: interior (``expert_gate_proj`` and friends). ⚠️ On
    #: ``tiny-random/qwen3.5-moe`` all three widths are 32, so the fixture
    #: cannot tell a wrong choice from a right one — which is exactly why this
    #: is its own field rather than a fallback through one of the others.
    moe_intermediate_size: int | None = None
    #: The Gated DeltaNet mixer's dimensions. Its q/k live in
    #: *key-head* space and its v/gate/state in *value-head* space — two
    #: different head counts, the linear-attention analogue of GQA, and the
    #: same silent-empty-slice hazard if one bound is used for the other.
    #: ⚠️ Four independent numbers, deliberately not derived from each other:
    #: the fixture has 2× GVA tiling (``num_value_heads == 2 · num_key_heads``)
    #: and equal head dims, and a table that assumed either coupling would be
    #: silently wrong on a family that breaks it.
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    #: The mixer stream at each layer, ``num_layers`` long, on a family whose
    #: config declares it (HF ``layer_types``). A hybrid tower alternates
    #: streams per layer, and a component that exists on only one of them
    #: (its capability row's ``stream``; ``COMPONENT_STREAMS`` is the view) is
    #: refused at load against this table —
    #: which is what lets the pure verbs refuse ``attention_premix`` at a Gated
    #: DeltaNet layer offline, instead of the run doing it against the loaded
    #: modules. ``None`` means the model declares no pattern the protocol can
    #: read; the run-time check still applies, so a dense family loses nothing
    #: by leaving it unset. Values are the protocol's ``STREAMS`` — an HF
    #: config's own spelling is mapped by :func:`model_info_from_hf_config`
    #: (``_HF_LAYER_STREAMS``), never copied in.
    layer_types: tuple[Stream, ...] | None = None
    #: The experts implementation the loaded model dispatches on
    #: (``experts_implementation``: ``"grouped_mm"``, ``"eager"``,
    #: ``"batched_mm"``) — a **load-time knob**, not a config fact, so a
    #: hand-declared entry leaves it ``None`` and only an entry adapted from a
    #: *loaded* model's config carries it. It is what lets ``validate`` decide
    #: the ``grouped_mm`` predicate (the routed interior's dispatch pin) during
    #: model-capability validation instead of at tap time (the rule: resolve
    #: implementation knobs during model-capability validation); the tap-time
    #: probe stays as the last-line check. Not in
    #: any canonical form.
    experts_implementation: str | None = None

    def __post_init__(self) -> None:
        if self.layer_types is None:
            return
        if len(self.layer_types) != self.num_layers:
            raise ValueError(
                f"model {self.key!r}: layer_types has {len(self.layer_types)} "
                f"entries for a {self.num_layers}-layer model"
            )
        unknown = sorted(set(self.layer_types) - set(STREAMS))
        if unknown:
            raise ValueError(
                f"model {self.key!r}: layer_types names {unknown}, not in "
                f"{list(STREAMS)}"
            )


_REGISTRY: dict[str, ModelInfo] = {}


def register_model(info: ModelInfo) -> None:
    """Register (or replace) one model's static metadata."""
    _REGISTRY[info.key] = info


def get_model_info(key: str) -> ModelInfo:
    """Look up a model key; a missing entry is a load error (the alternative
    — fetching a config from the network mid-canonicalization — would make
    digests depend on connectivity)."""
    info = _REGISTRY.get(key)
    if info is None:
        raise ValidationError(
            4,
            f"model {key!r} is not in the protocol model registry — register "
            "its static config (causalab.protocol.registry.register_model, or "
            "model_info_from_hf_config on a loaded HF config)",
            path="model.key",
        )
    return info


def model_info_from_hf_config(key: str, config: Any) -> ModelInfo:
    """Adapt a loaded HF config object (its text config, on multimodal
    wrappers) into a :class:`ModelInfo`. The caller owns where the config
    came from; this function only reads attributes."""
    text = getattr(config, "text_config", None) or config
    num_heads = int(getattr(text, "num_attention_heads"))
    hidden = int(getattr(text, "hidden_size"))
    head_dim = int(getattr(text, "head_dim", None) or hidden // num_heads)
    # transformers 5 renamed this to ``dtype``; ``torch_dtype`` still resolves but
    # warns on every access, and is scheduled for removal.
    dtype = str(
        getattr(text, "dtype", None) or getattr(text, "torch_dtype", None) or "float32"
    )
    return ModelInfo(
        key=key,
        hidden_size=hidden,
        num_layers=int(getattr(text, "num_hidden_layers")),
        num_heads=num_heads,
        num_kv_heads=int(getattr(text, "num_key_value_heads", None) or num_heads),
        head_dim=head_dim,
        intermediate_size=_intermediate_size(text, hidden),
        vocab_size=int(getattr(text, "vocab_size")),
        family=(
            str(getattr(text, "model_type"))
            if getattr(text, "model_type", None)
            else None
        ),
        native_dtype={"bfloat16": "bf16", "float16": "fp16"}.get(
            dtype.removeprefix("torch."), "fp32"
        ),
        # Two spellings in the wild, and neither is universal: mixtral and
        # qwen3_moe carry both, while qwen2_moe and qwen3_5_moe carry only
        # ``num_experts``. Reading ``num_local_experts`` alone silently left
        # num_experts=None on those, which makes component_width refuse
        # router_logits on a model that plainly has a router.
        num_experts=(
            getattr(text, "num_experts", None)
            or getattr(text, "num_local_experts", None)
        ),
        num_experts_per_tok=getattr(text, "num_experts_per_tok", None),
        # ⚠️ Three spellings, and on `tiny-random/qwen3.5-moe` all three are 32,
        # so the fixture CANNOT tell a wrong choice from a right one. Ordered
        # most-specific first and never silently defaulted to the dense
        # `intermediate_size`, because that is the one that would be wrong on a
        # real checkpoint while still producing a plausible number.
        shared_expert_intermediate_size=(
            getattr(text, "shared_expert_intermediate_size", None)
            or getattr(text, "moe_intermediate_size", None)
        ),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", None),
        linear_num_key_heads=getattr(text, "linear_num_key_heads", None),
        linear_num_value_heads=getattr(text, "linear_num_value_heads", None),
        linear_key_head_dim=getattr(text, "linear_key_head_dim", None),
        linear_value_head_dim=getattr(text, "linear_value_head_dim", None),
        layer_types=_layer_types(text),
        # set on a loaded model's config by the modeling code's dispatch
        # (``from_pretrained(experts_implementation=...)``, default grouped_mm);
        # absent on a bare text config, which is the honest ``None``
        experts_implementation=(
            str(getattr(text, "_experts_implementation"))
            if getattr(text, "_experts_implementation", None) is not None
            else None
        ),
    )


#: HF ``layer_types`` spellings and the protocol stream each one is. HF's
#: vocabulary names attention *variants* — a Gemma2/Gemma3 tower alternates
#: ``sliding_attention`` with ``full_attention``, Llama4 has
#: ``chunked_attention``, the sparse-attention families each have their own —
#: while the protocol's ``stream`` names the *mixer*: softmax attention or a
#: linear-attention kernel. A sliding window is still a ``self_attn`` child
#: computing an attention matrix, which is what the run-time probe
#: (``neural/shared/streams.py``) answers for it, so both halves of the stream
#: check agree on ``full_attention``. 🐞 The adapter used to copy the HF strings
#: straight into the ``STREAMS``-validated field, so every engine load of
#: ``google/gemma-2-2b-it`` — a built-in entry and a golden-protocol model —
#: raised on its ``sliding_attention`` layers.
_HF_LAYER_STREAMS: dict[str, Stream] = {
    "full_attention": "full_attention",
    "sliding_attention": "full_attention",
    "linear_attention": "linear_attention",
}


def _layer_types(text: Any) -> tuple[Stream, ...] | None:
    """The per-layer stream pattern an HF text config declares, in the
    protocol's vocabulary — or ``None`` when it declares none the protocol can
    read.

    The pinned transformers' llama and gpt2 configs carry no ``layer_types``;
    qwen3_5_moe and gemma2/gemma3 do. A pattern naming any spelling outside
    ``_HF_LAYER_STREAMS`` (``mamba``, ``chunked_attention``, a sparse-attention
    kind) is left unset as a whole rather than guessed at per layer: an
    unmapped kind is a family this table has not met, and the documented
    fallback for an entry without a pattern — the run-time check against the
    module the layer actually carries — is the honest answer for it. Deferring
    beats a wrong offline refusal, or a wrong offline pass.
    """
    declared = getattr(text, "layer_types", None)
    if declared is None:
        return None
    kinds = [str(kind) for kind in declared]
    if any(kind not in _HF_LAYER_STREAMS for kind in kinds):
        return None
    return tuple(_HF_LAYER_STREAMS[kind] for kind in kinds)


#: Components whose tensor is the residual stream: an RMSNorm maps it to itself,
#: and both MoE branches write into it, so every one of these is hidden-wide.
_HIDDEN_COMPONENTS: frozenset[str] = frozenset(
    {
        "embeddings",
        "block_input",
        "block_output",
        "attention_output",
        "mlp_input",
        "mlp_output",
        "ln_final",
        "attention_input_norm",
        "block_mid",
        "mlp_input_norm",
    }
)

#: Both MoE branches write into the residual stream, so both are hidden-wide —
#: but the block reshapes to ``(-1, hidden)`` before the router, so their
#: tensors are flattened over (batch, position) like the rest of its interior.
#: 🐞 The width table and the layout table used to say these two things in
#: different places and the descriptor now says them together; they had drifted,
#: and nothing checked, because no code compared a declared width against a real
#: tensor.
_FLAT_HIDDEN_COMPONENTS: frozenset[str] = frozenset(
    {"routed_output", "shared_expert_output"}
)

_SHARED_EXPERT_INNER: frozenset[str] = frozenset(
    {
        "shared_expert_gate_proj",
        "shared_expert_up_proj",
        "shared_expert_activation",
    }
)


def _intermediate_size(text: Any, hidden: int) -> int | None:
    """The dense MLP's inner width, resolved the way the modeling code resolves
    it — or ``None`` when the config declares none, which is what an all-MoE
    text config does (``Qwen3_5MoeTextConfig`` has no ``intermediate_size``
    attribute at all; the ``4304`` in the A3B's config.json is
    ``vision_config.intermediate_size``). 🐞 Falling back to ``4 · hidden``
    there gave ``mlp_activation`` a width on a tower with no dense MLP.

    🐞 Reading ``intermediate_size`` unconditionally is wrong on the GPT-2
    family: ``GPT2Config`` spells the field ``n_inner`` and the block computes
    ``config.n_inner if config.n_inner is not None else 4 * hidden_size``
    (transformers ``models/gpt2/modeling_gpt2.py:250``), ignoring any
    ``intermediate_size`` in the config. 📐 ``hf-internal-testing/tiny-random-gpt2``
    carries a stray ``intermediate_size: 37`` next to ``n_inner: null`` and a
    128-wide MLP, so the adapter reported 37 for a tensor that is 128 wide — a
    featurizer on ``mlp_activation`` would have been sized against nothing. It
    went unnoticed because no code compared a declared width to a real tensor
    until :func:`component_shape` did.
    """
    if hasattr(text, "n_inner"):  # the GPT-2 family's spelling, authoritative
        n_inner = getattr(text, "n_inner")
        return int(n_inner) if n_inner is not None else 4 * hidden
    declared = getattr(text, "intermediate_size", None)
    return int(declared) if declared else None


def component_shape(info: ModelInfo, component: str) -> FeatureShape:
    """The axes of one component's tensor (the table in the module docstring).

    This is the single description everything else derives from: the feature
    width, whether a featurizer may attach, whether ``head`` means anything and
    how many heads it selects among, and the native↔contract conversion the
    backend performs. It replaced a set of parallel answers — a width function
    with hand-written refusal texts, a five-string layout vocabulary, and a head
    bound that read ``info.num_heads`` regardless of component — that could and
    did disagree with each other.
    """
    if component in _HIDDEN_COMPONENTS:
        return shapes.bsd(info.hidden_size)
    if component in _FLAT_HIDDEN_COMPONENTS:
        return shapes.flat_td(info.hidden_size)
    if component in {"mlp_activation", "mlp_neuron_output"}:
        if info.intermediate_size is None:
            routed = (
                "expert_neuron_output"
                if component == "mlp_neuron_output"
                else "expert_activation"
            )
            raise ValidationError(
                4,
                f"model {info.key!r} declares no dense MLP inner width — every "
                f"layer is a sparse-MoE block, so {component!r} names no tensor "
                f"here. Its analogues are {routed!r} (inside the routed "
                "experts) and 'shared_expert_activation' (the shared expert's "
                "down-projection input).",
                reason="component_unavailable",
            )
        return shapes.bsd(info.intermediate_size)
    if component == "attention_premix":
        # the per-head o-projection input: query-head space (num_heads *
        # head_dim = the o_proj input width), NOT the GQA KV-head space of
        # v_proj. Head-major and already flattened — `(b, s, H*d)`.
        return shapes.bs_flat_heads(info.num_heads, info.head_dim)
    if component == "attention_query_pre_rope":
        # q_norm's output on a family that has one, q_proj's otherwise — the
        # queries as the mixer computes them, BEFORE RoPE rotates them. Query
        # space, so `head` runs 0..num_heads.
        return shapes.bs_flat_heads(info.num_heads, info.head_dim)
    if component == "attention_key_pre_rope":
        # ⚠️ KV-head space, which is narrower than query space by the GQA ratio.
        # This is the component §2.2's head-bound fix exists for: bounding it by
        # num_heads does not raise, it yields an EMPTY feature slice.
        return shapes.bs_flat_heads(info.num_kv_heads, info.head_dim)
    if component == "attention_value_states":
        # v_proj's output — the actual value vectors, KV-head space, and NOT
        # what `attention_premix` (the o_proj input, query space, post-gate)
        # names. The tap is before `past_key_values.update`, so a write reaches
        # the cache.
        return shapes.bs_flat_heads(info.num_kv_heads, info.head_dim)
    if component == "attention_gate":
        # 📐 Qwen3.5/3.6's q-projection emits `[q_h | gate_h]` per head in one
        # tensor of width H·2·d — measured (1, 5, 512) for H 8, d 32 on
        # `tiny-random/qwen3.5-moe`. This component is split 1 of 2, and the
        # fused descriptor is what keeps a write to it from disturbing q.
        return shapes.bs_fused_heads(info.num_heads, 2, 1, info.head_dim)
    if component == "attention_query":
        # 📐 the attention interface's first argument: (b, H, s, d), post-RoPE.
        return shapes.bhsd(info.num_heads, info.head_dim)
    if component == "attention_key":
        # 📐 its second argument: (b, H_kv, s, d), post-RoPE and BEFORE
        # `repeat_kv` — so KV-head space. The position axis is named because it
        # runs over the positions being attended *to*: under a KV cache that is
        # the whole prefix, growing by one per decode step, which is what makes
        # a continuation read of it meaningless rather than merely awkward.
        return shapes.bhsd(info.num_kv_heads, info.head_dim, position_name="key")
    if component == "attention_scores":
        # the softmax's INPUT — same axes as the pattern, one step earlier, and
        # the difference is everything: nothing downstream assumes scores are
        # normalized, because the model's own softmax has yet to run.
        return shapes.attention_pattern(
            info.num_heads,
            note=(
                "Read it whole at pos: \"all\". Unlike 'attention_probs', "
                "every write mechanism is legal here — the model's own softmax "
                "runs after the edit, so rows still sum to 1 by construction."
            ),
        )
    if component == "attention_z":
        # 📐 the interface's return[0]: (b, s, H, d) — already transposed back,
        # and BEFORE the gate multiply and the o-projection.
        return shapes.bshd(info.num_heads, info.head_dim)
    if component == "attention_result":
        # ⚠️ The shape of the component's **value**, which is not the shape of
        # the tensor its tap captures: the model never computes this one. Each
        # head's contribution to the residual stream is hidden-wide, so the
        # whole thing is `heads · hidden` — `heads` times `attention_output`,
        # which is why naming a `head` is strongly encouraged and why the read
        # is derived after the position gather rather than before it.
        return shapes.bs_flat_heads(info.num_heads, info.hidden_size)
    if component == "lm_head":
        return shapes.bsd(info.vocab_size)
    if component == "attention_probs":
        return shapes.attention_pattern(
            info.num_heads,
            note=(
                "Round 1 exposes the whole pattern, which is what an "
                'interchange on attention needs: read it at pos: "all", '
                "without a featurizer and without 'dims'."
            ),
        )
    if component == "input_ids":
        return shapes.bs(
            integral=True,
            note=(
                "Read it directly, or read 'embeddings' if you want the vector "
                "the ids look up."
            ),
        )
    if component == "router_logits":
        if info.num_experts is None:
            raise ValidationError(
                4, f"model {info.key!r} declares no experts; router_logits has no width"
            )
        return shapes.flat_td(info.num_experts)
    if component in ("router_scores", "expert_idx"):
        if info.num_experts_per_tok is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no num_experts_per_tok; "
                f"{component} has no width",
            )
        if component == "expert_idx":
            return shapes.flat_topk(
                info.num_experts_per_tok,
                integral=True,
                note=(
                    "It is the MoE routing table: integer expert ids on a "
                    "top-k axis. Read or write it directly to inspect or edit "
                    "routing."
                ),
            )
        # ⚠️ Dimensionally well defined, and a plain read of it is meaningful —
        # but column *k* is the *k*-th ranked expert, a different expert for
        # different tokens, so the axis is a ranking rather than a basis. That
        # is what `ranking` says, and what makes `subspace`/`pca`/`sae` refuse.
        return shapes.flat_topk(
            info.num_experts_per_tok,
            ranking=True,
            note=(
                "Its axis is a per-token ranking: column k is the k-th ranked "
                "expert, a different expert for different tokens, so a basis "
                "fitted across positions is fitted across a shuffled basis. "
                "Read it directly, or featurize 'router_logits', whose axis is "
                "the fixed all-experts one."
            ),
        )
    if component in ("expert_gate_proj", "expert_up_proj"):
        # the two halves of the routed up-projection's fused [gate_e | up_e]
        # output — one capture, two addresses (the `attention_gate` precedent),
        # token-major and ranked like the rest of the interior
        if info.num_experts_per_tok is None or info.moe_intermediate_size is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no routed-expert inner width "
                f"(moe_intermediate_size) or top-k; {component} has no width",
            )
        return shapes.flat_topk_fused_features(
            info.num_experts_per_tok,
            2,
            0 if component == "expert_gate_proj" else 1,
            info.moe_intermediate_size,
            ranking=True,
            note=(
                "Its slot axis is a per-token ranking (see 'expert_activation'); "
                "join slots to experts through 'expert_idx'."
            ),
        )
    if component == "expert_permutation":
        if info.num_experts_per_tok is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no num_experts_per_tok; "
                f"{component} has no top-k axis",
            )
        return shapes.flat_topk(
            info.num_experts_per_tok,
            integral=True,
            note=(
                "It is the serving kernel's row bookkeeping: for each "
                "(token, slot) pair, the row index in expert-sorted order. "
                "Read it to align raw kernel-order tensors; the per-expert "
                "components themselves are already presented token-major."
            ),
        )
    if component == "expert_output":
        # the down-projection's output BEFORE the routing weight — hidden-wide
        # per (token, slot), token-major. The registry identity:
        # routed_output == sum over slots of expert_output · router_scores,
        # exact (the model computes precisely this sum, in this order).
        if info.num_experts_per_tok is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no num_experts_per_tok; "
                f"{component} has no width",
            )
        return shapes.flat_topk_features(
            info.num_experts_per_tok,
            info.hidden_size,
            ranking=True,
            note=(
                "Its slot axis is a per-token ranking (see 'expert_activation'); "
                "join slots to experts through 'expert_idx'. The value is "
                "pre-routing-weight: routed_output == the slot-sum of "
                "expert_output · router_scores."
            ),
        )
    if component in {"expert_activation", "expert_neuron_output"}:
        # Each slot holds one expert's neurons. expert_activation captures
        # act(gate). expert_neuron_output captures act(gate) * up.
        if info.num_experts_per_tok is None or info.moe_intermediate_size is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no routed-expert inner width "
                f"(moe_intermediate_size) or top-k; {component} has no width",
            )
        return shapes.flat_topk_features(
            info.num_experts_per_tok,
            info.moe_intermediate_size,
            ranking=True,
            note=(
                "Its slot axis is a per-token ranking: slot k belongs to the "
                "k-th ranked expert, a different expert for different tokens, "
                "so a basis fitted across positions is fitted across a "
                "shuffled basis. Join slots to experts through 'expert_idx', "
                "which has the same (token, slot) rows."
            ),
        )
    if component in _SHARED_EXPERT_INNER:
        if info.shared_expert_intermediate_size is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no shared-expert inner width; "
                f"{component} has no width",
            )
        return shapes.flat_td(info.shared_expert_intermediate_size)
    if component == "shared_expert_gate":
        # one scalar per token: how much of the shared expert to mix in
        return shapes.flat_td(1)
    if component in (
        "delta_qkv",
        "delta_gate",
        "delta_premix",
        "delta_conv",
        "delta_query",
        "delta_key",
        "delta_value",
        "delta_beta",
        "delta_decay",
        "delta_kernel_output",
        "delta_kv_mem",
        "delta_state_update",
        "delta_state",
    ):
        missing = [
            name
            for name in (
                "linear_num_value_heads",
                "linear_num_key_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
            )
            if getattr(info, name) is None
        ]
        if missing:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no linear-attention stream "
                f"(missing {', '.join(missing)}); {component} has no width",
            )
        assert info.linear_num_value_heads is not None  # for the type-checker
        assert info.linear_num_key_heads is not None
        assert info.linear_key_head_dim is not None
        assert info.linear_value_head_dim is not None
        if component == "delta_qkv":
            # 📐 in_proj_qkv's fused [q | k | v] output: widths key_dim,
            # key_dim, value_dim — UNEQUAL (128/128/256 on the fixture), so
            # there is no head packing to declare and no `head:` here;
            # whole-tensor and `dims` only. The kernel-boundary components
            # are the per-head faces of the same information.
            key_dim = info.linear_num_key_heads * info.linear_key_head_dim
            value_dim = info.linear_num_value_heads * info.linear_value_head_dim
            return shapes.bsd(
                2 * key_dim + value_dim,
                note=(
                    "It is the fused [q | k | v] projection, widths "
                    f"{key_dim}/{key_dim}/{value_dim} — unequal, so it has no "
                    "head axis. The per-head faces are the kernel-boundary "
                    "components ('delta_query'/'delta_key'/'delta_value')."
                ),
            )
        if component == "delta_conv":
            # 📐 causal_conv1d_fn's return: (batch, conv_dim, position) —
            # channels-first, the existing bds layout, as #48 predicted. Same
            # fused unequal widths as delta_qkv, so no head axis here either.
            key_dim = info.linear_num_key_heads * info.linear_key_head_dim
            value_dim = info.linear_num_value_heads * info.linear_value_head_dim
            return shapes.bds(
                2 * key_dim + value_dim,
                note=(
                    "It is the convolved fused [q | k | v], channels-first and "
                    "with unequal split widths, so it has no head axis. The "
                    "per-head faces are 'delta_query'/'delta_key'/'delta_value'."
                ),
            )
        if component in ("delta_query", "delta_key"):
            # 📐 kernel args 0/1: (b, s, heads, d_k) — already tiled to the
            # v-head count (GVA repeat_interleave happens BEFORE the kernel)
            # and PRE-l2norm (the kernel normalizes and scales internally).
            return shapes.bshd(
                info.linear_num_value_heads,
                info.linear_key_head_dim,
                note=(
                    "Captured pre-l2norm: the kernel applies l2norm and the "
                    "1/sqrt(d) scale internally, so this is the tensor a write "
                    "can actually steer."
                ),
            )
        if component in ("delta_value", "delta_kernel_output"):
            # kernel arg 2, and return[0] — v-head space, (b, s, heads, d_v).
            # The output is pre-norm, pre-gate core_attn_out.
            return shapes.bshd(info.linear_num_value_heads, info.linear_value_head_dim)
        if component == "delta_state":
            # ⚠️ On a real checkpoint this is the expensive read: a full-seq
            # all-layers delta_state is layers · seq · heads · d_k · d_v floats
            # (30 · seq · 32·128·128 on the A3B). Address positions early — the
            # gather runs on the steps axis before anything is kept.
            return shapes.state_matrix(
                info.linear_num_value_heads,
                info.linear_key_head_dim,
                info.linear_value_head_dim,
                note=(
                    "It is the recurrent state S_t: one d_k × d_v matrix per "
                    "head per step. Read it whole (optionally with 'head:'); "
                    "its per-step faces are 'delta_kv_mem' (what the decayed "
                    "state recalls for k̂_t) and 'delta_state_update' (what is "
                    "written in)."
                ),
            )
        if component in ("delta_kv_mem", "delta_state_update"):
            # per-step d_v vectors per head, stacked over steps — derived from
            # adjacent states and pinned by the reconstruction identity
            # S_t == S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t
            return shapes.bshd(info.linear_num_value_heads, info.linear_value_head_dim)
        if component == "delta_beta":
            return shapes.bsh(
                info.linear_num_value_heads,
                note=(
                    "One scalar gate per head per position — "
                    "sigmoid(in_proj_b), in (0, 1)."
                ),
            )
        if component == "delta_decay":
            return shapes.bsh(
                info.linear_num_value_heads,
                note=(
                    "The log-decay g — negative reals (the state multiplies by "
                    "exp(g) per step), not a probability."
                ),
            )
        # delta_gate (in_proj_z's output) and delta_premix (out_proj's input)
        # are both value-head space: v-heads · v-head-dim, head-major, flat.
        return shapes.bs_flat_heads(
            info.linear_num_value_heads, info.linear_value_head_dim
        )
    if component.startswith("deltanet_"):
        return _deltanet_shape(info, component)
    raise ValidationError(
        4,
        f"component {component!r} has no declared feature shape — the protocol "
        "layer cannot size it, and featurizers cannot attach to it",
    )


def _deltanet_shape(info: ModelInfo, component: str) -> FeatureShape:
    """The Gated DeltaNet interior's shapes.

    All widths derive from the mixer's four ``linear_*`` dimensions: q/k live
    in key-head space, v/gate/state/output in value-head space, and the fused
    qkv projection is ``2·key_dim + value_dim`` wide.
    """
    if (
        info.linear_num_key_heads is None
        or info.linear_num_value_heads is None
        or info.linear_key_head_dim is None
        or info.linear_value_head_dim is None
    ):
        raise ValidationError(
            4,
            f"model {info.key!r} declares no linear-attention dimensions "
            f"(linear_num_key_heads and friends); {component} has no shape",
        )
    h_k, h_v = info.linear_num_key_heads, info.linear_num_value_heads
    d_k, d_v = info.linear_key_head_dim, info.linear_value_head_dim
    key_dim, value_dim = h_k * d_k, h_v * d_v
    if component == "deltanet_qkv":
        # the fused q|k|v projection, pre-conv: [key | key | value]
        return shapes.bsd(2 * key_dim + value_dim)
    if component == "deltanet_qkv_conv":
        # ⚠️ channels-first: the causal conv works in (batch, width, position)
        return shapes.bds(2 * key_dim + value_dim)
    if component == "deltanet_query":
        # pre repeat_interleave — key-head space, like attention_key under GQA
        return shapes.bshd(h_k, d_k)
    if component == "deltanet_key":
        return shapes.bshd(h_k, d_k)
    if component == "deltanet_value":
        return shapes.bshd(h_v, d_v)
    if component == "deltanet_beta":
        # one write-strength scalar per value head per token — σ(b)
        return shapes.bsd(h_v)
    if component == "deltanet_decay":
        # the log-decay g the kernel consumes, one per value head per token
        return shapes.bsd(h_v)
    if component == "deltanet_gate":
        # the output gate z, consumed by the gated norm after the kernel
        return shapes.bshd(h_v, d_v)
    if component == "deltanet_core_out":
        # the kernel's return, pre-gate — the DeltaNet analogue of attention_z
        return shapes.bshd(h_v, d_v)
    if component == "deltanet_gated_out":
        # after the gated norm, flattened — what the out-projection consumes
        return shapes.bs_flat_heads(h_v, d_v)
    if component == "deltanet_state":
        return shapes.chunked_state(
            h_v,
            d_k,
            d_v,
            note=(
                "Its position axis is the kernel's 64-token chunk index: read "
                'it whole (pos: "all") or at an integer chunk index. Per-token '
                "prefill state does not exist — the recurrent kernel runs only "
                "in single-token decode (the modeling code's own dispatch)."
            ),
        )
    raise ValidationError(
        4,
        f"component {component!r} has no declared feature shape — the protocol "
        "layer cannot size it, and featurizers cannot attach to it",
    )


def _no_axis(component: str, axis: str, shape: FeatureShape) -> str:
    """ "component X has no <axis> axis — its shape is (…)".

    Factored because two refusals share it — ``head:`` on a headless component
    (§2.2) and ``group: "head"`` on one (§2.5) — and the whole point of a
    generated refusal is that the two cannot come to describe the same absent
    axis differently.
    """
    return (
        f"component {component!r} has no {axis} axis — its shape is {shape.describe()}"
    )


def _with_note(message: str, shape: FeatureShape) -> str:
    """``message`` plus the shape's "…so do this instead" half.

    A descriptor can generate *why* an axis is absent but not what to do about
    it — ``delta_qkv``'s note names the per-head faces of the same information —
    so the note is appended rather than regenerated.
    """
    return f"{message} {shape.note}" if shape.note else message


def head_space_refusal(component: str, head: int, shape: FeatureShape) -> str:
    """Why ``head`` does not apply to ``component`` — the §2.2 refusal.

    Shared by the canonicalizer (which refuses at load) and
    :func:`component_width` (which refuses if anything reaches it another way),
    so the two cannot drift into disagreeing about what a head means.
    """
    return _with_note(
        f"{_no_axis(component, 'head', shape)} — so head {head} would be "
        "validated and then silently dropped. Name a component that has heads "
        "('attention_premix'), or drop the 'head' field.",
        shape,
    )


def component_width(info: ModelInfo, component: str, *, head: int | None = None) -> int:
    """The feature width at one site.

    A thin reading of :func:`component_shape`: the product of the feature axes,
    or one head's slice of it. Kept as a function because three call sites want
    exactly this number and nothing else about the shape.
    """
    shape = component_shape(info, component)
    if not shape.is_feature_space:
        raise ValidationError(4, shape.refusal(f"component {component!r}"))
    width = shape.width
    assert width is not None  # is_feature_space implies a feature axis
    if head is None:
        return width
    space = shape.head_space
    if space is None:
        raise ValidationError(4, head_space_refusal(component, head, shape))
    return width // space


def head_group_map(
    shape: FeatureShape, width: int, *, component: str
) -> tuple[int, int]:
    """The ``(groups, group_width)`` a head-grouped gate has on ``component``
    at a site ``width`` coordinates wide (§2.5 ``group: head``).

    One group per head, each ``head_dim`` coordinates wide, in the order the
    component's head-major axis lays them out — the same slices
    ``sites._head_slice`` names, so a group *is* the head a ``head`` field
    would select. ``width`` is the site's own width: the whole component, or
    one head's slice of it (then the map is a single group). Derived here from
    the shape alone, so the canonicalizer (offline, from the registry) and the
    executor (from the resolved site) cannot disagree about it.

    Raises:
        ValidationError: rule 23 (group legality). The component has no head
            axis — there is nothing to group by, and a gate that silently fell
            back to one parameter per coordinate would report a coordinate
            count as a head count. Or ``width`` is not a whole number of heads,
            which happens when the gate sits after a stage that changed the
            coordinates (a rotation), where "head" no longer names anything.
    """
    space = shape.head_space
    if space is None:
        raise ValidationError(
            23,
            f"gate group 'head' on component {component!r}: the component has "
            f"no head axis — its shape is {shape.describe()} — so there are no "
            "heads to group by. Name a head-major component "
            "('attention_premix' on a full-attention layer, 'delta_premix' on "
            "a Gated DeltaNet layer), or drop 'group'."
            + (f" {shape.note}" if shape.note else ""),
        )
    assert shape.width is not None  # a head axis implies a feature axis
    group_width = shape.width // space
    if width <= 0 or width % group_width:
        raise ValidationError(
            23,
            f"gate group 'head' on component {component!r}: the gate's input is "
            f"{width} wide, which is not a whole number of {group_width}-wide "
            "heads — a head-grouped gate acts on the component's own "
            "coordinates, so no stage before it may change them",
        )
    return width // group_width, group_width


def expert_neuron_group_map(
    info: ModelInfo | None, shape: FeatureShape, width: int, *, component: str
) -> tuple[int, int]:
    """The ``(num_experts, d_expert)`` an expert-keyed gate has on
    ``component`` at a site ``width`` coordinates wide (§2.5 ``group:
    expert_neuron``).

    One parameter per ``(expert, neuron)`` of the routed interior — the whole
    expert table, ``num_experts × d_expert``, whatever ``top_k`` experts a
    token activates. The site itself is token-major, ``top_k · d_expert`` wide
    with slot *k* the *k*-th ranked expert, so a slot's coordinates find their
    parameters through ``expert_idx`` at run time; the map only says how big
    the table is. ``expert_activation`` and ``expert_neuron_output`` each
    hold one expert's neurons in each slot.

    Raises:
        ValidationError: rule 23 rejects an unsupported component, a missing
            expert table, or an input width changed by an earlier stage.
            The error names the component and the required shape.
    """
    if component not in {"expert_activation", "expert_neuron_output"}:
        raise ValidationError(
            23,
            f"gate group 'expert_neuron' on component {component!r}: an "
            "expert-keyed gate holds one parameter per (expert, neuron) of the "
            "routed interior. 'expert_activation' and 'expert_neuron_output' "
            "contain one expert's neurons per routed slot. Name either, or "
            "drop 'group' (on 'shared_expert_activation' the per-coordinate "
            "gate is already one parameter per shared-expert neuron).",
        )
    if info is None or info.num_experts is None or info.moe_intermediate_size is None:
        raise ValidationError(
            23,
            f"gate group 'expert_neuron' on component {component!r}: the "
            "model declares no expert table (num_experts and "
            "moe_intermediate_size), so there is nothing to key the parameters by",
        )
    if width != shape.width:
        raise ValidationError(
            23,
            f"gate group 'expert_neuron' on component {component!r}: the "
            f"gate's input is {width} wide but the component is {shape.width} — "
            "an expert-keyed gate acts on the component's own routed slots, so "
            "no stage before it may change them",
        )
    return info.num_experts, info.moe_intermediate_size


def site_group_map_whole(
    shape: FeatureShape, width: int, *, component: str
) -> tuple[int, int]:
    """The ``(1, width)`` map of a site-grouped gate (§2.5 ``group: site``):
    one parameter over every coordinate of the site, so the site is one unit —
    the way a node-level circuit benchmark scores an MLP block or the input
    embedding. It is the ``head`` map with a single group, and everything
    downstream (the mask broadcast, the L1 mean, the ``rank`` rows, the
    size-matched control) reads it through the same code. ``width`` is the
    gate's own input width — the whole component, or one head's slice of it
    when the site names a ``head``; either is legitimately one unit.

    Raises:
        ValidationError: rule 23 (group legality). The component is not a
            feature space (an attention pattern, the routing table), so there is
            no coordinate axis to cover.
    """
    if not shape.is_feature_space or width <= 0:
        raise ValidationError(
            23,
            f"gate group 'site' on component {component!r}: the component has no "
            f"feature axis to cover — its shape is {shape.describe()} — so there "
            "is nothing for one parameter to gate. Name a feature-space "
            "component, or drop 'group'." + (f" {shape.note}" if shape.note else ""),
        )
    return 1, width


def gate_group_map(
    group: str,
    shape: FeatureShape,
    width: int,
    *,
    component: str,
    info: ModelInfo | None = None,
) -> tuple[int, int]:
    """The derived group map of a grouped gate, by group kind (§2.5):
    :func:`head_group_map` for ``head``, :func:`expert_neuron_group_map`
    for ``expert_neuron``, which additionally needs the model's expert table
    (``info``), and :func:`site_group_map_whole` for ``site``. One entry point
    so the canonicalizer, the loader and the executor derive the same map from
    the same facts."""
    if group == "head":
        return head_group_map(shape, width, component=component)
    if group == "expert_neuron":
        return expert_neuron_group_map(info, shape, width, component=component)
    if group == "site":
        return site_group_map_whole(shape, width, component=component)
    raise ValueError(f"unknown gate group {group!r}")  # the schema's enum is closed


#: Which site field selects a *single* member of what each group groups over
#: (§5.23): a ``head`` on the site leaves a head-grouped gate one group, an
#: ``expert`` leaves an expert-keyed gate one expert's neurons — a per-coordinate
#: gate wearing a grouped gate's name, refused rather than resolved. Keyed by
#: group so a group added to the vocabulary without a row here fails the census
#: (``test_every_group_has_a_site_selector``) instead of skipping the check.
#: ``None`` says the group has no such selector: a ``site`` gate over one head's
#: slice is still one parameter over one unit, exactly what it claims to be.
GROUP_SITE_SELECTORS: dict[str, str | None] = {
    "head": "head",
    "expert_neuron": "expert",
    "site": None,
}


def site_group_map(
    info: ModelInfo,
    group: str,
    component: str,
    *,
    head: int | None = None,
    expert: int | None = None,
) -> tuple[int, int]:
    """The group map of a grouped gate at one *declared* site — ``component``
    and, if the site names them, ``head`` and ``expert`` — from the registry
    alone (:func:`gate_group_map` over :func:`component_shape` and
    :func:`component_width`), with no model loaded.

    The offline reading of the map: what the canonicalizer stamps into a
    document's ``params``, what the loader expects a fitted bundle to carry,
    and what a test derives to compare with the executor's own reading from
    the resolved site. One function so the three cannot disagree on how a
    site's fields become a map.

    Raises:
        ValidationError: rule 23 (group legality). The site already selects a
            single member of what the group groups over (``head: 3`` under
            ``group: head``, ``expert: 7`` under ``group: expert_neuron``) —
            H groups over one head is one group, a coordinate-wise gate under
            a name that claims otherwise — or the component has no such axis
            (:func:`gate_group_map`).
    """
    field = GROUP_SITE_SELECTORS[group]  # the schema's enum is closed
    selected = None if field is None else {"head": head, "expert": expert}[field]
    if selected is not None:
        raise ValidationError(
            23,
            f"site component {component!r} already selects {field} {selected}, "
            f"so group {group!r} has exactly one group — a per-{field} gate over "
            f"one {field} is a coordinate-wise gate. Drop the site's {field!r}, "
            "or drop the group.",
        )
    return gate_group_map(
        group,
        component_shape(info, component),
        component_width(info, component, head=head),
        component=component,
        info=info,
    )


def gate_param_shape(
    group: str | None, group_map: tuple[int, int] | None, width: int
) -> tuple[int, ...]:
    """The shape of a gate's ``theta`` (§2.5): one entry per coordinate with no
    group, one per head (``[heads]``) under ``head``, exactly one (``[1]``)
    under ``site``, and the whole expert table (``[num_experts, d_expert]``)
    under ``expert_neuron`` — two-dimensional so a saved bundle reads as
    ``theta[expert, neuron]``."""
    if group is None or group_map is None:
        return (width,)
    if group in ("head", "site"):
        return (group_map[0],)
    if group == "expert_neuron":
        return tuple(group_map)
    raise ValueError(f"unknown gate group {group!r}")


# --------------------------------------------------------------------------- #
# the capability registry — one row per component
# --------------------------------------------------------------------------- #
#
# Everything that used to be a *second table* of component truth reads from
# here: which engines serve a component (``Engine.components``, generated),
# which mechanisms a write may use (the executor's write policy, and the
# load-time twin in ``validate``), which mixer stream it exists on
# (``COMPONENT_STREAMS``, now a view of the rows), which architectural facts it
# needs (``requires`` — the module-tree probes in ``neural/shared/sites.py`` are
# the run-time evaluators), which engines serve its ragged ``expert:`` face,
# and the retired spellings that fold onto it. The docs tables
# (``docs/running_experiments.md`` §5, spec §8's component row) are rendered
# from the rows by :func:`render_component_tables`; the census guards in
# ``tests/protocol/test_vocabulary_census.py`` hold the rendering, the engine
# sets and the row count to the rows.
#
# The design: one row per component keyed by name, predicates declared rather
# than families enumerated, and an ``overrides`` slot per family — the
# per-family tap table, filled for
# the three families the attention interior was *measured* on. Nothing here
# enters a document's canonical form (reading (a): zero pins move) — the rows
# only decide what is refused, where a tap lands, and what is rendered.

#: The engines a row may name. ``engine.py`` derives ``ENGINE_CHOICES`` from
#: this (plus ``"auto"``), so a third engine is a name here, a class that
#: declares it, and nothing else.
ENGINES: tuple[str, ...] = ("pytorch_hooks", "nnsight")

_BOTH: frozenset[str] = frozenset(ENGINES)
_HOOKS: frozenset[str] = frozenset({"pytorch_hooks"})
_NNSIGHT: frozenset[str] = frozenset({"nnsight"})
_NEITHER: frozenset[str] = frozenset()

#: Every ``do`` mechanism — the ``writes`` cell of a row that accepts any.
_ANY_MECHANISM: frozenset[str] = frozenset(MECHANISMS)
_SWAP_ONLY: frozenset[str] = frozenset({"swap"})

#: Architectural facts a component needs, evaluated against :class:`ModelInfo`
#: at load where the entry can decide them (``moe``, ``shared_expert`` — the
#: MoE widths) and against the loaded module tree at run
#: (``neural/shared/sites.py``, all five). Order is the order the run-time
#: check evaluates them in, which is the order the refusal texts were pinned
#: in: a fused-qkv family is refused before its missing gate is, a dense MLP
#: before its missing shared expert.
Predicate = Literal[
    "moe", "shared_expert", "grouped_mm", "split_qkv", "gated_attention"
]
PREDICATES: tuple[Predicate, ...] = get_args(Predicate)

#: How the serving engine reaches the tensor — documentation for the rendered
#: table, closed so a typo cannot render. The operational dispatch is
#: ``neural/shared/sites.resolve_site``; this names its kinds for a reader.
TapKind = Literal[
    "module input",
    "module output",
    "attention-function slot",
    "delta-kernel boundary",
    "grouped-experts dispatch",
    "`.source` line (fused forward)",
    "derived from `attention_premix`",
]
TAP_KINDS: tuple[TapKind, ...] = get_args(TapKind)

#: The keys a per-family override may carry — the **address** of one
#: attention-interior component on one family (the per-family tap table).
#: Closed, so a misspelt key cannot
#: silently mean "no override":
#:
#: ``module``
#:     the mixer child whose *output* is tapped (``q_proj``, ``q_norm``,
#:     ``c_attn``, …) — a name, never a module: the protocol layer is
#:     torch-free;
#: ``packing``
#:     how that module's native tensor packs the component's value
#:     (:data:`PACKINGS`);
#: ``splits`` / ``split``
#:     for a fused packing only: how many logical tensors share the module's
#:     output, and which one this component is.
OverrideKey = Literal["module", "packing", "splits", "split"]
OVERRIDE_KEYS: tuple[OverrideKey, ...] = get_args(OverrideKey)

#: How a tapped module's native tensor packs the component's logical value,
#: whose axes :func:`component_shape` describes family-independently. 📐 All
#: four measured (transformers 5.16):
#:
#: ``flat``
#:     the module's whole output *is* the value, ``(b, s, heads·d)`` —
#:     llama's bare ``q_proj`` (16 = 4·4);
#: ``head_axis``
#:     the whole output with the head axis kept, ``(b, s, heads, d)`` —
#:     ``Qwen3_5MoeAttention.q_norm`` emits ``(1, 5, 8, 32)``;
#: ``fused_heads``
#:     ``splits`` logical tensors interleaved *per head*,
#:     ``(b, s, heads·splits·d)`` — qwen3.5-moe's ``q_proj`` packs
#:     ``[q_h | gate_h]`` (512 = 8·2·32);
#: ``fused_blocks``
#:     ``splits`` logical tensors as contiguous *blocks*,
#:     ``(b, s, splits·heads·d)`` — GPT-2's ``c_attn`` emits ``[q | k | v]``
#:     (96 = 3·4·8) and the mixer splits it with ``.split(split_size, dim=2)``.
Packing = Literal["flat", "head_axis", "fused_heads", "fused_blocks"]
PACKINGS: tuple[Packing, ...] = get_args(Packing)
_FUSED_PACKINGS: frozenset[str] = frozenset({"fused_heads", "fused_blocks"})


def _check_override(component: str, family: str, address: Mapping[str, Any]) -> None:
    """One override is a well-formed address, or the row cannot be built."""
    where = f"{component}: override for family {family!r}"
    unknown = sorted(set(address) - set(OVERRIDE_KEYS))
    if unknown:
        raise ValueError(f"{where} has keys {unknown} outside {list(OVERRIDE_KEYS)}")
    module = address.get("module")
    if not isinstance(module, str) or not module.isidentifier():
        raise ValueError(f"{where} needs a module name, got {module!r}")
    packing = address.get("packing")
    if packing not in PACKINGS:
        raise ValueError(f"{where} names packing {packing!r}, not in {list(PACKINGS)}")
    splits, split = address.get("splits"), address.get("split")
    if packing in _FUSED_PACKINGS:
        if not (isinstance(splits, int) and splits >= 2):
            raise ValueError(
                f"{where}: a fused packing needs splits >= 2, got {splits!r}"
            )
        if not (isinstance(split, int) and 0 <= split < splits):
            raise ValueError(f"{where}: split {split!r} is not in range({splits})")
    elif splits is not None or split is not None:
        raise ValueError(f"{where}: packing {packing!r} takes no splits/split")


@dataclasses.dataclass(frozen=True)
class Capability:
    """One component's row: what exists, who serves it, what a write may do.

    ``reads`` is the set of engine names whose site resolver serves the
    component — and therefore its write surface too: ``Engine.components`` and
    ``Engine.writable_components`` are both generated from it. Write *policy*
    (``writes``) is deliberately not an engine fact: declaring ``router_logits``
    unwritable on one engine would turn "a write here reaches nothing — write
    ``router_scores``" into "try another engine", the wrong answer everywhere.

    ``writes`` is the closed set of mechanisms a write may use; ``None`` means
    read-only. ``why`` is the refusal text for either, verbatim from the tables
    it replaced, and ``reason`` the code that refusal carries
    (:data:`~causalab.protocol.errors.REASON_CODES`). ``write_capability`` is
    the coarse §8 verb a write charges beyond the generated
    ``component:<name>:write`` — today only the pattern's, whose write goes
    through the attention function rather than a hook.

    ``overrides`` is the per-family tap table: for each family
    (:attr:`ModelInfo.family`) the row has been measured on, the **address**
    of the component on that family — the mixer child to tap and how its
    native tensor packs the value (:data:`OVERRIDE_KEYS`, :data:`PACKINGS`).
    Only the attention interior carries any: it is the one place the families
    disagree about *where* a component is (GPT-2 fuses q, k and v into one
    ``c_attn``; qwen3.5-moe normalizes q and k before RoPE and packs a gate
    beside q). The site resolver reads the address; a family absent from a row
    is served by measurement where that is unambiguous (a bare projection, a
    norm) and refused where it is not (a fused projection whose block order
    only a row can state). The same rows let :func:`predicate_holds` decide
    ``split_qkv`` and ``gated_attention`` offline for a family that has them.
    """

    component: str
    stream: Stream | None
    reads: frozenset[str]
    writes: frozenset[str] | None
    expert_selection: frozenset[str]
    requires: frozenset[Predicate]
    reason: ReasonCode | None
    why: str
    write_capability: str | None
    tap: TapKind
    aliases: tuple[str, ...]
    deprecated_in: str | None
    overrides: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.reads <= _BOTH:
            raise ValueError(f"{self.component}: unknown engine in {self.reads}")
        if not self.expert_selection <= self.reads:
            raise ValueError(
                f"{self.component}: expert_selection names an engine that does "
                "not serve the component"
            )
        if self.writes is not None and not self.writes <= _ANY_MECHANISM:
            raise ValueError(f"{self.component}: unknown mechanism in {self.writes}")
        restricted = self.writes is None or self.writes != _ANY_MECHANISM
        if restricted != bool(self.why):
            raise ValueError(
                f"{self.component}: a restricted write policy and its 'why' text "
                "come together"
            )
        if (self.reason == "unsupported_mechanism") != restricted:
            raise ValueError(
                f"{self.component}: reason {self.reason!r} does not match the "
                "write policy"
            )
        for family, address in self.overrides.items():
            _check_override(self.component, family, address)
        # within one row, a module name means one packing: the measured
        # fallback for a family without an address picks among the declared
        # modules the mixer has, which is only well defined if they agree
        packings: dict[str, Any] = {}
        for address in self.overrides.values():
            seen = packings.setdefault(address["module"], address)
            if seen != address:
                raise ValueError(
                    f"{self.component}: module {address['module']!r} is declared "
                    "with two different packings across families"
                )

    @property
    def read_only(self) -> bool:
        return self.writes is None

    def address_on(self, info: ModelInfo) -> Mapping[str, Any] | None:
        """The row's address of the component on ``info``'s family — or
        ``None`` for a family the table has not met (or an entry with none)."""
        if info.family is None:
            return None
        return self.overrides.get(info.family)


def _row(
    component: str,
    *,
    tap: TapKind,
    stream: Stream | None = None,
    reads: frozenset[str] = _BOTH,
    writes: frozenset[str] | None = _ANY_MECHANISM,
    expert_selection: frozenset[str] = _NEITHER,
    requires: tuple[Predicate, ...] = (),
    why: str = "",
    write_capability: str | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Capability:
    """One row. ``reason`` is derived from the policy (one code per refusal
    kind, never authored twice) and the aliases come from the vocabulary's own
    retired-spelling map, so a name lives in exactly one place."""
    restricted = writes is None or writes != _ANY_MECHANISM
    aliases = tuple(
        sorted(old for old, new in DEPRECATED_COMPONENTS.items() if new == component)
    )
    # every alias records the protocol version it was retired under
    # (schema.DEPRECATED_IN); a row's aliases share one, which the census holds
    versions = sorted({DEPRECATED_IN[alias] for alias in aliases})
    if len(versions) > 1:  # pragma: no cover - the census names it
        raise AssertionError(
            f"{component}: its aliases {aliases} were retired under different "
            f"protocol versions {versions}; one row, one deprecation version"
        )
    return Capability(
        overrides=MappingProxyType(
            {
                family: MappingProxyType(dict(address))
                for family, address in (overrides or {}).items()
            }
        ),
        component=component,
        stream=stream,
        reads=reads,
        writes=writes,
        expert_selection=expert_selection,
        requires=frozenset(requires),
        reason="unsupported_mechanism" if restricted else None,
        why=why,
        write_capability=write_capability,
        tap=tap,
        aliases=aliases,
        deprecated_in=versions[0] if versions else None,
    )


# The write-policy texts, verbatim from the three tables they replace
# (``sites.READ_ONLY_COMPONENTS``, ``sites.SWAP_ONLY_COMPONENTS``,
# ``sites.NORMALIZED_TAPS``). Each names the alternative rather than just
# saying no; the executor and the validator wrap them in one template.
#
# The two read-only entries that are easiest to get wrong are refused for
# *different* reasons: ``router_logits`` is a **silent no-op** (📐 the MoE block
# destructures the router as ``_, routing_weights, selected_experts`` and
# never reads element 0 again — patching it moves the logits by exactly 0.0
# while a patch to ``router_scores`` or ``expert_idx`` moves them), and
# ``input_ids`` is the opposite: a write there *does* land (the embedding's
# pre-hook input, mutated in place), and is refused because token ids are not
# an activation — editing them is a change to the dataset.
_WHY_INPUT_IDS = (
    "the model's token input is not an activation; change the row's text "
    "instead, or write 'embeddings' to edit the vector the ids look up"
)
_WHY_ATTENTION_RESULT = (
    "it is derived, not computed: the model never forms the per-head "
    "contribution at all — it forms their sum, by projecting the whole "
    "'attention_premix' at once — so there is no tensor here for a write "
    "to change. Write 'attention_premix' instead, with the same 'head'; "
    "'attention_result' is a linear function of it, so a write there moves "
    "this by exactly the projection of what you wrote"
)
_WHY_ROUTER_LOGITS = (
    "the MoE block discards the router's logits (it destructures them into "
    "'_') and routes on the scores and indices it computed from them, so a "
    "write here cannot reach anything — write 'router_scores' to reweight "
    "the chosen experts, or 'expert_idx' to change which experts fire"
)
_WHY_DELTA_KV_MEM = (
    "a memory readout has no independent existence: it is "
    "(S_{t-1}·exp(g_t) · k̂_t) summed, recomputed from the state at every "
    "step, so there is no tensor a write could persist into. Write "
    "'delta_state' to change what the memory holds, or 'delta_value' to "
    "change what is stored into it"
)
_WHY_DELTA_STATE_UPDATE = (
    "its write lowers exactly onto a state edit through the reconstruction "
    "identity S_t = S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t, and that lowering is "
    "deferred — write 'delta_state' instead"
)
_WHY_EXPERT_PERMUTATION = (
    "it is the serving kernel's row bookkeeping (where each (token, slot) "
    "row sits in expert-sorted order), not routing: the kernel derives it "
    "from the routing table, and an edited copy would describe rows that "
    "were never sorted that way. Write 'expert_idx' to change which "
    "experts fire, or 'router_scores' to reweight them"
)
# 📐 ``expert_idx`` measured: an ``add_scaled`` write over the int64 routing
# table ran to completion with no refusal anywhere; on CUDA an out-of-range id
# is a device-side assert far from the write that caused it.
_WHY_EXPERT_IDX = (
    "the routing table carries integer expert ids, not features: a delta, a "
    "scale or a clamp over them yields ids chosen by arithmetic on labels, "
    "which route to arbitrary experts where they stay in range and fail at "
    "the gather where they do not. Swap in an index tensor read from "
    "elsewhere to change which experts fire, or write 'router_scores' to "
    "reweight the experts already chosen. Refusing rather than doing "
    "arithmetic on values that are labels"
)
# The distinction ``attention_scores`` exists to remove: same axes, and a
# delta on either is arithmetically fine — the difference is entirely in what
# happens *next*. After the pattern, the value multiply assumes rows summing
# to 1; after the scores, the model's own softmax renormalizes by construction.
_WHY_ATTENTION_PROBS = (
    "its rows are a probability distribution and the value multiply "
    "immediately downstream assumes they sum to 1 — nothing renormalizes "
    "them after an edit. Write 'attention_scores' instead: it is the same "
    "tensor one step earlier, upstream of the model's own softmax, so every "
    "mechanism is legal there and the rows still sum to 1 by construction"
)

_MOE: tuple[Predicate, ...] = ("moe",)
_SHARED: tuple[Predicate, ...] = ("moe", "shared_expert")
_ROUTED: tuple[Predicate, ...] = ("moe", "grouped_mm")
_SPLIT: tuple[Predicate, ...] = ("split_qkv",)

# The per-family tap table: the three families the attention interior has
# been measured on, keyed by the HF ``model_type`` the adapter reads
# (``GPT2Config.model_type == "gpt2"``, ``LlamaConfig.model_type == "llama"``,
# ``Qwen3_5MoeTextConfig.model_type == "qwen3_5_moe_text"``). The nnsight
# engine's address table for the function interiors
# (``nnsight_tracing/addresses.py``) has the same shape; this is the
# module-boundary half, on the rows. 📐 Per family:
#
# ==========================  =====================  ================  ======================
# component                   ``gpt2``               ``llama``         ``qwen3_5_moe_text``
# ==========================  =====================  ================  ======================
# ``attention_query_pre_rope``  ``c_attn`` block 0/3   ``q_proj`` flat   ``q_norm`` (b,s,H,d)
# ``attention_key_pre_rope``    ``c_attn`` block 1/3   ``k_proj`` flat   ``k_norm`` (b,s,H_kv,d)
# ``attention_value_states``    ``c_attn`` block 2/3   ``v_proj`` flat   ``v_proj`` flat
# ``attention_gate``            —                      —                 ``q_proj`` split 1/2 per head
# projection width            3·H·d = 96             H·d = 16          H·2·d = 512
# ==========================  =====================  ================  ======================
#
# GPT-2's ``GPT2Attention.forward`` computes ``query, key, value =
# self.c_attn(hidden_states).split(self.split_size, dim=2)`` with
# ``split_size = H·d``, so the blocks are contiguous column ranges
# ``[0:H·d] | [H·d:2H·d] | [2H·d:3H·d]`` (measured equal on the fixture).
# ``docs/running_experiments.md`` §5 renders this table from the rows
# (:func:`render_family_table`); nothing else restates it.
_GPT2 = "gpt2"
_LLAMA = "llama"
_QWEN35_MOE = "qwen3_5_moe_text"


def _fused_blocks(split: int) -> dict[str, Any]:
    return {"module": "c_attn", "packing": "fused_blocks", "splits": 3, "split": split}


def _flat(module: str) -> dict[str, Any]:
    return {"module": module, "packing": "flat"}


def _head_axis(module: str) -> dict[str, Any]:
    return {"module": module, "packing": "head_axis"}


_ROWS: tuple[Capability, ...] = (
    # --- the model boundary (layer-less) ---------------------------------- #
    _row("input_ids", tap="module input", writes=None, why=_WHY_INPUT_IDS),
    _row("embeddings", tap="module output"),
    _row("ln_final", tap="module output"),
    _row("lm_head", tap="module output"),
    # --- the residual stream, every layer ---------------------------------- #
    _row("block_input", tap="module input"),
    _row("attention_input_norm", tap="module output"),
    _row("attention_output", tap="module output"),
    _row("block_mid", tap="module input"),
    _row("mlp_input_norm", tap="module output"),
    _row("mlp_input", tap="module input"),
    _row("mlp_output", tap="module output"),
    _row("block_output", tap="module output"),
    # the dense MLP's inner activation — llama's ``act_fn`` output, GPT-2's
    # ``c_proj`` input. Its availability is the dense inner width itself
    # (``ModelInfo.intermediate_size``): an all-MoE tower declares none, and
    # :func:`component_shape` refuses it there.
    _row("mlp_activation", tap="module output"),
    _row("mlp_neuron_output", tap="module input"),
    # --- the full-attention mixer ----------------------------------------- #
    _row(
        "attention_query_pre_rope",
        tap="module output",
        stream="full_attention",
        requires=_SPLIT,
        overrides={
            _GPT2: _fused_blocks(0),
            _LLAMA: _flat("q_proj"),
            _QWEN35_MOE: _head_axis("q_norm"),
        },
    ),
    _row(
        "attention_key_pre_rope",
        tap="module output",
        stream="full_attention",
        requires=_SPLIT,
        overrides={
            _GPT2: _fused_blocks(1),
            _LLAMA: _flat("k_proj"),
            _QWEN35_MOE: _head_axis("k_norm"),
        },
    ),
    _row(
        "attention_value_states",
        tap="module output",
        stream="full_attention",
        requires=_SPLIT,
        overrides={
            _GPT2: _fused_blocks(2),
            _LLAMA: _flat("v_proj"),
            _QWEN35_MOE: _flat("v_proj"),
        },
    ),
    _row(
        "attention_gate",
        tap="module output",
        stream="full_attention",
        requires=("split_qkv", "gated_attention"),
        # the one family with a gate; its absence on the other two is what
        # lets `gated_attention` refuse them at load
        overrides={
            _QWEN35_MOE: {
                "module": "q_proj",
                "packing": "fused_heads",
                "splits": 2,
                "split": 1,
            }
        },
    ),
    _row("attention_query", tap="attention-function slot", stream="full_attention"),
    _row("attention_key", tap="attention-function slot", stream="full_attention"),
    _row("attention_scores", tap="attention-function slot", stream="full_attention"),
    _row("attention_z", tap="attention-function slot", stream="full_attention"),
    _row(
        "attention_probs",
        tap="module output",
        stream="full_attention",
        writes=_SWAP_ONLY,
        why=_WHY_ATTENTION_PROBS,
        # the write goes through the eager attention function, not a hook —
        # a capability an engine may lack, so a coarse verb routes on it
        write_capability="writable_attention_probs",
    ),
    _row("attention_premix", tap="module input", stream="full_attention"),
    _row(
        "attention_result",
        tap="derived from `attention_premix`",
        stream="full_attention",
        writes=None,
        why=_WHY_ATTENTION_RESULT,
    ),
    # --- the Gated DeltaNet mixer ------------------------------------------ #
    # One semantic name per tensor. The module boundaries and
    # the kernel boundary are served by BOTH engines: the reference engine by
    # hooks and by swapping the modeling file's kernel globals, the nnsight
    # engine by envoys and by its `.source` address table — each translating
    # the one name to its own mechanism (the eight retired `deltanet_*`
    # spellings fold onto these at parse). The `tap` cell names the reference
    # engine's mechanism; the nnsight one is `.source` for the kernel boundary.
    _row("delta_qkv", tap="module output", stream="linear_attention"),
    _row("delta_gate", tap="module output", stream="linear_attention"),
    _row("delta_premix", tap="module input", stream="linear_attention"),
    _row("delta_conv", tap="delta-kernel boundary", stream="linear_attention"),
    # post GVA `repeat_interleave` — value-head space; the nnsight engine's
    # pre-tiling face is `deltanet_query` (BACKEND_PAIRS: `gva_tile`)
    _row(
        "delta_query",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
    ),
    _row(
        "delta_key",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
    ),
    _row("delta_value", tap="delta-kernel boundary", stream="linear_attention"),
    _row("delta_beta", tap="delta-kernel boundary", stream="linear_attention"),
    _row("delta_decay", tap="delta-kernel boundary", stream="linear_attention"),
    _row(
        "delta_kernel_output",
        tap="delta-kernel boundary",
        stream="linear_attention",
    ),
    _row(
        "delta_kv_mem",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
        writes=None,
        why=_WHY_DELTA_KV_MEM,
    ),
    _row(
        "delta_state_update",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
        writes=None,
        why=_WHY_DELTA_STATE_UPDATE,
    ),
    # per step — the nnsight engine's per-chunk face is `deltanet_state`
    # (BACKEND_PAIRS: `chunk_boundary`)
    _row(
        "delta_state",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
    ),
    # --- the three DeltaNet faces only the nnsight engine serves ----------- #
    # Two names stay two names where the tensors differ in shape or timing:
    # q/k before the GVA tiling (key-head space, where `delta_query`/`delta_key`
    # are tiled to value heads) and the state once per 64-token chunk (where
    # `delta_state` is per step). An alias here would rebind, not redirect —
    # `alias_would_rebind` refuses it, and the pair's typed relation is a row
    # (BACKEND_PAIRS) the test helpers read rather than own.
    *(
        _row(
            name,
            tap="`.source` line (fused forward)",
            stream="linear_attention",
            reads=_NNSIGHT,
        )
        for name in ("deltanet_query", "deltanet_key", "deltanet_state")
    ),
    # --- the sparse MoE block and its shared expert ----------------------- #
    _row(
        "router_logits",
        tap="module output",
        requires=_MOE,
        writes=None,
        why=_WHY_ROUTER_LOGITS,
    ),
    _row("router_scores", tap="module output", requires=_MOE),
    _row(
        "expert_idx",
        tap="module output",
        requires=_MOE,
        writes=_SWAP_ONLY,
        why=_WHY_EXPERT_IDX,
    ),
    _row(
        "expert_permutation",
        tap="`.source` line (fused forward)",
        requires=_MOE,
        reads=_NNSIGHT,
        writes=None,
        why=_WHY_EXPERT_PERMUTATION,
    ),
    # the routed interior: the grouped experts dispatch is the reference
    # engine's tap and the only ragged ``expert:`` face served today; the
    # nnsight engine lands the token-major form through its ``.source`` table
    *(
        _row(
            name,
            tap="grouped-experts dispatch",
            requires=_ROUTED,
            expert_selection=_HOOKS,
        )
        for name in (
            "expert_gate_proj",
            "expert_up_proj",
            "expert_activation",
            "expert_neuron_output",
            "expert_output",
        )
    ),
    _row("routed_output", tap="module output", requires=_MOE),
    _row("shared_expert_gate_proj", tap="module output", requires=_SHARED),
    _row("shared_expert_up_proj", tap="module output", requires=_SHARED),
    _row("shared_expert_activation", tap="module input", requires=_SHARED),
    _row("shared_expert_output", tap="module output", requires=_SHARED),
    _row("shared_expert_gate", tap="module output", requires=_SHARED),
)

#: The registry: one row per name in the closed ``Component`` vocabulary, in
#: the vocabulary's order. ``tests/protocol/test_vocabulary_census.py`` holds
#: it to exactly ``set(COMPONENTS)``.
CAPABILITIES: Mapping[str, Capability] = MappingProxyType(
    {
        component: next(row for row in _ROWS if row.component == component)
        for component in COMPONENTS
    }
)
if len(_ROWS) != len(CAPABILITIES):  # pragma: no cover - the census names it
    raise AssertionError("a capability row names a component twice or not at all")


def capability(component: str) -> Capability:
    """The row for ``component``. Every caller already holds a name from the
    closed vocabulary (the parser rejects others), so a miss is a bug here,
    not a document error."""
    try:
        return CAPABILITIES[component]
    except KeyError:
        raise AssertionError(
            f"component {component!r} has no capability row — add one to "
            "registry.CAPABILITIES"
        ) from None


def components_served_by(engine: str) -> frozenset[str]:
    """The components whose row names ``engine`` — what that engine's class
    declares as ``components`` (and ``writable_components``: write policy is
    protocol-wide, not an engine gap)."""
    if engine not in ENGINES:
        raise AssertionError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    return frozenset(c for c, row in CAPABILITIES.items() if engine in row.reads)


def write_capabilities(engine: str | None = None) -> frozenset[str]:
    """The coarse §8 verbs the rows charge for a write — all of them, or the
    ones ``engine`` acquires by serving the component."""
    served = None if engine is None else components_served_by(engine)
    return frozenset(
        row.write_capability
        for c, row in CAPABILITIES.items()
        if row.write_capability is not None and (served is None or c in served)
    )


#: The mixer stream a component exists on, for the components that exist on
#: only one — a **view of the rows**, and the one table both halves of the
#: stream check read: the canonicalizer refuses against the registry entry's
#: ``layer_types`` at load, the engines' shared site resolver against the
#: module the layer really carries at run. A component absent here exists at
#: every layer.
COMPONENT_STREAMS: Mapping[str, Stream] = MappingProxyType(
    {c: row.stream for c, row in CAPABILITIES.items() if row.stream is not None}
)


#: The rows that carry per-family addresses — the attention interior.
INTERIOR_ROWS: tuple[str, ...] = tuple(
    c for c, row in CAPABILITIES.items() if row.overrides
)


def family_in_table(info: ModelInfo) -> bool:
    """Whether the per-family tap table has met ``info``'s family — some
    interior row carries an address for it. A family it has not met is served
    by measurement at run and decided nothing about at load."""
    return info.family is not None and any(
        info.family in CAPABILITIES[c].overrides for c in INTERIOR_ROWS
    )


def predicate_holds(info: ModelInfo, predicate: Predicate) -> bool | None:
    """Whether the registry entry can decide ``predicate`` — ``None`` when the
    fact lives in the module tree and only the run can read it.

    ``moe`` and ``shared_expert`` read the entry's MoE widths. ``split_qkv``
    and ``gated_attention`` read the per-family tap table: for a family it has
    met, the interior is addressable (a fused projection included, through
    the row's declared slices) and the gate exists exactly when the
    ``attention_gate`` row has an address for the family; for a family it has
    not met, ``None`` — the run measures. ``grouped_mm`` is a *load-time knob*
    (``experts_implementation``), not a config fact: an entry adapted from a
    loaded model's config carries it (:attr:`ModelInfo.experts_implementation`)
    and decides; a hand-declared entry leaves it ``None`` and the run's tap-time
    probe is the last-line check (implementation knobs resolve during
    model-capability validation).
    """
    if predicate == "moe":
        return info.num_experts is not None
    if predicate == "shared_expert":
        return info.shared_expert_intermediate_size is not None
    if predicate == "grouped_mm":
        if info.experts_implementation is None:
            return None
        return info.experts_implementation == "grouped_mm"
    if predicate in ("split_qkv", "gated_attention"):
        if not family_in_table(info):
            return None
        if predicate == "split_qkv":
            return True
        return CAPABILITIES["attention_gate"].address_on(info) is not None
    return None


#: What each load-decidable predicate means, for the refusal.
_PREDICATE_MEANS: dict[str, str] = {
    "moe": "a sparse-MoE block (the entry declares no experts)",
    "shared_expert": "a shared expert (the entry declares no shared-expert width)",
    "grouped_mm": (
        "the grouped experts dispatch (the loaded model runs another "
        "experts_implementation — a different factorization whose "
        "intermediates are different tensors; load it with "
        "experts_implementation='grouped_mm', the default)"
    ),
    "split_qkv": "addressable q/k/v projections (the per-family tap table has none)",
    "gated_attention": (
        "an output gate on its attention mixer (the per-family tap table "
        "declares none for this family: only Qwen3.5/3.6's q-projection emits "
        "[q | gate] per head)"
    ),
}


def unavailable_at_load(info: ModelInfo, component: str) -> str | None:
    """Why ``component`` has no tensor on the model ``info`` describes, from
    the row's predicates the entry can decide — or ``None``. The run makes the
    same refusal from the module tree; this is the half ``validate`` can make
    offline, so a dense model's document naming ``routed_output`` is refused
    before a GPU is spent on it."""
    row = capability(component)
    for predicate in PREDICATES:
        if predicate in row.requires and predicate_holds(info, predicate) is False:
            if predicate == "grouped_mm":
                # the knob, not the architecture: the tensor would exist under
                # the default dispatch, so the refusal names the knob to turn
                return (
                    f"component {component!r} needs {_PREDICATE_MEANS[predicate]}; "
                    f"model {info.key!r} was loaded with experts_implementation="
                    f"{info.experts_implementation!r}"
                )
            return (
                f"component {component!r} needs {_PREDICATE_MEANS[predicate]}, "
                f"which model {info.key!r} does not have — there is no such "
                "tensor on this model"
            )
    return None


def expert_axis_refusal(component: str, expert: Any) -> str:
    """Why ``expert`` does not apply to ``component`` — shared by the validator
    (which refuses at load) and the site resolver (which refuses if a document
    arrives unvalidated), so the two cannot describe the absent axis
    differently."""
    faces = [c for c, row in CAPABILITIES.items() if row.expert_selection]
    listed = ", ".join(f"'{c}'" for c in faces[:-1]) + f" and '{faces[-1]}'"
    return (
        f"site names expert {expert!r} on component {component!r}, which has no "
        "per-expert axis: the router's axes are all-experts or top-k, and the "
        "shared expert is not one of the routed experts. The per-expert "
        f"interior components are {listed}."
    )


def write_policy_refusal(ename: str, component: str, mechanism: str) -> str | None:
    """Why ``mechanism`` may not be written to ``component`` — or ``None`` when
    it may. **The** write policy: the validator applies it at load and the
    executor at the plan, so the two cannot disagree about what a write may do.
    A read-only row refuses every mechanism; a restricted row refuses the
    mechanisms outside its set. The text names the alternative (``why``)."""
    row = capability(component)
    if row.writes is None:
        return (
            f"write {ename!r} targets {component!r}, which no write may change: "
            f"{row.why}. Refusing at the plan, before anything runs."
        )
    if mechanism not in row.writes:
        return (
            f"write {ename!r} applies {mechanism!r} to {component!r}, which only "
            f"a whole-value 'swap' may change: {row.why}."
        )
    return None


# --------------------------------------------------------------------------- #
# the rendered docs tables
# --------------------------------------------------------------------------- #

#: The model the component table in ``docs/running_experiments.md`` §5 is
#: drawn for — the one hybrid, MoE checkpoint the vocabulary was built to
#: address, so every row has a "blocks" count.
DOCS_TABLE_MODEL = "Qwen/Qwen3.6-35B-A3B"

_TABLE_HEADER = (
    "| component | blocks | shape | tap | engines | write |\n|---|---|---|---|---|---|"
)

#: The five groups the table draws, in order, with their headings.
_TABLE_GROUPS: tuple[tuple[str, str], ...] = (
    ("boundary", "**Model boundary (no `layer`)**"),
    ("residual", "**Residual stream and dense MLP — every layer**"),
    (
        "full_attention",
        "**Full-attention mixer interior — the {full} `full_attention` layers**",
    ),
    (
        "linear_attention",
        "**Gated DeltaNet mixer interior — the {linear} `linear_attention` layers**",
    ),
    ("moe", "**Sparse MoE + shared expert — every layer**"),
)


def _table_group(row: Capability) -> str:
    if row.component in LAYERLESS_COMPONENTS:
        return "boundary"
    if row.stream is not None:
        return row.stream
    if "moe" in row.requires:
        return "moe"
    return "residual"


def _write_cell(row: Capability) -> str:
    if row.writes is None:
        return f"read-only — {row.why}"
    if row.writes == _ANY_MECHANISM:
        return "any mechanism"
    allowed = " ".join(f"`{m}`" for m in sorted(row.writes))
    return f"{allowed} only — {row.why}"


def _engines_cell(row: Capability) -> str:
    if row.reads == _BOTH:
        return "both"
    return " ".join(f"`{e}`" for e in ENGINES if e in row.reads)


def render_component_tables(info: ModelInfo | None = None) -> str:
    """The §5 component table of ``docs/running_experiments.md``, rendered from
    the rows (and from :func:`component_shape` for the shape column) for
    ``info`` — the A3B entry by default. The committed table is checked
    against this rendering row for row."""
    info = get_model_info(DOCS_TABLE_MODEL) if info is None else info
    assert info.layer_types is not None  # the docs model declares its pattern
    n_full = info.layer_types.count("full_attention")
    n_linear = info.layer_types.count("linear_attention")
    blocks = {
        "boundary": "— (layer-less)",
        "residual": f"every layer ({info.num_layers})",
        "full_attention": f"full-attn ({n_full})",
        "linear_attention": f"DeltaNet ({n_linear})",
        "moe": f"every layer ({info.num_layers})",
    }
    sections: list[str] = []
    for group, heading in _TABLE_GROUPS:
        lines = [heading.format(full=n_full, linear=n_linear), "", _TABLE_HEADER]
        for component in COMPONENTS:
            row = CAPABILITIES[component]
            if _table_group(row) != group:
                continue
            try:
                shape = f"`{component_shape(info, component).describe()}`"
                where = blocks[group]
            except ValidationError:
                # no tensor on this architecture: the row exists, the box does
                # not (the A3B's MLP is a sparse-MoE block at every layer)
                shape = "—"
                where = "none — no such tensor on this architecture"
            lines.append(
                f"| `{component}` | {where} | {shape} | {row.tap} | "
                f"{_engines_cell(row)} | {_write_cell(row)} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def engine_component_summary(engine: str) -> str:
    """The ``N of M`` cell of spec §8's generated component row."""
    return f"{len(components_served_by(engine))} of {len(COMPONENTS)}"


# --------------------------------------------------------------------------- #
# the per-family tap table: native packing → shape, and its rendering
# --------------------------------------------------------------------------- #


def native_shape(address: Mapping[str, Any], value: FeatureShape) -> FeatureShape:
    """The native tensor the address's module emits, as a shape — the
    component's family-independent ``value`` shape (:func:`component_shape`,
    ``(batch, position, head·feature)``) re-packed the way the row says this
    family's module packs it. The executor converts by it in both directions
    (:mod:`causalab.neural.shared.layout`), so a module whose real tensor
    disagrees raises rather than being reinterpreted."""
    head = next(a for a in value.axes if a.kind == "head")
    feature = next(a for a in value.axes if a.kind == "feature")
    assert head.width is not None and feature.width is not None
    packing = address["packing"]
    if packing == "flat":
        return dataclasses.replace(value, flat_inner=True)
    if packing == "head_axis":
        return dataclasses.replace(value, flat_inner=False)
    if packing == "fused_heads":
        return shapes.bs_fused_heads(
            head.width, address["splits"], address["split"], feature.width
        )
    assert packing == "fused_blocks", packing
    return shapes.bs_fused_blocks(
        address["splits"], address["split"], head.width, feature.width
    )


def families_in_table() -> tuple[str, ...]:
    """Every family some interior row has an address for, sorted."""
    return tuple(sorted({f for c in INTERIOR_ROWS for f in CAPABILITIES[c].overrides}))


def _address_cell(component: str, family: str) -> str:
    address = CAPABILITIES[component].address_on(
        # any widths: the rendered shape carries axis names, not numbers
        ModelInfo(
            key="render",
            hidden_size=1,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            head_dim=1,
            intermediate_size=None,
            vocab_size=1,
            family=family,
        )
    )
    if address is None:
        return "— (no such tensor: refused at load and at run)"
    shape = native_shape(address, shapes.bs_flat_heads(1, 1)).describe()
    cell = f"`{address['module']}` output `{shape}`"
    if address["packing"] in _FUSED_PACKINGS:
        cell += f", split {address['split']} of {address['splits']}"
    return cell


def render_family_table() -> str:
    """The per-family attention-interior table of ``docs/running_experiments.md``
    §5, rendered from the rows' ``overrides``: one row per interior component,
    one column per family the table has met. The committed table is checked
    against this rendering row for row (``tests/protocol/test_vocabulary_census.py``)."""
    families = families_in_table()
    header = "| component | " + " | ".join(f"`{f}`" for f in families) + " |"
    rule = "|---|" + "---|" * len(families)
    lines = [header, rule]
    for component in INTERIOR_ROWS:
        cells = " | ".join(_address_cell(component, f) for f in families)
        lines.append(f"| `{component}` | {cells} |")
    return "\n".join(lines) + "\n"


def render_widthless_components(info: ModelInfo | None = None) -> str:
    """One paragraph for the method pages: how many components a featurizer
    may attach to on the docs model — every component with a feature width
    (:func:`component_width`) — and which have none, decided by calling the
    derivation the loader calls rather than by a list."""
    info = get_model_info(DOCS_TABLE_MODEL) if info is None else info
    without: list[str] = []
    for component in COMPONENTS:
        try:
            component_width(info, component)
        except (ValidationError, ValueError):
            without.append(f"`{component}`")
    return (
        f"On `{info.key}` a featurizer attaches to "
        f"{len(COMPONENTS) - len(without)} of the {len(COMPONENTS)} components — "
        "every one whose component has a feature width to derive its shape "
        "from, which is every one but " + ", ".join(without) + ".\n"
    )


def render_gate_group_table(info: ModelInfo | None = None) -> str:
    """The §2.5 ``group`` table of the DBM method page: per group, the axis it
    shares one parameter across (``schema.GATE_GROUP_AXES``) and the
    components it is legal on, on the docs model — decided by calling
    :func:`site_group_map`, the derivation the canonicalizer, the loader and
    the executor share, on every component."""
    info = get_model_info(DOCS_TABLE_MODEL) if info is None else info
    lines = ["| group | axis | components it is legal on |", "|---|---|---|"]
    for group in GATE_GROUPS:
        legal: list[str] = []
        for component in COMPONENTS:
            try:
                site_group_map(info, group, component)
            except (ValidationError, ValueError):
                continue
            legal.append(f"`{component}`")
        lines.append(f"| `{group}` | `{GATE_GROUP_AXES[group]}` | {', '.join(legal)} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# the family plugin contract
# --------------------------------------------------------------------------- #
#
# A model *family* is a module tree, and everything an engine needs to know
# about one is declared here, beside the rows, as data: **detection** (one
# predicate over the loaded tree — never a config-string match), **component
# resolution** (semantic name → the family's tap: which module, which side,
# which function slot), the **mixer children** and the stream each means,
# **reconstruction identities** with dtype-keyed tolerances, and optional
# per-family evaluators for the rows' predicates. Tensor-shape contracts stay
# :func:`component_shape` (family-independent), supported mechanisms and
# aliases stay the rows' ``reads`` / ``writes`` / ``aliases`` cells, and the
# attention interior's per-family *address* stays the rows' ``overrides``
# — the adapter says the interior is tapped at the mixer *from the row*, and
# nothing is stated twice.
#
# The vocabulary is one global closed ``Component`` literal and the adapter
# declares **per-family availability** over it: a
# component with no tap on a family is refused by name, at the registry, not by
# an ``AttributeError`` out of a module lookup. The stated limit follows: a
# third party registers a family that serves the *existing* names on a new
# tree (``register_family``, from any module — the site resolver reads the
# registry and names no family), and cannot mint a component; a new tensor is
# a row here first.

#: Where a tap's module lives: the model root's four addressed children (the
#: adapter's :class:`TreeAddress`), one decoder block, that block's mixer
#: (whichever stream it carries) or its MLP.
TapScope = Literal["embedding", "final_norm", "lm_head", "block", "mixer", "mlp"]
TAP_SCOPES: tuple[TapScope, ...] = get_args(TapScope)

#: How the resolved site is landed — :attr:`ResolvedSite.kind` in
#: ``neural/shared/sites.py``: a module's input or output (a hook, an envoy),
#: or one of the function-boundary mechanisms: a slot of the attention
#: function, of the delta kernel's globals, of the grouped experts dispatch,
#: or a line inside a fused forward that only ``.source`` addressing reaches.
HookKind = Literal["in", "out", "interface", "delta", "experts", "interior"]
HOOK_KINDS: tuple[HookKind, ...] = get_args(HookKind)
_SLOT_KINDS: frozenset[str] = frozenset({"interface", "delta", "experts"})


@dataclasses.dataclass(frozen=True)
class Tap:
    """One family's tap for one component: the module (a scope and a dotted
    child path under it — ``""`` is the scope module itself), the hook kind,
    and the details the resolver hands the executor unchanged.

    ``from_row`` marks the attention interior: the child is not named here but
    read off the component row's per-family address (``Capability.overrides``),
    so a family the tap table has met is served from the row and one
    it has not is served by measurement or refused — exactly as before.
    """

    scope: TapScope
    path: str = ""
    kind: HookKind = "out"
    #: which element of a tuple payload the tap means (a router's 3-tuple)
    tuple_index: int | None = None
    #: the function slot for a slot kind — and ``"probs"`` on the pattern,
    #: whose read is a module tap and whose write goes through the function
    slot: str | None = None
    #: the **component** whose output receives an input write's delta. One
    #: name, and the engines read everything off it: the module (that
    #: component's own tap), the forward depth the delta lands at
    #: (``plan.COMPONENT_RANK``) and the payload element it rewrites (that
    #: tap's ``tuple_index``) — none of the three is assumed.
    #:
    #: ``block_mid`` is the component that needs it: the norm's input is the
    #: readable residual *and* what the MLP consumes, but the block has
    #: already saved that same value for its residual addition, so a write
    #: that only rewrote the norm's input would be dropped by the skip.
    #: Required rather than remembered — see :func:`register_family`. Only
    #: meaningful on an ``in`` tap.
    writeback: str | None = None
    #: the component's value is *computed from* the capture (``attention_result``)
    derivation: str | None = None
    #: the component whose shape the capture has, when it is not this one's
    shape_of: str | None = None
    from_row: bool = False

    def __post_init__(self) -> None:
        if self.scope not in TAP_SCOPES:
            raise ValueError(f"tap scope {self.scope!r} is not in {list(TAP_SCOPES)}")
        if self.kind not in HOOK_KINDS:
            raise ValueError(f"tap kind {self.kind!r} is not in {list(HOOK_KINDS)}")
        if self.path and not all(part.isidentifier() for part in self.path.split(".")):
            raise ValueError(f"tap path {self.path!r} is not a dotted child path")
        if self.kind in _SLOT_KINDS and self.slot is None:
            raise ValueError(f"a {self.kind!r} tap names its slot")
        if self.from_row and (self.path or self.scope != "mixer"):
            raise ValueError("a from_row tap is the mixer's, with the row's child")
        if self.writeback is not None:
            if self.kind != "in":
                raise ValueError(
                    f"a writeback tap lands an input rewrite's delta, so it is an "
                    f"'in' tap, not {self.kind!r}"
                )
            if self.writeback not in COMPONENTS:
                raise ValueError(
                    f"tap writeback {self.writeback!r} is not a component — it "
                    "names the component whose output receives the delta"
                )


@dataclasses.dataclass(frozen=True)
class TreeAddress:
    """The four children a family's model root is addressed by, plus the
    block's MLP child — dotted paths from the model root (``model.layers``,
    ``transformer.h``), names rather than modules: the protocol layer is
    torch-free."""

    blocks: str
    embedding: str
    final_norm: str
    lm_head: str = "lm_head"
    mlp: str = "mlp"

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            path = getattr(self, field.name)
            if not path or not all(p.isidentifier() for p in path.split(".")):
                raise ValueError(f"tree {field.name}={path!r} is not a dotted path")


def walk(root: Any, path: str) -> Any:
    """``root.a.b.c`` for ``path == "a.b.c"`` — ``None`` where a child is
    missing, so the caller can refuse by name instead of an ``AttributeError``."""
    module = root
    for name in path.split(".") if path else ():
        module = getattr(module, name, None)
        if module is None:
            return None
    return module


@dataclasses.dataclass(frozen=True)
class Identity:
    """A reconstruction identity a family declares: ``component`` is
    recomputed from ``inputs`` by ``formula`` to within the tolerance the
    dtype allows. A test that pins the identity reads the row rather than its
    own literal, which is what makes a new family testable by the same suite;
    a dtype with no declared tolerance is refused, not guessed."""

    name: str
    component: str
    inputs: tuple[str, ...]
    formula: str
    #: dtype (the protocol's ``native_dtype`` spellings) → ``(atol, rtol)``
    tolerance: Mapping[str, tuple[float, float]]
    #: ``component`` is the plain **sum** of ``inputs`` — a residual add, so
    #: each input is an addend of a value the enclosing forward saved a copy
    #: of. This is the one property :class:`FamilyAdapter`'s write-back check
    #: argues from, and it is not structural: a *functional* identity over the
    #: same scope/child/target shape (``attention_output == attention_premix @
    #: W_o``) needs no write-back at all, because a write at the child's input
    #: propagates through the child on its own and adding the delta again
    #: would apply it twice.
    #:
    #: Checked, not trusted: the formula has to be exactly
    #: ``component == a + b [+ ...]`` over ``inputs``. That is also what makes
    #: the engines' unit coefficient safe — they land ``out + delta``, so a
    #: block computing ``α·x + f(x)`` cannot claim this flag.
    additive: bool = False

    def __post_init__(self) -> None:
        for c in (self.component, *self.inputs):
            if c not in COMPONENTS:
                raise ValueError(f"identity {self.name!r} names {c!r}, not a component")
        if not self.tolerance:
            raise ValueError(f"identity {self.name!r} declares no tolerance")
        plain = f"{self.component} == {' + '.join(self.inputs)}"
        # whitespace-normalized, so `a == b+c` is the *same* claim as
        # `a == b + c` rather than a near-miss that reads as a non-sum and
        # silently drops the write-back requirement with the flag omitted
        is_sum = len(self.inputs) >= 2 and "".join(self.formula.split()) == "".join(
            plain.split()
        )
        if self.additive != is_sum:
            raise ValueError(
                f"identity {self.name!r} declares additive={self.additive} but its "
                f"formula is {self.formula!r}"
                + (
                    f", not {plain!r}"
                    if self.additive
                    else " — which is the plain sum of its inputs, so it is additive"
                )
                + ": the flag and the formula say the same thing, and the flag is "
                "what carries a write at one addend to the sum as an unscaled "
                "delta (FamilyAdapter._check_writebacks). Declaring one without "
                "the other is how that requirement goes missing."
            )

    def tolerance_for(self, dtype: str) -> tuple[float, float]:
        try:
            return self.tolerance[dtype]
        except KeyError:
            raise ValueError(
                f"identity {self.name!r} declares no tolerance for dtype {dtype!r} "
                f"(declared: {sorted(self.tolerance)}) — measure it before asserting"
            ) from None


#: A per-family evaluator of one of the rows' predicates over the loaded
#: model: ``(bundle, component, layer) -> refusal text or None``.
Probe = Any


@dataclasses.dataclass(frozen=True)
class FamilyAdapter:
    """One model family's plugin: what a family declares, and all of it.

    ``detect`` is a predicate over the loaded top-level module — ``hasattr`` /
    child-name structure, never ``config.model_type`` — and exactly one
    registered family may detect a tree (:func:`family_for` refuses none and
    several). ``mixers`` maps each mixer child name the family's blocks may
    carry to the stream it means; the shared stream table
    (``neural/shared/streams.py``) reads the union over registered families and
    keeps refusing a block that carries children of two streams. ``taps`` is
    the family's availability over the global vocabulary: a component absent
    here does not exist on the family and is refused by name. ``identities``
    may be empty — a family with none declares none. ``probes`` overrides the
    resolver's shared module-tree evaluators per predicate, for a family whose
    tree spells an architectural fact differently.
    """

    family: str
    detect: Any
    tree: TreeAddress
    mixers: Mapping[str, Stream]
    taps: Mapping[str, Tap]
    identities: tuple[Identity, ...] = ()
    probes: Mapping[Predicate, Probe] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.family.isidentifier():
            raise ValueError(f"family name {self.family!r} is not an identifier")
        if not callable(self.detect):
            raise ValueError(f"family {self.family!r}: detect is not callable")
        if not self.mixers:
            raise ValueError(f"family {self.family!r} declares no mixer child")
        for child, stream in self.mixers.items():
            if not child.isidentifier() or stream not in STREAMS:
                raise ValueError(
                    f"family {self.family!r}: mixer {child!r} → {stream!r} is not a "
                    f"child name and a stream in {list(STREAMS)}"
                )
        unknown = sorted(set(self.taps) - set(COMPONENTS))
        if unknown:
            raise ValueError(
                f"family {self.family!r} declares taps for {unknown}, which are not "
                "in the component vocabulary — a new tensor is a capability row "
                "first (causalab/protocol/registry.py), then a tap here"
            )
        for component, tap in self.taps.items():
            if not isinstance(tap, Tap):
                raise ValueError(f"family {self.family!r}: {component} needs a Tap")
            if tap.shape_of is not None and tap.shape_of not in COMPONENTS:
                raise ValueError(
                    f"family {self.family!r}: {component} shape_of unknown"
                )
            if tap.from_row and component not in INTERIOR_ROWS:
                raise ValueError(
                    f"family {self.family!r}: {component} is not an interior row, "
                    "so no per-family address exists to read its child from"
                )
        unknown = sorted(set(self.probes) - set(PREDICATES))
        if unknown:
            raise ValueError(f"family {self.family!r}: probes for {unknown}")
        names = [identity.name for identity in self.identities]
        if len(set(names)) != len(names):
            raise ValueError(f"family {self.family!r} declares an identity twice")
        self._check_writebacks()
        object.__setattr__(self, "mixers", MappingProxyType(dict(self.mixers)))
        object.__setattr__(self, "taps", MappingProxyType(dict(self.taps)))
        object.__setattr__(self, "probes", MappingProxyType(dict(self.probes)))

    def _check_writebacks(self) -> None:
        """Both halves of the write-back contract, on the declaration alone.

        **Declared ones resolve.** ``sites._writeback`` reads the target's
        module, forward depth and payload element off *that component's own
        tap*, for every tap that declares a write-back — so a target this
        family does not tap is fatal at resolve time regardless of identities,
        and is refused here where the reason can be stated.

        **Required ones are declared.** This guards a real bug class. A component
        tapped as the input of a *child* module, and named as an addend of an
        **additive** identity, has had its value saved by the enclosing scope
        for that addition before the tap fires: a write that rewrites only the
        child's input is dropped, and the identity then fails under a write
        while still holding on a clean forward — which is what makes the
        omission easy to ship.

        ``additive`` is load-bearing and not structural. ``attention_premix``
        has the identical shape — an ``in`` tap on a child whose module's
        output is another component — but ``attention_output`` is a
        *function* of it, not a sum over it: a write at the child's input
        propagates through the child on its own, and a write-back would apply
        the delta twice. An ``in`` tap on the scope module itself
        (``block_input``) is exempt for the same kind of reason: nothing has
        been saved yet when it fires.
        """
        for component, tap in self.taps.items():
            if tap.writeback is None:
                continue
            if tap.writeback == component:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} with a write-back "
                    "to itself — the delta is carried to a value the enclosing "
                    "forward saved *before* this tap, which cannot be this tap"
                )
            if tap.writeback not in self.taps:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} with "
                    f"writeback={tap.writeback!r}, which the family does not tap "
                    "— the delta's module, forward depth and payload element are "
                    "all read off that component's own tap, so it has to exist"
                )
            target = self.taps[tap.writeback]
            # `sites._writeback` resolves the target with `_tap_module` alone,
            # so the only target it reproduces faithfully is one `resolve_site`
            # would also serve from that call — the generic module-boundary
            # branch. Every earlier branch of that dispatch yields a different
            # module or a different payload element than the component names,
            # and each is visible from this Tap plus CAPABILITIES.
            disqualifier = None
            if target.kind != "out":
                disqualifier = (
                    f"its tap is {target.kind!r}, and an input tap resolves to the "
                    "enclosing module — the landing would be ordered before the "
                    "write it is a delta of"
                )
            elif target.slot is not None:
                disqualifier = (
                    f"its tap names function slot {target.slot!r}, whose write goes "
                    "through the attention function rather than a module boundary"
                )
            elif target.from_row:
                disqualifier = (
                    "its tap is from_row, an attention interior whose module is the "
                    "row's per-family child, not the mixer `_tap_module` returns"
                )
            elif "moe" in CAPABILITIES[tap.writeback].requires:
                disqualifier = (
                    "it is a routed (MoE) component, resolved through the experts "
                    "dispatch rather than as a module boundary"
                )
            if disqualifier is not None:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} with "
                    f"writeback={tap.writeback!r}, which is not a plain "
                    f"module-output boundary: {disqualifier}. The delta's module, "
                    "forward depth and payload element are all read off that tap."
                )
        addends: dict[str, list[Identity]] = {}
        for identity in self.identities:
            if identity.additive:
                for component in identity.inputs:
                    addends.setdefault(component, []).append(identity)
        for component, tap in self.taps.items():
            over = addends.get(component)
            if tap.kind != "in" or not tap.path or over is None:
                continue
            if len(over) > 1:
                raise ValueError(
                    f"family {self.family!r} makes {component!r} an addend of "
                    f"{sorted(i.name for i in over)} — a write there would owe a "
                    "delta to each, and one tap declares one write-back target"
                )
            identity = over[0]
            if tap.writeback is None:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} as the input of "
                    f"child {tap.path!r}, and declares additive identity "
                    f"{identity.name!r} ({identity.formula}) over it — so a write "
                    f"there must also reach {identity.component!r}, which the "
                    "enclosing forward has already saved. Declare it: "
                    f"Tap(..., writeback={identity.component!r})."
                )
            if tap.writeback != identity.component:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} with "
                    f"writeback={tap.writeback!r}, but additive identity "
                    f"{identity.name!r} ({identity.formula}) makes "
                    f"{identity.component!r} the value a write there has to reach"
                )

    def serves(self, component: str) -> bool:
        return component in self.taps

    def tap_for(self, component: str) -> Tap | None:
        return self.taps.get(component)

    def blocks_of(self, model: Any) -> Any:
        """The decoder-layer list of a model this family detected."""
        blocks = walk(model, self.tree.blocks)
        if blocks is None:
            raise ProtocolError(
                "P4",
                f"family {self.family!r} addresses its blocks at "
                f"{self.tree.blocks!r}, but this model ({type(model).__name__}) has "
                "no such child — the family's tree declaration and the model "
                "disagree",
            )
        return blocks

    def identity_for(self, component: str) -> Identity | None:
        for identity in self.identities:
            if identity.component == component:
                return identity
        return None


_FAMILIES: dict[str, FamilyAdapter] = {}

#: The registered families, by name — read-only view; ``register_family`` is
#: the one way in, from any module.
FAMILIES: Mapping[str, FamilyAdapter] = MappingProxyType(_FAMILIES)


def register_family(adapter: FamilyAdapter) -> None:
    """Register (or replace) one family. Refused if another family already
    declares one of its mixer children as a *different* stream — the shared
    stream table reads the union, and a child name must mean one stream."""
    for other in _FAMILIES.values():
        if other.family == adapter.family:
            continue
        for child, stream in adapter.mixers.items():
            declared = other.mixers.get(child)
            if declared is not None and declared != stream:
                raise ValueError(
                    f"family {adapter.family!r} declares mixer child {child!r} as "
                    f"{stream!r}, but family {other.family!r} declares it as "
                    f"{declared!r} — a mixer child name means one stream"
                )
    _FAMILIES[adapter.family] = adapter


def family(name: str) -> FamilyAdapter:
    """The registered family ``name``, or an error naming the registered ones."""
    try:
        return _FAMILIES[name]
    except KeyError:
        raise ProtocolError(
            "P4",
            f"no model family named {name!r} is registered (registered: "
            f"{sorted(_FAMILIES)}) — causalab.protocol.registry.register_family",
        ) from None


def _children(module: Any) -> list[str]:
    named = getattr(module, "named_children", None)
    if named is None:
        return []
    return sorted(name for name, _ in named())


def family_for(model: Any) -> FamilyAdapter:
    """The one registered family whose predicate recognizes ``model``'s
    module tree — refusing a tree no family detects, and one several do,
    rather than probing in a fixed order (a wrong tree produces plausible
    numbers; the stream table's rule, applied to families)."""
    hits = [adapter for adapter in _FAMILIES.values() if adapter.detect(model)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ProtocolError(
            "P4",
            f"no registered model family detects this module tree "
            f"({type(model).__name__}, children={_children(model)}); the registered "
            f"families are {sorted(_FAMILIES)}. Register one whose predicate "
            "recognizes the tree (causalab.protocol.registry.register_family) — "
            "detection is structural, never a config-string match",
        )
    raise ProtocolError(
        "P4",
        f"families {sorted(a.family for a in hits)} all detect this module tree "
        f"({type(model).__name__}) — a family's predicate must recognize its own "
        "tree alone, so the registry refuses to pick one by order",
    )


def mixer_children() -> dict[str, Stream]:
    """Every mixer child name a registered family declares, and its stream —
    the table the shared stream check reads."""
    out: dict[str, Stream] = {}
    for adapter in _FAMILIES.values():
        out.update(adapter.mixers)
    return out


def identities_for(name: str) -> tuple[Identity, ...]:
    return family(name).identities


def identity(name: str, component: str) -> Identity:
    """The identity family ``name`` declares for ``component`` — or a refusal
    naming what it does declare (a family with none declares none)."""
    found = family(name).identity_for(component)
    if found is None:
        raise ProtocolError(
            "P4",
            f"family {name!r} declares no reconstruction identity for "
            f"{component!r} (declared: "
            f"{[i.name for i in family(name).identities]})",
        )
    return found


# --- the built-in families: the two module trees the engines were built on -- #

#: The delta kernel's slots, the attention function's, and the experts
#: dispatch's — the function-boundary vocabulary the resolver and the engines
#: share, declared once here and read by ``neural/shared/sites.py``.
DELTA_KERNEL_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "delta_conv": "conv",
        "delta_query": "query",
        "delta_key": "key",
        "delta_value": "value",
        "delta_beta": "beta",
        "delta_decay": "decay",
        "delta_kernel_output": "kernel_output",
        "delta_kv_mem": "kv_mem",
        "delta_state_update": "state_update",
        "delta_state": "state",
    }
)
ATTENTION_FUNCTION_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "attention_query": "query",
        "attention_key": "key",
        "attention_scores": "scores",
        "attention_z": "z",
    }
)
EXPERTS_FUNCTION_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "expert_gate_proj": "gate_up",
        "expert_up_proj": "gate_up",
        "expert_activation": "activation",
        "expert_neuron_output": "neuron_output",
        "expert_output": "down",
    }
)

#: The taps every block-shaped tree shares: the block's own sides and the
#: mixer's and MLP's outer boundaries, plus the function-boundary interiors
#: whose module is only the anchor of the function tapped. A tree no family
#: detects still serves these (the nnsight_nnterp engine's standard adapter).
BLOCK_TAPS: Mapping[str, Tap] = MappingProxyType(
    {
        "input_ids": Tap("embedding", kind="in"),
        "embeddings": Tap("embedding"),
        "ln_final": Tap("final_norm"),
        "lm_head": Tap("lm_head"),
        "block_input": Tap("block", kind="in"),
        "block_output": Tap("block"),
        "attention_output": Tap("mixer"),
        "mlp_input": Tap("mlp", kind="in"),
        "mlp_output": Tap("mlp"),
        # element 1 of the mixer's (attn_output, attn_weights); the write goes
        # through the attention function (slot "probs")
        "attention_probs": Tap("mixer", tuple_index=1, slot="probs"),
        **{
            c: Tap("mixer", kind="interface", slot=s)
            for c, s in ATTENTION_FUNCTION_SLOTS.items()
        },
        # the module-boundary interior: the child is the row's per-family address
        **{c: Tap("mixer", from_row=True) for c in INTERIOR_ROWS},
    }
)

#: The Llama tree (Llama / Qwen / Mistral / Gemma, and the Qwen3.5-MoE hybrid
#: whose DeltaNet and MoE interiors are declared here because its blocks live
#: in this tree): ``model.layers``, ``input_layernorm`` /
#: ``post_attention_layernorm``, ``self_attn.o_proj``, a SwiGLU ``mlp.act_fn``.
_LLAMA_TAPS: dict[str, Tap] = {
    **BLOCK_TAPS,
    "attention_input_norm": Tap("block", "input_layernorm"),
    "block_mid": Tap(
        "block", "post_attention_layernorm", "in", writeback="block_output"
    ),
    "mlp_input_norm": Tap("block", "post_attention_layernorm"),
    "attention_premix": Tap("mixer", "o_proj", "in"),
    "attention_result": Tap(
        "mixer",
        "o_proj",
        "in",
        derivation="attention_result",
        shape_of="attention_premix",
    ),
    # act_fn's OUTPUT — act(gate_proj(x)), NOT the down-projection's input
    # (the oracle's semantics, inherited from the pyvene era)
    "mlp_activation": Tap("mlp", "act_fn"),
    "mlp_neuron_output": Tap("mlp", "down_proj", "in"),
    # the Gated DeltaNet mixer: three module sides, then the kernel boundary
    "delta_qkv": Tap("mixer", "in_proj_qkv"),
    "delta_gate": Tap("mixer", "in_proj_z"),
    "delta_premix": Tap("mixer", "out_proj", "in"),
    **{c: Tap("mixer", kind="delta", slot=s) for c, s in DELTA_KERNEL_SLOTS.items()},
    **{
        c: Tap("mixer", kind="interior")
        for c in ("deltanet_query", "deltanet_key", "deltanet_state")
    },
    # the sparse MoE block: a 3-tuple router, a fused experts module, the
    # shared expert's SwiGLU and its mixing scalar
    "router_logits": Tap("mlp", "gate", tuple_index=0),
    "router_scores": Tap("mlp", "gate", tuple_index=1),
    "expert_idx": Tap("mlp", "gate", tuple_index=2),
    "routed_output": Tap("mlp", "experts"),
    **{
        c: Tap("mlp", "experts", kind="experts", slot=s)
        for c, s in EXPERTS_FUNCTION_SLOTS.items()
    },
    "expert_permutation": Tap("mlp", "experts", kind="interior"),
    "shared_expert_gate_proj": Tap("mlp", "shared_expert.gate_proj"),
    "shared_expert_up_proj": Tap("mlp", "shared_expert.up_proj"),
    # down_proj's INPUT: silu(gate_proj(x)) * up_proj(x)
    "shared_expert_activation": Tap("mlp", "shared_expert.down_proj", "in"),
    "shared_expert_output": Tap("mlp", "shared_expert"),
    "shared_expert_gate": Tap("mlp", "shared_expert_gate"),
}

#: The GPT-2 tree: ``transformer.h``, ``ln_1`` / ``ln_2``, a fused
#: ``attn.c_attn`` (the rows' ``fused_blocks`` address) and ``attn.c_proj``,
#: an MLP whose ``c_proj`` INPUT is the activation (the down-projection's
#: input — a different tensor than the llama tree's ``act_fn`` output, and
#: pinned as such by the oracle). No DeltaNet or MoE interior exists on this
#: tree, so none is declared: the stream check and the ``moe`` probe refuse
#: those first, by architecture, exactly as before.
_GPT2_TAPS: dict[str, Tap] = {
    **BLOCK_TAPS,
    "attention_input_norm": Tap("block", "ln_1"),
    "block_mid": Tap("block", "ln_2", "in", writeback="block_output"),
    "mlp_input_norm": Tap("block", "ln_2"),
    "attention_premix": Tap("mixer", "c_proj", "in"),
    "attention_result": Tap(
        "mixer",
        "c_proj",
        "in",
        derivation="attention_result",
        shape_of="attention_premix",
    ),
    "mlp_activation": Tap("mlp", "c_proj", "in"),
    "mlp_neuron_output": Tap("mlp", "c_proj", "in"),
}

_EXACT: Mapping[str, tuple[float, float]] = MappingProxyType({"fp32": (0.0, 0.0)})

#: The residual identities every block satisfies (spec §2.4: ``block_mid =
#: block_input + attention_output``, ``block_output = block_mid +
#: mlp_output``) — exact, because the taps capture the very tensors the block
#: adds. Declared per family so a new family's plugin is tested by them.
_RESIDUAL_IDENTITIES: tuple[Identity, ...] = (
    Identity(
        "residual_mid",
        "block_mid",
        ("block_input", "attention_output"),
        "block_mid == block_input + attention_output",
        _EXACT,
        additive=True,
    ),
    Identity(
        "residual_out",
        "block_output",
        ("block_mid", "mlp_output"),
        "block_output == block_mid + mlp_output",
        _EXACT,
        additive=True,
    ),
)

LLAMA_TREE = FamilyAdapter(
    family="llama_tree",
    detect=lambda model: (
        walk(model, "model.layers") is not None
        and walk(model, "model.embed_tokens") is not None
    ),
    tree=TreeAddress(
        blocks="model.layers", embedding="model.embed_tokens", final_norm="model.norm"
    ),
    mixers={"self_attn": "full_attention", "linear_attn": "linear_attention"},
    taps=_LLAMA_TAPS,
    identities=(
        *_RESIDUAL_IDENTITIES,
        # 📐 the model computes precisely this sum in this order — exact
        Identity(
            "routed_sum",
            "routed_output",
            ("expert_output", "router_scores"),
            "routed_output == Σ_slot expert_output · router_scores",
            _EXACT,
        ),
        # 📐 pinned against the kernel's own returned states — exact in fp32
        Identity(
            "delta_state_recurrence",
            "delta_state",
            ("delta_decay", "delta_key", "delta_state_update"),
            "S_t == S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t  (k̂ = l2norm(delta_key))",
            _EXACT,
        ),
    ),
)

GPT2_TREE = FamilyAdapter(
    family="gpt2_tree",
    detect=lambda model: walk(model, "transformer.h") is not None,
    tree=TreeAddress(
        blocks="transformer.h",
        embedding="transformer.wte",
        final_norm="transformer.ln_f",
    ),
    mixers={"attn": "full_attention"},
    taps=_GPT2_TAPS,
    identities=_RESIDUAL_IDENTITIES,
)

register_family(LLAMA_TREE)
register_family(GPT2_TREE)


# --- typed backend pairs: two spellings, one tensor, a declared relation ---- #

#: How the reference engine's and the nnsight engine's captures of one
#: DeltaNet tensor line up (📐 measured on ``tiny-random/qwen3.5-moe``; the
#: golden tier repeats it on the A3B):
#:
#: ``identical``
#:     same shape, max abs diff 0.0 — **one name**: the nnsight spelling is an
#:     alias of the reference one (``schema.DEPRECATED_COMPONENTS``);
#: ``gva_tile``
#:     the reference engine's tensor is post ``repeat_interleave`` over the
#:     head axis (value-head space), the nnsight one pre (key-head space);
#:     exact after tiling — **two names**;
#: ``chunk_boundary``
#:     per step versus per 64-token chunk; the chunk's state is the step
#:     state at the chunk's last position — **two names**.
Relation = Literal["identical", "gva_tile", "chunk_boundary"]
RELATIONS: tuple[Relation, ...] = get_args(Relation)


@dataclasses.dataclass(frozen=True)
class BackendPair:
    """One DeltaNet tensor as the two engines reach it, and the typed relation
    between the two captures. An ``identical`` pair is an alias (one name,
    two mechanisms); any other relation is a **typed backend requirement**:
    two names, each served by the engine named, related by the declared
    transform — which the test helpers read from here rather than own."""

    hooks: str
    nnsight: str
    relation: Relation
    why: str
    #: the kernel's chunk length, for ``chunk_boundary`` (📐 read off the
    #: kernel's own loop, not off config)
    chunk: int | None = None

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"relation {self.relation!r} not in {list(RELATIONS)}")
        if (self.relation == "chunk_boundary") != (self.chunk is not None):
            raise ValueError(
                "chunk_boundary pairs declare the chunk length; others none"
            )

    @property
    def aliased(self) -> bool:
        return self.relation == "identical"

    @property
    def names(self) -> frozenset[str]:
        return frozenset({self.hooks, self.nnsight})


BACKEND_PAIRS: tuple[BackendPair, ...] = (
    *(
        BackendPair(hooks, nnsight, "identical", "same shape, max abs diff 0.0")
        for hooks, nnsight in (
            ("delta_qkv", "deltanet_qkv"),
            ("delta_conv", "deltanet_qkv_conv"),
            ("delta_gate", "deltanet_gate"),
            ("delta_value", "deltanet_value"),
            ("delta_beta", "deltanet_beta"),
            ("delta_decay", "deltanet_decay"),
            ("delta_kernel_output", "deltanet_core_out"),
            ("delta_premix", "deltanet_gated_out"),
        )
    ),
    BackendPair(
        "delta_query",
        "deltanet_query",
        "gva_tile",
        "delta_query is the kernel's argument, tiled to the value-head count "
        "(post repeat_interleave); deltanet_query is the projection before the "
        "tiling, in key-head space — different shapes, exact after tiling",
    ),
    BackendPair(
        "delta_key",
        "deltanet_key",
        "gva_tile",
        "delta_key is the kernel's argument, tiled to the value-head count "
        "(post repeat_interleave); deltanet_key is the projection before the "
        "tiling, in key-head space — different shapes, exact after tiling",
    ),
    BackendPair(
        "delta_state",
        "deltanet_state",
        "chunk_boundary",
        "delta_state is the recurrent state per step; deltanet_state is the "
        "chunked kernel's state once per 64-token chunk — different timing; the "
        "chunk's state is the step state at the chunk's last position",
        chunk=64,
    ),
)


def backend_pair(component: str) -> BackendPair:
    """The pair ``component`` (either spelling) belongs to."""
    for pair in BACKEND_PAIRS:
        if component in pair.names:
            return pair
    raise AssertionError(f"{component!r} is not a spelling of any backend pair")


def alias_would_rebind(alias: str, canonical: str) -> str | None:
    """Why folding ``alias`` onto ``canonical`` would **rebind** rather than
    redirect — or ``None`` when the two name one tensor in one shape at one
    time. The rule the alias table is held to (``test_registry_shapes.py``:
    an alias that redirects is safe; one that rebinds lets a document load
    and silently mean a different tensor): a declared backend relation other
    than ``identical`` refuses by name, and two vocabulary names whose
    declared shapes differ on the reference entry refuse by shape."""
    for pair in BACKEND_PAIRS:
        if {alias, canonical} == pair.names and not pair.aliased:
            return (
                f"aliasing {alias!r} to {canonical!r} would rebind, not redirect: "
                f"the two are related by {pair.relation!r} — {pair.why}. Two "
                "names with a typed backend requirement stay two names."
            )
    if alias in CAPABILITIES and canonical in CAPABILITIES:
        info = get_model_info(DOCS_TABLE_MODEL)
        try:
            left, right = component_shape(info, alias), component_shape(info, canonical)
        except ValidationError:
            return None
        if left.describe() != right.describe() or left.width != right.width:
            return (
                f"aliasing {alias!r} to {canonical!r} would rebind, not redirect: "
                f"their shapes differ on {info.key!r} — {left.describe()} "
                f"(width {left.width}) versus {right.describe()} (width "
                f"{right.width})"
            )
    return None


# --- the inventory: one producer of "what exists at which layer" ----------- #


@dataclasses.dataclass(frozen=True)
class LayerInventory:
    """One layer: the mixer stream it carries, the components that exist there
    and, per component, the engines that read it and the mechanisms a write
    may use (``None``: read-only)."""

    layer: int
    stream: Stream
    components: tuple[str, ...]
    reads: Mapping[str, frozenset[str]]
    writes: Mapping[str, frozenset[str] | None]


@dataclasses.dataclass(frozen=True)
class Inventory:
    """The tower's public inventory: every layer, plus the
    layer-less components at the model boundary."""

    model: str
    layers: tuple[LayerInventory, ...]
    layerless: tuple[str, ...]

    def where(self, component: str) -> tuple[int, ...]:
        """The layers ``component`` exists at."""
        return tuple(li.layer for li in self.layers if component in li.components)

    def count(self, stream: Stream) -> int:
        return sum(1 for li in self.layers if li.stream == stream)


def _exists(info: ModelInfo, component: str) -> bool:
    if unavailable_at_load(info, component) is not None:
        return False
    try:
        component_shape(info, component)
    except ValidationError:
        return False
    return True


def inventory(
    target: Any, *, adapter: FamilyAdapter | None = None, serves: Any = None
) -> Inventory:
    """Per layer, the mixer stream, the components present and their read /
    write mechanisms — one producer for ``dry-run``, the generated support
    tables and the inventory test.

    ``target`` is a :class:`ModelInfo` (offline: the stream pattern is the
    entry's ``layer_types``, availability is what the rows and the entry can
    decide) or a loaded bundle (anything with ``.info`` and ``.streams``; then
    the streams are the loaded tower's and the bundle's family adapter, if it
    has one, decides which components its tree serves). A component is listed
    at a layer when its row's stream is the layer's or unbound, the entry
    declares no fact against it (``unavailable_at_load``, ``component_shape``)
    and the family, when known, declares a tap for it. ``serves(component,
    layer)`` (``layer`` ``None`` at the model boundary), when given, refines
    that with what a loaded model actually serves — the engines' shared site
    resolver, through ``neural.shared.sites.inventory`` — so a fact only the
    module tree knows (📐 ``tiny-random/qwen3.5-moe``'s config declares a dense
    inner width its MoE blocks do not have) is read off the tree rather than
    off the entry. An entry that declares no layer pattern and is not loaded
    has no inventory — refused, not guessed.
    """
    info: ModelInfo = getattr(target, "info", target)
    streams = getattr(target, "streams", None)
    if streams is None:
        streams = info.layer_types
    if adapter is None:
        adapter = getattr(target, "adapter", None)
    if streams is None:
        raise ValidationError(
            4,
            f"model {info.key!r} declares no layer pattern (layer_types) and is not "
            "loaded, so which mixer each layer carries is unknown — the inventory "
            "is per layer, so there is none to give; load the model, or register "
            "the entry with its layer_types",
        )
    streams = tuple(streams)
    if len(streams) != info.num_layers:
        raise ValidationError(
            4,
            f"model {info.key!r}: {len(streams)} streams for {info.num_layers} layers",
        )
    layerless = tuple(
        c
        for c in COMPONENTS
        if c in LAYERLESS_COMPONENTS
        and _exists(info, c)
        and (adapter is None or adapter.serves(c))
        and (serves is None or serves(c, None))
    )
    layers: list[LayerInventory] = []
    for layer, stream in enumerate(streams):
        present = tuple(
            c
            for c in COMPONENTS
            if c not in LAYERLESS_COMPONENTS
            and CAPABILITIES[c].stream in (None, stream)
            and _exists(info, c)
            and (adapter is None or adapter.serves(c))
            and (serves is None or serves(c, layer))
        )
        layers.append(
            LayerInventory(
                layer=layer,
                stream=stream,
                components=present,
                reads=MappingProxyType({c: CAPABILITIES[c].reads for c in present}),
                writes=MappingProxyType({c: CAPABILITIES[c].writes for c in present}),
            )
        )
    return Inventory(model=info.key, layers=tuple(layers), layerless=layerless)


# --------------------------------------------------------------------------- #
# built-in entries — the models the repo's configs and corpus name.
# Sources: HF config.json of each checkpoint (static metadata, no weights).
# ``family`` is the config class's ``model_type`` (``GPT2Config.model_type ==
# "gpt2"``, ``LlamaConfig`` → ``llama``, ``Qwen3Config`` → ``qwen3``,
# ``Gemma2Config`` → ``gemma2``), i.e. what ``model_info_from_hf_config`` reads
# at run — so ``validate`` and the run key the per-family tap table identically.
# --------------------------------------------------------------------------- #

register_model(
    ModelInfo(
        key="meta-llama/Llama-3.1-8B",
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
    )
)
register_model(
    ModelInfo(
        key="meta-llama/Llama-3.1-8B-Instruct",
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
    )
)
register_model(
    ModelInfo(
        # The onboarding demos' model (demos/onboarding_tutorial/). Registered
        # so `validate` and `explain` can size their documents offline — the
        # pure verbs read this table rather than fetching a config, so a demo
        # naming an unregistered key is checkable only by running it.
        # Source: the checkpoint's own config.json, revision
        # 9213176726f574b556790deb65791e0c5aa438b6.
        key="meta-llama/Llama-3.2-1B-Instruct",
        hidden_size=2048,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=8192,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
    )
)
register_model(
    ModelInfo(
        # The pretrained sibling of the demo model above — same architecture,
        # same tokenizer, different weights. 07_cross_model names both, which
        # is the only reason a second 1B entry exists: a document that reads
        # one checkpoint and writes another has to size both offline.
        # Source: the checkpoint's own config.json, revision
        # 4e20de362430cd3b72f300e6b0f18e50e7166e08.
        key="meta-llama/Llama-3.2-1B",
        hidden_size=2048,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=8192,
        vocab_size=128256,
        native_dtype="bf16",
        family="llama",
    )
)
register_model(
    ModelInfo(
        key="gpt2",
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        intermediate_size=3072,
        vocab_size=50257,
        native_dtype="fp32",
        family="gpt2",
    )
)
register_model(
    ModelInfo(
        key="gpt2-xl",
        hidden_size=1600,
        num_layers=48,
        num_heads=25,
        num_kv_heads=25,
        head_dim=64,
        intermediate_size=6400,
        vocab_size=50257,
        native_dtype="fp32",
        family="gpt2",
    )
)
register_model(
    ModelInfo(
        key="Qwen/Qwen3-4B-Instruct-2507",
        hidden_size=2560,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=9728,
        vocab_size=151936,
        native_dtype="bf16",
        family="qwen3",
    )
)
register_model(
    ModelInfo(
        key="google/gemma-2-2b-it",
        hidden_size=2304,
        num_layers=26,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        intermediate_size=9216,
        vocab_size=256000,
        native_dtype="bf16",
        family="gemma2",
    )
)
# The two MIB circuit-track models (Mueller et al. 2025, ``MIB_circuit_track/
# utils.py``: ``qwen2.5 → Qwen/Qwen2.5-0.5B``, ``gemma2 → google/gemma-2-2b``)
# a circuit-discovery replication runs on. Source: each
# checkpoint's config.json (``Qwen2Config``: hidden 896, 24 layers, 14 query /
# 2 KV heads of 64, intermediate 4864, vocab 151936; ``Gemma2Config`` for the
# base model equals the ``-it`` row above — same architecture, different
# weights). ``tests/protocol/test_registry_shapes.py`` cross-checks each row
# against ``model_info_from_hf_config`` on the cached config when present.
register_model(
    ModelInfo(
        key="Qwen/Qwen2.5-0.5B",
        hidden_size=896,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=4864,
        vocab_size=151936,
        native_dtype="bf16",
        family="qwen2",
    )
)
register_model(
    ModelInfo(
        key="google/gemma-2-2b",
        hidden_size=2304,
        num_layers=26,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        intermediate_size=9216,
        vocab_size=256000,
        native_dtype="bf16",
        family="gemma2",
    )
)
register_model(
    ModelInfo(
        # A hybrid Gated DeltaNet / gated full-attention tower with a sparse MoE
        # block in every layer, loaded as the Qwen3.5-MoE text tower (the same
        # class as the ``tiny-random/qwen3.5-moe`` fixture). Registered so a
        # document naming it validates and digests offline, without
        # ``--register-from-hf``.
        # Source: the checkpoint's own config.json, revision
        # 995ad96eacd98c81ed38be0c5b274b04031597b0, read through
        # ``model_info_from_hf_config`` — every value here equals what the run
        # re-registers from the loaded config, so validate and run size the
        # same document identically. Cross-checked against
        # docs/qwen36-35b-a3b-architecture.html.
        key="Qwen/Qwen3.6-35B-A3B",
        hidden_size=2048,
        num_layers=40,
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
        # The text config carries no dense ``intermediate_size`` at all: the
        # block is MoE at every layer, so ``mlp_activation`` names no tensor
        # here, and the adapter reads ``None`` for it too (validate and run
        # agree). Was the adapter's ``4 · hidden`` fallback, 8192, which sized
        # a featurizer against a tensor that does not exist.
        intermediate_size=None,
        vocab_size=248320,
        native_dtype="bf16",
        # 📐 what the adapter reads off this checkpoint's config: the wrapper
        # ``Qwen3_5MoeConfig`` says ``qwen3_5_moe``, but the adapter reads the
        # *text* config, whose ``model_type`` is ``qwen3_5_moe_text`` — the same
        # string the ``tiny-random/qwen3.5-moe`` fixture loads with. 🐞 Was
        # ``qwen3_5_moe`` (the wrapper's spelling), which nothing keyed on;
        # the per-family tap table does, and validate and run must agree.
        family="qwen3_5_moe_text",
        num_experts=256,
        num_experts_per_tok=8,
        # the routed and the shared inner widths happen to coincide on this
        # checkpoint (``d_expert 512`` for both); they are two fields because
        # they are two config keys
        shared_expert_intermediate_size=512,
        moe_intermediate_size=512,
        # Gated DeltaNet: q/k in 16 key heads of 128, v/gate/state in 32 value
        # heads of 128 — the 2× GVA tiling the fixture also has
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        # ``full_attention_interval: 4``: layers 3, 7, …, 39 carry the gated
        # full-attention mixer, the other 30 the DeltaNet one
        layer_types=(("linear_attention",) * 3 + ("full_attention",)) * 10,
    )
)


# --------------------------------------------------------------------------- #
# the alias table is held to the redirect rule at import
# --------------------------------------------------------------------------- #


def _check_aliases(table: Mapping[str, str] | None = None) -> None:
    """Every retired spelling redirects: it is out of the vocabulary, its
    replacement is in, it has a deprecation version, and no declared backend
    relation or shape says it would rebind. Every backend pair agrees with the
    alias table: an ``identical`` pair is an alias, any other pair is two
    single-engine rows. Refused at import — a vocabulary defect is a bug in
    this module, never a document error. ``table`` is the alias table under
    test (the vocabulary's own by default; a test hands in a mutated one)."""
    table = DEPRECATED_COMPONENTS if table is None else table
    for alias, target in table.items():
        if (
            alias in COMPONENTS
            and alias != target
            and alias not in {p.nnsight for p in BACKEND_PAIRS}
        ):
            raise AssertionError(f"alias {alias!r} is still in the vocabulary")
        if target not in COMPONENTS:
            raise AssertionError(f"alias {alias!r} redirects to unknown {target!r}")
        if alias not in DEPRECATED_IN and table is DEPRECATED_COMPONENTS:
            raise AssertionError(f"alias {alias!r} has no deprecation version")
        reason = alias_would_rebind(alias, target)
        if reason is not None:
            raise AssertionError(reason)
    for pair in BACKEND_PAIRS:
        if pair.aliased:
            if table.get(pair.nnsight) != pair.hooks:
                raise AssertionError(
                    f"identical pair {pair.hooks!r}/{pair.nnsight!r} is not an alias"
                )
        else:
            for name, engine in ((pair.hooks, _HOOKS), (pair.nnsight, _NNSIGHT)):
                if name not in CAPABILITIES or CAPABILITIES[name].reads != engine:
                    raise AssertionError(
                        f"{name!r} of the {pair.relation!r} pair must be a row "
                        f"served by exactly {sorted(engine)}"
                    )


_check_aliases()
