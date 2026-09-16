"""SiteResolver: the spec's component vocabulary → concrete module taps.

Each site record resolves to ``(module, io side, feature-axis slice)``.
The map is engine-shared: both engines tap the same modules —
pytorch_hooks with ``register_forward_hook`` / ``register_forward_pre_hook``,
the nnsight engine by handing the same tree access to envoys whose
``.input`` / ``.output`` it reads and assigns in-trace. Writes replace the
same tensor either way. The table mirrors the hook-oracle reference
(``tests/neural/activations/hook_oracle.py``) for the two supported
families:

* **Llama-tree** (Llama/Qwen/Mistral/Gemma): ``model.layers[L]``,
  separate ``self_attn.{q,k,v,o}_proj``, SwiGLU MLP;
* **GPT-2-tree**: ``transformer.h[L]``, fused ``attn.c_attn`` (its three
  ``H·d`` column blocks are the logical q, k and v — addressed as such by
  the per-family tap table), ``attn.c_proj``.

Two semantics deliberately preserved from the oracle:

* ``mlp_activation`` names *different tensors per family* — Llama taps
  ``act_fn``'s output (``act(gate_proj(x))``, NOT the down-projection's
  input), GPT-2 taps ``c_proj``'s input (which IS the down-projection's
  input). Inherited 1:1 from the pyvene era and pinned by the oracle.
* the mixer's **interior** is four module boundaries, not four chunk ops
  (``attention_query_pre_rope``, ``attention_key_pre_rope``,
  ``attention_value_states``, ``attention_gate``) — see
  :func:`_attention_interior_site`, which reads *where* each lives on each
  family off the component rows' per-family addresses (the per-family tap
  table, ``registry.Capability.overrides``): no family is named in
  this module;
* ``attention_premix`` with a ``head`` is the ``[H*d, (H+1)*d]`` column
  slice of the o-projection's **input** — query-head space (``head_dim``
  honours a decoupled ``config.head_dim``). 📐 On a **gated** attention
  family (Qwen3.5/3.6's ``self_attn``, where the mixer multiplies by a
  learned gate before projecting out) that input is therefore
  **post-gate**: it is ``gate * z``, not the attention output ``z``. The
  tap is unchanged and correct — this note exists because "value" reads
  like the pre-gate tensor, and the two differ by an elementwise factor
  that a subspace fit will happily absorb without complaining.

Unsupported components refuse with the registry-extension message style. Which
components those are is per *engine*, and not this module's to say: the map is
engine-neutral, so it carries tap addresses and leaves capability routing to
:func:`causalab.protocol.engine.choose_engine`. **Every fact about a component
that is not an address is read from its capability row**
(:data:`causalab.protocol.registry.CAPABILITIES`): the mixer stream it needs,
the architectural predicates it requires (``moe``, ``shared_expert``,
``grouped_mm``, ``split_qkv``, ``gated_attention`` — this module holds their
*module-tree probes*, :data:`_PREDICATE_PROBES`), which engines serve its
ragged ``expert:`` face, and its write policy (applied by the shared executor
and, at load, by ``validate``). The tables that used to live here — the
read-only, swap-only and normalized-tap dicts, the three stream sets, the MoE
and attention-interior sets — were restatements of those rows and are gone.
In particular ``attention_probs`` is **served**, by both engines
— the reference one taps inside the eager attention call
(``engines/pytorch_hooks/attention_interface.py``), the nnsight one through
``.source``, and both declare ``writable_attention_probs``. Its write is
swap-only by its row's ``writes`` cell, not by any engine's absence. The MoE
interior is served too (``engines/pytorch_hooks/experts_interface.py``), with
``expert_permutation`` routing to the nnsight engine.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from causalab.protocol.errors import ProtocolError
from causalab.protocol import registry
from causalab.protocol.registry import (
    ATTENTION_FUNCTION_SLOTS,
    CAPABILITIES,
    COMPONENT_STREAMS,
    DELTA_KERNEL_SLOTS,
    EXPERTS_FUNCTION_SLOTS,
    INTERIOR_ROWS,
    PREDICATES,
    FamilyAdapter,
    Tap,
    capability,
    component_shape,
    expert_axis_refusal,
    family_for,
    family_in_table,
    head_space_refusal,
    native_shape,
    walk,
)
from causalab.protocol.schema import (
    DEPRECATED_COMPONENTS,
    LAYERLESS_COMPONENTS,
    SiteSpec,
)
from causalab.protocol.shapes import FeatureShape


__all__ = [
    "ATTENTION_FUNCTION_SLOTS",
    "DELTA_KERNEL_SLOTS",
    "EXPERTS_FUNCTION_SLOTS",
    "ResolvedSite",
    "adapter_of",
    "inventory",
    "resolve_band",
    "resolve_site",
]


@dataclasses.dataclass(frozen=True)
class ResolvedSite:
    """One tapped location: the module, which side of it carries the
    activation, an optional feature-axis slice (per-head views), and how the
    module's own tensor shape relates to the executor's ``(batch, position,
    feature)`` contract.

    ``shape`` is never chosen per tap: :func:`resolve_site` reads it from
    :func:`~causalab.protocol.registry.component_shape`, so the description each
    engine converts by and the description the protocol layer validates against
    are the same object. ``tuple_index`` defaults to the historical rule —
    element 0 of a tuple payload. See :mod:`causalab.neural.shared.layout` for
    how the conversion is computed from the declared axes.
    """

    module: Any
    kind: str  # "in" | "out"
    #: The module's native tensor shape; converted to/from the executor's
    #: contract at the hook boundary rather than special-cased per component.
    shape: FeatureShape
    feature_slice: slice | None = None
    layer: int = 0
    component: str = "block_output"
    #: Which element of a tuple payload the tap means. None keeps the historical
    #: rule (element 0 of a tuple, else the payload itself); an explicit index
    #: is required for e.g. a router returning (logits, scores, indices).
    tuple_index: int | None = None
    #: Where inside the attention function this component lives, when it is not
    #: a module boundary at all — see
    #: :mod:`causalab.neural.pytorch_hooks.attention_interface`.
    #:
    #: Set together with ``kind == "interface"`` for the four function-interior
    #: components. ``attention_probs`` is the one site that sets it while
    #: keeping an ordinary ``kind``: the mixer *returns* the pattern, so reading
    #: it is a plain module tap, and only the write has to go through the
    #: function.
    interface_slot: str | None = None
    #: Where an input write's delta has to land as well — see
    #: :class:`Writeback`. Set from the family's declared ``Tap.writeback``;
    #: ``block_mid`` is the component that has one. Reads ignore this field.
    writeback: "Writeback | None" = None
    #: The head the site named, kept alongside ``feature_slice`` because a
    #: *derived* component slices in a space the raw tensor does not have.
    head: int | None = None
    #: The expert the site named — the ragged face of a routed-interior tap:
    #: select the (position, slot) pairs the router sent to this
    #: expert. Carried on the site rather than lowered to a slice, because
    #: which rows it selects is a *runtime* fact (the routing), not a static
    #: one.
    expert: int | None = None
    #: Set when the component's value is **computed from** the tapped tensor
    #: rather than being it. Then ``shape`` describes what is captured and
    #: :func:`~causalab.protocol.registry.component_shape` describes the value —
    #: the one place in the backend where those two differ, and the field exists
    #: so that difference is declared rather than inferred.
    derivation: str | None = None

    @property
    def depth(self) -> tuple[int, int]:
        """(layer, intra-order) — matches the protocol planner's ranks."""
        from causalab.protocol.plan import COMPONENT_RANK, UNRANKED  # one table

        rank = COMPONENT_RANK.get(self.component, UNRANKED)
        if self.component in ("ln_final", "lm_head"):
            return (1_000_000, rank)
        return (self.layer, rank)


def adapter_of(bundle: Any) -> FamilyAdapter:
    """The family plugin serving ``bundle``'s model: the bundle's own
    (detected once, ``ModelBundle.adapter``) or, for a bundle without the
    attribute, detected here (``registry.family_for``) — structurally, never
    off the config."""
    adapter = getattr(bundle, "adapter", None)
    return adapter if adapter is not None else family_for(bundle.model)


def _blocks(bundle: Any) -> Any:
    return adapter_of(bundle).blocks_of(bundle.model)


def _attn(bundle: Any, layer: int) -> Any:
    """The mixer at ``layer`` — ``self_attn``, ``attn`` or ``linear_attn``.

    Was ``block.self_attn`` for every non-GPT-2 model, which AttributeErrors on
    a hybrid tower: 📐 on ``tiny-random/qwen3.5-moe`` three of four layers carry
    ``linear_attn`` (Gated DeltaNet) and only one carries ``self_attn``. The
    per-layer answer lives on the bundle (§5.2)."""
    return bundle.mixer_at(layer)


@dataclasses.dataclass(frozen=True)
class Writeback:
    """Where an input write's delta lands, all of it read off the declared
    target component's own tap: the ``module`` whose output receives the
    delta, the ``component`` that output *is* (so an engine orders the
    landing by that component's forward rank instead of assuming
    ``block_output``'s), and the ``tuple_index`` of the payload element to
    rewrite. ``module`` is an ``nn.Module`` in the hook engine and an nnsight
    ``Envoy`` in the trace engine."""

    module: Any
    component: str
    tuple_index: int | None = None


def _declared_tap(adapter: FamilyAdapter, component: str, layer: int, key: str) -> Tap:
    """The family's tap for ``component`` — or the registry's refusal by name.

    This is the per-family availability row: a component the
    family does not declare is refused *here*, with the family, the component
    and what the family does serve, never as a bare ``AttributeError`` out of a
    module lookup (the failure ``_FULL_ATTENTION_ONLY``'s note records).
    """
    tap = adapter.tap_for(component)
    if tap is None:
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {key!r}: model family "
            f"{adapter.family!r} declares no tap for it — the family's plugin "
            "(registry.FamilyAdapter.taps) does not serve this component, so "
            "there is no such tensor on this family. It serves "
            f"{sorted(adapter.taps)}. Declare the tap in the family's adapter "
            "(causalab.protocol.registry.register_family) if the tree has it.",
            reason="component_unavailable",
        )
    return tap


def _scope_module(bundle: Any, adapter: FamilyAdapter, tap: Tap, layer: int) -> Any:
    """The module a tap's scope names on this model, at ``layer``."""
    if tap.scope in ("embedding", "final_norm", "lm_head"):
        return walk(bundle.model, getattr(adapter.tree, tap.scope))
    block = _blocks(bundle)[layer]
    if tap.scope == "block":
        return block
    if tap.scope == "mixer":
        return _attn(bundle, layer)
    assert tap.scope == "mlp", tap.scope
    return walk(block, adapter.tree.mlp)


def _writeback(
    bundle: Any, adapter: FamilyAdapter, tap: Tap, layer: int
) -> Writeback | None:
    """A tap's declared writeback target, resolved through the *target
    component's own tap* — so the module, the forward depth and the payload
    element come off one declaration and no engine re-derives any of them.

    ``None`` where the tap declares no writeback. ``FamilyAdapter`` has
    already refused, at construction, a family that needs one and omits it
    and one whose target it does not tap, so what is left to fail here is the
    module tree — and that refusal is about a component the document never
    named, so it says which write-back asked for it."""
    if tap.writeback is None:
        return None
    try:
        target = _declared_tap(adapter, tap.writeback, layer, bundle.key)
        module = _tap_module(bundle, adapter, target, tap.writeback, layer)
    except ProtocolError as error:
        raise ProtocolError(
            "P4",
            f"a write at this site has to carry its delta to "
            f"{tap.writeback!r} (the family's declared write-back target), and "
            f"that component does not resolve on this model: {error}",
            reason="component_unavailable",
        ) from error
    return Writeback(
        module=module, component=tap.writeback, tuple_index=target.tuple_index
    )


def _tap_module(
    bundle: Any, adapter: FamilyAdapter, tap: Tap, component: str, layer: int
) -> Any:
    """The module ``tap`` addresses — or a refusal naming the family's claim
    and the tree it disagrees with. ``mlp_activation`` keeps its pre-PR text:
    the load-time twin is ``component_shape`` refusing an entry with no dense
    inner width (an all-MoE tower), and this is the same fact read off the
    module tree for a document arriving unvalidated."""
    scope = _scope_module(bundle, adapter, tap, layer)
    module = walk(scope, tap.path) if scope is not None else None
    if module is not None:
        return module
    if component == "mlp_activation" and scope is not None:
        raise ProtocolError(
            "P4",
            f"mlp_activation: this MLP (children={_children(scope)}) matches no "
            "known family — extend the tap table in pytorch_hooks/sites.py "
            "(and mirror it in the hook oracle).",
            reason="component_unavailable",
        )
    where = f"{tap.scope}.{tap.path}" if tap.path else tap.scope
    children = _children(scope) if scope is not None else None
    raise ProtocolError(
        "P4",
        f"component {component!r} at layer {layer} of {bundle.key!r}: family "
        f"{adapter.family!r} taps it at {where!r}, but this model has no such "
        f"module (children of the {tap.scope}: {children}) — the family's "
        "declaration and the loaded tree disagree; correct the tap in the "
        "family's adapter (causalab.protocol.registry.register_family).",
        reason="component_unavailable",
    )


def _head_slice(bundle: Any, component: str, head: int | None) -> slice | None:
    """The feature-axis slice a ``head`` names — or a refusal.

    The bound comes from the component's own shape, not from
    ``info.num_heads``. 📐 That distinction is not cosmetic: under GQA the
    KV-space components are ``num_key_value_heads`` wide, and a query-space
    bound over them produces a slice that is *empty* rather than out of range.
    Python does not raise on that — the read saves a ``(b, n_pos, 0)`` tensor
    and the write mutates nothing — which is the silent no-op the read-only
    rows of the capability registry exist to prevent elsewhere.
    """
    if head is None:
        return None
    shape = component_shape(bundle.info, component)
    space = shape.head_space
    if space is None:
        raise ProtocolError("P4", head_space_refusal(component, head, shape))
    if not 0 <= head < space:
        raise ProtocolError(
            "P4",
            f"site names head {head} on component {component!r}, which has "
            f"{space} heads ({shape.describe()})",
        )
    width = shape.width
    assert width is not None  # a head axis implies a feature axis
    per_head = width // space
    return slice(head * per_head, (head + 1) * per_head)


#: The mixer's interior at module boundaries: the rows that require
#: addressable q/k/v projections (``split_qkv``), read off the registry rather
#: than listed again. Where each lives on each family is the rows' per-family
#: address (``Capability.overrides``) — the same set as
#: ``registry.INTERIOR_ROWS``, and a census test holds the two equal.
#:
#: 📐 One might expect these to need function-level taps inside the mixer
#: forward. Measured on ``tiny-random/qwen3.5-moe``, three of the four are
#: ordinary ``nn.Module`` outputs: ``Qwen3_5MoeAttention`` runs ``q_norm`` and
#: ``k_norm`` **before** RoPE, so their outputs *are* the pre-RoPE projections,
#: and ``v_proj``'s output is ``v`` itself. Only the gate needs a descriptor
#: trick, and only because it shares a projection with ``q`` — the same trick
#: (a fused axis the layout conversion selects and scatters back through) that
#: serves the three blocks of GPT-2's ``c_attn``.
_ATTENTION_INTERIOR: frozenset[str] = frozenset(
    c for c, row in CAPABILITIES.items() if "split_qkv" in row.requires
)


def _projection_width(module: Any) -> int | None:
    """The output width a projection module declares — ``nn.Linear``'s
    ``out_features``, GPT-2's ``Conv1D.nf`` — or ``None`` for a module that
    declares none (a norm: its output has its input's shape, and the layout
    conversion checks that tensor at hook time)."""
    for attr in ("out_features", "nf"):
        width = getattr(module, attr, None)
        if isinstance(width, int):
            return width
    return None


def _check_projection_width(
    bundle: Any, module: Any, address: Mapping[str, Any], component: str, layer: int
) -> None:
    """The width rule, by name and before any hook: a projection the row
    addresses must emit exactly ``splits × (heads · head_dim)`` in the
    component's own head space (📐 ``H·d = 16`` on llama, ``H·2·d = 512`` on
    qwen3.5-moe's gated q-projection, ``3·H·d = 96`` on GPT-2's ``c_attn``).
    The layout conversion would catch the same disagreement at hook time as an
    internal error; this names the row and the module instead."""
    out = _projection_width(module)
    if out is None:
        return
    value = component_shape(bundle.info, component)
    assert value.width is not None  # every interior component has a feature axis
    splits = int(address.get("splits", 1))
    if out != splits * value.width:
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {bundle.key!r}: the "
            f"per-family tap table says {address['module']!r} emits "
            f"{splits * value.width} features on family {bundle.info.family!r} "
            f"({splits} × {value.width}, {value.describe()}), but this module "
            f"emits {out}. The row and the loaded module disagree — re-measure "
            "the family (causalab/protocol/registry.py, the interior rows' "
            "overrides) before trusting either.",
            reason="component_unavailable",
        )


def _declared_modules(component: str, *, packings: frozenset[str]) -> list[str]:
    """The module names the rows declare for ``component`` under ``packings``,
    across every family — the vocabulary the measured fallback picks from."""
    return sorted(
        {
            address["module"]
            for address in capability(component).overrides.values()
            if address["packing"] in packings
        }
    )


_UNFUSED: frozenset[str] = frozenset({"flat", "head_axis"})


def _has_separate_projections(attn: Any) -> bool:
    """Measured: for each of q, k and v the mixer carries one of the bare
    projections some family's row names (📐 ``q_proj``/``k_proj``/``v_proj``
    on the llama tree). GPT-2's mixer carries none — only ``c_attn``."""
    return all(
        any(
            hasattr(attn, module)
            for module in _declared_modules(component, packings=frozenset({"flat"}))
        )
        for component in INTERIOR_ROWS
        if component != "attention_gate"
    )


def _measured_address(
    bundle: Any, attn: Any, component: str, layer: int
) -> Mapping[str, Any]:
    """The address of ``component`` on a mixer whose family the per-family tap
    table has **not** met — today's measured behaviour, kept exactly, but
    picking among the addresses the rows declare rather than a local table.

    * a norm after the projection wins where the mixer has one (📐 measured on
      qwen3.5-moe: ``q_norm``/``k_norm`` run before ``apply_rotary_pos_emb``,
      so their output *is* the pre-RoPE tensor, ``(b, s, H, d)``);
    * else the bare projection, whose width must be the value's — a projection
      emitting two splits per head with no norm to tap after it is refused
      (its output is not the queries alone), as is one of neither width;
    * a **block order is never inferred**: which contiguous block of a fused
      ``c_attn`` is q is a family fact only a row can state, so a family with
      a fused projection and no row is refused (``_probe_split_qkv``).
    """
    row = capability(component)
    present = [
        address
        for address in row.overrides.values()
        if address["packing"] != "fused_blocks" and hasattr(attn, address["module"])
    ]
    norms = [a for a in present if a["packing"] == "head_axis"]
    if norms:
        return norms[0]
    if not present:
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {bundle.key!r}: family "
            f"{bundle.info.family!r} has no row in the per-family tap table, and "
            f"this mixer (children={_children(attn)}) carries none of the modules "
            f"the rows declare for it ({_declared_modules(component, packings=_UNFUSED)}). "
            "Measure the family and add its addresses to the interior rows' "
            "overrides in causalab/protocol/registry.py.",
            reason="component_unavailable",
        )
    address = present[0]
    module = getattr(attn, address["module"])
    value = component_shape(bundle.info, component)
    assert value.width is not None
    out = _projection_width(module)
    splits = int(address.get("splits", 1))
    if out is not None and out == 2 * value.width and splits == 1:
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {bundle.key!r}: this "
            f"mixer has no norm after {address['module']!r}, so the projection's "
            "output would have to be the pre-RoPE tensor — but that projection "
            "is fused ([q | gate] per head), so its output is not the queries "
            "alone. Addressing a split of a projection with no norm to tap after "
            "it is a row of the per-family tap table: measure the "
            "family and add it.",
            reason="component_unavailable",
        )
    if out is not None and out not in (value.width, 2 * value.width):
        raise ProtocolError(
            "P4",
            f"the projection {address['module']!r} at layer {layer} of "
            f"{bundle.key!r} emits {out} features, which is neither "
            f"{value.width} (heads·head_dim) nor {2 * value.width} (a gated "
            "family's [q | gate] per head). This backend cannot say which columns "
            f"are the {component!r} — measure the family and add a row to the "
            "per-family tap table (causalab/protocol/registry.py).",
            reason="component_unavailable",
        )
    return address


def _interior_address(
    bundle: Any, attn: Any, component: str, layer: int
) -> Mapping[str, Any]:
    """Where ``component`` is on this mixer: the row's address for the family
    (``Capability.overrides``, the per-family tap table), or — for a family the
    table has not met — the measured one."""
    declared = capability(component).address_on(bundle.info)
    if declared is not None:
        return declared
    return _measured_address(bundle, attn, component, layer)


def _attention_interior_site(
    bundle: Any,
    attn: Any,
    component: str,
    layer: int,
    head: int | None,
    tap: Any,
) -> ResolvedSite:
    """Resolve one module-boundary tap inside the mixer, from the rows.

    This is the per-family tap table: the family differences —
    which child of the mixer carries the component, and how that child's
    tensor packs it — are the ``overrides`` of the four interior rows in
    :mod:`causalab.protocol.registry`, keyed by the family the loaded config
    declares. No family is named here. The measured three-family table is
    rendered from those rows into ``docs/running_experiments.md`` §5
    (``registry.render_family_table``) and checked against them.

    📐 What the rows say, in one line each (the rendering has the rest):
    llama taps the bare projections; qwen3.5-moe taps ``q_norm``/``k_norm``
    (before RoPE, ``(b, s, H, d)``), ``v_proj``, and the gate as split 1 of 2
    of ``q_proj``; GPT-2 taps the three ``H·d``-wide blocks of ``c_attn``'s
    output — so the **same logical site** reads and writes on a fused and a
    split projection alike, the fused one through the layout conversion's
    scatter into the native tensor (``layout.from_contract``).

    The ``split_qkv`` and ``gated_attention`` predicates were checked by
    :func:`_check_requires` before this is reached; a family without a row was
    served or refused there by measurement.
    """
    feature_slice = _head_slice(bundle, component, head)
    address = _interior_address(bundle, attn, component, layer)
    module = getattr(attn, address["module"], None)
    if module is None:
        # the row is a claim about the family's module tree; a loaded mixer
        # that lacks the named child is the table disagreeing with the model,
        # refused by name rather than as a bare AttributeError out of the tap
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {bundle.key!r}: the "
            f"per-family tap table's row for family {bundle.info.family!r} taps "
            f"{address['module']!r}, but this mixer (children={_children(attn)}) "
            "has no child of that name. Correct the family's address in the "
            "interior rows' overrides in causalab/protocol/registry.py.",
            reason="component_unavailable",
        )
    _check_projection_width(bundle, module, address, component, layer)
    # the row says how this family's module packs the value; the executor
    # converts by it in both directions, so a tensor that disagrees raises
    shape = native_shape(address, component_shape(bundle.info, component))
    return tap(module, "out", feature_slice=feature_slice, shape=shape)


# The mixer's interior *inside the attention function* is
# ``ATTENTION_FUNCTION_SLOTS`` (declared in the registry beside the family
# taps, re-exported here). 📐 These four are not module boundaries:
# ``transformers`` computes them within one ``attention_interface(...)`` call,
# so ``query`` and ``key`` are its arguments (post-RoPE, and for ``key``
# before ``repeat_kv``), the scores are the softmax's input inside it, and
# ``z`` is its return. See :mod:`causalab.neural.pytorch_hooks.attention_interface`.


#: Components that only exist on a full-attention mixer. A Gated DeltaNet layer
#: has no attention matrix at all — there is nothing to read and nothing to
#: write — so naming one at such a layer is an error about the *architecture*,
#: not a missing feature (§5.3).
#: 🐞 ``attention_premix`` and ``attention_result`` belong here too, and did not
#: before. Both are the o-projection's input, and 📐 a Gated DeltaNet layer has
#: no ``o_proj`` at all — its children are
#: ``[conv1d, in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, norm, out_proj]`` —
#: so naming either at such a layer raised a bare
#: ``AttributeError: 'Qwen3_5MoeGatedDeltaNet' object has no attribute 'o_proj'``
#: out of the tap table instead of the architectural refusal that says why the
#: box does not exist there. ``attention_output`` is deliberately *not* here: a
#: DeltaNet layer does produce a mixer output, and it resolves.
#: Read off the capability rows' ``stream`` cell (``registry.COMPONENT_STREAMS``
#: is their view) rather than declared again here, because the canonicalizer
#: refuses from the same rows against the registry's ``layer_types`` — two
#: tables would be two answers.
_FULL_ATTENTION_ONLY: frozenset[str] = frozenset(
    component
    for component, stream in COMPONENT_STREAMS.items()
    if stream == "full_attention"
)

# The mirror: components that only exist on a Gated DeltaNet mixer.
# A full-attention layer computes no delta-rule state — its mixer has no
# ``in_proj_qkv``/``in_proj_z``/``out_proj`` children at all — and a family
# with no linear stream anywhere (llama, gpt2) hits the same refusal at every
# layer, which is the architectural refusal by name.
# The kernel boundary *inside* the DeltaNet forward is
# ``DELTA_KERNEL_SLOTS`` (the registry's, re-exported). 📐 These are not
# module boundaries: the forward calls two module-global functions
# (``causal_conv1d_fn`` and the delta-rule kernel), so the taps swap those
# globals for the dynamic extent of the tapped mixer's forward; the per-step
# interior is produced by stepping the library's own recurrent
# kernel in the chunked call's shadow. See
# :mod:`causalab.neural.engines.pytorch_hooks.delta_interface`. The nnsight
# engine lands the same names as ``.source`` lines of the fused forward.

#: The mirror set: the Gated DeltaNet interior only exists on a
#: linear-attention mixer — a softmax-attention layer has no recurrent state,
#: no delta kernel and no causal conv, so naming one of these there is the
#: same architectural error in the other direction. It is the part of the
#: protocol's linear-attention components the reference engine does **not**
#: serve (``deltanet_query`` / ``deltanet_key`` / ``deltanet_state``: the
#: pre-tiling and per-chunk faces, ``.source`` lines inside the fused
#: forward); ``_LINEAR_ATTENTION_ONLY`` is the part it does — the module and
#: kernel boundaries, which both engines serve under one name.
#: Both read off the rows.
_DELTANET_INTERIOR: frozenset[str] = frozenset(
    component
    for component, stream in COMPONENT_STREAMS.items()
    if stream == "linear_attention"
    and "pytorch_hooks" not in CAPABILITIES[component].reads
)

_LINEAR_ATTENTION_ONLY: frozenset[str] = (
    frozenset(
        component
        for component, stream in COMPONENT_STREAMS.items()
        if stream == "linear_attention"
    )
    - _DELTANET_INTERIOR
)


def _check_stream(bundle: Any, component: str, spec: SiteSpec, layer: int) -> None:
    """Refuse a site whose stream the layer does not carry, before hooking.

    Two ways to get this wrong, and both are caught here rather than as an
    AttributeError from inside a hook:

    * the site *declares* a ``stream`` the layer does not have — ``stream`` has
      parsed since ``schema.py`` gained it and nothing read it until now (§5.2);
    * the site names a full-attention-only component at a linear-attention
      layer, which no ``stream`` spelling can make true (§5.3).
    """
    actual = bundle.stream_at(layer)
    declared = spec.stream if isinstance(spec.stream, str) else None
    if declared is not None and declared != actual:
        raise ProtocolError(
            "P4",
            f"site names stream {declared!r} at layer {layer}, but that layer "
            f"carries {actual!r} — this is a hybrid tower ({', '.join(bundle.streams)}), "
            "so the stream is a per-layer fact, not a model-wide one",
            reason="component_unavailable",
        )
    if component in _FULL_ATTENTION_ONLY and actual != "full_attention":
        raise ProtocolError(
            "P4",
            f"component {component!r} needs a full-attention mixer, but layer "
            f"{layer} of {bundle.key!r} carries {actual!r} — a Gated DeltaNet "
            "block computes no attention matrix, so there is no such tensor at "
            f"this layer. This tower is ({', '.join(bundle.streams)}).",
            reason="component_unavailable",
        )
    if component in _LINEAR_ATTENTION_ONLY and actual != "linear_attention":
        raise ProtocolError(
            "P4",
            f"component {component!r} needs a Gated DeltaNet (linear-attention) "
            f"mixer, but layer {layer} of {bundle.key!r} carries {actual!r} — a "
            "gated-attention mixer computes no delta-rule state, so there is no "
            f"such tensor at this layer. This tower is "
            f"({', '.join(bundle.streams)}).",
            reason="component_unavailable",
        )
    if component in _DELTANET_INTERIOR and actual != "linear_attention":
        raise ProtocolError(
            "P4",
            f"component {component!r} needs a Gated DeltaNet mixer, but layer "
            f"{layer} of {bundle.key!r} carries {actual!r} — a softmax-attention "
            "block computes no recurrent state and runs no delta kernel, so "
            "there is no such tensor at this layer. This tower is "
            f"({', '.join(bundle.streams)}).",
            reason="component_unavailable",
        )


# --------------------------------------------------------------------------- #
# the architectural predicates — the module-tree half of `Capability.requires`
# --------------------------------------------------------------------------- #
#
# A row declares what a component *needs* (``registry.PREDICATES``); this is
# where each predicate is read off the loaded modules, and what the refusal
# says when it does not hold. The canonicalizer evaluates the ones the registry
# entry can decide (``moe``, ``shared_expert``) at load; every one is evaluated
# here at run, so a document arriving unvalidated is refused by the same rows.
# The texts are the ones the per-branch checks this replaces carried (the
# refusal snapshot pins them); the three that were ``NotImplementedError`` are
# protocol refusals now, which is what they always described.


def _mlp(bundle: Any, layer: int) -> Any:
    """The block's MLP child, as the family's tree names it."""
    block = _blocks(bundle)[layer]
    mlp = walk(block, adapter_of(bundle).tree.mlp)
    if mlp is None:
        raise ProtocolError(
            "P4",
            f"layer {layer} of {bundle.key!r}: family {adapter_of(bundle).family!r} "
            f"names the block's MLP {adapter_of(bundle).tree.mlp!r}, but this block "
            f"(children={_children(block)}) has no such child",
            reason="component_unavailable",
        )
    return mlp


def _children(module: Any) -> list[str]:
    return sorted(name for name, _ in module.named_children())


def _probe_moe(bundle: Any, component: str, layer: int) -> str | None:
    mlp = _mlp(bundle, layer)
    if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
        return None
    return (
        f"component {component!r} needs a sparse-MoE block at layer {layer}, "
        f"but this MLP (children={_children(mlp)}) is not one — extend the tap "
        "table in pytorch_hooks/sites.py."
    )


def _probe_shared_expert(bundle: Any, component: str, layer: int) -> str | None:
    if getattr(_mlp(bundle, layer), "shared_expert", None) is not None:
        return None
    return (
        f"component {component!r} needs a shared expert, which this MoE block "
        f"at layer {layer} does not have."
    )


def _probe_grouped_mm(bundle: Any, component: str, layer: int) -> str | None:
    # the dispatch pin: the interior tensors these
    # components name are the *grouped* function's locals. Another
    # implementation — the "eager" per-expert loop, "batched_mm" — computes
    # the same block output (📐 to 4.2e-7) by a different factorization,
    # whose intermediates are different tensors. Same numbers, wrong
    # provenance: refused by name, naming the knob.
    impl = _experts_implementation(bundle)
    if impl == "grouped_mm":
        return None
    return (
        f"component {component!r} taps the interior of the grouped experts "
        f"dispatch, but this model runs experts_implementation={impl!r} — a "
        "different factorization whose intermediates are different tensors, "
        "even though the block's output agrees. Load the model with "
        "experts_implementation='grouped_mm' (the default), or extend "
        "experts_interface.py for this implementation."
    )


def _probe_split_qkv(bundle: Any, component: str, layer: int) -> str | None:
    """The mixer's q, k and v are addressable: the per-family tap table has met
    the family (its rows address the interior, a fused projection included —
    GPT-2's ``c_attn`` as three logical column blocks), or, for a family it has
    not met, the mixer carries separate projections (measured). A fused
    projection without a row is refused by name: which block is which is a
    family fact only a row can state."""
    if family_in_table(bundle.info):
        return None
    attn = _attn(bundle, layer)
    if _has_separate_projections(attn):
        return None
    if hasattr(attn, "c_attn"):
        return (
            f"component {component!r} needs separate q/k/v projections, and this "
            f"mixer fuses them into one 'c_attn' (children="
            f"{_children(attn)}). Splitting a fused qkv projection is the "
            f"per-family tap table, and family "
            f"{bundle.info.family!r} has no row in it — measure the family and "
            "add its addresses to the interior rows' overrides in "
            "causalab/protocol/registry.py; 'attention_premix' and "
            "'attention_output' read on this family today."
        )
    return (
        f"component {component!r} needs addressable q/k/v projections, and this "
        f"mixer (children={_children(attn)}) has neither separate ones nor a row "
        f"for family {bundle.info.family!r} in the per-family tap table — measure "
        "the family and add its addresses to the interior rows' overrides in "
        "causalab/protocol/registry.py."
    )


def _no_gate(bundle: Any, component: str, layer: int) -> str:
    return (
        f"component {component!r} at layer {layer} of {bundle.key!r}: this "
        "mixer computes no output gate. The box exists only on the "
        "gated-attention family (Qwen3.5/3.6), whose q-projection emits "
        "[q | gate] per head and which multiplies the mixer's output by "
        "sigmoid(gate) before projecting out. On this family there is no such "
        "tensor to read or write."
    )


def _probe_gated_attention(bundle: Any, component: str, layer: int) -> str | None:
    """The mixer computes an output gate: the row addresses one for the family
    (the per-family tap table), or, for a family the table has not met, the
    q-projection measures ``H·2·d`` wide (📐 512 on qwen3.5-moe for H 8, d 32,
    against ``H·d = 16`` on llama) — the doubled width is the gate."""
    row = capability(component)
    if family_in_table(bundle.info):
        return (
            None
            if row.address_on(bundle.info) is not None
            else _no_gate(bundle, component, layer)
        )
    attn = _attn(bundle, layer)
    value = component_shape(bundle.info, component)
    assert value.width is not None
    for module in _declared_modules(component, packings=frozenset({"fused_heads"})):
        projection = getattr(attn, module, None)
        if projection is None:
            continue
        out = _projection_width(projection)
        if out == 2 * value.width:
            return None
        if out is not None and out != value.width:
            raise ProtocolError(
                "P4",
                f"the q-projection at layer {layer} of {bundle.key!r} emits {out} "
                f"features, which is neither {value.width} (heads·head_dim) nor "
                f"{2 * value.width} (a gated family's [q | gate] per head). This "
                "backend cannot say which columns are the queries — measure the "
                "family and add a row to the per-family tap table "
                "(causalab/protocol/registry.py).",
                reason="component_unavailable",
            )
    return _no_gate(bundle, component, layer)


#: One probe per predicate in the registry's vocabulary — the shared
#: module-tree evaluators, which every family uses unless its adapter declares
#: its own cell for a predicate (``FamilyAdapter.probes``). The census guard
#: asserts the keys are exactly ``registry.PREDICATES``; a predicate added to a
#: row without a probe here fails that test, not a document.
_PREDICATE_PROBES: dict[str, Any] = {
    "moe": _probe_moe,
    "shared_expert": _probe_shared_expert,
    "grouped_mm": _probe_grouped_mm,
    "split_qkv": _probe_split_qkv,
    "gated_attention": _probe_gated_attention,
}


def _check_requires(bundle: Any, component: str, layer: int) -> None:
    """Refuse a component whose row requires an architectural fact the loaded
    model does not have, in the rows' predicate order (a fused-qkv family is
    named before its missing gate; a dense MLP before its missing shared
    expert). Each predicate is evaluated by the family's own cell where its
    adapter declares one, by the shared module-tree probe otherwise."""
    row = capability(component)
    probes = adapter_of(bundle).probes
    for predicate in PREDICATES:
        if predicate not in row.requires:
            continue
        probe = probes.get(predicate, _PREDICATE_PROBES[predicate])
        refusal = probe(bundle, component, layer)
        if refusal is not None:
            raise ProtocolError("P4", refusal, reason="component_unavailable")


#: The MoE surface: every row that requires a sparse-MoE block. The
#: module-boundary taps (📐 the router is a module returning a 3-tuple and the
#: experts are a fused module), the dispatch interior, and the kernel's
#: ``expert_permutation`` — read off the rows, not listed again.
_MOE_COMPONENTS: frozenset[str] = frozenset(
    c for c, row in CAPABILITIES.items() if "moe" in row.requires
)

# The routed-expert interior *inside the experts dispatch* is
# ``EXPERTS_FUNCTION_SLOTS`` (the registry's, re-exported). 📐 These are not
# module boundaries: ``Qwen3_5MoeExperts`` stores its weights as 3-D
# parameters and computes the whole interior inside one dispatched
# ``ALL_EXPERTS_FUNCTIONS["grouped_mm"]`` call (its only child is the one
# shared ``act_fn``, which the wrapper hooks for the duration of that call).
# The reference engine taps them by wrapping that dispatch
# (:mod:`causalab.neural.engines.pytorch_hooks.experts_interface`); the
# nnsight engine lands the same components through its `.source` address
# table — both consume the ``kind="experts"`` resolution below.


def _experts_implementation(bundle: Any) -> str:
    """The experts implementation the loaded model dispatches on — read from
    the config the modeling code itself reads."""
    config = getattr(bundle.model.config, "text_config", None) or bundle.model.config
    return str(getattr(config, "_experts_implementation", "<undeclared>"))


def _moe_site(
    bundle: Any,
    adapter: FamilyAdapter,
    tap: Tap,
    component: str,
    spec: SiteSpec,
    layer: int,
) -> ResolvedSite:
    """Resolve one MoE tap, from the family's declared tap.

    📐 Every tap here is ``flat_td``: ``Qwen3_5MoeSparseMoeBlock`` reshapes to
    ``(-1, hidden)`` before the router, so the whole interior is flattened over
    (batch, position) and only the block's own input and output are contract
    shaped. Measured on ``tiny-random/qwen3.5-moe`` at 1x6 tokens, hidden 8,
    128 experts, top-10::

        gate       out -> ((6,128) logits, (6,10) scores, (6,10) int64 indices)
        experts    out -> (6, 8)
        shared_expert.gate_proj / up_proj out -> (6, 32)
        shared_expert.down_proj       in  -> (6, 32)
        shared_expert                 out -> (6, 8)
        shared_expert_gate            out -> (6, 1)

    The router is the reason ``tuple_index`` exists: ``Qwen3_5MoeTopKRouter``
    returns three tensors and the historical "element 0 of a tuple" rule would
    have silently handed back the logits for all three.

    The ``moe`` / ``shared_expert`` / ``grouped_mm`` predicates were checked by
    :func:`_check_requires` before this is reached.
    """
    # The `expert` sub-axis is the ragged face of the routed interior: select
    # the (position, slot) pairs the router sent to one expert. Only the rows
    # whose `expert_selection` names an engine carry it — the router's own axes
    # are all-experts (logits) or top-k (scores, indices), and the shared
    # expert is not one of the routed experts, so `expert` on those is refused
    # rather than silently ignored (the mistake `stream` made). `validate`
    # makes the same refusal at load from the same row; this is for a document
    # arriving unvalidated.
    expert = spec.expert if isinstance(spec.expert, int) else None
    if spec.expert is not None and not capability(component).expert_selection:
        raise ProtocolError(
            "P4",
            expert_axis_refusal(component, spec.expert),
            reason="component_unavailable",
        )
    if expert is not None:
        total = bundle.info.num_experts
        if total is None or not 0 <= expert < total:
            raise ProtocolError(
                "P4",
                f"site names expert {expert} on component {component!r}, but "
                f"{bundle.key!r} routes over {total} experts — the sub-axis "
                "selects one of them by its id.",
            )

    shape = component_shape(bundle.info, component)
    module = _tap_module(bundle, adapter, tap, component, layer)

    if tap.kind == "experts":
        if spec.head is not None and isinstance(spec.head, int):
            # no head axis anywhere in the MoE interior; refuse rather than drop
            _head_slice(bundle, component, spec.head)
        return ResolvedSite(
            module=module,
            kind="experts",
            layer=layer,
            component=component,
            shape=shape,
            interface_slot=tap.slot,
            expert=expert,
        )
    if tap.kind == "interior":
        # the serving kernel's row bookkeeping, inside the fused experts
        # forward — no module boundary and no dispatch slot; only the
        # nnsight engine's `.source` address table lands it, so it resolves
        # to the interior kind and the reference engine refuses by name.
        return ResolvedSite(
            module=module,
            kind="interior",
            layer=layer,
            component=component,
            shape=shape,
        )
    return ResolvedSite(
        module=module,
        kind=tap.kind,
        layer=layer,
        component=component,
        shape=shape,
        tuple_index=tap.tuple_index,
    )


def resolve_band(bundle: Any, spec: SiteSpec) -> tuple[ResolvedSite, ...]:
    """Every module a site addresses, one :class:`ResolvedSite` per layer of
    its band (§2.4 ``layers``), in band order — the one-layer band is the
    one-tuple of :func:`resolve_site`. The band is fanned out here and the
    resolved record stays scalar (``ResolvedSite.layer``): every engine
    consumer of a resolved site — hooks, address tables, the resume check —
    reasons about one module at a time."""
    band = spec.layers if isinstance(spec.layers, tuple) else None
    if band is None or len(band) <= 1:
        return (resolve_site(bundle, spec),)
    return tuple(
        resolve_site(bundle, dataclasses.replace(spec, layers=(layer,)))
        for layer in band
    )


def resolve_site(bundle: Any, spec: SiteSpec) -> ResolvedSite:
    """Resolve one site record to its tap, from the family's declared taps.

    The family is the bundle's plugin (:func:`adapter_of`); *where* a component
    is on it — which module, which side, which function slot — is the
    adapter's tap for the component (``registry.FamilyAdapter.taps``), and a
    component the family does not declare is refused by name. What this
    module keeps is the order of the checks and everything that is not an
    address: the stream check, the predicate probes, the head and expert
    sub-axes, the shape and the derivations — the same for every family.
    Refuses honestly on components this engine does not implement yet.
    """
    component = spec.component
    if not isinstance(component, str):
        raise ProtocolError("P2", f"unresolved site component {component!r}")
    if component in DEPRECATED_COMPONENTS:
        # the parser folds a retired spelling before anything downstream sees
        # it; a SiteSpec built by hand that still carries one is named rather
        # than falling through to "no capability row"
        raise ProtocolError(
            "P2",
            f"site component {component!r} is a retired spelling of "
            f"{DEPRECATED_COMPONENTS[component]!r}, which the parser folds at "
            "load — a site built without the parser must name the current "
            "component",
        )
    band = spec.layers if isinstance(spec.layers, tuple) else ()
    if len(band) > 1:
        # a band is one site across N layers, and a ResolvedSite is one
        # module: the fan-out is `resolve_band`, and an executor lowers a
        # band to its per-layer members (`plan.lower_bands`) before it asks
        # for modules — so a band reaching here is a caller that skipped that
        raise ProtocolError(
            "P2",
            f"site spans layers {list(band)} — a band resolves to one module "
            "per layer (resolve_band), not to one ResolvedSite; the executor "
            "lowers a band to its members before resolving",
        )
    layer = band[0] if band else 0
    head = spec.head if isinstance(spec.head, int) else None
    adapter = adapter_of(bundle)

    def tap(
        module: Any,
        kind: str,
        *,
        feature_slice: slice | None = None,
        tuple_index: int | None = None,
        interface_slot: str | None = None,
        writeback: "Writeback | None" = None,
        shape: FeatureShape | None = None,
        derivation: str | None = None,
    ) -> ResolvedSite:
        """One tap, with its shape read from the component table.

        The shape is resolved *here* rather than at each branch so that adding a
        component means adding a table entry and a module, never a third place
        that has an opinion about the tensor's axes.

        ``shape`` is overridden in exactly two cases, both declared elsewhere
        rather than decided per branch: a *derived* component, whose tap
        captures a different tensor than the one the component names
        (``attention_result``: the tap's ``shape_of``), and the attention
        interior, whose native **packing** is a fact about the family's
        *module*, not the component — 📐 ``Qwen3_5MoeAttention.q_norm`` emits
        ``(b, s, H, d)``, llama's bare ``q_proj`` ``(b, s, H·d)``, GPT-2's
        ``c_attn`` three ``H·d`` blocks — and is read off the component row's
        per-family address (``registry.native_shape``). Same component, same
        width, same head space; only the packing differs, and packing is the
        half of the descriptor the backend owns. Everything the protocol layer
        validates against is family-independent and stays in the one table.
        """
        if shape is None:
            shape = component_shape(bundle.info, component)
        return ResolvedSite(
            module=module,
            kind=kind,
            shape=shape,
            feature_slice=feature_slice,
            layer=layer,
            component=component,
            tuple_index=tuple_index,
            interface_slot=interface_slot,
            writeback=writeback,
            head=head,
            derivation=derivation,
        )

    if component in LAYERLESS_COMPONENTS:
        # the model boundary: the ids as the embedding's INPUT (the one module
        # boundary they cross, read-only and layer-less, §5.4), the embedding's
        # output, the final norm and the head — the family's tree names them
        declared = _declared_tap(adapter, component, layer, bundle.key)
        return tap(
            _tap_module(bundle, adapter, declared, component, layer), declared.kind
        )

    # Order matters: the stream check runs FIRST so that a full-attention-only
    # component at a Gated DeltaNet layer refuses with the architectural reason
    # ("there is no attention matrix here") rather than an engine-shaped one
    # ("this engine has not implemented it yet"). The first is permanent and
    # stayed true once attention_probs landed; the second was a roadmap
    # statement, and is now moot — both engines serve the component. The
    # predicate probes come second, the family's availability third: a MoE
    # component on a dense block is refused as "not a sparse-MoE block", which
    # is the fact, before any family is asked whether it declares a tap.
    _check_stream(bundle, component, spec, layer)
    _check_requires(bundle, component, layer)
    declared = _declared_tap(adapter, component, layer, bundle.key)

    if declared.from_row:
        # the module-boundary interior: the row's per-family address
        return _attention_interior_site(
            bundle, _attn(bundle, layer), component, layer, head, tap
        )
    if component in _MOE_COMPONENTS:
        return _moe_site(bundle, adapter, declared, component, spec, layer)

    module = _tap_module(bundle, adapter, declared, component, layer)

    if declared.kind == "interior":
        # inside a fused forward and its kernel — no module boundary anywhere;
        # the mixer is carried because it identifies whose forward to tap. Same
        # marker as the expert interior: the engine with `.source` addressing
        # serves it, the reference engine refuses by name.
        return tap(module, "interior")
    if component == "attention_probs":
        # element 1 of the mixer's (attn_output, attn_weights). Reading is an
        # ordinary tap; WRITING is not — see attention_interface.py, which
        # owns that half. Its shape has two position axes and so no contract
        # form, which is what makes the executor refuse to gather or featurize
        # it without any component name appearing in that refusal.
        return tap(
            module,
            declared.kind,
            tuple_index=declared.tuple_index,
            interface_slot=declared.slot,
        )
    if declared.derivation is not None:
        # 📐 `attention_result`: the model never computes this. Each head's
        # contribution to the residual stream is
        # `premix[..., h·d:(h+1)·d] @ W_o[:, h·d:(h+1)·d].T`, and the model
        # forms only their sum — by projecting the whole premix at once. So
        # the tap is `attention_premix`'s, and the value is derived from it
        # after the position gather, which keeps the cost
        # `n_positions · H · hidden` rather than `seq · H · hidden`.
        #
        # No `feature_slice`: a `head` here selects in the *result's* space
        # (hidden-wide blocks), not the captured tensor's (head_dim-wide), and
        # naming one makes the derivation cheap rather than making it a slice.
        if head is not None:
            _head_slice(bundle, component, head)  # bound-check, discard
        assert declared.shape_of is not None
        return tap(
            module,
            declared.kind,
            shape=component_shape(bundle.info, declared.shape_of),
            derivation=declared.derivation,
        )
    if declared.kind == "delta":
        # no module boundary: the tensor is an argument or return of the
        # kernel-boundary globals. The mixer is carried as the module — it
        # is what identifies *which* forward's calls to tap.
        if declared.slot == "state":
            # the state has a head axis but no feature axis to slice — the
            # bound is checked here and the executor selects the head on
            # the native matrix after the position gather
            shape = component_shape(bundle.info, component)
            space = shape.head_space
            if head is not None:
                assert space is not None
                if not 0 <= head < space:
                    raise ProtocolError(
                        "P4",
                        f"site names head {head} on component "
                        f"{component!r}, which has {space} heads "
                        f"({shape.describe()})",
                    )
            return tap(module, "delta", interface_slot=declared.slot)
        # the conv's fused unequal widths refuse a head by shape, like the qkv
        return tap(
            module,
            "delta",
            feature_slice=_head_slice(bundle, component, head),
            interface_slot=declared.slot,
        )
    if declared.kind == "interface":
        # No module boundary to hook: these four live inside one call. The
        # module is carried anyway, because it is what identifies *which*
        # mixer's call to tap.
        return tap(
            module,
            "interface",
            feature_slice=_head_slice(bundle, component, head),
            interface_slot=declared.slot,
        )
    # a module side: the head, where the site names one, is a slice of the
    # component's own head space — and a refusal by shape on a component with
    # none (`delta_qkv`'s fused unequal widths; a residual-stream tensor)
    return tap(
        module,
        declared.kind,
        feature_slice=_head_slice(bundle, component, head),
        tuple_index=declared.tuple_index,
        writeback=_writeback(bundle, adapter, declared, layer),
    )


def inventory(bundle: Any) -> registry.Inventory:
    """The loaded model's inventory as this resolver serves it: the registry's
    per-layer inventory (``registry.inventory``), refined by whether each
    ``(component, layer)`` actually resolves on the loaded tree. Offline and
    loaded agree wherever the registry entry knows the fact; where only the
    module tree does (📐 the tiny qwen3.5-moe fixture's config declares a dense
    inner width no MoE block has, so the entry sizes ``mlp_activation`` and the
    tree refuses it), this is the answer a run gives."""

    def serves(component: str, layer: int | None) -> bool:
        try:
            resolve_site(bundle, SiteSpec(component=component, layers=(layer,)))
        except ProtocolError:
            return False
        return True

    return registry.inventory(bundle, serves=serves)
