"""The family adapter for nnterp's standardized tree.

Sites resolve on the **standard** tree rather than on the raw one because
that is what nnterp buys: every architecture it supports exposes its block
list as ``layers``, its embedding as ``embed_tokens``, its final norm as
``ln_final``, its head as ``lm_head`` and each block's mixer as exactly one of
``self_attn`` / ``linear_attn`` — so the shared block-shaped taps
(``block_input`` … ``lm_head``, ``registry.BLOCK_TAPS``) address any such
model through one :class:`~causalab.protocol.registry.TreeAddress` and one
mixer table, with no per-family branch.

What nnterp does *not* rename is anything below a block: ``input_layernorm``,
``self_attn.o_proj``, ``mlp.act_fn``, ``mlp.gate`` keep their own names on the
envoy tree. Those sub-child taps are the registry families' knowledge, so the
adapter built here takes them from the family whose predicate recognizes the
raw module tree (``registry.family_for`` on the wrapped ``nn.Module``) — the
same declaration the reference engine resolves against. A tree no registered
family detects gets the shared taps alone, and every other component is
refused by name with the registry's own ``component_unavailable`` wording,
because the resolver reads availability off ``adapter.taps``.

The adapter is built once per loaded model and handed to the bundle; it is
never registered (``register_family`` keys the shared stream table, and a
per-bundle adapter is not a family).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import (
    BLOCK_TAPS,
    FamilyAdapter,
    TreeAddress,
    family_for,
    walk,
)
from causalab.protocol.schema import Stream

__all__ = ["STANDARD_MIXERS", "STANDARD_TREE", "matched_family", "standard_adapter"]

#: nnterp's standard names, as the resolver walks them from the model root.
STANDARD_TREE = TreeAddress(
    blocks="layers", embedding="embed_tokens", final_norm="ln_final", lm_head="lm_head"
)

#: The two mixer names nnterp guarantees — each block exposes exactly one of
#: them (``nnterp.rename_utils._check_attention_layers``), whatever the raw
#: tree called it (GPT-2's ``attn`` is ``self_attn`` here).
STANDARD_MIXERS: Mapping[str, Stream] = MappingProxyType(
    {"self_attn": "full_attention", "linear_attn": "linear_attention"}
)


def _detect_standard_tree(model: Any) -> bool:
    return all(
        walk(model, path) is not None
        for path in (
            STANDARD_TREE.blocks,
            STANDARD_TREE.embedding,
            STANDARD_TREE.final_norm,
        )
    )


def matched_family(raw_module: Any) -> FamilyAdapter | None:
    """The registered family whose predicate recognizes the raw module tree,
    or ``None`` when no family (or several) does."""
    try:
        return family_for(raw_module)
    except ProtocolError:
        return None


def standard_adapter(raw_module: Any) -> FamilyAdapter:
    """The adapter serving one standardized model: the standard tree for the
    scopes, nnterp's two mixer names for the stream table, and the matched
    family's taps (or the shared block taps alone) for the components.

    ``mlp`` is the one tree entry a family may spell differently, so it
    follows the matched family; the ``identities`` and ``probes`` are the
    matched family's, because they describe the same modules under other
    top-level names.
    """
    matched = matched_family(raw_module)
    if matched is None:
        return FamilyAdapter(
            family="nnterp_standard",
            detect=_detect_standard_tree,
            tree=STANDARD_TREE,
            mixers=STANDARD_MIXERS,
            taps=BLOCK_TAPS,
        )
    return FamilyAdapter(
        family=f"nnterp_{matched.family}",
        detect=_detect_standard_tree,
        tree=TreeAddress(
            blocks=STANDARD_TREE.blocks,
            embedding=STANDARD_TREE.embedding,
            final_norm=STANDARD_TREE.final_norm,
            lm_head=STANDARD_TREE.lm_head,
            mlp=matched.tree.mlp,
        ),
        mixers=STANDARD_MIXERS,
        taps=matched.taps,
        identities=matched.identities,
        probes=matched.probes,
    )
