"""The per-layer hybrid stream table — one answer, shared by every engine.

A hybrid architecture varies its mixer *per layer*: 📐 on
``tiny-random/qwen3.5-moe`` the text tower is ``['linear_attention',
'linear_attention', 'linear_attention', 'full_attention']``. Which stream a
layer carries is read off the module that is really there (what a tap has to
attach to), never off a family flag or the config alone — and because every
per-layer tap downstream depends on the answer, it must never diverge between
engines. Hence one module. Which child names mean which stream is what each
registered family declares (``registry.FamilyAdapter.mixers``); this module
reads the union and keeps the one rule that makes it right: a block carrying
children of both kinds is refused by name, never probed in a fixed order.
"""

from __future__ import annotations

from typing import Any, Mapping

from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import FAMILIES, mixer_children
from causalab.protocol.schema import Stream

__all__ = [
    "FULL_ATTENTION_CHILDREN",
    "LINEAR_ATTENTION_CHILDREN",
    "mixer_at",
    "stream_at",
]


def _built_in(stream: str) -> tuple[str, ...]:
    """The mixer children the built-in families declare for ``stream``, in
    declaration order — the view a reader of this module expects."""
    return tuple(
        child
        for adapter in FAMILIES.values()
        for child, declared in adapter.mixers.items()
        if declared == stream
    )


#: The mixer children that mean a layer runs full (softmax) attention, and the
#: ones that mean it runs a linear-attention kernel — as the built-in families
#: declare them (``registry.FamilyAdapter.mixers``). A family registered
#: later adds its own child names to the table :func:`stream_at` reads
#: (``registry.mixer_children``), not to these two constants.
FULL_ATTENTION_CHILDREN: tuple[str, ...] = _built_in("full_attention")
LINEAR_ATTENTION_CHILDREN: tuple[str, ...] = _built_in("linear_attention")


def stream_at(
    blocks: Any, layer: int, *, key: str, mixers: Mapping[str, Stream] | None = None
) -> str:
    """Which mixer stream ``blocks[layer]`` actually carries.

    Returns one of ``"full_attention"`` (a ``self_attn``/``attn`` child) or
    ``"linear_attention"`` (a ``linear_attn`` child), reading the child →
    stream table every registered family declares (``mixers`` overrides it:
    a bundle passes its own family's declaration). ``key`` names the model
    in refusals.

    Raises:
        ProtocolError: the block has no recognised mixer child, or has
            children of *both* kinds. The second case is hypothetical — no
            built-in family ships it — but probing in a fixed
            order would answer "full_attention" for it silently, and every
            per-layer tap downstream would then attach to the wrong module
            and still produce plausible numbers. A named refusal is the same
            trade this vocabulary makes everywhere else — and the template
            the family predicates follow (``registry.family_for``).
    """
    table = mixer_children() if mixers is None else mixers
    block = blocks[layer]
    full = [
        name
        for name, s in table.items()
        if s == "full_attention" and hasattr(block, name)
    ]
    linear = [
        name
        for name, s in table.items()
        if s == "linear_attention" and hasattr(block, name)
    ]
    if full and linear:
        raise ProtocolError(
            "P4",
            f"layer {layer} of {key!r} carries both a full-attention "
            f"child ({', '.join(full)}) and a linear-attention child "
            f"({', '.join(linear)}) — the stream of a layer must be one or "
            "the other, so extend the stream table in "
            "neural/shared/streams.py to say which this family means",
        )
    if full:
        return "full_attention"
    if linear:
        return "linear_attention"
    raise ProtocolError(
        "P4",
        f"layer {layer} of {key!r} has no recognised mixer child "
        f"(children={sorted(name for name, _ in block.named_children())}) — "
        "extend the stream table in neural/shared/streams.py",
    )


def mixer_at(
    blocks: Any, layer: int, *, key: str, mixers: Mapping[str, Stream] | None = None
) -> Any:
    """The attention/mixer module at ``layer``, whichever stream it is.

    Resolved *through* :func:`stream_at` rather than by its own probe, so the
    two can never disagree about a block: one answer, one place."""
    table = mixer_children() if mixers is None else mixers
    stream = stream_at(blocks, layer, key=key, mixers=table)
    block = blocks[layer]
    for name, declared in table.items():
        if declared == stream:
            child = getattr(block, name, None)
            if child is not None:
                return child
    raise AssertionError("unreachable")  # stream_at only answers if one exists
