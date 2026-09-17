"""The on-demand switch to eager attention — the one implementation whose
mixer returns its weights — shared by the engines that read the pattern off
a model loaded under the checkpoint's default."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from causalab.protocol.errors import ProtocolError

__all__ = ["eager_attention"]


@contextlib.contextmanager
def eager_attention(
    model: Any, applied_requirements: set[str], *, needed: bool
) -> Iterator[None]:
    """Run the body under eager attention when ``needed``.

    ``model`` owns the ``config`` and the ``set_attn_implementation`` the
    switch goes through (a transformers model, or the nnsight wrapper that
    forwards both). When ``needed`` and the model is not already eager, the
    switch is applied, verified on ``config._attn_implementation`` — a model
    that cannot switch dynamically is refused rather than run under the
    wrong kernel — and reversed on exit. ``"attn_eager"`` is stamped into
    ``applied_requirements`` only when the switch was actually applied: a
    model already eager needed nothing, and the receipt records what the
    run did, not what it would have done.
    """
    if not needed or model.config._attn_implementation == "eager":
        yield
        return
    previous = model.config._attn_implementation
    model.set_attn_implementation("eager")
    try:
        if model.config._attn_implementation != "eager":
            raise ProtocolError(
                "P4",
                "attention-interior taps require eager attention, but this "
                "model cannot switch implementations dynamically; load it "
                'with attn_implementation="eager"',
            )
        applied_requirements.add("attn_eager")
        yield
    finally:
        model.set_attn_implementation(previous)
