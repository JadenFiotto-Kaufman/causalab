"""The on-demand switch to eager attention — the one implementation whose
mixer returns its weights and whose scores exist — for a model loaded under
the checkpoint's default. The nnterp engine's block
(``engines/nnsight_nnterp/landers.py``) is its one caller; the reference
engine wraps its forwards in its own context manager.

The switch is two functions rather than a context manager around the
forward, because the forward may run in another process: a block that ships
to NDIF mutates the *server's* module, as its own first statement, and puts
it back after the forward — a config mutated on the client never reaches a
server. In one process the same two calls bracket the same forward.
"""

from __future__ import annotations

from typing import Any

from causalab.protocol.errors import ProtocolError

__all__ = ["restore_attention", "switch_to_eager"]


def switch_to_eager(model: Any) -> str | None:
    """Switch ``model`` to eager attention; the implementation to restore, or
    ``None`` when it was already eager and nothing was switched.

    ``model`` owns the ``config`` and the ``set_attn_implementation`` the
    switch goes through (a transformers model). The switch is verified on
    ``config._attn_implementation``: a model that cannot switch dynamically
    is put back and refused rather than run under the wrong kernel. A caller
    stamps ``"attn_eager"`` into its receipt only on a non-``None`` return:
    a model already eager needed nothing, and the receipt records what the
    run did, not what it would have done.
    """
    previous = model.config._attn_implementation
    if previous == "eager":
        return None
    model.set_attn_implementation("eager")
    if model.config._attn_implementation != "eager":
        model.set_attn_implementation(previous)
        raise ProtocolError(
            "P4",
            "attention-interior taps require eager attention, but this "
            "model cannot switch implementations dynamically; load it "
            'with attn_implementation="eager"',
        )
    return previous


def restore_attention(model: Any, previous: str) -> None:
    """Put back the implementation :func:`switch_to_eager` replaced."""
    model.set_attn_implementation(previous)
