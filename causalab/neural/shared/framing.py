"""The frame a row is encoded in (spec §2.2.1): plain text, or the chat frame
rendered through the tokenizer's **own** chat template — and where each
declared segment sits in it.

**Honest location.** Every segment boundary here is computed from the rendered
text and the tokenizer's offset mapping — the machinery a ``variable`` anchor
already uses — never from a column's serialized length or a template string
this package knows. The chat frame renders the prompt column as the one user
turn (plus a system turn when the section names a column for it) **twice**:
with and without the template's generation prompt. The generation prompt is
the ``assistant_prefix`` segment, and it is located as the *difference* of the
two renderings, so no template's spelling is assumed. The ``user`` and
``system`` segments (and any column-sourced segment) are located as the
occurrences of their text in the rendered string: none is ``absent`` and
several are ``ambiguous`` (§2.3's cardinalities), through the same path a
variable takes. ``continuation`` is the greedy continuation (§2.3
``generated``) and is not a prompt-frame segment.

**The real prefix.** ``prefix_lengths`` was 0 for every row before this
module existed. Under ``frame: chat`` it is the number of the row's real
tokens before the user turn's first token — BOS, role markers, the system turn
— so ``{"index": 0}`` is the first token of the user's text and ``{"all":
true}`` runs from there to the row's end, as §2.3 always said. A template that
does not carry the prompt verbatim (one that strips or rewrites it) cannot
locate the user turn, and the row is refused with the alignment reason rather
than given a guessed prefix. The rendered text is encoded with
``add_special_tokens=False``: the template owns its specials, and adding BOS a
second time is the double-BOS refusal ``encode`` already carries.

**Fail-closed twins.** A tokenizer with no chat template under ``frame: chat``
is refused with reason ``chat_template_missing`` — the same document runs on
a templated tokenizer, and a document without a ``segments`` section never
reaches this module (``ExecutorBase._batch`` encodes it exactly as before).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping, Sequence

from causalab.neural.shared.encoding import (
    EncodedBatch,
    _segment_run,
    column_value,
    encode,
    select_field,
)
from causalab.protocol.errors import ProtocolError
from causalab.protocol.segments import CONTINUATION_SEGMENT, SegmentsSpec

__all__ = ["FramedRows", "encode_framed", "frame_rows", "render_chat"]

Spans = tuple[tuple[int, int], ...]


@dataclasses.dataclass(frozen=True)
class FramedRows:
    """One role's rows, framed: the texts the tokenizer receives, whether it
    may add its own specials (plain text) or the template already did (chat),
    and per row every declared segment's candidate char spans."""

    texts: tuple[str, ...]
    add_special_tokens: bool
    segments: tuple[Mapping[str, Spans], ...]


def _occurrences(text: str, value: str) -> Spans:
    """Every non-overlapping occurrence of ``value`` in ``text`` as a char
    span; an empty value occurs nowhere (it would match everywhere)."""
    if not value:
        return ()
    return tuple((m.start(), m.end()) for m in re.finditer(re.escape(value), text))


def render_chat(
    tokenizer: Any, prompt: str, *, system: str | None = None
) -> tuple[str, dict[str, Spans]]:
    """Render one row through the tokenizer's chat template and locate the
    chat segments in the rendering (module docstring)."""
    if not getattr(tokenizer, "chat_template", None):
        raise ProtocolError(
            "P2",
            "the document declares segments.frame 'chat', but this tokenizer "
            "carries no chat template to render it with — there is no honest "
            "chat frame to locate segments in. Use a tokenizer that ships one "
            "(or set one on it), or drop the frame and address the plain text",
            path="segments.frame",
            reason="chat_template_missing",
        )
    if not prompt:
        raise ProtocolError(
            "P2",
            "an empty prompt has no user turn to locate under the chat frame",
            path="segments.frame",
        )
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    without = str(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    )
    rendered = str(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    )
    if not rendered.startswith(without):
        raise ProtocolError(
            "P2",
            "this chat template does not append its generation prompt to the "
            "rendering without one, so the assistant prefix cannot be located "
            f"honestly: {without!r} is not a prefix of {rendered!r}",
            path="segments.frame",
        )
    located: dict[str, Spans] = {
        "user": _occurrences(rendered, prompt),
        "assistant_prefix": (
            ((len(without), len(rendered)),) if len(rendered) > len(without) else ()
        ),
    }
    if system is not None:
        located["system"] = _occurrences(rendered, system)
    return rendered, located


def frame_rows(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    field: str,
    spec: SegmentsSpec,
) -> FramedRows:
    """Frame one role's rows under ``spec``: the plain text of ``field``, or
    its chat rendering, with every declared segment located per row."""
    texts: list[str] = []
    located_rows: list[Mapping[str, Spans]] = []
    for row in rows:
        text = str(select_field(row, field))
        located: dict[str, Spans]
        if spec.frame == "chat":
            system = (
                column_value(row, spec.system.column)
                if spec.system is not None
                else None
            )
            text, located = render_chat(tokenizer, text, system=system)
        else:
            located = {}
        for name, source in spec.declare.items():
            located[name] = _occurrences(text, column_value(row, source.column))
        located.pop(CONTINUATION_SEGMENT, None)  # the generated frame, never here
        texts.append(text)
        located_rows.append(located)
    return FramedRows(
        texts=tuple(texts),
        add_special_tokens=spec.frame != "chat",
        segments=tuple(located_rows),
    )


def encode_framed(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    field: str,
    spec: SegmentsSpec,
    *,
    device: str = "cpu",
) -> EncodedBatch:
    """Encode one role's rows under a ``segments`` section: the framed texts
    through :func:`~causalab.neural.shared.encoding.encode`, the segment table
    on the batch, and — in the chat frame — the real ``prefix_lengths``: the
    count of each row's real tokens before the user turn's first token."""
    framed = frame_rows(tokenizer, rows, field, spec)
    batch = encode(
        tokenizer,
        framed.texts,
        device=device,
        add_special_tokens=framed.add_special_tokens,
        segments=framed.segments,
    )
    if spec.frame != "chat":
        return batch
    # the prefix is where the user turn starts, and the user turn is located
    # like any segment: exactly one verbatim occurrence, or the alignment
    # refusal (a template that strips or rewrites the prompt has no honest
    # prefix to report)
    prefixes = [
        _segment_run(batch, row, "user")[0] - batch.first_real(row)
        for row in range(len(framed.texts))
    ]
    return dataclasses.replace(batch, prefix_lengths=tuple(prefixes))
