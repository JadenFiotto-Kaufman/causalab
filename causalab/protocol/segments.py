"""The ``segments`` section (spec §2.2.1) and rule 27.

A row's text has **named segments** — in a chat frame the system turn, the
user turn, the template's generation prompt and the greedy continuation; in
any frame the value of a column the task serialized — and §2.3's anchors gain
``segment`` beside ``variable`` and ``column`` so
``{"index": -1, "scope": {"segment": "assistant_prefix"}}`` is one authored
address. This module owns the section's object model and parser, the closed
vocabularies (:data:`SEGMENT_FRAMES`, :data:`CHAT_SEGMENTS`), and rule 27
(:func:`check`): what the document alone can decide about segments and spans.

**Honest location is the engine's.** A segment is *declared* here and
*located* when the run encodes its inputs (``neural/shared/framing.py``): every
boundary is computed from the rendered text and the tokenizer's offset
mapping, never from a column's serialized length or a template string this
package knows. A declared segment the rendered text does not contain is
``absent`` and one it contains twice is ``ambiguous`` (§2.3's cardinalities),
through the same path a ``variable`` takes.

**Optional and digest-neutral.** A document without a ``segments`` section is
plain text exactly as before — ``prefix_lengths`` is 0 for every row and no
byte of any existing document's canonical form moves, because the section is
copied through canonicalization only when authored and nothing here has a
materialized default (no ``frame: plain``, no ``segments: {}``).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Mapping

from causalab.protocol.errors import ParseError, ValidationError, suggest
from causalab.protocol.schema import PositionSpec
from causalab.protocol.spans import (
    SpanSpec,
    segment_anchors,
    selector,
    static_indices,
    walk,
)

if TYPE_CHECKING:
    from causalab.protocol.schema import Document

__all__ = [
    "CHAT_SEGMENTS",
    "CONTINUATION_SEGMENT",
    "SEGMENT_FRAMES",
    "SEGMENT_SOURCE_KEYS",
    "SegmentSource",
    "SegmentsSpec",
    "check",
    "parse_segments",
]

#: The ``frame`` vocabulary (§2.2.1). Absent is plain text — there is no
#: literal spelling of that default.
SEGMENT_FRAMES: tuple[str, ...] = ("chat",)
#: The chat frame's segment names, in reading order (§2.2.1). ``system`` is
#: declared only when the section gives it a source; ``continuation`` names
#: the greedy continuation (§2.3 ``generated``) in every frame.
CHAT_SEGMENTS: tuple[str, ...] = ("system", "user", "assistant_prefix", "continuation")
CONTINUATION_SEGMENT: str = "continuation"
#: How a declared segment is located (§2.2.1): from a dataset column — the
#: row's value for it, found in the row's rendered text.
SEGMENT_SOURCE_KEYS: tuple[str, ...] = ("column",)


@dataclasses.dataclass(frozen=True)
class SegmentSource:
    """Where a declared segment's text comes from: a top-level row column."""

    column: str


@dataclasses.dataclass(frozen=True)
class SegmentsSpec:
    """§2.2.1 — the parsed section. ``frame`` is ``None`` for plain text;
    ``system`` is the chat frame's optional system turn; ``declare`` names the
    column-sourced segments, in authored order."""

    frame: str | None = None
    system: SegmentSource | None = None
    declare: Mapping[str, SegmentSource] = dataclasses.field(default_factory=dict)

    def declared(self) -> tuple[str, ...]:
        """Every segment name a ``segment:`` anchor may use under this
        section: the chat names (``system`` only with a source) when the frame
        is ``chat``, ``continuation`` always, then the declared ones."""
        names: list[str] = []
        if self.frame == "chat":
            names.extend(
                name
                for name in CHAT_SEGMENTS
                if name != "system" or self.system is not None
            )
        else:
            names.append(CONTINUATION_SEGMENT)
        names.extend(name for name in self.declare if name not in names)
        return tuple(names)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParseError(
            "P2", f"expected an object, got {type(value).__name__}", path=path
        )
    return value


def _parse_source(raw: Any, path: str) -> SegmentSource:
    obj = _mapping(raw, path)
    for key in obj:
        if key not in SEGMENT_SOURCE_KEYS:
            raise ParseError(
                "P3",
                f"unknown key {key!r}{suggest(key, SEGMENT_SOURCE_KEYS)}",
                path=path,
            )
    if "column" not in obj or not isinstance(obj["column"], str) or not obj["column"]:
        raise ParseError(
            "P2",
            'a segment source is {"column": "<name>"} — the row column whose '
            "value the segment is located from",
            path=path,
        )
    return SegmentSource(column=obj["column"])


def parse_segments(raw: Any, path: str = "segments") -> SegmentsSpec:
    """Strict-parse the ``segments`` section. Shape only: that ``frame`` is
    in the vocabulary, that ``system`` needs the chat frame and that declared
    names are legal are rule 27's, so one rule names the field for every way
    the section can be wrong."""
    obj = _mapping(raw, path)
    allowed = ("frame", "system", "declare")
    for key in obj:
        if key not in allowed:
            raise ParseError(
                "P3", f"unknown key {key!r}{suggest(key, allowed)}", path=path
            )
    if not obj:
        raise ParseError(
            "P2",
            "an empty segments section declares nothing — name a frame or "
            "declare a segment, or omit the section (plain text is the "
            "absence of it)",
            path=path,
        )
    frame = None
    if "frame" in obj:
        if not isinstance(obj["frame"], str):
            raise ParseError("P2", "frame is a string", path=f"{path}.frame")
        frame = obj["frame"]
    system = _parse_source(obj["system"], f"{path}.system") if "system" in obj else None
    declare: dict[str, SegmentSource] = {}
    if "declare" in obj:
        table = _mapping(obj["declare"], f"{path}.declare")
        for name, source in table.items():
            if not isinstance(name, str) or not name:
                raise ParseError(
                    "P2", "a segment name is a non-empty string", path=f"{path}.declare"
                )
            declare[name] = _parse_source(source, f"{path}.declare.{name}")
    return SegmentsSpec(frame=frame, system=system, declare=declare)


# --------------------------------------------------------------------------- #
# rule 27 — a segment anchor names a declared segment; a span is well-formed
# --------------------------------------------------------------------------- #


def _authored_positions(doc: "Document") -> list[tuple[str, PositionSpec]]:
    out: list[tuple[str, PositionSpec]] = []
    for name, spec in doc.positions.items():
        if isinstance(spec, PositionSpec):
            out.append((f"positions.{name}", spec))
    for section, table in (("reads", doc.reads), ("writes", doc.writes)):
        for name, entry in table.items():
            if isinstance(entry.pos, PositionSpec):
                out.append((f"{section}.{name}.pos", entry.pos))
    return out


def check(doc: "Document") -> None:
    """Rule 27 — document-decidable facts about segments and spans (§2.2.1,
    §2.3), each named on its field.

    * ``segments.frame`` is in :data:`SEGMENT_FRAMES`; ``segments.system``
      needs the chat frame (there is no system turn in plain text); a
      declared name is not one of the chat frame's (those are the frame's to
      locate, and a column cannot be one of them);
    * every ``segment:`` anchor — a whole-segment selector, a ``scope`` or a
      ``relative_to`` — names a segment the section declares (no section, no
      segment anchors);
    * ``continuation`` is the greedy continuation, so an anchor on it carries
      ``generated`` (the decode budget lives on the position, §2.3) and a
      whole-``continuation`` span is spelled ``{"generated": …, "all": true}``;
    * an ``atomic`` span whose member set the document alone fixes has at
      least two members — one token is an ``index``, not a joint address.
    """
    segments = doc.segments
    declared: tuple[str, ...] = ()
    if segments is not None:
        if segments.frame is not None and segments.frame not in SEGMENT_FRAMES:
            raise ValidationError(
                27,
                f"frame {segments.frame!r} is not one of {list(SEGMENT_FRAMES)}"
                + suggest(segments.frame, SEGMENT_FRAMES)
                + " — plain text is the absence of a frame",
                path="segments.frame",
            )
        if segments.system is not None and segments.frame != "chat":
            raise ValidationError(
                27,
                "a system turn exists only in the chat frame: declare "
                '"frame": "chat", or drop "system"',
                path="segments.system",
            )
        for name in segments.declare:
            if name in CHAT_SEGMENTS:
                raise ValidationError(
                    27,
                    f"{name!r} is a chat-frame segment the frame locates itself; "
                    "a declared segment takes another name",
                    path=f"segments.declare.{name}",
                )
        declared = segments.declared()
    for path, spec in _authored_positions(doc):
        for inner, name in segment_anchors(spec):
            if segments is None:
                raise ValidationError(
                    27,
                    f"segment {name!r} is named but the document has no segments "
                    "section — declare the frame or the segment there (sec. 2.2.1)",
                    path=path,
                )
            if name not in declared:
                raise ValidationError(
                    27,
                    f"segment {name!r} is not declared; this section declares "
                    f"{list(declared)}" + suggest(name, declared),
                    path=path,
                )
            if name == CONTINUATION_SEGMENT:
                if isinstance(inner, SpanSpec) and inner.segment is not None:
                    raise ValidationError(
                        27,
                        "the whole continuation is spelled "
                        '{"generated": {"max_new_tokens": n}, "all": true} — the '
                        "frame carries its decode budget on the position (sec. 2.3)",
                        path=path,
                    )
                if spec.generated is None:
                    raise ValidationError(
                        27,
                        "an anchor inside the 'continuation' segment addresses "
                        "the greedy continuation, so the position carries "
                        '"generated": {"max_new_tokens": n} — the budget the '
                        "frame exists by (sec. 2.3)",
                        path=path,
                    )
        for inner in walk(spec):
            if not isinstance(inner, SpanSpec) or not inner.atomic:
                continue
            fixed = static_indices(inner)
            if fixed is not None and len(fixed) < 2:
                raise ValidationError(
                    27,
                    f"an atomic {selector(inner)} of {len(fixed)} member(s) is not "
                    "a joint address — one token is an 'index'; atomic needs two "
                    "or more (sec. 2.3)",
                    path=path,
                )
