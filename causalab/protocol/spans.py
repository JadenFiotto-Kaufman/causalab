"""Semantic spans (spec §2.3): the span algebra over the position anchors.

A :class:`~causalab.protocol.schema.PositionSpec` names **one** anchor —
``index`` / ``span`` / ``variable`` / ``column`` / ``all`` — and resolves to
one run of tokens per row. A :class:`SpanSpec` is accepted wherever a position
spec is (``positions.<name>``, an inline ``pos``) and adds the forms an author
otherwise has to hold by hand: a **noncontiguous set** (``indices``), a whole **segment**
(``segment``, the ``segments`` section's names), **unions** and
**intersections** of member specs, and the **relative predicates** ``before`` /
``after`` / ``between`` over any prompt-frame anchor. ``atomic: true`` makes
the resolved set **one address**: a two-token number written as one atomic
span is *one* write for rule 8's "≤ 1 absolute write per (site, overlapping
pos, model)" and one joint ``one_to_one`` run for §2.3's alignment
cardinality; the same two tokens as two ``index`` specs — or as a non-atomic
``indices`` set — are two constituent locations, each classified on its own.

**Torch-free by construction.** The algebra here (:func:`resolve_span`) is a
pure function of a row's *frame* — the row's first real token, its content
start and the padded length — plus two callbacks the engine supplies: how to
resolve a member spec and how to locate a named segment. The engine's
``encoding.resolve_position`` is the one caller; the load-time checks
(:func:`static_indices`, :func:`constituents`, :func:`walk`) read the document
alone, which is what keeps them behind the encode-time boundary (no tokenizer in
``ResolutionEnv``, §2.3).

**Nothing about cardinality is decided here.** A span's observed alignment is
:func:`~causalab.protocol.alignment.alignment_of` over the candidate runs the
engine hands it — one run per side for an atomic span, one run per constituent
otherwise — and its static half is ``plan.static_alignment``. This module
spells no member of that vocabulary and calls none of its functions.

Names published here are frozen (the workflow layer reads them): the keys in
:data:`SPAN_KEYS`, and the shape each takes, are listed in §2.3's span table.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterator, Mapping

from causalab.protocol.errors import ParseError, ProtocolError
from causalab.protocol.schema import PositionSpec, concrete_int

__all__ = [
    "COMPOSITE_KEYS",
    "PREDICATE_KEYS",
    "SPAN_KEYS",
    "SpanSpec",
    "constituents",
    "is_span_object",
    "parse_span_spec",
    "plain",
    "resolve_span",
    "segment_anchors",
    "selector",
    "static_indices",
    "walk",
]

#: The keys whose presence makes a position object a :class:`SpanSpec` (§2.3
#: span table). ``atomic`` alone promotes an ordinary ``span`` / ``variable`` /
#: ``column`` anchor into one joint address.
SPAN_KEYS: tuple[str, ...] = (
    "segment",
    "indices",
    "union",
    "intersection",
    "before",
    "after",
    "between",
    "atomic",
)
#: The two set-composition selectors, each over ≥ 2 member specs.
COMPOSITE_KEYS: tuple[str, ...] = ("union", "intersection")
#: The three relative predicates, each over one (``between``: two) anchor spec.
PREDICATE_KEYS: tuple[str, ...] = ("before", "after", "between")
#: Every selector a span spec may carry — exactly one per spec.
_SELECTORS: tuple[str, ...] = (
    "segment",
    "indices",
    "union",
    "intersection",
    "before",
    "after",
    "between",
    "span",
    "variable",
    "column",
)
#: The anchors that may take ``scope`` / ``relative_to``.
_SCOPABLE: frozenset[str] = frozenset({"indices", "span"})


@dataclasses.dataclass(frozen=True)
class SpanSpec(PositionSpec):
    """§2.3 — a span: one of the selectors below, optionally ``atomic``.

    Exactly one selector is set: ``segment`` (every token of a declared
    segment), ``indices`` (a set of content-frame indices, or of indices
    inside ``scope``'s anchor), ``union`` / ``intersection`` (over ≥ 2 member
    specs), ``before`` / ``after`` / ``between`` (every real token of the row
    strictly before / after / between the anchor runs), or the inherited
    ``span`` / ``variable`` / ``column`` with ``atomic: true``. ``atomic`` is
    ``False`` unless authored ``true`` — there is no literal spelling of the
    default, so no existing document's canonical form moves. A span addresses
    the prompt frame: ``generated`` is refused on it.
    """

    segment: str | None = None
    indices: tuple[int, ...] | None = None
    union: tuple[PositionSpec, ...] | None = None
    intersection: tuple[PositionSpec, ...] | None = None
    before: PositionSpec | None = None
    after: PositionSpec | None = None
    between: tuple[PositionSpec, PositionSpec] | None = None
    atomic: bool = False


def is_span_object(obj: Mapping[str, Any]) -> bool:
    """Whether a raw position object spells a span (any key of
    :data:`SPAN_KEYS`) — the parse dispatch ``schema._parse_position_spec``
    takes."""
    return any(key in obj for key in SPAN_KEYS)


def selector(spec: SpanSpec) -> str:
    """Which selector a span spec carries."""
    for key in _SELECTORS:
        if getattr(spec, key) is not None:
            return key
    raise AssertionError(f"span spec {spec!r} carries no selector")


def plain(spec: SpanSpec) -> PositionSpec:
    """The ordinary position spec under an ``atomic`` ``span`` / ``variable``
    / ``column`` — the same anchor, resolved by the same code path, then made
    one address by the caller."""
    return PositionSpec(
        span=spec.span,
        variable=spec.variable,
        column=spec.column,
        scope=spec.scope,
        relative_to=spec.relative_to,
        anchor_source=spec.anchor_source,
    )


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

_ParsePosition = Callable[[Any, str], PositionSpec]


def parse_span_spec(
    obj: Mapping[str, Any],
    path: str,
    *,
    parse_position: _ParsePosition,
    parse_anchor_ref: Callable[[Any, str], tuple[str, str]],
) -> SpanSpec:
    """Parse one span object (a mapping with at least one of
    :data:`SPAN_KEYS`). ``parse_position`` is the schema's position parser,
    used for members and anchors so a member may itself be a span; the
    callbacks keep this module importable without the schema importing it.

    Shape is decided here (``P2``): one selector, list shapes, member forms.
    What needs the rest of the document — that a ``segment`` is declared, that
    an ``atomic`` set has two members — is rule 27's (``segments.check``).
    """
    allowed = (
        *SPAN_KEYS,
        "span",
        "variable",
        "column",
        "scope",
        "relative_to",
        "alignment",
    )
    for key in obj:
        if key not in allowed:
            if key in ("index", "all"):
                raise ParseError(
                    "P2",
                    f"{key!r} does not combine with a span key: an index is one "
                    "token and 'all' is every token, and neither has a set to "
                    "compose — spell the set with 'indices' (§2.3)",
                    path=path,
                )
            if key == "generated":
                raise ParseError(
                    "P2",
                    "a span addresses the prompt frame; 'generated' does not "
                    "combine with span keys in v1 — address the continuation "
                    "with index/span/variable/all inside 'generated' (§2.3)",
                    path=path,
                )
            raise ParseError("P3", f"unknown key {key!r} in a span spec", path=path)
    selectors = [key for key in _SELECTORS if key in obj]
    if len(selectors) != 1:
        raise ParseError(
            "P2",
            f"a span spec needs exactly one of {list(_SELECTORS)}, got {selectors}",
            path=path,
        )
    which = selectors[0]
    atomic = False
    if "atomic" in obj:
        if obj["atomic"] is not True:
            raise ParseError(
                "P2",
                f'atomic is the flag {{"atomic": true}} — got {obj["atomic"]!r}; '
                "an unauthored span is its constituent locations, and there is "
                "no literal spelling of that default",
                path=f"{path}.atomic",
            )
        atomic = True
    if which in ("span", "variable", "column") and not atomic:
        raise ParseError(
            "P2",
            f"a bare {which!r} is an ordinary position; the only span key that "
            f"combines with it is 'atomic': true",
            path=path,
        )
    if ("scope" in obj or "relative_to" in obj) and which not in _SCOPABLE:
        raise ParseError(
            "P2",
            f"scope/relative_to modify 'indices' or an atomic 'span', not {which!r}",
            path=path,
        )
    if "scope" in obj and "relative_to" in obj:
        raise ParseError(
            "P2", "scope and relative_to are mutually exclusive", path=path
        )

    def member(raw: Any, where: str) -> PositionSpec:
        if not isinstance(raw, dict):
            raise ParseError(
                "P2",
                "a span member or anchor is a position object, spelled out — "
                'no int or "all" sugar inside a span (got '
                f"{type(raw).__name__})",
                path=where,
            )
        if "alignment" in raw:
            raise ParseError(
                "P2",
                "a member carries no alignment — declare it on the span that "
                "composes the members (§2.3)",
                path=f"{where}.alignment",
            )
        if "generated" in raw:
            raise ParseError(
                "P2",
                "a member addresses the prompt frame; 'generated' cannot appear "
                "inside a span (§2.3)",
                path=f"{where}.generated",
            )
        return parse_position(raw, where)

    def members(raw: Any, where: str, *, exactly: int | None = None) -> tuple:
        if not isinstance(raw, list):
            raise ParseError("P2", "expected a list of position objects", path=where)
        if exactly is not None and len(raw) != exactly:
            raise ParseError(
                "P2", f"expected exactly {exactly} position objects", path=where
            )
        if exactly is None and len(raw) < 2:
            raise ParseError(
                "P2",
                f"a {which} composes two or more members; one member is the "
                "member itself",
                path=where,
            )
        return tuple(member(item, f"{where}[{i}]") for i, item in enumerate(raw))

    segment = None
    indices = None
    union = intersection = None
    before = after = None
    between = None
    span = variable = column = None
    if which == "segment":
        if not isinstance(obj["segment"], str) or not obj["segment"]:
            raise ParseError(
                "P2", "segment names a declared segment", path=f"{path}.segment"
            )
        segment = obj["segment"]
    elif which == "indices":
        raw_indices = obj["indices"]
        if (
            not isinstance(raw_indices, list)
            or not raw_indices
            or not all(
                isinstance(v, int) and not isinstance(v, bool) for v in raw_indices
            )
        ):
            raise ParseError(
                "P2", "indices is a non-empty list of integers", path=f"{path}.indices"
            )
        if len(set(raw_indices)) != len(raw_indices):
            raise ParseError(
                "P2",
                f"indices repeats a member: {raw_indices}",
                path=f"{path}.indices",
            )
        indices = tuple(raw_indices)
    elif which in COMPOSITE_KEYS:
        parsed = members(obj[which], f"{path}.{which}")
        if which == "union":
            union = parsed
        else:
            intersection = parsed
    elif which == "between":
        first, second = members(obj["between"], f"{path}.between", exactly=2)
        between = (first, second)
    elif which in ("before", "after"):
        anchor = member(obj[which], f"{path}.{which}")
        if which == "before":
            before = anchor
        else:
            after = anchor
    elif which == "span":
        raw_span = obj["span"]
        if (
            not isinstance(raw_span, list)
            or len(raw_span) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in raw_span)
        ):
            raise ParseError(
                "P2", "a span is [a, b) — exactly two integers", path=f"{path}.span"
            )
        span = (raw_span[0], raw_span[1])
    elif which == "variable":
        if not isinstance(obj["variable"], str):
            raise ParseError("P2", "expected a string", path=f"{path}.variable")
        variable = obj["variable"]
    else:
        if not isinstance(obj["column"], str):
            raise ParseError("P2", "expected a string", path=f"{path}.column")
        column = obj["column"]

    scope_ref = (
        parse_anchor_ref(obj["scope"], f"{path}.scope") if "scope" in obj else None
    )
    relative_ref = (
        parse_anchor_ref(obj["relative_to"], f"{path}.relative_to")
        if "relative_to" in obj
        else None
    )
    anchor_ref = scope_ref if scope_ref is not None else relative_ref
    alignment = None
    if "alignment" in obj:
        if not isinstance(obj["alignment"], str):
            raise ParseError("P2", "expected a string", path=f"{path}.alignment")
        alignment = obj["alignment"]
    return SpanSpec(
        span=span,
        variable=variable,
        column=column,
        scope=scope_ref[1] if scope_ref is not None else None,
        relative_to=relative_ref[1] if relative_ref is not None else None,
        anchor_source=anchor_ref[0] if anchor_ref is not None else "variable",
        alignment=alignment,
        segment=segment,
        indices=indices,
        union=union,
        intersection=intersection,
        before=before,
        after=after,
        between=between,
        atomic=atomic,
    )


# --------------------------------------------------------------------------- #
# what the document alone decides
# --------------------------------------------------------------------------- #


def walk(spec: PositionSpec) -> Iterator[PositionSpec]:
    """``spec`` and every spec nested inside it (members, anchors), depth
    first — what a load-time check iterates to see every anchor a position
    reaches."""
    yield spec
    if not isinstance(spec, SpanSpec):
        return
    nested: list[PositionSpec] = []
    nested.extend(spec.union or ())
    nested.extend(spec.intersection or ())
    if spec.before is not None:
        nested.append(spec.before)
    if spec.after is not None:
        nested.append(spec.after)
    nested.extend(spec.between or ())
    for inner in nested:
        yield from walk(inner)


def segment_anchors(spec: PositionSpec) -> Iterator[tuple[PositionSpec, str]]:
    """Every ``(spec, segment name)`` a position reaches: a whole-segment
    selector, or a ``scope`` / ``relative_to`` spelled as a segment."""
    for inner in walk(spec):
        if isinstance(inner, SpanSpec) and inner.segment is not None:
            yield inner, inner.segment
        anchor = inner.scope if inner.scope is not None else inner.relative_to
        if inner.anchor_source == "segment" and isinstance(anchor, str):
            yield inner, anchor


def static_indices(spec: PositionSpec) -> tuple[int, ...] | None:
    """The content-frame index set a spec addresses **by construction**, or
    ``None`` when only the tokenizer can say (an anchored or text-located
    selector). Sorted, distinct. An unscoped ``index`` / ``span`` /
    ``indices`` is static; a union or intersection of static members is."""
    if spec.scope is not None or spec.relative_to is not None:
        return None
    if isinstance(spec, SpanSpec):
        if spec.indices is not None:
            return tuple(sorted(set(spec.indices)))
        if spec.span is not None and isinstance(spec.span, tuple):
            lo, hi = (int(v) for v in spec.span)
            return tuple(range(lo, hi))
        if spec.union is not None or spec.intersection is not None:
            members = spec.union if spec.union is not None else spec.intersection
            assert members is not None
            sets = [static_indices(m) for m in members]
            if any(s is None for s in sets):
                return None
            first, *rest = (set(s) for s in sets if s is not None)
            for other in rest:
                first = first | other if spec.union is not None else first & other
            return tuple(sorted(first))
        return None
    if spec.generated is not None:
        return None
    if spec.index is not None and isinstance(spec.index, int):
        return (spec.index,)
    if spec.span is not None and isinstance(spec.span, tuple):
        lo, hi = (int(v) for v in spec.span)
        return tuple(range(lo, hi))
    return None


def constituents(spec: PositionSpec) -> tuple[PositionSpec, ...]:
    """The addresses a spec is classified as, for §2.3's cardinality: an
    ``atomic`` span is one; a non-atomic ``indices`` set is one per index
    (each a single-token ``index`` in the same frame); a non-atomic ``union``
    is its members; everything else is itself."""
    if not isinstance(spec, SpanSpec) or spec.atomic:
        return (spec,)
    if spec.indices is not None:
        return tuple(
            PositionSpec(
                index=n,
                scope=spec.scope,
                relative_to=spec.relative_to,
                anchor_source=spec.anchor_source,
            )
            for n in spec.indices
        )
    if spec.union is not None:
        return tuple(spec.union)
    return (spec,)


# --------------------------------------------------------------------------- #
# resolution — a pure function over one row's frame
# --------------------------------------------------------------------------- #


def resolve_span(
    spec: SpanSpec,
    *,
    frame: tuple[int, int, int],
    resolve: Callable[[PositionSpec], list[int]],
    segment_run: Callable[[str], list[int]],
    where: str = "",
) -> list[int]:
    """Resolve one span for one row into sorted, distinct padded-frame indices.

    ``frame`` is ``(first_real, content_start, padded_len)``: the row's first
    real token (past the left padding), its content start (past any chat
    prefix, §2.3) and the padded length. ``indices`` counts like ``index``
    does — ``n ≥ 0`` from the content start, ``n < 0`` from the end — while
    the relative predicates range over the row's **real** tokens, chat prefix
    included, because "before the user turn" is exactly the prefix. Members
    and anchors resolve through ``resolve`` (the engine's position resolver,
    so a member may be any prompt-frame spec, spans included) and a named
    segment through ``segment_run``.

    A span that resolves to **no** token is refused with reason
    ``empty_selector`` rather than gathering nothing: ``between`` two runs
    that touch, or an intersection of disjoint members, is an authoring fact
    the row exposes, and a silent empty read would report a number over it.
    """
    first_real, start, padded = frame
    which = selector(spec)
    label = f"span {where}" if where else "span"
    out: list[int]
    if which == "segment":
        assert spec.segment is not None
        out = list(segment_run(spec.segment))
    elif which == "indices":
        assert spec.indices is not None
        if spec.scope is None and spec.relative_to is None:
            out = [padded + n if n < 0 else start + n for n in spec.indices]
        else:
            out = [
                token
                for n in spec.indices
                for token in resolve(
                    PositionSpec(
                        index=n,
                        scope=spec.scope,
                        relative_to=spec.relative_to,
                        anchor_source=spec.anchor_source,
                    )
                )
            ]
    elif which == "union":
        assert spec.union is not None
        found: set[int] = set()
        for member in spec.union:
            found.update(resolve(member))
        out = sorted(found)
    elif which == "intersection":
        assert spec.intersection is not None
        runs = [set(resolve(member)) for member in spec.intersection]
        out = sorted(set.intersection(*runs)) if runs else []
    elif which == "before":
        assert spec.before is not None
        anchor = resolve(spec.before)
        out = list(range(first_real, min(anchor))) if anchor else []
    elif which == "after":
        assert spec.after is not None
        anchor = resolve(spec.after)
        out = list(range(max(anchor) + 1, padded)) if anchor else []
    elif which == "between":
        assert spec.between is not None
        one, two = (resolve(anchor) for anchor in spec.between)
        if one and two:
            earlier, later = (one, two) if min(one) <= min(two) else (two, one)
            out = list(range(max(earlier) + 1, min(later)))
        else:
            out = []
    else:  # an atomic span / variable / column: the plain anchor, made one
        out = list(resolve(plain(spec)))
    out = sorted(set(int(concrete_int(i, "resolved position")) for i in out))
    if not out:
        raise ProtocolError(
            "P2",
            f"{label} ({which}) resolves to no token in this row — a span that "
            "selects nothing is refused rather than gathered silently (§2.3)",
            reason="empty_selector",
        )
    return out
