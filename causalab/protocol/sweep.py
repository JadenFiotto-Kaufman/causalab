"""Sweep-axis discovery and deterministic expansion (spec §3).

A document with ``{"sweep": …}`` wrappers denotes one compiled intervention
per point: every axis is an explicit wrapper on a field (or whole entry) of
a named table, the axes cross-multiply, and expansion is a pure function of
the document — one document ⇒ the same ordered point list, always.

This module works on the **raw mapping** (the order-preserving tree
:func:`causalab.protocol.schema.load_raw` produces, after artifact
resolution), not on the parsed dataclasses: substituting one axis value is a
tree edit, and re-parsing each concrete point through
:func:`~causalab.protocol.schema.parse_document` afterwards guarantees a
point is exactly as valid as the same document written by hand.

Axis identity is name identity (§3): the axis id is the dotted path of the
wrapped field (``sites.target.layers``, ``positions.tap``,
``featurizers.rot.k``, ``train.seed``), axes are ordered by first appearance
in the document, and the cross product iterates the *last* axis fastest —
so point order is the lexicographic order of coordinate indices.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, Mapping, Sequence

from causalab.protocol.errors import ValidationError
from causalab.protocol.schema import dotted_path

__all__ = [
    "AT_ONCE_KEY",
    "SWEEP_KEY",
    "Axis",
    "Expansion",
    "Point",
    "coordinate_label",
    "axis_values",
    "expand",
    "find_axes",
    "label_value",
    "short_coords",
]


#: The wrapper that declares a sweep axis.
SWEEP_KEY = "sweep"

#: Its sibling (§3.1), which declares an axis *inside* one point rather than
#: across points. Named here rather than in :mod:`causalab.protocol.families`
#: so the nested-wrapper check below can see both without importing the module
#: that expands it — and so the two keywords are declared side by side, which
#: is where a reader comparing them will look.
AT_ONCE_KEY = "at_once"

#: Refuse a cross product larger than this without an explicit override
#: (§5.14 — "may be capped without an explicit override flag").
DEFAULT_POINT_CAP = 4096

#: Hard sanity bound on one axis's value count, checked BEFORE the range
#: sugar materializes — the point cap is overridable, this is not.
MAX_AXIS_VALUES = 1_000_000


@dataclasses.dataclass(frozen=True)
class Axis:
    """One sweep axis: the path of the wrapped value and its values.

    ``path`` addresses the wrapper's location in the raw tree
    (``("method", "sites", "target", "layers")``); ``id`` is its section-rooted
    dotted spelling (``sites.target.layers``, §1) — the coordinate key in
    results and derived names, and the spelling a workflow's ``emit``,
    ``order`` and plot axes use.
    """

    path: tuple[str, ...]
    values: tuple[Any, ...]

    @property
    def id(self) -> str:
        return dotted_path(self.path)


@dataclasses.dataclass(frozen=True)
class Point:
    """One expanded point: coordinates plus the concrete raw document."""

    coords: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Expansion:
    """The result of expanding one document: its axes (document order) and
    the full cross product (last axis fastest)."""

    axes: tuple[Axis, ...]
    points: tuple[Point, ...]

    @property
    def is_swept(self) -> bool:
        return bool(self.axes)


def _is_sweep_node(node: Any) -> bool:
    return isinstance(node, dict) and SWEEP_KEY in node


def axis_values(
    spec: Any,
    *,
    path: str,
    rule: int | str = 14,
    keyword: str = SWEEP_KEY,
    article: str = "a",
) -> tuple[Any, ...]:
    """The value list an axis wrapper's payload denotes (§3).

    A list is itself; a ``{"range": [start, stop, step?]}`` object is sugar for
    the half-open range it expands to. One grammar, one implementation, two
    callers: ``{"sweep": …}`` here and ``{"at_once": …}`` in
    :mod:`causalab.protocol.families`, which is what keeps the two keywords
    from drifting into two dialects of "a list of values". ``rule`` and
    ``keyword`` are what differ — the checklist rule each cites, and the word
    its messages use. ``article`` goes with the keyword: "a sweep wrapper", "an
    at_once wrapper", and a reader who wrote one keyword should never be told
    about the other in bad grammar.
    """
    if isinstance(spec, Mapping):
        if set(spec) != {"range"}:
            raise ValidationError(
                rule,
                f"{article} {keyword} object form takes exactly {{'range': [...]}}",
                path=path,
            )
        rng = spec["range"]
        if (
            not isinstance(rng, list)
            or not 2 <= len(rng) <= 3
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in rng)
        ):
            raise ValidationError(
                rule,
                f"{keyword} range must be [start, stop] or [start, stop, step] "
                "of integers",
                path=path,
            )
        step = rng[2] if len(rng) == 3 else 1
        if step == 0:
            raise ValidationError(
                rule, f"{keyword} range step must be non-zero", path=path
            )
        n = len(range(rng[0], rng[1], step))  # O(1) — no materialization yet
        if n > MAX_AXIS_VALUES:
            raise ValidationError(
                rule,
                f"{keyword} range denotes {n} values, over the per-axis bound of "
                f"{MAX_AXIS_VALUES} — refuse before materializing (§5.14)",
                path=path,
            )
        values: tuple[Any, ...] = tuple(range(rng[0], rng[1], step))
    elif isinstance(spec, list):
        values = tuple(spec)
    else:
        raise ValidationError(
            rule,
            f"{article} {keyword} wrapper takes a list or a range object, got "
            f"{type(spec).__name__}",
            path=path,
        )
    if not values:
        raise ValidationError(
            rule, f"{article} {keyword} axis must have at least one value", path=path
        )
    if _any_nested_wrapper(values):
        raise ValidationError(
            rule, f"{keyword} values may not contain nested axis wrappers", path=path
        )
    return values


def _sweep_values(node: Mapping[str, Any], path: tuple[str, ...]) -> tuple[Any, ...]:
    """The value list a sweep wrapper denotes (§3). Shape errors are §5.14
    rejections — the parser already checks the shapes it can see, but expansion
    may encounter wrappers in positions the schema types as free-form."""
    if len(node) != 1:
        raise ValidationError(
            14, "a sweep wrapper holds nothing but the axis", path=".".join(path)
        )
    return axis_values(node[SWEEP_KEY], path=".".join(path))


def _any_nested_wrapper(node: Any) -> bool:
    """Whether an axis wrapper of *either* kind sits inside ``node``.

    Both are refused inside an axis's values: nested sweeps because a value is
    a value, and an ``at_once`` because families expand *before* axes are
    found, so one surviving in here is one that had no name identity to expand
    against (§3.1).
    """
    if _is_sweep_node(node) or (isinstance(node, Mapping) and AT_ONCE_KEY in node):
        return True
    if isinstance(node, Mapping):
        return any(_any_nested_wrapper(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_any_nested_wrapper(v) for v in node)
    return False


def find_axes(raw: Mapping[str, Any]) -> tuple[Axis, ...]:
    """Every sweep axis in the document, in order of first appearance.

    Wrappers are only discovered under string keys (table entries and their
    fields) — a wrapper *inside a list* (e.g. inside an authored ``dims``
    list) has no name identity and is rejected.
    """
    axes: list[Axis] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if _is_sweep_node(node):
            axes.append(Axis(path=path, values=_sweep_values(node, path)))
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, path + (str(key),))
        elif isinstance(node, list):
            for item in node:
                if _any_nested_wrapper(item):
                    raise ValidationError(
                        14,
                        "a sweep wrapper inside a list has no name identity; "
                        "declare the axis on a named field",
                        path=".".join(path),
                    )

    walk(raw, ())
    return tuple(axes)


def _substitute(
    node: Any, assignment: Mapping[tuple[str, ...], Any], path: tuple[str, ...]
) -> Any:
    if _is_sweep_node(node):
        return assignment[path]
    if isinstance(node, Mapping):
        return {
            key: _substitute(value, assignment, path + (str(key),))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_substitute(item, assignment, path) for item in node]
    return node


def expand(
    raw: Mapping[str, Any], *, point_cap: int | None = DEFAULT_POINT_CAP
) -> Expansion:
    """Expand a raw document into its compiled interventions (§3).

    The cross product of all axes, last axis fastest; a document with no
    axes expands to exactly itself. ``point_cap`` refuses accidental
    combinatorial explosions (§5.14) — pass ``None`` for an explicit
    override.
    """
    axes = find_axes(raw)
    if not axes:
        return Expansion(axes=(), points=(Point(coords={}, raw=raw),))
    total = 1
    for axis in axes:
        total *= len(axis.values)
    if point_cap is not None and total > point_cap:
        raise ValidationError(
            14,
            f"sweep expands to {total} points, over the cap of {point_cap}; "
            "pass an explicit override to run a campaign this large",
        )
    points: list[Point] = []
    for combo in _cross(axes):
        assignment = {axis.path: value for axis, value in zip(axes, combo)}
        coords = {axis.id: value for axis, value in zip(axes, combo)}
        points.append(Point(coords=coords, raw=_substitute(raw, assignment, ())))
    return Expansion(axes=axes, points=tuple(points))


def _cross(axes: tuple[Axis, ...]) -> Iterator[tuple[Any, ...]]:
    if not axes:
        yield ()
        return
    head, *rest = axes
    for value in head.values:
        for tail in _cross(tuple(rest)):
            yield (value, *tail)


def short_coords(
    coords: Mapping[str, Any], *, entry: str | None = None
) -> dict[str, Any]:
    """Coordinates keyed by their **short** names — the names that appear in
    labels, in saved tensor keys, and therefore in an ``entry`` selector
    (:mod:`causalab.protocol.bundles`).

    Coordinates on the named entry itself drop the entry prefix
    (``k``, not ``featurizers.rot.k``); transitive coordinates keep
    ``<entry>.<field>``; the table name is always dropped. ``train`` axes
    read best bare (``seed``)."""
    out: dict[str, Any] = {}
    for axis_id, value in coords.items():
        segments = axis_id.split(".")
        if len(segments) >= 2 and segments[0] in (
            "positions",
            "sites",
            "featurizers",
            "params",
            "code",
            "reads",
            "writes",
            "intervened_models",
            "metrics",
            "train",
        ):
            segments = segments[1:]
        if entry is not None and len(segments) >= 2 and segments[0] == entry:
            segments = segments[1:]
        out[".".join(segments)] = value
    return out


def coordinate_label(coords: Mapping[str, Any], *, entry: str | None = None) -> str:
    """The ``[k=8]`` / ``[target.layers=5]`` suffix for derived names (§3),
    over :func:`short_coords`."""
    parts = [
        f"{name}={label_value(value)}"
        for name, value in short_coords(coords, entry=entry).items()
    ]
    return f"[{','.join(parts)}]" if parts else ""


def label_value(value: Any) -> str:
    """One coordinate value as it appears in a label — and therefore in a
    saved tensor key, which :mod:`causalab.protocol.bundles` matches
    against, so this rendering is part of the artifact contract."""
    if isinstance(value, Mapping):
        # a swept spec (e.g. a position): label by its single distinguishing pair
        pairs = ",".join(f"{k}:{v}" for k, v in value.items())
        return "{" + pairs + "}"
    if isinstance(value, (list, tuple)):
        return band_label(value)
    return str(value)


def band_label(layers: Sequence[Any]) -> str:
    """A layer band (§2.4 ``layers``) as it appears in a label: the one-layer
    band ``[3]`` is ``3`` — the scalar case is the scalar spelling, so a
    swept ``sites.target.layers`` labels ``[target.layers=3]`` exactly as the
    index does — a contiguous band ``[10, …, 19]`` is ``10..19``, and any
    other band joins its members with ``+`` (``10+12+15``). Neither the comma
    nor the brackets of the ``[k=8,seed=0]`` syntax appear, so a band label
    parses back like any other coordinate value."""
    members = [str(v) for v in layers]
    if len(members) == 1:
        return members[0]
    if all(isinstance(v, int) for v in layers) and list(layers) == list(
        range(layers[0], layers[0] + len(layers))
    ):
        return f"{layers[0]}..{layers[-1]}"
    return "+".join(members)
