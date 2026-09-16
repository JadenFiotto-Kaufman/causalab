"""Named axes — correlated row tuples and dependent axes (spec §3.2).

A sweep wrapper (§3) is one field, one axis: two wrapped fields are two axes
and their points are the cross product. Two experiments need more than that
and both are shapes ``sweep.py`` cannot spell without a second dialect:

* **correlated rows** — three candidate locations, each a (``layers``,
  ``component``, ``pos``) tuple whose fields move *together*: three rows, not
  the twenty-seven points three independent axes would give;
* **dependent axes** — ROME's clipped ten-layer restoration window centred at
  each of 48 layers: the window is a function of the centre, computed at
  load, clipped to the tower.

Both are declared once, by name, in an optional top-level ``axes`` group, and
referenced at the fields they move through an ``{"axis": "<name>"}`` or
``{"axis": "<name>.<field>"}`` wrapper — legal exactly where a ``{"sweep": …}``
wrapper is, so §3's first bullet stays true: you can see at the field that it
varies.

**Where this runs, and why no digest moves.** This is a compile stage of its
own — ``axes``, after ``families`` and before the shape ``gate``
(:data:`causalab.protocol.compile.STAGES`) — and it is sugar in the exact
sense ``families`` is: every stage after it sees a document the author could
have written by hand. The stage *lowers* the authored form to the **display
form** — every wrapper becomes ``{"sweep": [column]}`` and the block is
removed — which is what the gate parses and what ``CompiledPoints.explicit``
reports. The *expansion*, though, is not
the display form's cross product (that is the 3·3·3·2 the whole feature
exists to avoid): :func:`expand_axes` walks the named axes' entries in
declaration order, substitutes each entry's values at every referencing
wrapper, hands the concrete tree to :func:`~causalab.protocol.sweep.expand`
for the ordinary axes it still carries, and concatenates — so a named axis
is the slowest coordinate and the inner sweep keeps "last axis fastest". A
point reached this way is byte-for-byte the hand-written point (its tree is
that tree), so its canonical form and digest are that point's — the property
``canonical.py`` states for every point.

The *campaign* is a different matter: the display form reads as the cross
product, which is not what was authored, so the canonical document carries
the ``axes`` block when — and only when — one was authored
(:func:`canonical_axes`; ``canonicalize`` emits it between ``data`` and
``method``). Every document without the block keeps its canonical bytes.

This module is imported by ``compile.py`` alone (the ``paths.py`` shape): it
sits outside the hashed script closure (``tests/workflow/test_closure_census.py``
``SHARED``), so it moves no workflow pin; it edits no byte of ``sweep.py``,
``schema.py`` or ``errors.py``, and it adds no §5 rule — its refusals are the
parser's ``P2`` / ``P3`` / ``P4`` codes on paths under ``axes.<name>`` (or the
wrapper's own section-rooted path), and the point cap is rule 14 over the
**true** count, taken after correlated rows are applied.

What it refuses by design rather than by omission: a rule kind nothing here
knows (``routed_experts`` — "one intervention per routed expert selected for
that row") is ``P4`` with a message naming run-time fan-out. Routing is known
only after a forward, and expansion is a pure function of the document (§3);
that fan-out is the workflow's (``fan_out``), never a compile's.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Callable, Iterator, Mapping, Sequence

from causalab.protocol.errors import ParseError, ValidationError, suggest
from causalab.protocol.registry import ModelInfo
from causalab.protocol.schema import dotted_path
from causalab.protocol.sweep import (
    AT_ONCE_KEY,
    MAX_AXIS_VALUES,
    SWEEP_KEY,
    Axis,
    Expansion,
    Point,
    expand,
    find_axes,
)

__all__ = [
    "AXES_KEY",
    "AXIS_KEY",
    "AXIS_KINDS",
    "CLIP_TARGETS",
    "RULE_KINDS",
    "Axes",
    "NamedAxis",
    "canonical_axes",
    "expand_axes",
    "has_axes",
    "lower_axes",
    "parse_axes",
]


#: The optional fifth top-level group (§1, §3.2). Not in ``GROUP_ORDER``: the
#: stage strips it before the gate, which is what keeps ``schema.py`` untouched.
AXES_KEY = "axes"

#: The wrapper that references a named axis at a field — ``{"axis": "center"}``
#: or ``{"axis": "location.layers"}`` — legal wherever ``{"sweep": …}`` is.
AXIS_KEY = "axis"

#: The key that decides what kind of axis a declaration is — exactly one of
#: these per declaration. Closed; spec §3.2's ``key`` table is the census.
AXIS_KINDS: tuple[str, ...] = ("rows", "range", "values", "dependent_on")

#: The rules a dependent axis may be computed by. Closed; spec §3.2's ``kind``
#: table is the census. ``clipped_band`` is ROME's window: a ``width``-wide band
#: around the parent's value, clipped to the model's layer count.
RULE_KINDS: tuple[str, ...] = ("clipped_band",)

#: What a ``clipped_band`` may clip to. ``layers`` reads its bound from the
#: registry entry of the document's one ``model.key`` (``ModelInfo.num_layers``),
#: exactly as ``canonicalize`` bounds a site's layers.
CLIP_TARGETS: tuple[str, ...] = ("layers",)

#: Every key a declaration of each kind may carry.
_DECLARATION_KEYS: Mapping[str, tuple[str, ...]] = {
    "rows": ("rows", "key"),
    "range": ("range",),
    "values": ("values",),
    "dependent_on": ("dependent_on", "rule"),
}

_CLIPPED_BAND_KEYS: tuple[str, ...] = ("width", "clip_to")

#: The message the cap shares with ``sweep.expand`` — one wording, so a reader
#: who hit the cap through either path is told the same thing.
_CAP_MESSAGE = (
    "sweep expands to {total} points, over the cap of {cap}; "
    "pass an explicit override to run a campaign this large"
)


@dataclasses.dataclass(frozen=True)
class NamedAxis:
    """One declared axis.

    ``keys`` are its coordinate values, one per entry — a ``rows`` axis's
    ``key`` field (or the row index), a ``range`` / ``values`` axis's values;
    empty for a dependent axis, which records nothing (its value is in the
    point, as the parent's coordinate already names it). ``columns`` hold what
    a referencing wrapper receives per entry: keyed by field for a ``rows``
    axis, by ``None`` for the axis itself otherwise. ``parent`` is the axis a
    dependent one follows; ``declared`` is the declaration as authored."""

    name: str
    kind: str
    keys: tuple[Any, ...]
    columns: Mapping[str | None, tuple[Any, ...]]
    parent: str | None
    declared: Mapping[str, Any]

    @property
    def is_dependent(self) -> bool:
        return self.kind == "dependent_on"

    @property
    def size(self) -> int:
        return len(next(iter(self.columns.values())))


@dataclasses.dataclass(frozen=True)
class Axes:
    """The parsed ``axes`` group of one document, with the authored tree it
    was parsed from — the tree with the block and the wrappers still in
    place, which is what :func:`expand_axes` substitutes into.

    ``references`` are the wrappers, in first-appearance order: the tree path
    of each, the axis it names and the row field, if any."""

    source: Mapping[str, Any]
    named: tuple[NamedAxis, ...]
    references: tuple[tuple[tuple[str, ...], str, str | None], ...]

    @property
    def independent(self) -> tuple[NamedAxis, ...]:
        """The axes that are coordinates — every one that is not dependent —
        in declaration order, slowest first."""
        return tuple(axis for axis in self.named if not axis.is_dependent)

    def __getitem__(self, name: str) -> NamedAxis:
        for axis in self.named:
            if axis.name == name:
                return axis
        raise KeyError(name)


def has_axes(raw: Mapping[str, Any]) -> bool:
    """Whether ``raw`` declares an ``axes`` group — the cheap guard that keeps
    the stage off every document written before §3.2, which is every shipped
    one."""
    return AXES_KEY in raw


# --------------------------------------------------------------------------- #
# parsing the block
# --------------------------------------------------------------------------- #


def _refuse(code: str, message: str, path: str) -> ParseError:
    return ParseError(code, message, path=path)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _wrapper_inside(node: Any) -> bool:
    """Whether a ``sweep``, ``at_once`` or ``axis`` wrapper sits anywhere in
    ``node`` — none may: an entry's value is a value."""
    if isinstance(node, Mapping):
        if SWEEP_KEY in node or AT_ONCE_KEY in node or _is_axis_node(node):
            return True
        return any(_wrapper_inside(v) for v in node.values())
    if isinstance(node, list):
        return any(_wrapper_inside(v) for v in node)
    return False


def _fold(field: str, value: Any) -> Any:
    """A row's ``layers`` spelled as an index is the one-layer band ``[n]``
    (§2.4) — the same fold the parser and the canonical form make, applied
    here so the display form's column and the substituted point both carry
    the band spelling, and the point is exactly the hand-written one."""
    if field == "layers" and _is_int(value):
        return [value]
    return value


def parse_axes(raw: Mapping[str, Any], model_info: Callable[[str], ModelInfo]) -> Axes:
    """Parse the ``axes`` group of ``raw`` and find every wrapper that
    references it. ``model_info`` is the registry lookup a ``clipped_band``
    reads its bound from. Every refusal is a :class:`ParseError` on a path
    under ``axes.<name>`` or at the wrapper's own section-rooted path."""
    block = raw[AXES_KEY]
    if not isinstance(block, Mapping) or not block:
        raise _refuse(
            "P2",
            "'axes' is a non-empty object of named axes (§3.2)"
            + ("" if isinstance(block, Mapping) else f", got {_type_name(block)}"),
            AXES_KEY,
        )
    kinds: dict[str, str] = {}
    for name, decl in block.items():
        kinds[str(name)] = _kind_of(str(name), decl)
    named: dict[str, NamedAxis] = {}
    # independent axes first, so a dependent one may name a parent declared
    # after it — declaration order is still the coordinate order
    for name, kind in kinds.items():
        if kind != "dependent_on":
            named[name] = _parse_independent(name, kind, block[name])
    for name, kind in kinds.items():
        if kind == "dependent_on":
            named[name] = _parse_dependent(name, block[name], named, raw, model_info)
    ordered = tuple(named[name] for name in kinds)
    references = _find_references(raw, ordered)
    _check_everything_is_used(ordered, references)
    return Axes(source=raw, named=ordered, references=references)


def _type_name(value: Any) -> str:
    return type(value).__name__


def _kind_of(name: str, decl: Any) -> str:
    path = f"{AXES_KEY}.{name}"
    if not isinstance(decl, Mapping):
        raise _refuse(
            "P2",
            f"axis {name!r} is an object declaration, got {_type_name(decl)}",
            path,
        )
    present = [kind for kind in AXIS_KINDS if kind in decl]
    if len(present) != 1:
        raise _refuse(
            "P2",
            f"axis {name!r} is exactly one of {list(AXIS_KINDS)} — "
            + (
                f"it declares {present}"
                if present
                else f"it declares none (keys: {sorted(str(k) for k in decl)})"
            ),
            path,
        )
    kind = present[0]
    allowed = _DECLARATION_KEYS[kind]
    for key in decl:
        if key not in allowed:
            raise _refuse(
                "P3",
                f"unknown key {key!r} in the {kind} axis {name!r}"
                f"{suggest(str(key), allowed)}",
                f"{path}.{key}",
            )
    return kind


def _parse_independent(name: str, kind: str, decl: Mapping[str, Any]) -> NamedAxis:
    path = f"{AXES_KEY}.{name}"
    if kind == "rows":
        return _parse_rows(name, decl, path)
    if kind == "range":
        values = _range_values(decl["range"], f"{path}.range")
    else:
        values = _list_values(decl["values"], f"{path}.values")
    return NamedAxis(
        name=name,
        kind=kind,
        keys=values,
        columns={None: values},
        parent=None,
        declared=decl,
    )


def _parse_rows(name: str, decl: Mapping[str, Any], path: str) -> NamedAxis:
    rows = decl["rows"]
    if not isinstance(rows, list) or not rows:
        raise _refuse(
            "P2",
            f"'rows' of axis {name!r} is a non-empty list of row objects"
            + ("" if isinstance(rows, list) else f", got {_type_name(rows)}"),
            f"{path}.rows",
        )
    fields: tuple[str, ...] | None = None
    for i, row in enumerate(rows):
        where = f"{path}.rows[{i}]"
        if not isinstance(row, Mapping):
            raise _refuse(
                "P2",
                f"row {i} of axis {name!r} is an object, got {_type_name(row)}",
                where,
            )
        if not row:
            raise _refuse("P2", f"row {i} of axis {name!r} names no field", where)
        these = tuple(str(k) for k in row)
        if fields is None:
            fields = these
        elif set(these) != set(fields):
            missing = sorted(set(fields) - set(these))
            extra = sorted(set(these) - set(fields))
            raise _refuse(
                "P2",
                f"row {i} of axis {name!r} does not cover the fields row 0 "
                f"declares ({list(fields)}): "
                + ", ".join(
                    part
                    for part in (
                        f"missing {missing}" if missing else "",
                        f"extra {extra}" if extra else "",
                    )
                    if part
                ),
                where,
            )
        for field, value in row.items():
            if _wrapper_inside(value):
                raise _refuse(
                    "P2",
                    f"a row value is a value: no sweep, at_once or axis wrapper "
                    f"inside axis {name!r}'s rows (the row *is* the axis)",
                    f"{where}.{field}",
                )
    assert fields is not None
    key = decl.get("key")
    if "key" in decl:
        if not isinstance(key, str) or key not in fields:
            raise _refuse(
                "P2",
                f"'key' of axis {name!r} names one of its row fields "
                f"{list(fields)}"
                + (suggest(key, fields) if isinstance(key, str) else ""),
                f"{path}.key",
            )
        keys = tuple(row[key] for row in rows)
        for i, value in enumerate(keys):
            if not _is_scalar(value):
                raise _refuse(
                    "P2",
                    f"the key field {key!r} of axis {name!r} is a scalar per row — "
                    f"row {i} carries {_type_name(value)}; a coordinate is a scalar",
                    f"{path}.rows[{i}].{key}",
                )
        _check_distinct(keys, name, f"{path}.key", what=f"key field {key!r}")
    else:
        keys = tuple(range(len(rows)))
    columns: dict[str | None, tuple[Any, ...]] = {
        field: tuple(_fold(field, row[field]) for row in rows) for field in fields
    }
    _check_rows_distinct(columns, fields, name, path)
    return NamedAxis(
        name=name, kind="rows", keys=keys, columns=columns, parent=None, declared=decl
    )


def _check_rows_distinct(
    columns: Mapping[str | None, tuple[Any, ...]],
    fields: tuple[str, ...],
    name: str,
    path: str,
) -> None:
    """Rows are distinct, compared folded (so ``layers: 8`` and ``[8]`` are one
    row): two identical rows would enumerate as two points with one name, keyed
    or not — a keyed pair is already caught by its repeated key, a keyless pair
    only here."""
    seen: list[tuple[Any, ...]] = []
    for i in range(len(next(iter(columns.values())))):
        row = tuple(columns[field][i] for field in fields)
        if row in seen:
            raise _refuse(
                "P2",
                f"rows {seen.index(row)} and {i} of axis {name!r} are one row "
                f"({dict(zip(fields, row, strict=True))!r}): two identical rows "
                "would be two points with one name — rows are distinct",
                f"{path}.rows[{i}]",
            )
        seen.append(row)


def _check_distinct(values: Sequence[Any], name: str, path: str, *, what: str) -> None:
    seen: list[Any] = []
    for value in values:
        if value in seen:
            raise _refuse(
                "P2",
                f"the {what} of axis {name!r} repeats {value!r}: two entries with one "
                "coordinate would be two points with one name",
                path,
            )
        seen.append(value)


def _range_values(spec: Any, path: str) -> tuple[Any, ...]:
    if (
        not isinstance(spec, list)
        or not 2 <= len(spec) <= 3
        or not all(_is_int(v) for v in spec)
    ):
        raise _refuse(
            "P2", "'range' is [start, stop] or [start, stop, step] of integers", path
        )
    step = spec[2] if len(spec) == 3 else 1
    if step == 0:
        raise _refuse("P2", "'range' step must be non-zero", path)
    n = len(range(spec[0], spec[1], step))
    if n == 0:
        raise _refuse("P2", f"'range' {spec} denotes no value", path)
    if n > MAX_AXIS_VALUES:
        raise _refuse(
            "P2",
            f"'range' denotes {n} values, over the per-axis bound of "
            f"{MAX_AXIS_VALUES} — refused before materializing (§5.14)",
            path,
        )
    return tuple(range(spec[0], spec[1], step))


def _list_values(spec: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(spec, list) or not spec:
        raise _refuse(
            "P2",
            "'values' is a non-empty list of scalars"
            + ("" if isinstance(spec, list) else f", got {_type_name(spec)}"),
            path,
        )
    for i, value in enumerate(spec):
        if not _is_scalar(value):
            raise _refuse(
                "P2",
                f"a named axis's value is a scalar (it is the coordinate), got "
                f"{_type_name(value)}; a tuple of fields is a 'rows' axis",
                f"{path}[{i}]",
            )
    values = tuple(spec)
    _check_distinct(values, path.split(".")[1], path, what="values")
    return values


def _parse_dependent(
    name: str,
    decl: Mapping[str, Any],
    named: Mapping[str, NamedAxis],
    raw: Mapping[str, Any],
    model_info: Callable[[str], ModelInfo],
) -> NamedAxis:
    path = f"{AXES_KEY}.{name}"
    parent_name = decl["dependent_on"]
    if not isinstance(parent_name, str) or parent_name not in named:
        candidates = [n for n, axis in named.items() if axis.kind != "rows"]
        raise _refuse(
            "P2",
            f"axis {name!r} depends on {parent_name!r}, which is not a declared "
            f"range or values axis"
            + (
                suggest(parent_name, candidates) if isinstance(parent_name, str) else ""
            ),
            f"{path}.dependent_on",
        )
    parent = named[parent_name]
    if parent.kind == "rows":
        raise _refuse(
            "P2",
            f"axis {name!r} depends on the rows axis {parent_name!r}: a rule "
            "takes one scalar per entry, and a row's fields already move "
            "together — compute the value and write it as one more field",
            f"{path}.dependent_on",
        )
    if "rule" not in decl:
        raise _refuse(
            "P2",
            f"a dependent axis declares its 'rule' (one of {list(RULE_KINDS)})",
            path,
        )
    rule = decl["rule"]
    if not isinstance(rule, Mapping) or len(rule) != 1:
        raise _refuse(
            "P2",
            f"'rule' of axis {name!r} is an object with one key, the rule kind "
            f"({list(RULE_KINDS)})",
            f"{path}.rule",
        )
    kind = str(next(iter(rule)))
    if kind not in RULE_KINDS:
        raise _refuse(
            "P4",
            f"unknown axis rule {kind!r}; the rules are {list(RULE_KINDS)}"
            f"{suggest(kind, RULE_KINDS)}. A per-row expert, or a per-row "
            "causally-later site, is a run-time fan-out: routing is known only "
            "after a forward, and expansion is a pure function of the document "
            "(§3); declare it in the workflow (fan_out)",
            f"{path}.rule",
        )
    values = _clipped_band(
        name, rule[kind], parent, raw, model_info, f"{path}.rule.{kind}"
    )
    return NamedAxis(
        name=name,
        kind="dependent_on",
        keys=(),
        columns={None: values},
        parent=parent_name,
        declared=decl,
    )


def _clipped_band(
    name: str,
    spec: Any,
    parent: NamedAxis,
    raw: Mapping[str, Any],
    model_info: Callable[[str], ModelInfo],
    path: str,
) -> tuple[Any, ...]:
    """``{"width": w, "clip_to": "layers"}``: for a centre *c*, the band
    ``[max(0, c − ⌊w/2⌋) … min(L − 1, c + ⌈w/2⌉ − 1)]`` over the model's *L*
    layers — ROME's window, ten wide, so centre 0 is ``[0..4]`` and centre 47
    of 48 is ``[42..47]``."""
    if not isinstance(spec, Mapping):
        raise _refuse(
            "P2",
            f"'clipped_band' is an object {{width, clip_to}}, got {_type_name(spec)}",
            path,
        )
    for key in spec:
        if key not in _CLIPPED_BAND_KEYS:
            raise _refuse(
                "P3",
                f"unknown key {key!r} in clipped_band{suggest(str(key), _CLIPPED_BAND_KEYS)}",
                f"{path}.{key}",
            )
    for key in _CLIPPED_BAND_KEYS:
        if key not in spec:
            raise _refuse("P2", f"clipped_band declares {key!r}", f"{path}.{key}")
    width = spec["width"]
    if not _is_int(width) or width < 1:
        raise _refuse("P2", "'width' is a positive integer", f"{path}.width")
    clip_to = spec["clip_to"]
    if clip_to not in CLIP_TARGETS:
        raise _refuse(
            "P4",
            f"unknown clip target {clip_to!r}; the targets are {list(CLIP_TARGETS)}"
            + (suggest(clip_to, CLIP_TARGETS) if isinstance(clip_to, str) else ""),
            f"{path}.clip_to",
        )
    if not all(_is_int(v) for v in parent.keys):
        raise _refuse(
            "P2",
            f"clipped_band centres on the integer values of {parent.name!r}, "
            "which is not an integer axis",
            f"{path}.width",
        )
    model = raw.get("model")
    key = model.get("key") if isinstance(model, Mapping) else None
    if not isinstance(key, str):
        raise _refuse(
            "P2",
            f"clip_to 'layers' reads the layer count of one model, and "
            f"model.key is {'swept' if isinstance(key, Mapping) else 'not a string'}; "
            "clip against one model per document",
            f"{path}.clip_to",
        )
    num_layers = model_info(key).num_layers
    below, above = width // 2, width - width // 2
    bands: list[list[int]] = []
    for centre in parent.keys:
        lo = max(0, centre - below)
        hi = min(num_layers - 1, centre + above - 1)
        bands.append(list(range(lo, hi + 1)))
    assert all(bands), "a clipped band is never empty: the centre is inside the tower"
    return tuple(bands)


# --------------------------------------------------------------------------- #
# the wrappers
# --------------------------------------------------------------------------- #


def _is_axis_node(node: Any) -> bool:
    """A wrapper is a mapping holding ``axis`` and nothing else — the shape
    :func:`canonical._is_sweep` reads a sweep wrapper by. A mapping that
    carries ``axis`` beside other keys is a value (a ``gaussian`` write's
    payload, sec. 2.8, has one), so the walk needs no knowledge of the method
    vocabulary; a wrapper that meant to reference an axis and mis-spelled a
    key leaves that axis unreferenced, which is refused by name."""
    return isinstance(node, Mapping) and set(node) == {AXIS_KEY}


def _find_references(
    raw: Mapping[str, Any], named: tuple[NamedAxis, ...]
) -> tuple[tuple[tuple[str, ...], str, str | None], ...]:
    """Every ``{"axis": …}`` wrapper outside the block, in first-appearance
    order, resolved against the declarations. Like sweep wrappers, one inside
    a list has no name identity and is refused."""
    by_name = {axis.name: axis for axis in named}
    found: list[tuple[tuple[str, ...], str, str | None]] = []

    def walk(node: Any, path: tuple[str, ...], in_list: bool) -> None:
        if _is_axis_node(node):
            where = dotted_path(path)
            if in_list:
                raise _refuse(
                    "P2",
                    "an axis wrapper inside a list has no name identity; "
                    "reference the axis on a named field",
                    where,
                )
            found.append(_resolve_reference(node, where, by_name) + (path,))
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, path + (str(key),), in_list)
        elif isinstance(node, list):
            for item in node:
                walk(item, path, True)

    for key, value in raw.items():
        if key != AXES_KEY:
            walk(value, (str(key),), False)
    return tuple((path, name, field) for name, field, path in found)


def _resolve_reference(
    node: Mapping[str, Any], where: str, by_name: Mapping[str, NamedAxis]
) -> tuple[str, str | None]:
    ref = node[AXIS_KEY]
    if not isinstance(ref, str) or not ref:
        raise _refuse(
            "P2",
            f"'axis' names a declared axis, '<name>' or '<name>.<field>', got "
            f"{_type_name(ref)}",
            where,
        )
    name, _, field = ref.partition(".")
    axis = by_name.get(name)
    if axis is None:
        raise _refuse(
            "P2",
            f"{ref!r} names no declared axis (declared: {sorted(by_name)})"
            f"{suggest(name, by_name)}",
            where,
        )
    if axis.kind == "rows":
        fields = [str(f) for f in axis.columns if f is not None]
        if not field:
            raise _refuse(
                "P2",
                f"{name!r} is a rows axis and is referenced by field: "
                f"'{name}.<field>' with a field from {fields}",
                where,
            )
        if field not in axis.columns:
            raise _refuse(
                "P2",
                f"{ref!r}: axis {name!r} has no row field {field!r} "
                f"(fields: {fields}){suggest(field, fields)}",
                where,
            )
        return name, field
    if field:
        raise _refuse(
            "P2",
            f"{ref!r}: axis {name!r} is a {axis.kind} axis and has no fields; "
            f'write {{"axis": "{name}"}}',
            where,
        )
    return name, None


def _check_everything_is_used(
    named: tuple[NamedAxis, ...],
    references: tuple[tuple[tuple[str, ...], str, str | None], ...],
) -> None:
    """A declared axis nothing references, or a row field no wrapper reads,
    is a typo that would otherwise multiply points silently or drop a field
    the author meant to move. The ``key`` field may go unreferenced: it may
    be a label."""
    referenced = {(name, field) for _path, name, field in references}
    parents = {axis.parent for axis in named if axis.parent is not None}
    for axis in named:
        path = f"{AXES_KEY}.{axis.name}"
        if axis.kind == "rows":
            fields = [str(f) for f in axis.columns if f is not None]
            if not any((axis.name, f) in referenced for f in fields):
                raise _refuse(
                    "P2",
                    f"axis {axis.name!r} is declared but nothing references it "
                    f'({{"axis": "{axis.name}.<field>"}} on a field)',
                    path,
                )
            key = axis.declared.get("key")
            for field in fields:
                if field != key and (axis.name, field) not in referenced:
                    raise _refuse(
                        "P2",
                        f"row field {field!r} of axis {axis.name!r} is referenced "
                        f'nowhere ({{"axis": "{axis.name}.{field}"}} on the '
                        "field it moves) — or name it as the row 'key'",
                        f"{path}.rows[0].{field}",
                    )
        elif (axis.name, None) not in referenced and axis.name not in parents:
            raise _refuse(
                "P2",
                f"axis {axis.name!r} is declared but nothing references it "
                f'({{"axis": "{axis.name}"}} on a field, or a dependent axis)',
                path,
            )


# --------------------------------------------------------------------------- #
# lowering, expansion, the canonical block
# --------------------------------------------------------------------------- #


def lower_axes(raw: Mapping[str, Any], axes: Axes) -> dict[str, Any]:
    """The **display form**: ``raw`` with every wrapper replaced by the
    ``{"sweep": [column]}`` it stands for and the ``axes`` group removed — a
    document the shape gate parses as it parses any swept one. Its cross
    product is *not* the expansion (:func:`expand_axes` is); it is what
    ``CompiledPoints.explicit`` reports."""
    columns = {
        path: list(axes[name].columns[field]) for path, name, field in axes.references
    }
    lowered = _substitute(
        raw, {path: {SWEEP_KEY: column} for path, column in columns.items()}, ()
    )
    assert isinstance(lowered, dict)
    del lowered[AXES_KEY]
    return lowered


def _substitute(
    node: Any, assignment: Mapping[tuple[str, ...], Any], path: tuple[str, ...]
) -> Any:
    if _is_axis_node(node):
        return assignment[path]
    if isinstance(node, Mapping):
        return {
            str(key): _substitute(value, assignment, path + (str(key),))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_substitute(item, assignment, path) for item in node]
    return node


def _entries(axes: Axes) -> Iterator[tuple[int, ...]]:
    """Every combination of entry indices over the independent axes, the
    first declared slowest — the order the points come out in."""
    return itertools.product(*(range(axis.size) for axis in axes.independent))


def _row_tree(axes: Axes, combo: tuple[int, ...]) -> dict[str, Any]:
    """The concrete tree one combination denotes: every wrapper replaced by
    its entry's value, the block removed — the hand-written document."""
    index = {axis.name: i for axis, i in zip(axes.independent, combo, strict=True)}
    assignment: dict[tuple[str, ...], Any] = {}
    for path, name, field in axes.references:
        axis = axes[name]
        at = index[axis.parent] if axis.is_dependent else index[name]
        assert axis.parent is None or axis.parent in index
        assignment[path] = axis.columns[field][at]
    tree = _substitute(axes.source, assignment, ())
    assert isinstance(tree, dict)
    del tree[AXES_KEY]
    return tree


def expand_axes(axes: Axes, *, point_cap: int | None) -> Expansion:
    """Expand a document with named axes (§3.2).

    For each combination of entries over the independent named axes (first
    declared slowest), the concrete tree is built and handed to
    :func:`~causalab.protocol.sweep.expand` for the ordinary sweep axes it
    still carries; the points are concatenated in that order, each prefixed
    with one coordinate per named axis (``axes.<name>``: the entry's key).
    A dependent axis and a row's substituted fields record no coordinate of
    their own. The point cap (§5.14) is checked once, over the true count —
    rows × the inner cross product — so a correlated axis is counted as the
    rows it is, not as the cross product it replaced.
    """
    combos = list(_entries(axes))
    first = _row_tree(axes, combos[0])
    inner_axes = find_axes(first)
    inner_count = 1
    for axis in inner_axes:
        inner_count *= len(axis.values)
    total = len(combos) * inner_count
    if point_cap is not None and total > point_cap:
        raise ValidationError(14, _CAP_MESSAGE.format(total=total, cap=point_cap))
    named = tuple(
        Axis(path=(AXES_KEY, axis.name), values=axis.keys) for axis in axes.independent
    )
    points: list[Point] = []
    for combo in combos:
        inner = expand(_row_tree(axes, combo), point_cap=None)
        prefix = {
            named_axis.id: named_axis.values[i]
            for named_axis, i in zip(named, combo, strict=True)
        }
        for point in inner.points:
            points.append(Point(coords={**prefix, **point.coords}, raw=point.raw))
    return Expansion(axes=(*named, *inner_axes), points=tuple(points))


def canonical_axes(axes: Axes) -> dict[str, Any]:
    """The ``axes`` block as the canonical document carries it (§7): rows
    recorded folded — each field takes the value its display column received
    through :func:`_fold`, in row 0's field order, so ``layers: 8`` and
    ``layers: [8]`` are one row and one digest — with ``key`` when authored, a
    ``range`` materialized to its values, a dependent axis with its rule as
    authored and the entries it computed — so two spellings of one campaign
    are one digest, and the campaign never reads as the display form's cross
    product."""
    out: dict[str, Any] = {}
    for axis in axes.named:
        if axis.kind == "rows":
            fields = [field for field in axis.columns if field is not None]
            entry: dict[str, Any] = {
                "rows": [
                    {field: axis.columns[field][i] for field in fields}
                    for i in range(axis.size)
                ]
            }
            if "key" in axis.declared:
                entry["key"] = axis.declared["key"]
        elif axis.is_dependent:
            entry = {
                "dependent_on": axis.parent,
                "rule": axis.declared["rule"],
                "values": list(axis.columns[None]),
            }
        else:
            entry = {"values": list(axis.keys)}
        out[axis.name] = entry
    return out
