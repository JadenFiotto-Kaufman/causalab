"""``at_once`` families: one authored entry, N entries in one forward (spec §3.1).

A *band* patches several layers during the **same** forward pass. The sweep
language cannot express that and should not: a sweep axis denotes independent
points (§3), and the whole point of a band is that its writes are in force
together. `intervened_models` already says which writes those are — what was
missing was a way to *declare* the per-layer tables without writing one entry
per layer by hand, which is why the shipped `attention_band_patch` preset
carried 44 entries for three bands and the whole-tower campaign its description
describes is ~300.

**The rule.** One field of one entry in a named table may be wrapped
``{"at_once": [...]}`` or ``{"at_once": {"range": [start, stop, step?]}}`` — the
same value grammar ``{"sweep": ...}`` uses, and literally the same validator
(:func:`causalab.protocol.sweep.axis_values`). The entry then denotes one entry
per value, named ``a[layers=10]`` — the ``rot[k=8]`` convention §3 already
uses for derived names — or by an optional ``names`` template naming exactly the
axis field, ``"a{layers}"``.

**On ``layers`` the wrapper composes.** A site's ``layers`` is a band (§2.4), and
an ``at_once`` axis on it is indexed by layer: each member value is a layer
index and denotes the one-layer band ``[n]`` (the parser's and the canonical
form's rule for a bare index), so the shipped ``{"at_once": {"range": [10,
20]}}`` is ten one-layer sites in one point — N sites, N reads, N writes, one
forward. A ``layers`` **band** is the other shape: one site, one read, one write
across N layers. A member value that is itself a list is a band site per member
(``[[10, 11], [12, 13]]`` — two two-layer sites in one point), labelled as
:func:`~causalab.protocol.sweep.band_label` spells a band (``10..11``).

**Fan-out is by reference**, the same name-identity rule sweeps use (§3): an
entry that references a family fans out over that family's axis, and member *i*
references member *i*. So one `reads` entry over a site family is a family of
reads, and one `writes` entry over both is a family of writes. An
`intervened_models` write list then **windows** one, in the words that declared
it: ``{"w": {"layers": {"at_once": {"range": [10, 15]}}}}`` is the five writes of
that band. A window is an ``at_once`` axis over the family's field whose values
the family must carry — the same wrapper, the same payload grammar, the same
validator, so this module has one spelling of "a list of values" and not a
second one for the subset.

**Sweep or family?** If the values do not have to be in force at the same time,
**sweep** them: the planner interns the shared forward anyway (§3), so a
32-layer harvest is one forward either way, and a sweep keeps the coordinates in
the results table instead of multiplying declarations. A family is for when one
forward has to carry them all.

**Where this runs, and why the digest does not move.** Expansion is a pure tree
edit and a compile stage of its own — ``families``, between ``resolve`` and the
authored-form shape ``gate`` (:data:`causalab.protocol.compile.STAGES`). Everything downstream — the parser, the
§5 checklist, sweep expansion, the canonical form, the digest — sees exactly the
document the author would have written by hand. That is what makes ``at_once``
*sugar* in the §7 sense ("the canonical form materializes … sugar expanded")
rather than a second dialect: rewriting a specification's tables in ``at_once``
compiles to the same bytes, so the *mechanism* costs no digest. Be exact about
what that does and does not cover — a v1 digest is over the whole canonical
form, ``description`` included, so a file whose prose is also rewritten moves,
for the prose. The shipped preset is that case, deliberately: its old
``description`` told the next author to copy-paste.

This version deliberately stops at the tables above. A family *of* intervened
models — an ``at_once`` on a model entry's own field — is refused by name, and is
told apart from a window by position: a window sits inside a write-list item,
under the family it selects from (:func:`_model_axis`). `metrics`, `save` and
`train` do not fan out, and a family reaching any of them is refused by name: a
saved family needs a rule for per-member ``file_path``, and §2.12's answer for
swept documents — coordinates become columns of one table — is very likely the
better one than N files, which makes it its own change rather than a line here.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from causalab.protocol.errors import ValidationError
from causalab.protocol.schema import (
    OBJECTIVE_WEIGHT_PREFIX,
    NAMED_SECTIONS,
    REGULARIZER_KINDS,
    RESERVED_NAMES,
)
from causalab.protocol.sweep import AT_ONCE_KEY, SWEEP_KEY, axis_values

#: Re-exported for the callers that read a document rather than expand one.
#: The declaration lives in :mod:`causalab.protocol.sweep`, beside ``sweep``'s
#: own — one spelling, and the nested-wrapper check there sees both keywords
#: without importing this module. A second literal here would let
#: ``sweep._any_nested_wrapper`` and :func:`has_families` drift into disagreeing
#: about what a family is, which no test could have caught.
__all__ = ["AT_ONCE_KEY", "NAMES_KEY", "expand_families", "has_families"]

#: The optional member-naming template that sits beside it.
NAMES_KEY = "names"

#: The rule every refusal here carries (§5).
RULE = "family_wrappers"

#: Tables whose entries may declare or inherit a family. `intervened_models` is
#: handled separately — it *selects* from a write family rather than joining its
#: axis — and `metrics` is absent for the reason the module docstring gives.
FAMILY_SECTIONS: tuple[str, ...] = (
    "positions",
    "sites",
    "featurizers",
    "params",
    "reads",
    "writes",
)

#: Tables a family may not reach in this version, each for the reason the
#: module docstring gives: what a per-member ``file_path`` should be.
CONSUMER_SECTIONS: tuple[str, ...] = ("metrics", "save", "train")

#: Fields that hold the *name of another entry*. Fan-out rewrites only these,
#: rather than every string equal to a family name, because several metric
#: fields (``a``, ``b``, ``token``, ``target``) name **dataset columns** and not
#: document entities (§2.10) — a column called ``a`` must not be captured by a
#: family called ``a``.
NAME_FIELDS: frozenset[str] = frozenset({"site", "pos", "model", "featurizer"})

#: Operand slots inside a ``do`` block (§2.8): a read name, a param name or a
#: literal. ``b`` here is ``affine``'s bias param, unrelated to ``metrics.b``.
DO_OPERAND_FIELDS: frozenset[str] = frozenset({"swap", "op", "A", "b"})

#: The most members one family may denote. `sweep`'s per-axis bound is a
#: million because :data:`~causalab.protocol.sweep.DEFAULT_POINT_CAP` refuses
#: the expansion afterwards; a family materializes entries *directly*, with no
#: later cap to catch it, so it needs its own — and a table of a thousand
#: addresses in one forward is past the point where the answer is a sweep.
MAX_FAMILY_MEMBERS = 1024


def has_families(raw: Mapping[str, Any]) -> bool:
    """Whether anything in ``raw`` — a document, or its ``method`` group alone
    — declares a family.

    :func:`expand_families` is a no-op on a document without one, and every
    document written before §3.1 is such a document — so this is the cheap
    guard that keeps the new stage off the existing path entirely.
    """
    return _wrapper_path(_method_of(raw)) is not None


def _method_of(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """The tree families live in (§1, §3.1): a document's ``method`` group,
    or ``raw`` itself when it already *is* that group (no group key at all).
    A ``method`` that is not an object is left to the shape gate to refuse."""
    if "method" in raw and isinstance(raw["method"], Mapping):
        return raw["method"]
    return raw


def _wrapper_path(raw: Mapping[str, Any]) -> str | None:
    """The path of the first ``at_once`` wrapper in ``raw``, or ``None``.

    A **table key** spelled ``at_once`` is skipped: §1 reserves four names and
    this is not one of them, so an entry may legitimately be called that, and
    reading its name as a wrapper would refuse a valid document with a message
    about something the author never wrote.
    """
    for section, value in raw.items():
        if section in NAMED_SECTIONS and isinstance(value, Mapping):
            for name, entry in value.items():
                found = _find_key(entry, AT_ONCE_KEY, f"{section}.{name}")
                if found is not None:
                    return found
        else:
            found = _find_key(value, AT_ONCE_KEY, str(section))
            if found is not None:
                return found
    return None


def _find_key(node: Any, key: str, path: str = "") -> str | None:
    """The path of the first occurrence of ``key`` anywhere in ``node``."""
    if isinstance(node, Mapping):
        for name, value in node.items():
            here = f"{path}.{name}" if path else str(name)
            if name == key:
                return path or here
            found = _find_key(value, key, here)
            if found is not None:
                return found
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found = _find_key(item, key, f"{path}[{index}]")
            if found is not None:
                return found
    return None


# --------------------------------------------------------------------------- #
# one family
# --------------------------------------------------------------------------- #


class _Family:
    """One declared or inherited family: its axis, and how members are named."""

    __slots__ = ("field", "name", "names", "section", "values")

    def __init__(
        self,
        section: str,
        name: str,
        field: str,
        values: tuple[Any, ...],
        names: str | None,
    ) -> None:
        self.section = section
        self.name = name
        self.field = field
        self.values = values
        self.names = names

    @property
    def axis(self) -> tuple[str, tuple[Any, ...]]:
        """Axis identity — the field name plus the values.

        Two families over the same field and the same values are **one** axis
        and align member for member; the same field over different values is a
        different axis, and one entry referencing both is refused, because
        which pairs are meant is exactly what alignment cannot guess.
        """
        # a member value may itself be a band (a list, on `layers`); the axis
        # is compared by value, so it is spelled hashably
        return (self.field, tuple(_label(v) for v in self.values))

    def member(self, value: Any) -> str:
        if self.names is not None:
            return self.names.replace("{" + self.field + "}", _label(value))
        return f"{self.name}[{self.field}={_label(value)}]"

    def members(self) -> tuple[tuple[str, Any], ...]:
        return tuple((self.member(value), value) for value in self.values)


def _label(value: Any) -> str:
    """One axis value as it appears in a member's name — the same rendering
    sweeps use for a coordinate (:func:`causalab.protocol.sweep.label_value`),
    so a family member and a swept name are spelled alike."""
    from causalab.protocol.sweep import label_value

    return label_value(value)


def _refuse(message: str, path: str) -> ValidationError:
    return ValidationError(RULE, message, path=path)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


def _declared_axis(section: str, name: str, entry: Mapping[str, Any]) -> str | None:
    """The field this entry wraps in ``at_once``, if any."""
    wrapped = [key for key, value in entry.items() if _is_wrapper(value, AT_ONCE_KEY)]
    if not wrapped:
        return None
    if len(wrapped) > 1:
        raise _refuse(
            f"{len(wrapped)} fields carry an at_once wrapper "
            f"({', '.join(sorted(wrapped))}) — one axis per entry, so that a "
            "member has one index and one name",
            f"{section}.{name}",
        )
    return wrapped[0]


def _is_wrapper(node: Any, key: str) -> bool:
    return isinstance(node, Mapping) and key in node


def _template(
    entry: Mapping[str, Any], field: str, section: str, name: str
) -> str | None:
    template = entry.get(NAMES_KEY)
    if template is None:
        return None
    placeholder = "{" + field + "}"
    if not isinstance(template, str) or placeholder not in template:
        raise _refuse(
            f"a names template names exactly the axis field, {placeholder!r} — "
            f"got {template!r}. Without it two members would share a name",
            f"{section}.{name}.{NAMES_KEY}",
        )
    return template


def _is_name_field(key: Any, *, in_do: bool) -> bool:
    return key in NAME_FIELDS or (in_do and key in DO_OPERAND_FIELDS)


def _referenced(node: Any, *, in_do: bool = False) -> Iterable[str]:
    """Every entry name a subtree references, by :data:`NAME_FIELDS`.

    A name field may hold a **list** of names — a `featurizer` composition is
    written either way (§2.5) — so both spellings of one reference are read
    here. Whether a family works must not depend on which the author picked.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            if _is_name_field(key, in_do=in_do):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, list):
                    yield from (item for item in value if isinstance(item, str))
            elif isinstance(value, (Mapping, list)):
                yield from _referenced(value, in_do=in_do or key == "do")
    elif isinstance(node, list):
        for item in node:
            yield from _referenced(item, in_do=in_do)


def _find_families(raw: Mapping[str, Any]) -> dict[str, _Family]:
    """Declared families, then inherited ones to a fixpoint."""
    families: dict[str, _Family] = {}

    for section in FAMILY_SECTIONS:
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for name, entry in table.items():
            if not isinstance(entry, Mapping):
                continue
            field = _declared_axis(section, name, entry)
            if field is None:
                continue
            wrapper = entry[field]
            if len(wrapper) != 1:
                raise _refuse(
                    "an at_once wrapper holds nothing but the axis",
                    f"{section}.{name}.{field}",
                )
            values = axis_values(
                wrapper[AT_ONCE_KEY],
                path=f"{section}.{name}.{field}",
                rule=RULE,
                keyword=AT_ONCE_KEY,
                article="an",
            )
            _check_axis(values, f"{section}.{name}.{field}")
            families[name] = _Family(
                section,
                name,
                field,
                values,
                _template(entry, field, section, name),
            )

    # inheritance is a fixpoint over the reference graph, which §2.9 makes a
    # DAG — so it terminates on its own, and a bound would only ever
    # under-expand a chain deeper than whatever number was picked
    grew = True
    while grew:
        grew = False
        for section in FAMILY_SECTIONS:
            table = raw.get(section)
            if not isinstance(table, Mapping):
                continue
            for name, entry in table.items():
                if name in families or not isinstance(entry, Mapping):
                    continue
                donors = [
                    families[ref] for ref in _referenced(entry) if ref in families
                ]
                if not donors:
                    continue
                axes = {donor.axis for donor in donors}
                if len(axes) > 1:
                    fields = sorted({field for field, _ in axes})
                    raise _refuse(
                        f"this entry references families on {len(axes)} different "
                        f"axes ({', '.join(fields)}) — a cross product inside one "
                        "forward is not what a family means. Sweep the second axis "
                        "(§3), which is what a cross product is for",
                        f"{section}.{name}",
                    )
                donor = donors[0]
                families[name] = _Family(
                    section,
                    name,
                    donor.field,
                    donor.values,
                    _template(entry, donor.field, section, name),
                )
                grew = True
        if not grew:
            break
    return families


def _check_one_axis_per_entry(
    raw: Mapping[str, Any], families: Mapping[str, _Family]
) -> None:
    """Every family-capable entry sits on exactly one axis.

    The inheritance fixpoint checks the entries it *derives* an axis for, but
    an entry that declares its own was skipped — and there the mismatch is
    quieter, not louder: :func:`_member_entry` substitutes a reference by
    *value*, so a donor family that happens to carry the same values yields the
    diagonal of a cross product silently, and one that does not yields a name
    nothing declares for rule 4 to report as a dangling reference. Neither is
    the experiment anyone wrote.
    """
    for section in FAMILY_SECTIONS:
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for name, entry in table.items():
            own = families.get(name)
            if own is None or not isinstance(entry, Mapping):
                continue
            foreign = {
                families[ref].axis
                for ref in _referenced(entry)
                if ref in families and families[ref].axis != own.axis
            }
            if foreign:
                fields = sorted({field for field, _ in foreign} | {own.field})
                raise _refuse(
                    f"this entry is on the {own.field!r} axis and references a "
                    f"family on another ({', '.join(fields)}) — a cross product "
                    "inside one forward is not what a family means. Sweep the "
                    "second axis (§3), which is what a cross product is for",
                    f"{section}.{name}",
                )


def _check_axis(values: tuple[Any, ...], path: str) -> None:
    """The bound first, then distinctness.

    In that order on purpose: :func:`~causalab.protocol.sweep.axis_values` caps
    the ``{"range": …}`` form and nothing caps an explicit list, so checking
    distinctness first meant a 200k-element list was counted before anything
    mentioned the bound.
    """
    if len(values) > MAX_FAMILY_MEMBERS:
        raise _refuse(
            f"this axis denotes {len(values)} members, over the bound of "
            f"{MAX_FAMILY_MEMBERS}. A family materializes entries directly, so "
            "nothing downstream caps it the way the point cap caps a sweep — and "
            "a table of this many addresses in one forward is the point where "
            "the answer is a sweep (§3.1)",
            path,
        )
    counts = Counter(_label(value) for value in values)
    duplicated = sorted(label for label, n in counts.items() if n > 1)
    if duplicated:
        raise _refuse(
            f"at_once values are a member index, so they must be distinct — "
            f"{', '.join(duplicated)} appears twice",
            path,
        )


# --------------------------------------------------------------------------- #
# materialization
# --------------------------------------------------------------------------- #


def _member_entry(
    node: Any,
    families: Mapping[str, _Family],
    value: Any,
    *,
    axis: str | None = None,
    in_do: bool = False,
    top: bool = True,
) -> Any:
    """One member's copy of an authored entry.

    ``axis`` — the entry's own declared field, and *only* at the entry's top
    level — becomes this member's index value; every family reference becomes
    the member at the same index. A wrapper anywhere else is deliberately left
    in place: it has no name identity, and substituting it here would silently
    give it this member's index instead (the bug the orphan check now catches).
    """
    if isinstance(node, Mapping):
        out: dict[str, Any] = {}
        for key, child in node.items():
            if top and key == NAMES_KEY:
                continue  # the template is authoring metadata; the member is named
            if top and key == axis and _is_wrapper(child, AT_ONCE_KEY):
                out[key] = _member_value(key, value)
            elif _is_name_field(key, in_do=in_do) and isinstance(child, str):
                out[key] = families[child].member(value) if child in families else child
            elif _is_name_field(key, in_do=in_do) and isinstance(child, list):
                out[key] = [
                    families[item].member(value)
                    if isinstance(item, str) and item in families
                    else item
                    for item in child
                ]
            else:
                out[key] = _member_entry(
                    child,
                    families,
                    value,
                    in_do=in_do or key == "do",
                    top=False,
                )
        return out
    if isinstance(node, list):
        return [
            _member_entry(item, families, value, in_do=in_do, top=False)
            for item in node
        ]
    return node


def _member_value(field: str, value: Any) -> Any:
    """The value a member carries in its axis field. On ``layers`` (§2.4) an
    index denotes the one-layer band ``[n]``, so the expanded tree *is* the
    hand-written one entry for entry — the parser and the canonical form make
    the same fold, but the expansion is what an author reads back."""
    if field == "layers" and isinstance(value, int) and not isinstance(value, bool):
        return [value]
    return value


def _window(spec: Any, path: str) -> tuple[Any, ...]:
    """The values a write-list selector picks out of a family's index.

    Spelled ``{"at_once": [...]}`` or ``{"at_once": {"range": [a, b, step?]}}``
    — the wrapper that declared the family, carrying the payload grammar
    :func:`axis_values` already checks. A first draft spelled an interval
    ``{"span": [a, b]}`` after the position window (§2.3) and let a bare list
    through beside it: two dialects of one selection, the interval the weaker
    (no step), and the bare form's refusals talking about a keyword the author
    never wrote. One word per object (§11.1). The empty window — ``{"range":
    [15, 10]}`` names no write, so the band would compile as the un-intervened
    model and report no effect — is the empty axis, which the shared grammar
    already refuses.
    """
    if isinstance(spec, Mapping) and AT_ONCE_KEY in spec and len(spec) > 1:
        extra = sorted(set(spec) - {AT_ONCE_KEY})
        raise _refuse(
            f"a window holds nothing but its {AT_ONCE_KEY} axis — got "
            f"{', '.join(extra)} too",
            path,
        )
    if not (isinstance(spec, Mapping) and AT_ONCE_KEY in spec):
        raise _refuse(
            "a window is spelled like the axis it selects from: "
            f'{{"{AT_ONCE_KEY}": [...]}} or '
            f'{{"{AT_ONCE_KEY}": {{"range": [start, stop, step?]}}}}',
            path,
        )
    return axis_values(
        spec[AT_ONCE_KEY], path=path, rule=RULE, keyword=AT_ONCE_KEY, article="an"
    )


def _selected_writes(
    name: str,
    selector: Mapping[str, Any],
    families: Mapping[str, _Family],
    path: str,
) -> list[str]:
    if name not in families:
        raise _refuse(
            f"{name!r} is not a family, so there is nothing to select from — "
            "a write list names a write, or windows a family",
            path,
        )
    family = families[name]
    fields = sorted(selector)
    if fields != [family.field]:
        raise _refuse(
            f"family {name!r} is indexed by {family.field!r}, so a window is "
            f"declared on {family.field!r} — got {', '.join(fields) or 'nothing'}",
            path,
        )
    index = {_label(value): member for member, value in family.members()}
    wanted = _window(selector[family.field], f"{path}.{family.field}")
    missing = [_label(value) for value in wanted if _label(value) not in index]
    if missing:
        raise _refuse(
            f"the window names {family.field}={', '.join(missing)}, which family "
            f"{name!r} does not carry (it runs {_label(family.values[0])}.."
            f"{_label(family.values[-1])}) — a band that resolves to fewer writes "
            "than the interval it declares is the silent bug this form exists to "
            "refuse",
            path,
        )
    return [index[_label(value)] for value in wanted]


def _write_list(
    model: str, entry: Mapping[str, Any], families: Mapping[str, _Family]
) -> list[Any]:
    """One intervened model's `writes`, with families resolved to member names.

    A bare family name is every member, and follows the family if it grows; a
    window is the explicit form, and refuses to resolve to a different number
    of writes than the interval it names.
    """
    writes: list[Any] = []
    listed = entry.get("writes")
    if not isinstance(listed, list):
        # A whole write list may be swept (`_parse_im` wraps it), and that is
        # legal and untouched — unless a family name is inside it, where the
        # member it means depends on the point and this stage has none. Refused
        # by name, like every other shape §3.1 leaves for later.
        held = next((name for name in _mentions(listed) if name in families), None)
        if held is not None:
            raise _refuse(
                f"this write list is swept and names the family {held!r}. A "
                "family materializes before axes are found (§3.1), so a member "
                "chosen per point is not something this stage can resolve — "
                "sweep something other than the write list, or name members",
                f"intervened_models.{model}.writes",
            )
        # otherwise absent or the wrong type: the parser's rejection to make,
        # not ours — and injecting `writes: None` would take away its message
        return listed
    for index, item in enumerate(listed):
        path = f"intervened_models.{model}.writes[{index}]"
        if isinstance(item, str):
            if item in families:
                writes.extend(member for member, _ in families[item].members())
            else:
                writes.append(item)
        elif isinstance(item, Mapping):
            if len(item) != 1:
                raise _refuse(
                    "a write-list entry names one family, "
                    '{"<family>": {"<field>": {"at_once": ...}}}',
                    path,
                )
            name, selector = next(iter(item.items()))
            if not isinstance(selector, Mapping):
                raise _refuse(
                    f"the selector for {name!r} is an object on the family's "
                    'index, e.g. {"layers": {"at_once": {"range": [10, 15]}}}',
                    path,
                )
            writes.extend(_selected_writes(name, selector, families, path))
        else:
            writes.append(item)
    counts = Counter(w for w in writes if isinstance(w, str))
    duplicated = sorted(name for name, n in counts.items() if n > 1)
    if duplicated:
        raise _refuse(
            f"{', '.join(duplicated)} listed twice — overlapping selectors, not "
            "overlapping bands. Bands overlap by sharing writes between models, "
            "never by naming one twice in one model",
            f"intervened_models.{model}.writes",
        )
    return writes


# --------------------------------------------------------------------------- #
# the entry point
# --------------------------------------------------------------------------- #


def expand_families(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``raw`` with every ``at_once`` family materialized (§3.1).

    A pure tree edit, and a no-op on any document that declares no family — so
    the compiled intervention, and therefore the digest, is exactly what the
    same experiment written out by hand produces. Idempotent: the result
    carries no ``at_once``, so expanding twice changes nothing.

    Families live in the ``method`` group (§1): a wrapper declares one field of
    one named method entry, and every refusal path is section-rooted, the way
    every other path in the protocol is spelled. Handed a document, this
    expands its ``method`` and returns the document; handed the group alone,
    it returns the expanded group. The header, ``model`` and ``data`` never
    see a wrapper.
    """
    if "method" in raw and isinstance(raw["method"], Mapping):
        return {**raw, "method": _expand_method(raw["method"])}
    return _expand_method(raw)


def _expand_method(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not has_families(raw):
        _check_orphan_names(raw, {})
        return dict(raw)

    families = _find_families(raw)
    _check_orphan_names(raw, families)
    _check_one_axis_per_entry(raw, families)
    _check_no_sweep_on_a_family(raw, families)
    declared = _declared_names(raw)

    out: dict[str, Any] = {}
    for section, value in raw.items():
        if section in FAMILY_SECTIONS and isinstance(value, Mapping):
            table: dict[str, Any] = {}
            for name, entry in value.items():
                family = families.get(name)
                if family is None:
                    table[name] = entry
                    continue
                for member, index_value in family.members():
                    _check_member_name(member, name, section, table, declared)
                    table[member] = _member_entry(
                        entry, families, index_value, axis=family.field
                    )
            out[section] = table
        elif section == "intervened_models" and isinstance(value, Mapping):
            models: dict[str, Any] = {}
            for name, entry in value.items():
                if (
                    name in families
                    or _model_axis(entry, f"intervened_models.{name}") is not None
                ):
                    raise _refuse(
                        "a family of intervened models is not in this version: "
                        "one model per band, and its metric and save entry with "
                        "it, are what a saved family still needs a `file_path` "
                        "rule for (§3.1)",
                        f"intervened_models.{name}",
                    )
                if not isinstance(entry, Mapping) or "writes" not in entry:
                    models[name] = entry  # rule 10's message to give, not ours
                    continue
                models[name] = {
                    **entry,
                    "writes": _write_list(name, entry, families),
                }
            out[section] = models
        else:
            _check_no_consumer_reference(section, value, families)
            out[section] = value

    orphan = _wrapper_path(out)
    if orphan is not None:
        raise _refuse(
            "an at_once wrapper sits where it has no name identity: it declares "
            "one field of one entry of "
            f"{', '.join(FAMILY_SECTIONS)}, and nowhere else (§3.1)",
            orphan,
        )
    return out


def _declared_names(raw: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for section in NAMED_SECTIONS:
        table = raw.get(section)
        if isinstance(table, Mapping):
            names.update(str(name) for name in table)
    return names


def _check_member_name(
    member: str,
    family: str,
    section: str,
    table: Mapping[str, Any],
    declared: set[str],
) -> None:
    where = f"{section}.{family}.{NAMES_KEY}"
    if member in table:
        raise _refuse(
            f"two members are both named {member!r} — check the names template",
            where,
        )
    if member in RESERVED_NAMES:
        raise _refuse(
            f"a member is named {member!r}, which is reserved (§1)",
            where,
        )
    if member in declared and member != family:
        raise _refuse(
            f"a member is named {member!r}, which the document already declares "
            "— one name, one entry (§1)",
            where,
        )


def _check_orphan_names(
    raw: Mapping[str, Any], families: Mapping[str, _Family]
) -> None:
    """``names`` with no axis to name is a typo worth catching by name: the
    parser would report it as an unknown key on whichever table it sits in.

    Checked against the discovered families rather than against the entry's own
    wrapper, because ``names`` is equally legal on an entry that *inherits* its
    axis by reference — which is most of them. And run on every document, not
    only family-free ones: the typo is likeliest in a document that has
    families elsewhere, which is exactly where the first version of this check
    did not look.
    """
    for section in FAMILY_SECTIONS:
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for name, entry in table.items():
            if (
                isinstance(entry, Mapping)
                and NAMES_KEY in entry
                and name not in families
            ):
                raise _refuse(
                    f"{NAMES_KEY!r} names the members of a family, but this entry "
                    f"declares no {AT_ONCE_KEY!r} axis (§3.1)",
                    f"{section}.{name}.{NAMES_KEY}",
                )


def _model_axis(entry: Any, path: str) -> str | None:
    """The path of an ``at_once`` wrapper on an intervened model *itself* — a
    family of models, which this version refuses by name — as distinct from a
    **window** inside its write list, which :func:`_write_list` resolves.

    The two are told apart by position, not by keyword: a window is a one-key
    object *keyed by a family name* inside a list-shaped write list,
    ``writes[i].<family>.<field>``, and ``_write_list`` consumes every such
    object and refuses anything that is not exactly that shape. A wrapper
    anywhere else in the entry — on ``input``, wrapping a write item itself —
    is an axis over models. A swept ``writes`` is left to ``_write_list`` too:
    its refusal names the sweep, which is what the author wrote.
    """
    if not isinstance(entry, Mapping):
        return _find_key(entry, AT_ONCE_KEY, path)
    for key, value in entry.items():
        if key == AT_ONCE_KEY:
            return path
        if key == "writes":
            if not isinstance(value, list):
                continue
            for index, item in enumerate(value):
                here = f"{path}.writes[{index}]"
                if isinstance(item, Mapping):
                    if AT_ONCE_KEY in item:
                        return here
                    continue
                found = _find_key(item, AT_ONCE_KEY, here)
                if found is not None:
                    return found
            continue
        found = _find_key(value, AT_ONCE_KEY, f"{path}.{key}")
        if found is not None:
            return found
    return None


def _mentions(node: Any) -> Iterable[str]:
    """Every string anywhere in a subtree, mapping keys included — for the
    check that asks whether a family is *named*, which a window does by key:
    ``{"w": {"layers": {"at_once": ...}}}`` mentions ``w``."""
    yield from _flat_strings(node)
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield str(key)
            yield from _mentions(value)
    elif isinstance(node, list):
        for item in node:
            yield from _mentions(item)


def _flat_strings(node: Any) -> Iterable[str]:
    """Every string anywhere in a subtree, for the checks that only ask whether
    a name is *mentioned*."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, Mapping):
        for value in node.values():
            yield from _flat_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _flat_strings(item)


def _extra_names(entry: Any) -> Iterable[str]:
    """Name-bearing fields outside :data:`NAME_FIELDS`, in the tables that only
    consume: a metric's ``of`` and a save entry's ``value``."""
    if not isinstance(entry, Mapping):
        return
    for field in ("of", "value"):
        held = entry.get(field)
        if isinstance(held, str):
            yield held
    for nested in entry.values():
        if isinstance(nested, Mapping):
            yield from _extra_names(nested)
        elif isinstance(nested, list):
            for item in nested:
                yield from _extra_names(item)


def _train_names(block: Any) -> Iterable[str]:
    """Every entry a ``train`` block references, by the **owning entry's** name
    (§2.11).

    Four spellings, and each was its own small trap:

    * ``params`` is a plain name list — but its members are param *slots*, so
      ``"rot.weight"`` names the entry ``rot``. Compared on the first dotted
      segment, which is what :func:`validate._check_train_references` does;
    * a **regularizer is keyed by its kind** — ``{"l1": ["rot"]}`` — which is
      why scanning for a ``names`` key found nothing: §2.11 never emits one;
    * that mapping's value may be a bare string rather than a list
      (``_parse_regularizer_names`` accepts both);
    * and in the positional objective form the whole thing sits inside a
      ``[weight, …]`` list inside the objective list, so the walk has to
      descend lists as well as mappings.

    ``anneal`` is the fourth: it is a mapping *keyed* by the slot it anneals,
    so its names are in its keys and nowhere else.
    """
    if not isinstance(block, Mapping):
        return
    params = block.get("params")
    if isinstance(params, list):
        yield from (_owner(item) for item in params if isinstance(item, str))
    anneal = block.get("anneal")
    if isinstance(anneal, Mapping):
        # a `train.objective.<name>.weight` key anneals a term, not a slot —
        # its first segment names no entry (§2.11)
        yield from (
            _owner(key)
            for key in anneal
            if isinstance(key, str) and not key.startswith(OBJECTIVE_WEIGHT_PREFIX)
        )
    phases = block.get("phases")
    if isinstance(phases, list):
        # a phase narrows `params` and may anneal — the same two spellings
        for phase in phases:
            if isinstance(phase, Mapping):
                yield from _train_names(
                    {k: v for k, v in phase.items() if k != "optimizer"}
                )
                freeze = phase.get("freeze_masks")
                if isinstance(freeze, list):
                    yield from (name for name in freeze if isinstance(name, str))
    yield from _regularizer_names(block.get("objective"))


def _owner(slot: str) -> str:
    """The entry a param slot belongs to: ``rot.weight`` is ``rot``'s."""
    return slot.split(".", 1)[0]


def _regularizer_names(node: Any) -> Iterable[str]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in REGULARIZER_KINDS:
                if isinstance(value, str):
                    yield _owner(value)
                elif isinstance(value, list):
                    yield from (_owner(item) for item in value if isinstance(item, str))
            else:
                yield from _regularizer_names(value)
    elif isinstance(node, list):
        for item in node:
            yield from _regularizer_names(item)


def _check_no_sweep_on_a_family(
    raw: Mapping[str, Any], families: Mapping[str, _Family]
) -> None:
    """A family entry may not also carry a sweep axis.

    Expansion copies the entry once per member, so the sweep wrapper would be
    copied too — and a sweep axis is identified by its *path* (§3), so N copies
    on N paths are N independent axes whose cross product is exponential rather
    than the one axis the author meant.
    """
    for section in FAMILY_SECTIONS:
        table = raw.get(section)
        if not isinstance(table, Mapping):
            continue
        for name, entry in table.items():
            if name not in families or not isinstance(entry, Mapping):
                continue
            # anywhere in the entry, not only on its own fields: a `sweep`
            # inside a `do` block is copied to every member exactly the same
            # way, and ten members of the shipped preset is 2**10 points —
            # under the point cap, so nothing downstream would have caught it
            swept = _find_key(entry, SWEEP_KEY, f"{section}.{name}")
            if swept is not None:
                raise _refuse(
                    f"this entry declares an at_once family and also sweeps "
                    f"{swept} — expansion would copy the sweep wrapper onto all "
                    f"{len(families[name].values)} members, and a sweep axis is "
                    "its path (§3), so that is one axis per member and a cross "
                    "product of them. Sweep an entry off the family instead",
                    f"{section}.{name}",
                )


def _check_no_consumer_reference(
    section: str, value: Any, families: Mapping[str, _Family]
) -> None:
    """`metrics`, `save` and `train` do not fan out in this version, so a
    family reaching any of them is refused by name rather than silently kept as
    a dangling reference for rule 4 to report as a missing entry.

    `train` is here for the same reason as the other two and one more: it
    references entries by *param slot* — ``params``, a regularizer's names, and
    ``anneal``'s keys — so a fit over a family is the shape most likely to look
    as though it had worked. Its names are collected by :func:`_train_names`,
    which knows the four spellings §2.11 actually uses.
    """
    if section not in CONSUMER_SECTIONS or not isinstance(value, (Mapping, list)):
        return
    # `metrics` is a table of entries and `save` a list of them, so each can be
    # named in the path; `train` is one block, not a table, and iterating *its*
    # items would look inside `params` one element at a time and see no names
    entries: Any = (
        [(None, value)]
        if section == "train"
        else value.items()
        if isinstance(value, Mapping)
        else enumerate(value)
    )
    for key, entry in entries:
        named = list(_referenced(entry))
        named.extend(_train_names(entry) if section == "train" else _extra_names(entry))
        hit = next((name for name in named if name in families), None)
        if hit is not None:
            raise _refuse(
                f"this references the family {hit!r}. `metrics`, `save` and "
                "`train` "
                "do not fan out in this version: a saved family needs a rule for "
                "per-member `file_path`, and §2.12's answer for swept documents — "
                "coordinates become columns of one table — is the better one, so "
                "it is its own change. Name one member, or sweep the axis (§3)",
                section if key is None else f"{section}.{key}",
            )
