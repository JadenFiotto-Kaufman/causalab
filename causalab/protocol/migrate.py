"""``causalab migrate`` — the rewrite of an intervention specification to the
current ``protocol_version``.

Spec §7 calls a change to the canonical form a *loader migration*: bump the
version, ship the rewrite. This module carries every step so far, chained, so
a document of any earlier version comes out at :data:`PROTOCOL_VERSION`:

**v1 → v2** (§1): a flat v1 document — fifteen top-level sections — becomes the
four groups ``header`` / ``model`` / ``data`` / ``method``, with nothing inside
a section changed:

* ``version`` → ``header.protocol_version``; ``description`` →
  ``header.description``; ``type`` is dropped (a file's kind is its shape);
* ``model`` (or its retired alias ``neural_model``) and ``data`` stay where
  they are;
* the remaining sections — the method's — move under ``method``, in their §1 order.

**v2 → v3** (§2.4): a site's depth index ``layer`` became ``layers``, a
**band** of layer indices — ``{"component": "block_output", "layer": 18}``
is ``{"component": "block_output", "layers": [18]}``. The rename is carried
everywhere the field is spelled and nothing else inside a section changes:

* ``method.sites.<name>.layer: n`` → ``layers: [n]``; a wrapped field
  (``{"sweep": …}``, ``{"at_once": …}``, ``{"artifact": …}``) keeps its wrapper
  under the new key — an axis over ``layers`` is indexed by layer (§3, §3.1);
* an ``at_once`` window in an ``intervened_models`` write list,
  ``{"w": {"layer": {"at_once": …}}}``, and a ``names`` template's ``{layer}``
  placeholder re-spell the field;
* every dotted id ``sites.<name>.layer`` — a workflow step's ``set``, ``emit``,
  ``group_by`` or plot ``x``, a ``layer_column``, the same id quoted in a
  ``description`` — becomes ``sites.<name>.layers``, wherever in the tree the
  string sits, as a key or as a value.

A **workflow** document has no ``protocol_version`` of its own (its
``version`` is the workflow spec's); it is rewritten exactly when it spells a
dotted ``sites.<name>.layer`` id, and comes back as it went in otherwise.

The rewrite is a pure function of the document and idempotent: a document
already at the current version comes back as it went in. A **split** v1
document (``application`` + ``method`` halves, or a ``type: method`` file) is
refused rather than composed — composition was the v1 loader's, and the six
such documents this repository shipped were composed by it before it was
retired. Compose one with a v1 release and migrate the composition.

Markdown files are rewritten too: every fenced ``json`` block that parses as a
whole document needing migration is migrated in place, so a guide's examples
move with the documents they describe (``tests/docs/test_docs.py`` loads them).

:func:`format_document` is the authoring format the rewrite emits — two-space
indentation, with an object or list written on one line when it is shallow and
fits, which is how the shipped documents read: one site, one read, one save
entry per line.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from causalab.protocol.errors import ParseError, ProtocolError, suggest
from causalab.protocol.schema import (
    METHOD_SECTIONS,
    MIGRATABLE_PROTOCOL_VERSIONS,
    PROTOCOL_VERSION,
    load_raw,
)

__all__ = [
    "format_document",
    "is_v1_protocol",
    "main",
    "migrate_document",
    "migrate_markdown",
    "needs_migration",
]

#: The version the v1 → v2 step writes; the v2 → v3 step reads it.
_V2: str = "2"
#: A dotted site-field id in the retired spelling, wherever a string sits.
_DOTTED_LAYER = re.compile(r"\bsites\.([A-Za-z0-9_]+)\.layer\b")
#: The retired ``names`` placeholder (§3.1).
_TEMPLATE_LAYER = "{layer}"

#: v1's top-level bookkeeping (spec v1 §1, rows 1–3).
_V1_HEADER: tuple[str, ...] = ("version", "type", "description")
#: The two spellings v1 accepted for the network (v1 §2.1).
_V1_MODEL: tuple[str, ...] = ("model", "neural_model")
#: The v1 split's own sections (v1 §1.1), which this rewrite refuses.
_V1_SPLIT: tuple[str, ...] = ("application", "method")

#: Longest line :func:`format_document` writes an object or list on one line
#: within — the repository's own documents wrap there.
LINE_WIDTH = 88


def is_v1_protocol(raw: Any) -> bool:
    """Whether ``raw`` is a v1 intervention specification: a mapping with a
    top-level ``version`` and no ``header``, and not a workflow (``steps``)."""
    return (
        isinstance(raw, Mapping)
        and "header" not in raw
        and "steps" not in raw
        and "version" in raw
    )


def needs_migration(raw: Any) -> bool:
    """Whether :func:`migrate_document` would change ``raw``: a v1 document, a
    document declaring a :data:`~causalab.protocol.schema.MIGRATABLE_PROTOCOL_VERSIONS`
    member, or a workflow spelling a dotted ``sites.<name>.layer`` id."""
    if not isinstance(raw, Mapping):
        return False
    if is_v1_protocol(raw):
        return True
    if "steps" in raw:
        return _spells_old_dotted(raw)
    header = raw.get("header")
    return isinstance(header, Mapping) and (
        header.get("protocol_version") in MIGRATABLE_PROTOCOL_VERSIONS
    )


def migrate_document(raw: Any) -> dict[str, Any]:
    """The current-version form of one document tree.

    A current-version document is returned as it is (the function is
    idempotent); a v1 document is regrouped and then carried through every
    later step; a protocol_version 2 document gets the ``layer`` → ``layers``
    rename; a workflow gets its dotted ids re-spelled; a split v1 document or
    a method file is refused (see the module docstring). Beyond the renamed
    field nothing inside a section is inspected — the strict parse is the
    compiler's job, and a document that was wrong in v1 is the same wrong
    document now, with the same refusal from ``causalab validate``.
    """
    if not isinstance(raw, Mapping):
        raise ParseError("P1", "the top level must be a JSON object")
    if "steps" in raw:
        return _rewrite_dotted(copy.deepcopy(dict(raw)))
    if "header" in raw:
        header = raw["header"]
        version = (
            header.get("protocol_version") if isinstance(header, Mapping) else None
        )
        if version == PROTOCOL_VERSION:
            return dict(raw)
        if version == _V2:
            return _v2_to_v3(raw)
        raise ParseError(
            "P2",
            f"unsupported protocol_version {version!r}; migrate reads "
            f"{', '.join(repr(v) for v in ('1', *MIGRATABLE_PROTOCOL_VERSIONS))} "
            f"and writes protocol_version {PROTOCOL_VERSION!r}",
            path="header.protocol_version",
        )
    return _v2_to_v3(_v1_to_v2(raw))


def _v1_to_v2(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The regroup (module docstring, first step)."""
    if any(key in raw for key in _V1_SPLIT) or raw.get("type") == "method":
        raise ParseError(
            "P2",
            "a split method/application document (or a method file) is not "
            "migrated: composition was the v1 loader's. Compose it with a v1 "
            "release (`causalab explain` printed the composition) and migrate "
            "the composed document",
            path="method",
        )
    version = raw.get("version")
    if version != "1":
        raise ParseError(
            "P2",
            f"unsupported version {version!r}; migrate reads v1 documents "
            f'("version": "1") and writes protocol_version {PROTOCOL_VERSION!r}',
            path="version",
        )
    known = (*_V1_HEADER, *_V1_MODEL, "data", *METHOD_SECTIONS)
    for key in raw:
        if key not in known:
            raise ParseError(
                "P3", f"unknown section {key!r}{suggest(key, known)}", path=key
            )
    if all(key in raw for key in _V1_MODEL):
        raise ParseError("P2", "both 'model' and 'neural_model' are present")
    for required in ("data",):
        if required not in raw:
            raise ParseError("P2", f"missing required section {required!r}")
    model = raw.get("model", raw.get("neural_model"))
    if model is None:
        raise ParseError("P2", "missing required section 'model'")

    header: dict[str, Any] = {"protocol_version": _V2}
    if raw.get("description") is not None:
        header["description"] = raw["description"]
    return {
        "header": header,
        "model": model,
        "data": raw["data"],
        "method": {
            section: raw[section] for section in METHOD_SECTIONS if section in raw
        },
    }


def _v2_to_v3(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The ``layer`` → ``layers`` rename (module docstring, second step), on
    a deep copy; key order is kept so the rewritten file reads as the author
    laid it out, with ``layers`` where ``layer`` stood."""
    out = copy.deepcopy(dict(raw))
    header = dict(out.get("header") or {})
    header["protocol_version"] = PROTOCOL_VERSION
    out["header"] = header
    method = out.get("method")
    if isinstance(method, Mapping):
        method = dict(method)
        sites = method.get("sites")
        if isinstance(sites, Mapping):
            method["sites"] = {
                name: _rename_layer(site) if isinstance(site, Mapping) else site
                for name, site in sites.items()
            }
        for section, table in method.items():
            if not isinstance(table, Mapping):
                continue
            for name, entry in table.items():
                if isinstance(entry, Mapping) and isinstance(entry.get("names"), str):
                    entry["names"] = entry["names"].replace(_TEMPLATE_LAYER, "{layers}")
        models = method.get("intervened_models")
        if isinstance(models, Mapping):
            for entry in models.values():
                writes = entry.get("writes") if isinstance(entry, Mapping) else None
                if not isinstance(writes, list):
                    continue
                entry["writes"] = [
                    {
                        family: _rename_layer(selector)
                        for family, selector in item.items()
                    }
                    if isinstance(item, Mapping)
                    and all(isinstance(v, Mapping) for v in item.values())
                    else item
                    for item in writes
                ]
        out["method"] = method
    return _rewrite_dotted(out)


def _rename_layer(entry: Mapping[str, Any]) -> dict[str, Any]:
    """``layer`` → ``layers`` in one site (or window selector), in place in
    the key order; a bare index becomes the one-layer band, a wrapper keeps
    its wrapper."""
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "layer":
            if isinstance(value, int) and not isinstance(value, bool):
                value = [value]
            out["layers"] = value
        else:
            out[key] = value
    return out


def _rewrite_dotted(node: Any) -> Any:
    """Every ``sites.<name>.layer`` dotted id in a tree — as a key or a string
    value — re-spelled ``sites.<name>.layers``; everything else as it is."""
    if isinstance(node, Mapping):
        return {
            (
                _DOTTED_LAYER.sub(r"sites.\1.layers", key)
                if isinstance(key, str)
                else key
            ): _rewrite_dotted(value)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_dotted(item) for item in node]
    if isinstance(node, str):
        return _DOTTED_LAYER.sub(r"sites.\1.layers", node)
    return node


def _spells_old_dotted(node: Any) -> bool:
    if isinstance(node, Mapping):
        return any(
            (isinstance(key, str) and _DOTTED_LAYER.search(key) is not None)
            or _spells_old_dotted(value)
            for key, value in node.items()
        )
    if isinstance(node, list):
        return any(_spells_old_dotted(item) for item in node)
    return isinstance(node, str) and _DOTTED_LAYER.search(node) is not None


# --------------------------------------------------------------------------- #
# the authoring format
# --------------------------------------------------------------------------- #


def format_document(document: Mapping[str, Any], *, width: int = LINE_WIDTH) -> str:
    """One document as the text the repository authors it in: two-space
    indentation, key order preserved, and any object or list that is at most
    two levels deep and fits in ``width`` columns written on one line — so a
    site, a read, a write or a save entry is one line, and a table of them is
    one entry per line. The output ends with a newline."""
    return _render(document, 0, width) + "\n"


def _depth(node: Any) -> int:
    if isinstance(node, Mapping):
        return 1 + max((_depth(v) for v in node.values()), default=0)
    if isinstance(node, list):
        return 1 + max((_depth(v) for v in node), default=0)
    return 0


def _inline(node: Any) -> str:
    return json.dumps(node, ensure_ascii=False, separators=(", ", ": "))


def _render(node: Any, indent: int, width: int) -> str:
    if not isinstance(node, (Mapping, list)) or not node:
        return _inline(node)
    flat = _inline(node)
    if _depth(node) <= 2 and indent + len(flat) <= width:
        return flat
    pad = " " * (indent + 2)
    close = " " * indent
    if isinstance(node, Mapping):
        items = [
            f"{pad}{json.dumps(key, ensure_ascii=False)}: {_render(value, indent + 2, width)}"
            for key, value in node.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + close + "}"
    items = [f"{pad}{_render(value, indent + 2, width)}" for value in node]
    return "[\n" + ",\n".join(items) + "\n" + close + "]"


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #

#: A fenced ``json`` block: the opening fence (any indentation, any info string
#: after ``json``), its body, the closing fence at the same indentation.
_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```json\b(?P<info>[^\n]*)\n(?P<body>.*?)\n(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def migrate_markdown(text: str) -> str:
    """``text`` with every fenced ``json`` block that is a whole document
    needing migration (:func:`needs_migration`) rewritten at the current
    version (the fence lines and their indentation kept). A block that is not
    JSON, already current, or one the rewrite refuses is left exactly as it
    is — a fragment with an ellipsis, a run receipt, a deliberately malformed
    example a guide shows to explain a refusal."""

    def rewrite(match: re.Match[str]) -> str:
        indent = match.group("indent")
        body = match.group("body")
        lines = body.split("\n")
        stripped = "\n".join(
            line[len(indent) :] if line.startswith(indent) else line for line in lines
        )
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError:
            return match.group(0)
        if not needs_migration(raw):
            return match.group(0)
        try:
            migrated = migrate_document(raw)
        except ParseError:
            return match.group(0)  # a split example, or a deliberately broken one
        rendered = format_document(migrated).rstrip("\n")
        body_out = "\n".join(indent + line for line in rendered.split("\n"))
        return f"{indent}```json{match.group('info')}\n{body_out}\n{indent}```"

    return _FENCE.sub(rewrite, text)


# --------------------------------------------------------------------------- #
# the verb
# --------------------------------------------------------------------------- #


def _rewrite(path: Path) -> str | None:
    """The migrated text of one file, or ``None`` when nothing would change.

    JSON and markdown only. A ``.yaml`` document is an authoring surface the
    loader reads, but a rewrite of it would drop its comments and formatting
    — the reasons to author YAML — so it is refused with that reason, and the
    author regroups it by hand (or converts it to JSON first).
    """
    if path.suffix in (".yaml", ".yml"):
        raise ParseError(
            "P2",
            "a YAML document is not rewritten in place: the migration would drop "
            "its comments and formatting. Regroup it by hand (§1), or convert it "
            "to JSON and migrate that",
            path=str(path),
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        out = migrate_markdown(text)
        return None if out == text else out
    raw = load_raw(text)
    if not needs_migration(raw):
        return None  # current version, or a workflow with nothing to re-spell
    return format_document(migrate_document(raw))


def main(args: argparse.Namespace) -> int:
    """``causalab migrate PATH... [--check]`` — rewrite in place, or with
    ``--check`` report what would change and exit 1 if anything would."""
    changed = 0
    refused = 0
    for path in args.paths:
        try:
            out = _rewrite(path)
        except (ProtocolError, OSError) as err:
            # a missing path or a directory is a refusal like a malformed
            # file: the files before it in `paths` stay migrated, and the
            # exit code says the run was not clean
            print(f"refused: {path}: {err}", file=sys.stderr)
            refused += 1
            continue
        if out is None:
            continue
        changed += 1
        if args.check:
            print(f"would migrate {path}")
        else:
            path.write_text(out, encoding="utf-8")
            print(f"migrated {path}")
    if refused:
        return 1
    if args.check and changed:
        return 1
    return 0
