"""The workflow's ``pins`` section: what a workflow touches, stamped into the
document itself, and held to on every later load (docs/workflow_protocol.md
§1, §5 rule 21, §7).

A workflow is the unit of reproducibility, and ``--resume`` is a workflow
verb: an intervention specification run on its own has no step boundaries
to resume at, and a table, a script or a code module has nothing to be
pinned *by* except the workflow that consumes it. So the workflow document
is where every pin lives — **not** a sidecar beside a dataset, not a stamp
inside an intervention specification, not a file beside the workflow. The
section is a census of the closure the load resolved:

.. code-block:: json

    "pins": {
      "documents": {"intervention_protocol.json": "<sha256 of the file>"},
      "scripts":   {"scripts/figure.py": "<sha256 of the module>",
                    "scripts/figure.py#helper.py": "<sha256 of a sibling it imports>"},
      "datasets":  {"data": "<content digest of the whole table>"},
      "code":      {"causalab.tasks.x.causal_models": "<sha256 of the module>"},
      "files":     {"params/basis.safetensors": "<content digest>"}
    }

Only non-empty categories are written; every key is the resource as the
document names it — a document path relative to the workflow file, a
script's authored locator (a sibling member as ``<script>#<member>``), a
dataset ref with its ``#split`` fragment stripped (the pin is the *table*),
a ``code`` reference's resolved module, an artifact's ``file_path`` as the
canonical form resolved it. A nested ``workflow`` step's document is pinned
by its own §7 digest rather than its bytes, because the inner file carries
its own ``pins`` section and stamping it must not move the outer's pin.

Three verbs over one census (:func:`collect_pins`):

* **stamp** — :func:`stamp_pins` writes the census into the file, as the
  last section. ``causalab pin <wf>`` does it on demand; ``causalab run``
  does it on the first run of an unpinned document, so a workflow never has
  to be stamped by hand before it is run once. A run under ``--set`` stamps
  nothing: the overrides describe a different closure than the file does.
* **check** — :func:`check_pins`, at load, so every door (``validate``,
  ``explain``, ``digest``, ``run``) refuses a stale pin before anything is
  compiled further: a pinned resource whose bytes moved, a resource the load
  touched that the section does not pin, and a pinned resource the load no
  longer touches are three distinct refusals, each naming
  ``pins.<category>.<key>`` (rule 21). The fix is deliberate and named in the
  refusal: ``causalab pin <wf>`` re-stamps after a change the author meant —
  the one verb a stale section cannot refuse (``load_workflow(hold_pins=
  False)``), because it exists to replace the section.
* **absent** — a document with no ``pins`` section loads and runs; the first
  ``run`` stamps it (above).

**Pins are not identity.** The section is excluded from the canonical form
(§7): a stamped and an unstamped copy of one workflow have the same step
identities, so stamping moves no shipped digest, no demo digest and no
``--resume`` decision. The identities already carry what the pins record —
a script step's ``script_sha256``, a protocol step's ``document_digest`` and
through it every dataset digest and ``code`` hash — and ``--resume``
compares those against the run tree. The pins compare the same facts
against the *document*, which is what lets an author state "this is the
closure I validated" and have the executor hold the next run to it.

Torch-free, engine-free; imports the document model and nothing above it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from causalab.protocol.loader import load_text
from causalab.protocol.resolve import split_dataset_ref
from causalab.workflow.document import (
    BehavioralStep,
    ProtocolStep,
    Reference,
    ScriptStep,
    WorkflowDocument,
    WorkflowError,
    WorkflowStep,
    producer_of,
)

__all__ = [
    "PINS_KEY",
    "PINS_RULE",
    "PIN_CATEGORIES",
    "check_pins",
    "collect_pins",
    "parse_pins",
    "stamp_pins",
]

#: The document key.
PINS_KEY: str = "pins"

#: The checklist rule a stale, missing or surplus pin is refused under (§5).
PINS_RULE: int = 21

#: The closed category vocabulary, in the order the section is written.
PIN_CATEGORIES: tuple[str, ...] = ("documents", "scripts", "datasets", "code", "files")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- #
# parse (rule 1: strict keys, closed categories, sha256 values)
# --------------------------------------------------------------------------- #


def parse_pins(raw: Any, path: str) -> dict[str, dict[str, str]]:
    """The authored section, checked for shape: an object whose keys are
    :data:`PIN_CATEGORIES` and whose values map a resource name to one
    sha256 hex digest. Nothing here reads a file — that is
    :func:`collect_pins` at load."""
    if not isinstance(raw, Mapping):
        raise WorkflowError(1, "'pins' is an object of categories", path=path)
    out: dict[str, dict[str, str]] = {}
    for category, entries in raw.items():
        if category not in PIN_CATEGORIES:
            raise WorkflowError(
                1,
                f"unknown pins category {category!r}; the categories are "
                f"{list(PIN_CATEGORIES)}",
                path=f"{path}.{category}",
            )
        if not isinstance(entries, Mapping):
            raise WorkflowError(
                1, f"'pins.{category}' maps a resource to its digest", path=path
            )
        pinned: dict[str, str] = {}
        for key, digest in entries.items():
            if not isinstance(digest, str) or not _HEX64.match(digest):
                raise WorkflowError(
                    1,
                    f"pins.{category}.{key}: a pin is one sha256 hex digest, "
                    f"got {digest!r}",
                    path=f"{path}.{category}",
                )
            pinned[str(key)] = digest
        out[str(category)] = pinned
    return out


# --------------------------------------------------------------------------- #
# the census
# --------------------------------------------------------------------------- #


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table_digest(datasets: Any, base: str) -> str:
    """The whole table's digest — the pin is the *table*, whichever split a
    document reads. A resolver that knows the distinction
    (:meth:`~causalab.protocol.resolve.FileDatasets.table_digest`) answers
    it; one that does not (a store of unsplit tables) digests the ref."""
    whole = getattr(datasets, "table_digest", None)
    return whole(base) if whole is not None else datasets.digest(base)


def _walk(node: Any, code: dict[str, str], files: dict[str, str], step_names) -> None:
    """One pass over a canonical intervention document: every ``code``
    reference (``source_module`` / ``source_sha256`` and its sibling closure)
    and every resolved artifact (``file_path`` / ``content_digest``, and a
    code reference's ``data_inputs``) that is not a run-tree product."""
    if isinstance(node, Mapping):
        module = node.get("source_module")
        if isinstance(module, str) and isinstance(node.get("source_sha256"), str):
            code[module] = node["source_sha256"]
            closure = node.get("closure")
            if isinstance(closure, Mapping):
                for member, digest in closure.items():
                    code[f"{module}#{member}"] = str(digest)
            inputs = node.get("data_inputs")
            digests = node.get("data_input_digests")
            if isinstance(inputs, Mapping) and isinstance(digests, Mapping):
                for name, target in inputs.items():
                    if name in digests and producer_of(str(target), step_names) is None:
                        files[str(target)] = str(digests[name])
        file_path = node.get("file_path")
        digest = node.get("content_digest")
        if isinstance(file_path, str) and isinstance(digest, str):
            if producer_of(file_path, step_names) is None:
                files[file_path] = digest
        for value in node.values():
            _walk(value, code, files, step_names)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _walk(value, code, files, step_names)


def collect_pins(
    document: WorkflowDocument,
    workflow_dir: Path,
    inner: Mapping[str, Any],
    nested: Mapping[str, Any],
    datasets: Any,
    step_names,
) -> dict[str, dict[str, str]]:
    """The census: what this load touched, by category, keys sorted, empty
    categories omitted — the value a stamp writes and a check compares.

    ``document`` is the authored document (the outer's own steps — a
    fan-out's children and a nested workflow's steps are derived and pinned
    through their parent); ``inner`` the compiled protocol and behavioral
    documents by step; ``nested`` the loaded inner workflows by step;
    ``datasets`` the run's resolver; ``step_names`` every step of the
    flattened run, so a run-tree path is recognised as a product and never
    pinned as a file."""
    documents: dict[str, str] = {}
    scripts: dict[str, str] = {}
    tables: dict[str, str] = {}
    code: dict[str, str] = {}
    files: dict[str, str] = {}
    for name, step in document.steps.items():
        if isinstance(step, (ProtocolStep, BehavioralStep)):
            documents[step.document] = _file_sha256(workflow_dir / step.document)
            loaded = inner.get(name)
            compiled = getattr(loaded, "compiled", None)
            if compiled is not None:
                for ref in compiled.data:
                    base, _fragment = split_dataset_ref(str(ref))
                    if base not in tables:
                        tables[base] = _table_digest(datasets, base)
                _walk(compiled.canonical, code, files, step_names)
        elif isinstance(step, WorkflowStep):
            documents[step.document] = nested[name].digest
        elif isinstance(step, ScriptStep):
            scripts[step.script] = step.script_sha256
            for member, digest in step.closure.items():
                scripts[f"{step.script}#{member}"] = digest
            for value in step.inputs.values():
                if not isinstance(value, Reference) or value.path is None:
                    continue
                target = str(value.path)
                if target.startswith("/"):
                    continue  # rule 4: not existence-checked at load, not pinned
                files[target] = _file_sha256(workflow_dir / target)
    census = {
        "documents": documents,
        "scripts": scripts,
        "datasets": tables,
        "code": code,
        "files": files,
    }
    return {
        category: dict(sorted(entries.items()))
        for category in PIN_CATEGORIES
        if (entries := census[category])
    }


# --------------------------------------------------------------------------- #
# the check (rule 21)
# --------------------------------------------------------------------------- #


def check_pins(
    authored: Mapping[str, Mapping[str, str]],
    actual: Mapping[str, Mapping[str, str]],
) -> None:
    """Hold the authored section to the census, exactly: every pinned
    resource is touched with the pinned digest, and every touched resource
    is pinned. The first disagreement, in category then key order, is the
    refusal — three distinct facts, each naming ``pins.<category>.<key>``."""
    for category in PIN_CATEGORIES:
        want = authored.get(category, {})
        got = actual.get(category, {})
        for key in sorted(set(want) | set(got)):
            where = f"pins.{category}.{key}"
            if key not in want:
                raise WorkflowError(
                    PINS_RULE,
                    f"the workflow touches {key!r} but does not pin it "
                    f"(its digest is {got[key][:12]}…). The pins section describes "
                    "another closure than the one this document loads; re-stamp it "
                    "with `causalab pin <workflow>` if the change was meant (§7)",
                    path=where,
                )
            if key not in got:
                raise WorkflowError(
                    PINS_RULE,
                    f"pinned, but nothing in the workflow touches {key!r} "
                    "any more. Re-stamp the section with `causalab pin <workflow>` "
                    "if the change was meant (§7)",
                    path=where,
                )
            if want[key] != got[key]:
                raise WorkflowError(
                    PINS_RULE,
                    f"{key!r} moved — pinned {want[key][:12]}…, but the "
                    f"bytes the load resolved digest to {got[key][:12]}…. The "
                    "workflow was stamped against another version of this "
                    "resource; re-stamp it with `causalab pin <workflow>` if the "
                    "change was meant, else restore the resource (§7)",
                    path=where,
                )


# --------------------------------------------------------------------------- #
# the stamp
# --------------------------------------------------------------------------- #


def _indent_of(text: str) -> int:
    """The indent the file already uses (its first indented line), else 2."""
    match = re.search(r"^( +)\S", text, re.M)
    return len(match.group(1)) if match else 2


def stamp_pins(path: Path, pins: Mapping[str, Mapping[str, str]]) -> Path:
    """Write ``pins`` into the workflow file at ``path`` as its last section,
    replacing any section already there, and return ``path``.

    A JSON document keeps its key order, indent and escaping style (ASCII
    files stay ASCII); a YAML document is rewritten by the YAML emitter,
    which keeps keys and values and drops comments — JSON is the normative
    surface (IM spec §9), and a YAML author who wants comments kept stamps
    by hand from ``causalab explain``.
    """
    text = path.read_text()
    raw: dict[str, Any] = dict(load_text(path))
    raw.pop(PINS_KEY, None)
    raw[PINS_KEY] = {
        category: dict(entries) for category, entries in pins.items() if entries
    }
    if path.suffix in (".yaml", ".yml"):
        import yaml  # the optional authoring surface, as in `load_text`

        path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
        return path
    body = json.dumps(raw, indent=_indent_of(text), ensure_ascii=text.isascii())
    path.write_text(body + ("\n" if text.endswith("\n") or not text else ""))
    return path
