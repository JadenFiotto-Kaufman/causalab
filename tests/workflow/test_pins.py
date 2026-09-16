"""The workflow's ``pins`` section (workflow spec §1, §5 rule 21, §7).

Pins are the one place a digest of something a run touches is *authored*: a
workflow stamps the census of its closure — documents, scripts (and their
sibling members), tables, ``code`` modules, files — into itself, and every
later load holds the document to it. Nothing beside a dataset, and nothing
inside an intervention specification, carries a pin; ``--resume`` is a
workflow verb. What is pinned here:

* **the census** — a protocol step pins its document's bytes and the whole
  table it reads (fragment stripped); a script step pins its module and
  every sibling in its closure, and a relative ``path`` input as a file; an
  absolute path is not pinned (rule 4 defers it to run time);
* **stamping** — ``stamp_pins`` writes the section last, keeps the file's
  indent and key order, and is idempotent; a stamped document loads clean;
* **rule 21** — a moved table, a moved document, a moved script, an unpinned
  resource and a pin nothing touches any more are each refused naming
  ``pins.<category>.<key>``; the refusal reaches every door (``validate``);
* **not identity** — a stamped and an unstamped copy have equal step
  identities and equal canonical forms, so stamping busts no ``--resume``;
* **the CLI** — ``pin`` stamps; ``run`` stamps an unpinned document on its
  first run and refuses to under ``--set``; ``pin`` and ``--resume`` on an
  intervention specification are refused; ``explain`` reports the state;
* **the census of the spec** — §5 numbers rule 21 and §1 lists the section.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks import TASKS_ROOT
from causalab.workflow.document import (
    MAX_RULE,
    SECTION_ORDER,
    WorkflowError,
    load_workflow,
)
from causalab.workflow.pins import (
    PIN_CATEGORIES,
    PINS_KEY,
    PINS_RULE,
    check_pins,
    collect_pins,
    parse_pins,
    stamp_pins,
)
from tests.protocol._env import FIXTURES

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
LOCATE = REPO / "causalab/configs/protocols/weekdays_locate_scan.json"
SPEC = REPO / "docs/workflow_protocol.md"
TABLE_REF = "natural_domains_arithmetic/data/weekdays"
SCRIPT = (
    "import json\nfrom pathlib import Path\n\n\n"
    "def main(inputs, outputs):\n"
    "    Path(outputs['out']).write_text(json.dumps([{'n': 1}]))\n"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _document() -> dict[str, Any]:
    return {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "locate": {
                "type": "intervention_protocol",
                "document": "methods/locate.json",
            },
            "count": {
                "type": "script",
                "script": {"path": "scripts/count.py"},
                "inputs": {"notes": {"path": "notes.json"}},
                "outputs": {"out": {"file": "count.json", "columns": {"n": "int64"}}},
            },
        },
    }


@pytest.fixture
def site(tmp_path: Path) -> dict[str, Any]:
    """A workflow directory with its own copy of everything it touches — the
    method preset, a script, a file input and the shipped table it reads
    (shadowing the shipped one root-first, so a test can move a byte)."""
    wf_dir = tmp_path / "wf"
    (wf_dir / "methods").mkdir(parents=True)
    shutil.copyfile(LOCATE, wf_dir / "methods" / "locate.json")
    (wf_dir / "scripts").mkdir()
    (wf_dir / "scripts" / "count.py").write_text(SCRIPT)
    (wf_dir / "notes.json").write_text('{"k": 1}\n')
    data = tmp_path / "data"
    table = data / f"{TABLE_REF}.json"
    table.parent.mkdir(parents=True)
    shutil.copyfile(TASKS_ROOT / f"{TABLE_REF}.json", table)
    workflow = wf_dir / "workflow.json"
    workflow.write_text(json.dumps(_document(), indent=2) + "\n")
    env = ResolutionEnv(
        datasets=FileDatasets(
            root=data, fallback_roots=(FIXTURES / "data", TASKS_ROOT)
        ),
        artifacts=FileArtifacts(root=tmp_path),
    )
    return {
        "dir": wf_dir,
        "workflow": workflow,
        "table": table,
        "env": env,
        "data": data,
    }


def _load(site: dict[str, Any]):
    return load_workflow(site["workflow"], site["env"])


# --------------------------------------------------------------------------- #
# the census
# --------------------------------------------------------------------------- #


def test_the_census_names_every_resource_the_load_touched(site) -> None:
    loaded = _load(site)
    pins = loaded.pins
    assert set(pins) == {"documents", "scripts", "datasets", "files"}
    assert pins["documents"] == {
        "methods/locate.json": _sha256(site["dir"] / "methods/locate.json")
    }
    assert pins["scripts"] == {
        "scripts/count.py": _sha256(site["dir"] / "scripts/count.py")
    }
    # the pin is the *table*: the fragment the document reads is stripped
    assert pins["datasets"] == {TABLE_REF: _sha256(site["table"])}
    assert pins["files"] == {"notes.json": _sha256(site["dir"] / "notes.json")}
    # an unpinned document loads: the section is None, the census is there
    assert loaded.document.pins is None


def test_an_absolute_path_input_is_not_pinned(site, tmp_path: Path) -> None:
    """Rule 4 defers an absolute path to run time; a pin over it would be a
    claim the load cannot make on another host."""
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}")
    doc = _document()
    doc["steps"]["count"]["inputs"]["notes"] = {"path": str(elsewhere)}
    site["workflow"].write_text(json.dumps(doc))
    assert "files" not in _load(site).pins


def test_a_sibling_the_script_imports_is_pinned_under_the_script(site) -> None:
    helper = site["dir"] / "scripts" / "pins_probe_helper.py"
    helper.write_text("def value():\n    return 1\n")
    (site["dir"] / "scripts" / "count.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent))\n"
        "import pins_probe_helper  # noqa: E402\n\n\n"
        "def main(inputs, outputs):\n    pass\n"
    )
    pins = _load(site).pins["scripts"]
    assert pins["scripts/count.py#pins_probe_helper.py"] == _sha256(helper)


# --------------------------------------------------------------------------- #
# stamping
# --------------------------------------------------------------------------- #


def test_stamp_writes_the_section_last_and_the_document_loads_clean(site) -> None:
    before = json.loads(site["workflow"].read_text())
    stamp_pins(site["workflow"], _load(site).pins)
    raw = json.loads(site["workflow"].read_text())
    assert list(raw) == [*before, PINS_KEY] == [k for k in SECTION_ORDER if k in raw]
    assert raw[PINS_KEY] == _load(site).pins
    stamped = _load(site)
    assert stamped.document.pins == raw[PINS_KEY]
    # idempotent, byte for byte, and the file's indent survives
    text = site["workflow"].read_text()
    stamp_pins(site["workflow"], stamped.pins)
    assert site["workflow"].read_text() == text
    assert text.startswith('{\n  "version"') and text.endswith("\n")


def test_pins_are_not_identity(site) -> None:
    """A stamped and an unstamped copy of one workflow have the same step
    identities and the same canonical form: stamping moves no digest and
    busts no ``--resume``."""
    unstamped = _load(site)
    stamp_pins(site["workflow"], unstamped.pins)
    stamped = _load(site)
    assert stamped.step_digests == unstamped.step_digests
    assert stamped.inner_digests == unstamped.inner_digests
    assert stamped.canonical == unstamped.canonical
    assert stamped.digest == unstamped.digest
    assert PINS_KEY not in stamped.canonical


# --------------------------------------------------------------------------- #
# rule 21
# --------------------------------------------------------------------------- #


def _stamp(site) -> None:
    stamp_pins(site["workflow"], _load(site).pins)


def test_a_moved_table_is_refused_naming_the_dataset(site) -> None:
    _stamp(site)
    rows = json.loads(site["table"].read_text())
    rows[0]["input"] += " "
    site["table"].write_text(json.dumps(rows))
    with pytest.raises(
        WorkflowError,
        match=rf"W{PINS_RULE}.*pins\.datasets\.{re.escape(TABLE_REF)}.*moved",
    ) as info:
        _load(site)
    assert info.value.rule == PINS_RULE


def test_a_moved_document_and_a_moved_script_are_refused(site) -> None:
    _stamp(site)
    locate = site["dir"] / "methods" / "locate.json"
    locate.write_text(locate.read_text() + "\n")
    with pytest.raises(
        WorkflowError, match=r"pins\.documents\.methods/locate\.json.*moved"
    ):
        _load(site)
    locate.write_text(locate.read_text()[:-1])
    (site["dir"] / "scripts" / "count.py").write_text(SCRIPT + "# edited\n")
    with pytest.raises(WorkflowError, match=r"pins\.scripts\.scripts/count\.py.*moved"):
        _load(site)


def test_an_unpinned_resource_and_a_pin_nothing_touches_are_refused(site) -> None:
    _stamp(site)
    raw = json.loads(site["workflow"].read_text())
    del raw[PINS_KEY]["files"]["notes.json"]
    site["workflow"].write_text(json.dumps(raw))
    with pytest.raises(WorkflowError, match=r"pins\.files\.notes\.json.*does not pin"):
        _load(site)
    raw[PINS_KEY]["files"]["notes.json"] = "0" * 64
    raw[PINS_KEY]["files"]["gone.json"] = "0" * 64
    site["workflow"].write_text(json.dumps(raw))
    with pytest.raises(
        WorkflowError, match=r"pins\.files\.gone\.json.*nothing in the workflow touches"
    ):
        _load(site)


def test_the_section_is_strict(site) -> None:
    with pytest.raises(WorkflowError, match=r"unknown pins category 'models'") as info:
        parse_pins({"models": {}}, "pins")
    assert info.value.rule == 1
    with pytest.raises(WorkflowError, match="one sha256 hex digest"):
        parse_pins({"datasets": {"x": "abc"}}, "pins")
    assert parse_pins({}, "pins") == {}
    raw = _document()
    raw[PINS_KEY] = {"datasets": {"x": "not-a-digest"}}
    site["workflow"].write_text(json.dumps(raw))
    with pytest.raises(WorkflowError, match="one sha256 hex digest"):
        _load(site)


def test_check_is_exact_and_category_ordered() -> None:
    actual = {"documents": {"a.json": "1" * 64}, "datasets": {"t": "2" * 64}}
    check_pins(actual, actual)
    with pytest.raises(WorkflowError, match=r"pins\.documents\.a\.json"):
        check_pins({"datasets": {"t": "2" * 64}}, actual)  # documents come first
    assert PIN_CATEGORIES == ("documents", "scripts", "datasets", "code", "files")


def test_collect_omits_empty_categories(site) -> None:
    loaded = _load(site)
    census = collect_pins(
        loaded.document,
        site["dir"],
        loaded.inner,
        loaded.nested,
        site["env"].datasets,
        frozenset(loaded.document.steps),
    )
    assert "code" not in census and all(census.values())


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #


def _argv(verb: str, site, *extra: str) -> list[str]:
    return [
        verb,
        str(site["workflow"]),
        "--data-root",
        str(site["data"]),
        "--artifacts-root",
        str(site["dir"].parent),
        *extra,
    ]


def test_pin_stamps_and_validate_then_holds_the_document_to_it(site, capsys) -> None:
    assert main(_argv("pin", site)) == 0
    out = capsys.readouterr().out
    assert "pinned" in out and "1 dataset" in out and "1 document" in out
    assert json.loads(site["workflow"].read_text())[PINS_KEY] == _load(site).pins
    assert main(_argv("validate", site)) == 0
    assert main(_argv("explain", site)) == 0
    assert "pins      checked" in capsys.readouterr().out
    rows = json.loads(site["table"].read_text())
    rows[0]["input"] += " "
    site["table"].write_text(json.dumps(rows))
    assert main(_argv("validate", site)) == 1
    assert f"[W{PINS_RULE}]" in capsys.readouterr().err
    # `pin` is the one verb a stale section cannot refuse — it exists to
    # replace it — and the re-stamped document validates again
    assert main(_argv("pin", site)) == 0
    # the pin is the table's *canonical* digest, not the file's bytes — the
    # tampered file above was written compact
    assert json.loads(site["workflow"].read_text())[PINS_KEY]["datasets"] == {
        TABLE_REF: site["env"].datasets.table_digest(TABLE_REF)
    }
    assert main(_argv("validate", site)) == 0


def test_explain_says_how_an_unpinned_document_gets_pinned(site, capsys) -> None:
    assert main(_argv("explain", site)) == 0
    assert "pins      none" in capsys.readouterr().out


def test_pin_refuses_set(site, capsys) -> None:
    assert main(_argv("pin", site, "--set", "output_dir=elsewhere")) == 1
    assert "--set describes another closure" in capsys.readouterr().err
    assert PINS_KEY not in json.loads(site["workflow"].read_text())


def test_pin_and_resume_are_refused_on_an_intervention_specification(
    site, tmp_path: Path, capsys
) -> None:
    doc = site["dir"] / "methods" / "locate.json"
    common = ["--data-root", str(site["data"]), "--artifacts-root", str(tmp_path)]
    assert main(["pin", str(doc), *common]) == 1
    assert "pins are a workflow's" in capsys.readouterr().err
    assert (
        main(["run", str(doc), "--out", str(tmp_path / "o"), "--resume", *common]) == 1
    )
    assert "--resume is a workflow flag" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the spec's census
# --------------------------------------------------------------------------- #


def test_rule_21_is_numbered_and_the_section_is_listed() -> None:
    assert PINS_RULE == 21 and MAX_RULE >= PINS_RULE
    text = SPEC.read_text()
    section = text[text.index("## 5. Validation") : text.index("## 6. Derived")]
    items = {int(n) for n in re.findall(r"^(\d+)\. ", section, re.M)}
    assert PINS_RULE in items and "pins" in re.search(
        rf"^{PINS_RULE}\. (.+)$", section, re.M
    ).group(1)
    layout = text[text.index("## 1. Document layout") : text.index("### 1.1")]
    assert re.search(r"^\| 5 \| `pins` \|", layout, re.M)
    assert SECTION_ORDER[-1] == PINS_KEY
