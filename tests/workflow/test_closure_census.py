"""The shipped script modules' repository import closures, frozen (workflow
spec §4.2) — the layering census.

The repository-wide walk (``import_closure`` with ``repository=True``) is how
the suite proves what a shipped script reaches: the protocol core and the step
I/O, never an engine, never numerics, never a layer the script has no business
importing (``events.py``, ``derived.py``, ``fan_out.py``, ``nested.py`` —
each has a guard below or in its own test module). An import added to
``causalab/io/step_io.py`` widens what every shipped script pulls into a
torch-free ``validate``, so each hashed module's closure is listed here, member
by member, and compared. A change is a deliberate edit to this table, never a
drift.

None of it is identity. The *identity* walk (``repository=False``) drops every
member under the ``causalab`` package — those bytes are runtime identity, the
``tree_digest`` every step record carries and ``--resume`` compares (§7) — so
a ``{"module": …}`` step carries no closure keys at all, and no edit to a
module in this table moves a shipped or demo workflow's digest. The pins
below hold that: every shipped and demo script entry names a module and
carries neither key; the parent-package switch is **off** (a parent
``__init__`` is executed, not declared); and the isolated-runtime script with
no repository imports carries no closure keys either (valid work is not
refused and gets an identity that says so).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.code import closure_sha256, import_closure
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.workflow.document import load_workflow

from tests.protocol.test_vocabulary_census import HASHED_SCRIPTS
from tests.workflow.test_isolation import (
    SCRIPT as ISOLATED_SCRIPT,
    _document as isolated_document,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / "causalab" / "configs" / "workflows"
DEMOS = REPO / "demos"

#: The modules every shipped script reaches: ``io/step_io.py`` and, through
#: it, the resolve → registry → schema chain of the protocol core (``schema.py``
#: imports ``estimand.py``, so the estimand vocabulary is a member, and
#: ``spans.py`` / ``segments.py``, so the span grammar is one).
SHARED: tuple[str, ...] = (
    # the safetensors implementation (the Python surface over the Rust
    # extension `_core`, which has no `.py` and so is not a member — imported
    # as `import …._core as _core`, since `from . import _core` would resolve
    # to the package `__init__`): reached through tensor_files.py below,
    # torch-importing modules every one — which is why step_io imports
    # tensor_files function-locally, so a torch-free `validate` still executes
    # none of it
    "causalab/io/fastersafetensors/_buffers.py",
    "causalab/io/fastersafetensors/_distributed.py",
    "causalab/io/fastersafetensors/_dtypes.py",
    "causalab/io/fastersafetensors/_files.py",
    "causalab/io/fastersafetensors/_header.py",
    "causalab/io/fastersafetensors/_select.py",
    "causalab/io/fastersafetensors/_serialize.py",
    "causalab/io/fastersafetensors/_slices.py",
    "causalab/io/fastersafetensors/errors.py",
    "causalab/io/fastersafetensors/torch.py",
    "causalab/io/step_io.py",
    # the one import for the repository's tensor files; reached
    # function-locally from step_io
    "causalab/io/tensor_files.py",
    "causalab/protocol/bundles.py",
    "causalab/protocol/errors.py",
    "causalab/protocol/estimand.py",
    "causalab/protocol/registry.py",
    "causalab/protocol/resolve.py",
    "causalab/protocol/schema.py",
    "causalab/protocol/segments.py",
    "causalab/protocol/shapes.py",
    "causalab/protocol/spans.py",
    "causalab/protocol/sweep.py",
    "causalab/protocol/tables.py",
    "causalab/tables.py",
)

#: hashed module → its declared closure, sorted by path. Exactly the set the
#: vocabulary census exempts (``HASHED_SCRIPTS``), asserted below.
CLOSURES: dict[str, tuple[str, ...]] = {
    "causalab/analysis/fit_pca.py": SHARED,
    "causalab/analysis/harvest_difference.py": SHARED,
    "causalab/analysis/head_stats.py": SHARED,
    "causalab/io/plots/workflow_figures.py": tuple(
        sorted(
            (
                *SHARED,
                "causalab/io/plots/figure_format.py",
                "causalab/io/step_record.py",
            )
        )
    ),
    "causalab/workflow/scripts/select.py": tuple(
        sorted((*SHARED, "causalab/io/step_record.py"))
    ),
}

#: The built-in reduction step: a hashed module named by no shipped workflow —
#: frozen all the same, because its closure is what a `reduce` step pulls into
#: a torch-free load.
REDUCE = "causalab/workflow/scripts/reduce.py"
REDUCE_CLOSURE: tuple[str, ...] = tuple(
    sorted((*SHARED, "causalab/workflow/reduction.py"))
)


def _closure(module: str, **kwargs: Any) -> dict[str, str]:
    return import_closure(REPO / module, root=REPO, **kwargs)


def test_fan_out_is_in_no_hashed_closure() -> None:
    """§2.9: the fan-out layer is a member of no hashed script's
    closure, of the reduce script's and of SHARED — the same guard
    `test_conditional.py` holds for conditional.py, so no digest moves."""
    module = "causalab/workflow/fan_out.py"
    assert module not in SHARED
    for hashed in (*CLOSURES, REDUCE):
        assert module not in _closure(hashed), hashed


def test_nested_is_in_no_hashed_closure() -> None:
    """§2.10: the nested-workflow layer is a member of no hashed
    script's closure, of the reduce script's and of SHARED — the same guard as
    for conditional.py and fan_out.py, so no digest moves."""
    module = "causalab/workflow/nested.py"
    assert module not in SHARED
    for hashed in (*CLOSURES, REDUCE):
        assert module not in _closure(hashed), hashed


def _demo_env(document: Path) -> ResolutionEnv:
    demo = document.parents[1]
    return ResolutionEnv(
        datasets=FileDatasets(root=demo / "data"),
        artifacts=FileArtifacts(root=REPO),
    )


# --------------------------------------------------------------------------- #
# the frozen table
# --------------------------------------------------------------------------- #


def test_the_frozen_table_is_the_hashed_set() -> None:
    assert set(CLOSURES) == HASHED_SCRIPTS


@pytest.mark.parametrize("module", sorted(CLOSURES), ids=lambda m: Path(m).stem)
def test_a_hashed_module_has_the_frozen_closure(module: str) -> None:
    got = tuple(_closure(module))
    want = CLOSURES[module]
    assert got == want, (
        f"{module}'s repository import closure moved — what a torch-free "
        "`validate` pulls in changed. If the import is right, update this table "
        "deliberately (docs/workflow_protocol.md §4.2); no digest moves with it.\n"
        f"new members: {sorted(set(got) - set(want))}\n"
        f"gone: {sorted(set(want) - set(got))}"
    )


@pytest.mark.parametrize(
    "module", sorted(CLOSURES) + [REDUCE], ids=lambda m: Path(m).stem
)
def test_the_identity_walk_admits_nothing_from_the_package(module: str) -> None:
    """The two walks, side by side on every hashed module: the layering walk
    reaches the frozen table; the identity walk — the one the loaders call —
    reaches nothing, because every member is under the package and the
    package is runtime identity (§4.2, §7)."""
    assert _closure(module), "the layering walk found nothing"
    assert _closure(module, repository=False) == {}


def test_the_reduce_script_has_the_frozen_closure() -> None:
    assert tuple(_closure(REDUCE)) == REDUCE_CLOSURE


def test_every_member_is_a_tracked_module_under_the_repo_root() -> None:
    """The manifest keys are repo-relative paths to files that exist, so a
    reader can open the file a moved digest names."""
    for members in (*CLOSURES.values(), REDUCE_CLOSURE):
        for member in members:
            assert (REPO / member).is_file(), member
            assert member.startswith("causalab/") and member.endswith(".py")


# --------------------------------------------------------------------------- #
# the switch, pinned off
# --------------------------------------------------------------------------- #


def test_parent_packages_are_not_declared(env: Any) -> None:
    """The declared closure has no ``__init__.py`` in it; the switch adds
    them — ``causalab/protocol/__init__.py`` and what it eagerly imports — and
    the loader does not use the switch."""
    select = "causalab/workflow/scripts/select.py"
    declared = _closure(select)
    assert not any(member.endswith("__init__.py") for member in declared)

    executed = _closure(select, include_parents=True)
    assert set(declared) < set(executed)
    assert "causalab/protocol/__init__.py" in executed
    assert "causalab/protocol/loader.py" in executed, (
        "the eager fan-out of protocol/__init__.py should follow the parents"
    )

    loaded = load_workflow(WORKFLOWS / "weekdays_8b.json", env)
    assert "closure" not in loaded.canonical["steps"]["best"]


# --------------------------------------------------------------------------- #
# valid work
# --------------------------------------------------------------------------- #


def test_an_isolated_script_carries_no_closure_keys(tmp_path: Path, env: Any) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "count.py").write_text(ISOLATED_SCRIPT)
    document = isolated_document(tmp_path, deps=["packaging"])
    loaded = load_workflow(document, env, workflow_dir=tmp_path)
    entry = loaded.canonical["steps"]["count"]
    assert "closure" not in entry and "closure_sha256" not in entry
    assert closure_sha256({}) == hashlib.sha256(b"").hexdigest()  # the fixed value
    assert load_workflow(document, env, workflow_dir=tmp_path).digest == loaded.digest


SHIPPED = sorted(WORKFLOWS.glob("*.json"))
DEMO_WORKFLOWS = sorted(DEMOS.glob("*/workflows/*.json"))


def test_the_workflow_census_found_something() -> None:
    assert len(SHIPPED) >= 2 and len(DEMO_WORKFLOWS) >= 9


@pytest.mark.parametrize(
    "path", SHIPPED + DEMO_WORKFLOWS, ids=[p.stem for p in SHIPPED + DEMO_WORKFLOWS]
)
def test_every_shipped_and_demo_workflow_loads_with_no_closure_keys(
    path: Path, env: Any
) -> None:
    """Every script step names a module in the frozen table and carries neither
    closure key — so no edit to the package moves its digest — and no other
    kind of entry carries one either."""
    loaded = load_workflow(path, env if path.parent == WORKFLOWS else _demo_env(path))
    for entry in loaded.canonical["steps"].values():
        assert "closure" not in entry and "closure_sha256" not in entry
        if entry["type"] == "script":
            module = entry["script"]["module"].replace(".", "/") + ".py"
            assert module in CLOSURES, module
