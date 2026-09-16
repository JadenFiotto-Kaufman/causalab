"""Every module-level third-party import in ``causalab/`` is a declared
runtime dependency.

The failure this pins was invisible from a checkout: ``receptive_field.py``
imported ``plotly`` at module scope while ``pyproject.toml`` never declared it.
``uv sync`` installs the dev group, whose ``dash`` depends on plotly, so every
test and every ``uv run causalab`` saw it — and a ``uv pip install`` of the
wheel did not, so a user's ``import causalab.io.plots.receptive_field`` was a
``ModuleNotFoundError`` and, on the pre-PEP-562 ``plots/__init__``, so was
every demo workflow with a figure step (``find_spec`` on a ``module:`` locator
imports the parent package).

The rule is the one ``pyproject.toml`` already states for ``scipy``: "an op's
dependency belongs in the record of what this package requires, not in another
library's install tree." A package imported at module scope must be in
``[project.dependencies]``. A package an *extra* provides (dash, nnsight) may
only be imported lazily, inside the function that needs it — which is also
what keeps ``import causalab.io.plots`` headless (``test_notebooks_extra.py``)
and ``validate`` torch-free (``test_load_is_torch_free.py``).

Import names are mapped to distributions through the installed metadata
(``RECORD``), not ``top_level.txt``, which numpy, matplotlib and pandas no
longer ship. The check is therefore about *this* environment's install of each
distribution — the dev environment, whose dependency closure is a superset of
the wheel's — but the assertion is against the *declaration*, so a transitive
package imported directly is refused even though it would import here.
"""

from __future__ import annotations

import ast
import sys
from importlib.metadata import distributions
from pathlib import Path

import pytest

from tests._helpers.pyproject import REPO, core_dependencies, normalize

pytestmark = pytest.mark.unit

PACKAGE = REPO / "causalab"

#: Import names that resolve to no distribution: the standard library, and
#: the package under test.
_STDLIB = frozenset(sys.stdlib_module_names) | {"__future__"}


def _import_name_to_distributions() -> dict[str, set[str]]:
    """``{top-level import name: {normalized distribution names}}`` from every
    installed distribution's file list."""
    table: dict[str, set[str]] = {}
    for dist in distributions():
        name = normalize(dist.metadata["Name"])
        for file in dist.files or ():
            top = file.parts[0]
            if top.endswith((".dist-info", ".data")) or top == "__pycache__":
                continue
            if top.endswith(".py"):
                top = top[: -len(".py")]
            elif "." in top and len(file.parts) == 1:
                continue  # a .pth, a .so at the root, a license
            table.setdefault(top, set()).add(name)
    return table


def module_level_imports(source: Path) -> set[str]:
    """Top-level import names bound at module scope — ``import a.b`` and
    ``from a.b import c`` directly in the module body. An import inside a
    function, a ``try`` or an ``if`` is a deliberate deferral and is not one."""
    names: set[str] = set()
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_every_module_level_import_is_a_declared_runtime_dependency() -> None:
    declared = core_dependencies()
    table = _import_name_to_distributions()
    undeclared: dict[str, list[str]] = {}
    for module in sorted(PACKAGE.rglob("*.py")):
        for name in module_level_imports(module):
            if name in _STDLIB or name == "causalab":
                continue
            providers = table.get(name)
            assert providers, (
                f"{module.relative_to(REPO)} imports {name!r}, which no installed "
                "distribution provides — the dev environment cannot even run it"
            )
            if not providers & declared:
                undeclared.setdefault(
                    f"{name} ({', '.join(sorted(providers))})", []
                ).append(str(module.relative_to(REPO)))
    assert not undeclared, (
        "imported at module scope but not in [project.dependencies] — declare it, "
        "or import it inside the function that needs it if an extra provides it:\n"
        + "\n".join(f"  {name}: {files}" for name, files in sorted(undeclared.items()))
    )
