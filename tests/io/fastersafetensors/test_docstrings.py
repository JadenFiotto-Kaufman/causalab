"""Every attribute docstring sits directly under its assignment.

A constant inserted between a name and its docstring leaves a bare string
expression that documents nothing; no linter flags it (``B018`` exempts string
literals so that docstrings pass), and it has happened twice in review.
"""

import ast
from pathlib import Path

import pytest

import causalab.io.fastersafetensors as fastersafetensors

pytestmark = pytest.mark.unit

MODULES = sorted(Path(fastersafetensors.__file__).parent.glob("*.py*"))  # .py and .pyi
assert MODULES, "an empty parametrize would skip, not fail"


def _orphans(body: list[ast.stmt]) -> list[int]:
    """Line numbers of string expressions in ``body`` (after the docstring) not
    directly preceded by an assignment."""
    found = []
    for i, node in enumerate(body):
        if i == 0 or not isinstance(node, ast.Expr):
            continue
        if not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            continue
        if not isinstance(body[i - 1], ast.Assign | ast.AnnAssign):
            found.append(node.lineno)
    return found


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_attribute_docstrings_follow_their_assignment(module: Path) -> None:
    tree = ast.parse(module.read_text(), filename=str(module))
    orphans = _orphans(tree.body)
    for node in ast.walk(tree):
        # a function body's first statement is its docstring (skipped); any
        # other bare string there is dead code, so flagging it is right too
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            orphans += _orphans(node.body)
    assert orphans == [], (
        f"{module.name}: string expressions documenting nothing at lines {orphans}"
    )
