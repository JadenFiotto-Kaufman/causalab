"""Read ``pyproject.toml``'s requirement arrays without a TOML parser.

``tomllib`` is 3.11 and this project floors at 3.10, so the two tests that
compare the tree against its declared dependencies
(``tests/io/plots/test_notebooks_extra.py``, ``tests/test_declared_dependencies.py``)
share this hand-rolled read instead of each carrying one. Comment lines go
first so a package named in prose inside one cannot be mistaken for a
requirement.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"


def normalize(name: str) -> str:
    """PEP 503 normalization: case-insensitive, ``-``/``_``/``.`` are one."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(block: str) -> set[str]:
    """The normalized package names in one TOML array of requirement strings."""
    body = "\n".join(
        line for line in block.splitlines() if not line.strip().startswith("#")
    )
    return {
        normalize(re.split(r"[<>=!\[; @]", item)[0].strip())
        for item in re.findall(r'"([^"]+)"', body)
    }


def requirement_array(name: str) -> str:
    """The raw body of the top-level ``<name> = [ ... ]`` array."""
    text = PYPROJECT.read_text()
    match = re.search(rf"^{re.escape(name)} = \[(.*?)^\]", text, re.S | re.M)
    assert match is not None, f"{name} array not found in pyproject.toml"
    return match.group(1)


def core_dependencies() -> set[str]:
    """``[project.dependencies]`` — what every install of the wheel carries."""
    return requirement_names(requirement_array("dependencies"))
