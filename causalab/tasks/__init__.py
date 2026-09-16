"""Task definitions for causal abstraction experiments.

See ``causalab/tasks/README.md`` for task structure, required exports,
and how to create a new task.
"""

from pathlib import Path

#: ``causalab/tasks/`` — every task ships the table(s) it generates under
#: ``<task>/data/<variant>.json`` (README §2), so this directory is also a data
#: root: the CLI's default ``--data-root``, and the fallback behind any other.
#: Kept import-light on purpose: ``causalab.cli`` reads it and must stay torch-free.
TASKS_ROOT = Path(__file__).resolve().parent

__all__ = ["TASKS_ROOT"]
