"""The serialized table format, shared by the two layers that touch it.

A dataset table is written by :mod:`causalab.tasks.serialize` and read back by
:class:`causalab.protocol.resolve.FileDatasets`. Those two layers are
deliberately decoupled — ``causalab.protocol`` never imports ``causalab.tasks``
and ``causalab.tasks`` never imports ``causalab.protocol``, which is what keeps
resolution stdlib-only and a document's digest independent of task code or a
tokenizer (§2.2).

They still have to agree on exactly two things:

* **The bytes a table serializes to.** The content digest stamped into a
  canonical form (§7) is taken over them, so writer and reader must produce
  byte-identical output from the same rows.
* **The name of the column that declares a row's split.** The writer stamps it;
  the resolver selects on it.

Both live here, once, so that agreement is a shared import rather than two
copies free to drift. This module imports nothing but the standard library, so
either layer can depend on it without acquiring the other's dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

__all__ = ["SPLIT_COLUMN", "table_bytes"]

#: The column every row carries to declare which split it belongs to (§2.2).
#:
#: A dataset is one table and the split is a property of the row, not of the
#: file: a document selects one with the ``<ref>#<split>`` fragment. Values are
#: arbitrary strings — ``train``/``val``/``test`` is convention, not vocabulary,
#: so a k-fold table may use ``fold0``…``fold4`` and an undivided pool one
#: uniform value. Nothing in the library hardcodes a split name.
SPLIT_COLUMN = "split"


def table_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """The exact bytes a table serializes to: sorted keys, fixed indent,
    trailing newline. Deterministic on purpose — the content digest stamped
    into a canonical form (§7) has to be reproducible from the build's
    command line, on any machine."""
    return (json.dumps(list(rows), indent=1, sort_keys=True) + "\n").encode()
