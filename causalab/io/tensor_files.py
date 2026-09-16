"""``save_file`` / ``load_file`` / ``safe_open`` for the repository's tensor files.

The one place that names which safetensors implementation writes and reads
them: :mod:`causalab.io.fastersafetensors` — the reference library's API and
bytes on disk, the I/O planned per machine and run with the GIL released.
Every module that touches a ``.safetensors`` file imports from here,
function-locally where the module must stay free of numerics at import time
(step scripts, ``step_io``).
"""

from __future__ import annotations

try:
    from causalab.io.fastersafetensors.torch import load_file, safe_open, save_file
except ImportError as exc:
    # seven modules reach this import; a missing extension should name its cause
    # here rather than as a bare `cannot import name '_core'` from whichever
    # script touched a tensor first
    raise ImportError(
        f"causalab.io.fastersafetensors could not be imported ({exc}). If the "
        "Rust extension (_core) is not built for this interpreter or platform, "
        "run `uv sync` with a Rust toolchain on PATH "
        "(docs/fastersafetensors.md)"
    ) from exc

__all__ = ["load_file", "safe_open", "save_file"]
