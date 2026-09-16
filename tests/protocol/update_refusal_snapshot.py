"""Regenerate ``tests/protocol/fixtures/refusal_snapshot.json``.

Runs every trigger in ``tests/_helpers/refusal_snapshot.py`` and records what
it raised: the exception class, the machine-readable code (``V<n>`` / ``P<n>``),
the path and the full rendered message. The run-time half loads the tiny
fixtures (``tiny-random/qwen3.5-moe``, tiny-llama, tiny-gpt2 — set
``HF_HUB_OFFLINE=1``), so run it as a job, not inline::

    uv run python tests/protocol/update_refusal_snapshot.py [--check]

``--check`` re-captures and exits non-zero if the committed file differs,
without writing. The snapshot was captured once, on the base the capability
registry refactored (``BASE``); regenerating it *after* a refactor and committing the result
would make the acceptance test vacuous, which is why the two test modules
compare against the committed file rather than re-capturing. The one
legitimate edit after that is **retiring** an entry (``table.RETIRED``)::

    uv run python tests/protocol/update_refusal_snapshot.py --retire-only

which rewrites *only* the retired rows as ``captured: false`` with the
decision and leaves every other entry byte-identical — a full re-capture
would also re-pin the upgrades ``ALLOWED_UPGRADES`` records, and so quietly
make that list vacuous.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from causalab.protocol.errors import ProtocolError

from tests._helpers import refusal_snapshot as table

BASE = "fd61afb9"


def _record(entry_id: str, layer: str, exc: BaseException) -> dict[str, Any]:
    code = exc.code if isinstance(exc, ProtocolError) else None
    path = exc.path if isinstance(exc, ProtocolError) else None
    return {
        "id": entry_id,
        "layer": layer,
        "captured": True,
        "exc_class": type(exc).__name__,
        "code": code,
        "path": path,
        "message": str(exc),
    }


def capture() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for entry_id, trigger in table.LOAD_TRIGGERS.items():
        try:
            trigger()
        except Exception as exc:  # noqa: BLE001 - the point is to record it
            entries.append(_record(entry_id, "load", exc))
        else:
            raise AssertionError(f"load trigger {entry_id} did not refuse")
    fixtures = table.Fixtures()
    for entry_id, run_trigger in table.RUN_TRIGGERS.items():
        try:
            run_trigger(fixtures)
        except Exception as exc:  # noqa: BLE001
            entries.append(_record(entry_id, "run", exc))
        else:
            raise AssertionError(f"run trigger {entry_id} did not refuse")
    for entry_id, reason in {**table.NOT_RUNNABLE, **table.RETIRED}.items():
        entries.append(
            {"id": entry_id, "layer": "run", "captured": False, "reason": reason}
        )
    entries.sort(key=lambda entry: int(entry["id"]))
    return {"base": BASE, "entries": entries}


def retire_only() -> dict[str, Any]:
    """The committed snapshot with the ``RETIRED`` rows replaced, nothing else
    re-captured — torch-free, no fixture loads."""
    data = json.loads(table.SNAPSHOT.read_text())
    for entry in data["entries"]:
        reason = table.RETIRED.get(entry["id"])
        if reason is not None:
            retired = {
                "id": entry["id"],
                "layer": entry["layer"],
                "captured": False,
                "reason": reason,
            }
            entry.clear()
            entry.update(retired)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify, do not write")
    parser.add_argument(
        "--retire-only",
        action="store_true",
        help="rewrite only the RETIRED rows; every other entry stays byte-identical",
    )
    args = parser.parse_args(argv)
    data = retire_only() if args.retire_only else capture()
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        current = table.SNAPSHOT.read_text() if table.SNAPSHOT.exists() else ""
        if current != text:
            print(f"{table.SNAPSHOT} is stale", file=sys.stderr)
            return 1
        print(f"{table.SNAPSHOT} is current")
        return 0
    table.SNAPSHOT.write_text(text)
    captured = sum(1 for e in json.loads(text)["entries"] if e["captured"])
    print(f"wrote {table.SNAPSHOT} ({captured} captured refusals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
