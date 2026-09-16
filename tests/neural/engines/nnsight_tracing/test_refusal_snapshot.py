"""Every snapshotted **run-time** refusal still refuses, with the same message.

The run-time half of the refusal snapshot (see
``tests/protocol/test_refusal_snapshot.py`` for the rule and the shared
trigger table). Lives here because it loads the tiny fixtures on both engines;
the nnsight conftest's session bundles share ``load_model``'s cache with the
table's lazy fixtures, so nothing is loaded twice.
"""

from __future__ import annotations

import pytest

from tests._helpers import refusal_snapshot as table
from tests.protocol.test_refusal_snapshot import ENTRIES, check_entry

pytestmark = pytest.mark.smoke

RUN_IDS = sorted(
    (i for i, e in ENTRIES.items() if e["layer"] == "run" and e["captured"]), key=int
)


@pytest.fixture(scope="module")
def fixtures() -> table.Fixtures:
    return table.Fixtures()


@pytest.mark.parametrize("entry_id", RUN_IDS)
def test_every_snapshotted_run_refusal_still_refuses(
    entry_id: str, fixtures: table.Fixtures
) -> None:
    entry = ENTRIES[entry_id]
    with pytest.raises(Exception) as excinfo:
        table.RUN_TRIGGERS[entry_id](fixtures)
    check_entry(entry, excinfo.value)
