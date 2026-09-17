"""Every snapshotted **run-time** refusal still refuses on the nnsight +
nnterp engine, with the same message wherever the refusal is the shared
layer's.

The run-time half of the refusal snapshot (``tests/protocol/
test_refusal_snapshot.py`` holds the rule and the shared trigger table),
re-run with this engine's bundles and executor in place of the reference
engine's. Refusals raised by shared code — the write policy, the stream
check, the predicate probes, the head and expert sub-axes, the whole-tensor
read rule, the ``dims`` refusal on the ragged ``expert:`` face (33) — reach
the same words. One entry is retired for this engine: the previous nnsight
engine refused the ``expert:`` face outright (31), and this one serves it
through the routing table it captures beside the experts interior, so the
trigger runs to a value. A recorded decision, pinned by running it.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import pytest

from tests._helpers import refusal_snapshot as table
from tests.neural.engines.nnsight_nnterp.conftest import TINY_LLAMA, TINY_QWEN35_MOE
from tests.protocol.test_refusal_snapshot import ENTRIES, check_entry

pytestmark = pytest.mark.smoke

#: Snapshotted refusals this engine deliberately turned into a pass.
SERVED_NOW: dict[str, str] = {
    "31": (
        "The ragged 'expert:' face of the routed interior is served: the "
        "executor captures the experts module's routing table beside its "
        "`.source` interior and hands it to the shared `_expert_selected`, "
        "the same landing the reference engine's dispatch wrapper feeds."
    ),
}

RUN_IDS = sorted(
    (
        i
        for i, e in ENTRIES.items()
        if e["layer"] == "run" and e["captured"] and i not in SERVED_NOW
    ),
    key=int,
)


class NnterpFixtures(table.Fixtures):
    """The trigger table's fixtures, served by this engine's loader."""

    @functools.cached_property
    def hooks_qwen(self) -> Any:
        from causalab.neural.engines.nnsight_nnterp.loading import load_model

        return load_model(TINY_QWEN35_MOE, attn_implementation="eager")

    @functools.cached_property
    def hooks_llama(self) -> Any:
        from causalab.neural.engines.nnsight_nnterp.loading import load_model

        return load_model(TINY_LLAMA, attn_implementation="eager")

    @functools.cached_property
    def hooks_qwen_eager_experts(self) -> Any:
        """The fixture dispatched on the per-expert loop, for the dispatch
        pin — the predicate reads the model's config, so a standardized
        model built with that selection stands in for the loaded one."""
        import torch
        from nnterp import StandardizedTransformer

        model = StandardizedTransformer(
            TINY_QWEN35_MOE,
            dtype=torch.float32,
            device="cpu",
            attn_implementation="eager",
            experts_implementation="eager",
            dispatch=True,
        )
        return dataclasses.replace(self.hooks_qwen, model=model)

    @functools.cached_property
    def trace_qwen(self) -> Any:
        return self.hooks_qwen


@pytest.fixture(scope="module")
def fixtures(request: pytest.FixtureRequest) -> table.Fixtures:
    from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor

    original = table._executor

    def executor(doc_raw: dict[str, Any], bundle: Any, *, trace: bool = False) -> Any:
        return _with_class(original, NnterpExecutor, doc_raw, bundle)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(table, "_executor", executor)
    request.addfinalizer(monkeypatch.undo)
    return NnterpFixtures()


def _with_class(original: Any, cls: type, doc_raw: dict[str, Any], bundle: Any) -> Any:
    """The table's own executor builder, over this engine's executor class:
    the same parsed-but-unvalidated document, rows and role fields."""
    from causalab.protocol.schema import parse_document

    from tests.protocol._docs import in_order

    doc = parse_document(in_order(doc_raw))
    rows = [{"input": table.TEXT, "counterfactual_inputs": [table.COUNTERFACTUAL_TEXT]}]
    return cls(
        doc,
        bundle,
        role_rows={"base": rows, "counterfactual": rows},
        role_fields={"base": "input", "counterfactual": "counterfactual_inputs[0]"},
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


@pytest.mark.parametrize("entry_id", RUN_IDS)
def test_every_snapshotted_run_refusal_still_refuses(
    entry_id: str, fixtures: table.Fixtures
) -> None:
    entry = ENTRIES[entry_id]
    with pytest.raises(Exception) as excinfo:
        table.RUN_TRIGGERS[entry_id](fixtures)
    check_entry(entry, excinfo.value)


@pytest.mark.parametrize("entry_id", sorted(SERVED_NOW, key=int))
def test_a_retired_run_refusal_now_runs(
    entry_id: str, fixtures: table.Fixtures
) -> None:
    assert ENTRIES[entry_id]["captured"]
    table.RUN_TRIGGERS[entry_id](fixtures)  # a value, or an unavailable cell
