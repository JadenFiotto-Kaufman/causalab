"""Every pre-registry **load-time** refusal still refuses, with the same message.

The snapshot (``fixtures/refusal_snapshot.json``) was captured on the base
before the capability registry consolidated the tables those refusals read
from. Each entry is re-triggered here and compared under the **superset
rule**: the old message must appear inside the new one, and the exception
class must be the same — or one of the upgrades ``ALLOWED_UPGRADES`` lists by
entry, each of which is a recorded decision, not a
tolerance. The run-time half lives beside the engine tests
(``tests/neural/engines/nnsight_nnterp/test_refusal_snapshot.py``), because
it loads models; the shared trigger table and the rule are the same.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests._helpers import refusal_snapshot as table

pytestmark = pytest.mark.unit

#: Deliberate after-states that are not "same class, superset message". Each
#: names the entry, what changed, and the substrings that must still be there
#: (the pins the existing tests carry). Was empty at the snapshot commit; every
#: entry here is a recorded decision.
ALLOWED_UPGRADES: dict[str, dict[str, Any]] = {
    # The new complete neuron site also accepts expert selection.
    "18": {
        "exc_class": "ProtocolError",
        "pins": [
            "site names expert 3 on component 'router_scores'",
            "which has no per-expert axis",
            "'expert_activation', 'expert_neuron_output' and 'expert_output'",
        ],
    },
    # The attention-pattern write policy is now one check with the routing
    # table's (`registry.write_policy_refusal`), so the two swap-only texts
    # share one template: "whole-value 'swap'" (was "whole-tensor 'swap'" for
    # the pattern alone). Everything the tests pin — the mechanism, the
    # component, the alternative and why — is unchanged. Its load-time twin is
    # `validate`'s V4 (test_capability_registry.py).
    "13": {
        "exc_class": "ProtocolError",
        "pins": [
            "write 'patch' applies 'add_scaled' to 'attention_probs', which only a",
            "'swap' may change: its rows are a probability distribution and the "
            "value multiply immediately downstream assumes they sum to 1",
            "Write 'attention_scores' instead",
        ],
    },
    # Three `NotImplementedError`s became protocol refusals (`ProtocolError`,
    # P4, reason `component_unavailable`) — what they always described: an
    # architectural fact the loaded model lacks, not a missing feature. The
    # texts are byte-identical; only the class moved.
    # (entry 23 — GPT-2's fused `c_attn` — is *retired*, not upgraded: the
    # per-family tap table addresses the interior there now, see
    # `table.RETIRED`.)
    "27": {
        "exc_class": "ProtocolError",
        "pins": [
            "component 'router_logits' needs a sparse-MoE block at layer 0, but "
            "this MLP (children=['act_fn', 'down_proj', 'gate_proj', 'up_proj']) "
            "is not one",
        ],
    },
    # One name per DeltaNet tensor: 'deltanet_qkv' folds onto
    # 'delta_qkv' at parse, so the full-attention-layer refusal it meets is the
    # one every component both engines serve on a DeltaNet mixer meets
    # (`_LINEAR_ATTENTION_ONLY`: "computes no delta-rule state") rather than
    # the nnterp-only interior's ("computes no recurrent state and runs no
    # delta kernel"). Same class, same code, same reason, same layer and tower
    # named; the component in the text is the canonical spelling.
    "17": {
        "exc_class": "ProtocolError",
        "pins": [
            "component 'delta_qkv' needs a Gated DeltaNet (linear-attention) "
            "mixer, but layer 3 of 'tiny-random/qwen3.5-moe' carries "
            "'full_attention'",
            "This tower is (linear_attention, linear_attention, "
            "linear_attention, full_attention).",
        ],
    },
    # The same upgrade for `mlp_activation` on an all-MoE
    # tower, whose load-time twin is `component_shape` refusing a model with
    # no dense inner width.
    "29": {
        "exc_class": "ProtocolError",
        "pins": [
            "mlp_activation: this MLP (children=['experts', 'gate', "
            "'shared_expert', 'shared_expert_gate']) matches no known family",
        ],
    },
}


def _entries() -> dict[str, dict[str, Any]]:
    data = json.loads(table.SNAPSHOT.read_text())
    return {entry["id"]: entry for entry in data["entries"]}


ENTRIES = _entries()
LOAD_IDS = sorted(
    (i for i, e in ENTRIES.items() if e["layer"] == "load" and e["captured"]), key=int
)


def test_the_snapshot_covers_the_census() -> None:
    """Vacuity floor and completeness: every census row 1–34 is either captured
    or recorded as not runnable, and every trigger has an entry."""
    assert set(ENTRIES) == {str(i) for i in range(1, 35)}
    captured = {i for i, e in ENTRIES.items() if e["captured"]}
    assert captured == set(table.LOAD_TRIGGERS) | set(table.RUN_TRIGGERS)
    assert {i for i, e in ENTRIES.items() if not e["captured"]} == set(
        table.NOT_RUNNABLE
    ) | set(table.RETIRED)
    assert not set(table.NOT_RUNNABLE) & set(table.RETIRED)
    for entry_id in table.RETIRED:
        assert ENTRIES[entry_id]["reason"] == table.RETIRED[entry_id]
    # 30 at the snapshot; entry 23 retired by the per-family tap table, entry 30
    # by the DeltaNet alias fold, entry 31 by the nnterp engine serving the
    # ragged `expert:` face
    assert len(captured) >= 27


def check_entry(entry: dict[str, Any], exc: BaseException) -> None:
    """The superset rule, with the recorded upgrades."""
    upgrade = ALLOWED_UPGRADES.get(entry["id"])
    if upgrade is None:
        assert type(exc).__name__ == entry["exc_class"], (
            f"entry {entry['id']}: {entry['exc_class']} became {type(exc).__name__}"
        )
        assert entry["message"] in str(exc), (
            f"entry {entry['id']}: the pre-PR message is no longer a substring "
            f"of the refusal.\n  was: {entry['message']}\n  now: {exc}"
        )
        return
    assert type(exc).__name__ == upgrade["exc_class"], (
        f"entry {entry['id']}: expected {upgrade['exc_class']}, got "
        f"{type(exc).__name__}"
    )
    for pin in upgrade["pins"]:
        assert pin in str(exc), f"entry {entry['id']}: {pin!r} not in {exc}"


@pytest.mark.parametrize("entry_id", LOAD_IDS)
def test_every_pre_pr_load_refusal_still_refuses(entry_id: str) -> None:
    entry = ENTRIES[entry_id]
    with pytest.raises(Exception) as excinfo:
        table.LOAD_TRIGGERS[entry_id]()
    check_entry(entry, excinfo.value)
