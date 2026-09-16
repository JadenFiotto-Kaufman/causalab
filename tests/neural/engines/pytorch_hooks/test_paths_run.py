"""Path blocks, executed (spec §3.2): the engine half of T9 and T10 on the
tiny fixtures.

T9 — an ordered receiver set is **one joint pass**. Two receivers at two
addresses lower to two ``inject_*`` writes in the one ``final`` intervened
model; the receipt's ``fires`` block lists both under one forward-group label,
and the joint effect on the metric is **not** the sum of the two
single-receiver effects. The gap the test measures is **receiver nesting**,
not a nonlinearity claim: the receivers are the block input of layer 1 and the
MLP output of the same layer, so with the block input injected the MLP
recomputes to the harvested value anyway — the joint pass equals the
upstream-only run exactly, and a sum of separate runs counts the downstream
effect twice. The anti-vacuity half is asserted separately: the receivers sit
inside the model (layer 1 exists, so the downstream receiver is recomputed
from the injected upstream one), each receiver moves the metric on its own,
the joint value is the upstream-only value, and the sum-of-separate value is
off it by more than tolerance — otherwise the mutation this test exists for
(one intervened model per receiver, values added afterwards) would pass
vacuously. (A tiny-random model is nearly linear, and nothing here asserts
otherwise: an attention-head sender's whole effect on the logit difference is
~5e-4, which is why the fixture uses a residual sender and a tolerance set
from the probed gap rather than a round number.)

T10 — the restoration policy changes the numbers. On the five-layer GPT-2
fixture a sender at layer 0 and a receiver at layer 2 leave one layer to
restore; ``attention_only`` and ``attention_and_mlp`` run to different logits,
have different point digests, and each policy reads back off the receipt's
``derived`` block. Dropping the ``freeze_m*`` emission collapses the two.

T12's run half — the hand-written ``03_path_patching_im.json`` still runs
unchanged — is ``test_run_corpus.py::test_03_path_patching_runs`` and
``test_write_set_fires.py`` (T19), untouched.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import register_model_key
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.shared.fires import group_label
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.paths import BLOCK
from causalab.protocol.registry import get_model_info
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.run import FIRES_KEY

from tests.neural.engines.pytorch_hooks.conftest import TINY_GPT2, TINY_LLAMA
from tests.protocol._env import FIXTURES
from tests.tables import frame as table_frame

pytestmark = pytest.mark.smoke

#: The joint-versus-sum gap below which the T9 mutation would pass vacuously:
#: a hundred times the fp32 noise of a tiny-random logit difference (~1e-7),
#: and a fifteenth of the gap the fixture actually shows (~1.6e-4, probed).
GAP = 1e-5


def _env(tmp_path: Path) -> ResolutionEnv:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    for key in (TINY_LLAMA, TINY_GPT2):
        register_model_key({"model": {"key": key, "revision": "main"}})
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )


def _doc(
    model_key: str,
    sender: dict[str, Any],
    receivers: list[dict[str, Any]],
    restoration: str,
    *,
    save: str,
) -> dict[str, Any]:
    """A path block over the IOI fixture rows, with the readout authored the
    way the shipped document authors it: ``lm_head``, ``logits`` in the
    injection model, and a clean twin on ``original``. ``save`` is ``"metric"``
    for the two ``logit_diff`` tables or ``"logits"`` for the raw tensors."""
    if save == "metric":
        metrics: dict[str, Any] = {
            name: {
                "kind": "logit_diff",
                "of": of,
                "a": "answer",
                "b": "cf_answer",
                "token_form": "space_prefixed",
            }
            for name, of in (("logit_diff", "logits"), ("ld_clean", "logits_clean"))
        }
        manifest = [
            {
                "value": "logit_diff",
                "model": "final",
                "input": "base",
                "file_path": "logit_diff.json",
            },
            {
                "value": "ld_clean",
                "model": "original",
                "input": "base",
                "file_path": "ld_clean.json",
            },
        ]
    else:
        metrics = {}
        manifest = [
            {
                "value": "logits",
                "model": "final",
                "input": "base",
                "file_path": "logits.safetensors",
            },
            {
                "value": "logits_clean",
                "model": "original",
                "input": "base",
                "file_path": "logits_clean.safetensors",
            },
        ]
    method: dict[str, Any] = {
        BLOCK: {
            "sender": sender,
            "source": "counterfactual",
            "receivers": receivers,
            "pos": -1,
            "restoration": restoration,
        },
        "sites": {"lm_head": {"component": "lm_head"}},
        "reads": {
            "logits": {"site": "lm_head", "pos": -1, "model": "final", "input": "base"},
            "logits_clean": {
                "site": "lm_head",
                "pos": -1,
                "model": "original",
                "input": "base",
            },
        },
    }
    if metrics:
        method["metrics"] = metrics
    method["save"] = manifest
    return {
        "header": {"protocol_version": "3"},
        "model": {"key": model_key, "revision": "main", "dtype": "fp32"},
        "data": {
            "base": {"dataset": "ioi/test", "field": "input"},
            "counterfactual": {
                "dataset": "ioi/test",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": method,
    }


def _run(doc: dict[str, Any], env: ResolutionEnv, out: Path) -> dict[str, Any]:
    run_protocol(doc, env, [PytorchHooksEngine()], out)
    return json.loads((out / RUN_RECORD_NAME).read_text())


def _values(out: Path, name: str) -> np.ndarray:
    return table_frame(out / name)["value"].to_numpy(dtype=float)


def _fires(receipt: dict[str, Any]) -> dict[str, dict[str, int]]:
    (point,) = receipt["points"]
    return receipt[FIRES_KEY][point["digest"]]


# --------------------------------------------------------------------------- #
# T9 — one joint pass, not a sum
# --------------------------------------------------------------------------- #


def test_two_receivers_are_one_joint_pass_and_not_a_sum(tmp_path: Path) -> None:
    """T9's run half. The two receivers are **nested** — ``mlp_output@1`` is
    downstream of ``block_input@1`` on the same path — and the gap is that
    nesting, not a nonlinearity: with the block input injected, the MLP
    recomputes to the harvested value anyway, so the joint pass equals the
    upstream-only run *exactly*, while a sum of separate runs counts the
    downstream receiver's effect twice. Mutation: emit one
    intervened model per receiver (``final_0``, ``final_1``) and add the two
    effects — the fires block then shows two ``final`` groups and the joint
    number is exactly the sum, so both halves below fail."""
    env = _env(tmp_path)
    sender = {"component": "block_output", "layers": [0]}
    upstream = {"component": "block_input", "layers": [1]}
    downstream = {"component": "mlp_output", "layers": [1]}
    joint = _run(
        _doc(
            TINY_LLAMA, sender, [upstream, downstream], "attention_only", save="metric"
        ),
        env,
        tmp_path / "joint",
    )
    only_upstream = _run(
        _doc(TINY_LLAMA, sender, [upstream], "attention_only", save="metric"),
        env,
        tmp_path / "upstream",
    )
    only_downstream = _run(
        _doc(TINY_LLAMA, sender, [downstream], "attention_only", save="metric"),
        env,
        tmp_path / "downstream",
    )

    # anti-vacuity, structural: the receivers' layer exists in the model, so
    # the downstream receiver is recomputed from the injected upstream one —
    # the nesting the gap below rests on (nothing here asserts nonlinearity)
    assert upstream["layers"][0] < get_model_info(TINY_LLAMA).num_layers

    # one forward group carries both injections, each firing once; the harvest
    # model is the sender swap alone (adjacent layers: nothing to restore)
    fires = _fires(joint)
    assert fires == {
        group_label("patched", "base"): {"swap_sender": 1},
        group_label("final", "base"): {"inject_0": 1, "inject_1": 1},
    }
    assert [g for g in fires if g.startswith("final")] == [group_label("final", "base")]
    for single in (only_upstream, only_downstream):
        assert _fires(single) == {
            group_label("patched", "base"): {"swap_sender": 1},
            group_label("final", "base"): {"inject": 1},
        }

    # the joint effect is not the sum of the separate effects
    clean = _values(tmp_path / "joint", "ld_clean.json")
    np.testing.assert_allclose(
        _values(tmp_path / "upstream", "ld_clean.json"), clean, atol=1e-6
    )
    joint_value = _values(tmp_path / "joint", "logit_diff.json")
    up = _values(tmp_path / "upstream", "logit_diff.json")
    down = _values(tmp_path / "downstream", "logit_diff.json")
    summed = up + down - clean
    gap = float(np.max(np.abs(joint_value - summed)))
    assert gap > GAP, f"joint {joint_value} vs sum-of-separate {summed}: gap {gap}"
    # anti-vacuity, numerical: the downstream receiver has an effect of its
    # own — the gap above is exactly that effect, counted twice by the sum —
    # and the nesting prediction holds: the joint pass is the upstream run
    assert float(np.max(np.abs(down - clean))) > GAP
    assert float(np.max(np.abs(up - clean))) > GAP
    np.testing.assert_allclose(joint_value, up, atol=1e-6)

    # M1: the receipt says what was lowered, beside the canonical form it names
    derived = joint["derived"][BLOCK]
    assert derived["receivers"] == ["receiver_0", "receiver_1"]
    assert derived["restorers"] == []
    for section, names in derived["emitted"].items():
        assert set(names) <= set(joint["canonical"]["method"][section]), section
    assert only_upstream["derived"][BLOCK]["receivers"] == ["receiver"]


# --------------------------------------------------------------------------- #
# T10 — the policy changes the numbers
# --------------------------------------------------------------------------- #


def test_the_restoration_policy_changes_the_numbers(tmp_path: Path) -> None:
    """T10's run half on the five-layer GPT-2 fixture: sender ``attention_premix``
    at layer 0 (head 1), receiver ``block_input`` at layer 2, one layer to
    restore. Mutation: drop the ``freeze_m*`` emission — the two runs then
    execute the same write set and the logits coincide."""
    env = _env(tmp_path)
    assert get_model_info(TINY_GPT2).num_layers == 5
    sender = {"component": "attention_premix", "layers": [0], "head": 1}
    receiver = {"component": "block_input", "layers": [2]}
    attention = _run(
        _doc(TINY_GPT2, sender, [receiver], "attention_only", save="logits"),
        env,
        tmp_path / "attention",
    )
    both = _run(
        _doc(TINY_GPT2, sender, [receiver], "attention_and_mlp", save="logits"),
        env,
        tmp_path / "both",
    )

    assert attention["derived"][BLOCK]["restoration"] == "attention_only"
    assert both["derived"][BLOCK]["restoration"] == "attention_and_mlp"
    assert attention["derived"][BLOCK]["restorers"] == [[1, "attention_output", "a1"]]
    assert both["derived"][BLOCK]["restorers"] == [
        [0, "mlp_output", "m0"],
        [1, "attention_output", "a1"],
        [1, "mlp_output", "m1"],
    ]
    assert _fires(attention)[group_label("patched", "base")] == {
        "swap_sender": 1,
        "freeze_1": 1,
    }
    assert _fires(both)[group_label("patched", "base")] == {
        "swap_sender": 1,
        "freeze_1": 1,
        "freeze_m0": 1,
        "freeze_m1": 1,
    }
    # different estimand identities: the point digest is over the write set
    assert attention["points"][0]["digest"] != both["points"][0]["digest"]
    assert attention["document_digest"] != both["document_digest"]

    # different numbers
    la = load_file(str(tmp_path / "attention" / "logits.safetensors"))
    lb = load_file(str(tmp_path / "both" / "logits.safetensors"))
    assert set(la) == set(lb) and la
    assert any(not torch.allclose(la[key], lb[key], atol=1e-5) for key in la)
    # and both patched something: neither equals the clean logits (each saved
    # bundle carries its read's one tensor under the read's own name)
    (clean,) = load_file(
        str(tmp_path / "attention" / "logits_clean.safetensors")
    ).values()
    for patched in (la, lb):
        (tensor,) = patched.values()
        assert not torch.allclose(tensor, clean, atol=1e-5)
