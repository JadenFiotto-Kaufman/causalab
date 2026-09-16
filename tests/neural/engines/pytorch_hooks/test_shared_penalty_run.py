"""One regularizer over many gates (spec §2.11, §3): the shared, sweepable
sparsity penalty a joint-layer DBM fit trains under.

The numeric half pins what the train loop computes for ``{"l1": [g0, g1]}``:
the mean of ``σ(θ/T)`` over the **concatenation** of the gates' units — not the
mean of the per-gate means, which the two differ on as soon as the gates differ
in size — so the one weight counts selected units across layers, and a single
name still gives the value it always did.

The end-to-end half fits two head-grouped gates on the tiny MoE fixture — query
heads at the full-attention layer 3, value heads at the DeltaNet layer 0 —
through ``run_protocol`` with the shared weight swept: the run has exactly one
coordinate column, ``train.objective.sparsity.weight``, and every saved bundle
is keyed by it alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import load_engines, register_model_key
from causalab.neural.engines.pytorch_hooks.train import _regularizer
from causalab.neural.shared.featurizers import Gate
from causalab.protocol import run_protocol
from causalab.protocol.loader import load
from causalab.protocol.tables import read_table

from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE
from tests.protocol._env import build_env

WEIGHTS = [0.01, 0.1]
AXIS = "train.objective.sparsity.weight"


def _two_gates() -> dict[str, Gate]:
    """A 4-head gate and a 12-coordinate gate, at different temperatures,
    with units on both sides of zero so no mean is trivially 0.5."""
    heads = Gate(4 * 16, group="head", groups=(4, 16))
    heads.temperature = 0.5
    plain = Gate(12)
    plain.temperature = 2.0
    with torch.no_grad():
        heads.theta.copy_(torch.tensor([3.0, -3.0, 0.5, -0.5]))
        plain.theta.copy_(torch.linspace(-2.0, 4.0, 12))
    return {"heads": heads, "plain": plain}


@pytest.mark.numerical_unit
class TestSharedPenaltyTerm:
    def test_l1_over_two_gates_is_the_mean_over_all_their_units(self) -> None:
        stages = _two_gates()
        term = _regularizer("l1", ["heads", "plain"], stages)
        soft = torch.cat(
            [torch.sigmoid(g.theta / g.temperature).flatten() for g in stages.values()]
        )
        assert torch.equal(term, soft.mean())
        assert term.numel() == 1 and term.requires_grad

    def test_it_is_not_the_mean_of_the_per_gate_means(self) -> None:
        """4 and 12 units: the concatenation weighs the larger gate three
        times as much, which is the point — the penalty counts units."""
        stages = _two_gates()
        term = _regularizer("l1", ["heads", "plain"], stages)
        per_gate = torch.stack(
            [_regularizer("l1", [name], stages) for name in ("heads", "plain")]
        )
        assert not torch.isclose(term, per_gate.mean())
        counts = torch.tensor([4.0, 12.0])
        assert torch.isclose(term, (per_gate * counts).sum() / counts.sum())

    def test_a_single_name_is_the_penalty_it_always_was(self) -> None:
        stages = _two_gates()
        heads = stages["heads"]
        assert torch.equal(
            _regularizer("l1", ["heads"], stages),
            torch.sigmoid(heads.theta / heads.temperature).mean(),
        )

    def test_l2_concatenates_the_parameters_themselves(self) -> None:
        stages = _two_gates()
        term = _regularizer("l2", ["plain", "heads"], stages)
        squares = torch.cat([g.theta.pow(2).flatten() for g in stages.values()])
        assert torch.isclose(term, squares.mean())

    def test_order_does_not_matter(self) -> None:
        stages = _two_gates()
        assert torch.equal(
            _regularizer("l1", ["heads", "plain"], stages),
            _regularizer("l1", ["plain", "heads"], stages),
        )


def _document() -> dict:
    """Two head-grouped gates, one at each of the fixture's head-major
    families, trained jointly under one swept sparsity weight."""
    return {
        "header": {
            "protocol_version": "3",
            "description": "joint two-layer head DBM under one shared, swept L1 weight",
        },
        "model": {"key": TINY_QWEN35_MOE, "revision": "main", "dtype": "fp32"},
        "data": {
            "base": {"dataset": "weekdays/train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "sites": {
                "attn": {"component": "attention_premix", "layers": [3]},
                "delta": {"component": "delta_premix", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "featurizers": {
                "attn_gate": {"kind": "gate", "group": "head"},
                "delta_gate": {"kind": "gate", "group": "head"},
            },
            "reads": {
                "attn_cf": {
                    "site": "attn",
                    "pos": -1,
                    "model": "original",
                    "input": "counterfactual",
                    "featurizer": "attn_gate",
                },
                "delta_cf": {
                    "site": "delta",
                    "pos": -1,
                    "model": "original",
                    "input": "counterfactual",
                    "featurizer": "delta_gate",
                },
                "logits": {
                    "site": "lm_head",
                    "pos": -1,
                    "model": "masked",
                    "input": "base",
                },
            },
            "writes": {
                "mask_attn": {
                    "site": "attn",
                    "pos": -1,
                    "featurizer": "attn_gate",
                    "do": {"swap": "attn_cf"},
                },
                "mask_delta": {
                    "site": "delta",
                    "pos": -1,
                    "featurizer": "delta_gate",
                    "do": {"swap": "delta_cf"},
                },
            },
            "intervened_models": {
                "masked": {"input": "base", "writes": ["mask_attn", "mask_delta"]}
            },
            "metrics": {
                "iia": {
                    "kind": "logit_diff",
                    "of": "logits",
                    "a": "cf_answer",
                    "b": "base_answer",
                    "token_form": "space_prefixed",
                },
                "ce": {
                    "kind": "cross_entropy",
                    "of": "logits",
                    "target": "label",
                    "token_form": "space_prefixed",
                },
            },
            "train": {
                "objective": {
                    "fit": {"weight": 1.0, "metric": "ce"},
                    "sparsity": {
                        "weight": {"sweep": WEIGHTS},
                        "l1": ["delta_gate", "attn_gate"],
                    },
                },
                "params": ["attn_gate", "delta_gate"],
                "optimizer": {"name": "adamw", "lr": 1e-2, "weight_decay": 0.0},
                "steps": {"epochs": 1},
                "batch": {"pairs": 2},
                "anneal": {
                    "attn_gate.theta.temperature": [1.0, 0.1, 0.5],
                    "delta_gate.theta.temperature": [1.0, 0.1, 0.5],
                },
                "seed": 0,
            },
            "save": [
                {
                    "value": "iia",
                    "model": "masked",
                    "input": "base",
                    "file_path": "iia.json",
                },
                {
                    "value": "ce",
                    "model": "masked",
                    "input": "base",
                    "file_path": "ce.json",
                },
                {
                    "value": "attn_gate",
                    "site": "attn",
                    "file_path": "attn_gate.safetensors",
                },
                {
                    "value": "delta_gate",
                    "site": "delta",
                    "file_path": "delta_gate.safetensors",
                },
            ],
        },
    }


@pytest.fixture(scope="module")
def swept_fit(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("shared_penalty")
    env = build_env(root / "artifacts")
    document = _document()
    register_model_key(document)  # the CLI's step: the registry learns the tiny key
    loaded = load(document, env)
    result = run_protocol(loaded, env, load_engines("auto", "cpu"), root / "out")
    return loaded, result


@pytest.mark.smoke
class TestSweptSharedWeight:
    def test_the_document_has_one_axis(self, swept_fit) -> None:
        loaded, _ = swept_fit
        assert [a.id for a in loaded.expansion.axes] == [AXIS]
        assert loaded.canonical_document["method"]["train"]["objective"]["sparsity"][
            "l1"
        ] == [
            "attn_gate",
            "delta_gate",
        ]

    def test_the_metric_table_has_exactly_that_coordinate_column(
        self, swept_fit
    ) -> None:
        _, result = swept_fit
        rows = read_table(Path(result.files["iia.json"]))
        assert len(rows) == 4 * len(WEIGHTS)  # the fixture's rows, once per point
        assert all(
            set(row)
            == {
                "example_id",
                "metric",
                "value",
                AXIS,
                "unit",
                "estimand_version",
                "eligible",  # the eligibility record (§2.10)
                "produced_by",
            }
            for row in rows
        )
        assert sorted({row[AXIS] for row in rows}) == WEIGHTS
        assert len({row["produced_by"] for row in rows}) == len(WEIGHTS)

    @pytest.mark.parametrize(
        "bundle", ["attn_gate.safetensors", "delta_gate.safetensors"]
    )
    def test_each_bundle_holds_one_head_mask_per_weight(
        self, swept_fit, bundle
    ) -> None:
        _, result = swept_fit
        tensors = load_file(str(result.files[bundle]))
        assert set(tensors) == {
            f"theta[objective.sparsity.weight={w}]" for w in WEIGHTS
        }
        assert all(t.shape == (8,) for t in tensors.values())  # 8 heads per family

    def test_the_diagnostics_report_both_gates_per_point(self, swept_fit) -> None:
        _, result = swept_fit
        records = json.loads(Path(result.files["fit_diagnostics.json"]).read_text())
        assert len(records) == len(WEIGHTS)
        for record in records:
            assert set(record["featurizers"]) == {"attn_gate", "delta_gate"}
            assert all(g["groups"] == 8.0 for g in record["featurizers"].values())
