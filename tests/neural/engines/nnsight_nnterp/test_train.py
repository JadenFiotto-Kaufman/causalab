"""The train runner on the tiny Llama (spec §2.11), CPU fp32: the reference
suite's smallest end-to-end cases through this engine's traces, and the same
fit through both engines.

Random weights carry no task signal, so these are mechanism tests: the fit
runs, moves exactly the declared params, honors the seed, keeps the rotation
orthonormal, anneals the gate temperature, reduces its own objective, and
returns what ``early_stop`` selected. The documents are the reference
suite's (``tests/_helpers/train_docs.py``)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.engines.nnsight_nnterp.engine import NnterpEngine
from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.engines.nnsight_nnterp.loading import NnterpBundle
from causalab.neural.engines.nnsight_nnterp.train import run_training
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.engines.pytorch_hooks.train import (
    run_training as run_hooks_training,
)
from causalab.neural.shared.execution import TrainOutcome
from causalab.neural.shared.loading import torch_module
from causalab.neural.shared.metrics import compute_metric
from causalab.neural.shared.training import loop as loop_module
from causalab.protocol.engine import requires
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import parse_document

from tests._helpers import a3b_sweep as sweep
from tests._helpers.train_docs import (
    ROWS,
    controlled_dbm_doc,
    das_doc,
    dbm_doc,
    phased_chain_doc,
    train_request,
)
from tests.protocol._docs import in_order

pytestmark = pytest.mark.unit

EVAL_SPLIT = "weekdays/data#test"
#: what the optimizer steps under a ``cayley`` rotation
ORIGINAL = "parametrizations.weight.original"


def _executor(bundle: NnterpBundle, doc_raw: dict[str, Any]) -> NnterpExecutor:
    return sweep.make_executor(NnterpExecutor, doc_raw, bundle, rows=ROWS, with_cf=True)


def _fit(bundle: NnterpBundle, doc_raw: dict[str, Any], **request: Any) -> TrainOutcome:
    executor = _executor(bundle, doc_raw)
    (outcome,) = run_training([executor.doc], [executor], train_request(**request))
    return outcome


def _slots(outcome: TrainOutcome) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{slot}": param.detach().clone()
        for name, stage in outcome.stages.items()
        for slot, param in stage.slot_params().items()
    }


def test_das_fit_moves_only_the_rotation_and_is_seeded(nnterp_llama):
    first = _slots(_fit(nnterp_llama, das_doc(seed=0)))
    again = _slots(_fit(nnterp_llama, das_doc(seed=0)))
    other = _slots(_fit(nnterp_llama, das_doc(seed=1)))
    assert set(first) == {"rot.weight"}
    torch.testing.assert_close(
        first["rot.weight"], again["rot.weight"], atol=0.0, rtol=0.0
    )
    assert not torch.allclose(first["rot.weight"], other["rot.weight"], atol=1e-6)


def test_das_fit_leaves_the_model_frozen_and_moves_the_rotation(nnterp_llama):
    """The gradient reaches the rotation through the trace and nothing else:
    the fitted weight differs from the one the seed initialises, and no model
    parameter takes a gradient."""
    untrained = _executor(nnterp_llama, das_doc(seed=0)).stage("rot")
    start = untrained.weight.detach().clone()
    fitted = _slots(_fit(nnterp_llama, das_doc(seed=0)))["rot.weight"]
    assert not torch.allclose(fitted, start, atol=1e-6)
    assert all(
        p.grad is None and not p.requires_grad
        for p in torch_module(nnterp_llama.model).parameters()
    )


def test_das_weight_stays_orthonormal(nnterp_llama):
    weight = _slots(_fit(nnterp_llama, das_doc(seed=0)))["rot.weight"]
    gram = weight.T @ weight
    torch.testing.assert_close(gram, torch.eye(weight.shape[1]), atol=1e-5, rtol=1e-4)


def test_das_fit_reduces_its_own_objective(nnterp_llama):
    """Optimizing CE on the training rows must reduce CE on those rows — the
    document's metric before and after the fit, read on the point executor
    whose stage cache the fit trained."""

    def mean_ce(fit: bool) -> float:
        executor = _executor(nnterp_llama, das_doc(seed=0, epochs=4))
        if fit:
            run_training([executor.doc], [executor], train_request())
            executor.reset_reads()
        values = compute_metric(
            executor.doc.metrics["ce"],
            executor.dense_value("logits"),
            executor.rows_for_metrics(),
            nnterp_llama.tokenizer,
        )
        return sum(values) / len(values)

    assert mean_ce(fit=True) < mean_ce(fit=False)


def test_dbm_fit_trains_theta_and_anneals_temperature(nnterp_llama):
    gate = _fit(nnterp_llama, dbm_doc()).stages["gate"]
    assert not torch.allclose(gate.theta, torch.zeros_like(gate.theta))
    assert gate.temperature < 1.0  # the anneal ran
    assert not gate.training  # left in (hard) eval mode


def _with_eval(doc_raw: dict[str, Any], every: dict[str, int]) -> dict[str, Any]:
    doc_raw["method"]["train"]["eval"] = {
        "every": every,
        "split": EVAL_SPLIT,
        "metrics": ["ce"],
    }
    return doc_raw


def test_an_eval_pass_scores_the_split_on_the_trained_stages(nnterp_llama):
    """A real pass: the eval executor is built once over the split's rows on
    the point's stage cache, and every epoch's score is the document's metric
    over it — the last one the outcome's."""
    doc_raw = _with_eval(das_doc(seed=0, epochs=3), {"epochs": 1})
    executor = _executor(nnterp_llama, doc_raw)
    built: list[NnterpExecutor] = []
    real = loop_module.score

    def recording(doc, eval_executor):
        built.append(eval_executor)
        return real(doc, eval_executor)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loop_module, "score", recording)
        (outcome,) = run_training(
            [executor.doc], [executor], train_request({EVAL_SPLIT: ROWS})
        )
    assert len(built) == 3 and len({id(e) for e in built}) == 1
    assert built[0].stage_cache is executor.stage_cache
    assert not built[0].grad_enabled
    assert outcome.eval_score is not None
    assert outcome.eval_score.passes == 3
    assert outcome.eval_score.selected == "last"
    # the split is the training rows here, so the last pass's score is the
    # point executor's own metric over the fitted stages
    executor.reset_reads()
    values = compute_metric(
        executor.doc.metrics["ce"],
        executor.dense_value("logits"),
        executor.rows_for_metrics(),
        nnterp_llama.tokenizer,
    )
    assert outcome.eval_score.metrics["ce"] == pytest.approx(
        sum(values) / len(values), rel=1e-6
    )


def test_early_stop_returns_the_best_fit_not_the_last(nnterp_llama, monkeypatch):
    """``early_stop`` selects a fit by its eval score, so the fit it selected
    is the one that comes back. The score is scripted — a deterministic peak
    at the second eval — because what is under test is the selection."""
    doc_raw = _with_eval(das_doc(seed=0, epochs=5), {"epochs": 1})
    doc_raw["method"]["train"]["early_stop"] = {
        "metric": "ce",
        "patience": 10,
        "mode": "max",
    }
    scores = [0.1, 0.9, 0.5, 0.4, 0.3]
    seen: list[dict[str, torch.Tensor]] = []

    def scripted(doc, eval_executor):
        stage = eval_executor.stage_cache["rot"]
        seen.append({k: v.detach().clone() for k, v in stage.state_dict().items()})
        return {"ce": scores[len(seen) - 1]}

    monkeypatch.setattr(loop_module, "score", scripted)
    outcome = _fit(nnterp_llama, doc_raw, splits={EVAL_SPLIT: ROWS})

    assert len(seen) == len(scores)  # patience never fires: five evals ran
    peak, final = seen[1], seen[-1]
    assert not torch.allclose(peak[ORIGINAL], final[ORIGINAL])  # the fit moved on
    returned = outcome.stages["rot"].state_dict()
    torch.testing.assert_close(returned[ORIGINAL], peak[ORIGINAL], atol=0.0, rtol=0.0)
    assert outcome.eval_score is not None
    assert outcome.eval_score.selected == "early_stop.best"
    assert outcome.eval_score.metrics["ce"] == 0.9


def test_an_update_counted_eval_is_refused_rather_than_never_run(nnterp_llama):
    doc_raw = _with_eval(das_doc(seed=0, epochs=1), {"updates": 1})
    with pytest.raises(ProtocolError, match="must count epochs"):
        _fit(nnterp_llama, doc_raw)


def test_a_controlled_weight_follows_the_fit_and_is_recorded(nnterp_llama):
    outcome = _fit(nnterp_llama, controlled_dbm_doc())
    target = "train.objective.sparsity.weight"
    record = outcome.controls[target]
    trace = outcome.control_trace[target]
    # ten epochs of two batches: one controller update per optimizer update
    assert record["updates"] == 20.0 and len(trace) == 20
    assert record["initial"] == 0.01
    assert record["final"] > record["initial"]
    assert trace[0]["signal"] == 16.0 and trace[0]["setpoint"] == pytest.approx(15.2)
    assert record["setpoint_final"] == 0.0


def test_phases_narrow_what_trains_and_pin_the_named_masks(nnterp_llama):
    """Phase 0 trains the gate alone, phase 1 the rotation alone under the
    gate's pinned hard mask with the phase's own anneal; every update is
    photographed (a ``trajectory`` entry)."""
    outcome = _fit(nnterp_llama, phased_chain_doc())
    assert [c.step for c in outcome.checkpoints] == [1, 2, 3, 4]
    assert [c.record["phase"] for c in outcome.checkpoints] == [0, 0, 1, 1]
    rot = [c.slots["rot"]["weight"] for c in outcome.checkpoints]
    theta = [c.slots["gate"]["theta"] for c in outcome.checkpoints]
    assert torch.equal(rot[0], rot[1]) and not torch.equal(theta[0], theta[1])
    assert torch.equal(theta[1], theta[2]) and torch.equal(theta[2], theta[3])
    assert not torch.equal(rot[1], rot[2]) and not torch.equal(rot[2], rot[3])
    weights = [c.record["weight.sparsity"] for c in outcome.checkpoints]
    assert weights[:2] == [0.01, 0.01]
    assert weights[2] == pytest.approx(1.0) and weights[3] == pytest.approx(1.5)
    assert outcome.stages["gate"].frozen_mask is None


# --------------------------------------------------------------------------- #
# the same fit through both engines
# --------------------------------------------------------------------------- #


def _hooks_fit(bundle: ModelBundle, doc_raw: dict[str, Any]) -> TrainOutcome:
    executor = sweep.make_executor(
        PointExecutor, doc_raw, bundle, rows=ROWS, with_cf=True
    )
    return run_hooks_training(executor.doc, executor, train_request())


@pytest.mark.parametrize(
    "doc_raw",
    [das_doc(seed=0, epochs=4), das_doc(seed=3, epochs=4), dbm_doc()],
    ids=["das-seed0", "das-seed3", "dbm"],
)
def test_the_reference_runner_and_this_one_fit_the_same_weights(
    hooks_llama, nnterp_llama, doc_raw
):
    hooked = _slots(_hooks_fit(hooks_llama, doc_raw))
    traced = _slots(_fit(nnterp_llama, doc_raw))
    assert hooked.keys() == traced.keys()
    for name in hooked:
        torch.testing.assert_close(traced[name], hooked[name], atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #


def test_a_train_document_routes_to_this_engine():
    """``train`` needs ``grad`` (§8) and nothing this engine leaves
    undeclared, so a pinned list of this engine alone serves it."""
    from causalab.protocol.engine import choose_engine

    for raw in (das_doc(), dbm_doc()):
        doc = parse_document(in_order(raw))
        needed = requires(doc)
        assert "grad" in needed
        assert needed <= NnterpEngine().effective_capabilities
        assert isinstance(choose_engine(doc, [NnterpEngine()]), NnterpEngine)
    assert {
        "train_free_params",
        "train_loss_precision",
        "train_eval_updates",
    }.isdisjoint(NnterpEngine.capabilities)


def test_a_remote_engine_neither_claims_nor_runs_a_fit():
    """Training needs the graph of a forward in this process: a remote engine
    drops ``grad``, so routing passes it over, and refuses a ``train``
    document handed to it anyway."""
    engine = NnterpEngine(remote="local")
    needed = requires(parse_document(in_order(das_doc())))
    assert "grad" in needed - engine.effective_capabilities
    with pytest.raises(ProtocolError, match="remote forward returns detached") as err:
        engine._train([], [], train_request())  # pyright: ignore[reportPrivateUsage]
    assert err.value.code == "P4"


def test_a_weight_free_bundle_makes_the_engine_remote_for_training(remote_llama):
    """``remote=None`` inherits the bundle's: an engine handed a weight-free
    bundle runs on NDIF, so it drops ``grad`` and refuses a fit the same."""
    engine = NnterpEngine(bundle=remote_llama)
    needed = requires(parse_document(in_order(das_doc())))
    assert "grad" in needed - engine.effective_capabilities
    with pytest.raises(ProtocolError, match="remote forward returns detached") as err:
        engine._train([], [], train_request())  # pyright: ignore[reportPrivateUsage]
    assert err.value.code == "P4"
