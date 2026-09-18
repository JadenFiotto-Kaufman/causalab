"""A whole fit as one NDIF job (``nnterp_engine/fit.py``), through a server
that runs the deserialized session (``tests/_helpers/faithful_server.py``).

The client is the production configuration — a weight-free bundle, so
``remote=True`` — and the server a separately loaded tiny Llama, CPU fp32.
The stages, the optimizer, the controllers and the early-stop snapshot are
built and stepped in the session; what comes home is plain data, loaded into
the client's own stage objects. Every fit is compared with the same document
fitted locally on the server's bundle — the same ``fit_body`` with no session
around it — to the bit.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import pytest
import torch

from causalab.neural.engines.nnterp_engine import train as train_module
from causalab.neural.engines.nnterp_engine.executor import NnterpExecutor
from causalab.neural.engines.nnterp_engine.fit import run_fit, select_rows
from causalab.neural.engines.nnterp_engine.train import plan_fit, run_training
from causalab.neural.shared import featurizers
from causalab.neural.shared.execution import TrainOutcome
from causalab.neural.shared.loading import torch_module
from causalab.neural.shared.training.executors import fit_spec, seeded_stages
from causalab.neural.shared.services import TensorBundle
from causalab.protocol.errors import ProtocolError, ValidationError

from tests._helpers import a3b_sweep as sweep
from tests._helpers.train_docs import (
    ROWS,
    TINY_QWEN35_MOE,
    chain_doc,
    controlled_dbm_doc,
    das_doc,
    dbm_doc,
    expert_dbm_fit_doc,
    phased_chain_doc,
    train_request,
)

pytestmark = pytest.mark.smoke

EVAL_SPLIT = "weekdays/data#test"
ORIGINAL = "parametrizations.weight.original"


def _early_stopped(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` with a real eval pass every epoch and ``early_stop`` on it: the
    split is scored, the best fit snapshotted and restored, all server-side."""
    doc["method"]["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": EVAL_SPLIT,
        "metrics": ["ce"],
    }
    # `max` on a loss the fit reduces: the first pass stays the best, and
    # patience runs out — the fit stops early and the first epoch's weights
    # are restored, which a fit that ignored either would not return
    doc["method"]["train"]["early_stop"] = {
        "metric": "ce",
        "patience": 1,
        "mode": "max",
    }
    return doc


def _dbm_eval_early_stop() -> dict[str, Any]:
    return _early_stopped(dbm_doc())


def _chain_eval_early_stop() -> dict[str, Any]:
    doc = chain_doc(0.05)
    doc["method"]["train"]["steps"] = {"epochs": 4}
    return _early_stopped(doc)


FITS = {
    "das": lambda: das_doc(seed=0, epochs=3),
    "dbm": dbm_doc,
    "das_eval_early_stop": lambda: _early_stopped(das_doc(seed=0, epochs=6)),
    # the gate and the two-stage chain through the same eval and early stop:
    # an annealed θ snapshotted and restored, and a stack of two trained
    # stages whose second is sized by the first
    "dbm_eval_early_stop": _dbm_eval_early_stop,
    "chain_eval_early_stop": _chain_eval_early_stop,
    "pid_controlled_dbm": controlled_dbm_doc,
    "phased_chain_trajectory": phased_chain_doc,
}

#: What no test here reaches, and what would have to run it: a hybrid model
#: (a remote GPT-2 or Qwen fit — the fixtures are one tiny Llama), a
#: featurizer starting from `init.from_scores` (an artifact-backed start; only
#: `init.file_path` is covered), an objective with a `constraint` and its dual
#: term, and a failure raised mid-fit on the server (what the client sees when
#: a job dies between updates).


def _executor(bundle: Any, doc_raw: dict[str, Any], **kwargs: Any) -> NnterpExecutor:
    return sweep.make_executor(
        NnterpExecutor, doc_raw, bundle, rows=ROWS, with_cf=True, **kwargs
    )


def _request():
    return train_request({EVAL_SPLIT: ROWS})


def _fit(bundle: Any, doc_raw: dict[str, Any], **kwargs: Any) -> TrainOutcome:
    executor = _executor(bundle, doc_raw, **kwargs)
    (outcome,) = run_training([executor.doc], [executor], _request())
    return outcome


def _init_state(executor: NnterpExecutor) -> dict[str, dict[str, torch.Tensor]]:
    """Every trained stage's ``state_dict`` as the document builds it, before
    any update — ``seeded_stages`` on a fresh executor, which is what a fit
    starts from."""
    spec = fit_spec(executor.doc, executor)
    return {
        name: {key: value.detach().clone() for key, value in stage.state_dict().items()}
        for name, stage in seeded_stages(spec, executor).items()
    }


def _assert_same_fit(
    here: TrainOutcome,
    there: TrainOutcome,
    init: dict[str, dict[str, torch.Tensor]],
) -> None:
    """The two fits agree on everything they returned — and both of them
    moved: two sides that each did nothing would agree just as well, so every
    trained stage is held against its own start (``init``)."""
    assert here.stages.keys() == there.stages.keys()
    assert init.keys() == here.stages.keys()
    for name, stage in here.stages.items():
        other = there.stages[name]
        assert stage.state_dict().keys() == other.state_dict().keys()
        for key, value in stage.state_dict().items():
            assert torch.equal(value, other.state_dict()[key]), f"{name}.{key}"
        assert any(
            not torch.equal(value, init[name][key])
            for key, value in stage.state_dict().items()
        ), f"{name} is where it started: this fit trained nothing"
        assert not other.training
        assert getattr(stage, "temperature", None) == getattr(
            other, "temperature", None
        )
    for field in (
        "eval_score",
        "diagnostics",
        "controls",
        "control_trace",
        "constraints",
        "constraint_trace",
        "anneals",
        "phases",
    ):
        assert getattr(here, field) == getattr(there, field), field
    assert len(here.checkpoints) == len(there.checkpoints)
    for a, b in zip(here.checkpoints, there.checkpoints):
        assert (a.step, a.epoch, a.record) == (b.step, b.epoch, b.record)
        for name, slots in a.slots.items():
            for slot, value in slots.items():
                assert torch.equal(value, b.slots[name][slot]), (a.step, name, slot)


@pytest.mark.parametrize("name", list(FITS))
def test_a_remote_fit_is_the_local_fit_to_the_bit(
    name, remote_llama, nnterp_llama_default_impl, ndif_llama
):
    """One job, one saved container, and the fit the local path fits: every
    ``state_dict`` entry of every trained stage, the eval score and what
    ``early_stop`` selected, the PID controller's trace, the checkpoints."""
    there = _fit(remote_llama, FITS[name]())
    (job,) = ndif_llama.jobs
    assert job.returned == ("result",)
    here = _fit(nnterp_llama_default_impl, FITS[name](), remote=False)
    assert len(ndif_llama.jobs) == 1  # the local fit is no job
    _assert_same_fit(
        here, there, _init_state(_executor(nnterp_llama_default_impl, FITS[name]()))
    )
    client = torch_module(remote_llama.model)
    assert {p.device.type for p in client.parameters()} == {"meta"}
    assert all(
        p.grad is None and not p.requires_grad
        for p in torch_module(nnterp_llama_default_impl.model).parameters()
    )
    # server-side bookkeeping did run
    if name.endswith("eval_early_stop"):
        assert there.eval_score is not None
        # `mode: max` on a loss the fit reduces: the first pass stays the
        # best, so what comes back is its snapshot and not the last update's
        assert there.eval_score.selected == "early_stop.best"
        assert there.eval_score.passes > 1
        if name == "das_eval_early_stop":
            assert there.eval_score.passes < 6  # patience stopped it short
    if name == "pid_controlled_dbm":
        (trace,) = there.control_trace.values()
        assert len(trace) == 20 and trace[0]["signal"] == 16.0
    if name == "phased_chain_trajectory":
        assert [c.step for c in there.checkpoints] == [1, 2, 3, 4]


def test_the_loss_moves_across_updates_on_the_server(remote_llama, ndif_llama):
    """The historical NDIF failure was a fit whose every update was a no-op
    (one autocast cache per request froze the first forward's weights): here
    the loss of each update differs from the last, the trained parameter left
    its init, and the waiting client was told, one line per epoch."""
    executor = _executor(remote_llama, das_doc(seed=0, epochs=3))
    planned = plan_fit(executor.doc, executor, _request())
    start = executor.stage_cache["rot"].state_dict()[ORIGINAL].detach().clone()
    result = run_fit(remote_llama.model, planned.plan, remote=True, progress=True)
    losses = result["loss_trace"]
    assert result["steps_run"] == len(losses) == 6
    assert all(a != b for a, b in zip(losses, losses[1:])), losses
    trained = result["state"]["rot"][ORIGINAL]
    assert trained.device.type == "cpu" and not trained.requires_grad
    assert not torch.equal(trained, start)
    (job,) = ndif_llama.jobs
    assert len(job.logs) == 3 and "6 of 6 updates" not in job.logs[0]
    assert job.logs[-1].endswith(f"loss {losses[-1]:.6g}")


def test_several_points_are_several_jobs_and_each_is_its_own_fit(
    remote_llama, nnterp_llama_default_impl, ndif_llama
):
    """``run_training`` takes a point per document: each is planned, fitted
    and loaded on its own, in order, one job each, and a point's fit is what
    it would be alone — a cohort of one, whatever runs beside it. The waiting
    client is told how each one is going."""
    docs = [das_doc(seed=0, epochs=2), das_doc(seed=3, epochs=2)]
    executors = [_executor(remote_llama, raw) for raw in docs]
    outcomes = run_training([ex.doc for ex in executors], executors, _request())

    assert len(outcomes) == len(ndif_llama.jobs) == 2
    assert all(job.returned == ("result",) and job.logs for job in ndif_llama.jobs)
    first, second = (
        outcome.stages["rot"].state_dict()[ORIGINAL] for outcome in outcomes
    )
    assert not torch.equal(first, second)  # two seeds, two fits
    for raw, outcome in zip(docs, outcomes):
        alone = _fit(nnterp_llama_default_impl, raw, remote=False)
        _assert_same_fit(
            alone, outcome, _init_state(_executor(nnterp_llama_default_impl, raw))
        )


def test_a_fit_s_program_outside_a_fit_refuses_by_name(nnterp_llama):
    """A fit's programs carry their featurizer stacks by name, and the names
    resolve against the fit's own stage table. Run one without that table —
    any forward that is not a fit's — and it refuses rather than reaching for
    a stack that is not there."""
    from causalab.neural.engines.nnterp_engine.landers import run_program

    executor = _executor(nnterp_llama, das_doc())
    planned = plan_fit(executor.doc, executor, _request())
    with pytest.raises(ProtocolError, match="resolve against a fit's stage table"):
        run_program(nnterp_llama.model, planned.plan.train[0], {})


def test_a_fit_returns_the_attributes_its_document_moves(remote_llama, ndif_llama):
    """What comes back beside a stage's ``state_dict`` is what the document's
    schedules move — the annealed ``temperature`` of the DBM's gate, nothing
    for a DAS fit that anneals nothing — not every number the stage happens
    to keep. `width` and `hard_eval` are the client's own, built from the
    same spec, and are never overwritten from the wire."""
    executor = _executor(remote_llama, dbm_doc())
    planned = plan_fit(executor.doc, executor, _request())
    result = run_fit(remote_llama.model, planned.plan, remote=True)
    assert set(result["attrs"]) == {"gate"}
    assert set(result["attrs"]["gate"]) == {"temperature"}
    assert result["attrs"]["gate"]["temperature"] < 1.0  # the anneal ran

    plain = _executor(remote_llama, das_doc(seed=0, epochs=2))
    result = run_fit(
        remote_llama.model, plan_fit(plain.doc, plain, _request()).plan, remote=True
    )
    assert result["attrs"] == {"rot": {}}


def test_the_fit_payload_is_the_plan(remote_llama, ndif_llama):
    """A whole DAS fit ships as ~103 KB pickled, ~27 KB under the zstd a
    request travels in — ~85 KB of it nnterp itself, which a remote
    ``StandardizedTransformer`` registers for by-value pickling. None of it
    is the executor or a stage (50 MB hung on each reaches nothing), and
    epochs cost nothing: a minibatch is a row selection made on the server."""

    def fit(epochs: int, *, ballast: bool = False) -> None:
        executor = _executor(remote_llama, das_doc(seed=0, epochs=epochs))
        if ballast:
            executor.ballast = torch.zeros(50_000_000 // 4)
            executor.stage("rot").ballast = torch.zeros(50_000_000 // 4)
        run_training([executor.doc], [executor], _request())

    fit(2)
    fit(2, ballast=True)
    fit(20)
    plain, laden, long = ndif_llama.jobs
    # not byte-equal: tap keys carry `id(module)` integers of varying length
    assert abs(plain.raw_bytes - laden.raw_bytes) < 1024, ndif_llama.jobs
    assert abs(plain.raw_bytes - long.raw_bytes) < 256, ndif_llama.jobs
    assert plain.raw_bytes < 128 * 1024, plain
    assert plain.zstd_bytes < 40 * 1024, plain


def _reachable(root: Any):
    seen: set[int] = set()
    stack = [root]
    while stack:
        item = stack.pop()
        if id(item) in seen or isinstance(item, (str, bytes, int, float, bool)):
            continue
        seen.add(id(item))
        yield item
        if isinstance(item, torch.Tensor):
            continue
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            stack.extend(getattr(item, f.name) for f in dataclasses.fields(item))


@pytest.mark.parametrize("name", list(FITS))
def test_a_plan_holds_no_stage_and_nothing_that_moves(name, remote_llama):
    """Stages travel by name: no plan reaches a stage, an optimizer, a
    generator or a tensor that requires grad."""
    executor = _executor(remote_llama, FITS[name]())
    plan = plan_fit(executor.doc, executor, _request()).plan
    for item in _reachable(plan):
        assert not isinstance(
            item, (featurizers.Stage, torch.optim.Optimizer, torch.Generator)
        ), item
        if isinstance(item, torch.Tensor):
            assert not item.requires_grad


def _stiefel_doc() -> dict[str, Any]:
    doc = das_doc()
    doc["method"]["featurizers"]["rot"]["parametrization"] = "stiefel"
    return doc


def _matrix_exp_doc() -> dict[str, Any]:
    doc = das_doc()
    doc["method"]["featurizers"]["rot"]["parametrization"] = "matrix_exp"
    return doc


def _untrained_member_doc() -> dict[str, Any]:
    """The chain with only the gate trained: ``rot`` is a stage the programs
    name and nobody trains, built on the server from its recipe all the same
    — and a ``stiefel`` one, whose base is a completion torch would draw from
    the global RNG and the stage draws from its own."""
    doc = chain_doc(0.05)
    doc["method"]["featurizers"]["rot"]["parametrization"] = "stiefel"
    doc["method"]["train"]["params"] = ["gate"]
    doc["method"]["save"] = [
        entry for entry in doc["method"]["save"] if entry.get("value") != "rot"
    ]
    return doc


BUILT = {
    "cayley": das_doc,
    "stiefel": _stiefel_doc,
    "matrix_exp": _matrix_exp_doc,
    "gate": dbm_doc,
    "chain": lambda: chain_doc(0.05),
    "untrained_member": _untrained_member_doc,
}


@pytest.mark.parametrize("name", list(BUILT))
def test_server_built_stages_are_the_client_s_to_the_bit(
    name, remote_llama, ndif_llama
):
    """The fit builds its stages from the plan's spec, in the session; the
    client built its own from the same spec on its executor. With each side's
    global RNG left somewhere different, the two agree to the bit — the
    digests match and nothing warns — and the server's process keeps the RNG
    it had."""
    executor = _executor(remote_llama, BUILT[name]())
    torch.manual_seed(1234)
    planned = plan_fit(executor.doc, executor, _request())
    torch.manual_seed(4321)
    state = torch.random.get_rng_state()
    result = run_fit(remote_llama.model, planned.plan, remote=True)
    assert result["init_digest"] == planned.init_digest
    assert set(planned.init_digest) == set(executor.stage_cache)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train_module._load(planned, result)  # pyright: ignore[reportPrivateUsage]
    # the server process's own global RNG is left where it was
    assert torch.equal(torch.random.get_rng_state(), state)


def test_a_stage_built_with_other_bits_warns_and_the_returned_state_stands(
    remote_llama, ndif_llama, monkeypatch
):
    executor = _executor(remote_llama, das_doc(seed=0, epochs=2))
    planned = plan_fit(executor.doc, executor, _request())
    planned.init_digest["rot"] = "0" * 64
    result = run_fit(remote_llama.model, planned.plan, remote=True)
    with pytest.warns(UserWarning, match=r"\['rot'\] were built where the fit ran"):
        outcome = train_module._load(planned, result)  # pyright: ignore[reportPrivateUsage]
    assert torch.equal(
        outcome.stages["rot"].state_dict()[ORIGINAL], result["state"]["rot"][ORIGINAL]
    )
    assert outcome.stages["rot"] is executor.stage_cache["rot"]


def test_a_saved_start_ships_with_the_plan(
    remote_llama, nnterp_llama_default_impl, ndif_llama
):
    """A gate that starts from a saved theta (§2.5 ``init.file_path``): the
    client records the entry its loader answered, the plan carries that entry
    alone, and the server-built gate starts from it."""
    theta = torch.linspace(-1.0, 1.0, 16)
    bundle = TensorBundle(
        tensors={"theta": theta, "bystander": torch.zeros(1_000_000)},
        entry_coords={},
        header={"produced_by": "f" * 64},
    )
    doc_raw = dbm_doc()
    doc_raw["method"]["featurizers"]["gate"]["init"] = {
        "file_path": "start.safetensors"
    }

    def fit(model_bundle: Any, **kwargs: Any) -> TrainOutcome:
        executor = _executor(model_bundle, doc_raw, **kwargs)
        executor.load_tensors = lambda path: {"start.safetensors": bundle}[path]
        (outcome,) = run_training([executor.doc], [executor], _request())
        return outcome

    there = fit(remote_llama)
    (job,) = ndif_llama.jobs
    assert job.raw_bytes < 256 * 1024  # the entry, not the 4 MB bundle
    starting = _executor(nnterp_llama_default_impl, doc_raw, remote=False)
    starting.load_tensors = lambda path: {"start.safetensors": bundle}[path]
    _assert_same_fit(
        fit(nnterp_llama_default_impl, remote=False), there, _init_state(starting)
    )
    assert not torch.equal(there.stages["gate"].theta.detach().reshape(-1), theta)


def test_a_row_selection_is_the_minibatch_s_own_plan(nnterp_llama):
    """``select_rows(template, rows)`` is field for field what a minibatch
    executor over the same rows plans: a fit ships the template once."""
    from causalab.neural.shared.training.draw import slice_rows

    executor = _executor(nnterp_llama, das_doc())
    planned = plan_fit(executor.doc, executor, _request())
    rows = [2, 0, 3]
    frames = {role: executor.frame(role) for role in executor.role_rows}
    minibatch = train_module._inner_executor(  # pyright: ignore[reportPrivateUsage]
        executor.doc,
        executor,
        role_rows=slice_rows(executor.role_rows, rows),
        grad_enabled=True,
        rows=tuple(rows),
        batches={role: frame.select(rows) for role, frame in frames.items()},
    )
    direct = minibatch.fit_programs(planned.plan.spec.objective_reads)
    assert len(direct) == len(planned.plan.train) == 2
    for template, program in zip(planned.plan.train, direct):
        selected = select_rows(template, rows)
        for field in dataclasses.fields(program):
            ours, theirs = getattr(selected, field.name), getattr(program, field.name)
            if field.name == "inputs":
                assert ours.keys() == theirs.keys()
                assert all(torch.equal(ours[k], theirs[k]) for k in ours)
            elif field.name == "position_ids":
                assert torch.equal(ours, theirs)
            else:
                assert ours == theirs, field.name


def test_a_scope_a_killed_request_left_open_does_not_reach_the_fit(
    remote_llama, ndif_llama, monkeypatch
):
    """``featurizers._SCOPE`` is a process global, and a request killed
    inside a ``featurizer_cache`` scope on a shared server leaves it open:
    the store is then never dropped, and a fit that nested in it would reuse
    its first step's rotation on every later step. A fit's scopes are
    isolated, so it is the clean fit."""
    clean = _fit(remote_llama, das_doc(seed=0, epochs=3))
    monkeypatch.setattr(featurizers, "_SCOPE", featurizers._Scope(depth=1, entries={}))  # pyright: ignore[reportPrivateUsage]
    dirty = _fit(remote_llama, das_doc(seed=0, epochs=3))
    _assert_same_fit(clean, dirty, _init_state(_executor(remote_llama, das_doc())))
    assert featurizers._SCOPE.depth == 1  # pyright: ignore[reportPrivateUsage]


def test_a_skewed_server_costs_a_fit_no_job(remote_llama, ndif_llama):
    ndif_llama.serve_env({"causalab": "9.9.9"})
    with pytest.raises(ProtocolError, match="has causalab '9.9.9'") as err:
        _fit(remote_llama, das_doc())
    assert err.value.code == "P4"
    assert not ndif_llama.jobs


def test_a_fit_on_a_meta_model_refuses(remote_llama, monkeypatch):
    """A sandboxed NDIF deployment hands the block a weight-free copy: the
    fit refuses by name before it builds anything."""
    from tests._helpers.faithful_server import FaithfulServer

    FaithfulServer(remote_llama.model, monkeypatch)
    with pytest.raises(ProtocolError, match="trusted, in-process NDIF"):
        _fit(remote_llama, das_doc())


def test_the_expert_neuron_dbm_is_one_remote_job_and_the_local_fit(monkeypatch):
    """The headline document, remotely. ``dbm_expert_neuron.json`` at tiny
    scale: its operand reads ``expert_activation``, a routed interior, so
    the value comes with a routing table, its gate is keyed by that table,
    and the write joins its slots to the read's by expert. Both tables now
    live in the session's flow, so the fit is **one job** — and the weights
    it trains are the local fit's to the bit.

    The weight-free MoE bundle builds in bf16 only (torch's meta
    ``grouped_mm`` has no fp32 kernel) and a client's dtype is structure,
    not numerics: the server here is the fp32 CPU model, which is what both
    sides of the comparison run on."""
    from causalab.neural.engines.nnterp_engine.loading import load_model
    from tests._helpers.faithful_server import FaithfulServer

    server = load_model(TINY_QWEN35_MOE, attn_implementation="eager")
    client = load_model(TINY_QWEN35_MOE, dtype="bf16", remote=True)
    ndif = FaithfulServer(server.model, monkeypatch)
    doc_raw = expert_dbm_fit_doc(pairs=2, epochs=2)
    there = _fit(client, doc_raw)
    (job,) = ndif.jobs
    assert job.returned == ("result",)
    here = _fit(server, doc_raw, remote=False)
    assert len(ndif.jobs) == 1  # the local fit is no job
    _assert_same_fit(here, there, _init_state(_executor(server, doc_raw)))
    assert {p.device.type for p in torch_module(client.model).parameters()} == {"meta"}


def test_the_engine_carries_no_second_refusal_for_a_fit(nnterp_llama):
    """``fit_programs`` refuses nothing of its own any more: the block
    finishes every kind of read the way the client does.

    The one frame a fit's forwards are not — the continuation frame — is
    refused a layer up and for a better reason than this engine could give:
    §5 rule 16 refuses `train` beside any generated position, on any engine,
    because a greedy decode is an argmax chain with no gradient path. The
    engine is never reached, which is why it carries no duplicate."""
    doc = das_doc()
    doc["method"]["reads"]["logits"]["pos"] = {
        "index": 0,
        "generated": {"max_new_tokens": 2},
    }
    with pytest.raises(ValidationError) as err:
        _fit(nnterp_llama, doc)
    assert err.value.rule == 16
    assert "no gradient path" in str(err.value)


def test_the_dry_run_mode_fits(nnterp_llama):
    """``remote="local"`` — nnsight's serialize → deserialize dry run over a
    loaded bundle — runs the same session body and fits the same weights."""
    from nnsight.intervention.backends import local

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(local, "_SERVER_MODULES", {*local._SERVER_MODULES, "causalab"})
        dry = _fit(nnterp_llama, das_doc(seed=0, epochs=2), remote="local")
    _assert_same_fit(
        _fit(nnterp_llama, das_doc(seed=0, epochs=2)),
        dry,
        _init_state(_executor(nnterp_llama, das_doc(seed=0, epochs=2))),
    )


def test_the_engine_fits_a_corpus_document_as_one_job_through_the_front_door(
    remote_llama, nnterp_llama_default_impl, ndif_llama, tmp_path
):
    """``run_protocol`` with ``NnterpEngine(bundle=<weight-free bundle>)``:
    the corpus DAS document — eval every epoch, ``early_stop`` — is fitted in
    one job, the finish phase evaluates the loaded stage in a second, and the
    run writes the bundle and the tables the local engine writes, byte for
    byte."""
    import shutil

    from causalab.neural.engines.nnterp_engine.engine import NnterpEngine
    from causalab.protocol import run_protocol
    from causalab.protocol.loader import load
    from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv

    from tests.neural.engines.nnterp_engine.conftest import TINY_LLAMA
    from tests.protocol._env import CORPUS_DIR, FIXTURES

    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts)
    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )
    loaded = load(
        CORPUS_DIR / "04_das_im.json",
        env,
        overrides={
            "model.key": TINY_LLAMA,
            "model.dtype": "fp32",
            "sites.target.layers": 1,
            "featurizers.rot.k": 4,
            "train.steps": {"epochs": 2},
            "train.batch": {"pairs": 2},
        },
    )
    here, there = tmp_path / "local", tmp_path / "remote"
    run_protocol(loaded, env, [NnterpEngine(bundle=nnterp_llama_default_impl)], here)
    assert not ndif_llama.jobs
    run_protocol(loaded, env, [NnterpEngine(bundle=remote_llama)], there)
    assert [job.returned for job in ndif_llama.jobs] == [("result",), ("results",)]
    names = sorted(p.name for p in here.iterdir())
    assert names == sorted(p.name for p in there.iterdir())
    assert "rot.safetensors" in names
    for name in names:
        if name.endswith((".safetensors", "iia.json", "ce.json")):
            assert (here / name).read_bytes() == (there / name).read_bytes(), name
