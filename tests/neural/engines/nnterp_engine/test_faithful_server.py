"""Remote parity through a server that runs the deserialized program.

The client is the production configuration — a weight-free bundle under
``remote=True`` — and the server a separately loaded model under the
checkpoint's own attention (``tests/_helpers/faithful_server.py``): the
restored tracer runs on its restored frame, and the saves come home through a
``torch.save`` / ``torch.load`` round trip. Every case is compared with the
local path on the server's own bundle: reads identical to the bit (CPU,
fp32), fires equal, the server's attention implementation put back, and the
client's parameters still on ``meta`` — no forward ran there.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch

from causalab.neural.engines.nnterp_engine.executor import NnterpExecutor
from causalab.neural.engines.nnterp_engine.loading import load_model
from causalab.neural.shared.loading import torch_module
from causalab.protocol.errors import ProtocolError

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnterp_engine.conftest import (
    ROWS,
    TINY_LLAMA,
    TINY_QWEN35_MOE,
)
from tests.neural.engines.nnterp_engine.test_generate_frame import _gen_doc, _window

pytestmark = pytest.mark.smoke


def _ragged_swap() -> dict[str, Any]:
    """A whole-sequence swap over rows of different lengths: the operand is
    ragged, so the server cannot finish it into the session's flow."""
    doc = sweep.interchange_doc("mlp_output", 0, pos="all")
    doc["method"]["writes"]["patch"]["ragged"] = {"policy": "padded_masked"}
    return doc


#: name → (document, has a counterfactual role, jobs, what the block reports)
CASES: dict[str, tuple[Callable[[], dict[str, Any]], bool, int, set[str]]] = {
    # two groups, one session: the operand flows between the traces
    "dense_session": (lambda: sweep.interchange_doc("block_output", 1), True, 1, set()),
    # a `.source` interior, navigated on the server's own forward
    "attention_query": (
        lambda: sweep.interchange_doc("attention_query", 1),
        True,
        1,
        set(),
    ),
    # an operand the server cannot finish: one job per group, shipped by value
    "ragged_fallback": (_ragged_swap, True, 2, set()),
    # the eager switch on the server's model, which the client never touches
    "eager_read": (
        lambda: sweep.read_doc("attention_scores", 1, pos="all"),
        False,
        1,
        {"attn_eager"},
    ),
    "eager_pattern_swap": (
        lambda: sweep.interchange_doc("attention_scores", 1, pos="all"),
        True,
        2,
        {"attn_eager"},
    ),
    # one generate trace; the continuation frame is built where the decode ran
    "generated_block_output": (lambda: _gen_doc("block_output", 1), False, 1, set()),
    "generated_lm_head": (
        lambda: _gen_doc("lm_head", None, pos=_window(index=-1)),
        False,
        1,
        set(),
    ),
    # projected through the server's head from the gathered ln_final rows
    "projected_lm_head": (lambda: sweep.read_doc("lm_head", None), False, 1, set()),
}


def _flat(value: Any) -> torch.Tensor:
    return getattr(value, "flat", value)


def _assert_parity(reference: NnterpExecutor, remote: NnterpExecutor) -> None:
    assert reference._read_values.keys() == remote._read_values.keys()
    for name, value in reference._read_values.items():
        other = remote._read_values[name]
        assert getattr(value, "widths", None) == getattr(other, "widths", None), name
        assert _flat(other).device.type == "cpu", name
        assert torch.equal(_flat(value), _flat(other)), f"read {name!r} differs"
    assert reference.fires == remote.fires
    assert reference._read_steps == remote._read_steps
    assert reference.applied_requirements == remote.applied_requirements


@pytest.mark.parametrize("case", list(CASES))
def test_a_remote_point_matches_the_local_path_to_the_bit(
    case, remote_llama, nnterp_llama_default_impl, ndif_llama
):
    build, with_cf, jobs, requirements = CASES[case]
    server = nnterp_llama_default_impl
    default = server.model.config._attn_implementation
    assert default != "eager"  # or the eager cases restore nothing
    # the remote run first: a server whose modules the local path has not
    # already instrumented is the one a real block meets
    remote = sweep.make_executor(
        NnterpExecutor, build(), remote_llama, rows=ROWS, with_cf=with_cf
    )
    assert remote.remote is True  # inherited from the weight-free bundle
    remote.run_all()
    assert len(ndif_llama.jobs) == jobs, ndif_llama.jobs

    reference = sweep.make_executor(
        NnterpExecutor, build(), server, rows=ROWS, with_cf=with_cf, remote=False
    )
    reference.run_all()
    assert len(ndif_llama.jobs) == jobs  # the local path is no job

    _assert_parity(reference, remote)
    assert remote.applied_requirements == requirements
    # one saved container per job, bound at block level
    container = "results" if jobs == 1 and with_cf else None
    for job in ndif_llama.jobs:
        assert len(job.returned) == 1, job
        if container is not None:
            assert job.returned == (container,)
    assert server.model.config._attn_implementation == default
    client = torch_module(remote_llama.model)
    assert {p.device.type for p in client.parameters()} == {"meta"}
    assert client is not torch_module(server.model)


def test_a_write_changes_what_the_server_read(
    remote_llama, nnterp_llama_default_impl, ndif_llama
):
    """Anti-vacuity for the parity above: the swap the session lands on the
    server changes the logits that come back."""
    patched = sweep.make_executor(
        NnterpExecutor,
        sweep.interchange_doc("block_output", 1),
        remote_llama,
        rows=ROWS,
        with_cf=True,
    )
    clean = sweep.make_executor(
        NnterpExecutor,
        sweep.read_doc("lm_head", None),
        remote_llama,
        rows=ROWS,
        with_cf=False,
    )
    assert not torch.equal(patched.read_value("logits"), clean.read_value("r"))


def test_a_loaded_bundle_runs_remotely_with_its_weights_idle(
    nnterp_llama, nnterp_llama_default_impl, ndif_llama
):
    """``remote=True`` over a loaded bundle is supported: the block names the
    model by key, so the server's copy runs it. The client here is pinned to
    eager and the server is not — the values are the server's."""
    doc = sweep.interchange_doc("block_output", 1)
    reference = sweep.make_executor(
        NnterpExecutor,
        doc,
        nnterp_llama_default_impl,
        rows=ROWS,
        with_cf=True,
        remote=False,
    )
    reference.run_all()
    remote = sweep.make_executor(
        NnterpExecutor, doc, nnterp_llama, rows=ROWS, with_cf=True, remote=True
    )
    remote.run_all()
    _assert_parity(reference, remote)
    assert len(ndif_llama.jobs) == 1


def _chain() -> dict[str, Any]:
    """Three groups in a chain: a dense read swapped into the base forward,
    a ragged whole-sequence read of *that* forward, swapped into a third."""
    doc = sweep.interchange_doc("block_output", 0)
    method = doc["method"]
    method["sites"]["mid"] = {"component": "mlp_output", "layers": 1}
    method["reads"]["v_mid"] = {
        "site": "mid",
        "pos": "all",
        "model": "patched",
        "input": "base",
    }
    method["writes"]["repatch"] = {
        "site": "mid",
        "pos": "all",
        "do": {"swap": "v_mid"},
        "ragged": {"policy": "padded_masked"},
    }
    method["intervened_models"]["repatched"] = {
        "input": "counterfactual",
        "writes": ["repatch"],
    }
    method["reads"]["logits"].update(model="repatched", input="counterfactual")
    method["save"][0].update(model="repatched", input="counterfactual")
    return doc


def test_one_unflowable_operand_cuts_the_session_where_it_is_consumed(
    remote_llama, nnterp_llama_default_impl, ndif_llama
):
    """Flowability is per operand. In a three-group chain whose second
    operand is ragged, the first two groups still run as one session with
    the dense operand flowing between them, and only the group that
    consumes the ragged one waits for it: two jobs, not three."""
    reference = sweep.make_executor(
        NnterpExecutor,
        _chain(),
        nnterp_llama_default_impl,
        rows=ROWS,
        with_cf=True,
        remote=False,
    )
    reference.run_all()
    remote = sweep.make_executor(
        NnterpExecutor, _chain(), remote_llama, rows=ROWS, with_cf=True
    )
    assert remote._flowable("v_cf") and not remote._flowable("v_mid")
    segments = remote._segments(remote._group_order())
    assert [(len(groups), set(flowing)) for groups, flowing in segments] == [
        (2, {"v_cf"}),
        (1, set()),
    ]
    remote.run_all()
    _assert_parity(reference, remote)
    assert len(ndif_llama.jobs) == 2, ndif_llama.jobs
    assert len(reference.fires) == 2  # both writes landed


def test_the_payload_is_the_program_not_the_executor(remote_llama, ndif_llama):
    """One job per point, and none of it the executor's: a 50 MB attribute
    hung on the executor does not reach the payload. What the weight-free
    client ships is ~99 KB pickled, ~26 KB under the zstd a request travels
    in. ~85 KB of that is nnterp itself, constant in the point: building a
    remote ``StandardizedTransformer`` registers nnterp for by-value pickling
    (it is not an NDIF server module), so its classes ride in every payload
    of the process; the program is the remaining ~13.5 KB."""
    doc = sweep.interchange_doc("block_output", 1)
    plain = sweep.make_executor(
        NnterpExecutor, doc, remote_llama, rows=ROWS, with_cf=True
    )
    plain.run_all()
    laden = sweep.make_executor(
        NnterpExecutor, doc, remote_llama, rows=ROWS, with_cf=True
    )
    laden.ballast = torch.zeros(50_000_000 // 4)
    laden.run_all()
    first, second = ndif_llama.jobs  # one session each: two groups, one job
    # not byte-equal: a program's tap keys carry `id(module)` integers, whose
    # pickled length varies by a byte or two between executors
    assert abs(first.raw_bytes - second.raw_bytes) < 1024, ndif_llama.jobs
    assert first.raw_bytes < 128 * 1024, first
    assert first.zstd_bytes < 40 * 1024, first


def test_a_flowing_operand_is_finished_on_the_server(
    remote_llama, nnterp_llama_default_impl, ndif_llama
):
    """The operand's ``dims`` are applied where it flows: a read that slices
    its features feeds a write of the same slice, and the session agrees
    with the local path, where the client finished the read."""
    doc = sweep.interchange_doc("block_output", 1)
    doc["method"]["reads"]["v_cf"]["dims"] = [0, 3, 5]
    doc["method"]["writes"]["patch"]["dims"] = [0, 3, 5]
    reference = sweep.make_executor(
        NnterpExecutor,
        doc,
        nnterp_llama_default_impl,
        rows=ROWS,
        with_cf=True,
        remote=False,
    )
    reference.run_all()
    remote = sweep.make_executor(
        NnterpExecutor, doc, remote_llama, rows=ROWS, with_cf=True
    )
    remote.run_all()
    _assert_parity(reference, remote)
    assert tuple(remote.read_value("v_cf").shape)[-1] == 3
    assert len(ndif_llama.jobs) == 1, ndif_llama.jobs
    remote.run_all()  # nothing left to run: no session, not an empty job
    assert len(ndif_llama.jobs) == 1, ndif_llama.jobs


def test_a_lazy_read_runs_a_job_per_group(
    remote_llama, nnterp_llama_default_impl, ndif_llama
):
    """``read_value`` before ``run_all`` still works remotely: each group is
    its own job and the operand ships by value."""
    doc = sweep.interchange_doc("block_output", 1)
    reference = sweep.make_executor(
        NnterpExecutor,
        doc,
        nnterp_llama_default_impl,
        rows=ROWS,
        with_cf=True,
        remote=False,
    )
    lazy = sweep.make_executor(
        NnterpExecutor, doc, remote_llama, rows=ROWS, with_cf=True
    )
    assert torch.equal(reference.read_value("logits"), lazy.read_value("logits"))
    assert len(ndif_llama.jobs) == 2, ndif_llama.jobs
    assert [job.returned for job in ndif_llama.jobs] == [("out",), ("out",)]


def test_the_engine_runs_a_point_as_one_job_through_the_front_door(
    remote_llama, nnterp_llama_default_impl, ndif_llama, tmp_path
):
    """``run_protocol`` with ``NnterpEngine(bundle=<weight-free bundle>)``
    and nothing else said: the engine inherits the bundle's ``remote``, the
    corpus interchange document runs as one session on the server, and it
    writes the files and the receipt the local engine writes."""
    import json
    import shutil

    from causalab.neural.engines.nnterp_engine.engine import NnterpEngine
    from causalab.protocol import RUN_RECORD_NAME, run_protocol
    from causalab.protocol.loader import load
    from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv

    from tests.protocol._env import CORPUS_DIR, FIXTURES

    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts)
    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )
    loaded = load(
        CORPUS_DIR / "02_interchange_im.json",
        env,
        overrides={"model.key": TINY_LLAMA, "sites.target.layers": 1},
    )
    here, there = tmp_path / "local", tmp_path / "remote"
    run_protocol(loaded, env, [NnterpEngine(bundle=nnterp_llama_default_impl)], here)
    assert not ndif_llama.jobs
    run_protocol(loaded, env, [NnterpEngine(bundle=remote_llama)], there)
    assert len(ndif_llama.jobs) == 1, ndif_llama.jobs
    assert json.loads((here / RUN_RECORD_NAME).read_text()) == json.loads(
        (there / RUN_RECORD_NAME).read_text()
    )
    names = sorted(p.name for p in here.iterdir())
    assert names == sorted(p.name for p in there.iterdir())
    for name in names:
        if name.endswith(".safetensors"):
            assert (here / name).read_bytes() == (there / name).read_bytes(), name


@pytest.mark.parametrize("remote", [False, "local"])
def test_a_weight_free_bundle_refuses_to_run_in_this_process(remote_llama, remote):
    """``remote=False`` over a weight-free bundle would have nnsight dispatch
    the whole checkpoint to serve the forward; the executor refuses instead,
    and the bundle stays weight-free."""
    with pytest.raises(ProtocolError, match="weight-free") as excinfo:
        sweep.make_executor(
            NnterpExecutor,
            sweep.read_doc("block_output", 1),
            remote_llama,
            rows=ROWS,
            with_cf=False,
            remote=remote,
        )
    assert excinfo.value.code == "P4"
    parameters = torch_module(remote_llama.model).parameters()
    assert {p.device.type for p in parameters} == {"meta"}


def test_a_forward_stops_after_its_last_operation(
    remote_llama, nnterp_llama_default_impl, ndif_llama
):
    """Nothing past a group's last operation is read, so the server's
    forward ends there: a layer-0 read never runs the last block, inside a
    session as in a lone trace, and the session's next trace still runs."""
    server = torch_module(nnterp_llama_default_impl.model)
    ran: list[int] = []
    last = server.model.layers[-1]
    handle = last.register_forward_hook(lambda *_: ran.append(1))
    try:
        shallow = sweep.make_executor(
            NnterpExecutor,
            sweep.read_doc("block_output", 0),
            remote_llama,
            rows=ROWS,
            with_cf=False,
        )
        shallow.run_all()
        assert not ran
        # a session: the counterfactual read stops at layer 0, the patched
        # forward that follows it runs to the head
        swap = sweep.make_executor(
            NnterpExecutor,
            sweep.interchange_doc("block_output", 0),
            remote_llama,
            rows=ROWS,
            with_cf=True,
        )
        swap.run_all()
        assert len(ran) == 1
    finally:
        handle.remove()
    assert len(ndif_llama.jobs) == 2


def test_a_whole_native_refusal_costs_no_job(remote_llama, ndif_llama):
    """A positioned read of the attention pattern is refused while the
    group is planned — not after a job has run and the whole ``(rows, heads,
    query, key)`` tensor has been downloaded."""
    executor = sweep.make_executor(
        NnterpExecutor,
        sweep.read_doc("attention_probs", 1, pos=-1),
        remote_llama,
        rows=ROWS,
        with_cf=False,
    )
    with pytest.raises(ProtocolError, match="position index would be ambiguous"):
        executor.run_all()
    assert not ndif_llama.jobs


def test_the_engine_inherits_remote_from_a_weight_free_bundle(remote_llama):
    """``NnterpEngine(bundle=load_model(key, remote=True))`` runs on NDIF:
    the engine's ``remote`` defaults to the bundle's own."""
    from causalab.neural.engines.nnterp_engine.engine import NnterpEngine

    assert NnterpEngine().remote is None
    assert NnterpEngine(bundle=remote_llama).remote is None
    executor = sweep.make_executor(
        NnterpExecutor,
        sweep.read_doc("block_output", 1),
        remote_llama,
        rows=ROWS,
        with_cf=False,
        remote=NnterpEngine(bundle=remote_llama).remote,
    )
    assert executor.remote is True


def test_a_block_handed_a_meta_model_refuses(remote_llama, monkeypatch):
    """A sandboxed NDIF deployment runs the block against a weight-free copy.
    Served that way — here, the client's own meta model as the server — the
    block refuses by name before it touches the model."""
    from tests._helpers.faithful_server import FaithfulServer

    FaithfulServer(remote_llama.model, monkeypatch)
    executor = sweep.make_executor(
        NnterpExecutor,
        sweep.read_doc("attention_scores", 1, pos="all"),
        remote_llama,
        rows=ROWS,
        with_cf=False,
    )
    with pytest.raises(ProtocolError, match="trusted, in-process NDIF"):
        executor.run_all()
    assert remote_llama.model.config._attn_implementation != "eager"
    parameters = torch_module(remote_llama.model).parameters()
    assert {p.device.type for p in parameters} == {"meta"}


# ---------------------------------------------------------------------- #
# the routed interior
# ---------------------------------------------------------------------- #


def test_a_routed_interior_operand_falls_back_on_the_server(monkeypatch):
    """A routed-interior operand travels with its routing table, which the
    write joins by expert, so the point runs one job per group with the
    operand and its table shipped by value.

    The weight-free MoE bundle builds in bf16 only (torch's meta
    ``grouped_mm`` has no fp32 kernel), and a client's dtype is structure,
    not numerics: the server here is the fp32 CPU model, so the values are
    compared to the bit with the local fp32 path."""
    from tests._helpers.faithful_server import FaithfulServer

    server = load_model(TINY_QWEN35_MOE)
    client = load_model(TINY_QWEN35_MOE, dtype="bf16", remote=True)
    ndif = FaithfulServer(server.model, monkeypatch)
    doc = sweep.interchange_doc("expert_output", sweep.stream_layers(server)[0])
    remote = sweep.make_executor(NnterpExecutor, doc, client, rows=ROWS, with_cf=True)
    remote.run_all()
    reference = sweep.make_executor(
        NnterpExecutor, doc, server, rows=ROWS, with_cf=True, remote=False
    )
    reference.run_all()
    _assert_parity(reference, remote)
    assert reference._read_routing.keys() == remote._read_routing.keys()
    for name, table in reference._read_routing.items():
        assert torch.equal(_flat(table), _flat(remote._read_routing[name])), name
    assert remote._read_routing  # or the fallback had no table to ship
    assert len(ndif.jobs) == 2, ndif.jobs
    clean = sweep.make_executor(
        NnterpExecutor,
        sweep.read_doc("lm_head", None),
        server,
        rows=ROWS,
        with_cf=False,
    )
    assert not torch.equal(clean.read_value("r"), remote.read_value("logits"))
    parameters = torch_module(client.model).parameters()
    assert {p.device.type for p in parameters} == {"meta"}
