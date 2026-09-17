"""The NDIF shape of the engine, pinned without a server.

nnsight's ``remote="local"`` runs the serialize → deserialize → execute path
in this process, but it runs the deserialized block **against the caller's
own frame**, so it cannot show the three ways a block that passes it still
fails on a real server:

1. a block that reads ``self`` ships the whole executor (measured: 50 MB
   against 11 KB) — the dry run pickles it happily;
2. a value saved into a dict slot, or a client object mutated in the block,
   never comes back — in one process the client's object *is* the block's;
3. a config mutated on the client (the eager switch) never reaches the
   server's model — in one process it is the same model.

So the properties are pinned directly: the structure of every trace body
(an AST tripwire), the size of the payload and its independence from the
executor, the parity of the session path with the local path, and the
weight-free bundle the remote tier loads.

``causalab`` is installed where a real block runs, so the dry run is told
it is a server module (the simulator otherwise hides it and the block's
by-reference functions fail to import). ``nnsight.register("causalab")``
is *not* the production configuration and does not survive the dry run:
cloudpickle ships a registered module's functions by value but an
``lru_cache`` wrapper (``gather._dense_index``) by reference, which the
simulator then cannot import.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

import causalab.neural.engines.nnsight_nnterp as engine_package
from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.engines.nnsight_nnterp.loading import load_model
from causalab.neural.shared.loading import torch_module
from causalab.protocol.errors import ProtocolError

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnsight_nnterp.conftest import ROWS, TINY_LLAMA

pytestmark = pytest.mark.smoke

_RUN_METHODS = {"trace", "generate", "session"}
_ENGINE_DIR = pathlib.Path(engine_package.__file__).parent


# ---------------------------------------------------------------------- #
# (i) the structural tripwire
# ---------------------------------------------------------------------- #


def _is_trace_block(node: ast.AST) -> bool:
    return isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr in _RUN_METHODS
        for item in node.items
    )


def _trace_blocks() -> list[tuple[str, ast.FunctionDef, ast.With]]:
    """Every ``with ….trace( / .generate( / .session(`` block of the engine
    package, with the function it sits in."""
    found = []
    for path in sorted(_ENGINE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if _is_trace_block(node):
                    found.append((path.name, function, node))
    return found


def _loaded_names(block: ast.With) -> set[str]:
    return {
        node.id
        for statement in block.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def test_the_engine_has_trace_blocks_to_check():
    blocks = _trace_blocks()
    methods = {
        item.context_expr.func.attr
        for _, _, block in blocks
        for item in block.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
    }
    assert _RUN_METHODS <= methods, methods  # or the tripwire is vacuous


def test_no_trace_body_reads_self():
    """nnsight ships every ``ast.Name`` a block loads as a whole pickled
    object, and an attribute is not a name: ``self.bundle.model`` ships
    ``self``. No trace body in the engine loads it."""
    for filename, function, block in _trace_blocks():
        names = _loaded_names(block)
        assert "self" not in names and "cls" not in names, (
            f"{filename}:{function.name} line {block.lineno} reads "
            f"{names & {'self', 'cls'}} inside a trace body — the whole "
            "executor would ship to NDIF"
        )


def test_a_trace_body_reads_only_its_function_s_own_data():
    """What a block may read: the enclosing function's parameters, what the
    block itself binds, a module the function imported, and module-level
    functions of the engine — never a closure over planning state."""
    for filename, function, block in _trace_blocks():
        module = ast.parse((_ENGINE_DIR / filename).read_text())
        module_level = {
            node.name
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        parameters = {
            arg.arg
            for arg in (
                *function.args.args,
                *function.args.kwonlyargs,
                *function.args.posonlyargs,
            )
        }
        imported = {
            alias.asname or alias.name
            for node in ast.walk(function)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        bound = {
            node.id
            for outer in ast.walk(function)
            if _is_trace_block(outer)
            for node in ast.walk(outer)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        allowed = module_level | parameters | imported | bound
        stray = _loaded_names(block) - allowed
        assert not stray, (
            f"{filename}:{function.name} line {block.lineno}: a trace body reads "
            f"{sorted(stray)}, which is neither a parameter, a block variable, "
            "an import nor a module-level function"
        )


def test_every_save_is_a_container_bound_at_block_level():
    """``push_result`` returns only *block variables* whose object is marked:
    ``saves[k] = nnsight.save(v)`` marks a value no variable names, and a
    real server returns nothing for it. Every ``nnsight.save`` in the engine
    is ``name = nnsight.save(…)``, a statement of a trace block's own body."""
    seen = 0
    for path in sorted(_ENGINE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        positions = {
            (statement.value.lineno, statement.value.col_offset)
            for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef)
            for block in ast.walk(function)
            if _is_trace_block(block)
            for statement in block.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        }
        for node in ast.walk(tree):
            is_save = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "save"
            )
            if is_save:
                seen += 1
                assert (node.lineno, node.col_offset) in positions, (
                    f"{path.name} line {node.lineno}: a save that is not a "
                    "block-level `name = nnsight.save(…)`"
                )
    assert seen  # or the check is vacuous


# ---------------------------------------------------------------------- #
# the dry run
# ---------------------------------------------------------------------- #


@pytest.fixture
def payloads(monkeypatch) -> list[int]:
    """The byte size of every request the test serializes, with the dry run
    treating ``causalab`` as installed on the server (module docstring)."""
    from nnsight.intervention.backends import local
    from nnsight.schema.request import RequestModel

    monkeypatch.setattr(local, "_SERVER_MODULES", {*local._SERVER_MODULES, "causalab"})
    sizes: list[int] = []
    serialize = RequestModel.serialize.__func__

    def measured(cls, tracer, compress=False):
        blob = serialize(cls, tracer, compress)
        sizes.append(len(blob))
        return blob

    monkeypatch.setattr(RequestModel, "serialize", classmethod(measured))
    return sizes


def _executor(doc, bundle, *, remote, rows=ROWS):
    return sweep.make_executor(
        NnterpExecutor, doc, bundle, rows=rows, with_cf=True, remote=remote
    )


def _values(executor: NnterpExecutor) -> dict[str, torch.Tensor]:
    executor.run_all()
    return dict(executor._read_values)


def _assert_identical(local: dict, remote: dict) -> None:
    assert local.keys() == remote.keys()
    for name, value in local.items():
        assert torch.equal(value, remote[name]), f"read {name!r} differs"


# (ii) ------------------------------------------------------------------ #


def test_the_payload_is_the_program_not_the_executor(nnterp_llama, payloads):
    """One job per point, kilobytes, and none of it the executor's: a 50 MB
    attribute hung on the executor does not reach the payload."""
    doc = sweep.interchange_doc("block_output", 1)
    plain = _executor(doc, nnterp_llama, remote="local")
    plain.run_all()
    laden = _executor(doc, nnterp_llama, remote="local")
    laden.ballast = torch.zeros(50_000_000 // 4)
    laden.run_all()
    assert len(payloads) == 2, payloads  # one session each: two groups, one job
    # not byte-equal: a program's tap keys carry `id(module)` integers, whose
    # pickled length varies by a byte or two between executors
    assert abs(payloads[0] - payloads[1]) < 1024, payloads
    assert payloads[0] < 256 * 1024, payloads


# (iii) ----------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component, layer",
    [("block_output", 1), ("mlp_output", 0), ("attention_query", 1)],
)
def test_a_session_point_matches_the_local_path_to_the_bit(
    nnterp_llama, payloads, component, layer
):
    """A two-group patching point — read the site on the counterfactual,
    swap it into the base forward, read the patched logits — through the
    session (the operand flows between the traces, never through the
    client) and through the local path: identical values, identical fires."""
    doc = sweep.interchange_doc(component, layer)
    local = _executor(doc, nnterp_llama, remote=False)
    session = _executor(doc, nnterp_llama, remote="local")
    _assert_identical(_values(local), _values(session))
    assert local.fires == session.fires and session.fires
    assert len(payloads) == 1, payloads
    # anti-vacuity: the patch changed the logits the session read
    clean = _executor(sweep.read_doc("lm_head", None), nnterp_llama, remote=False)
    assert not torch.equal(clean.read_value("r"), session.read_value("logits"))


def test_a_flowing_operand_is_finished_on_the_server(nnterp_llama, payloads):
    """The operand's ``dims`` are applied where it flows: a read that slices
    its features feeds a write of the same slice, and the session agrees
    with the local path, where the client finished the read."""
    doc = sweep.interchange_doc("block_output", 1)
    doc["method"]["reads"]["v_cf"]["dims"] = [0, 3, 5]
    doc["method"]["writes"]["patch"]["dims"] = [0, 3, 5]
    local = _executor(doc, nnterp_llama, remote=False)
    session = _executor(doc, nnterp_llama, remote="local")
    _assert_identical(_values(local), _values(session))
    assert tuple(session.read_value("v_cf").shape)[-1] == 3
    assert len(payloads) == 1, payloads


def test_a_lazy_read_runs_a_job_per_group(nnterp_llama, payloads):
    """``read_value`` before ``run_all`` still works remotely: each group is
    its own job and the operand ships by value."""
    doc = sweep.interchange_doc("block_output", 1)
    local = _executor(doc, nnterp_llama, remote=False)
    lazy = _executor(doc, nnterp_llama, remote="local")
    assert torch.equal(local.read_value("logits"), lazy.read_value("logits"))
    assert len(payloads) == 2, payloads
    assert max(payloads) < 256 * 1024, payloads


def test_an_operand_the_server_cannot_finish_falls_back(nnterp_qwen, payloads):
    """A routed-interior operand travels with its routing table, which the
    write joins by expert: the point runs one job per group, the operand
    shipped by value, and agrees with the local path."""
    doc = sweep.interchange_doc("expert_output", sweep.stream_layers(nnterp_qwen)[0])
    local = _executor(doc, nnterp_qwen, remote=False)
    remote = _executor(doc, nnterp_qwen, remote="local")
    _assert_identical(_values(local), _values(remote))
    assert len(payloads) == 2, payloads


def test_the_eager_switch_happens_in_the_block(nnterp_llama_default_impl, payloads):
    """A group whose address needs eager attention switches the model in its
    own block — on a server, the only model there is — restores it after the
    forward, and the client stamps what the block reports."""
    bundle = nnterp_llama_default_impl
    default = bundle.model.config._attn_implementation
    assert default != "eager"  # or this test is vacuous
    doc = sweep.read_doc("attention_scores", 1, pos="all")
    local = sweep.make_executor(
        NnterpExecutor, doc, bundle, rows=ROWS, with_cf=False, remote=False
    )
    remote = sweep.make_executor(
        NnterpExecutor, doc, bundle, rows=ROWS, with_cf=False, remote="local"
    )
    _assert_identical(_values(local), _values(remote))
    assert remote.applied_requirements == {"attn_eager"}
    assert bundle.model.config._attn_implementation == default


# ---------------------------------------------------------------------- #
# the remote tier's own surface
# ---------------------------------------------------------------------- #


def test_a_remote_bundle_holds_no_weights_and_still_plans():
    """The weight-free bundle: parameters on ``meta``, and everything before
    the forward — sites, positions, the program — works on the structure."""
    bundle = load_model(TINY_LLAMA, remote=True)
    assert bundle.remote and bundle.device == "cpu"
    assert {p.device.type for p in torch_module(bundle.model).parameters()} == {"meta"}
    executor = sweep.make_executor(
        NnterpExecutor,
        sweep.interchange_doc("attention_query", 1),
        bundle,
        rows=ROWS,
        with_cf=True,
    )
    assert executor.remote is True  # the bundle's own
    order = executor._group_order()
    assert order == [("original", "counterfactual"), ("patched", "base")]
    plans = [executor._plan(*group, flowing=frozenset({"v_cf"})) for group in order]
    assert not any(plan.unflowable for plan in plans)
    assert all(plan.program.offload for plan in plans)
    source, target = (plan.program for plan in plans)
    assert [op.reads[0].flow is not None for op in source.ops] == [True]
    # the write's operand is left to the session's flow, not shipped
    (write,) = [op.write for op in target.ops if op.write is not None]
    assert write.read_operands == {"v_cf"} and "v_cf" not in write.operands


def test_a_remote_bundle_refuses_a_device():
    with pytest.raises(ValueError, match="holds no weights"):
        load_model(TINY_LLAMA, device="cuda", remote=True)


def test_remote_mode_refuses_what_cannot_cross(nnterp_llama):
    with pytest.raises(ProtocolError, match="cannot run remotely"):
        sweep.make_executor(
            NnterpExecutor,
            sweep.interchange_doc("block_output", 1),
            nnterp_llama,
            rows=ROWS,
            with_cf=True,
            remote="local",
            grad_enabled=True,
        )


def test_the_engine_runs_a_point_as_one_job_through_the_front_door(
    nnterp_llama, payloads, tmp_path
):
    """``run_protocol`` with ``NnterpEngine(remote="local")``: the corpus
    interchange document runs as one session and writes the files and the
    receipt the local engine writes."""
    import json
    import shutil

    from causalab.neural.engines.nnsight_nnterp.engine import NnterpEngine
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
    run_protocol(loaded, env, [NnterpEngine(bundle=nnterp_llama)], here)
    assert not payloads
    run_protocol(
        loaded, env, [NnterpEngine(bundle=nnterp_llama, remote="local")], there
    )
    assert len(payloads) == 1, payloads
    assert json.loads((here / RUN_RECORD_NAME).read_text()) == json.loads(
        (there / RUN_RECORD_NAME).read_text()
    )
    names = sorted(p.name for p in here.iterdir())
    assert names == sorted(p.name for p in there.iterdir())
    for name in names:
        if name.endswith(".safetensors"):
            assert (here / name).read_bytes() == (there / name).read_bytes(), name
