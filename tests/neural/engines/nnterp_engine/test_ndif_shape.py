"""The NDIF shape of the engine: its structure, and its remote surface.

nnsight ships a traced block as source plus every name the block reads, each
pickled whole, and a server returns only the saved block-level variables. So
three properties of every trace body are pinned here, statically over the
package's AST and dynamically over what nnsight's own block reduction
captures:

1. no body reads ``self`` — a block that does ships the whole executor
   (measured: 50 MB against the program's 13.5 KB);
2. every save is a container bound at block level — a value saved into a
   dict slot never comes back;
3. a body reads only its function's own data — parameters, block variables,
   imports, module-level functions.

What a server does with the payload — the deserialized program on a
separate model, results through a ``torch.save`` round trip — is
``test_faithful_server.py``'s, over ``tests/_helpers/faithful_server.py``.
nnsight's ``remote="local"`` is not that: it deserializes the block and then
runs the original tracer **against the caller's own frame**, so it checks
pickling and imports only. It stays a mode a caller can ask for, and one test
here runs it.

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

import causalab.neural.engines.nnterp_engine as engine_package
from causalab.neural.engines.nnterp_engine.executor import NnterpExecutor
from causalab.neural.engines.nnterp_engine.loading import load_model
from causalab.neural.shared.loading import torch_module

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnterp_engine.conftest import ROWS, TINY_LLAMA

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


def test_the_fit_session_body_is_the_container_and_one_call():
    """A session returns every session-level variable bound to a saved
    object, and an inner trace pushes its saved container up by name: a loop
    written in the session body would download each step's graph-attached
    container. So the body of a fit's session is exactly the saved container
    and one call to a module-level function, whose locals are its own."""
    (block,) = [
        block
        for filename, function, block in _trace_blocks()
        if (filename, function.name) == ("fit.py", "run_fit")
    ]
    bind, call = block.body
    assert ast.unparse(bind) == "result = nnsight.save({})"
    assert ast.unparse(call) == "fit_body(model, plan, result, progress=progress)"


def test_nnsight_captures_only_the_program_for_each_body_kind(
    remote_llama, ndif_llama, monkeypatch
):
    """The tripwire's dynamic complement: what nnsight's own block reduction
    ships for each kind of body the engine opens — a point's session, a fit's
    session, a trace, a generate — is the model, the program(s) or the plan,
    and the function that runs them.
    """
    from nnsight.schema import request

    from causalab.neural.engines.nnterp_engine.train import run_training

    from tests._helpers.train_docs import ROWS as FIT_ROWS
    from tests._helpers.train_docs import das_doc, train_request
    from tests.neural.engines.nnterp_engine.test_generate_frame import _gen_doc

    captured: dict[str, set[frozenset[str]]] = {}
    reduce_block = request.reduce_block

    def spy(node, glbls, lcls):
        source, used_globals, used_locals = reduce_block(node, glbls, lcls)
        kind = node.items[0].context_expr.func.attr
        captured.setdefault(kind, set()).add(frozenset({*used_globals, *used_locals}))
        return source, used_globals, used_locals

    monkeypatch.setattr(request, "reduce_block", spy)
    swap = sweep.interchange_doc("block_output", 1)
    sweep.make_executor(
        NnterpExecutor, swap, remote_llama, rows=ROWS, with_cf=True
    ).run_all()  # one session
    sweep.make_executor(
        NnterpExecutor, swap, remote_llama, rows=ROWS, with_cf=True
    ).read_value("logits")  # a trace per group
    sweep.make_executor(
        NnterpExecutor,
        _gen_doc("block_output", 1),
        remote_llama,
        rows=ROWS,
        with_cf=False,
    ).read_value("r")  # a generate
    fit = sweep.make_executor(
        NnterpExecutor, das_doc(), remote_llama, rows=FIT_ROWS, with_cf=True
    )
    run_training([fit.doc], [fit], train_request())  # a whole fit: one session

    # `stages` is a fit's table; a block that ships is never a fit's — a fit's
    # forwards run inside its session — so what travels here is one `None`
    body = frozenset(
        {"model", "tracer", "program", "flow", "stages", "nnsight", "execute"}
    )
    assert captured == {
        "session": {
            frozenset({"model", "programs", "nnsight", "run_program"}),
            frozenset({"model", "plan", "progress", "nnsight", "fit_body"}),
        },
        "trace": {body},
        "generate": {body},
    }


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


def test_the_dry_run_mode_runs_a_session_point(nnterp_llama, payloads):
    """``remote="local"`` is a mode a caller can ask for — nnsight's own dry
    run over a loaded bundle, which checks that the program pickles and that
    everything it names imports with the caller's modules hidden. A two-group
    patching point runs through it as one session and agrees with the local
    path. (What a server does with the payload is
    ``test_faithful_server.py``'s.)"""
    doc = sweep.interchange_doc("block_output", 1)
    local = _executor(doc, nnterp_llama, remote=False)
    session = _executor(doc, nnterp_llama, remote="local")
    _assert_identical(_values(local), _values(session))
    assert local.fires == session.fires and session.fires
    assert len(payloads) == 1, payloads


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
    assert executor._flowable("v_cf")
    ((groups, flowing),) = executor._segments(order)  # one session
    assert groups == order and flowing == {"v_cf"}
    plans = [executor._plan(*group, flowing=flowing) for group in order]
    assert all(plan.program.offload for plan in plans)
    source, target = (plan.program for plan in plans)
    assert [op.reads[0].flow is not None for op in source.ops] == [True]
    # the write's operand is left to the session's flow, not shipped
    (write,) = [op.write for op in target.ops if op.write is not None]
    assert write.read_operands == {"v_cf"} and "v_cf" not in write.operands


def test_a_remote_bundle_refuses_a_device():
    with pytest.raises(ValueError, match="holds no weights"):
        load_model(TINY_LLAMA, device="cuda", remote=True)
