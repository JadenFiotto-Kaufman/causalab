"""The train loop's structural property: a fit is **plain data plus torch
state**, and the loop that steps it never sees an executor.

``FitSpec`` and ``FitState`` pickle with plain :mod:`pickle` and reach no
executor, document, tokenizer or model; a stage built from the spec alone
starts bit-identical to the one the point's executor builds; and
``fit_loop`` runs a whole fit — order, schedules, optimizer, eval, early stop
— against nothing but a ``read(name)`` callable.
"""

from __future__ import annotations

import ast
import dataclasses
import pickle
import pickletools
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
import transformers

from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.executor_base import ExecutorBase
from causalab.neural.shared.featurizers import Gate, StageRecipe
from causalab.neural.shared.training import (
    EarlyStop,
    FitSpec,
    ResolvedMetric,
    ScoreSpec,
    build_fit_state,
    fit_loop,
    score,
    step_loss,
)
from causalab.neural.shared.training.draw import Drawn
from causalab.neural.shared.training.executors import fit_spec, seeded_stages
from causalab.protocol.schema import (
    Document,
    FeaturizerSpec,
    MetricSpec,
    ObjectiveTerm,
)

from tests._helpers import a3b_sweep as sweep
from tests._helpers.train_docs import (
    ROWS,
    TINY_LLAMA,
    chain_doc,
    das_doc,
    dbm_doc,
)

pytestmark = pytest.mark.unit

TRAINING = Path(sys.modules[FitSpec.__module__].__file__).parent  # type: ignore[arg-type]
#: what the loop is made of — everything but the executor side of a fit
LOOP_MODULES = (
    "loop.py",
    "state.py",
    "spec.py",
    "objective.py",
    "schedules.py",
    "control.py",
    "diagnostics.py",
)
#: what no spec or state may reach
FORBIDDEN = (
    ExecutorBase,
    Document,
    ModelBundle,
    Drawn,
    transformers.PreTrainedTokenizerBase,
    transformers.PreTrainedModel,
)


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


def _stiefel_doc() -> dict[str, Any]:
    doc = das_doc()
    doc["method"]["featurizers"]["rot"]["parametrization"] = "stiefel"
    return doc


def _pooled_doc() -> dict[str, Any]:
    """Two budget gates at two sites sharing one pool (§2.5): a pool is built
    whole and linked, so the spec carries a recipe for every member."""
    doc = dbm_doc()
    method = doc["method"]
    gate = {
        "kind": "gate",
        "parametrization": "budget",
        "k_schedule": {"kind": "uniform", "low": 1, "high": 8, "eval": 4},
        "pool": "both",
    }
    method["featurizers"] = {"gate": dict(gate), "gate_b": dict(gate)}
    method["sites"]["mlp"] = {"component": "mlp_output", "layers": [1]}
    method["reads"]["v_cf_b"] = {
        "site": "mlp",
        "pos": {"index": -1},
        "model": "original",
        "input": "counterfactual",
        "featurizer": "gate_b",
    }
    method["writes"]["patch_b"] = {
        "site": "mlp",
        "pos": {"index": -1},
        "featurizer": "gate_b",
        "do": {"swap": "v_cf_b"},
    }
    method["intervened_models"]["patched"]["writes"].append("patch_b")
    method["train"]["objective"] = [[1.0, "ce"]]
    method["train"]["params"] = ["gate_b", "gate"]
    del method["train"]["anneal"]
    method["save"].append(
        {"value": "gate_b", "site": "mlp", "file_path": "gate_b.safetensors"}
    )
    return doc


DOCS = {
    "das": das_doc,
    "das-stiefel": _stiefel_doc,
    "dbm": dbm_doc,
    "chain": lambda: chain_doc(0.05),
    "pooled": _pooled_doc,
}
ALL_DOCS = sorted(DOCS)


def _executor(bundle: ModelBundle, name: str) -> PointExecutor:
    return sweep.make_executor(
        PointExecutor, DOCS[name](), bundle, rows=ROWS, with_cf=True
    )


def _reachable(root: Any) -> Iterator[Any]:
    """Every object reachable from ``root`` through containers, dataclass
    fields and instance dictionaries; a tensor is a leaf."""
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
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            stack.extend(getattr(item, f.name) for f in dataclasses.fields(item))
        stack.extend(getattr(item, "__dict__", {}).values())


def _pickled_globals(blob: bytes) -> set[str]:
    """The ``module.name`` of every class or function a pickle names."""
    names: set[str] = set()
    strings: list[str] = []
    for opcode, arg, _pos in pickletools.genops(blob):
        if isinstance(arg, str):
            strings.append(arg)
        if opcode.name == "GLOBAL":
            names.add(str(arg).replace(" ", "."))
        elif opcode.name == "STACK_GLOBAL":
            names.add(".".join(strings[-2:]))
    return names


@pytest.mark.parametrize("name", ["das", "dbm"])
def test_the_spec_and_state_pickle_and_reach_no_executor(bundle, name):
    executor = _executor(bundle, name)
    spec = fit_spec(executor.doc, executor)
    state = build_fit_state(spec, stages=seeded_stages(spec, executor))

    blob = pickle.dumps(spec)
    assert len(blob) < 256 * 1024, f"the {name} spec pickles to {len(blob)} bytes"
    assert pickle.loads(blob) is not None
    pickle.dumps(state)

    for root in (spec, state):
        for item in _reachable(root):
            assert not isinstance(item, FORBIDDEN), (
                f"{type(root).__name__} reaches a {type(item).__name__}"
            )
    for blob in (pickle.dumps(spec), pickle.dumps(state)):
        for named in _pickled_globals(blob):
            assert not named.startswith("causalab.neural.engines"), named
            assert not named.startswith(("transformers", "tokenizers")), named
            assert not named.startswith("causalab.neural.shared.executor_base"), named


@pytest.mark.parametrize("name", ALL_DOCS)
def test_a_pickled_state_holds_its_own_stages_parameters(bundle, name):
    """The optimizer of an unpickled state steps the unpickled stages: the
    parameter objects are shared across the round trip, as they were before
    it — a ``subspace`` included, whose parametrized module torch will not
    pickle whole."""
    executor = _executor(bundle, name)
    spec = fit_spec(executor.doc, executor)
    state = build_fit_state(spec, stages=seeded_stages(spec, executor))
    copy = pickle.loads(pickle.dumps(state))

    stepped = {id(p) for group in copy.optimizer.param_groups for p in group["params"]}
    owned = {id(p) for stage in copy.stages.values() for p in stage.parameters()}
    assert stepped and stepped <= owned
    for stage_name, stage in state.stages.items():
        theirs = copy.stages[stage_name].state_dict()
        for key, value in stage.state_dict().items():
            assert torch.equal(value, theirs[key]), f"{stage_name}.{key}"
        assert copy.stages[stage_name].training == stage.training


@pytest.mark.parametrize("name", ALL_DOCS)
def test_spec_built_stages_are_the_executors_to_the_bit(bundle, name):
    """``build_fit_state(spec)`` with no stages handed over builds them from
    the spec — the seed, then ``train.params`` order — and they start exactly
    where the point executor's own do, the global-RNG completion of a
    ``stiefel`` base included. The global RNG is left elsewhere on purpose
    before each build: the init must not depend on it."""
    executor = _executor(bundle, name)
    spec = pickle.loads(pickle.dumps(fit_spec(executor.doc, executor)))

    torch.manual_seed(1234)
    built = build_fit_state(spec).stages
    torch.manual_seed(4321)
    theirs = seeded_stages(spec, executor)

    assert list(built) == list(theirs)
    for stage_name, stage in theirs.items():
        assert type(built[stage_name]).__name__ == type(stage).__name__
        ours = built[stage_name].state_dict()
        for key, value in stage.state_dict().items():
            assert torch.equal(ours[key], value), f"{stage_name}.{key}"
        for slot, value in stage.slot_params().items():
            assert torch.equal(built[stage_name].slot_params()[slot], value)
    pools = [getattr(stage, "pool", None) for stage in built.values()]
    if name == "pooled":
        assert pools[0] is not None and all(pool is pools[0] for pool in pools)
        assert pools[0].units == theirs["gate"].pool.units  # type: ignore[union-attr]


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_the_loop_imports_no_engine_and_no_executor():
    """Layering, and the property itself: nothing the loop is made of
    imports an engine package, the executor base, or the executor side of a
    fit — at module level or inside a function."""
    for module in LOOP_MODULES:
        for imported in _imports(TRAINING / module):
            assert not imported.startswith("causalab.neural.engines"), (
                f"{module} imports {imported}"
            )
            assert imported not in (
                "causalab.neural.shared.executor_base",
                "causalab.neural.shared.training.executors",
                "causalab.neural.shared.training.draw",
            ), f"{module} imports {imported}"


def test_importing_the_loop_loads_no_engine():
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import causalab.neural.shared.training.loop; "
            "print([m for m in sys.modules if m.startswith('causalab.neural.engines')])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert loaded.stdout.strip() == "[]"


# --------------------------------------------------------------------------- #
# a whole fit on reads alone
# --------------------------------------------------------------------------- #

WIDTH, VOCABULARY, EXAMPLES = 8, 16, 6


def _handmade_spec() -> FitSpec:
    """A DBM-shaped fit written down as data: a gate over an 8-wide value, a
    cross-entropy on a read named ``logits`` and an ``l1`` on the mask."""
    labels = [(3 * i) % VOCABULARY for i in range(EXAMPLES)]
    ce = MetricSpec(
        kind="cross_entropy", of="logits", fields={"target": "label"}, token_form="id"
    )
    return FitSpec(
        seed=7,
        params=("gate",),
        trained_names=("gate",),
        optimizer={"name": "adamw", "lr": 0.5},
        objective=(
            ObjectiveTerm(weight=1.0, metric="ce", name="ce"),
            ObjectiveTerm(weight=0.01, regularizer=("l1", ("gate",)), name="l1"),
        ),
        metrics={
            "ce": ResolvedMetric(
                name="ce",
                metric=ce,
                rows=tuple({"label": label} for label in labels),
                vocabulary=VOCABULARY,
                token_ids={"target": torch.tensor(labels)},
            )
        },
        objective_reads=("logits",),
        batches=((0, 1), (2, 3), (4, 5)),
        epochs=4,
        total_steps=12,
        eval_split="held-out",
        eval_metrics=("ce",),
        eval_every_epochs=1,
        early_stop=EarlyStop(metric="ce", mode="min", patience=10),
        recipes=(StageRecipe(name="gate", width=WIDTH),),
        featurizers={"gate": FeaturizerSpec(kind="gate")},
    )


def test_a_fit_runs_on_reads_alone():
    """No document, no executor, no model: the spec is unpickled, the state
    built from it, and the callbacks answer ``read`` from a closed-form
    "model". The loop does the rest — and the fit moves, evaluates every
    epoch, and returns the early-stop selection."""
    spec = pickle.loads(pickle.dumps(_handmade_spec()))
    state = build_fit_state(spec)
    gate = state.stages["gate"]
    assert isinstance(gate, Gate)
    start = gate.theta.detach().clone()

    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(EXAMPLES, WIDTH, generator=generator)
    donors = torch.randn(EXAMPLES, WIDTH, generator=generator)
    head = torch.randn(WIDTH, VOCABULARY, generator=generator)

    def logits(rows: list[int]) -> torch.Tensor:
        base, _ = gate.featurize(inputs[rows])
        donor, _ = gate.featurize(donors[rows])
        return gate.inverse(base + (donor - base), None) @ head

    scores = ScoreSpec(
        split="held-out",
        metrics=(dataclasses.replace(spec.metrics["ce"], token_ids={}),),
    )
    stepped: list[tuple[int, ...]] = []

    def step(members):
        assert list(members) == [0]
        rows = spec.batches[state.order[state.position]]
        stepped.append(rows)
        step_loss(state, spec, lambda name: logits(list(rows))).backward()

    def evaluate(members):
        with torch.no_grad():
            return [score(scores, lambda name: logits(list(range(EXAMPLES))))]

    (outcome,) = fit_loop([state], [spec], step=step, evaluate=evaluate)

    assert len(stepped) == spec.total_steps
    assert sorted(stepped[:3]) == sorted(spec.batches)  # an epoch is a partition
    assert not torch.equal(gate.theta.detach(), start)
    assert outcome.stages["gate"] is gate and not gate.training
    assert outcome.eval_score is not None
    assert outcome.eval_score.split == "held-out"
    assert outcome.eval_score.passes == spec.epochs
    assert outcome.eval_score.selected == "early_stop.best"
    assert outcome.eval_score.metrics["ce"] == state.best


def test_two_runs_of_one_spec_are_one_fit():
    """The seed is the spec's: two states built from it, stepped by the same
    reads, end on the same bits — whatever the global RNG held."""

    def run(global_seed: int) -> torch.Tensor:
        torch.manual_seed(global_seed)
        spec = _handmade_spec()
        state = build_fit_state(spec)
        gate = state.stages["gate"]
        assert isinstance(gate, Gate)
        values = torch.randn(
            EXAMPLES, VOCABULARY, generator=torch.Generator().manual_seed(1)
        )

        def step(members):
            rows = list(spec.batches[state.order[state.position]])
            mask = gate.featurize(torch.ones(len(rows), WIDTH))[0].sum(-1, keepdim=True)
            step_loss(state, spec, lambda name: values[rows] * mask).backward()

        fit_loop(
            [state],
            [dataclasses.replace(spec, eval_every_epochs=None, early_stop=None)],
            step=step,
            evaluate=lambda members: [],
        )
        return gate.theta.detach().clone()

    assert torch.equal(run(0), run(99))
