"""A whole fit as one body: the plan that ships, and what runs where the
model is.

A fit on NDIF is **one job**. nnsight ships every name a session body reads,
pickled whole, and returns every session-level variable bound to a saved
object; an optimizer over a shipped stage trains a copy. So everything a fit
moves is **created where it runs** — the stages, the optimizer, the
controllers, the generators, the early-stop snapshot
(:func:`~causalab.neural.shared.training.state.build_fit_state`, from the
plan's :class:`~causalab.neural.shared.training.spec.FitSpec`) — and the body
of the session is three statements (:func:`run_fit`): open it, bind the one
saved container, call :func:`fit_body`. The loop's locals are a function's,
not the session's, so nothing but ``result`` comes home.

Locally the same :func:`fit_body` runs in this process with no session
around it: one code path, the same stages built from the same spec, the same
bits.

**What ships** (:class:`TrainPlan`) is plain data: the spec, one *template*
program per forward group over the point's **whole frame**
(:meth:`~causalab.neural.engines.nnterp_engine.executor.NnterpExecutor.
fit_programs`) with its stacks by name, the eval split's programs and its
:class:`~causalab.neural.shared.training.spec.ScoreSpec`, and the saved
tensors a featurizer starts from (:class:`Artifacts`). A minibatch is a **row
selection** of a template (:func:`select_rows`) — the partition is the
spec's ``batches`` — so epochs add nothing to the payload.

**A step** is the local design: the source forward, then the trained
forward, each its own trace; the read a later consumer needs *flows* between
them with its graph (``landers._reduce``); the loss is built from the flowing
objective reads; ``backward()`` runs after the traces have exited, as plain
PyTorch. The model is frozen where it is served, so the graph begins where a
trained featurizer enters the forward.

**What comes back** (:func:`fit_body`'s ``result``) is plain data: every
stage's ``state_dict`` and the plain attributes beside it (a gate's annealed
``temperature``), the digest of each stage as it was built, the loss per
update, and the outcome's records — eval score, control and constraint
traces, checkpoints, and ``fit_diagnostics``, which needs the live stages and
is therefore computed here. The client loads the state into its own stage
objects (``train.py``).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Callable, Mapping, Sequence

import torch

from causalab.neural.engines.nnterp_engine.landers import refuse_meta, run_program
from causalab.neural.engines.nnterp_engine.program import (
    GroupProgram,
    Op,
    ReadPlan,
    SlotRef,
    StackRef,
)
from causalab.neural.engines.nnterp_engine.versions import ensure_server_matches
from causalab.neural.shared.featurizers import (
    FeaturizerStack,
    Identity,
    Stage,
    featurizer_cache,
)
from causalab.neural.shared.fires import FireTally, check_fires
from causalab.neural.shared.services import BundlePoint
from causalab.neural.shared.training import build_fit_state, fit_loop, step_loss
from causalab.neural.shared.training.objective import score
from causalab.neural.shared.training.spec import FitSpec, ScoreSpec
from causalab.neural.shared.training.state import build_stages
from causalab.protocol.errors import ProtocolError

__all__ = [
    "Artifacts",
    "TrainPlan",
    "bind_stages",
    "fit_body",
    "run_fit",
    "select_rows",
    "stage_digest",
]


# ---------------------------------------------------------------------- #
# the plan
# ---------------------------------------------------------------------- #


@dataclasses.dataclass
class Artifacts:
    """The saved tensors and score tables a fit's featurizers start from
    (§2.5 ``file_path``, ``init.file_path``, ``init.from_scores``), recorded
    as the client's stages were built and replayed where the fit's are: the
    same ``load_tensors(path).point(…)`` / ``load_table(path)`` calls,
    answered from what the client's loaders answered. A point carries the
    tensors a builder took from it, never the bundle it was cut from."""

    points: dict[tuple[str, str, str, bool], BundlePoint] = dataclasses.field(
        default_factory=dict
    )
    tables: dict[str, Any] = dataclasses.field(default_factory=dict)

    def recording(
        self, load_tensors: Callable[[str], Any], load_table: Any
    ) -> tuple[Callable[[str], Any], Any]:
        """``load_tensors`` / ``load_table`` as given, every answer kept."""

        def tensors(path: str) -> Any:
            return _Bundle(self, path, load_tensors(path))

        def table(path: str) -> Any:
            self.tables[path] = load_table(path)
            return self.tables[path]

        return tensors, (None if load_table is None else table)

    def load_tensors(self, path: str) -> "_Bundle":
        return _Bundle(self, path, None)

    def load_table(self, path: str) -> Any:
        if path not in self.tables:
            raise ProtocolError("P2", f"the fit's plan carries no table {path!r}")
        return self.tables[path]


@dataclasses.dataclass
class _Bundle:
    """One bundle as a stage builder sees it (``TensorBundle.point``):
    recording over the client's ``source``, replaying without one."""

    artifacts: Artifacts
    path: str
    source: Any

    def point(
        self, slot: str, want: Any, *, what: str, implicit: bool = False
    ) -> BundlePoint:
        key = (self.path, slot, repr(want), implicit)
        if self.source is not None:
            point = self.source.point(slot, want, what=what, implicit=implicit)
            kept: dict[str, torch.Tensor] = {}
            self.artifacts.points[key] = dataclasses.replace(point, tensors=kept)
            return _RecordingPoint(
                **{f.name: getattr(point, f.name) for f in dataclasses.fields(point)},
                kept=kept,
            )
        if key not in self.artifacts.points:
            raise ProtocolError(
                "P2", f"{what}: the fit's plan carries no such entry of {self.path!r}"
            )
        return self.artifacts.points[key]


@dataclasses.dataclass(frozen=True)
class _RecordingPoint(BundlePoint):
    """A bundle point that keeps each tensor a stage builder takes from it —
    the recorded point's ``tensors`` — so the plan carries what was read."""

    kept: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

    def tensor(self, slot: str) -> torch.Tensor:
        value = super().tensor(slot)
        self.kept[f"{slot}{self.suffix}"] = value
        return value


@dataclasses.dataclass(frozen=True)
class TrainPlan:
    """One point's fit as the data its body runs on.

    ``spec`` is the fit (:class:`FitSpec`: objective, optimizer, schedules,
    the minibatch partition ``batches``, and a recipe per stage the programs
    name). ``train`` is one template program per forward group of a step, in
    dependency order, over the whole frame; ``eval`` the same over the
    ``train.eval`` split and ``score`` its metrics — ``None`` / ``()`` for a
    fit that never evaluates. ``artifacts`` answers a featurizer's saved
    start. ``progress`` prints a line per epoch where the fit runs — on NDIF
    a log line the waiting client shows."""

    label: str
    spec: FitSpec
    train: tuple[GroupProgram, ...]
    eval: tuple[GroupProgram, ...] = ()
    score: ScoreSpec | None = None
    artifacts: Artifacts = dataclasses.field(default_factory=Artifacts)
    progress: bool = False


def select_rows(program: GroupProgram, rows: Sequence[int]) -> GroupProgram:
    """``program`` over the rows ``rows`` of its frame, in that order — what a
    minibatch executor over the same selection plans: the same padded width,
    and each row's positions, which were resolved against the frame and hold
    whichever selection the row travels in. Only what has a row axis moves:
    the inputs, the position ids and the position tables."""
    index = torch.tensor(list(rows), dtype=torch.long)

    def of_rows(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(0, index.to(tensor.device))

    def of_positions(positions: list[list[int]] | None) -> list[list[int]] | None:
        return None if positions is None else [positions[i] for i in rows]

    def selected(op: Op) -> Op:
        reads = tuple(
            dataclasses.replace(plan, positions=of_positions(plan.positions))
            if isinstance(plan, ReadPlan)
            else plan
            for plan in op.reads
        )
        write = op.write
        if write is not None:
            write = dataclasses.replace(
                write,
                positions={
                    ename: of_positions(per_row)
                    for ename, per_row in write.positions.items()
                },
            )
        return dataclasses.replace(op, reads=reads, write=write)

    return dataclasses.replace(
        program,
        inputs={name: of_rows(tensor) for name, tensor in program.inputs.items()},
        position_ids=of_rows(program.position_ids),
        ops=tuple(selected(op) for op in program.ops),
    )


def bind_stages(program: GroupProgram, stages: Mapping[str, Stage]) -> GroupProgram:
    """``program`` with every name resolved against ``stages``, the fit's own
    table: a :class:`StackRef` becomes the stack of those stages, a
    :class:`SlotRef` the stage's slot as it stands now. Bound per forward —
    a slot of a trained stage moves with every update."""

    def stack_of(ref: Any) -> Any:
        if not isinstance(ref, StackRef):
            return ref
        if not ref.names:
            return FeaturizerStack(names=(), stages=(Identity(),))
        return FeaturizerStack(ref.names, tuple(stages[name] for name in ref.names))

    def slot_of(operand: Any) -> Any:
        if not isinstance(operand, SlotRef):
            return operand
        return stages[operand.featurizer].slot_params()[operand.slot]

    def bound(op: Op) -> Op:
        reads = tuple(
            dataclasses.replace(
                plan,
                flow=dataclasses.replace(plan.flow, stack=stack_of(plan.flow.stack)),
            )
            if isinstance(plan, ReadPlan) and plan.flow is not None
            else plan
            for plan in op.reads
        )
        write = op.write
        if write is not None:
            write = dataclasses.replace(
                write,
                stacks={ename: stack_of(ref) for ename, ref in write.stacks.items()},
                operands={
                    name: slot_of(operand) for name, operand in write.operands.items()
                },
            )
        return dataclasses.replace(op, reads=reads, write=write)

    return dataclasses.replace(program, ops=tuple(bound(op) for op in program.ops))


def stage_digest(stage: Stage) -> str:
    """sha256 over a stage's ``state_dict`` — names, shapes and bytes, on the
    CPU. The client's and the fit's own are compared as built: a differing
    BLAS can move the last bit of a QR-completed rotation."""
    digest = hashlib.sha256()
    for name, tensor in sorted(stage.state_dict().items()):
        host = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{host.dtype}:{tuple(host.shape)}".encode())
        digest.update(host.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------- #
# the run
# ---------------------------------------------------------------------- #


def run_fit(
    model: Any,
    plan: TrainPlan,
    *,
    remote: bool | str = False,
    redraw: Callable[[], tuple[GroupProgram, ...]] | None = None,
) -> dict[str, Any]:
    """Fit ``plan`` on ``model`` and hand back :func:`fit_body`'s result.

    Locally the body runs in this process. With ``remote`` it is the body of
    **one** session — one job, whatever the epochs — and that body is three
    statements on purpose: a session-level name bound to anything an inner
    trace saved is downloaded, so the loop lives in a function.

    ``redraw`` is a local fit's §2.2 ``draw``: the step's programs re-planned
    for a new epoch's draw, which only the client can do."""
    import nnsight

    if not remote:
        result: dict[str, Any] = {}
        fit_body(model, plan, result, redraw=redraw)
        return result
    if redraw is not None:
        raise ValueError("a redraw is the client's: it cannot cross into a session")
    ensure_server_matches(remote)
    with model.session(remote=remote):
        result = nnsight.save({})
        fit_body(model, plan, result)
    return result


def fit_body(
    model: Any,
    plan: TrainPlan,
    result: dict[str, Any],
    *,
    redraw: Callable[[], tuple[GroupProgram, ...]] | None = None,
) -> None:
    """The whole fit, where the model is: build the stages and the fit's
    state from the spec, step the shared loop with this engine's forwards,
    and fill ``result`` with plain data (module docstring)."""
    spec = plan.spec
    module = model._module
    refuse_meta(module, plan.train[0])
    device = next(module.parameters()).device
    stages = build_stages(
        spec,
        device=device,
        load_tensors=plan.artifacts.load_tensors,
        load_table=plan.artifacts.load_table,
    )
    init_digest = {name: stage_digest(stage) for name, stage in stages.items()}
    state = build_fit_state(spec, stages=stages, device=device)
    train = [plan.train]
    losses: list[torch.Tensor] = []

    def forward(programs: tuple[GroupProgram, ...], rows: Sequence[int] | None):
        """One pass over ``programs`` in order, each its own trace; the reads
        the pass is for, by name."""
        flow: dict[str, torch.Tensor] = {}
        for program in programs:
            if rows is not None:
                program = select_rows(program, rows)
            _check_fired(
                program, run_program(model, bind_stages(program, stages), flow)
            )
        return flow.__getitem__

    def step(_members: Sequence[int]) -> None:
        rows = spec.batches[state.order[state.position]]
        # isolated: a request killed inside a scope on a shared server leaves
        # the process-global store open, and the next fit must not read it
        with featurizer_cache(isolated=True):
            step_loss(state, spec, forward(train[0], rows)).backward()
        assert state.last_loss is not None
        losses.append(state.last_loss)

    def evaluate(_members: Sequence[int]) -> list[dict[str, float]]:
        assert plan.score is not None
        for stage in stages.values():
            stage.eval()
        with featurizer_cache(isolated=True):
            return [score(plan.score, forward(plan.eval, None))]

    def on_epoch(_members: Sequence[int]) -> None:
        if redraw is not None:
            train[0] = redraw()
        if plan.progress:
            print(
                f"fit {plan.label}: {state.epoch} of {spec.epochs} epochs, "
                f"{state.step} of {spec.total_steps} updates, "
                f"loss {float(losses[-1]):.6g}"
            )

    (outcome,) = fit_loop(
        [state], [spec], step=step, evaluate=evaluate, on_epoch=on_epoch
    )
    if plan.progress:
        print(
            f"fit {plan.label}: done after {state.step} updates, "
            f"loss {float(losses[-1]):.6g}"
        )
    result.update(
        state={
            name: {
                key: value.detach().cpu().clone()
                for key, value in stage.state_dict().items()
            }
            for name, stage in stages.items()
        },
        attrs={name: _plain_attrs(stage) for name, stage in stages.items()},
        init_digest=init_digest,
        steps_run=state.step,
        loss_trace=torch.stack([loss.reshape(()) for loss in losses]).tolist(),
        eval_score=(
            None
            if outcome.eval_score is None
            else dataclasses.asdict(outcome.eval_score)
        ),
        diagnostics=outcome.diagnostics,
        controls=outcome.controls,
        control_trace=outcome.control_trace,
        constraints=outcome.constraints,
        constraint_trace=outcome.constraint_trace,
        anneals=outcome.anneals,
        phases=outcome.phases,
        checkpoints=[dataclasses.asdict(c) for c in outcome.checkpoints],
    )


def _check_fired(program: GroupProgram, out: Mapping[str, Any]) -> None:
    """Every write of ``program`` landed exactly once — the check a point's
    finalization makes on the client, made where a fit's forward ran."""
    tally = FireTally()
    for op in program.ops:
        if op.write is not None:
            tally.declare(op.write.members, 1)
    for members, fire in out["fired"]:
        tally.fired(members, step=fire)
    check_fires(program.label, tally)


#: What ``torch.nn.Module`` itself keeps on an instance.
_MODULE_STATE = frozenset(vars(torch.nn.Module()))


def _plain_attrs(stage: Stage) -> dict[str, Any]:
    """The numbers a stage keeps beside its ``state_dict`` — what an anneal
    or a controller sets (a gate's ``temperature``), a budget gate's drawn
    ``k`` — and its budget pool's, under ``"pool"``."""

    def numbers(obj: Any) -> dict[str, Any]:
        return {
            name: value
            for name, value in vars(obj).items()
            if name not in _MODULE_STATE and isinstance(value, (bool, int, float))
        }

    attrs = numbers(stage)
    pool = getattr(stage, "pool", None)
    if pool is not None:
        attrs["pool"] = numbers(pool)
    return attrs
