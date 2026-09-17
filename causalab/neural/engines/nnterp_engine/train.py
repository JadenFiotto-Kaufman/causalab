"""This engine's train runner (spec §2.11): the forwards of a fit as traces,
under the shared loop.

What a ``train`` section means, and the update loop itself, are
:mod:`causalab.neural.shared.training`'s. What is here is how a step's
forward runs on this engine:

* a minibatch is an :class:`~causalab.neural.engines.nnterp_engine.executor.
  NnterpExecutor` over a row selection of the point's frame, built
  ``grad_enabled`` and on the point's own stage cache — its programs run
  under ``torch.enable_grad()`` and its reads come back as device tensors
  with their graph (``landers._offload``);
* a step asks the shared objective for its loss, whose reads run the
  executor's groups lazily — the operand's source forward, then the trained
  model's trace, where the write lands ``inverse(swap(featurize(·)))`` on the
  block's output in place and the head's logits are captured past it;
* ``backward()`` runs **after** the traces have exited, as plain PyTorch. The
  model's weights are frozen at load, so the graph begins where a trained
  featurizer enters the forward, and nothing a trace saves is detached —
  the capture-inside, backward-outside form nnsight supports for reading
  gradients, with no ordering rule between the tensors involved.

Every fit is a cohort of one: a member's step is its own traces, whatever
fits beside it. One :func:`~causalab.neural.shared.featurizers.
featurizer_cache` scope spans a member's forwards, loss and backward — a
rotation or a mask evaluated once per step, the gradient reaching the
parameter through that one evaluation, closed before the optimizer moves it.

**Local only.** A remote forward's saves come back detached, on the CPU, and
a client-side optimizer over parameters the server never sees trains
nothing; the executor refuses ``grad_enabled`` with ``remote`` and the engine
refuses the document before that.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from causalab.neural.engines.nnterp_engine.executor import NnterpExecutor
from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.execution import TrainOutcome
from causalab.neural.shared.executor_base import ExecutorBase
from causalab.neural.shared.featurizers import featurizer_cache
from causalab.neural.shared.training import (
    FitSpec,
    FitState,
    build_fit_state,
    fit_loop,
    step_loss,
)
from causalab.neural.shared.training.draw import Drawn
from causalab.neural.shared.training.executors import (
    eval_executor,
    eval_pass,
    fit_spec,
    minibatch_executors,
    seeded_stages,
)
from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.schema import Document

__all__ = ["run_training"]


def _inner_executor(
    doc: Document,
    executor: ExecutorBase,
    *,
    role_rows: Mapping[str, list[dict[str, Any]]],
    grad_enabled: bool,
    rows: tuple[int, ...] | str,
    batches: Mapping[str, EncodedBatch] | None = None,
    drawn: bool = False,
) -> NnterpExecutor:
    """This engine's ``ExecutorFactory`` (``shared/training/executors.py``):
    a fit's minibatch or eval executor
    over the point's. ``rows`` and ``drawn`` key a forward store, which this
    engine does not keep — every group an inner executor reads, it runs."""
    del rows, drawn
    assert isinstance(executor, NnterpExecutor)
    return NnterpExecutor(
        doc,
        executor.bundle,
        role_rows=role_rows,
        role_fields=executor.role_fields,
        load_tensors=executor.load_tensors,
        load_table=executor.load_table,
        stage_cache=executor.stage_cache,  # shared: one stage per name
        grad_enabled=grad_enabled,
        coords=executor.coords,
        batches=batches,
        remote=executor.remote,
    )


@dataclasses.dataclass(eq=False)
class _Fit:
    """One point's fit on this engine: the loop's ``spec`` and ``state``, and
    beside them what runs the forwards — the point's executor, one executor
    per minibatch, the eval executor once a pass has built it, and the drawn
    roles (§2.2 ``draw``). The loop sees none of it."""

    doc: Document
    executor: NnterpExecutor
    spec: FitSpec
    state: FitState
    minibatches: list[ExecutorBase]
    drawn: Drawn | None
    evaluator: ExecutorBase | None = None


def _prepare(doc: Document, executor: NnterpExecutor) -> _Fit:
    """The fit's plain data, its stages on the point's own cache — seeded and
    built in one breath, so a member's init is its document's whatever was
    prepared before it — its state, then its minibatch executors."""
    spec = fit_spec(doc, executor)
    state = build_fit_state(
        spec, stages=seeded_stages(spec, executor), device=executor.bundle.device
    )
    minibatches, drawn = minibatch_executors(doc, executor, spec, _inner_executor)
    return _Fit(doc, executor, spec, state, minibatches, drawn)


def _score(doc: Document, evaluator: ExecutorBase) -> dict[str, float]:
    """One eval pass on ``evaluator`` (``executors.eval_pass``)."""
    return eval_pass(doc, evaluator)


def run_training(
    docs: Sequence[Document],
    executors: Sequence[NnterpExecutor],
    request: ExecutionRequest,
) -> list[TrainOutcome]:
    """Fit each point; one outcome per point, in order.

    Each ``executor`` is its point's full-data executor — its stage cache is
    shared with every minibatch and with the eval executor, so the stages it
    later evaluates are the fitted ones, and the outcome's ``stages`` are
    those same objects. With ``train.early_stop`` they hold the best-scoring
    weights (``eval_score.selected``), otherwise the last."""
    if len(docs) != len(executors):
        raise ValueError(
            f"{len(docs)} documents but {len(executors)} executors — one per point"
        )
    fits = [_prepare(doc, executor) for doc, executor in zip(docs, executors)]

    def step(members: Sequence[int]) -> None:
        """One optimizer step's forwards, member by member: the loss's reads
        run the minibatch executor's traces under ``enable_grad``, and the
        backward runs once they have exited."""
        for fit in (fits[i] for i in members):
            state = fit.state
            minibatch = fit.minibatches[state.order[state.position]]
            minibatch.reset_reads()
            with featurizer_cache():
                step_loss(state, fit.spec, minibatch.dense_value).backward()

    def evaluate(members: Sequence[int]) -> list[dict[str, float]]:
        """One pass per member on its own executor over its ``train.eval``
        split — built on the first pass and kept, so the split is read and
        encoded once."""
        scores: list[dict[str, float]] = []
        for fit in (fits[i] for i in members):
            if fit.evaluator is None:
                fit.evaluator = eval_executor(
                    fit.doc, fit.executor, request, _inner_executor
                )
            scores.append(_score(fit.doc, fit.evaluator))
        return scores

    def on_epoch(members: Sequence[int]) -> None:
        # §2.2 `draw`: a new member per row for the new epoch
        for fit in (fits[i] for i in members):
            if fit.drawn is not None:
                fit.minibatches = fit.drawn.minibatches()

    outcomes = fit_loop(
        [fit.state for fit in fits],
        [fit.spec for fit in fits],
        step=step,
        evaluate=evaluate,
        on_epoch=on_epoch,
    )
    return [
        dataclasses.replace(
            outcome, draws=fit.drawn.record() if fit.drawn is not None else {}
        )
        for fit, outcome in zip(fits, outcomes)
    ]
