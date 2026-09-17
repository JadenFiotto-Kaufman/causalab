"""This engine's train runner (spec §2.11): the forwards of a fit as traces,
under the shared loop.

What a ``train`` section means, and the update loop itself, are
:mod:`causalab.neural.shared.training`'s. What is here is how a step's
forward runs on this engine:

* a minibatch is an :class:`~causalab.neural.engines.nnsight_nnterp.executor.
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

import functools
from typing import Any, Mapping, Sequence

from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.shared.encoding import EncodedBatch
from causalab.neural.shared.execution import TrainOutcome
from causalab.neural.shared.executor_base import ExecutorBase
from causalab.neural.shared.featurizers import featurizer_cache
from causalab.neural.shared.training import (
    Fit,
    evaluate_fits,
    finish,
    fit_loop,
    prepare_fit,
    step_loss,
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
    """This engine's ``ExecutorFactory``: a fit's minibatch or eval executor
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


def _step_forward(current: Sequence[tuple[Fit, ExecutorBase]]) -> None:
    """One optimizer step's forwards, member by member: the loss's reads run
    the minibatch executor's traces under ``enable_grad``, and the backward
    runs once they have exited."""
    for fit, minibatch in current:
        with featurizer_cache():
            step_loss(fit, minibatch).backward()


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
    fits = [
        prepare_fit(doc, executor, executor_factory=_inner_executor)
        for doc, executor in zip(docs, executors)
    ]
    fit_loop(
        fits,
        step_forward=_step_forward,
        evaluate=functools.partial(evaluate_fits, request=request),
    )
    return [finish(fit) for fit in fits]
