"""Training a document's featurizers (spec §2.11, §8): the document declares
the fit, an engine owns the forward, and everything between the two lives
here — so a fit means one thing whichever engine steps it.

Semantics implemented exactly as declared:

* ``seed`` drives featurizer init, data order, and every draw, so a
  ``{"sweep": [0,1,2]}`` on ``seed`` yields three genuinely different fits.
  Init reaches the featurizer as an explicit argument (``executor.seed`` →
  ``build_stack(seed=…)``, a *local* generator) rather than through the
  global RNG, because the same construction runs on apply paths where no
  loop entry ever executes; the batch order has its own local ``order_rng``.
  Nothing a fit does draws from the *global* RNG — not building its stages
  (``matrix_exp``/``stiefel`` complete torch's own orthogonal-parametrization
  basis from the stage's generator, and the ``cayley`` parametrization draws
  nothing), not the loop, whose model is in eval mode. That is what lets
  several fits share one loop, and what lets a fit run in a process it shares
  with other callers: a served model on NDIF. The one draw the loop does make
  is a
  ``hard_concrete`` gate's training mask, once per optimizer step from the
  member's own ``mask_rng`` (``Gate.resample``), so a member's samples are a
  function of its own document and seed whatever fits beside it;
* ``objective`` terms are differentiable metric tensors (``cross_entropy``,
  ``logit_diff``, ``soft_accuracy``, ``kl``, ``js`` — the last two toward
  another read, ``js`` optionally restricted to a per-row answer set; a
  term's weight is signed, so ``-1`` on a margin maximizes it) plus regularizers
  (``l1``/``l2`` over the params of one featurizer or of several together; a
  ``gate``'s l1 is the mean soft mask — the DBM sparsity semantics — and a
  list of gates is one mean over all their units, so one weight counts
  selected units across layers; ``l0`` is a ``hard_concrete`` gate's expected
  kept fraction, the penalty that goes with a sampled mask);
* ``anneal`` targets ``<featurizer>.<slot>.temperature`` linearly from
  start to end over the first ``frac`` of total steps, then holds;
* after every optimizer step each trained stage is **projected** back onto
  its feasible set (``Stage.project``) — a ``clamp`` gate clips its mask into
  ``[0, 1]``, everything else is a no-op;
* a ``trajectory`` save entry (§2.12) photographs the trained slots after the
  scheduled updates — ``count: n`` equally spaced, the last at the end — with
  the loss, every term's value and weight, every gate's mask numbers and every
  controlled value beside each photograph (``TrainOutcome.checkpoints``);
* ``control`` moves a hyperparameter — a named term's weight, or an
  anneal-style attribute — every update so a fit signal (the kept-unit count
  of one gate, or summed over several) follows its declared setpoint ramp
  (:mod:`.control`); the authored value is the start, the trajectory is
  recorded on the outcome. Each fit has its own controllers, checkpoints and
  live weights;
* ``eval`` runs on the declared split in eval mode (hard gate, no grad)
  every N epochs; ``early_stop`` tracks the eval metric with
  patience;
* ``batch.pairs`` counts base+counterfactual pairs; roles are sliced together
  (rows are paired by index, §2.2).

The model's weights are frozen at load; only featurizer slots optimize.

**Spec, state, and what an engine supplies.** A fit is three things, kept
apart so the loop that steps it never sees what runs a forward:

* :class:`~.spec.FitSpec` — the fit as **plain data**, frozen: the seed, the
  ``train`` section resolved against the point's rows (the minibatch
  partition as index lists, the update budget, the eval cadence, early stop,
  the schedules as authored, the checkpoint steps), the objective's terms
  with their metrics' answers already token ids (:class:`~.spec.
  ResolvedMetric`, ``metrics.metric_in_ids``), and a recipe per trained stage
  (``featurizers.StageRecipe``). No executor, document, tokenizer or model;
  it pickles with :mod:`pickle` in a few kilobytes. :class:`~.spec.ScoreSpec`
  is the same for one eval pass over one split;
* :class:`~.state.FitState` — everything the updates **move**: the stages,
  the optimizer and a constraint's duals, the two seeded generators, the
  step, epoch, order and position, the live weights, controllers, phases and
  their traces, the early-stop best and its snapshot, the checkpoints.
  :func:`~.state.build_fit_state` builds it from the spec — over stages an
  engine hands it (its point executor's own, so the finish phase sees the
  fitted objects by identity), or, given none, over stages it builds from
  the spec's recipes under the same seeding discipline, bit-identical to the
  executor's (:func:`~.state.build_stages`). It pickles too, a parametrized
  ``subspace`` included, with the optimizer still holding the stages' own
  parameters on the other side;
* the **engine's callbacks** to :func:`~.loop.fit_loop`, each handed member
  *indices* into the loop's states, so whatever an engine keeps per fit —
  executors, a graph pool, a store's tallies, the drawn roles — lives in a
  list of its own beside them: ``step(members)`` runs this update's grad
  forwards and the backward of each member's :func:`~.objective.step_loss`;
  ``evaluate(members)`` returns each member's :func:`~.objective.score`;
  ``on_epoch(members)`` hears of every epoch after the first, where a §2.2
  ``draw`` is redrawn.

The objective and the score are **functions of reads**: ``step_loss(state,
spec, read)`` and ``score(score_spec, read)``, where ``read(name)`` is the
dense value of a declared read for the rows in play — an executor's
``dense_value`` here, anything that answers elsewhere. The loop owns the
rest of an update: the epoch order, train mode and mask draws, phases and
anneals, ``zero_grad``, the lr schedule, the optimizer step, the projection,
the controllers, the checkpoints, early stop with its snapshot and restore,
and the outcome. The reference engine plugs in a cohort's row-bounded
windows or a CUDA-graph replay; the nnterp engine each member's own traces.

:mod:`.executors` is the executor side, shared because every engine does it
alike: making the spec from a document and its point executor, building the
stages on the executor's cache, cutting the minibatch executors through the
engine's :class:`~.executors.ExecutorFactory`, scoring an eval executor.
Nothing the loop imports reaches it.

| module | holds |
|---|---|
| :mod:`.spec` | ``FitSpec``, ``ScoreSpec``, ``ResolvedMetric``, ``EarlyStop`` — plain data |
| :mod:`.state` | ``FitState``, ``build_fit_state``, ``build_stages``, the optimizer and a constraint's dual groups |
| :mod:`.objective` | ``step_loss``, ``score``, ``metric_tensor``, ``regularizer`` |
| :mod:`.loop` | ``fit_loop``, ``begin_update``, ``record_eval``, ``read_signals``, ``after_update``, ``finish`` |
| :mod:`.schedules` | ``control`` / ``anneal`` / ``phases`` / the lr schedule, bound to a fit's state |
| :mod:`.control` | the PID law, pure Python |
| :mod:`.diagnostics` | ``fit_diagnostics``, ``checkpoint``, ``snapshot`` / ``restore`` |
| :mod:`.executors` | ``fit_spec``, ``score_spec``, ``seeded_stages``, ``minibatch_executors``, ``ExecutorFactory``, ``eval_executor``, ``eval_pass`` — engine-side |
| :mod:`.draw` | ``Drawn`` (§2.2 ``draw``), ``slice_rows`` — engine-side |
"""

from causalab.neural.shared.training.diagnostics import fit_diagnostics
from causalab.neural.shared.training.loop import finish, fit_loop, record_eval
from causalab.neural.shared.training.objective import (
    metric_tensor,
    regularizer,
    score,
    step_loss,
)
from causalab.neural.shared.training.spec import (
    EarlyStop,
    FitSpec,
    ResolvedMetric,
    ScoreSpec,
)
from causalab.neural.shared.training.state import (
    FitState,
    build_fit_state,
    build_stages,
)

__all__ = [
    "EarlyStop",
    "FitSpec",
    "FitState",
    "ResolvedMetric",
    "ScoreSpec",
    "build_fit_state",
    "build_stages",
    "finish",
    "fit_diagnostics",
    "fit_loop",
    "metric_tensor",
    "record_eval",
    "regularizer",
    "score",
    "step_loss",
]
