"""Training a document's featurizers (spec §2.11, §8): the document declares
the fit, an engine owns the forward, and everything between the two lives
here — so a fit means one thing whichever engine steps it.

Semantics implemented exactly as declared:

* ``seed`` drives featurizer init, data order, and every draw, so a
  ``{"sweep": [0,1,2]}`` on ``seed`` yields three genuinely different fits.
  Init reaches the featurizer as an explicit argument (``executor.seed`` →
  ``build_stack(seed=…)``, a *local* generator) rather than through the
  global RNG, because the same construction runs on apply paths where no
  loop entry ever executes; the batch order has its own local ``order_rng``;
  ``torch.manual_seed`` as each fit's stages are built — the one deliberate
  use of the global RNG — seeds what the init draws beyond the local
  generator (``matrix_exp``/``stiefel`` complete torch's own
  orthogonal-parametrization basis from it; the ``cayley`` parametrization
  draws nothing). Nothing draws from the *global* RNG during the loop itself
  — the model is in eval mode — which is what lets several fits share one
  loop: at loop entry the global RNG belongs to the last member prepared, and
  no member's numbers depend on it. The one draw the loop does make is a
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

**The engine seam.** :func:`~.loop.fit_loop` steps any number of
:class:`~.fit.Fit` objects in lockstep and owns every part of an update that
is not a forward. An engine supplies three things:

* an :class:`~.fit.ExecutorFactory` to :func:`~.fit.prepare_fit` — how a
  minibatch executor (a row selection of the point's frame, gradients on) and
  the eval executor are built over the point's executor and its stage cache;
* ``step_forward`` — this step's grad forwards and the backward of each
  member's :func:`~.loop.step_loss`. The reference engine packs a cohort's
  members into row-bounded windows, one forward each, or replays a CUDA
  graph; the nnterp engine runs each member's own traces;
* ``evaluate`` — the eval pass, ending in :func:`~.loop.record_eval`.
  :func:`~.loop.evaluate_fits` is the plain one.

| module | holds |
|---|---|
| :mod:`.objective` | ``metric_tensor``, ``regularizer`` |
| :mod:`.fit` | ``Fit``, ``prepare_fit``, ``ExecutorFactory``, the optimizer and a constraint's dual groups |
| :mod:`.draw` | ``Drawn`` (§2.2 ``draw``), ``slice_rows`` |
| :mod:`.schedules` | ``control`` / ``anneal`` / ``phases`` / the lr schedule, bound to a fit |
| :mod:`.control` | the PID law, pure Python |
| :mod:`.loop` | ``fit_loop``, ``step_loss``, ``score``, ``evaluate_fits``, ``record_eval``, ``finish`` |
| :mod:`.diagnostics` | ``fit_diagnostics``, ``checkpoint``, ``snapshot`` / ``restore`` |
"""

from causalab.neural.shared.training.diagnostics import fit_diagnostics
from causalab.neural.shared.training.fit import ExecutorFactory, Fit, prepare_fit
from causalab.neural.shared.training.loop import (
    evaluate_fits,
    finish,
    fit_loop,
    record_eval,
    score,
    step_loss,
)
from causalab.neural.shared.training.objective import metric_tensor, regularizer

__all__ = [
    "ExecutorFactory",
    "Fit",
    "evaluate_fits",
    "finish",
    "fit_diagnostics",
    "fit_loop",
    "metric_tensor",
    "prepare_fit",
    "record_eval",
    "regularizer",
    "score",
    "step_loss",
]
