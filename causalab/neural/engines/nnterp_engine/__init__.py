"""The nnterp engine: interventions over the standardized envoy
tree, locally or on NDIF.

The engine runs one nnsight trace per forward group over the tree
:class:`nnterp.StandardizedTransformer` builds — the same block list, embedding,
final norm and head under one set of names for every architecture nnterp
standardizes — and lands the protocol's writes and reads on the envoys'
``input`` / ``output``. Everything that is not *how a tensor is reached* is the
shared layer's (:mod:`causalab.neural.shared`): site resolution, position
frames, write math, featurizers and the run receipt.

Tiering, as built:

* **served at a module boundary** — the block-shaped taps every family shares
  (``block_input`` … ``lm_head``), the sub-child taps of the registry family
  whose predicate recognizes the *raw* module tree (``attention_input_norm``,
  ``block_mid``, ``mlp_activation``, the router and shared-expert boundaries
  of the Qwen3.5-MoE family, …), landed on the envoys' ``input`` / ``output``;
* **served through ``.source``** — the interiors the address table
  (:mod:`.sources`) reaches inside one forward: the attention function's
  slots and the pattern's write (``attention_query`` … ``attention_z``), the
  DeltaNet kernel boundary and its per-chunk state (``delta_conv`` …
  ``delta_kernel_output``, ``deltanet_query`` / ``key`` / ``state``), the
  routed-experts interior and its ragged ``expert:`` face
  (``expert_gate_proj`` … ``expert_output``, ``expert_permutation``);
* **served in the generated frame** — a group with a continuation read runs
  as one ``model.generate`` trace: writes and prompt-frame reads bind the
  prefill, the decode steps are walked with ``tracer.iter``, and module
  boundaries, the attention function's stackable slots and the DeltaNet
  state (through the recurrent kernel's own address) are read per step;
* **refused by name** — the per-token DeltaNet faces the chunked prefill
  kernel never materializes (``delta_kv_mem``, ``delta_state_update``,
  ``delta_state``), any interior on a tree the table has no rows for, and
  every other interior in the generated frame (the decode dispatches
  different kernels, so a prefill address is no evidence the tensor exists
  per step). The reference engine serves all of it;
* **trained, locally or on NDIF** — a ``train`` document (§2.11) is fitted
  on the shared loop (:mod:`causalab.neural.shared.training`) by one body,
  wherever it runs (:mod:`.fit`): :mod:`.train` plans the fit as data — the
  ``FitSpec``, one gradient-enabled template program per forward group over
  the point's whole frame with its featurizer stacks *by name*, the eval
  split's programs — and ``fit_body`` builds the stages, the optimizer and
  the controllers from it where the model is, runs each step's groups as
  traces (cacheless, as every prompt forward is) whose reads flow between
  them on the device with their graph, and runs the backward once the traces
  have exited. Locally that body runs in this process; with ``remote`` it is
  the body of **one** session — one job per fit — and the fitted state comes
  home as plain data, loaded into the client's own stages. Every fit is a
  cohort of one. Featurizer slots only, fp32 losses, evals on epoch
  boundaries — the reference engine's tier, and on CPU fp32 its weights to
  the bit, remote as local. A fit whose objective, eval or operand read the
  block cannot finish (ragged, routed, per-fire, generated) is refused by
  name, and a §2.2 ``draw`` — re-planned each epoch — fits locally only;
* **unclaimed** — cross-point interning (§3): the engine takes the
  campaign's handle and drops it, so every point runs its own forwards — a
  fit's source forward included, on every step.

The engine is **NDIF-shaped**, with one code path for local and remote
execution. nnsight ships a traced block as its source plus every name the
block reads, each pickled whole; a real server returns only saved
block-level variables, and sees neither a client-side mutation nor a
client-side config change. The structure follows from those rules:

* no trace body reads ``self`` — each group is planned into a frozen
  :class:`~.program.GroupProgram` before the trace (:mod:`.executor`), and the
  block is module-level functions over the model and that program
  (:mod:`.landers`);
* one saved container per block, bound at block level; reads are gathered at
  their positions inside the forward and detached there, fires and mismatch
  counts come back as data;
* the eager-attention switch is the block's own, before any operation;
* with ``remote`` set, a whole point runs as one ``model.session`` — the
  groups in dependency order, operands flowing between the traces on the
  server — against a weight-free bundle (:mod:`.loading`), whose ``remote``
  the executor and the engine inherit.

* a whole fit is one ``model.session`` too, whose body is the saved
  container and one call (:func:`.fit.run_fit`): everything the fit moves is
  created in that call, server-side.

Remote mode needs a **trusted, in-process NDIF deployment with the same
``causalab`` and ``nnterp`` installed server-side**: the block is
``causalab``'s functions working on the served model itself, and it refuses
a ``meta`` copy by name. The **version guard** (:mod:`.versions`) holds the
server to that before any job is submitted: the server's ``/env`` must
report this client's ``causalab`` and ``nnterp`` versions exactly, checked
once per host per process. :mod:`.executor` states the rest — the eager
switch a hard kill can strand, an inference point's featurizer stages
shipping by value, where bit-identity holds.

``tests/neural/engines/nnterp_engine/test_ndif_shape.py`` pins the
structure, and ``test_faithful_server.py`` / ``test_remote_fit.py`` run the
deserialized program — a point's, a whole fit's — against a separately
loaded model, results through a ``torch.save`` round trip — nnsight's
``remote="local"`` dry run executes against the caller's own frame and so
hides all three failure classes.

Requires the ``nnterp`` extra (``pip install 'causalab[nnterp]'``), which
carries both packages.
"""

from causalab.neural.engines.nnterp_engine.engine import NnterpEngine

__all__ = ["NnterpEngine"]
