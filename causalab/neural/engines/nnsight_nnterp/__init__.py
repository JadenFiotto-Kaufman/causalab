"""The nnsight + nnterp engine: interventions over the standardized envoy
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
* **unclaimed** — cross-point interning (§3): the engine takes the
  campaign's handle and drops it, so every point runs its own forwards.

Gradient-enabled groups run under ``torch.enable_grad()`` so the training
tier can build on the same ``_run_group``; nothing here trains yet.

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

Remote mode needs a **trusted, in-process NDIF deployment with the same
``causalab`` installed server-side**: the block is ``causalab``'s functions
working on the served model itself, and it refuses a ``meta`` copy by name.
:mod:`.executor` states the rest — the eager switch a hard kill can strand,
featurizer stages shipping by value, where bit-identity holds.

``tests/neural/engines/nnsight_nnterp/test_ndif_shape.py`` pins the
structure, and ``test_faithful_server.py`` runs the deserialized program
against a separately loaded model, results through a ``torch.save`` round
trip — nnsight's ``remote="local"`` dry run executes against the caller's
own frame and so hides all three failure classes.

Requires the ``nnsight`` extra (``pip install 'causalab[nnsight]'``), which
carries both packages.
"""

from causalab.neural.engines.nnsight_nnterp.engine import NnterpEngine

__all__ = ["NnterpEngine"]
