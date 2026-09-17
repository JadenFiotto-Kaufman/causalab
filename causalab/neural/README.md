# causalab.neural

The execution side of the intervention-protocol stack (spec §8 of
`docs/intervention_protocol.md`). Two engines implement the same contract over
one shared body of engine-neutral machinery; `docs/CODEBASE.md` §3 is the module
map, and `docs/running_experiments.md` §6 is the user-facing routing table.

- `shared/` — everything that is not *how a tensor is reached*: the component →
  tap map and its write-policy tables (`sites.py`), tokenization and position
  frames (`encoding.py`), the closed `do` set (`mechanisms.py`), featurizer
  application, metric lowering, output writing, tensor layouts, the
  `ExecutorBase` every executor inherits, and the engine-neutral half of
  training (`training/`: what a `train` section means, and the update loop an
  engine plugs its forward into).
- `engines/pytorch_hooks/` — the **reference** engine: raw module hooks, plus
  the interior taps no hook can reach (the eager attention call, the Gated
  DeltaNet kernel boundary, the routed-experts dispatch) and its `train`
  runner — cohorts, row budgets, captured graphs.
- `engines/nnterp_engine/` — the second engine, registry name `nnterp`: each
  forward group planned into a frozen program and run as one nnsight trace
  over nnterp's standardized tree — envoys for module boundaries, a `.source`
  address table for fused-forward interiors, one `model.generate` trace for a
  continuation read — in this process or, with `remote=`, on NDIF (one
  session per point, against a weight-free bundle; it needs a trusted
  deployment with the same `causalab` installed server-side). It declares
  `grad` too — its `train.py` fits a document on the shared loop, locally.
- `token_positions.py` — char→token position utilities (offset-mapping based,
  chat-prefix aware). Backbone-agnostic; the task packages' `token_positions.py`
  modules build on it. The protocol-native position service is
  `shared/encoding.py`; this module remains the home of the legacy declarative
  vocabulary the tasks encode against.
- `pytorch_hooks/` — **not** an engine: a deprecation shim that aliases
  `engines/pytorch_hooks/` through `sys.modules` for one deprecation beat.

The two engines' answers are asserted to agree over the whole shared component
vocabulary, read and written — on a tiny fixture in
`tests/neural/engines/nnterp_engine/test_parity_a3b_sweep.py` and on the real
Qwen3.6-35B-A3B in `tests/golden/test_a3b_engine_parity.py`.

Scheduling, planning and persistence are the protocol layer's
(`causalab/protocol`); the nnterp engine is a protocol engine like the
reference one, not a pipeline of its own.
