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
  DeltaNet kernel boundary, the routed-experts dispatch) and the `train`
  runner — cohorts, row budgets, captured graphs — which is why it is the only
  registered engine declaring `grad`.
- `engines/nnsight_tracing/` — the second engine: one trace over an envoy tree,
  with fused-forward interiors addressed through nnsight `.source`.
- `engines/nnsight_nnterp/` — the nnterp engine, outside the closed
  registry (a caller's explicit choice): one trace per forward group over
  nnterp's standardized tree, locally or on NDIF. It declares `grad` too — its
  `train.py` fits a document on the shared loop, locally.
- `token_positions.py` — char→token position utilities (offset-mapping based,
  chat-prefix aware). Backbone-agnostic; the task packages' `token_positions.py`
  modules build on it. The protocol-native position service is
  `shared/encoding.py`; this module remains the home of the legacy declarative
  vocabulary the tasks encode against.
- `pytorch_hooks/` — **not** an engine: a deprecation shim that aliases
  `engines/pytorch_hooks/` through `sys.modules` for one deprecation beat.

The two engines' answers are asserted to agree over the whole shared component
vocabulary, read and written — on a tiny fixture in
`tests/neural/engines/nnsight_tracing/test_parity_a3b_sweep.py` and on the real
Qwen3.6-35B-A3B in `tests/golden/test_a3b_engine_parity.py`.

Everything else that used to live here — the Plan IR and its scheduler, the old
nnsight *pipeline*, spec persistence — was replaced by the protocol layer
(`causalab/protocol`) plus the engines above. The nnterp engine here is not
that pipeline returning; it is a protocol engine like the reference one.
