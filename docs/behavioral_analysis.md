# Behavioral analysis execution

Use the existing [behavioral workflow step](workflow_protocol.md#27-behavioral-steps--the-declarative-behavioral-runner)
for generation, retention, task-bound scoring, and decisions. It executes a
no-intervention protocol and writes `continuations.json`, `outcomes.json`, and
`decision.json`. Use the registered top-level `causalab` CLI, not an internal
module's `main`. The worked fixture is
[qualify.json](../tests/workflow/fixtures/behavioral/qualify.json), with its
[protocol](../tests/workflow/fixtures/behavioral/protocols/qa_probe.json).

A CPU fixture smoke (random weights, not a behavioral qualification):

```bash
uv run causalab validate tests/workflow/fixtures/behavioral/qualify.json --data-root tests/workflow/fixtures/behavioral --artifacts-root tests/workflow/fixtures/behavioral
uv run causalab run tests/workflow/fixtures/behavioral/qualify.json --data-root tests/workflow/fixtures/behavioral --artifacts-root tests/workflow/fixtures/behavioral --out runs/behavioral-smoke --engine pytorch_hooks --device cpu --batch-rows 4
```

For real work, author a study workflow with the frozen model, task-bound checker,
explicit EOS IDs, all-row retention, and one step per scientific split. The fixture's
small thresholds and bounded retention are only for testing the interface.

## Freeze the measurement

Record model/tokenizer revisions, precision, literal input format, scorer and
accepted forms, generation cap, explicit EOS IDs, and disjoint parent splits.

Author `decoding.mode: deterministic` and `decoding.eos_token_ids` explicitly
for reproducible stopping. Otherwise the PyTorch engine resolves EOS from the
model generation configuration, then the tokenizer. It draws directly from raw
logits, without inheriting sampling, repetition, suppression, or forced-token
processors from Transformers generation defaults. It records the effective EOS
IDs with each continuation. A different engine must demonstrate the same output
contract before use.

Use `retain.generations: all`. The default retention is bounded and is
insufficient for an all-example behavioral audit. `token_ids` are content before
EOS; `emitted_ids` include terminal EOS; `padding_ids` are post-stop slots.
`terminal_eos_id`, `stop_reason`, `eos_token_ids`, and `decoding` make termination
explicit. `text` preserves non-EOS special tokens. Score that authoritative text
or the exact IDs, never a cleaned display string. First-token accuracy is a
separate measure from complete-answer accuracy. Top-k order is not the greedy
decision under tied logits: use the saved first emitted ID or unrestricted
argmax on prompt-end logits.

The behavioral step uses the task's content-bound `ScoringSpec`. Check its
complete-answer and truncation semantics against your frozen measurement; do not
silently substitute a prefix match for whole-answer equality. In this runner a
length-capped row is classified `truncated`. If the research contract instead
accepts a proven answer boundary before the cap, that needs an explicitly defined
scorer and outcome contract before execution.

## Plan and validate on CPU

Keep one scientific split per behavioral step, with a split-qualified dataset
reference. The workflow validates the reference against the step's split before
execution. Use `causalab.workflow.behavioral_plan.plan_batches` when distributing
rows into computational batches: it groups by split and prepared target
prefix/output length before chunking, rejects duplicate IDs and cross-split
parents, and returns exact source indices. Shards may mix batches from different
splits; their scientific summaries must remain separate.

The current behavioral step's split purposes are development, reserve, and
confirmation. Additional diagnostic panels need separate documents/analysis;
do not relabel them as confirmation to satisfy the parser. Run the complete
workflow's validation before allocation and compare every resolved row ID with
the batch plan. Compilation alone does not establish coverage.

Smoke the actual save/reload/scoring path, including early/alternative EOS,
length caps, control-token prefixes, ties, split boundaries, mixed answer lengths,
and a short final batch. Compare a bounded sample with ordinary generation under
identical settings. Test production-sized export separately from model fit.

## Memory, preparation, and recovery

Unsaved, untransformed continuation-head metrics project at most 16 positions
at a time, then reduce and release logits. Text-only metrics use emitted IDs
without projection. Full tensor saves and transformed reads explicitly request
materialization; size them separately. Forward microbatching alone does not bound
those requested tensors. Continuation position j reads the distribution **after**
consuming emitted token j; use prompt-end logits for the first emitted decision.

Use `causalab.neural.sequences.prepare_sequences` to prepare a batch with shared
immutable tokenizer metadata; it verifies the tokenizer did not change before
returning. Pass the continuation record's `eos_token_ids` when preparing emitted
IDs so alternate terminal tokens cannot hide post-stop padding. Cache immutable tokenizer metadata outside token loops. Prepare/audit on CPU
where possible; retain every raw generation, but expand selected-prompt target
views only after selection. Measure preparation, generation, projection, saving,
and auditing separately. Balance shards by measured work without changing splits.
Validate record equivalence before changing batching or claiming speedups.

Use workflow `--resume` for completed steps: it verifies artifact and
implementation identities. Make independent chunks separate workflow steps if
chunk-level recovery is required. Never reuse a partial file as a completed step,
or remove source checks to make an old recovery script run as a fresh study.
Aggregate chunk counts over the original full split before applying the study's
qualification rule; individual chunk decisions are not full-split qualification.

## Target validation and reporting

Before analysing locations, compare target locations with natural continuations:
separators, exact IDs, predictive positions, and all supplied prefix tokens.
Natural generation, supplied separators, and supplied correct answer tokens are
separate conditions. Verify a known input perturbation changes the intended
readout. Do not silently insert whitespace into old records and relabel them.

Report the results with the estimand and the acceptance criteria stated beside
them, and keep the raw examples and error cases reachable from the report.
