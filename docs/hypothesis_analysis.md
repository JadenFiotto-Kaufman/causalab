# Compare hypotheses on saved pairs

The [CPU example](../demos/hypothesis_testing/hypothesis_testing.md) exports native
pair tables and frozen predictions. Use the pair table directly in intervention
documents; its `label` is the target's intervened answer.

## Neural outputs

Save `top_k` with `k: 1` from the final, intervened `lm_head` read. Also save
`token_logits` over the declared answer vocabulary. The top-1 token comes from
the full vocabulary. Restricting the argmax to answer tokens would overstate
accuracy. Use an exact `match` against `label` during validation.

## Comparison

`causalab.analysis.compare_hypotheses.compare_saved_outputs` takes the pair
table, prediction table, native top-1 metric rows, and the run's tokenizer.
It resolves answer forms through the same token conversion used by native
exact-match metrics. Pass the run's `token_form` setting unchanged.

The workflow entry point accepts `pairs`, `predictions`, `neural`, `target`,
`alternatives`, `metric`, `token_form`, `tokenizer`, `tokenizer_revision`, and
an optional `split`. The tokenizer must be available locally. Its output is
`comparisons`, a native JSON table with one row per pair and alternative.

Keep the source run manifest and dataset pin with this table. The comparison
checks IDs, scoring identity, completeness, and duplicate rows; a metric table
alone cannot prove which dataset or tokenizer produced a run.

## Reduction and selection

Reduce by run point, family, split, and alternative. `value` is target agreement
minus alternative agreement; its mean times 100 is the percentage-point gap.
`target_score` and `alternative_score` retain absolute accuracy. Keep missing
outputs and their reasons visible. Report counts of both scored and excluded
pairs. For distinguishing-pair results, filter `distinguishing` first. An empty
subset is unavailable, not zero accuracy.

For each location and rank, average validation accuracy within each family.
Weight broad accuracy by 0.10 and divide 0.90 equally among narrow families.
Choose the smallest rank within 0.02 of that location's best weighted validation
accuracy. Freeze the chosen fit before testing. Do not weight individual pairs
again after computing family means.

Use native `workflow.scripts.reduce` and `workflow.scripts.select` for these
reductions and selection. Compare the selected fit with random subspaces at
the same site and rank. Preserve seed, fit, dataset, and run identities.
