# Saved hypothesis comparisons

| Field | Value |
|---|---|
| **Question** | Can saved outputs be compared against several hypotheses on the same pairs? |
| **Method** | Exact-pair export and comparison |
| **Model** | Symbolic addition modulo three; constructed neural outputs |
| **Data** | Four comparison pairs; generated broad and narrow families |
| **Documents** | Python models and generators linked below |
| **Cost** | CPU only; no model weights |
| **Reproduced** | ⚠ CPU handoff only; neural experiments require a model run |

## TL;DR

Export complete pairs and each hypothesis's predictions before intervening on
the neural model. Compare saved global top-1 outputs against those predictions.

## The protocol

The target swaps the intermediate value of A. Alternatives swap B, nothing,
or the output. Narrow families distinguish each alternative from the target.
The [models](models.py) share one scoring rule; the
[generators](counterfactuals.py) declare family and split membership.

## Run it

The check is CPU-only and runs against the two modules in this directory.
[`models.py`](models.py) declares `MODELS`, `DEFAULT_MODEL` and `HYPOTHESES`
(each hypothesis a model name plus the variables it intervenes on);
[`counterfactuals.py`](counterfactuals.py) declares `make_datasets(model, n=,
seed=)` and `random_pairs(model, n, seed=)`. The run here built the families
with `n=4` pairs each and 16 random pairs, exported every pair with its
per-hypothesis predictions through
`causalab.analysis.hypothesis_artifacts.export_hypotheses`, and scored the
families with `causalab.causal.causal_utils.distinguishability_report`, whose
JSON is the audit below.

Use a new export directory for another run. For research-sized tables, build
1,000 pairs per family and 1,000 random pairs: each family then has 1,000
training, 500 validation, and 500 test pairs.

## Experimental design

This CPU example checks the artifact handoff. Its case numbers keep prompts
separate across splits; they do not establish generalization to new arithmetic
rules. Use task-relevant groups for a research experiment.

For neural experiments, follow the [comparison guide](../../docs/hypothesis_analysis.md).
Use validation for selection and retain the selected fit for test evaluation.

## Results

The handoff test uses four pairs. Target accuracy is 75%, alternative accuracy
is 25%, and the gap is 50 percentage points. Three pairs distinguish the
hypotheses; target accuracy on those pairs is 2/3. One neural answer is outside
the declared answer vocabulary and counts as wrong for both hypotheses.
These are constructed test values, not model results.

## Limits

This example does not establish neural support for either intermediate variable.
The CPU checks do not verify GPU execution, fitted subspaces, or random controls.

## Next

Run full interventions and DAS on the exported pairs, then compare the saved
outputs. Report absolute accuracy, the gap from every alternative, and remaining
distance from the ceiling.
