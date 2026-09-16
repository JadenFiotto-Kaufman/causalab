# Shared execution for multi-token analysis

One scientific target is one next-token prediction. A model forward can serve
many targets: read the last prompt position for output zero, then the position
containing output j-1 for output j. A causal model cannot read later supplied
tokens at an earlier position. The library keeps target metrics separate while
sharing their forward, donor captures and available pre-intervention computation.

## Prepare exact sequences

The following uses a loaded PyTorch `ModelBundle` named `bundle`. Keep the model
revision, dtype, quantization, tokenizer and task splits fixed throughout an investigation.
For a quantized bundle, also include its `bundle.quantization` mapping in the
model dictionary passed to the workflow helper below.

```python
from causalab.neural.sequences import (
    prepare_sequence, pair_sequences, sequence_cohorts, write_sequence_workflow,
)

base = prepare_sequence(
    bundle.tokenizer, "Complete the sequence: 2, 4,", " 6, 8",
    example_id="even-base", split="eval", prefix_condition="correct",
)
donor = prepare_sequence(
    bundle.tokenizer, "Complete the sequence: 3, 6,", " 9, 12",
    example_id="triple-donor", split="eval", prefix_condition="correct",
)
# This automatic alignment is valid only if output lengths and semantic roles
# agree. Otherwise supply an explicit target_alignment or select a narrower
# target cohort. Do not assume that a number occupies one tokenizer token.
pair = pair_sequences(base, donor)

for cohort, rows in sequence_cohorts([pair]).items():
    length = rows[0]["output_length"]
    write_sequence_workflow(
        f"study/{cohort}",
        {"key": bundle.key, "revision": bundle.revision, "dtype": bundle.dtype},
        rows,
        layers=[0, 1, 2],       # expand after critical-location selection
        bands=[[0, 1], [1, 2]], # use the scientifically chosen layer bands
        positions=[{"index": -length - 1}],  # last original prompt token
    )
```

The helper writes `pairs.json`, `targets.json`, four intervention specifications
(`harvest`, `residual`, `attention`, `mlp`) and `workflow.json`. The harvest covers
all real input and output positions at every requested layer in one forward.
Each patching method includes all output readouts and an unchanged-base null
scored against the same donor targets. Each unique band is a separate
intervention; it is never combined with another band to save a forward.

Run from the CausaLab checkout, replacing the cohort path with the directory
returned above:

```bash
uv run causalab validate study/COHORT/workflow.json --data-root study/COHORT
uv run causalab run study/COHORT/workflow.json \
  --data-root study/COHORT --out study/runs --device cuda --batch-rows 16
```

The workflow runner executes its steps; external scheduling can instead launch
the independent method documents concurrently. Keep target readouts together.
Separate protocol requests or `--points` shards do not share an in-memory cache,
so choose shards with enough distinct interventions to benefit from shared donor
work. Bound batch rows and capture volume for long sequences.

Prepared rows preserve the exact IDs and offsets through an `input_encoding`
sibling. `counterfactual_inputs_encoding[0]` carries the donor's encoding. The
engine verifies the text and tokenizer identity, then left-pads those IDs without
retokenizing. All indices address this complete prepared frame, including any
special tokens. Do not add a `segments` section: preparation already owns the
frame. Use `chat=True` and optionally `system=...` during preparation when needed.

A text answer that changes the prompt's tokenization at its boundary is refused.
Revise that prompt or supply the exact continuation IDs. To analyze a recorded
baseline generation, pass its emitted IDs and `prefix_condition="baseline_generated"`.
Do not round-trip those IDs through text. Include EOS only as the last target;
tokens after EOS are refused. Empty or ambiguous semantic targets belong in the
research ledger as unavailable, not as invented labels in an executable cohort.

`sequence_cohorts` groups by split, prefix condition, and base/donor output
lengths, preserving full rows rather than repeating one row per target. It also
refuses a parent example appearing in conflicting splits, including as a donor.
Both callers and reports should use stable parent IDs, rather than assigning a
new ID to each token or pair. Choose train/evaluation membership before expanding
targets. Length cohorts are an execution detail; semantic roles still require
independent validation.

The generated patching documents apply the same location rule to source and
base. For unequal donor/base output lengths, a negative index relative to the
end may identify different roles. Use role-specific variable anchors or author
the source read's position separately. The donor's supplied prefix is part of
the intervention, including whether it follows its own reference or the base's.

## Add targets to existing experiments

`add_readouts(document, output_length, model="patched", name="targets", ...)`
returns a copy of an ordinary intervention specification (protocol_version 3,
`header` / `model` / `data` / `method`). It adds one `lm_head` read
per predictive position, with exact-ID accuracy, NLL, target logit and top-k
metrics. `target_prefix` names integer columns (default `output_0`, `output_1`,
...), and `alternative_prefix` adds target-minus-alternative logit differences.
Each metric has its own JSON file; all reads of the same model/input share one
forward. The same helper applies to DAS/DBM apply documents. Separate supervised
objectives still require separate fits; shared readouts do not change a fit.

Join metric rows with `targets.json` by their `example_id` and the target
ordinal in the metric name. Keep `produced_by` and sweep coordinates in every
join. NLL is negative log probability; `exp(-NLL)` is the target probability.
Do not replace missing scores with zero or mix units in one aggregate. Token
accuracy, semantic-slot success and full-generation success are different
statistics. Several targets sharing a run are not independent replications.

## PCA and logit lens use one harvest

For an aligned next-token view, use the workflow script
`causalab.analysis.sequence_activations` with inputs `acts` (one all-position
harvest) and `rows` (its prepared table). It gathers each target's predictive
position into `(examples, targets, features)`, including from flattened ragged
harvests. Preserve the original all-position artifact for input-token views and
high-dimensional neighbors. Do not use raw ragged padding as PCA observations.

`causalab.analysis.pca_by_position` accepts that tensor, explicit `train_rows`
indices, and `k`. It saves `mean`, `weight`, `coordinates` and a `spectrum` table
with a position column. Each position has a separate training-only mean and
basis. Supply concatenated train/evaluation activations in a declared common
order when projecting both; list only the training row indices. A new label or
target tab reuses coordinates rather than fitting again.

For an individual population, `fit_pca` retains its existing `weight` and
`spectrum` outputs and optionally saves `mean` and `coordinates`. The companion
`project_pca` script accepts `acts`, the frozen `mean`, and `weight`, and emits
`coordinates` for any split. `fit_pca` still pools all leading axes; use an
explicitly selected population or `pca_by_position` when pooling is not intended.
A grouped basis is an analysis artifact: select one position before treating it
as a localizer's initialization.

```python
from causalab.neural.shared.logit_lens import logit_lens

cells = logit_lens(
    bundle, "study/runs/sequence_analysis/harvest/residual_0.safetensors",
    k=10, batch_positions=128,
)
```

The logit lens uses the actual final normalization and output head, in bounded
position chunks. It runs no transformer blocks. Saved harvests must identify
the matching model, revision, dtype, quantization and `block_output` site. Tensor inputs are
caller-owned; preserve their provenance yourself. `target_ids` optionally supplies
one exact target ID per flattened vector, bounded by the decoder head's width
rather than the tokenizer's length (a padded vocabulary has more logit columns
than tokens). Records contain leading-axis indices,
top-k IDs/text/logits/probabilities, the log normalizer, and optional target logit
and log probability. Flattened ragged indices map back through the prepared
rows' token counts. Additional candidates can require another head projection,
but not another transformer pass. This API needs executable PyTorch modules,
not a tracing envoy outside a trace.

## Score one intervention-generated rollout

`add_rollout_readouts(document, output_length, ...)` adds all target scores and
decoded text to one greedy rollout. Supply prompt-only inputs, not a prepared
reference sequence. Target columns still name exact expected IDs.

```python
from causalab.neural.sequences import add_rollout_readouts

rollout = add_rollout_readouts(
    prompt_only_intervention_document, output_length=3,
    model="patched", name="rollout", target_prefix="cf_output_",
)
```

The first read uses prompt `-1`; target j>0 reads generated j-1. All reads of
the same model/input use one decode. A rollout remains sequential: the current
engine makes one prefill plus one step per requested generated activation.
Missing positions preserve the engine's unmatched/unavailable records. The
helper aligns ordinals; a variable semantic layout needs a separately verified
alignment before interpreting these scores as semantic-target effects.

Writes fire during prefill only. Generated-position writes, differentiating
through greedy decoding, and cross-point completed-rollout caching are not
implemented. Keep rollout targets in one group rather than a target sweep.
Neither the trajectory nor mutable KV/recurrent state may be reused as a clean
baseline for a different intervention merely because its emitted tokens match.

Correct fixed prefixes, frozen baseline-generated prefixes and intervention
rollouts answer different questions. Keep them in separate report tabs and
aggregates. Later rollout effects include changes to earlier emitted tokens;
their difference from fixed-prefix effects is not by itself a mediation estimate.

### Final PCA step: sparse concept probes

Append a script step using `causalab.analysis.sparse_pca_probe` after
`pca_by_position` (once per harvested layer). It consumes frozen coordinates;
PCA fitting and the probe's standardization must use the same training rows.
The row table is in coordinate order, with unique string `id`, a `split` of
`train`, `validation`, or `evaluation`, and a column for each concept. Use string
labels for categorical concepts and finite numbers for numeric concepts.
All categorical classes must occur in every split; numeric labels must vary
within each split. Hold out entities or input combinations through this table.
The relative `path` reference pins the label table. After intentionally changing
it, re-pin the workflow and execute without `--resume` to recompute the scores.

```json
"sparse_probes": {
  "type": "script",
  "script": {"module": "causalab.analysis.sparse_pca_probe"},
  "inputs": {
    "coordinates": {"step": "pca", "file": "coordinates.safetensors"},
    "rows": {"path": "data/probe_rows.json"},
    "concepts": {"output_label": "categorical", "operand_value": "numeric"},
    "layer": 12,
    "strengths": [0.1, 1.0, 10.0, 100.0]
  },
  "outputs": {
    "scores": "scores.json",
    "coefficients": "coefficients.json",
    "metadata": "metadata.json"
  }
}
```

The step fits L1 logistic regression (categorical) or Lasso (numeric) on
training-standardized PCs. It chooses the highest validation balanced accuracy
or R²; exact ties prefer fewer selected PCs, then the stronger penalty. It
retains that training fit for held-out evaluation. `strength` is `1/C` for
logistic regression and `alpha` for Lasso; scales are method-specific.
Nonconverged fits fail explicitly.

`scores` records one row per concept and position with the layer, split counts,
validation/evaluation scores, training-majority or training-mean baseline,
selected PCs, penalty, class labels and intercepts. `coefficients` retains all
PC coefficients and their selection indicators, per class where applicable.
Binary logistic coefficients describe the last class in `classes`. `metadata`
records split IDs, training means/scales, the candidate penalties and the fixed
solver seed. PC indices are zero-based in these artifacts.

Render each concept's evaluation score as a token-by-layer heatmap. Clicking a
site should show its selected PCs and standardized coefficients. Keep balanced
accuracy and R² on separate scales; R² may be negative. These scores measure
held-out decodability. A causal claim needs intervention evidence.

### Fourier probes on saved activations

`causalab.analysis.fit_fourier_probe` fits affine ridge readouts for one numeric
column at every supplied position. Use saved residuals, frozen PCA coordinates,
or the `coordinates` output of `project_subspace` for a frozen DAS basis. The
array has shape `(examples, positions, features)`; `(examples, features)` means
one position. Rows carry unique string `id`, a `split` of `train`, `validation`
or `evaluation`, and the numeric target. Their order must match the activations.

For period `T`, positive integer harmonic `k`, and `origin`, the targets are
`[cos(2*pi*k*(value-origin)/T), sin(2*pi*k*(value-origin)/T)]`. Defaults scan
every integer period from 2 through 150 at harmonic 1. Supply positive real
periods for a finer grid or another unit. Equal `k/T` values share one fit and
retain their aliases. Sampling can create further aliases, such as frequencies
above the Nyquist limit on integer labels; interpret them using the label grid.
A sine/cosine pair spans every phase origin. Separate fits for phase offsets
are redundant under this isotropic penalty and joint loss.

Metrics retain the number of observed phases and the shortest arc containing
them. Metadata retains numeric ranges per split. Inspect this coverage when
distinguishing a periodic readout from a fit to a short arc.

```json
"fourier": {
  "type": "script",
  "script": {"module": "causalab.analysis.fit_fourier_probe"},
  "inputs": {
    "acts": {"path": "acts.safetensors"},
    "rows": {"path": "data/probe_rows.json"},
    "target": "operand_value",
    "layer": 12,
    "representation": "residual",
    "origin": 0,
    "harmonics": [1],
    "alphas": [0.001, 0.01, 0.1, 1, 10, 100]
  },
  "outputs": {
    "weight": "weight.safetensors",
    "bias": "bias.safetensors",
    "plane": "plane.safetensors",
    "calibration": "calibration.safetensors",
    "predictions": "predictions.safetensors",
    "scores": "scores.json",
    "metadata": "metadata.json"
  }
}
```

Each position uses one training SVD for all frequencies, penalties and a seeded
shuffled-label control. The objective is summed squared error plus
`alpha * ||weight||²`, with a free intercept. Features are centered on training
rows. Isotropic ridge preserves equivalence between orthonormal DAS coordinates
`hQ` and their reconstruction `hQQᵀ`; per-coordinate rescaling would change
that penalty. PCA, DAS and any external preprocessing must use training data.
Keep a held-out probe evaluation set outside DAS fitting and rank selection.

Each frequency chooses its penalty by minimum validation pair MSE; exact ties
prefer the larger penalty. Report evaluation once using that frozen training
fit. The shuffled control permutes training target rows, then selects its own
penalty on the original validation targets. `shuffle_seed` defaults to zero.
This one shuffle is a diagnostic baseline. A broad scan needs independent
confirmation before a selected peak supports a research claim.

Outputs use float64. `weight` has shape `(positions, frequencies, features, 2)`;
`bias` has shape `(positions, frequencies, 2)`. Predictions have shape
`(examples, positions, frequencies, 2)`, with cosine first. The padded
orthonormal `plane` has the same shape as `weight`, and `calibration` has shape
`(positions, frequencies, 2, 2)`. Their product recovers the readout weight to
numerical tolerance. Read the recorded `rank` before using a plane: unused
columns are zero. These grouped analysis tensors need a selected position,
frequency and active rank before use as a geometric basis. The plane's rank
comes from the fitted weight. `target_rank` separately describes variation in
the training targets; a short arc can have a lower numerical target rank.

`scores` contains validation/evaluation pair MSE and variance-weighted joint
R², per-coordinate R², wrapped angular MAE in radians, phase counts and mean
radius, plus training-mean and shuffled baselines. Constant coordinates have
null R². Constant training targets have status `unavailable`; preserve the
reason when plotting. Period 2 on integer labels with origin zero has one
informative coordinate and a rank-one plane. Phase is defined only where
predicted radius exceeds `1e-8`. A phase locates the value modulo the effective
period `T/k`; it does not determine the original number across cycles.

`metadata` records the grid, label-row hash, split IDs, training means and
selection settings. Supply one unique string or integer in `position_labels`
per position for semantic token names, and
`source` for additional artifact references, units, label definitions and the
selected DAS rank/fit. The workflow records input digests and stamps tensor
provenance. Callers must keep the feature basis and position order consistent
when applying a fit, including in a workflow.

For a frozen readout, use `causalab.analysis.apply_fourier_probe` with `acts`,
`weight`, and `bias`. Its outputs are `predictions`, `radius`, `phase` in
`[0, 2*pi)`, and boolean `phase_defined`. An undefined phase has a zero storage
value and a false mask. Reuse the exact feature basis and position order from
the fit. Projection into the learned plane uses the saved calibration and bias
to recover cosine/sine coordinates.

Use the public artifact reader to verify a saved fit before further analysis:

```python
from pathlib import Path
from causalab.analysis.fourier_artifacts import load_fit

saved = load_fit(Path("fourier"), Path("acts.safetensors"), Path("data/probe_rows.json"))
```

The directory contains the seven outputs above. The reader checks the schema,
row hash and split IDs, tensor shapes, position labels, score grid, saved-plane
factorization and frozen predictions against the supplied original population.
It compares tensor identity stamps when present. Direct Python outputs can be
unstamped; their model provenance remains the caller's responsibility.

The returned mapping contains `metadata`, `scores`, `rows`, `identity`, and
float64 NumPy arrays `acts`, `weight`, `bias`, `plane`, `calibration`,
`predictions` and `truth`. `truth` has shape `(examples, frequencies, 2)`.
`artifacts` maps each native filename, plus `acts` and `rows`, to its absolute
`path` and `sha256`. All rows and frequencies are retained. Consumers choose
which measurements to display.

These probes follow the affine targets in
[Arithmetic in the Wild, §4 and Appendix G](https://arxiv.org/abs/2605.01148).
The ridge solver provides a direct fit for those targets. Optional PCA inputs
also support the setting studied by
[Engels et al.](https://arxiv.org/abs/2405.14860). The
[numeric and periodic encoding study](https://arxiv.org/abs/2502.00873) motivates
retaining scalar numeric probes alongside Fourier readouts. Angle decoding also
appears in [Probing for Arithmetic Errors](https://aclanthology.org/2025.emnlp-main.411/).

### Reconstruct a saved subspace component

`causalab.analysis.project_subspace` is a workflow script for any orthonormal
basis, including a fitted DAS subspace. Inputs are `acts` with shape `(..., d)`
and `weight` with shape `(d, k)`. Outputs are `coordinates` and `reconstructed`:

```text
coordinates = acts @ weight
reconstructed = coordinates @ weight.T
```

The operation uses CPU float64, preserves leading dimensions and does not
center, restore a mean or add the complementary residual. It refuses nonfinite
inputs, incompatible shapes and a basis outside the existing featurizer's
orthonormality tolerance. Select swept inputs with `slot` and `entry` on the
[workflow reference](workflow_protocol.md#3-cross-step-wiring--the-reference-grammar),
so the runner inherits the selected tensors' model and site metadata:

```json
"projection": {
  "type": "script",
  "script": {"module": "causalab.analysis.project_subspace"},
  "inputs": {
    "acts": {"path": "acts.safetensors", "slot": "acts", "entry": {"layer": 12}},
    "weight": {"path": "weight.safetensors", "slot": "weight", "entry": {"layer": 12, "k": 2}}
  },
  "outputs": {
    "coordinates": "coordinates.safetensors",
    "reconstructed": "reconstructed.safetensors"
  }
}
```

Single-entry files need no selector. For direct Python calls, select tensors
with `causalab.io.step_io.read_tensor` before passing them to `main`; the caller
owns provenance outside the workflow. Verify model, site and population
correspondence before combining activations and a basis.

To decode only this subspace component, pass the reconstructed tensor to
`causalab.neural.shared.logit_lens.logit_lens` with the matching loaded model
bundle and target IDs. The existing logit lens applies the model's final
normalization and vocabulary head. This readout contains no interchange and
no component outside the selected subspace. Save the basis/activation identities
and the reconstruction formula beside the result; do not present it as a raw
residual harvest.
