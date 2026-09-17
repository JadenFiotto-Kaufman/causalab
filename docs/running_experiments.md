# Running experiments

Causalab runs neural-network interventions from a **document**: a JSON file that
names the model, the data, the activations to read, the edits to make, and the
numbers to save. The document is the experiment — there is no Python config
layer, and nothing about a run is decided by code you write.

This page is the path from "I have a hypothesis" to "I have a saved,
digest-stamped result", plus the table of every hookpoint the
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) architecture
exposes.

Reference material this page points at rather than repeats:
[`intervention_protocol.md`](intervention_protocol.md) (the normative spec),
[`workflow_protocol.md`](workflow_protocol.md) (chaining documents),
[`CODEBASE.md`](CODEBASE.md) (module map), [`TESTS.md`](TESTS.md) (test tiers).
These pages describe the tree they are committed in; the run receipt names
the commit a result was produced from.

## Setup

```bash
uv sync                       # the nnterp engine ships in the dev group
uv run causalab --help
```

## 1. Serialize a dataset

A document names a dataset **by ref**; a ref resolves by reading bytes under
`--data-root`, which defaults to the task packages themselves — every task ships
its table under `causalab/tasks/<task>/data/`, so `<task>/data/<variant>#<split>`
resolves with no flag (`causalab/tasks/README.md` §2). Nothing is generated during a load, so `validate` needs no task
code, no tokenizer and no network — and a document's digest is a function of
committed bytes.

```bash
uv run python scripts/build_task_dataset.py \
    --task MCQA --n 32 --seed 0 --split all --target-variable answer \
    --out data/mcqa.json
# wrote data/mcqa.json (32 rows, digest 355e6b69d4b5…)
```

The command above is the record of the parameters the bytes came from: the
table is a build product, and nothing sits beside it. The workflow that names
the table pins its digest in its `pins` section (workflow spec §7).

## 2. Write the document

An interchange intervention — read the answer-slot residual stream from the
counterfactual prompt, patch it into the base prompt, score what changed. Save
as `patch.json`:

```json
{
  "header": {
    "protocol_version": "3",
    "description": "Interchange the answer-slot residual stream at one layer."
  },
  "model": {"key": "Qwen/Qwen3.6-35B-A3B", "revision": "main"},
  "data": {
    "base": {"dataset": "mcqa", "field": "input"},
    "counterfactual": {"dataset": "mcqa", "field": "counterfactual_inputs[0]"}
  },
  "method": {
    "sites": {
      "target": {"component": "block_output", "layers": [20]},
      "lm_head": {"component": "lm_head"}
    },
    "reads": {
      "v_cf": {"site": "target", "pos": -1, "model": "original", "input": "counterfactual"},
      "logits": {"site": "lm_head", "pos": -1, "model": "patched", "input": "base"}
    },
    "writes": {
      "patch": {"site": "target", "pos": -1, "do": {"swap": "v_cf"}}
    },
    "intervened_models": {
      "patched": {"input": "base", "writes": ["patch"]}
    },
    "metrics": {
      "iia": {
        "kind": "match",
        "of": "logits",
        "expected": "label",
        "token_form": "space_prefixed"
      },
      "logit_diff": {
        "kind": "logit_diff",
        "of": "logits",
        "a": "cf_answer",
        "b": "base_answer",
        "token_form": "space_prefixed"
      }
    },
    "save": [
      {"value": "iia", "model": "patched", "input": "base", "file_path": "iia.json"},
      {
        "value": "logit_diff",
        "model": "patched",
        "input": "base",
        "file_path": "logit_diff.json"
      }
    ]
  }
}
```

Reading it in section order (the recommended order — a document in another
order warns and runs identically; `save` is conventionally last):

- **`sites`** is the complete tap inventory. Every address a read or write
  names, `lm_head` included. There are no implicit site names.
- **`reads`** produce values, each bound to one (site, position, model, input).
- **`writes`** are inert definitions — an address and a mechanism, no model.
  They do nothing until an intervened model lists them, which is what makes one
  write reusable across several.
- **`intervened_models`** is where a write comes into force. `original` is
  reserved for the un-intervened model and is never declared.
- **`metrics`** reduce a read against dataset columns.
- **`save`** is the complete manifest of what leaves the run.

`v_cf` is read in one model and consumed by a write in force in another: that is
the single channel for cross-model data flow, and the graph it induces is the
execution schedule.

### Document sections

| section | required | declares |
|---|---|---|
| `version` | ✓ | `"1"` |
| `description` | – | intent, free text |
| `model` | ✓ | the network as a name: `key`, `revision`, `dtype`, `quantization`, optional `attn_implementation` |
| `data` | ✓ | input rows: `base`, optional `counterfactual` — dataset ref + field |
| `positions` | – | named token-position specs |
| `sites` | ✓ | named activation addresses — the complete tap inventory |
| `featurizers` | – | named feature-space maps |
| `params` | – | free/constant tensors owned by no featurizer |
| `code` | – | user functions a `pytorch_fn` write names: locator + source hash, args, declared file/env inputs, row roles |
| `reads` | ✓ | value producers: (site, pos, model, input) [+ featurizer, dims] |
| `writes` | – | inert effect definitions: (site, pos, `do`) |
| `intervened_models` | –* | which writes are in force on which input (*required with `writes`) |
| `metrics` | – | closed reductions over read values |
| `train` | – | the fit: objective, params, optimizer, steps, batch, seed |
| `save` | ✓ | the output manifest — non-empty, last |

### Closed vocabularies

Anything outside them is a load error, not a fallback.

| vocabulary | values |
|---|---|
| `sites.component` | 56 names; the 54 the A3B exposes are tabulated in [§5](#5-hookpoints-on-qwen36-35b-a3b) (`mlp_activation` and `mlp_neuron_output` have no tensor on this architecture); eight retired `deltanet_*` spellings are aliases (§5) |
| `sites.stream` | `full_attention` · `linear_attention` — a per-layer fact on a hybrid tower, refused at load if the layer carries the other one |
| `sites.head` / `sites.expert` | sub-axis selectors, legal only where the component has that axis |
| `writes.do` | `swap` · `add_scaled` · `lerp` · `affine` · `gaussian` · `renormalize` · `clamp` · `pytorch_fn` (local-only; names a `code` declaration) |
| `metrics.kind` | `logit_diff` · `token_logit` · `cross_entropy` · `kl` · `class_probs` · `token_logits` · `top_k` · `match` · `decode` |
| `featurizers.kind` | `identity` · `subspace` · `pca` · `sae` · `standardize` · `gate` |
| `pos` forms | `-1` (sugar for `{"index": n}`) · `"all"` · `{"variable": v}` · `{"column": c}` · `{"span": [a, b]}` (half-open), modified by `scope` / `relative_to` / `generated` |
| save formats | `.json` (per-example tables) · `.safetensors` (dense numerics) |

Three rules that catch most authoring mistakes:

| rule | consequence |
|---|---|
| one global namespace over the named sections | every name unique; `base`, `counterfactual`, `counterfactual[j]`, `original` are reserved |
| at most one **absolute** write per (site, overlapping pos, model) | any number of additive writes; absolute applies first, then the summed deltas — so write sets are order-free |
| `{"sweep": [v, …]}` / `{"sweep": {"range": [a, b]}}` is the only axis | bare arrays are never axes; axis identity is name identity, so sweeping `sites.target.layers` moves the read, the write and the metric together |

Derived, never authored: feature widths, `num_forwards`, the point count, the
`requires` capability set, digests. If it can be computed from the document, the
document must not say it.

### Generating documents

A document that repeats one shape at every layer or every executed condition
is generated, not typed. A generator writes an ordinary document — nothing at
run time knows it was generated, and the output validates like anything else.
Three rules make a generator safe to apply: it refuses rather than renames on a
name collision, so applying it twice is an error; it takes the model's facts
(layer count, `layer_types`, expert count) from the registry entry, with
`--register-from-hf` as the same opt-in `validate` takes
([§3](#3-check-it-before-you-spend-a-gpu)) for a model that has no built-in
entry; and it derives every name deterministically from the coordinates it
expands over. Qwen3.6-35B-A3B has a built-in entry and validates offline. Three
shapes recur:

**A per-layer harvest.** The template is a one-layer, one-site, pure-read
document (the shape of `mean_harvest.json`); the output declares, per layer,
`L{n}A` (`block_mid`, the residual after the mixer) and `L{n}M`
(`block_output`, after the MLP), and at every kept position one read
`acts_L{n}{A|M}_{position}` with its save entry. A save-time `reduce` carries
over, so a mean-harvest template yields per-layer means. Refuse `writes` or
`intervened_models` in the template (a harvest is a pure read), a read through
a featurizer, and more than one site.

**Routing capture as declared reads.** A run saves nothing implicitly, so
routing is captured by adding reads. For every MoE layer of the model and
every `(model, input)` pair the document executes — `original` on each input a
read names, plus every intervened model on its input — add an `expert_idx` and
a `router_scores` read at each position, named
`routing_L{n}_{model}_{input}_{position}_{idx|scores}`, and their save entries.
Positions default to the ones the pair's own reads use. Refuse a model whose
registry entry declares no experts. The result validates like any document:

```bash
uv run causalab validate patch_routed.json --data-root data --artifacts-root .
```

**A joint DBM fit across layers** fits gates at every layer under one sparsity
penalty. The mask unit picks the sites: whole heads (`attention_premix` on
full-attention layers, `delta_premix` on Gated DeltaNet layers), channels within
a head output (the same premix sites with a coordinate gate), complete MLP
neurons (`expert_neuron_output` with `group: expert_neuron` plus
`shared_expert_activation`; `mlp_neuron_output` on dense layers), or the union
of the last two. Routed and shared experts gate `act(gate) * up`, before the
down projection; a routed gate has one parameter per expert and neuron, and the
routing table maps those parameters to the active slots. Head families need the
registry's `layer_types` to identify each layer's mixer. Independent gates at
explicit aligned token positions give each layer and position its own
parameters under one objective; `all` broadcasts one gate across positions, and
the readout position (the answer logits, typically the final token) is
independent of the intervention positions. The fit document crosses penalties
with seeds; IIA compares the output with the label column, `logit_diff`
subtracts the base-answer logit from the gold-label logit in the intervened
output, and `ce` trains against that same gold label.

The **apply** document loads each gate's saved parameters and evaluates its hard
mask (`theta > 0`) on a confirmation split. One document evaluates every
penalty × seed cell with all gates linked to one `axes.fit_cell` axis: each row
holds the fit's exact bundle selector in `entry`, and metric rows carry the
corresponding `axes.fit_cell` row index. An optional `rank` save writes one
record per unit per evaluated cell — keep it to small audits, because an
all-token neuron sweep can produce millions of records. Report viewers must use
the mask evaluated for each saved cell; separate fits can select different
components at the same sparsity. Both documents pass through the loader before
writing; apply validation checks bundle identities under `--artifacts-root`.
The training defaults are `configs/protocols/dbm.json`.

**Exporting a DBM result.** `causalab.analysis.export_dbm.export(manifest_path,
register_from_hf=False)` joins saved apply metrics to the frozen gate bundles
used for evaluation. Its manifest groups apply runs by experiment; paths
resolve from the manifest's folder:

```json
{
  "experiments": [{
    "id": "output-neurons",
    "title": "Output DBM over neurons",
    "evaluations": [{
      "document": "neurons_apply.json",
      "run_dir": "runs/neurons_apply",
      "data_root": "data",
      "artifacts_root": "."
    }]
  }]
}
```

Set `register_from_hf=True` to resolve an unregistered model from its HF
config. Export checks document, point and bundle digests. Each point contains
the frozen hard masks, selected count, metrics and provenance. Gate positions
are integer token indices or `"all"`; named scalar definitions resolve to those
values. Other position forms are refused. The fit identity depends on bundle
content and entry selectors, so it survives a moved run folder.

IIA uses all eligible pairs. Logit difference uses eligible pairs whose gold
and original answers differ. Each metric carries its sample count, while
`omitted_points` records masks without eligible evaluations. A curve combines
runs only when their canonical model, data, gate layout and resolved metric
reads agree. The model identity includes dtype and quantization. Both scores
must read the base input under exactly the exported gate swaps. Each swap
reads the aligned counterfactual site through that gate.

Report applications add captions and token examples, then embed the JSON in
their own templates.

## 3. Check it before you spend a GPU

Both verbs are pure — no weights, no network, no accelerator — so they cost a
second and catch every load error.

⚠️ **"Load error" is narrower than "would have run".** The pure verbs hold no
tokenizer, so nothing whose answer is a token count is knowable here: a
`{"variable": …}` or `{"all": true}` position that turns out ragged across rows
is refused when the batch is encoded ([V19] for a write), not by `validate`.
`--data` does check that every column and every prompt variable a document
names *exists* in the resolved tables, at every expanded point of a sweep —
that half used to be checked at coordinate 0 only, and for columns only, which
is how a 64-point scan validated `OK` with 32 points that could not run.

⚠️ They are also **registry-only**: they derive featurizer widths from the
static metadata in `causalab/protocol/registry.py` rather than fetching a
config, so a digest never depends on connectivity. The A3B is a built-in entry,
with its hybrid layer pattern, so the document above validates offline — and so
does the refusal that matters most on this tower: a full-attention component
at a DeltaNet layer.

```bash
uv run causalab validate patch.json --data-root data --artifacts-root . --data
# OK: patch.json — 1 point, digest …

uv run causalab validate patch.json --data-root data --artifacts-root . --data \
    --set sites.target.component=attention_premix
# refused: [V4] at sites.target.component site 'target': component
#          'attention_premix' exists only on a 'full_attention' mixer, but layer
#          20 of 'Qwen/Qwen3.6-35B-A3B' carries 'linear_attention' — …
#          Layers carrying 'full_attention': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
```

A model that is **not** built in refuses with `[V4] … is not in the protocol
model registry`. Two ways forward, and which you want depends on what you are
checking:

- **pre-flighting on the real model** — `--register-from-hf` resolves the key
  from its HF config first. Opt-in, because it is the one thing that makes a
  pure verb touch the network:

  ```bash
  uv run causalab validate patch.json --data-root data --artifacts-root . \
      --data --register-from-hf
  # OK: patch.json — 1 point, digest …
  ```

  ⚠️ Do **not** substitute a similar registered model for this. It produces a
  *false* refusal — `[V4] … layer 36 out of range for the 36-layer model
  'Qwen/Qwen3-4B-Instruct-2507'` on a perfectly valid 40-layer A3B document.

- **checking the document alone** — point the pure verbs at a registered key.
  The structure, the reference graph and the save manifest are
  model-independent; only widths and layer bounds are not, so read a
  width-or-bounds refusal as being about the stand-in:

  ```bash
  uv run causalab validate patch.json --data-root data --artifacts-root . --data \
      --set model.key=Qwen/Qwen3-4B-Instruct-2507
  # OK: patch.json — 1 point, digest cc2e2500fac13029…

  uv run causalab explain patch.json --data-root data --artifacts-root . \
      --set model.key=Qwen/Qwen3-4B-Instruct-2507
  # digest    cc2e2500fac130298f0513e6f836da2d16056660147fdbd03f1e37e65427e4cf
  # model     Qwen/Qwen3-4B-Instruct-2507@main fp32
  # points    1
  # requires  ['component:block_output', 'component:block_output:write',
  #            'component:lm_head', 'paired_forward']
  # forwards  2 per point
  #   original on counterfactual: v_cf
  #   patched on base: logits
  # save
  #   iia (model=patched, input=base) -> iia.json
  #   logit_diff (model=patched, input=base) -> logit_diff.json
  ```

- **running it** — just run. `run` touches the model anyway, so it resolves an
  unregistered key from its HF config and registers it before canonicalizing.

`explain`'s `points` and `forwards` are what to size a job against: a sweep of
40 layers is 40 points, and the run cost is roughly points × forwards.
`requires` is the capability set routing matches engines on — every component
the document names appears there, `:write` suffixed where a write targets it.

`dry-run` is both verbs' answers in one report, plus what neither prints: per
site, what the registry entry says exists — shape, width, head space, which
engines read it and which mechanisms may write it — the forward count the
campaign actually owes once shared forwards are interned, the shard count for a
`--shard-size`, and, as its last line, the `undecided` facts only the run
decides (token windows and widths, answer tokens, pair validity, controls), so
a refusal at encoding time is never mistaken for a green here. It exits `1` on
any refusal and `0` otherwise; `--engine auto` also asks, per installed engine,
what `check_engine` would refuse (a `capability_shortfall`, reported not
raised), and `--data` folds in `validate --data`'s pass:

```bash
uv run causalab dry-run patch.json --data-root data --artifacts-root . \
    --set model.key=Qwen/Qwen3-4B-Instruct-2507 --shard-size 4
# dry-run   patch.json
# digest    cc2e2500fac130298f0513e6f836da2d16056660147fdbd03f1e37e65427e4cf
# model     Qwen/Qwen3-4B-Instruct-2507@main fp32
#   36 layers, hidden 2560, 32 heads (8 kv) x 128, vocab 151936, family qwen3; declares no layer pattern
# points    1
# forwards  2 per point, 2 interned
# shards    1 of at most 4 points (1 point)
# requires  ['component:block_output', 'component:block_output:write',
#            'component:lm_head', 'paired_forward']
# sites
#   target: block_output layer 20: available
#     shape (batch, position, feature), width 2560, no head axis
#     reads nnterp, pytorch_hooks; writes add_scaled, affine, clamp, gaussian, lerp, pytorch_fn, renormalize, swap
#   lm_head: lm_head: available
#     …
# undecided (decided when the run encodes its inputs): engines, inventory, tokenization, pair_validity, controls
```

The same document with an unavailable site refuses before any of that, with
the reason code beside the rule — the acceptance case of the dry run, and it
holds on a machine with no accelerator and no model cached:

```bash
uv run causalab dry-run patch.json --data-root data --artifacts-root . \
    --set sites.target.component=routed_output --set model.key=Qwen/Qwen3-4B-Instruct-2507
# refused: [V4] at sites.target.component site 'target': component 'routed_output'
#          needs a sparse-MoE block (the entry declares no experts), which model
#          'Qwen/Qwen3-4B-Instruct-2507' does not have — there is no such tensor on this model
#   code V4 (references_resolve) at sites.target.component, reason component_unavailable
```

## 4. Run it

Smoke it on a tiny random model of the same architecture — four layers, hidden
8, hybrid DeltaNet/attention tower, sparse MoE in every layer:

```bash
uv run causalab run patch.json --data-root data --artifacts-root . \
    --out runs/patch \
    --set model.key=tiny-random/qwen3.5-moe \
    --set sites.target.layers=1 \
    --device cpu
# saved iia.json -> runs/patch/iia.json
# saved logit_diff.json -> runs/patch/logit_diff.json
# cells 2 / 2 eligible
```

The numbers are meaningless — random weights answer nothing. What this proves is
that the document loads, plans, executes and saves. `--set` is for exploration
only: anything that matters about an experiment belongs in the file, where it
enters the digest.

The last line is the **denominator** (spec §4.1): one cell per `save` entry per
point, and how many of them measured something. A cell can be legal and still
measure nothing — a site scoped to `expert: 7` when the router sent expert 7 no
token at the addressed positions — and such a cell is *unavailable*, not an
error: the run writes it with `status: "unavailable"`, a reason code from the
spec's §2.4 table (`empty_selector` here) and the fact in `detail`, and counts
it as excluded. A sweep that prints `cells 155 / 157 eligible; 2 excluded:
empty_selector ×2` says, in one line and with nothing kept beside the result,
that two cells were excluded measurements rather than null localizations. The
same numbers are `RunResult.cells` / `RunResult.denominator` from Python, and
each excluded cell is repeated under `unavailable` in its point's summary.

That run also **refuses**, and the refusal is the point: MCQA's answers are
single letters, and ` Z` and `Z` are *different* tokens that both exist.
`token_form="auto"` cannot know which one the model emits, so it says so
instead of picking one. Set the metric's `token_form` to `bare` or
`space_prefixed` and run it again.

Then the real thing, on an accelerator:

```bash
uv run causalab run patch.json --data-root data --artifacts-root . \
    --out runs/patch --device cuda --dtype bf16
```

| flag | why |
|---|---|
| `--device` | placement is execution, not a document fact |
| `--dtype` | shorthand for `--set model.dtype=…`; precision **is** a document fact, so it enters the digest. Refused on a **workflow** — its steps each declare their own realization |
| `--artifacts-root` | where a relative artifact `file_path` resolves. It merely *defaults* to `.`, so passing it is what keeps absolute machine paths out of a digest — pass it in every invocation |
| `--engine` | `auto` (default: every installed engine, reference first, routed by `choose_engine`), or name one to pin it — see [§6](#6-engines-and-routing) |
| `--points START:STOP` | execute one half-open slice of an expanded sweep; the seam to shard a campaign on |
| `--batch-rows N` | reference engine: run a forward group over more than `N` rows as several forwards of at most `N` rows each, captures concatenated in row order. Execution only — the numbers equal the single-forward run up to dtype rounding, digests and stamps are unaffected, and the run receipt records the bound as `execution.batch_rows` — see [§6](#6-engines-and-routing). Refused together with `--engine nnterp`, which runs one batch per group and would honour no bound |
| `--resume` | reuse completed outputs whose inputs and code hash are unchanged. A **workflow** flag: refused on a single intervention specification, which has no step boundaries to resume at — wrap it in a workflow step. The first `run` of a workflow also stamps its `pins` section, the digests of everything it touches, which every later load is held to (`causalab pin` re-stamps after a meant change; workflow spec §7) |
| `--register-from-hf` | resolve an unregistered `model.key` from its HF config instead of refusing `[V4]`; `run` always does this, the pure verbs only on request |


## 5. Hookpoints on Qwen3.6-35B-A3B

The tower is **40 layers on a repeating 3+1 schedule**
(`full_attention_interval: 4`): layers 3, 7, … 39 carry a gated full-attention
mixer, the other 30 carry a Gated DeltaNet (linear-attention) mixer. Both kinds
carry the **same** MLP — a sparse MoE of 256 experts routed top-8, plus a shared
expert that runs on every token.
[`qwen36-35b-a3b-architecture.html`](qwen36-35b-a3b-architecture.html) draws
that forward pass one box per tensor, lavender where a hookpoint sits.

| | |
|---|---|
| layers | 40 — 30 `linear_attention`, 10 `full_attention` |
| hidden size | 2048 |
| full attention | 16 query heads, 2 KV heads (GQA), `head_dim` 256, output-gated, partial RoPE (0.25) |
| Gated DeltaNet | 16 key heads, 32 value heads (GVA), `d_k` = `d_v` = 128, causal conv kernel 4, 64-token chunked kernel |
| MoE | 256 experts, top-8, `moe_intermediate_size` 512; shared expert 512 |

Which mixer a layer carries is checked twice against one component→stream
table: at **load**, from the `layer_types` the registry entry declares (`[V4]`,
so `validate` refuses offline — the A3B entry declares its pattern), and at
**run**, against the module the layer really carries (`[P4]`), for a model
whose entry declares none. A site that names the wrong mixer is refused
either way:

```json
{"component": "attention_probs", "layers": [3]}    // ✓ layer 3 is full attention
{"component": "attention_probs", "layers": [4]}    // ✗ [V4] a Gated DeltaNet block
                                                   //     computes no attention matrix
{"component": "block_output", "layers": [4], "stream": "linear_attention"}  // ✓ optional, checked
```

### The table

**Reading the columns.** *blocks* is which of the two block types the tensor
exists in (and how many such layers the tower has). *shape* is the component's
declared axes — `head·feature` means head-major and already flattened,
`batch·position` means the MoE block's flattened token axis. *tap* is the
mechanism the engine reaches it by, which is what the engine column follows
from. *write* is the policy: a refusal names the alternative rather than just
saying no — and the text in that cell is the text the refusal prints.

**This table is generated.** Every cell is read off the capability registry
(`causalab/protocol/registry.py`, one row per component — spec §2.4) and the
A3B entry's shapes, by `registry.render_component_tables()`;
`tests/protocol/test_vocabulary_census.py` holds the committed text to that
rendering row for row, so it cannot drift from what the engines and the
validator actually do. The same rows generate each engine's component set (§6
below), the write policy `validate` and the executor apply, and the
component→stream table the mixer check reads. The table sits between
`generated: begin component-table` / `end` marker lines and is regenerated,
never edited by hand.

⚠️ **The engines column is information, not something to author.** It names
engines the way `--engine` does — `pytorch_hooks`, `nnterp` — since those are
the values the flag takes and the names the engines answer to. But a document
never names an engine: it declares the components it addresses, `requires`
derives the capabilities from those, and `choose_engine` picks (§8). So read a
single engine in that column as "only this one serves that component today",
and check the routing rather than copying the name:

```bash
uv run causalab explain patch.json --data-root data --artifacts-root . \
    --engine auto
# ... engine    nnterp
```


<!-- generated: begin component-table -->

**Model boundary (no `layer`)**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `input_ids` | — (layer-less) | `(batch, position)` | module input | both | read-only — the model's token input is not an activation; change the row's text instead, or write 'embeddings' to edit the vector the ids look up |
| `embeddings` | — (layer-less) | `(batch, position, feature)` | module output | both | any mechanism |
| `ln_final` | — (layer-less) | `(batch, position, feature)` | module output | both | any mechanism |
| `lm_head` | — (layer-less) | `(batch, position, feature)` | module output | both | any mechanism |

**Residual stream and dense MLP — every layer**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `block_input` | every layer (40) | `(batch, position, feature)` | module input | both | any mechanism |
| `attention_input_norm` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `attention_output` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `block_mid` | every layer (40) | `(batch, position, feature)` | module input | both | any mechanism |
| `mlp_input_norm` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `mlp_input` | every layer (40) | `(batch, position, feature)` | module input | both | any mechanism |
| `mlp_output` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `mlp_activation` | none — no such tensor on this architecture | — | module output | both | any mechanism |
| `mlp_neuron_output` | none — no such tensor on this architecture | — | module input | both | any mechanism |
| `block_output` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |

**Full-attention mixer interior — the 10 `full_attention` layers**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `attention_query_pre_rope` | full-attn (10) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `attention_key_pre_rope` | full-attn (10) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `attention_value_states` | full-attn (10) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `attention_gate` | full-attn (10) | `(batch, position, head·fused·feature)` | module output | both | any mechanism |
| `attention_query` | full-attn (10) | `(batch, head, position, feature)` | attention-function slot | both | any mechanism |
| `attention_key` | full-attn (10) | `(batch, head, position[key], feature)` | attention-function slot | both | any mechanism |
| `attention_scores` | full-attn (10) | `(batch, head, position[query], key_position[key])` | attention-function slot | both | any mechanism |
| `attention_z` | full-attn (10) | `(batch, position, head, feature)` | attention-function slot | both | any mechanism |
| `attention_result` | full-attn (10) | `(batch, position, head·feature)` | derived from `attention_premix` | both | read-only — it is derived, not computed: the model never forms the per-head contribution at all — it forms their sum, by projecting the whole 'attention_premix' at once — so there is no tensor here for a write to change. Write 'attention_premix' instead, with the same 'head'; 'attention_result' is a linear function of it, so a write there moves this by exactly the projection of what you wrote |
| `attention_premix` | full-attn (10) | `(batch, position, head·feature)` | module input | both | any mechanism |
| `attention_probs` | full-attn (10) | `(batch, head, position[query], key_position[key])` | module output | both | `swap` only — its rows are a probability distribution and the value multiply immediately downstream assumes they sum to 1 — nothing renormalizes them after an edit. Write 'attention_scores' instead: it is the same tensor one step earlier, upstream of the model's own softmax, so every mechanism is legal there and the rows still sum to 1 by construction |

**Gated DeltaNet mixer interior — the 30 `linear_attention` layers**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `delta_qkv` | DeltaNet (30) | `(batch, position, feature)` | module output | both | any mechanism |
| `delta_gate` | DeltaNet (30) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `delta_conv` | DeltaNet (30) | `(batch, feature, position)` | delta-kernel boundary | both | any mechanism |
| `delta_query` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | both | any mechanism |
| `delta_key` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | both | any mechanism |
| `delta_value` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | both | any mechanism |
| `delta_beta` | DeltaNet (30) | `(batch, position, head)` | delta-kernel boundary | both | any mechanism |
| `delta_decay` | DeltaNet (30) | `(batch, position, head)` | delta-kernel boundary | both | any mechanism |
| `delta_kv_mem` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | read-only — a memory readout has no independent existence: it is (S_{t-1}·exp(g_t) · k̂_t) summed, recomputed from the state at every step, so there is no tensor a write could persist into. Write 'delta_state' to change what the memory holds, or 'delta_value' to change what is stored into it |
| `delta_state_update` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | read-only — its write lowers exactly onto a state edit through the reconstruction identity S_t = S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t, and that lowering is deferred — write 'delta_state' instead |
| `delta_state` | DeltaNet (30) | `(batch, position[steps], head, state, state)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_kernel_output` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | both | any mechanism |
| `deltanet_query` | DeltaNet (30) | `(batch, position, head, feature)` | `.source` line (fused forward) | `nnterp` | any mechanism |
| `deltanet_key` | DeltaNet (30) | `(batch, position, head, feature)` | `.source` line (fused forward) | `nnterp` | any mechanism |
| `deltanet_state` | DeltaNet (30) | `(batch, position[chunk], head, feature)` | `.source` line (fused forward) | `nnterp` | any mechanism |
| `delta_premix` | DeltaNet (30) | `(batch, position, head·feature)` | module input | both | any mechanism |

**Sparse MoE + shared expert — every layer**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `router_logits` | every layer (40) | `(batch·position, feature)` | module output | both | read-only — the MoE block discards the router's logits (it destructures them into '_') and routes on the scores and indices it computed from them, so a write here cannot reach anything — write 'router_scores' to reweight the chosen experts, or 'expert_idx' to change which experts fire |
| `router_scores` | every layer (40) | `(batch·position, topk)` | module output | both | any mechanism |
| `expert_idx` | every layer (40) | `(batch·position, topk)` | module output | both | `swap` only — the routing table carries integer expert ids, not features: a delta, a scale or a clamp over them yields ids chosen by arithmetic on labels, which route to arbitrary experts where they stay in range and fail at the gather where they do not. Swap in an index tensor read from elsewhere to change which experts fire, or write 'router_scores' to reweight the experts already chosen. Refusing rather than doing arithmetic on values that are labels |
| `expert_gate_proj` | every layer (40) | `(batch·position, topk·fused·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_up_proj` | every layer (40) | `(batch·position, topk·fused·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_activation` | every layer (40) | `(batch·position, topk·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_neuron_output` | every layer (40) | `(batch·position, topk·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_permutation` | every layer (40) | `(batch·position, topk)` | `.source` line (fused forward) | `nnterp` | read-only — it is the serving kernel's row bookkeeping (where each (token, slot) row sits in expert-sorted order), not routing: the kernel derives it from the routing table, and an edited copy would describe rows that were never sorted that way. Write 'expert_idx' to change which experts fire, or 'router_scores' to reweight them |
| `expert_output` | every layer (40) | `(batch·position, topk·feature)` | grouped-experts dispatch | both | any mechanism |
| `routed_output` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_gate_proj` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_up_proj` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_activation` | every layer (40) | `(batch·position, feature)` | module input | both | any mechanism |
| `shared_expert_output` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_gate` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |

<!-- generated: end component-table -->

### Dense neuron sites

`mlp_activation` reads the dense MLP's activated gate. `mlp_neuron_output`
reads the complete neuron output at the input to its down projection. A gated
MLP forms this value as `act(gate) * up`. GPT-2 has one activation branch, so
both sites expose its complete activated neuron output.

Qwen3.6-35B-A3B has a sparse MoE block at each layer. Its registry entry
has no dense inner width, so `validate` refuses both dense sites with
`component_unavailable`. Use `expert_neuron_output` for routed experts and
`shared_expert_activation` for shared experts. Both expose complete neuron
outputs before the down projection. `expert_activation` remains the activated
gate branch inside routed experts.

### The attention interior, per family

The four module-boundary components of the full-attention mixer
(`attention_query_pre_rope`, `attention_key_pre_rope`,
`attention_value_states`, `attention_gate`) are the one place the model
families disagree about *where* a component is: GPT-2 fuses q, k and v into
one `c_attn`, llama keeps three projections, qwen3.5-moe normalizes q and k
before RoPE and packs a gate beside q. Each component's registry row carries
that address per family (spec §2.4, `overrides`, keyed by the entry's
`family` — the HF `model_type`), so the **same logical site** reads and writes
on all three: on GPT-2 the queries are the first `H·d` columns of `c_attn`'s
output, and a swap there moves the logits exactly as a direct write into
those columns does.

**This table is generated** from the rows by `registry.render_family_table()`
and held to it by `tests/protocol/test_vocabulary_census.py` (block
`family-table`):

<!-- generated: begin family-table -->

| component | `gpt2` | `llama` | `qwen3_5_moe_text` |
|---|---|---|---|
| `attention_query_pre_rope` | `c_attn` output `(batch, position, fused·head·feature)`, split 0 of 3 | `q_proj` output `(batch, position, head·feature)` | `q_norm` output `(batch, position, head, feature)` |
| `attention_key_pre_rope` | `c_attn` output `(batch, position, fused·head·feature)`, split 1 of 3 | `k_proj` output `(batch, position, head·feature)` | `k_norm` output `(batch, position, head, feature)` |
| `attention_value_states` | `c_attn` output `(batch, position, fused·head·feature)`, split 2 of 3 | `v_proj` output `(batch, position, head·feature)` | `v_proj` output `(batch, position, head·feature)` |
| `attention_gate` | — (no such tensor: refused at load and at run) | — (no such tensor: refused at load and at run) | `q_proj` output `(batch, position, head·fused·feature)`, split 1 of 2 |

<!-- generated: end family-table -->

The `fused` axis is the one the component does *not* span: the executor
selects the row's split on the way out and scatters it back on the way in, so
a write to `attention_key_pre_rope` on GPT-2 leaves the q and v columns of
`c_attn` untouched. `head` slices inside the logical value, in the
component's own head space (KV heads for `k` and `v` under GQA).

A family with no column here is one the table has not met. Its entry decides
nothing about the mixer interior offline, and the run serves it by measurement
where that is unambiguous — a bare projection of the value's width, or a norm
after it — and refuses by name where it is not: a fused projection's block
order. Because `validate` reads the same rows, a document naming
`attention_gate` on a `gpt2` or `llama` entry is refused before a GPU is
spent on it (`[V4]`, reason `component_unavailable`), the same refusal the
run makes from the module tree.

### `delta_*` and `deltanet_*`: one name per tensor, three typed pairs

The two engines reach the DeltaNet interior by unrelated mechanisms — the
reference engine swaps the modeling file's kernel globals for the extent of
one mixer forward, the nnterp engine drills `.source` inside the fused
forward. 📐 Measured on the fixture and asserted at both test tiers, eight of
the eleven `delta_*` / `deltanet_*` pairs are the *same tensor*: same shape,
same timing, max abs diff 0.0. Those eight carry **one name each**
(`delta_qkv`, `delta_conv`, `delta_gate`, `delta_value`, `delta_beta`,
`delta_decay`, `delta_kernel_output`, `delta_premix`), served by both engines,
and their `deltanet_*` spellings are aliases: a document that authors
`deltanet_core_out` parses and digests as `delta_kernel_output`
(`schema.DEPRECATED_COMPONENTS`, spec §2.4). The three pairs whose tensors
differ in shape or timing stay two names, with the relation declared as a row
(`registry.BACKEND_PAIRS`). The `deltanet_*` face is the nnterp engine's
alone; the `delta_*` one is the reference engine's, and the nnterp engine's
too where it reads the same tensor (the tiled q/k are the kernel call's
arguments):

| `delta_*` | `deltanet_*` (`nnterp` only) | relation |
|---|---|---|
| `delta_query` `delta_key` (both engines) | `deltanet_query` `deltanet_key` | `gva_tile` — `delta_*` is **post** GVA `repeat_interleave` (32 value heads); `deltanet_*` is **pre** (16 key heads). Exact after tiling. |
| `delta_state` (`pytorch_hooks` only) | `deltanet_state` | `chunk_boundary` — per **step** vs per 64-token **chunk**; the chunk's state is the step-state at the chunk's last position |

An alias across one of these would *rebind* rather than redirect — hand one
face's tensor to the other face's math — and `registry.alias_would_rebind`
refuses it by the declared relation (a census holds the alias table to it). So
name the tensor, not the engine: the one-name components route to
whichever engine is listed first; if you need per-step state (or
`delta_kv_mem` / `delta_state_update`, which the chunked prefill kernel
never materializes), that is the reference engine, and the pre-tiling q/k are
`deltanet_query` / `deltanet_key` on the nnterp engine.

### Reading state and attention is expensive

Two components have a position axis that is not the token axis, and both cost
real memory on the A3B:

- `delta_state` is one `d_k × d_v` matrix per head per step — 30 layers ×
  seq × 32 × 128 × 128 floats if you ask for every layer at `pos: "all"`.
  Address positions in the read, not afterwards: the gather runs before
  anything is kept.
- `attention_scores` / `attention_probs` have **two** position axes (query and
  key), so an integer `pos` is ambiguous and refused. Read them whole.

## 6. Engines and routing

Two engines implement the same protocol. A document does not name one — it
declares what it needs, and `choose_engine` takes the first engine in the list
whose capabilities cover it. **`--engine auto` is the default**: every installed
engine with the reference first. List order is preference, so anything the
reference serves behaves exactly as pinning `pytorch_hooks` would — while a
document only the nnterp engine can serve runs instead of refusing by name.
Pinning one engine is then a deliberate act (a parity check, or reproducing a
run that named one), not the thing you fall into by not passing a flag.

The table's `capabilities` and `components` rows are generated — from the
engine classes' `capabilities` sets and the registry's rows (block
`engine-summary`, carried here and in `docs/CODEBASE.md` §3); the other rows
are prose.

<!-- generated: begin engine-summary -->

| | `pytorch_hooks` | `nnterp` |
|---|---|---|
| how | `register_forward_hook` / pre-hook, plus global swaps for the delta kernel and the experts dispatch | one frozen program per forward group, run as one nnsight trace over nnterp's standardized tree — envoys for module boundaries, `.source` for fused-forward interiors — in this process or on NDIF |
| capabilities | `grad` `paired_forward` `full_logits` `writable_attention_probs` `pytorch_fn_local` `generate` `quantized_weights` | `grad` `paired_forward` `full_logits` `writable_attention_probs` `pytorch_fn_local` `generate` |
| components | 52 of 56 — every component but `deltanet_query`, `deltanet_key`, `deltanet_state` and `expert_permutation` | 53 of 56 — every component but `delta_kv_mem`, `delta_state_update` and `delta_state` |
| serves alone | the per-token DeltaNet faces `delta_kv_mem` / `delta_state_update` / `delta_state` (the last a typed backend pair, §5), quantized weights | the fused-forward faces `deltanet_query` / `deltanet_key` / `deltanet_state` and `expert_permutation`, remote execution on NDIF |
| install | always | `uv sync` (dev group) or the `nnterp` extra |

<!-- generated: end engine-summary -->

Both engines run **one device per run** (no `device_map` sharding). The
reference engine runs a forward group as one batch unless `--batch-rows N`
bounds it to row windows of at most `N` (§8, execution scale — the numbers are
equal up to dtype rounding and digests are unaffected; the run receipt records
the bound as `execution.batch_rows`); the nnterp engine always runs one batch
per group. Both engines declare `grad` and fit a `train` document on the same
shared loop (featurizer slots, fp32 losses, evals on epoch boundaries), so
under `auto` a fit runs on the reference engine, listed first, unless it
addresses a component only the nnterp engine serves.

The nnterp engine plans each forward group into a frozen program and runs it
as one nnsight trace over nnterp's standardized tree — the same block list,
embedding, final norm and head under one set of names for every architecture
nnterp standardizes; a tree no registered family detects still serves the
block-shaped taps, and an interior its `.source` address table has no row for
is refused by name. A continuation read runs the group as one
`model.generate` trace. The same program runs **remotely on NDIF**
(`NnterpEngine(remote=True)`, a Python-API option): a whole point is one
session, operands flowing between its traces on the server, against a
weight-free client bundle. Remote mode needs a trusted, in-process NDIF
deployment with the same `causalab` installed server-side, and a remote
engine neither declares `grad` nor fits a `train` document — a remote
forward returns detached saves.

The two engines' answers are asserted to agree over the whole shared vocabulary,
read and written, at both test tiers —
`tests/neural/engines/nnterp_engine/test_parity_a3b_sweep.py` on the tiny
fixture and `tests/golden/test_a3b_engine_parity.py` on the real checkpoint.

## 7. Running at scale

Scale is not document vocabulary: a document never names a device, a host or a
scheduler. Sharding is `--points`, and job dispatch is site tooling.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mcqa-patch
#SBATCH --gres=gpu:2          # ~70 GB of bf16 weights + KV/state headroom
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
set -euo pipefail

uv run causalab run patch.json \
    --data-root data --artifacts-root . --out "runs/patch" \
    --device cuda --dtype bf16
```

Shard a sweep by point range — `explain` tells you how many points there are,
and `dry-run --shard-size N` how many shards of at most `N` that is
(`ceil(points / N)`), so the array bound below is read off, not computed by
hand:

```bash
#SBATCH --array=0-9           # dry-run --shard-size 4 says 40 points -> 10 shards
START=$(( SLURM_ARRAY_TASK_ID * 4 ))
uv run causalab run scan.json \
    --data-root data --artifacts-root . \
    --out "runs/scan/shard_${SLURM_ARRAY_TASK_ID}" \
    --points "${START}:$(( START + 4 ))" \
    --device cuda --dtype bf16
```

Each point's digest is the provenance unit, so shards are independent and their
outputs merge by coordinate. Each shard prints its own `cells` line; the
campaign's denominator is the sum.

**If a generator writes your documents, have it skip an empty axis rather than
emit one.** `{"sweep": []}` is a load error — a campaign of zero points is
almost always a bug in whatever produced the list, and expanding it to "one
point, unswept" would silently change the experiment. The refusal names the
axis:

```
refused: [V14] at sites.target.layers a sweep axis must have at least one value
```

That is legible on its own; the reason it is worth planning for is the shape of
a job loop. Under `set -euo pipefail` one document that refuses at load takes
the whole loop with it, so a generator that emits `{"sweep": []}` for a filter
that matched nothing loses every *later* document in the batch too. Either drop
the field when the list is empty, or let the generator refuse where it can say
which filter came back empty.

## 8. Chaining documents: workflows

A workflow document chains protocol steps with `script` steps between them —
select the best layer from a scan, fit a PCA, plot a curve — and the runner
resolves the dependency graph. See
[`workflow_protocol.md`](workflow_protocol.md).

The shipped one, `causalab/configs/workflows/weekdays_8b.json`: scan 64 points
for the layer that carries the variable, select it, fit a DAS rotation over a
subspace sweep, select the best `k`, apply it — with two figures along the way.

```bash
uv run causalab explain causalab/configs/workflows/weekdays_8b.json --artifacts-root .
# schedule  5 levels
#   level 0: locate
#   level 1: best, scan_heatmap
#   level 2: fit
#   level 3: best_fit, iia_by_k
#   level 4: apply
#   locate: intervention_protocol ../protocols/weekdays_locate_scan.json — 64 point(s), campaign digest 0f80f33a03bb8356…
#   best: script causalab.workflow.scripts.select -> values.json
#   fit: intervention_protocol ../protocols/weekdays_das_sweep.json — 9 point(s), authored digest 17667939591e92a6…
#   best_fit: script causalab.workflow.scripts.select -> values.json
#   apply: intervention_protocol ../protocols/weekdays_das_apply.json — 1 point(s), authored digest 42b93035b7320066…
#   scan_heatmap: script causalab.io.plots.workflow_figures -> scan_iia.json, scan_iia.png
#   iia_by_k: script causalab.io.plots.workflow_figures -> iia_by_k.json, iia_by_k.png

uv run causalab run causalab/configs/workflows/weekdays_8b.json --artifacts-root . \
    --out runs/weekdays --device cuda
```

⚠️ No `--dtype` on that command, and that is not an omission: a workflow's
steps each declare their own realization, so `--dtype` is **refused** on one
(`refused: --dtype sets model.dtype on one intervention specification; a workflow's steps
each declare their own realization`). Precision for a workflow goes in the
step's document, or in that step's own `set` block — and the earlier version of
this example printed the refused command, which cost one GPU job to find out.

`explain` on a workflow is the same pre-flight as on a document, one level up:
the schedule is derived from the steps' references, so a level is what can run
in parallel, and each protocol step reports its own point count — 64 + 9 + 1
forward groups is what to size the job against.

`--resume` reuses a step whose inputs *and* script content hash are unchanged;
editing a script busts its reuse, which is why the hash is in the digest.

## 9. Where to look next

| you want | read |
|---|---|
| a worked experiment, end to end | [`../demos/`](../demos/) — one markdown demo per research question |
| a **fit** and the **apply** that scores it honestly | [`04_subspace`](../demos/onboarding_tutorial/04_subspace.md) (a trained rotation) · [`06_components`](../demos/onboarding_tutorial/06_components.md) (a trained mask) |
| the demo format | [`demos.md`](demos.md) |
| the normative document spec | [`intervention_protocol.md`](intervention_protocol.md) |
| chaining documents | [`workflow_protocol.md`](workflow_protocol.md) |
| the module map and layering rules | [`CODEBASE.md`](CODEBASE.md) |
| test tiers and pinned-artifact discipline | [`TESTS.md`](TESTS.md) |
| worked documents | `causalab/configs/protocols/*.json`, `causalab/configs/workflows/` |
| a picture of the hookpoints above | [`qwen36-35b-a3b-architecture.html`](qwen36-35b-a3b-architecture.html) |
