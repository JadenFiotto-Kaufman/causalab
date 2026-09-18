# Intervention Protocol — specification, `protocol_version` 3

The **Intervention Protocol** is the format this document specifies. It defines
causal intervention experiments on neural networks, with some useful properties.
The five nouns used throughout — *research pipeline*, *runtime implementation*,
*intervention specification*, *compiled intervention*, *run receipt* — are
defined once in §11.1 and used only in that sense.

- **Serializable, shareable, reproducible**: an **intervention specification** is
  one JSON document that fully describes an experiment. It can be diffed, shared
  and re-run, is self-contained, and enables exact reproduction. What is *hashed*
  is the **compiled intervention** it resolves to, which is why two
  specifications differing only in section order share a digest (§7).
- **Agnostic to neural network interfaces**: a specification defines the
  experiment, not the **runtime implementation**. It is passed to a compiler
  which resolves it to a **compiled intervention**, and an engine (pytorch-native
  hooks, NNsight/NNterp, SGLang, Megatron, …) executes that. The specification
  says *what*; the engine's planner derives *how* — forward count, fusion,
  batching, sweep parallelization.

## 1. Document layout

One file is one experiment, and it has **four groups**. The header says what
the file is; `model` and `data` say what the experiment ran on — the network
and the rows, which is exactly the part that does not transfer to another
network or another task; `method` is the experiment itself — what is read,
what is written into whom, how it is scored — and does transfer. All four are
required, in this order — **recommended, not enforced**. Order carries no
meaning: the canonical form emits the groups, and the method's sections, in
the order below however they were authored (§7), so a document written in
another order — one that has been through `json.dumps(..., sort_keys=True)`,
say, or a YAML round-trip — has the same digest, the same plan and the same
run. An unconventional order parses, with a warning naming this order (§5
rule 2).

| # | key | required | content |
|---|---|---|---|
| 1 | `header` | ✓ | what this file is — the three fields below |
| 2 | `model` | ✓ | the neural network ℒ, and how it is realized numerically (§2.1) |
| 3 | `data` | ✓ | input rows: `base` (+ `counterfactual`) (§2.2) |
| 4 | `method` | ✓ | the experiment: the twelve sections below (§2.2.1, §2.3–§2.12) |
| 5 | `axes` | – | **sugar**: named axes — correlated row tuples and dependent axes — declared once, referenced at the fields they move by `{"axis": …}` wrappers and lowered to sweep wrappers before the shape gate (sec. 3.2); in the canonical form when authored |

The header:

| # | key | required | content |
|---|---|---|---|
| 1 | `protocol_version` | ✓ | `"3"` — a string, compared and never ordered |
| 2 | `title` | – | free text, one line: what to call this experiment |
| 3 | `description` | – | free text, the file's intent (JSON has no comments) |

`title` and `description` are **authoring metadata**: they say what a file is
*for*, never what the experiment *is*, and the canonical form drops them (§7) —
renaming a document or rewording its intent moves no digest. `protocol_version`
is content, and stays.

The method:

| # | key | required | content |
|---|---|---|---|
| 1 | `segments` | – | the rows' named sequence segments and their frame (sec. 2.2.1) |
| 2 | `positions` | – | named token-position specs |
| 3 | `sites` | ✓ | named activation addresses — the complete tap inventory |
| 4 | `featurizers` | – | named feature-space maps |
| 5 | `params` | – | free/constant tensors owned by no featurizer |
| 6 | `code` | – | user functions a `pytorch_fn` write names, declared by content (sec. 2.8.1) |
| 7 | `reads` | ✓ | value producers |
| 8 | `writes` | – | effect definitions (inert until listed) |
| 9 | `intervened_models` | –* | which writes are in force on which input (*required if `writes` present) |
| 10 | `metrics` | – | closed reductions over read values |
| 11 | `train` | – | the fit, declared |
| 12 | `save` | ✓ | the complete output manifest — non-empty, last |

One more key may sit under `method` and is **not a section**: it never reaches
the parser, the canonical form or a digest, because the compiler lowers it to
the sections above before the shape gate (sec. 3.2, sec. 9.1):

| key | required | content |
|---|---|---|
| `path_patching` | – | **sugar**: a declarative path — sender, ordered receivers, one position, a restoration policy — lowered to `sites` / `reads` / `writes` / `intervened_models` before the shape gate (sec. 3.2); never in the canonical form, and not addressable by `--set` |

The smallest complete document — a harvest: one site, one read, one output:

```json
{
  "header": {"protocol_version": "3", "title": "Residual harvest at layer 18"},
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main", "dtype": "bf16"},
  "data": {
    "base": {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "input"}
  },
  "method": {
    "sites": {
      "target": {"component": "block_output", "layers": [18]}
    },
    "reads": {"acts": {"site": "target", "pos": -1, "model": "original", "input": "base"}},
    "save": [
      {
        "value": "acts",
        "model": "original",
        "input": "base",
        "file_path": "acts.safetensors"
      }
    ]
  }
}
```

- **One global namespace**: every name in method sections 2–10 must be unique
  across all of them; reserved names: `base`, `counterfactual`,
  `counterfactual[j]`, `original`. Segment names (sec. 2.2.1) are anchors, like
  prompt variables and columns, and live outside this namespace.
- All cross-references must resolve; references are by name, never inline
  duplication.
- **Artifact-valued fields**: anywhere a scalar or position is expected,
  `{"artifact": "<path>", "key": "<field>"}` reads one value from a prior
  run's artifact at load. Missing artifact = load error.
- **Paths name sections, never groups.** A section name is unique across the
  four groups, so every dotted path in the protocol — a `--set` override, a
  workflow step's `set`, a sweep axis id, an `emit` or a plot axis, the path in
  a refusal — starts at the section: `sites.target.layers`, `model.dtype`,
  `train.seed`. The group is implied (`method.sites.target.layers` is refused,
  with the spelling to use). The header's fields are addressed as
  `header.title`; nothing sweeps them.
- **One document, two digests.** A compile reports the document digest (the
  campaign) and one digest per point (the provenance units, §7), and nothing
  else: those are the identities `--resume` compares and a tensor is stamped
  with. There is no separate hash of the `method` group — "the same experiment
  on another network" is a diff of `model` (and, where addresses differ,
  `sites`), which the file shows and no digest needs to say.
- **Two version fields, two specifications.** `header.protocol_version` is
  this specification's; a workflow document (the workflow spec) carries its
  own top-level `version`, because the two formats change on their own
  schedules. A workflow handed to this loader is refused as a workflow, not
  as a bad version.
- **`causalab migrate <file>...`** rewrites a `protocol_version` 1 document
  (top-level `version`, every section at the top level) and a `protocol_version`
  2 document (a site's depth spelled as the scalar `layer`, sec. 2.4) into this
  shape in place — and the fenced JSON examples in a markdown file with it. The
  loader refuses a v1 or v2 document by name and points at the verb. Nothing inside a
  section changed between the two versions.

## 2. Section reference

### 2.1 `model`

| field | meaning |
|---|---|
| `model.key` | model name (HF key or registry name) — the network as a *name* |
| `model.revision` | checkpoint revision |
| `model.dtype` | the compute dtype the weights are realized in: `fp32` (default) \| `bf16` \| `fp16` |
| `model.quantization` | optional — load-time weight quantization (below) |
| `model.attn_implementation` | optional — `eager` \| `sdpa` \| `flash_attention_2`; the full-attention backend both engines load |

- **Precision is part of the experiment, not of the run.** The same specification at
  `bf16` and at `nf4` produces different numbers, so `dtype` and `quantization`
  are document vocabulary and enter the digest. An authored file may stay
  silent — the canonical form materializes `dtype` (§7), so no *record* is ever
  silent about the precision its numbers came out of. The CLI's `--dtype` is
  shorthand for `--set model.dtype=…` (§9): it changes the document, and the
  digest changes with it.
- **An explicit attention backend is also part of the experiment.**
  `model.attn_implementation` supports sweep and bind wrappers and enters
  canonical forms, campaign digests, forward identities, and artifact metadata
  (`model_attn_implementation`). Workflow steps may set it through
  `set: {"model.attn_implementation": "flash_attention_2"}`. Omission keeps
  the historical engine default (eager for `pytorch_hooks`, the checkpoint's
  default for `nnterp`)
  and adds no field to the canonical form. The chosen backend is the normal
  forward implementation; attention-interior operations can temporarily use
  eager and restore it afterward. See [attention backends](attention_backends.md).

```json
"model": {
  "key": "meta-llama/Llama-3.1-8B", "revision": "main", "dtype": "bf16",
  "quantization": {"scheme": "nf4", "method": "bitsandbytes", "compute_dtype": "bf16", "double_quant": true}
}
```

| quantization field | meaning |
|---|---|
| `scheme` | ✓ — `int8` (LLM.int8() mixed-precision decomposition) \| `nf4` \| `fp4` (the two 4-bit datatypes) |
| `method` | the quantizer: `bitsandbytes` (default, the only v1 entry) |
| `compute_dtype` | dtype the dequantized matmuls run in; defaults to `model.dtype` |
| `double_quant` | 4-bit only — quantize the quantization constants |
| `int8_threshold` | `int8` only — the outlier threshold (default `6.0`) |

- There is no bare `int4`: 4-bit is `nf4` or `fp4`, and a field whose point is
  to name one realization may not be ambiguous about which.
- Weights quantized **ahead of time** (GPTQ, AWQ) are a property of the
  checkpoint, so `model.key`/`revision` already name them; `quantization`
  describes quantization applied *at load* to an unquantized checkpoint.
- A document with `quantization` requires the `quantized_weights` capability
  (§8), so an engine that cannot realize it refuses instead of quietly running
  something else.

### 2.2 `data`

```json
"data": {
  "base":   {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "input"},
  "counterfactual": {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "counterfactual_inputs[0]"}
}
```

- `dataset`: a **ref — a local path under the data root, no digest**, optionally
  followed by `#<split>`. The parser resolves it at load and stamps the content
  digest into the canonical form (sec. 7). A resolver provides three things for
  a ref: its content digest, its columns, and its rows.
- **A dataset is one table, and the split is a column of it.** Every row carries
  a `split` column naming which split it belongs to, and a ref selects one with
  a URL-style fragment: `natural_domains_arithmetic/data/weekdays#train` is the rows of `natural_domains_arithmetic/data/weekdays.json`
  whose `split` is `"train"`. A table that is one undivided pool says so with a
  uniform value (`"all"`).

  This is a statement about where a guarantee lives. Two *files* called `train`
  and `test` assert their relationship in their names and nowhere a reader can
  reach; two *splits of one table* cannot share a row, because a row declares
  exactly one split, and whether they share an *endpoint* is a question anyone
  can answer from the bytes. The names stop being a promise and start being a
  fact. The same question is asked *across* the tables one fit names: when its
  training rows and its `train.eval.split` are two different refs, they must
  share no endpoint (sec. 5, rule 22's fourth refusal); one ref named for both
  roles is the visible train-equals-test ablation, and passes.

  The digest a ref stamps is therefore over the **selected rows**, not the file:
  two splits of one table carry two distinct digests, so run identity survives,
  and a run's digest depends only on the rows it consumed, so adding a split to
  a table does not invalidate a run over the splits already in it.
- Tables are **serialized ahead of the load, never generated during it**: a ref
  resolves by reading bytes, so no digest depends on importing task code, a
  tokenizer, or the network. Task-generated tables are built by
  `causalab.tasks.serialize` (`scripts/build_task_dataset.py` for one pool; a
  split table, whose splits are group-disjoint by construction and addressed as
  `<ref>#<split>`, the same way) and write the table alone; a Hub-hosted dataset enters
  the same way, by being materialized first. **Nothing sits beside a table.**
  There is no manifest, recipe or provenance sidecar: the parameters a table
  was built from are the builder's command line, which the task's README or
  the workflow's description records, and nothing at load or run time could
  rely on a claim beside the table anyway — that is exactly why the split
  lives in the table and not beside it.

  A table is exactly the bytes a document names, and it is held to nothing
  else: no code-identity block, no tokenizer stamp, no pre-forward comparison
  — an intervention specification run on its own consumes the table's bytes,
  whose content digest is already in its canonical form (sec. 7), and that is
  the whole of the table's identity to the run. The pin that says *this table,
  at these bytes, is what the experiment was validated against* lives in
  exactly one place: the **workflow** that consumes the table, in its `pins`
  section (workflow spec §7) — beside the pins of every document, script and
  code module the workflow touches, stamped on the first run and held to on
  every later load. A pin has no home outside a workflow, because `--resume`
  and the reproducibility it serves are workflow concepts (sec. 9).
- Roles are the keys: `base` (required) and `counterfactual` (optional). `counterfactual` is
  singular; if its value is an array, references index it as `counterfactual[j]`.
  Rows are paired: one base row + its counterfactual row(s) form one example.
- **`base` is the schema of a paired row**, and the only role column references
  are checked against. A counterfactual role carries **at most** the columns
  base carries, and — when it names a *different* dataset — exactly them
  (rule 20). This is not a preference: metric columns are served from base rows
  and nothing else, so a column only a counterfactual carries is a column no
  metric can read, and validating against the union of the roles accepted a
  superset of what a run can serve. The equality half closes the mirror case: a
  `column` position resolves against the role of the read it positions, so a
  base-only column on a counterfactual read would fail at run time instead of
  at load. Naming **one table for both sides** — every document shipped here,
  and the shape sec. 3 recommends — satisfies all of this for free.
- `field` selects the column; `[j]` indexes list-valued columns.
- **`example_id` is the row's label** (`protocol/examples.py`). A table may
  carry the column — the author's name for the row — and when it does, every
  row carries one, non-empty and unique within the table; `validate --data`
  refuses the rest under rule 4, and a run refuses it again before the first
  forward. A table without the column labels its rows by zero-based index,
  as strings. The label a per-example output carries is the **base** row's:
  rows are paired by index and base is never permuted, so the base label names
  the pair. No numerical value, output path or execution enters a label, and a
  rerun, a sharded run or a shared forward writes the same labels; editing the
  table's rows or order changes its content digest, and so the point digest,
  before it changes any label.
- **A counterfactual role may declare `shuffle: {seed: <int>}`** — the one data
  verb. Before rows are paired, that role's rows are permuted by
  `random.Random(seed).shuffle` over their indices (`neural/shared/services.py`,
  `shuffle_order`; stdlib, torch-free, a pure function of the seed and the row
  count): a seeded permutation of the role's row order, so an interchange over
  the shuffled role is the **label control** the workflow spec's
  `shuffled_source` kind declares. A permutation may leave **fixed points** —
  seed 0 over 4 rows gives `[2, 0, 1, 3]`, and row 3 still meets its own base
  row; only 9 of the 24 orders of 4 rows have none — and nothing here excludes
  them: a certifier of that kind must account for them. The seed is the
  permutation's only input; two seeds are two pairings and two digests.
  `shuffle` is refused on `base` (rule 1's parse
  gate, naming `data.base.shuffle`: the base role is the population and is
  never permuted), takes exactly the key `seed`, an integer that is not a
  boolean, and is not sweepable — one document is one pairing. It enters the
  canonical form **only when authored** (sec. 7), so no unshuffled document's
  canonical bytes or digest move; the row-count equality across roles is
  checked on the permuted list, whose length is unchanged.
- **A counterfactual role may declare `draw: {"kind": "uniform", "eval"?: j}`**
  — the second data verb, a **counterfactual set**. Its `field` then names a
  list-valued column **bare** (`"counterfactual_inputs"`, not
  `"counterfactual_inputs[0]"`; the indexed form is refused beside `draw`,
  naming `field`), and the role's members are the column's entries, row by row
  (rows may hold different counts; every row holds at least one). A **fit**
  draws one member per row **at every epoch** from its own seeded generator (the
  document seed — `train.seed` — hashed with the word `draw`, so its stream is
  distinct from the batch order's and the mask samples', which share the raw
  seed; two points at one seed take the same members, so a sweep of anything
  else — the objective, the optimizer, the step budget, which `cohort_key`
  leaves out — compares them on one draw sequence and not on two), and each row
  is visited once per epoch, so that is one draw per update the row takes part
  in — a counterfactual sampled per example per step. Every member of
  every row is encoded once into one frame, so a minibatch under any draw is a
  selection of it (the same padded width, and a cohort's members encode the same
  texts, so they still concatenate — by construction); the fit's inner store is
  not consulted by a minibatch that holds a drawn role — a source forward over
  it is constant for no two epochs, and the store is per executor, so the base
  role's source forwards are recomputed with it (narrowing the bypass to the
  drawn role is a follow-up). Inside the fit the drawn rows hold the drawn
  member at **every** index of the list column and of its per-member siblings —
  the two named ones, `<column>_variables` (the prompt-variable table) and
  `<column>_encoding` (refused, below); any other `<column>_…` column is the
  author's own and is left alone — so `<column>[eval]` and `<column>[0]` alike
  read it; a sibling of another length is refused at prepare (P2; the
  fixed-member path's fallback to the plain column has no one member to fall
  back to under a draw). Prepared token sequences (`<column>_encoding`, below)
  do not combine with `draw`: this engine re-encodes a drawn role from text, and
  the fit would train on other tokens than the point's reads — refused at
  prepare (P4). **Every other forward** — the point's own reads and metrics
  after the fit, `train.eval`, an apply document, either engine — reads the
  fixed member `eval` (`0` unless authored): a fixed member is one
  `counterfactual_inputs[j]`, spelled once. `fit_diagnostics.json` records,
  under `draws`, per drawn role the kind, `eval` and `members` — one list per
  epoch of the member each row took (under `train.steps.updates` a partial final
  epoch lists a member for every row, including rows whose minibatch never ran;
  `{"epochs": n}` is exact). `draw` is refused on `base` (the population has one
  input per row), takes exactly `kind` (closed: `uniform`) and `eval` (a
  non-negative integer), neither sweepable, and enters the canonical form **only
  when authored** (sec. 7, beside `shuffle`). In this engine a drawn role is
  plain text: it does not combine with `segments` (refused at prepare, P4), and
  a run asked to capture graphs runs a drawn *fit* eager (`unsupported_reason`,
  as for `control`) while an apply document, which reads the fixed member at one
  width, keeps capture: the shapes are epoch-invariant, but this engine rebuilds
  the minibatch executors each epoch, which a captured graph cannot follow. The
  members of a row are alternative **texts** for that row: every other column
  stays per-row, a metric's `target` included (a metric reads its answer columns
  off the row, not off the member), so a drawn set must be **label-preserving**
  — members that imply different answers would train against one target with no
  error. Each drawn role draws on its own: two drawn roles (two elements of a
  `counterfactual` list) take independent members per row per epoch, and their
  member counts need not agree — right for unrelated columns, and not a coupled
  draw over one shared set. Not here: `draw: all` (the mean over members at eval
  — a fixed member or several documents cover it today), a per-member answer
  column (a third per-member sibling — the answer is the row's, and a member
  that changes it is not label-preserving) and a draw shared across roles (one
  index per row per epoch read by two roles — two documents or one role cover
  it).
- **Anything per-row or task-semantic is a column**, computed when the table is
  built — answer forms for `match` (sec. 2.10), values that place a position per
  row (sec. 2.3). Documents reference columns; they never compute.
- **A table built from a task may carry its scoring identity**: two constant
  columns, `scoring_digest` (the content digest of the task's `ScoringSpec`,
  `causalab/causal/scoring.py` — its definition of correct, once) and
  `string_mode` (`exact` | `prefix`, whether a generated string must equal a
  declared form or merely start with one). They are columns and not manifest
  keys for the reason the split is: the content digest above covers them for
  free, so a table rebuilt under a changed definition of correct is a different
  dataset, and a claim in the sidecar could not be relied on at load. When the
  base table carries them, `validate --data` and the executor — before the first
  forward — hold every `match` metric's `mode` to the table's `string_mode`
  under sec. 2.10's translation table: a `prefix` table under `mode: exact` is
  refused (rule 4, naming both modes and the derivation). A table without the
  columns is **unrecorded**: nothing is compared, it loads and runs as it always
  did, and the run receipt's `scoring` block says so — `{<ref>: {"digest":
  …|null, "string_mode": …|null, "result": "ok" | "unrecorded"}}`, beside the
  `execution` block. Nothing here enters the canonical form (sec. 7): the
  identity lives in the table bytes and the receipt, so no document digest moves.
- **A row may declare which spans of its pair move together**: an optional
  `edit_groups` column (`causalab/causal/pairs.py`) — a list of groups, each
  `{"name": …, "atomic": true | false, "spans": {"base": [[start, end], …],
  "counterfactual": [[start, end], …]}}`, character spans into the row's `input`
  and `counterfactual_inputs[0]` in the offset convention `variable` positions
  resolve (sec. 2.3), one constituent per span pair. `atomic: true` says the
  constituents are one coordinated edit — a relation word and the total it
  changes, the two entries of a swapped mapping — validated as one; `false`
  declares the spans and asks for nothing. The serializer writes the column
  only when the example declares it, so no shipped or fixture table changes,
  and a row without it is unrecorded: nothing is held to it. What the column
  claims is checked in two halves, both under rule 27 (sec. 5): at `validate
  --data`, the shape — spans inside the texts, the same number of constituents
  on both sides, an `atomic` group with two or more; before the first forward,
  that no intervened model addresses a constituent of an `atomic` group
  without its siblings — through its writes' positions on its input, or the
  positions of the reads its writes take their operands from — because a
  jointly-validated edit applied component-wise would overstate what was
  validated. A non-`atomic` group is never refused; a fully addressed atomic
  group runs. The six pair-validity checks — answer change, correctness (the
  `grade` of sec. 2.10), intended token change, absence of unintended edits,
  tokenizer stability, full-set location coverage — are six functions of the
  same module; `scripts/build_task_dataset.py --validate-pairs --tokenizer
  <key> --revision <rev>` runs the three that need only the rows and a
  tokenizer when a table is built and prints the tokenizer it validated under
  — recorded wherever the table's command line is recorded, since nothing
  sits beside a table; no run-time check compares it. Nothing here enters the
  canonical form (sec. 7).
- Dataset **columns** referenced by metrics and by `column` positions are
  checked against **base's** table by `validate --data`, not at load. The
  prompt **variables** named by `variable` positions and by `scope` /
  `relative_to` anchors are checked against the **union** of the roles, because
  a variable lives in a `<field>_variables` sibling: two roles reading
  different fields of one table legitimately name different variables. The check covers **every expanded point**, so
  a reference that appears only at one coordinate of a sweep is checked there
  too.

**Prepared token sequences.** A text column may carry a `<column>_encoding`
sibling, aligned as a list when the text field is indexed (`input_encoding`,
`counterfactual_inputs_encoding[0]`). Each record has `version: 1`, a
`tokenizer_digest`, `text_sha256`, exact `input_ids`, and `offset_mapping` pairs.
The neural preparation helper also records `prompt_length`. The tokenizer digest
covers vocabulary, backend tokenization rules, specials and chat template; mutable
batch padding/truncation state is excluded. The runtime checks text/tokenizer
identity and validates IDs/offsets before left-padding the supplied sequence.
These bytes belong to the dataset identity like every other column. No decoding
and retokenizing occurs. A prepared role must cover every row and must not be
combined with `segments`: it already owns rendering and special tokens. Position
zero is its first real token, including specials, and `all` includes every real
token. Author with `causalab.neural.sequences.prepare_sequence`; the complete
recipe is [multi_token_analysis.md](multi_token_analysis.md).

### 2.2.1 `segments`

Optional. A row's text has **named sequence segments** — in a chat frame the
system turn, the user turn, the template's generation prompt and the greedy
continuation; in any frame the value of a column the task serialized — and this
section declares them, so a position can name one as an anchor
(`{"index": -1, "scope": {"segment": "assistant_prefix"}}`, `{"segment":
"user"}`, `{"before": {"segment": "user"}}`; sec. 2.3). It sits after `data`
in the sec. 1 order, and its names are anchors like prompt variables and
columns — outside the global namespace.

```json
"segments": {
  "frame": "chat",
  "system": {"column": "instructions"},
  "declare": {"list": {"column": "list_text"}}
}
```

| key | value | means |
|---|---|---|
| `frame` | one of the frame table | how each row's prompt is rendered before tokenization. **Absent is plain text** — the row's `field` as it is, `prefix_lengths` 0 — and there is no literal spelling of that default |
| `system` | `{"column": "c"}` | chat frame only: the row's value for column `c` is the system turn |
| `declare` | `{"<name>": <source>, …}` | segments located from the row's own columns, in any frame |

**Frames** — closed vocabulary:

| frame | renders |
|---|---|
| `chat` | each row's prompt as the one **user** turn (plus the `system` turn when one is named) through the **tokenizer's own chat template** — `apply_chat_template(…, tokenize=False, add_generation_prompt=True)` — and encodes the rendering with the template's own specials (no BOS added twice). `prefix_lengths` becomes the count of the row's tokens before the user turn's first token |

**The chat frame's segments** — closed vocabulary, in reading order. `system`
is declared only when the section names a column for it; `continuation` is the
greedy continuation of sec. 2.3's `generated` frame and is declared in every
frame (it exists wherever a decode does):

| segment | is |
|---|---|
| `system` | the system turn — the rendered text's occurrences of the `system` column's value |
| `user` | the user turn — the rendered text's occurrences of the prompt |
| `assistant_prefix` | the template's generation prompt — the **difference** between the rendering with `add_generation_prompt` and the one without |
| `continuation` | the greedy continuation (sec. 2.3 `generated`). An anchor on it carries `generated` — the decode budget lives on the position — and the whole continuation is `{"generated": …, "all": true}`, not a `segment` span |

**Sources** for a declared segment — closed vocabulary:

| source | locates |
|---|---|
| `column` | the row's value for the named top-level column, as the occurrences of that string in the row's (rendered) text — resolved like a `column` position (sec. 2.3) |

- **Honest location.** Every segment boundary is computed by the engine from
  the **rendered text and the tokenizer's offset mapping** — the machinery a
  `variable` anchor uses — never from a column's serialized length or from a
  template string the package knows. A segment that does not occur in the
  rendered text is `absent`, one that occurs twice is `ambiguous` (sec. 2.3's
  cardinalities): an `unavailable` cell for a read, a refusal before any
  forward for a write, with the matching reason code. A template that does not
  carry the prompt verbatim cannot locate the user turn, so the row is refused
  with that reason rather than given a guessed prefix; a tokenizer with **no
  chat template** under `frame: chat` is refused before any forward with reason
  `chat_template_missing` (sec. 2.4). Nothing about a segment is checked at
  load beyond rule 27 (sec. 5): the pure verbs hold no tokenizer.
- **Optional and digest-neutral.** A document without this section is plain
  text exactly as before it existed — `prefix_lengths` 0, the same tokens, the
  same digest. An authored section is part of the canonical form (it changes
  what the model sees), and nothing in it has a materialized default.
- **What `validate --data` checks**: the `system` column and every `declare`
  column exist in the resolved tables, like a `column` position.

### 2.3 `positions`

Named entries; a read/write `pos` is a name here, or an inline spec.

| form | resolves to |
|---|---|
| `-1` (bare int, sugar) | `{"index": -1}` |
| `"all"` (bare string, sugar) | `{"all": true}` |
| `{"index": n}` | one token per row. `n < 0` counts from the end of the sequence; `n ≥ 0` is rebased past any chat prefix |
| `{"variable": "x"}` | all tokens of prompt variable `x` — a per-row window, ragged across rows |
| `{"column": "c"}` | all tokens of the string in column `c` of the row — the per-row form |
| `{"span": [a, b]}` | fixed window `[a, b)` |
| `{"all": true}` | every content token of the row — ragged across rows |
| `{"segment": "s"}` | all tokens of the declared segment `s` (sec. 2.2.1) — a per-row window, located by the frame |
| + `"scope": {"variable": "x"}` / `{"column": "c"}` / `{"segment": "s"}` | interpret the index/span inside the anchor's span |
| + `"relative_to": {"variable": "x"}` / `{"column": "c"}` / `{"segment": "s"}` | offset from the anchor's span |
| + `"generated": {"max_new_tokens": n}` | resolve the anchor inside the row's greedy continuation instead of its prompt |

- Positions are **never resolved to integers in the document**. Resolution is
  an engine service against a `PositionFrame` (pad side, packing, sequence
  shard map) — sec. 8. What it resolves to is a **derived output** (sec. 6):
  a document that saves a `location_ledger` entry (sec. 2.12) gets the
  resolved indices back as a table, never as authored fields.

**Spans (`SpanSpec`).** A position object carrying any key of the table below
is a *span*: a set of tokens composed from the anchors above, accepted wherever
a position is (`positions.<name>`, an inline `pos`). Exactly one selector per
span; `scope` / `relative_to` modify `indices` (and an atomic `span`); a span
addresses the prompt frame (`generated` is refused on it); members and anchors
are position objects spelled out — no int or `"all"` sugar inside a span — and
carry no `alignment` of their own. A union or intersection has two or more
members; `indices` is a non-empty list of distinct integers, counting like
`index` does (`n ≥ 0` from the content start, `n < 0` from the end).

| span key | shape | selects |
|---|---|---|
| `segment` | `"s"` | every token of the declared segment `s` — as a selector, beside the anchor form in the table above |
| `indices` | `[n, …]` | a **noncontiguous set**: each `n` as `{"index": n}` would, inside `scope`'s anchor when one is given |
| `union` | `[<pos>, <pos>, …]` | every token any member selects |
| `intersection` | `[<pos>, <pos>, …]` | every token all members select |
| `before` | `<pos>` | every **real** token of the row strictly before the anchor's first token — the chat prefix included, so `{"before": {"segment": "user"}}` *is* the prefix |
| `after` | `<pos>` | every real token strictly after the anchor's last token |
| `between` | `[<pos>, <pos>]` | every token strictly between the two anchors' runs, whichever order they occur in |
| `atomic` | `true` | the resolved set is **one address** (below). Also the one key that promotes a bare `span` / `variable` / `column` into a span |

- **`atomic` is one address.** A two-token number is *either* two positions
  (two `index` specs, or a non-atomic `indices` set — two constituent
  locations, each classified on its own) *or* one span with `atomic: true` —
  a **single** address for rule 8's "≤ 1 absolute write per (site,
  overlapping pos, model)", one joint run for the alignment cardinality below
  (`one_to_one` when both inputs give it the same width), and one landed slice
  for a write. There is no `atomic: false`: an unauthored span is its
  constituent locations, and the default has no literal spelling, so no
  document's canonical form moves. An atomic set the document alone fixes (an
  unscoped `indices`, an unscoped `span`, a union of such) has two or more
  members — one token is an `index` (rule 27).
- **Width.** A span is as wide as its members make it on each row. A write at
  a span whose width differs across rows is rule 19's refusal, exactly as an
  `all` or `variable` write is, unless the write declares a `ragged` policy
  (sec. 2.8); a read carries it ragged. A span that resolves
  to **no** token on a row (two `between` anchors that touch, an intersection
  of disjoint members) is refused with reason `empty_selector` rather than
  gathered silently.
- **Rule 8 at load** compares what the document alone fixes: two spans (or a
  span and an index) with static index sets in one sign regime are provably
  disjoint when the sets share no member; anything text-located (`variable`,
  `column`, `segment`, a predicate) is assumed to overlap.
- **`alignment` on a span** is the same optional key with the same five
  members as on any position (below): an atomic span is one joint run per
  input, a non-atomic set is checked per constituent, and both go through
  `alignment_of` — the span module derives no cardinality of its own.

**The continuation frame (`generated`).** A decode produces a second frame, so
addressing it needs no new vocabulary: `generated` is a **frame selector**, not
an anchor, and exactly one anchor accompanies it. `{"generated": {…}, "index":
-1}` is the last generated token, `{…, "all": true}` every generated token,
`{…, "span": [0, 3]}` the first three, `{…, "variable": "x"}` the tokens where
the model said the row's value for `x`.

- The decode is **greedy** — argmax at every step. Sampling is not expressible:
  a document is a value (sec. 7), and a sampled continuation is not a function
  of it.
- `max_new_tokens` is required, ≥ 1, and sweepable. It is a mapping rather than a
  bare int so stopping conditions can join it later. Two positions on one
  (model, input) with different budgets are legal: the run decodes the deepest,
  and each read windows its own.
- The frame **ends at the row's first EOS**, so widths differ and continuation
  reads are ragged. The PyTorch engine resolves EOS IDs from an explicit
  behavioral decoding request, otherwise from the model generation configuration,
  then the tokenizer. It preserves terminal EOS separately from content and pads
  post-stop slots in its raw behavioral output. Non-EOS special tokens remain in
  continuation text. Unsaved, untransformed continuation-head metrics reduce
  bounded vocabulary projections; explicit tensor saves still materialize the
  requested logits. A window reaching past a row's end clips; a row that
  generated nothing contributes no positions. Unlike the prompt frame — where an
  out-of-range index is an authoring error and refused — how far a row generates
  is a *result*, and refusing on it would make a document fail on data.
- **`variable` in the continuation** — "the tokens where the model said the
  row's value for `x`" — differs from its prompt-side twin in two ways, both
  because the continuation is a *result* rather than an input. It takes the
  **first** occurrence instead of demanding exactly one, since a generation may
  repeat itself as a matter of course; and **zero** occurrences yield zero
  positions rather than refusing, because whether the model says the thing is
  usually the experiment. A metric over such a read reports the miss as a null
  value with `matched: false` (sec. 2.10), so it is data, not an exception.
  Character spans come from the decode's own incremental detokenization, so a
  match that starts inside a merged piece still lands on every token that
  produced it.
- **Reads only.** A write may not carry `generated` (rule 16): the continuation
  exists because the prefill already ran, and an intervention reaches it through
  the first token's logits and through what the prefill left in the KV cache —
  nothing re-fires per decode step. `train` and `generated` do not combine
  either: a greedy decode is an argmax chain with no gradient path.
- **`lm_head` at generated position `j` is the distribution *after* token `j`.**
  The one that *produced* token `j` sits at `j − 1`, and for `j = 0` that is the
  last prompt position — an ordinary `{"index": -1}`. Stated here so no document
  has to rediscover it.
- `column`, `scope` and `relative_to` are refused with `generated`: they resolve
  against the prompt, which the continuation does not contain.
- `variable` vs `column`. A **prompt variable** is looked up per role: the
  `<field>_variables` sibling of the role's text column first, then a
  same-named column. A **column** is looked up only as a top-level column of
  the row, so it is a property of the *row*, not of a role's text — the same
  `{"column": "c"}` resolves to the same string whichever role reads it. Use
  `column` when the value is computed by the task (per-row answer symbols,
  chosen entities) and every role must read the same string; use `variable`
  when each role's own text carries its own value, which is what a
  counterfactual pair usually needs.
- **What `validate --data` can and cannot say about a position.** Both
  spellings are checked for **existence** — a `column` against the resolved
  tables' columns, a `variable` against each role's `<field>_variables` sibling
  and the same-named-column fallback. Neither is checked for **width**: what a
  variable or a column resolves to is a char span, hence a token count, and the
  pure verbs hold no tokenizer (`ResolutionEnv` carries datasets and artifacts
  and stays torch- and network-free). So a per-row window that turns out ragged
  across rows is a *run-time* refusal — rule 19 for a write — and no amount of
  pre-flighting moves it earlier. When a document needs one token per row at a
  variable, say so: `{"index": -1, "scope": {"variable": "x"}}` is the last
  token of `x`'s span and is never ragged.
- The value in a column position is a **string**, resolved like a variable's
  value (it must occur exactly once in the row's text). Integer token indices
  are deliberately not a v1 spelling: they would bind a table to one
  tokenizer, and a task that can compute an index can serialize the substring
  instead.
- **The chat prefix is the `segments` section's (sec. 2.2.1).** The `n ≥ 0`
  rebase above and the "past any chat prefix" clause below are written against
  `prefix_lengths`, which is **0 for every row of a plain document** — one
  with no `segments` section, or one whose section names no `frame`. Under
  `"segments": {"frame": "chat"}` the engine renders each row's prompt through
  the tokenizer's **own** chat template, and `prefix_lengths` is the real
  count of the row's tokens before the user turn (BOS, role markers, the
  system turn), located from the rendered text and the offset mapping: then
  `{"index": 0}` is the first token of the user's text, `{"index": -1}` the
  last token of the template's generation prompt, and the prefix itself is
  `{"before": {"segment": "user"}}`. The `use_chat_template` field and
  chat-prefix hook in `neural/token_positions.py` belong to the pre-protocol
  pipeline surface the task packages annotate against; no engine implements
  them, and that module says so at the top. A dataset may still bake a
  *rendered* template into its `input` column under the plain frame, which
  works; what it must not do is leave the template's leading BOS in place,
  because the plain encoder adds special tokens as the tokenizer defines them
  and a second BOS shifts every position by one. That case is refused at
  encode.
- **`{"all": true}`** selects the row's real tokens only: padding is excluded,
  and so is any chat prefix — the same frame `{"index": n}` uses for `n ≥ 0`.
  It takes no `scope` or `relative_to` (there is nothing left to narrow), and
  `all` is a reserved name, so `"pos": "all"` is always the sugar and never a
  lookup in this table. Rows of unequal length make it ragged: reads carry
  that natively, a write at `all` needs every row to be the same length
  unless the write declares a `ragged` policy (sec. 2.8) — rule 19, decided
  on the encoded batch either way.

**Alignment cardinality (`alignment`).** A counterfactual pair resolves the same
address on the base input and on each `counterfactual_inputs` member; how the
two token runs map onto each other is a fact with five values, and it is
**data**: authored as the optional `alignment` key of a position entry — the
cardinality the author *declares* — and derived at encode time as the
*observed* one by one function, `alignment_of` in
`causalab/protocol/alignment.py`, which planning (`plan.static_alignment`),
execution (position resolution, and the executor's declared-vs-observed check)
and metrics (the answer-form pre-flight of sec. 2.10) all call and none
re-derives.

| `alignment` | means |
|---|---|
| `one_to_one` | one run on each input, of the same width — a single token, or one joint span |
| `one_to_many` | one token on the base input, several on the counterfactual |
| `many_to_one` | several tokens on the base input, one on the counterfactual |
| `absent` | one input resolved to nothing: the value does not occur in that row's text (reason `alignment_missing`, sec. 2.4) |
| `ambiguous` | more than one alignment fits and none was named: the value occurs several times, or both runs are wider than one token and of unequal widths (reason `alignment_ambiguous`) |

- **Per address form.** An `index` is one token per row on every input and an
  unscoped `span [a, b)` one joint window of width `b − a`: both are
  `one_to_one` by construction. **Two `index` specs are two locations; one
  `span` is one joint address** — so a two-token number is `one_to_one`
  addressed as a span of two, and two `one_to_one` locations addressed as two
  indices. A `variable` or `column` window is as wide as the tokenizer makes it
  on each input, so its cardinality is *observed* at encode. `all` carries no
  `alignment` (it takes no modifiers), and neither does a `generated` position
  (the continuation is a result, not one of the pair's inputs).
- **Declared against observed.** What the document alone decides is rule 26
  (sec. 5): the value is one of the five, the address can carry one, and a
  declaration on an `index` or an unscoped `span` says `one_to_one`. The rest
  needs the tokenizer and is checked when the run encodes its inputs — the same
  boundary rule 19 sits on — and a declared cardinality the pair contradicts is
  a refusal naming both, never a silent override.
- **Unalignable rows are cells, not drops.** With nothing declared, a row whose
  `variable` / `column` value occurs zero times (`absent`) or several times
  (`ambiguous`) makes a *read* an `unavailable` result cell (sec. 4.1) carrying
  the matching reason code and a `detail` that names the value, its count and
  the row; the row contributes no positions and the cell is counted in the
  denominator. A *write* on such a row is refused before any forward pass with
  the same reason code: a write that skipped a row would report a number for an
  intervention that did not happen. (Both used to be a bare `P2`.)
- **Optional, undefaulted, digest-neutral.** An unauthored `alignment` stays
  absent through the canonical form (sec. 7) — no document's digest moves — and
  an authored one is part of it. The observed cardinality is never recorded
  (sec. 6) except as the `reason` of an unavailable cell.
- **A span carries the same enum** (the span table above): an `atomic` span
  is one joint address — `one_to_one` when both inputs give it the same width
  — and a non-atomic set is its constituent locations, each classified on its
  own (a one-token constituent against a wider joint run on the other input is
  `one_to_many`), derived through `alignment_of` and not a second time.
- **The pair-difference validator** (it guards against one alignment error
  class: a legitimate change in a recomputed answer prefix read as an
  unintended prompt edit). `alignment.pair_differences(tokenizer, base_prompt,
  counterfactual_prompt, base_prefix=…, counterfactual_prefix=…)` reports three
  difference sets **separately**: the **prompt**, the **teacher-forced prefix**
  (the answer prefix a task recomputes per input and the model is forced through
  before the position a metric reads) and the **full context** (prompt + prefix,
  tokenized as one string, because a merge across the boundary makes it more
  than the other two's union). A pair differing only in its recomputed prefix
  validates clean — an empty prompt set beside non-empty prefix and full sets —
  where one folded set would read it as a prompt edit. The validator compares
  the strings `encode` receives: under the plain frame (`prefix_lengths` 0
  above) a row's full context is its prompt followed by its answer prefix and
  nothing before it; a chat-framed document (sec. 2.2.1) hands it the rendered
  turns. It is a Python entry point rather than a verb: the pure verbs hold no
  tokenizer.

### 2.4 `sites`

```json
"target": {"component": "block_output", "layers": [18]}
```

| field | meaning |
|---|---|
| `component` | one of the vocabulary below |
| `layers` | the **band** of depth indices the site spans (where the component has one): a non-empty list of integers in strictly increasing order, `[18]` for one layer |
| `head` / `expert` / `stream` | optional sub-axes: attention head, MoE expert, **mixer stream** |

**`layers` is a band.** A site is one address across every layer it names —
one read on it captures that many tensors, one write on it lands that many
times, one operand — which is the shape a clipped restoration window (ROME) or
a contiguous band has: "one point, N layers restored at once". The one-layer
band `[18]` is the ordinary site, and behaves as the scalar it replaced
everywhere: the same module, the same tensor keys and labels
(`[target.layers=18]`), the same artifact stamp (`site.layers` carries the
list). The engines run a band as its per-layer **members** — `a[layers=10]`,
…, the N-site document the author would have written by hand, with a member
write taking member *i* of an operand read on a band of the same length and
any other operand broadcast to every member (`protocol/plan.py`
`lower_bands`); the canonical form and the digest are over the band. A band
read saved, fed to a metric or featurized is refused by name (a band read is
N tensors; save each layer as its own read or sweep the layer, sec. 3), as is
a featurizer on a band write. A band is distinct from `at_once` (sec. 3.1),
which declares N one-layer sites in one point.

Refused at parse (`[P2]`): an empty list, an unsorted or repeated index, a
boolean, `null`; `layers` on a layer-less component and a layered component
without it. The `protocol_version` 2 spelling `layer` is refused as an unknown
key naming the rename and `causalab migrate` (`[P3]`). Every member is checked
against the model at load (rule 4: inside the tower, carrying the declared
`stream`). **A bare index is the one-layer band**: an axis over `layers`
(`{"sweep": {"range": [0, 32]}}`, `{"at_once": …}`) and a workflow `emit`
hand a point the index, and the canonical form writes the list either way, so
`"layers": 18` and `"layers": [18]` carry one digest — the same fold a retired
component spelling gets. The coordinate a swept `layers` records is the index.

`head`, `expert` and `stream` are checked against the component, not against
the model: `head` is refused on a component with no head axis, and bounded by
*that component's* head count — which under GQA is narrower for a KV-space
component than for a query-space one; `expert` is refused at load on every
component but the five routed-interior ones, whose capability row (below)
names an engine for the ragged `expert:` face; `stream` is one of
`full_attention` / `linear_attention`.

Component vocabulary (per-engine `SiteResolver` maps each to a tap). Listed in
reading order: the model's input, then one block walked from `block_input` to
`block_output`, then the head — with each mixer family's interior grouped where
it sits in that walk. That is a narrative order, not the tuple order of
`schema.COMPONENTS`, and nothing depends on either: the census
(`tests/protocol/test_vocabulary_census.py`) checks the *set*, all 56 of
them, and deliberately not the order:

`input_ids` · `embeddings` · `block_input` · `attention_input_norm` ·
`delta_qkv` · `delta_gate` · `delta_conv` · `delta_query` · `delta_key` ·
`delta_value` · `delta_beta` · `delta_decay` · `delta_kv_mem` ·
`delta_state_update` · `delta_state` · `delta_kernel_output` · `delta_premix` ·
`attention_query_pre_rope` · `attention_key_pre_rope` ·
`attention_value_states` · `attention_gate` · `attention_query` ·
`attention_key` · `attention_scores` · `attention_z` · `deltanet_query` ·
`deltanet_key` · `deltanet_state` · `attention_result` ·
`attention_output` · `attention_premix` · `attention_probs` · `block_mid` ·
`mlp_input_norm` · `mlp_input` · `router_logits` · `router_scores` ·
`expert_idx` · `expert_gate_proj` · `expert_up_proj` · `expert_activation` ·
`expert_neuron_output` ·
`expert_permutation` · `expert_output` · `routed_output` · `mlp_activation` ·
`mlp_neuron_output` ·
`shared_expert_gate_proj` · `shared_expert_up_proj` ·
`shared_expert_activation` · `shared_expert_output` · `shared_expert_gate` ·
`mlp_output` · `block_output` · `ln_final` · `lm_head`

- **`sites` is the complete inventory**: every site a read or write references
  must be declared here, including `lm_head` (`{"component": "lm_head"}`).
  There are no implicit site names.
- **The three norm taps** name the two RMSNorms every block carries:
  `attention_input_norm` is `input_layernorm`'s **output** (what the mixer
  consumes), `block_mid` is `post_attention_layernorm`'s **input** (the residual
  stream after the mixer is added), and `mlp_input_norm` is that same module's
  **output**. So a block satisfies
  `block_mid = block_input + attention_output` and
  `block_output = block_mid + mlp_output`.
- **`input_ids` is the model's token input, not an activation.** It is
  layer-less, **read-only**, and *not a feature space*: it carries integer ids
  on a position axis, so no featurizer may attach to it and it has no width.
  Read `embeddings` for the vector the ids look up.
- **The MoE surface** splits four ways. The **router** exposes
  `router_logits` (all experts), `router_scores` (the renormalized top-k) and
  `expert_idx` (which experts, integer ids on the same top-k axis);
  `router_probs` is *derived* — `softmax(router_logits)` — and is not a
  component. `routed_output` is the combined expert output, and the **shared
  expert** exposes its SwiGLU interior plus `shared_expert_gate`, the scalar
  that mixes it in. The *routed* per-expert interior has no module
  boundaries at all — the experts module stores its weights as 3-D parameters
  and computes the whole interior inside one dispatched
  `ALL_EXPERTS_FUNCTIONS["grouped_mm"]` call — so its components are reached by
  wrapping that dispatch entry, and they carry a **dispatch pin**: a model
  loaded with any other `experts_implementation` (the `"eager"` per-expert
  loop, `"batched_mm"`) is refused by name, because a different factorization
  computes different intermediates even where the block's output agrees (to
  4.2e-7 on the fixture). `expert_activation` is the activated gate half,
  `act_fn(gate_e)` — the same tensor `mlp_activation` names on the llama
  family — represented **token-major**: `(batch·position, top_k · d_expert)`,
  slot *k* the *k*-th ranked expert, joined to experts through `expert_idx`.
  `expert_neuron_output` contains the complete `act_fn(gate_e) * up_e`
  value before the down-projection. It uses the same token and slot axes.
  `mlp_neuron_output` captures the dense down-projection input. On a gated
  MLP this is `act_fn(gate) * up`; on GPT-2 it is the `c_proj` input.
  `shared_expert_activation` already names the shared down-projection input.
  The routed slot axis is a ranking, like `router_scores`, with the same
  basis-fitting refusal; a per-column featurizer is accepted, and a `gate`
  may be **expert-keyed** (`group: expert_neuron`, sec. 2.5), holding one
  parameter per (expert, neuron) and reading a token's slots through that
  same join — the one featurizer whose parameters follow the expert rather
  than the slot. So are the other interior slots ranked:
  `expert_gate_proj` and `expert_up_proj` are the two halves of the fused
  `[gate_e | up_e]` projection (one capture, two addresses — the
  `attention_gate` precedent), and `expert_output` is the down-projection's
  output **before** the routing weight, pinned by the identity
  `routed_output == Σ_slot expert_output · router_scores` (exactly, 0.0).
- **`expert: e` is the ragged face of the routed interior.** On the four
  interior components it selects the (position, slot) pairs the router sent to
  expert *e* and returns flat rows plus per-example widths (a ragged value);
  an expert no token chose returns width-0 rows — a data fact, not an error.
  A write under `expert: e` lands only on that expert's rows (and therefore
  lands nowhere when no addressed token chose it). `featurizer`/`dims` are
  refused on this face: they are sized against the token-major `top_k · d`
  axis and these rows are `d`-wide. On every other MoE component `expert`
  is still refused — those tensors have no per-expert axis.
  `expert_permutation` (integral, read-only) is the serving kernel's row
  bookkeeping for anyone aligning raw kernel-order tensors; it lives inside
  the fused forward where no module boundary exists, so only the nnterp
  engine's `.source` address table serves it. The other interior components
  are served by both engines — the reference engine through the dispatch
  wrapper above, the nnterp engine through its `.source` addresses — with
  the same token-major presentation, the same ragged `expert:` face and the
  same pre-routing-weight `expert_output`.
- **`expert_idx` is a routing table, not a feature space** — the same rule as
  `input_ids`: integer ids, no featurizer, no width. And `router_scores` has a
  width but its axis is a per-token **ranking**, not a basis: column *k* is the
  *k*-th ranked expert, a different expert for different tokens, so a basis
  fitted across positions is fitted across a basis that is itself shuffled per
  position. A **basis-fitting** featurizer there (`subspace`, `pca`, `sae`) is
  therefore refused; `identity`, `standardize` and `gate` act per column and are
  still accepted, because "how large is the top-ranked score, typically" is a
  meaningful question about a ranking.
- **`expert` is refused on every other component, at load.** The four
  routed-interior components above are the only rows whose `expert_selection`
  names an engine; on every other component — the router's (all-experts or
  top-k axes), the shared expert's (not one of the routed experts), and every
  non-MoE tensor — `sites.<name>.expert` is a reference to an axis the
  component does not have, and rule 4 refuses it with the same text the site
  resolver prints for a document that arrives unvalidated. The per-expert
  interior is otherwise indexed by routed *slot* (its top-k axis) —
  `expert_idx` says which expert fills each slot.
- **One name per DeltaNet tensor; three typed backend pairs.** The Gated
  DeltaNet interior is named once, engine-neutrally, by the `delta_*`
  components below — the delta rule the tensor belongs to, not the modeling
  file's variable names — and **both engines serve every one of them**: the
  reference engine at module boundaries and by swapping the kernel globals,
  the nnterp engine through envoys and its `.source` address table, each
  translating the one name to its own mechanism. Eight `deltanet_*` spellings
  of the same tensors
  (`deltanet_qkv` → `delta_qkv`, `deltanet_qkv_conv` → `delta_conv`,
  `deltanet_gate` → `delta_gate`, `deltanet_value` → `delta_value`,
  `deltanet_beta` → `delta_beta`, `deltanet_decay` → `delta_decay`,
  `deltanet_core_out` → `delta_kernel_output`, `deltanet_gated_out` →
  `delta_premix`) are **retired spellings**: a document that authors one
  parses and canonicalizes to the name on the right, so both digest
  identically (`schema.DEPRECATED_COMPONENTS`, each with the protocol version
  it was retired under in `schema.DEPRECATED_IN`). The rule an alias must
  satisfy is that it *redirects* — the two spellings name one tensor of one
  shape at one time — and never *rebinds*: three pairs fail it and therefore
  stay **two names with a typed backend requirement**, a row each
  (`registry.BACKEND_PAIRS`) saying which engine serves which spelling and how
  the two captures relate. `deltanet_query` and `deltanet_key` are the q/k
  splits **before** the GVA `repeat_interleave` — *key-head* space, the
  linear-attention analogue of GQA — where `delta_query` / `delta_key` are the
  kernel's arguments after it, which both engines read off the kernel call
  (`gva_tile`: exact after tiling); and
  **`deltanet_state`** is the recurrent state once per 64-token prefill
  chunk, its position axis the **kernel's chunk index**, where `delta_state`
  is per step (`chunk_boundary`: the chunk's state is the step-state at the
  chunk's last position). Those three are served by the nnterp engine
  alone; per-token prefill state does not exist there — the recurrent kernel
  runs only in single-token decode, by the modeling code's own dispatch — so
  it is refused by name rather than served at a granularity the kernel does
  not have. In the **generated frame** `deltanet_state` *is* per token (a
  separate, decode-verified address, because decode dispatches different
  kernels than prefill; interior components without one refuse by name in
  that frame). `registry.alias_would_rebind` is the guard: pointing an alias
  at a `gva_tile` or `chunk_boundary` pair is refused with the relation, and
  the census holds the alias table to it.
- **`attention_result` is the per-head contribution to the residual stream**,
  and the only component the model never computes: the block projects the whole
  `attention_premix` at once, so what the forward pass forms is the *sum* over
  heads. `sum_h attention_result == attention_output` (minus the o-projection's
  bias, which belongs to no head) is the identity that defines it.
  Naming a `head` is strongly encouraged — the dense form is `heads` times
  `attention_output`, which on a 64-head model at hidden 4096 is 64× the memory
  — but the whole tensor is not refused, only documented. The read is derived
  after the position gather, so the cost is `n_positions · heads · hidden`
  rather than `seq · heads · hidden`.
- **Six components are read-only, and a write to any is refused** rather than
  silently discarded — at load by rule 4, and again at the plan for a document
  that arrives unvalidated, from the same capability row. `input_ids` is the
  model's input. `router_logits` is discarded by the MoE block itself, which
  routes on the scores and indices it computed from them — so a write there
  could not reach anything. Write `router_scores` to reweight the chosen
  experts, or `expert_idx` to change which experts fire. `attention_result` is
  *derived* — there is no tensor there to change — so write `attention_premix`
  with the same `head` instead; the result is a linear function of it.
  `expert_permutation` is the serving kernel's bookkeeping, not routing;
  `delta_kv_mem` is a memory readout recomputed every step; and
  `delta_state_update` writes are deferred — write `delta_state` for
  either. Two more components take **only `swap`**: `expert_idx` (integer
  labels) and `attention_probs` (a normalized distribution, see below). The
  policy is the row's `writes` cell, and the refusal text is its `why`.
- **The mixer's interior is four module boundaries, not four chunk ops.**
  `attention_query_pre_rope` and `attention_key_pre_rope` are the queries and
  keys as the mixer computes them, *before* RoPE rotates them — on a family with
  `q_norm`/`k_norm` those norms run before RoPE, so their outputs are exactly
  these tensors. `attention_value_states` is `v_proj`'s output: the actual value
  vectors, in **KV-head space**, and the tap sits before the KV cache is
  updated, so a write there reaches it. `attention_gate` is the second split of
  the gated-attention family's fused `[q | gate]` projection.
- **`attention_value_states` is not `attention_premix`,** and the two are the
  reason the latter was renamed. `attention_premix` is the o-projection's
  *input* — the mixer's output after the gate, in **query-head** space, `heads ·
  head_dim` wide. `attention_value_states` is `v_proj`'s output, in **KV-head**
  space, `kv_heads · head_dim` wide. Under GQA those differ by the group ratio,
  so a `head` valid on one can be out of range on the other; the bound is the
  component's, and naming a head the component does not have is an error.
- **`attention_gate` exists only where the mixer computes one.** Qwen3.5/3.6's
  attention multiplies its output by `sigmoid(gate)` before projecting out, and
  packs the gate into the q-projection. A family without one refuses the
  component by name rather than returning a slice of `q` — at load, for a
  family the per-family tap table has met (`gpt2`, `llama`), and at run for
  every family. All four require a full-attention layer.
- **On a fused-qkv family the three are logical slices of the fused
  projection.** GPT-2 computes q, k and v as one `c_attn` output and splits it
  into three `H·d`-wide column blocks; the component rows' per-family
  addresses (`overrides`, below) name those blocks, so `attention_query_pre_rope`,
  `attention_key_pre_rope` and `attention_value_states` are the **same logical
  sites** on a fused and a split projection — same names, same `(b, s, H·d)`
  value, same module-output read and write — and equal the oracle's slice of
  `x @ W_c_attn + b` within the write oracle's tolerance
  (`tests/neural/engines/pytorch_hooks/test_family_tap_table.py`). A fused
  family the table has *not* met is still refused by name: which block is
  which is a family fact only a row can state.
- **Four more components live *inside* the attention function**, where no
  forward hook reaches: `attention_query` and `attention_key` are the post-RoPE
  queries and keys as that function receives them (`attention_key` before the
  GQA `repeat_kv`, so **KV-head space**), `attention_scores` is the softmax's
  input, and `attention_z` is the function's result — the mixer's output
  *before* the gate multiply and the o-projection. Unlike the module-boundary
  four, these do not depend on separate q/k/v projections, so they read on a
  fused-qkv family too.
- **`attention_scores` is the write surface `attention_probs` could not be.**
  They are the same tensor one step apart and have identical axes; what differs
  is what happens next. After the pattern comes the value multiply, which
  assumes rows summing to 1 and gets whatever an edit produced — so the pattern
  accepts only `swap`. After the scores comes the model's own softmax, which
  renormalizes by construction — so **every mechanism is legal**. Attention
  knockout is an `add_scaled` of a large negative mask; head boosting is a
  scale. Note that a *uniform* shift is a no-op, because softmax is invariant to
  a shift along the axis it normalizes: a knockout has to be targeted, which
  means a full-shape operand rather than a scalar. `gaussian` is refused, since
  its noise is drawn per feature axis and this tap has none.
- **Continuation reads are refused where the steps do not stack.** A decode step
  attends over the whole KV cache, so a tensor indexed by the positions being
  attended *to* grows by one per step while the query axis stays 1.
  `attention_probs`, `attention_scores` (two position axes) and `attention_key`
  (one position axis, over the keys) therefore refuse in the `generated` frame;
  `attention_query` and `attention_z` are query-axis-shaped and read normally.
  Writes never need the rule — rule 16 already makes them prefill-only.
- **`attention_probs` is the whole attention pattern**, `(batch, heads, query,
  key)`, exposed whole: `pos: "all"`. Both of its trailing axes
  are positions — its *feature* axis IS a position axis — so addressing one
  query row, attaching a featurizer, or slicing `dims` is **refused** rather
  than approximated. Those three refusals are not written per component: the
  tap declares its axes as `(batch, head, position[query], key_position[key])`,
  which has two position axes and therefore no `(batch, position, feature)`
  form, and each refusal follows from something that form would have provided.
  A write replaces the whole pattern, which is what an
  interchange on attention means, and both inputs must have the same number of
  positions. The edit is handed back to the model's own value multiply — nothing
  recomputes it — so a write here works on every family whose eager attention
  the backend can wrap, and every mechanism but `swap` is refused because
  nothing downstream restores rows summing to 1. That refusal is made at
  **load** (rule 4, reason `unsupported_mechanism`): the pattern *looked*
  writable to the validator and was refused only by the engine, which is the
  mismatch the capability row closes.
- **The Gated DeltaNet interior begins at its module boundaries**:
  `delta_qkv` is `in_proj_qkv`'s output — the fused `[q | k | v]` projection,
  whose three widths are *unequal* (`key_dim`/`key_dim`/`value_dim`), so it has
  no head axis and reads whole (or via `dims`); `delta_gate` is `in_proj_z`'s
  output, the output gate, value-head space; and `delta_premix` is `out_proj`'s
  **input** — the post-norm, post-gate mixer value, the exact analogue of
  `attention_premix`, which is why the name. All three require a
  `linear_attention` layer: at a full-attention layer they refuse with the
  mirror of the DeltaNet refusal ("a gated-attention mixer computes no
  delta-rule state"), and a family with no linear stream anywhere (llama,
  GPT-2) hits that refusal at every layer. The conv output and the kernel
  boundary are *function* taps on the reference engine — the
  `conv1d` module never fires — and `.source` lines on the nnterp engine;
  both engines serve all three module boundaries as ordinary module taps.
- **Seven more DeltaNet boxes live at the kernel boundary**, as
  arguments and returns of two module-global call sites the forward uses:
  `delta_conv` is `causal_conv1d_fn`'s return (channels-first, the fused
  unequal widths again, so no head axis); `delta_query`/`delta_key`/
  `delta_value` are the kernel's first three arguments — post-conv,
  GVA-**tiled** to the value-head count, and **pre**-l2norm (the kernel
  normalizes and scales internally, so these are the tensors a write can
  steer); `delta_beta` (`sigmoid(in_proj_b)`) and `delta_decay` (the
  log-decay `g`, negative reals) are its per-head gates, whose feature axis
  IS the head axis; `delta_kernel_output` is its return — the pre-norm,
  pre-gate `core_attn_out`, pinned by
  `norm(delta_kernel_output, delta_gate) == delta_premix` (exactly). The
  wrappers swap the modeling file's own globals for the dynamic extent of the
  tapped mixer's forward and call through to the originals — so whatever
  hub/`fla` dispatch the environment resolved keeps computing, and identity
  is bit-exact by construction. Both delta-rule kernels and both conv
  entry points are swapped together, so cached decode steps (which natively
  run the recurrent kernel and `causal_conv1d_update`) are tapped identically
  to prefill — `delta_key` therefore reads in the generated frame on the
  reference engine, unlike `attention_key` (the kernel receives one step's k,
  not the prefix); the nnterp engine serves the kernel boundary in the
  prompt frame only, until a decode address is verified. A `kernelize()`d
  mixer (a hub-kernel class forward) is refused by name, as is a family whose
  modeling file does not export the four globals. The untiled q/k are
  `deltanet_query` / `deltanet_key` (the typed pairs above); the post-split
  views are not components (one box, one address — they are `delta_conv`
  rows re-viewed).
- **The DeltaNet per-step interior is read by stepping the library's own
  recurrent kernel** — intercept, never transcribe. `delta_state` is the recurrent state `S_t`: one `d_k × d_v`
  matrix per head per step, the second shape with no feature space (after the
  attention pattern) — but unlike the pattern it keeps its one position axis,
  so positions gather on the *steps* axis, `head:` selects a matrix stack, and
  `featurizer`/`dims` refuse off the declared axes. `delta_kv_mem`
  (`(S_{t-1}·exp(g_t) · k̂_t).sum`) and `delta_state_update` (`(v_t −
  kv_mem_t)·β_t`, the diagram's `delta`) are derived from adjacent states and
  pinned by the reconstruction identity `S_t == S_{t-1}·exp(g_t) + k̂_t ⊗
  delta_t` against the kernel's own returned states, exactly. At **prefill** a
  read runs the stepwise loop in the chunked call's *shadow*: the base forward
  is bit-identical, and the cost is O(seq) extra kernel calls at the tapped
  layer only (on a real checkpoint a full-seq all-layers `delta_state` is
  `layers · seq · heads · d_k · d_v` floats — address positions early). At
  **decode** the model runs the recurrent kernel natively, so generated-frame
  reads are plain per-step captures, pinned cross-path against test-side
  stepping. A **write** to `delta_state` substitutes the stepwise loop for the
  chunked call so edits feed forward — the one deliberate path-forcing in the
  vocabulary, costing ~5e-7 on the fixture's logits, pinned per layer as a
  bound. Its tensor operand must cover exactly the write's addressed steps
  (step-for-step, no broadcasting). `delta_kv_mem` is **read-only** (a memory
  readout has no independent existence — write `delta_state` or `delta_value`)
  and `delta_state_update` writes are deferred (they lower exactly onto a
  state edit via the reconstruction identity).
- **`stream` names a mixer stream, and it is a per-layer fact.** It is one of
  `full_attention` / `linear_attention`. A hybrid tower carries a different mixer
  at different depths (Qwen3.6's text tower alternates Gated DeltaNet with gated
  full attention), so a site whose declared `stream` contradicts the layer it
  names is refused. Most components are stream-bound without any declaration
  (the `stream` cell of their capability row; `registry.COMPONENT_STREAMS` is
  the view of those cells): every `attention_*` name except the two the block
  produces whatever its mixer (`attention_input_norm`, `attention_output`)
  exists only on a full-attention layer — a linear-attention block computes no
  attention matrix and has no `o_proj` — and every `delta_*` / `deltanet_*` name
  only on a linear-attention one. Both refusals are about the architecture and
  are permanent. They are made twice from the one table: at **load**, when the
  registry entry declares the tower's `layer_types` (Qwen3.6-35B-A3B does; so
  does an entry adapted from an HF config whose `layer_types` the adapter can
  place — `full_attention` and `sliding_attention` are both the softmax mixer,
  `linear_attention` is the kernel; any other spelling leaves the pattern
  unset), so `validate` refuses `attention_premix` at a DeltaNet layer
  offline; and at **run**, against the module the layer actually carries, for
  a model whose entry declares no pattern.
- Sites are pure data — no behavior, no model handles.

**The capability registry — one row per component.** Everything above that is
a fact *about a component* rather than about a document — which engines serve
it, what a write may do to it, which mixer stream it needs, which architectural
facts it requires, whether it has a ragged `expert:` face, what its retired
spellings are — is one row of `causalab/protocol/registry.py`'s
`CAPABILITIES`, keyed by the component name. There is no second table: each
engine's `components` set is generated from the rows' `reads` cell, the write
policy `validate` (rule 4) and the executor apply is the rows' `writes` cell,
the stream check reads the rows' `stream` cell, the site resolver's family
refusals read the rows' `requires` cell, and the component tables in
`docs/running_experiments.md` §5 and in §8 below are rendered from the rows.
`tests/protocol/test_vocabulary_census.py` holds every one of those derived
statements to the rows and the rows to the 56-name vocabulary. A new
architecture, a new mechanism or a new refusal is a row here and a generated
docs row — never a table anywhere else. *Where* a component is on a model
family — which module, which side, which function slot — is the family's
**plugin** (sec. 8, the family contract): a `registry.FamilyAdapter`
registered beside the rows, whose per-family taps are the availability of the
one global vocabulary on that family.

**Per-component availability, from the rows.** The table below states that
once per component — whether it takes a `layers` band or is layer-less
(`schema.LAYERLESS_COMPONENTS`), its mixer stream, its engines, its write
policy, the predicates it requires, its `expert:` face and its retired
spellings — every cell a field of the component's `CAPABILITIES` row. It is
generated from those rows (the `availability-table` block) and held byte for
byte to that rendering, never edited by hand. Where a cell says a surface
is unavailable it says so in the refusal's own words — the row's `why`, the
predicate text the load-time refusal prints — never as a placeholder.

<!-- generated: begin availability-table -->

| component | layers | stream | engines | write policy | requires | expert face | aliases |
|---|---|---|---|---|---|---|---|
| `input_ids` | layer-less — refused with `layers` | — (layer-less) | both | read-only — the model's token input is not an activation; change the row's text instead, or write 'embeddings' to edit the vector the ids look up | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `embeddings` | layer-less — refused with `layers` | — (layer-less) | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `block_input` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `attention_input_norm` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `delta_qkv` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | `deltanet_qkv` (retired under protocol version 1) |
| `delta_gate` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | `deltanet_gate` (retired under protocol version 1) |
| `delta_conv` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | `deltanet_qkv_conv` (retired under protocol version 1) |
| `delta_query` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `delta_key` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `delta_value` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | `deltanet_value` (retired under protocol version 1) |
| `delta_beta` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | `deltanet_beta` (retired under protocol version 1) |
| `delta_decay` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | `deltanet_decay` (retired under protocol version 1) |
| `delta_kv_mem` | a `layers` band | `linear_attention` layers only — refused on the other mixer | `pytorch_hooks` only — `nnterp` refuses it by name | read-only — a memory readout has no independent existence: it is (S_{t-1}·exp(g_t) · k̂_t) summed, recomputed from the state at every step, so there is no tensor a write could persist into. Write 'delta_state' to change what the memory holds, or 'delta_value' to change what is stored into it | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `delta_state_update` | a `layers` band | `linear_attention` layers only — refused on the other mixer | `pytorch_hooks` only — `nnterp` refuses it by name | read-only — its write lowers exactly onto a state edit through the reconstruction identity S_t = S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t, and that lowering is deferred — write 'delta_state' instead | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `delta_state` | a `layers` band | `linear_attention` layers only — refused on the other mixer | `pytorch_hooks` only — `nnterp` refuses it by name | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `delta_kernel_output` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | `deltanet_core_out` (retired under protocol version 1) |
| `attention_query_pre_rope` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | any mechanism | `split_qkv` — needs addressable q/k/v projections (the per-family tap table has none) | none — `expert` is refused at load: no per-expert axis | none |
| `attention_key_pre_rope` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | any mechanism | `split_qkv` — needs addressable q/k/v projections (the per-family tap table has none) | none — `expert` is refused at load: no per-expert axis | none |
| `attention_value_states` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | any mechanism | `split_qkv` — needs addressable q/k/v projections (the per-family tap table has none) | none — `expert` is refused at load: no per-expert axis | none |
| `attention_gate` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | any mechanism | `gated_attention` — needs an output gate on its attention mixer (the per-family tap table declares none for this family: only Qwen3.5/3.6's q-projection emits [q \| gate] per head); `split_qkv` — needs addressable q/k/v projections (the per-family tap table has none) | none — `expert` is refused at load: no per-expert axis | none |
| `attention_query` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `attention_key` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `attention_scores` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `attention_z` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `deltanet_query` | a `layers` band | `linear_attention` layers only — refused on the other mixer | `nnterp` only — `pytorch_hooks` refuses it by name | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `deltanet_key` | a `layers` band | `linear_attention` layers only — refused on the other mixer | `nnterp` only — `pytorch_hooks` refuses it by name | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `deltanet_state` | a `layers` band | `linear_attention` layers only — refused on the other mixer | `nnterp` only — `pytorch_hooks` refuses it by name | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `attention_result` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | read-only — it is derived, not computed: the model never forms the per-head contribution at all — it forms their sum, by projecting the whole 'attention_premix' at once — so there is no tensor here for a write to change. Write 'attention_premix' instead, with the same 'head'; 'attention_result' is a linear function of it, so a write there moves this by exactly the projection of what you wrote | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `delta_premix` | a `layers` band | `linear_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | `deltanet_gated_out` (retired under protocol version 1) |
| `attention_output` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `attention_premix` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | `attention_value` (retired under protocol version 1) |
| `attention_probs` | a `layers` band | `full_attention` layers only — refused on the other mixer | both | `swap` only — its rows are a probability distribution and the value multiply immediately downstream assumes they sum to 1 — nothing renormalizes them after an edit. Write 'attention_scores' instead: it is the same tensor one step earlier, upstream of the model's own softmax, so every mechanism is legal there and the rows still sum to 1 by construction | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `block_mid` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `mlp_input_norm` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `mlp_input` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `mlp_output` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `mlp_activation` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `mlp_neuron_output` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `router_logits` | a `layers` band | either | both | read-only — the MoE block discards the router's logits (it destructures them into '_') and routes on the scores and indices it computed from them, so a write here cannot reach anything — write 'router_scores' to reweight the chosen experts, or 'expert_idx' to change which experts fire | `moe` — needs a sparse-MoE block (the entry declares no experts) | none — `expert` is refused at load: no per-expert axis | none |
| `router_scores` | a `layers` band | either | both | any mechanism | `moe` — needs a sparse-MoE block (the entry declares no experts) | none — `expert` is refused at load: no per-expert axis | none |
| `expert_idx` | a `layers` band | either | both | `swap` only — the routing table carries integer expert ids, not features: a delta, a scale or a clamp over them yields ids chosen by arithmetic on labels, which route to arbitrary experts where they stay in range and fail at the gather where they do not. Swap in an index tensor read from elsewhere to change which experts fire, or write 'router_scores' to reweight the experts already chosen. Refusing rather than doing arithmetic on values that are labels | `moe` — needs a sparse-MoE block (the entry declares no experts) | none — `expert` is refused at load: no per-expert axis | none |
| `expert_gate_proj` | a `layers` band | either | both | any mechanism | `grouped_mm` — needs the grouped experts dispatch (the loaded model runs another experts_implementation — a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe` — needs a sparse-MoE block (the entry declares no experts) | `expert:` served by both | none |
| `expert_up_proj` | a `layers` band | either | both | any mechanism | `grouped_mm` — needs the grouped experts dispatch (the loaded model runs another experts_implementation — a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe` — needs a sparse-MoE block (the entry declares no experts) | `expert:` served by both | none |
| `expert_activation` | a `layers` band | either | both | any mechanism | `grouped_mm` — needs the grouped experts dispatch (the loaded model runs another experts_implementation — a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe` — needs a sparse-MoE block (the entry declares no experts) | `expert:` served by both | none |
| `expert_neuron_output` | a `layers` band | either | both | any mechanism | `grouped_mm` — needs the grouped experts dispatch (the loaded model runs another experts_implementation — a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe` — needs a sparse-MoE block (the entry declares no experts) | `expert:` served by both | none |
| `expert_permutation` | a `layers` band | either | `nnterp` only — `pytorch_hooks` refuses it by name | read-only — it is the serving kernel's row bookkeeping (where each (token, slot) row sits in expert-sorted order), not routing: the kernel derives it from the routing table, and an edited copy would describe rows that were never sorted that way. Write 'expert_idx' to change which experts fire, or 'router_scores' to reweight them | `moe` — needs a sparse-MoE block (the entry declares no experts) | none — `expert` is refused at load: no per-expert axis | none |
| `expert_output` | a `layers` band | either | both | any mechanism | `grouped_mm` — needs the grouped experts dispatch (the loaded model runs another experts_implementation — a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe` — needs a sparse-MoE block (the entry declares no experts) | `expert:` served by both | none |
| `routed_output` | a `layers` band | either | both | any mechanism | `moe` — needs a sparse-MoE block (the entry declares no experts) | none — `expert` is refused at load: no per-expert axis | none |
| `shared_expert_gate_proj` | a `layers` band | either | both | any mechanism | `moe` — needs a sparse-MoE block (the entry declares no experts); `shared_expert` — needs a shared expert (the entry declares no shared-expert width) | none — `expert` is refused at load: no per-expert axis | none |
| `shared_expert_up_proj` | a `layers` band | either | both | any mechanism | `moe` — needs a sparse-MoE block (the entry declares no experts); `shared_expert` — needs a shared expert (the entry declares no shared-expert width) | none — `expert` is refused at load: no per-expert axis | none |
| `shared_expert_activation` | a `layers` band | either | both | any mechanism | `moe` — needs a sparse-MoE block (the entry declares no experts); `shared_expert` — needs a shared expert (the entry declares no shared-expert width) | none — `expert` is refused at load: no per-expert axis | none |
| `shared_expert_output` | a `layers` band | either | both | any mechanism | `moe` — needs a sparse-MoE block (the entry declares no experts); `shared_expert` — needs a shared expert (the entry declares no shared-expert width) | none — `expert` is refused at load: no per-expert axis | none |
| `shared_expert_gate` | a `layers` band | either | both | any mechanism | `moe` — needs a sparse-MoE block (the entry declares no experts); `shared_expert` — needs a shared expert (the entry declares no shared-expert width) | none — `expert` is refused at load: no per-expert axis | none |
| `block_output` | a `layers` band | either | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `ln_final` | layer-less — refused with `layers` | — (layer-less) | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |
| `lm_head` | layer-less — refused with `layers` | — (layer-less) | both | any mechanism | nothing | none — `expert` is refused at load: no per-expert axis | none |

<!-- generated: end availability-table -->

**The per-family tap table** is the rows' `overrides`: for each family the
mixer interior has been measured on (the registry entry's `family`, the HF
`model_type` the adapter reads — `gpt2`, `llama`, `qwen3_5_moe_text`), the
**address** of each of the four attention-interior components — the mixer
child whose output is tapped and how that child's native tensor packs the
component's logical value. Only those four rows carry any (the census holds
that), and `docs/running_experiments.md` §5 renders the table from them
(`registry.render_family_table`). The site resolver reads the address; a
family absent from a row is served by measurement where the mixer is
unambiguous (a bare projection, a norm after it) and refused where it is not
(a fused projection's block order). An address has these keys, closed:

| override key | meaning |
|---|---|
| `module` | the mixer child whose *output* is the tap — a name (`q_proj`, `q_norm`, `c_attn`), never a module: the protocol layer stays torch-free |
| `packing` | how that module's native tensor packs the component's value — one of the packings below |
| `splits` | for a fused packing: how many logical tensors share the module's output (2 for `[q \| gate]`, 3 for `[q \| k \| v]`) |
| `split` | for a fused packing: which of them this component is |

| packing | native tensor | measured on |
|---|---|---|
| `flat` | `(b, s, heads·d)` — the module's whole output *is* the value | llama's bare `q_proj`/`k_proj`/`v_proj` (16 = 4·4) |
| `head_axis` | `(b, s, heads, d)` — the whole output, head axis kept | qwen3.5-moe's `q_norm`/`k_norm`, which run before RoPE (`(1, 5, 8, 32)`) |
| `fused_heads` | `(b, s, heads·splits·d)` — the splits interleaved *per head* | qwen3.5-moe's `q_proj`, `[q_h \| gate_h]` (512 = 8·2·32) |
| `fused_blocks` | `(b, s, splits·heads·d)` — the splits as contiguous *blocks* | GPT-2's `c_attn`, `[q \| k \| v]` (96 = 3·4·8) |

The logical value's axes (`component_shape`) are family-independent; the
packing is the half of the shape the backend owns, and the layout conversion
selects the split on the way out and scatters it back into the native tensor
on the way in, so a write to one split leaves the others untouched.

A row's `requires` names the architectural facts the component needs. Each is
evaluated twice from the one row — at load against the registry entry where
the entry can decide it, and at run against the loaded module tree (the site
resolver's probes) — so `validate` refuses `routed_output` on a dense model
offline and the run refuses the same document if it arrives unvalidated:

| predicate | means | decided at load by | decided at run by |
|---|---|---|---|
| `moe` | the block is a sparse-MoE block (router + fused experts) | the entry's `num_experts` | the MLP has `gate` and `experts` children |
| `shared_expert` | the MoE block carries a shared expert | the entry's `shared_expert_intermediate_size` | the MLP has a `shared_expert` child |
| `grouped_mm` | the experts dispatch is the grouped kernel (the dispatch pin) | the entry's `experts_implementation`, when the entry was adapted from a *loaded* model's config (a load-time knob, not a config fact — a hand-declared entry carries none and decides nothing) | `experts_implementation == "grouped_mm"`, the last-line check |
| `split_qkv` | the mixer's q, k and v are addressable — separate projections, or a fused projection whose row declares the logical blocks (GPT-2's `c_attn`) | the per-family tap table has met the entry's `family` (`overrides`); `None` otherwise | the row's address for the family, else measured: the mixer carries separate projections |
| `gated_attention` | the q-projection emits `[q \| gate]` per head (Qwen3.5/3.6) | the `attention_gate` row has an address for the entry's `family`; `None` for a family the table has not met | the row's address for the family, else measured: `q_proj.out_features == 2·H·d` |

Every refusal about a component, a mechanism or a selector carries a **reason
code** next to its rule number — `ProtocolError.reason`, a closed vocabulary
(`errors.REASON_CODES`) so a caller can branch on *what kind of fact was
missing* without parsing prose. The same vocabulary names why a result cell is
`unavailable` (sec. 4.1), so one set of names covers a fact whether it surfaced
as a refusal or as a value. Seven are emitted today; the eighth is declared so
the code that will emit it cannot invent a spelling. A row whose emitter cell
begins "not yet emitted" is declared and not yet emitted — the census guard
holds that marker to the code:

| reason | emitted by | meaning |
|---|---|---|
| `unsupported_mechanism` | rule 4 (`validate`) and the executor's write policy, from the row's `writes` cell | a write names a mechanism the component's policy refuses — a read-only component, or arithmetic on a `swap`-only one |
| `component_unavailable` | the stream check (load and run), the row's `requires` (load and run), the engines' interior refusals, `component_shape` on an all-MoE tower's `mlp_activation`, an `expert` selector on a component without that axis, and the executor's fire-count check after each forward — a write whose module the forward never called, or called twice (sec. 4, "Fires") | the model, the layer or the engine has no such tensor — in this forward included |
| `alignment_missing` | position resolution when a `variable` / `column` value occurs nowhere in the row's text (an `unavailable` result cell for a read, sec. 4.1; a refusal under a write), the executor's answer-form pre-flight (a bare `token_form` over an answer the table carries space-prefixed, sec. 2.10), and a declared `alignment` the pair resolves as `absent` (sec. 2.3) | a cross-input operand has no alignment to pair on |
| `alignment_ambiguous` | position resolution when the value occurs several times (an `unavailable` cell for a read, a refusal under a write), and a declared `alignment` the pair resolves as `ambiguous` (sec. 2.3) | more than one alignment fits and none was named |
| `empty_selector` | the executor's `expert:` face when the router sent that expert no token at the addressed positions (an `unavailable` result cell, sec. 4.1), rule 15's entry selection when no bundle entry matches or none is selected (a load error), and a span (sec. 2.3) that resolves to no token on a row (a refusal, `protocol/spans.py`) | a selector resolved to nothing |
| `chat_template_missing` | the chat frame's encode (`neural/shared/framing.py`): a document declares `segments.frame: chat` and the tokenizer carries no chat template to render it with (a refusal before any forward, sec. 2.2.1) | the model's tokenizer cannot render the frame the document declares |
| `ragged_write_unsupported` | rule 19's `refuse` path (`neural/shared/executor_base.py`, `_ragged_write_error`): the pre-forward width check and the landing path, for a write whose rows address different numbers of positions under no `ragged` policy or `refuse`; a ragged operand paired into a write under `refuse`; and, under a landing policy, a ragged operand whose row widths disagree with the write's (sec. 2.8, sec. 5 rule 19) | a ragged write, or a ragged operand paired into a write, has no aligned shape to land on |
| `overlapping_write_unproven` | not yet emitted — rules 8 / 9 territory; the fire-count observable (sec. 4, "Fires") is what the executor checks today | two writes overlap at one address and their order is not proven |

### 2.5 `featurizers`

`featurize(x) → (f, err)`; `inverse(f, err) → x̂`; both defined per kind. The
how-to layer — one page per kind a practitioner runs as a method, every field
with its legality, the shipped templates — is [`docs/methods/`](methods/README.md);
this section is the normative text those pages point back at.

<!-- generated: begin featurizer-kind-table -->

| kind | featurize | param slots | authored fields |
|---|---|---|---|
| `identity` (default) | `(x, 0)` | — | — |
| `subspace` | `(Qᵀx, 0)` | `weight` | `k`, `parametrization` ∈ `cayley` \| `matrix_exp` \| `stiefel`, `init` (on a fit), `seed` (on a fit) |
| `pca` | `(Pᵀx, 0)` | `weight` | `k` |
| `sae` | `(enc(x), x − dec(enc(x)))` | `enc`, `dec`, `b_enc`, `b_dec` | — |
| `standardize` | `((x−μ)/σ, 0)` | `mu`, `sigma` | — |
| `gate` | `(m⊙x, (1−m)⊙x)`, `m` the soft mask in training and the hard mask at eval, by `parametrization` (the table below) | `theta` | `parametrization` ∈ `sigmoid` \| `clamp` \| `hard_concrete` \| `budget`, `group` ∈ `head` \| `expert_neuron` \| `site`, `axis` ∈ `position`, `init` (on a fit), `temperature` (under `hard_concrete`), `stretch` (under `hard_concrete`), `dead` ∈ `freeze_after` \| `leak` (on a fit), `top_k` (with `file_path`), `k_schedule` (under `budget`, on a fit), `stop_grad_shift` (under `budget`, on a fit), `pool` (under `budget` to fit; any map with `file_path`) |

<!-- generated: end featurizer-kind-table -->

- **Widths are derived** — never authored: from (model, site), and for a
  position gate (`axis`, below) from the **addressed window** instead, so one
  such gate spans sites of different feature widths while the chain's feature
  width flows past it unchanged. Only choices are authored: the fields column
  above, each legal where its note says (`schema.FEATURIZER_FIELD_CONDITIONS`),
  plus `file_path`, `entry`, `dtype` and `description` on every kind.
- **Params are auto-declared** per kind, named `<featurizer>.<slot>`.
- **A fit reports on itself** in `fit_diagnostics.json` beside its bundle, per
  kind: a `gate` its unit counts (the `group` bullets below); a `subspace` its
  saved rotation's `orthonormality_deviation` (`max|QᵀQ − I|`) and
  `within_tolerance` against the bar a later `init` load applies — so a
  rotation a later document cannot name as a start is flagged in the run that
  produced it, not the run refused it.
- **Composition**: a `featurizer` reference may be a list `["rot", "gate"]`,
  applied left-to-right with a per-stage `err` list. A stage's width is
  derived from its **position** in the chain (a gate after a `k=3` rotation is
  3-wide — a *position* gate is its window wide wherever it sits, the chain's
  feature width flowing past it), which makes two shapes load errors (rule
  12):
  - an **empty** list. `identity` is a declarable kind, so a document meaning
    "no featurizer" omits the key or names one; `[]` is a third spelling that
    no canonical form records.
  - a **repeated** stage. A name appearing twice has two positions and one
    derived width — the engine sizes a stage by walking the chain to that
    name, so the second application would be built at the first's input width
    — and both stages would share one parameter slot (for a position gate,
    whose two applications are one window wide, the slot alone is the reason).
    Two applications of one map are two declared featurizers.
- **One name, several sites**: the name a read or write puts in `featurizer`
  is a *parameter set*, and a name reached from several addresses is one — one
  θ (one rotation) read and written at every site it is named at, the
  gradients of every site accumulating on it. A budget mask tied across layers
  and a DAS rotation tied across spans are this spelling: declare the
  featurizer once and name it from each site's read and write; there is no
  tying field. `train.params` and `save` name it once (the bundle stamps the
  site its `save` entry names, and the loader expects a bundle's site only
  when the name is used at exactly one); rule 4 holds every site the name
  reaches to one width — for a loaded featurizer too, at load and not at the
  build after its weights are read — and rule 23 holds a grouped gate to one
  group map; a chain names it at most once (the repeated stage above). Two
  sites meant to carry two masks are two declared featurizers.
- **Error-term contract**: `err` and unselected dims always come from the
  pre-write value at the address — so a zero write ablates only the feature
  contribution, and a `dims` write is a subspace swap.
- **`seed`** (optional, `subspace` only): the draw its **initial** rotation
  comes from. Absent, it is the document's seed (`train.seed`, or 0 with no
  fit), so an existing document is unchanged and its canonical form does not
  grow a field. Illegal together with `file_path`: a loaded featurizer's
  weights are its bytes and it draws nothing.
  - This is what makes an *untrained* `subspace` a first-class **random rank-k
    basis**: `Q = qr(randn(d, k))` at that seed, orthonormal by construction.
    With `seed: {"sweep": [0, 1, 2]}` and no `train` block, one document is the
    matched-k random-subspace control — three draws at one cell, scored exactly
    as the fit was, in one model load. Without an authorable seed the control
    needed a `train` section it had nothing to train, which is why every study
    that wanted it built the draws by hand.
    See `configs/protocols/random_subspace_control.json`.
- **`init`** (optional, `subspace` and `gate`): where the fit **starts**. On a
  `subspace`, `{"file_path": <basis bundle>, "entry": …}` — the first `k` columns of a saved
  `(d, m)` basis, `m ≥ k`, typically a PCA basis `causalab.analysis.fit_pca`
  wrote over a harvest at the same site. `entry` is optional and has the
  semantics of the featurizer's own `entry`. Illegal together with
  `file_path`: a loaded featurizer draws nothing and trains nothing, so it has
  no start to set. Legal together with `seed`.
  - **What is checked.** At load, the basis's `ArtifactIdentity` must match
    the fit on model key, revision, model dtype, quantization and the site
    record, and must carry a `produced_by`; its own `k`, `dtype` and
    trained-on data are *not* compared — a wider basis seeds by its first `k`
    components, and a PCA over one corpus may start a fit on another. When
    the basis is a swept bundle and no `entry` is authored, the file-level
    stamp is what is checked — a field its entries differ on (a swept site)
    then needs an authored `entry` to be checkable, and the load says so. At
    build, the basis must be as wide as the site, hold at least `k` columns,
    and its first `k` columns must be orthonormal (they become the
    parametrization's base verbatim); the selected entry must carry its own
    `produced_by`. A mismatch refuses naming the field.
  - **What it means for the fit.** The `k` columns `P` are completed to a
    full orthonormal `d × d` basis `Q₀` by QR of `[P | randn(d, d−k)]`
    drawn at the document seed (or the featurizer's own `seed`), and the
    declared parametrization is applied *on top*: the trained subspace is
    the first `k` columns of `Q₀ · R(A)` with `A` starting at the identity
    map. So before any step the featurized read is exactly `Pᵀx`, the
    trainable surface is the same one an unseeded fit has, and a document
    without `init` is unchanged to the bit. The saved bundle's identity (sec.
    8) records `init_produced_by`, `init_trained_on` (the data ref the basis
    was fitted over — a harvested read stamps the dataset it read, and a
    `fit_pca` output inherits it), the component indices taken
    (`init_components`, `[0, k)`) and a digest of the seeding matrix
    (`init_digest`); the basis's bytes enter the
    canonical form as `init.content_digest`, exactly as a loaded featurizer's
    do. See `configs/protocols/das_pca_init.json`.
  - **On a `gate`**, three spellings and exactly one of them. `{"fill": p}`
    starts every unit at **mask value** `p ∈ [0, 1]` — `θ = logit(p)` under
    `sigmoid` (which therefore refuses the endpoints at build; a start at a
    pole is a `clamp` start), `θ = p` under `clamp` — the one number that means
    the same under both maps. It is **sweepable**: whether the start decides
    the mask is a real question, and `{"sweep": [0.5, 0.99]}` asks it in one
    document. `run.py`'s `--init_active 0.99`, "everything patched but a
    little", is `{"fill": 0.99}`. `{"file_path": <gate bundle>, "entry": …}`
    starts from a saved `theta` **verbatim** — an earlier fit, a trajectory
    checkpoint (sec. 2.12), a hand-built prior — which must be a theta *of this
    gate*: same `group` and map, same `parametrization`, the unit count this
    layout has, checked at load against the header and at build against the
    selected entry exactly as a loaded gate is; it must carry its own
    `produced_by`. The bundle a fit from a saved start writes records
    `init_produced_by`, `init_trained_on` and `init_digest` (no
    `init_components`: a gate's start is the whole theta), and the start's
    bytes enter the canonical form as `init.content_digest`. A `fill` enters the
    canonical form as authored and is recorded in `fit_diagnostics.json` as
    `init_fill`. Absent, the gate starts at the midpoint mask (`θ = 0`, or
    `½`), as it always has — no digest moves.
  - **`{"from_scores": {…}}`** starts a gate from a per-unit **score table** —
    a saved metric table (a JSON list of row objects, one per unit) such as
    `causalab.analysis.head_stats` writes or an attribution scan saves. A
    position gate's units are token positions, which no shipped *attribution*
    scan writes a table over (`head_stats` groups on two columns and writes
    `layer` / `head`), so its table is a prior position-gate fit's `rank.json`
    (`unit` / `theta`, with `axis`; §9 — the θ index is the unit, so the route
    is exact) or `pca_by_position`'s `spectrum` (one row per `position` of the
    harvested window per `pc`, `where` picking the component). A unit is an
    offset into the window, and neither table's score column is the `value`
    default below. Keys: `file_path` (the table); `unit` (the column holding
    each row's unit index — a list of columns for a two-axis theta such as
    `expert_neuron`'s; default `"unit"`); `value` (the score column; default
    `"value"`); `where` (an equality filter, `{"layer": 15}`, that picks this
    gate's rows out of a table over several sites); and exactly one of
    **`keep`** — the top-`keep` units by score start on the kept pole of the
    gate's map and the rest on the dropped pole (`(0, 1)` under `clamp`, one
    unit either side of the hard threshold under `sigmoid` and `hard_concrete` —
    the same poles `analysis.random_mask` writes its controls on), so an
    *untrained* gate with `keep` is the attribution- or magnitude-pruning
    baseline in one document — or **`scale`** — `θ = midpoint + scale · z`, `z`
    the population z-score of the values (clipped to `[0, 1]` under `clamp`), an
    attribution-initialised fit. `keep` and `scale` are sweepable; `unit` and
    `value` are materialized to their defaults in the canonical form. The
    table's bytes enter the canonical form as `init.from_scores.content_digest`
    and the bundle's identity as `init_digest` (no `init_produced_by`: a table
    has no `ArtifactIdentity`); `fit_diagnostics.json` records
    `init_from_scores` (path, mode, the units put on the kept pole). Rule 32
    refuses a `keep` above the unit count as the document canonicalizes and, at
    build, a table that does not name every unit of the gate exactly once after
    `where`.
- **`group`** (optional, `gate` only): the unit one `theta` entry covers. A
  closed vocabulary of three values, each naming the axis of the site's declared
  shape its coordinate→group map is derived over:

  | group | one θ per | axis the site must have |
  |---|---|---|
  | `head` | attention head — the map is `(heads, head_dim)` | `head`: a head-major component (`attention_premix`, `delta_premix`) |
  | `expert_neuron` | (expert, neuron) of the routed expert table — the map is `(num_experts, d_expert)` | `topk`: the routed-slot axis, which `expert_activation` and `expert_neuron_output` lay out as one expert's neurons per slot |
  | `site` | the whole site — the map is `(1, width)`, one parameter, so an MLP block or the embedding is a single unit (the node a circuit benchmark scores it as) | `feature`: any feature-space component; a site that names a `head` is one unit over that head's slice, which is still what it claims |

  **Absent, nothing changes.** A gate with no `group` has one parameter per
  coordinate — the gate it has always been — and the canonical form records
  only what was authored, so a document that names no `group` is byte for
  byte the document it was before the field existed (same params, same
  digest), and a per-coordinate bundle fitted before the field existed still
  loads under it: `group` and `group_map` enter the identity check only when
  the document authors a group. There is no literal spelling of the default —
  `coordinate` is not a value, and is refused as an unknown one.

  `head` gives a gate one
  parameter per **head** of a head-major component: every coordinate of head
  `h` receives `σ(θ_h/T)` in training and `θ_h > 0` in eval, so the fitted mask
  is a set of heads. The coordinate→head map — `(heads, head_dim)`, the same
  slices a `head` field selects — is derived from the component's shape and
  never authored; the canonical form records `params.theta` as `[heads]`. Both
  `attention_premix` (a full-attention layer's o-projection input, query-head
  space) and `delta_premix` (a Gated DeltaNet layer's out-projection input,
  value-head space) are head-major, so one declaration masks query heads on
  the one family and value heads on the other. Consequences:
  - an `l1` term over a grouped gate is the mean of `σ(θ/T)` over its units
    (heads here), so the penalty is the selected-unit count;
    `fit_diagnostics.json` counts units (`hard_mask_size`, `decisive_fraction`)
    and records `groups`, the unit count.
  - a group the site cannot honour is a **load error** (rule 23, group
    legality), decided with no model loaded: the component has no such axis
    (`group: "head"` on `block_output`, named in the refusal), the site
    already selects a single member of it (`head: 3` under `group: "head"` —
    H groups over one head is one group), the gate is not the **first stage**
    of its chain (a grouped gate acts on the component's own coordinates —
    after any other stage, a `subspace`, an `sae` or a per-coordinate
    `standardize` alike, coordinate 5 no longer names a unit of the component),
    or two sites sharing the featurizer would lay its units out differently. A loaded (`file_path`) grouped gate is checked
    the same way.
  - the saved bundle stamps `group` and `group_map` into its `ArtifactIdentity`
    (sec. 8). It reloads only through a gate declaring the same `group`, at a
    site whose derived map is the same — same component, layer, head count and
    head width — and anything else refuses naming the field that disagrees
    (rule 15). A per-coordinate document is refused against a grouped bundle,
    and a grouped document against a per-coordinate one.
  See `configs/protocols/dbm_head.json` (fit) and `dbm_head_apply.json`
  (replay).

  `expert_neuron` gives a gate on `expert_activation` or
  `expert_neuron_output` one parameter per
  **(expert, neuron)** of the routed interior: `num_experts × d_expert` in
  all, whatever `top_k` experts a token activates. The site is token-major —
  `top_k` slots of `d_expert`, slot *k* the *k*-th ranked expert — and a
  slot's coordinates use the parameters of the expert filling it, looked up
  through `expert_idx` at the same rows and positions; an expert a token did
  not activate has no slot there, so its parameters leave that token alone.
  The map is `(num_experts, d_expert)`, derived from the model and never
  authored; the canonical form records `params.theta` as `[num_experts,
  d_expert]`, and the saved `theta` is that two-dimensional table. Legal on
  `expert_activation` and `expert_neuron_output`. Other components fail
  rule 23 with the component name. On `shared_expert_activation` the
  ordinary per-coordinate gate is already one parameter per shared-expert
  neuron, with separate keys. Consequences:
  - the `l1` term is the mean of `σ(θ/T)` over the whole table and
    `fit_diagnostics.json` counts (expert, neuron) units.
  - **swap under routing mismatch.** A write through an expert-keyed gate
    joins its tensor operand to the written slots **by expert id**, not slot
    for slot: for a base slot holding expert `e`, the source is the operand's
    (counterfactual) activation of `e` at the same token when `e` is active
    there; otherwise the slot has no source and keeps its base value (the gate
    mixes base with base). The operand must therefore be a read at the routed
    interior — it carries its expert ids — over the same positions as the
    write; `dims` is refused through such a gate. The executor records, per
    write, layer and example, how many base slots had an expert inactive on
    the operand's side, out of the slots addressed, and writes the table as
    **`routing_mismatch.json`** beside `fit_diagnostics.json` (columns `point`,
    `coords`, `write`, `layer`, `example`, `mismatched`, `slots`) — the mismatch
    rate a report divides out. An apply document writes the same table.
  - the bundle stamps `group: expert_neuron` and `group_map: [num_experts,
    d_expert]`; the rule-15 checks above apply unchanged.
  See `configs/protocols/dbm_expert_neuron.json` (fit: an expert-keyed gate on
  `expert_activation` beside a plain gate on `shared_expert_activation`, both
  trained) and `dbm_expert_neuron_apply.json` (replay).
- **`parametrization`** (optional, `gate`): how `theta` maps to the mask. One
  field with one meaning across kinds — how the stored parameter maps to the
  object the fit is about — and an enum per kind: a `subspace` names its
  rotation map above, a gate one of these:

<!-- generated: begin gate-map-table -->

| parametrization | soft mask (train) | after every optimizer step | hard mask (eval, apply) | mask penalty (`train.objective`) | `anneal` on `theta.temperature` | default start |
|---|---|---|---|---|---|---|
| `sigmoid` (absent) | `σ(θ / T)` | nothing | `θ > 0` | **`l1`** = `mean σ(θ/T)`; `l0` is **refused** (rule 4) | legal | `θ = 0`, i.e. `m = ½` |
| `clamp` | `θ` itself | `θ ← clip(θ, 0, 1)` | `θ > ½` (`round`) | **`l1`** = `mean θ`; `l0` is **refused** (rule 4) | **refused** (rule 4): a clamp gate's mask is θ itself, projected into [0, 1] after every step — it has no temperature to anneal | `θ = ½` |
| `hard_concrete` | **sampled**, once per optimizer step: `u ~ U(0,1)`, `s = σ((log u − log(1−u) + θ)/β)`, then `clip(s·(ζ−γ)+γ, 0, 1)` | nothing | `clip(σ(θ)·(ζ−γ)+γ, 0, 1) > ½`, i.e. `θ > logit((½−γ)/(ζ−γ))` — exactly `θ > 0` at the default stretch | **`l0`** = `mean σ(θ − β·log(−γ/ζ))`, the expected kept fraction of the sampled mask; `l1` is **refused** (rule 4) | legal | `θ = 0`, i.e. `m = ½` |
| `budget` | `σ(θ + c_k)` with the step's budget `k` drawn from `k_schedule` and the scalar `c_k` solved so `Σ m = k` exactly | nothing | the **`top_k`** largest `θ` — `k_schedule.eval` inside the fit, the document's `top_k` on a loaded gate; there is no threshold | none — `l1` and `l0` are **refused** (rule 4): the mask's sum *is* the budget | **refused** (rule 4): a budget gate's mask is σ(θ + c_k) with the shift solved per step — it has no temperature to anneal; its sharpness is the budget's | `θ = 0` (`fill` ½) |

<!-- generated: end gate-map-table -->

  **`parametrization` may be a mapping**,
  `{"forward": "hard", "backward": <map>}`, the forward/backward split of
  a soft-versus-hard ablation grid: the training **forward** uses the map's training mask
  **thresholded at ½** — `θ > 0` under `sigmoid`, `θ > ½` under `clamp`, the
  step's sample at ½ under `hard_concrete`, `θ + c_k > 0` under `budget` — and
  the **backward** sees the map's own gradient, the straight-through idiom
  `hard + (soft − soft.detach())`. The gap it closes is real: a gate whose σ
  never leaves the relaxed band (no unit outside `[0.1, 0.9]`) can still score
  1.000 through its hard eval mask off a coin flip — `configs/protocols/dbm.json`'s
  description says why — and under the split that is a score the training loss
  also paid for. The split
  does not by itself make θ separate (`decisive_fraction` then reports the
  backward map's confidence), so the presets' *gate on `decisive_fraction`
  before believing a number* stands, and the preset is untouched. The eval-mode
  split, the `l1`/`l0` term, `anneal`, `dead` and every other field are the
  map's (the `backward`'s), unchanged — with five consequences worth knowing.
  Under `backward: budget` the forward mask `[θ + c_k > 0]` sums to the budget
  only approximately (`Σ σ = k` constrains the mass, not the count above ½), so
  that pairing has neither exact density in the forward nor a sparsity term.
  Under a sec. 2.11 `constraint` on an `l1` term the dual ascent holds the
  *soft* mass oscillating across the target (the undershoot-and-return cycle
  sec. 2.11 describes), and an equality's cheapest resting state is a plateau of
  mid-range σ — exactly the population near ½ — so the forward flips whole units
  on and off as their σ crosses ½: its density holds the target on average and
  jitters in unit steps; the pairing is legal (refusing it would take the
  paper-faithful comparison away), and the word for its forward density is
  *unstably*, not *approximately*. Under `backward: sigmoid` a
  `gate.theta.temperature` anneal changes neither the training forward nor the
  eval split — `σ(θ/T) > ½ ⟺ θ > 0` for every `T` — only the gradient the split
  hands back, `σ'(θ/T)/T`, and that two ways: for a confident unit an anneal
  toward zero drives it to zero (the split has already closed the train/eval gap
  the schedule exists to close), while at `θ = 0` it is `1/(4T)` and grows as
  `T` falls — `0.25 → 25` over `dbm.json`'s `1.0 → 0.01`, the crossover near
  `|θ| ≈ 0.06` — so the schedule freezes the decided units and sharpens the
  undecided. Composed with the `constraint` consequence above, whose resting
  population is that mid-range band, the jitter gets louder as `T` falls rather
  than freezing; and a constraint's target then bounds the split's *forward*
  density directly — `mean σ(θ/T) = t` forces `#{σ > ½}/W ≤ 2t` by Markov,
  reaching `t` only once the gate has committed, since `hard_mask_size` *is*
  that count under `sigmoid` and `clamp`; the pointwise limit
  `σ(θ/T) → 1[θ > 0]` says more only where `|θ| ≫ T`, and a θ scaled with `T`
  holds the target at every temperature with an empty forward (sec. 2.11's
  relaxed-mask paragraph — the same trajectory as the mid-range plateau above) —
  under `sigmoid` only: `backward: hard_concrete`'s limit as `β → 0` is `σ(θ)`,
  the stretched sample clipping to `Bernoulli(σ(θ))` exactly, so there the split
  is a no-op in the limit and `expected_l0 → σ(θ)` is that Bernoulli's keep
  probability — the same composition holds the forward's density *in
  expectation* over a per-step redraw, while eval stays the deterministic
  `θ > 0` (sec. 2.11's "buys no separation", read on the forward side). A fit of
  confident units freezes past the anneal's `frac` with nothing in the record
  saying so, `decisive_fraction` driven to 1 by the schedule alone; `dead.leak`
  (whose row below is spelled in this very quantity, and whose `ε` does not
  depend on `T`) is the remedy. Under every map the default init sits on the
  split's threshold — the midpoint mask, `σ(0) = ½` or `½` itself (`init` above)
  — and the split thresholds strictly above it, so a split fit's *first*
  training forward is the empty mask under `sigmoid` and `clamp`, and under
  `budget` all-or-nothing whatever `k` (`budget_shift` gives a uniform θ one
  value, `k/W`, so the count above ½ is `0` or `W`, never `k`, until θ has
  spread across `−c_k`); only `hard_concrete`'s sample escapes it. A warm-up
  transient, not a defect — `∂L/∂θ` is per-unit and largest at the midpoint, so
  the first update moves off it (the fit test asserts θ leaves zero) — with
  `init.fill` / `init.from_scores` as the lever. And a *loaded* gate with
  `top_k` that trains behind another stage thresholds the map at ½ in the
  training forward while its eval split cuts the `top_k` ranking — two different
  masks (a mismatch the string form shares, made confident here). `forward` is
  not a site coordinate (`docs/workflow_protocol.md` sec. 5, rule 16 — the
  site-equivalence rule): a soft-forward and a hard-forward fit at one site are
  one shape and compare equivalent — such a grid compares exactly those two,
  and as a coordinate the pair would be refused, liftable only by
  `non_equivalence`; the split decides which loss produced θ, not what θ is,
  which is what the bundle's provenance stamp records. `fit_diagnostics.json`
  records `forward` beside `parametrization`, because two of its numbers change
  meaning under the split: `decisive_fraction` (off the soft mask) reports the
  backward map's confidence, the forward being 0/1 by construction, and
  `hard_mask_size` is the eval split — equal to the forward's count above ½
  under `sigmoid` and `clamp`, not under `hard_concrete` (the forward is the
  step's sample) or `budget` (the eval cut). A budget pool's members agree on
  `forward` as they do on the map (rule 4). The mapping form is not swept
  (`{"sweep": ["sigmoid", {"forward": …}]}` is refused naming the rule): an
  ablation grid is one document per forward/backward pair. The bundle stamps
  `forward` as provenance only, since θ reads out through the same hard split
  either way. `forward` has the one value `hard`: `soft` *is* the string form
  and `sampled` *is* `hard_concrete`, so each is refused rather than spelled
  twice (sec. 7); the string form stays and means forward = backward. Gates
  only.

  `clamp` is the DCM relaxation: the mask is the
  parameter, held on `[0, 1]` by projection rather than by a squashing map, so
  the sparsity gradient is a constant `weight / units` per unit rather than
  `σ'(θ/T)/T`, and there is nothing to anneal — binarization is the eval-mode
  `round`. **Absent, nothing changes**: `sigmoid` is the gate as it always was
  and has no spelling in the canonical form, so no existing document's digest
  moves. Consequences:
  - `fit_diagnostics.json` reads `hard_mask_size` and `decisive_fraction`
    through the gate's own map (`θ > ½` and `|θ − ½|` under `clamp`) and
    records `parametrization` beside them.
  - the saved bundle stamps its **effective** parametrization (`sigmoid` when
    unauthored) into its `ArtifactIdentity` (sec. 8). A bundle reloads only
    through a gate declaring the same map — the two hard masks differ, so a
    mask under one is not a mask under the other — and a mismatch refuses
    naming both (rule 15 at load, again at build). A bundle fitted before the
    field existed carries no stamp and is read as `sigmoid` by both checks, so
    it still loads under a document that declares nothing.
  - `analysis.random_mask` draws its size-matched control at the bundle's own
    threshold and writes the control under the same map.

  `hard_concrete` is the stochastic L0 relaxation of Louizos, Welling &
  Kingma 2018 ([arXiv 1712.01312](https://arxiv.org/abs/1712.01312)), the map
  NeuroSurgeon's `HardConcrete*` layers implement. Its two constants are gate
  fields, legal only under this map: **`temperature`** (β, default `2/3`) and
  **`stretch`** (`[γ, ζ]`, default `[-0.1, 1.1]`, with `γ < 0 < 1 < ζ`);
  unauthored they are not materialized (the `group` precedent), so write
  neither or both. `temperature` is sweepable; `stretch` is not — the hard
  split is derived from it and the bundle stamps it, so it is a constant of
  the relaxation the whole fit is read through, not an axis. What the
  sampling buys: the mask a training forward uses is **drawn once per
  optimizer step** from the fit's own generator (seeded by `train.seed`, never
  the global RNG, so a cohort member's draws are its own and a fit reproduces
  across devices) and **shared by every read and write the gate sits on** in
  that step — the interchange `m·v_cf + (1−m)·v_base` is one mask, a
  partition, not two draws — then resampled at the next. At the default
  β = 2/3 and θ = 0 about a third of the draws land on a pole and the rest are
  fractional, so the mask is not near-binary; what a rotation trained *beside*
  the gate cannot do is fit a **fixed** fractional interpolation, because the
  interpolation changes every step, whatever the anneal does. Consequences:
  - the mask penalty is keyed on the map (rule 4): **`l0`** — the expected
    kept fraction `mean σ(θ − β·log(−γ/ζ))`, the quantity the sampled mask has
    an expectation of — is legal only under `hard_concrete`, and `l1` is
    refused there (it would penalize the deterministic mask the forward never
    uses); under the deterministic maps `l0` is refused, because the relaxed
    mask is already the kept probability and its mean is `l1` — one
    computation gets one spelling.
  - `init.fill p` means what it means under every map — *the mask value every
    unit starts at*: θ inverts the stretch, `θ = logit((p−γ)/(ζ−γ))`, so the
    deterministic mask starts at exactly `p` (a plain `logit(p)` would start
    `fill: 0.99` fully clipped at 1). `fill: ½` is `θ = 0`, exactly on the
    split, as under `sigmoid`.
  - the eval threshold `logit((½−γ)/(ζ−γ))` is computed in one place
    (`protocol.schema.hard_concrete_threshold`) for the gate and for
    `analysis.random_mask`, and is **exactly** `0` at any symmetric stretch —
    derived in floating point it is ≈ `−4e−16`, which would count a θ of
    exactly 0 as kept.
  - `fit_diagnostics.json` reads `hard_mask_size` and `decisive_fraction`
    through the deterministic mask and records `stretch` beside
    `parametrization`. `decisive_fraction` changes meaning here: the stretch
    scales `|m − ½|` by `ζ − γ`, so a unit is decisive at a smaller `|θ|` and
    saturates at exactly 0 or 1 once `|θ| ≳ logit((1−γ)/(ζ−γ))`; and β does
    not enter the deterministic mask, so an anneal leaves the number alone
    where a `sigmoid` anneal drives it to 1. Read it per map, not across maps.
  - the bundle stamps `stretch` into its `ArtifactIdentity` (sec. 8) and
    `analysis.random_mask` draws at that threshold. A document that **authors**
    a stretch is asked of the bundle (like `group`); a bundle fitted at a
    non-default stretch is refused by a document authoring none (rule 15) —
    the two would split θ at different thresholds — so an apply over such a fit
    re-authors its stretch, as it re-authors `dtype`. β is not compared: it
    does not enter the split.
  - authoring `temperature` beside an `anneal` on the same gate's
    `theta.temperature` is refused (rule 4): the schedule's start replaces the
    authored value before the first forward, so one of the two would silently
    win. Write one or the other.

  `budget` is a sparsity-loss-free mask: `θ` is learned as a **ranking**, not
  a set. Each optimizer step draws a budget `k` from the
  gate's **`k_schedule`** — `{"kind": "fixed", "k": n}` (the same cut every
  step: the plain sigmoid-top-k mask), or `{"kind": "uniform" | "log_uniform",
  "low": a, "high": b}` (an integer in `[a, b]` per step; `log_uniform` needs
  `a ≥ 1` and is the default curriculum, spending as many steps between
  1 and 2 units as between 24 and 48) — from the fit's own generator, so the
  sequence of budgets is a function of `train.seed`. The training mask is
  `σ(θ + c_k)` with the one scalar `c_k` solved by bisection so that
  `Σ_i σ(θ_i + c_k) = k` exactly; the shift carries its implicit gradient
  (`∂c/∂θ_i = −σ'_i / Σ_j σ'_j`, so `Σ m = k` holds to first order under any
  update) unless **`stop_grad_shift`** is `true` — the `−c_k` ablation —
  where the shift enters as a constant. `k_schedule` is required on a fit
  (`eval` beside it — the cut the fit's own held-out pass scores and its
  `hard_mask_size` counts — defaults to `k` under `fixed` and is required under
  a sampled kind, since a sampled schedule implies no single number; `k` and
  `eval` are sweepable, the bounds are not) and refused on a loaded gate, which
  is read out through `top_k`. Consequences:
  - there is **no sparsity penalty and no temperature**: `l1` and `l0` on a
    budget gate are refused (rule 4) — the mask's sum is the schedule's — and
    so is an `anneal` on its `theta.temperature`; `temperature` and `stretch`
    are refused as under every non-sampling map.
  - there is **no threshold**: the eval-mode split is the `top_k` largest `θ`
    (ties toward the lower index, sec. 2.5 `top_k`), at `k_schedule.eval`
    inside the fit and at the document's `top_k` on a loaded gate. A budget
    bundle is refused by a document that names no `top_k`, and
    `analysis.random_mask` needs the same `top_k` to size its control. The
    whole readout of a budget fit is therefore a **`top_k` sweep** over one
    bundle (a `k = ⌊p·|H|⌉` grid), plus a `rank` table (sec. 2.12).
  - a budget gate cannot be a `control` signal (rule 4): its `hard_mask_size`
    is `k_schedule.eval`, fixed by the document, so nothing the controller
    moves can move it.
  - `fit_diagnostics.json` records `k_schedule`, `eval_k`, `stop_grad_shift`
    and `k` — the last step's budget — beside `parametrization`; a
    `trajectory` checkpoint carries `<gate>.k` for its step. `decisive_fraction`
    is read through `σ(θ + c)` at the last step's shift (the eval cut's shift
    before any step), so it says how far the ranking has separated *at that
    budget*, not across budgets.
  - **`k_schedule.of`** (optional, `patched` | `kept`, `K_SCHEDULE_OF`) says
    what every number in the schedule counts. Absent, `patched`: units that
    take the counterfactual — the gate's own count, unchanged, no field in the
    canonical form. Under `kept` the schedule counts the units left **clean**
    (`k`, `low`, `high` and `eval` alike) and the gate complements each number
    against its unit count exactly once, so `{"kind": "log_uniform", "low": 1,
    "high": N − 1, "of": "kept"}` is the log-uniform curriculum on the kept count
    (a log-uniform draw on the kept count is not log-uniform on the patched
    one). `top_k` is not a schedule number and always counts patched units;
    `fit_diagnostics.json` reports `eval_k` in patched units and records `of`.
  - **`pool`** (optional, a name; fitted or loaded): every budget gate
    authoring the same `pool` shares **one** budget — one `k` per optimizer
    step, one shift `c_k` solved over the *concatenation* of the members' θ,
    one ranking cut at one count — so a mask over units living at different
    sites (every head, every MLP block and the embedding, `2L + 1` gates) is
    one budget fit over `N` units, not `2L + 1` fits over their own budgets.
    The implicit gradient is the pool's (`w_i = σ'_i / Σ_pool σ'_j`), so a
    member's mask carries a gradient into its co-members' θ, which is the true
    derivative of the shared shift. Members are built together by the executor
    before any mask is computed, and must agree on `k_schedule` and
    `stop_grad_shift` (a fit) or `top_k` (loaded) — one value on every
    member, or one `axes` entry they all reference (sec. 3.2) — and be all
    fitted or all loaded (rule 4); a schedule number above the pool's unit
    count is refused at the build. On a loaded pool `top_k` is the **pooled**
    count and each member keeps the units whose pooled rank falls below it;
    `rank` rows carry `pool` and `pool_rank`. A loaded pool is a **pooled
    readout** and takes gates of *any* map (each with `file_path` and the
    one `top_k`): a ranking method's curve cuts the joint ranking of every
    unit of a model whether the units were fitted as one budget or as one
    L1-penalised sigmoid mask, and nothing is drawn or solved. A budget fit
    stamps `pool` and `pool_units` into its bundles; a stamped pool must match
    the document's and a pooled bundle is refused by an unpooled document
    (rule 15) — a member's θ is a ranking only relative to its co-members —
    while an unstamped bundle may join any readout pool; a stamped
    `pool_units` must equal the pool the document assembles. Members of a
    readout pool agree on `parametrization` too: the joint ranking is over raw
    θ, and a `clamp` θ in `[0, 1]` does not sit on a sigmoid logit's scale.
    `analysis.random_mask` refuses a bundle *fitted* in a pool (its stamp says
    so); a readout pool is a document fact no bundle carries, so there its
    `top_k` is per entry and a pooled control is drawn per member at that
    member's share of the cut (a pooled control is a follow-up). Never
    sweepable: a pool is a name.
  - **`axis`** (optional; the one value `position`): a **position gate**. θ has
    one entry per **addressed token position** and applies to every coordinate
    of that position — `x'[t] = m_t · x_cf[t] + (1 − m_t) · x_base[t]` over the
    window — a position mask; `hard_mask_size` counts positions. The
    feature is write-side (and read-as-swap-source): a *read* through a position
    gate cannot feed a metric — rule 31 refuses a metric over a read wider than
    one token and rule 4 refuses a one-position window, so the two accepted sets
    are disjoint; reduce the model output instead. Every read or write through
    it addresses a **fixed `span` `[a, b)`** of two or more positions (rule 4:
    the window's length is θ's width and must be the same on every row and at
    every use; `all`, `variable`, `column` windows and the anchored `segment` /
    `before` / `after` / `between` spans are as wide as the row, a static
    `indices` set or `union` / `intersection` of static members is a
    non-contiguous window (deferred, as `all` is), a `generated` window is
    clipped to the row's decode, a `scope`d span is sliced out of the anchor's
    run, a `relative_to` span is not placed by its anchor at all today (the
    resolver offsets an `index` only; §2.3's span offset is unimplemented, and
    an unimplemented placement may not size a θ), and an `index` or a span of
    one position is one scalar at one position — `group: site` on that write,
    not a mask *over* positions — so all of these are refused; spell an
    unanchored prompt-frame span (`atomic` composes, below), which is **counted
    forward from the content frame's start**: a right-anchored window
    (`{"span": [-4, -1]}`, the negative-index idiom
    `configs/protocols/multi_position_patch.json` addresses those positions
    with) has no unscoped spelling in §2.3 today, so `axis` does not reach it —
    a deferral, like `all`). The canonical form records the gate's `width` and
    `params.theta` as the window length, `[b − a]`, not the site's feature width
    — so a position gate may be used at sites of different feature widths (θ is
    the window either way; named from several layers' writes it is one mask
    over positions tied across them, §2.5 "one name, several sites"), and rule
    32's `init.from_scores.keep` bound counts positions. `fit_diagnostics.json`
    records `axis` beside `width`, so a reader knows its counts are over
    positions, and `rank` rows carry `axis` so a reader knows their `unit` is a
    position (§9). In a chain with a feature gate the two masks multiply — the
    per-position neuron mask as an outer product, tied across positions instead
    of one gate per position. Either order composes while the feature gate is
    per-coordinate (`["posgate", "gate"]`); a **grouped** feature gate must come
    first (`["gate", "posgate"]` — positions ⊗ heads), since rule
    23 holds a grouped gate to the head of its chain. `axis` takes no `group` (a
    position gate is already one scalar per position over the whole width — what
    `group: site` would say of a feature gate) and no `pool` (a budget over
    positions is not written down); it combines with every `parametrization`,
    `init`, `dead`, `k_schedule` and `top_k`, all of which see θ as units. A
    saved position gate stamps `axis` into its identity and reloads
    only under a document that spells it (both directions). As with `group`, the
    default has no literal spelling — `feature` is refused as an unknown value —
    so no unauthored document's digest moves. Never sweepable: `axis` decides
    θ's shape, and a document has one. `atomic: true` on the window is inert: an
    unanchored fixed `span` resolves the same either way (`spans.resolve_span`
    takes the plain anchor), is one address either way, and is `b − a` long
    either way, so both window twins size it and rule 4 accepts it — θ ranks the
    positions of the one address rather than splitting it into several, `atomic`
    being rule 8's write cardinality and §2.3's alignment run, not the feature
    space. Rule 27's "atomic needs two or more" is rule 4's own bound, so the
    two cannot disagree about a window.
  - `init.fill p` is `θ = logit(p)` as under `sigmoid` (a zero initial `θ`
    is `fill: 0.5`); the stamped `parametrization` is `budget` and a bundle
    reloads only through a gate declaring it.

  See `configs/protocols/dbm_head.json` for the default and the DCM presets
  for `clamp`.
- **`dead`** (optional, `gate` only, on a gate in `train.params`): what the fit
  does about a unit whose hard mask has closed. Exactly one of two rules:

  | rule | what happens | where | eval-mode hard mask |
  |---|---|---|---|
  | `{"freeze_after": n}` | a unit hard-off (`θ ≤` the map's threshold) for `n` consecutive optimizer steps is **frozen**: its `θ` is photographed and restored after every later step, so the optimizer's momentum cannot reopen it and a pruned unit stays pruned | the post-step projection, beside `clamp`'s clip | unchanged |
  | `{"leak": ε}`, `0 < ε < 1` | the training mask's **derivative** is floored: the forward value is the map's own `m`, the backward sees `∂m/∂θ + ε` (the leaky-ReLU idiom, `m + ε·(θ − θ.detach())`) — so a unit whose map saturated at the zero pole (`σ'(θ/T) ≈ 0`, a concrete sample clipped to 0) still receives `ε·∂L/∂m` and can come back | the training backward only | unchanged |

  `freeze_after` is the original DCM behaviour (a dropped head stays dropped,
  which makes a controller-driven sweep a *nested* sequence by construction);
  `leak` is the opposite answer ("keep gradients alive on dead masks so they
  can reawaken"), and the two contradict each other on one gate, so authoring
  both is a parse refusal. It is a gradient leak and not a value floor on
  purpose: `ε + (1 − ε)·m` would leave `∂/∂θ` scaled by the same vanishing
  `σ'`, and would route `ε` of the counterfactual through a unit the eval
  mask drops — the fractional-mask co-adaptation a hard mask exists to
  prevent. Under `clamp` a leak is legal and nearly inert — `∂m/∂θ` is
  already 1 there and the projection, not a saturating map, is what pins `θ`
  at 0 — so it is allowed for the comparison and documented as such rather
  than refused. Neither value is sweepable: a
  sweep over how a unit dies is a sweep over the training procedure, not
  over an axis whose points are comparable. A `dead` rule on a gate loaded
  from `file_path` is refused at parse, and on a gate outside `train.params`
  by rule 4 — no step would ever apply it. **Absent, nothing changes**: no
  spelling enters the canonical form, no digest moves, and the bundle's
  `ArtifactIdentity` (sec. 8) does not carry it — the rule changed how `θ`
  moved, not how `θ` is read, so a bundle fitted under either rule reloads
  through a gate authoring none. `fit_diagnostics.json` records the rule as
  authored plus `frozen_units` and `reawakened_units` (units that were
  hard-off after some step and are kept at the end); the two counts are
  recorded for **every** gate, rule or none, so a leaking fit and a plain one
  are comparable on the number the leak exists to move.
- **`file_path`** (optional): load a fitted artifact instead of computing.
  Its `ArtifactIdentity` (sec. 8) is checked; mismatch refuses. A loaded
  featurizer may not appear in `train.params`.
  - ⚠️ **`model_dtype` is part of that identity, and `--dtype` is not a way to
    satisfy it.** The compared value is the document's `model.dtype`, and a
    `model` block with no `dtype` **implies `fp32`** — so an apply document that
    omits it is refused against a fit that declared `bf16`, with
    `[V15] … implies 'fp32' but the bundle was stamped 'bf16'`. `--dtype` is a
    `--set model.dtype=…` shorthand on a **document** run, and a *workflow* run
    does not accept it at all — so a chained fit → apply can only be repaired in
    the file. Write `"dtype": "bf16"` into the apply document's `model` block,
    next to the fit's. The refusal message says so, and names the stamped
    value.
- **`entry`** (optional, only with `file_path`): which entry of that bundle.
  A swept document writes one file across all its points, keyed by
  coordinate (`weight[k=8,seed=0]`, sec. 2.12), so "the fit at k=8, seed=0"
  is `{"entry": {"k": 8, "seed": 0}}` — coordinate *names*, as they appear
  in the key, not full axis ids.
  - Omitted, the entry is implied by the **consuming point's own
    coordinates**: axis identity is name identity (sec. 3), so a document
    swept on `featurizers.rot.k` selects the fit at *its* `k` and the two
    sweeps zip instead of crossing. Coordinates the producer never had are
    ignored; a bundle with a single entry needs nothing.
  - **Authored, it is used exactly as written and is not completed from those
    coordinates.** The two spellings are alternatives, not layers: completing
    one from the other would make a selector's meaning depend on which axes the
    consuming document happens to sweep. So a *partial* `entry` against a
    multi-axis bundle resolves only when it is already unique, and otherwise
    refuses naming the coordinates that would disambiguate — pinning `k`
    elsewhere in the document does **not** narrow an `entry` that omits `k`.
    Name every varying coordinate, or drop `entry` entirely.
  - A selection that matches no entry, or more than one, is a **load
    error** — never first-hit-wins. Inside a workflow it is caught before
    any step runs, since a producing document's entry names follow from its
    own expansion (workflow spec sec. 5.10).
  - All of a bundle's slots for one entry come from the same point: an
    SAE's `enc` and `dec` cannot be crossed between fits.
- **`top_k`** (optional, `gate` with `file_path` only): read the loaded
  `theta` out as its **`top_k` largest units** instead of through the map's
  threshold. The hard mask a fit reports is one cut through a ranking — `θ > 0`
  under `sigmoid`, `θ > ½` under `clamp`, the stretched split under
  `hard_concrete` — and a ranking method has no threshold at all: a budget
  gate, an attribution score written into `theta`, a magnitude order are
  *only* readable at a count. With `top_k`, the eval-mode split is the first
  `top_k` units of `theta` ordered descending (ties toward the lower index, so
  the cut is a function of `theta` alone); every map's relaxed mask is
  monotone in `θ`, so the order is the soft mask's, and ranking the parameter
  keeps units the hard-concrete clip has saturated to exactly 0 or 1 apart.
  Units are the gate's units: heads on a `head` gate, `(expert, neuron)`
  entries on an `expert_neuron` one. A non-negative integer, **sweepable** —
  `{"sweep": [1, 2, 4, 8, 16]}` is the kept-count → score curve every mask
  method reports, one document, one bundle; `0` keeps nothing and is the base
  run; a count above the unit count is refused at build by name.
  - Identities: at `top_k` = the fit's `hard_mask_size` the two readouts are
    the same mask, so the apply reproduces the threshold apply bit for bit;
    `causalab.analysis.random_mask` takes the same `top_k`, so the size-matched
    control is matched to the cut actually scored.
  - Absent, nothing changes: the hard mask is the map's own split and no field
    enters the canonical form. Refused without `file_path`: a fit's readout is
    decided by its map, and a top-k cut of a *training* mask would make the
    loss and the eval disagree about which units are on. `top_k` does not enter
    the bundle's ArtifactIdentity — it is a way of reading a bundle, not a
    property of one — and is recorded where the readout is: the document's
    canonical form, `fit_diagnostics.json` when a loaded gate sits in a fit,
    and every row of a `rank` table (sec. 2.12).

### 2.6 `params` (optional)

Free tensors owned by no featurizer (steering vectors, a free written value):

| field | meaning |
|---|---|
| `file_path` | constant tensor, loaded |
| `entry` | which entry of that bundle (sec. 2.5), plus the reserved `slot` key |
| `shape`, `init` | trainable free tensor (must then appear in `train.params`) |

- A loaded constant is read from the bundle's `value` tensor by convention.
  A bundle *harvested from a read* is keyed by that read's name instead, so
  `{"entry": {"slot": "acts"}}` names it. `slot` is a params-only key — a
  featurizer's slots are fixed by its kind.

### 2.7 `reads`

```json
"v_cf":  {"site": "target",  "pos": -1, "model": "original", "input": "counterfactual"},
"logits": {"site": "lm_head", "pos": -1, "model": "patched",  "input": "base"}
```

| field | meaning |
|---|---|
| `site`, `pos` | the address |
| `model` | `original` (un-intervened) or a declared intervened_model |
| `input` | `base` \| `counterfactual` \| `counterfactual[j]`. For an intervened model this is redundant with the IM's own `input` and is **cross-checked** — mismatch is a load error |
| `featurizer` | optional; value is read in feature space |
| `dims` | optional static index list into the feature axis; default = all |

- Value = `featurize(activation at (site, pos) in model)[dims]`.
- A read in model `M` sees the activation **with all of `M`'s writes applied**
  (upstream and at the same address). To read an un-written value, read in
  `original` (or an IM without that write).
- Reads never carry `do`.

### 2.8 `writes` and the `do` algebra

```json
"patch": {"site": "target", "pos": -1, "featurizer": "rot", "do": {"swap": "v_cf"}}
```

- A write is an **inert definition**: no `model`, no `input`, no conditions.
  It executes inside every intervened_model that lists it.
- Effect at its address: `write(inverse(scatter(do(f[dims]) into f), err))` —
  untouched dims and `err` from the pre-write value (sec. 2.5).
- `Operand` = a read name · a param name (`rot.weight`, or a `params` entry) ·
  a literal scalar. **Never a tensor, never a closure** — constant vectors
  enter as `params` entries.
- **`ragged`** (optional): how the write lands when its rows address
  different numbers of positions — an `all`, `variable`, `column` or span
  window over rows that tokenize to different lengths (sec. 2.3). Spelled
  `"ragged": {"policy": "refuse" | "exact_length_buckets" | "padded_masked"}`;
  one key today, an object so a later knob has a place. It is **authored
  here and resolved by the executor** on the encoded batch, before any
  forward — the parser checks the vocabulary and nothing else, because only
  the tokenizer can say how wide a row is (rule 19). Absent
  means `refuse` and **nothing is materialized**: a document that authors no
  policy canonicalizes byte for byte as before (sec. 7), and no existing
  document changes meaning. Never swept (rule 14) — how a window lands is an
  execution strategy, not a research variable. Both landing policies work
  inside the forward the batch already runs: the same rows per forward, the
  same fire counts, the same prefix keys, on both engines; every
  per-position mechanism writes the same values under either, and only a
  `gaussian` draw — shaped by the landed slice — differs between a bucket's
  width and the padded width. A ragged **operand** (a read at a ragged window
  named by `do`) pairs into the write row by row under a landing policy and
  is refused when any row's widths disagree; under `refuse` it is refused as
  the write is. Under either landing policy an operand must carry each row's
  own width (only a one-position operand broadcasts): a uniform dense operand
  as wide as the widest row is refused — `ragged_write_unsupported`, the same
  refusal under both policies — never truncated into a narrower row.

  | policy | what lands | what the receipt records |
  |---|---|---|
  | `refuse` (the absent field) | nothing — rule 19's refusal before any forward, reason `ragged_write_unsupported` | nothing |
  | `exact_length_buckets` | the rows grouped by width, one dense gather per width; each row at its own width | `execution.ragged["<model>/<write>"]`: the policy, every row's width, the `[width, rows]` buckets (sec. 8) |
  | `padded_masked` | one gather over the rows padded to the widest, the write computed once, and only the real slots scattered back — padding is read as a duplicate of a real activation and never written | the same block: policy, per-row widths, buckets |

Closed mechanism set (`do` has exactly one key):

| `do` | write | class |
|---|---|---|
| `{"swap": op}` | `f ← op` | absolute |
| `{"add_scaled": {"op": op, "alpha": a}}` | `f ← f + a·op` | additive |
| `{"lerp": {"op": op, "alpha": a}}` | `f ← (1−a)·f + a·op` | absolute |
| `{"affine": {"A": param, "b": param}}` | `f ← Af + b` | absolute |
| `{"gaussian": {"seed": s, "scale": c, "axis": "tp_duplicated" \| "tp_split"}}` | `f ← f + c·randn(s)` | additive |
| `{"renormalize": true}` | `f ← f·‖f₀‖/‖f‖` | absolute |
| `{"clamp": {"lo": a, "hi": b}}` | `f ← clip(f, a, b)` | absolute |
| `{"pytorch_fn": {"code": "…"}}` | arbitrary | absolute; names a `code` declaration (sec. 2.8.1); **local-only** — refused at load by any non-local engine |

- Per (site, overlapping pos, model): **at most one absolute write**; any
  number of additive writes. Application order: absolute first, then additive
  deltas summed. This replaces any commutativity analysis and makes write sets
  order-free.
- `gaussian.axis` tells a tensor-parallel engine whether the draw is
  replicated or sharded across ranks; `seed` is part of the hash.
- **There is no resampling mechanism** — no `{"swap": {"resample": …}}`,
  no per-forward permutation of a read's rows. Which counterfactual row a base
  row meets is a property of the **data**, decided before any forward and held
  for the whole run — every update of a fit included. The resampling control
  ("how much of the effect is *this* counterfactual, and how much is any
  in-distribution activation at this site?" — the activation-space twin of
  NeuroSurgeon's `random_ablate`, which keeps the site on its distribution
  where a mean or zero ablation does not) is therefore the plain `swap` over a
  counterfactual role that authors `shuffle: {seed}` (sec. 2.2): row *i*
  receives row `order[i]`'s value. Nothing about it is new: the pairing is in
  the canonical form and the digest (sec. 7), so an interned group's content
  is fixed by its digest; the workflow's `shuffled_source` kind certifies it;
  and every rule on this page — the absolute class, rule 19's landing, rule
  21, `dims`, the featurizer path, CUDA-graph capture, sec. 8's layout
  transparency — is the swap's, because it is a swap. What is deliberately not
  offered is a permutation redrawn per forward or per update (a per-step
  sample): a write whose value is not a function of the document would put a
  seed into every group's identity, exempt fits from the store, and need a
  receipt to replay what the data already states. A spread over pairings is
  several documents, one per `shuffle.seed`. `shuffle` may leave fixed points,
  and sec. 2.2 says who accounts for them.
- **An operand is read from at or above the address it lands on** (rule 21).
  Deeper is *executable* — the operand's model runs first, which is exactly
  what sec. 2.9's acyclicity licenses — but the network has no edge from the
  deeper address to the shallower one, so a write fed that way is attributable
  to no path, and a number attributable to no path is what this rule exists to
  refuse. Equal depth is the two-pass idiom: harvest a receiver's value in one
  intervened model, inject it at that same address in another. If the value is
  genuinely meant as an externally supplied constant rather than a routed
  activation, say so — harvest it into a bundle in its own run and load it as a
  `params` entry (sec. 2.6), where being a constant is the declared thing.
  Because the order is `(layer, intra-block rank)` and not layer alone, the
  rule is block-order aware: a head's contribution feeding the same layer's MLP
  is upstream, and the reverse is refused.

#### 2.8.1 `code` — user functions identified by content

A `pytorch_fn` runs user code inside the intervention, so the code is part of
the experiment. A bare qualified name says which function without saying
*what* it is: the body, the arguments, the files it opens, the environment it
reads and the row convention it assumes all sit outside the digest, and two
demonstrably different interventions can carry one protocol identity. The
`code` section is that declaration, and `pytorch_fn` names an entry in it.

```json
"code": {
  "corrupt": {
    "locator": "rome.corruption.add_noise",
    "args": {"sigma_multiple": 3.0},
    "data_inputs": {"scale": "stats/subject_embedding_std.json"},
    "env_inputs": ["ROME_NOISE_SCALE"],
    "row_roles": [{"role": "clean", "rows": 1}, {"role": "corrupted", "rows": 10}]
  }
}
```

| field | required | content |
|---|---|---|
| `locator` | ✓ | the importable dotted path to the function |
| `args` | – | typed JSON keyword arguments, passed on every call |
| `data_inputs` | – | name → file path the function may read; each is content-digested at load |
| `env_inputs` | – | environment variables the function is allowed to read (**names only** — a value is a property of the machine and belongs in the run receipt) |
| `row_roles` | – | what the rows of the batch it receives are, **in batch order**: a list of `{"role": …, "rows": n}` |
| `description` | – | free text |

Derived and stamped into the canonical form, never authored (sec. 6):
`source_module`, `source_sha256`, `data_input_digests`, and — only when the
module imports a sibling outside the `causalab` package — `closure` and
`closure_sha256`.

- **`source_sha256` is the defining module's bytes**, the same quantity a
  workflow script step hashes (`docs/workflow_protocol.md` sec. 4.2), through the
  same function. Not the function's own source segment: a function's behaviour
  depends on its module's helpers, constants and imports, so a segment hash
  reports *unchanged* for an edit that changes every output — and a function
  built by a factory at import time has no segment to hash at all. A
  consequence worth stating: a locator into an installed third-party package
  puts *that package's* file in the digest, which makes the document name the
  version it ran against.
- **`closure_sha256` is the defining module's declared sibling import
  closure**, and `closure` the manifest behind it — `{path: sha256}` for every
  module *beside* the defining module (under its own root, resolved by
  filesystem shape) that it reaches transitively through its `import` /
  `from … import` statements, sorted by path; `closure_sha256` is the sha256 of
  the newline-joined `"<path> <sha256>"` lines. Both are written only when the
  manifest is non-empty. The same two fields a workflow script step carries,
  from the same walk (`docs/workflow_protocol.md` sec. 4.2 states the
  boundaries: lazy imports in, `TYPE_CHECKING` blocks out, parent `__init__`
  execution out, dynamic imports invisible). `source_sha256` keeps its meaning
  — the defining module alone — and the closure never lists the module itself.
  What the closure deliberately excludes is everything that is runtime
  identity rather than document identity: the `causalab` package's own modules
  (the `tree_digest` every step record carries and `--resume` compares, so a
  locator into the package carries no closure and no edit to the protocol core
  moves a document), and anything resolving outside the repository — the
  third-party file a locator points into is hashed as `source_sha256` because
  the document named it, but a third-party module the code merely *imports*
  has a version that belongs to the run receipt, never the document.
- **Nothing is imported to compute them.** Resolution walks the package tree on
  disk, and the closure walk reads and parses each member without importing
  it, so `validate`, `explain` and `digest` stay torch-free — which is the
  property that lets the hash reach the digest at all.
- **The call.** The mechanism calls `fn(f, **args)`, and adds `row_roles` (as
  `role -> (start, stop)` half-open bounds) as a keyword **when, and only
  when, the declaration carries roles**. So a plain one-tensor function is
  still called `fn(f)`, and a function that needs to know which rows are which
  is *told* instead of assuming. `data_inputs` and `env_inputs` are not passed:
  they are identity and an allowlist, not a delivery channel — the function
  opens its own declared path, and the load refuses one it did not declare.
- **Row roles name the batch, and the batch is the resolved input rows.** The
  tensor a `pytorch_fn` sees has one row per row of the table the write's
  intervened_model reads (its `input` role, sec. 2.9), so `row_roles` is a
  claim the loader can check against that table's length — rule 25, in the
  `validate --data` pass and again inside `run_protocol` before an engine is
  chosen. It guards a failure a ROME replication met: a corruption function
  that assumed one clean row then ten corrupted, with nothing anywhere saying
  eleven, would have run happily on twelve.
- **Undeclared reads are refused where detectable** (rule 24). The loader
  reads the function's AST and refuses a literal `os.getenv("X")` /
  `os.environ["X"]` the `env_inputs` do not allow, and a literal path passed to
  `open` / `.read_text()` / `.load()` and friends that `data_inputs` do not
  name. This is a declaration checker, not a sandbox: a read routed through a
  variable is invisible to it, and a function whose signature cannot be read
  statically — a closure from a factory — is not refused, only unchecked.

### 2.9 `intervened_models`

```json
"patched": {"input": "base", "writes": ["swap_sender", "freeze_10", "freeze_11"]},
"final":   {"input": "base", "writes": ["inject"]}
```

| field | meaning |
|---|---|
| `input` | **mandatory** — `base` \| `counterfactual` \| `counterfactual[j]` |
| `writes` | the writes in force; **unordered** (canonical form sorts) |

- `original` is the reserved name for the un-intervened model (on any input);
  it is never declared.
- **Membership rule**: every declared write appears in ≥ 1 intervened_model.
- **Cross-model data flow has exactly one channel**: a read in model A may be
  the operand of a write in force in model B. No direct IM→IM wiring, no
  inheritance. The graph (IM → writes → operand reads → IMs) must be acyclic —
  it is the execution schedule's skeleton.

### 2.10 `metrics`

Closed vocabulary; `of` names a read. The other value fields do **not** all
mean the same thing, and the difference has cost real debugging time:

- `a` / `b` / `token` / `target` / `expected` name **dataset columns** — the
  answer is per row, so the document names the column and the table carries the
  string, or an exact integer vocabulary ID under `token_form: "id"`.
- `class_probs`'s `groups` holds **literal token strings**, `{name: [tokens]}`.
  A class is a property of the answer *space*, one for the whole run, so there
  is no column to read it from. `token_logits`'s `tokens` is the same thing as
  a flat list: the task's output vocabulary, saved as raw logits.

| kind | fields | what the value fields name | result per example | unit | estimand_version |
|---|---|---|---|---|---|
| `logit_diff` | `of, a, b` | columns | `logits[a] − logits[b]` | `logit` | `logit_diff/v1` |
| `soft_accuracy` | `of, a, b` | columns | `σ(logits[a] − logits[b])` — the same margin squashed to (0, 1), so a row counts as "a beats b" with a gradient that fades once it is decided (a soft-accuracy objective, the one budget masks train on) | `fraction` | `soft_accuracy/v1` |
| `token_logit` | `of, token` | column | `logits[token]` | `logit` | `token_logit/v1` |
| `cross_entropy` | `of, target` | column | CE against target | `nat` | `cross_entropy/v1` |
| `kl` | `of, target` | a **read** | KL between two reads' distributions — comparable ones: same effective width, same transform, same token-position frame (rule 29) | `nat` | `kl/v1` |
| `js` | `of, target` (+ optional `restrict`) | `target` a **read**; `restrict` a column (a per-row **list** of answer strings) or **literal token strings** | Jensen–Shannon divergence between two reads' distributions, both restricted to the answer set and renormalised when `restrict` is given (see below) | `nat` | `js/v1` |
| `class_probs` | `of, groups` | **literal token strings** | summed probability per group | `fraction` | `class_probs/v1` |
| `token_logits` | `of, tokens` | **literal token strings** | the raw logit of every listed token (see below) | `logit` | `token_logits/v1` |
| `top_k` | `of, k, by` | — | the k top-ranked entries of the read (see below) | — (a structure) | `top_k/v1` |
| `match` | `of, expected` (+ optional `mode`) | column (of a string, or a **list** of equivalent forms) | match indicator | `fraction` (a 0/1 indicator) | `match/v1` |
| `decode` | `of` | — | the addressed tokens as text | — (text) | `decode/v1` |

**Reads a kind may bind to.** Every kind but `kl`, an unrestricted `js` and
`top_k` names *vocabulary entries* — an authored string resolved to a token id
— so it binds to a *plain* `lm_head` read and a metric over anything else is a
load error. Plain means no `featurizer` and no `dims`: a featurizer
re-expresses the projection in its own latents and `dims` re-indexes a slice,
so under either one the read's entries are no longer token ids even though the
site says `lm_head`. `token_logits` is in that group even though its result
looks like `top_k`'s: it *starts* from authored strings and indexes the
projection by their ids, which is exactly the step a featurized or
`dims`-sliced read makes meaningless. `kl` and `js` compare two reads against
each other, and both reads must tap the same component; `top_k` reports indices
along whichever axis its read has. Those bind to a read at any component — a
`js` **with** `restrict` resolves its answer strings, so it binds like a
token-space kind.

**`js`, and `restrict`.** `js` is the symmetric, bounded twin of `kl`:
`JS(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m)` with `m = ½ (p + q)`, in nats, never
above `ln 2` — so its gradient does not grow as the `of` distribution gets
confidently wrong, which is why a mask fit toward a model's *own*
counterfactual distribution uses it. `restrict` narrows the comparison to an
**answer set**: both distributions are sliced to the set's token ids and
renormalised (a `log_softmax` over the slice — exact, no `eps`), so what is
compared is how the two models split their mass *among the answers*, not how
much of it leaks elsewhere. Two spellings, and the shape decides: a **string**
names a column whose per-row value is a list of answer strings (the answer set
varies per row — a task's valid next moves); a **list** is a literal answer
space, one for the whole run (`token_logits.tokens`'s shape — the ten digits).
Either way the strings resolve under the metric's `token_form`, which is
therefore **required** with `restrict` and **refused** without it (an
unrestricted `js` resolves no string, like `kl`); two strings landing on one id
are refused rather than counted twice (the `class_probs` rule); and a row
whose `restrict` column is empty is an excluded measurement (below), exactly as
an empty answer column is. `restrict` is not sweepable: an answer space is not
a research variable. It has no default and is absent from the canonical form
when unauthored, so no unrestricted document's digest moves. `js` is
objective-eligible (§2.11), through the same arithmetic that writes its table.

**Domains.** Every kind consumes one of two things from its read, and which
one is a property of the kind:

| domain | kinds | consumes |
|---|---|---|
| `distribution` | everything above except `decode` | the read's dense value at the addressed positions — the vocabulary projection for every kind but `top_k` |
| `ids` | `decode` | only the tokens the decode produced |

An `ids` kind therefore obliges **no** vocabulary projection anywhere (§8's
materialization requirement) — a text probe is cheap by construction, not by a
engine's cleverness. It also only means something where tokens were
*produced*: `decode` binds to a read whose position carries `generated` (§2.3),
and a `decode` over a prompt-frame read is a load error.

#### Estimand identity — `unit` and `estimand_version`

Every metric record says what it *is*, in two fields
(`causalab/protocol/estimand.py`, the one home both this spec and the workflow
spec import): its **`unit`** — which quantity the number measures — and its
**`estimand_version`** — an identifier for the *arithmetic* that produced it,
in the grammar `<estimand>/v<n>`: `<estimand>` is `snake_case` naming the
arithmetic, not the quantity, and `<n>` increments whenever the arithmetic
changes under an unchanged name. Both are **optional on a metric**, and both
are **derived** when unauthored: a kind is one arithmetic in one unit, so the
two columns above are what every kind's rows carry whether or not the
document says so. Authoring one is a statement, not a choice — a `match` may
say `"unit": "fraction"` and `"estimand_version": "match/v1"`; a `match`
saying `percentage_points`, or `ratio_of_sums/v1`, is refused at parse
(`P4`, naming what the kind computes), as is an off-vocabulary unit or a
malformed identifier (`P4`, with suggestions and the grammar). Neither field
is sweepable: an identity is not a research variable.

The unit vocabulary is closed and grows by PR; `tests/protocol/test_estimand.py`
holds this table to `UNITS` and the kind table's `unit` column to
`METRIC_UNITS`:

| unit | what it measures | produced by |
|---|---|---|
| `fraction` | a probability or proportion in [0, 1] | `class_probs` (softmax mass), `match` (a 0/1 indicator whose mean is the accuracy), `soft_accuracy` (a sigmoid of a margin) |
| `percentage_points` | the same quantity × 100 — a **different** unit, which is the point: a fraction is never compared to percentage points | no kind; a campaign's own table, or a declared rescaling |
| `count` | a number of things | the workflow `count` estimator (workflow spec §2.6) |
| `logit` | a raw, or differenced, pre-softmax score | `logit_diff`, `token_logit`, `token_logits` |
| `nat` | information in base *e* — `log_softmax` is a natural log | `cross_entropy`, `kl`, `js` |
| `bit` | information in base 2 | no kind; a campaign's own table |
| `dimensionless` | a ratio of two like-unit quantities that is not a proportion | no kind; a campaign's own table |

**Which kinds admit more than one arithmetic: none.** The identity would be
*required* on a kind or estimator that admits more than one arithmetic under
one name, because there the record is silent about which it is. No kind does:
each is exactly one gather-then-reduce once its fields are fixed (`match`'s
`mode` and `top_k`'s `by` are fields, in the canonical form), so a kind's
identifier is derived — `<kind>/v1` — and never required. The place two
arithmetics share a name is *after* the rows are on disk: a "normalized
recovery" can be a mean of per-row ratios in one report and a ratio of sums
in another. That is the workflow `reduction` block's object, and its
identifiers are the reduction's (workflow spec §2.6, rule 13); the table
there says which estimators admit which identifiers.

| kind or estimator | arithmetics under one name | identity |
|---|---|---|
| every metric kind above | one | derived, `<kind>/v1`; authored only as a statement of the same |
| the eight `reduction` estimators | one each, given the block (unit, weight, missing policy) | derived, `<estimator>/v1`, unless the block authors a campaign identifier the block **admits** — `mean_of_eligible_row_ratios/v1` for a row-unit `mean` with `missing: exclude`, `ratio_of_sums/v1` for a row-unit `weighted_mean` over the denominator — refused otherwise (workflow §5 rule 13) |

**Where the identity lives on disk.** A metric table has no envelope
(`causalab/protocol/tables.py`), so every row repeats `unit` and
`estimand_version` beside `produced_by` — `jq`-able, and consistent with
"labels repeat on every row". The *authoritative* copy is the canonical form
(§7), where both appear **only when authored**: a document that states neither
has the digest it had before the fields existed. The eligible rows the
`mean_of_eligible_row_ratios/v1` name refers to are the rows the reduction's
`missing: exclude` kept — `n` against `n_excluded` in its output; the metric's
own eligibility, and the one threshold a document may declare, are the next
subsection's.

#### Eligibility — `eligible`, `n_eligible`, `minimum_count`

Every result cell records how many rows its decision rule was evaluated over,
and a row the instrument could not measure reads as an **excluded
measurement**, not as negative evidence (a partial-signal disposition: cells
the instrument could not measure are excluded measurements, not null
localizations). The source is sec. 4.1's typed
`unavailable`, and this section adds no second spelling of it: a row is
excluded exactly when its value *is* an `unavailable`, and the reason code it
carries is that value's.

**Three sources, one record.** A metric's row is structurally unobservable —
a fact of the data or the model the document could not know — when

- the read it reduces aligned on nothing for that row (a `variable` / `column`
  position whose value occurs zero or several times, sec. 2.3): the row is
  excluded under `alignment_missing` / `alignment_ambiguous`, and the metric
  is computed over the rows that aligned — the read's cell stays as sec. 4.1
  describes it, the metric's cell is unavailable only when **no** row aligned;
- the table carries **no answer** for the row in a column the kind names —
  `null`, absent, or an empty list of forms: excluded under
  `alignment_missing` (the authored answer has no counterpart in the data),
  never scored against the string `"None"` and never a refusal of the run;
- a continuation row addressed nothing (`matched: false`, sec. 2.3): excluded
  under `alignment_missing`, as before with the `null` value it always had.

A malformed *specification* is still a refusal, not an excluded row: a
multi-token answer under `mode: exact`, an answer space that is not
first-token distinct, two literal tokens resolving to one id, an ambiguous
`token_form: "auto"`, a `top_k` `k` outside the read's width. Those are the
document's to fix, and a run that quietly excluded the rows they touch would
report a number for a metric it did not compute.

**On disk.** Every metric row carries `eligible` (`true` / `false`); an
excluded row alone also carries `reason_code` — one of sec. 2.4's codes — and
a `null` value, so an excluded row and a row that scored `null` for another
reason never look alike after a group-by. The aggregate cell — the point's
summary entry for the metric, and its `RunResult.cells` value — carries
`n_eligible` and `n_considered`, plus `excluded` by reason when any row was;
its value is the mean over the **eligible** rows only. All of it is derived
(sec. 6): none of these is authored, and none enters the canonical form, so
every result written before the record existed differs only by the new
columns.

**Three denominators, named apart.** `n_eligible` is the rows a metric's
*decision rule* was evaluated over — this section's. `save.reduce: "count"`
(sec. 2.12) is the rows a saved *read's* in-forward reduction collapsed — the
denominator that makes `sum` composable. A workflow reduction's `unit`
(workflow spec §2.6) is the *statistical unit* a saved table is later reduced
over — pairs, prompts, components — and its output's `n` / `n_excluded` count
those units. The three are not interchangeable and no field of one is read as
another's.

**The threshold: `minimum_count`.** A metric may declare the fewest eligible
rows its decision rule needs — optional, a positive integer, never sweepable,
in the canonical form **only when authored** (a document that declares none
keeps its digest). It is the one authored field here; the counts it is held
against are derived. `validate --data` refuses a threshold above the resolved
base table's **maximum eligible count** — the rows carrying a value in every
column the metric names, the one part of eligibility a table settles without a
model — under rule 4, naming the maximum, the table's size and the empty
columns (`[V4] at metrics.iia.minimum_count: … can make at most 2 of its 3 rows
eligible`). A threshold at exactly the maximum passes; a metric with none
makes no claim. The check is against the maximum and **not** against the row
count: a table three of whose ten answers are empty cannot make a threshold of
ten whatever the model does. Whether a *run's* `n_eligible` then met the
threshold is the cell's to report and a consumer's to read (the workflow's
controls and reductions); no run is refused over it. The controls themselves are declared
one layer up, on the workflow that applies this document — a step's `control`
and `waive` and the per-point statuses a dependent step inherits (workflow
spec §2.2, §8) — never inside `method`, so a control is a document like any
other and its seeds are the application's, not the method's.

**Comparisons and claims.** Two records whose units differ may not be compared
— by a reduction combining their rows, by `causalab.analysis.paired_ttest`, or
by a report binding — and the refusal names both units and both records. Two
records that declare nothing compare as they always did (an unknown unit is not
a wrong one); two arms of one campaign in the same unit and estimand compare
as `arm`; the same unit under two declared estimands is a **version
comparison**, allowed and labelled. A report claim binds a number to its record
— file, `produced_by` point digest, `estimand_version`, `unit`, value
(`estimand.Claim`) — and `estimand.check_claim` refuses a claim whose record no
longer produces that value, naming the record and the point digest: a rerun
cannot leave a stale value in prose. That is the binding and the refusal only;
generated report views are a later layer.

#### `top_k` — one kind over any read

`top_k` is the reduction for "I only want the largest few entries per row". Its
reason to exist is that the alternative is saving the whole tensor to disk and
argsorting it later: a 4k-wide residual stream, or a 100k-latent SAE/BSF
featurizer output, times every example and every position. Like `reduce: mean`
on a save entry (§2.12), it happens **where the rows are gathered**.

So `top_k` binds to any read — `lm_head`, `block_output`, `mlp_activation`, a
featurizer's output. There is deliberately no `top_dims` / `top_features`
sibling kind: one kind, disambiguated by mandatory fields.

- **`k`** (mandatory, integer) — how many entries per row. `1 ≤ k ≤ width`.
- **`by`** (mandatory, `value` | `abs_value` | `prob`) — the ranking rule. It
  is mandatory because only the author knows what the axis is, and the answers
  differ: a vocabulary projection has no meaningful negative entries, while a
  residual stream and a signed feature code do, so ranking an SAE code by
  signed value and by magnitude return different sets.
  - `value` — the k largest signed entries. Any read.
  - `abs_value` — the k largest by `|x|`; the reported value stays signed. Any
    read.
  - `prob` — softmax the last axis, then take the k largest probabilities.
    **Plain `lm_head` reads only** — a softmax across neurons, SAE latents, a
    featurizer's re-expression of the projection or a `dims` re-index of it
    normalizes over an axis that is not an event space, so its "probabilities"
    would be probabilities of nothing; validation refuses it elsewhere. A
    pre-`by` document that ranked logits meant `prob`.

**Result columns.** Each has one fixed meaning, in every document. A column is
*absent* when it does not apply — never reinterpreted:

| column | meaning | emitted when |
|---|---|---|
| `indices` | index along the read's last axis (a token id on `lm_head`, a neuron on `mlp_activation`, a latent on a featurizer output) | always |
| `tokens` | that index decoded as a token string | the read is a plain `lm_head` tap |
| `values` | the **raw** read value at that index | always |
| `probs` | the softmax probability over the vocabulary | `by: "prob"` |

`values` is always raw — a logit under `by: "prob"`, not the probability — so a
downstream reader never has to know the ranking rule to know what it is
holding. The normalized number lives in its own column.

#### `token_logits` — the task's answer space, saved

`token_logits` is what an intervention document saves so a causal variable
proposed *later* — whose answers lie in the task's output vocabulary — can be
rescored without a forward. The alternative it replaces is saving the whole
`lm_head` read, which at a 250k-wide vocabulary is the one tensor nobody wants
per example per position; `top_k` cannot stand in for it, because the token a
later hypothesis cares about is not necessarily in the top k.

- **`tokens`** (mandatory) — a non-empty list of **literal token strings**, one
  list for the whole run, for the same reason `class_probs`'s `groups` is: an
  answer space is a property of the run, not of a row. It is fixed per
  campaign — not sweepable, like `token_form` and `top_k.by` — because a sweep
  over it would fork the campaign on what gets saved rather than on a research
  variable. Each answer is listed once — `["X", " X"]` is refused at parse,
  since a leading space is normalized away before `token_form` decides the
  form — no entry may be empty or whitespace-only, and two *different* strings
  that the tokenizer maps to one id are refused when the metric resolves them.
- **`token_form`** (required, as on every string-resolving kind) — each listed
  string must be one token under the declared form; a multi-token entry refuses
  rather than scoring its first piece, the single-token rule every other
  string-resolving kind follows.
- Binds to a **plain `lm_head` read only** (above).

The result per example carries the same three columns `top_k` emits, with the
same fixed meanings, in the order the document listed the tokens:

| column | meaning |
|---|---|
| `indices` | the resolved token ids |
| `tokens` | the **decoded form of each resolved id** — the tokenizer's own string for the id in `indices`, not the authored entry — so the column shows what was *scored*, which is how a wrong `token_form` shows up in the table rather than in a number |
| `values` | the **raw** logit of each id |

Over a multi-position read it follows the per-position rules below: one row per
(example, position), each carrying the three lists.

**Metrics over several positions.** A read may address more than one position —
every generated token of a row, a window of them, the tokens where the model
said something (§2.3). A metric over such a read reduces **per position**, and
its table says which:

- one row per (example, position), carrying the `step` it scored and a
  `matched` flag; `decode` is the exception that reduces the whole window to
  one string, and its `step` is null because no single step owns it;
- an example that addressed **nothing** — it stopped generating, or never said
  what a `variable` anchor looked for — still gets exactly one row, with a null
  value and `matched: false`. "The model never said it" must not be
  indistinguishable from "it said it and scored 0";
- prompt-frame metrics are unchanged: one row per example, no `step` column.

- A metric binds to exactly one read → one (model, input). Same metric in two
  models = two reads + two metrics.
- Metrics are gather-then-reduce over read values and dataset columns —
  nothing else. Cross-read arithmetic (differences of saved metrics) is
  post-hoc analysis. The vocabulary stays closed so engines can lower kinds
  to fused/vocab-parallel implementations.
- **`token_form`** (**required**, `auto` | `bare` | `space_prefixed` | `id`) — how
  this metric's string answers become token ids. Required on every kind that
  names token strings (`logit_diff`, `token_logit`, `cross_entropy`,
  `class_probs`, `token_logits`, `match`); `kl` and `top_k` never resolve a
  string and refuse the key. That stays true under `top_k`'s any-read semantics: it *reports*
  indices it found and decodes them only when the read taps `lm_head` — it
  never turns an authored string into a token id, so the knob would have
  nothing to apply to.
  - **`id`** reads exact integer vocabulary IDs from dataset columns for
    `match`, `token_logit`, `logit_diff`, and `cross_entropy`. It never decodes,
    strips whitespace, or retokenizes a target. Strings, booleans, floats and
    out-of-vocabulary IDs are refused. An exact `match` may use a nonempty list
    of acceptable IDs. `first_token` matching and the literal-string kinds
    `class_probs` / `token_logits` refuse `id`. Differentiable objectives use
    the same ID resolution as evaluation. This is the form for contextual
    output-token targets, including whitespace and EOS tokens.
  - **Why required rather than defaulted.** How a string becomes a token id is
    a fact about the model's tokenizer, and a document that does not say which
    rule it means gets whichever one the library happened to prefer. That guess
    has been measurably wrong four ways in production: a leading space
    (`" ?"`=907 against `"?"`=30), punctuation that merges with the token
    before it, two authored forms resolving to one id and being summed twice,
    and the non-case of digits where `" 7"` really is two tokens and `auto` is
    right. `auto` remains available — as something a document *chooses*.
  - `auto` tries `" " + s` first and falls back to `s`. That is right when the
    answer follows a space in the prompt — weekdays, names, MCQA letters.
  - It is **wrong** when the answer does not follow a space and both forms
    happen to be single tokens. Under gpt2, `"?"` is token 30 and `" ?"` is
    token 5633: a `match` on a punctuation answer scored 5633, the model emits
    30, and the metric read a flat 0.000. Pin `token_form: "bare"` for those.
    **`auto` now refuses** rather than guessing whenever the two forms
    disagree — as a single token each (the case above), or, under
    `mode: "first_token"`, on which piece they credit. It warned before, and a
    warning that produces a wrong number anyway is not a check.
  - **A bare `token_form` over an answer the table carries space-prefixed is
    refused** before any forward pass (the other alignment error class: a bare
    token chosen where the model's answer is space-prefixed), with reason
    `alignment_missing` and **both surface forms decoded** — under a byte-level
    BPE `"Saturday"` and `" Saturday"` begin with different pieces, so the
    metric would credit a token the model never emits. It is the metrics half of
    sec. 2.3's one cardinality function: the bare form's tokens are located
    inside the answer's content tokens, and `absent` refuses. Where the two
    forms coincide (a sentencepiece family, whose `"Saturday"` *is* the
    `▁Saturday` piece) nothing fires; a bare table value says nothing about the
    model's form and is not checked.
  - The form applies to every token string in the metric, so a `class_probs`
    whose groups mix spaced and bare answers must stay on `auto`.
  - ⚠️ **A leading space in an authored value is normalized away, not
    honored.** The resolver strips it and then `token_form` alone decides the
    form, so `" X"` and `"X"` name the same answer. The consequence is worth
    stating because it is the opposite of what the spelling suggests: the
    common `["X", " X"]` idiom — written to "cover both forms" — is **inert**.
    Both entries resolve identically, and in a `class_probs` group, where the
    kind sums its members' ids, that used to be summed twice and report a
    probability above 1 (a measured **1.9927**). A group whose members collide
    on one id is now refused; list each answer once and say which form with
    `token_form`.
- A column value resolves to **one token**, space-prefixed form first; a
  multi-token value refuses rather than silently scoring its first piece.
- `match` is the exception, and only when told: its `expected` column may hold
  a **list of equivalent surface forms** (synonyms, casings — the argmax
  matching any of them scores 1.0), and `"mode": "first_token"` credits a
  form's first token instead of requiring the form to be one token. `mode` is
  `"exact"` by default and is materialized into the canonical form, so an
  omitted `mode` and an authored `"exact"` digest identically.
  - Which forms are equivalent is **task data**, serialized as a column when
    the table is built (from the task's `ScoringSpec` — its `forms`) —
    never a document-side string transform. Case folding included: a task that
    wants case-insensitivity serializes the casings as forms, because at this
    layer the comparison is between token ids, not strings.
  - `first_token` is what "prefix" means with logits at one position. It
    over-credits an answer space that is not first-token-distinct; whether a
    table's answer space *is* first-token-distinct is a property of the
    dataset, so the mode is opt-in per document and never a default — **and
    the metric refuses** a row set in which two different answers share a
    first token, which is the last place the claim can be checked before a
    number exists. `" 85"` is `[220, "8", "5"]` on Qwen, so an emitted `87`
    would otherwise score 1.000 against an expected `85`.

#### The task's scoring and the `match` mode — one translation table

A task declares what a correct answer *is* once, in its `ScoringSpec`
(`causalab/causal/scoring.py`, `causalab/tasks/README.md`): the surface
`forms` of every declared value, which variable the graded string is a form
of, and a `string_mode` — how a generated *string* compares to a form. The
protocol's `mode` compares an *argmax token* at one position. The two
vocabularies are related by a typed derivation, `ScoringSpec.protocol_mode`,
and this table is its census (`tests/protocol/test_vocabulary_census.py`
holds the left column to the task's `STRING_MODES`, the right to the protocol's
`MATCH_MODES`, and the map to `PROTOCOL_MODES`):

| task `string_mode` | protocol `mode` | why |
|---|---|---|
| `exact` | `exact` | the stripped string equals a form; with logits, the form is one token and the argmax is it |
| `prefix` | `first_token` | the string starts with a form; with logits at one position, a prefix is the answer's first token |

The derivation is what a table built from the task records as its
`string_mode` (sec. 2.2), and what a document over that table is held to: a
`prefix` table under `mode: exact` is a contradiction — the document says the
answer is one token and the table says it is not — refused before any forward.
An `exact` table under `first_token` is not one (`first_token` generalizes
`exact` on single-token answers); the half that can go wrong there is the
first-token-distinctness refusal above, which is the metric's own.

**A genuinely multi-token graded answer** needs neither mode. `first_token`
stays lossy even when the answer space is distinct — it credits one token of
an answer that is several — and `exact` refuses the value outright. The path
that grades the whole answer is the `ids`-domain kind: a `decode` read over
the continuation returns the text the model produced, obliging no vocabulary
projection at all (sec. 8), and the task's spec grades that text
(`ScoringSpec.grade`: `1.0`, `0.0`, or `null` for a generation that names no
declared value under `invalid_output: unscored`). The grade is a metric record
in the `fraction` unit under its own arithmetic name, `string_grade/v1` — the
same unit `match` produces, not the same arithmetic, so it is not called
`match/v1`. A task whose answers cannot be said as forms declares a
`full_string_checker` inside the spec, digested with everything else, and the
same path runs its function.

The per-example reading of that record is the **`grade`** vocabulary
(`causalab/causal/pairs.py`, `GRADES`; `tests/protocol/test_vocabulary_census.py`
holds this table to it), a one-to-one relabelling of what `ScoringSpec.grade`
returns:

| per-example `grade` | `ScoringSpec.grade` returns | meaning |
|---|---|---|
| `correct` | `1.0` | the generated string is a form of the expected value |
| `incorrect` | `0.0` | it is a form of another declared value, or names none under `invalid_output: incorrect` |
| `unscored` | `null` | it names no declared value under `invalid_output: unscored` |

`grade` is not `matched`: `matched: false` is a metric row saying the model
never said anything at the position the metric reads, while `unscored` says it
said something that names no declared value. A `grade` is emitted by
convention today — a `decode` read over a no-intervention document, graded
through the task's spec, as the pair-validity check "correctness" does
(`causalab/causal/pairs.py`) — not by a metric kind; the `string_grade/v1`
record above is where it lands.

#### Adding a metric kind

The vocabulary is closed and grows by PR — a reduction that does not fit it
belongs in a workflow **script step** (`docs/workflow_protocol.md` §4), which
is content-hashed and needs no vocabulary at all. When a kind does belong
here, it is eight steps, and the point of enumerating them is that a kind
added with three of the eight is a kind that validates and then fails at run:

1. Add the name to the `MetricKind` literal (`causalab/protocol/schema.py`).
   `METRIC_KINDS` derives from it, so there is no second list to edit.
2. Add its mandatory value fields to `METRIC_FIELDS` — beyond `of`, which
   every kind has. Any field that is *not* a dataset column name also goes in
   `NON_COLUMN_METRIC_FIELDS`, or `validate --data` will look for a column by
   that value.
3. Add its domain to `METRIC_DOMAINS`. `distribution` obliges a vocabulary
   projection at the read; `ids` obliges none, which is what makes a text
   probe cheap by construction (§8). Getting this wrong makes the planner
   materialize a tensor nobody reads, or fail to materialize one that is.
4. The optional tables, only if they apply: `OPTIONAL_METRIC_FIELDS` plus
   `METRIC_FIELD_DEFAULTS` (an omitted optional field **with a default** is
   materialized into the canonical form, so authored and omitted digest
   identically; one without — `js.restrict` — stays absent, so "not
   restricted" has no spelling to materialize); `READ_TARGET_METRIC_KINDS` if
   the kind's `target` is a read rather than a column;
   `TOKEN_COLUMN_METRIC_KINDS` if a value field carries an authored string
   that must resolve to a token id; `WHOLE_WINDOW_METRIC_KINDS` if the kind
   reduces the whole addressed window to one value per example rather than one
   per position.
5. Add one `if kind == …` branch to `causalab/neural/shared/metrics.py`.
6. Add a row to the kind table above, and — if the kind binds to reads other
   than a plain `lm_head` one — say so under **Reads a kind may bind to**.
7. Add a test. A kind whose result is a number needs one pinned value; a kind
   with a legality rule needs the document that violates it.
8. State the kind's unit: one entry in `METRIC_UNITS`
   (`causalab/protocol/estimand.py`) — a `UNITS` member, or `None` for a kind
   whose value is not a scalar — and the same word in the kind table's `unit`
   column. A kind that scores in a unit the vocabulary lacks grows the
   vocabulary by PR first; the identifier column is derived and needs no entry.

Steps 1 and 6 are checked against each other by
`tests/protocol/test_vocabulary_census.py`, so the table cannot fall behind
the code; step 8 by `tests/protocol/test_estimand.py`, the same way. Nothing checks that step 5 exists, which is why it is listed
explicitly: a kind that parses and has no branch is a load that succeeds and
a run that does not.

### 2.11 `train`

| field | meaning |
|---|---|
| `objective` | the weighted terms, in one of two spellings: positional `[[weight, term], …]`, or named `{name: {"weight": w, …term}}` where the term is `"metric": name` or a regularizer. A regularizer is `{"l1": names}` \| `{"l2": names}` \| `{"l0": names}`: one featurizer (all its params), one dotted slot, or a non-empty list of distinct featurizers penalized **together**, with an optional `"reduce": "mean"` (the default, unspelled) \| `"sum"` over the concatenated per-unit quantities, and an optional `"costs"`: `{<target>: c}` — a finite positive multiplier on that target's quantities before the concatenation (an unlisted target costs 1; the keys are the term's own targets, rule 4) — or the word `"parameter_count"`, which divides each target's quantities by its own element count. A **named** `l1` / `l0` term may carry `"constraint": {"target": t, "dual": {"lr": η, "init"?: [λ₁, λ₂]}}` **instead of** `weight` — Edge Pruning's Lagrangian target density (below); `weight` beside it, the positional form, `l2`, a metric term, `reduce: sum` and `costs: "parameter_count"` are refused (a `costs` table composes: the target is held on the cost-weighted density). Every featurizer named must be in `params`; `l0` names `hard_concrete` gates only, and `l1` on a `hard_concrete` gate is refused (rule 4) |
| `params` | what is optimized: featurizer names (all slots) or dotted slots; the **only** trainability declaration |
| `optimizer` | `{name, lr, …}` — lr/schedule/clip live here. `schedule` is `constant` (the default) or `linear_warmup_decay` — HF's `get_linear_schedule_with_warmup`: lr climbs from 0 over the first `warmup_frac` of the updates (0.1 unless authored; `warmup_frac` is legal only with this schedule and enters the canonical form only when authored) and decays linearly to 0 at the last update, pyvene's sigmoid-mask (DBM) recipe; refused beside `phases`, which rewrite `lr` themselves (rule 4). `lr` and `weight_decay` are one number for every trained parameter, **or a mapping keyed by the entries of `params`** — `{"lr": {"rot": 0.001, "gate": 0.1}}` — naming every entry exactly once (a key `params` does not train, or an entry left without a value, is refused at load): a rotation and a gate stepping at their own rates inside one fit, each entry its own optimizer parameter group |
| `steps` | `{"epochs": n}` or `{"updates": n}` |
| `batch` | `{"pairs": n}` — counts base+counterfactual **pairs**, not rows |
| `anneal` | dotted-path **open-loop** schedules: `{<target>: [start, end, frac]}` or `{<target>: {"from", "to", "frac", "shape": "linear" \| "geometric"}}` — the target is a trained featurizer's `<name>.<slot>.<hyperparameter>` (`gate.theta.temperature`) **or a named objective term's weight**, `train.objective.<name>.weight`, the address a `control` and a sweep use; the value walks from `from` to `to` over the first `frac` of the run and holds. `shape` defaults to `linear`; `geometric` multiplies by a constant per step (continuous sparsification's `T ← T·r`), so its endpoints share a sign and neither is zero. The list is the canonical spelling of a linear schedule (sec. 7) |
| `control` | **closed-loop** schedules: `{<target>: {"kind": "pid", "signal": {"hard_mask_size": <gate> \| [<gate>, …]}, "setpoint": {"ramp": [start, end, frac]}, "gains": {kp, ki, kd?}, space?, bounds?, d_clip?}}` — the target is a named term's weight (`train.objective.<name>.weight`) or an anneal-style dotted hyperparameter, and its authored value is the controller's start; a list of gates is one signal, their kept counts summed (see below) |
| `phases` | consecutive step windows narrowing the fit: `[{"until": {"frac": f} \| {"updates": n}, "params": [...], "optimizer"?: {lr, weight_decay}, "anneal"?: {...}, "freeze_masks"?: [<gate>, …]}, …]`. Inside a phase only its `params` (a non-empty subset of `train.params`, by entry) receive gradients — the rest are frozen, their optimizer groups kept so a later phase resumes; its `optimizer` overrides `lr` / `weight_decay` for its own params; its `anneal` runs over the *phase's* steps; `freeze_masks` pins each named gate's **hard** mask at the phase's start for every forward inside it (rule 4: gates only, none the phase trains). `until` is one unit for every phase, strictly increasing, the last `frac` `1.0` (an `updates` last phase must equal the run's update count — checked by the loop). Absent = the one-phase fit, and a document without it keeps its digest |
| `precision` | `{feature, loss}` dtypes — the *model's* dtype is `model.dtype` (§2.1), one home per fact. An engine whose loop cannot execute the declared precision (no `train_loss_precision` capability, sec. 8) refuses the document at load, rule 30 — never digests one precision and runs another |
| `eval` | `{every, split, metrics}` |
| `early_stop` | `{metric, patience, mode}` |
| `checkpoint` | transient training state (resume); the final artifact is the `save` entry |
| `seed` | init + data order |

- **A metric term's weight is signed, and the sign is the direction.** The loop
  minimizes `Σ wᵢ · termᵢ`, so `{"margin": {"weight": -1, "metric": "ld"}}`
  with `ld` a `logit_diff` **maximizes** the margin — MIB's `L_MIB = ŷ[y_b] − ŷ[y_s]`
  objective as one line — and `-1` on a `soft_accuracy` maximizes the soft accuracy.
  A metric whose smaller value is better (`cross_entropy`, `kl`, `js`) takes a
  positive weight; there is no `maximize` flag because the number already says
  it. Edge Pruning's faithfulness term is the same vocabulary: a `kl` toward a
  read of the *unpatched* run — `{"kind": "kl", "of": "logits", "target":
  "logits_clean"}` with `logits_clean` a read on `model: original` — weighted
  `+1`, beside the sparsity term.
- `train` present ⇒ the run requires gradients (sec. 8).
- Every trained featurizer must have a `save` entry (sec. 2.12).
- `eval` runs on `split` in eval mode (hard gate, no grad) every `every`
  epochs; an `every` counted in `updates` is refused at load (rule 30) unless
  the routed engine declares `train_eval_updates` (sec. 8) — a loop that only
  reaches an eval on an epoch boundary would otherwise run no eval at all and
  still save the fit. The split's rows are read and encoded **once per point**, and the
  groups the fit cannot change are run on it once (sec. 4, "Fits"); only the
  trained model's group is re-run per pass.
  `split` is a dataset ref like a `data` entry's (sec. 2.2), and its digest is
  stamped in the canonical form (sec. 7). When it names a different ref from
  the training rows, the two must be endpoint-disjoint — rule 22's fourth
  refusal, checked by `validate --data` and again before any forward; the same
  ref for both roles is the visible train-equals-test ablation.
- Points of one campaign that all declare `train` on the same model
  realization and the same rows fit **together** (sec. 4, "Cohorts"): their
  optimizer steps and eval passes are batched into one forward each, every
  member keeping its own `seed`, schedule, objective and `early_stop`
  decision. The fitted weights are those of the separate fits up to the
  rounding of a different batch shape.
- **A regularizer over a list is one penalty, not a sum of penalties**: the
  mean over the *concatenation* of what its featurizers are penalized on — a
  gate's `σ(θ/T)` per unit (per coordinate, head or `(expert, neuron)` as its
  `group` says), `|p|` (`l1`) or `p²` (`l2`) over every param otherwise — so a
  gate with more units weighs more, and one weight over forty layers' gates
  counts selected units across all of them. **`reduce`** picks the reduction:
  the `mean` (default, and not materialized when unauthored, so no digest
  moves) makes a kept unit cost `weight / units`, which is why a weight tuned
  on 8 heads is inert on 2048 coordinates; `sum` makes a kept unit cost
  `weight` whatever the unit count — NeuroSurgeon's `λ · Σ` convention — so one
  weight means one thing across gates of different sizes. **`costs`** scales
  each target's quantities *before* the concatenation: a table
  `{"gate_3": 1.0, "gate_15": 0.25}` makes a kept unit of `gate_15` a quarter
  as expensive as one of `gate_3` under either reduction (an unlisted target
  costs 1; a cost of 0 is refused — a target that should cost nothing is a
  target to leave out; a key that is not one of the term's targets is rule
  4's), and the word `"parameter_count"` costs each target
  `1 / its own element count` (`θ` as stored: a grouped gate's *units* — heads,
  experts — not the coordinates they span, the same per-unit reading the
  quantity itself has), so under `reduce: sum` the term is the **sum of
  per-featurizer means** — NeuroSurgeon's λ "scaled with the parameter count",
  written once instead of per gate — and a 40-layer sweep's gates each weigh
  the same whatever their width. Under the default `mean` the same word gives
  `(1/N) Σ_f S_f/n_f`: the per-featurizer equality kept, but the whole term
  scaled by the total unit count `N`, which is the width coupling the word
  exists to remove — so `parameter_count` pairs with `reduce: sum` (the other
  pairing is legal, and is that coupling). A cost is a literal, not swept
  (sweep the term's weight); a uniform table — every target at one cost — is
  the weight restated and is allowed as authored. Unauthored, nothing is scaled
  and no digest moves. **`l0`** is a `hard_concrete` gate's **expected kept
  fraction** per unit — Louizos et al.'s closed form `σ(θ − β·log(−γ/ζ))`, the
  quantity its sampled training mask has an expectation of, where the soft mask
  is the wrong surrogate; `reduce` applies to it as to `l1`. The pairing is by
  map, not by kind (rule 4): `l0` on a deterministic gate is refused — there
  the relaxed mask *is* the kept probability and its mean is `l1`, so `l0`
  would be a second spelling of one computation, digesting apart — and `l1` on
  a `hard_concrete` gate is refused, since it would penalize a mask the
  training forward never uses; on a featurizer with no mask `l0` is refused
  too; on a `budget` gate both are refused — its mask sums to the step's budget
  by construction (sec. 2.5 `k_schedule`), so there is nothing for a penalty to
  move. A one-name list is the name: `{"l1": ["gate"]}` canonicalizes to
  `{"l1": "gate"}`, and the list is sorted, so no document that named one
  featurizer moves its digest (sec. 7). Repeated or empty lists, and a name
  outside `params`, are load errors.
- **`anneal` — a hyperparameter, or a term's weight, on a declared path.**
  The target is a trained featurizer's hyperparameter (the temperature of a
  `sigmoid` gate, the β of a `hard_concrete` one) or a **named** objective
  term's weight, `train.objective.<name>.weight` — the open-loop twin of a
  `control` on the same address: a penalty weight walked from `0.01` to `30`
  over the first half of the run is the gain-free sparsity sweep, where the
  controller's is the closed-loop one. The schedule's `from` replaces the
  authored value before the first update, so the authored weight must be a
  plain number (rule 4), and a positional term has no name to address. The
  value the update *used* is what a `trajectory` checkpoint records under
  `weight.<name>` (sec. 2.12), and `fit_diagnostics.json` records each
  schedule's endpoints, shape and final value under `anneals`. Two shapes:
  `linear`, `from + (to − from) · min(1, step / (frac · steps))`, and
  `geometric`, `from · (to / from)^(min(1, step / (frac · steps)))` — the
  per-step constant factor continuous sparsification anneals its temperature
  by (Savarese et al. 2020), which has no meaning across zero, so `from` and
  `to` of one sign and neither zero (a load error otherwise). A linear
  schedule's mapping spelling canonicalizes to its list spelling, so no
  document authored before the mapping form existed moves (sec. 7).
- **`constraint` — a target density, held by ascent.** Edge Pruning holds a
  mask's density `s` (the `l1` term's soft-mask mean, or the `l0` term's
  expected kept fraction, under `mean`) to a target `t` by adding
  `λ₁·(s − t) + λ₂·(s − t)²` to the loss and **ascending** the dual pair
  `(λ₁, λ₂)` — stepped against its gradient, `(s − t)` and `(s − t)²`, at
  `dual.lr` from `dual.init` (`[0, 0]` unless authored, and then absent from
  the canonical form; `λ₁` may start negative — an equality multiplier ranges
  over ℝ, and the ascent takes it there itself — but a negative `λ₂` is refused
  at load: the ascent only ever raises it (its gradient `(s − t)²` is
  non-negative), so a negative `λ₂` is one no fit reaches, and it inverts the
  quadratic penalty) — with no weight to tune. It is an **equality**, not a
  budget: `target: 0.1` is "exactly 10 %", not "at most 10 %", since a mask
  sparser than `t` is pulled back up (a `weight` term is the one-sided
  pressure). The pull is not immediate — `λ₁` is still positive when the mask
  first crosses `t`, so the fit undershoots and returns once `λ₁` drops below
  `2λ₂·(t − s)` (early on, that is `λ₁` falling through zero; later the
  accumulated `λ₂` does it while `λ₁` is still positive), tighter each cycle as
  `λ₂` grows; a density below `t` in the trace is that turn, not a broken
  constraint. A `constraint` term has **no `weight`** (rule 4: nothing to
  `anneal` or `control` on it) and every target is a gate (rule 4). Our
  `control` (a PID on the weight) covers the same use; this is the
  paper-faithful comparison.
  - **The two multipliers.** `λ₁`'s gradient is the signed gap, so it grows
    while the mask is denser than `t` and falls once it is sparser; `λ₂`'s is
    `(s − t)²`, never negative, so `λ₂` only accumulates — the growing quadratic
    coefficient of a penalty method, and on a target the gate cannot reach
    (below what its `dead` rule or threshold permits, a density that `costs` has
    rescaled, or one a temperature `anneal` is carrying `s` away from — below)
    it grows without bound until the quadratic term dominates the loss;
    `constraints` records the endpoints, so a reader sees it after the fact.
  - **What density is held.** The term's value as `costs` leaves it: a `costs`
    table holds the cost-weighted density to `t`; `costs: "parameter_count"` is
    refused (the term is then `mean(σ)/N`, no longer a fraction, and the
    constraint would invert — `λ₁` negative, the gate driven to *full* density).
    The quantity is the gate's **relaxed** mask, so it moves with the gate's
    temperature: an `anneal` on `<gate>.theta.temperature` (the DBM recipe —
    `configs/protocols/dbm.json` authors it beside `l1`) changes `s` with θ
    standing still, and the duals answer that drift as if the gate had moved.
    The same target therefore names a different θ at each temperature — under
    `sigmoid` at `T = 1` a density of `0.1` is every θ ≈ −2.2, at `T = 0.01` it
    is 10 % of units with θ > 0 — and under `l0` the drift is monotone in β (at
    θ = 0, `β: 2/3 → 0.1` takes `s` from 0.83 to 0.56). Under `sigmoid` the
    limit is the useful one — `σ(θ/T) → 1[θ > 0]`, so as `T` falls the target
    stops being a claim about a relaxation and becomes one about the density the
    bundle saves — but it is a *pointwise* limit, and it bites only once
    `|θ| ≫ T`: the equality pins the relaxed **mean**, so a θ scaled with `T`
    satisfies it exactly at any temperature with an unseparated gate (at `T = 1`
    every θ ≈ −2.2 holds `s` at `0.1`, at `T = 0.01` every θ ≈ −0.022 does, both
    with an *empty* hard mask and `decisive_fraction` 0 — the signature that
    preset's header warns about, met from the other side: the preset failed at σ
    ≈ ½, a satisfied `0.1` fails at σ ≈ 0.1). What the target buys is the mask's
    *size*, not its margin: `hard_mask_size / N` is at most `2t` and reaches `t`
    only on a gate that has committed, so read `hard_mask_size` against `t · N`
    beside the duals — the trajectory row carries the count and the duals,
    `fit_diagnostics.json`'s row the `width` (or `groups`) it is read against.
    The preset's `decisive_fraction` does not substitute for it here: it asks
    whether σ is outside `[0.1, 0.9]`, so the degenerate gate reads 0 at
    `t = 0.1` (the band's own edge) but **1** at any `t < 0.1`, where a uniform
    σ = 0.05 is "decisive" and keeps nothing. Under `l0` the limit is different:
    `σ(θ − β·log(−γ/ζ)) → σ(θ)` as `β → 0`, the deterministic relaxed mask, not
    the `θ > 0` split, so a low `β` collapses the held quantity onto `l1`'s
    reading and buys no separation. Either way the anneal's own stretch is
    charged to the duals, `λ₂` at its largest, so an annealed fit reads its
    `dual.lr` against the schedule.
  - **How the duals step.** One more parameter group of the fit's optimizer
    with `maximize` set, no weight decay and no momentum, so they step with
    the featurizers under the optimizer's own arithmetic (exactly
    `lr · gradient` under `sgd` whatever the document's `momentum`,
    Adam-normalised under `adamw`) and are untouched by `optimizer.schedule`.
    `phases` does not touch them either — and that cuts the other way: a phase
    that does not name the constrained gate freezes `s` but not the ascent, so
    the duals keep accumulating the frozen gap for the whole phase and the
    gate meets it on its first active update. So a constrained term wants
    every phase to name its gate — a warm-up phase that leaves the gate out
    charges the whole warm-up to the duals (the objective is global; the phase
    schedule is the author's lever). `freeze_masks` is no way around it: rule
    4 refuses pinning a gate's mask in a phase that trains its θ, so a phase
    that pins the constrained gate is one that does not train it, and `s` is
    frozen for that phase like any other.
  - **What is recorded.** Each update records `term.<name>` (the density),
    `lambda1.<name>` and `lambda2.<name>` in the trajectory — the duals the
    update *stepped with*, as `weight.<name>` is the weight it used, so a
    checkpoint's pair is the previous trace row's (or `dual.init` at the
    first) and a join on `step` between the two records is off by one ascent.
    `fit_diagnostics.json` records the target, the initial and final duals and
    the density at the **last update** under `constraints` — under early
    stopping the mask the bundle saves is `early_stop.best`'s, whose hard size
    is in `diagnostics`, so `value_final` may not be the saved mask's density
    (as `controls` reports its last value).
  - **Not an axis.** Nothing in the block is swept (refused naming the
    field, `axis` spelling included): one constraint, one target — the
    density → score curve is one document per target, or one document and N
    `set` overrides of `train.objective.<name>.constraint.target` (sec. 1),
    each with its own digest.
  - **Eager only.** The duals step on the eager loop: a run asked to capture
    graphs runs a constrained point eager rather than refusing it — the whole
    point, since `unsupported_reason` chooses the point's executor; and in a
    cohort one ineligible member makes the whole cohort eager
    (`cohort_graph_reason`), as for `control` and `phases`.
- **`control` — a hyperparameter that follows the fit.** `anneal` moves a
  hyperparameter along a declared ramp whatever the fit does; `control` moves
  it every update so that a **signal the fit produces** follows a declared
  setpoint. Two signals: `hard_mask_size` — a trained gate's kept-unit
  count through its hard mask, the number `fit_diagnostics.json` reports under
  that name — and `hard_mask_fraction`, that count over the gates' total unit
  count, so a ramp `[1, 0, frac]` and one set of gains mean the same thing over
  8, 48 or 2048 units (the integral term is `ki · (signal − setpoint)` in the
  signal's units, so a count-valued signal needs `ki` rescaled per unit count);
  either of **one gate, or of a list of gates summed**: a fit over
  several layers' head gates under one list-valued `l1` (one penalty over all
  their units) is steered by one count over all of them, and the canonical
  form writes the list either way, so `"g"` and `["g"]` are one controller
  (the `layers` fold). The one controller is a `pid`. The target is
  either a **named** objective term's weight, `train.objective.<name>.weight`
  (the same address a sweep uses; a positional term has no name to address),
  or an anneal-style `<featurizer>.<slot>.<hyperparameter>` on a trained
  featurizer; the value the document authors there is the controller's
  **initial** value, and a path may not be both annealed and controlled. The
  `setpoint` is a `ramp` in the signal's units with the `anneal` shape —
  `[16, 0, 0.5]` walks a 16-head gate's kept count from all to none over the
  first half of the run, then holds — so "sweep the sparsity from everything
  patched to nothing patched at a steady pace" is a declaration, not a loop.
  The law, with `e_t = signal_t − setpoint_t`:

  ```
  u_t   = kp · (e_t − e_{t−1}) + ki · e_t + kd · Δ(e_t − e_{t−1})     (kd clipped at ±d_clip)
  log w ← clip(log w + u_t, log bounds)        space: log (default)
  w     ← clip(w + u_t, bounds)                space: linear
  ```

  This is a PID controller on the kept count: its
  `count_error` is `e_t` and its `rate_error` is `e_t − e_{t−1}`, sign
  included. The integral term is the count error itself and cannot wind up
  past the unit count; log space makes the gains relative to the weight's
  magnitude. `gains` are sweepable; `kind`, `signal` and `setpoint` are not
  (they say what is controlled, not how hard). `kd`, `space`, `bounds` and
  `d_clip` default to `0.0`, `log`, `[1e-8, 1e8]`, `5.0` and are materialized
  in the canonical form, so two spellings of one controller digest alike.
  The controlled value's start, end, last signal and setpoint and update count
  are recorded in `fit_diagnostics.json` under `controls`, per target; the
  per-update trace is what a `trajectory` checkpoint (sec. 2.12) records at
  its step. Engine-neutral: `causalab/neural/shared/training/control.py`.
- **`phases` — the fit in windows.** One `train.params` set for the whole run
  cannot say "gate only for the first tenth, then both, then the rotation
  alone at a frozen hard mask" — the shape every joint rotation-plus-gate
  fit asked for, and the warm-ups Edge Pruning and budget-mask fits run. `phases` says
  it: each entry owns the updates from the previous entry's end to its
  `until`, and within it (i) only its `params` step — the other entries'
  tensors take no gradient and their groups' `lr` / `weight_decay` are `0`,
  the groups themselves kept, so Adam's moments survive the boundary and a
  later phase resumes rather than restarts; (ii) its `optimizer` may set
  `lr` / `weight_decay` for its params (a scalar or a mapping over them),
  else the top-level values apply; (iii) its `anneal` schedules — the same
  two spellings — run over the phase's own steps, on featurizers the phase
  trains or on a named term's weight, and may not name a path the top-level
  `anneal` or a `control` already moves; (iv) `freeze_masks` snapshots each
  named gate's hard mask at the phase's first update and every forward in the
  phase, training mode included, uses that split — the fix for
  **co-adaptation**: a rotation trained under a fractional mask and scored
  through the hard one it never saw. A gate the phase trains cannot be
  pinned. A `frac` end is `round(frac · updates)` with the last phase taking
  the tail; a phase that resolves to no update, or an `updates` partition
  that does not end at the run's length, is refused before the first step.
  A `trajectory` checkpoint records `phase` (the window's index) beside its
  step, and `fit_diagnostics.json` records each phase's `start`, `end`,
  `params` and `freeze_masks` under `phases`.
- **Only a named term's weight can be swept.** A positional term's weight
  sits inside a list and has no name identity (sec. 3); in the named form it
  is `train.objective.<name>.weight`, a field of a named entry, so the penalty
  grid of a many-gate fit is one axis. Term names are local to `objective`
  (they address nothing outside it) and part of the canonical form, like every
  other entry name.

### 2.12 `save`

Mandatory, non-empty, the last section. The **complete manifest** of
everything that leaves the run. Three saveable values, two entry shapes, plus
the non-value kinds below:

| saved | entry |
|---|---|
| read / metric | `{"value": name, "model": …, "input": …, "file_path": …}` |
| trained featurizer | `{"value": name, "site": …, "file_path": …}` |
| a non-value kind | `{"kind": …, "file_path": …}` |

A **non-value kind** saves something the run derives rather than a declared
value, so its entry names no `value` and no binding; the kind is the whole
entry. Closed vocabulary:

| kind | writes | why it is here |
|---|---|---|
| `location_ledger` | the run's resolved token indices as a table (sec. 6): one row per (example, edit group, constituent, side, token index, token id, decoded token), plus the point's `point` digest and `coords` — a `.json` array of row objects | resolved indices are derived, never authored (sec. 7); this is the auditable record of the derivation. **Opt-in**: a run writes one only when the document saves this entry, and stamps its digest as `location_ledger_sha256` on every artifact it writes (sec. 8) as provenance — recorded, never compared: a run that loads a fitted parameter and saves its own ledger records the tokens it selected on its own rows. At most one entry per document (rule 10) |
| `trajectory` | every trained featurizer's slots at **checkpoints along the fit**, as one `.safetensors` bundle keyed by the featurizer and the `step` — `theta[featurizer=gate,step=40]`, `theta[l1=0.1,featurizer=gate,step=40]` under a sweep — each entry stamped with the featurizer's ArtifactIdentity and carrying, as numbers, the checkpoint's `step`, `epoch`, `loss`, every objective term's value and live weight (`term.<name>`, `weight.<name>`), a `constraint` term's density and dual pair (`term.<name>`, `lambda1.<name>`, `lambda2.<name>` — and no `weight.<name>`, it has none), every trained gate's `hard_mask_size` / `decisive_fraction`, and every controlled hyperparameter's value (`control.<target>`). Takes `every`: `{"count": n}` (n checkpoints equally spaced, the last at the end) or `{"updates": n}` / `{"epochs": n}` | a sweep that moves a mask from everything patched to nothing patched is a *sequence* of masks, and only the last would otherwise leave the run. A checkpoint is loadable exactly as a fitted bundle is — `{"file_path": "…/trajectory.safetensors", "entry": {"step": 40}}` on a `file_path` featurizer or a gate `init` (sec. 2.5) — through the same identity checks; a fit that trains several featurizers photographs each at every step, so a consumer names its own, `{"entry": {"featurizer": "gate_3", "step": 40}}`, and a bare `{"step": 40}` is refused as ambiguous (sec. 5 rule 15) rather than resolved to whichever was written last; the scalar trace is readable from the header without loading a tensor. Needs `train`; at most one entry per document (rule 10) |
| `rank` | every gate the point built — trained or loaded — as one row per **unit**: `featurizer`, `unit` (a flat index into `theta`: a head on a `head` gate, an `(expert, neuron)` entry on an `expert_neuron` one, an addressed token position on a position gate (sec. 2.5 `axis`), else a coordinate), `theta`, `rank` (the unit's position in the gate's descending-`theta` order, `0` = kept first, ties toward the lower index), `hard` (whether the eval-mode split keeps it), `parametrization`, `axis` (which axis `unit` indexes — `null` for a feature gate), `top_k` (the count the split was cut at, `null` under the map's own threshold), `pool` and `pool_rank` (a budget pool's name and the unit's position in the *pooled* ranking, `null` off a pool), plus the point's `point` digest and `coords` — a `.json` array of row objects | the ranking is the artifact of every method that orders units — a `top_k` readout (sec. 2.5), a budget gate, a magnitude or attribution order — and the hard mask is one cut through it; without the table a reader has the cut and not the order, and cannot tell a unit dropped at rank 5 from one dropped at rank 500. A fit's own hard split and a `top_k` replay of the same bundle write the same `rank` column, so the two readouts are comparable row for row. Needs at least one `gate` featurizer (rule 10); at most one entry per document (rule 10) |

- `model`/`input` (resp. `site`) **restate** the binding resolved from the
  declarations and are **cross-checked** — mismatch is a load error. They are
  drift-protected documentation, never a second source of truth.
- `file_path` is relative to the run's output directory. **JSON and
  safetensors are the only two formats**: dense numerics → `.safetensors`,
  per-example metric tables → `.json`, an array of row objects, every row
  `{example_id, metric, value, [step, matched], …coords, unit,
  estimand_version, eligible, [reason_code], produced_by}` — `example_id` the
  base row's label (sec. 2.2), `produced_by` the point digest; together with
  `metric` and `step` they key a row. Row labels
  repeat — that is the deliberate trade for a file `jq` and a human can both
  read. One file per metric, so a document saving three metrics writes three
  tables. In swept documents
  the path is unchanged; axis coordinates become columns / keyed entries
  (`weight[k=8,seed=0]`, one record per entry in the header's `entries`
  table — sec. 8). A read whose scoped slice selected no rows is still
  written, and its entry record carries the `unavailable` fields (sec. 4.1).
- **`reduce`** (optional, reads only): save a statistic over the read's
  gathered rows instead of the rows. Every verb collapses `(…, width)` to
  `(width,)`, the broadcast form a write operand takes (sec. 2.8) — mean
  ablation is a harvest with `reduce: mean` feeding a `params` constant. The
  reduction happens **where the rows are gathered**, so the un-reduced harvest
  never reaches disk: an ablation grid over an 8B model's layers is gigabytes
  of activations for kilobytes of means. A metric already reduces its read,
  and a featurizer bundle holds fitted parameters rather than rows; `reduce`
  on either is a load error. A reduction *after* the rows are on disk — over a
  saved table, with a declared statistical unit, grouping, missing policy and
  interval — is the workflow spec's `reduction` contract (workflow §2.6), not a
  `reduce` verb: the two run at different times, and neither vocabulary grows
  for the other.

  Closed vocabulary:

  | verb | value per column | why it is here |
  |---|---|---|
  | `mean` | arithmetic mean of the rows | mean ablation, and the default summary |
  | `sum` | sum of the rows | the numerator of a mean composed across points or shards — a mean of means is wrong once the point row counts differ |
  | `std` | **sample** standard deviation (`n−1`) | the spread the mean hides. One row gives `NaN`: the spread of a single observation is undefined, and `0.0` would read as "no variation" |
  | `median` | median, **no interpolation** — the lower of the two middle values at even `n` | survives an outlier row the mean does not, and the saved value is one a row actually held |
  | `count` | how many rows were reduced, broadcast to `width` | the denominator that makes `sum` composable, and the record of a truncated or ragged harvest that a `mean` alone hides |

  The accumulation is fp32 whatever the run's dtype: a bf16 sum over thousands
  of rows loses the low bits it exists to average. Over **zero** rows — an
  `unavailable` cell (sec. 4.1) — `mean`, `std` and `median` are `NaN` and
  `sum` and `count` are `0`: the statistic of no observations is undefined,
  and the pair that composes still composes.
- Rules (all load errors): every metric saved (no objective/eval exception —
  the loss trajectory is always in the results) · every trained featurizer
  saved · untrained or `file_path`-loaded featurizers not saveable · writes
  and intervened_models not saveable.

#### Adding a `reduce` verb

The vocabulary is closed and grows by PR. Five steps, and the last two are
what make the vocabulary a record rather than a habit:

1. Add the verb to `SAVE_REDUCTIONS` (`causalab/protocol/schema.py`).
2. Add one branch to `_reduce_rows`
   (`causalab/neural/shared/outputs.py`). It must return a `(width,)` fp32
   tensor: reducing anywhere but where the rows are gathered defeats the
   point.
3. Add a row to the table above, saying what the value is *and* why the verb
   exists. Where a verb has a choice to make — an estimator's correction, an
   interpolation rule, tie-breaking — the row states which one, because a
   reader cannot recover it from the name.
4. Add a test that the un-reduced rows never reach disk: the reduction's
   output is `(width,)`, not `(n, width)`.
5. Nothing else. The guard in `tests/protocol/test_vocabulary_census.py`
   fails if steps 1 and 3 disagree, so the table cannot silently fall behind
   the code.

## 3. Sweeps

- **Every axis is an explicit wrapper** on a field of a named table entry:
  `{"sweep": [v1, v2, …]}` or `{"sweep": {"range": [start, stop, step?]}}`.
  Bare arrays are never axes. Works on scalar- and list-typed fields alike.
- **Axis identity = name identity**: sweeping `sites.target.layers` moves every
  read/write/metric referencing `target` together. The same list written on
  two fields is two axes (a cross) — share by referencing one name.
- Axes **propagate through the reference graph**; entities off the axis stay
  singletons shared by all points.
- Multiple axes form the **cross product**; nothing else is zipped here.
  Correlated rows and dependent axes are the `axes` group (sec. 3.2): a named
  axis declared once, referenced at its fields by `{"axis": …}` wrappers and
  lowered to sweep wrappers before the shape gate — nothing else is zipped,
  and there are no conditionals.
  - ⚠️ **`data.base.dataset` and `data.counterfactual.dataset` are two names,
    hence two axes.** Sweeping both over the same *n* refs is an *n*×*n* cross
    that pairs every base table with every counterfactual table, and rows are
    paired **across roles by index** (sec. 2.2) — so *n*²−*n* of those points
    silently score mispaired rows rather than failing. One campaign hit the
    4096-point cap this way and read the cap as the symptom. There is no zip to
    reach for: the intended shape is **one table carrying both sides**, base
    reading `input` and the counterfactual role reading
    `counterfactual_inputs[j]` of the *same* ref — which is what every shipped
    preset does, and what leaves one axis to sweep. When the two sides really
    are separate files, that is one document per pair (or one `--set` per
    shard), not one document with two axes. Two fields sharing one axis is a
    `rows` axis (sec. 3.2), and it does not make two tables one: the intended
    shape stays one table.
- **A shared penalty is one axis.** A regularizer that names many
  featurizers (sec. 2.11) has one weight; written in the named form of
  `objective`, that weight is `train.objective.<name>.weight` and sweeping it
  moves every listed gate's penalty together — the sparsity curve of a
  forty-layer DBM fit is one axis with one coordinate column, not forty
  crossed ones.
- Coordinates suffix derived names (`rot[k=8]`) and key results.
- Expansion is **deterministic at load**: one document ⇒ one compiled
  intervention per point. The document digest names the campaign; each
  point's digest is the provenance unit. The planner content-dedups sub-values shared across
  points — shared harvests and forwards fall out automatically (identical
  reads intern to one read).
- **The planner derives the sharing; the engine claims it.** A forward
  group's `digest` is the identity of everything determining its activations
  — which includes **how the network is realized numerically**, not only which
  network it is: the group carries the whole of sec. 2.1's canonical `model`
  (`key`, `revision`, `dtype`, and the normalized `quantization` block), by the
  same function the canonical form uses. An fp32 harvest and a bf16 harvest of
  one address are different tensors, so they are different content and must not
  intern together. The same is true of a read's closure digest.

  The group likewise carries, per input role, the **content digest of the rows
  that role reads** — sec. 2.2's digest over the selected rows, as the canonical
  form stamps it — plus the field the role tokenizes, and **never the ref's
  name**. Two tables under one name therefore never intern, one table under two
  names does, and two splits of one table are two identities. The same digest
  is what a fitted bundle records as its trained-on digest (sec. 8), so the
  interning identity and the provenance identity name a table the same way.

  **Taps are deliberately not in it** — reading layer 3 or layer 23 of the
  same un-intervened forward is the same forward. So a 32-layer scan's
  counterfactual harvest is *one* group carrying 32 taps:
  `interned_groups(plans)` counts what a campaign owes (65) against the
  `sum(num_forwards)` a per-point loop pays (128). Spending that is execution's
  job, not the plan's — run each distinct digest **once**, capturing the
  **union** of the taps every point asked of it, and let each point gather and
  featurize its own value out of that one capture. The reference engine does
  this and reports what it ran as `RunResult.forwards`. The trade is memory: a
  shared pass holds every tapped address at once where a per-point loop held
  one — but only for as long as a point is still owed it. A capture lives from
  its first sharer to its last (the plan's count of instances per digest says
  when that is), and a group no other point shares is never kept at all: a
  swept *patched* forward tapping `lm_head` is a full-vocabulary tensor per
  point, and keeping one per point until the request ends is what a per-point
  loop never did. An engine that interns nothing is still correct, only
  slower — it leaves `RunResult.forwards` at 0, which reads as "not measured".

  The same store serves a **fit's** inner passes (sec. 4, "Fits"): a group
  no trained parameter can reach — `original`, and any intervened model
  whose writes are fed by nothing the fit moves — runs **once per (digest,
  row slice)** for its minibatches and once for the `train.eval` split, and
  is served from that raw capture on every later step, epoch, eval pass and
  point sharing the digest. Those passes are not forward groups and are never
  counted in `RunResult.forwards`; a group the fit *does* change is never
  cached inside it.

### 3.1 `at_once` — one axis inside one point

A sweep axis denotes independent points, so it cannot say "these ten layers are
patched during the **same** forward". `intervened_models` says *which* writes are
in force together (§2.9); `at_once` is how the table they come from is declared
without writing one entry per index by hand.

- **One field of one entry** in `positions`, `sites`, `featurizers`, `params`,
  `reads` or `writes` may be wrapped `{"at_once": [v1, v2, …]}` or
  `{"at_once": {"range": [start, stop, step?]}}` — **the same value grammar
  `sweep` uses, and the same validator**, so there is one notion of "a list of
  values" and one place it is checked. The entry then denotes one entry per
  value, all of them present in the same point.
- **Members are named** `a[layers=10]` — §3's derived-name convention — or by an
  optional `names` template naming exactly the axis field, `"a{layers}"`. The
  template is what lets an existing file adopt `at_once` without moving its
  digest, which is why it is worth having.
- **Fan-out is by reference**, the same name-identity rule §3 states for sweeps:
  an entry referencing a family fans out over its axis, and member *i*
  references member *i*. One `reads` entry over a site family is a family of
  reads; one `writes` entry over both is a family of writes.
- An `intervened_models` write list may then **window** a family, in the words
  that declared it: `{"w": {"layers": {"at_once": {"range": [10, 15]}}}}` is the
  five writes of that band, and `{"at_once": [10, 12, 14]}` three that need not
  be adjacent. A window is an `at_once` axis over the family's field, so it
  takes the axis's grammar and nothing else — no second spelling for the
  subset. A bare family name is every member and follows the family if it
  grows; a window is the explicit form and refuses to resolve to a different
  number of writes than the values it names.
- **On `layers` the wrapper composes with the band** (sec. 2.4): an `at_once`
  axis on a site's `layers` is indexed by layer — each member value is a layer
  index and denotes the one-layer band `[n]`, so the example below is ten
  one-layer sites in one point (N sites, N reads, N writes, one forward), where
  a single site with `"layers": [10, …, 19]` is one site across ten layers (one
  read, one write). A member value that is itself a list is a band site per
  member, labelled `10..11`.

```json
"sites":  {"a": {"component": "attention_output", "layers": {"at_once": {"range": [10, 20]}}, "names": "a{layers}"}},
"reads":  {"v": {"site": "a", "pos": "tap", "model": "original", "input": "counterfactual", "names": "v_a{layers}"}},
"writes": {"w": {"site": "a", "pos": "tap", "do": {"swap": "v"}, "names": "w{layers}"}},
"intervened_models": {
  "band5_L10": {"input": "base", "writes": [{"w": {"layers": {"at_once": {"range": [10, 15]}}}}]}
}
```

**`sweep` multiplies runs; `at_once` multiplies declarations.** Both declare an
axis with the same value grammar, and the difference is the denominator: a swept
layer list is *N* compiled interventions, an `at_once` layer list is *N*
addresses inside one. So if the values do not have to be in force at the same
time, **sweep them** — the planner interns the shared forward anyway (§3), so a
32-layer harvest is one forward either way, and a sweep keeps the coordinates in
the results table instead of multiplying declarations. Reach for `at_once` when
one forward has to carry them all.

**It is sugar, expanded before anything reads the tree** (§7): the parser, the §5
checklist, sweep expansion, the canonical form and the digest all see exactly the
document the author would otherwise have written out. So rewriting a
specification's tables in `at_once` compiles to the same bytes and costs no
digest — the mechanism is free. What that does *not* cover: a v1 digest is over
the whole canonical form, `description` included, so a file that also rewrites
its prose moves, for the prose.

What it refuses, each because the alternative is a number attributable to
nothing (rule 28):

| refused | why |
|---|---|
| two `at_once` fields on one entry | a member would have two indices and no name |
| an entry on one axis that references a family on another | a cross product inside one forward is not what a family means — that is what sweeping is for. Checked for an entry that declares its axis and one that inherits it alike |
| a family entry that also carries a `sweep` wrapper, anywhere in it | expansion copies the wrapper to every member, and a sweep axis *is* its path (§3) — so that is one axis per member, and their cross product. Ten members is 2¹⁰ points, under the point cap, so nothing later would catch it |
| a window naming an index the family does not carry | a band that resolves to fewer writes than the interval it declares is the silent bug the window form exists to refuse |
| an empty window, `{"at_once": {"range": [15, 10]}}` | the same bug from the other side: it names no write at all, so the band compiles as the un-intervened model and its metric reports no effect. A window is an axis, and the axis grammar refuses an empty one (§3) |
| a window spelled any other way — a bare list, a bare `range`, an interval keyword | one word per object (§11.1): the subset is written the way the axis was |
| a family of more than **1024** members | a sweep's per-axis bound is safe because the point cap refuses the expansion afterwards; a family materializes entries directly, with nothing behind it. A table of that many addresses in one forward is where the answer is a sweep |
| a swept write list that names a family | which member it means would depend on the point, and families materialize before axes are found |

A family *of* intervened models — an `at_once` on a model entry's own field —
is not in this version and is refused by name; it is told apart from a window
by position, since a window sits inside a write-list item under the family it
selects from. `metrics`, `save` and `train` do not fan out in this version, and
a family reaching any of them is refused by name: a saved family needs a rule for
per-member `file_path`, and §2.12's answer for swept documents — coordinates
become columns of one table — is very likely the better one, so it is its own
change. `train` is included for that reason and one more: it references entries
by *param slot* — `params`, a regularizer's names, and `anneal`'s keys, the last
of which are the mapping's *keys* rather than its values (a
`train.objective.<name>.weight` key names a term, not an entry) — so a fit over a
family is the shape most likely to look as though it had worked.

**One shape is inexpressible rather than unimplemented**, and it is worth
knowing which: axis identity is the field name plus the values, so two families
align only if they share a field. A site family on `layer` and a position family
on `index` cannot be *zipped* into a diagonal (layer 10 at −4, layer 11 at −3) —
and sweeping the second axis, which is what the refusal advises, gives
independent points rather than that diagonal. Emit the entries from a generator
if you need it inside one point; *across* points, the diagonal is a `rows` axis
(sec. 3.2).

### 3.2 Path blocks — path patching as a compile into the existing nouns

A path-patching experiment (`IOI`: sender → receiver, with everything off the
path held at its clean value) is written by hand as five kinds of entry: the
sender and receiver sites and one restorer site per layer in between; a read of
each; the sender swap and one freeze per restorer; the read of the receiver
under all of that and the write that injects it into an otherwise clean run;
and the two intervened models that group the writes. The shipped
`configs/protocols/path_patching.json` is those twenty entries. Only five of
them are choices — the sender, the receivers, the position, which data role the
sender's value comes from, and **which sites are restored** — and the last is
invisible in the hand-written form: it *is* which sites happen to appear in
`writes`.

A **path block** names exactly those choices, under `method`:

```json
"path_patching": {
  "sender":      {"component": "attention_premix", "layers": [9], "head": 9},
  "source":      "counterfactual",
  "receivers":   [{"component": "block_input", "layers": [12]}],
  "pos":         -1,
  "restoration": "attention_only",
  "harvest":     "patched",
  "inject":      "final"
}
```

| key | required | is |
|---|---|---|
| `sender` | ✓ | an inline site (sec. 2.4) whose `layers` names **one** layer, `L` or `[L]` — the one-layer band (one digest either way); a band of several layers is refused, below |
| `source` | – | the data role the sender's value is read from (sec. 2.2); default `counterfactual` |
| `receivers` | ✓ | a **non-empty, ordered list** of inline sites, each a one-layer band (`layers: L` or `[L]`) strictly above the sender's — injected together, in one intervened model |
| `pos` | ✓ | **one** position in the sec. 2.3 grammar — an integer, a `positions` name or an inline spec — copied verbatim into every emitted read and write |
| `restoration` | ✓ | `attention_only` or `attention_and_mlp`: which sites between sender and receivers are held at their clean value |
| `harvest`, `inject` | – | the names of the two intervened models; default `patched` and `final`, so a hand-authored `reads.logits.model` can name them |

**It is a compiler into the existing nouns, not a second dialect.** Every
field names something this specification already has; the block's own two
words (`restoration` and its policies, `receivers`) are the compiler's input
and nothing else's. It has no positions vocabulary of its own and takes no
list of layers to restore — the freeze layers are *derived* from the sender and
the receivers. **A path runs between two layers**: a sender or receiver site is
the one-layer band of sec. 2.4 (`layers: L` or `[L]`), and a band of several
layers is refused by name rather than lowered to a guess (the refusal table
below). The block is lowered by the `paths` compile stage (sec. 9.1), after
`families` and before the shape gate, so the parser, the sec. 5 checklist,
sweep expansion, the canonical form and the digest see **exactly the document
the author would have written by hand**. The shipped preset, re-expressed as
the block above, compiles to the same canonical bytes and the same digests
(`tests/protocol/test_paths.py`); the block appears in no canonical form and
moves no pin.

**What it emits**, in the shipped document's own names — the convention:

| from | sites | reads (`model`, `input`) | writes | in |
|---|---|---|---|---|
| the sender | `sender` | `v_sender` (`original`, `source`) | `swap_sender` = swap `v_sender` | `harvest` |
| each restored layer *L* | `a{L}` = `attention_output@L`, and under `attention_and_mlp` `m{L}` = `mlp_output@L` | `v_a{L}` / `v_m{L}` (`original`, `base`) | `freeze_{L}` / `freeze_m{L}` = swap the clean value | `harvest` |
| each receiver | `receiver` (one member) or `receiver_0`, `receiver_1`, … (several) | `v_receiver[_i]` (`harvest`, `base`) | `inject[_i]` = swap `v_receiver[_i]` | `inject` |

The two intervened models are `harvest` = `{input: base, writes: [swap_sender,
freeze_*]}` and `inject` = `{input: base, writes: [inject*]}`. Every emitted
site — the sender and receivers as normalized, each restorer — is written in
the canonical one-layer band form of sec. 2.4, `{"component": …, "layers":
[L]}`, so at its sites the lowered document is byte-for-byte what a
hand-written v3 document canonicalizes to, whichever spelling the author used.
Generated entries merge into the authored tables of the same name — generated first, the
author's after; an authored table the block does not touch (`lm_head`,
`logits`, the metrics, the manifest) is left as written — and a table the
author did not write is inserted where sec. 1 puts it, so the lowered document
raises no order warning the authored one did not.

**The freeze rule.** With sender layer *s* and farthest receiver layer *r*,
`attention_only` restores `attention_output` at every *L* in `[s+1, r)`;
`attention_and_mlp` additionally restores `mlp_output` at those layers **and at
the sender's own layer *s***: the sender's block's MLP reads the residual
stream after an attention sender has written into it, so it is downstream of
the sender and off the direct path. This is a choice, stated here and recorded
as data in the derived record (`restored`), not something the block decides
silently. A receiver one layer above the sender therefore restores nothing
under `attention_only` and the sender's own MLP under `attention_and_mlp`. The
restorers' **order** — which the canonical form does not carry (an intervened
model's write list is sorted, sec. 7) and execution does not read from the
document (writes install in module order) — is recorded as the **restorer
boundary**: `[layer, component, site]` per restored site, ordered by
`(layer, COMPONENT_RANK[component])`, the order the forward computes them in.

**Ordered receiver sets.** `receivers` is ordered data, every member is
injected in the **one** `inject` model — one forward, one joint intervention,
never a sum of separate runs, which is not equivalent in a nonlinear model —
and a one-element set is spelled `receiver` / `v_receiver` / `inject`, so it is
the scalar document to the byte. The order is data: it numbers the members
(`receiver_0` is the first authored), so the same sites in another order are a
different document, and the derived record repeats it (`receivers`) so no
reader has to parse names; the intervened model's own write list is sorted in
the canonical form (sec. 7).

**The restoration policy is part of the estimand** through the write set it
generates: two blocks differing only in `restoration` lower to different
`writes`, so their canonical bytes, point digests and `produced_by` differ
(sec. 7), and a comparison across them fails its provenance binding rather
than being silently made. The policy is not encoded in `estimand_version`,
which names a metric's arithmetic and never an intervention choice (sec. 2.10).

**The compiled form is saved.** The lowering is a *derived* record (sec. 6),
never a canonical section: `compile_protocol` returns it as its `lowered`
output (sec. 9.1) — `{"path_patching": {authored, restoration, source, pos,
harvest, inject, sender, receivers, restorers, restored, emitted}}` — `explain`
prints the receivers and the restorer boundary, and the run receipt writes it
as `derived` beside the canonical form whose keys `emitted` names and the
digests over that form. Two runs that both say "path patching" are compared
there, field by field. A document without a block has `lowered == {}` and no
`derived` key.

**What it refuses**, at the `paths` stage, each naming
`method.path_patching.<field>` with a parser code — no numbered rule, because
what the block cannot say is a shape property of the block and what the
*lowered* document gets wrong is the checklist's already:

| refused | code |
|---|---|
| a missing `sender`, `receivers`, `pos` or `restoration`; a site whose `layers` is missing or not one integer layer (`[]`, a non-integer member, a float, a boolean) | `P2` |
| an unknown key in the block or in one of its sites (the v2 `layer` is one: the suggestion names `layers`) | `P3`, with a suggestion |
| an unknown restoration policy, or an unknown component | `P4`, with a suggestion |
| **a band sender or receiver** — `layers` naming more than one layer. A path runs between two layers: a band has no defined freeze range, and reading it member-wise is a *set* of paths, which the author spells with one block per layer. Refused in one place (`paths._site`), so the decision can be reversed there | `P4`, at `sender.layers` / `receivers[i].layers`, naming the mechanism |
| an empty receiver list | `P2` |
| a path that runs upstream — a receiver at or below the sender's layer (a same-layer receiver is not in this version) | `P2` |
| a `sweep` wrapper anywhere inside the block — the freeze set depends on the layers, so a wrapped layer would need one lowering per point; wrap the emitted entries instead | `P2` |
| an `at_once` wrapper anywhere inside the block — the same reason, but the `families` stage runs first and walks every method key, so it is refused there as rule 28: the wrapper sits where it has no name identity | rule 28, at `families` |
| **two emitted sites at one address** — the lowering hands every site its own name and rule 8 compares absolute writes by site *name*, so none of these would be caught downstream: a duplicate receiver (two absolute swaps at one address in one model; a whole-component site covers each of its heads), a receiver at a site the policy freezes (`attention_output@L` for *s* < *L* < *r*, or `mlp_output@L` under `attention_and_mlp` — it would read the frozen clean value and inject clean for clean), or a sender inside the MLP of its own layer under `attention_and_mlp` (`mlp_input`, `mlp_activation` or `mlp_output` at *s*: the clean `freeze_m{s}` would overwrite everything it wrote, a path effect of exactly zero). Named at the `sender` or the later `receivers[i]`; the restorers are derived and never blamed | `P2` |
| one name for both intervened models (`harvest` = `inject`) — they would collapse into one entry and the harvest model would vanish | `P2`, at `inject` |
| a generated name the author also declares in the same table (at `harvest` / `inject` when that field chose the name) | `P2` |
| an authored `method.<table>` that is not an object where the block emits into it — refused by the table's path, before the derived record is built | `P2` |
| a generated name the author declares in another section | rule 3, on the lowered document |

`--set` **cannot address the block**: overrides are section-rooted (sec. 1)
and the block is not a section, so `--set path_patching.restoration=…` is
refused as a path that does not exist. Retarget a block document by editing
it; sweeping a field of the block, a per-receiver `pos`, a receiver in the
sender's own layer and a numbered rule for the upstream check are deferred
deliberately — each would make this a digest mover.


### 3.2 `axes` — correlated rows and dependent axes

A sweep wrapper is one field, one axis, and two wrapped fields are two axes: the
cross product. Two experiments need more than that. A study's three candidate
locations are each a (`layers`, `component`, `pos`) tuple whose fields move *together* —
three rows, not the twenty-seven points three independent axes would give — and
ROME's restoration window is a ten-layer band *computed from* the centre it is
swept over, clipped to the tower. Both are declared once, by name, in an
optional fifth top-level group, `axes` (§1), and referenced at the fields they
move by a wrapper legal exactly where `{"sweep": …}` is: `{"axis": "<name>"}`
for an axis that is one value per entry, `{"axis": "<name>.<field>"}` for a
field of a correlated row. §3's first bullet stays true — you can see at the
field that it varies — and the row fields spell the *target* field's name
(`layers`, `component`, `pos`), so a row's `layers: 8` is the one-layer band
`[8]` exactly as it is on a site (sec. 2.4):

```json
{
  "axes": {
    "location": {
      "rows": [
        {"layers": 8, "component": "attention_output", "pos": {"index": -1}},
        {"layers": 9, "component": "mlp_output", "pos": {"index": -1}},
        {"layers": 10, "component": "block_output", "pos": {"index": -2}}
      ],
      "key": "layers"
    },
    "center": {"range": [0, 48]},
    "window": {
      "dependent_on": "center",
      "rule": {"clipped_band": {"width": 10, "clip_to": "layers"}}
    }
  },
  "method": {
    "positions": {"tap": {"axis": "location.pos"}},
    "sites": {
      "target": {
        "component": {"axis": "location.component"},
        "layers": {"axis": "location.layers"}
      },
      "restore": {"component": "block_output", "layers": {"axis": "window"}}
    },
    "train": {"seed": {"sweep": [0, 1]}}
  }
}
```

Three kinds of axis, one closed vocabulary — the key a declaration carries is
the kind it is, and it carries exactly one of them
(`causalab/protocol/axes.py` `AXIS_KINDS`; `tests/protocol/test_vocabulary_census.py`
holds this table to it):

| key | coordinate | lowers to |
|---|---|---|
| `rows` | one per row: the scalar field `key` names, else the row's index. A **non-empty list of objects with one key set**; every field but `key` is referenced by some `{"axis": "<name>.<field>"}` wrapper, and the fields of one row are substituted **together** | at each referencing field, `{"sweep": [that field's column]}` |
| `range` | the value: `[start, stop, step?]` of integers, half-open, non-empty | `{"sweep": [the values]}` |
| `values` | the value: a non-empty list of **distinct scalars** — a tuple of fields is a `rows` axis | `{"sweep": [the values]}` |
| `dependent_on` | **none of its own** — its parent's coordinate already names the entry; the value is in the point. Names a `range` or `values` axis and a `rule` | `{"sweep": [one computed entry per parent value]}` |

The rules a dependent axis may follow (`RULE_KINDS`, censused likewise):

| kind | means |
|---|---|
| `clipped_band` | `{"width": w, "clip_to": "layers"}`: for a parent value *c*, the band `[max(0, c − ⌊w/2⌋) … min(L − 1, c + ⌈w/2⌉ − 1)]` over the model's *L* layers, read from the registry entry of the document's one `model.key` — so ten wide, centre 0 is `[0..4]` and centre 47 of 48 is `[42..47]`, and no member is ever outside the tower. `clip_to` is `layers` and nothing else in this version |

**Expansion order.** The named axes are the *slowest* coordinates, in
declaration order; inside each entry the ordinary sweep axes expand as §3 says,
last axis fastest. So the example's `location` × `train.seed` is exactly six
points, `[axes.location=8, seed=0]`, `[8, 1]`, `[9, 0]`, … — never the
3·3·3·2 = 54 the wrappers would give as independent axes — and the 48 centres
are 48 points, each with one band site. A point reached this way *is* the
hand-written point: its tree is that tree, so its canonical form and its digest
are that point's, as for every other point (sec. 7).

**Coordinates.** Exactly one per named axis, `axes.<name>` (the group is spelled,
since `axes` is not a method section): a `rows` axis records its `key` value, a
`range` / `values` axis its value; a dependent axis and a row's substituted
fields record nothing — no list or object ever becomes a coordinate. Labels
follow (`[axes.center=5]`, sec. 3); existing labels are untouched.

**Canonical form and digests.** The compiler lowers the group before the shape
gate (`axes` stage, sec. 9.1), so the parser and the checklist
see a swept document any author could have written by hand — the **display
form**. But that form's cross product is not the campaign, so the campaign's
canonical form carries the `axes` block, **digest-bearing when authored**
(sec. 7): rows recorded folded — each field the value its display column
carries, so `layers: 8` and `layers: [8]` are one row and one digest (with
`key` when authored) — a `range` materialized to its values, a dependent axis
with its rule as authored and the entries it computed. Every document without
the group keeps its canonical bytes; no version bump.

**The point cap** (rule 14) is taken over the true count — entries × the inner
cross product — after correlated rows are applied, and it names that count.

What it refuses, each with a valid twin that compiles (`P2`/`P3`/`P4` on
`axes.<name>…`, or at the wrapper's own section-rooted path; no rule number —
these are authoring-shape facts decided before the gate):

| refused | why |
|---|---|
| a row that is not an object; an empty row; rows whose key sets differ | a row *is* the axis, and every entry covers the same fields — a missing field would silently keep the document's value for that row |
| a `key` that names no row field, a non-scalar key value, or a repeated one | the key is the row's coordinate: a scalar, one per point |
| two identical rows, compared folded (`layers: 8` is `[8]`), keyed or not | rows are distinct: two identical rows would be two points with one name — `P2` at the repeated row, naming both indices |
| a `sweep`, `at_once` or `axis` wrapper inside a row value or a `values` list | an entry's value is a value |
| `{"axis": …}` naming an undeclared axis or row field; a rows axis referenced without a field; a `range` / `values` / dependent axis referenced with one; a wrapper inside a list | name identity: a reference names one declared column, and one inside a list has no name (sec. 3) |
| a declared axis nothing references (and that no dependent axis follows); a row field no wrapper reads, unless it is the `key` | the typo that would otherwise multiply points, or drop a field the author meant to move, silently. A wrapper is a mapping holding `axis` **and nothing else**; a mapping carrying `axis` beside other keys is a value (a `gaussian` write's payload, sec. 2.8, is one), so a mis-keyed wrapper is caught here, as the axis it left unreferenced |
| `dependent_on` a `rows` axis, or an undeclared one | a rule takes one scalar per entry; a row's fields already move together — compute the value and write it as one more field |
| an unknown rule kind — `routed_experts`, "one intervention per routed expert selected for that row" | `P4`, naming **run-time fan-out**: routing is known only after a forward, and expansion is a pure function of the document (§3). That fan-out is the workflow's (`fan_out`), never a compile's. Likewise "all sites causally later than each row's changed token" splits: the layer half is a band anchored at a declared layer (a dependent axis), the position half is per row through the position anchors of sec. 2.3 |
| `clip_to` with a swept `model.key`, or a `clip_to` other than `layers` | one model's layer count bounds the band |
| more points than the cap, counted after the rows are applied | rule 14, with the true count |

Nothing here distinguishes a generated document from a hand-written one: a
document a script expanded over every layer is an ordinary document, and stays
one.


## 4. Execution semantics

- **Models → forwards.** For each expanded point, the models are: `original`
  on every input it is read on, plus each intervened_model. Each (model)
  is one forward group over its input rows; fusion, batching, and staging
  across groups are the engine's choice. `num_forwards` is derived, never
  authored.
- **Within a model**: apply each in-force write at its address (absolute
  first, then additive sum); reads see the fully written state.
- **Across models**: operand values flow along the acyclic model graph;
  the engine stages them (fused multi-pass, saved constants, or microbatch
  wiring — its call).
- **Elision**: a model whose reads are all satisfied may stop its forward
  after the deepest tap; a full-depth pass is never owed. A group that
  decodes is the exception — every step needs the head, so nothing is
  elided. When several points **share** the group (sec. 3), the deepest tap is
  the deepest of the union: the one pass has to serve all of them, so read
  `stop_after` off the interned group, never off the point that ran first.
  Interning still wins against elision — 32 passes elided at layers 0..31 cost
  ~16x the single full-depth pass that replaces them.
  The vocabulary head is the one module past every block, and the reference
  engine elides it on its own terms: an `lm_head` read at named positions is
  served by gathering the head's input (`ln_final`) at those positions and
  running the head module over the gathered rows — the same weights, the same
  arithmetic per logit — so a forward on which nothing reads the whole
  sequence, writes at the head or decodes never runs the head at all. The
  read's value is unchanged; only where the projection is computed moves. A
  read the fit differentiates through keeps the head as the model runs it,
  so the training gradient is the model's to the bit.
- **Resume**: elision's mirror image. Below the first block any of its
  in-force writes lands in, an intervened model's forward *is* the
  un-intervened forward over the same rows, so an engine may **start** it at
  the residual entering the first block any write — or any tap the campaign
  asks of the group — touches, from a cached un-intervened pass over the same
  rows (an `original` group that ran anyway, or a shallower intervened model
  whose pass is still un-intervened down to there), with identical results:
  block `L` receives exactly the tensor it would have computed. The prefix's
  identity is the digest the same input would have under `original`
  (`base_digest`), so every intervened model on one input shares it whatever
  it writes; the depth is read off the interned group like `stop_after`,
  since a tap below the write still needs its block to run. `original` never
  resumes, nor does a group that decodes. Inside a fit this holds for the
  trained model's own group: its prefix is fit-constant even though the group
  is not, so every optimizer step, epoch and eval pass at layer 20 skips the
  twenty blocks below it. A prefix lives with its sharers, as sec. 3 says of
  a capture: it is dropped once every group that could start from it has
  run, and never stored once none is left to. Reported per point as
  `prefix_reuse` in the run summaries; `RunResult.forwards` is untouched — a
  resumed forward is still a forward.
- **Decoding groups.** A group whose reads address the continuation
  (sec. 2.3) runs one prefill plus `max_new_tokens` greedy steps: `n` tokens
  need `n` steps, because the last generated token must be consumed by a
  forward for its own activations to exist. Writes apply in the prefill only.
  The depth is derived from the group's positions, never authored.
- **Cohorts.** Points of one campaign that declare `train` on the same model
  realization over the same rows in the same frame — the network, the data
  identity per input role and the `segments` section; nothing a member
  trains, sweeps or schedules — form a **fit cohort** and fit together. The
  members step in lockstep, and each step is **one** forward over the
  concatenation of their minibatches, every member's writes landing on its
  own rows at its own address (the members may write at different layers),
  the taps the union; each member reads its own rows back out of the
  capture, the summed loss backpropagates into disjoint parameters, and each
  member's optimizer steps on exactly its own gradient. Eval passes batch the
  same way, grad off, among members whose split agrees. Every member keeps
  its own seed (init and minibatch order — members hold different rows of
  the shared frame at one step), step budget, eval cadence, objective,
  optimizer, anneal and `early_stop`: a member whose budget is spent or whose
  patience ran out leaves the batch while the rest go on. Nothing in the
  model couples rows, so each member's fit is the fit it would have run
  alone, up to the rounding of a different batch shape; the reference engine
  pins this on CPU to `1e-5`. Only groups the store cannot serve batch — the
  trained model's; `original` and every fit-constant group are served (sec.
  4, "Fits"), a group that decodes, one fed by a read off a non-constant
  model, or one with a per-step state write runs on its own. Resume composes:
  the cohort forward starts at the deepest block every member holds a prefix
  for and stores each member's wanted depths under that member's key. The
  rows of one grad forward are bounded by the engine's `fit_rows` (a
  member's minibatch is never split; sec. 8) — measured on the cohort's
  first step when none is authored — the eval passes by `batch_rows`. Outputs are the per-point loop's: every point still runs its
  own whole-role passes and writes its own entries, in point order, after
  the cohort has fitted.
- **Fits.** A `train` document re-runs its groups every optimizer step, but
  only the groups a trained parameter can **reach** change between steps.
  The rest — `original` always, and an intervened model whose in-force writes
  go through no trained featurizer, name no `train.params` root, and consume
  reads only off constant models and only unfeaturized by a trained
  featurizer — are *fit-constant*: an engine runs each once per (group
  digest, row slice) of the fit's minibatches and once for the `train.eval`
  split, and serves the raw capture (before gather and featurizer, so the
  trained featurizer still applies to it with gradient) on every later step,
  epoch, eval pass and point sharing the digest (sec. 3). A fit's inner passes
  are not forward groups and are never counted in `RunResult.forwards`; a
  group the fit changes is never cached inside it. Fitted weights are
  unchanged by the reuse — the served capture is what the frozen forward
  would have produced.
- **Fires.** Every write member of an intervened model is installed for one
  forward and *fires* when the forward reaches its module: once per forward
  for a write at a module boundary, an attention-interface slot, the experts
  interior or the DeltaNet kernel boundary; once per **addressed step** for a
  `delta_state` write, which the stepwise substitution hands every step's
  state. The engine counts each member's firings per forward and refuses the
  point — `P4`, reason `component_unavailable`, naming the member, the group
  and both counts — when any member fired other than its declared count.
  Zero is a module the forward never called (the measured `conv1d` shape:
  the forward reaches the weights through a module-global function), so the
  tensor the write addresses did not exist in this forward and the point
  would have scored an un-intervened forward as an intervention; more than
  the declared count is a module the forward visits twice. The unit is **one
  forward of the group**: a group run as several row windows (sec. 8)
  installs the set again per window and each window is checked; a forward
  that resumes from a cached prefix ("Resume") still runs every block a
  write lands in and is one forward. **The write set is one transaction**:
  it is resolved and built whole before any hook installs, every hook
  edits a clone rather than the module's storage, and captures are published
  and tables written only after the forward has returned and every count has
  been checked — so a member that fails to resolve, mismatches its operand's
  shape or fires the wrong number of times refuses the point with no table,
  no capture and no receipt recording a subset. The counts are recorded in
  the run receipt under `fires` (sec. 9), once the whole campaign has run:
  per point digest and forward group (`<model> on <input>`, the ledger's edit
  group, sec. 6), `{write: count}` — a module-kind member's per-forward
  count, a state write's number of distinct steps over the group's rows —
  the same number under any row layout, so two layouts still differ in their
  receipts at `execution.batch_rows` alone. A point served a shared group's
  captures (sec. 3) records the counts of the pass that ran it. A fire count
  is a receipt fact, never a canonical one: it enters no digest and no stamp.
- **Determinism**: `gaussian` draws from its declared seed; sweep expansion
  and canonicalization are pure functions of the document.

### 4.1 Resolution: `available`, `unavailable`, `invalid`

Every resolution a run performs — a component to a tensor, a selector to rows,
a bundle key to an entry — ends one of three ways, kept apart by **type**
(`causalab/protocol/resolution.py`, torch-free):

| value | fields | where it goes |
|---|---|---|
| `available(mapping, denominator_key)` | what resolved | the result, counted as eligible |
| `unavailable(reason, detail, denominator_key)` | a `reason` from sec. 2.4's table, the fact in prose, the key it is counted under | the result **and** the denominator, as an excluded cell |
| `invalid(error_code, detail)` | the existing `V<n>` / `P<n>` code | raised by the validator — never a result |

The rule: **`unavailable` belongs in ordinary results and denominators;
`invalid` stops validation.** Which one a site produces follows from who could
have known: a *document-decidable* defect — a selector that names no entry
(rule 15), an unknown component (rule 4), a non-positive authored width, a
`--points` shard outside the campaign — is `invalid`, with the code it always
had. A *structural fact of the data or the model that the document could not
know* — the router sent an expert no token at the addressed positions, so the
`expert:` face of that component (sec. 2.4) has width zero in this batch — is
`unavailable`: the document was legal, the instrument measured nothing here,
and the cell is an **excluded measurement, not a null localization**. A cell
unavailable because of a registry fact cites the capability row's `reason` /
`why` rather than restating it. The triple adds no rule numbers: `invalid`
reuses the refusal it wraps, and `unavailable` is not a rule.

**A result cell** is one `save` entry at one point. An available cell records
**nothing new** — no `status: "available"` — so every result written before
this value existed is byte-identical to one written after it, and the canonical
form (sec. 7) has no new field. An unavailable cell records four fields, in the
`entries` record of its tensor key (sec. 8) and in the point's summary:

| field | value |
|---|---|
| `status` | `"unavailable"` |
| `reason` | one of sec. 2.4's reason codes |
| `detail` | the fact, in prose — which expert, at how many positions |
| `denominator_key` | the key below |

The saved tensor is still written — a `(0, d)` gather with per-row widths of
zero, or for a `reduce`d read the `(width,)` vector with `NaN` for `mean` /
`std` / `median` and `0` for `sum` / `count` (sec. 2.12).

The same rule places the two alignment reasons (sec. 2.3). A row whose
`variable` / `column` value occurs zero or several times in its text is a fact of
the data the document could not know: for a **read** it is an `unavailable` cell
— `alignment_missing` / `alignment_ambiguous`, the `detail` naming the value, its
count and the row — counted in the denominator (`cells 0 / 1 eligible; 1
excluded: alignment_ambiguous ×1`), the row contributing no positions, and a
metric that reduces such a read inherits the cell **row by row** (sec. 2.10
"Eligibility"): the rows that aligned are scored, each excluded row carries the
reason code with a `null` value, and the metric's own cell is unavailable only
when no row was eligible. For a
**write** it stays a refusal, before any forward pass, with the same reason
code: an intervention that silently skipped a row would still report a number
for it. A malformed *document-level* value (an `alignment` outside the
vocabulary, a position that is not a mapping) stays `invalid` with its rule or
parse code — rule 26 and `P2` respectively.

**The denominator is data.** `denominator_key` is the string an aggregator
counts a cell under: the saved value's name plus the point's coordinate label
(sec. 3) — `iia[target.layers=3,pos=2]`, or bare `iia` for an un-swept
document — which is also the tensor entry's key in a bundle. Wherever results
are reduced, the reduction reports `eligible` (cells counted) and `unavailable`
(cells excluded, grouped by reason) **as numbers**: `RunResult.cells` holds one
value per cell and `RunResult.denominator` the counts; `run` prints them as
`cells 155 / 157 eligible; 2 excluded: empty_selector ×2`. "Two cells were
excluded, not null localizations" is therefore one line read from the result,
with no bookkeeping beside it. Shards (sec. 9, `--points`) each report their
own; the campaign's is the sum.

## 5. Validation — load-error checklist

A conforming loader rejects the document unless all of these hold — every item
but five, and each of the five says so where it stands: rule 2 recommends
rather than requires, rule 19 is checked when the run encodes its inputs,
rules 20 and 25 (like rule 4's column half) need the resolved tables, so they
run under `validate --data`, and rule 32's coverage half is checked at build,
when the score table is read.

A configuration accepted by preflight must either execute or fail with a
narrower runtime condition that preflight could not know. Per-row window
counts (`variable`, `column`, `all` in the generated frame) and the rows' token
widths are runtime conditions the compile cannot know (sec. 2.3); everything else in this list is
decided before a model is loaded — including what the routed engine can
execute (rules 13 and 30, run against the chosen engine by `check_engine`,
sec. 9.1, before it loads weights). An engine's own refusal of a fact this list
decides is for a document that arrived unvalidated, never a second place the
fact is decided.

**Independent violations are reported together.** A document breaking three
unrelated rules is refused once, naming all three with their own paths, rather
than costing three edit-and-rerun cycles to learn what one pass already knew.
Two rules are the exception and stop the pass where they fail, because the
rules after them are not *evaluable* rather than merely uninteresting: rule 3
produces the namespace the rest reads, and rule 4 is what makes every name in
the document dereferenceable. The same holds **across the points of a sweep**:
every expanded point is a document of its own, so the compiler (sec. 9.1)
validates all of them and reports the *distinct* violations together, in point
order; a sweep whose every point breaks one rule the same way is refused once,
with that one message.

**Rule ids are the slugs; the numbers are frozen labels.** Each item below
opens with its slug (`split_declaration`), which is the rule's identity in code
(`RULES` in `causalab/protocol/errors.py`) and in this list, and the number
that follows it is a display label frozen when the rule landed: `[V22]` is how
the refusal prints, and `§5.22` is how prose cites it. A new rule appends with
the next unused number; numbers are never reused or renumbered, and a retired
rule leaves a gap. Two PRs that both claim a number conflict in one dict
literal and one line of this list, and one side takes the next number — no
raise site, test or citation moves, because none of them names a position.

1. **strict_keys** — Strict keys: unknown fields anywhere are errors; closed
   enums reject with suggestions. Derived fields (sec. 7) may not be authored.
2. **section_order** — Sections in the sec. 1 order. **This one warns and parses
   on.** Order is a reading convention, not content: canonicalization emits the
   sec. 1 order whatever order the file is in, so a reordered document is byte
   for byte the same experiment and refusing it bought nothing. The warning
   names the recommended order. `save` last needs no rule of its own — it is
   last in the sec. 1 order, its presence is required by the parse, and its
   contents by rule 10.
3. **global_namespace** — Global namespace: no duplicate names across method
   sections 2–10 (§1); no reserved names (`base`, `counterfactual`,
   `counterfactual[j]`, `original`, `all`) declared.
4. **references_resolve** — Every reference resolves: sites (declared inventory
   only), positions, featurizers, params, reads, writes, intervened_models,
   metrics; a regularizer's `costs` keys name that term's own targets and a
   `constraint` term's targets are gates (sec. 2.11); every read or write
   through a position gate (`axis`) addresses a fixed `span`, one length per
   gate (sec. 2.5); an `anneal` or `control` target that is a term's weight
   names a
   **named** objective term with a numeric authored weight, a phase's
   `anneal` a featurizer the phase trains, and its `freeze_masks` gates the
   phase does not train (sec. 2.11) — and,
   from the component's capability row (sec. 2.4), the site
   sub-axes and the write mechanisms: `expert` on a component with no
   per-expert axis, a mechanism the component's write policy refuses, and a
   component the registry entry says the model does not have (a MoE tensor on
   a dense model, a full-attention tensor at a Gated DeltaNet layer, the
   routed-expert interior on a model whose entry records another
   `experts_implementation` than the grouped dispatch) are references that do
   not resolve. Each carries a reason code (sec. 2.4). A component the
   model's *family* does not serve (sec. 8) is the same refusal made by the
   run: the family is a property of the loaded module tree, which no load
   pass sees, so it is not a load rule. A metric's `minimum_count` (sec. 2.10
   "Eligibility") above the resolved base table's **maximum eligible count** —
   the rows carrying a value in every column the metric names — is a reference
   to more eligible rows than the data resolves, refused under `validate --data`
   naming the maximum; a threshold at exactly the maximum passes, and a metric
   with none makes no claim. A `match` metric whose `mode`
   contradicts a recorded table's `string_mode` under sec. 2.10's derivation
   is a reference the table's own bytes refuse to resolve, held at load and
   again before the first forward (sec. 2.2).
5. **read_bindings** — Reads: `model` ∈ `original` ∪ IMs; `input` a valid role;
   if `model` is an IM, `input` equals the IM's `input`.
6. **write_operands** — Writes carry no `model`/`input`/conditions; operands
   name reads, params, or literal scalars.
7. **write_membership** — Every write is in ≥ 1 intervened_model; every IM has a
   mandatory valid `input`; the model graph is acyclic.
8. **one_absolute_write** — Per (site, overlapping pos, model): ≤ 1 absolute
   write.
9. **dims_disjoint** — `dims` selections co-occurring at one address in one
   model are disjoint.
10. **save_manifest** — `save` non-empty; entry shapes exact; bindings match
    resolution; every metric and every trained featurizer saved; nothing else
    saveable.
11. **sink_rule** — Sink rule: every read is saved, a metric input, or an
    operand.
12. **featurizer_legality** — Featurizer legality: loaded featurizers
    (`file_path`) are not trained; trained featurizers are declared kinds with
    trainable slots; and a `featurizer` composition is non-empty with no
    repeated stage (sec. 2.5 — a stage's width comes from its position in the
    chain).
13. **pytorch_fn_local** — `pytorch_fn` present ⇒ refused unless the selected
    engine is local.
14. **sweep_wrappers** — Sweep wrappers well-formed; the expanded point count is
    reported (and may be capped without an explicit override flag; the count is
    taken after correlated rows are applied, sec. 3.2).
15. **artifact_fields_resolve** — Artifact-valued fields resolve (missing
    artifact = error, never a default).
16. **generation_read_only** — Generation is read-only and prefill-only: no
    write's `pos` carries `generated`, and `train` does not co-occur with a
    `generated` position.
17. **realization_coherent** — The model's realization is coherent: a
    `quantization` block carries only the knobs its own scheme has
    (`double_quant` is 4-bit vocabulary, `int8_threshold` is int8 vocabulary).
18. **shape** — Shape (§1): the four groups are present and nothing else is at
    the top level; the header holds `protocol_version` `"3"` and at most `title`
    and `description`, both free text; the method holds only its twelve
    sections. A document with a top-level `version` — the `protocol_version` 1
    spelling — and a document declaring `protocol_version` `"2"` are each
    refused by name, with `causalab migrate` as the answer; a workflow document
    (`steps`) handed to this loader is refused as a workflow.
    All of this is decided before anything addresses the tree by path, so a
    `--set` on the wrong kind of document is refused as that and not as a
    missing path.
19. **write_widths_uniform** — Write widths are uniform: every row a write
    addresses carries the same number of positions. Only an `all` or `variable`
    write can be ragged, and only the tokenizer can say how wide a row is — so,
    unlike the rest of this checklist, rule 19 is checked when the run encodes
    its inputs, **before any forward pass**, not at load. `validate` cannot
    decide it: the pure verbs hold no tokenizer, by design. A write may
    declare how a ragged window lands (`writes.<w>.ragged.policy`, sec. 2.8):
    `refuse` — the behaviour of an absent field — is this refusal;
    `exact_length_buckets` and `padded_masked` land every row at its own
    width, before any forward and with no change of batch geometry, and
    record the policy and the per-row widths in the run receipt's
    `execution.ragged` block. A `column` window is checked here like a
    `variable` one. A ragged operand paired into a write under `refuse`, or
    whose widths disagree with the write's, is the same refusal.
20. **base_is_paired_schema** — `base` is the schema of a paired row: every
    non-`base` data role's columns are a subset of `base`'s, and equal to them
    when the role names a different dataset (sec. 2.2). Like rule 4's column
    half this needs the resolved tables, so it belongs to the `validate --data`
    pass rather than the bare load.
21. **operand_reachability** — Operand reachability: every write operand that
    names a **read** is read at an address no deeper than the one the write
    lands on, in the `(layer, intra-block rank)` order of sec. 2.4's vocabulary.
    Equal is legal — that is the harvest/inject idiom. Params and literal-scalar
    operands have no address and are unconstrained. A band (sec. 2.4 `layers`)
    is held the way it runs: an operand read on a band of the same length as
    the write's feeds it member by member, so member *i* of the read is no
    deeper than member *i* of the write; any other operand is broadcast, so the
    read's **deepest** member is no deeper than the write's **shallowest**.
22. **split_declaration** — Split declaration (sec. 2.2), four refusals: three a
    resolver owes on any table it reads, and a fourth owed to a fit across the
    tables it names. The first three are the resolver's rather than this
    checklist's on purpose: each is a property of one table, and the resolver
    is the single place a ref becomes rows, so `run` cannot route around them
    the way it can route around a `validate --data` pass. The fourth needs two
    tables at once, so it is checked by the `validate --data` pass and again,
    per point, before any forward — the one seam every engine's `execute` goes
    through (`causalab/protocol/fit_splits.py`).
    - **Every table declares.** A table with no `split` column is refused, and
      so is one where only some rows carry it. An optional declaration is one
      that can be omitted, and an omitted one is the state the column exists to
      abolish; a single undivided pool says so with one uniform value.
    - **A bare ref may not name a partitioned table.** If a table carries more
      than one split, a ref without a `#<split>` fragment is refused — it would
      consume every split at once, which is the contamination this is all for.
    - **Splits of one table are endpoint-disjoint.** No prompt may appear in two
      splits, at either endpoint. Row-level disjointness is free (a row declares
      one split); this is the half that matters, because a prompt that is a
      training base and a test counterfactual reports a training score under a
      held-out name. There is no opt-out: a deliberate train-equals-test
      ablation names *one* split twice, where the document shows it.
    - **A fit's splits are endpoint-disjoint across tables.** A document with a
      `train` block names rows in two roles — the training rows its `data` refs
      select and the held-out rows `train.eval.split` names. When the two are
      splits of one table the bullet above already holds; when they are two
      different refs, no prompt may appear in both, at either endpoint, and
      the refusal names both refs, both roles and the first shared prompt. The
      same ref named for both roles is not refused: that is how a deliberate
      train-equals-test ablation is spelled, where the document shows it.
23. **group_legality** — Group legality: a gate's `group` (sec. 2.5) names an
    axis the site's component has, **in the site's own basis**, and its
    coordinate→group map is derived from the registry — `(heads, head_dim)` off
    the component's shape under `head`, `(num_experts, d_expert)` off the
    model's expert table under `expert_neuron`. So: the component carries the
    axis the group needs (`head` on a head-major component, `expert_neuron` on
    `expert_activation` or `expert_neuron_output`; `group: "head"` on `block_output` is refused, as
    `head:` on it is by sec. 2.2); the site does not already select a single
    member of it (`head: 3` under `group: "head"` is one group — a
    coordinate-wise gate under a grouped gate's name); the gate is the **first
    stage** of every chain that uses it (a grouped gate acts on the component's
    own coordinates — after any other stage, a `subspace`, an `sae` or a
    `standardize` alike, coordinate 5 is no longer a channel of head 0); and, when one
    featurizer is used at several sites, they all derive the same map. Like
    rule 4's width half this reads the model's declared axes, so it is decided
    as the document canonicalizes — a loaded `file_path` gate included — and
    **no model is loaded**: `ModelInfo` is static config. Whether a *fitted
    bundle* matches the document's group and map is rule 15's: the stamped
    `group` and `group_map`.
24. **code_declaration** — A `code` declaration agrees with the source it names (sec. 2.8.1): the
    `locator` resolves to a Python source file (there has to be something to
    hash); the declared `args` fit the function's signature, and an argument it
    requires and the document does not give is refused; and no read the
    function makes is both **statically detectable** and undeclared — a literal
    `os.getenv("X")` outside `env_inputs`, a literal path outside
    `data_inputs`. The last two halves need a readable `def`: a function built
    by a factory has none, and is hashed but not signature-checked. Resolution
    imports nothing.
25. **row_roles** — Declared row roles match the resolved data (sec. 2.8.1): a `code` entry
    whose `row_roles` cover *n* rows is refused when the input role of an
    intervened_model its write is in force on resolves to a table of a
    different length. Like rule 4's column half and rule 20 this needs the
    resolved tables, so it belongs to the `validate --data` pass rather than
    the bare load. A declaration with no `row_roles` makes no claim and is not
    checked.
26. **alignment_declared** — A declared `alignment` fits its address (sec. 2.3):
    the value is one of `one_to_one` | `one_to_many` | `many_to_one` | `absent` |
    `ambiguous`; the address can carry one (`all` takes no modifiers, and a
    `generated` position is a result rather than one of the pair's inputs); and
    where the document alone fixes the cardinality — an `index` is one token per
    row on every input, an unscoped `span` one joint window of one width — the
    declaration is `one_to_one`. Everything else about a declaration needs the
    tokenizer and is checked when the run encodes its inputs, as rule 19 is:
    the pure verbs hold none. Named on the field (`positions.<name>.alignment`,
    `reads.<r>.pos.alignment`, `writes.<w>.pos.alignment`).
27. **segment_declared** — A `segment` anchor names a declared segment and a
    span is well-formed (sec. 2.2.1, sec. 2.3): `segments.frame` is one of
    `chat`; `segments.system` needs the chat frame; a name in
    `segments.declare` is not one of the chat frame's own; every `segment`
    anchor — a whole-segment selector, a `scope` or a `relative_to` — names a
    segment the section declares (no section, no segment anchors); an anchor
    on `continuation` carries `generated` (the frame's decode budget lives on
    the position) and the whole continuation is spelled `{"generated": …,
    "all": true}`; an `atomic` span whose member set the document alone fixes
    has two or more members. Document-decidable only: whether a declared
    segment *occurs* in a row is the frame's to find when the run encodes its
    inputs (`absent` / `ambiguous`, sec. 2.3). Named on the field
    (`segments.frame`, `segments.system`, `segments.declare.<name>`,
    `positions.<name>`, `reads.<r>.pos`, `writes.<w>.pos`). A row's
    `edit_groups` declaration (sec. 2.2) is held to the same standard, at
    `data.base`: its spans lie inside the pair's texts, both sides declare the
    same number of constituents and an `atomic` group has two or more (checked
    at `validate --data`); and a position that addresses a constituent of an
    `atomic` group without its siblings, within one intervened model, is not a
    well-formed span for that pair — refused when the run encodes its inputs,
    naming the group and the missing siblings (`intervened_models.<m>`).
28. **family_wrappers** — `at_once` families well-formed (sec. 3.1): one axis
    per entry and one axis per reference, no `sweep` wrapper anywhere in a
    family entry, a `names` template naming the axis field, distinct values, at
    most 1024 members, member names that are neither reserved nor already
    declared, and a write-list window that is non-empty and resolves to exactly
    the interval it names. References to a family from `metrics`, `save`,
    `train` or a swept write list are refused here rather than left for rule 4
    to report as a missing entry.
29. **kl_operands_compatible** — `kl` operands are comparable (sec. 2.10): the
    two reads hand the metric distributions over the same **effective width**
    — the site's component width from static model config (one head's slice
    under `head`), folded through the read's featurizer chain and `dims` —
    through the same **transform** (the same featurizer composition and `dims`
    selection on both sides), in the same **token-position frame** (both
    prompt, or both `generated`). A shared component label is rule 4's
    precondition, not a distribution's shape. A loaded SAE's width is its
    bundle's, and the per-row count of two continuation windows is the
    tokenizer's; both stay the executor's.
30. **train_engine_supported** — The routed engine can execute the fit as
    authored (sec. 2.11): a free `params` tensor in `train.params`, a
    `train.precision` other than `fp32`, and an `eval` counted in `updates`
    each need the sec. 8 capability that says the engine's loop implements it
    (`train_free_params`, `train_loss_precision`, `train_eval_updates`), and
    the refusal names the field and the missing verb. Decided exactly when the
    engine is known — a compile handed `engine_capabilities`, and every run
    through `check_engine` (sec. 9.1) before the engine loads a model — so the
    rule refuses the engine, not the document: an engine declaring the verb
    runs the same document unchanged.
31. **metric_position_scalar** — A metric reduces one position per example
    (sec. 2.10): a prompt-frame read whose address the document fixes to more
    than one token — an `all` address, an unscoped `span` wider than one, an
    `indices` set or union of more than one — cannot feed a metric. A
    `variable` or `column` window is as wide as the tokenizer makes it and is
    the executor's to refuse; a continuation read is exempt by design, its
    metric reduces per decode step (sec. 2.3).
32. **scores_init** — A gate's `init.from_scores` fits the gate (sec. 2.5):
    `keep` is at most the gate's unit count (decided as the document
    canonicalizes, where the width and the group map are derived — like rule
    23), and the score table, after `where`, names every unit of the gate
    exactly once — one `unit` column per axis of `theta`, each index inside
    it, none repeated, every score a number (decided at build, when the table
    is read, like rule 19). A start that skipped or doubled a unit would seed
    a mask nobody authored.

## 6. Derived — never authored

| property | derivation |
|---|---|
| featurizer widths, param shapes | from (model config, site); parametrization internals are not authored |
| a grouped gate's group map | `(heads, head_dim)` from the component's shape, or `(num_experts, d_expert)` from the model's expert table (sec. 2.5 `group`); stamped as `group_map` |
| a hard-concrete gate's stretch | the authored or default `[γ, ζ]` (sec. 2.5 `parametrization: hard_concrete`); stamped as `stretch`, since the hard split `θ > logit((½−γ)/(ζ−γ))` depends on it; compared when the document authors one, and a non-default stamp is refused by a document authoring none |
| param slots | per featurizer kind (sec. 2.5) |
| `requires` | capability set, sec. 8 |
| `num_forwards`, fusion, staging | from the model graph; a compile property |
| decode depth, and what a continuation read obliges | from the group's `generated` positions, `save` and the metrics over it (sec. 8) |
| dataset content digest | resolved + stamped at load |
| observed alignment cardinality | from (tokenizer, row) when the run encodes its inputs (sec. 2.3); never recorded, except as the `reason` of an unavailable cell (sec. 4.1) |
| resolved token indices, and the **location ledger** | from (tokenizer, frame, row) when the run encodes its inputs (sec. 2.3): every position of every read and write, on every input — never authored (sec. 7). Recorded only when the document saves a `location_ledger` entry (sec. 2.12): one row per (example, edit group, constituent, side, token index, token id, decoded token) — `example` the row's index in its table, `edit group` the forward group `<model> on <input>`, `constituent` the position's name (or its inline path, with `[k]` per member of a non-atomic set), `side` the input role, `token index` the index in the row's own token sequence (0 = the row's first real token, chat prefix included, padding never), `token id` and `decoded token` what sat there. Its digest — `sha256` over the rows, each as sorted-key minimal JSON of the seven columns, the rows sorted as strings and joined by `\n`, UTF-8 — is stamped as `location_ledger_sha256` on every artifact the run writes (sec. 8) and recomputable from the saved table alone |
| chat segment locations and `prefix_lengths` | from the tokenizer's own chat template and its offset mapping when a document declares `segments.frame: chat` (sec. 2.2.1); 0 and none under the plain frame |
| `code` `source_module`, `source_sha256`, `data_input_digests`, and `closure` / `closure_sha256` when the module imports a sibling | resolved from the locator, its sibling import closure and the declared paths at load (sec. 2.8.1) |
| compiled interventions + digests | deterministic sweep expansion (one per point); the named axes of sec. 3.2 expand as the rows they declare, never as the cross product of their lowered wrappers |
| family member names | the entry name and the axis value, or its `names` template (sec. 3.1) |
| path-block lowering (the emitted entries, the restorer boundary, the policy) | from the `method.path_patching` block, at compile (sec. 3.2); recorded in the run receipt as `derived` and reported as the compile's `lowered` output, never in the canonical form — the emitted entries *are* the canonical `method`'s sites, reads, writes and intervened models, and the digests are over those |
| `ArtifactIdentity` | stamped into artifacts, sec. 8 |
| a metric row's `unit` and `estimand_version` | the metric's own when authored, else the kind's (`METRIC_UNITS`, `<kind>/v1`; sec. 2.10); repeated on every row of the table, never in the canonical form unless authored |
| a metric row's **eligibility record** (`eligible`, `reason_code`) and a metric cell's `n_eligible` / `n_considered` | from the rows when the run scores them (sec. 2.10 "Eligibility"): a row is `eligible: false` with the `reason_code` of the `unavailable` it became (sec. 4.1) — its address aligned on nothing, its answer column is empty, a continuation row addressed nothing — and `true` otherwise; the cell counts its rows. Three denominators, named apart: `n_eligible` is the rows a metric's **decision rule** was evaluated over, `save.reduce: "count"` (sec. 2.12) is the rows a saved read's reduction collapsed, and a workflow reduction's `unit` (workflow §2.6) is the statistical unit a table is later reduced over. None is authored; only the threshold `minimum_count` is |
| a table's `scoring_digest` / `string_mode`, and the receipt's `scoring` block | the task's `ScoringSpec` when the table is built (sec. 2.2); compared, never authored, at `validate --data` and before the first forward, and recorded per ref in the run receipt — never in the canonical form |

## 7. Canonical form and digests

- **An experiment is a value, not a program.** One JSON document fully
  describes an experiment: it can be hashed, diffed, shared, and re-run.
  It never contains tensors, closures, resolved token indices, or anything
  only one engine could interpret. Resolved indices are a derived *output*
  (sec. 6, the location ledger): a document addresses tokens semantically
  (sec. 2.3), the run derives where they fell, and a re-run whose derivation
  differs — a terminal separator that moves every index by one — has a
  different ledger digest, stamped on what it writes (sec. 8), so a reader can
  tell the two apart from the table alone. A `pytorch_fn` names user code, and that is
  a reference — which is why it is a `code` declaration carrying the source
  hash rather than a bare qualified name (sec. 2.8.1): a name is not a value,
  a name plus the sha256 of what it resolves to is.
- **The parser owns execution.** The document says *what*; the parser/planner
  derives *how* (forward count, fusion, batching, sweep parallelization) and
  `explain` reports it.
- **Everything declared must reach a sink; everything derivable is derived.**
  Dead declarations are load errors. Authored files are minimal; the stamped
  canonical form materializes every default (sec. 7).
- **Format**: strict JSON (unknown keys = error). YAML is accepted at the
  authoring surface; the object model is normative. JSON has no comments —
  use `description`.
- **v1 scope**: prefill-only *interventions*. Greedy decode is addressable as a
  position frame (sec. 2.3) and readable; there are no decode-step writes, no
  sampling, and one neural model per document.
- **Canonical-stamp principle**: the authored file may be minimal; the
  canonical form materializes *everything* — every default (constant LR,
  optimizer betas, `model.dtype` and the quantization scheme's own knobs),
  every resolved reference (dataset digests, artifact values, a `code` entry's
  source hash, its sibling-closure manifest when it has one, and its data
  inputs' content digests), every derived width, sugar expanded (int and `"all"` positions,
  `at_once` families materialized to their members, sec. 3.1), unordered
  lists sorted (IM write lists), sweeps expanded to points. **A named-axes
  group is digest-bearing when authored** (sec. 3.2): the campaign's canonical
  form carries the `axes` block — rows as authored, ranges and dependent
  entries materialized — beside its lowered sweep wrappers, so the campaign
  never reads as the cross product it is not; each point's canonical form is
  the hand-written point's, and every document without the group keeps its
  bytes, so this is no version bump. `rows` are recorded folded — each field
  takes the value the display column receives after the same folding — so
  `layers: 8` and `layers: [8]` are one row and one digest. The canonical form
  is the four groups of §1 (the `axes` group beside them when authored) with
  the header reduced to
  `protocol_version`: `title` and `description` are authoring metadata and are
  dropped — the canonical form is the experiment, not the file, so renaming a
  document or rewording its intent moves no digest.
- **There is no method digest** (§1). An earlier version hashed the `method`
  group as authored beside the document digest; nothing compared it, it moved
  on sugar-only respellings the canonical form folds, and a run receipt
  carrying two identities for one document invited the question of which one
  a reader should trust. The identities that exist are the ones something
  reads: the document digest and the point digests.
- `digest = sha256(canonical bytes)` — sorted keys, canonical floats; each
  param replaced by its content hash. Document digest = campaign; point
  digest = provenance unit, stamped on every artifact as `produced_by`.
- **Estimand identity is digest-bearing when authored** (sec. 2.10): a
  metric's `unit` and `estimand_version` enter its canonical entry exactly
  when the document states them, and are otherwise derived onto the rows and
  absent here — so every document that authors neither keeps its digest, and
  a document that states its unit is a different value from one that does not.
- **A role's `shuffle` is digest-bearing when authored** (sec. 2.2), the same
  way: it enters the role's canonical entry exactly when the document states it
  and adds nothing otherwise, so every unshuffled document keeps its digest and
  a shuffled role is a different value from the unshuffled one, per seed.
- **A role's `draw` is digest-bearing when authored** (sec. 2.2), as `shuffle`
  is: the block enters the role's canonical entry exactly when the document
  states it and adds nothing otherwise, so every undrawn document keeps its
  digest. Its `eval` member reaches the **data identity** too — a forward group
  is keyed on the field the role *tokenizes* (`DataRole.resolved_field`,
  `<column>[eval]`), so a drawn role at one `eval` and the same role at another
  never intern together, while a drawn role at `eval: j` and the fixed
  `<column>[j]` spelling over the same rows do — identical texts, and a fit's
  own forwards bypass the store. Two different documents either way: `draw` is
  in the canonical form and the bare `field` is not the indexed one.
- Any change to canonical form bumps `header.protocol_version` and ships a
  loader migration (`causalab migrate`, §9). Pin a golden corpus (canonical
  form + digest per example) in tests. Version 3 is that rule applied to
  sec. 2.4's `layer` → `layers` rename: the canonical bytes of every layered
  site changed (`{"layers": [18]}` is not `{"layer": 18}`), so every layered
  document's digest moved and the migration carries the rename — the field,
  an `at_once` window on it, a `names` template's placeholder and every
  dotted `sites.<name>.layer` id (a workflow's `set`/`emit`/`group_by`/`x`).

## 8. Engine contract

An engine implements these services:

| service | contract |
|---|---|
| `SiteResolver` | site record → tap in its execution engine (component vocabulary, sec. 2.4) |
| position resolution | Pos spec + `PositionFrame` (pad side, packing, sequence shard map) → indices; supports flat, per-row, and ragged windows |
| planner | model graph → forward groups; fusion/batching/staging; elision |
| cross-point interning | run each distinct group `digest` once, capturing the union of the taps every point asked of it; each point gathers and featurizes its own value from that capture, keeping its own per-entry provenance (sec. 3). A group's identity carries, per input role, the content digest of the rows it reads (sec. 2.2) plus the field — never the ref's name. Optional but expected — report what ran as `RunResult.forwards` |
| mechanisms | the closed `do` set, class order per address; refuse `pytorch_fn` if non-local; count each write member's firings per forward against the count its kind declares and refuse the point on a mismatch — the whole set, as one transaction (sec. 4, "Fires") |
| featurizers | kinds table with declared dtypes; error-term contract |
| metrics | lower kinds to native ops; derive minimal logit materialization (`logits_to_keep`, vocab-parallel CE) from `save` + metric needs |
| generation | greedy-decode a group to its derived depth; materialize a distribution only where `save` or a metric needs one (see below); writes stay in the prefill |
| training | own the `train` loop (optimizer, accumulation, anneal, early stop, checkpoints) — the document never changes across engines |
| RNG | realize `gaussian` per declared seed + axis semantics, bit-stable across parallelism layouts |
| stamping | write each point's compiled intervention + digest; `ArtifactIdentity` into every featurizer bundle's safetensors header |
| pre-forward checks | before the first forward pass of a point — each a runtime condition the compile could not know (sec. 5's invariant): rule 19 (write widths, on the encoded batch, dispatching on the write's `ragged` policy — sec. 2.8), and the answer-form pre-flight (sec. 2.10). A table's recipe sidecar is **not** among them: a run holds a table to nothing beside it (sec. 2.2) — the pin over a table is the consuming workflow's (`pins`, workflow spec §7) |

`ArtifactIdentity` (stamped, checked on any `file_path` load; mismatch
refuses): `produced_by` digest · model key + revision · **model dtype +
quantization** · tokenizer · site record · `k` · parametrization · a gate's
`group` and its derived `group_map` (`[heads, head_dim]` or `[num_experts,
d_expert]`, sec. 2.5) · a budget gate's `pool` name (compared both ways) and
`pool_units` (a record) · featurizer dtype · trained-on data ref + digest ·
engine · applied implementations · `commit` (the code revision). A rotation
fitted against bf16 weights is not the same artifact as one fitted against
fp32 weights, and the stamp is what says so. The trained-on digest is the
sha256 of the `table_bytes` of the rows the trained-on ref selected, as
canonicalized (sec. 2.2) — the same digest the fitting point's canonical
`data` carries. It is a **record, not a load-time expectation**: an apply
document legitimately reads a different split than the fit trained on, so
there is nothing in it to compare the digest against, and a load never refuses
on it. A `subspace` fit that started
from a saved basis (sec. 2.5 `init`) additionally stamps where it started:
`init_produced_by` (the basis's own digest), `init_trained_on` (the data ref
the basis was fitted over), `init_components` (the column indices taken,
`[0, k)`) and `init_digest` (a sha256 of the seeding matrix) — so two fits
from two bases are two artifacts even when everything the document says
agrees. A harvested read stamps the dataset it read (`trained_on` and its
digest) beside its site, which is how a basis fitted over it comes to carry
one. **Recorded on every such fit, expected only where authored:** the check
iterates the keys the *document* implies, so the four `init_*` fields enter a
load's expectation only for a document that authors `init` — a document
without one asks nothing about a start, and a subspace bundle stamped before
these keys existed loads under it exactly as before. No re-fit, no migration.
**Three of those are recorded, not compared.** `location_ledger_sha256`
(sec. 6) is **stamped only when the run emitted a ledger** (a
`location_ledger` save entry, sec. 2.12) and never turned into a refusal: it
names the ledger table the fit was made under, so a reader can find the tokens
the parameter was trained on. A run that loads the artifact and saves its own
ledger records the tokens it selected on *its* rows — applying a fitted
parameter to other data is the ordinary use, and the two tables are how the
researcher compares the selections. The check iterates the keys the
*document* implies, so `engine` and `applied implementations` are written into
the header and read back by a human, never turned into a refusal — no document
states either one, and neither could: both are facts about the run that
produced the artifact, not about the experiment that asked for it. They are
stamped because a tensor captured on a different engine, or through a forced
eager-attention path, is not self-evidently the same tensor — provenance a
reader needs even where the loader has no expectation to compare it against.

**The `commit` field comes from `causalab.provenance.runtime_identity()`, and
it is never the string `unknown`.** The key keeps its historical name; what
it carries is a *revision*, resolved as follows. That module reads the *installed
distribution's* metadata — PEP 610 `direct_url.json` — rather than importing the
package or shelling out to `git` in a guessed directory, which is what lets it
keep `requested` and `resolved` revisions apart: "the branch I asked for" and
"the commit that is installed" are different facts, and conflating them is how a
run can report a branch it did not execute. An install that records no revision
(a published wheel) is stamped with its **tree digest** instead — a deterministic
hash over the bytes that will run — so the field always identifies content.
A resolution *failure* raises rather than producing a placeholder.

**Per entry, not per file.** A swept document writes one file from many
points, so the file-level stamp carries only what every point agrees on;
whatever differs (`k`, the point digest, a swept site) is stamped per tensor
key in an `entries` table in the same header — `{key: {slot, coords, …identity}}`,
plus the four `unavailable` fields on a cell that measured nothing (sec. 4.1).
That table is what makes an entry selectable (sec. 2.5) and provable: the
check runs against the record of the entry a document actually selects. A
bundle with no table is a single-point or hand-made artifact and is checked
at file level, as before.

**The family contract.** The `SiteResolver` above is shared by both engines
(`causalab/neural/shared/sites.py`), and everything it knows about a *model
family* it reads from one declared plugin beside the capability rows — a
`registry.FamilyAdapter`, registered with `registry.register_family` from any
module. A family declares:

| field | what it declares | where it lives |
|---|---|---|
| `detect` | **model detection** — one predicate over the loaded module tree (`hasattr` / child-name structure), never a `config.model_type` match; exactly one registered family must detect a tree, and none or several is refused by name (`registry.family_for`) | the adapter |
| `tree` | the model root's addressed children — the block list, the embedding, the final norm, the head, the block's MLP child — as dotted paths (`model.layers`, `transformer.h`) | the adapter (`TreeAddress`) |
| `mixers` | the mixer child names its blocks may carry and the stream each means; the shared stream table reads the union over registered families and refuses a block carrying children of two streams | the adapter |
| `taps` | **component resolution** and **per-family availability** — semantic name → the family's tap: a scope (`embedding`, `final_norm`, `lm_head`, `block`, `mixer`, `mlp`), a dotted child path, a hook kind (`in`, `out`, or a function slot: `interface`, `delta`, `experts`, `interior`), a tuple index, a slot, a derivation. A component with no tap is refused by name, at the registry, never as an `AttributeError` out of a module lookup. The attention interior's tap says *from the row*: its per-family address stays the component row's `overrides` (sec. 2.4) | the adapter (`Tap`) |
| **tensor-shape contracts** | `registry.component_shape` — family-independent, on the rows | the rows |
| **supported mechanisms** | the rows' `reads` (which engines) and `writes` (which `do` mechanisms) cells | the rows |
| `identities` | **reconstruction identities** — which component is recomputed from which inputs by which formula, to a dtype-keyed tolerance (`block_mid == block_input + attention_output`; `routed_output == Σ_slot expert_output · router_scores`; `S_t == S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t`); the tests that pin an identity read the row, so a new family is held to what it declares; a family with none declares none | the adapter (`Identity`) |
| **aliases + deprecation version** | the rows' `aliases` / `deprecated_in` cells, from `schema.DEPRECATED_COMPONENTS` / `DEPRECATED_IN`; the typed backend pairs (`registry.BACKEND_PAIRS`) say which two-spelling tensors may not be aliased | the rows and the alias table |
| `probes` | optional per-family evaluators of the rows' `requires` predicates over the loaded tree; the resolver's shared module-tree probes otherwise | the adapter |

**The readout** — the final normalization, the unembedding and its accumulation dtype, and centering — is an engine-side service beside the adapter, not a document vocabulary: `causalab/neural/shared/readout.py` builds it from the adapter's `tree.final_norm` / `tree.lm_head` (the modules, called as the model calls them, never their weights) and a declaration keyed by the entry's `family` (`ModelInfo.family`, the HF `model_type` — finer than the tree family, because the Llama tree carries both a `weight` and a `1 + weight` RMSNorm gain): the norm's kind, its gain convention, where its epsilon lives, and the dtype the reference unembedding accumulates in, each held to the module's own forward and refused by name when the declaration and the module disagree. Nothing in a document names it — `lm_head`'s value is unchanged and a centered readout is a Python method — so no rule, component or metric field is added here.

Two families are built in — the Llama tree (Llama / Qwen / Mistral / Gemma
and the Qwen3.5-MoE hybrid, whose DeltaNet and MoE interiors live in it) and
the GPT-2 tree — and `tests/_helpers/synthetic_family.py` registers a third
from outside `causalab/neural/`, which is the contract's acceptance. **The
stated limit:** the component vocabulary is one global closed literal (sec.
2.4) and one execution order (`plan.COMPONENT_RANK`), because routing, the
canonical form and every census guard rest on that; a family declares which
of the existing names it serves and where, and cannot mint one — a new
tensor is a capability row first, then a tap. `registry.inventory(model)`
is the one producer of "what exists at which layer": per layer, the mixer
stream, the components present and their read / write mechanisms, from the
registry entry offline (its `layer_types`) or from a loaded model (its
modules and its family) — what `dry-run` (sec. 9), the generated support
tables and the inventory tests all read.

**Capabilities.** `requires` is derived from the document; an engine declares
what it supports; `choose_engine = first b where requires ⊆ b.capabilities`;
refusal messages generate from the missing capability.

Two kinds of entry, one comparison. The **coarse verbs** below are the closed
`CAPABILITIES` vocabulary. **Component entries** are generated, never listed:
every site a read or write references contributes `component:<name>` (a write
also `component:<name>:write`), and each engine declares the component sets it
serves — so a document touching a component outside one engine's site
vocabulary routes past it to an engine that serves it, and the generated
refusal names the entry. The closed vocabulary behind these entries is the
sec. 2.4 `Component` literal itself. Stream- and layer-level constraints (a
full-attention box on a DeltaNet layer, a read-only component) stay
engine-internal policy: they depend on the loaded model or are true of every
engine, so routing on them would be either impossible or misleading.

| capability | required when |
|---|---|
| `grad` | `train` present |
| `paired_forward` | a write's operand read has a different `input` than the write's model |
| `full_logits` | a full `lm_head` read is saved, or a `class_probs` / `top_k` metric reads `lm_head` other than through a `dims` slice — a *featurized* `lm_head` read still obliges the whole projection (the featurizer consumes it) even though its value is latents. A `top_k` over any other component obliges no vocabulary projection (sec. 2.10) and must not be charged for one |
| `generate` | any position carries `generated` (sec. 2.3) |
| `quantized_weights` | `model.quantization` present (sec. 2.1) |
| `writable_attention_probs` | a write targets `attention_probs` |
| `pytorch_fn_local` | any `pytorch_fn` |
| `train_free_params` | a `train.params` entry names a `params` entry — a free tensor (sec. 2.6) rather than a featurizer or a slot; refused by name under rule 30 when the routed engine lacks it |
| `train_loss_precision` | `train.precision.feature` or `.loss` is authored as anything but `fp32` (sec. 2.11); rule 30 |
| `train_eval_updates` | `train.eval.every` counts `updates` (sec. 2.11); rule 30 |
| `component:<name>`[`:write`] | generated — a read or write references a site with that component (writes add `:write`) |

**What the shipped engines declare.** Two engines implement this contract. The
ten verb rows below are exactly their `capabilities` frozensets, read off
`PytorchHooksEngine` and `NnterpEngine` — not a forecast of engines that
might exist. `tests/protocol/test_vocabulary_census.py` compares every row,
every ✓/✗ cell **and both component counts** back to those classes, so this
table cannot drift from them. The last row is a different attribute: component entries are *generated*
rather than listed (above), so it reports each engine's `components` /
`writable_components` instead — sets that are themselves generated from the
capability registry's rows (sec. 2.4), whose `reads` cell names the engines
serving each component; the `N of 56` cells are
`registry.engine_component_summary` and the census holds them to it.

| capability | `pytorch_hooks` (reference) | `nnterp` |
|---|---|---|
| `grad` | ✓ | ✓ on the same shared loop, in this process or — a `remote` engine — as one NDIF job per fit: the stages, the optimizer and the graph live in the job's session and the fitted state comes home (a §2.2 `draw`, re-planned every epoch, fits in this process only) |
| `paired_forward` | ✓ | ✓ |
| `full_logits` | ✓ | ✓ |
| `generate` | ✓ | ✓ one `model.generate` trace, decode steps walked with `tracer.iter` |
| `quantized_weights` | ✓ | ✗ |
| `writable_attention_probs` | ✓ inside the eager attention call | ✓ on the softmax's output inside the same call |
| `pytorch_fn_local` | ✓ | ✓ |
| `train_free_params` | ✗ — the shared loop (`neural/shared/training/`) optimizes featurizer slots only, and refuses a free tensor as arrived-unvalidated | ✗ — the same loop |
| `train_loss_precision` | ✗ — the shared objective casts logits and targets to fp32 unconditionally; the authored `train.precision` is digested, not executed | ✗ — the same loop |
| `train_eval_updates` | ✗ — the shared loop reaches an eval on epoch boundaries only, and refuses an `updates` counter as arrived-unvalidated | ✗ — the same loop |
| `component:<name>`[`:write`] | 52 of 56 — all but `deltanet_query` / `deltanet_key` / `deltanet_state` and `expert_permutation` | 53 of 56 — all but `delta_kv_mem` / `delta_state_update` / `delta_state` |

Neither engine is a superset of the other, which is the point of routing: the
reference engine alone reaches the per-token DeltaNet faces `delta_kv_mem` /
`delta_state_update` / `delta_state` (by stepping the recurrent kernel inside
swapped module-global call sites — the chunked prefill kernel never
materializes them) and loads quantized weights, and the nnterp engine alone
reaches the pre-tiling `deltanet_query` / `deltanet_key`, the per-chunk
`deltanet_state` and `expert_permutation` (through `.source`); the rest of the
DeltaNet interior is one name both serve, each by its own mechanism (sec. 2.4).
`--engine auto`, the default, lists every installed engine with the reference
first and lets `choose_engine` pick.

Capability entries this vocabulary carries for engine *classes* that do not
exist here yet — a vocab-parallel trainer with no `full_logits`, a serving stack
whose only write is additive steering — stay in the vocabulary deliberately: the
refusal an unsupported document gets is generated from the missing entry, so the
entry has to exist before the engine does.

**Materialization (generation).** A continuation read's cost is not the decode,
it is the vocabulary: at batch 32 and 16 steps, every step's distribution over a
128k vocabulary is ~260 MB in fp32, one step is ~16 MB, a site's activations
~8 MB, the token ids ~2 KB. The planner therefore derives, per group, the decode
depth and — per continuation read — whether anything downstream consumes a
distribution: the read is saved, or a metric in the `distribution` domain
reduces it (sec. 2.10). An `ids`-domain metric does **not** count, which is the
point of the domain: a text probe — `decode` over a continuation read, nothing
saved — obliges no vocabulary projection at all, and an engine **must not**
build one where the answer is no.

*How* it complies is its own business: keeping only the addressed steps,
projecting a narrower slice (`logits_to_keep` takes an index tensor), replaying
the sequence teacher-forced, or a vocab-parallel reduction. The reference
engine keeps `ln_final` activations across steps and projects through the head
only at the addressed positions, which needs no second pass — an implementation
note, not a requirement. `explain` prints the obligation so the bill is legible
before a run.

**Execution scale.** Documents and workflows are scheduler-agnostic — they
never name devices, hosts, or job systems. The division of labor:

- A **engine** owns all intra-run execution: device placement, batching, and
  any parallelism across a campaign's points or across its own
  accelerators — declared, like everything else, through its capability set
  and constructor. The reference engine takes `device` and runs points
  serially; sharded and multi-device engines are engine work, not document
  vocabulary. **Precision is not on this list**: `dtype` and `quantization`
  change the numbers, so they are the document's (§2.1), and an engine reads
  them per point rather than being told once.
- **Microbatching is the engine's too.** A forward group over `N` rows may
  execute as ceil(`N`/`b`) forwards of at most `b` rows each: each
  microbatch's captures concatenate in row order before a read gathers, a
  write whose operand was read in another group indexes the operand by row,
  ragged values keep their flat rows and widths, a `gaussian` draw is made
  over all `N` rows and sliced,
  and a decoding group decodes each microbatch from its own prefill. The
  result equals the single-forward run up to dtype rounding — the group is
  the unit the plan counts and `RunResult.forwards` reports, however many
  forwards it took. `b` is an execution parameter, never document
  vocabulary: the reference engine's `batch_rows` constructor argument and
  the CLI's `--batch-rows N`. It bounds no-grad forwards, `train.eval` passes
  included; a training minibatch keeps its `train.batch.pairs` rows, because
  that is the document's own batching knob for grad forwards, so the
  execution bound applies to the no-grad passes and to `train.eval`. Digests
  and artifact stamps are unaffected. **Batch geometry has exactly one
  recorder: the run receipt.** `protocol.json` (and a workflow step's
  `_step.json`) carries `execution.batch_rows` — the chosen engine's bound,
  `null` for a whole-batch run and for an engine that has no bound — so two
  layouts of one document produce receipts that differ there and nowhere
  else. The block gains `execution.ragged` **only** when some write ran under
  a non-`refuse` `ragged` policy (sec. 2.8): per `"<model>/<write>"`, the
  policy, every row's width and the `[width, rows]` buckets it landed by —
  recorded at the pre-forward width check (rule 19), never gated on, and
  absent from every receipt of a document that authors no policy. A fit
  cohort's batched eval passes are the one kind of no-grad
  forward that `null` does not leave whole: they pack under `batch_rows`
  when it is authored and otherwise under the fit's own `fit_rows` bound,
  measured or authored, so the receipt's `fit_rows` (or `fit_rows_resolved`)
  says their geometry too: the bound reported is the one every window of the
  fit ran under, after any shrink of either kind, so it is safe to pin, and a
  `fit_rows_shrinks` beside it says the measurement missed; only when the
  shrink was an eval window's is the report below the grad bound, so that a
  re-run pinned there packs its grad windows smaller than this one did. A
  grad budget that never resolved (off CUDA) stays `null` whatever its eval
  windows did.
  Pinning `--engine nnterp` together with `--batch-rows N` is refused
  at the command line, since nothing would honour the bound; under `auto`
  the receipt records `null` when the nnterp engine serves the document.
  It is *recorded, not gated*: a layout-dependent flip in a top-1
  token is something a reader of the two receipts can see and attribute,
  never something a run refuses over, and it stays out of the canonical
  document, the digests and every artifact stamp. Beside the receipt the run
  appends `events.jsonl`, the local event stream (workflow spec §4.3: seven
  event names, one JSON line each with a monotonic sequence, the document
  digest and the `--points` shard) — a sidecar that enters no receipt, no
  digest and no stamp, and in which a remote sink's failure is a `warning`
  line rather than anything the run does. The block's other key is
  `execution.model_source` — `loaded` when the engine loaded the document's
  model itself, `caller` when it ran a caller-owned bundle handed to its
  constructor (§9, the ownership contract; an engine that reports nothing
  loads). It is execution provenance exactly like `batch_rows`: the same
  weights loaded and handed in write byte-identical files and receipts that
  differ there and nowhere else, and it enters no canonical form, no digest
  and no stamp. The block's third key is `execution.fit_rows`, the grad
  forward's counterpart of `batch_rows`: the points of a swept campaign that
  declare `train` are fitted together as a **fit cohort** — one forward per
  optimizer step over the concatenation of every member's minibatch — and
  `fit_rows` bounds how many rows one such grad forward covers, the members
  packed into forwards under the bound. A member's own minibatch
  (`train.batch.pairs` rows, the document's knob) is never split. Unset
  (`null`), the bound is **measured**: the cohort's first step runs its first
  member alone as a probe under peak-memory tracking, and the bound is the
  rows the device's free memory holds at that slope with a tenth held back,
  never fewer than one member; every later window packs under it, and a
  window that still runs out of memory is retried at half the rows (never
  below one member — a single member that does not fit is the run's error).
  Off CUDA there is nothing to read and `null` packs every member into one
  forward. The measured bound is reported as `execution.fit_rows_resolved`
  beside the `null` request — the **smallest** over the run's cohorts, the
  number every cohort ran at or above and so the one to pin as `fit_rows`
  for a reproducible re-run, since what the probe measures depends on what
  else occupied the device; a run whose bound had to shrink says so beside it
  (`execution.fit_rows_shrinks`); a re-run pinned at that number is safe,
  and packs its grad windows smaller than the shrinking run did only when
  the shrink was an eval window's. A pinned numerical capture (the drift
  tier) should author `fit_rows`: the measured default packs by the device's free
  memory at that moment, and a different packing rounds differently. The
  eval passes of a cohort pack under `batch_rows` when it is authored and
  under the fit's `fit_rows` bound otherwise — so a `fit_rows` authored
  below one member's eval rows runs the eval members one per forward, and
  keeping the eval batching under a small `fit_rows` means authoring
  `batch_rows` beside it. An authored bound is never probed or shrunk — for
  the eval windows either. It is an execution parameter exactly like
  `batch_rows` — the reference engine's `fit_rows` constructor argument,
  the CLI's `--fit-rows N`, and a workflow step's `execution` block (workflow
  spec §2.2), which overrides the engine's bound for that step — recorded in
  the receipt as `execution.fit_rows` (`null` for an unbounded fit and for an
  engine with no such bound) and entering no canonical form, no digest and
  no stamp. `--engine nnterp` with `--fit-rows N` is refused for the same
  reason `--batch-rows` is: that engine runs each minibatch's forward whole. A later execution parameter adds its own key
  beside these; nothing else belongs in the block. What the run *observed*
  is recorded beside it, never in it: the `scoring` block (sec. 2.2) and
  the `fires` block (sec. 4) — both layout-invariant, so two layouts of one
  document still differ at `execution.batch_rows` and `execution.fit_rows`
  alone.
- **Job dispatch is site tooling outside this repository.** The one seam it
  needs is the CLI's `--points START:STOP` selector: an external scheduler
  expands nothing itself, launches `run` per index range, and recombines by
  digest — every shard stamps its artifacts as members of the same campaign
  (`document_digest` is unaffected by slicing).

## 9. CLI and the Python entry point

**The Python function is the primitive; the CLI is a wrapper over it.** Every
verb below is a function in `causalab.protocol` — `load`, `validate_document`,
`expand`, `canonicalize`, `digest`, `plan_point` — and so is `run`:

```python
from causalab.protocol import ResolutionEnv, FileDatasets, FileArtifacts, load, run_protocol

env = ResolutionEnv(datasets=FileDatasets(root=data), artifacts=FileArtifacts(root=arts))
result = run_protocol(document_path, env, engines, out)   # -> RunResult
for manifest_path, disk_path in result.files.items():
    ...
```

What stays in the CLI is argument parsing, what gets printed, and the exit
code. So a notebook, a script step or a test reaches a run without shelling
out and parsing stdout, and the two paths cannot disagree about what a run is
— there is only one.

### 9.1 One compiler, four doors

Everything between an authored file and an engine is **one function**,
`compile_protocol` in `causalab/protocol/compile.py`, and every entry point
calls it: `causalab validate` / `digest` / `explain` / `run`, `run_protocol`, a
workflow's `intervention_protocol` step when it runs, and workflow validation
when it loads that step's document. Nothing else composes the sequence — not
the CLI, not the workflow loader, not the runner — so a document cannot
validate under one resolution context and execute under another; the only
thing the four callers may vary is the six inputs below, and the compiled
result records what those decided.

```python
compile_protocol(
    authored_document,    # a path, the document as a tree, or a read prefix
    base_directory,       # where the document's relative references resolve from
    overrides,            # `--set` / a step's `set`, by section-rooted path (§1)
    dataset_resolver,     # the two services a load resolves against (§2.2, §1)
    artifact_resolver,    #   — a workflow's validation passes a deferring store
    engine_capabilities,  # what the engine can do, when known (rule 13; sec. 8)
)
```

It returns the eight outputs of a compile and nothing else — the seven compile outputs, and the derived record of what the compile lowered:

| output | is |
|---|---|
| `canonical` | the canonical document (sec. 7), sweep wrappers intact — the campaign |
| `points` | the expanded points with their coordinates (sec. 3), each parsed, validated and canonicalized, and the **explicit document** they came from — overrides applied, artifact-valued fields resolved |
| `data` | every dataset ref the points name, with its content digest and its columns |
| `artifacts` | every reference outside the document — a value reference or a `file_path` load — with the stamped identity read, and whether the store *deferred* the check |
| `capabilities` | the engine capabilities the campaign requires (sec. 8), derived from the registry rows — never a second table |
| `digests` | the document digest and every point digest (§7) |
| `diagnostics` | what the compile found and did not refuse on (below) |
| `lowered` | the derived record (sec. 6) of what the `paths` stage lowered: `{}` for a document without a `method.path_patching` block, else the block as written, its restoration policy, the receivers in injection order, the restorer boundary ordered by (layer, component rank) and the names of every emitted entry (sec. 3.2). Bound to the digests by reference — every emitted name is a key of `canonical.method` — and written into the run receipt as `derived`. Last, and stays last |

Violations are not an output: they are raised, one as itself and several
independent ones as `ValidationErrors`, so a returned compile is a valid
document. Byte for byte, the canonical form and every digest are what the
loader computed before the compiler existed — the corpus, golden, workflow and
demo pins are the proof.

**The order is data.** Authoring sugar — `--set`, artifact-valued fields,
sweeps — resolves to the explicit form *before* validation and hashing, once,
in this order (`compile.STAGES`; the census in
`tests/protocol/test_compile_protocol.py` holds this table to the code). A later
compiler stage is one more entry in the tuple and one more row here, inserted
where the order says it belongs:

| stage | does |
|---|---|
| `read` | read the source; refuse a workflow document, and a `protocol_version` this compiler does not read (§1) — before anything addresses the tree by path |
| `override` | apply `overrides` by section-rooted path (§1, sec. 9) |
| `resolve` | replace every artifact-valued field by its value (§1, rule 15); hold the tree to the JSON object model |
| `families` | `at_once` families into the entries they denote, all inside one point (sec. 3.1, rule 28) — before the gate, which is what makes them sugar |
| `paths` | a `method.path_patching` block into the sites, reads, writes and intervened models it denotes (sec. 3.2) — after families, so a wrapper inside the block is refused rather than expanded; before the gate, so the parser, the checklist, the canonical form and the digest see the hand-written document. Its record is the `lowered` output |
| `axes` | named axes — correlated row tuples and dependent axes (sec. 3.2) — parsed, and the document lowered to its display form: every `{"axis": …}` wrapper the `{"sweep": [column]}` it stands for, the group removed. `expand` walks the parsed axes as the rows they are, never as that cross product; `canonicalize` writes the block into the campaign. After families and paths, so a wrapper on a family entry reaches every member first and a path block's emitted entries are in the tree before the references are found; before the gate, which knows the four groups alone |
| `gate` | the strict parse of the explicit form, sweep wrappers intact (rules 1–2) |
| `expand` | sweeps into compiled interventions (sec. 3, rule 14) |
| `validate` | parse every point and run the checklist on each (sec. 5); check every `file_path` load's identity (§2.5, sec. 8); raise the distinct violations together |
| `canonicalize` | the canonical document and every point's canonical form (sec. 7); a per-point refusal here (a layer outside the model, rule 4) is collected across points like the checklist's |
| `digest` | `sha256` of the canonical bytes, document and points |
| `identify` | the dataset identities and schemas, the artifact references and what the store deferred |
| `route` | the capabilities the campaign requires (sec. 8); when `engine_capabilities` were given, a shortfall against them is refused (rule 13) — and `validate` has decided rules 13 and 30 from the same set |

**Diagnostics** are the closed set of things a compile reports without
refusing:

| kind | means |
|---|---|
| `deferred_check` | the artifact resolver deferred a `file_path`'s existence and identity check to run time — workflow validation of a step-dependent document (workflow spec §2.3). The compiled result says so instead of looking like a real resolution |
| `capability_shortfall` | the document requires a capability an engine lacks. A compile given `engine_capabilities` *refuses* it (rule 13, the `route` stage), as does `check_engine` for the routed engine; the kind is produced by `dry_run` per candidate engine (`causalab dry-run --engine`, below): what `check_engine` would refuse, reported before any weights load |

No door knows the engine when it compiles — routing needs the compiled points
— so every executing door compiles with `engine_capabilities = None`, chooses
an engine, and calls `check_engine(compiled, engine.effective_capabilities)`
**before** the engine loads a model: the same rule functions the `validate`
and `route` stages run when the set is given (rules 13 and 30, the shortfall),
re-entered for the engine routing chose, never a second pipeline. This is the
seam sec. 5's invariant hangs on, and the one `dry-run` asks per candidate
engine.

The read prefix — `read` and `override` — is also callable on its own
(`read_document(authored_document, base_directory, overrides)`), for the one
caller that has to see the overridden tree before it can pick a resolver: a
workflow decides whether an inner document depends on a step by walking that
tree, and then hands the prefix back to `compile_protocol` as the authored
document. One read, run in two halves, never a second implementation.

- `document` is a compiled document, a path, or the document as a tree. Pass a
  `CompiledProtocol` (sec. 9.1) when the compile itself needs options:
  `overrides` (`--set`) and the point cap are compile-time concerns, and a run
  must not re-decide them.
- `engines` are candidate implementations of §8's engine contract, supplied by
  the caller — which is what lets `causalab.protocol` import no engine and no
  torch.
- `points` is the shard selector, exactly as `--points` (below).
- There is **no `resume`**: `--resume` skips a workflow step whose outputs are
  already on disk with a matching stamped digest, and an intervention run has no
  step boundaries to resume at — the CLI refuses the flag on an intervention
  specification rather than ignoring it. Sharding a campaign is `points`; a
  resumable intervention is one wrapped in a workflow step, which is also the
  only place its inputs are pinned (workflow spec §7).

**A caller-owned model, and the ownership contract.** An engine normally loads
the document's model itself (`load_model`, cached, prepared: eager attention,
`.eval()`, `.requires_grad_(False)`, left padding with a pad token). A caller
who already holds a model hands it in instead:

```python
from causalab.neural.engines.pytorch_hooks import ModelBundle, PytorchHooksEngine

bundle = ModelBundle.from_model(model, tokenizer, key="gpt2", revision="main", device="cpu", dtype="fp32")
result = run_protocol(document_path, env, [PytorchHooksEngine(bundle=bundle)], out)
```

The contract both sides keep:

- **The library never loads, moves, frees or changes the training mode of a caller-owned model.**
  `from_model` derives the registry entry exactly as the loader does and
  mutates nothing. Where the loader would *prepare* the model —
  eval mode, frozen weights, left padding, a pad token, weights in
  the declared `dtype` — `from_model` **refuses** an object that lacks the
  setting, naming the one call the caller makes (`model.eval()`,
  `model.requires_grad_(False)`,
  `tokenizer.padding_side = "left"`, `tokenizer.pad_token = tokenizer.eos_token`).
  The refusals fire only where a loaded run's numbers would differ; a model
  prepared that way with the same attention backend is accepted and produces
  the loaded run's bytes. A declared `model.attn_implementation` must match
  the caller's model, or the engine refuses before any forward. Other
  attention backends are accepted: the hooks
  executor temporarily selects eager for forwards that need attention-function
  interiors, then restores the caller's selection, including after errors.
- **Every hook the engine installs is removed on every exit path**, a raise
  in the middle of a forward included: afterwards the model is the same
  object, on the same device, with the same parameters, and no module carries
  a hook. A caller bundle never enters the loader's cache.
- **`key` and `revision` are the caller's assertion.** Nothing can check them
  against the weights; the run receipt and every `ArtifactIdentity` stamp
  record them as given. What *is* checked, before any forward: the document's
  canonical `model` realization — `key`, `revision`, `dtype`, the materialized
  `quantization` block — against the bundle's, and the engine's `device`
  against the bundle's request. A disagreement refuses, naming both sides,
  because the record would otherwise describe a model that did not run.
- **The receipt says which way the model came in**: `execution.model_source`
  is `caller` for a bundle run and `loaded` otherwise (§8) — execution
  provenance, in no canonical form and no digest.

Both engines take `bundle=` (the nnterp engine an `NnterpBundle`); `from_model`
is the reference engine's constructor. The nnterp engine's `remote=` — its
forwards on NDIF, against a weight-free bundle — is a Python caller's option
too: it needs a trusted deployment with the same `causalab`, `nnterp`,
`nnsight` and `torch` installed server-side, which no command-line flag can state — the engine
compares the two installs against the server's reported environment before
it submits anything, and refuses (`P4`) on any difference. The CLI has no bundle flag: a
caller-owned model is a Python caller's situation.

The verbs dispatch on the document's shape: a **workflow** (it has `steps`)
runs its step graph, and an **intervention specification** runs the full
pipeline. `migrate` takes files, not a document, and compiles nothing.

| verb | effect |
|---|---|
| `run <doc>` | validate, expand, plan, execute, stamp; writes `<out>/protocol.json` — the canonical document, its digest, the per-point provenance digests, and an `execution` block: the row bounds the chosen engine was built with (`batch_rows` for no-grad forwards, `fit_rows` for the grad forwards of a fit; declared before execution, `null` when unbounded) and where its model came from (`model_source`: `loaded`, or `caller` for a bundle handed to the engine — sec. 8), and — added once the pre-forward checks have run — a `scoring` block per base dataset ref: the table's recorded scoring identity against the document's `match` modes (sec. 2.2), and — added once every point has run — a `fires` block: per point digest and forward group, how many times each write member fired per forward (sec. 4; a refused run's receipt carries none). Execution is recorded there and nowhere else: it enters no digest and no stamp; prints every saved file and the `cells` denominator (sec. 4.1) |
| `migrate <path>... [--check]` | rewrite earlier-version documents as the current version, in place: `protocol_version` 1's flat sections regrouped (§1), then `protocol_version` 2's scalar site `layer` renamed to the band `layers` (§2.4, §7) with every `at_once` window, `names` placeholder and dotted `sites.<name>.layer` id — a workflow document is rewritten exactly when it spells one; a markdown file's fenced JSON examples with them; a current document and a fragment are left alone. JSON and markdown only: a YAML document is refused with the reason (the rewrite would drop its comments), and a v1 *split* document (an `application` naming a `method` file) is refused too — its composition was the v1 loader's, so compose it with a v1 release first. `--check` writes nothing and exits 1 if anything would change |
| `validate <doc> [--data]` | sec. 5 checks; `--data` also checks column and prompt-variable references, at every point |
| `explain <doc>` | models + forward plan, expanded point count, derived `requires`, resolved bindings, digest, what `save` produces |
| `--engine` (explain) | also route the document and print which engine would serve it, or the sec. 8 refusal. Opt-in: engines are heavy, and without it `explain` stays torch-free |
| `dry-run <doc> [--data]` | everything a run decides before weights load, resolved and reported (the contract below): the composition (digest, overrides applied), every dataset ref with its content digest and columns, the model configuration from the registry entry, the axes and point count, the forwards per point and interned over the campaign, the derived `requires`, and per site — from the entry alone — availability, tensor shape, width, head space, which engines read it and which mechanisms may write it; the per-layer inventory where the entry declares `layer_types`; every read with the metrics that reduce it; every `save` entry; the compile's diagnostics. `--data` also runs `validate --data`'s pass and reports its refusal. The last line is always `undecided (decided when the run encodes its inputs): …`. Exit `0` = compiled and every fact resolved or explicitly undecided; `1` = any refusal — a compile refusal (printed as `validate` prints it, plus the reason-coded record: code, rule slug, field, reason), a `--data` refusal, a shortfall for the requested engine; `2` = argparse. Never loads weights, never fetches a config: an unregistered `model.key` is `[V4]`, and `--register-from-hf` is refused (`[P4]`) rather than inherited; a workflow document is refused with one line |
| `--engine` (dry-run) | also ask `check_engine`, per candidate engine, what it would refuse — reported per engine as a `capability_shortfall` diagnostic, never raised. A pinned engine's shortfall exits `1`; under `auto`, only no candidate serving does. Builds engines (constructing one loads no weights); without it `dry-run` stays torch-free |
| `--shard-size N` (dry-run) | plan `--points` shards of at most `N` points and report how many the campaign needs — `ceil(points / N)`, the arithmetic a scheduler recipe otherwise does by hand (`docs/running_experiments.md` §7); no producer of a shard count exists elsewhere |
| `digest <doc>` | the campaign digest |
| `--set path=value` | ad-hoc override — exploration only; promote anything that matters into the file |
| `--device` (run) | reference-engine placement: any torch device string (`cpu` default, `cuda`, `cuda:1`, `mps`). Placement is execution; precision is not (§8) |
| `--engine` (run) | `auto` (default) is every installed engine with the reference **first**, routed by `choose_engine` (sec. 8); name one to pin it. A document never names an engine — it declares what it needs — so the default is routing rather than a choice the document did not make |
| `--dtype` (run) | shorthand for `--set model.dtype=…` — it edits the document, so the run's digest is the overridden document's and the record never lies about what produced the numbers. Refused on a workflow, whose steps each declare their own |
| `--points START:STOP` (run) | execute one half-open point-index shard of the expanded campaign (sec. 8, execution scale); document runs only — digests and stamps are unaffected |
| `--batch-rows N` (run) | reference engine: run a forward group over more than `N` rows as several forwards of at most `N` rows each, captures concatenated in row order (sec. 8, execution scale). Execution only — the numbers equal the single-forward run up to dtype rounding; digests and stamps are unaffected, and the receipt records the bound as `execution.batch_rows`. Document and workflow runs; bounds no-grad forwards (`train.eval` included) while a training minibatch keeps its `train.batch.pairs` rows |
| `--fit-rows N` (run) | reference engine: bound how many rows one **grad** forward of a fit covers — the members of a fit cohort are packed into forwards of at most `N` rows each, a member's own minibatch (`train.batch.pairs` rows) is never split, and without the flag the bound is measured on the cohort's first step from the device's free memory (unbounded off CUDA) and recorded as `execution.fit_rows_resolved` (sec. 8, execution scale). Execution only — digests and stamps are unaffected, and the receipt records the bound as `execution.fit_rows`. Document and workflow runs; a step's own `execution` block overrides it for that step (workflow spec §2.2) |
| `--register-from-hf` | resolve an unregistered `model.key` from its HF config before loading, instead of refusing `[V4]`. Opt-in, so without it a digest never depends on the network; `run` always does it. On a workflow it pre-registers **every** inner document's key. Not on `dry-run`, which refuses it |

**The dry run.** There is no `protocol` sub-command group — the CLI is `causalab
<verb>` over both document types — so the verb is `causalab dry-run <doc>`,
and `dry_run(document, env, *, engines=(), shard_size=None, overrides=None,
check_data=False) -> DryRunReport` in `causalab/protocol/dry_run.py` is its
Python entry point (one caller, two doors: the compiler is the same
`compile_protocol`). It decides everything the compile decides, plus what the
registry entry decides about each site and, when candidate engines are handed
in, what `check_engine` would refuse per engine — and it **names what it
leaves to the run** instead of omitting it, so a refusal when the run encodes
its inputs (rule 19, the ragged-window refusals of sec. 2.3) is never mistaken
for a dry run that passed. A document that does not compile is re-raised: the
refusal *is* the report, printed as `validate` prints it plus its reason-coded
record. A compiled document whose engine falls short is reported, not raised.
It never calls a model loader, an engine's `execute`, or a config fetch: the
model facts are the entry's, so the report is the same on a machine with no
accelerator and no model cached.

Per site, the status is one of:

| status | means |
|---|---|
| `available` | the tensor exists on the entry, its shape and width are known, and the capability row says which engines read it and what a write may do |
| `undecided` | the entry cannot decide a fact the run will: which mixer the layer carries (a stream-bound component on an entry without `layer_types`), a predicate only the module tree answers (`grouped_mm` on a hand-declared entry, `split_qkv` / `gated_attention` on a family outside the tap table), an attention-interior address the tap table has not measured |
| `refused` | the row or the entry says there is no such tensor (`[V4]`, reason `component_unavailable`), or `head` names an axis the component has none of. The compile refuses these first, so a report of a compiled document never carries one — the status is what `site_report` answers for a site the compile has not seen |

The report's `undecided` list names the facts the run decides, by topic —
each one a line of the report, never an omission:

| topic | decided by |
|---|---|
| `tokenization` | the run's encoding of its inputs (sec. 2.3): the `variable`, `column` and `all` windows and the rows' token widths; the answer tokens a metric matches (sec. 2.10) |
| `pair_validity` | the rows and the tokenizer at run (sec. 2.2): the counterfactual pair's checks; `--data` decides column existence and the declared row roles here |
| `controls` | the workflow that applies the document: a document dry run decides none |
| `stream_at_layer` | the loaded module, on an entry that declares no `layer_types` |
| `module_tree` | the loaded modules: a predicate the entry cannot decide, an unmeasured attention-interior address |
| `inventory` | `registry.inventory`, which needs `layer_types` or a loaded model (sec. 8) |
| `engines` | routing, when no candidate was handed in — pass `--engine` |
| `model` | the first point, when `model.key` is swept |

## 10. Worked examples

An interchange intervention, complete: the header names the file, `model` and
`data` name what it ran on, and `method` is the experiment — the mechanism,
the scoring, and the addresses it reads and writes at.

```json
{
  "header": {
    "protocol_version": "3",
    "title": "Weekdays interchange, Llama-3.1-8B layer 18",
    "description": "Swap the answer-slot residual from the counterfactual into base at layer 18, in bf16, over the weekdays training pairs; IIA scoring."
  },
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main", "dtype": "bf16"},
  "data": {
    "base": {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "input"},
    "counterfactual": {
      "dataset": "natural_domains_arithmetic/data/weekdays#train",
      "field": "counterfactual_inputs[0]"
    }
  },
  "method": {
    "sites": {
      "target": {"component": "block_output", "layers": [18]},
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
        "expected": "cf_answer",
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

`causalab explain` on that document prints the plan and the document digest.
Moving the experiment to another network is an edit to `model` (and, where the
addresses differ, to `sites`), and a diff of the two files says whether the
experiment itself survived the move. A layer scan is a
one-line edit: `"sites": {"target": {"component": "block_output", "layers":
{"sweep": {"range": [0, 32]}}}}` — and the axis is `sites.target.layers`,
section-rooted (§1), in every coordinate label and every workflow reference.

Path patching (sender → receiver, off-path frozen; shows cross-model flow):

```json
{
  "header": {"protocol_version": "3"},
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
  "data": {
    "base": {"dataset": "IOI/data/default", "field": "input"},
    "counterfactual": {"dataset": "IOI/data/default", "field": "counterfactual_inputs[0]"}
  },
  "method": {
    "sites": {
      "sender": {"component": "attention_premix", "layers": [9], "head": 9},
      "receiver": {"component": "block_input", "layers": [12]},
      "a10": {"component": "attention_output", "layers": [10]},
      "a11": {"component": "attention_output", "layers": [11]},
      "lm_head": {"component": "lm_head"}
    },
    "reads": {
      "v_sender": {"site": "sender", "pos": -1, "model": "original", "input": "counterfactual"},
      "v_a10": {"site": "a10", "pos": -1, "model": "original", "input": "base"},
      "v_a11": {"site": "a11", "pos": -1, "model": "original", "input": "base"},
      "v_receiver": {"site": "receiver", "pos": -1, "model": "patched", "input": "base"},
      "logits": {"site": "lm_head", "pos": -1, "model": "final", "input": "base"}
    },
    "writes": {
      "swap_sender": {"site": "sender", "pos": -1, "do": {"swap": "v_sender"}},
      "freeze_10": {"site": "a10", "pos": -1, "do": {"swap": "v_a10"}},
      "freeze_11": {"site": "a11", "pos": -1, "do": {"swap": "v_a11"}},
      "inject": {"site": "receiver", "pos": -1, "do": {"swap": "v_receiver"}}
    },
    "intervened_models": {
      "patched": {"input": "base", "writes": ["swap_sender", "freeze_10", "freeze_11"]},
      "final": {"input": "base", "writes": ["inject"]}
    },
    "metrics": {
      "logit_diff": {
        "kind": "logit_diff",
        "of": "logits",
        "a": "answer",
        "b": "cf_answer",
        "token_form": "space_prefixed"
      }
    },
    "save": [
      {
        "value": "logit_diff",
        "model": "final",
        "input": "base",
        "file_path": "logit_diff.json"
      }
    ]
  }
}
```

DAS with a k × seed sweep (9 fits from one harvest; shows axes + train +
featurizer save):

```json
{
  "header": {"protocol_version": "3"},
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
  "data": {
    "base": {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "input"},
    "counterfactual": {
      "dataset": "natural_domains_arithmetic/data/weekdays#train",
      "field": "counterfactual_inputs[0]"
    }
  },
  "method": {
    "sites": {
      "target": {"component": "block_output", "layers": [18]},
      "lm_head": {"component": "lm_head"}
    },
    "featurizers": {
      "rot": {
        "kind": "subspace",
        "k": {"sweep": [8, 16, 32]},
        "parametrization": "cayley"
      }
    },
    "reads": {
      "v_cf": {
        "site": "target",
        "pos": -1,
        "model": "original",
        "input": "counterfactual",
        "featurizer": "rot"
      },
      "logits": {"site": "lm_head", "pos": -1, "model": "patched", "input": "base"}
    },
    "writes": {
      "patch": {"site": "target", "pos": -1, "featurizer": "rot", "do": {"swap": "v_cf"}}
    },
    "intervened_models": {
      "patched": {"input": "base", "writes": ["patch"]}
    },
    "metrics": {
      "iia": {
        "kind": "logit_diff",
        "of": "logits",
        "a": "cf_answer",
        "b": "base_answer",
        "token_form": "space_prefixed"
      },
      "ce": {
        "kind": "cross_entropy",
        "of": "logits",
        "target": "label",
        "token_form": "space_prefixed"
      }
    },
    "train": {
      "objective": [[1.0, "ce"]],
      "params": ["rot"],
      "optimizer": {"name": "adamw", "lr": 0.001},
      "steps": {"epochs": 10},
      "batch": {"pairs": 16},
      "eval": {
        "every": {"epochs": 1},
        "split": "natural_domains_arithmetic/data/weekdays#test",
        "metrics": ["iia"]
      },
      "early_stop": {"metric": "iia", "patience": 3, "mode": "max"},
      "seed": {"sweep": [0, 1, 2]}
    },
    "save": [
      {"value": "iia", "model": "patched", "input": "base", "file_path": "iia.json"},
      {"value": "ce", "model": "patched", "input": "base", "file_path": "ce.json"},
      {"value": "rot", "site": "target", "file_path": "rot.safetensors"}
    ]
  }
}
```

## 11. Glossary

### 11.1 One word per object — the five names

"Protocol" was doing the work of five different nouns: the methodology, the
installed package, a file on disk, the resolved form that file expands to, and
the metadata a run leaves behind. A sentence like "the protocol digest changed"
was therefore ambiguous between three of them, and a reader could not tell which
without reading the code.

These five names are **normative**. Each object has exactly one, and no other
term in this spec, `docs/workflow_protocol.md`, or a module docstring may be used
for it.

| term | means | is not |
|---|---|---|
| **research pipeline** | the methodology — the sequence of questions a study asks, of which one intervention is a step | any file, and nothing the runtime can hash |
| **runtime implementation** | the installed `causalab` package and the engine that executes an intervention | the specification, which is engine-agnostic by design |
| **intervention specification** | the authored, serializable JSON document — what an author writes, diffs and shares | the resolved form; sugar is unexpanded and derived fields are absent |
| **compiled intervention** | the canonical, sweep-expanded, digested form the runtime produces from a specification and a model — one per point | the specification, and never authored by hand |
| **run receipt** | the metadata a completed run leaves: what was requested, what was observed, what was verified | a scientific result; it records the conditions, not the finding |

Two consequences worth stating, because both were live confusions:

- **Digests belong to the compiled intervention, not the specification.** Two
  specifications that differ only in section order have the *same* digest (§7),
  because the digest is over the compiled form. "The document digest" in this
  spec always means the digest of what the document compiled to.
- **A file has no `type` field.** `protocol_version` 1 carried
  `type: protocol | method | workflow`; version 2 retired it (§1) — a file's
  kind is its shape, and `steps` is what makes a workflow. A workflow *step*
  still says `"type": "intervention_protocol"`, a serialized enum value in the
  workflow format, unchanged and unchangeable without breaking every existing
  workflow, and deliberately not renamed here. The demo format's
  `## The protocol` section heading is the same kind of thing — `docs/demos.md`
  §2 mandates it as one of the seven sections and `tests/demos/test_demos.py`
  pins the literal, so renaming it means every shipped demo, the format doc and
  the test — and it stands for the same reason.

**One place the old wording is deliberately left standing: prose inside a
hashed script.** A workflow script step hashes its module's *bytes*
(`docs/workflow_protocol.md` §7), so a module named by a step is part of an
experiment's identity down to its docstring. Renaming a noun in
`causalab/analysis/harvest_difference.py` moved
`demos/onboarding_tutorial/10_steering.md`'s `harvest` step identity and
invalidated a recorded H100 reproduction — for one word.
Prose in a hashed script is therefore **frozen**: it changes when the science
changes, not when the vocabulary does. Anything else spends a scientific digest
on an edit that means nothing.

`tests/protocol/test_vocabulary_census.py` guards this table, so a sixth name
cannot appear without appearing here. It scans **every `.md` and `.py` file in
the repository** and names its exemptions, rather than listing the directories
it covers. That direction is deliberate: an inclusion list makes each new
directory a silent blind spot, and three rounds of review each found one more
of them. The rule is wider than the scope sentence above — `demos/`,
`tests/` and the repo `README.md` are all prose a reader meets —
because the alternative is re-litigating the scope every time a directory is
added.

Two carve-outs, each with its reason:

| carve-out | why |
|---|---|
| the five hashed script modules | their prose is frozen (above). Listed by name, and the same file checks that list against the modules workflow `script` steps actually name — so a sixth step fails loudly, which this paragraph cannot do |
| the census file itself | it holds the banned phrases as *data*, so it necessarily contains all of them. A genuine offence in its own prose would not be caught |

One limit worth stating, because no wider corpus fixes it: the census reads
source *text*, so a message **composed from an enum value** is beyond it.
the module that implemented the retired method/application split printed one
of the banned phrases to users out of two literals that each looked innocent,
until the enum was quoted
(`` `type: {structural}` ``). Reading a command's real output —
`tests/workflow/test_cli_refusals.py` — is the instrument for that, not a
bigger glob.

### 11.2 Causal abstraction correspondence (Geiger et al., arXiv:2301.04709)

| this spec | causal abstraction |
|---|---|
| `model` | the low-level model ℒ (the high-level model ℋ lives with the task's dataset, not in documents) |
| `intervened_models.<name>` | ℒ_{b∪𝕀} — the intervened model |
| a write's `do` | an interventional 𝕀_X |
| `swap` from a counterfactual read | interchange intervention (`IntInv`; `DistIntInv` when featurized) |
| site + pos + dims | the target variable set **X** |
| featurizer | the translation τ |
| `match` metric | interchange-intervention accuracy (IIA) |
