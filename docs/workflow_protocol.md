# Workflow Protocol — specification v2

Self-contained specification of the second config type: a declarative format
for **chains of intervention runs plus the Python that processes their
outputs** — what the retired *analyses* chains used to be (that package was
deleted in the protocol refactor). A
workflow document composes
**intervention specifications** (`docs/intervention_protocol.md`, "the IM spec"
below — its §11.1 defines the five nouns used here); it never touches a neural
network itself.

Note on one word that is *not* renamed: a **`protocol` step** below always means
"a step whose `type` is `intervention_protocol`". That is the serialized name of
a step type in this format, not a sixth sense of the word, and renaming it would
break every existing workflow document.

## 0. Principles

- **A campaign is a value.** One JSON document describes the whole pipeline:
  which intervention specifications run, how files flow between them, and what
  Python turns one into another. It can be hashed, diffed, shared and re-run —
  a workflow run is a pure function of the document **and the scripts it
  names**, both hashed into the digest (§7).
- **The runner owns execution.** The document says *what depends on what*; the
  runner derives *how* — order, parallelism, resume. There is no
  `sequence:`/`parallel:` construct: independent steps **are** the parallelism,
  and `explain` reports the derived schedule.
- **Six step types, one wiring mechanism.** `protocol` is declarative because
  that is where load-time bite lives; `script` is an escape hatch wide enough
  that the pipeline never has to leave the record; `behavioral` (§2.7) is the
  declarative behavioral runner — a no-intervention document decoded under a
  declared spec, its generations checked against the task's own definition of
  correct and turned into a typed decision; `decision` and `conditional`
  (§2.8) are the two control kinds — a typed decision over a measured values
  object, and a closed predicate over a decision that skips one side of the
  graph — so a preregistered signal gates a downstream fit without the
  orchestration layer interpreting prose. A document step may declare its
  fan-out (§2.9): its children are derived at load and its own name is their
  join. A `workflow` step (§2.10) nests another workflow document — loaded
  once, at the outer's load, its steps join the run as `<step>/<inner>` and
  its digest enters the outer's by reference. Everything they consume is
  spelled one way (§3).
- **Everything declared is published.** A step's declared outputs land in its
  own directory and stay there. There is no `save` section to keep in step with
  the steps, and no sink rule — a terminal plot or report step is legitimate.
- **Two record formats, ever.** JSON for anything structured and readable,
  safetensors for dense numerics. Visualization formats (`.png`, `.pdf`,
  `.html`) are legal outputs but carry no record — a figure is a rendering, not
  an artifact (§2.5).
- **Format**: strict JSON (unknown keys = error); YAML accepted at the authoring
  surface. **v2 scope**: a finite acyclic step graph whose shape is known at
  load — no loops, a fan-out is declared and expands at load (§2.9; no
  run-time fan-out), one workflow per run — a nested workflow (§2.10)
  contributes steps to that run, never a second run; a conditional (§2.8)
  skips steps of that graph and never adds any.

## 1. Document layout

Sections in this order — **recommended, not enforced**, for the same reason
an intervention specification's are (intervention_protocol §5 rule 2): the canonical
form is built from the parsed document, so authored order changes neither the
digest nor the schedule. An unconventional order parses, with a warning.

| # | key | required | content |
|---|---|---|---|
| 1 | `version` | ✓ | `"1"` |
| 2 | `description` | – | free text, the pipeline's intent |
| 3 | `output_dir` | ✓ | the workflow's own directory name — one path segment |
| 4 | `steps` | ✓ | the named step table — the whole pipeline |
| 5 | `pins` | – | the stamped census of everything the workflow touches — documents, scripts, tables, code modules, files — by digest (§7). Written by `causalab pin` or by the first `run`; held to on every later load (§5 rule 21); never canonical |

- **One namespace**: step names are unique, filesystem-safe (`[A-Za-z0-9_-]+`),
  and become the run tree's subdirectories. A declared fan-out's children are
  derived steps named `<step>@<i>` (§2.9) — `@` is outside the authored
  alphabet, so a child can never collide with an authored step. A nested
  workflow's steps are derived steps named `<step>/<inner>` (§2.10) — `/` is
  outside it too.
- A workflow document is distinguished from an intervention specification by its
  `steps` section; the CLI verbs dispatch on it (§9).

### 1.1 `output_dir` and the run tree

`output_dir` is a single filesystem-safe path segment — not a nested path, not
absolute. The CLI supplies the root it sits under:

```
<out-root>/<output_dir>/<step>/<the step's declared outputs>
<out-root>/<output_dir>/<step>/_step.json      # the runner's per-step record
<out-root>/<output_dir>/<step>@<i>/            # child i of a fanned-out step, a step directory like any (§2.9)
<out-root>/<output_dir>/<step>/<inner>/        # a step of the workflow nested as <step>, a step directory like any (§2.10)
<out-root>/<output_dir>/<step>/.attempts/<inner>/<id>/  # its attempt, beside it under the nested workflow's own root
<out-root>/<output_dir>/workflow.json          # the run manifest
<out-root>/<output_dir>/events.jsonl           # the run's event stream, append-only (§4.3)
<out-root>/<output_dir>/.attempts/<step>/<id>/ # a step's attempt, until it is published (§8)
<out-root>/<output_dir>/.attempts/<step>/<id>/attempt.json  # what a failed attempt keeps
<out-root>/<output_dir>/.attempts/<step>/<n>.superseded/    # a prior unit a rerun displaced or a skip retired, retained (§8)
```

A step directory `<step>/` is a **complete unit or absent**: the runner writes
into an attempt directory and publishes it by one rename (§8). `.attempts/`
holds attempts in flight, the bounded metadata of failed ones, and the units a
rerun superseded; a clean first run leaves none, and a step a conditional
skipped (§2.8) has no directory at all. Step names match `[A-Za-z0-9_-]+` (§5, rule 3), so the leading dot
cannot collide with a step, and nothing under `.attempts/` is ever addressable
through the §3 grammar. A nested workflow's steps (§2.10) land under their
`workflow` step's directory, `<step>/<inner>/`, with their own `.attempts/`
beside them; the inner document's `output_dir` is not used — the outer step's
name is where it lands — and the `workflow` step itself has no directory
beyond that root, which is removed when every step under it was skipped.

The split is deliberate. The *name* of a workflow's directory is a property of
the workflow; the *root* is a property of the site. So documents stay free of
absolute paths (IM spec §8) while a workflow still owns where its own outputs
gather. `output_dir` is therefore **excluded from the digest** — it names where,
not what (§7).

## 2. Section reference

### 2.1 `steps` — common fields

Every step is an object with a `type` from the closed set
`intervention_protocol · script · behavioral · decision · conditional · workflow`, plus:

| field | meaning |
|---|---|
| `type` | ✓ — the step vocabulary below |
| `description` | – free text |
| `after` | – step names that must complete first, beyond the derived data dependencies (pure ordering; rare) |
| `requires_receipt` | – `{"step": S, "outcome": "pass" \| "fail"}`: the receipt this step's allocation waits on (§2.8). In the canonical form only when authored |

Data dependencies are **derived, never authored**: a step depends on every step
whose outputs it references (§3) — an `inputs` reference, and inside an
`intervention_protocol` step's document every **run-tree load**: a featurizer's
`file_path`, a params entry's `file_path`, and a featurizer's `init.file_path`
(the basis or theta a fit *starts* from, IM spec §2.5 — so harvest → `fit_pca` →
fit is one workflow). A document's `save` paths are outputs, never loads.
`after` adds ordering without data flow.

### 2.2 `intervention_protocol` steps — run one intervention specification

```json
"locate": {"type": "intervention_protocol", "document": "protocols/locate_scan.json"}
```

| field | meaning |
|---|---|
| `document` | ✓ — path to an intervention specification, relative to the workflow file |
| `set` | – dotted-path overrides applied before loading, same syntax and semantics as the CLI `--set` (IM spec §9). Unlike the CLI form these are **part of the record**: they enter the canonical form and the digest |
| `max_points` | – override of the sweep point cap for this document (IM spec §5.14) |
| `execution` | – this step's row bounds, overriding the engine's for this step only: `batch_rows` (rows per no-grad forward) and/or `fit_rows` (rows per grad forward of a fit — the members of a fit cohort are packed into forwards under it; IM spec §8), each a positive integer or `null` for unbounded. Execution, not identity: unlike `set` and `max_points` it enters **neither** the canonical form nor the digest, and the step's `_step.json` is its one recorder. Unknown keys and non-positive values are refused (rule 1) |
| `control` | – the step **is a control** of another step: `{"of": <step>, "kind": <kind>, "seam": <seam>, "seeds": [...], "min_draws": n, "non_equivalence": {"fields": [...], "reason": "…"}}` — `of` and `kind` required, the rest optional (controls, below; `non_equivalence` names the equivalence fields the control knowingly differs from its target in, rule 16). In the canonical form only when authored |
| `waive` | – the controls this step waives, `{<kind>: <reason>}` or `{<kind>: {"reason": <reason>, "reference": <ref>}}` (controls, below). Only when authored |
| `stop_after_failure_rate` | – on a control step only: the fraction of its points that may fail certification before the certifying step is `failed` and its dependents `blocked` (§8); `0.0` unauthored. Only when authored |
| `fan_out` | – `{"over": {"axis": A} \| {"shards": N}, "join": {"require": "all" \| "selected"}}`: the step's declared fan-out (§2.9) — its compiled points are partitioned into children `<step>@<i>` at load and its own name is their join. In the canonical form only when authored |

- The document is compiled through the IM spec's **one compiler**
  (`compile_protocol`, IM spec §9.1) — the same call `causalab validate` and
  `run_protocol` make — with the workflow's resolution environment (§3).
  Validation and execution differ in exactly one input, the artifact store
  (the deferring store at load, the run-tree overlay at run), so a step cannot
  validate under one resolution and execute under another; every load error of
  the inner document is a load error of the workflow.
- The step's outputs are exactly the inner document's `save` manifest, landing
  under `<step>/`. **The inner specification keeps its own `save` section** — that
  is what declares a protocol step's output names, and it is what §5 rule 4
  checks references against. It is a different section from the one v1 had at
  the workflow level, which is gone.

**Controls as native conditions.** A control is an ordinary intervention
specification run as an ordinary protocol step — it shares the primary
execution path by construction, and no control is ever materialized into a
document that did not author one. What this layer adds is the *declaration*
that a step is a control of another (`control`), the *requirement* that a fit
or a controlled step has each required kind declared or explicitly waived
(`waive`; §5 rule 14 — a caller may waive a control, never silently omit it),
and the *record* of each control's status per point, inherited by every
dependent step (§8). The declaration lives here, on the workflow — the
application protocol — and not inside the intervention document's `method`:
the seeds of a control are how a campaign applied a method, not part of the
method, so two campaigns differing only in control seeds share every inner
document digest and differ in the identity of the step that declares them.

```json
"fit":     {"type": "intervention_protocol", "document": "protocols/das.json",
            "waive": {"self_swap": {"reason": "external", "reference": "runs/2026-09-01/self_swap"}}},
"control": {"type": "intervention_protocol", "document": "protocols/random_subspace_control.json",
            "control": {"of": "fit", "kind": "matched_random", "seeds": [0, 1, 2], "min_draws": 3}}
```

`control.of` names the target and adds no edge to the schedule: a step
inherits a control's status (§8) only when the control or its certifier is
among its ancestors through `after` or an artifact reference — today a target
that names no `after` and reads nothing from the certifier inherits nothing.
An implicit `of → certifier` edge is a possible later addition, not part of this revision.

The kinds are a closed vocabulary (`CONTROL_KINDS`, `causalab/workflow/document.py`;
census `tests/workflow/test_controls.py`). Four, not seven: the source-to-source
no-op is the same document as the self-swap; the full-component and
untouched-layer controls are documents any author writes (a featurizer omitted,
a site swept) and materializing them is forbidden — `full_component` *declares*
such a document, so its site is held equivalent to its target's (rule 16) and
its measured ceiling is on the record, while the untouched-layer control is a
site swept and needs no kind; a cohort stratification is not an intervention.
Every fit and every step some control names is held to `self_swap` and
`matched_random` — each declared by a step or waived (rule 14); `full_component`
is never required: full-component controls stay authored documents.
A workflow that authors no `control` and no `waive` anywhere predates the
layer and is not held to it; declaring one control commits the workflow to the
requirement for every fit it runs in its own steps — the requirement is per
workflow document, and a nested workflow (§2.10) is held by its own document's
declarations, none in this version (rule 20).

| kind | what the control document is | what is checked at load, against the compiled documents |
|---|---|---|
| `self_swap` | the bit-exact no-op leg: an intervened model whose every write swaps in a read taken from `original`, on the model's **own** input, at the write's own address — site, `pos`, `featurizer`, `dims` equal (IM spec §5 rule 21 admits equal depth). A same-target donor by construction: rows pair by index, so the operand is the target's own row | the predicate above holds for some intervened model of the document — refused naming the first field that fails; and a script step certifies it (below) |
| `matched_random` | the size-matched random control of a fit: an untrained `subspace` at the fit's `k` with `seed` swept (`configs/protocols/random_subspace_control.json`), or a gate drawn by `causalab.analysis.random_mask` at a recorded seed and applied | `of` declares a fit; for every featurizer the fit trains, the control holds one of the same `kind`, `k` and `group`, written at the same site (unless `non_equivalence` declares the site field, rule 16); `seeds` are present, distinct, at least `min_draws` (20 unauthored — authored lower is *recorded* lower); and they are the draws the document makes — its featurizer's `seed`, or the `seed` of the script step that drew the bundle it loads |
| `shuffled_source` | a counterfactual role permuted under a declared seed — the label control: the target's document with `data.counterfactual.shuffle: {seed: <int>}` (IM spec §2.2) as its **one** difference: a seeded permutation of that role's row order (`random.Random(seed).shuffle`), the base role untouched. A permutation may leave **fixed points** — seed 0 over 4 rows gives `[2, 0, 1, 3]`, row 3 meeting its own base row; 9 of the 24 orders of 4 rows have none — and nothing excludes them; a certifier of this kind must account for them | the control's compiled document equals `of`'s canonical form with `shuffle` masked — refused naming the first differing field; at least one counterfactual role of the control authors `shuffle` and none of the target's does. Its points are `passed` when they ran, as `matched_random`'s are; its waiver is never required |
| `full_component` | the whole-component swap at the target's site — the parent of a learned intervention, through no featurizer and no `dims`; the full-component score is the measured ceiling and need not be 1.0; below the readout cell a sparse mask may beat it — its status is `passed` when the swap ran (a comparison against the fit is a reduction, §2.6) | every write its intervened models list goes through no featurizer (or only `identity`) and names no `dims`; and it is site-equivalent to its target (rule 16, below) with `featurizer` — and so `dims`, which index the featurized value — allowed to differ: that difference is the kind. Never required by rule 14 |

**Site equivalence — a comparison is refused unless its two sides act at one
address.** A control whose kind demands *coverage* of its target's site
(`full_component`, `matched_random`; `COVERAGE_KINDS`) is compared against its
target at load over the **expanded points of both**: every write an intervened
model lists, at every point, is a typed site tuple
(`causalab/protocol/equivalence.py`, torch-free, imported by the workflow
document model only), and the two are equivalent when every field below agrees
and they share no coordinate system. A parent and a learned intervention that
differ in layer coverage, pre/post-projection site, DeltaNet inclusion,
routed-rank identity or gate sharing are **refused** (rule 16) naming the
field, unless the control's declaration carries `non_equivalence: {"fields":
[<field>, …], "reason": "…"}` naming every differing field — and a declared
field the pair does not differ in is refused too, since a declaration records
a real difference. Not `self_swap`: a self-swap certifies per point, and
per-point *agreement* by coordinates (§8) is the right semantics there. The
fields are a closed vocabulary (`EQUIVALENCE_FIELDS`; census
`tests/workflow/test_equivalence.py`):

| equivalence field | what it compares | decided from |
|---|---|---|
| `component` | the component name, pre/post-projection sites included — `attention_premix` (the o-projection's input, head space) is not `attention_output` (its output, the residual stream); `delta_premix` is neither | the site |
| `shape` | the component's tensor shape on the model (`component_shape`) — two components of one name on two shapes are two spaces | the registry entry |
| `layers` | the band each point's site spans (IM spec §2.4), compared as the set of per-point bands — a control pinned to one layer against a target sweeping two is not equivalent, and a band `[3, 4]` at one point against a sweep over 3, 4 is not either: coverage-equal, intervention-different, since a band is one site and one intervention across its members; empty for a layerless component | the sites of every expanded point |
| `head` | the head a site names, or none — a head on a component with no head axis is said so | the site, bounded by the shape |
| `expert` | the routed expert a site names, or none | the site |
| `stream` | the mixer stream at each covered layer — DeltaNet inclusion: the site's declared `stream`, else the component's bound stream, else the registry's `layer_types` at the layer, else `full_attention` on a tower that declares no linear-attention mixer at all (it can carry no DeltaNet layer), else *unknown* — compared as such and said so | the site and the registry entry |
| `routed_rank` | `(num_experts, num_experts_per_tok, moe_intermediate_size)` on a component of the MoE block; none elsewhere | the registry entry |
| `featurizer` | the write's featurizer chain as shapes — each stage's `kind`, `k`, `parametrization`, `group`, `axis` and `units` (the axis a gate parameter indexes and how many units of it: a position gate's window length) and parameter `dtype`; **not** its `seed`, `init` basis or bundle bytes, which are values (the matched random control differs from the fit in exactly those) | the featurizers |
| `dims` | the coordinates of the featurized value the write covers — the authored `dims`, or every coordinate of the chain's output width when unauthored | the write and the shape |
| `sharing` | whether the two are expressed in **one** coordinate system — decided, not guessed: inside one document a featurizer *name* is one stage instance (`build_stack` caches by name), so two writes naming one featurizer share; across documents a name is only a name, and sharing is bytes — a document that loads the bundle another saves is scored in that other's basis. Listed as differing when the pair *shares*: a control in its target's own basis is no control | the featurizers, `file_path` and `save` |

The model's realization (`model.key`, `dtype`, `revision`) is not a site fact
and is compared under its own rule; what a script step does to a bundle it
reads is not decidable at load, so a bundle drawn *from* a fit's bundle
(`causalab.analysis.random_mask`) is `distinct` here and the run-time
`produced_by` stamp carries that chain. `13_random_subspace_control_im.json`
against `04_das_im.json` is equivalent with no declaration — only the
rotation's values differ. The verdict is on the control step's record (§8).

A waiver names why, from a closed set (`WAIVER_REASONS`):

| reason | waives | when it applies |
|---|---|---|
| `no_fit` | `matched_random` | the target declares no `train` — there is no rank or mask to match. Refused on a fit |
| `single_role` | `shuffled_source` | the document has no counterfactual role to permute |
| `external` | any kind | the control ran elsewhere; **must** carry a `reference` — the run, artifact or document it lives in |

A control's status per point (`CONTROL_STATUSES`), and the two words a
dependent inherits (`INHERITED_STATUSES`; §8):

| status | meaning |
|---|---|
| `passed` | the point certified — for `self_swap`, all three legs below held. For `matched_random`, `shuffled_source` and `full_component` the word means only that **the declared draw (the permutation, the swap) ran**: the pairing was checked at load, the comparison against the target is a reduction (§2.6), never a status, and nothing numerical was decided; the word stays `passed` because it is what a dependent inherits and the vocabulary is closed |
| `failed` | the point did not certify; on the stream it is a `warning` with `reason: instrument_failure` (§4.3) |
| `waived` | the kind is waived on the target step, with its reason, so no point carries a status for it |
| `not_run` | no certified point of the control agrees with this point's coordinates, or the certifier has not run — and a `self_swap` point's word on its control's own record until it has |
| `instrument_invalid` | a dependent point whose control `failed` at the agreeing coordinates — inherited, never authored |
| `instrument_failure` | the control point's own word on the event stream when it `failed` |

A replay control may name the seam its failure rate is measured on
(`CONTROL_SEAMS`); the seam is recorded beside the rate and nothing on this
tree checks a tolerance against it:

| seam | what changes between parent and replay |
|---|---|
| `A` | nothing — in-process re-execution |
| `B` | the artifact round-trip — a rotation saved and reloaded |
| `C` | batch order — the same rows reversed |
| `R1` | a parametrization re-materialized from its pre-image — not exact under TF32; a control crossing it declares a non-zero `stop_after_failure_rate` |

**Certification — a bit-exact no-op is necessary but insufficient.** A
self-swap that writes a value back where it was read leaves the receiver
untouched whether or not the intervention it controls does anything, so a
`self_swap` control certifies per point only when three legs hold together:
(i) **identity** — the receiver read under the self-swap model equals the same
read under `original` bit for bit; (ii) **positive sender effect** — the
operand the target model writes differs from the value it overwrites; (iii)
**changed receiver** — the receiver under the target model differs from the
receiver under `original` (legs ii and iii `not allclose`, `atol` `1e-4`
unless the campaign measured its own floor). The legs are decided from the
control document's **saved reads** by a *certifying script step* —
`causalab.analysis.certify_control`, which reads the five bundles (the
receiver under `original`, the self-swap model and the target model; the
operand; the overwritten value) and writes `controls.json`, one row per point.
A certifying step is recognized by its shape: it declares an output whose
file is `controls.json` and reads exactly one step declaring `control`; the
runner hands it that declaration under the input name `control` (rule 14
refuses a step authoring one, as rule 12 refuses `reduction`). A `self_swap`
control that no step certifies is refused: a no-op nobody certifies records
nothing. The fourth leg of the bar — agreement with an independent oracle at
`atol=1e-5 rtol=1e-4` — is **test-side only**
(`tests/neural/engines/pytorch_hooks/hook_oracle_lib.py`): nothing shipped
implements a write oracle, and no row claims one.

**Qualify once, before the fanout** (rule 15). A control qualifies its
target's *points*, and a target's rank × seed fanout is one step's expansion —
so the control runs **once** for the whole fanout, never once per fit. The
schedule makes that true without an authored `after`: every `control.of` adds
an implicit edge from the target to the control's certifying step (to the
control itself for a kind that needs none), so the target runs after the
qualification and its record inherits the statuses by coordinates (§8). A
control that already depends on its target — the `fit → random_mask → apply`
chain draws from the fit's bundle — is post-hoc by construction and keeps its
direction; that is `matched_random` only (`POST_HOC_CONTROL_KINDS`): a
`self_swap` or `shuffled_source` control authored to run after its target — an
`after` entry or a reference into it — is refused under rule 15 naming the
authored route, since its qualification is what the target's fanout inherits
and the authored direction would silently drop it; two steps that would each
qualify the other are refused naming both. The edge is derived, not authored:
it is in the schedule and not in the
canonical form (§7), so declaring a control moves no digest but its own.
Qualification is under the target's **exact production fingerprint**: a
control and its target are one realization — the compiled `model` as
`canonical_model` materializes it (`key`, `revision`, `dtype`, `quantization`,
and the attention backend when authored — the list is `canonical_model`'s, not
this spec's), equal field for field, refused naming the first field that
differs (`model.dtype`, `model.revision`, `model.attn_implementation`, …; a
backend authored on one side only is refused naming the omission, since the
omission is the engine's default and the two documents digest differently) —
and a campaign that changes the realization on **both** steps is re-qualified,
not refused. Realization is the model; packing (`execution`) is not — a
control at `fit_rows: 8` qualifies a target at `fit_rows: 512`. The identity
of a qualification is what a dependent's record carries beside each inherited
status (§8 `controls.identity`): the control's document digest, the
`tree_digest` of the code that ran it and the engine. Over-keying is
impossible by construction — nothing outside the compiled document and the
code that ran it — not the `execution` block (a step may author its row
bounds; the rest is measured), not a device, not an install path — is in the
triple, so two runs of one campaign on two machines share one qualification
identity, and a control record from another tree — or one under another
engine than the step would run under now — is re-run by `--resume` rather
than reused (§7, §8). The arithmetic follows: N points failing
qualification invalidate N points of the target — every fit at those
coordinates — not N × fanout fits attributed one by one.

**Why `protocol` stays declarative.** A protocol step is not "a script that
happens to take a document". Keeping it a type preserves, all at load time:
inner-document validation (every IM spec §5 rule, before any step runs); sweep
expansion, so point counts and per-point digests reach `explain`; engine
capability routing over the union of the points; `--points` shard dispatch; and
the producer's **sweep axes**, which is what makes a downstream reduction
meaningful at all (§6).

### 2.3 `script` steps — inputs, one Python script, declared outputs

```json
"steer_direction": {
  "type": "script",
  "script": "scripts/harvest_difference.py",
  "inputs": {
    "acts_pos": {"step": "harvest_pos", "file": "acts.safetensors"},
    "acts_neg": {"step": "harvest_neg", "file": "acts.safetensors"},
    "normalize": true
  },
  "outputs": {"direction": "direction.safetensors", "stats": "stats.json"}
}
```

| field | meaning |
|---|---|
| `script` | ✓ — a **locator**: `{"module": "causalab.analysis.fit_pca"}` or `{"path": "scripts/probe.py"}` (§2.4) |
| `inputs` | ✓ — `{name: value}` in the §3 grammar; each reference **is** a derived dependency edge |
| `outputs` | ✓ — `{slot: file}` under the step dir, non-empty |
| `runtime` | – dependency isolation (§4.1) |
| `reduction` | – the reduction contract (§2.6): what a number this step publishes *is* — statistical unit, grouping, weighting, missing-value policy, uncertainty procedure, resampling unit, repetitions, seed. Validated at load (§5 rule 12); in the canonical form and the digest **when authored** |
| `is_deterministic` | – default `true`; see §7 |

A script step is what makes the pipeline expressible without leaving the
record. It replaces v1's `transform` (a closed, versioned op registry),
`select` and `plot` step types: those were three vocabularies for "run some
Python over the previous step's output", and the registry's admission-by-pull-
request rule meant a one-off corpus-mean harvest could not be written at all.
The reductions v1 had as step types survive as **shipped scripts**, each filed
by subject rather than in one namespace: `causalab.workflow.scripts.select`,
`causalab.io.plots.workflow_figures`, and `causalab.analysis.*` (§2.4).

**Script steps are for deterministic Python analysis.** LLM and judge-style
work stays outside causalab — the protocol layer's determinism is what makes it
digestible, and judging lives in the **research pipeline** — the methodology
that consumes these outputs (§11.1 of the intervention protocol spec).

#### The output declaration

```json
"outputs": {
  "spectrum": {"file": "spectrum.json",
               "columns": {"component": "int64", "explained": "float64"}},
  "weight": "basis.safetensors"
}
```

The short form is a bare filename. `columns` is optional, allowed only on a
`.json` output, and is a promise about what the step publishes — verified **on
write** (§4). Because the digest covers outputs, a declared column set is part
of the record. Keeping the declaration in the *document* rather than inside the
file is what lets an empty table still be checked: there are no rows to infer
from, but there is always a declaration.

A `.json` output may instead declare **`keys`** — a flat *values object* rather
than a table, with one representative value per key:

```json
"outputs": {
  "values": {"file": "values.json",
             "keys": {"best_layer": 18, "best_pos": {"index": -1}}}
}
```

`columns` and `keys` are mutually exclusive: the first says "an array of row
objects", the second "one object mapping these names to values". `keys` is what
the `key` selector (§3) reads, and it is **load-bearing rather than
documentation**. A protocol step whose `set` pulls a value out of an earlier
step (`{"artifact": "best", "key": "best_layer"}`) cannot resolve it before the
run, so the loader substitutes the declared representative and validates the
inner document against *that* — which is why the representative is a value and
not a type: a position spec must type-check as a position spec, and `"int64"`
would not. The real value replaces it at run time and goes through the same
compiler (IM spec §9.1), whose result for the run equals the one validation
saw exactly when the representative equals the emitted value; a `file_path`
loaded from a step's run tree is reported by that compile as a
`deferred_check` diagnostic rather than checked against the wrong document.

v1 derived these representatives from a `select` step's `emit` table plus the
producing document's sweep axes. With `select` a script, the document has to
say it, and saying it is cheap.

### 2.4 Addressing a script, and `file` vs `path`

`script` is a **locator**, the same shape an `inputs` reference uses (§3):

| form | resolves to |
|---|---|
| `{"module": "causalab.analysis.fit_pca"}` | an importable module, found via `importlib.util.find_spec` — which resolves a dotted name to a file **without executing it** |
| `{"path": "scripts/probe.py"}` | a file beside the workflow document, contained, no parent escapes |

The shipped scripts are filed **by subject**, not in one flat namespace:

| module | what it is |
|---|---|
| `causalab.analysis.fit_pca` · `harvest_difference` · `head_stats` · `paired_ttest` · `random_mask` · `subspace_angles` | numerical analysis — fits, statistics, controls (a size-matched random mask for a DBM fit), the principal angles between two saved subspaces, and the operands an intervention consumes |
| `causalab.analysis.project_pca` · `pca_by_position` · `sequence_activations` | reuse a frozen training mean/basis; fit separate position populations; gather predictive activations from a shared sequence harvest |
| `causalab.io.plots.workflow_figures` | rendering, beside the rest of `io/plots/` |
| `causalab.workflow.scripts.select` · `reduce` | the two scripts whose purpose *is* the seam between steps: `select` picks the values a later document's `set` reads; `reduce` publishes a number under a declared reduction contract (§2.6) |

v1 spelled a shipped script `causalab:<name>`. That needed a registry — exactly
what this layer removes — and it hid *which* code ran behind a lookup. A module
path says it, and a script can then live where it belongs by subject instead of
where a resolver happens to search.

**`file` vs `path`.** Both words appear in a document and they are not
interchangeable:

- **`file`** is always *a name within some step's output directory*, declared by
  that step — `{"step": "harvest", "file": "acts.safetensors"}` on the way in,
  and a key of `outputs` on the way out. It is a filename, never a location;
  the runner owns where the step's directory lives. A `file` is checkable
  against a declaration.
- **`path`** is a location on disk that the workflow does not own — absolute, or
  relative to the directory of the document that names it (the base a script's
  `path` locator already uses, and the only one that exists for an installed
  package as well as a checkout). A `path` is checkable only against the
  filesystem.

### 2.5 Formats: two that carry the record, three that visualize it

**Record formats — two, and nothing else.** **JSON** for everything structured
and readable (metric tables, values objects, manifests, stats), **safetensors**
for dense numerical weights and tensor artifacts. A metric table is a **native
JSON array of row objects** (`causalab/protocol/tables.py`):

```json
[
  {"example": 0, "sites.target.layers": 18, "value": 0.83},
  {"example": 1, "sites.target.layers": 18, "value": 0.91}
]
```

Labels repeat on every row. That is the deliberate trade — a file `jq` and a
human can both read, at the cost of size. There is no envelope and no embedded
column header: inventing a format inside a format would defeat the point of
having only two. One file per metric, so a document saving three metrics writes
three tables. Non-finite floats are written as `null`: bare `NaN`/`Infinity`
tokens are not JSON, and a metric that computed nothing is exactly the "no
value" a null means.

**Visualization formats — `.png`, `.pdf`, `.html`.** These are legal declared
outputs but carry **no record**: a figure is a *rendering* of an artifact rather
than an artifact itself. So a visualization output may declare no
`columns`/`keys`, and the runner neither checks its shape nor stamps it with an
ArtifactIdentity — a non-empty file that starts with its format's signature
(`.png`, `.pdf`; `.html` has none) is the whole contract (§8).

- **`.png` is the default and is preferred over `.pdf`** unless a document asks
  for pdf explicitly. It is what a reviewer can open inline, in a PR, or in a
  notebook without a viewer. `.pdf` is for print or when a vector figure is
  genuinely needed; `.html` for interactive figures.
- The preference is implemented in one place —
  `causalab.io.plots.figure_format.normalize_figure_format(value, default="png")`
  — so every renderer inherits it rather than restating it.
- A step that renders should usually declare the **numbers as well**: the
  shipped `workflow_figures` script takes an optional `plotted` table output
  holding the exact rows it drew, which is what makes a figure checkable and
  lets a later step reference what it showed.

### 2.6 `reduction` — the reduction contract

A number published from a metric table is a *reduction*: rows became one
value, and eight decisions were made on the way — what one independent
observation was, how the rows were grouped, whether they were weighted, what
happened to a `null`, whether an interval was computed and by what procedure,
what was resampled, how often, and from which seed. Until now every one of
them was made by code (`aggregate`'s pandas defaults) and none was in the
record: a workflow review could not answer whether an interval was across
facts, tokens, rows or shards. A `script` step may now **declare** them:

```json
"facts": {
  "type": "script", "script": {"module": "causalab.workflow.scripts.reduce"},
  "inputs": {"table": {"step": "trace", "file": "aie.json"}},
  "reduction": {
    "estimator": {"kind": "mean"},
    "unit": {"kind": "example", "columns": ["fact"]},
    "group_by": ["sites.target.layers"],
    "weight": null,
    "missing": "exclude",
    "uncertainty": {"kind": "percentile_bootstrap",
                    "resample_unit": {"kind": "example", "columns": ["fact"]},
                    "repetitions": 2000, "seed": 42}
  },
  "outputs": {"table": {"file": "aie_by_layer.json",
                        "columns": {"value": "float64", "n": "int64"}}}
}
```

That block is ROME's "fact-level percentile bootstrap, 2,000 resamples, seed
42", authored rather than external. Every field is required — `weight` as an
explicit `null` — and every vocabulary is closed (`causalab/workflow/reduction.py`;
the docs↔code census is `tests/workflow/test_reduction_census.py`).

**The eight dimensions**

| dimension | field | value |
|---|---|---|
| statistical unit | `unit` | `{"kind": <unit>, "columns": [...]}` — the vocabulary member **plus** the column(s) that identify one observation in *this* table; only the campaign knows its key. `row` names no column |
| grouping | `group_by` | column names, one output row per distinct combination; `[]` for one row |
| weighting | `weight` | a column name, or `null` |
| missing-value policy | `missing` | one of the policies below |
| uncertainty procedure | `uncertainty.kind` | one of the procedures below |
| resampling unit | `uncertainty.resample_unit` | as `unit`; **may differ** from it (ROME resamples facts over fact × token rows) — on a curve under a `unit`, not `row`, and each unit within one cluster (decided on the table) |
| repetitions | `uncertainty.repetitions` | a positive integer |
| seed | `uncertainty.seed` | a non-negative integer — what seeds the `numpy.random.Generator`; `random` is never consulted |

plus the verb, `estimator` (`{"kind": <estimator>}`, with `q` for `quantile`,
and for the two curve estimators `x`, `y`, `x_scale`, `normalize`, `grid` —
"Curve estimators" below). The unit vocabulary makes a recurring confusion checkable — row, pair,
prompt, source family and connected component are **not** interchangeable,
especially when counterfactual roles reuse prompts, and a table reduced under
two of them is two different numbers with two different digests:

| unit | one observation is |
|---|---|
| `row` | one row of the table; nothing is collapsed |
| `example` | one example — the `example_id` column a protocol run stamps per example (the base row's label, IM spec §2.2), or a campaign's own key (`fact`) |
| `pair` | one counterfactual pair — its base and counterfactual rows together |
| `prompt` | one prompt, across every pair that reuses it |
| `source_family` | one source family of prompts |
| `component` | one connected component of the prompt-reuse graph |

**What `unit` does.** Rows sharing the unit key are collapsed to one
observation before the estimator runs: by their mean (weighted, when a weight
is declared) for `mean`, `weighted_mean`, `median` and `quantile`; by their sum
for `sum`; `count` counts units. `n` counts observations, never rows.

| policy | a `null` value (or weight; on a curve, a null `x` too) |
|---|---|
| `error` | refuses the table at run time, naming how many and how many came from `matched: false` |
| `exclude` | is dropped before the estimator runs, and the count is recorded (`n_excluded`) — what pandas' `skipna` did silently |
| `zero` | is scored `0.0` — the author's statement that "the model never said it" counts as a zero, not as absent; refused on a curve when the null is an `x` (an abscissa has no zero) |

| procedure | interval | takes |
|---|---|---|
| `none` | no interval | nothing — declaring `resample_unit`, `repetitions` or `seed` under `none` is refused: a field that governs nothing may not be declared |
| `percentile_bootstrap` | the 2.5th and 97.5th percentiles of the estimator over `repetitions` resamples, each drawing resample units **with replacement** and keeping each unit's rows together (a cluster bootstrap when `resample_unit` is coarser than `unit`; on a curve a resample unit that splits a unit is refused at run time); a unit drawn twice is two observations | `resample_unit`, `repetitions`, `seed` |
| `normal_approx` | estimate ± 1.96 × the sample standard deviation (`n−1`) of the estimator across resample units, over √k; fewer than two units gives `null` bounds | `resample_unit`; pairs only with `mean` / `weighted_mean` |

Both intervals are 95% — one convention, not a ninth dimension. Every group's
resampler is seeded from the declared seed plus a stable word derived from the
group's coordinates, so a group's interval does not depend on which other
groups the table holds.

| estimator | value | note |
|---|---|---|
| `mean` | arithmetic mean of the observations | `weight` must be `null` |
| `weighted_mean` | Σ w·v / Σ w over the observations | requires a `weight` column; a zero weight contributes nothing |
| `sum` | sum of the observations | the numerator of a mean composed across tables |
| `count` | the number of observations (units) | the denominator that makes `sum` composable |
| `median` | the lower of the two middle values at even `n`, **no interpolation** | the same choice as `save.reduce`'s `median`: one verb, one meaning |
| `quantile` | the `q`-quantile, linearly interpolated | `q` in (0, 1) |
| `auc` | the trapezoid area under the curve of per-`x` means of `y`, over the sorted distinct `x` | a curve estimator: `x` (required), `y` (optional, else the value column), `x_scale` (required: `linear` \| `log`), `normalize` (optional: `x` is divided by it — a positive number, or a column constant and positive within the group); `weight` must be `null`; one distinct `x` is refused (no area) |
| `cpr` | MIB's circuit-performance ratio: `faith(p) = (m(N − int(p·N)) − C)/(B − C)` over the kept-fraction grid `p`, integrated by trapezoid over the raw `p`; `B = m(0)`, `C = m(N)` | a curve estimator: `x` holds the **cuts** (units taking the counterfactual, `top_k`), `normalize` is **required** and is `N` (an integer ≥ 2, or a column holding it), `x_scale` `linear` (MIB) \| `log`, `grid` optional (default MIB's ten points `0.001 … 1.0`); `weight` must be `null` |

| scale | abscissa |
|---|---|
| `linear` | the abscissa as it is — `x` (÷ `normalize` when given) for `auc`, the raw kept fraction `p` for `cpr`; MIB's integral |
| `log` | its natural logarithm — `log x` for `auc` (an `x ≤ 0` is refused at run time: a cut of 0 has no logarithm), `log p` for `cpr`; equal weight per decade of the sweep |

**Curve estimators.** `auc` and `cpr` reduce a *curve* — a table with one row
per (example, cut), the metric table a `top_k` sweep through a §3.2 axis writes
(`axes.cut`, `value`) — to one area. Their observation is a point on the curve:
rows sharing the unit key **and** an `x` value collapse to one observation by
their mean (`unit: row` collapses nothing), and the curve is the per-`x` mean
of the observations, `m(x)`. `n` counts those observations. `auc` integrates
`m` over the sorted distinct `x` as they are, over `x ÷ normalize` when a
ceiling is given, and over `log x` under `x_scale: log` (an `x ≤ 0` is refused
at run time — a cut of 0 has no logarithm). `cpr` is
`MIB_circuit_track/evaluation.py`'s `evaluate_area_under_curve` cut by cut: `N`
is the unit count, the anchors are `m(0)` (the clean model, MIB's `B`) and
`m(N)` (fully corrupted, `C`), each grid point `p` reads the row at cut
`N − int(p·N)` — the polarity trap: MIB's `p` is the fraction kept *clean*,
causalab's `top_k` counts units *patched* — and the value is the trapezoid of
`faith` over the raw `p` (span 0.999, not normalised). Several `p` that floor
to one cut (MIB's three smallest, on 157 units) repeat that cut's ordinate at
distinct `p`: a flat segment, kept as MIB keeps it. An authored `grid` is read
the same way, `N − int(p·N)` in floating point, so a `p` whose product with `N`
is mathematically an integer can land one cut low — `0.29 × 100` is `28.999…`,
cut 72, the row for 28 kept, not 29 — check an authored point against the cut
it selects. A curve presumes a **balanced panel**: under a `unit` other than
`row`, every unit has a row at every `x` the curve reads — `cpr`'s anchors and
grid cuts, `auc`'s every distinct `x`; a unit missing one is refused naming the
`x`, rather than letting the per-`x` means (`B`, `C` and `m(cut)` under `cpr`)
average different populations, or a bootstrap draw over an `auc` panel lose an
`x` outright and integrate a different grid (MIB's evaluator averages whatever
is there; a declared estimand refuses — `missing: exclude` on a null row leaves
such a hole, and says so). The panel is balanced in *units*, not in rows: a
second axis a `group_by` leaves out (a `method` absent at one cut) keeps every
unit a row at every cut and still averages different populations — declare the
axis in `group_by`. Neither `group_by`, `unit.columns` nor
`uncertainty.resample_unit.columns` may name the abscissa: grouped by `x` every
curve is one point, a unit keyed on `x` holds no curve, and resampling cuts
resamples the grid so the interval would be a spread over grids, not over the
population — all refused at load, as is `resample_unit: row` under a declared
`unit` (rows are inside every unit). The panel makes a draw of whole units a
draw of whole curves only if each unit lies in one resample cluster, and that is
the table's to say, not the column names': a coarser unit declared by its own
column (examples nested in templates, the two-level cluster bootstrap) passes, a
resample unit that splits a unit across clusters (a `method` inside an
`example`; a null in a unit column) is refused at run time naming how many units
it splits — otherwise a draw could integrate a grid the table does not have,
`cpr` refusing the lost cut and `auc` integrating the shorter grid silently. A
missing anchor, a missing grid cut, a non-integer `x`, `B = C` (to within `1e-9`
of the largest |mean| among the anchors and the cuts the curve reads, a
scale-free floor), an `x` or a `normalize` column that is the column being
reduced (the default `y` is that column, known only at run time) and a hole in
the panel are run-time refusals naming the cut (the `x`, under `auc`). An
infinite `x` is refused once over the whole table, before any group is read and
before `missing` is applied — a null `x` is `missing`'s (`error` raises,
`exclude` drops the row; `zero` is refused on a curve — an abscissa has no
zero), an infinite one is the table's, named with its rows and values whatever
`missing` says; an infinite `normalize` column is refused at its group by the
parser's own finiteness rule. Neither reaches a table causalab wrote
(`write_table` turns a non-finite float into `null`), so an infinite value is a
foreign table's or a direct caller's. A refusal raised inside a bootstrap draw
says which repetition and that it is the draw's (a draw that lacks a point, or
whose anchors coincide) — under a declared `unit` a draw is whole curves and
cannot lose a point, so that refusal is `unit: row`'s, or coinciding anchors.
Under `unit: row` there is no panel and no cluster rule, so a `resample_unit`
there draws whatever `x` its clusters hold — `cpr` refuses a lost cut inside the
draw, `auc` integrates the `x` the draw contains and says nothing; declare a
`unit` for the draw to be whole curves. Neither takes a `weight` (MIB's means
are equal-weight; a weighted curve would be another estimand, not a flag) and
`normal_approx` stays a mean's; a `percentile_bootstrap` redraws whole resample
units and recomputes the curve each time — a cluster bootstrap of the area. The
`cpr` block for a gpt2 IOI sweep — the number a MIB Figure-1 comparison
reads against the paper:

```json
"cpr": {
  "type": "script", "script": {"module": "causalab.workflow.scripts.reduce"},
  "inputs": {"table": {"step": "apply", "file": "ld.json"}},
  "reduction": {
    "estimator": {"kind": "cpr", "x": "axes.cut", "x_scale": "linear", "normalize": 156},
    "unit": {"kind": "example", "columns": ["example"]},
    "group_by": [], "weight": null, "missing": "exclude",
    "uncertainty": {"kind": "none"}
  },
  "outputs": {"table": {"file": "cpr.json"}}
}
```

**Two reductions, two times.** `save.reduce` (IM spec §2.12) runs *in the
forward pass* over a read's gathered rows, so the un-reduced harvest never
reaches disk; its verbs stay exactly what they are, and nothing here grows
that vocabulary. `reduction` runs *after* the rows are on disk, over a table,
and is where a statistical unit, a grouping and an interval can be declared at
all. Where the verb is the same verb (`mean`, `sum`, `count`, `median`) it is
spelled the same and means the same.

**The output** is one row per group: the group coordinates, `value`, and the
counts — `n` (observations that contributed), `n_rows`, `n_missing` (rows
whose value or weight was `null`), `n_unmatched` (the part of `n_missing` a
`matched: false` row wrote: "the model never said it", distinguishable from a
non-finite value), `n_excluded` (rows the `exclude` policy dropped) — plus
`lower` and `upper` when a procedure is declared. `n` and `n_excluded` are the
honest denominator a minimum-count check consumes; **no threshold is checked
here** (that is the eligibility layer's). Item identity — which rows are
the same observation across runs — is the content-derived row identity's, not
this contract's: `unit` binds a column, and the binding is checked against the
table at run time.

**How the block reaches the script.** The runner hands an authored block to
`main` under `inputs["reduction"]` — the same channel every other input
travels, in process or across the isolation boundary — and records it in the
step's `_step.json` and in `workflow.json`, so a review reads the declaration
from the tree. A step that authors `reduction` may therefore declare no input
of that name (rule 12). The built-in `causalab.workflow.scripts.reduce` runs
the closed set; its optional `value` input names the column to reduce, so a
step running the built-in that declares `value` and a curve's `estimator.y` is
refused at load (rule 12 — the parser knows the step's module; the script keeps
the refusal as its run-time backstop). Anything outside it is a user `script`
step that authors the **same** block and gets the same validation — the escape
hatch declares its dimensions or it is not a reduction; its inputs are its own,
a `value` among them.

**What `select` does, declared.** The shipped `select` (and the `plot`
renderer) reduce through `causalab.io.step_record.aggregate`, whose
arithmetic is the implied reduction

```
{estimator: mean, unit: row, group_by: <the producer's sidecar axes>,
 weight: null, missing: exclude, uncertainty: none}
```

(`implied_reduction` in that module spells it out as data, and a test holds the
two equal). The unit is `row`, not `example`: the two coincide exactly when a
table holds one row per example per group — every non-windowed metric table —
and diverge on a windowed one, where an example with more positions weighs
more. An **unauthored** step keeps that arithmetic bit for bit; `aggregate` is
not changed by this section, and a table with no `example_id` column and no axes
is not reduced at all (the rows are already the unit a consumer chooses
between). **Protocol steps take no `reduction`**: their in-forward reduction is
the inner document's `save.reduce`.

**Column existence is data.** Load checks the block's shape (rule 12); whether
`unit`, `group_by`, `weight` or `resample_unit` name columns the table really
has is checked at run time by the built-in, refused naming the column and the
dimension that named it. A refusal here, as everywhere in this section, ships
with the legitimate case beside it: every shipped and demo workflow authors no
`reduction` and loads, runs and digests exactly as before.

**Estimand identity.** A reduced number is a record, and a record says what
it is (IM spec §2.10, `causalab/protocol/estimand.py` — the one home of the
unit vocabulary and the identifier grammar, imported by both layers). A block
may author a ninth, optional key, **`estimand_version`**: an identifier in the
grammar `<estimand>/v<n>` naming the *arithmetic* the campaign means. It is
checked at load (rule 13) against what the block computes — the identifiers
below are each admissible only for the block that runs them — and it enters the
canonical entry, and so the digest, only when authored. Unauthored, the
identity is the estimator's own, `<estimator>/v1`, recorded on the output rows
and never in the canonical form. `tests/workflow/test_estimand.py` holds this
table to `REDUCTION_ESTIMANDS`:

| identifier | computed by | arithmetic |
|---|---|---|
| `mean/v1` | `mean` | arithmetic mean of the observations |
| `weighted_mean/v1` | `weighted_mean` | Σ w·v / Σ w over the observations |
| `sum/v1` | `sum` | sum of the observations |
| `count/v1` | `count` | the number of observations |
| `median/v1` | `median` | the lower middle observation |
| `quantile/v1` | `quantile` | the `q`-quantile, interpolated |
| `auc/v1` | `auc` | the trapezoid area under the per-`x` means of `y` over the sorted `x`, against the block's declared `x_scale` (`x ÷ normalize` when given; `log x` under `x_scale: log`) — the grid and scale are the block's, carried by its digest, not by the name |
| `cpr/v1` | `cpr` | MIB's CPR arithmetic over the block's declared kept-fraction grid `p` (MIB's ten points unless a `grid` is authored) and `x_scale`: the trapezoid of `(m(N − int(p·N)) − C)/(B − C)`, `B` the mean at cut 0, `C` at cut `N` — MIB's number on MIB's grid, the block's otherwise |
| `mean_of_eligible_row_ratios/v1` | `mean`, `unit: row`, `weight: null`, `missing: exclude` | per row, numerator ÷ denominator (the value column); then the mean over the **eligible** rows — the rows `exclude` kept, `n` against `n_excluded`; every row weighs the same |
| `ratio_of_sums/v1` | `weighted_mean`, `unit: row`, a `weight` column | Σ numerator ÷ Σ denominator over the same rows — the value column is the per-row ratio and the weight column its denominator; rows weigh by denominator |

The last two share one pattern: the same value column, the same
rows, two questions, so they may not share a name — and a block declaring
`ratio_of_sums/v1` while running a plain `mean` is refused naming what it
declared, what it computes and what it may declare. An identifier is a name
for arithmetic the block already spells out; it is not a tenth dimension, and
authoring one does not change a number.

**The identity on the output.** Every reduced row carries, beside the counts,
`unit` — the input table's own, read from its rows (`count` is always in
`count`; a table without a `unit` column reduces to `null`, an *unknown* unit,
not a wrong one), `estimand_version` — the authored identifier or the
estimator's own — and `produced_by`, the one point digest the group's rows
share (`null` for a group spanning points: its provenance is the step's, not
a point's). A table whose rows are in **two units** is refused before anything
is summed, naming both units and the table: a fraction is not reduced together
with percentage points.

**Comparisons.** The shipped comparison, `causalab.analysis.paired_ttest`,
refuses two inputs whose rows are in different units, naming both units and
both inputs, and records on its output the shared `unit` and a `comparison`
label: `arm` when both sides carry the same estimand or neither declares one —
two arms of one campaign — and `version` when both declare and they differ, a
comparison *between arithmetics*, allowed and said. A check that refused an
arm-against-arm comparison because nothing was declared would be worse than
no check, so an undeclared side never refuses. Report claims bind to records by
file, `produced_by`, `estimand_version`, `unit` and value
(`estimand.Claim` / `check_claim`, IM spec §2.10); a generated report view
would be the full form of that binding, and none ships yet.

### 2.7 `behavioral` steps — the declarative behavioral runner

```json
"qualify": {
  "type": "behavioral",
  "document": "protocols/qa_probe.json",
  "set": {"data.base.dataset": "qa/data#development"},
  "decoding": {"mode": "sampled", "seed": 7, "temperature": 1.0, "top_p": 1.0},
  "checker": {"task": "natural_domains_arithmetic", "task_cfg": {"domain_type": "weekdays"},
              "scoring_digest": "961fabc779af7766d806961664dbf346bf54e932690c14d15b664a840b6adfdb"},
  "split": "development",
  "thresholds": {"min_examples": 10000, "min_valid_rate": 0.95, "min_correct_rate": 0.8},
  "retain": {"generations": {"max_rows": 1000}},
  "decision": {"on_pass": "advance", "on_fail": "narrow"}
}
```

A behavioral step runs a **free-form generation task with no
experiment-specific generation or scoring code**: without it every
investigation hand-implements runs from a document, and two investigations
following one pipeline could use materially different local generators. The step names an
authored **no-intervention document under the `generated` frame** — the shape
of `causalab/configs/protocols/probe_variable.json`: a `decode` metric over a
continuation read, a `variable` anchor that searches the continuation for the
answer — and that document is the prompt manifest: its document digest is the
content address, exactly as a dataset resolves to a stamped content digest.
Everything the document does not carry is authored on the step, and every
such field is in the step's canonical entry and digest (§7):

| field | meaning |
|---|---|
| `document` | ✓ — path to the no-intervention document, relative to the workflow file; it declares at least one `generated` position |
| `set` | – overrides, as on a protocol step (§2.2); in the digest |
| `max_points` | – the sweep point cap, as on a protocol step |
| `decoding` | ✓ — how the continuation is decoded (table below); the first place a decode seed exists in a record |
| `checker` | ✓ — `{"task", "scoring_digest", "task_cfg"?}`: the task's `ScoringSpec`, bound by **content digest** (`ScoringSpec.digest`, the `scoring_digest` column a built table carries); `task_cfg` is a factory task's config, the shipped manifest's own field. **Never a sixth spelling of correct**: `string_mode` is read from the spec and recorded, never authored |
| `split` | ✓ — the split purpose (table below); the document's base ref must select the same fragment (`qa/data#development`) |
| `thresholds` | ✓ — `{"min_examples", "min_valid_rate", "min_correct_rate"}`, declared numbers; every rate the decision reads is a count over `n`, and no statistic is re-implemented here — a reduction is §2.6's |
| `retain` | – `{"generations": "all"}` or `{"generations": {"max_rows": n}}`: how many raw generations the step keeps in `continuations.json`. **Bounded by default** (`max_rows` 1000): preserved generations are otherwise unbounded output, and an unbounded retention must be spelled `"all"` |
| `decision` | ✓ — `{"on_pass", "on_fail"}`, each from the decision table below: what a passing or failing qualification does to the question |
| `fan_out` | – the step's declared fan-out (§2.9), as on a protocol step: each child decides over its own points and the join decides once over the summed counts. In the canonical form only when authored |

**Decoding.** The document alone specifies a greedy decode; a behavioral step
says so, or says how to sample:

| mode | fields | meaning |
|---|---|---|
| `deterministic` | none | the argmax at every step — byte for byte the document's own decode, so the same document under this mode produces what it produces today |
| `sampled` | `seed` ✓, `temperature` (default `1.0`, `> 0`), `top_p` (default `1.0`, in `(0, 1]`) | each token is one draw from `softmax(logits / temperature)` restricted to the smallest set whose mass reaches `top_p`, from a generator seeded once per decode window with `seed`. The same seed under the same batch geometry draws the same tokens; sampled decoding is **not** bit-reproducible across geometries on a GPU, so "same seed ⇒ same bytes" is a CPU-fixture claim and a differing-geometry run is recorded, not gated. Only the reference engine samples: a `sampled` step routed to the `nnsight` engine is refused before its model loads, naming the engine and `deterministic` |

Both modes accept optional `eos_token_ids`, a nonempty list of distinct
nonnegative token IDs. The PyTorch engine otherwise uses the model generation
configuration's EOS IDs, then the tokenizer's EOS. It stops each row at its
first matching EOS, pads post-stop slots, and stops model calls once the whole
batch has terminated. Explicit EOS settings enter the behavioral step identity.
See [the behavioral execution recipe](behavioral_analysis.md) for production
validation and exact records.

**Outcomes.** Every generated row lands in exactly one of four **terminal
outcomes**, a closed vocabulary held to this table by
`tests/workflow/test_behavioral.py`; the order is the derivation's
precedence:

| outcome | means | derived from |
|---|---|---|
| `truncated` | the row never emitted EOS inside the decode budget | the row's width equals the budget (`Continuation.widths`); takes precedence over every other outcome |
| `no_final_answer` | the continuation ended, but the answer variable's value appears nowhere in it | the `{"generated": …, "variable": <answer>}` anchor resolves to no steps |
| `invalid_format` | the value appears, but the graded string is not a declared form under the spec's string mode (`exact` / `prefix`) — or the continuation is empty | the spec's own grader matches no declared value; a width of `0` |
| `valid` | the graded string is a declared form | graded `correct` / `incorrect` by the spec, the `per-example grade` vocabulary of the IM spec §2.10 |

**Splits.** A qualification names what its split is *for* — three values, not
a free string — and the document's base ref selects that split by fragment
(IM spec §2.2, `<ref>#<split>`); a `development` step over a `#confirmation`
ref is refused at load:

| purpose | meaning |
|---|---|
| `development` | the split a question is developed on — iterate freely |
| `reserve` | held back while developing; spent once, to check a development result before committing |
| `confirmation` | the confirmatory split — the decision a report rests on |

Because `split` is in the step's canonical entry, a `development`
qualification cannot be replayed against `confirmation` under `--resume`: the
step identity differs, the step runs again and writes a new decision record.

**Decisions.** The step is the *producer* of the typed decision the
conditional layer reads (the `DecisionRecord`):

| decision | meaning |
|---|---|
| `advance` | the question stands — proceed on it |
| `revise` | the question stands but its framing or apparatus does not — change it before proceeding |
| `narrow` | the question is too wide — restrict its scope |

**What the step publishes** (§8): the document's own `save` files, plus
`continuations.json` — the engine's raw generations, one row per generated
row with `point`, `point_digest`, `model`, `input`, `example_id` (the row's
label, IM spec §2.2), `steps`,
`split`, exact unpadded `input_ids`, `width`, `truncated`, the real `token_ids`, `emitted_ids` (including terminal
EOS), `terminal_eos_id`, `padding_ids`, `stop_reason`, effective `eos_token_ids`,
`decoding`, `greedy_token_id` (null for sampled decoding), `text` and per-token char
`offsets`, bounded by `retain` — `outcomes.json` — one row per example with
its `outcome`, `grade` (for a `valid` row), `expected`, `width` and `steps` —
and `decision.json` beside `_step.json`: `decision_type`, `schema_version`
(`1`), `measured_inputs` (`n`, the counts per outcome and `correct`,
`valid_rate`, `correct_rate`), `rule` (the thresholds block verbatim),
`outcome` (`pass` iff `n ≥ min_examples`, `valid_rate ≥ min_valid_rate` and
`correct_rate ≥ min_correct_rate`, else `fail`), `evidence_identity` (the
sha256 of `outcomes.json` joined to the step identity), `split` and `step`.
`continuations.json` is a request-keyed engine output written when the
request declares a `decoding`, not a `save` kind (IM spec §2.12 is
unchanged); nothing about a behavioral step enters the inner document, its
canonical form or its digest.

**Cohorts.** The defaults — 10,000 examples for a single-input document,
1,000 pairs for a paired one — are what a qualification is expected to run
over. A smaller split is **recorded** in the step record's `cohort` block and
**warned**, never refused: the thresholds the decision is held to are the
authored ones.

### 2.8 `decision` and `conditional` steps — typed decisions gate the graph

```json
"gate_k": {"type": "decision",
           "values": {"step": "best_fit", "file": "values.json"},
           "rule": {"best_k": {"le": 16}, "best_seed": {"in": [0, 1, 2]}},
           "decision": {"on_pass": "advance", "on_fail": "narrow"}},

"gate":   {"type": "conditional",
           "predicate": {"decision": {"step": "qualify"}, "field": "outcome", "eq": "pass"},
           "on_true": ["fit", "apply"],
           "on_false": ["narrow_probe"],
           "scope": "global"},

"fit":    {"type": "intervention_protocol", "document": "protocols/fit.json",
           "requires_receipt": {"step": "qualify", "outcome": "pass"}}
```

A preregistered signal gates a downstream step **without the orchestration
layer interpreting prose**: the signal is a typed decision record, the gate is
a closed predicate over one of its fields, and what the verdict does to the
graph is declared. Two producers write the record (the `DecisionRecord`:
`decision_type`, `schema_version`, `measured_inputs`, `rule`, `outcome`,
`evidence_identity`, plus `step`): a `behavioral` step (§2.7), from its
outcome counts, and a `decision` step, from a script step's values object.
Prose gates stay legal — a knee `select` with no decision
downstream is unchanged: a `decision` step *wraps* its result, it does
not replace it.

**The `decision` step** reads one values object through the §3 grammar and
holds each named key to one clause:

| field | meaning |
|---|---|
| `values` | ✓ — a §3 step reference to a `.json` values object (`{"step": S, "file": F}`), with no selector: the `rule` names the keys it reads. Rule 4 holds the file and every key to the producer's `outputs.<slot>.keys` declaration |
| `rule` | ✓ — one clause per declared key, `{<key>: {<comparator>: <literal>}}`: exactly one comparator from the table below with a JSON literal operand (`in` takes a list). No expression language, no arithmetic, no reference to another step. `pass` iff every clause holds |
| `decision` | ✓ — `{"on_pass", "on_fail"}`, each from §2.7's decision table: what a pass and a fail do to the question |

| comparator | holds when |
|---|---|
| `eq` | the measured value equals the literal |
| `ne` | the measured value differs from the literal |
| `lt` | the measured value is a number below the literal |
| `le` | the measured value is a number at most the literal |
| `gt` | the measured value is a number above the literal |
| `ge` | the measured value is a number at least the literal |
| `in` | the measured value is one of the listed literals |

The record it writes is `decision.json` beside `_step.json`, in §2.7's shape:
`measured_inputs = {key: value}` for the rule's keys only — the record's shape
is declared —, `rule` verbatim, `outcome`, `decision_type` through `on_pass` /
`on_fail`, `schema_version` `1`, `step`, and `evidence_identity` = the sha256
of the values file's bytes joined by `:` to the identity of the step that
wrote it (read from the record beside the file); no `split` — a script
producer has none. A decision step declares no `keys`: it is consumed by
name, by a conditional's predicate or by a receipt, never by `key`.

**The `conditional` step** decides between two sides of the graph:

| field | meaning |
|---|---|
| `predicate` | ✓ — `{"decision": {"step": S}, "field": F, "eq" \| "ne" \| "in": L}`: `S` a `behavioral` or `decision` step, `F` from the decision-field table, `L` from that field's closed vocabulary (a list for `in`) |
| `on_true` | ✓ — the steps that run when the predicate holds: a non-empty list of declared step names |
| `on_false` | ✓ — the steps that run when it does not: non-empty, disjoint from `on_true` |
| `scope` | ✓ — what the verdict decides for (table below) |

| decision field | vocabulary |
|---|---|
| `outcome` | `pass` · `fail` — the boolean the producer's rule decided |
| `decision_type` | `advance` · `revise` · `narrow` (§2.7) — the authored consequence |

| scope | executes | meaning |
|---|---|---|
| `global` | ✓ | one verdict for the run: the side not chosen is skipped whole |
| `per_target` | ✓ | one verdict per target: the producer is fanned out over an axis under `sites.` and the conditional expands with it (§2.9) |
| `per_variable` | ✓ | one verdict per variable: the producer is fanned out over an axis under `positions.` and the conditional expands with it (§2.9) |

The conditional's predicate (the noun is the registry's too, for an
architectural capability; here it is always *the conditional's predicate*)
reads `<run_root>/<S>/decision.json`, checks `schema_version` `1`, and
evaluates. A `true` verdict **skips** every `on_false` step and, transitively,
every step that depends on one; `false` the `on_true` side. Every step named
in `on_true` / `on_false` is scheduled after the conditional (a derived edge,
§3, §6); one already upstream of it closes a cycle rule 5 refuses. The two
sides must be **dependency-disjoint**: no step of one side may depend, directly
or transitively, on a step of the other — the verdict that skipped the one
would skip the other with it, and the side authored to run never would — so a
conditional whose sides cross is refused at load (rule 18). The
conditional's own record carries `predicate`, `scope`, `verdict`, `evidence`
(`step`, `decision_type`, `outcome`, `evidence_identity`) and `skipped`; it
publishes no data file, so a §3 reference to a conditional is refused (rule
4).

**Skipped is the runner's third outcome** (§8) beside `completed` and
`reused` — never `failed`, never `blocked`: two words meaning "no output" for
two different reasons would be one known defect in a new place. A skipped step
gets **no step directory** — a file whose existence implied nothing is exactly
what must not be left behind — and its manifest entry names the decision that
skipped it: `{"type", "status": "skipped", "skipped_by": {"conditional",
"decision_step", "decision_type", "outcome", "evidence_identity",
"transitive_from"}}`. The decision is named by `evidence_identity`, so the
entry cannot be produced from a metric table or a values object: only a
`decision.json` carries one. On the stream, a skipped step has
`phase_started` and `phase_completed {status: skipped, skipped_by}` and no
`result_committed` (§4.3). An earlier published unit of a step a rerun skips
is retained as `superseded` (§8).

**The scope requirement.** A conditional declares its `scope` because
what a verdict decides *for* is part of the design, not of the run: `global`
is the whole graph; `per_target` and `per_variable` are one verdict per child
of a declared fan-out (§2.9). A per-child conditional names a fanned-out
`behavioral` producer, gates fanned-out steps whose `over` equals the
producer's, and expands with them at load: `gate@i` reads `P@i`'s
`decision.json` and gates `S@i`, so verdict `i` decides for child `i` and no
other; the conditional's own record then carries one `verdicts` entry per
child. `per_target` is declared over an axis under `sites.`, `per_variable`
over an axis under `positions.`, neither over `shards` — a range of points is
neither a target nor a variable. A scope that does not fit its producer is
refused at load under rule 19, naming `scope`. What stays derived at load is
the graph's shape: which steps exist — children included — and their edges.
What a conditional adds is derived at run time: which of those steps run.
A skipped step's children are skipped with it (§2.9), and a *dependent* of a
skipped step that the run never reaches derives `skipped` (§8), while a step
a verdict skipped directly but the run never reached derives `pending` — the
skip is not on the stream until the step's turn, and §4.3 makes the stream
authoritative. Transitive skips propagate over `after` edges too: they are
in the derived dependencies like any edge.

**`requires_receipt` — a smoke receipt is a prerequisite.** Any step
(`intervention_protocol`, `script`, `behavioral`, `decision`, `conditional`)
may declare `{"step": S, "outcome": "pass" | "fail"}`:

| field | meaning |
|---|---|
| `step` | ✓ — a `behavioral` or `decision` step of this workflow, scheduled before this one (a derived edge) |
| `outcome` | ✓ — the outcome `S`'s `decision.json` must carry: `pass` · `fail` |

The check runs in the runner **before the step is scheduled** — before an
attempt directory exists, before any engine is chosen (`route_engine`), before
any device is touched. A **missing** receipt
(`<run_root>/<S>/decision.json` absent — `S` was skipped, or never published)
and a **failed** receipt (its `outcome` is not the one required) are two
distinct refusals with distinct messages, both rule 18, both naming
`requires_receipt`. Neither is a skip: a skip is a decision, this is an unmet
precondition — the step is `failed` and its dependents `blocked`. `--resume`
reuses a receipt-bearing step only while the producer's *current*
`decision.json` still carries the required outcome (the evidence clause
below); a receipt that flipped since the step ran sends it back through the
check, which refuses — a reused entry never stands on a failed receipt.

**`--resume` and the evidence.** `evidence_identity` is a
run-time value — the sha256 of a table that exists only after a step ran — so
it cannot enter a load-time digest. It binds through the reuse decision
instead (§7, §8): a conditional's record names the `evidence_identity` it
read, and it is reused only if the producer's *current* `decision.json`
carries the same one; a decision step is reused only if the values file's
bytes and its producer's identity still digest to the `evidence_identity` its
own record carries. Otherwise the step is re-evaluated — a reused verdict can
never gate a step whose evidence changed. A receipt refusal changes no file
under the step's directory: the earlier published unit stays as it was
published (`status: completed`, `disposition: accepted`) and the manifest's
`failed` is the authority — the run tree records what each attempt produced,
`workflow.json` records what the run concluded; a rerun that re-qualifies the
receipt reuses or supersedes it as §8 says.

### 2.9 `fan_out` — a declared fan-out and its join

```json
"qualify": {"type": "behavioral", "document": "protocols/qa_probe.json",
            "decoding": {"mode": "deterministic"}, "checker": {"…": "…"}, "split": "development",
            "thresholds": {"…": "…"}, "decision": {"on_pass": "advance", "on_fail": "narrow"},
            "fan_out": {"over": {"axis": "sites.target.layers"}, "join": {"require": "all"}}},

"gate":    {"type": "conditional",
            "predicate": {"decision": {"step": "qualify"}, "field": "outcome", "eq": "pass"},
            "on_true": ["apply"], "on_false": ["probe"], "scope": "per_target"},

"apply":   {"type": "intervention_protocol", "document": "protocols/apply.json",
            "fan_out": {"over": {"axis": "sites.target.layers"}, "join": {"require": "selected"}}},

"fit":     {"type": "intervention_protocol", "document": "protocols/fit.json",
            "fan_out": {"over": {"shards": 4}, "join": {"require": "all"}}}
```

A fan-out whose width is known only at run time — one intervention per
routed expert, say — is refused where it is spelled (IM spec §3.2, `P4`:
"declare it in the workflow (`fan_out`)"), because a shape that grows
as it runs is not a workflow: rule 5's acyclicity and every load-time check
rest on knowing the graph before it runs. So **the fan-out is declared when
the workflow is written and its width is a pure function of the document at
load**: a document step — `intervention_protocol` or `behavioral` —
declares `fan_out`, the runner partitions the step's *compiled point list*
into children at load, and the step's own name is their join.

| over | children |
|---|---|
| `axis` | `{"axis": A}` — one child per value of `A`, an axis the step's compiled document expands (`sites.target.layers`, `positions.tap`, a named axis `axes.<name>`), in compiled coordinate order; a child holds the points whose coordinate on `A` equals its value. Refused naming the axis and the ids the document does expand |
| `shards` | `{"shards": N}` — `N` contiguous ranges of the compiled index list, sizes differing by at most one, the first ranges longer; `2 ≤ N ≤` the point count, a literal integer (never `"auto"`, never a `bool`) |

| require | the join publishes when |
|---|---|
| `all` | every child published every one of its points; a child a verdict skipped skips the join with it (§2.8's transitive rule) |
| `selected` | every child a per-child conditional (§2.8) left unskipped published every one of its points; the skipped children are named on the receipt. Declared only on a step such a conditional gates; a join every child of which is skipped is skipped with them |

`join` is required when `fan_out` is authored: no default is materialized, so
nothing is written a reader did not see. A failed child blocks either join.

**The children are derived, never authored (§6).** Child `i` is `<step>@<i>`
— `@` is outside rule 3's alphabet, so no authored step can collide with one
— a real step with the parent's compiled document, its digest and its
composition, and a point selection (`shard`: `{index, of, over, value |
range, points}`, the parent's compiled indices it runs). The children are
scheduled after the parent's dependencies and the parent after its children;
they are in the derived order, the dependencies, the schedule `explain`
prints and the run manifest, and in the canonical form never: the parent's
entry carries `fan_out` (only when authored, §7), and that is the whole
identity of the expansion. A reference names the join, never a child
(rule 19): `{"step": "fit@0", …}` is refused naming `fit`. A child's record
is a protocol (or behavioral) step's record over its points — `points`,
`point_digests` sliced, `engine` and `execution` its own, plus `shard` — with
`identity` the digest of the parent's entry and its shard, so `--resume`
never reuses `fit@0` from a run of another width (§7).

**The join is the parent's name, and the receipt is the parent's
`_step.json`**: a multi-shard join rejects missing or duplicate points and
emits one normalized receipt. The join runs no engine.
It reads each child's record and tables, refuses a **missing** point and a
**duplicate** point **by point digest** — `point_digests` on the child's
record and, on every table row, `produced_by` (a metric row), `point_digest`
(a continuation row) or a digest-valued `point` (the engine's side tables
`train_eval.json`, `fit_diagnostics.json`, `routing_mismatch.json`); never by
coordinate spelling — as two distinct rule-19 refusals (the join is `failed`,
its dependents `blocked`, never `skipped`; the attempt is retained under
`.attempts/<step>/` like any failed attempt). The parent's point list is
**reconstructed from the children's own `point_digests`** — a deferred
document has no load-time list — so a *foreign* point is a digest a row names
that no child declared as its own, and a child declaring one digest twice is
a duplicate. A row naming no point, or a `point` index outside its child's
points, is refused naming the file and the row — never appended out of order:

- missing — `join 'fit' (require all): point <digest> (coords …) was published by no child — 'fit@1' is failed or absent, or published <k> of its <m> points`;
- duplicate — `join 'fit': point <digest> was published by 'fit@0' and 'fit@1' — one point, one child`.

Then it re-assembles every save file **in the parent's point order** (a table
row is placed by its point; a behavioral `outcomes.json` row's `point` index
and a `continuations.json` row's `point` index are re-based to the parent's —
its `point_digest` places it, and a row whose `point` index and `point_digest`
disagree about which point it describes is refused naming the file and the
row) into its own attempt, and publishes by the one rename like any step. Only
`produced_by` and `point_digest` promise a row per point, so only a file whose
rows they place is held to the per-file missing check; a digest-valued `point`
places a side-table row without enrolling its file, which the engine writes
per occurrence and only when there is something to write. A file some child
did not publish is missing unless it is one of those three side tables
(`train_eval.json`, `fit_diagnostics.json`, `routing_mismatch.json`), which
the join declares sparse and joins from the children that have it — density
is the writer's, never read off the rows found, so a dense table published
empty beside a child that omitted it is missing, not joined empty. A
fanned-out step's document cannot claim one of those three names for its own
save files (compared by the path's final component): the names are the
join's, so a step whose document saves one declares no `fan_out` in this
version, refused at load naming the step and the file (rule 19). A bundle
(`.safetensors`) is not
re-assembled: a step whose document saves one declares no `fan_out` in this
version (refused at load naming the file — its entries are written by one
engine call, and no shared writer merges them byte-identically). Nor is a
`location_ledger` save entry (IM spec §2.12, the one non-value save kind): a
step whose document declares one declares no `fan_out` in this version,
refused at load naming the step, the entry and its kind — a ledger row names
its point in a `point` column holding the digest string with no
`produced_by`, and the ledger is the run's per-point audit table of resolved
positions; the join re-assembles measurement tables, not audit tables. The
join holds every child's rows in memory before it
writes, so its peak is the unsharded table set, and the children's copies
stay on disk beside the joined table (the run tree is the publication): a
shard count buys forward-pass memory, not join memory. The receipt's
keys, beside a step record's own (`type`, `status`, `implementation`,
`files`, `digests`, `checks`, `disposition`):

| key | value |
|---|---|
| `identity` | the digest of the parent's canonical entry, `fan_out` included — not the inner document digest an unfanned protocol step carries |
| `document`, `document_digest`, `method` | the parent's, as any protocol step's |
| `points`, `point_digests`, `axes` | the parent's **full** lists — what `select` groups by, so a downstream reference reads a fanned-out step exactly as an unsharded one |
| `fan_out` | `{"over", "width", "children"}` — the declaration and the children it derived |
| `join` | `{"require", "consumed": {child: {"identity", "points", "digests"}}, "skipped": [{"child", "skipped_by"}] (selected only), "n_points", "n_missing": 0, "n_duplicate": 0}` — the counts are always `0` on a published receipt (a non-zero is a refusal), recorded so a reader sees the check was made |
| `decision` | behavioral parent only: `decision.json`, written over the **summed** counts of the children through the one writer (§2.7), with `evidence_identity` the sha256 of the joined `outcomes.json` joined to the join's identity; beside it the summed `outcomes`, `retain` and `cohort`, and `thresholds`, `checker`, `split` as on any behavioral record |

A `selected` join's receipt carries the parent's **full** `points` and
`point_digests` by design — so `select` and every downstream reference read a
fanned-out step unchanged — while its joined tables hold only the published
children's rows; `join.n_points` is the published count and `join.skipped`
names the rest. The skipped slots hold the load-time digests, not `null` (an
open question, not resolved here). No reader takes a
join's `point_digests` as "the points this step's files describe":
`_ControlLedger.restore` (`runner.py`) does so for controls, and controls ×
fan-out is refused at load (rule 19, `check_fan_out`).

No `engine` and no `execution` block: the join ran no forward, and each
child's record carries its own. A `per_target` / `per_variable` conditional's
parent record is likewise a join of its children's verdicts: `verdicts`
(`{child: bool}`), `evidence` (the joined producer's `decision.json`) and the
union of what the children `skipped`, with `files: []`.

**`--resume` and the join.** A published join is reused only while every
consumed child's current record carries the identity, points and digests the
receipt names and every child it named as skipped is still absent; otherwise
the join is re-made — the children stay reused when they can be.

**What "GPUs" means here.** The width is the declaration. The runner runs the
children sequentially in one process, as it runs every step, and `explain`
reports each fan-out's width; a dispatcher outside the repo maps children to
devices (§4). Nothing here consumes a device count, so no such key exists.

### 2.10 `workflow` — a nested reusable workflow

```json
"tail":   {"type": "workflow", "document": "workflows/locate_and_fit.json",
           "set": {"fit": {"sites.target.layers": [12]}}, "after": ["baseline"]},
"report": {"type": "script", "script": {"module": "causalab.io.plots.workflow_figures"},
           "inputs": {"table": {"step": "tail/fit", "file": "iia.json"}}, "outputs": {"figure": "iia.png"}},
"apply":  {"type": "intervention_protocol", "document": "protocols/apply.json",
           "requires_receipt": {"step": "tail/qualify", "outcome": "pass"}}
```

A `workflow` step's body is another workflow document, **loaded once, at the
outer's load, through the same loader and the same checklist** (§5) — so an
inner refusal is an outer refusal — and **digested by reference**: the outer
entry carries the inner document's own digest, so the outer digest moves when
the inner one does and nothing else moves. It is what makes a sub-graph
reusable — the `locate → best → fit → iia_by_k` tail two shipped workflows
spell out, or a `fit → apply` pair one workflow runs three times at three
layer bands — without copying its steps, and parametrised the way the tree
parametrises anything: by `set`.

| field | meaning |
|---|---|
| `document` | ✓ — path to a **workflow** document (one with a `steps` section, §1), relative to the workflow file; an intervention specification here is refused (rule 20) |
| `set` | – the nested form `{"<inner step>": {"<dotted path>": value}}`: laid over the named inner step's own `set` (outer wins, key by key) **before the inner document is parsed**, so rule 8 checks every path against the intervention specification exactly as it checks an authored one. Names a document step (`intervention_protocol` · `behavioral`) of the inner workflow; an unknown name, or a `script`, `decision`, `conditional` or nested `workflow` step, is refused naming `set.<inner>`. In the canonical form only when authored |
| `requires_receipt` | – as on any step (§2.8): checked before any step of the nested workflow is allocated; a failure is that step's failed attempt, naming this field |
| `after` | – as on any step: every step of the nested workflow runs after the named steps |

**Naming — one namespace, no shadowing.** The inner workflow's steps join
the run as `<step>/<inner>`: `tail/measure`, `tail/gate_k`, `tail/best`. `/`
is outside rule 3's alphabet, so no authored name collides with a flattened
one; it composes with a fan-out's `@` (`tail/fit@0`, §2.9) and with depth
(`a/b/c` — a nested document may itself nest). A path's producer is the
**longest** step name it starts with, so `tail/best/values.json` is
`tail/best`'s `values.json`, never the container's. The outer names an inner
step by its flattened name wherever a step name is spelled — a `{"step": …}`
reference, `after`, a conditional's `on_true` / `on_false`,
`requires_receipt.step`, `predicate.decision.step`, a decision's `values.step`
— and the `workflow` step's own name is legal in `after` and on a conditional's
sides (the whole nested workflow) and refused where a file or a receipt is
meant (rule 20; it publishes no file and writes no receipt). Inward, nothing
crosses: the inner document is loaded standalone first, against its own step
table, so a reference inside it names its own steps — `measure` inside
`tail.json` is `tail/measure`, even when the outer has a `measure` too — and
can never name an outer step. That is the whole of what makes it reusable;
what it receives from outside is `set`.

**The run tree.** The inner workflow's steps execute rooted at
`<run_root>/<step>/` (§1.1): each publishes to `<step>/<inner>/`, attempts
under `<step>/.attempts/<inner>/`, and the inner document's own `artifact:
"<inner>/file"` and `{"step": "<inner>", …}` strings resolve under that root,
so the document's bytes and digests are untouched. The inner document's
`output_dir` is not used. There is one run: one `events.jsonl` whose lines
carry the flattened names, one `workflow.json` whose `steps` carry them, one
`campaign_terminal`; the manifest gains a record-only `nested` map (§8). The
`workflow` step is a container — never attempted, never reused, in no
schedule, with no record, no status word and no directory beyond that root.

**The digest binding** (§7). The canonical entry is `{type, document, set,
workflow_digest, requires_receipt, after}` — `set`, `requires_receipt` and
`after` only when authored — with `workflow_digest` the inner document's own
§7 digest, computed after the outer's `set` is laid over it: two `workflow`
steps nesting one document with different `set` carry two digests (two
campaigns); with the same `set` they share one and still run under two names.
The inner's steps are **never entries of the outer** — editing an inner
`rule`, a script or a `description` (a workflow's description is canonical,
§7) moves the `workflow` step's identity through that one key; the inner's
`output_dir` or its whitespace moves nothing; and a workflow nesting nothing
carries the key on no entry, which `tests/workflow/test_nested.py` holds for
every shipped and demo workflow. `--resume` reuses each inner step by its own
identity, so an inner edit re-runs exactly the inner steps it moved.

**Refusals** (rule 20, each naming the field): a document that includes
itself, directly or through a chain of `workflow` steps, naming the chain
(`a.json -> b.json -> a.json`) — the one cap on depth, since files are
finite; a `document` that is not a workflow document; a `set` naming a step
that is not the inner workflow's or has no `set` of its own; a reference, a
`requires_receipt`, a `predicate` or a `values` naming the `workflow` step
itself. A `conditional` with scope `per_target` or `per_variable` whose
producer or gated step is a nested step is refused too: a per-child
conditional and the fan-out it follows sit in one document, so it is declared
inside the nested document, where §2.9 expands it under the sub-root.
**Controls stay at one root in this version**: a nested workflow
declares no `control` and no `waive`, and no step declares a `workflow` step
or a nested step as its `control.of` — the controls ledger is keyed by step
name at one root, and keying it by flattened name is a later, contained
change; the outer's rule 14 runs over the outer's own steps, so its coverage
requirement is over the outer's own fits — and because a fit inside a nested
workflow can carry no control and no waiver while the outer's cannot reach
it, an outer that engages the layer (any own step declares `control` or
`waive`) is refused under rule 20 when a nested document contains a fit,
naming the inner fit by its flattened name, rather than loading a fit no
declaration holds (fail-closed; lifted once the ledger keys by flattened
name). `fan_out` **on**
a `workflow` step is not a key of the kind (rule 1) and is left for a later
version; `fan_out` **inside** the inner document is §2.9's, unchanged (its
children `tail/fit@i` join under the sub-root). Also left out, by name: an
`exports` list or any opaque form (a nested workflow exposes every step under
its name — the tree wants the inner's tables by name, not a façade), an
`include` that splices text (the inner's own digest is what makes the binding
explainable), a per-nested-step engine, and an honoured inner `output_dir`.

## 3. Cross-step wiring — the reference grammar

**A reference is a locator plus an optional selector.** One grammar, used by
every `inputs` entry.

| locator | resolves to |
|---|---|
| `{"path": P}` | a file on disk: **absolute** if `P` starts with `/`, otherwise **relative to the workflow document's directory** |
| `{"step": S, "file": F}` | the file `F` that step `S` declares, in the run tree |

| selector | requires | yields |
|---|---|---|
| *none* | — | the resolved absolute path — the script opens it itself |
| `"key": K` | the locator names a `.json` | the scalar at `K` inside it |
| `"entry": {…}` | the locator names a `.safetensors` | one tensor of a bundle, by coordinate match (IM spec §2.5) |

Anything carrying none of these keys is a **JSON literal**, passed through
unchanged. References are recognized **only at the top level of an `inputs`
entry**, so a nested object is always a literal: `{"cfg": {"step": 3}}` is
unambiguous and the loader never guesses.

```json
"inputs": {
  "layer_in_run":  {"step": "best", "file": "best_cell.json", "key": "best_layer"},
  "layer_on_disk": {"path": "/data/fits/best_cell.json",      "key": "best_layer"},
  "layer_in_repo": {"path": "configs/pinned_cell.json",       "key": "best_layer"},
  "k": 8
}
```

- **Why `path` carries a tag.** Nothing distinguishes a string that is a path
  from a string that is data: `"meta-llama/Llama-3.1-8B"` and
  `"scripts/prompt.txt"` are both strings. One tag settles it, and earns a
  load-time existence check a bare string could not justify.
- **Why `{"step": …}` cannot be a path.** The run tree's root is a CLI argument
  (§1.1), so `<out-root>/<output_dir>/<step>/…` is unknowable when the document
  is written. A cross-step reference names the step symbolically and lets the
  runner resolve it. That is the one irreducible indirection.
- **A reference names a join, never a child.** A fanned-out step's children
  (`<step>@<i>`, §2.9) are derived; `{"step": "fit@0", …}` is refused at load
  (rule 19) naming `fit`, whose published files are the children's joined in
  point order.
- **`S` may be a nested step.** `{"step": "tail/best", …}` names the step
  `best` of the workflow nested as `tail` (§2.10), and resolves in the run
  tree to `tail/best/<file>`; a `workflow` step itself declares no file, so
  `{"step": "tail", …}` is refused (rule 20) naming `tail/<step>`.
- **`key` and `entry` are one selector per format.** `key` is a name lookup in a
  JSON object and yields a scalar. `entry` is a coordinate match inside a
  safetensors bundle and yields the file plus the selected entry — the tensor
  stays in the file. `entry` cannot collapse into `key`: bundle keys are
  composite (`weight[k=8,seed=0]`) and matching is on **(name, value) pairs,
  never the rendered label**, because the label's field order follows the
  *producing* document's axis order, which the consumer cannot know
  (`causalab/protocol/bundles.py`).
- **No implicit filename.** `file` is always explicit. v1 could omit it because
  a `select` step always wrote `values.json`; with select a script that declares
  its own output names, there is no canonical filename left to default to.
- Inside a protocol step's *inner document*, the IM spec's own reference
  grammar applies unchanged (`{"artifact": …, "key": …}` and `file_path`
  against the artifacts root, with step names shadowing it). A `set` block on a
  protocol step authors those, so it is the one place both grammars meet.
- These references **are** the derived dependency edges, together with `after`.
  The step graph must be acyclic; its topological order is the schedule
  skeleton, and steps with no path between them may run in parallel — the
  runner's choice, never authored.

## 4. Execution semantics

- Steps run in a topological order of the derived graph; each step's outputs
  land under `<step>/`.
- A protocol step executes through the standard engine routing (IM spec §8) —
  capabilities derive from the union over the inner document's points.
- A script step is invoked **in-process by default**:

  ```python
  def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None: ...
  ```

  The script must create every declared output file. The runner then verifies
  each one exists (a missing one fails the step, named by slot), verifies
  declared `columns` against the JSON actually written, and **stamps
  `ArtifactIdentity`** on safetensors outputs — inherited from the step's tensor
  inputs, plus `produced_by` with the step's own digest. Identity stamping stays
  the runner's job so a script cannot forget it.
- A step's writes land in an **attempt directory**, not in `<step>/`. The
  runner verifies every declared output there (§8), writes the per-step record
  (`<step>/_step.json`: the declared files, their content digests and the check
  each passed, and for a protocol step its sweep axes, its point digests and an
  `execution` block — the row bounds the step ran under, `batch_rows` and
  `fit_rows`: the engine's, overridden by the step's own `execution` block
  (§2.2); declared before execution and `null` when unbounded — plus
  `fit_rows_resolved`, the grad-forward bound the engine measured when
  `fit_rows` was `null` — the number to pin, with `fit_rows_shrinks` beside it
  when a window had to be re-packed, and `ragged` only when some write landed
  a ragged window under a declared `ragged` policy — per `<model>/<write>`
  the policy, the per-row widths and the width buckets (IM spec §2.8, §5 rule
  19); the same block a document run's `protocol.json` carries; IM spec §8),
  and only then
  publishes the attempt as `<step>/` by one rename. The run manifest
  (`workflow.json`) is written when the run ends, however it ends — see §7, §8.
- **What the runner knows about execution** (IM spec §8, execution scale): only
  the step dependency graph. It may run independent steps concurrently, but it
  owns no device, host, or job-system knowledge — those belong to the engines
  it is handed and to site tooling outside the repo (job dispatch, which shards
  *document* runs via the CLI's `--points`; a workflow run is never sharded as a
  unit). A declared fan-out (§2.9) shards a *step's* document into children
  the runner schedules like any steps, sequentially in one process; what a
  dispatcher does with the width is its own business.
- A nested workflow's steps (§2.10) are scheduled and narrated like any
  steps, under their `<step>/<inner>` names, and land under
  `<step>/<inner>/`; the `workflow` step itself is never scheduled.

### 4.1 `runtime` — dependency isolation

```json
"runtime": {"isolate": true, "deps": ["umap-learn>=0.5"]}
```

For a step whose dependency set differs from the runner's. An isolated step runs
in a subprocess with `deps` installed — the runner's own environment with the
declared packages layered on top (`uv run --no-project --python <the runner's
interpreter> --with <deps>`): the deps take precedence, the runner's environment
is not modified, and `causalab` is the same install that is running the
workflow, checkout or wheel. `env` lists variable **names** to pass through,
never values — a secret never appears in a document, a canonical form or a
manifest.

`runtime` **is** part of the canonical form and the digest: which interpreter
and dependency set a step ran under changes what the step *is*, so `--resume`
must not skip across a change to it. That is distinct from IM spec §8's
"execution parameters never enter documents", which is about `--device` /
`--points` — the same computation on different hardware.

### 4.2 Script resolution and the torch-free guarantee

`validate` and `digest` **never import a user script**, and **import no
numerics**: importing either would pull torch (or anything else) into a verb
that must stay cheap and runnable on a machine with no accelerator. So load-time
checking of a script is deliberately shallow — the file exists, `ast.parse`
succeeds, a module-level `def main` is present, and its bytes are hashed
(hashing needs no import). Everything else about the script is a run-time
contract. The hash is `causalab.protocol.code.source_sha256`, shared with an
intervention specification's `code` references (IM spec sec. 2.8.1) so a module
that is both a script step and a referenced function's home cannot acquire two
identities.

**What enters the identity** is the script's own bytes and, for a
`{"path": …}` script, *its declared sibling import closure*: every module
beside the script — under the script's own directory, resolved by filesystem
shape and never through `sys.path` — that it reaches transitively through its
`import` / `from … import` statements, function-local imports included (a lazy
import is static text and a real run-time dependence), `if TYPE_CHECKING:`
blocks excluded (they never execute). When the manifest is non-empty it is
materialized in the canonical entry as `closure: {path: sha256}`, sorted by
path, and hashed once as `closure_sha256` = sha256 of the newline-joined
`"<path> <sha256>"` lines; both sit beside `script_sha256`, which keeps its
meaning — the script alone — so a record still carries one hash a reader can
`sha256sum`. When it is empty neither key is written. The walk is
`causalab.protocol.code.import_closure(..., repository=False)`, shared with the
`code` section, and it is as static as the hash: every member is read and
parsed, never imported.

Its boundaries, stated. **The `causalab` package is never in a closure**: its
bytes are runtime identity — `causalab.provenance.runtime_identity().tree_digest`,
which every step record carries and `--resume` compares before reusing
anything (§7) — so a `{"module": …}` step declares no closure at all, and an
edit to `causalab/protocol/schema.py` moves no shipped workflow's identity.
(Putting the package into document identity as well was tried, and only made
every edit to the protocol core re-pin every workflow digest in the
repository while buying `--resume` nothing the tree digest did not already
refuse.) Anything resolving outside the repository — site-packages, the
stdlib, `torch`, `numpy`, `pandas` — is **third-party and excluded**, because a
dependency's version is the same runtime identity (the lockfile) and never
document identity; a parent package's `__init__.py` is executed by Python but
not declared, and is excluded; a dynamic import (`importlib.import_module`, a
module `__getattr__` such as `causalab/io/plots/__init__.py`'s) is invisible to
a static read — the same boundary the `code` section's declaration checker
states. What remains is exactly the code nothing else covers: a workflow's own
helpers beside its `{"path": …}` scripts. The repository-wide walk
(`repository=True`) still exists as the suite's layering tool —
`tests/workflow/test_closure_census.py` freezes what each shipped script
reaches, ten to twelve modules with the whole `protocol/` core among them, so
a torch-free `validate` stays one — and enters no digest.

The two clauses are not the same, and the second is the one with a cost. A
`{"module": …}` locator is resolved with `importlib.util.find_spec`, which does
not execute the named module but **does import its parent packages** — the
stdlib's documented behaviour. So the guarantee obliges, as weakly as it can
while still holding: *every package that may contain a shipped script is
importable without numerics.* `causalab/io/plots/` is lazy (PEP 562) because a
shipped script lives under it. `tests/protocol/test_load_is_torch_free.py`
enforces both clauses — the second by importing each script-holding package in
a subprocess, and by validating the shipped `weekdays_8b.json` end to end. This is a real reduction in load-time
bite compared with v1's op records, and it is the price of a vocabulary wide
enough to hold the work.

### 4.3 Event stream

Beside the run manifest — never inside a step directory — the runner appends
`events.jsonl`: one JSON line per event, written as the run proceeds and never
rewritten. It is the run's *history*, where the manifest is its *state*: a
reader who wants to know what happened and when reads the stream; a reader who
wants the outcome reads `workflow.json`. The stream is a sidecar in the strict
sense — it is an input to no `--resume` decision and
no identity, and the one thing read from it is the manifest's status words
(below) — so appending to it can never move the checksum of a scientific
output. A document run (IM spec §9) appends the same stream beside
`protocol.json`.

Each line carries `schema_version` (`1`), `event`, `seq` (monotonic within the
file and gap-free; a `--resume` run continues the first run's sequence rather
than starting a second stream), `ts` (UTC, ISO 8601), the run's identity —
`document_digest` and the `--points` shard as `points: [start, stop]` for a
document run; nothing for a workflow run, whose identities are its steps' (§7)
— and a `payload`. The identities are the ones that exist: the spec's own
"document digest = campaign, point digest = provenance unit" (IM spec §7); a
workflow step is named in the payload. Nothing in a payload is an
identity-bearing fact the run receipt does not already carry.

```
{"schema_version": 1, "event": "result_committed", "seq": 2, "ts": "2026-09-03T12:00:00.000000+00:00", "payload": {"step": "fit", "files": ["basis.safetensors"]}}
```

The event vocabulary is **closed** — seven names, held to `EVENTS` in
`causalab/io/events.py` by a census test (`tests/io/test_events.py`); the
writer refuses any other name, and so does the reader:

| event | when | payload |
|---|---|---|
| `phase_started` | workflow: a step's turn begins, before the reuse decision · document: before the engine runs | workflow: `step`, `type` · document: `phase`, `n_points` |
| `progress` | document: a point has run — reported once the engine returns, one line per point in run order | `point_digest`, `index`, `completed`, `total` |
| `metric` | document: a metric was summarized for a point | `point_digest`, `name`, `value` — the value `explain` prints, never a new fact |
| `warning` | workflow: an attempt failed and is retained under `.attempts/` · workflow: a control point failed certification (§2.2, §8) · either: the event sink raised | `step`, `reason: attempt_failed`, `error` · `step` (the control), `reason: instrument_failure`, `point` (its digest), `coords`, `certified_by` · `reason: sink_failed`, `event`, `seq`, `error` |
| `result_committed` | workflow: the **publish** moment (§8) — the verified attempt was renamed onto `<step>/` · document: the engine's files are written | workflow: `step`, `files` · document: `files` |
| `phase_completed` | workflow: a step was published, reused or skipped (§2.8) · document: the engine returned | workflow: `step`, `status`, for a protocol step `forwards` — the forward groups its engine ran (§8), so the stream shows a qualification ran once for its target's whole fanout — and for a skipped step `skipped_by` · document: `phase`, `forwards` |
| `campaign_terminal` | workflow: after the manifest is written by a run that ran to its end — every step completed, or a step failed (an interrupted run writes its manifest and no terminal line) · document: the last line of a finished run | workflow: `outcome` (`completed` / `failed`), `steps` · document: `outcome` |

A run that did not finish leaves a stream **without** `campaign_terminal` —
that absence is what "did not finish" reads as, and
`causalab.io.events.terminal(path)` is the one question a reader asks of it.
A stream cut mid-line, or one whose `seq` has a gap (a deleted or a
duplicated line), is reported by the reader, never skipped, and a writer
opening over one refuses rather than burying the damage.

**Local first, remote optional — as code, not as a sentence.** A run may be
handed an *event sink*: a callable receiving each line after its local write
(`run_workflow(..., sink=…)`, `run_protocol(..., sink=…)`; nothing is wired to
the CLI and no remote implementation ships). The seam's contract is what
matters: whatever the sink raises becomes a `warning` line (`reason:
sink_failed`) written locally and not re-delivered, and the outputs, the
manifest and the receipt are byte for byte what a run with no sink writes. A
remote adapter's failure **cannot** change scientific execution
(`tests/workflow/test_events_workflow.py`, `tests/protocol/test_events_run.py`).

**The stream is the authority for status.** A manifest cannot disagree with
execution history, because it is derived from it: each step's `status` in
`workflow.json` is `derive_statuses` (`causalab/workflow/derived.py`) over the
lines the run appended — `phase_completed` gives `completed` or `reused`, a
`warning` with `reason: attempt_failed` gives `failed`, and a step with no
terminal line is classified by the manifest's own rule (§8: `blocked` below a
failure, else `pending`; a step whose only line is `phase_started` is
`pending`). A `warning` with any other reason — `instrument_failure`,
`sink_failed` — is history and moves no step's word: a control point that
failed certification is a per-point fact in the step record, never a step
status (§8). The runner's memory of a step supplies every other field and must
agree on the word; a disagreement is refused before the manifest is written,
naming the step and both words, so a manifest is never written saying what
the stream did not record (`tests/workflow/test_derived_manifest.py`). A
stream the runner cannot read back at write time, or an interrupt that
landed between a memory assignment and its emit, likewise writes no manifest
and no terminal line and leaves the failure in flight as it was — a
`ProtocolError` on a clean run, a `ProtocolWarning` beside the step failure
or the interrupt otherwise; a manifest is never written from memory in the
stream's place. The `_step.json` beside a step's outputs is **not** derived:
it is written into the attempt directory before the publish, when the stream
holds only `phase_started` for that step, and its `status: completed` is the
fact that the attempt verified — `result_committed` narrates its publication.
That is the two lifecycles as files: the stream is the mutable sidecar,
appended to after scientific completion (a tracker line, a debug link) and
moving the checksum of nothing else, while a step's outputs and `_step.json`
are immutable and `workflow.json` is the view derived from the stream —
appending to the stream after a run flips `terminal()` to false without
changing a byte of the manifest. A run whose existing stream cannot be read
when the runner opens it — cut mid-line, a foreign line, a `seq` gap — is
refused before it starts (a `ProtocolError` chaining the reader's error, which
names `path:line`) and nothing is written; to run into the directory again,
move the sidecar aside to start a fresh stream or restore it byte-for-byte.

## 5. Validation — load-error checklist

**Scope: workflow documents.** The IM spec's own checklist for *intervention
specifications* is untouched; rules 8 and 9 below reach into an inner
specification, but they check the workflow's references to it — the document's own
validity stays with the IM loader, run in full.

1. Strict keys everywhere; closed `type` enum rejects with suggestions; derived
   fields may not be authored.
2. `output_dir` is present and is a single filesystem-safe path segment — not
   nested, not absolute, no parent escapes. Section order per §1 is part of
   this rule but **warns and parses on**, as it does for an intervention
   specification.
3. Step names unique and filesystem-safe; no step may be named `workflow.json`,
   and no step directory may collide with the run manifest.
4. Every reference resolves: `step` names a declared step; `file` names a file
   that step really declares (a protocol step's inner `save` manifest, a script
   step's `outputs`); `after` names declared steps. A **document-relative
   `path` must exist at load**; an **absolute `path` is not existence-checked**,
   because validation and execution routinely run on different hosts, so an
   absolute path naming another machine's data would fail a check it should
   pass — it becomes a run-time refusal, and `explain` lists them so the gap is
   visible before dispatch. A **selector must match its locator's format**:
   `key` only on a `.json`, `entry` only on a `.safetensors`, at most one
   selector per reference. A `key` selector on a locator naming an **in-run
   step's** output additionally requires that output to declare `keys`
   containing `K` — that declaration is what makes the reference checkable
   and is what a step-dependent inner document validates against (§2.3).
5. The derived step graph (`inputs` + inner-document artifact refs + `after`) is
   acyclic.
6. `script` names exactly one of `module` or `path`; it resolves — a dotted
   importable module, or a contained relative path — parses, and declares
   `main`. It is never imported (§4.2).
7. `outputs` is non-empty; each output's file is contained inside the step's own
   directory (relative, no parent escapes) and unique within the step; every
   output ends in `.json`, `.safetensors`, `.png`, `.pdf` or `.html` (§2.5);
   `columns` and `keys` are allowed only on a `.json` output, and are
   mutually exclusive.
8. A protocol step's `set` paths must exist in the target document (an override
   that would create structure is a typo), and the document must load as a valid
   intervention specification with `set` applied.
9. A tensor `entry` selection is checkable at load when the producer is a
   `protocol` step — a producing document's entry names follow from its own
   expansion, which is deterministic at load (IM spec §3), so a mis-aimed tensor
   handoff fails before any step runs rather than after the producing step has
   spent its compute. Against a *script* producer it is a run-time check.
10. An isolated step declares its `deps`; `env` lists names only.
11. `is_deterministic` is a boolean if present.
12. An authored `reduction` block is **complete and in vocabulary** (§2.6):
    `estimator`, `unit`, `group_by`, `weight` (a column or an explicit `null`),
    `missing` and `uncertainty` are all present; `unit` and `resample_unit` name
    a vocabulary member and, except for `row`, at least one column;
    `estimator.kind`, `missing` and `uncertainty.kind` are closed; `uncertainty`
    carries exactly the fields its procedure takes (`percentile_bootstrap`:
    `resample_unit`, a positive-integer `repetitions`, a non-negative-integer
    `seed`; `normal_approx`: `resample_unit`, and only with a mean; `none`:
    nothing); `weight` is a column iff the estimator is `weighted_mean`;
    `quantile` carries `q` in (0, 1); a curve estimator (`auc`, `cpr`) carries
    `x` and `x_scale` (closed: `linear` \| `log`), `y` and `normalize` only as a
    column name or (for `normalize`) a positive number — an integer ≥ 2,
    required, for `cpr` — with `y` not `x` and `normalize` naming neither,
    `grid` (at least two increasing fractions in (0, 1]) only on `cpr`, and none
    of `group_by`, `unit.columns` and `uncertainty.resample_unit.columns` naming
    `x` (grouped by its abscissa every curve is one point; a unit keyed on it
    holds no curve; resampling it resamples the grid), and on a curve under a
    `unit` other than `row` a `resample_unit` that is not `row` (a row draw
    splits every unit's curve; a column resample unit that splits a unit is the
    table's refusal, at run time); no other estimator carries any of them; the
    step declares no input named `reduction`; and a step running the built-in
    `causalab.workflow.scripts.reduce` does not declare its `value` input beside
    a curve's `estimator.y` (two spellings of one ordinate). The refusal
    **names the field**. Column *existence* is data and is checked at run time
    against the real table, never here.
13. An authored `reduction.estimand_version` is an identifier in the grammar
    `<estimand>/v<n>` **that the block computes** (§2.6): one of the
    identifiers the block's estimator, unit, weight and missing policy admit.
    `ratio_of_sums/v1` on a `mean` block — one arithmetic declared while another
    runs — is refused naming the declared identifier, the computed one and the
    admissible set. An unauthored identity is derived and never refused.
14. **Controls are declared or waived, and true to their kind** (§2.2). In a
    workflow that authors any `control` or `waive`, every protocol step whose
    compiled document has `train`, and every step some `control.of` names, has
    each of `self_swap` and `matched_random` either declared — the `kind` of a
    step whose `control.of` is that step — or named in its own `waive`; else
    refused naming the step and the kind (`step 'fit' declares a fit; control
    'matched_random' is neither declared by a step nor waived`). A declaration
    names a protocol step other than its own; `kind` is from the closed set; a
    `self_swap` document holds a self-swap model (the predicate of §2.2, refused
    naming the failing field) and is read by exactly one certifying step, which
    writes `controls.json` and authors no input named `control`; a
    `shuffled_source` document is its target's canonical form with a
    counterfactual role's `shuffle` as its one difference (refused naming the
    first differing field, or the missing `shuffle`); a `matched_random`
    control names a fit, pairs its trained featurizer's `kind`, `k`, `group`
    and site, and declares distinct `seeds` — at least `min_draws` of them, and
    the draws its document actually makes. A waiver's reason is from the closed
    set — an empty or unknown reason is refused, `external` carries a
    `reference`, `no_fit` waives only `matched_random` and never on a fit,
    `single_role` waives only `shuffled_source` — and a kind is declared or
    waived, never both. `stop_after_failure_rate` is a number in `[0, 1]` on a
    step that declares `control`. A `full_component` control's writes go
    through no featurizer (or only `identity`) and name no `dims`.
15. **A control runs before its target, once, under one realization** (§2.2).
    A step some `control.of` names depends on that control's certifying step
    (on the control itself, for a kind that needs none) unless the control
    already depends on the target — the schedule derives the edge, the
    canonical form never carries it, and two steps that would each qualify the
    other are refused naming both. The post-hoc direction is `matched_random`'s
    alone (`POST_HOC_CONTROL_KINDS`): a `self_swap` or `shuffled_source`
    control authored to run after its target, by an `after` entry or a
    reference into it, is refused naming the route. Post-hoc means a data hop:
    a `matched_random` control keeps its direction only when its route to the
    target is one of values references, script input references and run-tree
    loads — an `after` entry orders and reads nothing, so ordering alone is
    refused naming the route (`ordering alone does not make a control
    post-hoc; a post-hoc 'matched_random' reads its target's bundle`). A
    control and its target agree on the compiled `model` as `canonical_model`
    materializes it — `key`, `revision`, `dtype`, `quantization`, and the
    attention backend when authored; the list
    is `canonical_model`'s, not this spec's — else refused naming the first
    field that differs (`control 'ctl' runs the model at model.dtype = 'fp32';
    its target 'fit' runs it at 'bf16'`; a backend authored on one side only
    is refused naming the omission); a campaign that changes the realization
    on both steps loads and is re-qualified.
16. **A control and its target are site-equivalent, or declare why not**
    (§2.2). For every control whose kind demands coverage (`full_component`,
    `matched_random`; not `self_swap`, whose per-point agreement is §8's), the
    writes of the control's expanded points and of its target's are compared
    as typed site tuples — `component`, `shape`, `layers`, `head`, `expert`,
    `stream`, `routed_rank`, `featurizer`, `dims`, `sharing` — over every
    coordinate; each field the two differ in (for `sharing`: that they share
    one coordinate system) must be named in the control's
    `non_equivalence.fields`, with a non-empty `reason`, else refused naming
    the field and what separates the two (`control 'ctl' and its target 'fit'
    differ in layers: …`); `layers` compares bands (IM spec §2.4) — a band is
    one site, so a target authored `layers: [3, 4]` against a control swept
    over 3, 4 is refused naming `layers` (coverage-equal,
    intervention-different), lifted like every field by `non_equivalence`
    naming it; a `full_component` control's `featurizer` and
    `dims` may differ without a declaration (that is the kind); a declared
    field the pair does not differ in is refused; `fields` are from the closed
    set — an unknown one is refused naming it — and `reason` is a non-empty
    string.
17. **A behavioral step is complete and bound** (§2.7). It declares
    `decoding`, `checker`, `split`, `thresholds` and `decision`; `decoding.mode`
    is from the closed set, a `sampled` decode declares its `seed`,
    `temperature` is `> 0` and `top_p` in `(0, 1]`, and a `deterministic`
    decode authors neither; `checker` binds by a 64-hex `scoring_digest` that
    **equals the task's `ScoringSpec.digest`** and, when the split's rows
    record one, the table's `scoring_digest` — a missing or mismatched checker
    is refused at load, before any model exists, naming every digest involved;
    `split` is from the closed set and **equals the fragment of the
    document's base dataset ref**; the document declares a `generated`
    position and a literal base ref; `thresholds` carries exactly
    `min_examples` (a positive integer), `min_valid_rate` and
    `min_correct_rate` (numbers in `[0, 1]`); every `decision` value is from
    the closed set; `retain.generations` is `"all"` or `{max_rows: n}` — an
    unbounded retention is never implicit. At run time, a `sampled` step
    routed to an engine that only decodes greedily (`nnsight`) is refused
    under this rule before its model loads, naming the engine and
    `deterministic`. The refusal **names the field**.
18. **A decision, a conditional and a receipt are typed and bound** (§2.8). A
    `decision` step's `values` is a step reference to a `.json` values object
    with no selector; its producer writes that file and declares every key the
    `rule` reads (rule 4); each `rule` clause is exactly one comparator from
    the closed set with a JSON literal operand (`in` a list) — no expression,
    no arithmetic, no reference to another step — and its `decision` values
    are from the closed set. A `conditional`'s `predicate` names a
    `behavioral` or `decision` step, a `field` from the closed set and one of
    `eq` / `ne` / `in` with a literal from that field's vocabulary; `on_true`
    and `on_false` are non-empty, disjoint lists of declared steps, none the
    conditional itself or its producer, and none upstream of the conditional
    (the derived edge makes that a cycle, rule 5); `scope` is from the closed
    set — `per_target` and `per_variable` are held to their producer's
    fan-out by rule 19 (§2.9). `requires_receipt` names a `behavioral` or
    `decision` step of this workflow and an outcome from `pass` · `fail`. At
    run time, before the step is allocated, a **missing** receipt and a
    **failed** receipt are two distinct refusals under this rule — both
    `failed` steps, never `skipped`. The refusal **names the field**.
19. **A fan-out is declared, finite and joined** (§2.9). `fan_out.over` is
    `{"axis": A}` with `A` an axis the step's compiled document expands, or
    `{"shards": N}` with `2 ≤ N ≤` the point count — a literal, never a value
    read at run time; `fan_out.join.require` is from `all · selected`; the
    children `<step>@<i>` are derived, never authored, and never named by a
    reference (a reference names the join); a step that declares `control`,
    is named by a `control.of`, or certifies a control declares no `fan_out`,
    and neither does a step whose document saves a bundle or a
    `location_ledger` entry. Nor does a step whose document saves a file
    named like one of the engine's sparse side tables (§2.9: those names are
    the join's). A `conditional`
    with scope `per_target` or `per_variable` names a fanned-out `behavioral`
    producer and gates fanned-out steps whose `over` equals the producer's —
    `per_target` over an axis under `sites.`, `per_variable` under
    `positions.`, neither over `shards`; `selected` is declared only on a
    step such a conditional gates. At run time the join refuses a **missing**
    point and a **duplicate** point as two distinct refusals under this rule,
    and a table row it cannot place (no point named, a `point` index
    outside its child's points, or a `point` index and a `point_digest` that
    disagree about the row's point) naming the file and the row — the join is
    `failed`, its dependents `blocked`, never `skipped`. The refusal
    **names the field**.
20. **A nested workflow is a document, loaded once, named by prefix**
    (§2.10). A `workflow` step's `document` is a workflow document (it has a
    `steps` section) relative to the workflow file; it loads under this
    checklist in full, with the step's `set` laid over the named inner steps'
    own `set` — an inner name that does not exist or names a step with no
    `set` is refused; a document that includes itself, directly or through a
    chain of `workflow` steps, is refused naming the chain. Its steps join the
    run as `<step>/<inner>` (`/` is outside rule 3's alphabet, so no authored
    name collides), keep their own identities, and are addressed by a
    reference, an `after`, a conditional's sides or a `requires_receipt`
    exactly as any step is; the `workflow` step itself publishes no file and
    writes no receipt, so a reference, a `requires_receipt`, a `predicate` or
    a `values` naming it is refused. A nested workflow declares no `control`
    and no `waive`, and no step declares a `workflow` step as a control, in
    this version; so a workflow that engages rule 14 (an own step declares
    `control` or `waive`) is refused when a nested document contains a fit
    (a protocol step whose document has `train`), naming the inner fit —
    the outer's controls do not hold it, and an unheld fit does not load;
    a `workflow` step declares no `fan_out` (left for a later
    version). A `conditional` with scope `per_target` or `per_variable` names
    no nested step as its producer or on a side — a per-child conditional is
    declared inside the document that declares the fan-out it follows.
    The outer canonical entry carries the inner document's own
    digest as `workflow_digest`, so the `workflow` step's identity moves
    exactly when the inner document does. The refusal **names the field**.
21. **The pins hold** (§1, §7). A document that carries a `pins` section is
    held to it exactly, once everything it names has resolved: every pinned
    resource is one the load touched, with the pinned digest, and every
    resource the load touched is pinned. Three distinct refusals, each naming
    `pins.<category>.<key>`: a resource that **moved** (the bytes the load
    resolved digest differently — a table edited, a script or document
    changed, a `code` module rewritten), a resource the workflow touches but
    **does not pin** (a step added after the stamp), and a pin that **nothing
    touches** any more (a step removed after it). The section itself is
    strict (rule 1): the closed categories `documents`, `scripts`,
    `datasets`, `code`, `files`, each mapping a resource to one sha256 hex
    digest. A document with no section is not refused — it loads, and the
    first `run` stamps it (§7). The fix for a meant change is deliberate and
    named in the refusal: `causalab pin <wf>` re-stamps.

**Two v1 rules are deliberately gone.** The sink rule ("every step is consumed
by a later step or by `save`") and every `save`-manifest rule die with the
`save` section. Consequence, accepted: nothing flags a step whose outputs
nobody reads. That was the rule's value, and losing it is the price of
"everything is published" — in v1 a terminal plot step needed a `save` entry to
be blessed.

## 6. Derived — never authored

| property | derivation |
|---|---|
| step dependencies, schedule, parallelism | the reference graph (§3) |
| a protocol step's sweep axes and point digests | the IM spec's expansion, republished in `_step.json` |
| the columns a script step's table actually has | verified against its declaration on write |
| the columns a `reduction` binds (`unit`, `group_by`, `weight`, `resample_unit`; a curve's `estimator.x`, `estimator.y`, a column `estimator.normalize`) | checked against the real table at run time; refused naming the column and the dimension (§2.6) |
| a reduced row's `unit`, `estimand_version`, `produced_by` | the input table's unit (`count` → `count`, `cpr` → `dimensionless`, `auc` → `null`), the authored identifier or `<estimator>/v1`, the group's one point digest or `null` (§2.6) |
| a script step's digest, and the identity it stamps | its canonical entry — script hash, a `{"path": …}` script's sibling closure when it has one (§4.2), inputs, outputs, `runtime` — plus what its tensor inputs agree on |
| inner-document digests | the IM spec's canonicalization |
| the run manifest | stamped at execution |
| a step's `skipped` status | a conditional's verdict over its producer's `decision.json`, transitively over every step that depends on a skipped step (§2.8) |
| the edges a decision, a conditional or a receipt adds to the schedule | `values.step`, `predicate.decision.step`, `requires_receipt.step`, and conditional → every step it gates (§2.8); schedule only, never canonical |
| a fan-out's children, their selections and their identities | the parent's `fan_out` over its compiled points (§2.9): one child per axis value or per shard, each with its point indices and the digest of the parent's entry plus its shard; the children's edges (after the parent's dependencies, before the join); a per-child conditional's children likewise. Schedule and identity, never canonical |
| a nested workflow's steps, their names and their edges | its document, loaded at the outer's load (§2.10): every inner step under `<step>/<inner>` with its own identity, its inner edges, and the `workflow` step's own edges inherited by the inner steps nothing inside precedes |

**The shape is known at load.** Which steps exist and how they depend on one
another is derived at load, conditionals, fan-outs and nested workflows
included: a declared fan-out's children, their selections and their
identities are derived at load from the parent's `fan_out` over its compiled
points (§2.9), and a nested workflow's steps, their names and their edges
from its document (§2.10); which of them run is derived at run time, from the
verdicts. A shape that grows as it runs is
not a workflow (IM spec `P4`): the width is never read from a forward, a
table or the run tree.

**The axes still exist.** Because `protocol` stays declarative, the runner knows
each protocol step's sweep axes at load. It republishes them in the step's
`_step.json`, and the shipped `select`/`plot` scripts read them there. So
group-by-coordinates-then-mean survives as *behaviour*; it stops being derived
magic in the document model and becomes a documented thing a script does with
data the runner published. A user script gets the same record. What that
behaviour *is* — a mean over rows grouped by the axes, `null`s dropped — is
the implied reduction §2.6 declares; a step that wants another unit, another
estimator or an interval authors a `reduction` rather than inferring one.

## 7. Canonical form, digests, and `--resume`

Every identity in a workflow is a **step's**: the sha256 of its canonical
entry (a script, behavioral, decision, conditional, fanned-out or `workflow`
step), or its inner document's digest (an unfanned protocol step). There is
**no whole-workflow digest**: nothing compares one — `--resume` compares step
identities, a tensor is stamped with the step or point that produced it, and a
nested document's identity in its parent is `workflow_digest` on the
`workflow` step's entry (§2.10), the one place a fold over a whole document
survives, because that is what the parent names. `validate` and `explain`
print none; `digest <wf>` prints the step identities (§9). Nor is there a
method digest (IM spec §7): the identities that exist are the ones something
reads.

A script step's canonical entry is

```
{type, script, script_sha256, closure, closure_sha256, inputs, outputs, runtime, reduction, is_deterministic, after}
```

(`closure` and `closure_sha256` only when a `{"path": …}` script imports a
sibling, §4.2); a behavioral step's (§2.7) is

```
{type, document, set, max_points, document_digest, decoding, checker, split, thresholds, retain, decision, fan_out, after}
```

a protocol step's (§2.2) is

```
{type, document, set, max_points, control, waive, stop_after_failure_rate, fan_out, document_digest, requires_receipt, after}
```

with `fan_out` on either **only when authored** (§2.9),

a decision step's (§2.8) is

```
{type, values, rule, decision, requires_receipt, after}
```

a conditional step's is

```
{type, predicate, on_true, on_false, scope, requires_receipt, after}
```

and a `workflow` step's (§2.10) is

```
{type, document, set, workflow_digest, requires_receipt, after}
```

- **A behavioral step's fields sit on behavioral entries only.** `decoding`
  (the seed included), `checker`, `split`, `thresholds` and `decision` are
  identity — a changed seed or split is a step that runs again — and `retain`
  appears when authored; a `set` and `max_points` as on a protocol step. None
  of these keys exists on a protocol or script entry, so a workflow using
  neither step type carries none of them on any entry — the per-kind key
  census in `tests/workflow/test_behavioral.py` holds that for every shipped
  and demo workflow. The **step identity** `--resume` compares is the digest of
  this entry, as for a script step; the inner document digest alone would
  reuse a `development` run for a `confirmation` step.

- **A fan-out's `fan_out` appears only when authored, and its children are
  never canonical** (§2.9). The parent's entry carries the `over` and the
  `join` as written; the children it derives are in no entry, so the width is
  in the identity exactly once, and a workflow declaring no fan-out carries
  the key on no entry. The step
  identity `--resume` compares is, for a fanned-out protocol step, the digest
  of this entry (an unfanned protocol step keeps its inner document digest);
  for a child, the digest of the parent's entry and its shard.
- **`workflow_digest` is the inner document's own §7 fold, so the `workflow`
  step's identity moves exactly when the inner document does** (§2.10); the
  inner's steps are not entries of the outer, `set` appears only when
  authored, and a workflow nesting nothing carries the key on no entry. The
  inner steps' identities are their own, under their
  flattened names; the `workflow` step has an entry digest and no identity to
  compare, since it is never attempted.
- **A decision's and a conditional's fields sit on their own entries only**,
  and **`requires_receipt` appears only when authored**, on any kind (§2.8).
  `values`, `rule`, `predicate`, `on_true` (sorted), `on_false` (sorted) and
  `scope` exist on no protocol, script or behavioral entry, and a
  receipt-less step's entry carries no `requires_receipt`, so a workflow using
  none of them has the digest it had before the kinds existed — both shipped
  pins hold, which is the check that the layer leaked into no other document.
  The step identity `--resume` compares is the digest of this entry, as for a
  script step; `evidence_identity` is a run-time value and binds through the
  reuse decision instead (§2.8, §8).

- **A protocol step's `control`, `waive` and `stop_after_failure_rate` appear
  only when authored** (§2.2), in their parsed form (a bare waiver reason and
  its object form digest identically); a workflow declaring none has the
  digest it had before the layer existed — every shipped and demo workflow
  does — while one that declares a control carries its `of`, `kind`, `seam`,
  `seeds`, `min_draws` and `non_equivalence` (fields sorted, so two spellings
  digest identically) in its digest, which is where control seeds belong:
  the application, not the method. The rule-16 *verdict* is derived and is
  on the record only, never canonical.
- **The schedule's derived edges are not canonical.** Rule 15's implicit
  control → target edge (§2.2) lives in the derived schedule (§6) beside the
  reference graph; `order` and `deps` were never in the canonical form, so
  declaring a control moves the digest by its declaration alone.
- **`runtime` and `reduction` appear only when authored.** A document that
  declares neither has the digest it had before either field existed; one
  that authors a `reduction` carries all eight dimensions in its step's
  canonical entry, so the unit, the missing policy, the procedure, the
  repetitions and the seed each move the digest (§2.6). This is deliberate:
  a materialized default would have moved every script step's identity in the
  repo to say nothing new about any of them.
- **`reduction.estimand_version` appears inside the block only when authored**
  (§2.6): it moves the digest of the step that names its arithmetic and no
  other; the derived `<estimator>/v1` is recorded on the output rows and is not
  canonical.
- **The script's content hash is in the digest.** Without it `--resume` is
  incorrect: a step whose script changed would be skipped as up to date.
  Hashing needs no import, so it costs the torch-free guarantee nothing.
- **So is a `{"path": …}` script's sibling closure** (§4.2): `closure`, the
  manifest of every module beside the script that it imports, and
  `closure_sha256`, one hash over it — written only when the manifest is
  non-empty, so a `{"module": …}` step and a sibling-free script carry neither
  key. This is the one code identity `--resume` would otherwise miss: a
  sibling's arithmetic could change every downstream number while the script's
  own hash and the package's `tree_digest` both stayed still. The package's
  own modules are deliberately *not* here — they are the `tree_digest` below,
  compared on every reuse — so an edit to the protocol core re-runs every
  step of a resumed run and moves no document's identity. The manifest is what
  makes a move *explainable* — a changed identity names the sibling that moved
  it.
- The canonical form materializes every default (`is_deterministic: true`, an
  output's long form), sorts `after` lists, and **stamps each protocol step with
  its document's digest**, computed with `set` applied: for a document with no
  in-run references this is the IM spec §7 campaign digest; a document that
  references step outputs (its values exist only at run time) stamps the digest
  of its overridden authored form, and the fully resolved per-point digests land
  in the run manifest. Either way a protocol step's identity changes exactly
  when the campaign it runs changes, without inlining documents.
- `output_dir` is **excluded**: it names where, not what.
- A step's identity is `sha256(canonical bytes of its entry)`, same byte rules
  as the IM spec; a nested document's `workflow_digest` is the same fold over
  the inner canonical form.

**`is_deterministic`** defaults `true` and is part of the digest. It buys two
things: `explain` reports "this workflow is not replayable" and names the steps
responsible, and `--resume` refuses to reuse a non-deterministic step's outputs
unless told to. A step that sets it false is asking for review.

**`--resume`** reuses a published step when its recorded identity (the step
digest above) matches, **and** it was produced under the same implementation —
the record's `tree_digest` equals the running package's; a record without one
is never reused — **and** every file the record lists is present with the
**sha256 content digest** the record stamped for it. Reuse verifies content,
never existence: a truncated output, a file overwritten in place, a missing
file, or a record that carries no digests (one written before they were
recorded) all mean the step runs again. A reused step is reported `reused`
with the record it was reused from.

The implementation is the **bytes** of the `causalab` package that ran the
step — `causalab.provenance.runtime_identity().tree_digest`, a deterministic
digest over every file that executes — not its revision and not its dirty
state. The two halves of "same code" are this digest and the content hashes
already inside the step identity: a script step's `script_sha256` (and a
`{"path": …}` script's `closure_sha256` over its siblings), and the
`source_sha256` of every `code` reference an intervention document carries
(and its sibling closure, when it has one; IM spec §2.8.1). The package's own
modules are in the first half only. A dirty tree whose bytes differ has a different tree digest
already; two trees with equal digests are the same code whatever `dirty`
says, so `dirty` alone never refuses. A refused reuse is a silent
re-execution, reported `completed` like any digest mismatch — there is no
status for it. And no record carries a path: moving the run tree alone, to
another parent or under another `output_dir`, changes nothing `--resume`
compares.

**`workflow.json`** is the run-time record: per step the resolved input values,
the script path and hash (and a `{"path": …}` script's sibling closure), the
`runtime` block, the `reduction` block when authored, the step digest, the
`implementation` block, the content digests of its files, and its status (§8).
It carries no whole-workflow digest — there is none (above); a nested
workflow's `workflow_digest` appears in the record-only `nested` map (§8). It is written in a `finally`, so a run that
failed or was interrupted still leaves one classifying every step — unless
the derived status and the runner's memory disagree, or the stream cannot be
read (§4.3); then no manifest is written rather than a wrong one.

**What the identity covers, and what it does not.** A script step's code
identity is its own module's bytes and, for a `{"path": …}` script, its
sibling closure (§4.2). The `causalab` package is covered by the other half:
`causalab.io.step_record.aggregate`, imported by the shipped `select` and
`workflow_figures`, and `causalab.workflow.reduction`, imported by `reduce`,
are package modules, so a change to either changes the `tree_digest` and
re-runs every step of a resumed run — and moves no document's identity. The
consequence to hold on to: **a shipped script's own bytes are the only
package bytes in any document identity.** A docstring reword in
`causalab/analysis/fit_pca.py` moves the `fit_pca` step's identity and every
demo that quotes it; a docstring reword in `causalab/protocol/schema.py` moves
nothing. `tests/protocol/test_vocabulary_census.py`'s frozen-prose exemption
therefore stays at exactly the five hashed script modules, and
`tests/workflow/test_closure_census.py` freezes what each of them reaches in
the repository as a layering census. Not covered, by design: third-party code
(runtime identity, §4.2), parent-package `__init__` execution, and dynamic
imports.

**`pins` — the closure, stamped into the document.** The identities above
are *derived*: computed at load, compared against the run tree. The `pins`
section (§1) is the same closure *authored* — the workflow's own statement of
what it was validated against, held to on every later load (§5 rule 21). It is
a census, by category, of everything the load touched, each keyed by the name
the document gives it:

```json
{"pins": {
  "documents": {"intervention_protocol.json": "<sha256 of the file's bytes>",
                "inner_workflow.json":        "<the inner document's own §7 digest>"},
  "scripts":   {"scripts/figure.py":          "<script_sha256>",
                "scripts/figure.py#helper.py": "<a sibling in its closure>",
                "causalab.analysis.fit_pca":   "<a module-addressed script>"},
  "datasets":  {"weekdays/data":              "<content digest of the whole table>"},
  "code":      {"causalab.tasks.x.causal_models": "<source_sha256 of a code reference>"},
  "files":     {"params/basis.safetensors":   "<content_digest of a resolved artifact>"}
}}
```

- **What is pinned.** Every protocol and behavioral step's document (its
  bytes); every table those documents read, as the *whole table* with the
  `#split` fragment stripped — the pin is the table, whichever split a
  document consumes; every `code` reference's module and sibling closure
  (§2.8.1 of the IM spec) and every artifact the canonical form resolved to a
  `content_digest` that is not a run-tree product; every script step's module
  and its sibling closure (§4.2), and every relative `path` input as a file.
  A `workflow` step's document is pinned by its own digest rather than its
  bytes, because the inner file carries its own `pins` and stamping it must
  not move the outer's pin. Not pinned: absolute `path` inputs (rule 4 defers
  them to run time), run-tree products (they are outputs, verified by the
  step record), and the `causalab` package itself — that is runtime identity,
  the `tree_digest` above, and pinning it in the document would make every
  edit to the package a stale pin in every workflow.
- **Stamping.** `causalab pin <wf>` writes the census into the file as its
  last section, replacing any section there. `causalab run <wf>` does the same
  on the **first run of an unpinned document**, so a workflow is pinned by the
  act of running it once — never stamped by hand before it may run. Neither
  stamps under `--set`: the overrides describe a closure the file does not,
  and `run` says so and proceeds unpinned. Only non-empty categories are
  written; keys are sorted; a JSON file keeps its indent and key order, so a
  second stamp of an unchanged closure changes no byte.
- **Checking.** Rule 21, at load — so `validate`, `explain`, `digest` and
  `run` all refuse a stale pin before compiling further, naming
  `pins.<category>.<key>` and which of the three facts it is (moved, unpinned,
  no longer touched). A pinned workflow whose table was rebuilt with a changed
  generator, whose script was edited, or whose inner document was retargeted
  does not run until its author re-stamps it — which is the point: the change
  is acknowledged in the document that depends on it. `pin` is the one verb a
  stale section cannot refuse: it exists to replace the section, so it loads
  without holding the document to it and writes the fresh census.
- **Pins are not identity.** The section is excluded from the canonical form,
  like `output_dir`: a stamped and an unstamped copy of one workflow have the
  same step identities and the same canonical form, so stamping moves no
  shipped digest, no demo digest, and no `--resume` decision. The facts a pin
  records are already inside the identities — `script_sha256`, `closure`, a
  protocol step's `document_digest` and through it every dataset digest and
  `code` hash — and `--resume` compares *those*, against the run tree. Pins
  compare the same facts against the *document*. One closure, two readers.
- **Where pins do not live.** Not in an intervention specification (it
  carries no pins section; `causalab pin` refuses one), not beside a dataset
  (nothing sits beside a table — no manifest, no recipe, IM spec §2.2), not
  in a file beside the workflow. A resumable, pinned intervention is one
  wrapped in a workflow step.

## 8. Runner contract

The runner is causalab-owned. A workflow document names no engine: `--engine`
is a flag on the *run*, and it supplies one engine list that every protocol step
routes against — `choose_engine` picks per step, from the same list. So under
the default `--engine auto` two steps may well execute on different engines
(whichever each step's document requires), but an author cannot *pin* one engine
for `fit` and another for `apply`; see §11.

| service | contract |
|---|---|
| schedule | topological order; independent steps may parallelize |
| stores | the run-tree/external artifact overlay (§3), one output root |
| inputs | resolve the §3 grammar to paths and scalars |
| scripts | invoke `main(inputs, outputs)`, in-process or isolated |
| attempt | every write of a step goes to `.attempts/<step>/<id>/` (§1.1), never into `<step>/`; the id is a per-step counter |
| outputs | before publish, verify every declared output: it exists, is non-empty, parses under its format (JSON, a safetensors header whose promised length is the file's, a `.png`/`.pdf` signature; `.html` is checked for non-emptiness only and recorded as such), and matches its declared `keys`/`columns`; stamp identity on safetensors; record each file's sha256 |
| publish | one atomic rename of the verified attempt onto `<step>/`; a stale `<step>/` from an earlier run is moved aside inside `.attempts/` first and, once the new unit is published and narrated, **retained** as `.attempts/<step>/<n>.superseded/` — its `_step.json` rewritten with `status: superseded`, `disposition: superseded` and `superseded_by` (the replacing attempt, the published record's identity, the step directory) — never deleted; until then it is recoverable, and the next run restores it if the new unit never landed or retains it if it did |
| stamping | `_step.json` per step (files, digests, checks, for a protocol step `forwards` — the forward groups the engine actually ran, recorded and never compared, and an `implementation` block — `tree_digest`, `version`, `resolved_revision`, `dirty`, from `runtime_identity()`, asked once per run; only `tree_digest` is compared on `--resume`, the other three are recorded for a reader); `workflow.json` for the run, written in a `finally` — always, `KeyboardInterrupt` included, unless the derived status and the runner's memory disagree or the stream cannot be read (§4.3), when no manifest is written rather than a wrong one; if the manifest itself cannot be written, the step failure is what propagates; inner runs stamp per the IM spec |
| failure | a failed attempt keeps `attempt.json` (start/end, exception type and message, an isolated script's stderr tail bounded to 64 KB, which declared outputs it had written); at most the last 3 failed attempts of a step are kept, partial outputs included — older ones are deleted |
| resume | a step is reused (`--resume`) only when its recorded identity matches, its record's `implementation.tree_digest` is the running package's (§7; a record without one is never reused, and `dirty` is never a reason), a protocol step's recorded `engine` is the one the step would run under now (the routed engine's name — the third member of the qualification identity, compared, not merely carried; a record without one is never reused; resuming under another `--engine` re-runs every protocol step, and under `auto` "another" can mean the host's install changed; a host that cannot route the step — no configured engine covers it — has no engine to compare against and reuses the record; a fanned-out parent's record is the join, names no engine and is bound to its children's records instead — §2.9), and every recorded file is present with its recorded content digest; never on existence alone; never a non-deterministic step unless asked; a conditional or decision step also only while its evidence holds, and **any step declaring `requires_receipt` only while the producer's current `decision.json` still carries the required outcome — a flipped receipt is never reused** (§2.8), a join only while every consumed child's current record is the one it joined (§2.9) |

**A behavioral step's record** (§2.7) is a protocol step's — `document`,
`engine`, `document_digest`, `points`, `point_digests`, `axes`, `files`,
`method`, `execution` — with `identity` the step's own digest and the
`execution` block carrying `decoding` (`mode`, and for a sampled decode
`seed`, `temperature`, `top_p`) beside `batch_rows` and `model_source`: the
same block, the same one recorder, one more execution parameter. Beside them
`checker` (`task`, `task_cfg` when authored, `scoring_digest`, and the spec's
`string_mode`), `split`, `outcomes` (`n` and the count per outcome, plus
`correct`), `thresholds`, `retain` (what was authored, `retained` of `n`),
`cohort` (the default for the document's shape, `n`, `below_default`) and
`decision`, the path of `decision.json`.

**A join's record** (§2.9) is a protocol (or behavioral) step's with the
parent's full `points`, `point_digests` and `axes`, `identity` the digest of
the parent's entry, a `fan_out` block (`over`, `width`, `children`) and a
`join` block (`require`, `consumed` per child — its `identity`, `points`,
`digests` — `skipped` under `selected`, `n_points`, `n_missing: 0`,
`n_duplicate: 0`), and **no** `engine` or `execution` block; a behavioral
join adds `decision` over the summed counts. **A child's record** is its
kind's record over its points plus `shard`. A skipped parent's children are
skipped with it and appear in `workflow.json` as `skipped` entries; a
`selected` join whose child is skipped is not.

**A nested workflow's steps' records** (§2.10) are their own kinds' records,
at `<step>/<inner>/_step.json`, each with its own identity; the `workflow`
step has no record and no directory of its own beyond `<step>/`, which holds
its steps' directories and their `.attempts/`. `workflow.json` gains a
top-level `nested` map — `{"<step>": {"document", "workflow_digest", "steps":
[…]}}`, one entry per `workflow` step at any depth — record-only, like
`nondeterministic`: never canonical, never compared on `--resume`. The status
table below is unchanged: a nested step carries one of the six words under its
flattened name; the `workflow` step carries none. A nested conditional's
record names its steps as its own document does (`skipped`, `evidence.step`
are local names); the run's `skipped_by` blocks and the stream carry
flattened names.

**A decision step's record** (§2.8) is `type`, `status`, `identity` (its own
digest), `implementation`, `values` (the reference it read), `rule`,
`measured`, `outcome`, `decision_type`, `evidence_identity`, `files`
(`decision.json`) and `decision`, its path. **A conditional's** is `type`,
`status`, `identity`, `implementation`, `predicate`, `scope`, `verdict`,
`evidence` (`step`, `decision_type`, `outcome`, `evidence_identity`) and
`skipped` — the steps its verdict took out of the run — with `files: []`.
**Every published record carries `disposition`**, a closed vocabulary held to
this table by `tests/workflow/test_conditional.py`:

| disposition | meaning |
|---|---|
| `candidate` | the record as written into the attempt, before its publish |
| `accepted` | the published unit at `<step>/` — the one a reader beside its files (`read_sidecar`) finds, and the only one analysis accepts by default |
| `inadmissible` | a failed attempt's `attempt.json` |
| `superseded` | a unit a rerun displaced, or a skip took out of the run, retained under `.attempts/<step>/<n>.superseded/` with `superseded_by` |

A reader never infers `accepted` from absence. **Supersession preserves**: a
rerun does not overwrite — the prior unit is retained and marked,
`workflow.json` lists it under the step's `superseded`, and the hashed
`select` reads only the published unit because the record beside its input is
the accepted one, with no change to any hashed script. Retention is unbounded
by design: displaced units are preserved, and `.superseded`
units are never pruned (`prune_attempts` touches numeric attempt directories
only).

**Controls in the record** (§2.2). The layer lives inside the attempt, so no
step writes into another step's directory: a control step's `_step.json`
carries its `control` declaration with `by_point` (`{point digest: {coords,
status}}` over every point it expanded — `passed` for a `matched_random`,
`shuffled_source` or `full_component` point that ran, `not_run` for a
`self_swap` point until its certifier says) and `n_points` (`n_failed: 0` for
the kinds that pass by running); for a coverage kind, `equivalence` —
`status: equivalent | declared`, the `fields` the pair differs in and
`sharing: shared | distinct`, the rule-16 verdict against its target (§2.2); a
reused control is re-seated from this block on `--resume`, coordinates
included; a step's `waive` is recorded when authored; a certifying step's
record carries
`certifies` — `control`, `of`, `kind`, `by_point` (`{point digest: {coords,
status}}` over the control's points, joined from its `controls.json` rows by
coordinates), `n_failed`, `n_points` and the `stop_after_failure_rate` it was
held to. The rows' `coords` are a saved bundle's header spelling — short axis
names against the saved read's entity, non-scalar values as JSON text — and
the runner spells the control's points the same way (`coords_token`), so a
control swept on any authorable axis certifies. A row matching no point, or
two rows for one point, is refused; a point with **no row** is a
`ControlFailure` — a control that did not run on a point cannot certify it,
and it is never filled in as `not_run` to dilute the rate below. A certifier
whose `n_failed / n_points` exceeds the bound is a
**failed attempt** — `ControlFailure`, retained under `.attempts/` with its
`controls.json` — so every step downstream of it is `blocked` by the rule
below; under the default `0.0` the first failure stops. Each failed point is
narrated on the stream first (`warning`, `reason: instrument_failure`, §4.3).
Every protocol step downstream of a control (or of its certifier) carries
`controls` — `inherited_from` (the control steps), `identity` (`{control:
{document_digest, tree_digest, engine}}` — the qualification each status came
from: the control's resolved document digest, the `implementation.tree_digest`
of the code that ran it and the engine, one triple for the whole fanout and
nothing from `execution` in it; §2.2), `by_point` (`{point digest:
{coords, controls: {control: passed | failed | not_run}, status}}`),
`n_invalid` and `n_points` — where a point's `status` is `instrument_invalid`
when any control `failed` at the agreeing coordinates, `not_run` when a control
has no certified point there, else `passed`. Two points **agree** when, on
every axis either sweeps, the other has the same value as its coordinate or
as the value its document authors there — so a control pinned to one layer by
`set` agrees with the dependent's point at that layer. Values are compared as
canonical values: a control pinned by `set` or authored as a one-layer band
agrees with a dependent's point at that layer whichever spelling either side
used (IM spec §2.4: `L` ≡ `[L]`). A dependent fit
therefore inherits its target control's status per point rather than computing
its own, and a passing subset cannot silently stand in for the population:
`n_invalid` of `n_points` is on the record. What this layer does **not** do is
stop a campaign's expansion — the runner runs a fixed schedule; a failure
rate over the bound blocks the steps below it, and "does not expand further"
is dependent-sweep work outside this layer.

Every step in `workflow.json` carries exactly one of six statuses — a closed
vocabulary, held to this table by a census test. The word is derived from the
run's event stream at write time (§4.3), never copied from the runner's
memory, and a manifest that would disagree with the stream is refused rather
than written:

| status | meaning |
|---|---|
| `completed` | this run attempted the step, verified its outputs and published them |
| `reused` | `--resume` found a published unit whose identity, `implementation` and content digests match; the record is the earlier run's |
| `failed` | this run's attempt raised — a script error, a verification refusal, or an interruption mid-attempt (`KeyboardInterrupt`); nothing was published, the earlier unit if any is untouched |
| `blocked` | not attempted because a step it depends on is `failed` or `blocked` (`blocked_by` names them) |
| `pending` | not reached — the run stopped before its turn and nothing upstream failed |
| `skipped` | a conditional's verdict took the step out of this run (§2.8), directly or through a step it depends on; `skipped_by` names the decision by `evidence_identity`; no attempt, no directory, nothing published |

## 9. CLI

The same four verbs (IM spec §9) accept workflow documents — dispatch on the
`steps` section — and one verb, `pin`, is a workflow's alone:

| verb | effect |
|---|---|
| `run <wf> --out <root>` | validate, schedule, execute steps, stamp the manifest; an unpinned document is stamped with its `pins` section first (§7) — not under `--set`, which says so and runs unpinned |
| `validate <wf>` | the §5 checklist, including every inner document and, for a pinned document, rule 21 |
| `pin <wf>` | stamp the `pins` section — the census of every document, script, table, code module and file the workflow touches, by digest — into the workflow file, replacing any section there (§7). Refused under `--set`, and refused on an intervention specification: pins are a workflow's |
| `explain <wf>` | the derived schedule (levels of parallel steps), per-step inner digests/point counts, the width and join of each fan-out and each child's shard (§2.9), a nested workflow's steps indented under their `workflow` step (§2.10), non-deterministic steps, unchecked absolute paths |
| `digest <wf>` | the identities `--resume` compares, one `<step>  <digest>` line per step in schedule order (§7); there is no whole-workflow digest |

## 10. Worked example — the weekdays-8b pipeline

Locate a layer × position cell, fit DAS rotations at it, apply the best fit on
the test split, and plot — as one workflow over the golden-corpus documents
07/08/09.

```json
{
  "version": "1",
  "description": "weekdays-8b: locate -> DAS k x seed fits at the best cell -> apply on test; scan heatmap + IIA-vs-k curves.",
  "output_dir": "weekdays_8b",
  "steps": {
    "locate": {"type": "intervention_protocol", "document": "../protocols/weekdays_locate_scan.json"},
    "best": {
      "type": "script", "script": {"module": "causalab.workflow.scripts.select"},
      "inputs": {
        "table": {"step": "locate", "file": "iia.json"},
        "choose": "max",
        "emit": {"best_layer": "sites.target.layers", "best_pos": "positions.tap"}
      },
      "outputs": {"values": "values.json"}
    },
    "fit": {"type": "intervention_protocol", "document": "../protocols/weekdays_das_sweep.json",
             "set": {"positions.best": {"artifact": "best", "key": "best_pos"},
                     "sites.target.layers": {"artifact": "best", "key": "best_layer"}}},
    "best_fit": {
      "type": "script", "script": {"module": "causalab.workflow.scripts.select"},
      "inputs": {
        "table": {"step": "fit", "file": "iia.json"},
        "choose": "max",
        "emit": {"best_k": "featurizers.rot.k", "best_seed": "train.seed"}
      },
      "outputs": {"values": "values.json"}
    },
    "apply": {"type": "intervention_protocol", "document": "../protocols/weekdays_das_apply.json",
               "set": {"featurizers.rot.file_path": "fit/rot.safetensors",
                       "featurizers.rot.k": {"artifact": "best_fit", "key": "best_k"},
                       "featurizers.rot.entry": {
                         "k": {"artifact": "best_fit", "key": "best_k"},
                         "seed": {"artifact": "best_fit", "key": "best_seed"}}}},
    "scan_heatmap": {
      "type": "script", "script": {"module": "causalab.io.plots.workflow_figures"},
      "inputs": {
        "table": {"step": "locate", "file": "iia.json"},
        "plot": "heatmap", "x": "sites.target.layers", "y": "positions.tap"
      },
      "outputs": {"figure": "scan_iia.png"}
    }
  }
}
```

Derived schedule: `locate` → `best` → `fit` → `best_fit` → `apply`, with
`scan_heatmap` free to run as soon as `locate` finishes — parallelism nobody
authored. The `set` overrides on `fit` re-point the corpus document's artifact
refs at the in-run `best` step. `fit` sweeps k × seed, so its bundle holds nine
rotations: `best_fit` names the winning cell and `apply` selects that entry —
one fit applied, provably the one the numbers chose, with its ArtifactIdentity
checked exactly as in a standalone run.

Note the one wart the example makes visible: a `set` block authors *IM-spec*
references (`{"artifact": …}`), while the step's own `inputs` use the §3
grammar (`{"step": …}`). Both name the same referent. Aligning the two layers is
tracked as future work.

## 11. Open

- **`select` aggregation**: the shipped script hard-codes §2.6's implied
  reduction (a mean over rows grouped by the sidecar's axes). Variants are not
  `select` parameters: a step that wants another estimator, unit or an interval
  authors a `reduction` on `causalab.workflow.scripts.reduce` and points
  `select` at the reduced table. Whether `select`'s implied unit should become
  `example` on windowed tables is a numbers-moving question, open.
- **Plot vocabulary**: the shipped `plot` script covers heatmap and lines.
  Anything else is now a user script rather than a spec change, which is the
  point of the step type.
- **Cross-workflow threading** stays by external path or artifact reference (the
  same answer the IM spec §8 gives for cross-document threading).
- **Two reference grammars** (§3): a `set` block is where the workflow grammar
  and the IM spec's meet. Options are to align the IM spec on the simpler form,
  or to let `set` accept the workflow grammar and translate.
- **Per-step engine pinning**: the second engine exists (`nnsight`, beside the
  reference `pytorch_hooks`), so this is live work rather than a hypothetical.
  `--engine` is run-level: it hands the runner one engine list and every
  protocol step routes against it, which is enough for a step that *requires* a
  particular engine and not enough to author "fit here, apply there" — a
  parity run across engines, or a fit whose engine differs from the apply's.
  The field would be one optional `engine` per protocol step, overriding the
  run's list for that step; the open question is what it does to the step
  digest, since an engine is provenance (`ArtifactIdentity` stamps it) but not
  today part of what a document hashes.
