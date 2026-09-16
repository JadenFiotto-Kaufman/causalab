# Documentation

How causalab is documented, for two readers at once: a researcher who runs
experiments without reading the code, and an agent or engineer who extends it.
The code is the source of truth. Everything else is rendered from it or checked
against it.

**Status:** proposal, 2026-09-15. Nothing below is enforced yet.

## 1. Documentation types and where they live

Four kinds of page. Each answers one question for one reader, and each has one
home in the repo.

| Kind | Question | Reader | Home in causalab |
|---|---|---|---|
| Tutorial | Teach me the primitives, in an order you choose. | newcomer | `demos/onboarding_tutorial/`: a numbered sequence, read once, front to back. |
| How-to | I want to run method X. Show me every option X has and what each one does. | practitioner with a goal | `docs/methods/<method>.md` (§2, §7): one page per method family, hand-written prose around pointer blocks that pull every table and field description from the protocol object model in `causalab/protocol/schema.py`. The shipped templates in `causalab/configs/protocols/` are linked from it, held to the templates by a test. |
| Explanation | Why is it designed this way? What does it guarantee? | anyone extending it | Package docstrings in `__init__.py` (the layer map), module docstrings (the file's purpose and invariants), the prose of the two specs, and the `Experimental design` section of each demo. |
| Reference | What does this symbol, field, or option do? | both | Two layers. Python: function, class, and attribute docstrings, rendered into an API reference under `docs` (planned). Document format: the typed object model in `causalab/protocol/schema.py`, whose attribute docs are the field tables of the spec. |

**What the API reference is.** The Python reference, generated from
docstrings by a generator to come (§5 phase 5), one markdown file per package
under `docs`, committed to the repo.
It is committed rather than built on demand so that it exists on a fresh
checkout, reads on GitHub, and is greppable by an agent without a build step.
It is never edited by hand: a stale or edited copy fails CI. It documents the
Python surface an engineer extends, not the JSON a practitioner authors. A
practitioner who wants to run DAS does not read the API reference; they read
`docs/methods/das.md`.

**What `demos/` is.** The executable layer. The onboarding sequence is the
tutorial. The other demo folders are worked research questions, each one
reproducible or marked stale (`docs/demos.md`). A demo is not where a method
is defined; it is where a method is shown answering a question.

**What docstrings are.** The single authored source for explanation and
Python reference. Package level says what the layer is for and names its entry
points. Module level says what the file guarantees. Function and class level
give the contract. Attribute docs on the protocol object model give the
meaning of each JSON field.

## 2. Method pages: the how-to layer

### 2.1 The problem with templates

- A template is one point in a method's option space. `dbm.json` is a
  per-coordinate sigmoid gate on `block_output` under an `l1` term with a
  temperature anneal. That is one DBM.
- The repo ships five DBM templates today (`dbm`, `dbm_head`,
  `dbm_expert_neuron`, `dbm_apply`, `dbm_head_apply`), and the option space
  is larger than any of them: ten authorable gate fields, four
  parametrizations, three group vocabularies, three regularizer kinds, open-
  and closed-loop schedules, phases, and a fit-versus-apply split.
- A page generated from a template documents the point. A practitioner wants
  the space, with the points marked on it.

### 2.2 What a method family is

- A family is a featurizer kind. `subspace` is DAS, `gate` is DBM, `pca` is
  the PCA baseline. The closed set is `FeaturizerKind` in `schema.py`.
- Templates are classified by the kinds their featurizers author, so no list
  of "which templates belong to DBM" is maintained by hand. A template that
  chains a rotation and a gate appears on both pages.
- Everything the page says about the family is the closure of the object
  model over the predicate `kind == <family>`: the fields legal on that kind,
  the enums those fields draw from, the train-section fields that can name a
  featurizer of that kind, the components it may sit on, and the rules that
  refuse combinations.

### 2.3 What already exists in code

Everything below carries a `#:` attribute doc or a docstring today.

- The per-kind field table: `FEATURIZER_FIELDS` says which of the featurizer
  fields a `gate` may author (`group`, `parametrization`, `init`,
  `temperature`, `stretch`, `top_k`, `k_schedule`, `stop_grad_shift`,
  `pool`, `dead`), and `FEATURIZER_SLOTS` which parameter slots it owns.
- The field meanings: the attribute docs on `FeaturizerSpec`, one per field,
  each stating what it does, what happens when absent, and under which
  parametrization or fit state it is legal.
- The closed vocabularies, each with a doc: `GATE_PARAMETRIZATIONS`,
  `GATE_GROUPS`, `GATE_GROUP_AXES`, `GATE_DEAD_RULES`, `K_SCHEDULE_KINDS`,
  `K_SCHEDULE_OF`, `HARD_CONCRETE_TEMPERATURE`, `HARD_CONCRETE_STRETCH`,
  `REGULARIZER_KINDS`, `REGULARIZER_REDUCTIONS`, `ANNEAL_SHAPES`,
  `CONTROL_KINDS`, `CONTROL_SIGNALS`, `CONTROL_SPACES`, `CONTROL_DEFAULTS`,
  `PHASE_UNTIL_UNITS`.
- The train section: `TrainSpec`, `ObjectiveTerm`, `AnnealSchedule`,
  `PhaseSpec`, with docs on `anneal`, `control`, `phases`, `reduce`.
- Where a group is legal: `GATE_GROUP_AXES` plus the group-map functions in
  `causalab/protocol/registry.py`, whose error text already says which
  components have a head axis.
- The refusals: numbered rules in `causalab/protocol/validate.py`, each with
  a message written for the author who tripped it.
- Per template: `header.description`, a paragraph of rationale, and
  `causalab explain`, which reports the digest, required capabilities, and
  forward count.
- The precedent: `tests/protocol/test_vocabulary_census.py` already asserts
  that four closed vocabularies and the spec tables documenting them agree,
  because tables that drifted from code were the observed failure.

### 2.4 What is missing

- **Conditional legality is prose and imperative code, not a table.** "Under
  `hard_concrete` only", "on a trained gate only", "with `file_path` only"
  live in attribute docs and as `if` branches in the validator. A generator
  cannot render a correct field-by-parametrization matrix from either. The
  fix is one declarative table, `FEATURIZER_FIELD_CONDITIONS`, mapping each
  field to the parametrizations and fit states it is legal under, which the
  validator reads for rule 4 and the generator renders. The census test
  asserts the validator's branches and the table agree.
- **The per-parametrization table is hand-written.** Spec §2.5 has a seven
  column table per gate map: soft mask, post-step projection, hard mask,
  legal regularizer, whether temperature is legal, default start. That is a
  record per map, `GATE_MAPS`, and belongs in `schema.py` beside
  `GATE_PARAMETRIZATIONS`. The spec then includes the rendered table instead
  of carrying its own copy.
- **The §2.5 kinds table duplicates `FEATURIZER_FIELDS`.** Same fix: render
  it, include it.
- **Rules are not tagged by family.** To list "what a DBM document can be
  refused for", each rule needs to declare the kinds it concerns. A field on
  the rule record, checked by the census test.
- **No family paragraph.** One sentence per kind saying what the method is,
  as a doc on the `FeaturizerKind` entry or a small `FAMILIES` table.

### 2.5 The page

`docs/methods/dbm.md`, generated. Sections in order, each with its source.

1. **What it is.** The family paragraph. The formula the gate computes, from
   `GATE_MAPS`.
2. **Where it sits.** Components a gate may attach to, and for each `group`
   the components that have the axis it groups over. From `GATE_GROUP_AXES`
   and the shape registry.
3. **Fields.** The ten authorable fields. Per field: domain (enum values or
   type), legal under which parametrizations and fit states, value when
   absent, meaning. From `FEATURIZER_FIELDS`, `FEATURIZER_FIELD_CONDITIONS`,
   and the attribute docs.
4. **Parametrizations.** The per-map table. From `GATE_MAPS`.
5. **Training a gate.** The `train` fields that can name a gate: objective
   terms (`l1`, `l2`, `l0`, with `reduce`), anneal targets
   (`<gate>.theta.temperature`, objective weights), control signals
   (`hard_mask_size`, `hard_mask_fraction`), phases and `freeze_masks`,
   `dead` rules, `early_stop`, `eval`, `checkpoint`. From `TrainSpec` and the
   vocabularies.
6. **Applying a fitted gate.** `file_path`, `top_k`, `pool`, what the bundle
   stamps and checks on reload. From the attribute docs.
7. **What is refused.** The rules tagged `gate`, with their messages. From
   `validate.py`.
8. **Shipped variants.** One row per template that authors a gate: the
   fields it sets, its `header.description`, and the `explain` summary. Each
   row links to the file. This is the matrix the templates alone could not
   give. For DBM today:

   | template | component | group | parametrization | objective | anneal | fit or apply |
   |---|---|---|---|---|---|---|
   | `dbm` | `block_output` | per coordinate | `sigmoid` | `ce` + 0.01 `l1` | temperature 1.0 to 0.01 | fit |
   | `dbm_head` | `attention_premix` | `head` | `sigmoid` | `ce` + 0.01 `l1` | temperature 1.0 to 0.01 | fit |
   | `dbm_expert_neuron` | `expert_activation` + `shared_expert_activation` | `expert_neuron` + per coordinate | `sigmoid` | `ce` + 0.01 `l1` over both gates | both temperatures | fit |
   | `dbm_apply` | `block_output` | per coordinate | `sigmoid` | none | none | apply from `fit/gate.safetensors` |
   | `dbm_head_apply` | `attention_premix` | `head` | `sigmoid` | none | none | apply from `fit/gate.safetensors` |

   The matrix also shows what no template covers: `clamp`, `hard_concrete`,
   `budget`, `site` grouping, `l0`, `control`, `phases`, `dead`, `top_k`
   readout. Those rows of §3 to §6 are what makes the page a reference and
   not a catalogue.
9. **Demos.** Every demo whose documents author a gate. Today:
   `demos/onboarding_tutorial/06_components.md`.
10. **Spec.** Links to §2.5, §2.11, and the §5 rules from item 7.

### 2.6 What the practitioner does with it

- Reads items 1 to 4 once to learn the space.
- Picks the nearest shipped variant from item 8 and copies it.
- Changes fields with item 3 open, and knows before running which
  combinations item 7 will refuse.
- Runs `causalab explain` on the result and compares with the row they
  started from.

## 3. Design guideline

- **Code is the source of truth.** A fact about the code that can be generated
  is generated. A fact that cannot be generated is tested. A fact that is
  neither is an opinion, and lives in a file that says so.
- **One artifact, two readers.** Humans and agents read the same markdown.
  HTML is a view built from it, never the place content is authored.
- **Four kinds of page, never mixed.** Each file is one of the four kinds in
  §1 and says which.
- **Locality.** The explanation of a thing sits next to the thing. Module
  purpose in the module docstring, package purpose in `__init__.py`, function
  contract in the function, field meaning on the field.
- **Executable beats descriptive.** A runnable document, doctest, or demo is
  worth more than prose describing the same behaviour, because it fails when
  it goes stale.
- **No second copy.** A concept is explained once. Other places link to it by
  fully qualified name. A second explanation is a future contradiction.
- **Closed sets are tables.** A vocabulary or a legality rule the parser
  enforces is a data table in code, read by the validator and rendered by the
  generator. An `if` branch is a second copy of a table.
- **Prose earns its place.** A docstring that restates the signature is
  deleted. A paragraph that restates a JSON field is deleted (already the rule
  in `docs/demos.md`).
- **Nothing outside the repo.** No wiki, no vault, no notebook server. A doc a
  fresh checkout cannot see does not exist.

## 4. Rules

Each rule names the check that enforces it. A rule without a check is a
guideline and belongs in §3.

| # | Rule | Check |
|---|---|---|
| R1 | Every module opens with a docstring: what it is for, what it guarantees, which names are the entry points. | ruff `D100` |
| R2 | Every package `__init__.py` has a docstring and an `__all__`. The docstring is the package's explanation page. | ruff `D104`, test asserts `__all__` present |
| R3 | Every public function, class, and method has a Google-style docstring. Private names (`_x`) are exempt. | ruff `D1xx` with `convention = "google"` |
| R4 | Types live in signatures, never repeated in docstrings. Docstrings describe meaning, units, and invariants. | basedpyright strict (exists), ruff `DOC` rules |
| R5 | Cross-references are fully qualified names in backticks, e.g. `causalab.protocol.validate.validate_document`. Never "see the validator". | test: every backticked `causalab.*` path in `docs/` and in docstrings resolves |
| R6 | Docstring examples are doctests. Examples that need weights or an accelerator are marked `# doctest: +SKIP`. | `pytest --doctest-modules` in the CPU gate |
| R7 | The API reference (planned) is generated and committed. `docs/methods/` is hand-written prose around pointer blocks; every block is a rendering of the object it points at. A stale block fails CI. | test: regenerate and diff |
| R8 | Every narrative file in `docs/` declares its kind on line 2: tutorial, how-to, explanation, reference, or spec. | test: header present |
| R9 | A demo carries a `Reproduced` date or is marked stale. | already in `docs/demos.md`; add a CI run for the CPU demos |
| R10 | Any doc over 30 KB opens with a table of contents whose anchors are stable section numbers. | test: TOC present and every `§N` referenced exists |
| R11 | An `llms.txt` under `docs` (planned) lists every page with one line on what question it answers. `CLAUDE.md` points there, not at individual files. | generated with the API reference |
| R12 | New public surface without a docstring does not merge. | ruff in pre-commit, which CI already runs |
| R13 | Every closed vocabulary and every conditional-legality rule the parser enforces is a table in `schema.py` with an attribute doc. The validator reads it; the method pages and the spec render it. | `tests/protocol/test_vocabulary_census.py`, extended |
| R14 | Every field on the protocol object model has an attribute doc, and every validation rule declares the featurizer kinds it concerns. | census test |

## 5. Implementation plan

Baseline, measured 2026-09-15 on `staging`:

| | |
|---|---|
| Public defs with a docstring | 1182 of 1547 (76%) |
| `causalab/analysis` | 3 of 19 (16%) |
| `causalab/neural` | 338 of 515 (66%) |
| `causalab/causal`, `io`, `protocol` | 83% to 90% |
| Top-level `causalab/__init__.py` | empty |
| Docstring style | 37 files Google, 3 files numpy |
| Protocol object model | `causalab/protocol/schema.py`, 4431 lines, fields carry `#:` attribute docs |
| Closed vocabularies with a census guard | 4 of roughly 30 |
| Shipped templates | 24 files in `causalab/configs/protocols/`, 5 of them DBM |
| Prose in `docs/` | ~830 KB, largest file 338 KB |
| Renderer, docstring lint | none |

Phases are independent PRs. Each leaves CI green.

- **Phase 0, audit (no code change).**
  - Render the package once with `pdoc` to see what a docs-from-code site
    contains today. Keep the HTML as a review artifact, do not commit it.
  - Record per-package coverage with `interrogate` as the baseline for R12.
- **Phase 1, conventions (R3, R4).**
  - Enable ruff `D` rules with `convention = "google"` per package, starting
    where the gate is already green: `causal`, `io`, `protocol`.
  - Convert the three numpy-style files to Google style.
  - Keep the existing task exclusions (`entity_binding`, `IOI`).
- **Phase 2, public surface (R2).**
  - Add `__all__` to every package. Fill `causalab/__init__.py` with the
    primitives a practitioner uses, which is the list the researcher guide
    documents.
  - Bring `causalab/analysis` to the gate.
- **Phase 3, package docstrings (R1, R2).**
  - Move each section of `docs/CODEBASE.md` into the matching `__init__.py`
    docstring, section by section. `CODEBASE.md` shrinks to the layering rules
    and invariants that span packages, plus links.
- **Phase 4, declarative legality (R13, R14).** A protocol-layer PR, no docs
  tooling yet.
  - Add `FEATURIZER_FIELD_CONDITIONS` and `GATE_MAPS` to `schema.py`. Rewrite
    the rule 4 branches in `validate.py` to read them. Behaviour unchanged,
    which the existing validator tests establish.
  - Tag each rule record with the featurizer kinds it concerns.
  - Extend the census test: the two new tables agree with the validator, the
    spec §2.5 tables agree with them, every `FeaturizerSpec` field has an
    attribute doc.
- **Phase 5, generators (R7, R11).**
  - A generator built on `griffe`: walks the package, writes one markdown
    file per package into the API reference directory, plus `llms.txt`.
  - The same script writes `docs/methods/<family>.md` per §2.5, one file per
    `FeaturizerKind`, and the two §2.5 spec tables as includes.
  - A CPU-tier test regenerates into a temp dir and diffs against the
    committed copy.
  - Add `griffe` to a new `docs` extra, not to the default install. The
    `#:` attribute docs in `schema.py` are the Sphinx convention; `pdoc`
    reads them and `griffe` does not, so either convert them to attribute
    docstrings or read them directly. Decide in this phase.
- **Phase 6, checks on prose (R5, R8, R10).**
  - One test module, `tests/docs/test_docs.py`, in the pattern of
    `tests/test_architecture_diagram.py`: symbol references resolve, kind
    header present, TOC present on the two specs.
  - Add a generated TOC to `intervention_protocol.md` and
    `workflow_protocol.md`.
- **Phase 7, doctests (R6).**
  - Add `--doctest-modules` to the CPU gate. Mark accelerator examples
    `+SKIP`. Start with `causal/`, where examples need no weights.
- **Phase 8, narrative split.**
  - Split `running_experiments.md` into titled how-to pages, one goal each,
    or fold its method-specific parts into the generated method pages. The
    onboarding tutorial sequence stays as is.
- **Phase 9, site (optional).**
  - `mkdocs.yml` with `mkdocs-material` and `mkdocstrings`, over the same
    markdown. Adds nothing that is not already on disk. Deferred until the
    markdown is worth browsing.

Non-goals:

- No Sphinx. Existing prose is markdown and the cross-reference needs are met
  by R5.
- The spec prose stays hand-written. The object model in `schema.py` is a
  schema in all but name, so its tables are generated and included, but the
  normative text around them (the `do` algebra, sweep semantics, the engine
  contract) is explanation, not reference, and is checked rather than
  generated.

## 6. Overhead for contributors

What each rule costs per PR, and what it buys.

- **Writing docstrings (R1 to R4).** Three to ten lines per public symbol, one
  short paragraph per module. Ruff cannot autofix a missing docstring, so this
  is the one rule that blocks a commit on human or agent effort. A coding agent
  given R1 to R4 in `CLAUDE.md` produces compliant docstrings at negligible
  cost. The residual cost is reviewing them for content, which is the point.
- **Regenerating the reference (R7).** One command before commit. CI already
  auto-commits lint fixes, so a stale API reference or a stale `docs/methods/` block can be
  auto-fixed the same way rather than failing the PR. Then the cost is zero.
- **Adding a method variant.** One link line on the method page. A test holds
  the page's template links to the shipped templates that author the kind, so
  a template added without its line fails CI naming the page.
- **Adding a featurizer field or a parametrization (R13, R14).** One row in
  `FEATURIZER_FIELD_CONDITIONS` or `GATE_MAPS`, one attribute doc, and the
  rule tags. This replaces writing the same facts as `if` branches plus a
  spec table row plus a docstring, so it is less work than today, not more,
  and the census test says when a row is missing.
- **Doctests (R6).** Only examples that run on CPU without weights count.
  Everything else is `+SKIP`. In practice this is `causal/` and the protocol
  layer. No contributor has to make a GPU example runnable in CI.
- **Kind headers and TOCs (R8, R10).** One line per new doc. The TOC is
  generated.
- **Review load.** Reviewers now read prose as well as code. The mitigation is
  §3: prose that restates the signature or the JSON is deleted, so what
  remains is short and worth reading.
- **Migration, one-time.** About 365 missing docstrings, concentrated in
  `neural` and `analysis`. Done package by package behind the per-package ruff
  enable, so no PR has to fix more than its own package. Phase 4 is the one
  refactor with behavioural risk, and it is covered by the validator's own
  tests.
- **Dependencies.** None for the method pages: the pointer readers are a
  small `ast` and `inspect` reader inside the tool (§7). `griffe` and `mkdocs`
  are options for the API reference and a built site later, in a `docs` extra.
- **Where the rules do not apply.** The excluded tasks stay excluded.
  Experiment outputs and run trees are not shipped code and are out of scope.
- **The trade.** A few minutes per PR, against a reference that cannot be
  stale, a public surface that is declared rather than guessed, and prose that
  fails a test when it stops being true. For an agent, this is the difference
  between reading a docstring and reading the function body.

## 7. Decisions from the review of PR 744 (2026-09-16)

The first cut of the method pages was a whole-file generator
(`gen_docs.py`, 900 lines, retired in PR 744) with a dozen branches on the gate kind,
hand-written field domains, paraphrased refusal texts and page prose as
Python strings. It was a third copy of knowledge the parser and the schema
already held, and it drifted once inside its own PR. The review converged on
the following, in order of what was decided.

**Where each kind of content lives.**

- A fact the runtime enforces (legality per state and map, the penalty a map
  admits, whether a map anneals or is ranked) is a data table the runtime
  reads. `FEATURIZER_FIELD_CONDITIONS` and `GATE_MAPS` stay; the census
  sweep holds them to the parser.
- A refusal text is documentation. It is written once, in the table the
  parser or validator prints from, and quoted on the page verbatim. Never
  paraphrased in a generator.
- Prose about a code object (what a kind is, what a field means) is the
  object's docstring or `#:` attribute doc, and the page pulls it.
- A derived fact (which components a gate attaches to, which have a group
  axis) is probed by calling the registry function the loader calls, never
  written down.
- Page-level framing prose is markdown in the page, written by hand. Python
  strings in a generator are the worst home for prose: nobody reviews them as
  prose, and they drift like any other copy.

**The mechanism: pointer blocks, not renderers.** The support-tables tool
gained four method-invariant readers. A marker names a reader and a dotted
object under `causalab`:

    <!-- generated: begin doc   causalab.protocol.schema.GATE_MAPS -->
    <!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec init seed -->
    <!-- generated: begin value causalab.protocol.schema.TRAINABLE_KINDS -->
    <!-- generated: begin call  causalab.protocol.schema.render_field_legality_table gate -->

`doc` pulls a docstring or `#:` doc, `attrs` the `#:` docs of a class as
bullets, `value` an object's value by shape, `call` a package function's
markdown. The tool contains no kind name and no field name; a page needing a
table the generic readers cannot shape gets a small render function in the
package, beside the data, and points at it with `call`. A renamed object
fails `--check` naming the page and the marker. This is autodoc into
committed markdown — the reading surface stays GitHub and the checkout, and
no docs dependency is added. The hand-rolled reader is 25 lines of `ast`;
`griffe` would save little here and reads only string-form attribute
docstrings. Revisit a built site (`mkdocs` + `mkdocstrings` + `mkdocs-macros`
covers all four readers with no custom code) only if the team wants docs read
from a site rather than the repo.

**What the pages give up, deliberately.** The variants matrix (templates
classified by parsing them through the compile prefix) is gone: it was the
heaviest part of the generator, re-implemented the compile stage order, and
carried two crashes. A page lists template links by hand and a test holds
the list to the templates authoring the kind. Framing sentences can rot;
links and commands are held by `tests/docs/test_docs.py`, sentences are not.
That is the trade for having prose at all.

**Deferred: the code reorganisation.** The per-kind knowledge is still
spread over nine modules in two layers (string comparisons against kind and
map names: 10 in the stage module, 8 in the validator, 7 in the
canonicaliser, 27 map comparisons in the stage module alone). The direction
agreed, for after this PR:

- One record per featurizer kind — fields (references into a shared field
  table, plus per-kind legality and enum), slots, trainability, maps with
  their facts and prose, the width rule, the identity fields, a canonicalise
  hook, a train-check hook. The generic parser loop, the validator and the
  canonicaliser dispatch through it. The doc pages then point at the record.
- A kind contributes data and typed hooks, never its own parser: parsing is
  cross-field work (unknown keys with suggestions, sweep and artifact
  wrapping on every leaf, the fit-or-loaded state, swept maps, uniform error
  paths) and stays one function.
- Shared fields (`k`, `init`, `parametrization`, `file_path`, `entry`,
  `dtype`, `description`) are defined once and referenced; the census refuses
  a redefinition. The spec's "one field, one meaning" survives.
- Do the record first with every kind in one file, then split files. Split
  the schema module by document section (sites, featurizers, train, …), with
  kinds as the second level under featurizers; featurizers alone are a fifth
  of it.
- Possibly a `methods` package under `causalab` holding featurizer kinds and
  lowerings such as path patching under one contract with optional parts
  (document surface, lowering hook, validation hook, runtime stage, docs).
  The protocol layer imports only the torch-free half; hold that with a test
  that imports the protocol package with torch blocked. Discovery by scan
  plus a census requiring one complete record per subpackage. The train
  section, the chain rule, the canonical form and the digest stay core.
- The schema library question (pydantic with a discriminated union on
  `kind`: parsing, descriptions and JSON Schema from one declaration) is open.
  Keep the hand-rolled generic parser for now; revisit when a tool needs a
  machine-readable schema to author documents against.
