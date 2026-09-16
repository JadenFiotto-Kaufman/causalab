# Sparse autoencoder — the `sae` featurizer

*How-to. Prose by hand; every block between `generated` markers is rendered from `causalab/protocol/` — edit the source, not the block.*

An `sae` is a loaded encoder/decoder pair whose latents are the feature space; the reconstruction error rides along as `err`, so the inverse is exact. `featurize(x) → (f, err)` is `(enc(x), x − dec(enc(x)))`. Its parameter slots are `<name>.enc`, `<name>.dec`, `<name>.b_enc` and `<name>.b_dec`, declared from the bundle. It is a trainable kind — `train.params` may name it whole or by slot — but a shipped one is loaded.

Normative text: [§2.5 `featurizers`](../intervention_protocol.md#25-featurizers), [§2.11 `train`](../intervention_protocol.md#211-train), [§5 validation](../intervention_protocol.md#5-validation--load-error-checklist).

## 1. Where it sits

An autoencoder attaches to any site whose component has a feature width; [the index](README.md) lists them. The bundle's own site must match the document's (rule 15).

## 2. Fields

<!-- generated: begin call causalab.protocol.schema.render_field_legality_table sae -->

A `sae` authors no field of its own. `file_path`, `entry`, `dtype` and `description` are legal on every kind (§2.5).

<!-- generated: end call causalab.protocol.schema.render_field_legality_table sae -->

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

- **`file_path`** — Load a fitted artifact instead of fitting one. Legal on every kind, and what makes a featurizer *loaded*: it trains nothing (rule 12), authors no start (`init`, `seed`) and no training rule (`dead`, `k_schedule`), and a gate whose map has no threshold is read out through `top_k` (`FEATURIZER_FIELD_CONDITIONS`). The bundle's `ArtifactIdentity` is checked against the document at load and again at build (rule 15). Artifact-valued: a sweep over bundles is a sweep over fits.
- **`entry`** — With `file_path` only: which entry of a swept bundle to load — the coordinate values that pick one fit out of a bundle holding several (`_entry_selector`). Absent, the bundle must hold one.
- **`dtype`** — The precision the featurizer's parameters are held and saved in (`PRECISION_DTYPES`); absent, the model's. Legal on every kind and stamped into a fitted bundle's identity, so an apply document re-authors the fit's.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

## 3. What is refused

Beyond the checklist every document meets (spec §5) and an unknown key (rule 1):

<!-- generated: begin call causalab.protocol.schema.render_field_refusals sae -->

- nothing beyond the checklist every document meets (§5): a `sae` has no field that is legal in one state and refused in another

<!-- generated: end call causalab.protocol.schema.render_field_refusals sae -->

## 4. Shipped templates and demos

No template in `causalab/configs/protocols/` and no demo authors an `sae` today (a test holds the template half to the templates).
