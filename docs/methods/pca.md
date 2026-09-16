# PCA baseline — the `pca` featurizer

*How-to. Prose by hand; every block between `generated` markers is rendered from `causalab/protocol/` — edit the source, not the block.*

A `pca` is a fixed basis fitted over a harvest (`causalab.analysis.fit_pca`) and loaded from its bundle; its first `k` components are the subspace an interchange acts in — the untrained, variance-maximizing control a DAS fit is compared with. `featurize(x) → (f, err)` is `(Pᵀx, 0)`. Its one parameter slot is `<name>.weight`. It is not a trainable kind: naming it in `train.params` is refused (rule 12), and it is always authored with a `file_path`.

Normative text: [§2.5 `featurizers`](../intervention_protocol.md#25-featurizers), [§5 validation](../intervention_protocol.md#5-validation--load-error-checklist).

## 1. Where it sits

A basis attaches to any site whose component has a feature width; [the index](README.md) lists them. The bundle's own site must match the document's (rule 15).

## 2. Fields

<!-- generated: begin call causalab.protocol.schema.render_field_legality_table pca -->

| field | without `file_path` | with `file_path` |
|---|---|---|
| `k` | legal | legal |

<!-- generated: end call causalab.protocol.schema.render_field_legality_table pca -->

What it means, and the fields every kind has:

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec k file_path entry dtype -->

- **`k`** — `subspace` and `pca`: the width of the feature space — the first `k` columns of the rotation or basis are the subspace an interchange acts in, and the site's other `d − k` directions pass through untouched. Sweepable: the rank curve every localization reports.
- **`file_path`** — Load a fitted artifact instead of fitting one. Legal on every kind, and what makes a featurizer *loaded*: it trains nothing (rule 12), authors no start (`init`, `seed`) and no training rule (`dead`, `k_schedule`), and a gate whose map has no threshold is read out through `top_k` (`FEATURIZER_FIELD_CONDITIONS`). The bundle's `ArtifactIdentity` is checked against the document at load and again at build (rule 15). Artifact-valued: a sweep over bundles is a sweep over fits.
- **`entry`** — With `file_path` only: which entry of a swept bundle to load — the coordinate values that pick one fit out of a bundle holding several (`_entry_selector`). Absent, the bundle must hold one.
- **`dtype`** — The precision the featurizer's parameters are held and saved in (`PRECISION_DTYPES`); absent, the model's. Legal on every kind and stamped into a fitted bundle's identity, so an apply document re-authors the fit's.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec k file_path entry dtype -->

## 3. What is refused

Beyond the checklist every document meets (spec §5) and an unknown key (rule 1):

<!-- generated: begin call causalab.protocol.schema.render_field_refusals pca -->

- nothing beyond the checklist every document meets (§5): a `pca` has no field that is legal in one state and refused in another

<!-- generated: end call causalab.protocol.schema.render_field_refusals pca -->

## 4. Shipped templates

No template in `causalab/configs/protocols/` authors a `pca` (a test holds this to the templates). A DAS fit started from a PCA basis is [`das_pca_init`](../../causalab/configs/protocols/das_pca_init.json), which reads the basis through the subspace's `init` rather than as a featurizer of its own.

## 5. Demos

- [`onboarding_tutorial/05_variance_vs_cause.md`](../../demos/onboarding_tutorial/05_variance_vs_cause.md)
