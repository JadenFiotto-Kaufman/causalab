# DAS (distributed alignment search) — the `subspace` featurizer

*How-to. Prose by hand; every block between `generated` markers is rendered from `causalab/protocol/` — edit the source, not the block.*

A `subspace` is a trained orthonormal rotation whose first `k` columns are the subspace an interchange acts in; the site's other `d − k` directions pass through untouched, so the fitted subspace *is* the localization. `featurize(x) → (f, err)` is `(Qᵀx, 0)`. Its one parameter slot is `<name>.weight`, auto-declared from (model, site) and never authored (spec §6); `train.params` names it whole or by slot. An *untrained* subspace with an authored `seed` is a random rank-`k` basis — the matched-`k` control every localization report needs.

Normative text: [§2.5 `featurizers`](../intervention_protocol.md#25-featurizers), [§2.11 `train`](../intervention_protocol.md#211-train), [§5 validation](../intervention_protocol.md#5-validation--load-error-checklist).

## 1. Where it sits

A subspace attaches to any site whose component has a feature width; [the index](README.md) lists them.

## 2. Fields

Which fields a document may author on a subspace being fitted or drawn (no `file_path`) and on one loaded from a bundle:

<!-- generated: begin call causalab.protocol.schema.render_field_legality_table subspace -->

| field | without `file_path` | with `file_path` |
|---|---|---|
| `k` | legal | legal |
| `parametrization` ∈ `cayley` \| `matrix_exp` \| `stiefel` | legal | legal |
| `init` | legal | **refused** |
| `seed` | legal | **refused** |

<!-- generated: end call causalab.protocol.schema.render_field_legality_table subspace -->

What each one means:

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec k parametrization init seed -->

- **`k`** — `subspace` and `pca`: the width of the feature space — the first `k` columns of the rotation or basis are the subspace an interchange acts in, and the site's other `d − k` directions pass through untouched. Sweepable: the rank curve every localization reports.
- **`parametrization`** — `subspace`: the rotation map (`PARAMETRIZATIONS`); `gate`: the theta→mask map (`GATE_PARAMETRIZATIONS`, `sigmoid` when absent). One field because it is one idea — how the stored parameter maps to the object the fit is about — and the enum follows the kind.
- **`init`** — Where the fit **starts** (§2.5). `subspace`: `{"file_path": …, "entry": …}` naming a saved basis whose first `k` columns are the starting subspace (a PCA basis at the same site, typically). `gate`: `{"fill": p}`, every unit at mask value `p` (`θ = logit(p)` or `θ = p` by parametrization), the `file_path` form naming a saved `theta` taken verbatim, or `{"from_scores": …}` — a per-unit score table (`_parse_scores_init`) whose top `keep` units start on the kept pole, or whose z-scored values become `theta` under `scale`. `entry` has the semantics of the featurizer's own `entry`. Illegal with `file_path` on the featurizer itself: a loaded featurizer draws nothing and trains nothing, so it has no start to set.
- **`seed`** — `subspace` only: the draw its initial rotation comes from. Absent, it is the document's seed (`train.seed`, or 0 with no fit) — so nothing about an existing document changes. Authoring it is what makes an *untrained* subspace a **random rank-k basis a document can sweep**, which is the matched-k control the localization report requires and no preset could express.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec k parametrization init seed -->

## 3. Parametrizations

<!-- generated: begin doc causalab.protocol.schema.PARAMETRIZATIONS -->

The rotation maps of a `subspace` (§2.5 `parametrization`): how the stored parameter becomes the orthonormal `(d, k)` basis `Q`. `cayley` is the Cayley transform from the start basis, at rank `k` (`O(d k²)` per access; the default the shipped DAS templates author); `matrix_exp` and `stiefel` are torch's `orthogonal` maps — the matrix exponential of a skew-symmetric parameter and a product of Householder reflections. Each bundle stamps the map it was fitted under, so a loading document re-authors it (rule 15).

<!-- generated: end doc causalab.protocol.schema.PARAMETRIZATIONS -->

## 4. Training

`train.params` names the rotation. `l1` and `l2` on a subspace are `|p|` and `p²` over its weight; `l0` counts a mask and is refused (rule 4). The `train` fields beyond the objective:

<!-- generated: begin attrs causalab.protocol.schema.TrainSpec anneal phases -->

- **`anneal`** — §2.11 `anneal`: open-loop schedules keyed by what they move — a trained featurizer's `<name>.<slot>.<hyperparameter>`, or a named objective term's `weight` (`train.objective.<name>.weight`).
- **`phases`** — §2.11 `phases`: consecutive step windows, each narrowing what trains and what is annealed; `None` is the one-phase fit every document before the field was.

<!-- generated: end attrs causalab.protocol.schema.TrainSpec anneal phases -->

## 5. Applying a fitted subspace

`file_path` loads the bundle and the rotation comes back in eval mode. The bundle's `ArtifactIdentity` (model, site, `k`, `parametrization`, `dtype`) is checked at load and again at build, so an apply document re-authors the fit's fields (rule 15). The fields that matter then:

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

- **`file_path`** — Load a fitted artifact instead of fitting one. Legal on every kind, and what makes a featurizer *loaded*: it trains nothing (rule 12), authors no start (`init`, `seed`) and no training rule (`dead`, `k_schedule`), and a gate whose map has no threshold is read out through `top_k` (`FEATURIZER_FIELD_CONDITIONS`). The bundle's `ArtifactIdentity` is checked against the document at load and again at build (rule 15). Artifact-valued: a sweep over bundles is a sweep over fits.
- **`entry`** — With `file_path` only: which entry of a swept bundle to load — the coordinate values that pick one fit out of a bundle holding several (`_entry_selector`). Absent, the bundle must hold one.
- **`dtype`** — The precision the featurizer's parameters are held and saved in (`PRECISION_DTYPES`); absent, the model's. Legal on every kind and stamped into a fitted bundle's identity, so an apply document re-authors the fit's.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

## 6. What is refused

Beyond the checklist every document meets (spec §5) and an unknown key (rule 1), in the refusals' own words:

<!-- generated: begin call causalab.protocol.schema.render_field_refusals subspace -->

- `init` with `file_path` — 'init' sets the *starting point* of a fit; a featurizer loaded from a file_path has its weights in the file — it draws nothing and trains nothing, so there is no start to set
- `seed` with `file_path` — 'seed' picks an *initial* rotation; a featurizer loaded from a file_path has its weights in the file and draws nothing

<!-- generated: end call causalab.protocol.schema.render_field_refusals subspace -->

## 7. Shipped templates

Every template in `causalab/configs/protocols/` whose featurizers author a `subspace` (a test holds this list to the templates):

- [`das`](../../causalab/configs/protocols/das.json) — one rotation on `block_output`, fitted under `ce`.
- [`das_pca_init`](../../causalab/configs/protocols/das_pca_init.json) — a rank sweep, every fit started from the first `k` principal components of a PCA bundle at the same site (`init`).
- [`random_subspace_control`](../../causalab/configs/protocols/random_subspace_control.json) — the untrained control: `k` matched to the fit, `seed` swept, no `train` section.
- [`weekdays_das_apply`](../../causalab/configs/protocols/weekdays_das_apply.json) — a fitted rotation loaded from its bundle and scored on the test split.
- [`weekdays_das_sweep`](../../causalab/configs/protocols/weekdays_das_sweep.json) — `k` × `seed` fits from one harvest, at a layer an earlier stage located.

## 8. Demos

- [`onboarding_tutorial/04_subspace.md`](../../demos/onboarding_tutorial/04_subspace.md)
- [`weekdays_geometry/weekdays_geometry.md`](../../demos/weekdays_geometry/weekdays_geometry.md)
