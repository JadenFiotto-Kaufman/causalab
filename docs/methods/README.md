# Method pages

One page per featurizer kind a practitioner runs as a method: what it is, where it attaches, every field it may author and in which state, what a document is refused for, and the shipped templates and demos to copy from. The prose is written by hand. Every table and every field description between `generated` markers is rendered from the code and held byte for byte to that rendering, so a table cannot go stale and a renamed object fails the check naming the page.

| page | kind | in one line |
|---|---|---|
| [DBM](dbm.md) | `gate` | a per-unit mask over a site, trained soft and read hard; the fitted mask is the localization |
| [DAS](das.md) | `subspace` | a trained orthonormal rotation whose first `k` columns are the subspace an interchange acts in |
| [PCA baseline](pca.md) | `pca` | a fixed basis fitted over a harvest and loaded; the untrained control a DAS fit is compared with |
| [Sparse autoencoder](sae.md) | `sae` | a loaded encoder/decoder pair whose latents are the feature space |

`identity` and `standardize` have no page: nothing to author and nothing to fit, so the kinds table below says all there is.

## The kinds

<!-- generated: begin call causalab.protocol.schema.render_featurizer_kind_table -->

| kind | featurize | param slots | authored fields |
|---|---|---|---|
| `identity` (default) | `(x, 0)` | — | — |
| `subspace` | `(Qᵀx, 0)` | `weight` | `k`, `parametrization` ∈ `cayley` \| `matrix_exp` \| `stiefel`, `init` (on a fit), `seed` (on a fit) |
| `pca` | `(Pᵀx, 0)` | `weight` | `k` |
| `sae` | `(enc(x), x − dec(enc(x)))` | `enc`, `dec`, `b_enc`, `b_dec` | — |
| `standardize` | `((x−μ)/σ, 0)` | `mu`, `sigma` | — |
| `gate` | `(m⊙x, (1−m)⊙x)`, `m` the soft mask in training and the hard mask at eval, by `parametrization` (the table below) | `theta` | `parametrization` ∈ `sigmoid` \| `clamp` \| `hard_concrete` \| `budget`, `group` ∈ `head` \| `expert_neuron` \| `site`, `axis` ∈ `position`, `init` (on a fit), `temperature` (under `hard_concrete`), `stretch` (under `hard_concrete`), `dead` ∈ `freeze_after` \| `leak` (on a fit), `top_k` (with `file_path`), `k_schedule` (under `budget`, on a fit), `stop_grad_shift` (under `budget`, on a fit), `pool` (under `budget` to fit; any map with `file_path`) |

<!-- generated: end call causalab.protocol.schema.render_featurizer_kind_table -->

The kinds `train.params` may name — the ones with gradient-trainable slots (rule 12):

<!-- generated: begin value causalab.protocol.schema.TRAINABLE_KINDS -->

`gate`, `sae`, `subspace`

<!-- generated: end value causalab.protocol.schema.TRAINABLE_KINDS -->

## Where a featurizer attaches

<!-- generated: begin call causalab.protocol.registry.render_widthless_components -->

On `Qwen/Qwen3.6-35B-A3B` a featurizer attaches to 48 of the 56 components — every one whose component has a feature width to derive its shape from, which is every one but `input_ids`, `delta_state`, `attention_scores`, `attention_probs`, `mlp_activation`, `mlp_neuron_output`, `expert_idx`, `expert_permutation`.

<!-- generated: end call causalab.protocol.registry.render_widthless_components -->

## Fields every kind has

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec kind file_path entry dtype description -->

- **`kind`** — The kind (`FEATURIZER_KINDS`), `identity` when absent — the family the method page is about (`FEATURIZER_FAMILIES`). Sweepable.
- **`file_path`** — Load a fitted artifact instead of fitting one. Legal on every kind, and what makes a featurizer *loaded*: it trains nothing (rule 12), authors no start (`init`, `seed`) and no training rule (`dead`, `k_schedule`), and a gate whose map has no threshold is read out through `top_k` (`FEATURIZER_FIELD_CONDITIONS`). The bundle's `ArtifactIdentity` is checked against the document at load and again at build (rule 15). Artifact-valued: a sweep over bundles is a sweep over fits.
- **`entry`** — With `file_path` only: which entry of a swept bundle to load — the coordinate values that pick one fit out of a bundle holding several (`_entry_selector`). Absent, the bundle must hold one.
- **`dtype`** — The precision the featurizer's parameters are held and saved in (`PRECISION_DTYPES`); absent, the model's. Legal on every kind and stamped into a fitted bundle's identity, so an apply document re-authors the fit's.
- **`description`** — Free text for the reader; not part of the canonical form or the digest.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec kind file_path entry dtype description -->

## Refreshing a page

Edit the prose in the page and the docs in the code; then re-render every generated block from the objects its markers point at. A block is `<!-- generated: begin <reader> <pointer> … -->`, the reader one of `doc` (a docstring or `#:` attribute doc), `attrs` (the `#:` docs of a class, as bullets), `value` (an object's value) or `call` (a function's markdown). Normative text for everything here: [§2.5 `featurizers`](../intervention_protocol.md#25-featurizers), [§2.11 `train`](../intervention_protocol.md#211-train), [§5 validation](../intervention_protocol.md#5-validation--load-error-checklist).
