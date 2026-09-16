# DBM (differentiable binary masking) — the `gate` featurizer

*How-to. Prose by hand; every block between `generated` markers is rendered from `causalab/protocol/` — edit the source, not the block.*

A `gate` is a per-unit mask `m` over a site, trained soft and read hard: an interchange through it takes the counterfactual on the kept units and the base on the rest, so the fitted mask *is* the localization. `featurize(x) → (f, err)` is `(m⊙x, (1−m)⊙x)`. Its one parameter slot is `<name>.theta`, auto-declared from (model, site) and never authored (spec §6); `train.params` names it whole or by slot. Read §1 to §3 once, copy the nearest template in §7, and read §6 before a run.

Normative text: [§2.5 `featurizers`](../intervention_protocol.md#25-featurizers), [§2.11 `train`](../intervention_protocol.md#211-train), [§5 validation](../intervention_protocol.md#5-validation--load-error-checklist).

## 1. Where it sits

A gate attaches to any site whose component has a feature width ([the index](README.md) lists them). A `group` shares one parameter across the units of one axis of the site's shape; which components have that axis is decided by the same derivation the loader runs (`causalab.protocol.registry.site_group_map`), on the reference model:

<!-- generated: begin call causalab.protocol.registry.render_gate_group_table -->

| group | axis | components it is legal on |
|---|---|---|
| `head` | `head` | `delta_gate`, `delta_query`, `delta_key`, `delta_value`, `delta_beta`, `delta_decay`, `delta_kv_mem`, `delta_state_update`, `delta_kernel_output`, `attention_query_pre_rope`, `attention_key_pre_rope`, `attention_value_states`, `attention_gate`, `attention_query`, `attention_key`, `attention_z`, `deltanet_query`, `deltanet_key`, `deltanet_state`, `attention_result`, `delta_premix`, `attention_premix` |
| `expert_neuron` | `topk` | `expert_activation`, `expert_neuron_output` |
| `site` | `feature` | `embeddings`, `block_input`, `attention_input_norm`, `delta_qkv`, `delta_gate`, `delta_conv`, `delta_query`, `delta_key`, `delta_value`, `delta_beta`, `delta_decay`, `delta_kv_mem`, `delta_state_update`, `delta_kernel_output`, `attention_query_pre_rope`, `attention_key_pre_rope`, `attention_value_states`, `attention_gate`, `attention_query`, `attention_key`, `attention_z`, `deltanet_query`, `deltanet_key`, `deltanet_state`, `attention_result`, `delta_premix`, `attention_output`, `attention_premix`, `block_mid`, `mlp_input_norm`, `mlp_input`, `mlp_output`, `router_logits`, `router_scores`, `expert_gate_proj`, `expert_up_proj`, `expert_activation`, `expert_neuron_output`, `expert_output`, `routed_output`, `shared_expert_gate_proj`, `shared_expert_up_proj`, `shared_expert_activation`, `shared_expert_output`, `shared_expert_gate`, `block_output`, `ln_final`, `lm_head` |

<!-- generated: end call causalab.protocol.registry.render_gate_group_table -->

<!-- generated: begin doc causalab.protocol.schema.GATE_GROUPS -->

The units a `gate` may share one parameter across (§2.5 `group`). A closed set: each value names a derivation of the coordinate→group map from the component's shape, and an unknown one has no map to derive. `head` is one group per head on a head-major component (`attention_premix`, `delta_premix`), so the fitted mask selects heads rather than coordinates. `expert_neuron` is one parameter per `(expert, neuron)` of the routed interior (`expert_activation` or `expert_neuron_output`). A token's slots look their parameters up through `expert_idx`, so the mask is stable across tokens that route differently. `site` is one parameter for the whole site. Its map is `(1, width)`, so an MLP block or the embedding is one unit, the node a circuit-discovery benchmark (MIB) scores it as; it is the `head` map with a single group and shares its code path.

<!-- generated: end doc causalab.protocol.schema.GATE_GROUPS -->

## 2. Fields

Which fields a document may author on a gate being fitted (no `file_path`) and on one loaded from a bundle, and under which `parametrization` where that matters:

<!-- generated: begin call causalab.protocol.schema.render_field_legality_table gate -->

| field | without `file_path` | with `file_path` |
|---|---|---|
| `parametrization` ∈ `sigmoid` \| `clamp` \| `hard_concrete` \| `budget` | any map | any map |
| `group` ∈ `head` \| `expert_neuron` \| `site` | any map | any map |
| `axis` ∈ `position` | any map | any map |
| `init` | any map | **refused** |
| `temperature` | `hard_concrete` | `hard_concrete` |
| `stretch` | `hard_concrete` | `hard_concrete` |
| `dead` ∈ `freeze_after` \| `leak` | any map | **refused** |
| `top_k` | **refused** | any map |
| `k_schedule` | `budget` | **refused** |
| `stop_grad_shift` | `budget` | **refused** |
| `pool` | `budget` | any map |

<!-- generated: end call causalab.protocol.schema.render_field_legality_table gate -->

What each one means:

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec parametrization group axis init temperature stretch dead top_k k_schedule stop_grad_shift pool -->

- **`parametrization`** — `subspace`: the rotation map (`PARAMETRIZATIONS`); `gate`: the theta→mask map (`GATE_PARAMETRIZATIONS`, `sigmoid` when absent). One field because it is one idea — how the stored parameter maps to the object the fit is about — and the enum follows the kind.
- **`group`** — `gate` only: the unit one parameter covers (`GATE_GROUPS`). Absent, the gate is per coordinate — the gate it has always been, with no field added to its canonical form, so no existing document's digest moves. There is no literal spelling of that default: the vocabulary is exactly `head`, `expert_neuron` and `site`. The coordinate→group map is derived from the model and the site's component and never authored (§6).
- **`axis`** — §2.5 `axis` (`GATE_AXES`): `"position"` for a gate whose θ runs over the addressed token positions; `None` is the feature gate.
- **`init`** — Where the fit **starts** (§2.5). `subspace`: `{"file_path": …, "entry": …}` naming a saved basis whose first `k` columns are the starting subspace (a PCA basis at the same site, typically). `gate`: `{"fill": p}`, every unit at mask value `p` (`θ = logit(p)` or `θ = p` by parametrization), the `file_path` form naming a saved `theta` taken verbatim, or `{"from_scores": …}` — a per-unit score table (`_parse_scores_init`) whose top `keep` units start on the kept pole, or whose z-scored values become `theta` under `scale`. `entry` has the semantics of the featurizer's own `entry`. Illegal with `file_path` on the featurizer itself: a loaded featurizer draws nothing and trains nothing, so it has no start to set.
- **`temperature`** — `gate` under `hard_concrete` only: the concrete temperature β and the stretch `[γ, ζ]` of the relaxation (`HARD_CONCRETE_TEMPERATURE` and `HARD_CONCRETE_STRETCH` when absent). Refused under any other map — they name constants of a distribution the other maps do not sample from. `temperature` is sweepable like any hyperparameter; `stretch` deliberately is not: the eval-mode split is derived from it (`hard_concrete_threshold`) and the bundle stamps it into its ArtifactIdentity, so a stretch is a constant of the relaxation the whole fit is read through, not an axis one bundle's entries may differ on. Authoring `temperature` beside an `anneal` on the same gate's temperature is refused (rule 4): the schedule's start would overwrite it before the first step.
- **`stretch`** — The stretch `[γ, ζ]` of the hard-concrete relaxation (`HARD_CONCRETE_STRETCH` when absent), legal exactly where `temperature` is and, unlike it, never swept: the eval-mode split is derived from it and the bundle stamps it (see `temperature`).
- **`dead`** — `gate` only, on a *trained* gate: the dead-unit rule (§2.5 `GATE_DEAD_RULES`) — `{"freeze_after": n}` or `{"leak": ε}`, exactly one. A training rule, so it is refused on a loaded gate at parse and on a gate outside `train.params` by rule 4: there is no step for it to act in. Not stamped into the bundle's ArtifactIdentity — it changes how θ moved, not how θ is read.
- **`top_k`** — `gate` with `file_path` only: read the loaded `theta` out as its `top_k` largest units instead of through the map's threshold (§2.5). The threshold is one cut through a ranking; a ranking method (a budget gate, an attribution score, a magnitude order) has no threshold at all and is *only* readable this way, and a sweep over `top_k` is the kept-count → score curve every mask method reports. Sweepable, an integer in `[0, units]`; `0` keeps nothing (the base run). Absent, the hard mask is the map's own split and no field enters the canonical form. Refused without `file_path`: a fit's readout is decided by its map, and a top-k cut of a *training* mask would make the loss and the eval disagree about which units are on.
- **`k_schedule`** — `gate` under `budget` only, and required there on a fit: how each optimizer step draws its budget `k` (`K_SCHEDULE_KINDS`) — `{"kind": "fixed", "k": n}` or `{"kind": "uniform" | "log_uniform", "low": a, "high": b}` — plus `eval`, the cut the fit's own held-out pass and its `hard_mask_size` are read at (`k` when `fixed`; required under a sampled kind, since no single number is implied). `k` and `eval` are sweepable; the bounds are not. Refused on a loaded gate: a schedule is a training-time object, and a loaded budget gate is read out through `top_k`.
- **`stop_grad_shift`** — `gate` under `budget` only: the `−c_k` ablation — the solved shift enters the mask as a constant, so `θ` receives only the direct `σ'` gradient and the mask's sum is free to drift within a step. Absent, the shift carries its implicit gradient (`∂c/∂θ_i = −σ'_i / Σ σ'_j`), which keeps `Σ m = k` to first order under any update — the default.
- **`pool`** — `gate` under `budget` only, fitted or loaded: the name of the budget pool this gate shares one `k`, one shift and one ranking with (§2.5). Every gate authoring the same name is one budget fit over the union of their units — heads, MLP blocks and the embedding across all layers as one `N` — with one `k_schedule` (a fit) or one `top_k` (loaded), which the members must agree on (rule 4). On a *loaded* gate the pool is a pooled **readout** — one `top_k` cut through the members' joint ranking — and is legal under any map, since a ranking method's curve cuts every unit of a model together whatever fitted them. Never sweepable: a pool is a name. Absent, the gate budgets alone, and no field enters the canonical form. A budget fit stamps `pool` and `pool_units` into its bundle; a stamped pool must match the document's (and a pooled bundle is refused by an unpooled document), while an unstamped bundle may join any readout pool.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec parametrization group axis init temperature stretch dead top_k k_schedule stop_grad_shift pool -->

## 3. Parametrizations

How `theta` becomes the mask, one row per map. The penalty and anneal columns are what rule 4 refuses per map:

<!-- generated: begin doc causalab.protocol.schema.GATE_MAPS -->

How a `gate`'s `theta` maps to its mask (§2.5 `parametrization` on a gate). `sigmoid` is the gate as it has always been: soft `σ(θ/T)`, hard `θ > 0`, the temperature annealable. `clamp` is the raw mask `m = θ` held in `[0, 1]` by a projection after every optimizer step, hard `θ > ½` (`round`), with no temperature to anneal — the DCM relaxation. Absent, `sigmoid`: there is no spelling of the default in the canonical form, so no existing document's digest moves. The `subspace` parametrizations are `PARAMETRIZATIONS`; the field is one word with one meaning — how the stored parameter maps to the object the fit is about — and the enum is chosen by kind. `hard_concrete` is the stochastic L0 relaxation of Louizos, Welling & Kingma 2018 (arXiv 1712.01312), the map NeuroSurgeon's `HardConcrete*` layers implement: in training the mask is *sampled* — `u ~ U(0, 1)`, `s = σ((log u − log(1−u) + θ) / β)`, stretched to `(γ, ζ)` and clipped to `[0, 1]` — one draw per optimizer step, shared by every read and write the gate sits on, so an interchange stays a partition — and resampled at the next step, so a fit that trains a rotation beside the gate cannot learn to exploit a fixed fractional mask the eval-mode hard split will never deliver; at eval the mask is the deterministic `σ(θ)` stretched and clipped, hard above ½, which at the default stretch is `θ > 0` — the sigmoid gate's split (`hard_concrete_threshold`), so a bundle reloads through the same check. Its two constants are the gate fields `temperature` (β) and `stretch` (`[γ, ζ]`), legal only under this map; its penalty is the `l0` regularizer, the expected kept fraction `mean σ(θ − β·log(−γ/ζ))` — and `l0` is legal only under this map, as `l1` is legal only under the deterministic ones (rule 4): the two would otherwise be two spellings of one computation. `budget` is a sparsity-loss-free budget parametrization: each optimizer step draws a budget `k` from the gate's `k_schedule` and the training mask is `σ(θ + c_k)` with the scalar `c_k` solved (bisection) so the mask sums to exactly `k` — so `θ` is learned as a **ranking**, no penalty is authored (`l1`/`l0` are refused, rule 4), there is no temperature and nothing to anneal, and the eval-mode split is a cut of the ranking at a count: `k_schedule.eval` inside the fit, `top_k` on a loaded gate — a budget bundle has no threshold and is refused by a document that names no cut. `stop_grad_shift` is the `−c_k` ablation: the shift enters the mask as a constant instead of carrying its implicit gradient.

<!-- generated: end doc causalab.protocol.schema.GATE_MAPS -->

<!-- generated: begin call causalab.protocol.schema.render_gate_map_table -->

| parametrization | soft mask (train) | after every optimizer step | hard mask (eval, apply) | mask penalty (`train.objective`) | `anneal` on `theta.temperature` | default start |
|---|---|---|---|---|---|---|
| `sigmoid` (absent) | `σ(θ / T)` | nothing | `θ > 0` | **`l1`** = `mean σ(θ/T)`; `l0` is **refused** (rule 4) | legal | `θ = 0`, i.e. `m = ½` |
| `clamp` | `θ` itself | `θ ← clip(θ, 0, 1)` | `θ > ½` (`round`) | **`l1`** = `mean θ`; `l0` is **refused** (rule 4) | **refused** (rule 4): a clamp gate's mask is θ itself, projected into [0, 1] after every step — it has no temperature to anneal | `θ = ½` |
| `hard_concrete` | **sampled**, once per optimizer step: `u ~ U(0,1)`, `s = σ((log u − log(1−u) + θ)/β)`, then `clip(s·(ζ−γ)+γ, 0, 1)` | nothing | `clip(σ(θ)·(ζ−γ)+γ, 0, 1) > ½`, i.e. `θ > logit((½−γ)/(ζ−γ))` — exactly `θ > 0` at the default stretch | **`l0`** = `mean σ(θ − β·log(−γ/ζ))`, the expected kept fraction of the sampled mask; `l1` is **refused** (rule 4) | legal | `θ = 0`, i.e. `m = ½` |
| `budget` | `σ(θ + c_k)` with the step's budget `k` drawn from `k_schedule` and the scalar `c_k` solved so `Σ m = k` exactly | nothing | the **`top_k`** largest `θ` — `k_schedule.eval` inside the fit, the document's `top_k` on a loaded gate; there is no threshold | none — `l1` and `l0` are **refused** (rule 4): the mask's sum *is* the budget | **refused** (rule 4): a budget gate's mask is σ(θ + c_k) with the shift solved per step — it has no temperature to anneal; its sharpness is the budget's | `θ = 0` (`fill` ½) |

<!-- generated: end call causalab.protocol.schema.render_gate_map_table -->

## 4. Training

`train.params` names the gate. The mask penalty is a regularizer term of `train.objective`, `{"l1": ["<gate>"]}` or `{"l0": …}`, one per map as the table above says; `l2` is legal on any trained featurizer. The temperature anneals as `<gate>.theta.temperature` under the maps whose anneal column says legal, and not beside an authored `temperature` (rule 4). The `train` fields a gate uses beyond the objective:

<!-- generated: begin attrs causalab.protocol.schema.TrainSpec anneal control phases -->

- **`anneal`** — §2.11 `anneal`: open-loop schedules keyed by what they move — a trained featurizer's `<name>.<slot>.<hyperparameter>`, or a named objective term's `weight` (`train.objective.<name>.weight`).
- **`control`** — §2.11 `control`: closed-loop schedules — `{<target>: {kind, signal, setpoint, gains, …}}` where the target is a named objective term's `weight` (`train.objective.<name>.weight`) or an anneal-style dotted hyperparameter, and the authored value of the target is the controller's initial value.
- **`phases`** — §2.11 `phases`: consecutive step windows, each narrowing what trains and what is annealed; `None` is the one-phase fit every document before the field was.

<!-- generated: end attrs causalab.protocol.schema.TrainSpec anneal control phases -->

What a fit does about a unit whose mask goes hard-off, the `dead` field:

<!-- generated: begin doc causalab.protocol.schema.GATE_DEAD_RULES -->

§2.5 `dead` — what a trained gate does about a unit whose mask has gone hard-off, the two answers the DBM literature gives and exactly one of them per gate. `freeze_after: n` freezes a unit once it has been hard-off for `n` consecutive optimizer steps (the original DCM: a pruned head stays pruned, so the sweep is a nested sequence by construction). `leak: ε` adds `ε` to the training mask's derivative `∂m/∂θ` — the forward value is unchanged — so a unit whose map has saturated at the zero pole still receives gradient and can come back ("keep gradients alive on dead masks"); the eval-mode hard mask is unchanged. Neither has a spelling in the canonical form when unauthored, so no digest moves.

<!-- generated: end doc causalab.protocol.schema.GATE_DEAD_RULES -->

## 5. Applying a fitted gate

`file_path` loads the bundle and the gate comes back in eval mode, split by its map's hard mask. A `budget` gate has no threshold and is read out at `top_k`, alone or through a `pool` with other loaded gates. The bundle's `ArtifactIdentity` (model, site, `group`, `parametrization`, `dtype`) is checked at load and again at build, so an apply document re-authors the fit's fields (rule 15). The fields that matter then:

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

- **`file_path`** — Load a fitted artifact instead of fitting one. Legal on every kind, and what makes a featurizer *loaded*: it trains nothing (rule 12), authors no start (`init`, `seed`) and no training rule (`dead`, `k_schedule`), and a gate whose map has no threshold is read out through `top_k` (`FEATURIZER_FIELD_CONDITIONS`). The bundle's `ArtifactIdentity` is checked against the document at load and again at build (rule 15). Artifact-valued: a sweep over bundles is a sweep over fits.
- **`entry`** — With `file_path` only: which entry of a swept bundle to load — the coordinate values that pick one fit out of a bundle holding several (`_entry_selector`). Absent, the bundle must hold one.
- **`dtype`** — The precision the featurizer's parameters are held and saved in (`PRECISION_DTYPES`); absent, the model's. Legal on every kind and stamped into a fitted bundle's identity, so an apply document re-authors the fit's.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

## 6. What is refused

Beyond the checklist every document meets (spec §5) and an unknown key (rule 1), in the refusals' own words:

<!-- generated: begin call causalab.protocol.schema.render_field_refusals gate -->

- `init` with `file_path` — 'init' sets the *starting point* of a fit; a featurizer loaded from a file_path has its weights in the file — it draws nothing and trains nothing, so there is no start to set
- `temperature` under another map — 'temperature' and 'stretch' are the constants of the hard_concrete relaxation (§2.5) — this gate maps theta through the authored map, which samples nothing
- `stretch` under another map — 'temperature' and 'stretch' are the constants of the hard_concrete relaxation (§2.5) — this gate maps theta through the authored map, which samples nothing
- `dead` with `file_path` — 'dead' is a rule for units that go hard-off *during a fit*; a gate loaded from a file_path takes no step, so there is nothing for it to act on (§2.5)
- `top_k` without `file_path` — 'top_k' reads a *loaded* gate out as its k largest units — it needs a file_path; a gate being fitted is read through its map's own split, so the loss and the eval agree on which units are on (§2.5)
- `k_schedule` with `file_path` — 'k_schedule' draws a *training* budget; a gate loaded from a file_path trains nothing and is read out at 'top_k' (§2.5)
- `k_schedule` under another map — 'k_schedule' and 'stop_grad_shift' belong to the budget parametrization (§2.5) — this gate maps theta through the authored map, which draws no budget
- `stop_grad_shift` with `file_path` — 'stop_grad_shift' is the budget's training-time ablation — how the solved shift enters the mask's gradient; a gate loaded from a file_path solves no shift and is read out at 'top_k' (§2.5)
- `stop_grad_shift` under another map — 'k_schedule' and 'stop_grad_shift' belong to the budget parametrization (§2.5) — this gate maps theta through the authored map, which draws no budget
- `pool` under another map — 'pool' on a gate being fitted shares one budget, which only the budget parametrization draws (§2.5) — this gate maps theta through the authored map; a pooled readout needs a file_path and a top_k
- [V23] `group_legality` — Group legality (§5.23)
- [V32] `scores_init` — A gate's `init.from_scores` fits the gate (§5.32)

<!-- generated: end call causalab.protocol.schema.render_field_refusals gate -->

Per map, the penalty and anneal columns of the table in §3 are refused as they say (rule 4).

## 7. Shipped templates

Every template in `causalab/configs/protocols/` whose featurizers author a `gate` (a test holds this list to the templates):

- [`dbm`](../../causalab/configs/protocols/dbm.json) — one gate on `block_output`, per coordinate, fitted under `ce` and an `l1` term with the temperature annealed.
- [`dbm_apply`](../../causalab/configs/protocols/dbm_apply.json) — the same gate loaded from its bundle, scored on a split.
- [`dbm_head`](../../causalab/configs/protocols/dbm_head.json) — `group: head` on `attention_premix`: the fitted mask selects heads.
- [`dbm_head_apply`](../../causalab/configs/protocols/dbm_head_apply.json) — the head gate loaded.
- [`dbm_expert_neuron`](../../causalab/configs/protocols/dbm_expert_neuron.json) — `group: expert_neuron` on `expert_activation` beside a per-coordinate gate on `shared_expert_activation`, fitted together.
- [`dbm_expert_neuron_apply`](../../causalab/configs/protocols/dbm_expert_neuron_apply.json) — both loaded.

Every shipped gate uses the default `sigmoid` map in its string form, is a feature gate (no `axis`) and authors none of the conditional fields of §2; the fields table is the reference for those.

## 8. Demos

- [`onboarding_tutorial/06_components.md`](../../demos/onboarding_tutorial/06_components.md)
