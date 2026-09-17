"""The intervention-protocol object model and its strict parser.

This module is the authoring surface of ``docs/intervention_protocol.md``:
it turns a raw JSON/YAML mapping into a typed, frozen
:class:`Document` — or refuses with a structured
:class:`~causalab.protocol.errors.ParseError` /
:class:`~causalab.protocol.errors.ValidationError`. Everything here is
engine-free and torch-free: sites, positions, featurizers and writes are
pure data records; an engine interprets them (spec §8).

Parsing owns the *shape* rules of the spec:

* strict keys — an unknown field anywhere is an error with suggestions
  (§5.1); closed enums reject with suggestions; derived fields (§6) may not
  be authored;
* section order — a *recommendation* (§1), not a rule: an unconventional
  order warns and parses on (§5.2);
* sugar — a bare int ``pos`` means ``{"index": n}`` and the bare string
  ``"all"`` means ``{"all": true}`` (§2.3); sugar is expanded here, so the
  object model only ever holds the canonical spelling;
* the two value wrappers — ``{"sweep": …}`` (§3) and
  ``{"artifact": …, "key": …}`` (§1) — are accepted anywhere a scalar-,
  list- or spec-typed *leaf* is expected and preserved as
  :class:`Sweep` / :class:`ArtifactRef` values; expansion and resolution
  happen in :mod:`causalab.protocol.sweep` / :mod:`causalab.protocol.resolve`.

Cross-reference and semantic checks (the §5 checklist items that need the
whole document) live in :mod:`causalab.protocol.validate`, not here.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    Union,
    get_args,
)

from causalab.protocol.errors import (
    RULES,
    ParseError,
    ProtocolWarning,
    ValidationError,
    suggest,
)
from causalab.protocol.estimand import (
    IDENTITY_COLUMNS,
    METRIC_UNITS,
    UNITS,
    EstimandError,
    metric_identity,
    parse_identifier,
)

if TYPE_CHECKING:
    from causalab.protocol.segments import SegmentsSpec

__all__ = [
    "ALIGNMENT_CARDINALITIES",
    "ALL_POSITIONS",
    "AlignmentCardinality",
    "ArtifactRef",
    "COMPONENTS",
    "DEPRECATED_COMPONENTS",
    "DEPRECATED_IN",
    "CodeSpec",
    "Component",
    "DataRole",
    "Do",
    "Document",
    "WriteSpec",
    "FEATURIZER_KINDS",
    "FeaturizerKind",
    "GATE_GROUP_AXES",
    "GATE_GROUPS",
    "SCORES_INIT_DEFAULTS",
    "SCORES_INIT_KEYS",
    "FORWARD_MASKS",
    "GATE_AXES",
    "GATE_DEAD_RULES",
    "span_length",
    "GATE_PARAMETRIZATIONS",
    "OPTIMIZER_SCHEDULES",
    "K_SCHEDULE_KINDS",
    "K_SCHEDULE_OF",
    "HARD_CONCRETE_STRETCH",
    "HARD_CONCRETE_TEMPERATURE",
    "hard_concrete_theta",
    "hard_concrete_threshold",
    "ANNEAL_SHAPES",
    "AnnealSchedule",
    "CONTROL_DEFAULTS",
    "CONTROL_KINDS",
    "CONTROL_SIGNALS",
    "CONTROL_SPACES",
    "OBJECTIVE_WEIGHT_PREFIX",
    "FeaturizerSpec",
    "IMSpec",
    "MATCH_MODES",
    "MECHANISMS",
    "METRIC_FIELD_DEFAULTS",
    "MINIMUM_COUNT_FIELD",
    "RAGGED_FIELD",
    "RAGGED_POLICIES",
    "METRIC_DOMAINS",
    "METRIC_KINDS",
    "Mechanism",
    "MetricKind",
    "MetricDomain",
    "MetricSpec",
    "GROUP_ORDER",
    "HEADER_FIELDS",
    "METHOD_SECTIONS",
    "ConstraintSpec",
    "DRAW_KINDS",
    "ObjectiveTerm",
    "PER_PARAMS_OPTIMIZER_FIELDS",
    "PHASE_UNTIL_UNITS",
    "PhaseSpec",
    "REGULARIZER_COSTS",
    "REGULARIZER_KINDS",
    "REGULARIZER_REDUCTIONS",
    "MODEL_DTYPE_DEFAULT",
    "MIGRATABLE_PROTOCOL_VERSIONS",
    "PROTOCOL_VERSION",
    "REQUIRED_METHOD_SECTIONS",
    "REQUIRED_SECTIONS",
    "ModelRef",
    "NON_COLUMN_METRIC_FIELDS",
    "OPTIONAL_METRIC_FIELDS",
    "ParamSpec",
    "QUANT_METHODS",
    "QUANT_SCHEMES",
    "QuantizationSpec",
    "PositionSpec",
    "ReadSpec",
    "RESERVED_NAMES",
    "READ_TARGET_METRIC_KINDS",
    "SAVE_KINDS",
    "TRAJECTORY_EVERY_UNITS",
    "RowRole",
    "SECTION_ORDER",
    "SaveEntry",
    "STREAMS",
    "SiteSpec",
    "Stream",
    "TOKEN_COLUMN_METRIC_KINDS",
    "TOKEN_FORMS",
    "TOP_K_RANKINGS",
    "VOCAB_TOP_K_RANKING",
    "WHOLE_WINDOW_METRIC_KINDS",
    "TokenForm",
    "check_protocol_version",
    "concrete_int",
    "concrete_str",
    "dotted_path",
    "metric_column_fields",
    "metric_reads_vocabulary",
    "Sweep",
    "TrainSpec",
    "parse_document",
    "load_raw",
    "tree_path",
]


# --------------------------------------------------------------------------- #
# closed vocabularies (spec §2.4, §2.5, §2.8, §2.10)
# --------------------------------------------------------------------------- #

Component = Literal[
    "input_ids",
    "embeddings",
    "block_input",
    "attention_input_norm",
    "delta_qkv",
    "delta_gate",
    "delta_conv",
    "delta_query",
    "delta_key",
    "delta_value",
    "delta_beta",
    "delta_decay",
    "delta_kv_mem",
    "delta_state_update",
    "delta_state",
    "delta_kernel_output",
    "attention_query_pre_rope",
    "attention_key_pre_rope",
    "attention_value_states",
    "attention_gate",
    "attention_query",
    "attention_key",
    "attention_scores",
    "attention_z",
    "deltanet_query",
    "deltanet_key",
    "deltanet_state",
    "attention_result",
    "delta_premix",
    "attention_output",
    "attention_premix",
    "attention_probs",
    "block_mid",
    "mlp_input_norm",
    "mlp_input",
    "mlp_output",
    "mlp_activation",
    "mlp_neuron_output",
    "router_logits",
    "router_scores",
    "expert_idx",
    "expert_gate_proj",
    "expert_up_proj",
    "expert_activation",
    "expert_neuron_output",
    "expert_permutation",
    "expert_output",
    "routed_output",
    "shared_expert_gate_proj",
    "shared_expert_up_proj",
    "shared_expert_activation",
    "shared_expert_output",
    "shared_expert_gate",
    "block_output",
    "ln_final",
    "lm_head",
]

#: The closed site component vocabulary (§2.4). The order here is the
#: literal's, not §2.4's: the spec lists the same 56 names in a reading
#: order that walks a block, and nothing depends on either ordering — the
#: census in tests/protocol/test_vocabulary_census.py compares the *sets*.
COMPONENTS: tuple[Component, ...] = get_args(Component)

#: Retired spellings, mapped to the name that replaced them. Applied at parse,
#: so ``SiteSpec.component`` is always current and nothing downstream — the
#: shape table, the rank table, the tap table — carries a second name.
#:
#: 🔤 ``attention_value`` named the o-projection's **input**: on a gated
#: attention family (Qwen3.5/3.6) that is ``z · σ(gate)``, on an ungated one it
#: is ``z``, and on neither is it the value vectors. Round 2 exposes those
#: separately, as ``attention_value_states`` — so the old name had to move
#: before the collision, not after it (nnterp#51 is the same mistake, made
#: after).
#:
#: The replacement name is **not** reused for the new box, deliberately: a
#: document written against the old vocabulary would then load, and silently
#: mean a different tensor. An alias that redirects is safe; an alias that
#: rebinds is the failure this whole rename exists to avoid.
#:
#: 🔤 The eight ``deltanet_*`` spellings below named the nnterp engine's
#: ``.source`` reach into the Gated DeltaNet forward, while ``delta_*`` named
#: the reference engine's kernel-boundary reach — two vocabularies for **the
#: same physical tensors** (📐 measured identical in shape and value on
#: ``tiny-random/qwen3.5-moe``, ``tests/_helpers/a3b_sweep.py``'s pair table).
#: That was a leak: a document had to name the *backend* to name the tensor.
#: The schema keeps one canonical, engine-neutral spelling
#: per tensor — ``delta_*`` names the delta rule the tensor belongs to, where
#: ``deltanet_core_out`` / ``deltanet_gated_out`` / ``deltanet_qkv_conv`` echo
#: the modeling file's variable names — and each engine translates it to its
#: own mechanism. Only pairs whose two spellings agree in **shape and timing**
#: are aliased: ``deltanet_query`` / ``deltanet_key`` (pre GVA tiling,
#: key-head space) and ``deltanet_state`` (per 64-token chunk, not per step)
#: stay their own names with a typed backend requirement
#: (``registry.BACKEND_PAIRS``), because an alias there would *rebind*, not
#: redirect — ``registry.alias_would_rebind`` is the guard, and the census
#: holds every entry here to it.
DEPRECATED_COMPONENTS: dict[str, Component] = {
    "attention_value": "attention_premix",
    "deltanet_qkv": "delta_qkv",
    "deltanet_qkv_conv": "delta_conv",
    "deltanet_gate": "delta_gate",
    "deltanet_value": "delta_value",
    "deltanet_beta": "delta_beta",
    "deltanet_decay": "delta_decay",
    "deltanet_core_out": "delta_kernel_output",
    "deltanet_gated_out": "delta_premix",
}

#: The protocol ``version`` under which each retired spelling became an alias
#: — the version *from which* a document may still author it and canonicalize
#: to the replacement. Every entry is ``"1"`` today: the one protocol version
#: there is, and both folds were made within it. The field is
#: per alias so that a spelling retired under a later version records that
#: version, not the table's oldest.
DEPRECATED_IN: dict[str, str] = {alias: "1" for alias in DEPRECATED_COMPONENTS}

#: The mixer a layer carries. A hybrid tower has both — 📐 on
#: ``tiny-random/qwen3.5-moe`` three of four layers are Gated DeltaNet
#: (``linear_attention``) and one is ``full_attention`` — so a site may name the
#: stream it means and be refused at load if the layer it names carries the
#: other one.
#:
#: 🐞 This once parsed as an **integer**, which made the field
#: unusable from either side: ``sites._check_stream`` only reads a *string*
#: (``bundle.stream_at`` returns one), so an authored ``"full_attention"`` was
#: rejected by the parser before the check could see it, and an authored ``0``
#: parsed and was then silently ignored — precisely the failure ``_moe_site``'s
#: ``expert`` refusal was written to avoid. Round 1's tests missed it because
#: they construct ``SiteSpec(stream="full_attention")`` directly, exercising a
#: path no document can reach.
Stream = Literal["full_attention", "linear_attention"]

#: Every stream a site may name.
STREAMS: tuple[Stream, ...] = get_args(Stream)

#: Components that carry no ``layers`` field.
#: ``input_ids`` joins these because it is the model's *input* (§5.4), not an
#: activation inside a block — there is no layer at which to read it.
LAYERLESS_COMPONENTS: frozenset[str] = frozenset(
    {"input_ids", "embeddings", "ln_final", "lm_head"}
)

FeaturizerKind = Literal["identity", "subspace", "pca", "sae", "standardize", "gate"]
FEATURIZER_KINDS: tuple[FeaturizerKind, ...] = get_args(FeaturizerKind)

#: Auto-declared param slots per featurizer kind (§2.5) — ``<name>.<slot>``.
FEATURIZER_SLOTS: dict[str, tuple[str, ...]] = {
    "identity": (),
    "subspace": ("weight",),
    "pca": ("weight",),
    "sae": ("enc", "dec", "b_enc", "b_dec"),
    "standardize": ("mu", "sigma"),
    "gate": ("theta",),
}

#: Authorable choice fields per featurizer kind (§2.5) — everything else about
#: a featurizer (width, param shapes, slots) is derived and may not be
#: authored (§6). ``file_path`` (load a fitted artifact) and ``dtype`` are
#: legal on every kind; ``description`` is legal everywhere.
FEATURIZER_FIELDS: dict[str, frozenset[str]] = {
    "identity": frozenset(),
    "subspace": frozenset({"k", "parametrization", "init", "seed"}),
    "pca": frozenset({"k"}),
    "sae": frozenset(),
    "standardize": frozenset(),
    "gate": frozenset(
        {
            "group",
            "parametrization",
            "init",
            "temperature",
            "stretch",
            "top_k",
            "k_schedule",
            "stop_grad_shift",
            "pool",
            "dead",
            "axis",
        }
    ),
}

#: The kinds with gradient-trainable slots (§5.12) — the ones ``train.params``
#: may name. ``pca`` and ``standardize`` are computed from data and loaded,
#: ``identity`` has nothing to fit.
TRAINABLE_KINDS: frozenset[str] = frozenset({"subspace", "gate", "sae"})


@dataclasses.dataclass(frozen=True)
class FeaturizerFamily:
    """One method family (§2.5): a featurizer kind as the practitioner who
    runs it meets it. ``sentence`` is what the kind does in one line;
    ``featurize`` the map ``featurize(x) → (f, err)`` computes, as the spec's
    kinds table prints it. The method page under ``docs/methods/`` is written
    by hand and pulls its tables from here."""

    sentence: str
    featurize: str


#: One entry per :data:`FEATURIZER_KINDS` (the census holds the two to one
#: set): what each kind does in one line, and the ``featurize`` cell of the
#: §2.5 kinds table (:func:`render_featurizer_kind_table`).
FEATURIZER_FAMILIES: dict[str, FeaturizerFamily] = {
    "identity": FeaturizerFamily(
        "The default: the site's own coordinates, unchanged — ``f = x``, "
        "nothing to fit and nothing to load.",
        "`(x, 0)`",
    ),
    "subspace": FeaturizerFamily(
        "A trained orthonormal rotation whose first ``k`` columns are the "
        "subspace an interchange acts in; the site's other directions pass "
        "through untouched, so the fitted subspace *is* the localization.",
        "`(Qᵀx, 0)`",
    ),
    "pca": FeaturizerFamily(
        "A fixed basis fitted over a harvest (``causalab.analysis.fit_pca``) "
        "and loaded from its bundle; its first ``k`` components are the "
        "subspace — the untrained control a DAS fit is compared with.",
        "`(Pᵀx, 0)`",
    ),
    "sae": FeaturizerFamily(
        "A loaded encoder/decoder pair whose latents are the feature space; "
        "the reconstruction error rides along as ``err`` so the inverse is "
        "exact.",
        "`(enc(x), x − dec(enc(x)))`",
    ),
    "standardize": FeaturizerFamily(
        "A loaded per-coordinate ``(μ, σ)``: the site in z-scored units, so a "
        "write in feature space is a write in standard deviations.",
        "`((x−μ)/σ, 0)`",
    ),
    "gate": FeaturizerFamily(
        "A per-unit mask ``m`` over the site, trained soft and read hard: an "
        "interchange through it takes the counterfactual on the kept units "
        "and the base on the rest, so the fitted mask *is* the localization.",
        "`(m⊙x, (1−m)⊙x)`, `m` the soft mask in training and the hard mask "
        "at eval, by `parametrization` (the table below)",
    ),
}

#: The rotation maps of a ``subspace`` (§2.5 ``parametrization``): how the
#: stored parameter becomes the orthonormal ``(d, k)`` basis ``Q``. ``cayley``
#: is the Cayley transform from the start basis, at rank ``k`` (``O(d k²)`` per
#: access; the default the shipped DAS templates author); ``matrix_exp`` and
#: ``stiefel`` are torch's ``orthogonal`` maps — the matrix exponential of a
#: skew-symmetric parameter and a product of Householder reflections. Each
#: bundle stamps the map it was fitted under, so a loading document re-authors
#: it (rule 15).
PARAMETRIZATIONS: tuple[str, ...] = ("cayley", "matrix_exp", "stiefel")

#: §2.5 ``dead`` — what a trained gate does about a unit whose mask has gone
#: hard-off, the two answers the DBM literature gives and exactly one of them
#: per gate. ``freeze_after: n`` freezes a unit once it has been hard-off for
#: ``n`` consecutive optimizer steps (the original DCM: a pruned head stays
#: pruned, so the sweep is a nested sequence by construction). ``leak: ε``
#: adds ``ε`` to the training mask's derivative ``∂m/∂θ`` — the forward value
#: is unchanged — so a unit whose map has saturated at the zero pole still
#: receives gradient and can come back ("keep gradients alive on dead
#: masks"); the eval-mode hard mask is unchanged. Neither has a spelling
#: in the canonical form when unauthored, so no digest moves.
GATE_DEAD_RULES: tuple[str, ...] = ("freeze_after", "leak")


@dataclasses.dataclass(frozen=True)
class GateMap:
    """One ``gate`` parametrization (§2.5): how ``theta`` becomes the mask, as
    the columns of the spec's table and the three facts the checklist reads
    off it. ``penalty`` is the one mask regularizer the map admits — ``l1``
    for a deterministic soft mask, ``l0`` for a sampled one, ``None`` when the
    mask's sum is fixed by construction — and ``penalty_formula`` what it
    computes; ``anneals_temperature`` says whether ``<gate>.theta.temperature``
    is an ``anneal`` target, with ``no_temperature_because`` the refusal when
    it is not; ``ranked`` marks a map whose eval-mode split is a count cut
    through a ranking rather than a threshold, so it needs a ``k_schedule`` to
    fit, a ``top_k`` to read, and cannot be a ``control`` signal. The prose
    columns are the spec's cells, rendered by :func:`render_gate_map_table`;
    the validator reads the facts (rule 4)."""

    name: str
    soft_mask: str
    post_step: str
    hard_mask: str
    penalty: str | None
    penalty_formula: str
    anneals_temperature: bool
    default_start: str
    no_temperature_because: str = ""
    ranked: bool = False

    def __post_init__(self) -> None:
        if self.anneals_temperature == bool(self.no_temperature_because):
            raise ValueError(
                f"gate map {self.name!r}: a map that does not anneal its "
                "temperature says why, and one that does says nothing"
            )
        if (self.penalty is None) == bool(self.penalty_formula):
            raise ValueError(
                f"gate map {self.name!r}: a penalty comes with its formula"
            )


#: How a ``gate``'s ``theta`` maps to its mask (§2.5 ``parametrization`` on a
#: gate). ``sigmoid`` is the gate as it has always been: soft ``σ(θ/T)``,
#: hard ``θ > 0``, the temperature annealable. ``clamp`` is the raw mask
#: ``m = θ`` held in ``[0, 1]`` by a projection after every optimizer step,
#: hard ``θ > ½`` (``round``), with no temperature to anneal — the DCM
#: relaxation. Absent, ``sigmoid``: there is no spelling of the default in the
#: canonical form, so no existing document's digest moves. The ``subspace``
#: parametrizations are :data:`PARAMETRIZATIONS`; the field is one word with
#: one meaning — how the stored parameter maps to the object the fit is about
#: — and the enum is chosen by kind.
#: ``hard_concrete`` is the stochastic L0 relaxation of Louizos, Welling &
#: Kingma 2018 (arXiv 1712.01312), the map NeuroSurgeon's ``HardConcrete*``
#: layers implement: in training the mask is *sampled* — ``u ~ U(0, 1)``,
#: ``s = σ((log u − log(1−u) + θ) / β)``, stretched to ``(γ, ζ)`` and clipped
#: to ``[0, 1]`` — one draw per optimizer step, shared by every read and
#: write the gate sits on, so an interchange stays a partition — and
#: resampled at the next step, so a fit that trains a rotation beside the gate
#: cannot learn to exploit a fixed fractional mask the eval-mode hard split
#: will never deliver; at eval the mask is the deterministic ``σ(θ)`` stretched
#: and clipped, hard above ½, which at the default stretch is ``θ > 0`` — the
#: sigmoid gate's split (:func:`hard_concrete_threshold`), so a bundle reloads
#: through the same check. Its two constants are the gate fields
#: ``temperature`` (β) and ``stretch`` (``[γ, ζ]``), legal only under this
#: map; its penalty is the ``l0`` regularizer, the expected kept fraction
#: ``mean σ(θ − β·log(−γ/ζ))`` — and ``l0`` is legal only under this map, as
#: ``l1`` is legal only under the deterministic ones (rule 4): the two would
#: otherwise be two spellings of one computation.
#: ``budget`` is a sparsity-loss-free budget parametrization: each optimizer
#: step draws a budget
#: ``k`` from the gate's ``k_schedule`` and the training mask is
#: ``σ(θ + c_k)`` with the scalar ``c_k`` solved (bisection) so the mask sums
#: to exactly ``k`` — so ``θ`` is learned as a **ranking**, no penalty is
#: authored (``l1``/``l0`` are refused, rule 4), there is no temperature and
#: nothing to anneal, and the eval-mode split is a cut of the ranking at a
#: count: ``k_schedule.eval`` inside the fit, ``top_k`` on a loaded gate — a
#: budget bundle has no threshold and is refused by a document that names no
#: cut. ``stop_grad_shift`` is the ``−c_k`` ablation: the shift enters
#: the mask as a constant instead of carrying its implicit gradient.
GATE_MAPS: dict[str, GateMap] = {
    "sigmoid": GateMap(
        "sigmoid",
        soft_mask="`σ(θ / T)`",
        post_step="nothing",
        hard_mask="`θ > 0`",
        penalty="l1",
        penalty_formula="`mean σ(θ/T)`",
        anneals_temperature=True,
        default_start="`θ = 0`, i.e. `m = ½`",
    ),
    "clamp": GateMap(
        "clamp",
        soft_mask="`θ` itself",
        post_step="`θ ← clip(θ, 0, 1)`",
        hard_mask="`θ > ½` (`round`)",
        penalty="l1",
        penalty_formula="`mean θ`",
        anneals_temperature=False,
        no_temperature_because=(
            "a clamp gate's mask is θ itself, projected into [0, 1] after every "
            "step — it has no temperature to anneal"
        ),
        default_start="`θ = ½`",
    ),
    "hard_concrete": GateMap(
        "hard_concrete",
        soft_mask=(
            "**sampled**, once per optimizer step: `u ~ U(0,1)`, "
            "`s = σ((log u − log(1−u) + θ)/β)`, then `clip(s·(ζ−γ)+γ, 0, 1)`"
        ),
        post_step="nothing",
        hard_mask=(
            "`clip(σ(θ)·(ζ−γ)+γ, 0, 1) > ½`, i.e. `θ > logit((½−γ)/(ζ−γ))` — "
            "exactly `θ > 0` at the default stretch"
        ),
        penalty="l0",
        penalty_formula=(
            "`mean σ(θ − β·log(−γ/ζ))`, the expected kept fraction of the sampled mask"
        ),
        anneals_temperature=True,
        default_start="`θ = 0`, i.e. `m = ½`",
    ),
    "budget": GateMap(
        "budget",
        soft_mask=(
            "`σ(θ + c_k)` with the step's budget `k` drawn from `k_schedule` "
            "and the scalar `c_k` solved so `Σ m = k` exactly"
        ),
        post_step="nothing",
        hard_mask=(
            "the **`top_k`** largest `θ` — `k_schedule.eval` inside the fit, "
            "the document's `top_k` on a loaded gate; there is no threshold"
        ),
        penalty=None,
        penalty_formula="",
        anneals_temperature=False,
        no_temperature_because=(
            "a budget gate's mask is σ(θ + c_k) with the shift solved per step "
            "— it has no temperature to anneal; its sharpness is the budget's"
        ),
        default_start="`θ = 0` (`fill` ½)",
        ranked=True,
    ),
}

#: The map an unauthored ``parametrization`` means on a gate. It has no
#: spelling in the canonical form, so no existing document's digest moves.
GATE_DEFAULT_MAP: str = "sigmoid"

#: The closed vocabulary of gate maps, in :data:`GATE_MAPS`' order.
GATE_PARAMETRIZATIONS: tuple[str, ...] = tuple(GATE_MAPS)


@dataclasses.dataclass(frozen=True)
class FieldLegality:
    """Where one authorable featurizer field is legal (§2.5), as two sets of
    gate maps: ``fit`` for a featurizer being fitted (no ``file_path``) and
    ``loaded`` for one read from a bundle. ``None`` is every map — and, on a
    kind with no maps, simply legal; an empty set is refused in that state.
    The three ``why`` texts are the refusals, verbatim: ``why_fit`` when the
    field is authored on a fit and ``fit`` is empty, ``why_loaded`` beside a
    ``file_path`` when ``loaded`` is empty, ``why_map`` when the gate's map
    (every arm of a swept one) is outside the set, with ``{maps}`` naming the
    arms. The parser reads this table, the method pages render it, and the
    census holds the two to each other."""

    fit: frozenset[str] | None
    loaded: frozenset[str] | None
    why_fit: str = ""
    why_loaded: str = ""
    why_map: str = ""

    def __post_init__(self) -> None:
        for state, legal, why in (
            ("fit", self.fit, self.why_fit),
            ("loaded", self.loaded, self.why_loaded),
        ):
            if legal is not None and not legal <= frozenset(GATE_MAPS):
                raise ValueError(f"{state}: {sorted(legal)} are not gate maps")
            if (legal == frozenset()) != bool(why):
                raise ValueError(
                    f"{state}: a state the field is refused in says why, and "
                    "only such a state does"
                )
        restricted = any(
            legal is not None and legal and legal != frozenset(GATE_MAPS)
            for legal in (self.fit, self.loaded)
        )
        if restricted != bool(self.why_map):
            raise ValueError(
                "a field legal under some maps and not others says why, and "
                "only such a field does"
            )

    def legal(self, *, loaded: bool) -> frozenset[str] | None:
        """The maps the field is legal under in one state."""
        return self.loaded if loaded else self.fit


_HARD_CONCRETE: frozenset[str] = frozenset({"hard_concrete"})
_BUDGET: frozenset[str] = frozenset({"budget"})
_NEVER: frozenset[str] = frozenset()
_HARD_CONCRETE_ONLY = (
    "'temperature' and 'stretch' are the constants of the hard_concrete "
    "relaxation (§2.5) — this gate maps theta through {maps}, which samples "
    "nothing"
)
_BUDGET_ONLY = (
    "'k_schedule' and 'stop_grad_shift' belong to the budget parametrization "
    "(§2.5) — this gate maps theta through {maps}, which draws no budget"
)

#: §2.5's conditional legality, one row per authorable featurizer field
#: (every field of :data:`FEATURIZER_FIELDS`), in the order the spec's tables
#: and the method pages list them. "Fit only", "with ``file_path`` only" and
#: "under ``hard_concrete`` only" used to be ``if`` branches in the parser
#: and prose in the attribute docs; they are this table, which the parser
#: applies (:func:`_parse_featurizer`) and the generator renders.
FEATURIZER_FIELD_CONDITIONS: dict[str, FieldLegality] = {
    "k": FieldLegality(None, None),
    "parametrization": FieldLegality(None, None),
    "group": FieldLegality(None, None),
    "axis": FieldLegality(None, None),
    "init": FieldLegality(
        None,
        _NEVER,
        why_loaded=(
            "'init' sets the *starting point* of a fit; a featurizer loaded "
            "from a file_path has its weights in the file — it draws nothing "
            "and trains nothing, so there is no start to set"
        ),
    ),
    "seed": FieldLegality(
        None,
        _NEVER,
        why_loaded=(
            "'seed' picks an *initial* rotation; a featurizer loaded from a "
            "file_path has its weights in the file and draws nothing"
        ),
    ),
    "temperature": FieldLegality(
        _HARD_CONCRETE, _HARD_CONCRETE, why_map=_HARD_CONCRETE_ONLY
    ),
    "stretch": FieldLegality(
        _HARD_CONCRETE, _HARD_CONCRETE, why_map=_HARD_CONCRETE_ONLY
    ),
    "dead": FieldLegality(
        None,
        _NEVER,
        why_loaded=(
            "'dead' is a rule for units that go hard-off *during a fit*; a gate "
            "loaded from a file_path takes no step, so there is nothing for it "
            "to act on (§2.5)"
        ),
    ),
    "top_k": FieldLegality(
        _NEVER,
        None,
        why_fit=(
            "'top_k' reads a *loaded* gate out as its k largest units — it needs "
            "a file_path; a gate being fitted is read through its map's own "
            "split, so the loss and the eval agree on which units are on (§2.5)"
        ),
    ),
    "k_schedule": FieldLegality(
        _BUDGET,
        _NEVER,
        why_loaded=(
            "'k_schedule' draws a *training* budget; a gate loaded from a "
            "file_path trains nothing and is read out at 'top_k' (§2.5)"
        ),
        why_map=_BUDGET_ONLY,
    ),
    "stop_grad_shift": FieldLegality(
        _BUDGET,
        _NEVER,
        why_loaded=(
            "'stop_grad_shift' is the budget's training-time ablation — how the "
            "solved shift enters the mask's gradient; a gate loaded from a "
            "file_path solves no shift and is read out at 'top_k' (§2.5)"
        ),
        why_map=_BUDGET_ONLY,
    ),
    "pool": FieldLegality(
        _BUDGET,
        None,
        why_map=(
            "'pool' on a gate being fitted shares one budget, which only the "
            "budget parametrization draws (§2.5) — this gate maps theta through "
            "{maps}; a pooled readout needs a file_path and a top_k"
        ),
    ),
}


def _legality_note(legality: FieldLegality) -> str:
    """The parenthetical the kinds table hangs on a conditionally legal field:
    where it is legal, in the spec's words, or ``""`` when everywhere."""
    every = frozenset(GATE_MAPS)

    def maps(legal: frozenset[str]) -> str:
        return " \\| ".join(f"`{m}`" for m in GATE_PARAMETRIZATIONS if m in legal)

    def state(legal: frozenset[str] | None) -> str:
        if legal is None or legal == every:
            return "any"
        return "none" if not legal else maps(legal)

    fit, loaded = state(legality.fit), state(legality.loaded)
    if fit == "any" and loaded == "any":
        return ""
    if loaded == "none":
        return "on a fit" if fit == "any" else f"under {fit}, on a fit"
    if fit == "none":
        return (
            "with `file_path`"
            if loaded == "any"
            else f"under {loaded}, with `file_path`"
        )
    if fit == loaded:
        return f"under {fit}"
    fit_part = "any map" if fit == "any" else fit
    loaded_part = "any map" if loaded == "any" else loaded
    return f"under {fit_part} to fit; {loaded_part} with `file_path`"


def _field_name_cell(kind: str, field: str) -> str:
    """One authored field's name and, when it has one, its closed vocabulary."""
    domain: tuple[str, ...] | None = None
    if field == "parametrization":
        domain = GATE_PARAMETRIZATIONS if kind == "gate" else PARAMETRIZATIONS
    elif field == "group":
        domain = GATE_GROUPS
    elif field == "axis":
        domain = GATE_AXES
    elif field == "dead":
        domain = GATE_DEAD_RULES
    cell = f"`{field}`"
    if domain is not None:
        cell += " ∈ " + " \\| ".join(f"`{v}`" for v in domain)
    return cell


def _field_cell(kind: str, field: str) -> str:
    """One authored field as the kinds table spells it: its name, its closed
    vocabulary when it has one, and where it is legal when that is not
    everywhere."""
    cell = _field_name_cell(kind, field)
    note = _legality_note(FEATURIZER_FIELD_CONDITIONS[field])
    return f"{cell} ({note})" if note else cell


def render_featurizer_kind_table() -> str:
    """Spec §2.5's kinds table, one row per :data:`FEATURIZER_KINDS`: the
    family's ``featurize`` map, the slots :data:`FEATURIZER_SLOTS` declares
    and the fields :data:`FEATURIZER_FIELDS` lets the kind author, each with
    its vocabulary and its legality note. ``file_path``, ``entry``, ``dtype``
    and ``description`` are legal on every kind and are not repeated per row."""
    lines = [
        "| kind | featurize | param slots | authored fields |",
        "|---|---|---|---|",
    ]
    for kind in FEATURIZER_KINDS:
        family = FEATURIZER_FAMILIES[kind]
        name = f"`{kind}` (default)" if kind == "identity" else f"`{kind}`"
        slots = ", ".join(f"`{s}`" for s in FEATURIZER_SLOTS[kind]) or "—"
        fields = ", ".join(
            _field_cell(kind, f)
            for f in FEATURIZER_FIELD_CONDITIONS
            if f in FEATURIZER_FIELDS[kind]
        )
        lines.append(f"| {name} | {family.featurize} | {slots} | {fields or '—'} |")
    return "\n".join(lines) + "\n"


def render_gate_map_table() -> str:
    """Spec §2.5's parametrization table, one row per :data:`GATE_MAPS` entry,
    every cell a field of the record — the penalty and anneal columns spelled
    from the facts the validator reads (rule 4)."""
    lines = [
        "| parametrization | soft mask (train) | after every optimizer step "
        "| hard mask (eval, apply) | mask penalty (`train.objective`) "
        "| `anneal` on `theta.temperature` | default start |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, gate_map in GATE_MAPS.items():
        label = f"`{name}` (absent)" if name == GATE_DEFAULT_MAP else f"`{name}`"
        if gate_map.penalty is None:
            others = " and ".join(f"`{k}`" for k in ("l1", "l0"))
            penalty = (
                f"none — {others} are **refused** (rule 4): the mask's sum *is* "
                "the budget"
            )
        else:
            other = "l0" if gate_map.penalty == "l1" else "l1"
            penalty = (
                f"**`{gate_map.penalty}`** = {gate_map.penalty_formula}; "
                f"`{other}` is **refused** (rule 4)"
            )
        anneal = (
            "legal"
            if gate_map.anneals_temperature
            else f"**refused** (rule 4): {gate_map.no_temperature_because}"
        )
        cells = (
            label,
            gate_map.soft_mask,
            gate_map.post_step,
            gate_map.hard_mask,
            penalty,
            anneal,
            gate_map.default_start,
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _kind_fields(kind: str) -> list[str]:
    """One kind's authorable fields, in the legality table's order."""
    if kind not in FEATURIZER_FIELDS:
        raise KeyError(f"{kind!r} is not a featurizer kind: {FEATURIZER_KINDS}")
    return [f for f in FEATURIZER_FIELD_CONDITIONS if f in FEATURIZER_FIELDS[kind]]


def _legality_cell(legal: frozenset[str] | None, *, has_maps: bool) -> str:
    if legal is None:
        return "any map" if has_maps else "legal"
    if not legal:
        return "**refused**"
    return ", ".join(f"`{m}`" for m in GATE_PARAMETRIZATIONS if m in legal)


def render_field_legality_table(kind: str) -> str:
    """One kind's fields against the two states a featurizer is in — authored
    without a ``file_path`` (fitted, or untrained) and with one (loaded) —
    and, where the kind has maps, the maps each is legal under: the rows of
    :data:`FEATURIZER_FIELD_CONDITIONS` for :data:`FEATURIZER_FIELDS`
    ``[kind]``, as the method page under ``docs/methods/`` carries them."""
    fields = _kind_fields(kind)
    if not fields:
        return (
            f"A `{kind}` authors no field of its own. `file_path`, `entry`, "
            "`dtype` and `description` are legal on every kind (§2.5).\n"
        )
    has_maps = kind == "gate"
    lines = ["| field | without `file_path` | with `file_path` |", "|---|---|---|"]
    for field in fields:
        legality = FEATURIZER_FIELD_CONDITIONS[field]
        lines.append(
            f"| {_field_name_cell(kind, field)} "
            f"| {_legality_cell(legality.fit, has_maps=has_maps)} "
            f"| {_legality_cell(legality.loaded, has_maps=has_maps)} |"
        )
    return "\n".join(lines) + "\n"


def render_field_refusals(kind: str) -> str:
    """What a document authoring one kind is refused for, in the refusals'
    own words: every ``why`` text of the kind's legality rows (the parser
    prints these), then the checklist rules tagged with the kind
    (``Rule.kinds``). A kind nothing is specific to says so."""
    bullets: list[str] = []
    for field in _kind_fields(kind):
        legality = FEATURIZER_FIELD_CONDITIONS[field]
        if legality.why_fit:
            bullets.append(f"- `{field}` without `file_path` — {legality.why_fit}")
        if legality.why_loaded:
            bullets.append(f"- `{field}` with `file_path` — {legality.why_loaded}")
        if legality.why_map:
            bullets.append(
                f"- `{field}` under another map — "
                + legality.why_map.format(maps="the authored map")
            )
    for rule in RULES.values():
        if kind in rule.kinds:
            bullets.append(
                f"- [{rule.code}] `{rule.slug}` — {rule.title} (§5.{rule.number})"
            )
    if not bullets:
        bullets.append(
            f"- nothing beyond the checklist every document meets (§5): a `{kind}` "
            "has no field that is legal in one state and refused in another"
        )
    return "\n".join(bullets) + "\n"


#: §2.5 the mapping form of a gate's ``parametrization``: ``{"forward":
#: "hard", "backward": <map>}`` — the forward pass uses the map's training
#: mask **thresholded at ½** (0/1) while the backward pass sees the map's own
#: gradient, the straight-through idiom ``hard + (soft − soft.detach())``;
#: the hard-forward ablation. The map is any of
#: :data:`GATE_PARAMETRIZATIONS`. ``forward`` has the one value ``hard``: a
#: soft forward is the string form, and the sampled forward is
#: ``hard_concrete`` itself — either would be a second spelling of one
#: computation, digesting apart, so both are refused.
FORWARD_MASKS: tuple[str, ...] = ("hard",)

#: The kinds a ``budget`` gate's ``k_schedule`` draws its per-step budget by
#: (§2.5): ``fixed`` (``k`` every step — the plain sigmoid-top-k mask),
#: ``uniform`` over the integers ``[low, high]``, ``log_uniform`` over them
#: (``low ≥ 1``: a budget curriculum that spends as many
#: steps between 1 and 2 units as between 24 and 48).
K_SCHEDULE_KINDS: tuple[str, ...] = ("fixed", "uniform", "log_uniform")

#: What a ``k_schedule``'s numbers count (§2.5 ``k_schedule.of``): ``patched``
#: — units that take the counterfactual, the gate's own count and the default
#: when absent (no canonical-form change) — or ``kept``, the units left clean.
#: A log-uniform draw on one is not log-uniform on the other, and a schedule
#: written ``k ~ LogUniform{1, N − 1}`` over kept units is a different draw;
#: every reader inside the gate works in patched units, so a ``kept`` schedule
#: is complemented exactly once, at
#: the gate (``eval`` included; ``top_k`` is not a schedule number and always
#: counts patched units).
K_SCHEDULE_OF: tuple[str, ...] = ("patched", "kept")

#: The hard-concrete constants when unauthored (§2.5): Louizos et al.'s
#: ``β = 2/3``, ``(γ, ζ) = (−0.1, 1.1)`` — NeuroSurgeon's defaults verbatim.
#: Not materialized into the canonical form (the ``group`` precedent: no
#: spelling of the default), so an authored default and an absent one are
#: two spellings that digest apart — write neither, or both.
HARD_CONCRETE_TEMPERATURE: float = 2.0 / 3.0
HARD_CONCRETE_STRETCH: tuple[float, float] = (-0.1, 1.1)


def hard_concrete_theta(mask: float, stretch: tuple[float, float]) -> float:
    """The θ whose **deterministic** hard-concrete mask is ``mask`` (§2.5): the
    inverse of ``clip(σ(θ)·(ζ − γ) + γ, 0, 1)`` strictly inside the poles,
    ``logit((mask − γ) / (ζ − γ)) = log(mask − γ) − log(ζ − mask)``. It is
    both what ``init.fill p`` starts a unit at and, at ``mask = ½``, the
    eval-mode threshold (:func:`hard_concrete_threshold`) — one map, so a fill
    of ½ starts exactly on the split.

    The balanced case — ``mask`` at the midpoint of the stretch, which is ½ at
    every symmetric stretch, the default included — is answered with an exact
    ``0.0`` rather than derived: in binary floating point ``1.1 − (−0.1)`` is
    not ``1.2`` and the derived value is about ``−4e−16``, which would count a
    θ of exactly ``0`` — every unit at the default start — as kept, where the
    sigmoid gate's ``θ > 0`` does not."""
    lo, hi = float(stretch[0]), float(stretch[1])
    if not lo < mask < hi:
        raise ValueError(f"a mask value of {mask} is outside the stretch ({lo}, {hi})")
    below, above = mask - lo, hi - mask
    if math.isclose(below, above, rel_tol=1e-9, abs_tol=0.0):
        return 0.0
    return math.log(below) - math.log(above)


def hard_concrete_threshold(stretch: tuple[float, float]) -> float:
    """The θ above which a hard-concrete gate keeps a unit in eval mode (§2.5):
    where the stretched σ(θ) crosses ½, ``logit((½ − γ) / (ζ − γ))`` — exactly
    ``0`` at a symmetric stretch (:func:`hard_concrete_theta`). One function
    for the two readers of a fitted bundle — the gate's own hard mask
    (``Gate.hard_threshold``) and the size-matched control
    (``analysis.random_mask``) — so the two counts agree bit for bit."""
    return hard_concrete_theta(0.5, stretch)


#: §2.11 ``train.control`` — the closed-loop counterpart of ``anneal``. A
#: closed vocabulary of controllers, of the fit signals one may observe, and
#: of the spaces it may move the controlled value in; and the defaults the
#: canonical form materializes so two spellings of one controller digest
#: identically (the ``train.optimizer`` treatment).
CONTROL_KINDS: tuple[str, ...] = ("pid",)
#: ``hard_mask_size``: a trained gate's kept-unit count through its hard mask
#: — the number ``fit_diagnostics.json`` reports under the same name.
#: ``hard_mask_fraction``: the same count divided by the number of units the
#: named gates have — kept units over all units — so a setpoint ramp
#: ``[1, 0, frac]`` and one set of gains mean the same thing over 8, 48 or
#: 2048 units, where a count-valued signal needs its ``ki`` rescaled per unit
#: count (the integral term is ``ki · (kept − setpoint)`` in the signal's units).
CONTROL_SIGNALS: tuple[str, ...] = ("hard_mask_size", "hard_mask_fraction")
CONTROL_SPACES: tuple[str, ...] = ("log", "linear")
CONTROL_DEFAULTS: dict[str, Any] = {
    "kd": 0.0,
    "space": "log",
    "bounds": [1e-8, 1e8],
    "d_clip": 5.0,
}
#: The prefix a control target uses to name an objective term's weight:
#: ``train.objective.<name>.weight`` — the same address a sweep uses (§3).
OBJECTIVE_WEIGHT_PREFIX = "train.objective."

#: §2.11 ``train.anneal`` — the shapes an open-loop schedule may take between
#: its endpoints. ``linear`` is the list spelling ``[start, end, frac]`` and
#: the default; ``geometric`` multiplies by a constant factor per step
#: (continuous sparsification's ``T ← T · r``, Savarese et al. 2020), so it
#: needs endpoints of one sign and neither zero.
ANNEAL_SHAPES: tuple[str, ...] = ("linear", "geometric")


@dataclasses.dataclass(frozen=True)
class AnnealSchedule:
    """One §2.11 ``anneal`` entry, parsed: the value walks from ``start`` to
    ``end`` over the first ``frac`` of the run and holds, along ``shape``.
    Two spellings — ``[start, end, frac]`` and ``{"from", "to", "frac",
    "shape"}`` — land here alike, and the canonical form writes the list
    whenever the shape is linear (:mod:`causalab.protocol.canonical`), so the
    longer spelling of a linear schedule digests as the shorter one."""

    start: float
    end: float
    frac: float
    shape: str = "linear"

    def value_at(self, step: int, total_steps: int) -> float:
        """The scheduled value before update ``step`` of ``total_steps``:
        ``start`` at step 0, ``end`` from ``frac · total_steps`` on."""
        ramp_steps = max(1, int(self.frac * total_steps))
        progress = min(1.0, step / ramp_steps)
        if self.shape == "geometric":
            return self.start * (self.end / self.start) ** progress
        return self.start + (self.end - self.start) * progress


#: §2.11 ``train.phases[i].until`` — how a phase's end is counted: a fraction
#: of the run's updates, or an absolute update count. One form per document.
PHASE_UNTIL_UNITS: tuple[str, ...] = ("frac", "updates")


@dataclasses.dataclass(frozen=True)
class PhaseSpec:
    """One §2.11 ``train.phases`` entry: a window of the run ending at
    ``until`` (``{"frac": f}`` or ``{"updates": n}``) inside which only
    ``params`` (a subset of ``train.params``) receive gradients, the phase's
    own ``anneal`` schedules run over the phase's steps, ``freeze_masks``
    names gates whose *hard* mask is snapshotted at the phase's start and used
    by every forward inside it, and ``optimizer`` may override ``lr`` /
    ``weight_decay`` for the phase's params. Everything not spelled is
    inherited from the top-level ``train``."""

    until: Mapping[str, int | float]
    params: tuple[str, ...]
    optimizer: Mapping[str, Any] | None = None
    anneal: Mapping[str, AnnealSchedule] | None = None
    freeze_masks: tuple[str, ...] = ()


#: The units a ``gate`` may share one parameter across (§2.5 ``group``). A
#: closed set: each value names a derivation of the coordinate→group map from
#: the component's shape, and an unknown one has no map to derive. ``head`` is
#: one group per head on a head-major component (``attention_premix``,
#: ``delta_premix``), so the fitted mask selects heads rather than coordinates.
#: ``expert_neuron`` is one parameter per ``(expert, neuron)`` of the routed
#: interior (``expert_activation`` or ``expert_neuron_output``). A token's slots
#: look their parameters up through ``expert_idx``, so the mask is stable
#: across tokens that route differently. ``site`` is one parameter for the whole
#: site. Its map is ``(1, width)``, so an MLP block or the embedding is one unit, the node a
#: circuit-discovery benchmark (MIB) scores it as; it is the ``head`` map with a
#: single group and shares its code path.
GATE_GROUPS: tuple[str, ...] = ("head", "expert_neuron", "site")

#: §2.5 ``axis`` on a gate: which axis of the addressed window θ runs over.
#: Absent, the gate is the feature gate it has always been — one θ per
#: coordinate (or per ``group``) — and, as with ``group``, the default has no
#: literal spelling. ``position`` gives one θ per addressed **token
#: position**, applied to every coordinate of that position: a position
#: mask, and in a chain before a feature gate the outer product
#: ``m_t · m_j``. A position gate's every use addresses a fixed ``span``
#: window (its width is the window's length, known at load), and it takes no
#: ``group`` (it is already one scalar per position over the whole width —
#: what ``group: site`` would say) and no ``pool``.
GATE_AXES: tuple[str, ...] = ("position",)

#: The axis of the site's declared shape each group groups over (§2.5's
#: ``group`` table, third column; §5.23): ``head`` needs a head axis — a
#: head-major component — and ``expert_neuron`` the routed-expert slot axis
#: (``topk``) on ``expert_activation`` or ``expert_neuron_output``. ``site``
#: uses the feature axis itself. Every site a featurizer may attach to has
#: one, so the map always exists and is one group wide.
#: Spelled in :mod:`causalab.protocol.shapes`' ``AxisKind``
#: vocabulary, kept as plain strings here because ``schema`` sits below
#: ``shapes`` in the import order; the census guard asserts the two agree and
#: that every group has a row, so a group value with no axis behind it fails
#: CI rather than resolving to "no grouping".
GATE_GROUP_AXES: dict[str, str] = {
    "head": "head",
    "expert_neuron": "topk",
    "site": "feature",
}

Mechanism = Literal[
    "swap",
    "add_scaled",
    "lerp",
    "affine",
    "gaussian",
    "renormalize",
    "clamp",
    "pytorch_fn",
]
#: The closed ``do`` mechanism set (§2.8).
MECHANISMS: tuple[Mechanism, ...] = get_args(Mechanism)

#: Mechanisms whose write is a delta added after the absolute write (§2.8).
ADDITIVE_MECHANISMS: frozenset[str] = frozenset({"add_scaled", "gaussian"})

MetricKind = Literal[
    "logit_diff",
    "soft_accuracy",
    "token_logit",
    "cross_entropy",
    "kl",
    "js",
    "class_probs",
    "token_logits",
    "top_k",
    "match",
    "decode",
]
METRIC_KINDS: tuple[MetricKind, ...] = get_args(MetricKind)

#: Value fields per metric kind beyond ``of`` (§2.10). ``kl.target`` and
#: ``js.target`` name a read (:data:`READ_TARGET_METRIC_KINDS`); every other
#: value field names a dataset column (checked at run time by
#: ``validate --data``, §2.2) — except the fields in
#: :data:`NON_COLUMN_METRIC_FIELDS`. :func:`metric_column_fields` is the one
#: function that applies those exceptions, so the loader, the eligibility
#: predicate and the run cannot disagree about which fields are columns.
METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "logit_diff": ("a", "b"),
    "soft_accuracy": ("a", "b"),
    "token_logit": ("token",),
    "cross_entropy": ("target",),
    "kl": ("target",),
    "js": ("target",),
    "class_probs": ("groups",),
    "token_logits": ("tokens",),
    "top_k": ("k", "by"),
    "match": ("expected",),
    "decode": (),
}

#: Mandatory value fields that are *not* dataset column names: ``top_k.k`` is
#: an integer, ``top_k.by`` is a closed enum, and ``token_logits.tokens`` is a
#: list of literal token strings — an answer space is a property of the run,
#: exactly as ``class_probs.groups`` is (a mapping, which the column check
#: already skips by shape). Everything else in :data:`METRIC_FIELDS` (bar
#: ``kl.target``, a read) is checked against the resolved datasets' columns.
NON_COLUMN_METRIC_FIELDS: frozenset[str] = frozenset({"k", "by", "tokens"})

#: Metric kinds whose ``target`` is a **read**, not a column (§2.10): ``kl``
#: and ``js`` compare two reads' distributions against each other. The
#: planner materializes the target read for them, the executor hands its
#: value to the reduction, and the column checks skip the field.
READ_TARGET_METRIC_KINDS: frozenset[str] = frozenset({"kl", "js"})

#: What each metric kind consumes from its read (§2.10). ``distribution``
#: kinds reduce the read's dense value at the addressed positions (the
#: vocabulary projection for an ``lm_head`` read, which is every kind but
#: ``top_k``); ``ids`` kinds consume only the tokens the decode produced. The
#: split is what lets the planner (§8) tell a text probe — which obliges no
#: vocabulary projection at all — from a scoring one, so it is a property of
#: the kind, never of the document.
MetricDomain = Literal["distribution", "ids"]
METRIC_DOMAINS: dict[str, MetricDomain] = {
    "logit_diff": "distribution",
    "soft_accuracy": "distribution",
    "token_logit": "distribution",
    "cross_entropy": "distribution",
    "kl": "distribution",
    "js": "distribution",
    "class_probs": "distribution",
    "token_logits": "distribution",
    "top_k": "distribution",
    "match": "distribution",
    "decode": "ids",
}

#: Metric kinds that reduce the whole addressed window to one value per
#: example rather than one per position: ``decode`` joins its tokens into a
#: string, so a per-step row would be a per-character-ish lie.
WHOLE_WINDOW_METRIC_KINDS: frozenset[str] = frozenset({"decode"})


TokenForm = Literal["auto", "bare", "space_prefixed", "id"]

#: How a metric's string answers become token ids (§2.10). ``auto`` is the
#: historical resolver — try ``" " + s`` first, fall back to ``s`` — which is
#: right for answers that follow a space (weekdays, names, MCQA letters) and
#: wrong for answers that do not (punctuation: gpt2 emits ``"?"`` = 30, but
#: ``" ?"`` = 5633 is also one token and wins). ``bare`` and ``space_prefixed``
#: pin one form, so a document can say which one it means.
TOKEN_FORMS: tuple[TokenForm, ...] = get_args(TokenForm)

#: How one position maps across a pair's inputs (§2.3): the cardinality of
#: the token runs the same address resolves to on the base input and on a
#: counterfactual input. Authored as the optional ``alignment`` key of a
#: position entry — the member the author *declares* — and derived at encode
#: time as the *observed* one by ``causalab.protocol.alignment.alignment_of``,
#: the one function planning, execution and metrics all call. An ``index`` and
#: an unscoped ``span`` are ``one_to_one`` by construction (one token, or one
#: joint span of the same width, on every input — two ``index`` specs are two
#: locations, one ``span`` is one joint address); a ``variable`` or ``column``
#: window is whatever the tokenizer says. ``absent`` and ``ambiguous`` are the
#: two unalignable values, and they are §2.4's ``alignment_missing`` and
#: ``alignment_ambiguous`` reason codes. Optional with **no default**: an
#: unauthored key stays absent through the canonical form, so no digest moves.
AlignmentCardinality = Literal[
    "one_to_one", "one_to_many", "many_to_one", "absent", "ambiguous"
]
ALIGNMENT_CARDINALITIES: tuple[AlignmentCardinality, ...] = get_args(
    AlignmentCardinality
)

#: Metric kinds whose value fields carry string answers that must resolve to
#: token ids — the kinds ``token_form`` applies to. ``kl`` compares two reads'
#: distributions and ``top_k`` reports indices it found (decoding them only
#: when the read happens to tap ``lm_head``), so neither resolves an authored
#: string and neither accepts the key.
TOKEN_COLUMN_METRIC_KINDS: frozenset[str] = frozenset(
    {
        "logit_diff",
        "soft_accuracy",
        "token_logit",
        "cross_entropy",
        "class_probs",
        "token_logits",
        "match",
    }
)

#: Optional value fields per metric kind (§2.10). An omitted optional field
#: **with a default** (:data:`METRIC_FIELD_DEFAULTS`) is materialized to it in
#: the canonical form (§7), so an authored default and an omitted one digest
#: identically — the same treatment ``train.optimizer`` defaults get. An
#: optional field with **no** default (``js.restrict``) stays absent when
#: unauthored, like ``minimum_count``: absent means "unrestricted", and an
#: absent field keeps the digest.
OPTIONAL_METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "match": ("mode",),
    "js": ("restrict",),
}

#: Defaults for the optional fields above, by ``(kind, field)``.
METRIC_FIELD_DEFAULTS: dict[tuple[str, str], Any] = {
    ("match", "mode"): "exact",
}

#: The optional decision threshold on any metric kind (§2.10 "Eligibility"):
#: the fewest eligible rows the metric's decision rule needs. Not in
#: :data:`OPTIONAL_METRIC_FIELDS` because it has no default to materialize —
#: absent means "no threshold", and an absent field keeps the digest (§7).
#: Checked by ``validate --data`` against the resolved base table's maximum
#: eligible count (``loader.check_data_columns``, rule 4).
MINIMUM_COUNT_FIELD = "minimum_count"

#: The optional ragged-window policy of a write (§2.8, §5 rule 19): how a
#: write whose rows address different numbers of positions (an ``all``,
#: ``variable``, ``column`` or span window) lands. Spelled as a one-key
#: object, ``{"policy": <one of RAGGED_POLICIES>}``, so a later knob has a
#: place beside the policy. Like :data:`MINIMUM_COUNT_FIELD` it has no
#: default to materialize — absent means ``refuse`` (rule 19's refusal, the
#: behaviour every document had before the field existed), and an absent
#: field keeps the digest (§7). Vocabulary-checked at parse; resolved by the
#: executor on the encoded batch, where the tokenizer is (the encode-time
#: boundary).
RAGGED_FIELD = "ragged"

#: The closed policy set (§2.8): ``refuse`` — rule 19 as before;
#: ``exact_length_buckets`` — the rows land grouped by width, one gather per
#: width; ``padded_masked`` — the rows land through one padded gather and a
#: mask, padding never written. Both non-refusing policies land every row at
#: its own width before any forward and change no batch geometry.
RAGGED_POLICIES: tuple[str, ...] = ("refuse", "exact_length_buckets", "padded_masked")

#: How ``match`` compares the argmax token to an expected form (§2.10):
#: ``exact`` needs the form to be one token; ``first_token`` credits the
#: form's first token, which is what "prefix" means with logits at one
#: position (a multi-token answer's first piece).
MATCH_MODES: tuple[str, ...] = ("exact", "first_token")

#: How ``top_k`` ranks the entries of its read's last axis (§2.10). Mandatory,
#: because the right answer depends on what the axis *is* and only the author
#: knows: a vocabulary projection has no meaningful negative entries, while a
#: residual stream and a signed feature code do — ranking a 100k-latent SAE
#: code by signed value and by magnitude give different top-k sets, and
#: silently picking one for the author is how a plot ends up meaning something
#: other than its caption.
#:
#: * ``value`` — the k largest signed entries. Any read.
#: * ``abs_value`` — the k largest by ``|x|``; the reported value stays signed.
#:   Any read.
#: * ``prob`` — softmax the last axis, then take the k largest probabilities.
#:   Legal **only** on an ``lm_head`` read: a softmax across neurons or SAE
#:   latents normalizes over an axis that is not an event space, and the
#:   resulting "probabilities" would mean nothing (validation refuses it).
TOP_K_RANKINGS: tuple[str, ...] = ("value", "abs_value", "prob")

#: The ``top_k.by`` ranking that normalizes, and so is vocabulary-only.
VOCAB_TOP_K_RANKING: str = "prob"

#: The bare-string spelling of an all-positions spec (§2.3 sugar). Reserved as
#: a name so a ``positions`` entry can never shadow the sugar.
ALL_POSITIONS: str = "all"

#: Names no section may declare (§1): the input roles, the un-intervened
#: model, the indexed-counterfactual family (checked by prefix for
#: ``counterfactual[``), and the all-positions sugar.
RESERVED_NAMES: frozenset[str] = frozenset(
    {"base", "counterfactual", "original", ALL_POSITIONS}
)

#: The one value ``header.protocol_version`` may hold (§1). A string, as v1's
#: ``version`` was: the loader compares it, never orders it, and a JSON
#: integer would make ``"3"`` versus ``3`` a refusal of its own.
PROTOCOL_VERSION: str = "3"

#: Earlier ``protocol_version`` values ``causalab migrate`` rewrites (§7, §9):
#: a document declaring one is refused by name and told the verb. ``"2"`` is
#: the four-group form whose sites spelled a scalar ``layer``; version 3
#: renamed the field to ``layers``, a band of layer indices (§2.4).
MIGRATABLE_PROTOCOL_VERSIONS: tuple[str, ...] = ("2",)

#: The four groups of an intervention specification, in recommended order
#: (§1): the header names the file, ``model`` and ``data`` name what the
#: experiment ran on, and ``method`` is the experiment itself.
GROUP_ORDER: tuple[str, ...] = ("header", "model", "data", "method")

#: What the ``header`` group may hold (§1). ``title`` and ``description`` are
#: authoring metadata — they say what a file is *for* — and canonicalization
#: drops them; ``protocol_version`` is content, and stays.
HEADER_FIELDS: tuple[str, ...] = ("protocol_version", "title", "description")

#: The twelve sections of the ``method`` group, in recommended order (§1).
METHOD_SECTIONS: tuple[str, ...] = (
    "segments",
    "positions",
    "sites",
    "featurizers",
    "params",
    "code",
    "reads",
    "writes",
    "intervened_models",
    "metrics",
    "train",
    "save",
)

#: The ``save`` entry kinds that are not a saved read / metric / featurizer
#: (§2.12): ``location_ledger`` — the run's resolved token indices as a
#: table (``protocol/ledger.py``); ``trajectory`` — a fit's checkpoints as a
#: bundle; ``rank`` — every gate's units ordered by ``theta``, as a table
#: (``neural/shared/execution.py``), the object a top-k readout
#: (:attr:`FeaturizerSpec.top_k`) reads off and the record that makes a
#: ranking method's result inspectable without the bundle. Opt-in: an entry
#: of a kind is the only thing that makes a run write one.
SAVE_KINDS: tuple[str, ...] = ("location_ledger", "trajectory", "rank")

#: How a ``trajectory`` entry (§2.12) spaces its checkpoints: ``count`` — n
#: checkpoints equally spaced over the run, the last at its end — or every n
#: ``updates`` / ``epochs`` (the ``_parse_counter`` units), the last update
#: always included.
TRAJECTORY_EVERY_UNITS: tuple[str, ...] = ("count", "updates", "epochs")

REQUIRED_METHOD_SECTIONS: frozenset[str] = frozenset({"sites", "reads", "save"})

#: Every addressable section, in recommended order: the two input sections and
#: the method's ten. A section name is unique across the groups, so a dotted
#: path (``--set``, a sweep axis id, a workflow's ``emit``) starts at the
#: section and never spells the group — :func:`tree_path` finds it (§1).
SECTION_ORDER: tuple[str, ...] = ("model", "data", *METHOD_SECTIONS)

REQUIRED_SECTIONS: frozenset[str] = (
    frozenset({"model", "data"}) | REQUIRED_METHOD_SECTIONS
)


def tree_path(dotted: str) -> tuple[str, ...]:
    """A section-rooted dotted path as a path into the document tree (§1):
    ``sites.target.layers`` → ``("method", "sites", "target", "layers")``,
    ``model.dtype`` → ``("model", "dtype")``. A path that starts at a group or
    at nothing known is returned as written, so the caller's "does not exist"
    refusal names what was typed. An index on the first segment
    (``save[0].file_path`` — ``save`` is the method's one list) is part of
    that segment, not of the section's name."""
    parts = tuple(dotted.split("."))
    if parts and parts[0].split("[", 1)[0] in METHOD_SECTIONS:
        return ("method", *parts)
    return parts


def dotted_path(path: Sequence[str]) -> str:
    """The inverse of :func:`tree_path`: a tree path as its section-rooted
    dotted spelling, the group dropped."""
    parts = tuple(path)
    if (
        len(parts) >= 2
        and parts[0] == "method"
        and parts[1].split("[", 1)[0] in METHOD_SECTIONS
    ):
        parts = parts[1:]
    return ".".join(parts)


#: The name-bearing sections sharing one global namespace (§1: method sections 2–10).
NAMED_SECTIONS: tuple[str, ...] = (
    "positions",
    "sites",
    "featurizers",
    "params",
    "code",
    "reads",
    "writes",
    "intervened_models",
    "metrics",
)

PRECISION_DTYPES: tuple[str, ...] = ("fp32", "bf16", "fp16")

#: The compute dtype a document runs in when it authors none (§2.1). This is
#: the value canonicalization materializes, so every canonical form names a
#: dtype and every digest covers it; ``fp32`` keeps an unauthored document
#: running exactly as it did when dtype was an execution flag.
MODEL_DTYPE_DEFAULT: str = "fp32"

#: Full-attention backends supported by the document vocabulary. Omission
#: retains the engine's default; an explicit choice is part of the experiment.
ATTENTION_IMPLEMENTATIONS: tuple[str, ...] = ("eager", "sdpa", "flash_attention_2")

#: Weight-quantization schemes (§2.1). Each names one *load-time* scheme the
#: reference engine can realize through bitsandbytes: ``int8`` is LLM.int8()
#: mixed-precision decomposition, ``nf4`` / ``fp4`` are the two 4-bit
#: quantization types. There is no bare ``int4``: bitsandbytes' 4-bit is one
#: of these two datatypes, and "int4" would not say which — the whole point
#: of putting the field in the record is that it names one realization.
#: Weights quantized *ahead of time* (GPTQ, AWQ) are a property of the
#: checkpoint, so they are named by ``model.key``/``revision``, not here.
QUANT_SCHEMES: tuple[str, ...] = ("int8", "nf4", "fp4")

#: Quantizers (§2.1). One entry in v1 — the library the reference engine
#: calls; naming it keeps a document honest when a second one appears.
QUANT_METHODS: tuple[str, ...] = ("bitsandbytes",)

#: Quantization fields that only make sense for some schemes (rule 17).
_QUANT_4BIT_FIELDS: tuple[str, ...] = ("double_quant",)
_QUANT_INT8_FIELDS: tuple[str, ...] = ("int8_threshold",)

#: Optimizer field vocabulary and per-name defaults, materialized into the
#: canonical form (§7: "every default (constant LR, optimizer betas, dtypes)").
#: The vocabulary is closed like every other enum here; extending it is a
#: schema change, not a free-form pass-through.
OPTIMIZER_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "momentum",
        "clip_grad_norm",
        "schedule",
        "warmup_frac",
    }
)
#: The learning-rate schedules an optimizer may follow (§2.11 ``optimizer.schedule``):
#: ``constant`` — the authored ``lr`` at every update, the default and the only
#: value the field had before 2026-09-10 (it was parsed and ignored); and
#: ``linear_warmup_decay`` — HF's ``get_linear_schedule_with_warmup``: the lr
#: climbs linearly from 0 over the first ``warmup_frac`` of the updates (0.1
#: unless authored) and decays linearly to 0 at the last, the schedule pyvene's
#: sigmoid-mask recipe (DBM) trains under. ``warmup_frac`` is legal only with it,
#: and enters the canonical form only when authored.
OPTIMIZER_SCHEDULES: tuple[str, ...] = ("constant", "linear_warmup_decay")

OPTIMIZER_DEFAULTS: dict[str, dict[str, Any]] = {
    # torch.optim.AdamW defaults (torch 2.x): betas=(0.9, 0.999), eps=1e-8,
    # weight_decay=1e-2 — but the protocol default is 0.0: a regularizer is an
    # objective term here (§2.11), never an optimizer side-effect.
    "adamw": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "schedule": "constant",
    },
    "adam": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "schedule": "constant",
    },
    "sgd": {"momentum": 0.0, "weight_decay": 0.0, "schedule": "constant"},
}


# --------------------------------------------------------------------------- #
# value wrappers
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Sweep:
    """An explicit sweep axis (§3): ``{"sweep": [v1, v2]}`` or
    ``{"sweep": {"range": [start, stop, step?]}}``. ``values`` holds the
    expanded value list either way (a range is expanded eagerly — it is sugar
    for the list it denotes)."""

    values: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class ArtifactRef:
    """An artifact-valued field (§1): one value read from a prior run's
    artifact at load. Unresolved in the authored object model; resolution
    (and the missing-artifact load error, §5.15) is
    :mod:`causalab.protocol.resolve`'s job."""

    artifact: str
    key: str


#: A leaf that may still be swept or artifact-valued in the authored model.
Leaf = Union[Any, Sweep, ArtifactRef]


def concrete_int(value: Leaf, what: str) -> int:
    """Narrow a leaf that must be concrete by now (a point document) to int."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ParseError("P2", f"{what} is not a concrete integer (got {value!r})")


def concrete_str(value: Leaf, what: str) -> str:
    """Narrow a leaf that must be concrete by now (a point document) to str."""
    if isinstance(value, str):
        return value
    raise ParseError("P2", f"{what} is not a concrete string (got {value!r})")


# --------------------------------------------------------------------------- #
# section records
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class QuantizationSpec:
    """§2.1 — how the weights are quantized at load.

    Quantization changes what the network computes, so it is document
    vocabulary, not an execution flag: two runs of the same protocol at
    ``nf4`` and at ``bf16`` are two experiments, and their digests say so.
    Every field that moves a number is here — the scheme, the quantizer, the
    dtype the dequantized matmuls run in, and the scheme's own knobs.
    """

    scheme: Leaf
    method: Leaf = "bitsandbytes"
    compute_dtype: Leaf | None = None
    double_quant: Leaf | None = None
    int8_threshold: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class ModelRef:
    """§2.1 — the network as a name, plus how it is realized numerically.

    ``revision`` defaults to ``"main"`` and ``dtype`` to
    :data:`MODEL_DTYPE_DEFAULT` (both materialized by canonicalization when
    unauthored, §7).
    """

    key: Leaf
    revision: Leaf = "main"
    dtype: Leaf | None = None
    quantization: QuantizationSpec | None = None
    attn_implementation: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class DataRole:
    """§2.2 — one input-row column: a dataset ref (local path or HF key, no
    digest — the content digest is stamped at load) plus the column selector
    ``field`` (``[j]`` indexes list-valued columns).

    ``shuffle`` is the first data verb: ``{"seed": <int>}`` on a **counterfactual**
    role permutes that role's rows by ``random.Random(seed)`` over their indices
    before rows are paired, so the same rows meet different base rows — the
    ``shuffled_source`` control (workflow spec §2.2). The base role is the
    population and is never permuted (refused at parse). ``None`` when
    unauthored, and in the canonical form exactly when authored (§7), so no
    unshuffled document's digest moves.

    ``draw`` is the second data verb, on a **counterfactual** role whose
    ``field`` names a list-valued column bare (no ``[j]``): ``{"kind":
    "uniform", "eval"?: j}``. A fit redraws one member per row from its own
    seeded generator at every epoch (each row is visited once per epoch, so
    that is once per update it takes part in); every other forward — the
    point's reads, ``train.eval``, an apply document — reads the fixed member
    ``eval`` (``0`` unless authored). ``None`` when unauthored, canonical only
    when authored."""

    dataset: Leaf
    field: Leaf
    shuffle: Mapping[str, Any] | None = None
    draw: Mapping[str, Any] | None = None

    @property
    def draw_column(self) -> str | None:
        """The bare list column a drawn role reads, or ``None``."""
        return str(self.field) if self.draw is not None else None

    @property
    def eval_member(self) -> int:
        """The member every non-training forward of a drawn role reads."""
        if self.draw is None:
            return 0
        return int(self.draw.get("eval", 0))

    @property
    def resolved_field(self) -> str:
        """The field a forward tokenizes out of this role's rows: the authored
        field, or ``<column>[eval]`` for a drawn role (§2.2). The one spelling
        every mirror of "the field this role reads" asks — ``resolve_roles``,
        the fit's own minibatch executors, the loader's prompt-variable check
        and the data identity — so a drawn role cannot be one thing to the
        engine and another to a checker."""
        if self.draw is None:
            return str(self.field)
        return f"{self.draw_column}[{self.eval_member}]"


@dataclasses.dataclass(frozen=True)
class PositionSpec:
    """§2.3 — a token-position spec. Exactly one of ``index`` / ``span`` /
    ``variable`` / ``column`` / ``all`` is set; ``scope`` /
    ``relative_to`` name an anchor (a prompt variable, or a dataset
    column when ``anchor_source`` is ``"column"``), only modify
    ``index``/``span``, and are mutually exclusive. ``all`` selects every
    content token of the row and takes no modifiers. Positions are never
    resolved to integers in the document — resolution is an engine
    service against a ``PositionFrame`` (§2.3, §8).

    ``generated`` is a **frame selector**, not an anchor: it says the
    anchor resolves inside the row's greedy continuation instead of its
    prompt, and it carries the decode budget (``{"max_new_tokens": n}``).
    The anchor vocabulary is unchanged inside that frame.

    ``alignment`` is the cardinality the author *declares* for how this
    address maps across the pair's inputs (``ALIGNMENT_CARDINALITIES``,
    §2.3). Optional and undefaulted — ``None`` declares nothing. Rule 26
    checks what the document alone can decide about it; the executor checks
    the rest against the tokenizer and refuses a contradiction."""

    index: Leaf | None = None
    span: Leaf | None = None
    variable: Leaf | None = None
    column: Leaf | None = None
    all: Leaf | None = None
    scope: Leaf | None = None
    relative_to: Leaf | None = None
    alignment: str | None = None
    #: The continuation frame and its decode budget (§2.3). ``None`` is
    #: the prompt frame — where every position lived before generation.
    generated: Mapping[str, Leaf] | None = None
    #: Where ``scope``/``relative_to`` resolve from: ``"variable"`` (the
    #: role's prompt variables) or ``"column"`` (a top-level row column).
    #: Not authored on its own — it comes from the anchor's spelling.
    anchor_source: str = "variable"


@dataclasses.dataclass(frozen=True)
class SiteSpec:
    """§2.4 — a named activation address: pure data, no behavior.

    ``layers`` is the **band** the site spans: a non-empty, strictly
    increasing tuple of layer indices, ``(18,)`` for the ordinary one-layer
    site. A band is one site — one read, one write, one operand — across
    every layer it names (ROME's clipped restoration window is the shape);
    the engines fan it out to one module per member
    (:func:`causalab.protocol.plan.lower_bands`). Distinct from ``at_once``
    (§3.1), which declares N one-layer sites in one point. ``None`` on the
    layer-less trunk components (:data:`LAYERLESS_COMPONENTS`).
    """

    component: Leaf
    layers: Leaf | None = None
    head: Leaf | None = None
    expert: Leaf | None = None
    stream: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class FeaturizerSpec:
    """§2.5 — a named feature-space map. Only choices are authored; widths
    and param shapes derive from (model, site). ``file_path`` loads a fitted
    artifact (its ``ArtifactIdentity`` is checked; a loaded featurizer may
    not be trained)."""

    #: The kind (:data:`FEATURIZER_KINDS`), ``identity`` when absent — the
    #: family the method page is about (:data:`FEATURIZER_FAMILIES`). Sweepable.
    kind: Leaf = "identity"
    #: ``subspace`` and ``pca``: the width of the feature space — the first
    #: ``k`` columns of the rotation or basis are the subspace an interchange
    #: acts in, and the site's other ``d − k`` directions pass through
    #: untouched. Sweepable: the rank curve every localization reports.
    k: Leaf | None = None
    #: ``subspace``: the rotation map (:data:`PARAMETRIZATIONS`); ``gate``:
    #: the theta→mask map (:data:`GATE_PARAMETRIZATIONS`, ``sigmoid`` when
    #: absent). One field because it is one idea — how the stored parameter
    #: maps to the object the fit is about — and the enum follows the kind.
    parametrization: Leaf | None = None
    #: Where the fit **starts** (§2.5). ``subspace``: ``{"file_path": …,
    #: "entry": …}`` naming a saved basis whose first ``k`` columns are the
    #: starting subspace (a PCA basis at the same site, typically). ``gate``:
    #: ``{"fill": p}``, every unit at mask value ``p`` (``θ = logit(p)`` or
    #: ``θ = p`` by parametrization), the ``file_path`` form naming a saved
    #: ``theta`` taken verbatim, or ``{"from_scores": …}`` — a per-unit score
    #: table (:func:`_parse_scores_init`) whose top ``keep`` units start on the
    #: kept pole, or whose z-scored values become ``theta`` under ``scale``.
    #: ``entry`` has the semantics of the featurizer's own ``entry``. Illegal
    #: with ``file_path`` on the featurizer itself: a loaded featurizer draws
    #: nothing and trains nothing, so it has no start to set.
    init: Mapping[str, Any] | None = None
    #: ``subspace`` only: the draw its initial rotation comes from. Absent, it
    #: is the document's seed (``train.seed``, or 0 with no fit) — so nothing
    #: about an existing document changes. Authoring it is what makes an
    #: *untrained* subspace a **random rank-k basis a document can sweep**,
    #: which is the matched-k control the localization report requires and no
    #: preset could express.
    seed: Leaf | None = None
    #: ``gate`` only: the unit one parameter covers (:data:`GATE_GROUPS`).
    #: Absent, the gate is per coordinate — the gate it has always been, with
    #: no field added to its canonical form, so no existing document's digest
    #: moves. There is no literal spelling of that default: the vocabulary is
    #: exactly ``head``, ``expert_neuron`` and ``site``. The coordinate→group map is
    #: derived from the model and the site's component and never authored (§6).
    group: Leaf | None = None
    #: ``gate`` under ``hard_concrete`` only: the concrete temperature β and
    #: the stretch ``[γ, ζ]`` of the relaxation (:data:`HARD_CONCRETE_TEMPERATURE`
    #: and :data:`HARD_CONCRETE_STRETCH` when absent). Refused under any other
    #: map — they name constants of a distribution the other maps do not
    #: sample from. ``temperature`` is sweepable like any hyperparameter;
    #: ``stretch`` deliberately is not: the eval-mode split is derived from it
    #: (:func:`hard_concrete_threshold`) and the bundle stamps it into its
    #: ArtifactIdentity, so a stretch is a constant of the relaxation the whole
    #: fit is read through, not an axis one bundle's entries may differ on.
    #: Authoring ``temperature`` beside an ``anneal`` on the same gate's
    #: temperature is refused (rule 4): the schedule's start would overwrite it
    #: before the first step.
    temperature: Leaf | None = None
    #: The stretch ``[γ, ζ]`` of the hard-concrete relaxation
    #: (:data:`HARD_CONCRETE_STRETCH` when absent), legal exactly where
    #: ``temperature`` is and, unlike it, never swept: the eval-mode split is
    #: derived from it and the bundle stamps it (see ``temperature``).
    stretch: tuple[float, float] | None = None
    #: ``gate`` only, on a *trained* gate: the dead-unit rule (§2.5
    #: :data:`GATE_DEAD_RULES`) — ``{"freeze_after": n}`` or ``{"leak": ε}``,
    #: exactly one. A training rule, so it is refused on a loaded gate at
    #: parse and on a gate outside ``train.params`` by rule 4: there is no
    #: step for it to act in. Not stamped into the bundle's ArtifactIdentity —
    #: it changes how θ moved, not how θ is read.
    dead: Mapping[str, Any] | None = None
    #: §2.5 ``axis`` (:data:`GATE_AXES`): ``"position"`` for a gate whose θ
    #: runs over the addressed token positions; ``None`` is the feature gate.
    axis: Leaf | None = None
    #: §2.5 the mapping form of ``parametrization``: ``"hard"`` when the
    #: forward pass uses the map's training mask thresholded at ½ with the
    #: map's gradient behind it (:data:`FORWARD_MASKS`); ``parametrization``
    #: then holds the map (the ``backward``). ``None`` is the plain map.
    forward: str | None = None  # never swept: the mapping form is not a leaf
    #: ``gate`` with ``file_path`` only: read the loaded ``theta`` out as its
    #: ``top_k`` largest units instead of through the map's threshold (§2.5).
    #: The threshold is one cut through a ranking; a ranking method (a budget
    #: gate, an attribution score, a magnitude order) has no threshold at all
    #: and is *only* readable this way, and a sweep over ``top_k`` is the
    #: kept-count → score curve every mask method reports. Sweepable, an
    #: integer in ``[0, units]``; ``0`` keeps nothing (the base run). Absent,
    #: the hard mask is the map's own split and no field enters the canonical
    #: form. Refused without ``file_path``: a fit's readout is decided by its
    #: map, and a top-k cut of a *training* mask would make the loss and the
    #: eval disagree about which units are on.
    top_k: Leaf | None = None
    #: ``gate`` under ``budget`` only, and required there on a fit: how each
    #: optimizer step draws its budget ``k`` (:data:`K_SCHEDULE_KINDS`) —
    #: ``{"kind": "fixed", "k": n}`` or ``{"kind": "uniform" | "log_uniform",
    #: "low": a, "high": b}`` — plus ``eval``, the cut the fit's own held-out
    #: pass and its ``hard_mask_size`` are read at (``k`` when ``fixed``;
    #: required under a sampled kind, since no single number is implied).
    #: ``k`` and ``eval`` are sweepable; the bounds are not. Refused on a
    #: loaded gate: a schedule is a training-time object, and a loaded budget
    #: gate is read out through ``top_k``.
    k_schedule: Mapping[str, Any] | None = None
    #: ``gate`` under ``budget`` only: the ``−c_k`` ablation — the solved
    #: shift enters the mask as a constant, so ``θ`` receives only the direct
    #: ``σ'`` gradient and the mask's sum is free to drift within a step.
    #: Absent, the shift carries its implicit gradient
    #: (``∂c/∂θ_i = −σ'_i / Σ σ'_j``), which keeps ``Σ m = k`` to first order
    #: under any update — the default.
    stop_grad_shift: Leaf | None = None
    #: ``gate`` under ``budget`` only, fitted or loaded: the name of the budget
    #: pool this gate shares one ``k``, one shift and one ranking with (§2.5).
    #: Every gate authoring the same name is one budget fit over the union of
    #: their units — heads, MLP blocks and the embedding across all layers as
    #: one ``N`` — with one ``k_schedule`` (a fit) or one ``top_k`` (loaded),
    #: which the members must agree on (rule 4). On a *loaded* gate the pool
    #: is a pooled **readout** — one ``top_k`` cut through the members' joint
    #: ranking — and is legal under any map, since a ranking method's curve
    #: cuts every unit of a model together whatever fitted them. Never
    #: sweepable: a pool is a name. Absent, the gate budgets alone, and no
    #: field enters the canonical form. A budget fit stamps ``pool`` and
    #: ``pool_units`` into its bundle; a stamped pool must match the document's
    #: (and a pooled bundle is refused by an unpooled document), while an
    #: unstamped bundle may join any readout pool.
    pool: Leaf | None = None
    #: The precision the featurizer's parameters are held and saved in
    #: (:data:`PRECISION_DTYPES`); absent, the model's. Legal on every kind and
    #: stamped into a fitted bundle's identity, so an apply document re-authors
    #: the fit's.
    dtype: Leaf | None = None
    #: Load a fitted artifact instead of fitting one. Legal on every kind, and
    #: what makes a featurizer *loaded*: it trains nothing (rule 12), authors
    #: no start (``init``, ``seed``) and no training rule (``dead``,
    #: ``k_schedule``), and a gate whose map has no threshold is read out
    #: through ``top_k`` (:data:`FEATURIZER_FIELD_CONDITIONS`). The bundle's
    #: ``ArtifactIdentity`` is checked against the document at load and again
    #: at build (rule 15). Artifact-valued: a sweep over bundles is a sweep
    #: over fits.
    file_path: Leaf | None = None
    #: With ``file_path`` only: which entry of a swept bundle to load — the
    #: coordinate values that pick one fit out of a bundle holding several
    #: (:func:`_entry_selector`). Absent, the bundle must hold one.
    entry: Any = None
    #: Free text for the reader; not part of the canonical form or the digest.
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    """§2.6 — a free tensor owned by no featurizer: either a loaded constant
    (``file_path``, optionally narrowed to one bundle entry by ``entry``) or
    a trainable free tensor (``shape`` + ``init``, which must then appear in
    ``train.params``)."""

    file_path: Leaf | None = None
    entry: Any = None
    shape: Leaf | None = None
    init: Leaf | None = None
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class RowRole:
    """§2.8.1 — one named block of rows in the batch a referenced function
    receives, in batch order. ``rows`` is how many; the roles' order is the
    physical order, so ``[clean:1, corrupted:10]`` *is* ROME's eleven-row
    convention, written down."""

    role: str
    rows: int


@dataclasses.dataclass(frozen=True)
class CodeSpec:
    """§2.8.1 — a user function, declared rather than merely named.

    ``locator`` is the importable dotted path. Everything else is what a bare
    qualname left outside the digest: the arguments it is called with, the
    files it may read (content-digested at load), the environment variables
    it is allowed to read, and what the rows of its batch are. The source
    hash is derived, never authored (§6).
    """

    locator: Leaf
    args: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    data_inputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    env_inputs: tuple[str, ...] = ()
    row_roles: tuple[RowRole, ...] = ()
    description: str | None = None

    @property
    def declared_rows(self) -> int | None:
        """How many rows the declaration says the batch has, or ``None`` when
        it says nothing about rows."""
        return sum(r.rows for r in self.row_roles) if self.row_roles else None


@dataclasses.dataclass(frozen=True)
class ReadSpec:
    """§2.7 — a value producer. ``pos`` is a positions-table name or an
    inline :class:`PositionSpec` (int sugar already expanded). ``featurizer``
    is a single name or a left-to-right composition tuple."""

    site: Leaf
    pos: Leaf
    model: Leaf
    input: Leaf
    featurizer: Leaf | None = None
    dims: Leaf | None = None


@dataclasses.dataclass(frozen=True)
class Do:
    """§2.8 — one mechanism from the closed set. ``payload`` holds the
    mechanism's single value exactly as authored (operand name / literal
    scalar for ``swap``; the option mapping for the structured mechanisms;
    ``True`` for ``renormalize``)."""

    mechanism: Leaf
    payload: Any


@dataclasses.dataclass(frozen=True)
class WriteSpec:
    """§2.8 — an inert effect definition: no model, no input, no conditions;
    it executes inside every intervened model that lists it."""

    site: Leaf
    pos: Leaf
    do: Do
    featurizer: Leaf | None = None
    dims: Leaf | None = None
    #: The ragged-window policy, one of :data:`RAGGED_POLICIES`, or ``None``
    #: when the document authors none — which the executor reads as
    #: ``refuse`` (§5 rule 19). Never swept: how a window lands is an
    #: execution strategy, not a research variable.
    ragged: str | None = None


@dataclasses.dataclass(frozen=True)
class IMSpec:
    """§2.9 — an intervened model ℒ_{b∪𝕀}: a mandatory input role plus the
    writes in force (unordered; canonical form sorts)."""

    input: Leaf
    writes: tuple[str, ...] | Sweep | ArtifactRef


@dataclasses.dataclass(frozen=True)
class MetricSpec:
    """§2.10 — a closed-vocabulary reduction over one read (``of``) plus
    dataset columns. ``fields`` holds the kind's extra value fields.
    ``token_form`` says how this metric's string answers become token ids
    (``TOKEN_FORMS``). It is **required** on every kind in
    ``TOKEN_COLUMN_METRIC_KINDS``: the field defaults to ``"auto"`` here only
    so a spec constructed in code stays valid, while a *document* must name a
    form. ``"auto"`` is still the historical space-prefixed-first resolver —
    now something a document chooses rather than inherits. ``top_k`` additionally
    carries a mandatory ``by`` (``TOP_K_RANKINGS``) in ``fields``, because a
    top-k over a signed feature code and one over a vocabulary projection are
    different questions.

    ``unit`` and ``estimand_version`` are the record's identity
    (``causalab/protocol/estimand.py``): optional, in the canonical form
    **only when authored**, and — because every kind is one arithmetic in
    one unit — checked at parse to be the kind's own (``METRIC_UNITS``,
    ``<kind>/v1``). A document may state what its metric is; it may not
    declare it to be something else. Unauthored, both are derived onto every
    row the metric writes (``outputs.py``) and never into the digest.

    ``minimum_count`` is the metric's **decision threshold** (§2.10
    "Eligibility"): the fewest eligible rows its decision rule needs.
    Optional, never sweepable, in the canonical form only when authored;
    ``validate --data`` refuses one above the resolved base table's maximum
    eligible count (rule 4), and the cell's derived ``n_eligible`` is what a
    consumer holds it against. The count itself is derived at run time
    (``outputs.py``), never authored — this field is only the bar."""

    kind: Leaf
    of: Leaf
    fields: Mapping[str, Leaf]
    token_form: Leaf = "auto"
    unit: str | None = None
    estimand_version: str | None = None
    minimum_count: int | None = None


@dataclasses.dataclass(frozen=True)
class ConstraintSpec:
    """§2.11 ``constraint`` on a named mask-density regularizer: Edge
    Pruning's Lagrangian target. The term's value ``s`` (the mask mean under
    ``l1``, the expected kept fraction under ``l0``) is held to ``target`` by
    ``λ₁·(s − t) + λ₂·(s − t)²`` added to the loss, with the dual pair
    ``(λ₁, λ₂)`` **ascended** — stepped against its gradient — at ``dual_lr``
    from ``dual_init`` (``(0, 0)`` when unauthored, and then absent from the
    canonical form). The term has no ``weight``: the duals are its weight.
    ``λ₂`` is non-negative: the parser refuses a negative one, and the ascent
    never makes one (its gradient ``(s − t)²`` is non-negative) — a
    constructor that bypasses the parser owes the same bound, since
    ``_add_dual_groups`` reads ``init`` into a tensor without re-checking."""

    target: float
    dual_lr: float
    dual_init: tuple[float, float] | None = None

    @property
    def init(self) -> tuple[float, float]:
        return self.dual_init if self.dual_init is not None else (0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class ObjectiveTerm:
    """One weighted term of ``train.objective`` (§2.11): a differentiable
    metric by name, or a regularizer ``(kind, names)`` — ``kind`` is one of
    :data:`REGULARIZER_KINDS` (``l1``, ``l2``, ``l0``) and ``names`` the
    featurizers (or the one dotted slot) it penalizes together. ``name`` is the
    term's key in the named mapping form and ``None`` in the positional list
    form; it is what the term's ``weight`` is addressed by when swept
    (``train.objective.<name>.weight``). A ``constraint`` term (named form
    only) has no weight — ``None`` — and carries its :class:`ConstraintSpec`
    instead."""

    weight: Leaf | None
    metric: str | None = None
    regularizer: tuple[str, tuple[str, ...]] | None = None
    name: str | None = None
    #: a regularizer's reduction over the concatenated per-unit quantities
    #: (:data:`REGULARIZER_REDUCTIONS`): ``mean`` when unauthored — a kept
    #: unit then costs ``weight / units`` — or ``sum``, where it costs
    #: ``weight`` whatever the unit count (NeuroSurgeon's ``λ · Σ``, with λ
    #: "scaled with parameter count" by the author instead of by the mean).
    #: Kept ``None`` when unauthored so no canonical form materializes it.
    reduce: str | None = None
    #: §2.11 ``costs``: a per-target multiplier on the penalized quantities
    #: before they are concatenated — ``{target: c}`` (an unlisted target
    #: costs 1), or a word from :data:`REGULARIZER_COSTS`:
    #: ``"parameter_count"`` divides each target's quantities by its own
    #: element count, NeuroSurgeon's λ scaled with the parameter count, so
    #: under ``reduce: sum`` the term is the sum of per-featurizer means.
    #: ``None`` when unauthored.
    costs: Mapping[str, float] | str | None = None
    #: §2.11 ``constraint``: a Lagrangian target density on a named ``l1`` /
    #: ``l0`` term (:class:`ConstraintSpec`); ``None`` for every other term.
    constraint: ConstraintSpec | None = None

    def path(self, index: int) -> str:
        """The term's address in error messages: its name, or its list index."""
        if self.name is not None:
            return f"train.objective.{self.name}"
        return f"train.objective[{index}]"


@dataclasses.dataclass(frozen=True)
class TrainSpec:
    """§2.11 — the fit, declared. ``objective`` is the weighted terms, in
    authored order, whichever form spelled them."""

    objective: tuple[ObjectiveTerm, ...]
    params: tuple[str, ...]
    optimizer: Mapping[str, Leaf]
    steps: Mapping[str, Leaf]
    batch: Mapping[str, Leaf]
    #: §2.11 ``anneal``: open-loop schedules keyed by what they move — a
    #: trained featurizer's ``<name>.<slot>.<hyperparameter>``, or a named
    #: objective term's ``weight`` (``train.objective.<name>.weight``).
    anneal: Mapping[str, AnnealSchedule] | None = None
    #: §2.11 ``control``: closed-loop schedules — ``{<target>: {kind, signal,
    #: setpoint, gains, …}}`` where the target is a named objective term's
    #: ``weight`` (``train.objective.<name>.weight``) or an anneal-style dotted
    #: hyperparameter, and the authored value of the target is the
    #: controller's initial value.
    control: Mapping[str, Mapping[str, Any]] | None = None
    #: §2.11 ``phases``: consecutive step windows, each narrowing what trains
    #: and what is annealed; ``None`` is the one-phase fit every document
    #: before the field was.
    phases: tuple[PhaseSpec, ...] | None = None
    precision: Mapping[str, Leaf] | None = None
    eval: Mapping[str, Any] | None = None
    early_stop: Mapping[str, Leaf] | None = None
    checkpoint: Mapping[str, Leaf] | None = None
    seed: Leaf = 0


@dataclasses.dataclass(frozen=True)
class SaveEntry:
    """§2.12 — one manifest entry. Read/metric entries carry
    ``model``/``input``; trained-featurizer entries carry ``site``. The
    restated binding is cross-checked at validation, never trusted.
    ``reduce`` (reads only) saves a statistic over the gathered rows
    instead of the rows themselves (§2.12)."""

    value: str
    file_path: str
    model: str | None = None
    input: str | None = None
    site: str | None = None
    reduce: str | None = None
    #: A non-value entry kind (:data:`SAVE_KINDS`, §2.12): ``location_ledger``
    #: saves the run's resolved token indices, ``trajectory`` the trained
    #: featurizers at checkpoints along the fit. ``value`` then repeats the
    #: kind, so the one-entry-per-value rule holds for it too.
    kind: str | None = None
    #: ``trajectory`` only: how the checkpoints are spaced —
    #: ``{"count": n}`` | ``{"updates": n}`` | ``{"epochs": n}``
    #: (:data:`TRAJECTORY_EVERY_UNITS`).
    every: Mapping[str, int] | None = None


@dataclasses.dataclass(frozen=True)
class Document:
    """One parsed intervention specification (authored form, sugar
    expanded, wrappers preserved). The four groups of the file (§1) flatten
    to attributes here — a consumer reads ``document.sites``, never
    ``document.method.sites`` — while ``raw`` keeps the mapping the parser
    consumed, grouped as authored: the substrate sweep expansion and
    canonicalization operate on."""

    protocol_version: str
    model: ModelRef
    data: Mapping[str, DataRole | tuple[DataRole, ...]]
    sites: Mapping[str, SiteSpec]
    reads: Mapping[str, ReadSpec]
    save: tuple[SaveEntry, ...]
    title: str | None = None
    description: str | None = None
    positions: Mapping[str, PositionSpec | Sweep | ArtifactRef] = dataclasses.field(
        default_factory=dict
    )
    featurizers: Mapping[str, FeaturizerSpec] = dataclasses.field(default_factory=dict)
    params: Mapping[str, ParamSpec] = dataclasses.field(default_factory=dict)
    code: Mapping[str, CodeSpec] = dataclasses.field(default_factory=dict)
    writes: Mapping[str, WriteSpec] = dataclasses.field(default_factory=dict)
    intervened_models: Mapping[str, IMSpec] = dataclasses.field(default_factory=dict)
    metrics: Mapping[str, MetricSpec] = dataclasses.field(default_factory=dict)
    train: TrainSpec | None = None
    #: §2.2.1 — the row's named segments and their frame (``protocol/
    #: segments.py``); ``None`` is plain text, which is every document that
    #: does not author the section.
    segments: "SegmentsSpec | None" = None
    raw: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def named_entries(self) -> dict[str, str]:
        """Every declared name → the section that declares it. Duplicates are
        a validation concern (§5.3); the parser reports the first section."""
        seen: dict[str, str] = {}
        for section in NAMED_SECTIONS:
            table: Mapping[str, Any] = getattr(self, section)
            for name in table:
                seen.setdefault(name, section)
        return seen


def metric_reads_vocabulary(doc: Document, metric: MetricSpec) -> bool:
    """Whether the last axis of ``metric``'s read is the vocabulary.

    True exactly when the ``of`` read taps ``lm_head`` **and hands the
    projection on unchanged**. The site alone is not enough: a ``featurizer``
    re-expresses the value in its own latents and ``dims`` re-indexes a slice
    of it, so under either one the read's entries are no longer token ids — a
    softmax over them normalizes an axis that is not the vocabulary, and
    decoding index *j* as token *j* names the wrong token.

    Three places need the same answer and must not disagree: capability
    derivation (does this document oblige ``full_logits``?, §8), validation
    (may this ``top_k`` normalize, may a token-space kind bind here?, §2.10)
    and the reduction itself (are these indices token ids worth decoding?). A
    dangling ``of`` is validation's error to report, so it answers False here
    rather than raising."""
    read = doc.reads.get(str(metric.of))
    if read is None:
        return False
    if read.featurizer is not None or read.dims is not None:
        return False
    site = doc.sites.get(str(read.site))
    return site is not None and site.component == "lm_head"


def metric_column_fields(metric: MetricSpec) -> dict[str, str]:
    """The value fields of ``metric`` that name **dataset columns**, as
    ``field → column`` (§2.10).

    One predicate for three callers — ``validate --data``'s column check and
    its maximum-eligible count (``loader.py``) and the run-time eligibility
    predicate (``metrics.excluded_rows``) — so a row the loader counts as
    eligible is a row the run scores, and a field the loader skips is a field
    the run never looks up. Skipped: the non-column fields (``k``, ``by``,
    ``tokens``), ``groups`` (a mapping of literals), a ``target`` that is a
    read (:data:`READ_TARGET_METRIC_KINDS`), the enum-valued optional fields
    (``match.mode``), and any field whose value is not a string — a literal
    ``restrict`` list, or a swept field the caller resolves per point. A
    string-valued ``restrict`` **is** a column: its per-row value is the
    answer list the comparison is restricted to.
    """
    kind = str(metric.kind)
    optional = OPTIONAL_METRIC_FIELDS.get(kind, ())
    out: dict[str, str] = {}
    for field, value in metric.fields.items():
        if field in NON_COLUMN_METRIC_FIELDS or field == "groups":
            continue
        if field == "target" and kind in READ_TARGET_METRIC_KINDS:
            continue
        if field in optional and field != "restrict":
            continue
        if not isinstance(value, str):
            continue
        out[field] = value
    return out


# --------------------------------------------------------------------------- #
# raw loading
# --------------------------------------------------------------------------- #


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ParseError("P2", f"duplicate key {key!r} in one object")
        out[key] = value
    return out


def load_raw(text: str) -> dict[str, Any]:
    """Parse strict JSON text into an order-preserving mapping.

    YAML is accepted at the CLI surface (it parses to the same object model);
    this function is the JSON path and the normative behavior: duplicate keys
    and non-object top levels are errors, NaN/Infinity are rejected.
    """
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ParseError:
        raise
    except json.JSONDecodeError as err:
        raise ParseError("P1", f"not valid JSON: {err}") from err
    if not isinstance(raw, dict):
        raise ParseError("P1", "the top level must be a JSON object")
    return raw


def _reject_constant(name: str) -> Any:
    raise ParseError("P1", f"non-finite JSON constant {name!r} is not allowed")


# --------------------------------------------------------------------------- #
# parse helpers
# --------------------------------------------------------------------------- #


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParseError(
            "P2", f"expected an object, got {type(value).__name__}", path=path
        )
    return value


def _check_keys(mapping: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    allowed_set = set(allowed)
    for key in mapping:
        if key not in allowed_set:
            raise ParseError(
                "P3",
                f"unknown key {key!r}{suggest(key, allowed_set)}",
                path=path,
            )


def _enum(value: Any, options: Sequence[str], path: str) -> str:
    if not isinstance(value, str) or value not in options:
        raise ParseError(
            "P4",
            f"{value!r} is not one of {list(options)}"
            + (suggest(value, options) if isinstance(value, str) else ""),
            path=path,
        )
    return value


def _wrapped(
    value: Any,
    elem: Callable[[Any, str], Any],
    path: str,
    *,
    allow_sweep: bool = True,
) -> Any:
    """Parse a leaf that may be a ``{"sweep": …}`` wrapper around
    ``elem``-typed values (§3). Artifact references (§1) resolve *before*
    the parse gate (loader.load), so one reaching the parser is a loader
    misuse, not an authoring surface."""
    if isinstance(value, dict) and "sweep" in value:
        if not allow_sweep:
            raise ValidationError(14, "a sweep wrapper is not allowed here", path=path)
        _check_keys(value, ("sweep",), path)
        return _parse_sweep(value["sweep"], elem, path)
    if isinstance(value, dict) and isinstance(value.get("artifact"), str):
        raise ParseError(
            "P2",
            "unresolved artifact reference reached the parser — load through "
            "causalab.protocol.loader.load, which resolves artifact fields first",
            path=path,
        )
    return elem(value, path)


def _parse_sweep(spec: Any, elem: Callable[[Any, str], Any], path: str) -> Sweep:
    if isinstance(spec, dict):
        _check_keys(spec, ("range",), f"{path}.sweep")
        rng = spec.get("range")
        if (
            not isinstance(rng, list)
            or not 2 <= len(rng) <= 3
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in rng)
        ):
            raise ValidationError(
                14,
                "sweep range must be [start, stop] or [start, stop, step] of integers",
                path=f"{path}.sweep",
            )
        start, stop = rng[0], rng[1]
        step = rng[2] if len(rng) == 3 else 1
        if step == 0:
            raise ValidationError(
                14, "sweep range step must be non-zero", path=f"{path}.sweep"
            )
        if len(range(start, stop, step)) > 1_000_000:  # O(1); before materializing
            raise ValidationError(
                14,
                "sweep range denotes over 1,000,000 values — refuse before "
                "materializing (§5.14)",
                path=f"{path}.sweep",
            )
        values = list(range(start, stop, step))
    elif isinstance(spec, list):
        values = spec
    else:
        raise ValidationError(
            14,
            f"a sweep wrapper takes a list or a range object, got {type(spec).__name__}",
            path=f"{path}.sweep",
        )
    if not values:
        raise ValidationError(
            14, "a sweep axis must have at least one value", path=path
        )
    return Sweep(
        values=tuple(elem(v, f"{path}.sweep[{i}]") for i, v in enumerate(values))
    )


#: Reductions a ``save`` entry may apply to a read (§2.12). Closed: the
#: vocabulary grows by PR, with a §2.12 row and a test, which is what the
#: docs↔code guard in ``tests/protocol/test_vocabulary_census.py`` enforces.
#:
#: All five collapse ``(rows, width)`` to ``(width,)``, so the un-reduced
#: harvest never reaches disk — that is the whole reason ``reduce`` exists.
#: ``mean`` and ``sum`` are the pair a sharded run needs (a weighted mean
#: across points is ``sum``s over ``count``s); ``std`` reports the spread the
#: mean hides; ``median`` survives an outlier row that the mean does not.
SAVE_REDUCTIONS: tuple[str, ...] = ("mean", "sum", "std", "median", "count")


def _entry_selector(
    value: Any, path: str, *, allow_slot: bool = False
) -> dict[str, Any]:
    """§2.5/§2.6 ``entry``: a mapping of coordinate name to scalar value,
    naming one entry inside a loaded bundle
    (:mod:`causalab.protocol.bundles`). Names are the coordinate names as
    they appear in the producer's keys (``k``, ``seed``,
    ``target.layers``) — not full axis ids, which the consuming document has
    no reason to know."""
    obj = _require_mapping(value, path)
    if not obj:
        raise ParseError(
            "P2", "an 'entry' selector names at least one coordinate", path=path
        )
    selector: dict[str, Any] = {}
    for name, coord in obj.items():
        if name == "slot":
            if not allow_slot:
                raise ParseError(
                    "P2",
                    "a featurizer bundle's slots are fixed by its kind — "
                    "'slot' selects only inside a params bundle",
                    path=f"{path}.slot",
                )
            if not isinstance(coord, str):
                raise ParseError("P2", "'slot' names one tensor", path=f"{path}.slot")
            selector[name] = coord
            continue
        if isinstance(coord, (dict, list)):
            raise ParseError(
                "P2",
                f"entry coordinate {name!r} must be a scalar — a bundle key "
                "records one value per coordinate",
                path=f"{path}.{name}",
            )
        selector[name] = coord
    return selector


def _scalar_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ParseError(
            "P2", f"expected a string, got {type(value).__name__}", path=path
        )
    return value


def _scalar_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ParseError(
            "P2", f"expected an integer, got {type(value).__name__}", path=path
        )
    return value


def _scalar_number(value: Any, path: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(
            "P2", f"expected a number, got {type(value).__name__}", path=path
        )
    return value


def _scalar_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ParseError("P2", f"expected true or false (got {value!r})", path=path)
    return value


def _int_list(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        raise ParseError("P2", "expected a list of integers", path=path)
    return tuple(value)


def _str_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ParseError("P2", "expected a list of strings", path=path)
    return tuple(value)


def _any_leaf(value: Any, path: str) -> Any:
    return value


def _token_list(value: Any, path: str) -> tuple[str, ...]:
    """``token_logits.tokens`` (§2.10): a non-empty list of literal token
    strings, each answer listed once and none of them blank.

    "Once" is judged after the leading-space normalization the resolver
    applies — ``" X"`` and ``"X"`` name the same answer, and ``token_form``
    alone decides the form — so the ``["X", " X"]`` idiom is refused here,
    torch-free, rather than at run time as two entries carrying one logit.
    An empty or whitespace-only entry is refused for the same reason: it
    names no answer, and under ``space_prefixed`` it would resolve to the
    lone space token. Two *different* strings that a tokenizer maps to one id
    can only be seen with the tokenizer in hand, and are refused where the
    metric resolves them (:mod:`causalab.neural.shared.metrics`).
    """
    tokens = _str_list(value, path)
    if not tokens:
        raise ParseError(
            "P2", "expected at least one token string — nothing to save", path=path
        )
    seen: dict[str, str] = {}
    for token in tokens:
        if not token.strip():
            raise ParseError(
                "P2",
                f"{token!r} is empty or whitespace-only — a token_logits entry "
                "names an answer token, and a blank names none (§2.10)",
                path=path,
            )
        bare = token.lstrip(" ")
        if bare in seen:
            raise ParseError(
                "P2",
                f"{token!r} and {seen[bare]!r} name the same token — a leading "
                "space is normalized away before token_form decides the form "
                "(§2.10), so list each answer once",
                path=path,
            )
        seen[bare] = token
    return tokens


# --------------------------------------------------------------------------- #
# section parsers
# --------------------------------------------------------------------------- #


def _parse_model(raw: Any, path: str) -> ModelRef:
    obj = _require_mapping(raw, path)
    _check_keys(
        obj, ("key", "revision", "dtype", "quantization", "attn_implementation"), path
    )
    if "key" not in obj:
        raise ParseError("P2", "model needs a 'key'", path=path)
    key = _wrapped(obj["key"], _scalar_str, f"{path}.key")
    revision = (
        _wrapped(obj["revision"], _scalar_str, f"{path}.revision")
        if "revision" in obj
        else "main"
    )
    dtype = (
        _wrapped(
            obj["dtype"],
            lambda v, p: _enum(v, PRECISION_DTYPES, p),
            f"{path}.dtype",
        )
        if "dtype" in obj
        else None
    )
    quantization = (
        _parse_quantization(obj["quantization"], f"{path}.quantization")
        if "quantization" in obj
        else None
    )
    attn_implementation = (
        _wrapped(
            obj["attn_implementation"],
            lambda v, p: _enum(v, ATTENTION_IMPLEMENTATIONS, p),
            f"{path}.attn_implementation",
        )
        if "attn_implementation" in obj
        else None
    )
    return ModelRef(
        key=key,
        revision=revision,
        dtype=dtype,
        quantization=quantization,
        attn_implementation=attn_implementation,
    )


def _parse_quantization(raw: Any, path: str) -> QuantizationSpec:
    obj = _require_mapping(raw, path)
    _check_keys(
        obj,
        ("scheme", "method", "compute_dtype", *_QUANT_4BIT_FIELDS, *_QUANT_INT8_FIELDS),
        path,
    )
    if "scheme" not in obj:
        raise ParseError(
            "P2",
            f"quantization needs a 'scheme' — one of {list(QUANT_SCHEMES)}",
            path=path,
        )
    return QuantizationSpec(
        scheme=_wrapped(
            obj["scheme"], lambda v, p: _enum(v, QUANT_SCHEMES, p), f"{path}.scheme"
        ),
        method=_wrapped(
            obj["method"], lambda v, p: _enum(v, QUANT_METHODS, p), f"{path}.method"
        )
        if "method" in obj
        else "bitsandbytes",
        compute_dtype=_wrapped(
            obj["compute_dtype"],
            lambda v, p: _enum(v, PRECISION_DTYPES, p),
            f"{path}.compute_dtype",
        )
        if "compute_dtype" in obj
        else None,
        double_quant=_wrapped(obj["double_quant"], _scalar_bool, f"{path}.double_quant")
        if "double_quant" in obj
        else None,
        int8_threshold=_wrapped(
            obj["int8_threshold"], _scalar_number, f"{path}.int8_threshold"
        )
        if "int8_threshold" in obj
        else None,
    )


#: §2.2 ``draw.kind``: how a fit picks one member of a list-valued
#: counterfactual column per row per epoch. Closed; censused against §2.2.
DRAW_KINDS: tuple[str, ...] = ("uniform",)

_INDEXED_FIELD = re.compile(r"\[\d+\]$")


def _parse_data_role(raw: Any, path: str, *, shuffleable: bool) -> DataRole:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("dataset", "field", "shuffle", "draw"), path)
    for field in ("dataset", "field"):
        if field not in obj:
            raise ParseError("P2", f"data role needs a {field!r}", path=path)
    shuffle: dict[str, Any] | None = None
    if "shuffle" in obj:
        if not shuffleable:
            raise ParseError(
                "P2",
                "'shuffle' permutes a counterfactual role; the base role is the "
                "population and is never permuted (§2.2)",
                path=f"{path}.shuffle",
            )
        shuffle = _parse_shuffle(obj["shuffle"], f"{path}.shuffle")
    field_leaf = _wrapped(obj["field"], _scalar_str, f"{path}.field")
    draw: dict[str, Any] | None = None
    if "draw" in obj:
        if not shuffleable:
            raise ParseError(
                "P2",
                "'draw' samples a counterfactual role's members; the base role is "
                "the population and has one input per row (§2.2)",
                path=f"{path}.draw",
            )
        draw = _parse_draw(obj["draw"], f"{path}.draw")
        if isinstance(field_leaf, str) and _INDEXED_FIELD.search(field_leaf):
            raise ParseError(
                "P2",
                f"a drawn role names its list column bare — {field_leaf!r} already "
                "picks one member, so there is nothing to draw; drop the index "
                "(the fixed member every non-training forward reads is 'draw.eval')",
                path=f"{path}.field",
            )
    return DataRole(
        dataset=_wrapped(obj["dataset"], _scalar_str, f"{path}.dataset"),
        field=field_leaf,
        shuffle=shuffle,
        draw=draw,
    )


def _parse_draw(raw: Any, path: str) -> dict[str, Any]:
    """``draw: {kind: uniform, eval?: j}`` (§2.2). ``kind`` is closed and not
    sweepable (what is drawn is what the document *is*, not a knob); ``eval``
    is the member index every non-training forward reads, a non-negative
    integer that is not a ``bool``, absent for ``0``."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("kind", "eval"), path)
    if "kind" not in obj:
        raise ParseError(
            "P2", f"'draw' needs a 'kind' — one of {list(DRAW_KINDS)}", path=path
        )
    kind = _wrapped(
        obj["kind"],
        lambda v, p: _enum(_scalar_str(v, p), DRAW_KINDS, p),
        f"{path}.kind",
        allow_sweep=False,
    )
    out: dict[str, Any] = {"kind": kind}
    if "eval" in obj:
        member = _wrapped(obj["eval"], _scalar_int, f"{path}.eval", allow_sweep=False)
        if int(member) < 0:
            raise ParseError(
                "P2",
                f"'eval' indexes the list column — a non-negative integer, got {member!r}",
                path=f"{path}.eval",
            )
        out["eval"] = member
    return out


def _parse_shuffle(raw: Any, path: str) -> dict[str, Any]:
    """``shuffle: {seed: <int>}`` (§2.2) — the seed is the permutation's only
    input, an integer that is not a ``bool``, and not sweepable: one document
    is one pairing, and a swept seed would make the control document differ
    from its target by an axis rather than by one field."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("seed",), path)
    if "seed" not in obj:
        raise ParseError(
            "P2", "'shuffle' needs a 'seed' — the permutation's only input", path=path
        )
    seed = _wrapped(obj["seed"], _scalar_int, f"{path}.seed", allow_sweep=False)
    return {"seed": seed}


def _parse_data(raw: Any, path: str) -> dict[str, DataRole | tuple[DataRole, ...]]:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("base", "counterfactual"), path)
    if "base" not in obj:
        raise ParseError("P2", "data needs a 'base' role", path=path)
    out: dict[str, DataRole | tuple[DataRole, ...]] = {
        "base": _parse_data_role(obj["base"], f"{path}.base", shuffleable=False)
    }
    if "counterfactual" in obj:
        cf = obj["counterfactual"]
        if isinstance(cf, list):
            out["counterfactual"] = tuple(
                _parse_data_role(s, f"{path}.counterfactual[{j}]", shuffleable=True)
                for j, s in enumerate(cf)
            )
        else:
            out["counterfactual"] = _parse_data_role(
                cf, f"{path}.counterfactual", shuffleable=True
            )
    return out


def _parse_position_spec(raw: Any, path: str) -> PositionSpec:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return PositionSpec(index=raw)  # §6.1 int sugar
    if raw == ALL_POSITIONS:
        return PositionSpec(all=True)  # §6.1 "all" sugar
    obj = _require_mapping(raw, path)
    # §2.3 spans: any span key dispatches to the span module, which owns the
    # algebra's grammar (sets, unions, intersections, predicates, `atomic`).
    from causalab.protocol.spans import is_span_object, parse_span_spec

    if is_span_object(obj):
        return parse_span_spec(
            obj,
            path,
            parse_position=_parse_position_spec,
            parse_anchor_ref=_parse_anchor_ref,
        )
    _check_keys(
        obj,
        (
            "index",
            "span",
            "variable",
            "column",
            "all",
            "scope",
            "relative_to",
            "generated",
            "alignment",
        ),
        path,
    )
    anchors = [k for k in ("index", "span", "variable", "column", "all") if k in obj]
    if len(anchors) != 1:
        raise ParseError(
            "P2",
            "a position spec needs exactly one of "
            f"index/span/variable/column/all, got {anchors}",
            path=path,
        )
    if "all" in obj and obj["all"] is not True:
        raise ParseError(
            "P2",
            f'all is the flag {{"all": true}} — got {obj["all"]!r}; there is no '
            "other all-positions selection to spell",
            path=path,
        )
    index = (
        _wrapped(obj["index"], _scalar_int, f"{path}.index") if "index" in obj else None
    )
    span = _wrapped(obj["span"], _parse_span, f"{path}.span") if "span" in obj else None
    variable = (
        _wrapped(obj["variable"], _scalar_str, f"{path}.variable")
        if "variable" in obj
        else None
    )
    column = (
        _wrapped(obj["column"], _scalar_str, f"{path}.column")
        if "column" in obj
        else None
    )
    every = True if "all" in obj else None
    scope_ref = (
        _wrapped(obj["scope"], _parse_anchor_ref, f"{path}.scope")
        if "scope" in obj
        else None
    )
    relative_ref = (
        _wrapped(obj["relative_to"], _parse_anchor_ref, f"{path}.relative_to")
        if "relative_to" in obj
        else None
    )
    anchor_ref = scope_ref if scope_ref is not None else relative_ref
    anchor_source = anchor_ref[0] if anchor_ref is not None else "variable"
    scope = scope_ref[1] if scope_ref is not None else None
    relative_to = relative_ref[1] if relative_ref is not None else None
    if (scope is not None or relative_to is not None) and (
        variable is not None or column is not None or every is not None
    ):
        raise ParseError(
            "P2",
            "scope/relative_to modify an index or span, not a variable/column/all spec",
            path=path,
        )
    if scope is not None and relative_to is not None:
        raise ParseError(
            "P2", "scope and relative_to are mutually exclusive", path=path
        )
    generated = (
        _parse_generated(obj["generated"], f"{path}.generated")
        if "generated" in obj
        else None
    )
    # A declared cardinality is a string and never swept (a sweep over how an
    # address pairs is not a sweep over anything the model sees). Membership
    # in ALIGNMENT_CARDINALITIES and fit to the address are rule 26's
    # (`validate`), so one rule names the field for every way it can be wrong.
    alignment = (
        _wrapped(obj["alignment"], _scalar_str, f"{path}.alignment", allow_sweep=False)
        if "alignment" in obj
        else None
    )
    if generated is not None:
        # The continuation frame carries no prompt-frame notions: a `column`
        # holds a substring of the *input* text, and scope/relative_to anchor
        # on a prompt variable's token run. Both are meaningless in a frame
        # the prompt does not contain (§2.3).
        # …except a scope naming the `continuation` segment, which *is* the
        # frame `generated` selects (§2.2.1): the name and the frame agree.
        in_continuation = anchor_source == "segment" and scope == "continuation"
        offenders = [
            key
            for key, present in (
                ("column", column is not None),
                ("scope", scope is not None and not in_continuation),
                ("relative_to", relative_to is not None),
            )
            if present
        ]
        if offenders:
            raise ParseError(
                "P2",
                f"{offenders} resolve against the prompt, so they cannot combine "
                "with 'generated' — anchor inside the continuation with "
                "index/span/variable/all instead",
                path=path,
            )
    if isinstance(span, tuple):
        lo, hi = span
        if scope is None and (lo < 0 or hi <= lo):
            raise ParseError(
                "P2",
                f"span [{lo}, {hi}) is not a forward window — unscoped spans are "
                "content-frame, non-negative, non-empty",
                path=path,
            )
        if (
            scope is not None
            and (lo < 0) == (hi < 0 or hi == 0 and lo < 0)
            and lo >= hi
        ):
            raise ParseError(
                "P2", f"scoped span [{lo}, {hi}) is statically empty", path=path
            )
    return PositionSpec(
        index=index,
        span=span,
        variable=variable,
        column=column,
        all=every,
        scope=scope,
        relative_to=relative_to,
        anchor_source=anchor_source,
        generated=generated,
        alignment=alignment,
    )


def _parse_generated(value: Any, path: str) -> dict[str, Any]:
    """§2.3 — the continuation frame selector: ``{"max_new_tokens": n}``.

    A mapping rather than a bare int on purpose: stopping conditions
    (``stop``, ``min_new_tokens``) join this object later without any
    ambiguity about what a bare number would have meant.
    """
    obj = _require_mapping(value, path)
    _check_keys(obj, ("max_new_tokens",), path)
    if "max_new_tokens" not in obj:
        raise ParseError(
            "P2", "generated needs 'max_new_tokens' — the decode budget", path=path
        )
    budget = _wrapped(obj["max_new_tokens"], _scalar_int, f"{path}.max_new_tokens")
    if isinstance(budget, int) and budget < 1:
        raise ParseError(
            "P2",
            f"max_new_tokens is {budget} — a continuation frame needs at least "
            "one generated token",
            path=f"{path}.max_new_tokens",
        )
    return {"max_new_tokens": budget}


def _parse_span(value: Any, path: str) -> tuple[int, int]:
    ints = _int_list(value, path)
    if len(ints) != 2:
        raise ParseError("P2", "a span is [a, b) — exactly two integers", path=path)
    return (ints[0], ints[1])


def span_length(spec: Any) -> int | None:
    """The number of positions a fixed ``span`` [a, b) addresses on every row,
    or ``None`` when the spec is not such a window — an ``index``, a
    ``variable``/``column`` (as wide as the row's value), ``all`` (as wide as
    the row), a span set, a ``generated`` window (clipped to the row's decode
    width) or a ``scope``d one (sliced out of the anchor's run, so shorter on a
    short anchor) or a ``relative_to`` one (not placed by its anchor at all
    today: the resolver offsets an ``index`` only and a span so spelled falls
    through to the content frame — §2.3's span offset is unimplemented, and an
    unimplemented placement may not size a θ). Stricter than
    ``spans.static_indices`` by design: an ``index``, a static ``indices`` set,
    a static ``union`` and a one-position span all answer there and are
    declined here — a position gate's window is a contiguous span of two or
    more (a non-contiguous window is deferred, as ``all`` is). Kept here rather
    than deferring to ``spans``: a module-scope import would be circular
    (``spans`` imports ``PositionSpec``; the parser reaches it by a
    function-local import where it must), and the two answer different
    questions — ``static_indices`` asks whether an address set is static, this
    asks whether a window is a contiguous row-independent span of two or more.
    What sizes a position gate (§2.5 ``axis``);
    ``canonical._window_length_raw`` is its twin over the raw
    mapping, and a change to what counts as a window is made in both."""
    if not isinstance(spec, PositionSpec) or not isinstance(spec.span, tuple):
        return None
    if spec.generated is not None:
        return None
    if spec.scope is not None or spec.relative_to is not None:
        return None
    a, b = spec.span
    if not (isinstance(a, int) and isinstance(b, int)):
        return None
    # a window of one position is one scalar at one position — `group: site`
    # on a one-token write, not a mask *over* positions; an `index` is the
    # same address under the other spelling and is refused alike.
    # `b - a` is the realized length because an unscoped span is non-negative
    # (`_parse_position`: "content-frame, non-negative"). A §2.3 that admitted
    # a right-anchored `[-4, -1)` would have to refuse a straddling `[-2, 1)`
    # here — its length is the row's — and in `_window_length_raw` with it.
    return b - a if b - a >= 2 else None


def _parse_anchor_ref(value: Any, path: str) -> tuple[str, str]:
    """A ``scope``/``relative_to`` anchor: ``(source, name)`` where source is
    ``"variable"`` (per-role prompt variable), ``"column"`` (a top-level row
    column) or ``"segment"`` (a declared segment, §2.2.1) — §2.3."""
    obj = _require_mapping(value, path)
    _check_keys(obj, ("variable", "column", "segment"), path)
    named = [key for key in ("variable", "column", "segment") if key in obj]
    if len(named) != 1 or not isinstance(obj[named[0]], str):
        raise ParseError(
            "P2",
            'expected {"variable": "<name>"}, {"column": "<name>"} or '
            '{"segment": "<name>"}',
            path=path,
        )
    return named[0], obj[named[0]]


def _parse_positions(
    raw: Any, path: str
) -> dict[str, PositionSpec | Sweep | ArtifactRef]:
    obj = _require_mapping(raw, path)
    return {
        name: _wrapped(value, _parse_position_spec, f"{path}.{name}")
        for name, value in obj.items()
    }


def _current_component(value: Any) -> Any:
    """Fold a retired component spelling onto the name that replaced it.

    Done here rather than in canonicalization so that *nothing* downstream ever
    sees the old name: the canonical form, the digest and every table are in one
    vocabulary, and the alias is a parse-time courtesy with no second code path
    behind it.
    """
    return DEPRECATED_COMPONENTS.get(value, value) if isinstance(value, str) else value


def _band(value: Any, path: str) -> tuple[int, ...]:
    """§2.4 ``layers``: the band a site spans, as a tuple of layer indices.

    Authored as a non-empty list of integers in strictly increasing order —
    a band is a *set* of layers, so a repeated or unsorted index is a typo,
    not a second spelling. A bare index is the one-layer band ``[n]``: it is
    what an axis over ``layers`` (``{"sweep": {"range": [0, 32]}}``,
    ``{"at_once": …}``) and a workflow ``emit`` hand a point, and the
    canonical form writes the list either way (``canonical._canon_site``), so
    the two spellings carry one digest. ``true``/``false`` are refused as
    they are everywhere an integer is expected.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if not isinstance(value, list):
        raise ParseError(
            "P2",
            "'layers' is a band: a list of layer indices, [18] for one layer "
            f"(got {type(value).__name__})",
            path=path,
        )
    if not value:
        raise ParseError(
            "P2",
            "'layers' names at least one layer — an empty band is no address",
            path=path,
        )
    for i, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ParseError(
                "P2",
                f"expected an integer layer index, got {type(item).__name__}",
                path=f"{path}[{i}]",
            )
    if any(b <= a for a, b in zip(value, value[1:])):
        raise ParseError(
            "P2",
            f"'layers' is a band, listed in strictly increasing order — got "
            f"{value}" + (" (a layer repeats)" if len(set(value)) < len(value) else ""),
            path=path,
        )
    return tuple(value)


def _parse_site(raw: Any, path: str) -> SiteSpec:
    obj = _require_mapping(raw, path)
    if "layer" in obj:
        # the protocol_version 2 spelling — named, with the rename and the
        # verb that carries it, rather than left to `suggest`'s guess
        raise ParseError(
            "P3",
            "unknown key 'layer' — did you mean 'layers'? protocol_version 3 "
            "renamed a site's depth index to 'layers', a band of layer "
            "indices ([18] for one layer, §2.4); `causalab migrate <file>` "
            "rewrites a protocol_version 2 document",
            path=path,
        )
    _check_keys(obj, ("component", "layers", "head", "expert", "stream"), path)
    if "component" not in obj:
        raise ParseError("P2", "a site needs a 'component'", path=path)
    component = _wrapped(
        obj["component"],
        lambda v, p: _enum(_current_component(v), COMPONENTS, p),
        f"{path}.component",
    )
    layers = (
        _wrapped(obj["layers"], _band, f"{path}.layers") if "layers" in obj else None
    )
    if isinstance(component, str):  # un-swept: layer presence is checkable now
        if component in LAYERLESS_COMPONENTS and layers is not None:
            raise ParseError("P2", f"{component} is layer-less", path=path)
        if component not in LAYERLESS_COMPONENTS and layers is None:
            raise ParseError("P2", f"{component} needs 'layers'", path=path)
    return SiteSpec(
        component=component,
        layers=layers,
        head=_wrapped(obj["head"], _scalar_int, f"{path}.head")
        if "head" in obj
        else None,
        expert=_wrapped(obj["expert"], _scalar_int, f"{path}.expert")
        if "expert" in obj
        else None,
        stream=_wrapped(
            obj["stream"], lambda v, p: _enum(v, STREAMS, p), f"{path}.stream"
        )
        if "stream" in obj
        else None,
    )


def _parse_sites(raw: Any, path: str) -> dict[str, SiteSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_site(value, f"{path}.{name}") for name, value in obj.items()}


def _authored_gate_maps(parametrization: Any) -> list[str]:
    """The concrete map names a gate's ``parametrization`` authors —
    :data:`GATE_DEFAULT_MAP` when absent, every string arm of a sweep — so a
    legality check holds each arm. An artifact-valued arm is resolved at run
    time and is not checked here."""
    if parametrization is None:
        return [GATE_DEFAULT_MAP]
    values = (
        list(parametrization.values)
        if isinstance(parametrization, Sweep)
        else [parametrization]
    )
    return [v for v in values if isinstance(v, str)]


def _is_sweep_like(value: Mapping[str, Any]) -> bool:
    """A `{"sweep": …}` wrapper (§3), which `_wrapped` owns — as opposed to a
    field's own mapping form. `{"axis": …}` is lowered to a sweep before the
    parse gate (§3.2), so the second test is defensive: one that reached here
    is refused by `_wrapped`'s callback as not swept, not read as a form."""
    return "sweep" in value or "axis" in value


def _parse_featurizer(raw: Any, path: str) -> FeaturizerSpec:
    obj = _require_mapping(raw, path)
    kind_raw = obj.get("kind", "identity")
    kind = _wrapped(
        kind_raw,
        lambda v, p: _enum(v, FEATURIZER_KINDS, p),
        f"{path}.kind",
    )
    kind_key = kind if isinstance(kind, str) else "identity"
    allowed = {"kind", "file_path", "entry", "dtype", "description"} | set(
        FEATURIZER_FIELDS.get(kind_key, frozenset())
    )
    _check_keys(obj, allowed, path)
    if "entry" in obj and "file_path" not in obj:
        raise ParseError(
            "P2",
            "'entry' selects inside a loaded bundle — it needs a file_path",
            path=path,
        )
    parametrization = None
    forward = None
    if (
        "parametrization" in obj
        and isinstance(obj["parametrization"], Mapping)
        and not _is_sweep_like(obj["parametrization"])
    ):
        # §2.5 the mapping form: {"forward": "hard", "backward": <map>} — a
        # gate's straight-through split of the forward and backward masks
        p_param = f"{path}.parametrization"
        if kind_key != "gate":
            raise ParseError(
                "P2",
                "the mapping form of 'parametrization' splits a gate's forward and "
                "backward masks — "
                + (
                    f"a {kind_key!r} has one rotation map"
                    if isinstance(kind, str)
                    else "a swept 'kind' cannot take it (the form is a gate's)"
                ),
                path=p_param,
            )
        mapping = obj["parametrization"]
        _check_keys(mapping, ("forward", "backward"), p_param)
        for field in ("forward", "backward"):
            if field not in mapping:
                raise ParseError(
                    "P2",
                    'the mapping form is {"forward": "hard", "backward": <map>} — '
                    f"{field!r} is missing",
                    path=p_param,
                )
        forward_raw = mapping["forward"]
        if forward_raw in ("soft", "sampled"):
            raise ParseError(
                "P2",
                f"forward {forward_raw!r} is the map's own forward — spell the map "
                "alone ('sampled' is hard_concrete itself); the mapping form is for "
                "'hard'",
                path=f"{p_param}.forward",
            )
        forward = _enum(
            _scalar_str(forward_raw, f"{p_param}.forward"),
            FORWARD_MASKS,
            f"{p_param}.forward",
        )
        if isinstance(mapping["backward"], Mapping) and "sweep" in mapping["backward"]:
            raise ParseError(
                "P2",
                'the mapping form of \'parametrization\' ({"forward", "backward"}) '
                "is not swept — author one document per arm (§2.5)",
                path=f"{p_param}.backward",
            )
        parametrization = _enum(
            _scalar_str(mapping["backward"], f"{p_param}.backward"),
            GATE_PARAMETRIZATIONS,
            f"{p_param}.backward",
        )
    elif "parametrization" in obj:
        # one field, one meaning, an enum per kind: a subspace's rotation map
        # or a gate's theta→mask map (§2.5); a swept kind admits either
        vocabulary = (
            GATE_PARAMETRIZATIONS
            if kind_key == "gate"
            else PARAMETRIZATIONS
            if isinstance(kind, str)
            else (*PARAMETRIZATIONS, *GATE_PARAMETRIZATIONS)
        )

        def _one_map(v: Any, p: str) -> str:
            if isinstance(v, Mapping):
                # the mapping form is not a leaf, so a sweep arm may not be one:
                # an ablation grid is one document per forward/backward pair
                raise ParseError(
                    "P2",
                    "the mapping form of 'parametrization' ({\"forward\", "
                    '"backward"}) is not swept — author one document per arm '
                    "(§2.5)",
                    path=p,
                )
            return _enum(v, vocabulary, p)

        parametrization = _wrapped(
            obj["parametrization"], _one_map, f"{path}.parametrization"
        )
    # §2.5's conditional legality, read off FEATURIZER_FIELD_CONDITIONS: a
    # field authored in a state (fitted or loaded) or under a map the table
    # does not list is refused in the table's own words. A swept map is legal
    # only if every arm is — compile would refuse the whole run on the
    # offending arm anyway (every expanded point is validated before weights
    # load), and naming the arm here is the better message.
    loaded = "file_path" in obj
    maps = _authored_gate_maps(parametrization) if kind_key == "gate" else None
    for field, legality in FEATURIZER_FIELD_CONDITIONS.items():
        if field not in obj:
            continue
        legal = legality.legal(loaded=loaded)
        if legal is None:
            continue
        if not legal:
            why = legality.why_loaded if loaded else legality.why_fit
            raise ParseError("P2", why, path=f"{path}.{field}")
        if maps is not None:
            outside = [m for m in maps if m not in legal]
            if outside:
                named = ", ".join(repr(m) for m in outside)
                raise ParseError(
                    "P2", legality.why_map.format(maps=named), path=f"{path}.{field}"
                )
    group = None
    if "group" in obj:
        group = _wrapped(
            obj["group"],
            lambda v, p: _enum(v, GATE_GROUPS, p),
            f"{path}.group",
        )
    axis = None
    if "axis" in obj:
        axis = _wrapped(
            obj["axis"],
            lambda v, p: _enum(v, GATE_AXES, p),
            f"{path}.axis",
            allow_sweep=False,
        )
        if "group" in obj:
            raise ParseError(
                "P2",
                "a position gate is one θ per addressed position over every "
                "coordinate — already the scalar `group: site` would name — so "
                "`axis` and `group` do not combine (§2.5)",
                path=f"{path}.group",
            )
        if "pool" in obj:
            raise ParseError(
                "P2",
                "a budget pool over position gates is not written down (§2.5) — "
                "pool feature gates, or budget one position gate alone",
                path=f"{path}.pool",
            )
    temperature = (
        _wrapped(obj["temperature"], _positive_number, f"{path}.temperature")
        if "temperature" in obj
        else None
    )
    stretch = (
        _parse_stretch(obj["stretch"], f"{path}.stretch") if "stretch" in obj else None
    )
    k_schedule = None
    stop_grad_shift = None
    pool = None
    if kind_key == "gate":
        assert maps is not None
        if "pool" in obj and loaded and "top_k" not in obj:
            raise ParseError(
                "P2",
                "'pool' on a loaded gate is a pooled readout — one 'top_k' cut "
                "through the members' joint ranking — so it needs a top_k (§2.5)",
                path=f"{path}.pool",
            )
        ranked = [m for m in maps if GATE_MAPS[m].ranked]
        if ranked and not loaded and "k_schedule" not in obj:
            raise ParseError(
                "P2",
                "a budget gate draws its per-step budget from 'k_schedule' (§2.5) "
                "— {'kind': 'fixed', 'k': n} or {'kind': 'uniform' | "
                "'log_uniform', 'low': a, 'high': b, 'eval': n}",
                path=path,
            )
        if "k_schedule" in obj:
            k_schedule = _parse_k_schedule(obj["k_schedule"], f"{path}.k_schedule")
        if "stop_grad_shift" in obj:
            stop_grad_shift = _wrapped(
                obj["stop_grad_shift"], _scalar_bool, f"{path}.stop_grad_shift"
            )
        if "pool" in obj:
            # a name, never swept: the members of a pool are whoever authors
            # the same string, and a sweep over names would be a sweep over
            # which gates share a budget — a different document per arm
            pool = _scalar_str(obj["pool"], f"{path}.pool")
            if not pool:
                raise ParseError(
                    "P2", "a budget pool needs a name", path=f"{path}.pool"
                )
    dead = _parse_gate_dead(obj["dead"], f"{path}.dead") if "dead" in obj else None
    return FeaturizerSpec(
        kind=kind,
        k=_wrapped(obj["k"], _scalar_int, f"{path}.k") if "k" in obj else None,
        parametrization=parametrization,
        init=_parse_featurizer_init(obj["init"], f"{path}.init", kind_key)
        if "init" in obj
        else None,
        seed=_wrapped(obj["seed"], _scalar_int, f"{path}.seed")
        if "seed" in obj
        else None,
        group=group,
        temperature=temperature,
        stretch=stretch,
        dead=dead,
        top_k=_wrapped(obj["top_k"], _non_negative_int, f"{path}.top_k")
        if "top_k" in obj
        else None,
        k_schedule=k_schedule,
        stop_grad_shift=stop_grad_shift,
        pool=pool,
        axis=axis,
        forward=forward,
        dtype=_wrapped(
            obj["dtype"], lambda v, p: _enum(v, PRECISION_DTYPES, p), f"{path}.dtype"
        )
        if "dtype" in obj
        else None,
        file_path=_wrapped(obj["file_path"], _scalar_str, f"{path}.file_path")
        if "file_path" in obj
        else None,
        entry=_wrapped(obj["entry"], _entry_selector, f"{path}.entry")
        if "entry" in obj
        else None,
        description=obj.get("description"),
    )


def _scalar_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ParseError(
            "P2", f"expected true or false, got {type(value).__name__}", path=path
        )
    return value


def _parse_k_schedule(raw: Any, path: str) -> dict[str, Any]:
    """§2.5 ``k_schedule`` of a ``budget`` gate: how each optimizer step draws
    its budget ``k``. ``{"kind": "fixed", "k": n}`` — the same cut every step;
    ``{"kind": "uniform" | "log_uniform", "low": a, "high": b}`` — an integer
    drawn from ``[a, b]`` per step (``log_uniform`` needs ``a ≥ 1``: the draw
    is ``round(exp(U(log a, log b)))``). ``eval`` is the cut the fit's own
    held-out pass scores and its ``hard_mask_size`` counts: defaults to ``k``
    under ``fixed`` and is **required** under a sampled kind, since a sampled
    schedule implies no single number. ``k`` and ``eval`` may be swept; the
    bounds are one schedule. ``of`` (:data:`K_SCHEDULE_OF`) says what every
    number here counts — patched units (absent, the default) or kept ones."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("kind", "k", "low", "high", "eval", "of"), path)
    if "kind" not in obj:
        raise ParseError(
            "P2",
            f"k_schedule needs a 'kind': one of {list(K_SCHEDULE_KINDS)}",
            path=path,
        )
    kind = _enum(obj["kind"], K_SCHEDULE_KINDS, f"{path}.kind")
    out: dict[str, Any] = {"kind": kind}
    if kind == "fixed":
        if "k" not in obj or "low" in obj or "high" in obj:
            raise ParseError(
                "P2", "a fixed k_schedule names 'k' and no bounds", path=path
            )
        out["k"] = _wrapped(obj["k"], _non_negative_int, f"{path}.k")
    else:
        if "k" in obj or "low" not in obj or "high" not in obj:
            raise ParseError(
                "P2",
                f"a {kind} k_schedule names 'low' and 'high' and no 'k'",
                path=path,
            )
        low = _non_negative_int(obj["low"], f"{path}.low")
        high = _non_negative_int(obj["high"], f"{path}.high")
        if low > high:
            raise ParseError(
                "P2",
                f"k_schedule bounds are ordered, got low={low} > high={high}",
                path=path,
            )
        if kind == "log_uniform" and low < 1:
            raise ParseError(
                "P2",
                "a log_uniform k_schedule draws round(exp(U(log low, log high))) — "
                f"low must be at least 1, got {low}",
                path=f"{path}.low",
            )
        out["low"], out["high"] = low, high
        if "eval" not in obj:
            raise ParseError(
                "P2",
                f"a {kind} k_schedule samples its budget, so it names 'eval': the "
                "cut the fit's held-out pass scores and its hard_mask_size counts",
                path=path,
            )
    if "eval" in obj:
        out["eval"] = _wrapped(obj["eval"], _non_negative_int, f"{path}.eval")
    if "of" in obj:
        out["of"] = _enum(obj["of"], K_SCHEDULE_OF, f"{path}.of")
    return out


def _non_negative_int(value: Any, path: str) -> int:
    number = _scalar_int(value, path)
    if number < 0:
        raise ParseError(
            "P2", f"expected a non-negative integer, got {number}", path=path
        )
    return number


def _positive_number(value: Any, path: str) -> float | int:
    number = _scalar_number(value, path)
    if number <= 0:
        raise ParseError("P2", f"expected a positive number, got {number}", path=path)
    return number


def _parse_stretch(raw: Any, path: str) -> tuple[float, float]:
    """``stretch: [γ, ζ]`` of a hard-concrete gate: the interval the concrete
    sample is stretched onto before clipping to ``[0, 1]``, so ``γ < 0 < 1 < ζ``
    — a stretch that does not cover the unit interval has no mass on the poles
    and its hard split would never be reached."""
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw)
    ):
        raise ParseError("P2", "stretch is a two-number list [γ, ζ]", path=path)
    lo, hi = float(raw[0]), float(raw[1])
    if not lo < 0.0 < 1.0 < hi:
        raise ParseError(
            "P2",
            f"stretch [γ, ζ] must satisfy γ < 0 < 1 < ζ, got [{lo}, {hi}]",
            path=path,
        )
    return (lo, hi)


def _parse_gate_dead(raw: Any, path: str) -> dict[str, float | int]:
    """§2.5 ``dead``: one of :data:`GATE_DEAD_RULES`, never both — a frozen
    unit takes no gradient and a leaking one exists to keep taking it, so the
    two answers to a dead unit contradict each other on the same gate.
    ``freeze_after`` is a positive integer count of consecutive hard-off
    steps; ``leak`` is a gradient slope strictly inside ``(0, 1)`` — ``0`` is
    no rule and ``1`` makes the mask's backward that of an unsquashed θ.
    Neither is sweepable: a
    sweep over how a unit dies is a sweep over the training *procedure*, and
    the two rules' results are not points on one axis."""
    obj = _require_mapping(raw, path)
    if not obj:
        raise ParseError(
            "P2",
            '\'dead\' names a rule: {"freeze_after": n} or {"leak": eps} (§2.5)',
            path=path,
        )
    _check_keys(obj, set(GATE_DEAD_RULES), path)
    if len(obj) != 1:
        raise ParseError(
            "P2",
            "'dead' names exactly one rule — a frozen unit takes no gradient and "
            "a leaking one exists to keep taking it, so 'freeze_after' and 'leak' "
            "cannot both hold on one gate (§2.5)",
            path=path,
        )
    if "freeze_after" in obj:
        n = obj["freeze_after"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ParseError(
                "P2",
                f"'freeze_after' counts consecutive hard-off optimizer steps — a "
                f"positive integer, got {n!r}",
                path=f"{path}.freeze_after",
            )
        return {"freeze_after": int(n)}
    eps = obj["leak"]
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not 0.0 < eps < 1.0
    ):
        raise ParseError(
            "P2",
            f"'leak' is a gradient slope strictly inside (0, 1), got {eps!r} — "
            "0 is no rule and 1 is the backward of an unsquashed theta",
            path=f"{path}.leak",
        )
    return {"leak": float(eps)}


def _parse_featurizer_init(raw: Any, path: str, kind: str) -> dict[str, Any]:
    """§2.5 ``init`` — where a fit **starts**.

    On a ``subspace``: ``{"file_path": <basis bundle>, "entry": <selector>}``,
    ``entry`` optional with the same semantics as the featurizer's own.
    Neither field is sweepable — a basis is one artifact, and which of its
    columns seed the fit follows from ``k``.

    On a ``gate``: either ``{"fill": p}`` — every unit starts at **mask value**
    ``p ∈ [0, 1]``, which is ``θ = logit(p)`` under ``sigmoid`` and ``θ = p``
    under ``clamp``, the one number that means the same under both maps, and
    sweepable (whether the start decides the mask is a real question) — or
    the ``file_path`` form, a saved ``theta`` taken verbatim as the start (an
    earlier fit, a trajectory checkpoint, a hand-built prior); or
    ``{"from_scores": …}``, a start read off a per-unit **score table**
    (:func:`_parse_scores_init`). Exactly one of the three."""
    obj = _require_mapping(raw, path)
    keys = (
        ("file_path", "entry", "fill", "from_scores")
        if kind == "gate"
        else ("file_path", "entry")
    )
    _check_keys(obj, keys, path)
    starts = [key for key in ("fill", "file_path", "from_scores") if key in obj]
    if len(starts) > 1 or ("entry" in obj and starts and "file_path" not in obj):
        # `entry` belongs to the file_path start; beside another start it is
        # two starts spelled at once
        raise ParseError(
            "P2",
            "'init' names one start: {'fill': p}, {'file_path': …, 'entry': …} "
            "or {'from_scores': …} — not both",
            path=path,
        )
    if "fill" in obj:
        return {"fill": _wrapped(obj["fill"], _mask_value, f"{path}.fill")}
    if "from_scores" in obj:
        return {
            "from_scores": _parse_scores_init(obj["from_scores"], f"{path}.from_scores")
        }
    if "file_path" not in obj:
        raise ParseError(
            "P2",
            "'init' names the start to fit from — it needs a file_path"
            + (" (or, on a gate, a fill or from_scores)" if kind == "gate" else ""),
            path=path,
        )
    init: dict[str, Any] = {
        "file_path": _scalar_str(obj["file_path"], f"{path}.file_path")
    }
    if "entry" in obj:
        init["entry"] = _entry_selector(obj["entry"], f"{path}.entry")
    return init


#: §2.5 ``init.from_scores`` — the keys, and the two the table is read by
#: when unauthored. ``unit`` and ``value`` default to the column names a
#: per-unit table conventionally carries; naming them is what lets a
#: ``head_stats.json`` (``head`` / ``mean``) seed a head gate without a
#: rewrite. They are materialized into the parsed form, so an authored
#: default and an omitted one digest identically.
SCORES_INIT_KEYS: tuple[str, ...] = (
    "file_path",
    "unit",
    "value",
    "where",
    "keep",
    "scale",
)
SCORES_INIT_DEFAULTS: dict[str, str] = {"unit": "unit", "value": "value"}


def _parse_scores_init(raw: Any, path: str) -> dict[str, Any]:
    """§2.5 ``init.from_scores`` — a gate's start read off a **score table**:
    a saved metric table (one row per unit, a JSON list of row objects) such
    as ``causalab.analysis.head_stats`` writes, or an attribution scan's own
    output. ``file_path`` names it; ``unit`` is the column (or, for a
    two-axis theta such as ``expert_neuron``'s, the list of columns) holding
    each row's unit index; ``value`` the column holding its score; ``where``
    an equality filter (``{"layer": 15}``) that picks this gate's rows out of
    a table over several sites. Exactly one of ``keep`` — the top-``keep``
    units by score start on the kept pole of the gate's map, the rest on the
    dropped pole (the ``random_mask`` convention), the attribution- or
    magnitude-pruning baseline as one document — or ``scale`` — ``theta`` is
    the table's z-scored values times ``scale``, centred on the midpoint mask,
    an attribution-initialised fit (the first SGD step of a mask *is*
    path-weighted IG). Both are sweepable: how many units
    a ranking needs is the question a ``keep`` sweep asks. The coverage checks
    — ``keep`` at most the unit count, every unit named exactly once — are
    rule 32, decided where the width and the table are known."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, SCORES_INIT_KEYS, path)
    if "file_path" not in obj:
        raise ParseError(
            "P2", "from_scores names the score table: it needs a file_path", path=path
        )
    out: dict[str, Any] = {
        "file_path": _scalar_str(obj["file_path"], f"{path}.file_path")
    }
    unit = obj.get("unit", SCORES_INIT_DEFAULTS["unit"])
    if isinstance(unit, list):
        if not unit or not all(isinstance(column, str) for column in unit):
            raise ParseError(
                "P2",
                "from_scores.unit is a column name, or a non-empty list of them "
                "(one per axis of the gate's theta)",
                path=f"{path}.unit",
            )
        out["unit"] = list(unit)
    else:
        out["unit"] = _scalar_str(unit, f"{path}.unit")
    out["value"] = _scalar_str(
        obj.get("value", SCORES_INIT_DEFAULTS["value"]), f"{path}.value"
    )
    if "where" in obj:
        where = _require_mapping(obj["where"], f"{path}.where")
        for column, literal in where.items():
            if isinstance(literal, bool) or not isinstance(literal, (str, int, float)):
                raise ParseError(
                    "P2",
                    "from_scores.where maps column names to the scalar each row "
                    f"must equal, got {literal!r}",
                    path=f"{path}.where.{column}",
                )
        out["where"] = dict(where)
    modes = [key for key in ("keep", "scale") if key in obj]
    if len(modes) != 1:
        raise ParseError(
            "P2",
            "from_scores reads the table one way: 'keep' (the top-k units start "
            "kept) or 'scale' (theta is the z-scored score times scale) — exactly "
            "one of the two",
            path=path,
        )
    if "keep" in obj:
        out["keep"] = _wrapped(obj["keep"], _positive_int, f"{path}.keep")
    else:
        out["scale"] = _wrapped(obj["scale"], _positive_number, f"{path}.scale")
    return out


def _positive_int(value: Any, path: str) -> int:
    number = _scalar_int(value, path)
    if number < 1:
        raise ParseError("P2", f"expected a positive integer, got {number}", path=path)
    return number


def _mask_value(value: Any, path: str) -> float:
    """A gate's ``init.fill`` (§2.5): a number in ``[0, 1]``, the mask value
    every unit starts at. The endpoints are legal here — a clamp gate may
    start fully patched, and so may a hard-concrete gate, whose start
    ``logit((fill − γ)/(ζ − γ))`` is finite at both poles — and a sigmoid gate,
    whose start is ``logit(fill)``, refuses them at build where the map is
    known."""
    number = _scalar_number(value, path)
    if not 0.0 <= float(number) <= 1.0:
        raise ParseError(
            "P2",
            f"a gate's init.fill is a mask value in [0, 1], got {number!r}",
            path=path,
        )
    return float(number)


def _parse_featurizers(raw: Any, path: str) -> dict[str, FeaturizerSpec]:
    obj = _require_mapping(raw, path)
    return {
        name: _parse_featurizer(value, f"{path}.{name}") for name, value in obj.items()
    }


def _parse_param(raw: Any, path: str) -> ParamSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("file_path", "entry", "shape", "init", "description"), path)
    loaded = "file_path" in obj
    trainable = "shape" in obj or "init" in obj
    if "entry" in obj and not loaded:
        raise ParseError(
            "P2",
            "'entry' selects inside a loaded bundle — it needs a file_path",
            path=path,
        )
    if loaded == trainable:
        raise ParseError(
            "P2",
            "a params entry is either loaded (file_path) or trainable (shape + init)",
            path=path,
        )
    if trainable and not ("shape" in obj and "init" in obj):
        raise ParseError(
            "P2", "a trainable params entry needs both shape and init", path=path
        )
    return ParamSpec(
        file_path=_wrapped(obj["file_path"], _scalar_str, f"{path}.file_path")
        if loaded
        else None,
        entry=_wrapped(
            obj["entry"],
            lambda v, p: _entry_selector(v, p, allow_slot=True),
            f"{path}.entry",
        )
        if "entry" in obj
        else None,
        shape=_wrapped(obj["shape"], _int_list, f"{path}.shape")
        if "shape" in obj
        else None,
        init=_wrapped(obj["init"], _scalar_str, f"{path}.init")
        if "init" in obj
        else None,
        description=obj.get("description"),
    )


def _parse_params(raw: Any, path: str) -> dict[str, ParamSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_param(value, f"{path}.{name}") for name, value in obj.items()}


#: Fields of a ``code`` entry the loader derives and no one may author (§6):
#: the resolved module and its source hash, the manifest and hash of its
#: declared import closure (present only when the module imports a sibling
#: outside the ``causalab`` package), and the content digests of the declared
#: data inputs.
DERIVED_CODE_FIELDS: tuple[str, ...] = (
    "source_module",
    "source_sha256",
    "closure",
    "closure_sha256",
    "data_input_digests",
)


def _parse_row_roles(raw: Any, path: str) -> tuple[RowRole, ...]:
    """§2.8.1 — the batch's row convention, in batch order. A list, not a
    mapping: ``[clean, corrupted]`` and ``[corrupted, clean]`` are different
    conventions and JSON objects are unordered."""
    if not isinstance(raw, list):
        raise ParseError(
            "P2",
            "row_roles is a list of {'role': …, 'rows': n} in batch order",
            path=path,
        )
    roles: list[RowRole] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        where = f"{path}[{index}]"
        obj = _require_mapping(item, where)
        _check_keys(obj, ("role", "rows"), where)
        for field in ("role", "rows"):
            if field not in obj:
                raise ParseError("P2", f"a row role needs {field!r}", path=where)
        role = _scalar_str(obj["role"], f"{where}.role")
        rows = _scalar_int(obj["rows"], f"{where}.rows")
        if rows < 1:
            raise ParseError(
                "P2",
                f"row role {role!r} covers {rows} rows; it must be ≥ 1",
                path=f"{where}.rows",
            )
        if role in seen:
            raise ParseError("P2", f"duplicate row role {role!r}", path=where)
        seen.add(role)
        roles.append(RowRole(role=role, rows=rows))
    if not roles:
        raise ParseError(
            "P2",
            "row_roles is empty — omit it to say nothing about the rows, "
            "rather than saying there are none",
            path=path,
        )
    return tuple(roles)


def _parse_code_entry(raw: Any, path: str) -> CodeSpec:
    obj = _require_mapping(raw, path)
    for field in DERIVED_CODE_FIELDS:
        if field in obj:
            raise ParseError(
                "P5",
                f"{field!r} is derived from the resolved source and stamped at "
                "load, never authored (§6)",
                path=f"{path}.{field}",
            )
    _check_keys(
        obj,
        ("locator", "args", "data_inputs", "env_inputs", "row_roles", "description"),
        path,
    )
    if "locator" not in obj:
        raise ParseError("P2", "a code entry needs a 'locator'", path=path)
    args = obj.get("args", {})
    if not isinstance(args, Mapping):
        raise ParseError(
            "P2", "args is a JSON object of keyword values", path=f"{path}.args"
        )
    data_inputs = obj.get("data_inputs", {})
    if not isinstance(data_inputs, Mapping) or not all(
        isinstance(v, str) for v in data_inputs.values()
    ):
        raise ParseError(
            "P2",
            "data_inputs maps a name to a file path",
            path=f"{path}.data_inputs",
        )
    description = obj.get("description")
    if description is not None and not isinstance(description, str):
        raise ParseError("P2", "description is free text", path=f"{path}.description")
    return CodeSpec(
        locator=_wrapped(obj["locator"], _scalar_str, f"{path}.locator"),
        args=dict(args),
        data_inputs=dict(data_inputs),
        env_inputs=tuple(_str_list(obj["env_inputs"], f"{path}.env_inputs"))
        if "env_inputs" in obj
        else (),
        row_roles=_parse_row_roles(obj["row_roles"], f"{path}.row_roles")
        if "row_roles" in obj
        else (),
        description=description,
    )


def _parse_code(raw: Any, path: str) -> dict[str, CodeSpec]:
    obj = _require_mapping(raw, path)
    return {
        name: _parse_code_entry(value, f"{path}.{name}") for name, value in obj.items()
    }


def _parse_pos_field(value: Any, path: str) -> Any:
    """A read/write ``pos``: a positions-table name or an inline spec. The
    bare string ``"all"`` is the all-positions sugar, never a name — it is
    reserved (§5.3), so no entry can be declared under it."""
    if isinstance(value, str) and value != ALL_POSITIONS:
        return value
    return _parse_position_spec(value, path)


def _parse_featurizer_ref(value: Any, path: str) -> Any:
    """A ``featurizer`` reference: one name or a composition list (§2.5)."""
    if isinstance(value, str):
        return value
    return _str_list(value, path)


def _parse_read(raw: Any, path: str) -> ReadSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("site", "pos", "model", "input", "featurizer", "dims"), path)
    for field in ("site", "pos", "model", "input"):
        if field not in obj:
            raise ParseError("P2", f"a read needs {field!r}", path=path)
    return ReadSpec(
        site=_wrapped(obj["site"], _scalar_str, f"{path}.site"),
        pos=_wrapped(obj["pos"], _parse_pos_field, f"{path}.pos"),
        model=_wrapped(obj["model"], _scalar_str, f"{path}.model"),
        input=_wrapped(obj["input"], _scalar_str, f"{path}.input"),
        featurizer=_wrapped(
            obj["featurizer"], _parse_featurizer_ref, f"{path}.featurizer"
        )
        if "featurizer" in obj
        else None,
        dims=_wrapped(obj["dims"], _int_list, f"{path}.dims")
        if "dims" in obj
        else None,
    )


def _parse_reads(raw: Any, path: str) -> dict[str, ReadSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_read(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_operand(value: Any, path: str) -> Any:
    """A write operand: a read/param name or a literal scalar (§2.8)."""
    if isinstance(value, str):
        return value
    return _scalar_number(value, path)


def _parse_do(raw: Any, path: str) -> Do:
    obj = _require_mapping(raw, path)
    if len(obj) != 1:
        raise ParseError("P2", "'do' has exactly one mechanism key", path=path)
    ((mech, payload),) = obj.items()
    if mech not in MECHANISMS:
        raise ParseError(
            "P4", f"unknown mechanism {mech!r}{suggest(mech, MECHANISMS)}", path=path
        )
    p = f"{path}.{mech}"
    if mech == "swap":
        return Do(mechanism=mech, payload=_wrapped(payload, _parse_operand, p))
    if mech in ("add_scaled", "lerp"):
        options = _require_mapping(payload, p)
        _check_keys(options, ("op", "alpha"), p)
        for field in ("op", "alpha"):
            if field not in options:
                raise ParseError("P2", f"{mech} needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "op": _wrapped(options["op"], _parse_operand, f"{p}.op"),
                "alpha": _wrapped(options["alpha"], _parse_operand, f"{p}.alpha"),
            },
        )
    if mech == "affine":
        options = _require_mapping(payload, p)
        _check_keys(options, ("A", "b"), p)
        for field in ("A", "b"):
            if field not in options:
                raise ParseError("P2", f"affine needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "A": _wrapped(options["A"], _scalar_str, f"{p}.A"),
                "b": _wrapped(options["b"], _scalar_str, f"{p}.b"),
            },
        )
    if mech == "gaussian":
        options = _require_mapping(payload, p)
        _check_keys(options, ("seed", "scale", "axis"), p)
        for field in ("seed", "scale", "axis"):
            if field not in options:
                raise ParseError("P2", f"gaussian needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "seed": _wrapped(options["seed"], _scalar_int, f"{p}.seed"),
                "scale": _wrapped(options["scale"], _scalar_number, f"{p}.scale"),
                "axis": _wrapped(
                    options["axis"],
                    lambda v, pp: _enum(v, ("tp_duplicated", "tp_split"), pp),
                    f"{p}.axis",
                ),
            },
        )
    if mech == "renormalize":
        if payload is not True:
            raise ParseError(
                "P2", 'renormalize is written {"renormalize": true}', path=p
            )
        return Do(mechanism=mech, payload=True)
    if mech == "clamp":
        options = _require_mapping(payload, p)
        _check_keys(options, ("lo", "hi"), p)
        for field in ("lo", "hi"):
            if field not in options:
                raise ParseError("P2", f"clamp needs {field!r}", path=p)
        return Do(
            mechanism=mech,
            payload={
                "lo": _wrapped(options["lo"], _scalar_number, f"{p}.lo"),
                "hi": _wrapped(options["hi"], _scalar_number, f"{p}.hi"),
            },
        )
    # pytorch_fn — names a `code` declaration, never a bare qualname (§2.8.1).
    # A qualname alone left the function's body, arguments, file reads,
    # environment reads and row convention outside the digest; the declaration
    # is what carries them.
    options = _require_mapping(payload, p)
    if "qualname" in options:
        raise ParseError(
            "P3",
            "pytorch_fn no longer takes a bare 'qualname': write "
            '{"pytorch_fn": {"code": "<name>"}} and declare the function in '
            "the document's 'code' section, so its source, arguments, "
            "declared inputs and row roles are in the digest (§2.8.1)",
            path=f"{p}.qualname",
        )
    _check_keys(options, ("code",), p)
    if "code" not in options:
        raise ParseError(
            "P2",
            "pytorch_fn names a 'code' declaration — {\"pytorch_fn\": "
            '{"code": "<name>"}} — so the function\'s source, arguments, '
            "declared inputs and row roles are in the digest (§2.8.1)",
            path=p,
        )
    return Do(
        mechanism=mech,
        payload={"code": _wrapped(options["code"], _scalar_str, f"{p}.code")},
    )


def _parse_ragged(raw: Any, path: str) -> str:
    """``{"policy": <RAGGED_POLICIES>}`` (§2.8): the one-key object a write's
    ``ragged`` field holds. Vocabulary only — whether the window *is* ragged
    is the executor's to decide on the encoded batch (§5 rule 19)."""
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("policy",), path)
    if "policy" not in obj:
        raise ParseError(
            "P2",
            f"a {RAGGED_FIELD!r} declaration needs 'policy' "
            f"(one of {list(RAGGED_POLICIES)})",
            path=path,
        )
    return _enum(obj["policy"], RAGGED_POLICIES, f"{path}.policy")


def _parse_write(raw: Any, path: str) -> WriteSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("site", "pos", "featurizer", "dims", "do", RAGGED_FIELD), path)
    for field in ("site", "pos", "do"):
        if field not in obj:
            raise ParseError("P2", f"a write needs {field!r}", path=path)
    ragged: str | None = None
    if RAGGED_FIELD in obj:
        # how a ragged window lands is an execution strategy, fixed per
        # campaign like `minimum_count` — never a research variable, so a
        # sweep wrapper is rule 14 here
        ragged = _wrapped(
            obj[RAGGED_FIELD],
            _parse_ragged,
            f"{path}.{RAGGED_FIELD}",
            allow_sweep=False,
        )
    return WriteSpec(
        site=_wrapped(obj["site"], _scalar_str, f"{path}.site"),
        pos=_wrapped(obj["pos"], _parse_pos_field, f"{path}.pos"),
        do=_parse_do(obj["do"], f"{path}.do"),
        featurizer=_wrapped(
            obj["featurizer"], _parse_featurizer_ref, f"{path}.featurizer"
        )
        if "featurizer" in obj
        else None,
        dims=_wrapped(obj["dims"], _int_list, f"{path}.dims")
        if "dims" in obj
        else None,
        ragged=ragged,
    )


def _parse_writes(raw: Any, path: str) -> dict[str, WriteSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_write(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_im(raw: Any, path: str) -> IMSpec:
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("input", "writes"), path)
    for field in ("input", "writes"):
        if field not in obj:
            raise ParseError("P2", f"an intervened_model needs {field!r}", path=path)
    return IMSpec(
        input=_wrapped(obj["input"], _scalar_str, f"{path}.input"),
        writes=_wrapped(obj["writes"], _str_list, f"{path}.writes"),
    )


def _parse_intervened_models(raw: Any, path: str) -> dict[str, IMSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_im(value, f"{path}.{name}") for name, value in obj.items()}


def _parse_metric(raw: Any, path: str) -> MetricSpec:
    obj = _require_mapping(raw, path)
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in METRIC_KINDS:
        raise ParseError(
            "P4",
            f"unknown metric kind {kind!r}{suggest(str(kind), METRIC_KINDS)}",
            path=f"{path}.kind",
        )
    extra = METRIC_FIELDS[kind]
    takes_token_form = kind in TOKEN_COLUMN_METRIC_KINDS
    optional = OPTIONAL_METRIC_FIELDS.get(kind, ())
    # a kind that may carry `restrict` resolves answer strings only when it
    # does, so the key is *allowed* here and *required or refused* below
    _check_keys(
        obj,
        (
            "kind",
            "of",
            *extra,
            *optional,
            *(("token_form",) if takes_token_form or "restrict" in optional else ()),
            *IDENTITY_COLUMNS,
            MINIMUM_COUNT_FIELD,
        ),
        path,
    )
    unit, estimand_version = _parse_metric_identity(obj, kind, path)
    minimum_count: int | None = None
    if MINIMUM_COUNT_FIELD in obj:
        # a decision threshold is a preregistered bar, not a research
        # variable — fixed per campaign like `token_form`, never swept
        minimum_count = _wrapped(
            obj[MINIMUM_COUNT_FIELD],
            _scalar_int,
            f"{path}.{MINIMUM_COUNT_FIELD}",
            allow_sweep=False,
        )
        if minimum_count < 1:
            raise ParseError(
                "P2",
                f"{MINIMUM_COUNT_FIELD} must be a positive integer, got "
                f"{minimum_count} — a decision rule that needs no eligible row "
                "declares no threshold",
                path=f"{path}.{MINIMUM_COUNT_FIELD}",
            )
    if "of" not in obj:
        raise ParseError("P2", "a metric needs 'of' (a read name)", path=path)
    token_form: Any = "auto"
    if takes_token_form and "token_form" not in obj:
        raise ParseError(
            "P2",
            f"metric kind {kind!r} needs 'token_form' — how its string answers "
            f"become token ids is a property of the model's tokenizer, not "
            f"something a document may leave to a default. One of "
            f"{list(TOKEN_FORMS)}: 'space_prefixed' for an answer the model "
            f"emits after a space (the common case), 'bare' for one it does "
            f"not, 'auto' to keep the historical space-prefixed-first guess. "
            f"The guess has been wrong in four measured ways — a leading space "
            f"(' ?' vs '?'), punctuation that merges with the token before it, "
            f"two authored forms resolving to one id, and multi-token digits — "
            f"so 'auto' is now something a document says on purpose",
            path=path,
        )
    if "token_form" in obj:
        token_form = _wrapped(
            obj["token_form"],
            lambda v, p: _enum(v, TOKEN_FORMS, p),
            f"{path}.token_form",
            allow_sweep=False,
        )
    fields: dict[str, Any] = {}
    for field in extra:
        if field not in obj:
            hint = (
                " — the ranking rule is mandatory because the read decides it: "
                f"{VOCAB_TOP_K_RANKING!r} (softmax the vocabulary, lm_head reads "
                "only), 'value' (largest signed entries) or 'abs_value' (largest "
                "magnitude). A pre-'by' document that scored logits meant "
                f"{VOCAB_TOP_K_RANKING!r}"
                if (kind, field) == ("top_k", "by")
                else ""
            )
            raise ParseError(
                "P2", f"metric kind {kind!r} needs {field!r}{hint}", path=path
            )
        if field == "k":
            fields[field] = _wrapped(obj[field], _scalar_int, f"{path}.{field}")
        elif field == "by":
            # ranking rule, not a dataset column — and not sweepable: a sweep
            # over `by` would fork a campaign on how a plot is read rather
            # than on a research variable (same reasoning as `token_form`).
            fields[field] = _wrapped(
                obj[field],
                lambda v, p: _enum(v, TOP_K_RANKINGS, p),
                f"{path}.{field}",
                allow_sweep=False,
            )
        elif field == "groups":
            fields[field] = _wrapped(obj[field], _any_leaf, f"{path}.{field}")
        elif field == "tokens":
            # the run's answer space, not a research variable — fixed per
            # campaign like `token_form` and `top_k.by`: a sweep over it would
            # fork the campaign on what gets saved rather than on a hypothesis
            fields[field] = _wrapped(
                obj[field], _token_list, f"{path}.{field}", allow_sweep=False
            )
        else:
            fields[field] = _wrapped(obj[field], _scalar_str, f"{path}.{field}")
    for field in optional:
        if field == "restrict":
            # no default: absent means unrestricted, and stays absent (§2.10)
            if field in obj:
                fields[field] = _parse_restrict(obj[field], f"{path}.{field}")
            continue
        value = obj.get(field, METRIC_FIELD_DEFAULTS[(kind, field)])
        fields[field] = _wrapped(value, _scalar_str, f"{path}.{field}")
        if (kind, field) == ("match", "mode") and fields[field] not in MATCH_MODES:
            raise ParseError(
                "P4",
                f"unknown match mode {fields[field]!r} — one of "
                f"{list(MATCH_MODES)}{suggest(str(fields[field]), MATCH_MODES)}. "
                "(A task's 'prefix' string_mode is 'first_token' here — the "
                "translation table in sec. 2.10.)",
                path=f"{path}.{field}",
            )
    if token_form == "id" and (
        kind in {"class_probs", "token_logits"} or fields.get("mode") == "first_token"
    ):
        raise ParseError(
            "P2",
            "token_form='id' scores exact integer token IDs from dataset columns; "
            "it does not accept literal token lists or first_token matching",
            path=f"{path}.token_form",
        )
    if "restrict" in optional:
        # `restrict` is what makes the kind resolve a string to a token id, so
        # it decides `token_form` per document: required with it, meaningless
        # without — the rule the token-column kinds and `kl` each follow by
        # kind, applied here by field
        if "restrict" in fields and "token_form" not in obj:
            raise ParseError(
                "P2",
                f"metric kind {kind!r} with 'restrict' needs 'token_form' — the "
                "answer strings resolve to token ids, and how is a property of "
                f"the model's tokenizer, one of {list(TOKEN_FORMS)} (§2.10)",
                path=path,
            )
        if "restrict" not in fields and "token_form" in obj:
            raise ParseError(
                "P3",
                f"metric kind {kind!r} without 'restrict' compares two whole "
                "distributions and resolves no string — 'token_form' has nothing "
                "to decide; drop it, or add 'restrict'",
                path=f"{path}.token_form",
            )
    return MetricSpec(
        kind=kind,
        of=_wrapped(obj["of"], _scalar_str, f"{path}.of"),
        fields=fields,
        token_form=token_form,
        unit=unit,
        estimand_version=estimand_version,
        minimum_count=minimum_count,
    )


def _parse_restrict(value: Any, path: str) -> Any:
    """``js.restrict`` (§2.10): the answer set the two distributions are
    restricted to and renormalised over — a **column name** (a string), whose
    per-row value is a list of answer strings, or a **literal list** of answer
    strings for an answer space that is one for the whole run (the
    ``token_logits.tokens`` shape). The shape decides, as it does for
    ``class_probs.groups``. Not sweepable in either form: an answer space is
    not a research variable, so a sweep wrapper is refused as a shape."""
    if isinstance(value, str):
        return _scalar_str(value, path)
    if isinstance(value, list):
        return _token_list(value, path)
    raise ParseError(
        "P2",
        "restrict is a column name (whose per-row value is a list of answer "
        "strings) or a literal list of answer strings — not a sweep: an answer "
        "space is not a research variable",
        path=path,
    )


def _parse_metric_identity(
    obj: Mapping[str, Any], kind: str, path: str
) -> tuple[str | None, str | None]:
    """The optional ``unit`` / ``estimand_version`` of a metric (§2.10).

    Both are parse-level rejections of the existing kinds, no §5 rule: an
    off-vocabulary unit or a malformed identifier is ``P4`` with suggestions
    (``estimand.py`` owns both vocabularies), and a value the kind does not
    compute — ``percentage_points`` on a ``match``, ``ratio_of_sums/v1`` on a
    ``kl`` — is ``P4`` too, naming what the kind computes: the admissible set
    for a kind is exactly its own unit and its own identifier. Neither field
    is sweepable — an identity is not a research variable."""
    unit: str | None = None
    if "unit" in obj:
        unit = _wrapped(
            obj["unit"],
            lambda v, p: _enum(v, UNITS, p),
            f"{path}.unit",
            allow_sweep=False,
        )
        own = METRIC_UNITS[kind]
        if own is None:
            raise ParseError(
                "P4",
                f"metric kind {kind!r} produces no scalar and has no unit — "
                f"drop 'unit' ({unit!r})",
                path=f"{path}.unit",
            )
        if unit != own:
            raise ParseError(
                "P4",
                f"metric kind {kind!r} computes a value in {own!r}, not {unit!r} — "
                "a document may state a kind's unit, not change it",
                path=f"{path}.unit",
            )
    estimand_version: str | None = None
    if "estimand_version" in obj:
        authored = _wrapped(
            obj["estimand_version"],
            lambda v, _p: v,
            f"{path}.estimand_version",
            allow_sweep=False,
        )
        try:
            parse_identifier(authored)
        except EstimandError as err:
            raise ParseError("P4", str(err), path=f"{path}.estimand_version") from err
        own_identity = metric_identity(kind)
        if authored != own_identity:
            raise ParseError(
                "P4",
                f"metric kind {kind!r} computes {own_identity!r}, not {authored!r} — "
                "a kind is one arithmetic, so its identifier is derived; a "
                "reduction over the saved table names its own (workflow spec §2.6)",
                path=f"{path}.estimand_version",
            )
        estimand_version = str(authored)
    return unit, estimand_version


def _parse_metrics(raw: Any, path: str) -> dict[str, MetricSpec]:
    obj = _require_mapping(raw, path)
    return {name: _parse_metric(value, f"{path}.{name}") for name, value in obj.items()}


#: ``l0`` is the expected kept fraction of a ``hard_concrete`` gate's sampled
#: mask (§2.11), the Louizos et al. expected-L0 ``σ(θ − β·log(−γ/ζ))`` — legal
#: only under that map, as ``l1`` is legal only under the deterministic ones
#: (rule 4): on a deterministic map the relaxed mask is already the kept
#: probability and its mean is ``l1``, so ``l0`` there would be a second
#: spelling of one computation. Gates only.
REGULARIZER_KINDS: tuple[str, ...] = ("l1", "l2", "l0")
#: §2.11 ``reduce`` on a regularizer term: how the per-unit penalized
#: quantities, concatenated over the term's featurizers, become one number.
REGULARIZER_REDUCTIONS: tuple[str, ...] = ("mean", "sum")
#: §2.11 ``costs`` on a regularizer term, the word form: a rule for the
#: per-target multiplier instead of a table. ``parameter_count`` is ``1 /
#: (the target's penalized element count)``.
REGULARIZER_COSTS: tuple[str, ...] = ("parameter_count",)


def _parse_regularizer_names(value: Any, path: str) -> tuple[str, ...]:
    """The featurizers one regularizer penalizes together (§2.11): a name, or
    a non-empty list of distinct names. The list is one penalty over the
    concatenation of their parameters, so a repeated name would count a
    featurizer twice and an empty list would penalize nothing — both refuse
    here rather than fit quietly."""
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ParseError(
            "P2",
            "a regularizer names one featurizer (or dotted slot) or a list of "
            "featurizer names",
            path=path,
        )
    if not value:
        raise ParseError(
            "P2", "a regularizer list names at least one featurizer", path=path
        )
    seen: set[str] = set()
    for name in value:
        if name in seen:
            raise ParseError(
                "P2",
                f"a regularizer list names each featurizer once ({name!r} repeats)",
                path=path,
            )
        seen.add(name)
    return tuple(value)


def _parse_reduce(value: Any, path: str) -> str:
    return _enum(_scalar_str(value, path), REGULARIZER_REDUCTIONS, path)


#: The optional fields a regularizer term carries beside its kind.
_REGULARIZER_OPTIONS: tuple[str, ...] = ("reduce", "costs")


def _parse_costs(value: Any, path: str) -> Mapping[str, float] | str:
    """§2.11 ``costs``: ``{target: c}`` with every ``c`` a finite positive
    number — a cost of 0 would name a target and penalize nothing, so name
    fewer targets instead — or one word of :data:`REGULARIZER_COSTS`. Whether
    the keys are the term's own targets is a reference and rule 4's.

    The two sweep refusals below are reachable at the authoring gate
    because ``compile.STAGES`` runs ``gate`` (the strict parse, sweep
    wrappers intact) before ``expand``; were the stages ever reordered, a
    ``{"sweep": …}`` cost would be expanded into points and swept silently
    instead of refused here."""
    if isinstance(value, str):
        return _enum(value, REGULARIZER_COSTS, path)
    if not isinstance(value, Mapping):
        raise ParseError(
            "P2",
            "'costs' is {target: positive number} — a multiplier per penalized "
            f"featurizer — or one of {list(REGULARIZER_COSTS)}",
            path=path,
        )
    if not value:
        raise ParseError("P2", "'costs' names at least one target", path=path)
    out: dict[str, float] = {}
    for key, cost in value.items():
        if not isinstance(key, str) or not key:
            raise ParseError(
                "P2", f"'costs' keys are target names, got {key!r}", path=path
            )
        if key == "sweep":
            raise ParseError(
                "P2",
                "'costs' is not swept — a cost is a literal number per target; "
                "sweep the term's weight instead",
                path=path,
            )
        # convert inside the refusal: an integer too large for a double
        # (JSON keeps it an int) must leave as P2, not as OverflowError
        try:
            scaled = float(cost)
        except (OverflowError, TypeError, ValueError):
            scaled = float("nan")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(scaled)
            or scaled <= 0.0
        ):
            swept = isinstance(cost, Mapping) and "sweep" in cost
            raise ParseError(
                "P2",
                f"a cost is a finite positive number, got {cost!r} — "
                + (
                    "a cost is a literal, not swept; sweep the term's weight instead"
                    if swept
                    else "a target that should cost nothing is a target to leave out"
                ),
                path=f"{path}.{key}",
            )
        out[key] = scaled
    return out


def _parse_regularizer(
    value: Any, path: str
) -> tuple[tuple[str, tuple[str, ...]], str | None, Mapping[str, float] | str | None]:
    """The positional regularizer, ``{"l1"|"l2"|"l0": names}`` with optional
    ``"reduce"`` and ``"costs"`` (§2.11): returns ``((kind, names), reduce,
    costs)``."""
    reg = _require_mapping(value, path)
    if "constraint" in reg:
        raise ParseError(
            "P2",
            "a 'constraint' term is addressed by name (its duals are traced under "
            "it) — spell the objective in its named form",
            path=f"{path}.constraint",
        )
    kinds = [key for key in reg if key in REGULARIZER_KINDS]
    extra = [
        key
        for key in reg
        if key not in REGULARIZER_KINDS and key not in _REGULARIZER_OPTIONS
    ]
    if len(kinds) != 1 or extra:
        odd = extra[0] if extra else (next(iter(reg)) if len(reg) == 1 else None)
        raise ParseError(
            "P2",
            'a regularizer is {"l1": names}, {"l2": names} or {"l0": names}, '
            'optionally with "reduce" and "costs"'
            + (suggest(odd, REGULARIZER_KINDS) if isinstance(odd, str) else ""),
            path=path,
        )
    (kind,) = kinds
    reduce = _parse_reduce(reg["reduce"], f"{path}.reduce") if "reduce" in reg else None
    costs = _parse_costs(reg["costs"], f"{path}.costs") if "costs" in reg else None
    return (kind, _parse_regularizer_names(reg[kind], f"{path}.{kind}")), reduce, costs


def _parse_objective_term(value: Any, path: str) -> ObjectiveTerm:
    """The positional form: ``[weight, metric-name]`` or
    ``[weight, {<regularizer kind>: names}]`` (:data:`REGULARIZER_KINDS`)."""
    if not isinstance(value, list) or len(value) != 2:
        raise ParseError(
            "P2", "an objective term is [weight, metric-or-regularizer]", path=path
        )
    weight = _wrapped(value[0], _scalar_number, f"{path}[0]")
    if isinstance(value[1], str):
        return ObjectiveTerm(weight=weight, metric=value[1])
    regularizer, reduce, costs = _parse_regularizer(value[1], f"{path}[1]")
    return ObjectiveTerm(
        weight=weight, regularizer=regularizer, reduce=reduce, costs=costs
    )


def _refuse_sweep(value: Any, path: str, what: str) -> None:
    """A §2.11 ``constraint`` field is not an axis: one constraint, one target.
    A ``{"sweep": …}`` wrapper anywhere in the block is refused by name, not
    as ``_scalar_number``'s "expected a number, got dict" or ``_check_keys``'s
    "unknown key 'sweep'", so the author learns the rule, not the type.
    Reachable for the reason ``_parse_costs`` records: the gate parses before
    ``expand``, wrappers intact.

    The ``{"axis": …}`` spelling of §3.2 reaches here two ways. A document
    that declares an ``axes`` group has it lowered to the sweep it stands for
    by ``compile._axes`` *before* the gate, so the message names ``axis`` too
    — the only word that tells that author what was refused. A document with
    no ``axes`` group (``has_axes`` is the stage's guard) hands the wrapper to
    the gate intact, in a real compile as much as in a test that skips it —
    that is what the ``"axis" in value`` check is for."""
    if isinstance(value, Mapping) and ("sweep" in value or "axis" in value):
        raise ParseError(
            "P2",
            f"a constraint's {what} is not swept (nor an `axis`) — one constraint, "
            "one target; author one document per value, or override it per run "
            "with `set` (§2.11)",
            path=path,
        )


def _parse_constraint(raw: Any, path: str) -> ConstraintSpec:
    """§2.11 ``constraint``: ``{"target": t, "dual": {"lr": η, "init"?: [λ₁,
    λ₂]}}``. The target is a density — a fraction in (0, 1); the dual lr is
    a positive number; ``init`` is two finite numbers with ``λ₂ ≥ 0``, absent
    for ``(0, 0)``."""
    _refuse_sweep(raw, path, "block")
    obj = _require_mapping(raw, path)
    _check_keys(obj, ("target", "dual"), path)
    for field in ("target", "dual"):
        if field not in obj:
            raise ParseError("P2", f"a constraint needs {field!r}", path=path)
    _refuse_sweep(obj["target"], f"{path}.target", "'target'")
    _refuse_sweep(obj["dual"], f"{path}.dual", "'dual' pair")
    target = _scalar_number(obj["target"], f"{path}.target")
    if not 0.0 < float(target) < 1.0:
        raise ParseError(
            "P2",
            f"a target density is a fraction in (0, 1) — the mask mean the fit is "
            f"held to — got {target!r}",
            path=f"{path}.target",
        )
    dual = _require_mapping(obj["dual"], f"{path}.dual")
    _check_keys(dual, ("lr", "init"), f"{path}.dual")
    if "lr" not in dual:
        raise ParseError(
            "P2",
            "the dual pair needs 'lr' — the rate (λ₁, λ₂) ascend at",
            path=f"{path}.dual",
        )
    _refuse_sweep(dual["lr"], f"{path}.dual.lr", "'dual.lr'")
    lr = _scalar_number(dual["lr"], f"{path}.dual.lr")
    if not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ParseError(
            "P2", f"dual.lr is a positive number, got {lr!r}", path=f"{path}.dual.lr"
        )
    init: tuple[float, float] | None = None
    if "init" in dual:
        raw_init = dual["init"]
        _refuse_sweep(raw_init, f"{path}.dual.init", "'dual.init'")
        if (
            not isinstance(raw_init, list)
            or len(raw_init) != 2
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(float(v))
                for v in raw_init
            )
        ):
            raise ParseError(
                "P2",
                "dual.init is [λ₁, λ₂] — two finite numbers — or absent for [0, 0]",
                path=f"{path}.dual.init",
            )
        if float(raw_init[1]) < 0.0:
            # λ₂ is the quadratic penalty's coefficient: negative, the term is
            # concave and the mask gradient points away from the target until
            # the ascent carries λ₂ back above zero. λ₁ ranges over ℝ — an
            # equality multiplier — and the fit takes it negative itself
            raise ParseError(
                "P2",
                "dual.init's λ₂ is the quadratic penalty's coefficient — negative "
                "inverts it (the gate driven away from the target); λ₁ may be "
                "negative, the fit takes it there itself (§2.11)",
                path=f"{path}.dual.init",
            )
        init = (float(raw_init[0]), float(raw_init[1]))
    return ConstraintSpec(target=float(target), dual_lr=float(lr), dual_init=init)


def _parse_named_objective_term(name: str, value: Any, path: str) -> ObjectiveTerm:
    """The named form: ``{"weight": w, "metric": name}``, ``{"weight": w,
    "l1"|"l2"|"l0": names}`` or — on a mask term — ``{"l1"|"l0": names,
    "constraint": {…}}``, which carries its dual pair instead of a weight
    (§2.11). A named term's weight sits under mapping keys, so it is a field
    of a named entry and may be swept (§3); the positional form's weight is
    inside a list and may not, and nothing inside ``constraint`` is swept
    (:func:`_refuse_sweep`)."""
    obj = _require_mapping(value, path)
    _check_keys(
        obj,
        ("weight", "metric", *REGULARIZER_KINDS, *_REGULARIZER_OPTIONS, "constraint"),
        path,
    )
    constraint = (
        _parse_constraint(obj["constraint"], f"{path}.constraint")
        if "constraint" in obj
        else None
    )
    if constraint is None and "weight" not in obj:
        raise ParseError("P2", "an objective term needs 'weight'", path=path)
    weight = (
        _wrapped(obj["weight"], _scalar_number, f"{path}.weight")
        if "weight" in obj
        else None
    )
    kinds = [
        key
        for key in obj
        if key not in ("weight", "constraint") and key not in _REGULARIZER_OPTIONS
    ]
    if len(kinds) != 1:
        raise ParseError(
            "P2",
            "an objective term is a weight and exactly one of 'metric', 'l1', 'l2', 'l0'",
            path=path,
        )
    reduce = _parse_reduce(obj["reduce"], f"{path}.reduce") if "reduce" in obj else None
    costs = _parse_costs(obj["costs"], f"{path}.costs") if "costs" in obj else None
    if kinds == ["metric"]:
        # each option refused for its own reason: P2 text is the interface
        for option, parsed, why in (
            (
                "reduce",
                reduce,
                "a metric term is already one number per row, reduced by the "
                "metric's own rule",
            ),
            (
                "costs",
                costs,
                "a metric term has no per-target quantities to scale — scaling "
                "one metric is what its weight is",
            ),
            (
                "constraint",
                constraint,
                "a metric has no mask density to hold to a target",
            ),
        ):
            if parsed is not None:
                raise ParseError(
                    "P2",
                    f"'{option}' is a regularizer's field — {why}",
                    path=f"{path}.{option}",
                )
        metric = obj["metric"]
        if not isinstance(metric, str):
            raise ParseError("P2", "'metric' is a metric name", path=f"{path}.metric")
        return ObjectiveTerm(weight=weight, metric=metric, name=name)
    (kind,) = kinds
    if constraint is not None:
        if weight is not None:
            raise ParseError(
                "P2",
                "a constraint term has no weight — its multipliers are the dual pair "
                "(λ₁, λ₂) the fit ascends (§2.11); drop 'weight' or drop 'constraint'",
                path=f"{path}.weight",
            )
        if kind not in ("l1", "l0"):
            raise ParseError(
                "P2",
                "a target density is a mask quantity — 'l1' (the soft mask's mean) "
                f"or 'l0' (the expected kept fraction), not {kind!r}",
                path=f"{path}.constraint",
            )
        if reduce == "sum":
            raise ParseError(
                "P2",
                "a constraint's target is a density (a fraction), and under 'sum' "
                "the term is a count — spell 'mean' or leave reduce unauthored",
                path=f"{path}.reduce",
            )
        if costs == "parameter_count":
            # the same category error as `reduce: sum`, and worse: with the
            # density divided by N the gap is negative from the first update,
            # λ₁ descends and the fit drives the gate toward *full* density
            raise ParseError(
                "P2",
                "a constraint's target is a density, and 'parameter_count' divides "
                "each target's quantities by its element count — the term is no "
                "longer a fraction; drop 'costs' or drop 'constraint' (a costs "
                "table is fine: the target is then held on the cost-weighted "
                "density)",
                path=f"{path}.costs",
            )
    names = _parse_regularizer_names(obj[kind], f"{path}.{kind}")
    return ObjectiveTerm(
        weight=weight,
        regularizer=(kind, names),
        name=name,
        reduce=reduce,
        costs=costs,
        constraint=constraint,
    )


def _parse_objective(raw: Any, path: str) -> tuple[ObjectiveTerm, ...]:
    if isinstance(raw, list) and raw:
        return tuple(
            _parse_objective_term(t, f"{path}[{i}]") for i, t in enumerate(raw)
        )
    if isinstance(raw, dict) and raw and "sweep" not in raw:
        return tuple(
            _parse_named_objective_term(name, term, f"{path}.{name}")
            for name, term in raw.items()
        )
    raise ParseError(
        "P2",
        "train.objective is a non-empty list of [weight, term] pairs or a "
        "non-empty object of named {weight, term} entries",
        path=path,
    )


def _parse_counter(value: Any, path: str) -> dict[str, Any]:
    obj = _require_mapping(value, path)
    _check_keys(obj, ("epochs", "updates"), path)
    if len(obj) != 1:
        raise ParseError("P2", "expected exactly one of epochs/updates", path=path)
    ((unit, count),) = obj.items()
    return {unit: _wrapped(count, _scalar_int, f"{path}.{unit}")}


#: The optimizer fields a fit may set **per trained parameter** (§2.11): one
#: number for everything in ``train.params``, or a mapping keyed by the entries
#: of ``train.params`` — a rotation at 1e-3 beside a gate at 0.1 in one fit.
PER_PARAMS_OPTIMIZER_FIELDS: tuple[str, ...] = ("lr", "weight_decay")


def _parse_per_params_number(value: Any, params: Sequence[str], path: str) -> None:
    """``lr`` / ``weight_decay``: a scalar (or a sweep of one) for every trained
    parameter, or a mapping ``{<params entry>: number}`` naming **every** entry
    of ``train.params`` exactly once — a key that is not a trained parameter
    is refused (it would silently apply to nothing), and an entry left out is
    refused rather than given a hidden default (there is none to give)."""
    if isinstance(value, Mapping) and "sweep" not in value:
        keys = set(value)
        declared = set(params)
        unknown = sorted(keys - declared)
        if unknown:
            raise ParseError(
                "P2",
                f"per-parameter optimizer setting names {unknown}, which train.params "
                f"does not train — keys are entries of train.params ({sorted(declared)})",
                path=path,
            )
        missing = sorted(declared - keys)
        if missing:
            raise ParseError(
                "P2",
                f"per-parameter optimizer setting leaves {missing} without a value — "
                "name every entry of train.params, or give one number for all",
                path=path,
            )
        for key, number in value.items():
            _wrapped(number, _scalar_number, f"{path}.{key}")
        return
    _wrapped(value, _scalar_number, path)


def _parse_train(raw: Any, path: str) -> TrainSpec:
    obj = _require_mapping(raw, path)
    _check_keys(
        obj,
        (
            "objective",
            "params",
            "optimizer",
            "steps",
            "batch",
            "anneal",
            "control",
            "phases",
            "precision",
            "eval",
            "early_stop",
            "checkpoint",
            "seed",
        ),
        path,
    )
    for field in ("objective", "params", "optimizer", "steps", "batch"):
        if field not in obj:
            raise ParseError("P2", f"train needs {field!r}", path=path)
    objective = _parse_objective(obj["objective"], f"{path}.objective")
    params = _str_list(obj["params"], f"{path}.params")
    optimizer = _require_mapping(obj["optimizer"], f"{path}.optimizer")
    _check_keys(optimizer, OPTIMIZER_FIELDS, f"{path}.optimizer")
    if "name" not in optimizer or "lr" not in optimizer:
        raise ParseError(
            "P2", "train.optimizer needs 'name' and 'lr'", path=f"{path}.optimizer"
        )
    _enum(optimizer["name"], tuple(OPTIMIZER_DEFAULTS), f"{path}.optimizer.name")
    for field in ("lr", "weight_decay"):
        if field in optimizer:
            _parse_per_params_number(
                optimizer[field], params, f"{path}.optimizer.{field}"
            )
    for field in ("eps", "momentum", "clip_grad_norm"):
        if field in optimizer:
            _wrapped(optimizer[field], _scalar_number, f"{path}.optimizer.{field}")
    if "betas" in optimizer:
        betas = optimizer["betas"]
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or not all(
                isinstance(b, (int, float)) and not isinstance(b, bool) for b in betas
            )
        ):
            raise ParseError(
                "P2",
                "optimizer betas is a two-number list",
                path=f"{path}.optimizer.betas",
            )
    schedule = "constant"
    if "schedule" in optimizer:
        schedule = _enum(
            optimizer["schedule"], OPTIMIZER_SCHEDULES, f"{path}.optimizer.schedule"
        )
    if "warmup_frac" in optimizer:
        if schedule != "linear_warmup_decay":
            raise ParseError(
                "P2",
                "'warmup_frac' belongs to schedule 'linear_warmup_decay' — a constant "
                "schedule warms nothing up (§2.11)",
                path=f"{path}.optimizer.warmup_frac",
            )
        frac = _scalar_number(optimizer["warmup_frac"], f"{path}.optimizer.warmup_frac")
        if not 0.0 <= float(frac) < 1.0:
            raise ParseError(
                "P2",
                f"warmup_frac is a fraction of the updates in [0, 1), got {frac}",
                path=f"{path}.optimizer.warmup_frac",
            )
    steps = _parse_counter(obj["steps"], f"{path}.steps")
    batch = _require_mapping(obj["batch"], f"{path}.batch")
    _check_keys(batch, ("pairs",), f"{path}.batch")
    if "pairs" not in batch:
        raise ParseError(
            "P2",
            "train.batch counts base+counterfactual pairs: {'pairs': n}",
            path=f"{path}.batch",
        )
    anneal = None
    if "anneal" in obj:
        anneal_raw = _require_mapping(obj["anneal"], f"{path}.anneal")
        anneal = {
            key: _wrapped(
                value,
                lambda v, p: _parse_anneal_entry(v, p),
                f"{path}.anneal.{key}",
            )
            for key, value in anneal_raw.items()
        }
    control = None
    if "control" in obj:
        control_raw = _require_mapping(obj["control"], f"{path}.control")
        if not control_raw:
            raise ParseError(
                "P2", "train.control names at least one target", path=f"{path}.control"
            )
        control = {
            key: _parse_control(value, f"{path}.control.{key}")
            for key, value in control_raw.items()
        }
    phases = None
    if "phases" in obj:
        phases = _parse_phases(obj["phases"], params, f"{path}.phases")
    precision = None
    if "precision" in obj:
        precision_raw = _require_mapping(obj["precision"], f"{path}.precision")
        _check_keys(precision_raw, ("feature", "loss"), f"{path}.precision")
        precision = {
            key: _wrapped(
                value,
                lambda v, p: _enum(v, PRECISION_DTYPES, p),
                f"{path}.precision.{key}",
            )
            for key, value in precision_raw.items()
        }
    eval_spec = None
    if "eval" in obj:
        eval_raw = _require_mapping(obj["eval"], f"{path}.eval")
        _check_keys(eval_raw, ("every", "split", "metrics"), f"{path}.eval")
        for field in ("every", "split", "metrics"):
            if field not in eval_raw:
                raise ParseError(
                    "P2", f"train.eval needs {field!r}", path=f"{path}.eval"
                )
        eval_spec = {
            "every": _parse_counter(eval_raw["every"], f"{path}.eval.every"),
            "split": _wrapped(eval_raw["split"], _scalar_str, f"{path}.eval.split"),
            "metrics": _str_list(eval_raw["metrics"], f"{path}.eval.metrics"),
        }
    early_stop = None
    if "early_stop" in obj:
        es_raw = _require_mapping(obj["early_stop"], f"{path}.early_stop")
        _check_keys(es_raw, ("metric", "patience", "mode"), f"{path}.early_stop")
        for field in ("metric", "patience", "mode"):
            if field not in es_raw:
                raise ParseError(
                    "P2", f"train.early_stop needs {field!r}", path=f"{path}.early_stop"
                )
        early_stop = {
            "metric": _wrapped(
                es_raw["metric"], _scalar_str, f"{path}.early_stop.metric"
            ),
            "patience": _wrapped(
                es_raw["patience"], _scalar_int, f"{path}.early_stop.patience"
            ),
            "mode": _wrapped(
                es_raw["mode"],
                lambda v, p: _enum(v, ("min", "max"), p),
                f"{path}.early_stop.mode",
            ),
        }
    checkpoint = None
    if "checkpoint" in obj:
        ck_raw = _require_mapping(obj["checkpoint"], f"{path}.checkpoint")
        _check_keys(ck_raw, ("every", "file_path"), f"{path}.checkpoint")
        checkpoint = {
            key: (
                _parse_counter(value, f"{path}.checkpoint.every")
                if key == "every"
                else _wrapped(value, _scalar_str, f"{path}.checkpoint.file_path")
            )
            for key, value in ck_raw.items()
        }
    seed: Any = 0
    if "seed" in obj:
        seed = _wrapped(obj["seed"], _scalar_int, f"{path}.seed")
    return TrainSpec(
        objective=objective,
        params=params,
        optimizer=dict(optimizer),
        steps=steps,
        batch=dict(batch),
        anneal=anneal,
        control=control,
        phases=phases,
        precision=precision,
        eval=eval_spec,
        early_stop=early_stop,
        checkpoint=checkpoint,
        seed=seed,
    )


def _parse_signal_target(raw: Any, path: str) -> str | list[str]:
    """What a control signal observes (§2.11): one gate's name, or a
    **list** of gate names whose kept-unit counts are summed — one signal over
    several layers' gates, as a list-valued ``l1`` is one penalty over them.
    Kept as authored here; the canonical form writes the list either way, so
    ``"g"`` and ``["g"]`` are one controller (the ``layers`` fold)."""
    if isinstance(raw, str):
        return _scalar_str(raw, path)
    if not isinstance(raw, list) or not raw:
        raise ParseError(
            "P2",
            "a control signal names a trained gate, or a non-empty list of them "
            "whose kept counts are summed",
            path=path,
        )
    names = [_scalar_str(item, f"{path}[{i}]") for i, item in enumerate(raw)]
    if len(set(names)) != len(names):
        raise ParseError(
            "P2",
            f"a control signal lists each gate once — got {names}",
            path=path,
        )
    return names


def _parse_control(raw: Any, path: str) -> dict[str, Any]:
    """One ``train.control`` entry (§2.11): a closed-loop schedule on the
    hyperparameter the key names. ``kind`` is the controller
    (:data:`CONTROL_KINDS`), ``signal`` the one fit quantity it observes —
    ``{<signal>: <featurizer>}`` over :data:`CONTROL_SIGNALS` — ``setpoint``
    the ramp the signal should follow (the ``anneal`` schedule shape, in the
    signal's units) and ``gains`` the ``kp`` / ``ki`` / optional ``kd``. The
    optional ``space``, ``bounds`` and ``d_clip`` default per
    :data:`CONTROL_DEFAULTS` and are materialized in the canonical form. The
    gains are sweepable; the kind, the signal and the setpoint are not — they
    say *what* is controlled, not how hard."""
    obj = _require_mapping(raw, path)
    _check_keys(
        obj, ("kind", "signal", "setpoint", "gains", "space", "bounds", "d_clip"), path
    )
    for field in ("kind", "signal", "setpoint", "gains"):
        if field not in obj:
            raise ParseError("P2", f"a control entry needs {field!r}", path=path)
    kind = _wrapped(
        obj["kind"],
        lambda v, p: _enum(v, CONTROL_KINDS, p),
        f"{path}.kind",
        allow_sweep=False,
    )
    signal_raw = _require_mapping(obj["signal"], f"{path}.signal")
    if len(signal_raw) != 1:
        raise ParseError(
            "P2",
            "a control observes exactly one signal: {<signal>: <featurizer>}",
            path=f"{path}.signal",
        )
    ((signal_name, signal_target),) = signal_raw.items()
    _enum(signal_name, CONTROL_SIGNALS, f"{path}.signal")
    signal = {
        signal_name: _parse_signal_target(signal_target, f"{path}.signal.{signal_name}")
    }
    setpoint_raw = _require_mapping(obj["setpoint"], f"{path}.setpoint")
    _check_keys(setpoint_raw, ("ramp",), f"{path}.setpoint")
    if "ramp" not in setpoint_raw:
        raise ParseError(
            "P2",
            "a control setpoint is {'ramp': [start, end, frac]}",
            path=f"{path}.setpoint",
        )
    ramp = _parse_anneal_schedule(setpoint_raw["ramp"], f"{path}.setpoint.ramp")
    if not 0.0 < ramp[2] <= 1.0:
        raise ParseError(
            "P2",
            f"a setpoint ramp's frac is the fraction of the run it spans, in (0, 1]; got {ramp[2]}",
            path=f"{path}.setpoint.ramp",
        )
    gains_raw = _require_mapping(obj["gains"], f"{path}.gains")
    _check_keys(gains_raw, ("kp", "ki", "kd"), f"{path}.gains")
    for gain in ("kp", "ki"):
        if gain not in gains_raw:
            raise ParseError("P2", f"control gains need {gain!r}", path=f"{path}.gains")
    gains = {
        gain: _wrapped(value, _scalar_number, f"{path}.gains.{gain}")
        for gain, value in gains_raw.items()
    }
    out: dict[str, Any] = {
        "kind": kind,
        "signal": signal,
        "setpoint": {"ramp": list(ramp)},
        "gains": gains,
    }
    if "space" in obj:
        out["space"] = _enum(obj["space"], CONTROL_SPACES, f"{path}.space")
    if "bounds" in obj:
        bounds = obj["bounds"]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(
                isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds
            )
            or not bounds[0] < bounds[1]
        ):
            raise ParseError(
                "P2",
                "control bounds are an increasing two-number list",
                path=f"{path}.bounds",
            )
        out["bounds"] = [float(bounds[0]), float(bounds[1])]
    if "d_clip" in obj:
        d_clip = _scalar_number(obj["d_clip"], f"{path}.d_clip")
        if d_clip <= 0:
            raise ParseError(
                "P2", "control d_clip is a positive number", path=f"{path}.d_clip"
            )
        out["d_clip"] = float(d_clip)
    return out


def _parse_anneal_schedule(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ParseError("P2", "an anneal schedule is [start, end, frac]", path=path)
    start, end, frac = (_scalar_number(v, f"{path}[{i}]") for i, v in enumerate(value))
    return (float(start), float(end), float(frac))


def _parse_anneal_entry(value: Any, path: str) -> AnnealSchedule:
    """§2.11 ``anneal.<target>``: the list ``[start, end, frac]`` (linear), or
    the mapping ``{"from", "to", "frac", "shape"?}`` whose ``shape`` is one of
    :data:`ANNEAL_SHAPES`. A geometric schedule multiplies by a constant per
    step, so its endpoints must share a sign and neither may be zero — a ramp
    through zero has no ratio to walk."""
    if isinstance(value, list):
        return AnnealSchedule(*_parse_anneal_schedule(value, path))
    if not isinstance(value, Mapping):
        raise ParseError(
            "P2",
            'an anneal schedule is [start, end, frac] or {"from", "to", "frac", "shape"}',
            path=path,
        )
    _check_keys(value, ("from", "to", "frac", "shape"), path)
    for field in ("from", "to", "frac"):
        if field not in value:
            raise ParseError("P2", f"an anneal schedule needs {field!r}", path=path)
    start = float(_scalar_number(value["from"], f"{path}.from"))
    end = float(_scalar_number(value["to"], f"{path}.to"))
    frac = float(_scalar_number(value["frac"], f"{path}.frac"))
    shape = "linear"
    if "shape" in value:
        shape = _enum(value["shape"], ANNEAL_SHAPES, f"{path}.shape")
    if shape == "geometric" and start * end <= 0:
        raise ParseError(
            "P2",
            f"a geometric anneal multiplies by a constant per step, so 'from' and "
            f"'to' share a sign and neither is zero; got {start!r} → {end!r}",
            path=path,
        )
    return AnnealSchedule(start, end, frac, shape)


def _parse_phases(raw: Any, params: Sequence[str], path: str) -> tuple[PhaseSpec, ...]:
    """§2.11 ``phases``: a non-empty list of windows that **partition** the
    run. Each names ``until`` — one of :data:`PHASE_UNTIL_UNITS`, the same
    unit for every phase — strictly increasing, the last ``frac`` exactly
    ``1.0`` (an ``updates``-counted last phase is checked against the run's
    update count by the loop, which alone knows it); and ``params``, a
    non-empty subset of ``train.params`` by entry. What ``params`` leaves out
    is frozen for the phase. A phase's ``optimizer`` may carry only the
    per-params fields, keyed by the phase's own params; its ``anneal`` is
    parsed as the top-level one; ``freeze_masks`` is a name list the
    validator resolves to gates (rule 4)."""
    if not isinstance(raw, list) or not raw:
        raise ParseError("P2", "train.phases is a non-empty list of phases", path=path)
    declared = set(params)
    out: list[PhaseSpec] = []
    unit: str | None = None
    last_end: float | None = None
    for i, entry in enumerate(raw):
        p = f"{path}[{i}]"
        obj = _require_mapping(entry, p)
        _check_keys(obj, ("until", "params", "optimizer", "anneal", "freeze_masks"), p)
        for field in ("until", "params"):
            if field not in obj:
                raise ParseError("P2", f"a phase needs {field!r}", path=p)
        until = _require_mapping(obj["until"], f"{p}.until")
        _check_keys(until, PHASE_UNTIL_UNITS, f"{p}.until")
        if len(until) != 1:
            raise ParseError(
                "P2",
                "a phase ends at exactly one of {'frac': f} | {'updates': n}",
                path=f"{p}.until",
            )
        ((this_unit, end_raw),) = until.items()
        if unit is None:
            unit = this_unit
        elif this_unit != unit:
            raise ParseError(
                "P2",
                f"every phase counts its end in one unit; phase 0 used {unit!r}, this one {this_unit!r}",
                path=f"{p}.until",
            )
        end = (
            float(_scalar_number(end_raw, f"{p}.until.frac"))
            if unit == "frac"
            else float(_scalar_int(end_raw, f"{p}.until.updates"))
        )
        if unit == "frac" and not 0.0 < end <= 1.0:
            raise ParseError(
                "P2",
                f"a phase's frac is a fraction of the run in (0, 1]; got {end}",
                path=f"{p}.until",
            )
        if unit == "updates" and end < 1:
            raise ParseError(
                "P2", "a phase spans at least one update", path=f"{p}.until"
            )
        if last_end is not None and end <= last_end:
            raise ParseError(
                "P2",
                f"phases are consecutive: this phase ends at {end}, the previous at {last_end}",
                path=f"{p}.until",
            )
        last_end = end
        phase_params = _str_list(obj["params"], f"{p}.params")
        if not phase_params:
            raise ParseError(
                "P2",
                "a phase trains at least one entry — a phase that trains nothing is a wait, not a fit",
                path=f"{p}.params",
            )
        outside = sorted(set(phase_params) - declared)
        if outside:
            raise ParseError(
                "P2",
                f"phase params {outside} are not entries of train.params ({sorted(declared)}) — "
                "a phase narrows the trained set, it never widens it",
                path=f"{p}.params",
            )
        if len(set(phase_params)) != len(phase_params):
            raise ParseError("P2", "a phase names each entry once", path=f"{p}.params")
        optimizer = None
        if "optimizer" in obj:
            optimizer = _require_mapping(obj["optimizer"], f"{p}.optimizer")
            _check_keys(optimizer, PER_PARAMS_OPTIMIZER_FIELDS, f"{p}.optimizer")
            for field, value in optimizer.items():
                _parse_per_params_number(value, phase_params, f"{p}.optimizer.{field}")
            optimizer = dict(optimizer)
        anneal = None
        if "anneal" in obj:
            anneal_raw = _require_mapping(obj["anneal"], f"{p}.anneal")
            anneal = {
                key: _wrapped(
                    value, lambda v, q: _parse_anneal_entry(v, q), f"{p}.anneal.{key}"
                )
                for key, value in anneal_raw.items()
            }
        freeze = ()
        if "freeze_masks" in obj:
            freeze = _str_list(obj["freeze_masks"], f"{p}.freeze_masks")
            if len(set(freeze)) != len(freeze):
                raise ParseError(
                    "P2", "freeze_masks names each gate once", path=f"{p}.freeze_masks"
                )
        out.append(
            PhaseSpec(
                until={unit: end if unit == "frac" else int(end)},
                params=phase_params,
                optimizer=optimizer,
                anneal=anneal,
                freeze_masks=freeze,
            )
        )
    if unit == "frac" and last_end != 1.0:
        raise ParseError(
            "P2",
            f"the last phase ends at frac {last_end}; phases partition the run, so it ends at 1.0",
            path=path,
        )
    return tuple(out)


def _parse_trajectory_every(value: Any, path: str) -> dict[str, int]:
    """``trajectory.every`` (§2.12): exactly one of
    :data:`TRAJECTORY_EVERY_UNITS`, a positive integer. Not sweepable — how
    often a fit is photographed is not a research variable."""
    obj = _require_mapping(value, path)
    _check_keys(obj, TRAJECTORY_EVERY_UNITS, path)
    if len(obj) != 1:
        raise ParseError(
            "P2", f"expected exactly one of {list(TRAJECTORY_EVERY_UNITS)}", path=path
        )
    ((unit, count),) = obj.items()
    n = _scalar_int(count, f"{path}.{unit}")
    if n < 1:
        raise ParseError(
            "P2", f"every.{unit} is a positive integer, got {n}", path=path
        )
    return {unit: n}


def _parse_save(raw: Any, path: str) -> tuple[SaveEntry, ...]:
    if not isinstance(raw, list):
        raise ParseError("P2", "save is a list of entries", path=path)
    entries: list[SaveEntry] = []
    for i, entry_raw in enumerate(raw):
        p = f"{path}[{i}]"
        obj = _require_mapping(entry_raw, p)
        if "kind" in obj:
            # a non-value entry (§2.12): the kind is the whole binding
            kind = _enum(obj["kind"], SAVE_KINDS, f"{p}.kind")
            _check_keys(
                obj,
                ("kind", "file_path", *(("every",) if kind == "trajectory" else ())),
                p,
            )
            if "file_path" not in obj:
                raise ParseError("P2", "a save entry needs 'file_path'", path=p)
            every = None
            if kind == "trajectory":
                if "every" not in obj:
                    raise ParseError(
                        "P2",
                        "a trajectory entry says how its checkpoints are spaced: "
                        "'every': {'count': n} | {'updates': n} | {'epochs': n}",
                        path=p,
                    )
                every = _parse_trajectory_every(obj["every"], f"{p}.every")
            entries.append(
                SaveEntry(
                    value=kind,
                    file_path=_scalar_str(obj["file_path"], f"{p}.file_path"),
                    kind=kind,
                    every=every,
                )
            )
            continue
        _check_keys(obj, ("value", "model", "input", "site", "file_path", "reduce"), p)
        for field in ("value", "file_path"):
            if field not in obj:
                raise ParseError("P2", f"a save entry needs {field!r}", path=p)
        has_binding = "model" in obj or "input" in obj
        has_site = "site" in obj
        if has_binding and has_site:
            raise ValidationError(
                10, "a save entry carries model/input or site, never both", path=p
            )
        if has_binding and not ("model" in obj and "input" in obj):
            raise ValidationError(
                10, "a read/metric save entry needs both model and input", path=p
            )
        if not has_binding and not has_site:
            raise ValidationError(
                10,
                "a save entry needs its binding: model+input (read/metric) or site (featurizer)",
                path=p,
            )
        entries.append(
            SaveEntry(
                value=_scalar_str(obj["value"], f"{p}.value"),
                file_path=_scalar_str(obj["file_path"], f"{p}.file_path"),
                model=_scalar_str(obj["model"], f"{p}.model")
                if "model" in obj
                else None,
                input=_scalar_str(obj["input"], f"{p}.input")
                if "input" in obj
                else None,
                site=_scalar_str(obj["site"], f"{p}.site") if "site" in obj else None,
                reduce=_enum(obj["reduce"], SAVE_REDUCTIONS, f"{p}.reduce")
                if "reduce" in obj
                else None,
            )
        )
    return tuple(entries)


# --------------------------------------------------------------------------- #
# the document parser
# --------------------------------------------------------------------------- #


def check_protocol_version(raw: Mapping[str, Any]) -> None:
    """The first thing asked of any tree handed to the parser: is it an
    intervention specification of the version this loader reads (§1)?

    Answered *before* anything addresses the tree by path, so a ``--set`` or a
    workflow ``set`` on a document of the wrong shape is refused as that,
    rather than as a path that "does not exist". A **workflow** document is
    recognised by its ``steps`` section (workflow spec §1); the v1 spelling by
    its top-level ``version``, and the refusal names the verb that rewrites it.
    """
    if "steps" in raw:
        raise ParseError(
            "P2",
            "this is a workflow document (it has a 'steps' section), not an "
            "intervention specification — the workflow verbs read it",
            path="steps",
        )
    if "header" not in raw and ("version" in raw or "application" in raw):
        raise ParseError(
            "P2",
            "this is an intervention protocol v1 document (top-level 'version'); "
            f"this loader reads protocol_version {PROTOCOL_VERSION!r} — rewrite it "
            "with `causalab migrate <file>`",
            path="version",
        )
    header = raw.get("header")
    if not isinstance(header, Mapping):
        raise ParseError("P2", "missing required group 'header'", path="header")
    if "protocol_version" not in header:
        raise ParseError(
            "P2", "header needs 'protocol_version'", path="header.protocol_version"
        )
    version = header["protocol_version"]
    if version in MIGRATABLE_PROTOCOL_VERSIONS:
        raise ParseError(
            "P2",
            f"this is a protocol_version {version!r} document; this loader "
            f"reads protocol_version {PROTOCOL_VERSION!r} — rewrite it with "
            "`causalab migrate <file>` (version 3 spells a site's depth as "
            "'layers', a band of layer indices; the verb carries the rename)",
            path="header.protocol_version",
        )
    if version != PROTOCOL_VERSION:
        raise ParseError(
            "P2",
            f"unsupported protocol_version {version!r}; this loader reads "
            f"protocol_version {PROTOCOL_VERSION!r}",
            path="header.protocol_version",
        )


def _warn_unconventional_order(
    keys: Sequence[str], order: Sequence[str], *, what: str
) -> None:
    """§5.2 — the §1 order is recommended, not required.

    Order carries no meaning downstream: :func:`canonical.canonicalize` walks
    the recommended order and emits what it finds however it was authored, so
    a document written in another order has the same canonical bytes, the same
    digest and the same run as one written conventionally. Refusing it
    therefore refused a document that was already, byte for byte, the same
    experiment — most often one that had been through
    ``json.dumps(..., sort_keys=True)`` or a YAML round-trip on the way.

    What the order is *for* is reading, so what is left of the rule is a
    warning that names the order it recommends — once for the four groups,
    once for the method's sections.

    The retired half of this rule — ``save`` last — is subsumed: ``save`` is
    last in :data:`METHOD_SECTIONS`, and its presence is required separately
    (:data:`REQUIRED_METHOD_SECTIONS`) and its contents by rule 10.
    """
    ranks = {name: i for i, name in enumerate(order)}
    if [ranks[k] for k in keys] == sorted(ranks[k] for k in keys):
        return
    recommended = [name for name in order if name in keys]
    warnings.warn(
        f"{what} are not in the recommended docs/intervention_protocol.md §1 "
        f"order: got {list(keys)}, recommended {recommended} — this parses, "
        f"digests and runs identically either way (§5 rule 2)",
        ProtocolWarning,
        stacklevel=3,
    )


def _free_text(header: Mapping[str, Any], field: str) -> str | None:
    value = header.get(field)
    if value is not None and not isinstance(value, str):
        raise ParseError("P2", f"{field} is free text", path=f"header.{field}")
    return value


def parse_document(raw: Mapping[str, Any]) -> Document:
    """Strict-parse a raw mapping into a :class:`Document`.

    Owns §5.1 (strict keys, closed enums, no authored derived fields) and
    §5.2 (group and section order, which it warns about rather than refuses);
    cross-reference rules are
    :func:`causalab.protocol.validate.validate_document`'s job. ``raw`` may
    be in any order; the order it *is* in is only used to warn (§5 rule 2).

    The tree is the four groups of §1. The header is checked first
    (:func:`check_protocol_version`), then the groups' key sets, then each
    section — every refusal names its path from the section, which is how
    every other dotted path in the protocol is spelled (§1).
    """
    check_protocol_version(raw)
    for key in raw:
        if key not in GROUP_ORDER:
            raise ParseError(
                "P3", f"unknown group {key!r}{suggest(key, GROUP_ORDER)}", path=key
            )
    _warn_unconventional_order(list(raw), GROUP_ORDER, what="groups")
    for group in GROUP_ORDER:
        if group not in raw:
            raise ParseError("P2", f"missing required group {group!r}", path=group)
    header = _require_mapping(raw["header"], "header")
    _check_keys(header, HEADER_FIELDS, "header")
    method = _require_mapping(raw["method"], "method")
    for key in method:
        if key not in METHOD_SECTIONS:
            raise ParseError(
                "P3",
                f"unknown section {key!r}{suggest(key, METHOD_SECTIONS)}",
                path=key,
            )
    _warn_unconventional_order(list(method), METHOD_SECTIONS, what="method sections")
    for section in METHOD_SECTIONS:
        if section in REQUIRED_METHOD_SECTIONS and section not in method:
            raise ParseError(
                "P2", f"missing required section {section!r}", path=section
            )
    save = _parse_save(method["save"], "save")
    if not save:
        raise ValidationError(10, "save must be non-empty", path="save")
    from causalab.protocol.segments import parse_segments

    segments = (
        parse_segments(method["segments"], "segments") if "segments" in method else None
    )
    return Document(
        protocol_version=header["protocol_version"],
        title=_free_text(header, "title"),
        description=_free_text(header, "description"),
        model=_parse_model(raw["model"], "model"),
        data=_parse_data(raw["data"], "data"),
        segments=segments,
        positions=_parse_positions(method.get("positions", {}), "positions"),
        sites=_parse_sites(method["sites"], "sites"),
        featurizers=_parse_featurizers(method.get("featurizers", {}), "featurizers"),
        params=_parse_params(method.get("params", {}), "params"),
        code=_parse_code(method.get("code", {}), "code"),
        reads=_parse_reads(method["reads"], "reads"),
        writes=_parse_writes(method.get("writes", {}), "writes"),
        intervened_models=_parse_intervened_models(
            method.get("intervened_models", {}), "intervened_models"
        ),
        metrics=_parse_metrics(method.get("metrics", {}), "metrics"),
        train=_parse_train(method["train"], "train") if "train" in method else None,
        save=save,
        raw=dict(raw),
    )
