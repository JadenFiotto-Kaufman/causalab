"""The planning skeleton: models → forward groups, shared work interned.

Execution semantics (spec §4) derive everything from the document: for each
expanded point, the models to run are ``original`` on every input it is
read on, plus each intervened model on its declared input — one **forward
group** each. ``num_forwards`` is a property of the plan, never authored.

Across the points of a swept document, the planner **content-dedups**: a
forward group's identity is the digest of its full dependency closure
(model + in-force writes + their operand reads' closures + input binding),
so a harvest shared by nine fits interns to one group and the sharing falls
out of value identity, not scheduling cleverness (§3). Engines consume
this plan as data; fusion, batching and staging stay their call (§8).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from causalab.protocol.alignment import alignment_of
from causalab.protocol.canonical import canonical_model_ref
from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import FAMILIES
from causalab.protocol.schema import (
    METRIC_DOMAINS,
    AlignmentCardinality,
    Do,
    Document,
    IMSpec,
    PositionSpec,
    ReadSpec,
    SiteSpec,
    WriteSpec,
    concrete_int,
)
from causalab.protocol.spans import SpanSpec, static_indices

__all__ = [
    "COMPONENT_RANK",
    "PAST_BLOCKS",
    "UNRANKED",
    "ForwardGroup",
    "Materialization",
    "PointPlan",
    "Tap",
    "cohort_key",
    "fit_cohorts",
    "fit_constant_models",
    "generated_budget",
    "interned_groups",
    "plan_point",
    "closure_digest",
    "site_depth",
    "static_alignment",
]

#: Intra-block execution order of the component vocabulary — engine-free
#: data used only to find a group's deepest tap (elision, §4). ``ln_final``
#: and ``lm_head`` sort after every block.
#:
#: **Numbered in hundreds, with the attention band deliberately spread.** The
#: values are ordinal — only their order is ever read, never the numbers — but
#: changing one changes group elision, therefore every closure digest, and the
#: operand-reachability comparison (§5.21, via :func:`site_depth`); so the point
#: of the spacing is that inserting a component never renumbers an existing one.
#: The attention interior is where the vocabulary grew most (the pre-RoPE
#: projections, the gate, the post-RoPE q/k, the scores, the mixer output and
#: the per-head result), and the spacing below is what let each of those land
#: as an insertion rather than a re-pin. The gaps that remain are for whatever
#: the MoE and DeltaNet interiors need next.
COMPONENT_RANK: dict[str, int] = {
    "input_ids": -10,  # the model's input: before every activation
    "embeddings": 0,
    "block_input": 100,
    "attention_input_norm": 150,  # input_layernorm, between resid_pre and mixer
    # The DeltaNet mixer's interior interleaves numerically with the
    # full-attention band below: a layer carries one stream or the other, so
    # only relative order *within* a stream is ever compared, and the numbers
    # avoid every attention slot so that neither stream renumbers the other.
    "delta_qkv": 152,  # in_proj_qkv's fused [q|k|v] output, pre-conv
    "delta_gate": 154,  # in_proj_z's output — the output gate, produced early
    "delta_conv": 156,  # causal_conv1d_fn's return, channels-first
    "delta_query": 158,  # kernel arg 0: post-conv, post-tiling, PRE-l2norm
    # The two pre-tiling faces only the nnsight engine serves (the typed
    # backend pairs), ranked where the forward computes them — the
    # q/k splits of the conv output, before the value split — because the
    # `.source` interiors refuse out-of-order requests and the one-name
    # `delta_*` band is requested by the same engine in the same forward.
    "deltanet_query": 159,
    # The mixer's interior, in the order the forward computes it. All four are
    # module boundaries: q_norm/k_norm run BEFORE RoPE and are nn.Modules, so
    # the pre-RoPE projections are ordinary forward hooks rather than taps
    # inside the attention function.
    "attention_query_pre_rope": 160,
    "deltanet_key": 161,
    "delta_key": 162,  # kernel arg 1
    "delta_value": 164,  # kernel arg 2
    "delta_beta": 166,  # kernel kwarg beta — sigmoid(in_proj_b), per head
    "delta_decay": 168,  # kernel kwarg g — the log-decay, negative reals
    "attention_key_pre_rope": 170,
    # the per-step interior, in loop order: readout, update, state
    "delta_kv_mem": 172,  # (S_{t-1}·exp(g_t) · k̂_t).sum — what the state recalls
    "delta_state_update": 174,  # (v_t − kv_mem_t)·β_t — the diagram's `delta`
    "delta_state": 176,  # S_t, one d_k × d_v matrix per head per step
    # the per-chunk state the nnsight engine reads inside the chunked kernel,
    # before its return (the third typed pair)
    "deltanet_state": 177,
    "delta_kernel_output": 178,  # kernel return[0]: pre-norm, pre-gate
    "attention_value_states": 180,
    # the DeltaNet post-norm, post-gate mixer input — the exact analogue of
    # attention_premix, which is why the name
    "delta_premix": 182,
    # produced with q (one fused projection) and consumed at the very end, at
    # `attn_output * sigmoid(gate)` — ranked where it is produced
    "attention_gate": 190,
    # ...then RoPE rotates q and k, and the attention function runs: scores,
    # softmax, and the weighted sum of values. These four are taps *inside* that
    # function rather than module boundaries — see pytorch_hooks/attention_interface.py.
    "attention_query": 200,
    "attention_key": 210,
    "attention_scores": 220,
    "attention_probs": 230,
    "attention_z": 240,
    # 🔤 `attention_premix` was once named `attention_value`. It is the
    # o-projection's INPUT — on a gated family `z · σ(gate)`, on an ungated one
    # `z` — which is the mixer's output just before it is mixed back into the
    # residual stream, and is not the value vectors that name suggested. The
    # interior names those separately, and two components a letter apart in meaning
    # and identical in name is nnterp#51's cautionary tale happening to us.
    "attention_premix": 300,
    # derived, not computed: the model never forms it, so it sorts where it
    # would be if it did — between the tensor it is a function of and the sum
    # of its own heads
    "attention_result": 350,
    "attention_output": 400,
    # resid_mid is post_attention_layernorm's INPUT and mlp_input_norm its
    # OUTPUT, so the two straddle that one module in this order
    "block_mid": 450,
    "mlp_input_norm": 470,
    "mlp_input": 500,
    # The MoE interior, between the block's input and its output: the router
    # fires first, then the experts, then the combine.
    "router_logits": 510,
    "router_scores": 520,
    "expert_idx": 530,
    # The per-expert interior, ranked where its ops fire inside
    # the fused experts forward: the fused [gate | up] projection's two halves
    # land at 532/534, the activation between them and the down-projection at
    # 536, then — just before the weighted combine — the kernel's inverse
    # permutation (538), which is what `expert_permutation` reads, and the
    # down-projection's (pre-routing-weight) output keeps its reserved 540.
    # The late permutation rank is deliberate: ranks are execution order, and
    # the `.source` interiors refuse out-of-order taps.
    "expert_gate_proj": 532,
    "expert_up_proj": 534,
    "expert_activation": 536,
    "expert_neuron_output": 537,
    "expert_permutation": 538,
    "expert_output": 540,
    "routed_output": 550,
    "mlp_activation": 600,
    "mlp_neuron_output": 605,
    # the shared expert runs beside the routed ones; its gate is *consumed*
    # last, at the multiply that produces the (derived) gated output
    "shared_expert_gate_proj": 610,
    "shared_expert_up_proj": 620,
    "shared_expert_activation": 630,
    "shared_expert_output": 640,
    "shared_expert_gate": 650,
    "mlp_output": 700,
    "block_output": 800,
    "ln_final": 900,
    "lm_head": 1000,
}

#: The rank of a component the table does not know. Deliberately past
#: ``lm_head``: an unranked tap sorts last, so it is treated as the deepest and
#: nothing is elided behind it. Being wrong in the other direction would elide a
#: forward that a later tap still needed.
UNRANKED = 10_000

#: The block index :func:`site_depth` gives the two trunk components past every
#: block (``ln_final``, ``lm_head``). Also what :attr:`ForwardGroup.write_depth`
#: reads when no write of the group lands in a block, so a resume bound can be
#: compared with tap depths in one arithmetic; an engine clamps it to the model.
PAST_BLOCKS = 1_000_000


@dataclasses.dataclass(frozen=True)
class Tap:
    """One value to materialize in a group's forward: a read's address."""

    read: str
    site: str
    depth: tuple[int, int]  # (layer, component rank) — the elision key


@dataclasses.dataclass(frozen=True)
class Materialization:
    """What one continuation read obliges an engine to build.

    ``needs_distribution`` is the expensive bit: a vocabulary-wide tensor
    per addressed position. It is false when the read is neither saved nor
    reduced by a metric that consumes distributions, in which case the
    engine must not build one (§8). *How* it avoids building one — a
    narrowed projection, per-step captures, a replay — is the engine's
    choice; this is the requirement, not the mechanism."""

    read: str
    site: str
    needs_distribution: bool


@dataclasses.dataclass(frozen=True)
class ForwardGroup:
    """One forward pass: a model (original or an IM) on one input role.

    ``digest`` is the content identity of everything that determines this
    group's activations — equal digests across points mean one shared
    forward. ``decode_depth`` is the greedy budget this group must decode
    for (0 = prefill only), and ``materialize`` states what its
    continuation reads oblige — both derived, never authored (§6).

    ``base_digest`` is the digest the same input role would have under
    ``original`` — the identity of the **un-intervened prefix** of this
    forward (§4 "Resume"). Every intervened model on one input shares it,
    whatever it writes, and it equals the ``original`` group's own digest
    whenever one is planned on that input, so a pass of either kind can leave
    the residual entering a block behind for the others. ``write_depth`` is
    the shallowest block any in-force write lands in, :data:`PAST_BLOCKS` when
    none does (``original``, or writes past the last block): blocks strictly
    below it — and the residual entering it — are exactly what ``original``
    computes on the same rows."""

    model: str
    input: str
    taps: tuple[Tap, ...]
    digest: str
    base_digest: str
    write_depth: int
    decode_depth: int = 0
    materialize: tuple[Materialization, ...] = ()

    @property
    def stop_after(self) -> tuple[int, int] | None:
        """The deepest tap's depth — an engine may end the forward there
        (§4 elision). ``None`` when the group has no taps, and also when it
        decodes: every decode step needs the head, so there is nothing to
        elide.

        Read this off the *interned* group, not a single point's, whenever
        several points share the forward: the shared pass has to reach every
        tap any of them asked for, so the depth it may stop at is the deepest
        of the union. :func:`interned_groups` builds exactly that group."""
        if self.decode_depth:
            return None
        return max((tap.depth for tap in self.taps), default=None)

    @property
    def resume_at(self) -> int:
        """The block an engine may *start* this forward at (§4 "Resume"),
        given the residual entering it from an un-intervened pass over the
        same rows — the mirror image of :attr:`stop_after`. 0 means never.

        The shallowest block any write **or any tap** touches: below the
        first write the forward is ``original``'s, but a tap below it still
        needs its block to run, so a read at layer 1 pins a write at layer 20
        to block 1. ``original`` never resumes — it is what the prefix *is* —
        and neither does a group that decodes (§4: nothing is elided there
        either). Past every block (:data:`PAST_BLOCKS`) is left for the
        engine to clamp, as it alone knows the model's depth.

        Like ``stop_after``, read this off the *interned* group: the one pass
        a shared digest earns serves every sharer's taps, so the block it may
        start at is the shallowest of the union, not of the point that ran
        first. :func:`interned_groups` builds exactly that group."""
        if self.model == "original" or self.decode_depth:
            return 0
        return min([self.write_depth, *(tap.depth[0] for tap in self.taps)])


@dataclasses.dataclass(frozen=True)
class PointPlan:
    """The derived execution shape of one concrete compiled intervention."""

    groups: tuple[ForwardGroup, ...]

    @property
    def num_forwards(self) -> int:
        return len(self.groups)


def plan_point(
    doc: Document, *, data_identity: Mapping[str, Any] | None = None
) -> PointPlan:
    """Derive the forward groups of one concrete document.

    ``data_identity`` (input role → the identity of the rows that role
    reads: the canonical form's content digest of the selected rows, §2.2,
    plus the field — never the ref's name) folds the input data into group
    digests so two points reading different data never intern together, and
    two points reading the same rows under two names do; omit it for a
    purely structural plan.

    A band site (§2.4 ``layers``) is planned as its per-layer members
    (:func:`lower_bands`) — the taps, depths and resume point are the
    hand-written N-site document's, which is what the executor runs."""
    doc = lower_bands(doc)
    groups: list[ForwardGroup] = []
    seen: set[tuple[str, str]] = set()
    # original, once per input it is read on (§4), in read declaration order
    for read in doc.reads.values():
        if read.model == "original" and ("original", str(read.input)) not in seen:
            seen.add(("original", str(read.input)))
            groups.append(_build_group(doc, "original", str(read.input), data_identity))
    for im_name, im in doc.intervened_models.items():
        groups.append(_build_group(doc, im_name, str(im.input), data_identity))
    return PointPlan(groups=tuple(groups))


def interned_groups(plans: Iterable[PointPlan]) -> tuple[ForwardGroup, ...]:
    """The campaign's forward groups once §3's content-dedup is applied:
    groups sharing a ``digest`` merge into **one**, whose taps are the union
    of theirs.

    ``sum(p.num_forwards for p in plans)`` is what a per-point loop pays;
    ``len(interned_groups(plans))`` is what the campaign actually owes. For a
    32-layer × 2-position interchange scan that is 65 rather than 128 — the
    64 patched forwards are genuinely distinct, but the counterfactual
    harvest depends on nothing swept, so its 64 instances become one forward
    carrying 32 taps (the position axis moves the gather, not the pass).
    Taps are absent from the digest precisely so this falls out
    of value identity (reading layer 3 or layer 23 of the same un-intervened
    forward is the same forward), which is also why the merged group must
    carry the union: the one pass it earns has to serve every point.

    A merged group's ``decode_depth`` is the deepest any sharer needs, since
    a decode changes what the group produces rather than what its prefill
    computes. Its ``model``/``input`` come from the first sharer — equal
    digests mean equal model *closures*, so two intervened models that differ
    only in name merge, and either name describes the pass.

    Callers build each :class:`PointPlan` with its own ``data_identity``, so
    points reading different data never merge here.
    """
    merged: dict[str, ForwardGroup] = {}
    for plan in plans:
        for group in plan.groups:
            first = merged.get(group.digest)
            if first is None:
                merged[group.digest] = group
                continue
            taps = list(first.taps)
            taps.extend(tap for tap in group.taps if tap not in taps)
            seen = {item.read for item in first.materialize}
            materialize = list(first.materialize)
            materialize.extend(
                item for item in group.materialize if item.read not in seen
            )
            merged[group.digest] = dataclasses.replace(
                first,
                taps=tuple(taps),
                decode_depth=max(first.decode_depth, group.decode_depth),
                materialize=tuple(materialize),
            )
    return tuple(merged.values())


def _build_group(
    doc: Document,
    model: str,
    input_role: str,
    data_identity: Mapping[str, Any] | None,
) -> ForwardGroup:
    taps = tuple(
        Tap(read=rname, site=str(read.site), depth=site_depth(doc, str(read.site)))
        for rname, read in doc.reads.items()
        if read.model == model and str(read.input) == input_role
    )
    identity = dict(data_identity or {})
    digest = _group_digest(doc, model, input_role, identity)
    # the same input under `original`: the identity of everything this
    # forward computes below its first write (§4 "Resume"). Equal to `digest`
    # for the original model itself
    base_digest = (
        digest
        if model == "original"
        else _group_digest(doc, "original", input_role, identity)
    )
    # The digest stays activation-identity: a decode changes what the group
    # *produces*, not what its prefill computes — the same reason taps are
    # not in it. Two points that differ only in decode depth share a prefill.
    depth = 0
    materialize: list[Materialization] = []
    for rname, read in doc.reads.items():
        if read.model != model or str(read.input) != input_role:
            continue
        budget = generated_budget(doc, read.pos)
        if budget is None:
            continue
        depth = max(depth, budget)
        materialize.append(
            Materialization(
                read=rname,
                site=str(read.site),
                needs_distribution=_needs_distribution(doc, rname),
            )
        )
    return ForwardGroup(
        model=model,
        input=input_role,
        taps=taps,
        digest=digest,
        base_digest=base_digest,
        write_depth=_write_depth(doc, model),
        decode_depth=depth,
        materialize=tuple(materialize),
    )


def _attention_requirement(
    doc: Document, model: str, input_role: str
) -> dict[str, str]:
    """Interior reads/writes change the implementation, not merely the taps.

    Include continuation reads too: prefill and decode must use one backend.
    The requirement follows operand closures so downstream edits cannot share
    results produced from different source implementations.
    """
    interiors = {
        component
        for adapter in FAMILIES.values()
        for component, tap in adapter.taps.items()
        if tap.slot is not None and tap.kind not in {"delta", "experts"}
    }
    sites = [
        read.site
        for read in doc.reads.values()
        if read.model == model and str(read.input) == input_role
    ]
    if model != "original":
        im = doc.intervened_models[model]
        if isinstance(im.writes, tuple):
            sites.extend(doc.writes[name].site for name in im.writes)
    if any(
        isinstance(doc.sites[str(site)].component, str)
        and doc.sites[str(site)].component in interiors
        for site in sites
    ):
        return {"attention_requirement": "eager"}
    return {}


def _group_digest(
    doc: Document, model: str, input_role: str, identity: Mapping[str, Any]
) -> str:
    body = {
        # the *realization*, not just the name: `canonical_model_ref` is the
        # canonical form's own function, so dtype and the quantization block
        # are in the group's identity exactly as they are in the document's
        "network": canonical_model_ref(doc.model),
        **_attention_requirement(doc, model, input_role),
        "model": _model_closure(doc, model, set(), identity),
        "input": input_role,
        "data": identity.get(input_role),
        # the frame the rows are encoded in (§2.2.1): a `segments.frame: chat`
        # row is rendered through the tokenizer's chat template before it is
        # tokenized, so the same rows under the same model are a different
        # token sequence — a different activation under what would otherwise
        # be the same digest. Campaign-invariant today (the section takes no
        # sweep wrapper), so this changes nothing within a request; it makes
        # "equal digest, equal tokens" a property of the digest rather than
        # of the parser.
        "segments": doc.segments,
    }
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), default=_encode
        ).encode()
    ).hexdigest()


def _write_depth(doc: Document, model: str) -> int:
    """The shallowest block one of ``model``'s in-force writes lands in —
    :data:`PAST_BLOCKS` for ``original`` and for writes that touch no block.

    The block is :func:`site_depth`'s layer coordinate: an in-block component
    at layer ``L`` is inside block ``L`` whichever side of it the hook rides,
    so ``block_output`` at ``L`` counts as ``L`` (block ``L`` has to run for
    its output hook to fire) and ``block_input`` at ``L`` as ``L`` too; the
    two layer-less trunk components count as 0 (``embeddings``,
    ``input_ids``: before every block) and past every block (``ln_final``,
    ``lm_head``). Conservative by construction — the residual entering the
    named block is pre-write on every one of them. A write the plan cannot
    see — a ``writes`` list, a site name or a site component still under a
    sweep wrapper — is 0."""
    if model == "original":
        return PAST_BLOCKS
    im = doc.intervened_models[model]
    if not isinstance(im.writes, tuple):
        # "no write I can see" must not read as "past every block": that is
        # the permissive direction — a resume past a write it cannot see, and
        # post-write residuals stored as un-intervened. `fit_constant_models`
        # refuses this shape outright; the plan is also asked about template
        # documents (`plan_point` with no data identity), so this answers
        # with the depth that resumes nothing and stores nothing.
        return 0
    depths: list[int] = []
    for ename in im.writes:
        site_name = str(doc.writes[ename].site)
        if site_name not in doc.sites or not isinstance(
            doc.sites[site_name].component, str
        ):
            # the neighbouring door: `site_depth` narrows a component it
            # cannot read to the trunk, i.e. `PAST_BLOCKS` — the same
            # permissive direction, so the same answer
            return 0
        depths.append(site_depth(doc, site_name)[0])
    return min([PAST_BLOCKS, *depths])


def static_alignment(doc: Document, pos: Any) -> AlignmentCardinality | None:
    """The cardinality a position has **by construction**, from the document
    alone (§2.3) — or ``None`` when only the tokenizer can say.

    An ``index`` (bare, scoped or relative) is one token per row on every
    input; an unscoped ``span [a, b)`` is one joint window of width ``b − a``
    on every input. Both pair ``one_to_one`` whatever the text says — two
    ``index`` specs are two locations, one ``span`` is one joint address — so
    a declaration that says otherwise is refusable at load (rule 26). This is
    the planning half of :func:`~causalab.protocol.alignment.alignment_of`'s
    three callers: what the plan knows about the pairing shape before any
    encode. A ``variable`` or ``column`` window, a scoped span (clipped to
    its anchor's width) and ``all`` are as wide as the tokenizer makes them,
    and the executor decides those at encode time.

    Takes the spelling a read or write carries: a ``positions`` name or an
    inline spec.
    """
    spec = doc.positions[pos] if isinstance(pos, str) else pos
    if not isinstance(spec, PositionSpec) or spec.generated is not None:
        return None
    if isinstance(spec, SpanSpec):
        # A span the document alone fixes (an unscoped `indices` set, an
        # atomic unscoped `span`, a union of such) is one set of one width on
        # every input — an atomic span as one joint address, a non-atomic set
        # as constituents that are each one token. Anything text-located
        # (`segment`, `variable`, a predicate) is the tokenizer's to decide.
        fixed = static_indices(spec)
        return alignment_of((fixed,), (fixed,)) if fixed is not None else None
    if spec.index is not None:
        run: tuple[int, ...] = (0,)
    elif (
        spec.span is not None
        and spec.scope is None
        and isinstance(spec.span, tuple)
        and len(spec.span) == 2
    ):
        lo, hi = (int(v) for v in spec.span)
        run = tuple(range(lo, hi))
    else:
        return None
    return alignment_of((run,), (run,))


def generated_budget(doc: Document, pos: Any) -> int | None:
    """The decode budget of a position, or ``None`` for the prompt frame.

    Takes the spelling a read carries (a positions-table name or an inline
    spec) and returns the concrete budget — points are concrete by the time
    they are planned, so a surviving sweep wrapper is a caller error."""
    spec = doc.positions.get(pos) if isinstance(pos, str) else pos
    if not isinstance(spec, PositionSpec) or spec.generated is None:
        return None
    return concrete_int(spec.generated["max_new_tokens"], "generated.max_new_tokens")


def _needs_distribution(doc: Document, read: str) -> bool:
    """Whether anything downstream of ``read`` consumes a full distribution.

    Saving the read is the obvious case. So is any metric in the
    ``distribution`` domain (§2.10). An ``ids`` kind does **not** count: it
    consumes the tokens the decode produced, so a text-only probe obliges no
    vocabulary projection anywhere — which is the whole point of stating the
    requirement rather than always paying it."""
    if any(entry.value == read for entry in doc.save):
        return True
    for metric in doc.metrics.values():
        domain = METRIC_DOMAINS.get(str(metric.kind), "distribution")
        if str(metric.of) == read and domain == "distribution":
            return True
        target = metric.fields.get("target")
        if isinstance(target, str) and target == read:
            return True
    return False


def closure_digest(
    doc: Document, read_name: str, *, data_identity: Mapping[str, Any] | None = None
) -> str:
    """The content identity of one read's value: its address plus the full
    closure of the model it reads in. Equal digests across points mean one
    shared harvest (§3).

    "The model it reads in" includes **how it is realized**: an fp32 and a bf16
    harvest of the same address are different tensors, so they are different
    content and must not share. See :func:`_build_group` for the bug this
    closed.
    """
    body = {
        "network": canonical_model_ref(doc.model),
        "read": _read_closure(doc, read_name, set(), dict(data_identity or {})),
    }
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), default=_encode
        ).encode()
    ).hexdigest()


def site_depth(doc: Document, site_name: str) -> tuple[int, int]:
    """One site's position in the forward pass, as a sortable ``(layer, rank)``.

    The total order the whole vocabulary shares: block depth first, then
    :data:`COMPONENT_RANK` inside the block, with the two layer-less trunk
    components sorting after every block. Three readers depend on it, and on
    nothing finer:

    * group elision (§4) — a forward may stop after its deepest tap;
    * operand reachability (§5.21) — a write's operand may not be read from
      strictly deeper than the address it lands on;
    * resume (§4, :func:`_write_depth`) — the shallowest block a write lands
      in.

    A **band** site (§2.4 ``layers`` with several members) answers with its
    **shallowest** member — the first block the site touches, which is the
    block a write on it first lands in (resume) and the depth a write's
    operand has to be at or above (rule 21's "at or above the write" reads
    the band's min). The deepest member — where a *read* on the band is
    complete, and the tap depth group elision stops after — is the last of
    :func:`site_depths`; the planner never sees a band in either role because
    :func:`plan_point` lowers bands to their members first
    (:func:`lower_bands`), and rule 21 compares member depths pairwise.

    An unresolved ``layers`` (an unexpanded sweep or artifact wrapper) reads
    as 0 and a non-string ``component`` as the trunk. The first two callers
    run on a *point* document, where every site is concrete, so for them that
    fallback is a type-narrowing convenience and not a semantic; the third
    can be asked about a template and refuses a site it cannot read before
    calling here, since the trunk fallback is the permissive direction for it.
    """
    return site_depths(doc, site_name)[0]


def site_depths(doc: Document, site_name: str) -> tuple[tuple[int, int], ...]:
    """Every member of a site's band as a :func:`site_depth` coordinate, in
    band order — one entry for a one-layer site or a trunk component. Rule 21
    (``validate._check_operand_reachability``) reads a band's members from
    here so a band operand feeding a band write is checked member by member."""
    site = doc.sites[site_name]
    layers = site.layers if isinstance(site.layers, tuple) and site.layers else (0,)
    component = site.component if isinstance(site.component, str) else "lm_head"
    rank = COMPONENT_RANK.get(component, UNRANKED)
    if component in ("ln_final", "lm_head"):
        return ((PAST_BLOCKS, rank),)  # after every block
    return tuple((int(layer), rank) for layer in layers)


# --------------------------------------------------------------------------- #
# bands — one site across N layers, lowered to its per-layer members
# --------------------------------------------------------------------------- #


def band_member(name: str, layer: int) -> str:
    """The name a band site's per-layer member — and the member of every read
    and write on it — carries once lowered: ``a[layers=10]``, §3's derived-name
    convention over the ``layers`` coordinate."""
    return f"{name}[layers={layer}]"


def lower_bands(doc: Document) -> Document:
    """``doc`` with every multi-layer band site (§2.4 ``layers``) fanned out to
    one site per member, and the reads and writes on it to one per member.

    A band is *one* address across N layers: one read on it captures N
    tensors, one write on it lands N times, and a write whose operand is a read
    on a band of the same length takes member *i*'s value at member *i* — the
    shape ROME's clipped restoration window has, "one point, N layers
    restored at once". The engines reason about one module at
    a time (``ResolvedSite`` is one module), so the executor lowers a point
    document to the N-site form the author would have written by hand before
    it resolves anything — the same form ``at_once`` (§3.1) compiles to, which
    is what lets a band and its hand-written equivalent run identically. The
    canonical form and the digest are over the authored band; only the
    execution sees the members.

    Lowering is a pure function of the document and idempotent (the members
    are one-layer sites). What the members do not cover is refused **by
    name** rather than guessed at: a band read saved, fed to a metric or
    named anywhere but as the operand of a band write of the same length has
    no single value; a featurizer on a band read or write would fit one map
    across N layers, which is a decision (a shared subspace? one per layer?)
    the document has to make explicitly, one site per layer, until the
    protocol gives it a word. Any other operand — a read on a one-layer site,
    a param, a literal — is broadcast to every member.
    """
    bands = {
        name: spec.layers
        for name, spec in doc.sites.items()
        if isinstance(spec.layers, tuple) and len(spec.layers) > 1
    }
    if not bands:
        return doc

    def label(site: str) -> str:
        from causalab.protocol.sweep import band_label  # one rendering

        return f"site {site!r} (layers {band_label(bands[site])})"

    sites: dict[str, SiteSpec] = {}
    for name, spec in doc.sites.items():
        if name in bands:
            for layer in bands[name]:
                sites[band_member(name, layer)] = dataclasses.replace(
                    spec, layers=(layer,)
                )
        else:
            sites[name] = spec

    band_reads: dict[str, str] = {}  # read → its band site
    reads: dict[str, ReadSpec] = {}
    for rname, read in doc.reads.items():
        site = str(read.site)
        if site not in bands:
            reads[rname] = read
            continue
        if read.featurizer is not None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} at {label(site)} names a featurizer — a "
                "featurizer on a band would be one map fitted across every "
                "layer of it, a choice this document has to make explicitly, "
                "one site per layer",
                reason="unsupported_mechanism",
            )
        band_reads[rname] = site
        for layer in bands[site]:
            reads[band_member(rname, layer)] = dataclasses.replace(
                read, site=band_member(site, layer)
            )

    band_writes: dict[str, str] = {}  # write → its band site
    writes: dict[str, WriteSpec] = {}
    for ename, write in doc.writes.items():
        site = str(write.site)
        operands = tuple(
            name for name in _operand_names(write.do) if name in band_reads
        )
        if site not in bands:
            if operands:
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} at site {site!r} takes its operand from "
                    f"read {operands[0]!r} on {label(band_reads[operands[0]])} — "
                    "a band read is N tensors, and a one-layer write lands "
                    "one; read the operand at a one-layer site, or make the "
                    "write a band of the same length",
                    reason="unsupported_mechanism",
                )
            writes[ename] = write
            continue
        if write.featurizer is not None:
            raise ProtocolError(
                "P4",
                f"write {ename!r} at {label(site)} names a featurizer — a "
                "featurizer on a band would be one map fitted across every "
                "layer of it, a choice this document has to make explicitly, "
                "one site per layer",
                reason="unsupported_mechanism",
            )
        for operand in operands:
            if len(bands[band_reads[operand]]) != len(bands[site]):
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} at {label(site)} takes its operand from "
                    f"read {operand!r} on {label(band_reads[operand])} — a "
                    "band operand feeds a band write member by member, so the "
                    "two bands must have the same number of layers",
                    reason="unsupported_mechanism",
                )
        band_writes[ename] = site
        for index, layer in enumerate(bands[site]):
            substitution = {
                operand: band_member(operand, bands[band_reads[operand]][index])
                for operand in operands
            }
            writes[band_member(ename, layer)] = dataclasses.replace(
                write,
                site=band_member(site, layer),
                do=_substitute_operands(write.do, substitution),
            )

    intervened_models: dict[str, IMSpec] = {}
    for mname, im in doc.intervened_models.items():
        if not isinstance(im.writes, tuple):
            intervened_models[mname] = im
            continue
        lowered: list[str] = []
        for ename in im.writes:
            if ename in band_writes:
                lowered.extend(
                    band_member(ename, layer) for layer in bands[band_writes[ename]]
                )
            else:
                lowered.append(ename)
        intervened_models[mname] = dataclasses.replace(im, writes=tuple(lowered))

    # every other mention of a band read or write has no member to go to
    named = {**band_reads, **band_writes}
    for entry in doc.save:
        if entry.value in named:
            raise ProtocolError(
                "P4",
                f"save entry {entry.value!r} names a read on "
                f"{label(named[entry.value])} — a band read is N tensors and "
                "the manifest saves one per entry; save each layer as its own "
                "read, or sweep the layer (§3) to get one table",
                reason="unsupported_mechanism",
            )
    for mname, metric in doc.metrics.items():
        of = str(metric.of)
        if of in named:
            raise ProtocolError(
                "P4",
                f"metric {mname!r} reduces read {of!r} on {label(named[of])} — "
                "a metric reads one tensor, and a band read is N",
                reason="unsupported_mechanism",
            )
    elsewhere = sorted(
        set(
            _strings(
                doc.raw.get("method", {}),
                skip=(
                    "sites",
                    "reads",
                    "writes",
                    "intervened_models",
                    "save",
                    "metrics",
                ),
            )
        )
        & set(named)
    )
    if elsewhere:
        raise ProtocolError(
            "P4",
            f"{elsewhere[0]!r} is a read or write on {label(named[elsewhere[0]])} "
            "and is named outside the tables a band lowers (reads, writes, "
            "intervened_models) — a band member has no value there",
            reason="unsupported_mechanism",
        )

    method = dict(doc.raw.get("method", {}))
    method["sites"] = {name: _site_raw(spec) for name, spec in sites.items()}
    raw = {**doc.raw, "method": method}
    return dataclasses.replace(
        doc,
        sites=sites,
        reads=reads,
        writes=writes,
        intervened_models=intervened_models,
        raw=raw,
    )


def _site_raw(spec: SiteSpec) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in dataclasses.asdict(spec).items()
        if value is not None
    }


def _operand_names(do: Do) -> tuple[str, ...]:
    """The read or param names a mechanism's payload references — the
    validator's rule (``validate._operand_names``) over the same payload
    shapes: ``swap``'s one operand, the ``op``/``alpha`` of ``add_scaled`` and
    ``lerp``, ``affine``'s ``A`` and ``b``."""
    payload = do.payload
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, Mapping):
        return tuple(v for v in payload.values() if isinstance(v, str))
    return ()


def _substitute_operands(do: Do, substitution: Mapping[str, str]) -> Do:
    if not substitution:
        return do
    payload = do.payload
    if isinstance(payload, str):
        return dataclasses.replace(do, payload=substitution.get(payload, payload))
    if isinstance(payload, Mapping):
        return dataclasses.replace(
            do,
            payload={
                key: substitution.get(value, value) if isinstance(value, str) else value
                for key, value in payload.items()
            },
        )
    return do


def _strings(node: Any, *, skip: tuple[str, ...] = ()) -> Iterable[str]:
    """Every string value in a subtree (mapping keys included), the sections
    in ``skip`` left out at the top level."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in skip:
                continue
            yield str(key)
            yield from _strings(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, str):
        yield node


def _model_closure(
    doc: Document, model: str, visiting: set[str], identity: Mapping[str, Any]
) -> Any:
    """Everything that determines a model's activations: for an IM, the
    in-force writes and, recursively, their operand reads' closures. The
    validated acyclicity (§5.7) bounds the recursion; ``visiting`` is a
    belt-and-braces guard."""
    if model == "original":
        return "original"
    if model in visiting:
        raise AssertionError(
            f"cycle through {model!r} — validation should have refused this"
        )
    im = doc.intervened_models[model]
    train_dep: Any = None
    writes: dict[str, Any] = {}
    for ename in sorted(im.writes if isinstance(im.writes, tuple) else ()):
        write = doc.writes[ename]
        operands: dict[str, Any] = {}
        for op in _write_operand_names(doc, ename):
            if op in doc.reads:
                operands[op] = _read_closure(doc, op, visiting | {model}, identity)
        for op in _write_param_names(doc, ename):
            operands[op] = _entry(doc.params, op) if op in doc.params else op
        writes[ename] = {
            "site": _entry(doc.sites, str(write.site)),
            "pos": _pos_entry(doc, write.pos),
            "featurizer": _featurizer_entry(doc, write.featurizer),
            "dims": write.dims,
            "do": {str(write.do.mechanism): write.do.payload},
            "operands": operands,
        }
        if doc.train is not None and (
            _uses_trained_featurizer(doc, write.featurizer)
            or any(
                op.split(".", 1)[0] in _trained_roots(doc)
                for op in _write_param_names(doc, ename)
            )
        ):
            # a trained featurizer's weights — or a trained free tensor the
            # write consumes — are a function of the whole fit: two points
            # differing only in train.seed must never intern
            train_dep = dataclasses.asdict(doc.train)
    return {
        "input": im.input,
        "data": identity.get(str(im.input)),
        "writes": writes,
        "train": train_dep,
    }


def fit_constant_models(doc: Document) -> frozenset[str]:
    """The models whose forward a ``train`` document **cannot change**.

    A fit re-runs its groups every optimizer step, but only the groups a
    trained parameter can reach actually differ between steps. ``original``
    never does — the network's weights are frozen at load (§2.11) — and an
    intervened model does not either when every in-force write is fed by
    nothing the fit moves: no trained featurizer on the write, no
    ``train.params`` root among its param operands, and every read operand
    taken raw (through no trained featurizer) off a model that is itself
    constant. That last clause is the recursion: a swap fed by a read *on*
    ``patched`` moves whenever the rotation does, even though the swap itself
    is unfeaturized.

    Without a ``train`` section nothing moves and every model is constant.

    What consumes this: an engine's train loop, to run a constant group once
    per row slice and serve its raw capture on every later step, epoch, eval
    pass and point (§3, §4). The shipped DAS/DBM methods reduce to
    ``{"original"}`` — their one write goes through the trained featurizer.
    Over-inclusion here would hand a stale activation to a gradient step,
    which is why the exclusions are stated per dependency rather than by
    "grad enabled".
    """
    models = ("original", *doc.intervened_models)
    if doc.train is None:
        return frozenset(models)
    trained_roots = _trained_roots(doc)

    def constant(model: str, visiting: set[str]) -> bool:
        if model == "original":
            return True
        if model in visiting:
            raise AssertionError(
                f"cycle through {model!r} — validation should have refused this"
            )
        im = doc.intervened_models[model]
        if not isinstance(im.writes, tuple):
            # "no writes I can see" must not read as "constant": that is the
            # unsafe direction, so a non-point document is refused loudly
            raise AssertionError(
                f"{model!r}.writes is unexpanded — expansion should have "
                "produced a point document before anything asks what a fit "
                "cannot change"
            )
        for ename in im.writes:
            write = doc.writes[ename]
            if _uses_trained_featurizer(doc, write.featurizer):
                return False
            for op in _write_param_names(doc, ename):
                if op.split(".", 1)[0] in trained_roots:
                    return False
            for op in _write_operand_names(doc, ename):
                read = doc.reads[op]
                if _uses_trained_featurizer(doc, read.featurizer):
                    return False
                if not constant(str(read.model), visiting | {model}):
                    return False
        return True

    return frozenset(model for model in models if constant(model, set()))


def _trained_roots(doc: Document) -> frozenset[str]:
    """The featurizer and params names a fit moves: ``train.params`` with any
    ``.slot`` suffix dropped. Empty without a ``train`` section."""
    if doc.train is None:
        return frozenset()
    return frozenset(p.split(".", 1)[0] for p in doc.train.params)


def _write_param_names(doc: Document, ename: str) -> tuple[str, ...]:
    """Operand names that resolve to params entries or featurizer slots —
    their specs are part of the written value's identity."""
    do = doc.writes[ename].do
    payload = do.payload
    names: list[str] = []
    if isinstance(payload, str):
        names.append(payload)
    elif isinstance(payload, Mapping):
        names.extend(v for v in payload.values() if isinstance(v, str))
    return tuple(n for n in names if n not in doc.reads)


def _uses_trained_featurizer(doc: Document, ref: Any) -> bool:
    if doc.train is None or ref is None:
        return False
    trained = _trained_roots(doc)
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    return any(name in trained for name in chain)


def _read_closure(
    doc: Document, read_name: str, visiting: set[str], identity: Mapping[str, Any]
) -> Any:
    read = doc.reads[read_name]
    return {
        **_attention_requirement(doc, str(read.model), str(read.input)),
        "site": _entry(doc.sites, str(read.site)),
        "pos": _pos_entry(doc, read.pos),
        "featurizer": _featurizer_entry(doc, read.featurizer),
        "dims": read.dims,
        "input": read.input,
        "data": identity.get(str(read.input)),
        "model": _model_closure(doc, str(read.model), visiting, identity),
        "train": dataclasses.asdict(doc.train)
        if doc.train is not None and _uses_trained_featurizer(doc, read.featurizer)
        else None,
    }


def _write_operand_names(doc: Document, ename: str) -> tuple[str, ...]:
    do = doc.writes[ename].do
    payload = do.payload
    names: list[str] = []
    if isinstance(payload, str):
        names.append(payload)
    elif isinstance(payload, Mapping):
        names.extend(v for v in payload.values() if isinstance(v, str))
    return tuple(n for n in names if n in doc.reads)


def _entry(table: Mapping[str, Any], name: str) -> Any:
    return dataclasses.asdict(table[name])


def _pos_entry(doc: Document, pos: Any) -> Any:
    if isinstance(pos, str):
        resolved = doc.positions[pos]
        return (
            dataclasses.asdict(resolved)
            if isinstance(resolved, PositionSpec)
            else str(resolved)
        )
    return dataclasses.asdict(pos) if isinstance(pos, PositionSpec) else pos


def _featurizer_entry(doc: Document, ref: Any) -> Any:
    if ref is None:
        return None
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    return [dataclasses.asdict(doc.featurizers[name]) for name in chain]


def cohort_key(doc: Document, data_identity: Mapping[str, str]) -> str | None:
    """The identity of the forward a fit's members can **share** (§4
    "Cohorts"), or ``None`` for a document that fits nothing.

    Points of a campaign whose keys agree may fit together: one forward per
    optimizer step over the concatenation of every member's minibatch, each
    member's writes landing on its own rows. That is one forward exactly when
    every member runs the same network realization over the same rows in the
    same frame — so the key is the realization (``canonical_model_ref``, dtype
    and quantization included), the data identity per input role (the content
    digest and field the rows are encoded from, as the group digests carry
    it), and the ``segments`` section that frames them. Nothing a member
    trains, sweeps or schedules is in it: featurizer specs, the seed, the
    objective, the optimizer, the step budget, the eval cadence and the
    write's address are each member's own, applied to its own rows — with one
    exception. Whether an intervened model's sites reach an **attention
    interior** (:func:`_attention_requirement`) decides the attention
    implementation its forward runs under, and a cohort forward runs under one
    implementation for every member; so that requirement, per intervened
    model, is in the key, and a member whose sites need eager never shares a
    forward with one whose sites do not.
    """
    if doc.train is None:
        return None
    body = {
        "network": canonical_model_ref(doc.model),
        "data": dict(data_identity),
        "segments": _method_section(doc, "segments"),
        "attention": {
            name: _attention_requirement(doc, name, str(im.input))
            for name, im in doc.intervened_models.items()
        },
    }
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), default=_encode
        ).encode()
    ).hexdigest()


def _method_section(doc: Document, name: str) -> Any:
    """One section of the document's ``method`` group as authored (§1), or
    ``None`` when absent."""
    method = doc.raw.get("method")
    return method.get(name) if isinstance(method, Mapping) else None


def fit_cohorts(
    docs: Sequence[Document], identities: Sequence[Mapping[str, str]]
) -> tuple[tuple[int, ...], ...]:
    """Partition a campaign's point indices into fit cohorts (§4 "Cohorts"):
    the train points sharing a :func:`cohort_key` form one cohort each, in
    first-appearance order; every other point stands alone. Every index
    appears exactly once, so a campaign loop can run the partition in order
    and cover the campaign."""
    if len(docs) != len(identities):
        raise ValueError(
            f"{len(docs)} documents but {len(identities)} data identities — the "
            "two are in lockstep per point"
        )
    cohorts: dict[str, list[int]] = {}
    order: list[tuple[int, ...] | str] = []
    for index, (doc, identity) in enumerate(zip(docs, identities)):
        key = cohort_key(doc, identity)
        if key is None:
            order.append((index,))
            continue
        if key not in cohorts:
            cohorts[key] = []
            order.append(key)
        cohorts[key].append(index)
    return tuple(
        tuple(cohorts[entry]) if isinstance(entry, str) else entry for entry in order
    )


def _encode(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"unencodable {type(obj).__name__} in a plan closure")
