"""The engine contract: capabilities, `requires`, and routing (spec §8).

An engine is anything that can execute compiled interventions — nnsight, Megatron,
SGLang, or the in-repo reference over native pytorch hooks
(:mod:`causalab.neural.engines.pytorch_hooks`). The document never knows which one
runs it: ``requires`` derives the capability set a document needs, each
engine declares what it supports, and ``choose_engine`` picks the first
engine whose capabilities cover the requirement — with refusal messages
generated from the missing capabilities, never hand-written per case.
"""

from __future__ import annotations

import abc
import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.errors import ValidationError
from causalab.protocol.plan import generated_budget
from causalab.protocol.registry import ENGINES, capability, write_capabilities
from causalab.protocol.resolution import Denominator, Resolution
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.schema import Document, MetricSpec

__all__ = [
    "Engine",
    "CAPABILITIES",
    "CONTINUATIONS_FILE",
    "ExecutionRequest",
    "RunResult",
    "choose_engine",
    "component_capability",
    "requires",
    "requires_campaign",
    "train_capabilities",
]

#: The engine *names* ``--engine`` accepts, and the default selection.
#: ``"auto"`` means "every installed engine, reference first" — routing rather
#: than a pin (§8). One definition because it had three: the parser's default
#: and two `getattr(args, "engine", …)` fallbacks, which had drifted to
#: disagree (`"auto"` vs `"pytorch_hooks"`), so a non-argparse caller got
#: routing on one path and a silent pin to the reference engine on the other.
#: The names themselves are the capability registry's
#: (:data:`~causalab.protocol.registry.ENGINES`) — the rows name engines, so
#: the flag and the rows cannot disagree about which exist.
ENGINE_CHOICES: tuple[str, ...] = (*ENGINES, "auto")
DEFAULT_ENGINE = "auto"

#: The closed capability vocabulary (§8). Component capabilities
#: (``component:<name>``, ``component:<name>:write``) are *generated*, one per
#: entry of the closed :data:`~causalab.protocol.schema.Component` vocabulary —
#: two engines with different site surfaces route on them, and the vocabulary
#: stays closed because ``Component`` already is. The one component-shaped
#: verb (``writable_attention_probs``: the pattern's write goes through the
#: attention function, not a hook) is likewise generated — from the
#: ``write_capability`` cell of the capability rows — so the verb exists
#: because a row says a write there costs more than the component entry.
CAPABILITIES: tuple[str, ...] = (
    "grad",
    "paired_forward",
    "full_logits",
    *sorted(write_capabilities()),
    "pytorch_fn_local",
    "generate",
    "quantized_weights",
    # the three training facts a fit can author that an engine's loop may not
    # honour (§2.11, rule 30): a free ``params`` tensor in ``train.params``, a
    # ``train.precision`` other than fp32, an ``updates``-counted ``eval``.
    # Neither shipped engine offers them, and each used to be refused inside
    # the train loop, after the weights had loaded.
    "train_free_params",
    "train_loss_precision",
    "train_eval_updates",
)


def component_capability(component: str, *, write: bool = False) -> str:
    """The generated capability entry for serving ``component`` (§8) —
    reading it, or with ``write=True``, landing a write on it."""
    return f"component:{component}:write" if write else f"component:{component}"


#: Metric kinds that need the whole vocabulary materialized (§8) — but only
#: when their read actually taps ``lm_head``. ``class_probs`` always does
#: (validation binds it to a vocabulary projection); ``top_k`` ranks whatever
#: axis its read has, and a top-k over a 4k-wide residual stream or a 100k-wide
#: SAE code obliges no vocabulary projection at all. Charging it ``full_logits``
#: would route such a document onto a full-vocab engine for nothing.
_FULL_VOCAB_METRICS = frozenset({"top_k", "class_probs"})


def _metric_read_obliges_full_projection(doc: Document, metric: MetricSpec) -> bool:
    """Whether serving ``metric``'s read means materializing the vocabulary.

    Deliberately NOT :func:`~causalab.protocol.schema.metric_reads_vocabulary`:
    that predicate asks what the read *hands the metric* (token ids, or a
    featurizer's latents / a ``dims`` re-index), which governs softmaxing and
    decoding. This one asks what the engine must *compute upstream* — and a
    featurized ``lm_head`` read still consumes the whole projection, its
    featurizer merely re-expresses it. The two questions diverge exactly
    there. A ``dims`` slice is the one transform that needs only its named
    rows, matching the saved-read rule above.
    """
    read = doc.reads.get(str(metric.of))
    if read is None:
        return False
    site = doc.sites.get(str(read.site))
    return site is not None and site.component == "lm_head" and read.dims is None


def requires(doc: Document) -> frozenset[str]:
    """The capability set one concrete document needs — derived, never
    authored (§6).

    Component needs are part of the set: every site a read or write
    references contributes ``component:<name>`` (writes also
    ``component:<name>:write``), so a document is routed by *what it touches*,
    not only by the coarse §8 verbs — the honest answer once two engines with
    different site surfaces exist. Stream- and layer-level constraints stay
    engine-internal: they depend on the loaded model, which routing never
    sees."""
    needed: set[str] = set()
    if doc.train is not None:
        needed.add("grad")
        needed.update(train_capabilities(doc))
    for read in doc.reads.values():
        needed.add(component_capability(doc.sites[str(read.site)].component))
    for write in doc.writes.values():
        component = doc.sites[str(write.site)].component
        needed.add(component_capability(component))
        needed.add(component_capability(component, write=True))
    if doc.model.quantization is not None:
        needed.add("quantized_weights")
    for im in doc.intervened_models.values():
        if not isinstance(im.writes, tuple):
            raise AssertionError(
                "requires() takes a concrete point document — expand sweeps first"
            )
        for ename in im.writes:
            write = doc.writes[ename]
            payload = write.do.payload
            operand_names = (
                [payload]
                if isinstance(payload, str)
                else [v for v in payload.values() if isinstance(v, str)]
                if isinstance(payload, Mapping)
                else []
            )
            for op in operand_names:
                read = doc.reads.get(op)
                if read is not None and str(read.input) != str(im.input):
                    needed.add("paired_forward")
            if write.do.mechanism == "pytorch_fn":
                needed.add("pytorch_fn_local")
            site = doc.sites[str(write.site)]
            if isinstance(site.component, str):
                verb = capability(site.component).write_capability
                if verb is not None:
                    needed.add(verb)
    saved = {entry.value for entry in doc.save}
    for rname, read in doc.reads.items():
        if rname in saved and read.dims is None:
            site = doc.sites[str(read.site)]
            if site.component == "lm_head":
                needed.add("full_logits")
    for metric in doc.metrics.values():
        if metric.kind in _FULL_VOCAB_METRICS and _metric_read_obliges_full_projection(
            doc, metric
        ):
            needed.add("full_logits")
    for read in doc.reads.values():
        if generated_budget(doc, read.pos) is not None:
            # a continuation to address means the engine must decode one
            needed.add("generate")
            break
    return frozenset(needed)


def train_capabilities(doc: Document) -> frozenset[str]:
    """The training verbs a fit's own fields oblige (§2.11): each is a fact
    the document decides and an engine's loop may not implement, so it is
    routed on here and refused by name under rule 30 when the routed engine
    lacks it (:func:`causalab.protocol.validate.check_engine_support`) — the
    one derivation both read, so routing and the rule cannot disagree.

    * ``train_free_params`` — a ``train.params`` entry names a ``params``
      entry (a free tensor, §2.6) rather than a featurizer or a slot;
    * ``train_loss_precision`` — ``train.precision.feature`` or ``.loss`` is
      authored as anything but ``fp32``;
    * ``train_eval_updates`` — ``train.eval.every`` counts ``updates``.
    """
    train = doc.train
    if train is None:
        return frozenset()
    needed: set[str] = set()
    if any(pname in doc.params for pname in train.params):
        needed.add("train_free_params")
    if train.precision is not None and any(
        isinstance(value, str) and value != "fp32" for value in train.precision.values()
    ):
        needed.add("train_loss_precision")
    if train.eval is not None and "updates" in train.eval["every"]:
        needed.add("train_eval_updates")
    return frozenset(needed)


#: The request-keyed engine output a decode writes when
#: :attr:`ExecutionRequest.decoding` is set: one row per generated row of every
#: decoding group — ``point``, ``point_digest``, ``model``, ``input``,
#: ``example``, ``steps`` (the budget), ``width``, ``truncated``, the real
#: ``token_ids``, ``text`` and per-token char ``offsets``. Not a ``save`` kind:
#: the document's ``save`` section is unchanged, so no document digest moves.
CONTINUATIONS_FILE = "continuations.json"


@dataclasses.dataclass(frozen=True)
class ExecutionRequest:
    """Everything an engine needs to run one document: the concrete points
    (raw trees, artifact fields resolved), their canonical forms and
    digests, coordinates per point, the resolution environment, and where
    outputs land.

    ``execution`` is the request's own execution parameters — ``batch_rows``
    (rows per no-grad forward) and ``fit_rows`` (rows per grad forward of a
    fit), each a positive integer or ``None`` for unbounded — which override
    the engine's constructor defaults for this request and nothing else; an
    absent key leaves the engine's value in force. A workflow step's
    ``execution`` block arrives here (workflow spec §2.2). It is execution,
    never identity (§8): it enters no canonical form, no digest and no
    stamp, and the run receipt is its one recorder."""

    points: tuple[Mapping[str, Any], ...]
    canonical: tuple[Mapping[str, Any], ...]
    digests: tuple[str, ...]
    coords: tuple[Mapping[str, Any], ...]
    document_digest: str
    env: ResolutionEnv
    output_dir: Path
    execution: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    #: How the request's ``generated`` frames are decoded — a workflow
    #: ``behavioral`` step's ``decoding`` block (workflow spec §2.7):
    #: ``{"mode": "deterministic"}`` or ``{"mode": "sampled", "seed", "temperature",
    #: "top_p"}``. ``None`` — every other door — is the greedy decode the
    #: document alone specifies, byte for byte what it produced before this
    #: field existed. Set, the engine also writes :data:`CONTINUATIONS_FILE`
    #: into its result. Execution, not identity: the block enters no
    #: canonical document, no document digest and no stamp; the step record
    #: is its one recorder (``execution.decoding``).
    decoding: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What an execution produced: saved files (save-manifest paths →
    absolute paths on disk) and per-point summaries for `explain`-style
    reporting."""

    files: Mapping[str, Path]
    summaries: tuple[Mapping[str, Any], ...] = ()
    #: Forward groups the engine actually ran across the whole campaign.
    #: ``num_forwards`` (§4) is what the *plan* derives per point; this is what
    #: execution cost, so the two together say how much of §3's cross-point
    #: interning an engine claimed. An engine that shares nothing reports
    #: points × groups; one that interns fully reports the campaign's distinct
    #: group digests (:func:`causalab.protocol.plan.interned_groups`). The
    #: inner passes of a fit are not forward groups and are not counted.
    forwards: int = 0
    #: One :data:`~causalab.protocol.resolution.Resolution` per result cell —
    #: every ``save`` entry of every executed point, in run order. An
    #: ``Unavailable`` cell is a legal cell with nothing to measure (a scoped
    #: slice that selected no rows); it is in the result with its reason code
    #: and it is in the denominator. Never an ``Invalid``: a defect stops
    #: validation before anything executes (spec §4.1).
    cells: tuple[Resolution, ...] = ()

    @property
    def denominator(self) -> Denominator:
        """``eligible`` of ``total`` cells, the excluded ones by reason — the
        numbers a summary reads instead of keeping its own books."""
        return Denominator.of(self.cells)


class Engine(abc.ABC):
    """One execution engine, described by data and entered through one
    method. Implementations own the §8 services (SiteResolver, position
    resolution, planning, mechanisms, featurizers, metrics, training, RNG,
    stamping) internally — the seam is the document, not the services."""

    #: Engine name, for routing messages and ArtifactIdentity stamping.
    name: str = "abstract"
    #: The §8 capability set this engine supports.
    capabilities: frozenset[str] = frozenset()
    #: Components this engine's site resolver serves. The matching
    #: ``component:<name>`` capabilities are generated (never listed in
    #: ``capabilities`` by hand), so the closed vocabulary stays
    #: :data:`~causalab.protocol.schema.Component`.
    components: frozenset[str] = frozenset()
    #: The subset of ``components`` this engine can land a write on.
    writable_components: frozenset[str] = frozenset()
    #: Local engines may run ``pytorch_fn`` writes (§2.8).
    is_local: bool = False

    @property
    def effective_capabilities(self) -> frozenset[str]:
        """``capabilities`` plus the generated component entries — what
        routing actually compares against :func:`requires`."""
        return (
            self.capabilities
            | {component_capability(c) for c in self.components}
            | {component_capability(c, write=True) for c in self.writable_components}
        )

    @abc.abstractmethod
    def execute(self, request: ExecutionRequest) -> RunResult:
        """Run every point and write everything the save manifests name."""


def requires_campaign(docs: Sequence[Document]) -> frozenset[str]:
    """The union of every point's capability needs — a heterogeneous sweep
    routes on the whole campaign, not its first point."""
    needed: frozenset[str] = frozenset()
    for doc in docs:
        needed |= requires(doc)
    return needed


def choose_engine(
    doc: Document | Sequence[Document], engines: Sequence[Engine]
) -> Engine:
    """The first engine whose capabilities cover the document's (or the
    whole campaign's) needs; the refusal message is generated from the
    missing capabilities (§8)."""
    needed = requires(doc) if isinstance(doc, Document) else requires_campaign(doc)
    shortfalls: list[str] = []
    for engine in engines:
        missing = needed - engine.effective_capabilities
        if not missing:
            return engine
        shortfalls.append(f"{engine.name} lacks {sorted(missing)}")
    raise ValidationError(
        13,
        f"no engine supports this document: it requires {sorted(needed)}; "
        + "; ".join(shortfalls),
    )
