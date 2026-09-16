"""The load-error checklist (spec §5) over one concrete document.

:func:`validate_document` runs rules 3–13, 16, 17, 21, 24, 26 and 27 on a parsed, *concrete*
:class:`~causalab.protocol.schema.Document` — a compiled intervention, or an
un-swept document — and reports every independent violation it finds, not
just the first (see the function's own docstring for which two rules gate the
rest). The other rules live where their information lives:

* rule 1 (strict keys) and rule 2 (section order, which warns rather than
  refuses) — :mod:`causalab.protocol.schema`;
* rule 14 (sweep wrappers, point cap) — :mod:`causalab.protocol.sweep`;
* rule 15 (artifact-valued fields) — :mod:`causalab.protocol.resolve`;
* rule 13 needs a selected engine, so it only fires when one is passed —
  and so does rule 30, the engine-supported fit: both run again for the
  routed engine through :func:`check_engine_support` (the compiler's
  ``check_engine``), before any weights load.

Interpretations this module commits to (each surfaced in the PR notes):

* **Rule 8 overlap is conservative.** Two positions "overlap" unless they
  are *provably* disjoint from the document alone: distinct indices of the
  same sign, non-intersecting spans in the same frame, or two different
  prompt variables (template variables occupy disjoint spans). An index
  compared against a variable window is unknowable at load and treated as
  overlapping — refuse-before-run rather than collide-at-run. An
  all-positions spec overlaps every other spec by construction, itself
  included: it is never provably disjoint. Note the rules compare *writes*
  only, so an all-positions read collides with nothing.
* **Rules 8 and 9 compose.** Full-width writes have ``dims = all``. At one
  (site, overlapping pos, model) address: at most one full-width absolute
  write (rule 8), and no two *absolute* writes may intersect in dims
  (rule 9) — additive deltas may overlap anything, absolute-then-additive
  order is the mechanism-class rule (§2.8).
* **Dead declarations** (§0) are reported under rule 11 — the sink rule
  names reads explicitly; unused sites, positions, featurizers and params
  are the same principle and share the rule number.
* **Rule 21 refuses the uninterpretable, not the unexecutable.** A write
  whose operand is read from strictly deeper than the write's own address is
  perfectly runnable — the operand's model runs first, and §2.9's acyclicity
  is what makes that staging legal. It is refused because no edge of the
  network carries information from the source to the target, so the number
  it produces is attributable to no path. Equal depth stays legal: that is
  the two-pass harvest/inject idiom, where the value is injected
  at the very address it was read from. The comparison is the same
  ``(layer, rank)`` order group elision uses, so it is block-order aware —
  a same-layer head → MLP operand is upstream, and the reverse is not.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Callable, Iterable, Mapping

from causalab.protocol.code import (
    CODE_RULE,
    RUNTIME_KEYWORDS,
    function_def,
    resolve_locator_or_refuse,
    signature_problems,
    undeclared_reads,
)
from causalab.protocol.errors import ValidationError, ValidationErrors, suggest
from causalab.protocol.plan import generated_budget, site_depths, static_alignment
from causalab.protocol.registry import (
    ModelInfo,
    capability,
    component_width,
    expert_axis_refusal,
    get_model_info,
    write_policy_refusal,
)
from causalab.protocol.segments import check as check_segments
from causalab.protocol.spans import SpanSpec, static_indices
from causalab.protocol.sweep import band_label
from causalab.protocol.schema import (
    span_length,
    AnnealSchedule,
    Sweep,
    OBJECTIVE_WEIGHT_PREFIX,
    READ_TARGET_METRIC_KINDS,
    ALIGNMENT_CARDINALITIES,
    CodeSpec,
    ADDITIVE_MECHANISMS,
    FEATURIZER_SLOTS,
    GATE_DEFAULT_MAP,
    GATE_MAPS,
    METRIC_DOMAINS,
    TRAINABLE_KINDS,
    Do,
    Document,
    FeaturizerSpec,
    WriteSpec,
    NAMED_SECTIONS,
    PositionSpec,
    ReadSpec,
    RESERVED_NAMES,
    SaveEntry,
    VOCAB_TOP_K_RANKING,
    metric_reads_vocabulary,
)

__all__ = ["check_engine_support", "im_writes", "validate_document"]

_COUNTERFACTUAL_INDEXED = re.compile(r"^counterfactual\[(\d+)\]$")

#: Metric kinds that do **not** name vocabulary entries, and so bind to a read
#: at any component. ``kl`` compares two reads' distributions against each
#: other; ``top_k`` reports indices along whichever axis the read has — a
#: token id on ``lm_head``, a neuron on ``mlp_activation``, a latent on an SAE
#: featurizer output. Every other kind resolves an authored string to a token
#: id, which only an ``lm_head`` read can be indexed by. ``js`` is decided per
#: document, not per kind: unrestricted it compares two whole distributions
#: like ``kl``; with ``restrict`` it resolves answer strings and binds like a
#: token-column kind (the check below).
_ANY_READ_METRIC_KINDS = frozenset({"kl", "top_k"})


def validate_document(
    doc: Document,
    *,
    engine_is_local: bool | None = None,
    engine_capabilities: frozenset[str] | None = None,
    model_info: Callable[[str], ModelInfo] = get_model_info,
) -> None:
    """Run checklist rules 3–13, 16, 17, 21, 24, 26, 27, 29 and 31, plus 30
    when the engine is known (13 only when ``engine_is_local`` or
    ``engine_capabilities`` is given, 30 only under ``engine_capabilities``).

    ``engine_capabilities`` is what the routed engine offers (§8), when the
    caller knows it: rule 13 is read off it (``pytorch_fn_local``) unless
    ``engine_is_local`` says so directly, and rule 30 holds the fit's authored
    training facts to it. ``model_info`` is the third resolution service —
    static model metadata, the registry's by default — which rule 29 reads
    the operands' widths from; no weights, no tokenizer.

    **Independent violations are reported together.** A document with three
    unrelated problems raises a :class:`ValidationErrors` naming all three,
    each with its own path, instead of the first one and a round trip per
    remaining problem. The rules below the gate do not depend on one another,
    so the second failure was knowable when the first was reported; reporting
    it was the only thing missing. One violation is still raised as itself, so
    a single-error message is unchanged.

    **Two rules gate the rest, and keep first-error behaviour**, because the
    checks after them are not merely uninteresting when they fail — they are
    not *evaluable*:

    * rule 3 produces the namespace map rules 4 and 6 read;
    * rule 4 is what makes every name in the document dereferenceable. The
      checks below index ``doc.writes[...]``, ``doc.sites[...]``,
      ``doc.featurizers[...]`` directly, so running them against a document
      whose references do not resolve would raise ``KeyError`` rather than a
      refusal.

    Anything that is not a :class:`ValidationError` propagates immediately: a
    collector that swallowed a ``KeyError`` would turn a bug in this module
    into a silent partial validation.
    """
    names = _check_namespace(doc)  # rule 3 — gates the rest
    _check_references(doc, names)  # rule 4 (+ the rule-5 read bindings) — gates

    independent = [
        functools.partial(_check_capabilities, doc),  # rule 4's capability half
        functools.partial(_check_writes_inert, doc, names),  # rule 6
        functools.partial(_check_membership_and_acyclic, doc),  # rule 7
        functools.partial(_check_write_collisions, doc),  # rules 8 + 9
        functools.partial(_check_save, doc),  # rule 10
        functools.partial(_check_budget_pools, doc),  # rule 4's pool half
        functools.partial(_check_position_gates, doc),  # rule 4's axis half
        functools.partial(_check_sinks, doc),  # rule 11
        functools.partial(_check_trainability, doc),  # rule 12
        functools.partial(_check_generation, doc),  # rule 16
        functools.partial(_check_model_realization, doc),  # rule 17
        functools.partial(_check_operand_reachability, doc),  # rule 21
        functools.partial(_check_code_declarations, doc),  # rule 24
        functools.partial(_check_alignment_declarations, doc),  # rule 26
        functools.partial(check_segments, doc),  # rule 27 (protocol/segments.py)
        functools.partial(_check_kl_operands, doc, model_info),  # rule 29
        functools.partial(_check_metric_positions, doc),  # rule 31
    ]
    if engine_is_local is None and engine_capabilities is not None:
        engine_is_local = "pytorch_fn_local" in engine_capabilities
    if engine_is_local is not None:
        independent.append(
            functools.partial(_check_pytorch_fn, doc, engine_is_local)  # rule 13
        )
    if engine_capabilities is not None:
        independent.append(
            functools.partial(
                _check_train_engine_supported, doc, engine_capabilities
            )  # rule 30
        )

    failures: list[ValidationError] = []
    for check in independent:
        try:
            check()
        except ValidationError as err:
            failures.append(err)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ValidationErrors(failures)


# --------------------------------------------------------------------------- #
# rule 26 — a declared alignment fits its address
# --------------------------------------------------------------------------- #


def _declared_positions(doc: Document) -> list[tuple[str, PositionSpec]]:
    """Every position spec the document authors, with its path: the
    ``positions`` table's concrete entries and the inline specs on reads and
    writes."""
    out: list[tuple[str, PositionSpec]] = []
    for name, spec in doc.positions.items():
        if isinstance(spec, PositionSpec):
            out.append((f"positions.{name}", spec))
    for section, table in (("reads", doc.reads), ("writes", doc.writes)):
        for name, entry in table.items():
            if isinstance(entry.pos, PositionSpec):
                out.append((f"{section}.{name}.pos", entry.pos))
    return out


def _check_alignment_declarations(doc: Document) -> None:
    """Rule 26 — a declared ``alignment`` fits its address (§2.3).

    Document-decidable only, and all of it named on the field: the value is
    one of the five cardinalities; the address can carry one (``all`` is
    every content token of the row and takes no modifiers, a ``generated``
    position is a result rather than one of the pair's inputs); and where the
    document alone fixes the cardinality — an ``index`` is one token per row
    on every input, an unscoped ``span`` one joint window of one width — the
    declaration agrees with it (:func:`~causalab.protocol.plan
    .static_alignment`). What a ``variable`` or ``column`` window resolves to
    needs the tokenizer, and the executor checks that declaration at encode
    time (the encode-time boundary): the pure verbs hold no tokenizer.
    """
    for path, spec in _declared_positions(doc):
        declared = spec.alignment
        if declared is None:
            continue
        where = f"{path}.alignment"
        if declared not in ALIGNMENT_CARDINALITIES:
            raise ValidationError(
                26,
                f"alignment {declared!r} is not one of {list(ALIGNMENT_CARDINALITIES)}"
                + suggest(declared, ALIGNMENT_CARDINALITIES),
                path=where,
            )
        if spec.all is not None:
            raise ValidationError(
                26,
                "an all-positions address carries no alignment: it is every "
                "content token of the row on each input, ragged by construction, "
                "and it takes no modifiers (sec. 2.3) — declare the cardinality "
                "on the index, span, variable or column it stands for",
                path=where,
            )
        if spec.generated is not None:
            raise ValidationError(
                26,
                "a generated position carries no alignment: the continuation is "
                "a result, not one of the pair's inputs, so there is nothing to "
                "pair it with (sec. 2.3)",
                path=where,
            )
        static = static_alignment(doc, spec)
        if static is not None and static != declared:
            shape = (
                "one token"
                if spec.index is not None
                else "one joint span of a fixed width"
            )
            raise ValidationError(
                26,
                f"this address is {static!r} by construction — {shape} per row "
                f"on every input — so declaring {declared!r} contradicts the "
                "document itself",
                path=where,
            )


# --------------------------------------------------------------------------- #
# rule 4, the capability half — a selector or a mechanism the component has not
# --------------------------------------------------------------------------- #


def _check_capabilities(doc: Document) -> None:
    """Every site sub-axis and every write mechanism resolves against the
    component's capability row (``registry.CAPABILITIES``, spec §2.4).

    Two references the reference rule used to let through:
    ``sites.<name>.expert`` *looks* generic while only the
    routed interior has a per-expert axis, and ``attention_probs`` *appears*
    writable while only ``swap`` is. Each was refused by the engine at run
    time, after the document had validated. The same two functions the
    executor calls answer here, so the validator and the engine cannot
    disagree; the run-time refusals stay for a document that arrives
    unvalidated.
    """
    for name, site in doc.sites.items():
        if not isinstance(site.component, str) or site.expert is None:
            continue
        if not capability(site.component).expert_selection:
            raise ValidationError(
                4,
                f"site {name!r}: " + expert_axis_refusal(site.component, site.expert),
                path=f"sites.{name}.expert",
                reason="component_unavailable",
            )
    for ename, write in doc.writes.items():
        site = doc.sites[str(write.site)]
        if not isinstance(site.component, str) or not isinstance(
            write.do.mechanism, str
        ):
            continue
        refusal = write_policy_refusal(ename, site.component, write.do.mechanism)
        if refusal is not None:
            raise ValidationError(
                4, refusal, path=f"writes.{ename}.do", reason="unsupported_mechanism"
            )


# --------------------------------------------------------------------------- #
# rules 13 + 30 — the rules that need the routed engine
# --------------------------------------------------------------------------- #


def check_engine_support(doc: Document, engine_capabilities: frozenset[str]) -> None:
    """The engine-aware half of the checklist — rule 13 (a ``pytorch_fn``
    needs a local engine, §2.8) and rule 30 (the fit as authored needs the
    engine's training verbs, §2.11) — against one engine's capability set.

    The same two rule functions :func:`validate_document` runs when it is
    handed ``engine_capabilities``; public so the compiler can re-enter them
    for the engine routing chose (``compile.check_engine``), which no door
    knows at compile time. Independent violations are reported together, as
    the checklist's are.
    """
    failures: list[ValidationError] = []
    for check in (
        functools.partial(
            _check_pytorch_fn, doc, "pytorch_fn_local" in engine_capabilities
        ),
        functools.partial(_check_train_engine_supported, doc, engine_capabilities),
    ):
        try:
            check()
        except ValidationError as err:
            failures.append(err)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ValidationErrors(failures)


def _check_train_engine_supported(doc: Document, capabilities: frozenset[str]) -> None:
    """Rule 30 — the routed engine can execute the fit as authored (§2.11).

    Three facts a fit authors are decided by the document and honoured, or
    not, by the engine's loop: a free ``params`` tensor in ``train.params``,
    a ``train.precision`` other than fp32, and an ``eval`` counted in
    ``updates``. Each maps to one §8 verb the engine declares or does not
    (``engine.train_capabilities`` is the one derivation routing shares), and
    each used to be discovered inside the loop — after the weights had
    loaded, or not at all: a bf16 loss was digested as bf16 and run at fp32
    with nothing said. The refusal names the field and the missing verb; the
    engine's twin refusals stay for a document that arrived unvalidated.
    """
    train = doc.train
    if train is None:
        return
    if "train_free_params" not in capabilities:
        for i, pname in enumerate(train.params):
            if pname in doc.params:
                raise ValidationError(
                    30,
                    f"train.params names the free tensor {pname!r}, which the "
                    "routed engine cannot train: it declares no "
                    "'train_free_params' capability — featurizer slots only "
                    "(sec. 2.11)",
                    path=f"train.params[{i}]",
                )
    if "train_loss_precision" not in capabilities and train.precision is not None:
        for key in ("feature", "loss"):
            value = train.precision.get(key)
            if isinstance(value, str) and value != "fp32":
                raise ValidationError(
                    30,
                    f"train.precision.{key} is {value!r}, but the routed engine "
                    "computes its loss path in fp32 only: it declares no "
                    "'train_loss_precision' capability, so the authored "
                    "precision would be digested and not executed (sec. 2.11)",
                    path=f"train.precision.{key}",
                )
    if (
        "train_eval_updates" not in capabilities
        and train.eval is not None
        and "updates" in train.eval["every"]
    ):
        raise ValidationError(
            30,
            "train.eval.every counts updates, but the routed engine evaluates "
            "on epoch boundaries only: it declares no 'train_eval_updates' "
            "capability, so the fit would run no eval at all and still save "
            "(sec. 2.11)",
            path="train.eval.every",
        )


# --------------------------------------------------------------------------- #
# rule 29 — kl operands are comparable
# --------------------------------------------------------------------------- #


def _read_chain(read: ReadSpec) -> tuple[str, ...]:
    """A read's featurizer composition as a tuple, ``()`` for none."""
    ref = read.featurizer
    if isinstance(ref, str):
        return (ref,)
    return tuple(ref) if isinstance(ref, tuple) else ()


def _stage_output_width(spec: FeaturizerSpec, input_width: int) -> int | None:
    """The §2.5 chain rule on one stage, torch-free: ``subspace`` / ``pca``
    project to ``k``, the width-preserving kinds pass ``input_width`` through,
    and a loaded SAE's dictionary size is in its bundle, not its spec
    (``None``). The twin of ``neural.shared.featurizers.stage_output_width``,
    which the engine sizes stages with."""
    kind = spec.kind if isinstance(spec.kind, str) else "identity"
    if kind in ("subspace", "pca"):
        return spec.k if isinstance(spec.k, int) else None
    if kind == "sae":
        return None
    return input_width


def _effective_width(
    doc: Document, read: ReadSpec, info: ModelInfo | None
) -> int | None:
    """The width of what a read hands its consumer, from static model config
    alone: the site's component width (one head's slice under ``head``),
    folded through the featurizer chain, then narrowed to a ``dims`` slice —
    or ``None`` where only a loaded bundle or the engine could say."""
    if isinstance(read.dims, tuple):
        return len(read.dims)
    if info is None:
        return None
    site = doc.sites[str(read.site)]
    if not isinstance(site.component, str) or not (
        site.head is None or isinstance(site.head, int)
    ):
        return None
    try:
        running: int | None = component_width(info, site.component, head=site.head)
    except ValidationError:
        return None  # not a feature space (a pattern, a ranking): the engine's call
    for member in _read_chain(read):
        if running is None:
            return None
        running = _stage_output_width(doc.featurizers[member], running)
    return running


def _frame(doc: Document, pos: Any) -> str:
    """The token-position frame a read addresses (§2.3): ``"prompt"``, or
    ``"generated"`` when its position carries a decode budget."""
    return "prompt" if generated_budget(doc, pos) is None else "generated"


def _check_kl_operands(doc: Document, model_info: Callable[[str], ModelInfo]) -> None:
    """Rule 29 — ``kl`` operands are comparable (§2.10): the two reads hand
    the metric distributions over the **same effective width**, through the
    **same transform**, in the **same token-position frame**.

    Rule 4 already holds both reads to one component label; a label is not a
    distribution's shape. A ``block_output`` read through a ``k=8`` subspace
    and the raw read at the same site share a label and are incomparable —
    the run discovered it as a shape error, after the weights had loaded. The
    three halves, all weights-free:

    * **width** — the component's width from the registry's static config,
      folded through each read's featurizer chain and ``dims`` (the engine's
      ``_featurizer_input`` walk, offline). Decided when both sides are
      derivable; a loaded SAE's width is its bundle's, and stays the engine's;
    * **transform** — the featurizer chains and ``dims`` selections are the
      same names. Two subspaces of one ``k`` are two bases, and a divergence
      across bases measures nothing;
    * **frame** — both reads address the prompt or both the continuation
      (:func:`~causalab.protocol.plan.generated_budget`). A prompt-frame
      read wider than one position is rule 31's; the per-row *count* of two
      continuation windows (a ``variable`` anchor said on one side and not
      the other) needs the tokenizer and is the executor's named residue
      (``metrics.compute_windowed_metric``, §5's invariant).
    """
    info: ModelInfo | None = None
    for qname, metric in doc.metrics.items():
        if metric.kind != "kl":
            continue
        target = metric.fields.get("target")
        if not isinstance(target, str) or target not in doc.reads:
            continue  # rule 4's report
        of_read, target_read = doc.reads[str(metric.of)], doc.reads[target]
        p = f"metrics.{qname}.target"
        of_chain, target_chain = _read_chain(of_read), _read_chain(target_read)
        if of_chain != target_chain or of_read.dims != target_read.dims:
            raise ValidationError(
                29,
                f"kl {qname!r} compares {metric.of!r} and {target!r} through "
                f"different transforms — featurizer {list(of_chain)} / dims "
                f"{of_read.dims!r} against featurizer {list(target_chain)} / dims "
                f"{target_read.dims!r} — so the two distributions are not over "
                "one axis; a comparison needs the same re-expression on both "
                "sides (sec. 2.10)",
                path=p,
            )
        if info is None and isinstance(doc.model.key, str):
            try:
                info = model_info(doc.model.key)
            except ValidationError:
                info = None  # an unregistered key: canonicalization's refusal
        of_width = _effective_width(doc, of_read, info)
        target_width = _effective_width(doc, target_read, info)
        if (
            of_width is not None
            and target_width is not None
            and of_width != target_width
        ):
            raise ValidationError(
                29,
                f"kl {qname!r} compares {metric.of!r} ({of_width} wide) with "
                f"{target!r} ({target_width} wide): two distributions over "
                "different widths have no position-for-position terms "
                "(sec. 2.10)",
                path=p,
            )
        of_frame, target_frame = _frame(doc, of_read.pos), _frame(doc, target_read.pos)
        if of_frame != target_frame:
            raise ValidationError(
                29,
                f"kl {qname!r} compares {metric.of!r} in the {of_frame} frame "
                f"with {target!r} in the {target_frame} frame — a prompt "
                "position and a continuation step are different token-position "
                "frames (sec. 2.3), so nothing pairs them (sec. 2.10)",
                path=p,
            )


# --------------------------------------------------------------------------- #
# rule 31 — a metric reduces one position per example
# --------------------------------------------------------------------------- #


def _check_metric_positions(doc: Document) -> None:
    """Rule 31 — a metric reduces one position per example (§2.10), so a
    prompt-frame read whose address the document fixes to more than one
    token cannot feed one.

    The engine reduces a metric's read at one position per row
    (``metrics._last_pos_rows``), and refused a wider read after the weights
    had loaded. Where the document alone fixes the width — an ``all``
    address, an unscoped ``span`` wider than one token, an ``indices`` set
    or union of more than one — the refusal belongs here. A ``variable`` or
    ``column`` window is as wide as the tokenizer makes it, and stays the
    executor's named residue (§5's invariant). A **continuation** read is
    exempt by design: it addresses as many steps as the row generated, and
    its metric reduces per step (``compute_windowed_metric``).
    """
    for qname, metric in doc.metrics.items():
        operands = [("of", metric.of)]
        if metric.kind == "kl":
            operands.append(("target", metric.fields.get("target")))
        for field, rname in operands:
            if not isinstance(rname, str) or rname not in doc.reads:
                continue  # rule 4's report
            pos = doc.reads[rname].pos
            spec = doc.positions.get(pos) if isinstance(pos, str) else pos
            if not isinstance(spec, PositionSpec) or spec.generated is not None:
                continue
            if spec.all is not None:
                shape = "every content token of the row"
            else:
                fixed = static_indices(spec)
                if fixed is None or len(fixed) <= 1:
                    continue
                shape = f"{len(fixed)} positions by construction"
            raise ValidationError(
                31,
                f"metric {qname!r} reduces one position per example, but its "
                f"read {rname!r} addresses {shape} — bind the metric to a "
                "one-token address, or address the continuation frame, whose "
                "metrics reduce per step (sec. 2.3, sec. 2.10)",
                path=f"metrics.{qname}.{field}",
            )


# rule 24 — a code declaration agrees with the source it names
# --------------------------------------------------------------------------- #


def _check_code_declarations(doc: Document) -> None:
    """§5 rule 24 — the declared function exists as Python source, its
    declared arguments fit its signature, and no read it makes is both
    detectable and undeclared.

    The three halves fail for three different reasons and are worth naming
    separately:

    * **the locator.** A code reference is identified by source bytes, so a
      name that resolves to no ``.py`` file has no identity to cover. Refused.
    * **the arguments.** An argument the function requires and the document
      does not give is a value that comes from somewhere else — a default, a
      module global, an environment — and none of those are in the digest.
      Refused when the signature is *readable*; a function built by a factory
      (the arithmetic golden's ``apply_target_<n>``) has no ``def`` to read
      and this half simply does not run.
    * **the reads.** ROME's corruption function took its noise scale from an
      externally selected file. A literal ``open("…")`` or ``os.getenv("…")``
      the declaration does not name is refused here — *statically*, from the
      AST. Nothing is imported and nothing is executed: this is a declaration
      checker, not a sandbox, and a read routed through a variable is invisible
      to it by design (:mod:`causalab.protocol.code`).
    """
    for name, spec in doc.code.items():
        if not isinstance(spec.locator, str):
            continue  # a swept locator is concrete at every point, checked there
        path = f"code.{name}"
        resolved = resolve_locator_or_refuse(spec.locator, path=f"{path}.locator")
        fn = function_def(resolved)
        if fn is None:
            continue
        supplied = _runtime_keywords(spec)
        problems = signature_problems(fn, args=spec.args, supplied=supplied)
        problems += [
            f"{spec.locator} {problem}"
            for problem in undeclared_reads(
                fn,
                env_inputs=spec.env_inputs,
                data_inputs=spec.data_inputs.values(),
            )
        ]
        if problems:
            raise ValidationError(
                CODE_RULE,
                f"code {name!r} does not agree with {spec.locator!r}: "
                + "; ".join(problems),
                path=path,
            )


def _runtime_keywords(spec: CodeSpec) -> tuple[str, ...]:
    """The keywords the mechanism will pass because the declaration asked for
    them. Declared, so the signature check expects them; not declared, so it
    does not (§2.8.1)."""
    declared = {"row_roles": bool(spec.row_roles)}
    return tuple(key for key in RUNTIME_KEYWORDS if declared[key])


# --------------------------------------------------------------------------- #
# rule 3 — one global namespace, no reserved names
# --------------------------------------------------------------------------- #


def _check_namespace(doc: Document) -> dict[str, str]:
    names: dict[str, str] = {}
    for section in NAMED_SECTIONS:
        table = getattr(doc, section)
        for name in table:
            if name in RESERVED_NAMES or _COUNTERFACTUAL_INDEXED.match(name):
                raise ValidationError(
                    3, f"{name!r} is a reserved name", path=f"{section}.{name}"
                )
            if name in names:
                raise ValidationError(
                    3,
                    f"name {name!r} is declared in both {names[name]!r} and {section!r} — "
                    "method sections 2–10 share one namespace",
                    path=f"{section}.{name}",
                )
            names[name] = section
    return names


# --------------------------------------------------------------------------- #
# rule 4 — every reference resolves (and rule 5, read bindings)
# --------------------------------------------------------------------------- #


def _valid_roles(doc: Document) -> set[str]:
    roles = {"base"}
    counterfactual = doc.data.get("counterfactual")
    if counterfactual is None:
        return roles
    if isinstance(counterfactual, tuple):
        roles.update(f"counterfactual[{j}]" for j in range(len(counterfactual)))
        # a singular reference to an array-valued counterfactual is not a role
    else:
        roles.add("counterfactual")
    return roles


def _check_pos_ref(doc: Document, pos: Any, path: str) -> None:
    if isinstance(pos, str):
        if pos not in doc.positions:
            raise ValidationError(
                4, f"position {pos!r} is not declared in positions", path=path
            )
    elif not isinstance(pos, PositionSpec):
        raise ValidationError(4, f"unresolvable position {pos!r}", path=path)


def _check_featurizer_ref(doc: Document, ref: Any, path: str) -> None:
    """Rule 4 for each stage; rule 12 for the composition itself (§2.5).

    A chain's stages are applied left-to-right, and a featurizer's width is
    **derived from its position in the chain** — the engine sizes a stage by
    walking the chain until it reaches the name it is sizing
    (``executor_base._featurizer_input``). That is what makes the two shapes
    below illegal rather than merely odd, and this check used to let both
    through:

    * an **empty** chain is a ``featurizer`` key that re-expresses nothing.
      ``identity`` is a declarable kind, so a document meaning "no featurizer"
      either omits the key or names it; ``[]`` is a third spelling of the same
      thing that no canonical form records, and this layer does not do silent
      defaults.
    * a **repeated** stage has two positions in the chain and one derived
      width. The width walk returns the running width at the *first*
      occurrence, so the second stage would be built at the wrong input width,
      and both stages would share one parameter slot. There is no spelling
      that fixes that: two applications of one map are two declared
      featurizers.
    """
    if ref is None:
        return
    if isinstance(ref, str):
        chain: tuple[str, ...] = (ref,)
    elif isinstance(ref, tuple):
        chain = tuple(ref)
    else:
        # a surviving sweep wrapper, or anything else the parse let past:
        # validation runs on concrete point documents (§3), so this is a
        # caller error rather than a document one — but it is not silence
        raise ValidationError(
            4, f"unresolvable featurizer reference {ref!r}", path=path
        )
    if not chain:
        raise ValidationError(
            12,
            "an empty featurizer composition re-expresses nothing — omit the "
            "key, or name a featurizer of the 'identity' kind if the stage is "
            "deliberate (§2.5)",
            path=path,
        )
    for name in chain:
        if name not in doc.featurizers:
            raise ValidationError(4, f"featurizer {name!r} is not declared", path=path)
    repeated = sorted({name for name in chain if chain.count(name) > 1})
    if repeated:
        raise ValidationError(
            12,
            f"featurizer composition {list(chain)} repeats {repeated} — a "
            "featurizer's width is derived from its position in the chain, so a "
            "name appearing twice has two input widths and one derived width, "
            "and both stages would share one parameter slot. Declare a second "
            "featurizer instead (§2.5).",
            path=path,
        )


def _param_slot_names(doc: Document) -> set[str]:
    """Every addressable param name: ``params`` entries plus the
    auto-declared ``<featurizer>.<slot>`` names (§2.5)."""
    out = set(doc.params)
    for fname, spec in doc.featurizers.items():
        kind = spec.kind if isinstance(spec.kind, str) else "identity"
        for slot in FEATURIZER_SLOTS.get(kind, ()):
            out.add(f"{fname}.{slot}")
    return out


def _check_references(doc: Document, names: dict[str, str]) -> None:
    roles = _valid_roles(doc)
    model_names = {"original", *doc.intervened_models}

    for rname, read in doc.reads.items():
        p = f"reads.{rname}"
        if read.site not in doc.sites:
            raise ValidationError(
                4, f"site {read.site!r} is not declared", path=f"{p}.site"
            )
        _check_pos_ref(doc, read.pos, f"{p}.pos")
        if read.model not in model_names:
            raise ValidationError(
                5,
                f"read model {read.model!r} is neither 'original' nor a declared "
                "intervened_model",
                path=f"{p}.model",
            )
        if read.input not in roles:
            raise ValidationError(
                5,
                f"read input {read.input!r} is not a valid role ({sorted(roles)})",
                path=f"{p}.input",
            )
        if read.model != "original":
            im = doc.intervened_models[str(read.model)]
            if read.input != im.input:
                raise ValidationError(
                    5,
                    f"read {rname!r} declares input {read.input!r} but its model "
                    f"{read.model!r} runs on {im.input!r} — the restated binding "
                    "must match (§2.7)",
                    path=f"{p}.input",
                )
        _check_featurizer_ref(doc, read.featurizer, f"{p}.featurizer")

    for ename, write in doc.writes.items():
        p = f"writes.{ename}"
        if write.site not in doc.sites:
            raise ValidationError(
                4, f"site {write.site!r} is not declared", path=f"{p}.site"
            )
        _check_pos_ref(doc, write.pos, f"{p}.pos")
        _check_featurizer_ref(doc, write.featurizer, f"{p}.featurizer")
        if str(write.do.mechanism) == "pytorch_fn":
            named = write.do.payload["code"]
            if isinstance(named, str) and named not in doc.code:
                raise ValidationError(
                    4,
                    f"pytorch_fn names code declaration {named!r}, which the "
                    f"'code' section does not declare{suggest(named, doc.code)}",
                    path=f"{p}.do.pytorch_fn.code",
                )

    for mname, im in doc.intervened_models.items():
        p = f"intervened_models.{mname}"
        if im.input not in roles:
            raise ValidationError(
                7,
                f"intervened_model input {im.input!r} is not a valid role",
                path=f"{p}.input",
            )
        for ename in im_writes(im.writes):
            if ename not in doc.writes:
                raise ValidationError(
                    4, f"write {ename!r} is not declared", path=f"{p}.writes"
                )

    for qname, metric in doc.metrics.items():
        p = f"metrics.{qname}"
        if metric.of not in doc.reads:
            raise ValidationError(
                4, f"metric 'of' {metric.of!r} is not a read", path=f"{p}.of"
            )
        of_read = doc.reads[str(metric.of)]
        of_site = doc.sites[str(of_read.site)]
        # What the read actually hands the metric. The site alone cannot
        # answer this: a featurizer re-expresses an lm_head projection in its
        # own latents and `dims` re-indexes a slice of it, so either one takes
        # the value out of token-id space even though the site says lm_head.
        is_vocabulary = metric_reads_vocabulary(doc, metric)
        if of_site.component != "lm_head":
            not_vocab_because = f"taps {of_site.component!r}"
        elif of_read.featurizer is not None:
            not_vocab_because = (
                "taps 'lm_head' through a featurizer, whose latents are not token ids"
            )
        else:
            not_vocab_because = (
                "taps 'lm_head' through a 'dims' slice, whose re-indexed "
                "entries are not token ids"
            )
        binds_any_read = metric.kind in _ANY_READ_METRIC_KINDS or (
            metric.kind == "js" and "restrict" not in metric.fields
        )
        if not binds_any_read and not is_vocabulary:
            # not in the §5 checklist: the token-space kinds name vocab
            # entries, which only a plain lm_head read produces (an
            # interpretation this loader commits to; surfaced in the PR notes)
            raise ValidationError(
                4,
                f"metric {qname!r} names vocabulary tokens, but its read "
                f"{metric.of!r} {not_vocab_because} — token-space metric "
                "kinds bind to plain lm_head reads (no featurizer, no dims)",
                path=f"{p}.of",
            )
        if (
            metric.kind == "top_k"
            and metric.fields.get("by") == VOCAB_TOP_K_RANKING
            and not is_vocabulary
        ):
            # `top_k` itself is axis-agnostic, but its normalizing ranking is
            # not: a softmax across neurons or SAE latents normalizes over an
            # axis that is not an event space, so the numbers it emits would
            # be probabilities of nothing.
            raise ValidationError(
                4,
                f"metric {qname!r} ranks by {VOCAB_TOP_K_RANKING!r}, which "
                f"softmaxes a vocabulary, but its read {metric.of!r} "
                f"{not_vocab_because} — rank by 'value' or 'abs_value' on a "
                "read that is not a vocabulary projection (§2.10)",
                path=f"{p}.by",
            )
        if METRIC_DOMAINS.get(str(metric.kind)) == "ids":
            # an ids-domain kind reduces the tokens a decode produced, so its
            # read has to be one that decoded something: in the prompt frame
            # there are no produced tokens to reduce, only given ones
            of_pos = of_read.pos
            of_spec = doc.positions.get(of_pos) if isinstance(of_pos, str) else of_pos
            if not (
                isinstance(of_spec, PositionSpec) and of_spec.generated is not None
            ):
                raise ValidationError(
                    4,
                    f"metric {qname!r} is a {metric.kind!r} over {metric.of!r}, but "
                    f"that read addresses the prompt — {metric.kind!r} reduces the "
                    "tokens a decode produced, so it binds to a read whose position "
                    "carries 'generated' (§2.3, §2.10)",
                    path=f"{p}.of",
                )
        if metric.kind in READ_TARGET_METRIC_KINDS:
            target = metric.fields.get("target")
            if target not in doc.reads:
                raise ValidationError(
                    4,
                    f"{metric.kind} target {target!r} is not a read (§2.10)",
                    path=f"{p}.target",
                )
            target_site = doc.sites[str(doc.reads[str(target)].site)]
            if target_site.component != of_site.component:
                raise ValidationError(
                    4,
                    f"{metric.kind} compares two reads' distributions, but "
                    f"{metric.of!r} taps {of_site.component!r} and {target!r} taps "
                    f"{target_site.component!r}",
                    path=f"{p}.target",
                )

    _check_dead_rules_have_a_fit(doc)
    if doc.train is not None:
        _check_train_references(doc)

    for i, entry in enumerate(doc.save):
        if entry.kind is not None:
            continue  # a non-value kind (§2.12) names no declaration
        if (
            entry.value not in doc.reads
            and entry.value not in doc.metrics
            and entry.value not in doc.featurizers
            and entry.value not in names  # declared-but-unsaveable is rule 10
        ):
            raise ValidationError(
                4,
                f"save value {entry.value!r} is not declared",
                path=f"save[{i}].value",
            )


def _check_dead_rules_have_a_fit(doc: Document) -> None:
    """§2.5 ``dead`` is a rule for units that close *during a fit*: a gate
    authoring one must be trained in this document — named (whole or by
    slot) in ``train.params``. On a gate no step ever moves the rule would
    be digested and never act, which is the silent kind of wrong; the parser
    already refuses it beside ``file_path``, and this is the same refusal for
    a gate that is merely not in the fit."""
    trained = (
        {pname.split(".", 1)[0] for pname in doc.train.params}
        if doc.train is not None
        else set()
    )
    for name, spec in doc.featurizers.items():
        if spec.dead is None or name in trained:
            continue
        rule = next(iter(spec.dead))
        why = (
            "this document has no train section"
            if doc.train is None
            else f"{name!r} is not in train.params"
        )
        raise ValidationError(
            4,
            f"featurizer {name!r} authors dead.{rule}, a rule for units that go "
            f"hard-off during a fit, but {why} — no step would ever apply it (§2.5)",
            path=f"featurizers.{name}.dead",
        )


def _check_budget_pools(doc: Document) -> None:
    """Rule 4 for §2.5 ``pool``: every *fitted* gate authoring one pool name
    is a ``budget`` gate (every arm of a swept map) — a loaded one may be any
    map, since a pooled readout only cuts a joint ranking — and the members
    agree on what makes them one pool: one ``k_schedule`` and one
    ``stop_grad_shift`` on a fit, one ``top_k`` when loaded, and not a mix of
    the two. A pool of one is legal (it is the lone gate). The unit-count bound
    on the schedule is the build's: only the executor knows every member's
    width."""
    pools: dict[str, list[str]] = {}
    for name, spec in doc.featurizers.items():
        if spec.kind == "gate" and isinstance(spec.pool, str):
            pools.setdefault(spec.pool, []).append(name)
    trained = (
        {p.split(".", 1)[0] for p in doc.train.params}
        if doc.train is not None
        else set()
    )
    for pool, names in pools.items():
        specs = [doc.featurizers[n] for n in names]
        for name, spec in zip(names, specs):
            if spec.file_path is not None:
                continue  # a pooled readout: the maps must agree (below), not be budget
            maps = (
                list(spec.parametrization.values)
                if isinstance(spec.parametrization, Sweep)
                else [spec.parametrization]
            )
            if any(m != "budget" for m in maps):
                raise ValidationError(
                    4,
                    f"featurizer {name!r} is in pool {pool!r} but is not a "
                    "budget gate on every arm — a pool shares one budget, which "
                    "only the budget map draws (§2.5)",
                    path=f"featurizers.{name}.pool",
                )
        loaded = {spec.file_path is not None for spec in specs}  # a Sweep is loaded too
        if len(loaded) != 1:
            raise ValidationError(
                4,
                f"pool {pool!r} mixes fitted and loaded gates {names} — a "
                "pool is one ranking, fitted together or read out together (§2.5)",
                path=f"featurizers.{names[0]}.pool",
            )
        if not loaded.pop():
            partial = [n for n in names if n not in trained]
            if partial and len(partial) != len(names):
                raise ValidationError(
                    4,
                    f"pool {pool!r}: {partial} are not in train.params while "
                    f"the rest of the pool is — one pool, one fit (§2.5)",
                    path="train.params",
                )
        # `forward` too: a straight-through member thresholds its share of the
        # pooled mask at ½ while its co-members keep the soft share — one pool,
        # one ranking on one scale (§2.5 the mapping form)
        for field in (
            "parametrization",
            "forward",
            "k_schedule",
            "stop_grad_shift",
            "top_k",
        ):
            values = {repr(getattr(spec, field)) for spec in specs}
            if len(values) != 1:
                raise ValidationError(
                    4,
                    f"pool {pool!r}: its members {names} disagree on "
                    f"{field!r} — one pool is one ranking on one scale, draws one "
                    "budget and is cut at one count (§2.5); author the same value on "
                    "every member"
                    # the mapping form is not swept, so an `axes` entry is not a
                    # remedy for `forward` — the parser would refuse it
                    + (
                        ""
                        if field == "forward"
                        else ", or one `axes` entry referenced by all of them (§3.2)"
                    ),
                    # `forward` is authored under `parametrization`, not as a key
                    path=f"featurizers.{names[0]}."
                    + ("parametrization.forward" if field == "forward" else field),
                )


def _check_position_gates(doc: Document) -> None:
    """Rule 4 for §2.5 ``axis: position``: every read or write that goes
    through a position gate addresses a **fixed span** — θ is one entry per
    addressed position, so the window's length must be known and the same
    on every row — and every use of one gate, at every site it is named
    from, addresses the same length (one name, one parameter set, one
    window)."""
    positional = {
        name
        for name, spec in doc.featurizers.items()
        if spec.kind == "gate" and spec.axis == "position"
    }
    if not positional:
        return
    lengths: dict[str, set[int]] = {name: set() for name in positional}
    for section, entries in (("reads", doc.reads), ("writes", doc.writes)):
        for ename, entry in entries.items():
            used = [
                member
                for member in _featurizer_chain(entry.featurizer)
                if member in positional
            ]
            if not used:
                continue
            pos = entry.pos
            pos_spec = doc.positions.get(pos) if isinstance(pos, str) else pos
            length = span_length(pos_spec)
            if length is None:
                raise ValidationError(
                    4,
                    f"{section}.{ename} goes through position gate {used[0]!r} but "
                    "its `pos` is not a fixed span [a, b) — a position gate's θ is "
                    "one entry per addressed position, so the window must be a span "
                    "of two or more positions whose length is the same on every "
                    "row: `all`, a variable/column and a `segment` / `before` / "
                    "`after` / `between` span are as wide as the row, a static "
                    "`indices` set or `union` / `intersection` of static members "
                    "is a non-contiguous window (deferred, as `all` is), a "
                    "`generated` window is clipped to the row's decode, a "
                    "`scope`d span is sliced out of the anchor's run, a "
                    "`relative_to` span is not placed by its anchor at all today "
                    "(the resolver offsets an `index` only; §2.3's span offset is "
                    "unimplemented, and an unimplemented placement may not size a "
                    "θ), and an `index` or a span of one position is one scalar at "
                    "one position — "
                    "`group: site` on that write, not a mask over positions "
                    "(§2.5 axis)",
                    path=f"{section}.{ename}.pos",
                )
            for member in used:
                lengths[member].add(length)
    for name, seen in lengths.items():
        if len(seen) > 1:
            raise ValidationError(
                4,
                f"position gate {name!r} is used over windows of lengths "
                f"{sorted(seen)} — one gate, one window (§2.5 axis)",
                path=f"featurizers.{name}.axis",
            )


def _check_train_references(doc: Document) -> None:
    assert doc.train is not None
    train = doc.train
    slot_names = _param_slot_names(doc)
    trained = {pname.split(".", 1)[0] for pname in train.params}
    for i, term in enumerate(train.objective):
        p = term.path(i)
        if term.metric is not None:
            if term.metric not in doc.metrics:
                raise ValidationError(
                    4, f"objective metric {term.metric!r} is not declared", path=p
                )
            continue
        assert term.regularizer is not None
        _kind, names = term.regularizer
        for reg_target in names:
            # a single name may be a dotted slot; a list names whole featurizers
            if reg_target not in doc.featurizers and (
                len(names) > 1 or reg_target not in slot_names
            ):
                raise ValidationError(
                    4,
                    f"regularizer target {reg_target!r} is not a featurizer"
                    + ("" if len(names) > 1 else " or a dotted param slot"),
                    path=p,
                )
            if reg_target.split(".", 1)[0] not in trained:
                raise ValidationError(
                    4,
                    f"regularizer target {reg_target!r} is not in train.params — "
                    "a penalty on a frozen featurizer moves nothing",
                    path=p,
                )
            if _kind in ("l0", "l1"):
                _check_mask_penalty_pairs_with_the_map(doc, _kind, reg_target, p)
        if isinstance(term.costs, Mapping):
            # §2.11 `costs`: a multiplier keyed by a target the term does not
            # penalize is a reference to nothing — the key is a target name
            for costed in term.costs:
                if costed not in names:
                    raise ValidationError(
                        4,
                        f"costs key {costed!r} is not one of this term's targets "
                        f"{list(names)} — a cost multiplies a penalized quantity, "
                        "so it names one of the term's own targets",
                        path=f"{p}.costs",
                    )
        if term.constraint is not None:
            # §2.11 `constraint`: a target density is a gate's mask mean, so
            # every target is a gate — `l1` over a rotation's |p| has no density
            # (every target is a declared featurizer or a slot of one by now —
            # an undeclared name was refused above — so the lookup holds)
            for reg_target in names:
                fspec = doc.featurizers[reg_target.split(".", 1)[0]]
                if fspec.kind != "gate":
                    raise ValidationError(
                        4,
                        f"constraint target {reg_target!r} is not a gate — a target "
                        f"density is a gate's mask mean (§2.11); a {fspec.kind!r} "
                        "has no density to hold",
                        path=f"{p}.constraint",
                    )
    for i, pname in enumerate(train.params):
        if (
            pname not in doc.featurizers
            and pname not in slot_names
            and pname not in doc.params
        ):
            raise ValidationError(
                4,
                f"train.params entry {pname!r} is neither a featurizer, a dotted "
                "slot, nor a params entry",
                path=f"train.params[{i}]",
            )
    if train.eval is not None:
        for mname in train.eval["metrics"]:
            if mname not in doc.metrics:
                raise ValidationError(
                    4,
                    f"eval metric {mname!r} is not declared",
                    path="train.eval.metrics",
                )
    if train.early_stop is not None:
        es_metric = train.early_stop["metric"]
        if es_metric not in doc.metrics:
            raise ValidationError(
                4,
                f"early_stop metric {es_metric!r} is not declared",
                path="train.early_stop.metric",
            )
    if train.control is not None:
        _check_control_references(doc)
    if train.anneal is not None:
        for dotted, schedule in train.anneal.items():
            if train.control is not None and dotted in train.control:
                raise ValidationError(
                    4,
                    f"{dotted!r} is both annealed and controlled — an open-loop "
                    "schedule and a closed-loop one cannot both set one "
                    "hyperparameter (§2.11)",
                    path="train.anneal",
                )
            _check_anneal_target(doc, dotted, schedule, "train.anneal")
    if train.phases is not None:
        if str(train.optimizer.get("schedule", "constant")) != "constant":
            raise ValidationError(
                4,
                "train.optimizer.schedule is not 'constant' while train.phases is "
                "authored — a phase rewrites each group's lr at its boundary and a "
                "schedule rewrites it every update; the two cannot both own it (§2.11)",
                path="train.optimizer.schedule",
            )
        _check_phase_references(doc)


def _check_anneal_target(
    doc: Document, dotted: str, schedule: AnnealSchedule, path: str
) -> None:
    """§2.11 ``anneal``: one target resolves — a named objective term's weight
    (the open-loop twin of a ``control`` on the same address: a *named* term,
    a numeric authored weight, since the schedule's start replaces it before
    the first step) or a ``<featurizer>.<slot>.<hyperparameter>`` path whose
    slot the kind has; a gate's temperature schedule stays positive, is not
    authored beside its anneal, and does not exist on a ``clamp`` gate."""
    if dotted.startswith(OBJECTIVE_WEIGHT_PREFIX):
        _check_objective_weight_target(doc, dotted, "anneal", path)
        return
    parts = dotted.split(".")
    if len(parts) < 3:
        raise ValidationError(
            4,
            f"anneal target {dotted!r} is neither a named objective term's "
            "weight ('train.objective.<name>.weight') nor a dotted "
            "<featurizer>.<slot>.<hyperparameter> path",
            path=path,
        )
    fname, slot = parts[0], parts[1]
    spec = doc.featurizers.get(fname)
    if spec is None:
        raise ValidationError(
            4, f"anneal target {dotted!r}: {fname!r} is not a featurizer", path=path
        )
    kind = spec.kind if isinstance(spec.kind, str) else "identity"
    if slot not in FEATURIZER_SLOTS.get(kind, ()):
        raise ValidationError(
            4,
            f"anneal target {dotted!r}: {kind!r} featurizers have no slot {slot!r}",
            path=path,
        )
    if kind == "gate" and parts[-1] == "temperature":
        # the schedule IS the temperature for an annealed fit (an authored
        # one is refused below), so it gets the authored field's check:
        # σ(θ/T) and the concrete sample divide by it, and a negative β
        # inverts the sample's monotonicity in θ
        if schedule.start <= 0 or schedule.end <= 0:
            start, end = schedule.start, schedule.end
            raise ValidationError(
                4,
                f"anneal target {dotted!r}: a gate's temperature is positive, "
                f"got a schedule from {start!r} to {end!r} — the mask divides "
                "by it, and a non-positive value is a division by zero or "
                "an inverted mask, not a sharper one (§2.5)",
                path=path,
            )
        if spec.temperature is not None:
            # `_set_anneal` writes the schedule's `start` onto the stage
            # before the first forward, so an authored β would never take
            # effect — two fields naming one number, one silently winning
            raise ValidationError(
                4,
                f"anneal target {dotted!r}: {fname!r} authors temperature "
                f"{spec.temperature!r} and anneals it — the schedule's start "
                "replaces the authored value before the first step, so write "
                "one or the other (§2.5)",
                path=path,
            )
        if isinstance(spec.parametrization, (str, type(None))):
            # a swept map is held arm by arm at compile, per expanded point
            gate_map = GATE_MAPS[spec.parametrization or GATE_DEFAULT_MAP]
            if not gate_map.anneals_temperature:
                raise ValidationError(
                    4,
                    f"anneal target {dotted!r}: {gate_map.no_temperature_because} "
                    "(§2.5 parametrization)",
                    path=path,
                )


def _check_phase_references(doc: Document) -> None:
    """§2.11 ``phases``: every name a phase carries resolves against the fit
    it narrows. A phase's ``params`` are entries of ``train.params`` (the
    parser holds that); a phase's ``anneal`` target is a featurizer the
    *phase* trains — a schedule on a frozen slot would move a number nothing
    reads — or a named term's weight, and names no path the top-level
    ``anneal`` or a ``control`` already moves (two schedules on one value);
    ``freeze_masks`` names gates, each outside the phase's ``params`` — a
    gate whose θ trains while its mask is pinned is a contradiction — and
    not loaded from a file, whose mask never moves anyway."""
    assert doc.train is not None
    train = doc.train
    assert train.phases is not None
    scheduled = set(train.anneal or ()) | set(train.control or ())
    for i, phase in enumerate(train.phases):
        p = f"train.phases[{i}]"
        owners = {pname.split(".", 1)[0] for pname in phase.params}
        for dotted, schedule in (phase.anneal or {}).items():
            if dotted in scheduled:
                raise ValidationError(
                    4,
                    f"{p}.anneal: {dotted!r} is already annealed or controlled at "
                    "the top level — one value, one schedule (§2.11)",
                    path=f"{p}.anneal",
                )
            _check_anneal_target(doc, dotted, schedule, f"{p}.anneal")
            if (
                not dotted.startswith(OBJECTIVE_WEIGHT_PREFIX)
                and dotted.split(".", 1)[0] not in owners
            ):
                raise ValidationError(
                    4,
                    f"{p}.anneal target {dotted!r}: {dotted.split('.', 1)[0]!r} does "
                    f"not train in this phase (params {list(phase.params)}) — a "
                    "schedule on a frozen featurizer moves a number nothing reads",
                    path=f"{p}.anneal",
                )
        for name in phase.freeze_masks:
            spec = doc.featurizers.get(name)
            kind = (
                spec.kind if spec is not None and isinstance(spec.kind, str) else None
            )
            if kind != "gate":
                raise ValidationError(
                    4,
                    f"{p}.freeze_masks names {name!r}, which is not a gate — only a "
                    "gate has a hard mask to pin (§2.5)",
                    path=f"{p}.freeze_masks",
                )
            if name in owners:
                raise ValidationError(
                    4,
                    f"{p}.freeze_masks pins {name!r}'s mask while the phase trains "
                    "its θ — pin it or train it, not both",
                    path=f"{p}.freeze_masks",
                )


def _check_mask_penalty_pairs_with_the_map(
    doc: Document, kind: str, reg_target: str, path: str
) -> None:
    """§2.11: a gate's mask penalty is keyed on its map, not on its kind —
    :data:`~causalab.protocol.schema.GATE_MAPS` says which one each map admits.
    ``l0`` is the expected kept fraction of a *sampled* mask — legal only on a
    map that samples, since on a deterministic map the relaxed mask is itself
    the kept probability and its mean is ``l1`` (two spellings of one
    computation would digest apart); ``l1`` is the mean of the deterministic
    soft mask — refused on a sampled map, whose training forward never uses
    it; a ranked map fixes the mask's sum by construction and admits neither.
    A swept ``parametrization`` is refused when *any* arm contradicts the
    penalty, naming the arm (the rule the parser applies to the hard-concrete
    constants): compile would refuse the whole run on that point anyway, and
    saying so here is the better message. ``l1`` on a non-gate is ``|p|`` over
    its params and is not this rule's."""
    fspec = doc.featurizers.get(reg_target.split(".", 1)[0])
    if fspec is None or fspec.kind != "gate":
        if kind == "l0":
            what = f"a {fspec.kind!r} featurizer" if fspec else "a dotted param slot"
            raise ValidationError(
                4,
                f"regularizer target {reg_target!r}: 'l0' is the expected kept "
                f"fraction of a gate's mask (§2.11) — {what} has no mask to count",
                path=path,
            )
        return
    authored = fspec.parametrization
    maps: list[Any] = (
        list(authored.values)
        if isinstance(authored, Sweep)
        else [GATE_DEFAULT_MAP if authored is None else authored]
    )
    sampled_maps = ", ".join(
        repr(name) for name, gate_map in GATE_MAPS.items() if gate_map.penalty == "l0"
    )
    for parametrization in maps:
        if not isinstance(parametrization, str):
            continue  # an artifact-valued map is resolved at run time
        admitted = GATE_MAPS[parametrization].penalty
        arm = " (one arm of the sweep)" if isinstance(authored, Sweep) else ""
        if admitted is None:
            raise ValidationError(
                4,
                f"regularizer target {reg_target!r}: a {parametrization} gate{arm} "
                "has no sparsity penalty — its mask sums to the step's budget by "
                f"construction (§2.5 k_schedule), so {kind!r} would penalize a "
                "quantity the schedule already fixes — spell none",
                path=path,
            )
        if kind == admitted:
            continue
        if kind == "l0":
            raise ValidationError(
                4,
                f"regularizer target {reg_target!r}: 'l0' is the expected kept "
                f"fraction of a sampled mask (§2.11), and a {parametrization!r} "
                f"gate{arm} has a deterministic relaxed mask — its mean is the 'l1' "
                "term, so spell it 'l1' (or fit the gate under parametrization "
                f"{sampled_maps})",
                path=path,
            )
        raise ValidationError(
            4,
            f"regularizer target {reg_target!r}: 'l1' is the mean of the "
            f"deterministic soft mask, which a {parametrization} gate{arm} never "
            "uses in its training forward — its penalty is 'l0', the expected "
            "kept fraction of the sampled mask (§2.11)",
            path=path,
        )


def _check_objective_weight_target(
    doc: Document, target: str, schedule: str, path: str
) -> None:
    """§2.11: an ``anneal`` or ``control`` target of the form
    ``train.objective.<name>.weight`` names a **named** objective term (a
    positional term has no name to address) whose authored weight is a
    number — it is the schedule's start (``anneal``) or the controller's
    initial value (``control``), so a swept or wrapped weight would be two
    values for one field."""
    assert doc.train is not None
    named_terms = {
        term.name: term for term in doc.train.objective if term.name is not None
    }
    rest = target[len(OBJECTIVE_WEIGHT_PREFIX) :]
    name, _, field = rest.rpartition(".")
    if field != "weight" or name not in named_terms:
        raise ValidationError(
            4,
            f"{schedule} target {target!r} is not a named objective term's "
            "weight — spell the objective in its named form and address "
            f"'train.objective.<name>.weight' (named terms: "
            f"{sorted(named_terms) or 'none'})",
            path=path,
        )
    if named_terms[name].constraint is not None:
        raise ValidationError(
            4,
            f"{schedule} target {target!r}: a constraint term has no weight — its "
            "multipliers are the dual pair (λ₁, λ₂) the fit ascends (§2.11), not a "
            "schedule's target",
            path=path,
        )
    weight = named_terms[name].weight
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        what = (
            "the schedule's start"
            if schedule == "anneal"
            else "the controller's initial value"
        )
        raise ValidationError(
            4,
            f"{schedule} target {target!r}: the term's authored weight is "
            f"{what}, so it must be a number here",
            path=path,
        )


def _check_control_references(doc: Document) -> None:
    """§2.11 ``control``: every target names something the loop can set — a
    **named** objective term's ``weight`` (``train.objective.<name>.weight``,
    the same address a sweep uses; a positional term has no name to address)
    or an anneal-style ``<featurizer>.<slot>.<hyperparameter>`` on a trained
    featurizer — and every signal names a trained gate, the one kind with a
    ``hard_mask_size``."""
    assert doc.train is not None
    train = doc.train
    assert train.control is not None
    trained = {pname.split(".", 1)[0] for pname in train.params}
    for target, spec in train.control.items():
        p = f"train.control.{target}"
        if target.startswith(OBJECTIVE_WEIGHT_PREFIX):
            _check_objective_weight_target(doc, target, "control", p)
        else:
            parts = target.split(".")
            fname = parts[0]
            spec_f = doc.featurizers.get(fname)
            if len(parts) < 3 or spec_f is None or fname not in trained:
                raise ValidationError(
                    4,
                    f"control target {target!r} is neither a named objective "
                    "term's weight ('train.objective.<name>.weight') nor a "
                    "<featurizer>.<slot>.<hyperparameter> path on a trained "
                    "featurizer",
                    path=p,
                )
            kind = spec_f.kind if isinstance(spec_f.kind, str) else "identity"
            if parts[1] not in FEATURIZER_SLOTS.get(kind, ()):
                raise ValidationError(
                    4,
                    f"control target {target!r}: {kind!r} featurizers have no slot {parts[1]!r}",
                    path=p,
                )
        ((signal, signal_target),) = spec["signal"].items()
        names = (
            [signal_target] if isinstance(signal_target, str) else list(signal_target)
        )
        for name in names:  # one gate, or several whose kept counts are summed
            signal_spec = doc.featurizers.get(str(name))
            signal_kind = (
                signal_spec.kind
                if signal_spec is not None and isinstance(signal_spec.kind, str)
                else None
            )
            if signal_kind != "gate" or str(name) not in trained:
                raise ValidationError(
                    4,
                    f"control signal {signal!r} of {target!r} names {name!r}, "
                    "which is not a trained gate — a kept-unit count is a gate's, "
                    "and a frozen gate's count never moves",
                    path=f"{p}.signal",
                )
            assert signal_spec is not None
            signal_maps = (
                list(signal_spec.parametrization.values)
                if isinstance(signal_spec.parametrization, Sweep)
                else [signal_spec.parametrization or GATE_DEFAULT_MAP]
            )
            ranked = [
                m for m in signal_maps if isinstance(m, str) and GATE_MAPS[m].ranked
            ]
            if ranked:
                raise ValidationError(
                    4,
                    f"control signal {signal!r} of {target!r} names {name!r}, a "
                    f"{ranked[0]} gate — its kept count is the schedule's cut "
                    "(k_schedule.eval), fixed by the document, so nothing the "
                    "controller moves can move the signal (§2.5)",
                    path=f"{p}.signal",
                )


# --------------------------------------------------------------------------- #
# rule 6 — writes are inert; operands are reads, params, or literal scalars
# --------------------------------------------------------------------------- #


def _operand_names(do: Do) -> tuple[str, ...]:
    """The names a mechanism's operands reference (literal scalars excluded)."""
    mech = do.mechanism
    if mech == "swap":
        return (do.payload,) if isinstance(do.payload, str) else ()
    if mech in ("add_scaled", "lerp"):
        out: list[str] = []
        for field in ("op", "alpha"):
            value = do.payload.get(field)
            if isinstance(value, str):
                out.append(value)
        return tuple(out)
    if mech == "affine":
        return tuple(
            v for v in (do.payload.get("A"), do.payload.get("b")) if isinstance(v, str)
        )
    return ()


def _check_writes_inert(doc: Document, names: dict[str, str]) -> None:
    slot_names = _param_slot_names(doc)
    for ename, write in doc.writes.items():
        if write.do.mechanism == "affine":
            for field in ("A", "b"):
                target = write.do.payload.get(field)
                if isinstance(target, str) and (
                    target in doc.reads
                    or (
                        target in names
                        and target not in doc.params
                        and target not in slot_names
                    )
                ):
                    raise ValidationError(
                        6,
                        f"affine {field!r} must name a param (§2.8 types both "
                        f"fields as params); {target!r} is a "
                        f"{names.get(target, 'read')}",
                        path=f"writes.{ename}.do",
                    )
        for operand in _operand_names(write.do):
            if operand in doc.reads or operand in doc.params or operand in slot_names:
                continue
            if operand in names:
                raise ValidationError(
                    6,
                    f"write operand {operand!r} names a {names[operand]} entry — "
                    "operands are reads, params, or literal scalars (§2.8)",
                    path=f"writes.{ename}.do",
                )
            raise ValidationError(
                4,
                f"write operand {operand!r} is not declared",
                path=f"writes.{ename}.do",
            )


# --------------------------------------------------------------------------- #
# rule 7 — membership and the acyclic model graph
# --------------------------------------------------------------------------- #


def im_writes(writes: Any) -> tuple[str, ...]:
    """The write names an intervened_model lists, or ``()`` while the list is
    still a sweep wrapper. Public because the ``--data`` pass needs the same
    write → intervened_model map rule 7 builds (loader's rule 25)."""
    return tuple(writes) if isinstance(writes, tuple) else ()


def _check_membership_and_acyclic(doc: Document) -> None:
    in_force: dict[str, set[str]] = {ename: set() for ename in doc.writes}
    for mname, im in doc.intervened_models.items():
        seen: set[str] = set()
        for ename in im_writes(im.writes):
            if ename in seen:
                raise ValidationError(
                    7,
                    f"write {ename!r} listed twice",
                    path=f"intervened_models.{mname}.writes",
                )
            seen.add(ename)
            in_force[ename].add(mname)
    for ename, hosts in in_force.items():
        if not hosts:
            raise ValidationError(
                7,
                f"write {ename!r} appears in no intervened_model — every declared "
                "write must be in force somewhere (§2.9)",
                path=f"writes.{ename}",
            )

    # model graph: an edge M -> M' when a read in M' is an operand of a write
    # in force in M (M' must run first). 'original' has no out-edges.
    graph: dict[str, set[str]] = {m: set() for m in doc.intervened_models}
    for mname, im in doc.intervened_models.items():
        for ename in im_writes(im.writes):
            for operand in _operand_names(doc.writes[ename].do):
                read = doc.reads.get(operand)
                if read is not None and read.model != "original":
                    graph[mname].add(str(read.model))

    state: dict[str, int] = {}  # 0 in-progress, 1 done

    def visit(node: str, trail: tuple[str, ...]) -> None:
        mark = state.get(node)
        if mark == 1:
            return
        if mark == 0:
            cycle = " -> ".join((*trail[trail.index(node) :], node))
            raise ValidationError(
                7,
                f"the intervened-model graph has a cycle: {cycle} — operand flow "
                "must be acyclic (§2.9)",
            )
        state[node] = 0
        for nxt in graph[node]:
            visit(nxt, trail + (node,))
        state[node] = 1

    for mname in graph:
        visit(mname, ())


# --------------------------------------------------------------------------- #
# rules 8 + 9 — absolute-write and dims collisions per address
# --------------------------------------------------------------------------- #


def _pos_key(doc: Document, pos: Any) -> PositionSpec:
    if isinstance(pos, str):
        entry = doc.positions[pos]
        if isinstance(entry, PositionSpec):
            return entry
        raise ValidationError(4, f"position {pos!r} did not resolve to a concrete spec")
    assert isinstance(pos, PositionSpec)
    return pos


def _provably_disjoint(a: PositionSpec, b: PositionSpec) -> bool:
    """True when two position specs cannot address a common token on any row
    (see the module docstring for the conservative reading)."""
    if a.all is not None or b.all is not None:
        return False  # every content token — nothing is disjoint from it
    if isinstance(a, SpanSpec) or isinstance(b, SpanSpec):
        # A span is one address (atomic) or a set of them; either way the
        # only thing provable at load is two *static* index sets in one sign
        # regime that share no member. Anything text-located assumes overlap.
        a_set, b_set = static_indices(a), static_indices(b)
        if a_set is None or b_set is None:
            return False
        regimes = {n < 0 for n in (*a_set, *b_set)}
        if len(regimes) != 1:
            return False  # mixed end-relative and forward — unknowable at load
        return not set(a_set) & set(b_set)
    if (
        a.scope != b.scope
        or a.relative_to != b.relative_to
        or a.anchor_source != b.anchor_source
    ):
        return False  # different frames — incomparable, assume overlap
    if a.column is not None or b.column is not None:
        # A column's value is data, not a template slot: two *different*
        # columns can hold the same string on the same row, so unlike two
        # variables nothing is provable at load. (An index/span *scoped* to
        # the same column anchor stays comparable — that is the branch below,
        # reached because the anchors compared equal.)
        return False
    if a.variable is not None and b.variable is not None:
        return a.variable != b.variable
    if a.variable is not None or b.variable is not None:
        return False
    a_index = a.index if isinstance(a.index, int) else None
    b_index = b.index if isinstance(b.index, int) else None
    a_span = tuple(a.span) if isinstance(a.span, tuple) else None
    b_span = tuple(b.span) if isinstance(b.span, tuple) else None
    if a_index is not None and b_index is not None:
        return a_index != b_index and (a_index < 0) == (b_index < 0)
    if a_span is not None and b_span is not None:
        # comparable only within one sign regime (end-relative bounds are a
        # different frame; mixed-sign pairs are unknowable at load)
        if _span_regime(a_span) is None or _span_regime(a_span) != _span_regime(b_span):
            return False
        (a0, a1), (b0, b1) = a_span, b_span
        return a1 <= b0 or b1 <= a0
    if a_index is not None and b_span is not None:
        return _index_outside_span(a_index, b_span)
    if b_index is not None and a_span is not None:
        return _index_outside_span(b_index, a_span)
    return False


def _span_regime(span: tuple[int, int]) -> str | None:
    """ "forward" for fully non-negative spans, "end" for fully end-relative
    ones, None for mixed (incomparable at load)."""
    lo, hi = span
    if lo >= 0 and hi >= 0:
        return "forward"
    if lo < 0 and hi <= 0:
        return "end"
    return None


def _index_outside_span(index: int, span: tuple[int, int]) -> bool:
    regime = _span_regime(span)
    if regime == "forward":
        if index < 0:
            return False  # end-relative index vs a forward window — unknowable
        lo, hi = span
        return not lo <= index < hi
    if regime == "end":
        if index >= 0:
            return False  # forward index vs an end-relative window — unknowable
        lo, hi = span
        return not lo <= index < (hi if hi != 0 else 0)
    return False  # mixed-sign span — unknowable


def _is_absolute(write: WriteSpec) -> bool:
    return write.do.mechanism not in ADDITIVE_MECHANISMS


def _dims_intersect(a: WriteSpec, b: WriteSpec) -> bool:
    a_dims = a.dims if isinstance(a.dims, tuple) else None
    b_dims = b.dims if isinstance(b.dims, tuple) else None
    if a_dims is None or b_dims is None:
        return True  # full width intersects everything
    return bool(set(a_dims) & set(b_dims))


def _check_write_collisions(doc: Document) -> None:
    for mname, im in doc.intervened_models.items():
        writes = [(ename, doc.writes[ename]) for ename in im_writes(im.writes)]
        for i in range(len(writes)):
            for j in range(i + 1, len(writes)):
                (name_a, a), (name_b, b) = writes[i], writes[j]
                if a.site != b.site:
                    continue
                if _provably_disjoint(_pos_key(doc, a.pos), _pos_key(doc, b.pos)):
                    continue
                both_dims = isinstance(a.dims, tuple) and isinstance(b.dims, tuple)
                if both_dims and _dims_intersect(a, b):
                    # §5.9, read literally: explicit dims selections at one
                    # address are pairwise disjoint — additive included
                    # (surfaced as a spec question; a steer inside a swapped
                    # subspace needs featurizer composition instead)
                    raise ValidationError(
                        9,
                        f"writes {name_a!r} and {name_b!r} select intersecting dims "
                        f"at site {a.site!r} in model {mname!r} — co-occurring "
                        "dims selections must be disjoint (§5.9)",
                    )
                if not (_is_absolute(a) and _is_absolute(b)):
                    continue  # one absolute + additive deltas is the §2.8 class order
                if not both_dims:
                    raise ValidationError(
                        8,
                        f"writes {name_a!r} and {name_b!r} are two absolute writes at "
                        f"site {a.site!r} with overlapping positions in model "
                        f"{mname!r} — at most one absolute write per address (§2.8)",
                    )


# --------------------------------------------------------------------------- #
# rule 10 — the save manifest
# --------------------------------------------------------------------------- #


def _trained_featurizers(doc: Document) -> set[str]:
    if doc.train is None:
        return set()
    trained: set[str] = set()
    for pname in doc.train.params:
        root = pname.split(".", 1)[0]
        if root in doc.featurizers:
            trained.add(root)
    return trained


def _check_save(doc: Document) -> None:
    trained = _trained_featurizers(doc)
    seen_values: set[str] = set()
    seen_paths: set[str] = set()
    for i, entry in enumerate(doc.save):
        p = f"save[{i}]"
        if entry.file_path in seen_paths:
            raise ValidationError(
                10,
                f"two save entries write {entry.file_path!r} — one file per "
                "entry, or the later silently clobbers the earlier",
                path=p,
            )
        seen_paths.add(entry.file_path)
        if entry.value in seen_values:
            raise ValidationError(
                10,
                f"{entry.value!r} is saved twice — one manifest entry per value",
                path=p,
            )
        seen_values.add(entry.value)
        if entry.kind is not None:
            # a non-value kind (§2.12): its binding is the whole run, so there
            # is nothing to cross-check — only the shape of what it writes
            if entry.kind == "trajectory":
                if not entry.file_path.endswith(".safetensors"):
                    raise ValidationError(
                        10,
                        "a 'trajectory' entry is a tensor bundle — its file_path "
                        "must end in '.safetensors' (§2.12)",
                        path=p,
                    )
                if doc.train is None:
                    raise ValidationError(
                        10,
                        "a 'trajectory' entry photographs a fit along its way — "
                        "it needs a train section (§2.12)",
                        path=p,
                    )
                # at most one per document: a non-value entry's `value` is its
                # kind, so the "saved twice" check above already holds it
            elif not entry.file_path.endswith(".json"):
                raise ValidationError(
                    10,
                    f"a {entry.kind!r} entry is a table — its file_path must end "
                    "in '.json' (§2.12)",
                    path=p,
                )
            if entry.kind == "rank" and not any(
                spec.kind == "gate" for spec in doc.featurizers.values()
            ):
                raise ValidationError(
                    10,
                    "a 'rank' entry orders a gate's units by theta — this "
                    "document declares no gate featurizer (§2.12)",
                    path=p,
                )
            continue
        if entry.value in doc.featurizers:
            _check_save_featurizer(doc, entry, trained, p)
        elif entry.value in doc.reads or entry.value in doc.metrics:
            _check_save_binding(doc, entry, p)
        else:
            raise ValidationError(
                10,
                f"{entry.value!r} is not saveable — only reads, metrics, and "
                "trained featurizers leave a run (§2.12)",
                path=p,
            )
    # these two name the *declaration* that is missing from `save`, not a save
    # entry — there is no entry to point at, which is the problem
    for qname in doc.metrics:
        if qname not in seen_values:
            raise ValidationError(
                10,
                f"metric {qname!r} is not saved — every metric must be (§2.12)",
                path=f"metrics.{qname}",
            )
    for fname in trained:
        if fname not in seen_values:
            raise ValidationError(
                10,
                f"trained featurizer {fname!r} is not saved — every fit must be (§2.12)",
                path=f"featurizers.{fname}",
            )


def _check_save_binding(doc: Document, entry: SaveEntry, path: str) -> None:
    if entry.site is not None or entry.model is None or entry.input is None:
        raise ValidationError(
            10, "a read/metric save entry binds with model + input", path=path
        )
    read = doc.reads.get(entry.value)
    if read is None:
        read = doc.reads[str(doc.metrics[entry.value].of)]
    if entry.model != read.model or entry.input != read.input:
        raise ValidationError(
            10,
            f"save entry for {entry.value!r} restates (model={entry.model!r}, "
            f"input={entry.input!r}) but the declaration chain resolves to "
            f"(model={read.model!r}, input={read.input!r}) — bindings are "
            "cross-checked, never trusted (§2.12)",
            path=path,
        )
    is_metric = entry.value in doc.metrics
    if entry.reduce is not None and is_metric:
        raise ValidationError(
            10,
            f"save entry for metric {entry.value!r} carries 'reduce' — a metric "
            "is already a reduction over its read; reduce applies to reads "
            "(§2.12)",
            path=path,
        )
    # JSON and safetensors are the only two formats (§2.12): a metric table is
    # structured and readable, a read is dense numerics
    expected_ext = ".json" if is_metric else ".safetensors"
    if not entry.file_path.endswith(expected_ext):
        raise ValidationError(
            10,
            f"{entry.value!r} is a {'metric' if is_metric else 'read'} — its "
            f"file_path must end in {expected_ext!r} (§2.12)",
            path=path,
        )


def _check_save_featurizer(
    doc: Document, entry: SaveEntry, trained: set[str], path: str
) -> None:
    if entry.site is None or entry.model is not None or entry.input is not None:
        raise ValidationError(
            10, "a featurizer save entry binds with 'site' alone", path=path
        )
    if entry.value not in trained:
        spec = doc.featurizers[entry.value]
        reason = (
            "a file_path-loaded featurizer is a pointless copy"
            if spec.file_path is not None
            else "an untrained featurizer has nothing to save"
        )
        raise ValidationError(
            10, f"{entry.value!r} is not trained — {reason} (§2.12)", path=path
        )
    used_sites = _featurizer_sites(doc, entry.value)
    if entry.site not in used_sites:
        raise ValidationError(
            10,
            f"featurizer {entry.value!r} is used at site(s) {sorted(used_sites)}, "
            f"not {entry.site!r} — the restated site is cross-checked (§2.12)",
            path=path,
        )
    if entry.reduce is not None:
        raise ValidationError(
            10,
            "a featurizer bundle carries fitted parameters, not gathered rows — "
            "'reduce' applies to reads (§2.12)",
            path=path,
        )
    if not entry.file_path.endswith(".safetensors"):
        raise ValidationError(
            10, "a featurizer bundle's file_path must end in '.safetensors'", path=path
        )


def _featurizer_sites(doc: Document, fname: str) -> set[str]:
    used: set[str] = set()
    for read in doc.reads.values():
        if _references_featurizer(read.featurizer, fname):
            used.add(str(read.site))
    for write in doc.writes.values():
        if _references_featurizer(write.featurizer, fname):
            used.add(str(write.site))
    return used


def _references_featurizer(ref: Any, fname: str) -> bool:
    if ref is None:
        return False
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    return fname in chain


# --------------------------------------------------------------------------- #
# rule 11 — sinks: nothing declared is dead
# --------------------------------------------------------------------------- #


def _check_sinks(doc: Document) -> None:
    saved = {entry.value for entry in doc.save}
    metric_inputs: set[str] = set()
    for metric in doc.metrics.values():
        metric_inputs.add(str(metric.of))
        if metric.kind in READ_TARGET_METRIC_KINDS:
            target = metric.fields.get("target")
            if isinstance(target, str):
                metric_inputs.add(target)
    operands: set[str] = set()
    for write in doc.writes.values():
        operands.update(_operand_names(write.do))
    for rname in doc.reads:
        if rname not in saved and rname not in metric_inputs and rname not in operands:
            raise ValidationError(
                11,
                f"read {rname!r} is dead: neither saved, nor a metric input, nor "
                "a write operand (§5.11)",
                path=f"reads.{rname}",
            )
    _check_dead_declarations(doc, operands)


def _check_dead_declarations(doc: Document, operands: set[str]) -> None:
    """§0's uniform rule, reported under the sink rule's number: every
    declared site/position/featurizer/param must be referenced."""
    used_sites = {str(r.site) for r in doc.reads.values()} | {
        str(e.site) for e in doc.writes.values()
    }
    for sname in doc.sites:
        if sname not in used_sites:
            raise ValidationError(
                11, f"site {sname!r} is declared but never used", path=f"sites.{sname}"
            )
    used_pos = {r.pos for r in doc.reads.values() if isinstance(r.pos, str)} | {
        e.pos for e in doc.writes.values() if isinstance(e.pos, str)
    }
    for pname in doc.positions:
        if pname not in used_pos:
            raise ValidationError(
                11,
                f"position {pname!r} is declared but never used",
                path=f"positions.{pname}",
            )
    used_feat: set[str] = set()
    for read in doc.reads.values():
        used_feat.update(_featurizer_chain(read.featurizer))
    for write in doc.writes.values():
        used_feat.update(_featurizer_chain(write.featurizer))
    for fname in doc.featurizers:
        if fname not in used_feat:
            raise ValidationError(
                11,
                f"featurizer {fname!r} is declared but never used",
                path=f"featurizers.{fname}",
            )
    train_params: set[str] = set(doc.train.params) if doc.train is not None else set()
    for pname in doc.params:
        if pname in operands or pname in train_params:
            continue
        if any(pname in _operand_names(e.do) for e in doc.writes.values()):
            continue
        raise ValidationError(
            11,
            f"params entry {pname!r} is declared but never used",
            path=f"params.{pname}",
        )
    read_models = {str(r.model) for r in doc.reads.values()}
    for mname in doc.intervened_models:
        if mname not in read_models:
            raise ValidationError(
                11,
                f"intervened_model {mname!r} is never read — its writes can reach "
                "no sink (§0)",
                path=f"intervened_models.{mname}",
            )


def _featurizer_chain(ref: Any) -> Iterable[str]:
    if ref is None:
        return ()
    return (ref,) if isinstance(ref, str) else tuple(ref)


# --------------------------------------------------------------------------- #
# rule 12 — trainability declarations are consistent
# --------------------------------------------------------------------------- #


def _check_trainability(doc: Document) -> None:
    trained = _trained_featurizers(doc)
    for fname in trained:
        spec = doc.featurizers[fname]
        if spec.file_path is not None:
            raise ValidationError(
                12,
                f"featurizer {fname!r} is loaded from file_path and appears in "
                "train.params — loading and fitting the same artifact is a "
                "contradiction (§2.5)",
                path=f"featurizers.{fname}",
            )
        kind = spec.kind if isinstance(spec.kind, str) else "identity"
        if kind not in TRAINABLE_KINDS:
            raise ValidationError(
                12,
                f"featurizer {fname!r} has kind {kind!r}, which has no trainable "
                "slots (§5.12)",
                path=f"featurizers.{fname}",
            )
    if doc.train is not None:
        for pname in doc.train.params:
            spec = doc.params.get(pname)
            if spec is not None and spec.file_path is not None:
                raise ValidationError(
                    12,
                    f"params entry {pname!r} is a loaded constant (file_path) and "
                    "appears in train.params — loading and fitting the same "
                    "tensor is a contradiction (§2.6)",
                    path=f"params.{pname}",
                )
        for pname, spec in doc.params.items():
            if spec.shape is not None and pname not in doc.train.params:
                raise ValidationError(
                    12,
                    f"params entry {pname!r} declares shape/init (trainable) but is "
                    "not in train.params (§2.6)",
                    path=f"params.{pname}",
                )
    else:
        for pname, spec in doc.params.items():
            if spec.shape is not None:
                raise ValidationError(
                    12,
                    f"params entry {pname!r} is trainable but the document has no "
                    "train section (§2.6)",
                    path=f"params.{pname}",
                )


# --------------------------------------------------------------------------- #
# rule 16 — generation is read-only and prefill-only
# --------------------------------------------------------------------------- #


def _generated_positions(doc: Document) -> dict[str, PositionSpec]:
    """Every declared position that selects the continuation frame, by the
    name (or ``"<section>.<entry>"`` path) it was authored under."""
    found: dict[str, PositionSpec] = {}
    for name, entry in doc.positions.items():
        if isinstance(entry, PositionSpec) and entry.generated is not None:
            found[name] = entry
    for section in ("reads", "writes"):
        for name, spec in getattr(doc, section).items():
            pos = spec.pos
            if isinstance(pos, PositionSpec) and pos.generated is not None:
                found[f"{section}.{name}"] = pos
    return found


def _check_generation(doc: Document) -> None:
    """§5.16 — a decode is something a read *addresses*, never something a
    write or a fit runs inside.

    Two refusals, one rule. **Writes stay prefill-only**: the continuation
    only exists because the prefill already ran, and a write there would be
    a decode-step intervention — out of scope for v1 by §7's scope clause,
    and the reason no other rule has to reason about two frames (rules 8/9
    compare writes only, so both sides are always prompt-frame). **Training
    cannot see a decode**: `train` differentiates through the graph a
    forward builds, and a greedy continuation is an argmax chain, not a
    differentiable path.
    """
    for ename, write in doc.writes.items():
        pos = write.pos
        spec = doc.positions.get(pos) if isinstance(pos, str) else pos
        if isinstance(spec, PositionSpec) and spec.generated is not None:
            where = f" (position {pos!r})" if isinstance(pos, str) else ""
            raise ValidationError(
                16,
                f"write {ename!r} addresses the continuation frame{where} — writes "
                "are prefill-only (§2.3): reads may address generated tokens, "
                "writes may not",
                path=f"writes.{ename}",
            )
    if doc.train is not None:
        generated = _generated_positions(doc)
        if generated:
            raise ValidationError(
                16,
                f"train and the continuation frame do not combine: {sorted(generated)} "
                "select generated tokens, and a greedy decode is an argmax chain "
                "with no gradient path (§2.11)",
                path="train",
            )


# --------------------------------------------------------------------------- #
# rule 13 — pytorch_fn is local-only
# --------------------------------------------------------------------------- #


def _check_pytorch_fn(doc: Document, engine_is_local: bool) -> None:
    if engine_is_local:
        return
    for ename, write in doc.writes.items():
        if write.do.mechanism == "pytorch_fn":
            raise ValidationError(
                13,
                f"write {ename!r} uses pytorch_fn, which only a local engine may "
                "run (§2.8) — the selected engine is not local",
                path=f"writes.{ename}.do",
            )


# --------------------------------------------------------------------------- #
# rule 17 — the model's numeric realization is coherent
# --------------------------------------------------------------------------- #


def _check_model_realization(doc: Document) -> None:
    """§2.1 — a ``quantization`` block only carries the knobs its own scheme
    has. The enums are the parser's job (rule 1); what needs the whole block
    in one place is the cross-field question: ``double_quant`` and
    ``compute_dtype`` are 4-bit vocabulary and ``int8_threshold`` is
    LLM.int8() vocabulary, so any of them under the wrong scheme is a document
    that reads as if it configured something it did not.

    📐 ``compute_dtype`` earns its place here by measurement: the engine
    reads it as ``bnb_4bit_compute_dtype``, and the int8 branch of
    ``_bitsandbytes_config`` builds ``BitsAndBytesConfig(load_in_8bit=True,
    llm_int8_threshold=…)`` — nowhere for it to go. Left admissible, two int8
    documents differing only in ``compute_dtype`` hashed differently while
    producing identical numbers."""
    quantization = doc.model.quantization
    if quantization is None:
        return
    scheme = quantization.scheme
    if not isinstance(scheme, str):
        return  # a swept scheme is checked per point
    wrong = {
        "double_quant": scheme not in ("nf4", "fp4"),
        "compute_dtype": scheme not in ("nf4", "fp4"),
        "int8_threshold": scheme != "int8",
    }
    for field, is_wrong in wrong.items():
        if getattr(quantization, field) is not None and is_wrong:
            raise ValidationError(
                17,
                f"model.quantization.{field} does not apply to scheme "
                f"{scheme!r} — it is "
                + (
                    "a 4-bit knob (nf4 / fp4)"
                    if field in ("double_quant", "compute_dtype")
                    else "an int8 knob"
                )
                + " (§2.1)",
                path=f"model.quantization.{field}",
            )


# --------------------------------------------------------------------------- #
# rule 21 — a write's operand is reachable from its address
# --------------------------------------------------------------------------- #


def _site_label(doc: Document, site_name: str) -> str:
    """``block_mid`` at layer 11 — or at layers 10..19 for a band — the way a
    refusal should name an address."""
    site = doc.sites[site_name]
    layers = site.layers
    if isinstance(layers, tuple) and len(layers) == 1:
        where = f" layer {layers[0]}"
    elif isinstance(layers, tuple) and layers:
        where = f" layers {band_label(layers)}"
    else:
        where = ""
    return f"site {site_name!r} ({site.component}{where})"


def _check_operand_reachability(doc: Document) -> None:
    """Refuse a write whose operand is read strictly deeper than it lands.

    The value is *computable* — §2.9 stages the operand's model first — but
    the network has no edge from the deeper address to the shallower one, so
    what the write measures is attributable to no path. Equal depth is the
    two-pass harvest/inject idiom and stays legal.

    A **band** (§2.4 ``layers``) is checked the way it executes
    (``plan.lower_bands``): an operand read on a band of the same length as
    the write's band feeds it member by member, so member *i* of the read
    must be at or above member *i* of the write; any other operand is
    broadcast to every member, so the read's **deepest** member must be at or
    above the write's **shallowest** — the band's max on the read side, its
    min on the write side.
    """
    for ename, write in doc.writes.items():
        target_site = str(write.site)
        targets = site_depths(doc, target_site)
        for operand in _operand_names(write.do):
            read = doc.reads.get(operand)
            if read is None:
                continue  # a param or a literal: no address, so no geometry
            source_site = str(read.site)
            sources = site_depths(doc, source_site)
            if len(sources) == len(targets) and len(targets) > 1:
                pairs = tuple(zip(sources, targets))  # member i feeds member i
            else:
                pairs = ((max(sources), min(targets)),)
            if all(source <= target for source, target in pairs):
                continue
            raise ValidationError(
                21,
                f"write {ename!r} takes its operand from read {operand!r} at "
                f"{_site_label(doc, source_site)}, which is strictly deeper in "
                f"the forward pass than the address it lands on, "
                f"{_site_label(doc, target_site)}. The operand's model runs "
                "first, so this is executable — but no edge of the network "
                "carries information from the deeper address to the shallower "
                "one, so the write is attributable to no path (§2.8). Read the "
                "operand at or above the write's address; if it is meant as an "
                "externally supplied constant rather than a routed activation, "
                "harvest it in its own run and load it as a `params` entry "
                "(§2.6).",
                path=f"writes.{ename}.do",
            )
