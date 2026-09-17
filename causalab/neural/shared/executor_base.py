"""The engine-neutral core of a point executor.

Everything here operates on the *contract* tensor shape ``(batch, position,
feature)`` and the document: position resolution, gathers, featurizer stacks,
operand lookup, and the class-ordered write math. What an engine adds is one
method — :meth:`ExecutorBase._run_group` — that produces contract tensors for
this group's taps and lands its writes (hooks in the reference engine, traces
in the nnterp engine). The public surface consumed by
:mod:`causalab.neural.shared.execution` lives here so the two engines cannot
drift apart on what a read means.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from typing import Any, Callable, Container, Hashable, Iterable, Mapping, Sequence

import torch

from causalab.neural.shared.encoding import (
    Continuation,
    EncodedBatch,
    constituent_candidate_runs,
    encode,
    resolve_position,
    select_field,
)
from causalab.neural.shared.featurizers import (
    FeaturizerStack,
    Stage,
    StageRecipe,
    build_recipe,
    build_stack,
    link_budget_pools,
)
from causalab.neural.shared.fires import FireTally
from causalab.neural.shared.head import ReadTap, resolve_read_taps
from causalab.neural.shared.gather import (
    dense_index,
    flat_index,
    gather_positions,
    splice_features,
)
from causalab.neural.shared.mechanisms import (
    apply_absolute,
    apply_delta,
    apply_renormalize,
    is_additive,
    operand_names,
)
from causalab.causal.pairs import (
    EDIT_GROUPS_COLUMN,
    EditGroup,
    EditGroupError,
    check_component_wise,
    parse_edit_groups,
    row_texts,
)
from causalab.causal.scoring import (
    ScoringCheck,
    ScoringError,
    ScoringMismatch,
    check_scoring,
    declared_modes,
)
from causalab.neural.shared.framing import encode_framed
from causalab.neural.shared.metrics import check_answer_forms
from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.protocol.ledger import LedgerRow, LocationLedger
from causalab.protocol.spans import SpanSpec, constituents
from causalab.protocol.alignment import (
    UnalignableError,
    alignment_of,
    check_declared,
    unalignable,
)
from causalab.protocol.bundles import entry_selection, selector_slot
from causalab.protocol.errors import ProtocolError, ValidationError
from causalab.protocol.plan import fit_constant_models, generated_budget, lower_bands
from causalab.protocol.registry import component_width, write_policy_refusal
from causalab.protocol.resolution import (
    Resolution,
    Unavailable,
    available,
    cell_key,
    unavailable,
)
from causalab.protocol.shapes import FeatureShape
from causalab.protocol.schema import (
    span_length,
    ALL_POSITIONS,
    Document,
    PositionSpec,
    ReadSpec,
    SiteSpec,
    WriteSpec,
)

__all__ = [
    "CaptureKey",
    "ExecutorBase",
    "ForwardCache",
    "Interning",
    "PrefixKey",
    "PrefixPlan",
    "RaggedValue",
    "RowWindow",
    "TapKey",
    "WriteServices",
    "align_by_expert",
    "apply_writes_to_contract",
    "document_seed",
    "gather_rows",
    "land_ragged",
    "read_features",
    "read_operand",
    "refuse_unstackable",
    "tap_key",
    "written_value",
]


def document_seed(doc: Document) -> int:
    """The one seed a document implies: ``train.seed``, or **0** when it
    declares no fit.

    Read in one place so the consumers cannot drift apart: the ``subspace``
    featurizer's initial rotation (:func:`build_stack`) and the batch-order
    RNG.

    The 0 for a document with no ``train`` block is deliberate rather than
    accidental: an apply/inference document has no seed to name, and pinning
    it means the same document builds the same (unfitted) featurizer whether
    or not a fit is running — which a global-RNG init could not promise."""
    train = doc.train
    if train is None:
        return 0
    return int(train.seed) if isinstance(train.seed, int) else 0


@dataclasses.dataclass(frozen=True)
class RaggedValue:
    """A read over per-row windows of unequal width: the flat
    ``(total_positions, d)`` gather plus per-row widths, re-nestable via
    ``torch.split(flat, widths)`` — the RaggedIndex contract of the old
    resolver, kept as the protocol's ragged-read surface."""

    flat: torch.Tensor
    widths: tuple[int, ...]

    def detach_cpu(self) -> "RaggedValue":
        return RaggedValue(flat=self.flat.detach().cpu(), widths=self.widths)


@dataclasses.dataclass(frozen=True)
class RowWindow:
    """The rows ``[start, stop)`` of a role's ``total`` rows that one forward
    covers — a microbatch.

    Everything row-indexed in the write math is addressed through this rather
    than through the tensor a hook happens to hold: resolved positions and
    tensor operands are sliced to the window, and a ``gaussian`` draw is made
    over all ``total`` rows and then sliced, so a row receives the same noise
    whether it ran in one forward or in the third of five (§8 asks the RNG to
    be bit-stable across layouts). :meth:`whole` is the single-forward case,
    where no slicing happens at all.

    ``members`` narrows the window to a **width bucket** (§5 rule 19,
    ``exact_length_buckets``): the rows of ``[start, stop)`` a ragged write
    lands together because they address the same number of positions, as
    indices into the role's rows. A bucket is not a forward — the forward is
    still the window's — so it never reaches the engine; it is what the write
    math slices operands, routing tables and the ``gaussian`` draw by
    (:attr:`index`), and how a routed write names its examples
    (:attr:`examples`). ``None`` is the whole window.
    """

    start: int
    stop: int
    total: int
    members: tuple[int, ...] | None = None

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def index(self) -> "slice | list[int]":
        """What selects this window's rows out of a role-wide tensor: the
        contiguous slice, or the bucket's row indices."""
        return self.slice if self.members is None else list(self.members)

    @property
    def examples(self) -> list[int]:
        """The role's row indices this window covers, in order."""
        return (
            list(range(self.start, self.stop))
            if self.members is None
            else list(self.members)
        )

    def bucket(self, local_rows: Sequence[int]) -> "RowWindow":
        """The bucket of this window's rows at ``local_rows`` (indices into the
        window, i.e. into the tensor the forward holds)."""
        return RowWindow(
            self.start,
            self.stop,
            self.total,
            members=tuple(self.start + i for i in local_rows),
        )

    @property
    def size(self) -> int:
        return self.stop - self.start if self.members is None else len(self.members)

    @property
    def whole(self) -> bool:
        return self.members is None and self.start == 0 and self.stop == self.total


#: What identifies a tap for capture-sink sharing — see :func:`tap_key`. The
#: last element is an engine's interior address: it must be hashable and
#: value-equal (both engines key their capture sinks and write groups on it).
TapKey = tuple[
    int, str, FeatureShape, int | None, str | None, int | None, Hashable | None
]


def tap_key(site: ResolvedSite, source: Any = None) -> TapKey:
    """Identity of a tap for capture-sink sharing.

    Two sites may share a module and side yet mean different tensors — a
    different tuple element, or the same tensor read through a different shape
    — so the shape and tuple index are part of the identity. Keying on the
    module alone would let one tap read another's tensor.

    ``expert`` is part of the identity for the write path's sake: two writes
    at the same interior slot naming *different* experts must land as two
    separately masked applications, and the address grouping keys on this.

    ``source`` is an engine-specific *interior* address (the nnterp engine's
    ``SourceAddress``, a frozen dataclass): two interior taps may share the
    module, side and even shape while meaning different ops inside its
    forward — the DeltaNet q and k reshapes — so the address itself joins the
    key. It defaults to ``None`` so a hook-engine tap's key is unchanged.
    """
    return (
        id(site.module),
        site.kind,
        site.shape,
        site.tuple_index,
        site.interface_slot,
        site.expert,
        source,
    )


#: What one entry of the :class:`ForwardCache` store is keyed by — a group
#: digest plus **which rows** of the role it was run on:
#:
#: * ``digest`` — the role's whole rows (a campaign point's own forward);
#: * ``(digest, (i0, i1, …))`` — the role's rows at those indices (a fit's
#:   minibatch);
#: * ``(digest, "<split ref>")`` — the role's field over ``train.eval``'s split.
#:
#: The digest is the same in all three: it is the identity of the *forward*,
#: and the rows it ran over are a second coordinate, not a different forward.
#: :attr:`ForwardCache.wanted` stays keyed by the bare digest because the tap
#: union applies to every row selection of one forward alike.
CaptureKey = str | tuple[str, tuple[int, ...] | str]

#: What one cached **prefix** is keyed by (§4 "Resume"): the un-intervened
#: identity of the forward (:attr:`~causalab.protocol.plan.ForwardGroup.base_digest`),
#: the rows coordinate of the :data:`CaptureKey` (``None`` for the whole role,
#: a fit's slice otherwise), the row window ``(start, stop)`` the forward ran
#: over, and the block whose incoming residual the entry holds. The window is
#: part of it because a window is padded to its own frame: rows ``[0, 2)`` of a
#: four-row role and the same two rows as a minibatch are different tensors.
#: Hooks may append the attention implementation to distinguish accelerated
#: prefixes from eager ones. The first four coordinates (including those used
#: to release all variants of a prefix) retain their meaning.
PrefixKey = (
    tuple[str, tuple[int, ...] | str | None, tuple[int, int], int]
    | tuple[str, tuple[int, ...] | str | None, tuple[int, int], int, str]
)


@dataclasses.dataclass(frozen=True)
class PrefixPlan:
    """Where one forward group's pass may start, and what it may leave behind
    (§4 "Resume") — the plan's arithmetic, per digest, in the form the engine
    reads at each window.

    ``resume_at`` is :attr:`~causalab.protocol.plan.ForwardGroup.resume_at` of
    the *interned* group (0 = never). ``write_depth`` is the group's: the
    residual entering block ``d`` is the un-intervened one iff
    ``d <= write_depth``, so a pass may store the prefix at ``d`` only up to
    there — an intervened pass at layer 1 must never hand a later point at
    layer 3 a post-write residual as its prefix. Both are in the plan's block
    coordinate (:data:`~causalab.protocol.plan.PAST_BLOCKS` for "past every
    block"); the engine clamps them to the model it loaded."""

    base_digest: str
    resume_at: int
    write_depth: int


@dataclasses.dataclass(frozen=True)
class ForwardCache:
    """The campaign-wide store that makes §3's interning real at run time.

    The planner already says which forward groups a swept document shares: a
    group's ``digest`` is the content identity of everything that determines
    its activations, and taps are deliberately **not** part of it, because
    reading layer 3 or layer 23 of the same un-intervened forward is the same
    forward. A per-point loop ignores that and re-runs the shared group once
    per point; this is where the plan's guarantee gets claimed.

    ``wanted`` is the union of tap sites the **whole campaign** asks of each
    digest, so the first point to reach a group captures every address a later
    point will want. ``captured`` holds those activations *raw* — before the
    positional gather and before any featurizer — which is why points tapping
    one address through different featurizers, dims or positions still share
    a single forward. ``routing`` carries the experts-interface sub-axis table
    beside them: an interface capture without the dispatch indices it joins on
    is not a replayable value, so the two travel together or not at all.

    The trade is compute for memory: a 32-layer harvest shared by 32 points
    holds 32 captures at once where the per-point loop held one. That is
    inherent in making one pass serve the union, and it is still a large net
    win — 32 separate passes cost ~16x one full pass even when each is elided
    at its own tap. A fit adds its own slices (§4 "Fits"): the constant groups
    of its minibatches and of its eval split are captured once each and served
    on every later step, epoch, eval pass and point, so the store also holds
    roughly one more copy of the training rows' tapped activations plus one of
    the eval rows', on the capture device — against ``2·B·E + 2·E`` forwards
    per point saved down to ``B + B·E + 1 + E`` for the first point and
    ``B·E + E`` for each later one. A slice pass captures the **campaign's**
    tap union for its digest, not only the fit's own taps, so a campaign that
    also taps ``lm_head`` on the source digest from an inference point makes
    the fit retain vocabulary-wide captures over the training rows too.

    A whole-role capture lives exactly as long as a pass is still **owed** it.
    ``owed`` is the plan's count of forward-group instances per digest across
    every point; a pass that runs or is served a digest settles one, and the
    capture is dropped when the count reaches zero. A digest only one pass
    keys into is never stored at all — publishing it would pin a capture no
    later point can ask for, and for a swept *patched* group tapping
    ``lm_head`` that is a full-vocabulary tensor per point (150 rows of
    gemma-2-2b-it hold 13 GiB each; three of them exhaust an 80 GB device
    where the per-point loop never held more than one). Sliced keys are not
    settled: a fit's inner passes re-read them every step, so their count is
    not a plan quantity. Those live as long as the request.

    ``prefixes`` adds, per (prefix identity, rows, window, depth), one
    residual of ``rows × seq × d_model``, and one per intervened *input*
    rather than per point, since every model writing on the same rows at or
    above a depth shares that entry. Their lifetime is the plan's as well:
    ``prefix_owed`` counts, per (prefix identity, depth), the group instances
    across every point whose interned ``resume_at`` reaches that depth — the
    passes that could still start from it. A whole-role pass settling its
    group (:meth:`ExecutorBase._settle`) settles every depth up to its own
    ``resume_at``, and at zero every entry of that identity and depth is
    dropped, sliced keys included, and no pass stores a depth nobody is owed
    any more. What that bounds: at most ``|wanted depths|`` residuals per
    (identity, rows, window) are live at once — for an *ascending* scan the
    first pass stores every wanted depth and each dies only when the last
    group reaching it settles, so the peak is the whole set (a 10-depth scan
    at 2k context on a 40-block A3B: ~6.5 GB); the refcount removes the tail
    after the last sharer, and a fit's per-minibatch and eval slices die with
    the point, since the point's own pass runs *after* the fit. A depth past
    every block (``PAST_BLOCKS``) and the last block's depth name the same
    residual and are stored twice — a deliberate duplicate that keeps the
    refcount in plan coordinates. Capping the live set to the deepest ``k``
    entries per identity is a possible follow-up, not done here.

    **Prefixes** (§4 "Resume"). ``prefix_plans`` says, per group digest, the
    block its pass may start at and how deep its pass stays un-intervened;
    ``wanted_prefix_depths`` says, per prefix identity, every depth some group
    of the campaign will resume at, so the first pass over those rows — an
    ``original`` group's, or a shallower intervened model's — stores each of
    them on its way. Resuming is independent of the capture interning above:
    a fit's minibatch pass of the trained model is never *served* (its
    activations move every step) yet still *resumes*, because the prefix
    below its write is fit-constant even when the group is not.

    The store is keyed by :data:`CaptureKey` (a group digest, plus the row
    slice it ran over) and :data:`TapKey`, both engine-neutral, but only an
    engine whose ``_run_group`` consults it actually interns;
    :attr:`~causalab.protocol.engine.RunResult.forwards` then reports
    ``len(executed)``. A fit's inner passes are tallied apart — the constant
    groups it ran in ``inner_executed``, the ones it was served in
    ``inner_served`` — because they are not forward groups and are not
    counted; :func:`~causalab.neural.shared.execution.execute_request`
    reports the per-point difference as ``fit_forwards`` in each point's
    summary — the ``explain``-style record :class:`RunResult` hands back to
    the caller, not a file in the run tree.
    """

    #: group digest -> every site the campaign taps in that group's forward
    wanted: Mapping[str, tuple[SiteSpec, ...]] = dataclasses.field(default_factory=dict)
    #: capture key -> the raw activations one forward left, per tap
    captured: dict[CaptureKey, dict[TapKey, torch.Tensor]] = dataclasses.field(
        default_factory=dict
    )
    #: capture key -> the experts routing table beside those captures
    routing: dict[CaptureKey, dict[TapKey, torch.Tensor]] = dataclasses.field(
        default_factory=dict
    )
    #: group digest -> whole-role passes (across every point's plan) that
    #: still owe this digest a run or a serving; the publisher's own pass
    #: included. Decremented by :meth:`ExecutorBase._settle`; at zero the
    #: digest's captures are dropped. A digest absent here is not tracked:
    #: its captures are stored and kept for the request (the hand-built
    #: caches of the unit tests, or an engine that plans no campaign).
    owed: dict[str, int] = dataclasses.field(default_factory=dict)
    #: one entry per forward group actually run, in order — the number
    #: :attr:`~causalab.protocol.engine.RunResult.forwards` reports
    executed: list[str] = dataclasses.field(default_factory=list)
    #: one entry per fit-constant group a fit's inner executor (a minibatch,
    #: the eval pass) actually ran and published under a sliced key — the
    #: passes the cache did not save it. Never part of ``forwards``: inner
    #: passes of a fit are not forward groups. The non-constant pass a fit
    #: differentiates through every step is not recorded here either — it
    #: is the fit's own cost, one per step, and no cache is consulted for it.
    inner_executed: list[str] = dataclasses.field(default_factory=list)
    #: one entry per fit-constant group a fit's inner executor asked for and
    #: was **served** from a sliced key instead of running — the passes the
    #: cache did save. ``inner_executed`` + ``inner_served`` is what a fit
    #: would have paid for its constant groups without the store.
    inner_served: list[str] = dataclasses.field(default_factory=list)
    #: group digest -> where its pass may start and how deep it stays
    #: un-intervened (§4 "Resume"); a digest absent here never resumes
    prefix_plans: Mapping[str, PrefixPlan] = dataclasses.field(default_factory=dict)
    #: prefix identity (``base_digest``) -> every block some group of the
    #: campaign resumes at, i.e. the residuals a pass over those rows stores
    wanted_prefix_depths: Mapping[str, frozenset[int]] = dataclasses.field(
        default_factory=dict
    )
    #: prefix key -> the residual entering that block, detached, on the
    #: capture device
    prefixes: dict[PrefixKey, torch.Tensor] = dataclasses.field(default_factory=dict)
    #: (prefix identity, depth) -> whole-role group instances across every
    #: point's plan whose ``resume_at`` reaches ``depth``; the plan's count of
    #: passes that may still start from that prefix. Decremented by
    #: :meth:`ExecutorBase._settle`; at zero every ``prefixes`` entry of that
    #: identity and depth is dropped. Absent pairs are not tracked and live
    #: for the request, as ``owed`` treats an absent digest
    prefix_owed: dict[tuple[str, int], int] = dataclasses.field(default_factory=dict)
    #: one entry per window a pass resumed, holding the block it started at —
    #: ``len`` is how many forwards resumed, ``sum`` how many blocks were
    #: skipped. Both inner and counted passes land here: a resume is a saving
    #: whichever kind of pass it happens in, and a forward that resumed is
    #: still one forward in ``executed`` / ``inner_executed``
    resumed: list[int] = dataclasses.field(default_factory=list)
    #: group digest -> ``{write member: fire count}`` of the pass that ran it
    #: (§4 "Fires"): a point served this digest's captures records the counts
    #: of the pass that produced them, not zeroes for a forward it never ran.
    #: Kept for the request — a count is a few bytes, and it outlives the
    #: captures it describes
    fires: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Interning:
    """One point's handle on a shared :class:`ForwardCache`.

    ``digests`` maps this point's ``(model, input)`` groups to the plan
    digests they key into; the cache itself belongs to the whole campaign.
    An executor built without one runs every group itself and touches the
    store not at all — the unit tests' reference path, and what "no reuse"
    means.

    ``rows`` says which rows of each role this executor runs: ``None`` for the
    whole role (a campaign point), a tuple of indices for a minibatch, a split
    ref for the ``train.eval`` pass. It becomes the second coordinate of every
    :data:`CaptureKey` this executor reads or writes, so a slice is never
    served a whole-role capture or vice versa. ``counted`` says whether the
    passes this executor runs are forward groups of the campaign
    (``executed``, hence :attr:`~causalab.protocol.engine.RunResult.forwards`)
    or the inner passes of a fit (``inner_executed``). A fit's inner handles
    come from :meth:`ExecutorBase.inner_interning`; ``digests`` stays the
    point's whole map there, because a group the fit changes may not be served
    from the store (:meth:`ExecutorBase._may_intern`) yet still resumes from
    its un-intervened prefix (§4 "Resume"), and both are looked up by digest."""

    digests: Mapping[tuple[str, str], str]
    cache: ForwardCache
    rows: tuple[int, ...] | str | None = None
    counted: bool = True


def whole_native_tensor(
    rname: str, read: "ReadSpec | WriteSpec", raw: Any, site: ResolvedSite
) -> Any:
    """A read of a tap with **no contract form**: the whole native tensor.

    The one such tap is the attention pattern, ``(batch, heads, query, key)``,
    and the reason this is a bypass rather than a gather is its second position
    axis: the gather (dim 0 batch, dim 1 position) would index the head axis
    with position indices, and ``dims`` would slice the key axis as though it
    were features. Both produce plausible numbers from the wrong tensor.

    All three refusals below are *generated* from
    :class:`~causalab.protocol.shapes.FeatureShape` — position addressing needs
    one position axis, a featurizer needs a feature space, ``dims`` needs a
    feature axis to index — so a later tap with the same problem is refused by
    declaring its axes rather than by adding a branch here. What remains — the
    whole tensor, at ``pos: "all"`` — is exactly what an interchange on
    attention needs, and what nnterp's own check exercises
    (``self[layer] = rnd``).
    """
    shape = site.shape
    what = f"{site.component!r} ({shape.describe()})"
    pos = read.pos
    whole = getattr(pos, "all", None) is True or pos == ALL_POSITIONS
    if not whole:
        axes = ", ".join(a.label for a in shape.position_axes)
        raise ProtocolError(
            "P4",
            f"read {rname!r} addresses positions on {what}, which has "
            f"{len(shape.position_axes)} position axes ({axes}) — a position "
            "index would be ambiguous between them. Read the whole tensor with "
            'pos: "all".',
        )
    if read.featurizer is not None:
        raise ProtocolError(
            "P4",
            f"read {rname!r} featurizes {what}: {shape.refusal('it')} A "
            "featurizer would be fitted across an axis that is not a basis.",
        )
    if isinstance(read.dims, tuple):
        raise ProtocolError(
            "P4",
            f"read {rname!r} slices 'dims' on {what}: that would select "
            f"{shape.axes[-1].label} entries as though they were features.",
        )
    return raw


def _derive(site: ResolvedSite, value: torch.Tensor, rname: str) -> torch.Tensor:
    """Compute a derived component from the tensor its tap captured."""
    if site.derivation == "attention_result":
        return _attention_result(site, value)
    raise ProtocolError("P2", f"read {rname!r}: unknown derivation {site.derivation!r}")


def _attention_result(site: ResolvedSite, premix: torch.Tensor) -> torch.Tensor:
    """Head ``h``'s contribution to the residual stream.

    ``result[..., h, :] = premix[..., h·d:(h+1)·d] @ W_o[:, h·d:(h+1)·d].T`` —
    the part of the block's attention output that head ``h`` is responsible for.
    The model never forms it: it projects the whole premix at once, so what it
    computes is the *sum* over heads (plus the o-projection's bias, if it has
    one). ``sum_h result == attention_output - bias`` is the identity that
    defines this component, and the test suite pins it.

    Computed by **masking and re-projecting** rather than by slicing the weight
    matrix. That is deliberate: ``nn.Linear`` stores ``(out, in)`` and
    transformers' ``Conv1D`` (GPT-2's ``c_proj``) stores ``(in, out)``, so a
    weight-slicing implementation has to know which family it is looking at and
    is silently wrong if it guesses. Running the projection the model's own
    module defines cannot be wrong about its own layout, and the bias — which is
    *not* attributable to any head — is subtracted back off explicitly.

    ⚠️ Calls ``site.module`` directly, so it needs something that runs the
    projection when called: a real ``nn.Module``, or an envoy inside a trace
    body, where the call runs the module the envoy resolves to — which is how
    the nnterp engine derives it, in its block, over the gathered
    rows.
    """
    module = site.module
    bias = getattr(module, "bias", None)
    heads = site.shape.head_space
    assert heads is not None  # the premix tap always has a head axis
    per_head = premix.shape[-1] // heads

    def contribution(head: int) -> torch.Tensor:
        masked = torch.zeros_like(premix)
        window = slice(head * per_head, (head + 1) * per_head)
        masked[..., window] = premix[..., window]
        out = module(masked)
        return out if bias is None else out - bias

    if site.head is not None:
        return contribution(site.head)
    # The whole tensor: `heads` times wider than `attention_output`. On a real
    # A3B that is 64x at hidden 4096, which is why naming a `head` is
    # encouraged — but it is a documented cost, not a refusal.
    return torch.cat([contribution(head) for head in range(heads)], dim=-1)


def _part(k: int, parts: Sequence[Any]) -> str:
    """`` constituent k`` for a spec classified as several addresses, nothing
    for one."""
    return f" constituent {k}" if len(parts) > 1 else ""


def _ragged_write_error(
    ename: str, widths: list[int], *, model: str | None = None
) -> ValidationError:
    """One message for rule 19's ``refuse`` path, raised from the pre-flight
    and from the landing path — the same refusal wherever it is noticed
    first. Typed ``ragged_write_unsupported`` (§2.4): a write that declares
    no ``ragged`` policy, or ``refuse``, meets this; a declared
    ``exact_length_buckets`` / ``padded_masked`` lands instead
    (:func:`land_ragged`)."""
    where = f" in intervened_model {model!r}" if model else ""
    return ValidationError(
        19,
        f"write {ename!r}{where} addresses ragged position widths "
        f"{widths} — an all-positions or variable write needs every row to "
        "address the same number of positions, because the landed slice has "
        "one shape for the whole batch (§5.19)",
        path=f"writes.{ename}.pos",
        reason="ragged_write_unsupported",
    )


def _ragged_operand_error(value: str) -> ValidationError:
    """Rule 19 for a ragged *operand* under ``refuse`` (the absent-field
    behaviour): the read it pairs into the write came back with unequal
    per-row widths, and there is no aligned shape to land it on."""
    return ValidationError(
        19,
        f"operand read {value!r} is ragged (unequal per-row "
        "position widths) — pairing ragged windows into a write "
        "has no aligned shape, so the write is refused rather "
        "than landed on a guess (§5.19)",
        reason="ragged_write_unsupported",
    )


def _operand_width_error(
    value: str, mismatches: list[tuple[int, int, int]]
) -> ValidationError:
    """Rule 19 for an operand whose row widths disagree with the write's
    under a landing policy — a ragged read re-nested row by row
    (:func:`nest_ragged_operand`), or a dense read whose one
    width is not every row's (:func:`check_dense_operand`):
    an operand pairs into a ragged write row by row, at each row's own width,
    and a row where the two windows differ has no aligned shape — it is
    refused, never truncated or left-aligned into the narrower window.
    ``mismatches`` is every such ``(row, operand width, write width)`` over
    the whole window — the same rows, and so the same message, whichever
    policy lands the write."""
    rows = "; ".join(
        f"row {row} (operand {got}, write {want})" for row, got, want in mismatches
    )
    return ValidationError(
        19,
        f"operand read {value!r} has a width that disagrees with the write's "
        f"on {rows} — an operand pairs into a ragged write row by row, at each "
        "row's own width (only a one-position operand broadcasts), so the write "
        "is refused rather than landed on a guess (§5.19)",
        reason="ragged_write_unsupported",
    )


def ragged_geometry_of(policy: str, per_row: Sequence[Sequence[int]]) -> dict[str, Any]:
    """The receipt's record of one ragged write (§8 ``execution.ragged``):
    the declared policy, every row's width, and
    the ``[width, rows]`` buckets — what ``exact_length_buckets`` lands by
    and what ``padded_masked`` masks by. Recorded, not gated."""
    widths = [len(row) for row in per_row]
    return {
        "policy": policy,
        "widths": widths,
        "buckets": [[width, widths.count(width)] for width in sorted(set(widths))],
    }


def _operand_reads(payload: Any, reads: Mapping[str, Any]) -> tuple[str, ...]:
    """The read names a write's operand payload spells (§2.8): the ``swap``
    operand itself, or the read-valued options of a structured mechanism.
    Literals and params are not positions and name none."""
    if isinstance(payload, str):
        return (payload,) if payload in reads else ()
    if isinstance(payload, Mapping):
        return tuple(
            value
            for value in payload.values()
            if isinstance(value, str) and value in reads
        )
    return ()


def _pair_offsets(
    batch: EncodedBatch, row: int, text: str, *, role: str
) -> tuple[tuple[int, int], ...]:
    """The row's token char offsets in the coordinates of ``text`` — the pair
    side the ``edit_groups`` spans are declared over. The plain frame encodes
    the text verbatim; a chat frame renders it inside a template, so the
    offsets shift by where the text sits in what was encoded. Refused as rule
    27 when the encoded text does not contain the declared text exactly once
    — spans over text the frame did not encode name nothing."""
    encoded = batch.texts[row]
    offsets = batch.offset_mapping[row]
    if encoded == text:
        return offsets
    start = encoded.find(text)
    if start < 0 or encoded.find(text, start + 1) >= 0:
        raise ValidationError(
            27,
            f"data.{role} row {row}: the row's edit_groups spans are declared over "
            f"{text!r}, which the frame did not encode verbatim ({encoded!r}) — the "
            "spans name nothing in what the model reads (sec. 2.2 `edit_groups`)",
            path=f"data.{role}",
        )
    return tuple(
        (0, 0) if (a == 0 and b == 0) else (a - start, b - start) for a, b in offsets
    )


# ---------------------------------------------------------------------- #
# the write math, as functions of data
#
# Everything a write needs from the document and the executor arrives
# through a `WriteServices`, so the same math runs from a method of a live
# executor (the hook engines: the services are its bound methods) and from a
# block that ships to another process (the nnterp engine on NDIF: the
# services are tables prepared before the trace, and no executor travels).
# ---------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class WriteServices:
    """What the write math asks of whoever lands a write.

    ``positions_of(ename, write)`` is the write's positions over **every**
    row of the role (the whole padded frame). ``lookup(value, *, rows=None,
    ragged=None)`` resolves one operand (:meth:`ExecutorBase._operand_lookup`
    is the contract). ``stack_of(ename, write, site)`` is the write's
    featurizer stack. ``routing_of(operand, rows)`` is the routing table a
    tensor operand was read beside, ``None`` when it carries none. ``reads``
    answers ``name in reads`` for the document's read names; ``code`` is the
    document's ``code`` table (only ``pytorch_fn`` reads it). ``mismatches``
    receives the expert alignment's counts, keyed ``(write, layer,
    examples)`` — the caller owns where they are recorded.
    """

    positions_of: Callable[[str, WriteSpec], "list[list[int]]"]
    lookup: Callable[..., "torch.Tensor | float"]
    stack_of: Callable[[str, WriteSpec, ResolvedSite], FeaturizerStack]
    routing_of: Callable[[Any, "RowWindow | None"], "torch.Tensor | None"]
    reads: Container[str]
    code: Mapping[str, Any] | None
    mismatches: "dict[tuple[str, int, tuple[int, ...]], tuple[torch.Tensor, int]]"


def gather_rows(
    tensor: torch.Tensor, per_row: Sequence[Sequence[int]]
) -> "torch.Tensor | RaggedValue":
    """``tensor`` at each row's positions: dense ``(rows, width, …)`` when
    every row addresses one width, else the flat rows plus their widths."""
    widths = {len(row) for row in per_row}
    if len(widths) == 1:
        return gather_positions(tensor, dense_index(per_row, tensor.device))
    # ragged: one flat advanced index, (total_positions, ...) + widths
    return RaggedValue(
        flat=gather_positions(tensor, flat_index(per_row, tensor.device)),
        widths=tuple(len(row) for row in per_row),
    )


def read_features(
    value: torch.Tensor,
    site: ResolvedSite,
    stack: FeaturizerStack,
    dims: Any,
    *,
    routing: torch.Tensor | None = None,
    grad_enabled: bool = False,
) -> torch.Tensor:
    """The feature tail of a read over its gathered rows: the head's slice,
    the featurizer stack, then ``dims`` — what a read's value is past the
    gather, and what a later write consumes as its operand."""
    if site.feature_slice is not None:
        value = value[..., site.feature_slice]
    if not stack.is_identity:
        # a read whose executor keeps no gradients is detached afterwards, so
        # its featurize builds no graph — and shares the no-grad
        # evaluation the write hooks of the same pass use (featurizer_cache);
        # a grad-enabled executor's read inherits the ambient mode rather
        # than forcing grad on as the write hooks do: a read never trains
        with contextlib.nullcontext() if grad_enabled else torch.no_grad():
            value, _errs = stack.featurize(value, routing=routing)
    if isinstance(dims, tuple):
        index = torch.tensor(list(dims), dtype=torch.long, device=value.device)
        value = value.index_select(-1, index)
    return value


def read_operand(
    value: str,
    stored: "torch.Tensor | RaggedValue",
    *,
    device: Any,
    rows: "RowWindow | None" = None,
    ragged: Sequence[int] | None = None,
    positioned: bool = True,
) -> torch.Tensor:
    """A read's stored value as one write's operand, on ``device``: sliced
    to ``rows`` (dim 0 is the example), and under a landing policy
    (``ragged``, the write's per-row widths) re-nested or width-checked row
    by row. ``positioned`` says the read has a position axis at dim 1 — a
    whole-tensor read or a state read has none and pairs by broadcast."""
    if isinstance(stored, RaggedValue):
        if ragged is None:
            raise _ragged_operand_error(value)
        return nest_ragged_operand(value, stored, rows, ragged, device)
    operand = stored.to(device)
    if rows is not None and not rows.whole:
        operand = operand[rows.index]
    if ragged is not None and positioned:
        check_dense_operand(value, operand, rows, ragged)
    return operand


def check_dense_operand(
    value: str,
    operand: torch.Tensor,
    rows: "RowWindow | None",
    widths: Sequence[int],
) -> None:
    """A dense, positioned read as a write operand under a landing policy
    (§5 rule 19): ``operand`` is ``(rows, width, …)`` already sliced to
    ``rows``, and its one ``width`` must be every row's own landed width in
    ``widths``, or one (a single position broadcasts over a row, as it
    always has). Any other row is :func:`_operand_width_error`, naming the
    row and both widths — the refusal a :class:`RaggedValue` operand's
    disagreeing row gets, so the two landing policies refuse one document
    identically."""
    if operand.dim() < 2:
        return
    got = int(operand.shape[1])
    if got == 1:
        return
    examples = list(range(len(widths))) if rows is None else rows.examples
    mismatches = [
        (row, got, int(width))
        for row, width in zip(examples, widths)
        if got != int(width)
    ]
    if mismatches:
        raise _operand_width_error(value, mismatches)


def nest_ragged_operand(
    value: str,
    stored: RaggedValue,
    rows: "RowWindow | None",
    widths: Sequence[int],
    device: Any,
) -> torch.Tensor:
    """A ragged read as a write operand under a landing policy (§5 rule
    19): its rows over ``rows``, each checked to be exactly as wide as the
    write's window on that row, stacked into ``(rows, max width, …)`` with
    zero padding past a row's width — the same frame the write's
    ``padded_masked`` gather uses, and a bucket's frame when every width
    is the same. Padding is never written back: the landing masks it."""
    chunks = torch.split(stored.flat.to(device), list(stored.widths))
    examples = list(range(len(chunks))) if rows is None else rows.examples
    picked = [chunks[i] for i in examples]
    want = [int(width) for width in widths]
    mismatches = [
        (row, int(chunk.shape[0]), width)
        for row, chunk, width in zip(examples, picked, want)
        if int(chunk.shape[0]) != width
    ]
    if mismatches:
        raise _operand_width_error(value, mismatches)
    out = stored.flat.new_zeros((len(picked), max(want), *stored.flat.shape[1:]))
    for i, chunk in enumerate(picked):
        out[i, : chunk.shape[0]] = chunk
    return out


def _class_rank(entry: tuple[str, WriteSpec, ResolvedSite]) -> int:
    do = entry[1].do
    if str(do.mechanism) == "renormalize":
        return 2  # after the deltas — the only order where it acts (§2.8 note)
    return 1 if is_additive(do) else 0  # absolute first, then additive


def apply_writes_to_contract(
    entries: Sequence[tuple[str, WriteSpec, ResolvedSite]],
    tensor: torch.Tensor,
    services: WriteServices,
    *,
    per_row: list[list[int]] | None = None,
    rows: "RowWindow | None" = None,
    routing: torch.Tensor | None = None,
) -> None:
    """Apply every write at one address, in class order, mutating the
    contract-shaped ``tensor`` in place — absolute first, additive deltas
    summed, renormalize last against the pre-write norm (§2.8).

    ``per_row`` overrides position resolution, the same override
    :meth:`ExecutorBase._finalize_read` takes: a caller whose position axis
    is not the token axis (a per-chunk state) has already worked the indices
    out.

    ``rows`` is the window of the role's rows ``tensor`` holds — the
    microbatch. Positions resolve against the whole padded frame (a
    window is a row slice of it, so the indices coincide), then both they
    and any tensor operand are sliced to the window; ``None`` is the whole
    batch.

    ``routing`` is the routing table of a routed-interior address,
    ``(batch, position, top_k)`` over the same rows as ``tensor`` — the
    expert ids an expert-keyed gate keys its parameters by, gathered at
    the write's positions alongside the value."""
    if rows is None:
        rows = RowWindow(0, tensor.shape[0], tensor.shape[0])
    lookup = functools.partial(services.lookup, rows=rows)

    for ename, write, site in sorted(entries, key=_class_rank):
        if not site.shape.has_contract_form:
            # Symmetric with the read (see whole_native_tensor): this
            # tensor's feature axis is a position axis, so the position
            # gather below would index heads with positions and `dims`
            # would slice key positions as features. Both are refused
            # there; what is left is the whole tensor, edited whole.
            whole_native_tensor(ename, write, tensor, site)
            mechanism = str(write.do.mechanism)
            if mechanism == "swap":
                replacement = lookup(write.do.payload)
                if not isinstance(replacement, torch.Tensor):
                    raise ProtocolError(
                        "P2",
                        f"write {ename!r} swaps {site.component!r} with "
                        "a scalar; a whole-tensor interchange needs a "
                        "tensor operand read from elsewhere",
                    )
                if replacement.shape != tensor.shape:
                    raise ProtocolError(
                        "P2",
                        f"write {ename!r} replaces the whole "
                        f"{site.component!r} tensor, but its operand has "
                        f"shape {tuple(replacement.shape)} and the tap is "
                        f"{tuple(tensor.shape)} — an interchange needs "
                        "both inputs to have the same number of positions",
                    )
                tensor.copy_(replacement.to(tensor.dtype))
            elif mechanism == "gaussian":
                # 📐 The noise is drawn as (batch, position, feature) and
                # its `axis` names the feature axis' tensor-parallel
                # semantics. This tap has no feature axis — its last axis
                # is key positions — so there is nothing for either to
                # mean, and the draw does not even fit (measured: "shape
                # '[1, 8, 5, 5]' is invalid for input of size 40").
                # Refused by name rather than reshaped into something that
                # would run.
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} applies 'gaussian' to "
                    f"{site.component!r}, whose shape is "
                    f"{site.shape.describe()}: the noise is drawn per "
                    "(batch, position, feature) and its 'axis' names how "
                    "the feature axis is sharded, and this tap has no "
                    "feature axis at all. Swap in a noise tensor of the "
                    "tap's own shape instead.",
                )
            else:
                # 📐 Arithmetic on the whole tensor, with no gather: for
                # `attention_scores` this is the point of the component.
                # `written_value` broadcasts a scalar operand over any
                # rank, and `dims` and featurizers are already refused
                # above, so there is no feature axis for it to mis-slice.
                tensor.copy_(
                    written_value(
                        ename, write, site, tensor, services, lookup=lookup, rows=rows
                    ).to(tensor.dtype)
                )
            continue
        if per_row is not None:
            positions = per_row
            pad_to = max((len(row) for row in positions), default=0)
        else:
            # resolved against the whole padded frame, then sliced to the
            # window; the widest row of the *whole* batch is what a masked
            # landing pads to, so the `gaussian` draw a row receives does
            # not depend on how the batch was cut (§8)
            every = services.positions_of(ename, write)
            positions = every[rows.slice]
            pad_to = max((len(row) for row in every), default=0)
        widths = {len(row) for row in positions}
        policy = write.ragged or "refuse"
        if len(widths) != 1:
            if policy == "refuse":
                raise _ragged_write_error(ename, sorted(widths))
            land_ragged(
                ename,
                write,
                site,
                tensor,
                positions,
                services,
                policy=policy,
                pad_to=pad_to,
                rows=rows,
                routing=routing,
            )
            continue
        (width,) = widths
        # this write's lookup alone: the ragged binding below must not
        # leak into a later write of the same landing call, whose operands
        # would then be held to *this* write's width (or, under `refuse`,
        # refused with the width message instead of the operand one)
        write_lookup = lookup
        if policy != "refuse":
            # uniform on this window (or on the whole batch): the dense
            # landing below, with a ragged operand welcome at this width
            write_lookup = functools.partial(
                services.lookup, rows=rows, ragged=[width] * len(positions)
            )
        index = dense_index(positions, tensor.device)
        fslice = site.feature_slice or slice(None)
        # one gather of the landed positions, sort-free backward when the
        # table repeats no element (gather.py); the write-back splices
        # the new features into it rather than gathering again
        landed = gather_positions(tensor, index)
        v_new = written_value(
            ename,
            write,
            site,
            landed[..., fslice],
            services,
            lookup=write_lookup,
            rows=rows,
            routing=None if routing is None else routing[index.pair],
        )
        tensor[index.pair] = splice_features(landed, fslice, v_new.to(tensor.dtype))


def land_ragged(
    ename: str,
    write: WriteSpec,
    site: ResolvedSite,
    tensor: torch.Tensor,
    positions: list[list[int]],
    services: WriteServices,
    *,
    policy: str,
    pad_to: int,
    rows: "RowWindow",
    routing: torch.Tensor | None,
) -> None:
    """Land one write whose rows address different numbers of positions,
    under its declared ``ragged`` policy (§2.8, §5 rule 19) — every row at
    its own width, inside the forward the window already runs, so nothing
    about batch geometry, fire counts or prefix keys changes:

    * ``exact_length_buckets`` groups the window's rows by width and lands
      one dense gather per width — a :meth:`RowWindow.bucket`, so a tensor
      operand, a routing table and the ``gaussian`` draw are indexed by
      the bucket's rows exactly as a window slices them;
    * ``padded_masked`` pads every row's positions to ``pad_to`` (the
      widest row of the batch) with its own last position, lands one
      gather over the padded frame, and scatters back **only** the real
      slots — the pad slot is read (a duplicate of a real activation) and
      never written.

    Every per-position mechanism writes the same values under either
    policy; only a ``gaussian`` draw, shaped by the landed slice, differs
    between a bucket's width and the padded width. The pre-flight
    (:meth:`ExecutorBase.check_write_widths`) has already recorded the
    geometry.
    """
    fslice = site.feature_slice or slice(None)
    widths = [len(row) for row in positions]
    # every operand that is a read is paired to the whole window first,
    # whichever policy lands it: an operand whose widths disagree with the
    # write's on any row is rule 19 here, naming the same rows under both
    # policies — not the first bucket's alone, and never `_coerce`'s P2
    for name in operand_names(write.do.payload):
        if name in services.reads:
            services.lookup(name, rows=rows, ragged=widths)
    if policy == "exact_length_buckets":
        for width in sorted(set(widths)):
            members = [i for i, w in enumerate(widths) if w == width]
            bucket = rows.bucket(members)
            index = dense_index(
                [positions[i] for i in members], tensor.device, rows=members
            )
            landed = gather_positions(tensor, index)
            v_new = written_value(
                ename,
                write,
                site,
                landed[..., fslice],
                services,
                lookup=functools.partial(
                    services.lookup, rows=bucket, ragged=[width] * len(members)
                ),
                rows=bucket,
                routing=None if routing is None else routing[index.pair],
            )
            tensor[index.pair] = splice_features(landed, fslice, v_new.to(tensor.dtype))
        return
    if policy != "padded_masked":
        raise AssertionError(f"unknown ragged policy {policy!r} reached the landing")
    pad_to = max(pad_to, max(widths))
    padded = [
        [*row, *([row[-1] if row else 0] * (pad_to - len(row)))] for row in positions
    ]
    # a pad slot duplicates a real position, so where any row is short
    # this table is not distinct and the gather keeps autograd's
    # accumulating backward; the index decides that itself (gather.py)
    index = dense_index(padded, tensor.device)
    landed = gather_positions(tensor, index)
    v_new = written_value(
        ename,
        write,
        site,
        landed[..., fslice],
        services,
        lookup=functools.partial(services.lookup, rows=rows, ragged=widths),
        rows=rows,
        routing=None if routing is None else routing[index.pair],
    )
    spliced = splice_features(landed, fslice, v_new.to(tensor.dtype))
    # only the real slots go back: a pad slot duplicates a real index, and
    # an advanced-index assignment with duplicates lands one of the two
    # values arbitrarily — so padding is never written, by construction.
    # The real (row, slot) pairs are known on the host, so selecting them
    # is a distinct gather rather than a boolean mask (which would need
    # the device to count its hits)
    real_slots = flat_index([list(range(width)) for width in widths], tensor.device)
    real_positions = flat_index(positions, tensor.device)
    tensor[real_positions.pair] = gather_positions(spliced, real_slots)


def written_value(
    ename: str,
    write: WriteSpec,
    site: ResolvedSite,
    v_pre: torch.Tensor,
    services: WriteServices,
    *,
    lookup: "Callable[[Any], torch.Tensor | float] | None" = None,
    rows: "RowWindow | None" = None,
    routing: torch.Tensor | None = None,
) -> torch.Tensor:
    """featurize → class-ordered do → inverse, honoring dims and the
    error-term contract.

    ``lookup`` overrides operand resolution — the state-write path slices
    tensor operands to one (row, step) so the same mechanism math applies
    per step; everything else uses ``services.lookup`` unchanged.

    ``rows`` is the window ``v_pre`` covers, which only a ``gaussian``
    write needs: its draw is made over the whole batch and sliced, so the
    noise a row receives does not depend on how the batch was cut.

    ``routing`` is the routing table at ``v_pre``'s rows and positions,
    ``(batch, position, top_k)``, when the address is the routed interior.
    A write through an expert-keyed gate featurizes with it and joins
    every tensor operand to it by expert id (:func:`align_by_expert`), so
    a slot receives the operand's value for the *same expert* and a slot
    whose expert the operand never activated is left unchanged.
    """
    if lookup is None:
        lookup = services.lookup
    stack = services.stack_of(ename, write, site)
    f0, errs = stack.featurize(v_pre, routing=routing)
    dims = None
    if isinstance(write.dims, tuple):
        if stack.needs_routing:
            raise ProtocolError(
                "P4",
                f"write {ename!r} slices 'dims' through an expert-keyed gate: "
                "the token-major axis is joined to experts per token, so a "
                "fixed coordinate subset names different neurons on "
                "different tokens. Select neurons with the gate instead.",
            )
        dims = torch.tensor(list(write.dims), dtype=torch.long, device=f0.device)

    def select(f: torch.Tensor) -> torch.Tensor:
        return f if dims is None else f.index_select(-1, dims)

    def aligned(fill: torch.Tensor) -> "Callable[[Any], torch.Tensor | float]":
        """Operand resolution that joins a tensor operand's slots to
        ``v_pre``'s by expert; ``fill`` is what a slot with no source
        receives, chosen so the mechanism leaves it unchanged."""
        assert lookup is not None and routing is not None

        def resolve(value: Any) -> torch.Tensor | float:
            operand = lookup(value)
            if not isinstance(operand, torch.Tensor):
                return operand
            joined, key, counts = align_by_expert(
                ename,
                value,
                operand,
                routing,
                fill,
                source_routing=services.routing_of(value, rows),
                layer=site.layer,
                rows=rows,
            )
            # re-inserted at the end so a flush lands the calls in order (the
            # last write of a row wins)
            services.mismatches.pop(key, None)
            services.mismatches[key] = counts
            return joined

        return resolve

    # `f` is written into in place only where a `dims` slice lands in it
    # (the three `index_copy_` below), and then into its own copy: `f0` is
    # the featurizer's output, which its error term saved for backward (a
    # subspace's `x - f @ Qᵀ`), and through the identity stack the gathered
    # slice itself. A whole-axis result is a fresh tensor already (the sum,
    # the broadcast operand, the renormalized product), and every consumer
    # copies it into the model's tensor rather than mutating it — so no
    # defensive clone there
    f = f0 if dims is None else f0.clone()
    do = write.do
    batch_size, n_pos = v_pre.shape[0], v_pre.shape[1]
    if str(do.mechanism) == "renormalize":
        pass  # applied last, below
    elif is_additive(do):
        delta = apply_delta(
            do,
            select(f0),
            aligned(torch.zeros_like(f0)) if stack.needs_routing else lookup,
            batch=batch_size if rows is None else rows.total,
            n_pos=n_pos,
            rows=None if rows is None else rows.index,
        )
        if dims is None:
            f = f0 + delta
        else:
            f.index_copy_(-1, dims, select(f0) + delta)
    else:
        written = apply_absolute(
            do,
            select(f0),
            aligned(f0) if stack.needs_routing else lookup,
            code=services.code,
        )
        written = written.broadcast_to(select(f0).shape).to(f0.dtype)
        if dims is None:
            f = written
        else:
            f.index_copy_(-1, dims, written)
    if str(do.mechanism) == "renormalize":
        if dims is None:
            f = apply_renormalize(f, f0)
        else:
            f.index_copy_(-1, dims, apply_renormalize(select(f), select(f0)))
    return stack.inverse(f, errs)


def align_by_expert(
    ename: str,
    operand_name: Any,
    operand: torch.Tensor,
    routing: torch.Tensor,
    fill: torch.Tensor,
    *,
    source_routing: torch.Tensor | None,
    layer: int | None,
    rows: "RowWindow | None" = None,
) -> "tuple[torch.Tensor, tuple[str, int, tuple[int, ...]], tuple[torch.Tensor, int]]":
    """Join a tensor operand's routed slots to the written slots by expert
    id (§2.5 ``expert_neuron``).

    Slot *k* of a token holds its *k*-th ranked expert, so the same slot
    on the operand's side may hold a different expert. For every written
    slot holding expert ``e``, the source is the operand's slot holding
    ``e`` at the same row and position when ``e`` is active there, and
    ``fill`` otherwise — the pre-write feature value for an absolute
    write, zero for an additive one, so a slot with no source keeps its
    base value.

    Returns the joined operand, and the count of written slots with no
    source per example — ``(write, layer, examples)`` and ``(missing,
    slots per example)`` — for the caller to record
    (:attr:`ExecutorBase.routing_mismatch`). The key names the role's rows,
    not the window's (nor the bucket's), so a microbatched layout names the
    same examples as a whole-batch one; the counts stay on the device — a
    host read inside a layer hook would stall the launch stream once per
    layer, for a record only the point's full-data pass is ever asked for.

    ``source_routing`` is the routing table the operand was read beside
    (``services.routing_of``): the operand must have been read at a
    routed-interior site (it carries expert ids) over the same rows and
    positions as the write — a broadcast operand has no slot-to-expert map
    to join on, and is refused rather than landed slot for slot.
    """
    if source_routing is None:
        raise ProtocolError(
            "P2",
            f"write {ename!r} hands {operand_name!r} to an expert-keyed gate, "
            "but that operand carries no routing table — the source of a "
            "write through group 'expert_neuron' is a read at the routed "
            "interior, whose expert ids say which of its slots matches which "
            "of the written ones",
        )
    operand = operand.to(device=fill.device, dtype=fill.dtype)
    if operand.shape != fill.shape or source_routing.shape != routing.shape:
        raise ProtocolError(
            "P2",
            f"write {ename!r}: operand {operand_name!r} covers "
            f"{tuple(operand.shape)} with routing {tuple(source_routing.shape)}, "
            f"but the write addresses {tuple(fill.shape)} with routing "
            f"{tuple(routing.shape)} — slots are joined by expert per (example, "
            "position), so both sides must address the same positions",
        )
    top_k = routing.shape[-1]
    per_slot = operand.shape[-1] // top_k
    # (…, written slot, operand slot): does the operand's slot hold the
    # written slot's expert? An expert appears at most once per token, so
    # at most one operand slot matches
    match = routing.unsqueeze(-1) == source_routing.unsqueeze(-2)
    found = match.any(-1)
    source_slot = match.to(torch.int8).argmax(-1)
    slots = operand.reshape(*operand.shape[:-1], top_k, per_slot)
    picked = slots.gather(
        -2, source_slot.unsqueeze(-1).expand(*source_slot.shape, per_slot)
    )
    aligned = torch.where(found.unsqueeze(-1), picked, fill.reshape(slots.shape))
    assert layer is not None  # the routed interior is a layered component
    missing = (~found).reshape(found.shape[0], -1).sum(-1)
    per_example = found[0].numel()
    examples = list(range(len(missing))) if rows is None else rows.examples
    key = (ename, layer, tuple(examples))
    return aligned.reshape(operand.shape), key, (missing, per_example)


class ExecutorBase:
    """Execute one concrete document against one loaded model.

    Subclasses implement :meth:`_run_group` — everything else is shared."""

    #: Whether a no-grad read's finalized value stays on its device instead
    #: of moving to the CPU (``_finalize_read``) — detached either way, only
    #: the placement is the flag's: a fit's eval executor when a metric
    #: selects from the read on the device and its scorer copies the columns
    #: rather than the vocabulary (``training.loop.score``), and a CUDA evaluation
    #: capture (``graph_cohort.EvaluationGraphs``). Off by default: a point's
    #: own passes hand CPU values to the writers.
    device_reads = False

    def __init__(
        self,
        doc: Document,
        bundle: Any,
        *,
        role_rows: Mapping[str, list[dict[str, Any]]],
        role_fields: Mapping[str, str],
        load_tensors: Callable[[str], Any],
        load_table: Callable[[str], Any] | None = None,
        stage_cache: dict[str, Stage] | None = None,
        grad_enabled: bool = False,
        coords: Mapping[str, Any] | None = None,
        interning: Interning | None = None,
        batch_rows: int | None = None,
        batches: Mapping[str, EncodedBatch] | None = None,
    ) -> None:
        #: the document in its execution form: a band site (§2.4 ``layers``)
        #: is one address across N layers, and every consumer below reasons
        #: about one module at a time, so a band is lowered to its per-layer
        #: members here — the N-site document the author would have written by
        #: hand (:func:`~causalab.protocol.plan.lower_bands`). Names the
        #: caller asks for (``read_value``, ``resolution``) are unchanged:
        #: what a band read has no single value for, lowering refuses by name
        self.doc = doc = lower_bands(doc)
        self.bundle = bundle
        self.role_rows = dict(role_rows)
        self.role_fields = dict(role_fields)
        self.load_tensors = load_tensors
        #: the score-table loader a gate's ``init.from_scores`` reads through
        #: (§2.5); ``None`` where no artifact store backs the executor
        self.load_table = load_table
        self.stage_cache: dict[str, Stage] = (
            stage_cache if stage_cache is not None else {}
        )
        self.grad_enabled = grad_enabled
        #: at most this many rows per forward (§8, execution scale): a group
        #: over more rows runs as several forwards over row windows whose
        #: captures are concatenated in row order. ``None`` is one forward per
        #: group. An execution parameter — it never enters a digest or a stamp
        #: (the run receipt records it, ``protocol/run.py``). Checked positive
        #: where it enters (the engine constructor and the CLI's argparse
        #: type), not again here
        self.batch_rows = batch_rows
        #: the campaign's shared forward groups, or ``None`` to run every
        #: group this point declares (§3 interning is opt-in per executor
        #: because only the campaign loop knows the points share one row set)
        self.interning = interning
        #: the models no trained parameter can reach (§4 "Fits") — the only
        #: groups an executor that runs *inside* a fit may serve from, or
        #: publish to, the shared store
        self.fit_constant_models = fit_constant_models(doc)
        # the seed every freshly built featurizer initialises from; the stage
        # cache is keyed by name alone, so it belongs to this one point
        self.seed = document_seed(doc)
        #: this point's sweep coordinates — they select the matching entry of
        #: a swept bundle a loaded featurizer/param points at (§2.5)
        self.coords = dict(coords or {})
        self._read_values: dict[str, torch.Tensor | RaggedValue] = {}
        #: a fit's eval metrics resolved to token ids over this executor's
        #: rows (``training.spec.ScoreSpec``, set by
        #: ``training.executors.score_executor``) — the rows never change, so
        #: a fit's eval executor resolves them once, not per pass
        self.score_spec: Any = None
        self._deferred_heads: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        #: per dense read at a routed-interior site, the routing table
        #: gathered at the read's own rows and positions, ``(batch, position,
        #: top_k)`` — what lets a write through an expert-keyed gate align that
        #: read's slots to its own by expert id (:func:`align_by_expert`)
        self._read_routing: dict[str, torch.Tensor] = {}
        #: ``(write, layer, example) -> (mismatched, slots)``: per write through
        #: an expert-keyed gate, how many of the example's base slots held an
        #: expert inactive on the operand's side and so kept their base value,
        #: out of the slots the write addressed. Saved as
        #: ``routing_mismatch.json`` beside the fit (§2.5 ``expert_neuron``).
        #: Keyed rather than appended so a re-run of the group overwrites
        self._routing_mismatch: dict[tuple[str, int, int], tuple[int, int]] = {}
        #: what the writes have not yet read back: per (write, layer, the
        #: window's examples) the latest per-example count of unmatched
        #: slots, still on the device (:meth:`routing_mismatch` flushes)
        self._routing_mismatch_pending: dict[
            tuple[str, int, tuple[int, ...]], tuple[torch.Tensor, int]
        ] = {}
        #: reads whose scoped slice selected no rows at all — legal cells with
        #: nothing to measure, reported as ``unavailable`` values rather than
        #: raised (spec §4.1). The read's value is still the width-0 gather.
        self._unavailable: dict[str, Unavailable] = {}
        #: per read, the rows an address could not be aligned on (§4.1): the
        #: row-level half of ``_unavailable``, so a metric over the read can
        #: exclude those rows and score the rest (§2.10 "Eligibility")
        self._row_unavailable: dict[str, dict[int, Unavailable]] = {}
        self._groups_run: set[tuple[str, str]] = set()
        self._write_widths_checked = False
        #: ``(intervened model, write) -> {"policy", "widths", "buckets"}``
        #: for every write whose rows address different numbers of positions
        #: and whose declared ``ragged`` policy lands them (§5 rule 19):
        #: filled by :meth:`check_write_widths` before any forward, the run
        #: receipt's ``execution.ragged`` entry for the point
        #: (:func:`ragged_geometry_of`). Empty for every document that
        #: authors no policy — such a write is refused there instead.
        self.ragged_geometry: dict[tuple[str, str], dict[str, Any]] = {}
        self._answer_forms_checked = False
        self._edit_groups_checked = False
        #: what :meth:`check_scoring` found for the base table — ``None``
        #: until it has run; the run receipt's ``scoring`` block per point
        self.scoring_check: ScoringCheck | None = None
        #: (position, input role) pairs whose declared ``alignment`` has been
        #: checked against the pair on this executor — once per batch, as the
        #: forward-group interning caches once per closure (§2.3)
        self._alignment_checked: set[tuple[str, str]] = set()
        #: the location ledger (§6), built on first request and only when the
        #: document saves one (:meth:`location_ledger`)
        self._ledger: LocationLedger | None = None
        #: the encoded frame per input role — built on first use from the
        #: role's rows, or handed in: a fit's minibatch executor runs a row
        #: *selection* of its point's frame (``EncodedBatch.select``) so every
        #: minibatch of every point in a cohort shares one padded width and
        #: their forwards concatenate (§4 "Cohorts"). What is handed in must
        #: describe exactly ``role_rows``, row for row; the frame's texts are
        #: checked against the rows' field on first use
        self._batches: dict[str, EncodedBatch] = dict(batches or {})
        self._batches_checked: set[str] = set()
        self._continuations: dict[tuple[str, str], Continuation] = {}
        #: per generate read, the decode steps each row addresses — the
        #: same list the gather used, kept because metrics need to know
        #: *which* steps a value covers (and that a row covered none)
        self._read_steps: dict[str, list[list[int]]] = {}
        #: implementation requirements the run's addresses imposed (§7.3,
        #: e.g. "attn_eager") — execution metadata, stamped into the artifact
        #: identity next to the engine name, never canonical form
        self.applied_requirements: set[str] = set()
        #: per forward group this executor ran or was served, ``{write
        #: member: fire count}`` (§4 "Fires", ``neural/shared/fires.py``):
        #: the run receipt's ``fires`` entry for the point. Only groups with
        #: writes appear; a group whose member fired other than its declared
        #: count never gets here — the point is refused before anything is
        #: published
        self.fires: dict[tuple[str, str], dict[str, int]] = {}

    # ------------------------------------------------------------------ #
    # public surface
    # ------------------------------------------------------------------ #

    def read_value(self, name: str) -> "torch.Tensor | RaggedValue":
        """The (featurized, dims-selected) value of one read; runs its
        group (and, transitively, operand groups) on first use."""
        if name not in self._read_values:
            self._preflight()
            read = self.doc.reads[name]
            self._run_group(str(read.model), str(read.input))
        value = self._read_values[name]
        if name in self._deferred_heads:
            value = (
                RaggedValue(self._project_deferred(name, value.flat), value.widths)
                if isinstance(value, RaggedValue)
                else self._project_deferred(name, value)
            )
            self._read_values[name] = value
            del self._deferred_heads[name]
        return value

    def _project_deferred(self, name: str, value: "torch.Tensor") -> "torch.Tensor":
        """Run read ``name``'s owed ``lm_head`` over ``value`` and hand the
        logits back where the read's value lives.

        A head is only deferred when gradients are off (the registration in
        the engine's generate tail), and that is exactly when
        :meth:`_finalize_read` has detached the kept ``ln_final`` rows to the
        CPU — so the projection runs on the bundle's device, where the head's
        weight is, and its result comes back to the CPU like every other
        stored read value. On a CPU bundle both moves are no-ops."""
        head = self._deferred_heads[name]
        with torch.no_grad():
            return head(value.to(self.bundle.device)).detach().cpu()

    def generated_metric(self, metric: Any) -> list[list[Any]]:
        """Reduce ordinary continuation metrics before releasing each projection.

        Only untransformed, unsaved lm_head reads are deferred. Explicit tensor
        consumers still get full logits through read_value. Sixteen positions
        bound each vocabulary projection independently of forward batch size.
        """
        from causalab.neural.shared.metrics import compute_windowed_metric
        from causalab.protocol.schema import (
            METRIC_DOMAINS,
            READ_TARGET_METRIC_KINDS,
            metric_reads_vocabulary,
        )

        name = str(metric.of)
        target = (
            str(metric.fields["target"])
            if metric.kind in READ_TARGET_METRIC_KINDS
            else None
        )
        rows = self.rows_for_metrics()
        ids = (
            self.generated_ids(name)
            if METRIC_DOMAINS.get(str(metric.kind)) == "ids"
            else None
        )
        if ids is not None:
            return compute_windowed_metric(
                metric, [], rows, self.bundle.tokenizer, generated_ids=ids
            )
        if name not in self._deferred_heads or (
            target is not None and target not in self._deferred_heads
        ):
            return compute_windowed_metric(
                metric,
                self.windowed_value(name),
                rows,
                self.bundle.tokenizer,
                target_windows=self.windowed_value(target) if target else None,
                vocab_axis=metric_reads_vocabulary(self.doc, metric),
            )
        value = self._read_values[name]
        windows = (
            list(torch.split(value.flat, list(value.widths)))
            if isinstance(value, RaggedValue)
            else list(value)
        )
        target_windows = None
        if target is not None:
            target_value = self._read_values[target]
            target_windows = (
                list(torch.split(target_value.flat, list(target_value.widths)))
                if isinstance(target_value, RaggedValue)
                else list(target_value)
            )
            if [len(w) for w in windows] != [len(w) for w in target_windows]:
                raise ProtocolError(
                    "P2",
                    "continuation metric and target windows have different lengths",
                )
        out = []
        for index, (row, window) in enumerate(zip(rows, windows)):
            values = []
            for start in range(0, len(window), 16):
                chunk = window[start : start + 16]
                if chunk.shape[0]:
                    logits = self._project_deferred(name, chunk)
                    target_logits = (
                        self._project_deferred(
                            target, target_windows[index][start : start + 16]
                        )
                        if target is not None and target_windows is not None
                        else None
                    )
                    values.extend(
                        compute_windowed_metric(
                            metric,
                            [logits],
                            [row],
                            self.bundle.tokenizer,
                            target_windows=[target_logits]
                            if target_logits is not None
                            else None,
                            vocab_axis=True,
                        )[0]
                    )
                    del logits, target_logits
            out.append(values)
        return out

    def resolution(self, name: str) -> Resolution:
        """How one read resolved: ``Available`` (the ordinary case) or the
        ``Unavailable`` a scoped slice that selected nothing produced. Runs the
        read's group on first use, like :meth:`read_value`. The denominator
        key is :func:`~causalab.protocol.resolution.cell_key` over this point's
        coordinates — the same key the saved entry takes."""
        value = self.read_value(name)
        if name in self._unavailable:
            return self._unavailable[name]
        rows = sum(value.widths) if isinstance(value, RaggedValue) else value.shape[0]
        return available({"read": name, "rows": rows}, cell_key(name, self.coords))

    def row_resolutions(self, name: str) -> list[Unavailable | None]:
        """Per row of one read's input role, the ``Unavailable`` the row became
        when its address could not be aligned (§4.1), else ``None``.

        The row-level half of :meth:`resolution`: a read whose cell is
        unavailable because *some* rows failed to align still has rows that
        did, and a metric over it scores those and reports the rest as
        excluded measurements (§2.10 "Eligibility"). A cell unavailable for a
        reason that is not a row's — an ``expert:`` face the router sent no
        token — has no row-level record, and every entry is ``None``.
        """
        self.read_value(name)
        per_row = self._row_unavailable.get(name, {})
        rows = len(self.role_rows[str(self.doc.reads[name].input)])
        return [per_row.get(i) for i in range(rows)]

    def dense_rows(self, name: str, rows: Sequence[int]) -> torch.Tensor:
        """A read's value at the given rows, as the dense ``(len(rows), …)``
        tensor a metric reduces — the eligible-row form of :meth:`dense_value`.

        A read some of whose rows aligned on nothing is ragged (those rows
        have width zero), and :meth:`dense_value` rightly refuses to reduce
        it. Over the rows that *did* align it is not ragged at all: each has
        exactly the one position a metric reduces, so selecting them gives
        the same dense value the read would have had without the excluded
        rows. A row of any other width is still the ragged refusal.
        """
        value = self.read_value(name)
        if not isinstance(value, RaggedValue):
            index = torch.tensor(list(rows), dtype=torch.long, device=value.device)
            return value[index]
        widths = list(value.widths)
        if any(widths[i] != 1 for i in rows):
            raise ProtocolError(
                "P2",
                f"read {name!r} is ragged (unequal per-row position widths) — "
                "metrics reduce one aligned position per example",
            )
        offsets = [sum(widths[:i]) for i in rows]
        index = torch.tensor(offsets, dtype=torch.long, device=value.flat.device)
        return value.flat[index].unsqueeze(1)

    def check_write_widths(self) -> None:
        """Rule 19, checked **before any forward pass**.

        A ragged write used to surface from the landing path — i.e. on the
        accelerator, after the weights had loaded, with no rule number. On a
        35 B model that is minutes of wasted compute for a fact the encoded
        batch already knows, and it shaped a whole corpus: the refusal run had
        to end every request in a ``.`` token so negative indices aligned
        across rows.

        Only the tokenizer can say how wide a row is, which is why this cannot
        live in ``validate --data`` — that verb reads the dataset but holds no
        tokenizer, and giving it one would break the pure verbs' network- and
        torch-free contract. Encoding is the earliest point the question has an
        answer.

        What a ragged write meets here is its declared ``ragged`` policy
        (§2.8): ``refuse`` — and an absent field — is the refusal above;
        ``exact_length_buckets`` and ``padded_masked`` land every row at its
        own width instead (:func:`land_ragged`), and this check records what
        they will land under (:attr:`ragged_geometry`) for the run receipt.
        The policy is authored in the pure layer and *resolved* here, on the
        encoded batch — rule 19's encode-time boundary is unchanged.
        """
        if self._write_widths_checked:
            return
        self._write_widths_checked = True
        for model, im in self.doc.intervened_models.items():
            input_role = str(im.input)
            names = tuple(im.writes) if isinstance(im.writes, tuple) else ()
            for ename in names:
                write = self.doc.writes[ename]
                spec = self._spec(write.pos)
                if (
                    spec.all is None
                    and spec.variable is None
                    and spec.column is None
                    and not isinstance(spec, SpanSpec)
                ):
                    continue  # an index (scoped or not) is uniform by shape
                # a `variable` or `column` window is as wide as the row's
                # value tokenizes, a span — atomic or not — as wide as its
                # members make it on each row; a ragged one is rule 19's
                # business exactly as an `all` window is (§2.3)
                if spec.generated is not None:
                    continue  # rule 16 already refuses a generated write
                site = resolve_site(self.bundle, self.doc.sites[str(write.site)])
                if not site.shape.has_contract_form:
                    # This tap's last axis is positions, not features, so the
                    # landing path edits the whole tensor and never gathers —
                    # there are no per-row widths to be ragged. Mirroring that
                    # skip here is what keeps the pre-flight from refusing an
                    # `attention_scores` write, which is the point of the
                    # component.
                    continue
                batch = self._batch(input_role)
                per_row = self._positions(write.pos, batch, input_role)
                widths = {len(row) for row in per_row}
                if len(widths) == 1:
                    continue
                policy = write.ragged or "refuse"
                if policy == "refuse":
                    raise _ragged_write_error(ename, sorted(widths), model=model)
                # a declared policy lands every row at its own width (the
                # landing path); what it lands under is recorded here, before
                # any forward, for the receipt's `execution.ragged` (§8)
                self.ragged_geometry[(model, ename)] = ragged_geometry_of(
                    policy, per_row
                )

    def check_answer_forms(self) -> None:
        """The answer-form check, run **before any forward
        pass**: a metric that pins ``token_form: "bare"`` over a column whose
        values the table carries space-prefixed, under a tokenizer where the
        two forms are different tokens
        (:func:`~causalab.neural.shared.metrics.check_answer_forms`).

        Encode-time for the same reason rule 19 is: only the tokenizer can
        say whether ``"Saturday"`` and ``" Saturday"`` share a first piece,
        and the pure verbs hold none. Refused with reason ``alignment_missing``
        naming both surface forms, so a wrong-form metric is a refusal here
        rather than a flat 0.000 after the forward.
        """
        if self._answer_forms_checked:
            return
        self._answer_forms_checked = True
        for qname, metric in self.doc.metrics.items():
            check_answer_forms(
                self.bundle.tokenizer,
                metric,
                self.rows_for_metrics(),
                where=f"metric {qname!r}",
            )

    def check_scoring(self) -> ScoringCheck:
        """The table's recorded scoring identity against every ``match``
        ``mode`` this document declares, **before any forward pass** — the
        same comparison ``validate --data`` makes
        (:func:`causalab.causal.scoring.check_scoring`, §2.2, §2.10), run
        again here because a run may start from rows the pure verbs never saw.
        A ``prefix`` table under ``mode: exact`` is refused under rule 4,
        naming both modes and the derivation; an unrecorded table compares
        nothing and the receipt says so. Torch-free arithmetic over the rows,
        so this costs nothing a forward would have paid for.
        """
        if self.scoring_check is not None:
            return self.scoring_check
        rows = self.rows_for_metrics()
        try:
            self.scoring_check = check_scoring(
                rows, declared_modes(self.doc.metrics), where="data.base"
            )
        except ScoringMismatch as err:
            raise ValidationError(
                4, str(err), path=f"metrics.{err.metric}.mode"
            ) from err
        except ScoringError as err:
            raise ValidationError(4, f"data.base: {err}", path="data.base") from err
        return self.scoring_check

    def check_edit_groups(self) -> None:
        """Rule 27's pair half, checked **before any forward pass** (§2.2,
        §5 item 27): an ``atomic`` edit group a row declares
        (:mod:`causalab.causal.pairs`) is addressed whole or not at all by
        each intervened model — the positions of its writes on its input, and
        of the reads its writes take their operands from, on theirs. A
        constituent addressed without its siblings is refused naming the
        group and the missing siblings; a non-atomic group, a fully addressed
        atomic group and a table without the column all run.

        Encode-time for the reason rule 19 is: which tokens a char span
        covers is the tokenizer's to say, and the pure verbs hold none. The
        char → token reading is the one ``variable`` positions take
        (``encoding._chars_to_tokens``), so a declared span and a ``variable``
        anchor over the same characters name the same tokens.
        """
        if self._edit_groups_checked:
            return
        self._edit_groups_checked = True
        base_rows = self.role_rows.get("base", [])
        if not any(row.get(EDIT_GROUPS_COLUMN) is not None for row in base_rows):
            return
        groups_by_row: list[tuple[EditGroup, ...]] = []
        for index, row in enumerate(base_rows):
            try:
                groups_by_row.append(parse_edit_groups(row))
            except EditGroupError as err:
                raise ValidationError(
                    27, f"data.base row {index}: {err}", path="data.base"
                ) from err
        for mname, im in self.doc.intervened_models.items():
            names = tuple(im.writes) if isinstance(im.writes, tuple) else ()
            addressed: dict[str, list[set[int]]] = {}
            for ename in names:
                write = self.doc.writes[ename]
                self._address(addressed, write.pos, str(im.input))
                for rname in _operand_reads(write.do.payload, self.doc.reads):
                    read = self.doc.reads[rname]
                    self._address(addressed, read.pos, str(read.input), cell=rname)
            for role, per_row in addressed.items():
                side = "counterfactual" if role == "counterfactual" else "base"
                batch = self._batch(role)
                for index, groups in enumerate(groups_by_row):
                    if not any(group.atomic for group in groups):
                        continue
                    text = row_texts(base_rows[index])[side]
                    offsets = _pair_offsets(batch, index, text, role=role)
                    try:
                        check_component_wise(
                            groups,
                            offsets,
                            per_row[index],
                            side=side,
                            where=f"intervened_models.{mname} (input {role!r}), row {index}",
                            text=text,
                        )
                    except EditGroupError as err:
                        raise ValidationError(
                            27,
                            f"{err} (sec. 2.2 `edit_groups`)",
                            path=f"intervened_models.{mname}",
                        ) from err

    def _address(
        self,
        addressed: dict[str, list[set[int]]],
        pos: Any,
        role: str,
        *,
        cell: str | None = None,
    ) -> None:
        """Fold one position's resolved runs on ``role`` into ``addressed``
        (per row, the padded-frame indices). A generated position addresses
        the continuation, not the prompt the spans are declared over, and is
        skipped. ``cell`` names a read, whose unalignable rows become its
        ``unavailable`` cell as they would in the run; a write's propagate."""
        spec = self._spec(pos)
        if spec.generated is not None:
            return
        rows = self._positions(pos, self._batch(role), role, cell=cell)
        per_row = addressed.setdefault(role, [set() for _ in rows])
        for index, run in enumerate(rows):
            per_row[index].update(run)

    def dense_value(self, name: str) -> torch.Tensor:
        """A read value that must be a dense tensor (metric inputs): a
        ragged read has no per-example position alignment to reduce."""
        value = self.read_value(name)
        if isinstance(value, RaggedValue):
            raise ProtocolError(
                "P2",
                f"read {name!r} is ragged (unequal per-row position widths) — "
                "metrics reduce one aligned position per example",
            )
        return value

    def is_generated(self, name: str) -> bool:
        """Whether this read addresses the continuation frame (§2.3)."""
        return generated_budget(self.doc, self.doc.reads[name].pos) is not None

    def windowed_value(self, name: str) -> list[torch.Tensor]:
        """One read's value split per example: ``(positions_i, …)`` each.

        The metric surface for a continuation read. Unlike
        :meth:`dense_value` it welcomes ragged widths, because in the
        continuation frame they are the answer rather than a
        misalignment — a row that stopped early, or never said the value a
        ``variable`` anchor looks for, contributes an **empty** tensor.
        """
        value = self.read_value(name)
        if isinstance(value, RaggedValue):
            widths = list(value.widths)
            return list(torch.split(value.flat, widths)) if widths else []
        if value.dim() == 2:  # one position per row, already squeezed
            return [value[i].unsqueeze(0) for i in range(value.shape[0])]
        return [value[i] for i in range(value.shape[0])]

    def addressed_steps(self, name: str) -> list[list[int]]:
        """Per example, the decode steps one generate read covers."""
        if name not in self._read_steps:
            read = self.doc.reads[name]
            self._run_group(str(read.model), str(read.input))
        return self._read_steps[name]

    def generated_ids(self, name: str) -> list[list[int]]:
        """Per example, the token ids at a generate read's addressed steps.

        The ``ids`` metric domain (§2.10): these come from the decode
        itself, so a metric that only needs them obliges no vocabulary
        projection anywhere.
        """
        read = self.doc.reads[name]
        steps = self.addressed_steps(name)
        continuation = self._continuations[(str(read.model), str(read.input))]
        return [
            [int(continuation.token_ids[row, step]) for step in row_steps]
            for row, row_steps in enumerate(steps)
        ]

    def _preflight(self) -> None:
        """The document-level checks every run makes before its first
        forward, whichever door it came in by."""
        self.check_write_widths()
        self.check_answer_forms()
        self.check_scoring()
        self.check_edit_groups()

    def run_all(self) -> None:
        """Run every group the document implies (all reads materialize)."""
        self._preflight()
        for read in self.doc.reads.values():
            self._run_group(str(read.model), str(read.input))

    def stage(self, name: str) -> Stage:
        """The (shared) featurizer stage instance for one declared name."""
        if name not in self.stage_cache:
            build_recipe(
                self.stage_recipe(name),
                self.doc.featurizers,
                load_tensors=self.load_tensors,
                load_table=self.load_table,
                stage_cache=self.stage_cache,
                device=self.bundle.device,
                seed=self.seed,
                coords=self.coords,
                model_info=self.bundle.info,
            )
            # a budget pool's members are built together (§2.5 `pool`): a
            # member's mask is solved over the whole pool, so none may be
            # computed before every co-member exists
            link_budget_pools(self.doc.featurizers, self.stage_cache, self.stage)
        return self.stage_cache[name]

    def stage_recipe(self, name: str) -> StageRecipe:
        """Where this document uses featurizer ``name``, as the plain data a
        stage is built from (:class:`~causalab.neural.shared.featurizers.
        StageRecipe`) — what :meth:`stage` builds with, and what a fit's spec
        carries so the same stage can be built where no executor is."""
        width, site, entry = self._featurizer_input(name)
        # §2.5 `axis`: the window the entry addresses sizes a position gate;
        # `build_stack` decides whether it matters and refuses a `None` for a
        # position gate — the same line `_read_stack` hands it, so a gate
        # reached through either path gets one message. Rule 4 already held
        # every use of the gate to one fixed span, so the first entry's
        # window is the gate's
        return StageRecipe(
            name=name,
            width=width,
            site_shape=site.shape,
            site_component=site.component,
            position_width=span_length(self._spec(entry.pos)),
        )

    def _featurizer_input(
        self, name: str
    ) -> "tuple[int, ResolvedSite, ReadSpec | WriteSpec]":
        """The input of one declared featurizer: the width it is sized to —
        the site width of a chain that uses it, folded through the stages
        before it (§2.5 composition — a gate after a k=3 rotation is 3-wide)
        — the resolved site that chain starts at, whose shape a grouped gate
        derives its map from, and the read or write entry itself, whose
        window sizes a position gate (§2.5 ``axis``). One scan, the first
        entry that uses the name."""
        from causalab.neural.shared.featurizers import stage_output_width

        for entry in (*self.doc.reads.values(), *self.doc.writes.values()):
            ref = entry.featurizer
            chain = (
                (ref,)
                if isinstance(ref, str)
                else tuple(ref)
                if isinstance(ref, tuple)
                else ()
            )
            if name not in chain:
                continue
            site = resolve_site(self.bundle, self.doc.sites[str(entry.site)])
            running = (
                site.feature_slice.stop - site.feature_slice.start
                if site.feature_slice is not None
                else component_width(self.bundle.info, site.component, head=None)
            )
            for member in chain:
                if member == name:
                    return running, site, entry
                out = stage_output_width(self.doc.featurizers[member], running)
                if out is None:
                    raise ProtocolError(
                        "P2",
                        f"cannot size {name!r}: {member!r} before it in the "
                        "chain has no spec-derivable output width",
                    )
                running = out
        raise ProtocolError("P2", f"featurizer {name!r} is used by no read or write")

    def input_token_ids(self, input_role: str) -> list[list[int]]:
        """Exact unpadded inputs for output provenance, without retokenizing."""
        batch = self._batch(input_role)
        # one host copy of each tensor, then per-row slicing on the host
        ids, mask = batch.input_ids.cpu(), batch.attention_mask.cpu()
        return [tokens[real.bool()].tolist() for tokens, real in zip(ids, mask)]

    def rows_for_metrics(self) -> list[dict[str, Any]]:
        """Metric columns resolve against the base rows — the pairing
        anchor (§2.2: rows are paired; one base row + its counterfactuals form one
        example)."""
        return self.role_rows["base"]

    def reset_reads(self) -> None:
        """Drop cached read values and group state (training steps re-run
        forwards with updated featurizer parameters).

        The encoded batches and the shared interning store are untouched: the
        rows have not changed, and a constant group's raw capture is exactly
        what the next step should be served rather than re-run."""
        self._read_values.clear()
        self._deferred_heads.clear()
        self._read_routing.clear()
        self._groups_run.clear()
        self._write_widths_checked = False
        self.ragged_geometry.clear()
        self._continuations.clear()
        self._read_steps.clear()

    # ------------------------------------------------------------------ #
    # what an engine implements
    # ------------------------------------------------------------------ #

    def _run_group(self, model: str, input_role: str) -> None:
        """Run one (model, input role) forward group: land its writes and
        fill ``self._read_values`` for its reads."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # cross-point interning (§3) — the half an engine's _run_group consults
    # ------------------------------------------------------------------ #

    def _group_digest(self, model: str, input_role: str) -> str | None:
        """This group's plan digest, or ``None`` when the executor was built
        without a shared cache to key into."""
        if self.interning is None:
            return None
        return self.interning.digests.get((model, input_role))

    def _read_taps(
        self, model: str, input_role: str, reads: Iterable[tuple[str, ReadSpec]]
    ) -> "dict[str, ReadTap]":
        """The taps of one group's prompt-frame reads on this executor
        (:func:`~causalab.neural.shared.head.resolve_read_taps`): an
        ``lm_head`` read at named positions projects the head over its
        gathered rows, except where this executor differentiates through
        it — a grad-enabled executor reading a model a trained parameter can
        reach — which keeps the head as the model runs it, so the training
        gradient is the model's to the bit. A fit-constant group's read on
        the same executor carries no graph (the network is frozen) and
        projects."""
        differentiable = self.grad_enabled and model not in self.fit_constant_models
        return resolve_read_taps(
            self.bundle,
            self.doc,
            model,
            input_role,
            reads,
            differentiable=differentiable,
        )

    def _may_intern(self, model: str) -> bool:
        """Whether this executor may serve ``model``'s group from — or publish
        it to — the shared store at all.

        Outside a fit every group is fair game: a campaign point's captures
        are final. Inside one — a grad-enabled minibatch, or any executor whose
        passes are not counted (the grad-free eval pass) — only a model no
        trained parameter can reach qualifies (:attr:`fit_constant_models`).
        Gating on grad alone would be wrong in both directions: the eval pass
        runs grad-free yet must never be served the trained model's capture
        from an earlier step, and the source forward runs grad-enabled yet is
        exactly what the fit should stop re-running.
        """
        if self.interning is None:
            return False
        if self.grad_enabled or not self.interning.counted:
            return model in self.fit_constant_models
        return True

    def inner_interning(self, rows: tuple[int, ...] | str) -> Interning | None:
        """The handle a fit's inner executor over ``rows`` of this point's
        roles gets: the same campaign cache and this point's digests, keyed
        by the row slice, and not counted as forward groups. ``None`` when
        this point interns nothing.

        The digests are **not** narrowed to the fit-constant groups: which
        groups may be served from or published to the store is
        :meth:`_may_intern`'s decision, and the trained model's group — never
        served — still needs its digest to find the un-intervened prefix it
        resumes from (§4 "Resume")."""
        if self.interning is None:
            return None
        return Interning(
            digests=self.interning.digests,
            cache=self.interning.cache,
            rows=rows,
            counted=False,
        )

    def _capture_key(self, digest: str) -> CaptureKey:
        """Where this executor's captures of ``digest`` live in the store: the
        bare digest for the whole role, ``(digest, rows)`` for a slice."""
        assert self.interning is not None
        rows = self.interning.rows
        return digest if rows is None else (digest, rows)

    def _prefix_plan(self, digest: str | None) -> PrefixPlan | None:
        """The plan's resume arithmetic for this group (§4 "Resume"), or
        ``None`` when nothing about it may resume or store a prefix: no shared
        store, no digest, or a digest the campaign planned no prefix for.

        Deliberately not gated on :meth:`_may_intern`: the prefix below a
        group's first write is ``original``'s activations over these rows,
        fit-constant even when the group itself is what the fit trains."""
        if digest is None or self.interning is None:
            return None
        return self.interning.cache.prefix_plans.get(digest)

    def _prefix_key(self, plan: PrefixPlan, window: RowWindow, depth: int) -> PrefixKey:
        """Where the residual entering block ``depth`` of ``plan``'s prefix
        lives for this executor's rows and ``window``."""
        assert self.interning is not None
        return (
            plan.base_digest,
            self.interning.rows,
            (window.start, window.stop),
            depth,
        )

    def _interned(
        self, digest: str | None, keys: Iterable[TapKey]
    ) -> tuple[dict[TapKey, torch.Tensor], dict[TapKey, torch.Tensor]] | None:
        """This group's raw captures (and their routing tables) if an earlier
        pass already produced **every** address it taps under the same
        capture key, else ``None``.

        All-or-nothing on purpose: a partial hit would still have to run the
        forward for the addresses it missed, and the pass it runs captures the
        campaign's whole union anyway."""
        if digest is None or self.interning is None:
            return None
        key = self._capture_key(digest)
        captured = self.interning.cache.captured.get(key)
        if captured is None:
            return None
        wanted = set(keys)
        if not wanted or any(k not in captured for k in wanted):
            return None
        if not self.interning.counted:
            # a fit's inner pass the store saved — the number `inner_executed`
            # is measured against
            self.interning.cache.inner_served.append(digest)
        routing = self.interning.cache.routing.get(key, {})
        return (
            {k: captured[k] for k in wanted},
            {k: routing[k] for k in wanted if k in routing},
        )

    def _publish(
        self,
        digest: str | None,
        label: str,
        capture: Mapping[TapKey, torch.Tensor],
        routing: Mapping[TapKey, torch.Tensor],
    ) -> None:
        """Record that one forward group ran, and hand its raw captures to
        the passes that share its capture key.

        Called once per pass an engine actually executes, so
        ``len(cache.executed)`` is what the run paid against what
        :func:`~causalab.protocol.plan.interned_groups` says it owed. A fit's
        inner passes are tallied apart (``inner_executed``, keyed passes only)
        so that number never moves with the fit's step count.

        Captures are stored detached: a grad-enabled pass may have produced
        them, and what a later pass gathers and featurizes from the store must
        be a leaf — the trained featurizer applied *after* the capture is where
        that pass's graph begins."""
        if self.interning is None:
            return
        if not self.interning.counted:
            if digest is not None:
                self.interning.cache.inner_executed.append(digest)
        else:
            self.interning.cache.executed.append(digest or label)
        if digest is None:
            return
        # a whole-role digest no *other* pass keys into has no reader to keep
        # a capture for: storing it would only pin this pass's raw activations
        # (a swept patched group tapping lm_head: the whole vocabulary, every
        # row) until the request ends
        if self._tracks(digest) and self.interning.cache.owed[digest] <= 1:
            return
        key = self._capture_key(digest)
        # only what a tap actually filled: publishing an address whose module
        # never ran would hand a later point an empty capture instead of
        # letting it run the forward
        self.interning.cache.captured.setdefault(key, {}).update(
            {k: value.detach() for k, value in capture.items()}
        )
        self.interning.cache.routing.setdefault(key, {}).update(
            {k: value.detach() for k, value in routing.items()}
        )

    def _tracks(self, digest: str) -> bool:
        """Whether ``digest``'s whole-role captures are lifetime-tracked: this
        executor runs a counted, whole-role pass and the plan counted the
        digest's instances into ``owed``."""
        assert self.interning is not None
        return (
            self.interning.counted
            and self.interning.rows is None
            and digest in self.interning.cache.owed
        )

    def _settle(self, digest: str | None) -> None:
        """One whole-role pass over ``digest`` is done with its captures —
        run or served, the plan owes this digest one pass fewer. When no pass
        is owed any more the store drops the digest's captures and routing:
        the capture's lifetime is the span between its first sharer and its
        last, not the request.

        Called by an engine's ``_run_group`` once the pass has gathered its
        reads; a ``None`` digest (no interning, a decoding group, a group a
        fit trains through) settles nothing, as does a sliced pass. The
        pass's prefixes are settled at the same moment (:meth:`_settle_prefixes`)."""
        if digest is None or self.interning is None:
            return
        self._settle_prefixes(digest)
        if not self._tracks(digest):
            return
        owed = self.interning.cache.owed
        owed[digest] -= 1
        if owed[digest] <= 0:
            key = self._capture_key(digest)
            self.interning.cache.captured.pop(key, None)
            self.interning.cache.routing.pop(key, None)

    def _settle_prefixes(self, digest: str) -> None:
        """One whole-role pass over ``digest`` is done resuming (§4 "Resume"):
        every prefix depth it could have started from — the wanted depths up
        to its plan's ``resume_at`` — is owed one pass fewer, and a depth no
        remaining instance can reach is dropped for every rows coordinate and
        window at once. A sliced pass settles nothing: a fit's inner passes
        re-read their prefixes every step, and the point's own pass, which
        follows the fit, settles on their behalf."""
        assert self.interning is not None
        if not self.interning.counted or self.interning.rows is not None:
            return
        cache = self.interning.cache
        plan = cache.prefix_plans.get(digest)
        if plan is None:
            return
        for depth in cache.wanted_prefix_depths.get(plan.base_digest, ()):
            pair = (plan.base_digest, depth)
            if depth > plan.resume_at or pair not in cache.prefix_owed:
                continue
            cache.prefix_owed[pair] -= 1
            if cache.prefix_owed[pair] <= 0:
                for key in [k for k in cache.prefixes if (k[0], k[3]) == pair]:
                    del cache.prefixes[key]

    # ------------------------------------------------------------------ #
    # shared plumbing
    # ------------------------------------------------------------------ #

    def frame(self, role: str) -> EncodedBatch:
        """The encoded frame of one input role — every row of the role, left
        padded to one width. What a fit's minibatch executors are built as
        selections of (``EncodedBatch.select``, §4 "Cohorts")."""
        return self._batch(role)

    def _batch(self, role: str) -> EncodedBatch:
        if role in self._batches and role not in self._batches_checked:
            # a handed-in frame (a minibatch's selection of its point's frame)
            # must be these rows, row for row: the row count always, and under
            # the plain frame the texts themselves — a chat frame's texts are
            # the rendered template, so there only the count is comparable
            handed = self._batches[role]
            rows = self.role_rows[role]
            expected = tuple(
                str(select_field(row, self.role_fields[role])) for row in rows
            )
            if len(handed.texts) != len(rows) or (
                self.doc.segments is None and handed.texts != expected
            ):
                raise AssertionError(
                    f"the frame handed in for input {role!r} encodes "
                    f"{len(handed.texts)} rows that are not this executor's "
                    f"{len(rows)} — a selection must be built from the "
                    "same rows it is handed to"
                )
            self._batches_checked.add(role)
        if role not in self._batches:
            rows = self.role_rows[role]
            field = self.role_fields[role]
            from causalab.neural.shared.prepared import encode_prepared, encoding_field

            prepared_field = encoding_field(field)
            prepared_column = prepared_field.split("[", 1)[0]
            prepared = [prepared_column in row for row in rows]
            if any(prepared):
                if not all(prepared) or self.doc.segments is not None:
                    raise ProtocolError(
                        "P2",
                        "prepared inputs must cover every row and already own their frame; "
                        "do not combine them with segments",
                    )
                self._batches[role] = encode_prepared(
                    self.bundle.tokenizer,
                    [str(select_field(row, field)) for row in rows],
                    [select_field(row, prepared_field) for row in rows],
                    device=self.bundle.device,
                )
            elif self.doc.segments is not None:
                # the document declares its frame and segments (§2.2.1): the
                # chat frame renders through the tokenizer's own template and
                # sets the real prefix; a plain frame with declared column
                # segments locates them on the text as encoded below
                self._batches[role] = encode_framed(
                    self.bundle.tokenizer,
                    rows,
                    field,
                    self.doc.segments,
                    device=self.bundle.device,
                )
            else:
                texts = [str(select_field(row, field)) for row in rows]
                self._batches[role] = encode(
                    self.bundle.tokenizer, texts, device=self.bundle.device
                )
        return self._batches[role]

    def location_ledger(self) -> LocationLedger:
        """The location ledger (§6, ``protocol/ledger.py``): one row per
        (example, edit group, constituent, side, token index, token id,
        decoded token) for every prompt-frame position every read and write
        of this point resolves, on every input it resolves it on — the same
        indices :meth:`_positions` hands the gathers, recorded once.

        Built on first use and cached; the caller (``execution.py``) asks for
        it only when the document saves a ``location_ledger`` entry (§2.12),
        so a document that does not opt in never pays for it. The edit group
        is the forward group — ``<model> on <input>`` (``plan.ForwardGroup``);
        the constituent is the position's name (or its inline path), with
        ``[k]`` per member of a non-atomic set; the token index counts from
        the row's first real token, so it is a fact of the row and not of
        the batch's padding. Continuation reads are not in the ledger: their
        steps are a result of the decode, recorded per read as ``steps``.
        """
        if self._ledger is not None:
            return self._ledger
        ledger = LocationLedger()
        tokenizer = self.bundle.tokenizer
        #: each role's ids on the host, copied once for every position the
        #: ledger records on that role rather than once per record
        host_ids: dict[str, torch.Tensor] = {}

        def record(
            pos: Any, role: str, group: str, label: str, *, cell: str | None
        ) -> None:
            spec = self._spec(pos)
            if spec.generated is not None:
                return
            batch = self._batch(role)
            parts = constituents(spec)
            if len(parts) == 1:
                addressed = [(label, self._positions(pos, batch, role, cell=cell))]
            else:
                addressed = [
                    (f"{label}[{k}]", self._positions(part, batch, role, cell=cell))
                    for k, part in enumerate(parts)
                ]
            ids = host_ids.get(role)
            if ids is None:
                ids = host_ids[role] = batch.input_ids.cpu()
            for constituent, per_row in addressed:
                for example, indices in enumerate(per_row):
                    first = batch.first_real(example)
                    for index in indices:
                        token_id = int(ids[example, index])
                        ledger.add(
                            LedgerRow(
                                example=example,
                                edit_group=group,
                                constituent=constituent,
                                side=role,
                                token_index=index - first,
                                token_id=token_id,
                                decoded_token=str(
                                    tokenizer.convert_ids_to_tokens(token_id)
                                ),
                            )
                        )

        def has_positions(site_name: str) -> bool:
            site = resolve_site(self.bundle, self.doc.sites[site_name])
            return site.shape.has_contract_form

        for rname, read in self.doc.reads.items():
            if not has_positions(str(read.site)):
                continue  # the tap's last axis is not positions; nothing gathers
            record(
                read.pos,
                str(read.input),
                f"{read.model} on {read.input}",
                read.pos if isinstance(read.pos, str) else f"reads.{rname}.pos",
                cell=rname,
            )
        for mname, im in self.doc.intervened_models.items():
            names = tuple(im.writes) if isinstance(im.writes, tuple) else ()
            for ename in names:
                write = self.doc.writes[ename]
                if not has_positions(str(write.site)):
                    continue
                record(
                    write.pos,
                    str(im.input),
                    f"{mname} on {im.input}",
                    write.pos if isinstance(write.pos, str) else f"writes.{ename}.pos",
                    cell=None,
                )
        self._ledger = ledger
        return ledger

    def _row_windows(self, total: int) -> list[RowWindow]:
        """The microbatches one group over ``total`` rows runs as: ceil(total /
        batch_rows) windows of at most ``batch_rows`` rows, in row order — or
        the one whole window when no bound is set."""
        if self.batch_rows is None or self.batch_rows >= total:
            return [RowWindow(0, total, total)]
        return [
            RowWindow(start, min(start + self.batch_rows, total), total)
            for start in range(0, total, self.batch_rows)
        ]

    def _spec(self, pos: Any) -> PositionSpec:
        spec = self.doc.positions[pos] if isinstance(pos, str) else pos
        if not isinstance(spec, PositionSpec):
            raise ProtocolError("P2", f"unresolved position {pos!r}")
        return spec

    def _positions(
        self,
        pos: Any,
        batch: EncodedBatch,
        input_role: str,
        *,
        cell: str | None = None,
    ) -> list[list[int]]:
        """Every row's positions for one spec on one input.

        ``cell`` names the *read* whose rows these are. A row the address
        cannot be aligned on — its ``variable`` / ``column`` value occurs zero
        or several times — then becomes that read's ``unavailable`` cell
        (spec §4.1, reason ``alignment_missing`` / ``alignment_ambiguous``,
        the detail naming the value, its count and the row) and contributes
        no positions, so the row is an excluded measurement in the
        denominator rather than a refusal of the run. Without ``cell`` (a
        write, the width pre-flight) the typed refusal propagates before any
        forward: a write that skipped a row would report a number for an
        intervention that did not happen.

        A spec with a declared ``alignment`` is checked against the pair's
        observed cardinality on first use (:meth:`_check_declared_alignment`).
        """
        spec = pos
        if isinstance(spec, str):
            spec = self.doc.positions[spec]
        if not isinstance(spec, PositionSpec):
            raise ProtocolError("P2", f"unresolved position {pos!r}")
        rows = self.role_rows[input_role]
        field = self.role_fields[input_role]
        out: list[list[int]] = []
        problems: list[tuple[int, UnalignableError]] = []
        for i in range(len(rows)):
            try:
                out.append(
                    resolve_position(spec, batch, i, dataset_row=rows[i], field=field)
                )
            except UnalignableError as err:
                if cell is None:
                    raise
                problems.append((i, err))
                out.append([])
        if problems and cell is not None:
            key = cell_key(cell, self.coords)
            # the row-level record (§2.10 "Eligibility"): each excluded row
            # under its own reason, so a metric over this read scores the
            # rows that aligned and reports these as excluded measurements
            per_row = self._row_unavailable.setdefault(cell, {})
            for i, problem in problems:
                row_cell = unalignable(problem.cardinality, problem.message, key)
                assert row_cell is not None
                per_row.setdefault(i, row_cell)
            first = problems[0][1]
            excluded = unalignable(
                first.cardinality,
                f"read {cell!r} at position {pos if isinstance(pos, str) else spec!r} "
                f"on input {input_role!r}: {len(problems)} of {len(rows)} rows "
                "could not be aligned and contribute no positions — "
                + "; ".join(problem.message for _, problem in problems),
                key,
            )
            assert excluded is not None
            self._unavailable[cell] = excluded
        if spec.alignment is not None:
            self._check_declared_alignment(pos, spec, input_role)
        return out

    def _check_declared_alignment(
        self, pos: Any, spec: PositionSpec, input_role: str
    ) -> None:
        """A declared ``alignment`` (§2.3) against the cardinality the pair
        actually has, once per (position, input role) on this executor.

        Per row, the address's candidate runs on ``input_role`` are paired
        with its candidate runs on every other input role the document reads
        — rows are paired by index (§2.2) — and
        :func:`~causalab.protocol.alignment.alignment_of` names the observed
        cardinality; :func:`~causalab.protocol.alignment.check_declared`
        refuses a contradiction, naming both. A single-role document is
        checked against its own candidates. Cached per batch: cardinality is
        cheap to make sound and expensive to recompute per point, so it is
        computed once here, as forward-group
        interning is.
        """
        key = (pos if isinstance(pos, str) else repr(spec), input_role)
        if key in self._alignment_checked:
            return
        self._alignment_checked.add(key)
        rows = self.role_rows[input_role]
        field = self.role_fields[input_role]
        batch = self._batch(input_role)
        others = [role for role in self.role_rows if role != input_role]
        label = f"position {pos!r}" if isinstance(pos, str) else f"position {spec!r}"
        # one classification per address the spec *is*: an ordinary position
        # or an atomic span is one, a non-atomic set is each of its
        # constituents (§2.3 "composable groups", `spans.constituents`)
        for i in range(len(rows)):
            mine_by_part = constituent_candidate_runs(
                spec, batch, i, dataset_row=rows[i], field=field
            )
            if not others:
                for k, mine in enumerate(mine_by_part):
                    check_declared(
                        spec.alignment,
                        alignment_of(mine),
                        where=f"{label}{_part(k, mine_by_part)} on input "
                        f"{input_role!r}, row {i},",
                    )
            for other in others:
                other_rows = self.role_rows[other]
                if i >= len(other_rows):
                    continue
                theirs_by_part = constituent_candidate_runs(
                    spec,
                    self._batch(other),
                    i,
                    dataset_row=other_rows[i],
                    field=self.role_fields[other],
                )
                for k, (mine, theirs) in enumerate(zip(mine_by_part, theirs_by_part)):
                    check_declared(
                        spec.alignment,
                        alignment_of(mine, theirs),
                        where=f"{label}{_part(k, mine_by_part)} across inputs "
                        f"{input_role!r} and {other!r}, row {i},",
                    )

    @staticmethod
    def _gather(
        tensor: torch.Tensor, per_row: list[list[int]], what: str
    ) -> "torch.Tensor | RaggedValue":
        return gather_rows(tensor, per_row)

    def _read_stack(
        self, read: ReadSpec | WriteSpec, site: ResolvedSite
    ) -> FeaturizerStack:
        # Lazily: `build_stack` ignores the width entirely when no featurizer is
        # referenced (it returns an Identity stack), and some components have no
        # width to give — `input_ids` carries integer ids on a position axis and
        # `expert_idx` a routing table, so both refuse rather than invent one
        # (§5.4). Asking for the width up front made an unfeaturized read of
        # those impossible, which is not what the refusal is for: it exists to
        # reject a *featurizer*, not a read.
        if read.featurizer is None:
            width = 0
        elif site.feature_slice is not None:
            width = site.feature_slice.stop - site.feature_slice.start
        else:
            # `head=site.head` matters only for a *derived* component, which
            # carries no feature_slice: a head there narrows the value's width
            # without narrowing the captured tensor's.
            width = component_width(self.bundle.info, site.component, head=site.head)
        stack = build_stack(
            read.featurizer,
            dict(self.doc.featurizers),
            width=width,
            load_tensors=self.load_tensors,
            load_table=self.load_table,
            stage_cache=self.stage_cache,
            device=self.bundle.device,
            seed=self.seed,
            coords=self.coords,
            site_shape=site.shape,
            site_component=site.component,
            model_info=self.bundle.info,
            # §2.5 `axis`: a position gate in this chain is sized by the
            # entry's fixed span; `None` for every other chain
            position_width=span_length(self._spec(read.pos)),
        )
        link_budget_pools(self.doc.featurizers, self.stage_cache, self.stage)
        return stack

    def _finalize_read(
        self,
        rname: str,
        read: ReadSpec,
        site: ResolvedSite,
        raw: torch.Tensor,
        batch: EncodedBatch,
        input_role: str,
        *,
        per_row: list[list[int]] | None = None,
        project: Callable[[torch.Tensor], torch.Tensor] | None = None,
        expert_idx: torch.Tensor | None = None,
        to_cpu: bool | None = None,
        pregathered: bool = False,
    ) -> "torch.Tensor | RaggedValue":
        """One read's value: gather at its positions, then featurize.

        ``per_row`` overrides position resolution — the continuation frame
        resolves to decode steps, which the caller has already worked out
        against the decode. ``project`` runs on the gathered slice before
        anything else, which is how an ``lm_head`` continuation read is
        served from kept ``ln_final`` activations: the vocabulary projection
        happens at the addressed positions and nowhere else.
        ``to_cpu`` defaults to ``not self.device_reads``: an eval executor and
        a CUDA evaluation capture keep their read values on the device, and
        their scorer copies out only what a metric selects.

        ``pregathered`` says the engine reduced inside its forward: ``raw``
        (and ``expert_idx``) arrive already gathered at ``per_row`` — a dense
        ``(rows, width, …)`` tensor or a :class:`RaggedValue` — and already
        projected and derived, which need the model's own modules. What is
        left is the part that needs only the document: the head's slice, the
        featurizer stack, ``dims``. An engine whose forward runs in another
        process (the nnterp engine on NDIF) downloads the slice this way
        rather than the contract tensor.
        """
        if to_cpu is None:
            to_cpu = not self.device_reads
        if not site.shape.has_contract_form:
            return whole_native_tensor(rname, read, raw, site)
        if per_row is None:
            per_row = self._positions(read.pos, batch, input_role, cell=rname)
        if site.expert is not None:
            return self._expert_selected(
                rname, read, site, raw, expert_idx, per_row, pregathered=pregathered
            )
        gathered = raw if pregathered else self._gather(raw, per_row, f"read {rname!r}")
        if project is not None:
            if isinstance(gathered, RaggedValue):
                gathered = RaggedValue(
                    flat=project(gathered.flat), widths=gathered.widths
                )
            else:
                gathered = project(gathered)
        ragged = isinstance(gathered, RaggedValue)
        value = gathered.flat if isinstance(gathered, RaggedValue) else gathered
        if site.shape.state_axes:
            return self._state_read(rname, read, site, value, gathered)
        if site.derivation is not None and not pregathered:
            # After the gather, deliberately: the value is `heads` times wider
            # than the tensor it comes from, so deriving it before the gather
            # would cost `seq · H · hidden` where this costs
            # `n_positions · H · hidden`.
            value = _derive(site, value, rname)
        routing = None
        if expert_idx is not None:
            # the routing table at the same rows and positions as the value —
            # what an expert-keyed gate keys its parameters by, and what a
            # later write through one aligns this read's slots to its own by
            idx_gathered = (
                expert_idx
                if pregathered
                else self._gather(expert_idx, per_row, f"read {rname!r}")
            )
            if isinstance(idx_gathered, RaggedValue):
                routing = idx_gathered.flat
            else:
                routing = idx_gathered
                self._read_routing[rname] = routing.detach()
        value = read_features(
            value,
            site,
            self._read_stack(read, site),
            read.dims,
            routing=routing,
            grad_enabled=self.grad_enabled,
        )
        if not self.grad_enabled:
            value = value.detach()
            if to_cpu:
                value = value.cpu()
        if ragged:
            assert isinstance(gathered, RaggedValue)
            return RaggedValue(flat=value, widths=gathered.widths)
        return value

    def _expert_selected(
        self,
        rname: str,
        read: ReadSpec,
        site: ResolvedSite,
        raw: torch.Tensor,
        expert_idx: torch.Tensor | None,
        per_row: list[list[int]],
        *,
        pregathered: bool = False,
    ) -> RaggedValue:
        """The ragged face of the routed interior: the (position, slot) pairs
        the router sent to ``site.expert``, as flat ``(selected, d)`` rows plus
        per-example widths.

        An expert no token chose returns width-0 rows — **a data fact, not an
        error** (there is no per-expert hook to have not fired; the router
        simply sent it nothing at these positions). When that is true of
        *every* addressed position, the read is recorded as an
        ``unavailable`` cell with reason ``empty_selector`` (spec §4.1): the
        document was legal, the selector selected nothing here, and the cell
        belongs in the result and in the denominator rather than in a
        refusal. Partial emptiness stays data — the per-row widths say which
        rows the expert served.

        ``featurizer`` and ``dims`` are refused here rather than resized: the
        document sized them against the token-major form (``top_k · d``), and
        these rows are ``d``-wide — silently applying either would index a
        different space than the author named.
        """
        if expert_idx is None:
            raise ProtocolError(
                "P2",
                f"read {rname!r} selects expert {site.expert}, but the engine "
                "captured no routing table alongside the tap — an executor bug, "
                "not a document error",
            )
        if read.featurizer is not None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} featurizes the 'expert: {site.expert}' face of "
                f"{site.component!r}, whose rows are d_expert-wide while the "
                "component (and any featurizer sized against it) is top_k·d "
                "wide. Featurize the token-major form — drop 'expert' — or read "
                "this face raw.",
            )
        if isinstance(read.dims, tuple):
            raise ProtocolError(
                "P4",
                f"read {rname!r} slices 'dims' on the 'expert: {site.expert}' "
                f"face of {site.component!r}: 'dims' indexes the token-major "
                "top_k·d axis, and these rows are d-wide. Drop 'expert' or "
                "drop 'dims'.",
            )
        if pregathered:
            gathered, idx_gathered = raw, expert_idx
        else:
            gathered = self._gather(raw, per_row, f"read {rname!r}")
            idx_gathered = self._gather(expert_idx, per_row, f"read {rname!r}")
        if isinstance(gathered, RaggedValue):
            assert isinstance(idx_gathered, RaggedValue)
            flat_value, pos_widths = gathered.flat, gathered.widths
            flat_idx = idx_gathered.flat
        else:
            assert isinstance(idx_gathered, torch.Tensor)
            rows, n_pos = gathered.shape[0], gathered.shape[1]
            flat_value = gathered.reshape(rows * n_pos, gathered.shape[-1])
            flat_idx = idx_gathered.reshape(rows * n_pos, idx_gathered.shape[-1])
            pos_widths = (n_pos,) * rows
        top_k = flat_idx.shape[-1]
        per_slot = flat_value.shape[-1] // top_k
        mask = flat_idx == site.expert  # (positions, top_k)
        selected = flat_value.reshape(-1, top_k, per_slot)[mask]
        # hits per (example, position) row — read to the host once, then
        # summed per example there rather than one device read per row
        counts = mask.sum(dim=-1).tolist()
        widths: list[int] = []
        offset = 0
        for width in pos_widths:
            widths.append(sum(counts[offset : offset + width]))
            offset += width
        if sum(widths) == 0:
            self._unavailable[rname] = unavailable(
                "empty_selector",
                f"read {rname!r} selects the 'expert: {site.expert}' face of "
                f"{site.component!r} at layer {site.layer}, and the router sent "
                f"expert {site.expert} no token at the addressed positions "
                f"({len(pos_widths)} rows, {sum(pos_widths)} positions) — a "
                "fact of this batch's routing, not of the document",
                cell_key(rname, self.coords),
            )
        if not self.grad_enabled:
            selected = selected.detach().cpu()
        return RaggedValue(flat=selected, widths=tuple(widths))

    def _state_read(
        self,
        rname: str,
        read: ReadSpec,
        site: ResolvedSite,
        value: torch.Tensor,
        gathered: "torch.Tensor | RaggedValue",
    ) -> "torch.Tensor | RaggedValue":
        """The tail of a read whose trailing axes form a state matrix.

        The tensor keeps its native layout — ``(batch, steps, heads, d_k,
        d_v)`` after the position gather — because there is no feature vector
        to flatten to. ``head:`` selects on the head axis directly;
        ``featurizer`` and ``dims`` are refused off the declared axes (the same
        generated refusals the attention pattern gets, with the position gather
        kept, which is what distinguishes the two shapes).
        """
        what = f"{site.component!r} ({site.shape.describe()})"
        if read.featurizer is not None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} featurizes {what}: {site.shape.refusal('it')}",
            )
        if isinstance(read.dims, tuple):
            raise ProtocolError(
                "P4",
                f"read {rname!r} slices 'dims' on {what}: that would select "
                "d_v columns of a matrix as though they were features.",
            )
        if site.head is not None:
            # dim 0 of a ragged flat is the gathered rows; dense keeps
            # (batch, steps) in front — the head axis is right after either way
            value = (
                value[:, site.head]
                if isinstance(gathered, RaggedValue)
                else value[:, :, site.head]
            )
        if not self.grad_enabled:
            value = value.detach().cpu()
        if isinstance(gathered, RaggedValue):
            return RaggedValue(flat=value, widths=gathered.widths)
        return value

    def _state_step_writer(
        self,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
        rows: RowWindow,
        tally: FireTally | None = None,
    ) -> Callable[[int, torch.Tensor], torch.Tensor]:
        """Per-step application of ``delta_state`` writes, for the stepwise
        substitution.

        A state edit must feed forward — step ``t``'s replacement is what step
        ``t+1`` decays and writes into — so the shared whole-tensor write math
        cannot land it. Instead each addressed step applies the same
        class-ordered mechanisms to that step's matrix, flattened to one
        row: ``v_pre`` is ``S_t`` as ``(1, 1, heads·d_k·d_v)``, and a tensor
        operand (a ``delta_state`` read) is sliced to the same (row, step)
        before the mechanism sees it, so ``swap`` interchanges step-for-step.

        ``dims`` is refused (a matrix has no feature columns); ``featurizer``
        refuses through the width lookup, as every state read does.

        ``rows`` is the window this forward covers: ``state`` holds its rows
        only, so the window offsets them back to role rows when a tensor
        operand is sliced.

        ``tally`` is this forward's fire count (§4 "Fires"): a state write
        declares one firing per distinct step its rows address and fires
        once per step it edits, so a kernel path that never reached the
        substitution — or skipped a step — is refused after the forward.
        """

        def class_rank(entry: tuple[str, WriteSpec, ResolvedSite]) -> int:
            do = entry[1].do
            if str(do.mechanism) == "renormalize":
                return 2
            return 1 if is_additive(do) else 0

        prepared: list[tuple[str, WriteSpec, ResolvedSite, list[list[int]]]] = []
        for ename, write, site in sorted(entries, key=class_rank):
            if isinstance(write.dims, tuple):
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} slices 'dims' on {site.component!r} "
                    f"({site.shape.describe()}): that would select d_v columns "
                    "of a matrix as though they were features.",
                )
            per_row = self._positions(write.pos, batch, input_role)[rows.slice]
            prepared.append((ename, write, site, per_row))
            if tally is not None:
                tally.declare((ename,), len({p for row in per_row for p in row}))

        def edit_state(step: int, state: torch.Tensor) -> torch.Tensor:
            edited = state
            for ename, write, site, per_row in prepared:
                if tally is not None and any(step in row for row in per_row):
                    tally.fired((ename,), step=step)
                for row, positions in enumerate(per_row):
                    if step not in positions:
                        continue
                    j = positions.index(step)
                    if edited is state:
                        edited = state.clone()
                    v_pre = edited[row : row + 1].reshape(1, 1, -1)
                    v_new = self._written_value(
                        ename,
                        write,
                        site,
                        v_pre,
                        lookup=self._state_operand(
                            ename, rows.start + row, j, len(positions), v_pre
                        ),
                    )
                    edited[row] = v_new.reshape(edited.shape[1:]).to(edited.dtype)
            return edited

        return edit_state

    def _state_operand(
        self, ename: str, row: int, step_index: int, n_steps: int, v_pre: torch.Tensor
    ) -> Callable[[Any], "torch.Tensor | float"]:
        """Operand lookup for one (row, addressed-step) state application: a
        tensor operand must be a state read — ``(batch, steps, heads, d_k,
        d_v)`` — covering **exactly the write's addressed steps** (the standard
        write path's elementwise rule, stated rather than broadcast), and is
        sliced to this row and step so the mechanism math sees two aligned
        single-step rows."""

        def lookup(value: Any) -> "torch.Tensor | float":
            operand = self._operand_lookup(value)
            if not isinstance(operand, torch.Tensor):
                return operand
            if operand.dim() != 5:
                raise ProtocolError(
                    "P2",
                    f"write {ename!r} hands {value!r} to a 'delta_state' "
                    f"write, but its shape is {tuple(operand.shape)} — a state "
                    "operand is a 'delta_state' read, (batch, steps, heads, "
                    "d_k, d_v), applied step for step",
                )
            if operand.shape[1] != n_steps:
                raise ProtocolError(
                    "P2",
                    f"write {ename!r}: operand {value!r} covers "
                    f"{operand.shape[1]} steps, but the write addresses "
                    f"{n_steps} — the j-th operand step lands on the j-th "
                    "addressed step, so both sides must cover the same steps "
                    "(read the operand at the write's own positions)",
                )
            sliced = operand[row : row + 1, step_index : step_index + 1]
            return sliced.reshape(1, 1, -1).to(v_pre.device)

        return lookup

    # ------------------------------------------------------------------ #
    # writes (the math; landing them is the engine's job)
    # ------------------------------------------------------------------ #

    def _operand_lookup(
        self,
        value: Any,
        *,
        rows: RowWindow | None = None,
        ragged: Sequence[int] | None = None,
    ) -> torch.Tensor | float:
        """Resolve one operand: a literal, a read's value, a featurizer slot,
        or a ``params`` entry.

        Only a **read** is row-indexed (dim 0 is the example), so it alone is
        sliced to ``rows`` — the window (or width bucket) of the forward that
        consumes it. A featurizer slot or a params tensor is one value for
        every row and a scalar has no rows; both pass through whole.

        ``ragged`` is the consuming write's per-row position widths over
        ``rows`` when that write lands under a ``ragged`` policy (§5 rule 19).
        A :class:`RaggedValue` operand is then re-nested row by row to those
        widths — refused when any row's widths disagree
        (:func:`_operand_width_error`) — and padded to the widest with zeros,
        which the landing never writes back; a **dense** read (one width for
        every row) is held to the same rule (:func:`check_dense_operand`):
        its width must be each row's own, or one — a uniform operand as wide
        as the widest row would otherwise satisfy the padded frame's broadcast
        and land truncated into every narrower row. Without ``ragged`` (a
        write under ``refuse``, the absent-field behaviour), a ragged operand
        is rule 19's refusal as before (:func:`_ragged_operand_error`)."""
        if not isinstance(value, str):
            return float(value)
        if value in self.doc.reads:
            return read_operand(
                value,
                self._read_values[value],
                device=self.bundle.device,
                rows=rows,
                ragged=ragged,
                # asked only under a landing policy, the one place it is used
                positioned=ragged is not None and self._positioned(value),
            )
        resolved = self._artifact_operand(value)
        if resolved is None:
            raise ProtocolError("P2", f"operand {value!r} did not resolve at run time")
        return resolved

    def _artifact_operand(self, value: str) -> torch.Tensor | None:
        """An operand the document's artifacts hold — a featurizer slot or a
        ``params`` entry — or ``None`` when ``value`` names neither."""
        if "." in value:
            fname, slot = value.split(".", 1)
            if fname in self.doc.featurizers:
                params = self.stage(fname).slot_params()
                if slot in params:
                    return params[slot]
        if value in self.doc.params:
            spec = self.doc.params[value]
            if isinstance(spec.file_path, str):
                want, implicit = entry_selection(spec.entry, self.coords, value)
                slot = selector_slot(spec.entry, "value")
                what = f"params entry {value!r} ({spec.file_path})"
                point = self.load_tensors(spec.file_path).point(
                    slot, want, what=what, implicit=implicit
                )
                return point.tensor(slot)
            raise NotImplementedError(
                f"trainable free params ({value!r}) arrive with the train loop"
            )
        return None

    def _operand_routing(
        self, value: Any, rows: RowWindow | None = None
    ) -> torch.Tensor | None:
        """The routing table a tensor operand was read beside — ``None`` when
        the operand is not a read at a routed-interior site (a literal, a
        params tensor, a read anywhere else) and so carries no expert ids."""
        if not isinstance(value, str) or value not in self._read_routing:
            return None
        routing = self._read_routing[value].to(self.bundle.device)
        # sliced to the consuming forward's window — or width bucket — like
        # the operand itself (§8 microbatching): the two are joined row by row
        return routing if rows is None or rows.whole else routing[rows.index]

    def _resolve_write_addresses(
        self, write_names: tuple[str, ...]
    ) -> dict[Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]]:
        """Resolve and policy-check this group's writes, grouped by address.

        Addresses are keyed by :func:`tap_key`, so two components that share a
        module but mean different tensors (a different tuple element, or a
        different shape) get their own application rather than one
        overwriting the other's view."""
        by_address: dict[
            Any, tuple[ResolvedSite, list[tuple[str, WriteSpec, ResolvedSite]]]
        ] = {}
        for ename in write_names:
            write = self.doc.writes[ename]
            site = resolve_site(self.bundle, self.doc.sites[str(write.site)])
            # The write policy is the capability row's (read-only, or a closed
            # mechanism set) — the same function `validate` applies at load,
            # here for a document that arrived unvalidated. One check, one
            # text: the read-only, routing-table and attention-pattern
            # refusals used to be three tables at two sites.
            refusal = write_policy_refusal(
                ename, site.component, str(write.do.mechanism)
            )
            if refusal is not None:
                raise ProtocolError("P4", refusal, reason="unsupported_mechanism")
            key = tap_key(site)
            if key not in by_address:
                by_address[key] = (site, [])
            by_address[key][1].append((ename, write, site))
        return by_address

    def _positioned(self, value: str) -> bool:
        """Whether read ``value`` has a position axis at dim 1 — what a
        landing policy's width check holds a dense operand to. A whole-tensor
        read (a tap with no contract form) or a state read has none, and
        pairs into a positioned write by broadcast alone."""
        site = resolve_site(
            self.bundle, self.doc.sites[str(self.doc.reads[value].site)]
        )
        return site.shape.has_contract_form and not site.shape.state_axes

    def _write_services(
        self,
        positions_of: "Callable[[str, WriteSpec], list[list[int]]] | None" = None,
    ) -> WriteServices:
        """The write math's services, bound to this executor: its position
        resolution, operand lookup, featurizer stacks and routing tables, and
        the pending mismatch record the expert alignment's counts land in."""

        def unresolved(ename: str, _write: WriteSpec) -> "list[list[int]]":
            raise AssertionError(f"write {ename!r}: no position frame was bound")

        return WriteServices(
            positions_of=positions_of or unresolved,
            lookup=self._operand_lookup,
            stack_of=lambda _ename, write, site: self._read_stack(write, site),
            routing_of=self._operand_routing,
            reads=self.doc.reads,
            code=self.doc.code,
            mismatches=self._routing_mismatch_pending,
        )

    def _apply_writes_to_contract(
        self,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        input_role: str,
        batch: EncodedBatch,
        tensor: torch.Tensor,
        *,
        per_row: list[list[int]] | None = None,
        rows: RowWindow | None = None,
        routing: torch.Tensor | None = None,
    ) -> None:
        """Apply every write at one address to the contract-shaped
        ``tensor``, in place (:func:`apply_writes_to_contract`), with this
        executor's services: positions resolved on ``batch``, operands,
        stacks and routing tables looked up on ``self``."""
        services = self._write_services(
            positions_of=lambda _ename, write: self._positions(
                write.pos, batch, input_role
            )
        )
        apply_writes_to_contract(
            entries, tensor, services, per_row=per_row, rows=rows, routing=routing
        )

    def _written_value(
        self,
        ename: str,
        write: WriteSpec,
        site: ResolvedSite,
        v_pre: torch.Tensor,
        *,
        lookup: "Callable[[Any], torch.Tensor | float] | None" = None,
        rows: RowWindow | None = None,
        routing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """featurize → class-ordered do → inverse
        (:func:`written_value`), with this executor's services."""
        return written_value(
            ename,
            write,
            site,
            v_pre,
            self._write_services(),
            lookup=lookup,
            rows=rows,
            routing=routing,
        )

    @property
    def routing_mismatch(self) -> dict[tuple[str, int, int], tuple[int, int]]:
        """Per (write, layer, example), how many of the example's written
        slots found no source slot holding their expert, of how many slots —
        what ``routing_mismatch.json`` records (§2.5 ``expert_neuron``). The
        writes leave their counts on the device (:func:`align_by_expert`);
        reading this brings every pending count over in one host read."""
        pending, self._routing_mismatch_pending = self._routing_mismatch_pending, {}
        if pending:
            device = next(iter(pending.values()))[0].device
            counts = iter(
                torch.cat(
                    [missing.to(device).reshape(-1) for missing, _ in pending.values()]
                ).tolist()
            )
            for (ename, layer, examples), (_missing, per_example) in pending.items():
                for example in examples:
                    self._routing_mismatch[(ename, layer, example)] = (
                        int(next(counts)),
                        per_example,
                    )
        return self._routing_mismatch


def refuse_unstackable(name: str, site: ResolvedSite) -> None:
    """Refuse a continuation read whose steps do not stack into a frame.

    📐 A decode step attends over the whole KV cache, so a tensor indexed by the
    positions being attended *to* is ``prompt + step`` long at step ``step``
    while the query axis stays 1. The accumulating sink concatenates steps on
    dim 1, which for an ordinary tap is the position axis, so such a tap either
    raises a bare torch size error (measured: "Expected size 9 but got size 10")
    or, for a single-step budget, silently returns one step shaped like a frame.
    Neither is a continuation read.

    Both conditions are read off the declared axes rather than a component list:

    * **two position axes** — the attention pattern and the scores. There is no
      non-arbitrary answer to which of them the steps stack along;
    * **one position axis, over the keys** — ``attention_key``. Its own length
      is what grows, so consecutive steps are different lengths.

    ``attention_query`` and ``attention_z`` are query-axis-shaped and accumulate
    correctly, which is why this is a property of the axes and not of "anything
    inside the attention function".
    """
    shape = site.shape
    if shape.has_contract_form and not any(
        axis.kind == "position" and axis.name == "key" for axis in shape.axes
    ):
        return
    why = (
        "it has two position axes, so there is no single axis the decode steps "
        "stack along"
        if not shape.has_contract_form
        else "its position axis runs over the positions being attended to, "
        "which under a KV cache is the whole prefix and grows by one per step"
    )
    raise ProtocolError(
        "P4",
        f"read {name!r} reads {site.component!r} in the generated frame, whose "
        f"shape is {shape.describe()}: {why}, so the steps do not stack into "
        "one tensor. Read it in the prompt frame.",
    )
