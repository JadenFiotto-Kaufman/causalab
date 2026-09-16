"""The position frame: one tokenization per batch, positions born padded.

The engine's ``PositionFrame`` (spec §2.3, §8) is the padded batch the
model actually runs, plus what position resolution needs to address it:
pad side (always left here), per-row content offsets, offset mappings for
char→token resolution, per-row prefix lengths (0 in the plain frame; the
real chat-prefix token count under a document's ``segments.frame: chat``,
computed by ``framing.encode_framed`` from where the tokenizer's own template
put the user turn), and per-row segment locations (§2.2.1).

Position rules implemented against this frame (spec §2.3, §6.1):

* ``{"index": n}`` — ``n < 0`` counts from the end of the row's real
  tokens; ``n ≥ 0`` counts from the row's content start (past padding and
  any chat prefix).
* ``{"variable": "x"}`` — all tokens overlapping the char span of the
  row's value for ``x``. The value comes from the dataset row: for a text
  column ``<col>`` the sibling ``<col>_variables`` mapping (aligned
  per-element for list columns), else a plain column named ``x``. The
  value must occur exactly once in the row's text — zero occurrences is the
  ``absent`` cardinality and several the ``ambiguous`` one (§2.3), refused
  with the matching reason code (``alignment_missing`` /
  ``alignment_ambiguous``) rather than addressing the wrong tokens; a read's
  executor records such a row as an ``unavailable`` cell instead (§4.1).
* ``{"column": "c"}`` — the same token run, from the row's top-level
  column ``c`` only (never the ``<col>_variables`` sibling). The column is
  a property of the *row*, so it resolves to the same string whichever
  role reads it; that is what makes it the spelling for values a task
  computes per row (§2.3).
* ``{"span": [a, b]}`` — the content-frame window ``[a, b)``.
* ``{"all": true}`` — every content token of the row: past the left
  padding and past any chat prefix, through the last real token. Rows of
  different lengths make this ragged, which reads carry natively and the
  v1 write path does not (see the executor's write refusal).
* ``scope`` — the index/span interpreted inside the anchor's token run;
  ``relative_to`` — an index offset from the run (``+1`` = first token
  after it, ``-1`` = last token before it; ``0`` is refused). The anchor is
  ``{"variable": …}``, ``{"column": …}`` or ``{"segment": …}`` — a declared
  segment (§2.2.1), located by the frame that encoded the batch and resolved
  here through the same offset mapping, with the same ``absent`` /
  ``ambiguous`` refusals a variable has.
* a **span** (``protocol/spans.py``: ``segment``, ``indices``, ``union``,
  ``intersection``, ``before`` / ``after`` / ``between``, ``atomic``) — the
  algebra is torch-free and pure over this row's frame; members and anchors
  resolve back through :func:`resolve_position`.

Every resolved index is bounds-checked in the padded frame — a stale or
impossible position must fail here as a legible error, never reach a
gather (the stale-index failure class the old resolver guarded the same way).

**The continuation frame.** A position carrying ``generated`` resolves
against a :class:`Continuation` instead: the greedy decode's steps, indexed
from 0, one per generated token. Step indices are not padded-frame indices
and the two never mix — a decode *step* is the unit here, and the same
anchors mean what they say inside it (``{"index": -1}`` is the last real
generated token, ``{"all": true}`` is every one, ``{"variable": "x"}`` is
the tokens where the model *said* the row's value for ``x``).

The ``variable`` anchor differs from its prompt-side twin in two ways, both
because the continuation is a result rather than an input: it takes the
**first** occurrence instead of demanding exactly one (a repetitive
generation is a normal outcome, not an authoring error), and **zero**
occurrences yield zero positions instead of refusing — whether the model
says the thing is usually the experiment, so it has to be a value the run
reports, not an exception that ends it.

Rows end where they end: the frame stops at a row's first EOS, so widths
differ and continuation reads are ragged. A window that reaches past a
row's end **clips** rather than refusing, and a row that generated nothing
contributes no positions at all. That is deliberate and it is where this
frame differs from the prompt: how far a row generates is a *result*, so
refusing on it would make a document fail on data rather than on authoring
(the prompt frame keeps its strict bounds check, where an out-of-range
index really is an authoring error).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.alignment import alignment_of, refuse_unalignable
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import PositionSpec, concrete_int, concrete_str
from causalab.protocol.spans import SpanSpec, constituents, resolve_span

__all__ = [
    "Continuation",
    "EncodedBatch",
    "candidate_runs",
    "constituent_candidate_runs",
    "continuation_frame",
    "encode",
    "first_real_indices",
    "resolve_position",
    "resolve_steps",
]


@dataclasses.dataclass(frozen=True)
class EncodedBatch:
    """One left-padded batch plus its position frame."""

    texts: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offset_mapping: tuple[tuple[tuple[int, int], ...], ...]
    prefix_lengths: tuple[int, ...]  # chat-prefix token counts; 0 = plain text
    #: Per row, each declared segment's **candidate** char spans in
    #: ``texts[row]`` (§2.2.1): none is ``absent``, several ``ambiguous``,
    #: exactly one is the segment. Empty for a document with no ``segments``
    #: section — the plain-text frame, byte-identical to before the field.
    segments: tuple[Mapping[str, tuple[tuple[int, int], ...]], ...] = ()
    #: Per row, the padded index of its first real token — the ``argmax`` of
    #: the mask row — read off the device **once** for the whole batch, so
    #: :meth:`first_real`, :meth:`content_start` and every position
    #: resolution built on them are host arithmetic. Resolved per row per
    #: read and per write at every layer, the per-call round trip used to be
    #: most of the cohort forward's synchronizations. Derived from
    #: ``attention_mask`` when left empty; :meth:`select` and the cohort's
    #: frame concatenation pass theirs through, since a row's index does not
    #: change when it travels. ``dataclasses.replace`` copies this field like
    #: any other, so a caller replacing ``attention_mask`` through it must
    #: pass ``first_reals=()`` explicitly to have it re-derived — or, better,
    #: build the new frame as a row selection (``graph_cohort.slotted_frame``
    #: is the worked example). On the CPU, where the check costs no
    #: synchronization, the constructor refuses a cache the mask disagrees
    #: with.
    first_reals: tuple[int, ...] = dataclasses.field(
        default=(), compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.first_reals:
            object.__setattr__(
                self, "first_reals", first_real_indices(self.attention_mask)
            )
            return
        rows = int(self.attention_mask.shape[0])
        if len(self.first_reals) != rows:
            raise ValueError(
                f"first_reals carries {len(self.first_reals)} entries for a "
                f"{rows}-row attention mask"
            )
        if self.attention_mask.device.type == "cpu" and (
            self.first_reals != first_real_indices(self.attention_mask)
        ):
            raise ValueError(
                "first_reals disagrees with attention_mask — a frame whose mask "
                "was replaced must leave the cache empty to be re-derived"
            )

    @property
    def padded_len(self) -> int:
        return int(self.input_ids.shape[1])

    def first_real(self, row: int) -> int:
        """First real token of ``row`` in the padded frame — past the left
        padding, any chat prefix included."""
        return self.first_reals[row]

    def content_start(self, row: int) -> int:
        """First real token of ``row`` in the padded frame, past any prefix."""
        return self.first_real(row) + self.prefix_lengths[row]

    def position_ids(self) -> torch.Tensor:
        """Left-pad position ids: ``cumsum(mask) - 1``, clamped at 0 — the
        plain-forward convention (RoPE is shift-blind, absolute embeddings
        like GPT-2's ``wpe`` are not, so this must always be passed)."""
        return (self.attention_mask.cumsum(dim=1) - 1).clamp(min=0)

    def select(self, indices: Sequence[int]) -> "EncodedBatch":
        """The rows ``indices`` of this batch, in that order, **in this frame**:
        the same padded width, every per-row field sliced in step.

        A fit's minibatch is a selection of its point's frame rather than a
        fresh encode of its rows (spec §4, "Cohorts"): minibatches of several
        points then share one frame and concatenate into one forward, and a
        row's position indices — resolved against the frame — hold whichever
        selection it travels in."""
        if not indices:
            raise ValueError("a selection names at least one row")
        rows = list(indices)
        index = torch.tensor(rows, dtype=torch.long, device=self.input_ids.device)
        return EncodedBatch(
            texts=tuple(self.texts[i] for i in rows),
            input_ids=self.input_ids.index_select(0, index),
            attention_mask=self.attention_mask.index_select(0, index),
            offset_mapping=tuple(self.offset_mapping[i] for i in rows),
            prefix_lengths=tuple(self.prefix_lengths[i] for i in rows),
            segments=tuple(self.segments[i] for i in rows) if self.segments else (),
            first_reals=tuple(self.first_reals[i] for i in rows),
        )


def first_real_indices(attention_mask: torch.Tensor) -> tuple[int, ...]:
    """Per row of a left-padded mask, the index of its first real token: the
    ``argmax`` of the row (its first maximal entry; a row with no real token
    reads 0, as ``argmax`` of zeros does). One reduction over the batch and
    one host read, however many rows."""
    return tuple(int(i) for i in attention_mask.int().argmax(dim=1).tolist())


@dataclasses.dataclass(frozen=True)
class Continuation:
    """One batch's greedy continuation: the frame ``generated`` addresses.

    ``token_ids`` is ``(batch, steps)`` as decoded — every row runs the same
    number of steps, because a batched decode has no way not to — and
    ``widths`` says how much of each row is real: the count before its first
    EOS, or every step for a row that never emitted one. Positions resolve
    against ``widths``, so a row that stopped early simply contributes fewer
    of them.

    ``texts`` and ``offsets`` describe the same tokens as characters (per
    row: the decoded continuation, and each token's ``[start, end)`` span
    inside it), which is what a ``variable`` anchor searches. They come from
    incremental detokenization rather than a tokenizer's offset mapping:
    re-encoding ``prompt + continuation`` is not the same token sequence the
    decode produced (merges cross the boundary), so the spans have to be
    built as the tokens arrive.
    """

    token_ids: torch.Tensor
    widths: tuple[int, ...]
    texts: tuple[str, ...] = ()
    offsets: tuple[tuple[tuple[int, int], ...], ...] = ()

    @property
    def steps(self) -> int:
        """How many steps the decode ran — the same for every row."""
        return int(self.token_ids.shape[1])

    def real_ids(self, row: int) -> list[int]:
        """``row``'s generated ids up to its first EOS."""
        return [int(t) for t in self.token_ids[row, : self.widths[row]]]


def encode(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: str = "cpu",
    add_special_tokens: bool = True,
    segments: Sequence[Mapping[str, tuple[tuple[int, int], ...]]] = (),
) -> EncodedBatch:
    """Tokenize one batch with the engine's single convention: left
    padding, special tokens as the tokenizer defines them, offset mapping
    kept for char→token position resolution.

    ``prefix_lengths`` is ``0`` for every row **here**: this is the plain-text
    frame, and every document without a ``segments`` section takes it. The
    chat frame (§2.2.1, ``framing.encode_framed``) renders each row through
    the tokenizer's own chat template, encodes the rendered text with
    ``add_special_tokens=False`` (the template owns its specials) and sets
    the real prefix length from where the user turn was located. A dataset
    that bakes a *rendered* template into its ``input`` column under the plain
    frame still works — but this call then adds special tokens as the
    tokenizer defines them, so a rendered template that already opens with
    BOS gets a second one. A double BOS is a wrong number, not a crash: every
    position shifts by one and nothing says so. :func:`refuse_double_bos`
    catches it here, which is the only place that can see both halves.

    ``segments`` is the per-row segment location table the frame computed
    (:attr:`EncodedBatch.segments`); the plain frame passes none.
    """
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
    )
    # the BOS check and the first-real cache read the tokenizer's host tensors
    # before they move, so an encode onto an accelerator makes no round trip
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    refuse_empty_rows(texts, attention_mask)
    refuse_double_bos(tokenizer, input_ids, attention_mask)
    return EncodedBatch(
        texts=tuple(texts),
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        offset_mapping=tuple(
            tuple((int(a), int(b)) for a, b in row)
            for row in enc["offset_mapping"].tolist()
        ),
        prefix_lengths=tuple(0 for _ in texts),
        segments=tuple(dict(row) for row in segments),
        first_reals=first_real_indices(attention_mask),
    )


def refuse_empty_rows(texts: Sequence[str], attention_mask: torch.Tensor) -> None:
    """Refuse a batch with a row that encodes to no token at all.

    Every position of such a row would address padding, and the engine's
    frame arithmetic assumes each row has a first real token: its cached
    index is 0 for an empty row as for a full one, so the two would share a
    mask signature. An empty text (a tokenizer adding no special token) is a
    data error, named by row rather than run."""
    for row, count in enumerate(attention_mask.sum(dim=1).tolist()):
        if count == 0:
            raise ProtocolError(
                "P2",
                f"row {row} ({texts[row]!r}) encodes to no token: a frame's row "
                "has at least one real token, or every position in it would "
                "address padding",
            )


def refuse_double_bos(
    tokenizer: Any,
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> None:
    """Refuse a batch whose rows start with the BOS token twice.

    Read off the encoded batch rather than by re-encoding: the ids are already
    computed, and "two BOS at the start of the content" is the exact condition
    — it needs no assumption about whether *this* tokenizer prepends one, or
    about which string spells it.

    Stripping instead of refusing was considered and rejected: the text is the
    document's data, its content digest is part of the canonical form (§7), and
    silently editing it would make the digest describe bytes that never ran.
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None or input_ids.shape[1] < 2:
        return
    # one host read of the ids and one reduction over the mask, then row
    # arithmetic — not two reads per row from wherever the batch lives
    ids = input_ids.tolist()
    for row, start in enumerate(first_real_indices(attention_mask)):
        if start + 1 >= input_ids.shape[1]:
            continue
        if ids[row][start] == bos_id == ids[row][start + 1]:
            bos = getattr(tokenizer, "bos_token", None) or f"id {bos_id}"
            raise ProtocolError(
                "P2",
                f"row {row} begins with {bos!r} twice: the text already carries "
                "a BOS — a rendered chat template, most likely — and the "
                "tokenizer added another. Every position in the row is then "
                "off by one and no error would be raised. Remove the leading "
                f"{bos!r} from the dataset's text; v1 has no chat field, so "
                "the rendered template is the data",
            )


_LIST_FIELD = re.compile(r"^([A-Za-z0-9_]+)\[(\d+)\]$")


def select_field(row: Mapping[str, Any], field: str) -> Any:
    """Apply a data-role ``field`` selector (§2.2): a column name, with
    ``[j]`` indexing list-valued columns."""
    match = _LIST_FIELD.match(field)
    if match is None:
        if field not in row:
            raise ProtocolError(
                "P2", f"row has no column {field!r} (has {sorted(row)})"
            )
        return row[field]
    column, index = match.group(1), int(match.group(2))
    values = row.get(column)
    if not isinstance(values, list) or index >= len(values):
        raise ProtocolError("P2", f"column {column!r} has no element [{index}]")
    return values[index]


def variable_value(row: Mapping[str, Any], field: str, variable: str) -> str:
    """The row's value for a prompt variable, for the text selected by
    ``field`` (module docstring: ``<col>_variables`` sibling first, plain
    column fallback)."""
    match = _LIST_FIELD.match(field)
    column = match.group(1) if match else field
    sibling = row.get(f"{column}_variables")
    if match and isinstance(sibling, list):
        index = int(match.group(2))
        if index < len(sibling) and isinstance(sibling[index], Mapping):
            sibling = sibling[index]
        else:
            sibling = None
    if isinstance(sibling, Mapping) and variable in sibling:
        return str(sibling[variable])
    if variable in row:
        return str(row[variable])
    raise ProtocolError(
        "P2",
        f"no value for prompt variable {variable!r}: neither {column}_variables "
        f"nor a {variable!r} column exists in the dataset row",
    )


def column_value(row: Mapping[str, Any], column: str) -> str:
    """The row's value for a ``column`` position (§2.3) — a top-level column
    only, never the per-role ``<field>_variables`` sibling, so the same
    reference resolves to the same string whichever role reads it."""
    if column not in row:
        raise ProtocolError(
            "P2",
            f"position column {column!r} is not a column of the dataset row "
            f"(has {sorted(row)})",
        )
    value = row[column]
    if not isinstance(value, str):
        raise ProtocolError(
            "P2",
            f"position column {column!r} holds {type(value).__name__}, not a "
            "string — v1 column positions resolve a substring of the row's "
            "text (§2.3)",
        )
    return value


def _variable_token_runs(
    batch: EncodedBatch, row: int, value: str
) -> tuple[list[int], ...]:
    """The padded-frame token runs covering **each** occurrence of ``value``
    in the row's text — the *candidate* runs one address has here, via the
    offset mapping ((0, 0) entries are specials/padding and never match).

    How many candidates there are is the address's cardinality on this input
    (:func:`~causalab.protocol.alignment.alignment_of`, §2.3): none is
    ``absent``, several is ``ambiguous``, exactly one is the run. An
    occurrence that overlaps no token is kept as an empty run, so it too
    reads as ``absent`` rather than as a second candidate.
    """
    text = batch.texts[row]
    return tuple(
        _chars_to_tokens(batch, row, match.start(), match.end())
        for match in re.finditer(re.escape(value), text)
    )


def _chars_to_tokens(batch: EncodedBatch, row: int, lo: int, hi: int) -> list[int]:
    """The padded-frame tokens overlapping the char span ``[lo, hi)`` of the
    row's text, via the offset mapping ((0, 0) entries are padding / specials
    the text does not spell and never match)."""
    return [
        idx
        for idx, (a, b) in enumerate(batch.offset_mapping[row])
        if not (a == 0 and b == 0) and a < hi and b > lo
    ]


def _segment_token_runs(
    batch: EncodedBatch, row: int, name: str
) -> tuple[list[int], ...]:
    """The candidate token runs of a declared segment in this row — one per
    char span the frame located it at (§2.2.1). The frame that encoded the
    batch located every declared segment; a batch with no location table is a
    plain-frame batch under a document that never declared one."""
    if row >= len(batch.segments) or name not in batch.segments[row]:
        raise ProtocolError(
            "P2",
            f"segment {name!r} was not located on this batch — the document "
            "declares no segments section, or the frame that encoded it never "
            "declared this name (§2.2.1)",
        )
    return tuple(
        _chars_to_tokens(batch, row, lo, hi) for lo, hi in batch.segments[row][name]
    )


def _segment_run(batch: EncodedBatch, row: int, name: str) -> list[int]:
    """The one run a declared segment has in this row, or the typed refusal —
    ``absent`` is ``alignment_missing``, ``ambiguous`` is
    ``alignment_ambiguous`` — through the same path a ``variable`` takes."""
    runs = _segment_token_runs(batch, row, name)
    observed = alignment_of(runs)
    refuse_unalignable(
        observed,
        f"segment {name!r} occurs {len(runs)} time(s) in the rendered text of "
        f"row {row} ({batch.texts[row]!r}) — a segment anchor needs exactly one "
        f"occurrence, and this one is {observed!r} here",
    )
    return runs[0]


def _unique_run(batch: EncodedBatch, row: int, value: str, what: str) -> list[int]:
    """The one run ``value`` has in this row, or the typed refusal for none
    or several — ``absent`` is ``alignment_missing``, ``ambiguous`` is
    ``alignment_ambiguous`` (§2.3, §2.4). The executor turns that refusal
    into an ``unavailable`` cell for a read (§4.1) and lets it stand for a
    write, which cannot skip a row and still report a number."""
    runs = _variable_token_runs(batch, row, value)
    observed = alignment_of(runs)
    refuse_unalignable(
        observed,
        f"{what} value {value!r} occurs {len(runs)} times in {batch.texts[row]!r} "
        f"(row {row}) — position resolution needs exactly one occurrence, and "
        f"this address is {observed!r} here",
    )
    return runs[0]


def _row_value(
    dataset_row: Mapping[str, Any] | None,
    field: str | None,
    name: str,
    *,
    from_column: bool,
) -> str:
    """The row's string for an anchor or an anchor-free reference — a
    top-level column (``column``) or a per-role prompt variable
    (``variable``), §2.3."""
    if dataset_row is None:
        raise ProtocolError("P2", "variable/column positions need a dataset row")
    if from_column:
        return column_value(dataset_row, name)
    if field is None:
        raise ProtocolError("P2", "variable positions need a dataset row")
    return variable_value(dataset_row, field, name)


def candidate_runs(
    spec: PositionSpec,
    batch: EncodedBatch,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
) -> tuple[list[int], ...]:
    """The candidate runs one prompt-frame spec has in one row — what
    :func:`~causalab.protocol.alignment.alignment_of` classifies when a
    declared ``alignment`` is checked against the pair (§2.3).

    A ``variable`` / ``column`` address has one candidate per occurrence of
    its value; an anchored ``index`` / ``span`` inherits its anchor's
    candidates when the anchor is not unique (the derived address is exactly
    as ambiguous as the anchor); everything else has the one run
    :func:`resolve_position` returns. A ``generated`` spec has no candidates
    here: the continuation is a result, not one of the pair's inputs.
    """
    if spec.generated is not None:
        raise ProtocolError(
            "P2",
            "a generated position has no pair alignment — the continuation is a "
            "result, not one of the pair's inputs (§2.3)",
        )
    if isinstance(spec, SpanSpec):
        # A whole-segment span has one candidate per occurrence of the segment;
        # every other span — atomic or not — is the one run its algebra
        # resolves to (its constituents are classified through
        # :func:`constituent_candidate_runs`).
        if spec.segment is not None:
            return _segment_token_runs(batch, row, spec.segment)
        return (
            resolve_position(spec, batch, row, dataset_row=dataset_row, field=field),
        )
    if spec.variable is not None:
        value = _row_value(dataset_row, field, str(spec.variable), from_column=False)
        return _variable_token_runs(batch, row, value)
    if spec.column is not None:
        value = _row_value(dataset_row, field, str(spec.column), from_column=True)
        return _variable_token_runs(batch, row, value)
    if spec.scope is not None or spec.relative_to is not None:
        anchor_name = str(spec.scope or spec.relative_to)
        if spec.anchor_source == "segment":
            anchors = _segment_token_runs(batch, row, anchor_name)
        else:
            anchor = _row_value(
                dataset_row,
                field,
                anchor_name,
                from_column=spec.anchor_source == "column",
            )
            anchors = _variable_token_runs(batch, row, anchor)
        if len(anchors) != 1:
            return anchors
    return (resolve_position(spec, batch, row, dataset_row=dataset_row, field=field),)


def constituent_candidate_runs(
    spec: PositionSpec,
    batch: EncodedBatch,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
) -> list[tuple[list[int], ...]]:
    """The candidate runs of each address a spec is classified as (§2.3):
    one entry for an ordinary position or an ``atomic`` span, one per
    constituent of a non-atomic set (``spans.constituents``) — the
    "composable groups" half: the same two tokens are one joint
    ``one_to_one`` address when atomic and two single-token addresses when
    not, and a declared ``alignment`` is checked against each."""
    return [
        candidate_runs(part, batch, row, dataset_row=dataset_row, field=field)
        for part in constituents(spec)
    ]


def _generated_variable_run(
    continuation: Continuation, row: int, value: str
) -> list[int]:
    """Step indices covering the **first** occurrence of ``value`` in the
    row's generated text, or ``[]`` when the model never said it.

    Char spans come from the decode's incremental detokenization
    (:class:`Continuation`), so a match that starts mid-piece still lands on
    the steps that produced it — the sentencepiece case a post-hoc
    ``offset_mapping`` cannot resolve.
    """
    if row >= len(continuation.texts):
        return []
    text = continuation.texts[row]
    start = text.find(value)
    if start < 0:
        return []
    lo, hi = start, start + len(value)
    width = continuation.widths[row]
    return [
        step
        for step, (a, b) in enumerate(continuation.offsets[row][:width])
        if a < hi and b > lo
    ]


def resolve_steps(
    spec: PositionSpec,
    continuation: Continuation,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
) -> list[int]:
    """Resolve one ``generated`` spec for one row into **decode-step** indices.

    Indices are 0-based into the decode, bounded by the row's real width —
    see the module docstring on why a window past a row's end clips and a
    row that generated nothing yields nothing.
    """
    width = continuation.widths[row]
    if width == 0:
        return []
    if spec.all is not None:
        return list(range(width))
    if spec.index is not None:
        n = concrete_int(spec.index, "position index")
        step = width + n if n < 0 else n
        return [step] if 0 <= step < width else []
    if spec.span is not None:
        span = spec.span
        if not isinstance(span, tuple) or len(span) != 2:
            raise ProtocolError("P2", f"span is not concrete: {span!r}")
        a, b = (int(v) for v in span)
        return list(range(min(a, width), min(b, width)))
    if spec.variable is not None:
        if dataset_row is None or field is None:
            raise ProtocolError(
                "P2",
                "a generated 'variable' position needs its dataset row — the "
                "value the model may have said comes from the table",
            )
        variable = concrete_str(spec.variable, "position variable")
        value = variable_value(dataset_row, field, variable)
        return _generated_variable_run(continuation, row, value)
    raise ProtocolError(
        "P2",
        f"anchor {spec!r} has no continuation-frame resolution — v1 addresses "
        "generated tokens by index, span, variable or all",
    )


def resolve_position(
    spec: PositionSpec,
    batch: EncodedBatch,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
    continuation: Continuation | None = None,
) -> list[int]:
    """Resolve one position spec for one row into padded-frame indices, or
    into decode-step indices when the spec selects the continuation frame."""
    if spec.generated is not None:
        if continuation is None:
            raise ProtocolError(
                "P2",
                "a generated position needs the decode's continuation — the "
                "frame it addresses does not exist until the model has run",
            )
        return resolve_steps(
            spec, continuation, row, dataset_row=dataset_row, field=field
        )
    padded = batch.padded_len
    start = batch.content_start(row)

    def check(indices: list[int]) -> list[int]:
        bad = [
            i for i in indices if not start - batch.prefix_lengths[row] <= i < padded
        ]
        if bad:
            raise ProtocolError(
                "P2",
                f"resolved position(s) {bad} out of bounds for row {row} "
                f"(content [{start}, {padded}) in the padded frame) — refusing "
                "rather than addressing the wrong token",
            )
        return indices

    if isinstance(spec, SpanSpec):
        # the span algebra (protocol/spans.py) is torch-free and pure over this
        # row's frame; members and anchors come back through this resolver
        return check(
            resolve_span(
                spec,
                frame=(batch.first_real(row), start, padded),
                resolve=lambda member: resolve_position(
                    member, batch, row, dataset_row=dataset_row, field=field
                ),
                segment_run=lambda name: _segment_run(batch, row, name),
                where=f"row {row}",
            )
        )

    anchor_run: list[int] | None = None
    if spec.scope is not None or spec.relative_to is not None:
        anchor_name = str(spec.scope or spec.relative_to)
        if spec.anchor_source == "segment":
            anchor_run = _segment_run(batch, row, anchor_name)
        else:
            anchor_value = _row_value(
                dataset_row,
                field,
                anchor_name,
                from_column=spec.anchor_source == "column",
            )
            anchor_run = _unique_run(
                batch, row, anchor_value, f"anchor {anchor_name!r}"
            )

    if spec.all is not None:
        # content_start is already past the pad and any chat prefix; left
        # padding right-aligns content, so the row runs to the padded end
        return check(list(range(start, padded)))

    if spec.variable is not None:
        value = _row_value(dataset_row, field, str(spec.variable), from_column=False)
        return check(_unique_run(batch, row, value, "prompt variable"))

    if spec.column is not None:
        value = _row_value(dataset_row, field, str(spec.column), from_column=True)
        return check(_unique_run(batch, row, value, "position column"))

    if spec.index is not None:
        n = concrete_int(spec.index, "position index")
        if spec.relative_to is not None:
            assert anchor_run is not None
            if n == 0:
                raise ProtocolError(
                    "P2", "relative_to index 0 is ambiguous — use scope"
                )
            target = anchor_run[-1] + n if n > 0 else anchor_run[0] + n
            return check([target])
        if spec.scope is not None:
            assert anchor_run is not None
            if not -len(anchor_run) <= n < len(anchor_run):
                raise ProtocolError(
                    "P2",
                    f"index {n} outside the {len(anchor_run)}-token variable window",
                )
            return check([anchor_run[n]])
        if n < 0:
            return check([padded + n])
        return check([start + n])

    assert spec.span is not None
    span = spec.span
    if not isinstance(span, tuple) or len(span) != 2:
        raise ProtocolError("P2", f"span is not concrete: {span!r}")
    a, b = (int(v) for v in span)
    if spec.scope is not None:
        assert anchor_run is not None
        window = anchor_run[a:b]
        if not window:
            raise ProtocolError(
                "P2", f"span [{a}, {b}) is empty inside the variable window"
            )
        return check(window)
    if a < 0 or b <= a:
        raise ProtocolError("P2", f"span [{a}, {b}) is not a forward window")
    return check(list(range(start + a, start + b)))


def continuation_frame(
    tokenizer: Any, generated: torch.Tensor, widths: tuple[int, ...]
) -> Continuation:
    """Build the frame the decode produced, characters included.

    Token spans come from incremental detokenization — decode the row's
    first ``k`` tokens, then ``k + 1``, and the growth is token ``k``'s
    span. Re-encoding the finished text would not do: a tokenizer is free
    to merge across a boundary the decode never saw, and the spans have to
    describe the tokens the model actually emitted.
    """
    texts: list[str] = []
    offsets: list[tuple[tuple[int, int], ...]] = []
    for row, width in enumerate(widths):
        ids = [int(t) for t in generated[row, :width]]
        spans: list[tuple[int, int]] = []
        text = ""
        for k in range(width):
            grown = tokenizer.decode(
                ids[: k + 1],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            spans.append((len(text), len(grown)))
            text = grown
        texts.append(text)
        offsets.append(tuple(spans))
    return Continuation(
        token_ids=generated.detach().cpu(),
        widths=widths,
        texts=tuple(texts),
        offsets=tuple(offsets),
    )
