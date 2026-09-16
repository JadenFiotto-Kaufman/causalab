"""Alignment cardinality as data (spec §2.3), and the pair-difference
validator (spec §2.3).

**One address, two inputs, one of five cardinalities.** A position (§2.3)
addresses tokens in a row; a counterfactual pair resolves the same address on
the base input and on each counterfactual input. How the two runs map onto
each other is a fact with five values — :data:`~causalab.protocol.schema
.ALIGNMENT_CARDINALITIES`:

* ``one_to_one`` — one run on each side, the same width: a single token, or a
  joint span written as one address (a two-token number addressed by
  ``{"span": [a, a+2]}`` is *one* address of width two on both sides);
* ``one_to_many`` — one token on the base side, several on the other (the
  entity that is one piece here and three pieces there);
* ``many_to_one`` — the reverse;
* ``absent`` — one side resolved to nothing: the value the address names does
  not occur in that row's text;
* ``ambiguous`` — more than one alignment fits and none was named: the value
  occurs several times, or the two runs are both wider than one token and of
  unequal widths, so no pairing is canonical.

:func:`alignment_of` is **the one function that derives it.** Planning
(``plan.static_alignment``: what the document alone decides for an ``index``
or an unscoped ``span``), execution (``encoding.resolve_position`` for one
input's candidate runs; ``ExecutorBase._positions`` for a declared cardinality
against the pair) and metrics (``metrics.check_answer_forms``: the authored
answer form against the form the data carries) all call it and none
re-derives the classification — ``tests/protocol/test_alignment.py`` holds the
set of callers to exactly those modules. Both arguments are the *candidate*
runs one address resolved to on one side: zero candidates is ``absent``,
several is ``ambiguous``, and only one candidate per side has a width to
compare.

**The two unalignable values are reason codes, not exceptions.** ``absent``
is ``alignment_missing`` and ``ambiguous`` is ``alignment_ambiguous`` (§2.4's
table — nothing new is minted here). A row a *read*
cannot align is an :class:`~causalab.protocol.resolution.Unavailable` cell
through :func:`unalignable` (§4.1): it appears in the result and in the
denominator, never as a silent drop. A row a *write* cannot align is refused
before any forward pass through :func:`refuse_unalignable`, because a write
that silently skipped a row would report a number for an intervention that
did not happen. A **declared** ``alignment`` (§2.3) that the observed
cardinality contradicts is refused by :func:`check_declared` — never
silently overridden.

**The difference validator.** Two error classes recur in authoring
counterfactual pairs: choosing a bare token where the model's
answer is space-prefixed (the metrics check above), and treating a legitimate
change in a *recomputed answer prefix* as an unintended prompt edit. The
second is a validator that reports differences in **three sets, separately**
— :func:`pair_differences`: the **prompt** (the texts before any answer
prefix), the **teacher-forced prefix** (the answer prefix each input carries,
recomputed per input by the task), and the **full context** (prompt +
prefix, tokenized as one string, because a tokenizer may merge across the
boundary and the full-context differences are then not the union of the
other two). A pair differing only in its recomputed prefix validates clean:
an empty prompt set, a non-empty prefix set, a non-empty full set. Folding
the three into one would make that pair indistinguishable from a prompt
edit, which is the error class.

The contexts compared here are the strings the engine's ``encode`` receives:
v1 has no chat prefix (``EncodedBatch.prefix_lengths`` is 0 for every row,
§2.3), so a row's full context is its prompt followed by its answer prefix
and nothing before it. The validator is torch-free and loads nothing: the
tokenizer is an argument (anything with ``encode`` / ``convert_ids_to_tokens``
/ ``decode``), which keeps it usable from the pure verbs and from a notebook
without an accelerator.
"""

from __future__ import annotations

import dataclasses
import difflib
from typing import Any, Sequence

from causalab.protocol.errors import ProtocolError, ReasonCode
from causalab.protocol.resolution import Unavailable, unavailable
from causalab.protocol.schema import AlignmentCardinality

__all__ = [
    "Hunk",
    "PairDifferences",
    "UnalignableError",
    "alignment_of",
    "check_declared",
    "pair_differences",
    "refuse_unalignable",
    "token_runs",
    "unalignable",
    "unalignable_reason",
]


def alignment_of(
    base: Sequence[Sequence[int]],
    counterfactual: Sequence[Sequence[int]] | None = None,
) -> AlignmentCardinality:
    """The observed cardinality of one address across a pair.

    ``base`` and ``counterfactual`` are the **candidate** runs the address
    resolved to on each side — one entry per occurrence of the value in the
    row's text, each a run of token indices. ``counterfactual`` is ``None``
    when the address is resolved on one input alone (a document with one
    role, or the per-input half of a pair), in which case the classification
    is over the candidates only: none is ``absent``, several is
    ``ambiguous``, one is ``one_to_one``.
    """
    sides = [base] if counterfactual is None else [base, counterfactual]
    for side in sides:
        if len(side) == 0:
            return "absent"
    for side in sides:
        if len(side) > 1:
            return "ambiguous"
    widths = [len(side[0]) for side in sides]
    if any(width == 0 for width in widths):
        return "absent"
    if len(widths) == 1 or widths[0] == widths[1]:
        return "one_to_one"
    if widths[0] == 1:
        return "one_to_many"
    if widths[1] == 1:
        return "many_to_one"
    return "ambiguous"


def token_runs(
    needle: Sequence[int], haystack: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Every occurrence of ``needle`` as a contiguous run inside ``haystack``,
    as runs of ``haystack`` indices — the candidate runs of one token
    sequence addressed inside another (the metrics half: the authored answer
    form inside the form the data carries)."""
    n = len(needle)
    if n == 0:
        return ()
    return tuple(
        tuple(range(start, start + n))
        for start in range(len(haystack) - n + 1)
        if tuple(haystack[start : start + n]) == tuple(needle)
    )


def unalignable_reason(observed: AlignmentCardinality) -> ReasonCode | None:
    """The reason code an unalignable cardinality carries (§2.4), or ``None``
    for the three that pair."""
    if observed == "absent":
        return "alignment_missing"
    if observed == "ambiguous":
        return "alignment_ambiguous"
    return None


def unalignable(
    observed: AlignmentCardinality, detail: str, denominator_key: str
) -> Unavailable | None:
    """The ``unavailable`` cell an unalignable *read* row becomes (§4.1), or
    ``None`` when the cardinality pairs. ``detail`` names the value, how many
    times it occurred and the row; the cell is counted in the denominator
    under ``denominator_key``."""
    if observed == "absent":
        return unavailable("alignment_missing", detail, denominator_key)
    if observed == "ambiguous":
        return unavailable("alignment_ambiguous", detail, denominator_key)
    return None


class UnalignableError(ProtocolError):
    """A row an address could not be aligned on, as a refusal.

    Raised where the row is not a *read's* to record as an unavailable cell
    — position resolution itself, a write's positions, a metric's answer
    form — with the reason code the cardinality maps to. ``cardinality`` is
    the observed value, so an executor catching this for a read can build the
    cell with :func:`unalignable` rather than re-deriving it.
    """

    def __init__(self, cardinality: AlignmentCardinality, message: str) -> None:
        reason = unalignable_reason(cardinality)
        if reason is None:
            raise AssertionError(f"{cardinality!r} is not an unalignable cardinality")
        self.cardinality: AlignmentCardinality = cardinality
        super().__init__("P2", message, reason=reason)


def refuse_unalignable(observed: AlignmentCardinality, message: str) -> None:
    """Raise :class:`UnalignableError` when ``observed`` is ``absent`` or
    ``ambiguous``; return otherwise. The refusal carries reason
    ``alignment_missing`` for the first and ``alignment_ambiguous`` for the
    second — the two spellings are written here and nowhere else."""
    if observed == "absent":
        raise UnalignableError(observed, message)
    if observed == "ambiguous":
        raise UnalignableError(observed, message)


def check_declared(
    declared: str | None, observed: AlignmentCardinality, *, where: str
) -> None:
    """Refuse a declared ``alignment`` the observed cardinality contradicts.

    ``declared`` is the position's authored value (``None`` declares
    nothing and is never refused). A contradiction is an encode-time refusal
    (the encode-time boundary, §2.3): the document *said* how the address pairs and
    the tokenizer says otherwise, which is a wrong number waiting to happen
    if the declaration were honored over the observation, and a silent
    override the other way. The message names both.
    """
    if declared is None or declared == observed:
        return
    reason = unalignable_reason(observed)
    raise ProtocolError(
        "P2",
        f"{where} declares alignment {declared!r}, but the pair resolves it as "
        f"{observed!r} — a declared cardinality is checked against the "
        "tokenizer's, never honored over it. Correct the declaration, or drop "
        "it to let the observed cardinality stand",
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# the pair-difference validator
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Hunk:
    """One differing stretch between two token sequences: ``op`` is
    difflib's ``replace`` / ``delete`` / ``insert``; ``base`` and
    ``counterfactual`` are the decoded tokens each side has there."""

    op: str
    base: tuple[str, ...]
    counterfactual: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class PairDifferences:
    """The three difference sets of one counterfactual pair, kept apart.

    ``prompt`` compares the two prompts; ``teacher_forced_prefix`` the two
    answer prefixes; ``full_context`` the two ``prompt + prefix`` strings,
    each tokenized as one string. The three are reported separately because
    the error class this exists for is reading a prefix change as a prompt
    edit: :attr:`prompt_edited` is the question "did the *prompt* change",
    and it is answered by the first set alone.
    """

    prompt: tuple[Hunk, ...]
    teacher_forced_prefix: tuple[Hunk, ...]
    full_context: tuple[Hunk, ...]

    @property
    def prompt_edited(self) -> bool:
        """Whether the pair differs in its prompt — the intended edit of a
        counterfactual pair, and the one thing a recomputed prefix is not."""
        return bool(self.prompt)

    @property
    def prefix_recomputed(self) -> bool:
        """Whether the pair differs in its teacher-forced answer prefix."""
        return bool(self.teacher_forced_prefix)


def _tokens(tokenizer: Any, text: str) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [str(t) for t in tokenizer.convert_ids_to_tokens(list(ids))]


def _hunks(tokenizer: Any, base: str, counterfactual: str) -> tuple[Hunk, ...]:
    a, b = _tokens(tokenizer, base), _tokens(tokenizer, counterfactual)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return tuple(
        Hunk(op=op, base=tuple(a[i1:i2]), counterfactual=tuple(b[j1:j2]))
        for op, i1, i2, j1, j2 in matcher.get_opcodes()
        if op != "equal"
    )


def pair_differences(
    tokenizer: Any,
    base_prompt: str,
    counterfactual_prompt: str,
    *,
    base_prefix: str = "",
    counterfactual_prefix: str = "",
) -> PairDifferences:
    """The three difference sets of one pair (module docstring).

    ``*_prefix`` is each input's teacher-forced answer prefix — the part of
    the context the task recomputes per input and the model is forced through
    before the position a metric reads. The default (no prefix on either
    side) is every v1 corpus document, whose rows end at the prompt.
    """
    return PairDifferences(
        prompt=_hunks(tokenizer, base_prompt, counterfactual_prompt),
        teacher_forced_prefix=_hunks(tokenizer, base_prefix, counterfactual_prefix),
        full_context=_hunks(
            tokenizer,
            base_prompt + base_prefix,
            counterfactual_prompt + counterfactual_prefix,
        ),
    )
