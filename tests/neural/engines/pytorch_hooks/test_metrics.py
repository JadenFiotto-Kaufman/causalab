"""Metric lowering (spec §2.10) on hand-built logits — exact formulas."""

from __future__ import annotations

import math

import pytest
import torch

from causalab.neural.shared.metrics import (
    column_first_token_id,
    column_token_id,
    column_token_ids,
    compute_metric,
    compute_windowed_metric,
)
from causalab.protocol.errors import ProtocolError
from causalab.protocol.resolution import Unavailable
from causalab.protocol.schema import MetricSpec

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.numerical_unit


@pytest.fixture(scope="module")
def tokenizer():
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    return load_model(TINY_LLAMA).tokenizer


def _logits(tokenizer, favored: str, disfavored: str) -> torch.Tensor:
    logits = torch.zeros(1, 1, 32000)
    logits[0, 0, column_token_id(tokenizer, favored)] = 4.0
    logits[0, 0, column_token_id(tokenizer, disfavored)] = 1.0
    return logits


def test_space_prefixed_first_resolution(tokenizer):
    # 📐 transformers 5.16.1: this sentencepiece tokenizer dropped the legacy
    # dummy prefix, so " Monday" and "Monday" BOTH encode to the single ▁Monday
    # piece. The two forms agree because they are now the same id — not, as
    # under transformers 4.x, because the spaced form was ▁+▁Monday and `auto`
    # fell back. Pinned so the reason cannot drift silently again.
    assert tokenizer.encode(" Monday", add_special_tokens=False) == tokenizer.encode(
        "Monday", add_special_tokens=False
    )
    assert column_token_id(tokenizer, " Monday") == column_token_id(tokenizer, "Monday")
    with pytest.raises(ProtocolError):
        column_token_id(tokenizer, " Tuesday")  # 3 pieces either way — refuse


def test_logit_diff(tokenizer):
    metric = MetricSpec(kind="logit_diff", of="logits", fields={"a": "x", "b": "y"})
    values = compute_metric(
        metric,
        _logits(tokenizer, " Monday", " Friday"),
        [{"x": " Monday", "y": " Friday"}],
        tokenizer,
    )
    assert values == [pytest.approx(3.0)]


def test_soft_accuracy_is_the_sigmoid_of_the_margin(tokenizer):
    """σ(logits[a] − logits[b]) on the same hand-built logits `test_logit_diff`
    pins at 3.0: σ(3) = 0.9526; swapping the columns gives 1 − σ(3), so the two
    rows of one table are complementary and the value lives in (0, 1)."""
    logits = _logits(tokenizer, " Monday", " Friday")
    forward = MetricSpec(kind="soft_accuracy", of="logits", fields={"a": "x", "b": "y"})
    values = compute_metric(
        forward, logits, [{"x": " Monday", "y": " Friday"}], tokenizer
    )
    assert values == [pytest.approx(1.0 / (1.0 + math.exp(-3.0)))]
    reverse = MetricSpec(kind="soft_accuracy", of="logits", fields={"a": "y", "b": "x"})
    flipped = compute_metric(
        reverse, logits, [{"x": " Monday", "y": " Friday"}], tokenizer
    )
    assert flipped == [pytest.approx(1.0 - values[0])]


def test_cross_entropy(tokenizer):
    metric = MetricSpec(kind="cross_entropy", of="logits", fields={"target": "label"})
    logits = _logits(tokenizer, " Monday", " Friday")
    values = compute_metric(metric, logits, [{"label": " Monday"}], tokenizer)
    want = -torch.log_softmax(logits[0, 0].float(), dim=-1)[
        column_token_id(tokenizer, " Monday")
    ]
    assert values == [pytest.approx(float(want))]


def test_match_is_an_argmax_indicator(tokenizer):
    metric = MetricSpec(kind="match", of="logits", fields={"expected": "ans"})
    logits = _logits(tokenizer, " Monday", " Friday")
    assert compute_metric(metric, logits, [{"ans": " Monday"}], tokenizer) == [1.0]
    assert compute_metric(metric, logits, [{"ans": " Friday"}], tokenizer) == [0.0]


def test_kl_of_identical_distributions_is_zero(tokenizer):
    metric = MetricSpec(kind="kl", of="p", fields={"target": "q"})
    logits = _logits(tokenizer, " Monday", " Friday")
    values = compute_metric(
        metric, logits, [{}], tokenizer, target_value=logits.clone()
    )
    assert values == [pytest.approx(0.0, abs=1e-6)]


def test_top_k_orders_by_probability(tokenizer):
    metric = MetricSpec(kind="top_k", of="logits", fields={"k": 2, "by": "prob"})
    logits = _logits(tokenizer, " Monday", " Friday")
    (entry,) = compute_metric(metric, logits, [{}], tokenizer)
    assert entry["tokens"][0].strip() == "Monday"
    assert entry["probs"][0] > entry["probs"][1]
    # `values` is the raw logit even under `by: prob` — the probability lives
    # in its own column, so neither ever changes identity (§2.10)
    assert entry["indices"][0] == column_token_id(tokenizer, " Monday")
    assert entry["values"] == [pytest.approx(4.0), pytest.approx(1.0)]


# --------------------------------------------------------------------------- #
# top_k over a read that is not a vocabulary projection (§2.10)
# --------------------------------------------------------------------------- #

#: A hand-built 1×6 "feature code": signed, with the largest magnitude on a
#: *negative* entry, so `value` and `abs_value` cannot agree.
_SIGNED_CODE = torch.tensor([[[0.5, -7.0, 3.0, -0.25, 6.0, -2.0]]])


def test_top_k_by_value_takes_the_largest_signed_entries(tokenizer):
    """Oracle: sorted descending the code is 6.0 (idx 4), 3.0 (idx 2),
    0.5 (idx 0) — the negatives never place."""
    metric = MetricSpec(kind="top_k", of="code", fields={"k": 3, "by": "value"})
    (entry,) = compute_metric(metric, _SIGNED_CODE, [{}], tokenizer, vocab_axis=False)
    assert entry["indices"] == [4, 2, 0]
    assert entry["values"] == [
        pytest.approx(6.0),
        pytest.approx(3.0),
        pytest.approx(0.5),
    ]


def test_top_k_by_abs_value_ranks_on_magnitude_and_reports_the_sign(tokenizer):
    """Oracle: by |x| the code is 7.0 (idx 1, negative), 6.0 (idx 4),
    3.0 (idx 2). The reported value stays signed — ranking by magnitude must
    not hide that the strongest entry pushed the other way."""
    metric = MetricSpec(kind="top_k", of="code", fields={"k": 3, "by": "abs_value"})
    (entry,) = compute_metric(metric, _SIGNED_CODE, [{}], tokenizer, vocab_axis=False)
    assert entry["indices"] == [1, 4, 2]
    assert entry["values"] == [
        pytest.approx(-7.0),
        pytest.approx(6.0),
        pytest.approx(3.0),
    ]


def test_top_k_off_lm_head_emits_no_token_or_probability_column(tokenizer):
    """A neuron index is not a token id and a softmax across neurons is not a
    distribution, so neither column is emitted rather than emitted wrong."""
    metric = MetricSpec(kind="top_k", of="code", fields={"k": 2, "by": "value"})
    (entry,) = compute_metric(metric, _SIGNED_CODE, [{}], tokenizer, vocab_axis=False)
    assert set(entry) == {"indices", "values"}


def test_top_k_by_value_on_lm_head_still_decodes_but_does_not_normalize(tokenizer):
    """`tokens` follows the read (lm_head), `probs` follows `by` — the two
    columns are gated independently."""
    metric = MetricSpec(kind="top_k", of="logits", fields={"k": 1, "by": "value"})
    (entry,) = compute_metric(
        metric, _logits(tokenizer, " Monday", " Friday"), [{}], tokenizer
    )
    assert entry["tokens"][0].strip() == "Monday"
    assert entry["values"] == [pytest.approx(4.0)]
    assert "probs" not in entry


def test_top_k_reduces_every_row_independently(tokenizer):
    """Two rows whose maxima sit at different indices — the reduction is
    per row, which is what makes it a drop-in for saving the tensor."""
    metric = MetricSpec(kind="top_k", of="code", fields={"k": 1, "by": "value"})
    batch = torch.tensor([[1.0, 9.0, 2.0], [8.0, -1.0, 3.0]])
    got = compute_metric(metric, batch, [{}, {}], tokenizer, vocab_axis=False)
    assert [entry["indices"] for entry in got] == [[1], [0]]
    assert [entry["values"] for entry in got] == [[9.0], [8.0]]


@pytest.mark.parametrize("k", [0, 7])
def test_top_k_refuses_a_k_outside_the_read_width(tokenizer, k):
    metric = MetricSpec(kind="top_k", of="code", fields={"k": k, "by": "value"})
    with pytest.raises(ProtocolError, match="k must be in"):
        compute_metric(metric, _SIGNED_CODE, [{}], tokenizer, vocab_axis=False)


def test_windowed_top_k_carries_vocab_axis_through_to_the_reduction(tokenizer):
    """The generated frame reduces through :func:`compute_windowed_metric`, and
    ``vocab_axis`` has to survive that hop.

    The prompt-frame cases above pin the reduction itself; this pins the
    *plumbing*, which is the half a windowed read could silently lose — a
    non-vocabulary read reduced with ``vocab_axis`` left at its ``True``
    default would decode neuron indices as token ids and softmax across
    neurons, and both wrong columns would look plausible in the saved table.

    Also pins the regrouping: rows address different position counts (2, 1, 0),
    and the flatten/cat/split round trip has to hand each row back its own.
    """
    windows = [
        torch.tensor(
            [[0.5, -7.0, 3.0, -0.25, 6.0, -2.0], [1.0, 2.0, 9.0, 0.0, -3.0, 0.5]]
        ),
        torch.tensor([[-8.0, 0.1, 0.2, 0.3, 0.4, 0.5]]),
        torch.zeros(0, 6),  # addressed no positions — a result, not a misalignment
    ]
    metric = MetricSpec(kind="top_k", of="code", fields={"k": 2, "by": "value"})
    got = compute_windowed_metric(
        metric, windows, [{}, {}, {}], tokenizer, vocab_axis=False
    )

    assert [len(row) for row in got] == [2, 1, 0]
    assert [[entry["indices"] for entry in row] for row in got] == [
        [[4, 2], [2, 1]],
        [[5, 4]],
        [],
    ]
    # neither vocabulary column may appear anywhere in the windowed output
    assert all(set(entry) == {"indices", "values"} for row in got for entry in row)


def test_class_probs_sums_group_members(tokenizer):
    metric = MetricSpec(
        kind="class_probs",
        of="logits",
        fields={"groups": {"days": [" Monday", " Friday"], "other": [" Sunday"]}},
    )
    logits = _logits(tokenizer, " Monday", " Friday")
    (entry,) = compute_metric(metric, logits, [{}], tokenizer)
    probs = torch.softmax(logits[0, 0].float(), dim=-1)
    want = float(
        probs[column_token_id(tokenizer, " Monday")]
        + probs[column_token_id(tokenizer, " Friday")]
    )
    assert entry["days"] == pytest.approx(want)
    assert math.isclose(
        entry["other"],
        float(probs[column_token_id(tokenizer, " Sunday")]),
        rel_tol=1e-6,
    )


# --------------------------------------------------------------------------- #
# §2.10 token_form — the answer-tokenization knob
#
# The bug this guards: `auto` returns the FIRST single-token candidate and tries
# the space-prefixed form first, so a punctuation answer resolves to a row the
# model never emits. Under gpt2 "?" is token 30 and " ?" is token 5633 — both
# single tokens — so a `match` metric on a punctuation answer read a flat 0.000
# at all 48 layers of a real gpt2-xl scan with no error raised anywhere.
#
# These use the real gpt2 tokenizer, not tiny-random-gpt2: the tiny stub's
# 1000-token vocabulary has no " ?" row, so it cannot express the ambiguity.
# The IOI suite already loads real gpt2 (tests/tasks/IOI/conftest.py).
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


def test_gpt2_punctuation_is_the_ambiguous_case(gpt2_tokenizer):
    """The premise, pinned: both forms are one token and they differ."""
    assert gpt2_tokenizer.encode("?", add_special_tokens=False) == [30]
    assert gpt2_tokenizer.encode(" ?", add_special_tokens=False) == [5633]


def test_auto_still_takes_the_space_prefixed_form(gpt2_tokenizer):
    """Backward compatibility: `auto` is exactly the historical resolver, so
    every document written before ``token_form`` existed is unchanged — this
    is also the CORRECT answer for the common case (an answer after a space)."""
    assert column_token_id(gpt2_tokenizer, "?") == 5633
    for word in (" Monday", "Monday", "Mary", " Mary"):
        spaced = gpt2_tokenizer.encode(" " + word.strip(), add_special_tokens=False)
        assert len(spaced) == 1
        assert column_token_id(gpt2_tokenizer, word) == spaced[0]


def test_bare_form_resolves_the_token_the_model_emits(gpt2_tokenizer):
    """The fix: a document that says `bare` gets the bare row."""
    assert column_token_id(gpt2_tokenizer, "?", token_form="bare") == 30
    assert column_token_id(gpt2_tokenizer, ".", token_form="bare") == 13
    # a leading space in the authored value is stripped, not honored
    assert column_token_id(gpt2_tokenizer, " ?", token_form="bare") == 30


def test_space_prefixed_form_is_pinnable(gpt2_tokenizer):
    assert column_token_id(gpt2_tokenizer, "?", token_form="space_prefixed") == 5633
    assert (
        column_token_id(gpt2_tokenizer, "Monday", token_form="space_prefixed")
        == (gpt2_tokenizer.encode(" Monday", add_special_tokens=False)[0])
    )


def test_a_pinned_form_refuses_rather_than_falling_back(gpt2_tokenizer):
    """Pinning a form means it: `space_prefixed` must refuse rather than quietly
    hand back the bare row it was told not to use.

    The witness moved to gpt2 in the transformers 5 bump. This contract needs a
    value whose bare form is ONE token while its space-prefixed form is several
    — 📐 under 5.16.1 the sentencepiece tokenizer no longer has one, because it
    dropped the legacy dummy prefix and now encodes both forms identically (see
    ``test_the_two_forms_collapse_on_sentencepiece``). gpt2's byte-level BPE
    still separates them and is unchanged across the bump: "haus" is [30404],
    " haus" is [387, 385]."""
    assert len(gpt2_tokenizer.encode("haus", add_special_tokens=False)) == 1
    assert len(gpt2_tokenizer.encode(" haus", add_special_tokens=False)) == 2

    with pytest.raises(ProtocolError):
        column_token_id(gpt2_tokenizer, "haus", token_form="space_prefixed")
    # and the bare row it refused to fall back to is genuinely resolvable
    assert column_token_id(gpt2_tokenizer, "haus", token_form="bare") == 30404
    # `auto` is the form that IS allowed to fall back, and it lands on that same
    # row — the contrast is what makes the refusal above a pin, not an accident
    assert column_token_id(gpt2_tokenizer, "haus", token_form="auto") == 30404


def test_the_two_forms_collapse_on_sentencepiece(tokenizer):
    """📐 The hazard the transformers 5 bump introduced, recorded as a test.

    Dropping the legacy dummy prefix means " X" and "X" encode identically on
    this family, so `token_form` cannot separate the two rows here and
    `_ambiguous_under_auto` can never fire — the once-per-column warning that
    exists because a punctuation `match` read a flat 0.000 across all 48 layers
    of a gpt2-xl scan is structurally dark on sentencepiece. Nothing to fix in
    the resolver (there is only one row to name); pinned so that the day a
    tokenizer separates them again, this test says so out loud."""
    assert column_token_id(tokenizer, "Monday", token_form="bare") == column_token_id(
        tokenizer, "Monday", token_form="space_prefixed"
    )


def test_match_refuses_a_punctuation_answer_under_auto_and_scores_it_under_bare(
    gpt2_tokenizer,
):
    """The end-to-end regression, at the metric level.

    The model emits "?" (token 30). `auto` resolved the space-prefixed form,
    token 5633, and read **0.0** — a wrong number that a pipeline gate scores
    as a dead stage. It warned while doing it and nobody read the warning, so
    the warning is now the refusal; `bare` still reads 1.0.
    """
    logits = torch.zeros(1, 1, gpt2_tokenizer.vocab_size)
    logits[0, 0, 30] = 4.0  # what the model actually emits

    metric = MetricSpec(kind="match", of="logits", fields={"expected": "ans"})
    with pytest.raises(ProtocolError) as err:
        compute_metric(metric, logits, [{"ans": "?"}], gpt2_tokenizer)
    assert err.value.code == "P2"
    assert "ambiguous under this tokenizer" in str(err.value)

    fixed = MetricSpec(
        kind="match", of="logits", fields={"expected": "ans"}, token_form="bare"
    )
    assert compute_metric(fixed, logits, [{"ans": "?"}], gpt2_tokenizer) == [1.0]


def test_auto_refuses_once_per_column_when_the_forms_disagree(gpt2_tokenizer):
    """`auto` stays the default, but it no longer guesses — at all.

    Aggregated per column: half the IOI name vocabulary is ambiguous on gpt2,
    and a per-value refusal would name one of them and hide the rest.
    """
    with pytest.raises(ProtocolError) as err:
        column_token_ids(gpt2_tokenizer, ["?", "?", ".", "!"])
    assert err.value.code == "P2"
    assert "3 of 3 distinct answers are ambiguous" in str(err.value)


def test_auto_is_accepted_when_there_is_nothing_to_disambiguate(tokenizer):
    """📐 Under transformers 5.16.1 this tokenizer encodes " Monday" and "Monday"
    to the SAME id, so the two forms cannot disagree — no choice was made, so
    nothing is refused. (Under 4.x the same silence held for the opposite
    reason: the spaced form was two pieces, so only the bare form resolved.)"""
    assert len(column_token_ids(tokenizer, [" Monday", "Monday"])) == 2


def test_a_pinned_form_is_never_refused(gpt2_tokenizer):
    """`bare` and `space_prefixed` are unchanged: the author has said which
    form the model emits, so there is nothing left to guess."""
    assert column_token_ids(gpt2_tokenizer, ["?", "."], token_form="bare") == [
        30,
        13,
    ]
    assert (
        len(column_token_ids(gpt2_tokenizer, ["?", "."], token_form="space_prefixed"))
        == 2
    )


#  match: answer-form groups and first-token grading (§2.10)                   #
# --------------------------------------------------------------------------- #


def test_match_accepts_any_form_in_a_group(tokenizer):
    """A list-valued expected column is a group of equivalent surface forms:
    the argmax matching any member scores 1.0 (the synonym channel — 'US' /
    'USA' / 'United States' — with the group serialized by the task)."""
    metric = MetricSpec(kind="match", of="logits", fields={"expected": "forms"})
    logits = _logits(tokenizer, " Monday", " Friday")
    assert compute_metric(
        metric, logits, [{"forms": [" Sunday", " Monday"]}], tokenizer
    ) == [1.0]
    assert compute_metric(
        metric, logits, [{"forms": [" Sunday", " Friday"]}], tokenizer
    ) == [0.0]


def test_match_scalar_column_is_a_group_of_one(tokenizer):
    """The pre-existing spelling keeps its meaning — a scalar column is a
    one-member group, so no existing document changes behaviour."""
    grouped = MetricSpec(kind="match", of="logits", fields={"expected": "ans"})
    logits = _logits(tokenizer, " Monday", " Friday")
    assert compute_metric(grouped, logits, [{"ans": " Monday"}], tokenizer) == [1.0]


def test_match_empty_group_is_an_excluded_row(tokenizer):
    """An empty form group is a row the table carries no answer for: an
    excluded measurement under ``alignment_missing`` (spec §2.10
    "Eligibility"), not a refusal of the run — it used to raise ``P2``."""
    metric = MetricSpec(kind="match", of="logits", fields={"expected": "forms"})
    (cell,) = compute_metric(
        metric, _logits(tokenizer, " Monday", " Friday"), [{"forms": []}], tokenizer
    )
    assert isinstance(cell, Unavailable)
    assert cell.reason == "alignment_missing"


def test_first_token_mode_credits_a_multi_token_answer(tokenizer):
    """``exact`` refuses " Thursday" (3 sentencepiece pieces either spelling);
    ``first_token`` credits its first *content* piece — what the model emits
    in context, and what the retired string-prefix grading meant."""
    exact = MetricSpec(kind="match", of="logits", fields={"expected": "ans"})
    first = MetricSpec(
        kind="match", of="logits", fields={"expected": "ans", "mode": "first_token"}
    )
    thursday_first = tokenizer.encode("Thursday", add_special_tokens=False)[0]
    logits = torch.zeros(1, 1, 32000)
    logits[0, 0, thursday_first] = 4.0

    with pytest.raises(ProtocolError):
        compute_metric(exact, logits, [{"ans": " Thursday"}], tokenizer)
    assert compute_metric(first, logits, [{"ans": " Thursday"}], tokenizer) == [1.0]


class _FormSplitTokenizer:
    """A tokenizer whose two surface forms credit *different* first tokens.

    The case `_ambiguous_under_auto` structurally cannot see: neither form is a
    single token, so by its definition nothing is "ambiguous" — while `auto`
    still picked a form, and under `first_token` the form decides which piece
    gets the credit.
    """

    _PIECES = {1: "Th", 2: "urs", 3: "day", 4: " Thu", 5: "rsday"}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [4, 5] if text.startswith(" ") else [1, 2, 3]

    def decode(self, ids) -> str:
        return "".join(self._PIECES[int(i)] for i in ids)


def test_auto_refuses_a_multi_token_value_whose_forms_credit_different_pieces():
    """The silent half of the `auto` trap, and the one the aggregated column
    check cannot reach. Pinning either form is accepted."""
    tok = _FormSplitTokenizer()
    with pytest.raises(ProtocolError) as err:
        column_first_token_id(tok, " Thursday")
    assert err.value.code == "P2"
    assert "credit different first tokens" in str(err.value)

    assert column_first_token_id(tok, " Thursday", token_form="bare") == 1
    assert column_first_token_id(tok, " Thursday", token_form="space_prefixed") == 4


def test_first_token_refuses_an_answer_space_that_is_not_first_token_distinct(
    tokenizer,
):
    """``first_token`` credits a *prefix*, so it means "the model answered"
    only where different answers begin with different tokens.

    Where they do not it over-credits in silence: ``" 85"`` is
    ``[220, "8", "5"]`` on Qwen, so a model emitting ``87`` scores 1.000
    against an expected ``85`` and nothing in the run says so. The two answers
    below share a first piece on this tokenizer for the same reason.
    """
    first = MetricSpec(
        kind="match",
        of="logits",
        fields={"expected": "ans", "mode": "first_token"},
        token_form="bare",
    )
    a, b = "Thursday", "Thursdays"
    assert column_first_token_id(tokenizer, a, token_form="bare") == (
        column_first_token_id(tokenizer, b, token_form="bare")
    ), "witness moved: these two answers no longer share a first token"

    logits = torch.zeros(2, 1, 32000)
    with pytest.raises(ProtocolError) as err:
        compute_metric(first, logits, [{"ans": a}, {"ans": b}], tokenizer)
    assert err.value.code == "P2"
    assert "not first-token distinct" in str(err.value)


def test_first_token_accepts_a_distinct_answer_space(tokenizer):
    """The weekdays answer space *is* distinct, so nothing is refused — the
    check is a guard on the metric's honesty, not a ban on prefix grading."""
    first = MetricSpec(
        kind="match",
        of="logits",
        fields={"expected": "ans", "mode": "first_token"},
        token_form="bare",
    )
    logits = torch.zeros(2, 1, 32000)
    monday = column_first_token_id(tokenizer, "Monday", token_form="bare")
    logits[0, 0, monday] = 4.0
    scores = compute_metric(
        first, logits, [{"ans": "Monday"}, {"ans": "Friday"}], tokenizer
    )
    assert scores == [1.0, 0.0]


class _LoneSpacePieceTokenizer:
    """A tokenizer whose leading space is its own piece — the trap, distilled.

    Which *values* trigger the trap is a property of a released tokenizer and it
    moved under transformers 5; the rule that `first_token` must never credit a
    whitespace-only piece is a property of :func:`column_first_token_id`. Pinning
    the rule against a stub keeps it honest across bumps, and
    :func:`test_first_token_skips_a_real_lone_space_piece` keeps a live witness."""

    _PIECES = {0: " ", 1: "Th", 2: "urs", 3: "day"}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0, 1, 2, 3] if text.startswith(" ") else [1, 2, 3]

    def decode(self, ids) -> str:
        return "".join(self._PIECES[int(i)] for i in ids)


def test_first_token_skips_the_lone_space_piece():
    """The trap: crediting the *first* id would credit the bare-space piece —
    which every space-prefixed answer shares, making every answer match."""
    tok = _LoneSpacePieceTokenizer()
    assert tok.decode([tok.encode(" Thursday")[0]]).strip() == ""  # the premise
    assert column_first_token_id(tok, " Thursday") == 1  # "Th", not the space


def test_first_token_skips_a_real_lone_space_piece(tokenizer):
    """The live witness on a real sentencepiece tokenizer.

    📐 Under transformers 5.16.1 this tokenizer stopped emitting a lone ▁ for an
    ordinary space-prefixed word — " Thursday" is now [Th, urs, day] — but still
    emits one whenever the first character has no merged ▁X piece: digits,
    non-Latin scripts, emoji, ligatures. Measured: encode(" 3.14") is
    [29871, 29941, 29889, 29896, 29946] and 29871 decodes to "". So the skip is
    load-bearing, not dead code. A failure of the premise below means the
    witness moved, not that the behaviour broke — the behaviour is pinned in
    :func:`test_first_token_skips_the_lone_space_piece`."""
    ids = tokenizer.encode(" 3.14", add_special_tokens=False)
    assert len(ids) > 1, "witness moved: ' 3.14' is a single token now"
    assert tokenizer.decode([ids[0]]).strip() == "", (
        "witness moved: ' 3.14' no longer leads with a whitespace-only piece"
    )
    assert column_first_token_id(tokenizer, "3.14") == ids[1]


def test_first_token_agrees_with_exact_on_single_token_answers(tokenizer):
    """``first_token`` is a strict generalization: where ``exact`` resolves, it
    resolves to the same id."""
    for value in (" Monday", " Friday", " Sunday", "Monday"):
        assert column_first_token_id(tokenizer, value) == column_token_id(
            tokenizer, value
        )


# --------------------------------------------------------------------------- #
# §2.10 class_probs — a group is a set of token ids, not a list of strings
#
# `_candidates` strips a leading space before `token_form` picks a form, so
# ["Sorry", " Sorry"] resolves to one id twice and `class_probs` — which SUMS a
# group's ids — counted it twice, so a "probability" above 1 was possible.
#
# The gpt2 tokenizer is the witness that makes this visible. On the
# sentencepiece fixture above, half the surface-form distinctions collapse
# anyway, so the CPU suite could not have seen the bug.
# --------------------------------------------------------------------------- #


def _one_hot(vocab: int, token: int, value: float = 4.0) -> torch.Tensor:
    logits = torch.zeros(1, 1, vocab)
    logits[0, 0, token] = value
    return logits


def test_class_probs_refuses_a_group_whose_members_are_one_token(gpt2_tokenizer):
    """The `["X", " X"]` idiom is inert, and in this kind it was worse than
    inert. Refusing beats returning 1.99."""
    metric = MetricSpec(
        kind="class_probs",
        of="logits",
        fields={"groups": {"refusal": ["Sorry", " Sorry"]}},
        token_form="space_prefixed",
    )
    with pytest.raises(ProtocolError) as err:
        compute_metric(
            metric,
            _one_hot(gpt2_tokenizer.vocab_size, 0),
            [{}],
            gpt2_tokenizer,
        )
    assert "resolve to token id" in str(err.value)
    assert "twice" in str(err.value)


def test_class_probs_over_a_deduplicated_group_stays_a_probability(gpt2_tokenizer):
    """The number the double-count was hiding: one member, one id, ≤ 1.0."""
    token = column_token_id(gpt2_tokenizer, "Sorry", token_form="space_prefixed")
    metric = MetricSpec(
        kind="class_probs",
        of="logits",
        fields={"groups": {"refusal": ["Sorry"]}},
        token_form="space_prefixed",
    )
    logits = _one_hot(gpt2_tokenizer.vocab_size, token, value=12.0)
    (entry,) = compute_metric(metric, logits, [{}], gpt2_tokenizer)
    probs = torch.softmax(logits[0, 0].float(), dim=-1)
    assert entry["refusal"] == pytest.approx(float(probs[token]))
    assert 0.0 <= entry["refusal"] <= 1.0


def test_class_probs_distinct_forms_under_bare_still_sum(gpt2_tokenizer):
    """The dedup must not eat a real two-member class: under `bare`, "Sorry"
    and "sorry" are different gpt2 ids and both belong in the group."""
    upper = column_token_id(gpt2_tokenizer, "Sorry", token_form="bare")
    lower = column_token_id(gpt2_tokenizer, "sorry", token_form="bare")
    assert upper != lower  # the premise
    metric = MetricSpec(
        kind="class_probs",
        of="logits",
        fields={"groups": {"refusal": ["Sorry", "sorry"]}},
        token_form="bare",
    )
    logits = _one_hot(gpt2_tokenizer.vocab_size, upper, value=6.0)
    logits[0, 0, lower] = 6.0
    (entry,) = compute_metric(metric, logits, [{}], gpt2_tokenizer)
    probs = torch.softmax(logits[0, 0].float(), dim=-1)
    assert entry["refusal"] == pytest.approx(float(probs[upper] + probs[lower]))


# --------------------------------------------------------------------------- #
# §2.10 token_logits — the task's answer space, saved as raw logits
#
# The oracle is `token_logit`: one listed token at a time, through a dataset
# column, is exactly what `token_logits` reports for every listed token at
# once. The pin is that the two agree entry for entry — same id, same raw value.
# --------------------------------------------------------------------------- #

_ANSWERS = (" Monday", " Friday", " Sunday")


def _token_logits_metric(tokens=_ANSWERS, token_form="space_prefixed"):
    return MetricSpec(
        kind="token_logits",
        of="logits",
        fields={"tokens": tuple(tokens)},
        token_form=token_form,
    )


def test_token_logits_equal_the_corresponding_token_logit_values(tokenizer):
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 32000)
    (first, second) = compute_metric(
        _token_logits_metric(), logits, [{}, {}], tokenizer
    )

    for entry, example in ((first, 0), (second, 1)):
        assert list(entry) == ["indices", "tokens", "values"]
        assert entry["indices"] == [
            column_token_id(tokenizer, t, token_form="space_prefixed") for t in _ANSWERS
        ]
        assert entry["tokens"] == [tokenizer.decode([i]) for i in entry["indices"]]
        for answer, value in zip(_ANSWERS, entry["values"]):
            oracle = MetricSpec(
                kind="token_logit",
                of="logits",
                fields={"token": "t"},
                token_form="space_prefixed",
            )
            (want,) = compute_metric(
                oracle, logits[example : example + 1], [{"t": answer}], tokenizer
            )
            assert value == pytest.approx(want)


def test_token_logits_values_are_raw_logits_not_probabilities(tokenizer):
    """`values` has one identity across `top_k` and this kind: the raw read
    value. Nothing is normalized."""
    logits = _logits(tokenizer, " Monday", " Friday")
    (entry,) = compute_metric(_token_logits_metric(), logits, [{}], tokenizer)
    assert entry["values"] == [
        pytest.approx(4.0),
        pytest.approx(1.0),
        pytest.approx(0.0),
    ]


def test_token_logits_refuses_a_multi_token_entry(tokenizer):
    """The single-token rule every string-resolving kind follows: " Tuesday"
    is three pieces on this tokenizer, and scoring its first piece would
    silently save the logit of a different token under the answer's name."""
    metric = _token_logits_metric(tokens=(" Monday", " Tuesday"))
    with pytest.raises(ProtocolError, match="not a single token"):
        compute_metric(metric, torch.zeros(1, 1, 32000), [{}], tokenizer)


def test_token_logits_refuses_two_entries_that_resolve_to_one_id(tokenizer):
    """A spec built in code bypasses the parse-time duplicate check, and a
    tokenizer can map two different strings to one id regardless; either way
    the same row would be saved twice under two names. On this sentencepiece
    fixture " Monday" and "Monday" are one id (see the resolution pin at the
    top of the file), which is the collision a `MetricSpec` can carry."""
    metric = _token_logits_metric(tokens=("Monday", " Monday"), token_form="auto")
    with pytest.raises(ProtocolError) as err:
        compute_metric(metric, torch.zeros(1, 1, 32000), [{}], tokenizer)
    assert "resolve to token id" in str(err.value)
    assert "twice" in str(err.value)


def test_windowed_token_logits_reduces_per_position(tokenizer):
    """Over a multi-position read the kind follows the per-position rules:
    one entry per addressed position, each carrying the three lists, and an
    example that addressed nothing hands back an empty row."""
    torch.manual_seed(1)
    windows = [torch.randn(2, 32000), torch.zeros(0, 32000), torch.randn(1, 32000)]
    got = compute_windowed_metric(
        _token_logits_metric(), windows, [{}, {}, {}], tokenizer
    )
    assert [len(row) for row in got] == [2, 0, 1]
    ids = [column_token_id(tokenizer, t, token_form="space_prefixed") for t in _ANSWERS]
    for window, row in zip(windows, got):
        for position, entry in enumerate(row):
            assert entry["indices"] == ids
            assert entry["values"] == [
                pytest.approx(float(window[position, i])) for i in ids
            ]


# --------------------------------------------------------------------------- #
# §2.10 `js` — Jensen–Shannon, optionally restricted to an answer set
# --------------------------------------------------------------------------- #


def _js_by_hand(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """The definition, spelled out in probability space over one row."""
    p = torch.softmax(p_logits.double(), dim=-1)
    q = torch.softmax(q_logits.double(), dim=-1)
    m = 0.5 * (p + q)
    kl_pm = float((p * (p / m).log()).sum())
    kl_qm = float((q * (q / m).log()).sum())
    return 0.5 * kl_pm + 0.5 * kl_qm


def test_js_of_identical_distributions_is_zero(tokenizer):
    metric = MetricSpec(kind="js", of="p", fields={"target": "q"})
    p = _logits(tokenizer, " Monday", " Friday")
    values = compute_metric(metric, p, [{}], tokenizer, target_value=p.clone())
    assert values == [pytest.approx(0.0, abs=1e-7)]


def test_js_is_symmetric_and_bounded_by_ln2(tokenizer):
    metric = MetricSpec(kind="js", of="p", fields={"target": "q"})
    # well-separated peaks, so the value is O(0.1) and a float32 log_softmax
    # over 32000 entries is compared at a tolerance it can meet (the definition
    # is evaluated in float64; the reduction runs in float32)
    p = torch.zeros(1, 1, 32000)
    q = torch.zeros(1, 1, 32000)
    p[0, 0, column_token_id(tokenizer, " Monday")] = 15.0
    q[0, 0, column_token_id(tokenizer, " Friday")] = 15.0
    pq = compute_metric(metric, p, [{}], tokenizer, target_value=q)[0]
    qp = compute_metric(metric, q, [{}], tokenizer, target_value=p)[0]
    assert pq == pytest.approx(qp, rel=1e-6)
    assert pq == pytest.approx(_js_by_hand(p[0, 0], q[0, 0]), rel=1e-4)
    assert 0.5 < pq < math.log(2)  # two near-point masses on different tokens
    # disjoint supports: the bound is reached
    far_p = torch.full((1, 1, 32000), -40.0)
    far_q = torch.full((1, 1, 32000), -40.0)
    far_p[0, 0, column_token_id(tokenizer, " Monday")] = 40.0
    far_q[0, 0, column_token_id(tokenizer, " Friday")] = 40.0
    assert compute_metric(metric, far_p, [{}], tokenizer, target_value=far_q)[
        0
    ] == pytest.approx(math.log(2), abs=1e-6)


def test_restricted_js_is_js_over_the_renormalised_slice(tokenizer):
    """`restrict` slices both distributions to the answer ids and renormalises
    — a `log_softmax` over the slice — so the value equals the definition
    applied to the sliced logits, and the literal and column spellings agree."""
    answers = [" Monday", " Friday"]
    ids = [column_token_id(tokenizer, a) for a in answers]
    p = _logits(tokenizer, " Monday", " Friday")
    q = _logits(tokenizer, " Friday", " Monday")
    p[0, 0, 5] = 6.0  # mass on a token outside the answer set
    literal = MetricSpec(
        kind="js",
        of="p",
        fields={"target": "q", "restrict": tuple(answers)},
        token_form="space_prefixed",
    )
    column = MetricSpec(
        kind="js",
        of="p",
        fields={"target": "q", "restrict": "valid"},
        token_form="space_prefixed",
    )
    by_literal = compute_metric(literal, p, [{}], tokenizer, target_value=q)[0]
    by_column = compute_metric(
        column, p, [{"valid": answers}], tokenizer, target_value=q
    )[0]
    expected = _js_by_hand(p[0, 0, ids], q[0, 0, ids])
    assert by_literal == pytest.approx(expected, rel=1e-5)
    assert by_column == pytest.approx(expected, rel=1e-5)
    # and it is not the unrestricted value: the off-set mass changes that one
    unrestricted = MetricSpec(kind="js", of="p", fields={"target": "q"})
    assert compute_metric(unrestricted, p, [{}], tokenizer, target_value=q)[
        0
    ] != pytest.approx(expected, rel=1e-3)


def test_js_restrict_column_empty_row_is_an_excluded_measurement(tokenizer):
    metric = MetricSpec(
        kind="js",
        of="p",
        fields={"target": "q", "restrict": "valid"},
        token_form="space_prefixed",
    )
    p = torch.cat([_logits(tokenizer, " Monday", " Friday")] * 2, dim=0)
    q = torch.cat([_logits(tokenizer, " Friday", " Monday")] * 2, dim=0)
    rows = [{"valid": []}, {"valid": [" Monday", " Friday"]}]
    values = compute_metric(metric, p, rows, tokenizer, target_value=q)
    assert isinstance(values[0], Unavailable)
    assert values[0].reason == "alignment_missing"
    assert isinstance(values[1], float) and values[1] > 0.0


def test_js_restrict_refuses_two_answers_on_one_id(tokenizer):
    """The `class_probs` rule: a duplicated id would give one answer double
    mass in the restricted softmax. On this sentencepiece tokenizer
    `" Monday"` and `"Monday"` are one id (pinned above), so the pair is the
    run-time collision the parse-time check cannot see."""
    metric = MetricSpec(
        kind="js",
        of="p",
        fields={"target": "q", "restrict": "valid"},
        token_form="auto",
    )
    p = _logits(tokenizer, " Monday", " Friday")
    with pytest.raises(ProtocolError) as err:
        compute_metric(
            metric, p, [{"valid": [" Monday", "Monday"]}], tokenizer, target_value=p
        )
    assert "double mass" in str(err.value)
