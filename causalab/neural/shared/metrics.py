"""Metric lowering (spec §2.10): the closed kinds over one read.

Every kind is gather-then-reduce over the read's value and dataset
columns. Per-example results come back as plain floats (or small
structures for ``top_k``), ready for a JSON metric table.

All but one kind name vocabulary entries, so validation binds them to an
``lm_head`` read and the value they reduce is a logit vector. ``top_k`` is
the exception: it ranks the entries of whatever axis its read has — a
vocabulary, a 4k-wide residual stream, a 100k-latent SAE code — and reduces
**where the rows are gathered**, which is the point of it (saving the whole
tensor just to argsort it later is the thing to avoid). Its mandatory ``by``
field says how to rank, and ``vocab_axis`` tells the reduction whether the
indices it found are token ids worth decoding.

Token resolution defaults to the repo's space-prefixed-first rule
(``token_form: "auto"``): a column value resolves to the single token of
``" " + s`` when that is one token, else of ``s`` itself; anything
multi-token refuses — a metric over a multi-token answer is not expressible
in v1's closed vocabulary and must not silently score the first piece.

``auto`` is right whenever the answer follows a space in the prompt
(weekdays, IOI names, MCQA letters), which is the common case, and it is
**wrong** whenever the answer does not. Punctuation is the canonical
counterexample: gpt2 encodes ``"?"`` as 30 and ``" ?"`` as 5633, both single
tokens, so ``auto`` scores 5633 while the model emits 30 — a ``match`` metric
then reads a flat 0.000 with no error anywhere. That is why a §2.10 metric
carries ``token_form``: set ``"bare"`` or ``"space_prefixed"`` to pin the
form instead of letting the tokenizer's vocabulary decide. Under ``auto``,
:func:`refuse_ambiguous_auto` **refuses** the column when both forms are single
tokens and disagree — exactly the condition under which ``auto`` can be
silently wrong. It used to warn; the warning printed, nobody read it, and the
run produced the wrong number anyway. A default whose failure mode is a wrong
number is a refusal, not a warning.

📐 One limit of that refusal, introduced by the transformers 5 bump: a
sentencepiece family that has dropped the legacy dummy prefix encodes ``" X"``
and ``"X"`` to the *same* id, so the two forms can never disagree there. On such
a tokenizer ``token_form`` has only one row to name, the check is structurally
dark, and neither is a defect — but it does mean the punctuation trap above is a
BPE-family hazard, and a document cannot be checked against it by pinning
``token_form`` on a sentencepiece model.

``match`` is the one kind that can be told otherwise, and only explicitly
(§2.10): its ``expected`` column may hold a **list** of equivalent surface
forms (synonyms, casings), and ``"mode": "first_token"`` credits a form's
first token instead of demanding the form be one token. Both are
task-data decisions — the table says which forms count, the document says
whether a prefix counts — so neither can happen by accident.

Crediting a prefix is only honest when the answer space is first-token
distinct, so ``first_token`` refuses a row set where two different answers
begin with the same token (:func:`_refuse_indistinct_first_tokens`). ``" 85"``
is ``[220, "8", "5"]`` on Qwen, and a model emitting ``87`` would otherwise
score 1.000 against an expected ``85``, with nothing in the run's outputs
saying so.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.alignment import alignment_of, refuse_unalignable, token_runs
from causalab.protocol.errors import ProtocolError
from causalab.protocol.resolution import Unavailable, unavailable
from causalab.protocol.schema import (
    metric_column_fields,
    VOCAB_TOP_K_RANKING,
    WHOLE_WINDOW_METRIC_KINDS,
    MetricSpec,
)

__all__ = [
    "check_answer_forms",
    "column_first_token_id",
    "column_token_id",
    "column_token_ids",
    "refuse_ambiguous_auto",
    "compute_metric",
    "compute_windowed_metric",
    "excluded_rows",
    "GATHERED_KINDS",
    "gathered_metric",
    "metric_token_ids",
]


def _single_token(tokenizer: Any, text: str) -> int | None:
    """The token id of ``text`` when it encodes to exactly one token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def _candidates(value: str, token_form: str = "auto") -> tuple[str, ...]:
    """The surface forms to try, in order, for one metric answer.

    ``token_form`` is the §2.10 knob: ``auto`` tries the space-prefixed form
    first (BPE families make ``" one"`` one token) and falls back to the bare
    form (sentencepiece's ``"one"`` IS the ``\u2581one`` piece);
    ``space_prefixed`` and ``bare`` pin one form. A leading space in the
    authored value is normalized away rather than honored, so ``" ?"`` and
    ``"?"`` mean the same answer and only ``token_form`` decides the form.
    """
    bare = value.lstrip(" ")
    return {
        "auto": (" " + bare, bare),
        "space_prefixed": (" " + bare,),
        "bare": (bare,),
    }[token_form]


def _checked_token_id(value: Any, vocabulary_size: int) -> int:
    # Preserve the integer-only contract: bool, floats and strings are not IDs.
    if type(value) is not int or not 0 <= value < vocabulary_size:
        raise ProtocolError(
            "P2",
            f"token_form='id' needs an integer token ID in the vocabulary, got {value!r}",
        )
    return value


def column_token_id(tokenizer: Any, value: Any, *, token_form: str = "auto") -> int:
    """The single token id a metric column value names (module docstring).

    ``token_form`` is the §2.10 knob: ``"auto"`` (the default, and what every
    document without the key gets) keeps the space-prefixed-first rule,
    ``"space_prefixed"`` pins ``" " + s``, ``"bare"`` pins ``s`` with leading
    spaces stripped.
    """
    if token_form == "id":
        return _checked_token_id(value, len(tokenizer))
    candidates = _candidates(str(value), token_form)

    resolved: int | None = None
    for candidate in candidates:
        resolved = _single_token(tokenizer, candidate)
        if resolved is not None:
            break
    if resolved is None:
        tried = ", ".join(repr(c) for c in candidates)
        raise ProtocolError(
            "P2",
            f"metric column value {value!r} is not a single token under this "
            f"tokenizer (token_form={token_form!r}, tried {tried}) — multi-token "
            "answers have no closed metric kind in v1",
        )
    return resolved


def _ambiguous_under_auto(tokenizer: Any, value: str) -> tuple[int, int] | None:
    """``(space_prefixed_id, bare_id)`` when both forms are single tokens and
    they name *different* rows — the condition under which ``auto`` picks one
    for the author and can silently pick the wrong one."""
    bare = value.lstrip(" ")
    spaced_id = _single_token(tokenizer, " " + bare)
    bare_id = _single_token(tokenizer, bare)
    if spaced_id is None or bare_id is None or spaced_id == bare_id:
        return None
    return spaced_id, bare_id


def refuse_ambiguous_auto(
    tokenizer: Any, values: Sequence[str], *, where: str = "metric column"
) -> None:
    """Refuse a column ``token_form: "auto"`` would have to guess at.

    The condition is the one :func:`_ambiguous_under_auto` names: both surface
    forms are single tokens and they name *different* vocabulary rows, so the
    space-prefixed-first rule picks one and the author never said which.

    This used to be a ``UserWarning``. It printed, nobody read it, and the run
    scored a flat 0.000 at all 48 layers of a real gpt2-xl scan — a wrong
    number, which a pipeline gate reads as a dead stage rather than an error.
    A default whose failure mode is a wrong number has to be a refusal.

    Aggregated per column on purpose: per-value refusals would name one of
    half the IOI name vocabulary and hide how wide the problem is.
    """
    ambiguous = {
        v: pair
        for v in dict.fromkeys(values)
        if (pair := _ambiguous_under_auto(tokenizer, v))
    }
    if not ambiguous:
        return
    examples = ", ".join(
        f"{v!r} → {spaced} (space-prefixed) vs {bare} (bare)"
        for v, (spaced, bare) in list(ambiguous.items())[:3]
    )
    raise ProtocolError(
        "P2",
        f"{where}: {len(ambiguous)} of {len(set(values))} distinct answers are "
        f"ambiguous under this tokenizer — both forms are single tokens and they "
        f"name different rows ({examples}). token_form='auto' cannot know which "
        "one the model emits; set the metric's token_form to 'bare' or "
        "'space_prefixed' to say so",
    )


def check_answer_forms(
    tokenizer: Any,
    metric: MetricSpec,
    rows: Sequence[Mapping[str, Any]],
    *,
    where: str = "metric",
) -> None:
    """Refuse a bare ``token_form`` over an answer the table carries
    space-prefixed — checked before any forward pass (§2.10, §2.3).

    A leading space in a table value is the data saying how the model's
    answer reads in context: ``" Saturday"`` is what follows ``"tomorrow
    is"``. A metric that pins ``token_form: "bare"`` over that column scores
    ``"Saturday"`` instead, and under a byte-level BPE the two are different
    tokens (``ĠS…`` against ``S…``), so the metric credits a token the model
    never emits and reads a flat 0.000 with nothing said. The check is the
    metrics half of :func:`~causalab.protocol.alignment.alignment_of`: the
    authored form's tokens are located inside the answer's content tokens
    (whitespace-only pieces skipped, as :func:`column_first_token_id` does),
    and ``absent`` is refused as ``alignment_missing``, naming **both surface
    forms** decoded. A tokenizer under which the two forms coincide (a
    sentencepiece family, where ``"Saturday"`` *is* the ``▁Saturday`` piece)
    aligns ``one_to_one`` and nothing fires; a bare table value says nothing
    about the model's form and is not checked; ``space_prefixed`` and
    ``auto`` are not this error class (``auto`` has its own refusal).
    """
    kind = str(metric.kind)
    if str(metric.token_form) != "bare":
        return
    # only a column carries what the data says: `groups` and `tokens` are
    # authored literals and a `kl`/`js` target is a read (§2.10,
    # `metric_column_fields`); a `js` `restrict` column is an answer column
    # like any other and is checked the same way
    for field, column in metric_column_fields(metric).items():
        values: list[str] = []
        for row in rows:
            held = row.get(column)  # a missing column is compute_metric's refusal
            if isinstance(held, list):
                values.extend(str(v) for v in held)
            elif held is not None:
                values.append(str(held))
        for value in dict.fromkeys(values):
            if not value.startswith(" "):
                continue
            bare = value.lstrip(" ")
            answer_ids = [
                int(t) for t in tokenizer.encode(value, add_special_tokens=False)
            ]
            content = [t for t in answer_ids if tokenizer.decode([t]).strip()]
            authored = [
                int(t) for t in tokenizer.encode(bare, add_special_tokens=False)
            ]
            observed = alignment_of(
                token_runs(authored, content), (tuple(range(len(content))),)
            )

            def pieces(ids: Sequence[int]) -> list[str]:
                return [tokenizer.decode([t]) for t in ids]

            refuse_unalignable(
                observed,
                f"{where} ({kind}.{field}): the table carries {value!r} "
                f"space-prefixed, and token_form='bare' scores the bare form "
                f"{bare!r} — under this tokenizer the two are different tokens: "
                f"bare {bare!r} → {pieces(authored)} vs space-prefixed {value!r} → "
                f"{pieces(answer_ids)}. The metric would credit a token the model "
                "never emits for this answer. Set token_form to 'space_prefixed', "
                "or strip the space from the table if the model's answer really "
                "is bare",
            )


def column_token_ids(
    tokenizer: Any,
    values: Sequence[Any],
    *,
    token_form: str = "auto",
    where: str = "metric column",
    vocabulary_size: int | None = None,
) -> list[int]:
    """Resolve a whole metric column, refusing if ``auto`` would have to guess.

    An enclosing metric computation may supply its tokenizer vocabulary size.
    It is a call-local snapshot including added tokens, never a model logit width
    or metadata retained across tokenizer mutations.
    """
    if token_form == "id":
        size = len(tokenizer) if vocabulary_size is None else vocabulary_size
        return [_checked_token_id(value, size) for value in values]
    if token_form == "auto":
        refuse_ambiguous_auto(tokenizer, values, where=where)
    return [column_token_id(tokenizer, v, token_form=token_form) for v in values]


def column_first_token_id(
    tokenizer: Any, value: str, *, token_form: str = "auto"
) -> int:
    """The first *content* token id of a value — ``match``'s ``first_token``
    mode.

    A single-token value resolves exactly as :func:`column_token_id` does, so
    ``first_token`` is a strict generalization of ``exact``. A multi-token
    value resolves to the first piece that carries text: a sentencepiece family
    can encode a leading space as its own ``▁`` piece, and crediting *that*
    would score every space-prefixed answer alike — the first piece an argmax
    can distinguish is the one after it, which is also what the model emits in
    context.

    📐 Which values trigger that is tokenizer- *and* version-dependent, so the
    skip is written as a property of the piece (does it decode to text?) rather
    than of a known value. Under transformers 4.x the tiny Llama tokenizer
    emitted the lone ``▁`` for any space-prefixed word (``" Thursday"`` →
    ``▁ Th urs day``); 5.16.1 dropped that legacy dummy prefix, so
    ``" Thursday"`` is now ``Th urs day`` — and the skip is what makes this
    function return the same id, ``Th``, across the bump. It is not dead code:
    5.16.1 still emits the lone ``▁`` whenever the first character has no
    merged ``▁X`` piece — digits, non-Latin scripts, emoji, ligatures
    (``" 3.14"`` → ``▁ 3 . 1 4``) — and byte-level BPE families still split a
    whitespace run off the front (gpt2 ``"  ?"`` → ``' '`` + ``' ?'``).

    What this cannot know is whether the table's answer space is
    first-token-distinct — two answers sharing a first piece would both score.
    That is a property of the dataset, checked where the dataset is built."""
    candidates = _candidates(value, token_form)
    encoded = [
        tokenizer.encode(candidate, add_special_tokens=False)
        for candidate in candidates
    ]
    for ids in encoded:
        if len(ids) == 1:
            return int(ids[0])
    if token_form == "auto" and len(candidates) > 1:
        # The case `_ambiguous_under_auto` structurally cannot see: it asks
        # whether the two forms are *different single tokens*, so a value that
        # is single-token in neither form is not "ambiguous" by its definition
        # and nothing fired — while `auto` still silently picked a form, and
        # under `first_token` the form decides which piece gets credited.
        firsts = [_first_content_id(tokenizer, ids) for ids in encoded]
        distinct = {f for f in firsts if f is not None}
        if len(distinct) > 1:
            pieces = " vs ".join(
                f"{candidate!r} → {[tokenizer.decode([int(i)]) for i in ids]}"
                for candidate, ids in zip(candidates, encoded)
            )
            raise ProtocolError(
                "P2",
                f"metric column value {value!r} is multi-token and its two "
                f"surface forms credit different first tokens ({pieces}) — "
                "token_form='auto' cannot know which one the model emits; set "
                "the metric's token_form to 'bare' or 'space_prefixed'",
            )
    for ids in encoded:
        first = _first_content_id(tokenizer, ids)
        if first is not None:
            return first
    raise ProtocolError(
        "P2",
        f"metric column value {value!r} encodes to no content tokens under "
        "this tokenizer — nothing to compare an argmax against",
    )


def _first_content_id(tokenizer: Any, ids: Sequence[int]) -> int | None:
    """The first piece of ``ids`` that decodes to text, or ``None``.

    Written as a property of the piece rather than of a known value because
    which values carry a whitespace-only first piece is tokenizer- and
    version-dependent (see :func:`column_first_token_id`).
    """
    for token_id in ids:
        if tokenizer.decode([int(token_id)]).strip():
            return int(token_id)
    return None


def _refuse_indistinct_first_tokens(
    groups: Sequence[Sequence[str]],
    resolved: Sequence[set[int]],
    *,
    where: str,
) -> None:
    """Refuse a ``first_token`` metric whose answer space is not first-token
    distinct.

    ``first_token`` credits a *prefix*, so it means "the model answered" only
    when different answers begin with different tokens. Where they do not, the
    metric over-credits silently: ``" 85"`` tokenizes to ``[220, "8", "5"]`` on
    Qwen, so a model emitting ``87`` scores 1.000 against an expected ``85``.
    Nothing in a run's saved outputs says so — the number is simply wrong.

    The distinctness of a *dataset's* answer space is checked where the dataset
    is built (:func:`column_first_token_id`); this is the same claim asserted
    against the rows a metric was actually handed, which is the last place it
    can be checked before a number is produced.
    """
    by_first: dict[int, set[str]] = {}
    for forms, ids in zip(groups, resolved):
        answer = "/".join(sorted(form.strip() for form in forms))
        for token_id in ids:
            by_first.setdefault(token_id, set()).add(answer)
    collisions = {
        token_id: sorted(answers)
        for token_id, answers in by_first.items()
        if len(answers) > 1
    }
    if not collisions:
        return
    shown = ", ".join(
        f"{answers} share first token {token_id}"
        for token_id, answers in list(collisions.items())[:3]
    )
    raise ProtocolError(
        "P2",
        f"{where}: mode='first_token' credits a prefix, and this answer space "
        f"is not first-token distinct ({shown}) — the metric would score a "
        "wrong answer as correct. Use mode='exact', or an answer space whose "
        "members begin with different tokens",
    )


def _last_pos_rows(value: torch.Tensor) -> torch.Tensor:
    """A read at one position arrives as (batch, 1, width); squeeze it.

    ``width`` is the vocabulary for an ``lm_head`` read and the site's own
    width otherwise — only ``top_k`` reduces the latter (every other kind is
    bound to a vocabulary projection by validation)."""
    if value.dim() == 3:
        if value.shape[1] != 1:
            raise ProtocolError(
                "P2",
                f"metric read spans {value.shape[1]} positions — metrics reduce "
                "one position per example",
            )
        return value[:, 0, :]
    return value


def _top_k(
    metric: MetricSpec,
    dense: torch.Tensor,
    tokenizer: Any,
    *,
    vocab_axis: bool,
) -> list[dict[str, Any]]:
    """``top_k`` over one read's rows — the reduction that happens **where the
    rows are gathered**, so a 100k-latent SAE code never reaches disk.

    ``by`` (mandatory, §2.10) is the ranking rule, and it is the author's call
    because only the author knows what the axis is: a vocabulary projection
    has no meaningful negative entries, a residual stream and a signed feature
    code do.

    The emitted columns have **fixed identities**, so a column never means one
    thing in one document and another in the next; a column is absent rather
    than reinterpreted:

    ==========  =====================================  ====================
    column      meaning                                emitted when
    ==========  =====================================  ====================
    ``indices`` index along the read's last axis       always
    ``tokens``  that index decoded as a token string   the read is a plain lm_head tap
    ``values``  the **raw** read value at that index   always
    ``probs``   softmax probability over the vocab     ``by == "prob"``
    ==========  =====================================  ====================

    ``values`` is always the raw value — a logit under ``by: "prob"``, not the
    probability — so a downstream reader never has to know the ranking rule to
    know what it is holding. The normalized number lives in its own column.
    """
    k = metric.fields["k"]
    assert isinstance(k, int)  # parse guarantees the shape
    by = str(metric.fields["by"])
    width = int(dense.shape[-1])
    if k < 1 or k > width:
        raise ProtocolError(
            "P2",
            f"top_k asks for k={k} of a read {width} wide — k must be in [1, width]",
        )
    if by == VOCAB_TOP_K_RANKING:
        # validation binds `prob` to an lm_head read (§2.10): a softmax across
        # neurons or latents normalizes over an axis that is not an event space
        scores = torch.softmax(dense, dim=-1)
    elif by == "abs_value":
        scores = dense.abs()
    else:
        scores = dense
    top = scores.topk(k, dim=-1)
    out: list[dict[str, Any]] = []
    for i in range(dense.shape[0]):
        indices = [int(j) for j in top.indices[i]]
        entry: dict[str, Any] = {"indices": indices}
        if vocab_axis:
            entry["tokens"] = [tokenizer.decode([j]) for j in indices]
        entry["values"] = [float(dense[i, j]) for j in indices]
        if by == VOCAB_TOP_K_RANKING:
            entry["probs"] = [float(p) for p in top.values[i]]
        out.append(entry)
    return out


def restrict_token_ids(
    metric: MetricSpec,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    vocabulary_size: int | None = None,
) -> list[list[int]] | None:
    """The per-row answer set a ``js`` is restricted to (§2.10 ``restrict``),
    as token ids — ``None`` when the metric is unrestricted.

    A column form yields each row's own list; a literal list yields the same
    ids for every row. Every string resolves under the metric's ``token_form``
    exactly as an answer column does (:func:`column_token_ids`), and two
    strings that land on one id are refused rather than counted twice — the
    ``class_probs`` rule, for the same reason: the restricted softmax would
    give that answer double mass. Rows whose column is empty are not here:
    :func:`excluded_rows` took them out before the reduction ran.

    Under ``token_form: id`` the values are the ids themselves and every row's
    set is checked against one vocabulary bound — the enclosing metric
    computation's snapshot when it passes ``vocabulary_size``, otherwise one
    query here (the training objective's call); never one per row.
    """
    restrict = metric.fields.get("restrict")
    if restrict is None:
        return None
    token_form = str(metric.token_form)
    where = f"metric {metric.kind}.restrict"
    if token_form == "id" and vocabulary_size is None:
        vocabulary_size = len(tokenizer)

    def resolve(values: Sequence[Any]) -> list[int]:
        by_id: dict[int, Any] = {}
        for value, token in zip(
            values,
            column_token_ids(
                tokenizer,
                values,
                token_form=token_form,
                where=where,
                vocabulary_size=vocabulary_size,
            ),
        ):
            if token in by_id:
                raise ProtocolError(
                    "P2",
                    f"{where}: {value!r} and {by_id[token]!r} both resolve to token "
                    f"id {token} under token_form={token_form!r}, and the restricted "
                    "softmax would give that answer double mass. List each answer "
                    "once (a leading space is normalized away before token_form "
                    "decides the form, §2.10)",
                )
            by_id[token] = value
        return list(by_id)

    def spelled(values: Sequence[Any]) -> list[Any]:
        # under ``id`` the values *are* the ids (an integer is refused as a
        # string, `_checked_token_id`); every other form spells a surface string
        return list(values) if token_form == "id" else [str(v) for v in values]

    if isinstance(restrict, str):
        return [resolve(spelled(row[restrict])) for row in rows]
    shared = resolve(spelled(restrict))
    return [list(shared) for _ in rows]


def js_divergence(
    of_logits: torch.Tensor,
    target_logits: torch.Tensor,
    restrict_ids: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Per-row Jensen–Shannon divergence, in nats, between the distributions
    two ``(batch, vocab)`` logit tensors define (§2.10 ``js``)::

        JS(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m),   m = ½ (p + q)

    Symmetric, and bounded by ``ln 2``. With ``restrict_ids`` each row's two
    distributions are first **restricted to its answer ids and renormalised**
    — a ``log_softmax`` over the sliced logits, which is exact and needs no
    ``eps``. Differentiable in both arguments, so the same function is the
    objective term (``shared.training.objective.metric_tensor``) and the saved
    record — one arithmetic, one unit."""
    if restrict_ids is None:
        return _js_from_log_probs(
            torch.log_softmax(of_logits, dim=-1),
            torch.log_softmax(target_logits, dim=-1),
        )
    values: list[torch.Tensor] = []
    for i, ids in enumerate(restrict_ids):
        index = torch.as_tensor(list(ids), dtype=torch.long, device=of_logits.device)
        p = torch.log_softmax(of_logits[i].index_select(-1, index), dim=-1)
        q = torch.log_softmax(target_logits[i].index_select(-1, index), dim=-1)
        values.append(_js_from_log_probs(p, q))
    return torch.stack(values)


def _js_from_log_probs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """JS over the last axis of two log-probability tensors; ``m`` is formed in
    log space (``logaddexp − ln 2``) so a near-zero probability never
    underflows the log."""
    m = torch.logaddexp(p, q) - math.log(2.0)
    return 0.5 * (p.exp() * (p - m)).sum(dim=-1) + 0.5 * (q.exp() * (q - m)).sum(dim=-1)


def excluded_rows(
    metric: MetricSpec, rows: Sequence[Mapping[str, Any]], denominator_key: str
) -> dict[int, Unavailable]:
    """The rows a metric cannot score because the table carries **no answer**
    for them: for every column the kind names, a row whose value is ``null``,
    absent, or an empty list of forms (spec §2.10 "Eligibility").

    A structural fact of the data the document could not know (§4.1): the
    column exists — ``validate --data`` checked that — but this row has
    nothing in it, so the row is an **excluded measurement** under
    ``alignment_missing`` (the authored answer has no counterpart in the
    data), not a refusal of the run and not a score of ``"None"``. Keyed by
    row index; every value is the typed ``unavailable`` with the cell's
    ``denominator_key``, so it flows into the denominator unchanged. The
    ``validate --data`` twin — the *maximum* eligible count a
    ``minimum_count`` is held to — is ``loader.check_data_columns``.
    """
    kind = str(metric.kind)
    out: dict[int, Unavailable] = {}
    # the predicate `validate --data` counted the maximum eligible rows with
    # (§2.10, `metric_column_fields`): a `kl`/`js` target is a read and
    # excludes nothing, a `js` `restrict` column excludes its empty rows
    for field, column in metric_column_fields(metric).items():
        for i, row in enumerate(rows):
            if i in out:
                continue
            held = row.get(column)
            if held is None or (isinstance(held, list) and not held):
                out[i] = unavailable(
                    "alignment_missing",
                    f"row {i} carries no value in column {column!r} "
                    f"({kind}.{field}) — nothing to score it against",
                    denominator_key,
                )
    return out


#: The kinds whose per-example value **selects** entries of the projection at
#: the answer token ids and computes on nothing else. For these, gathering
#: the entries where the value sits — the device — and reducing the gathered
#: values on the CPU in float is :func:`compute_metric`'s arithmetic to the
#: bit: an upcast commutes with a selection, and the per-example operation is
#: the same 0-d float op. A kind with a softmax, a log-sum-exp or an argmax
#: over the vocabulary is not in the set — reduced on the device it rounds
#: differently — and keeps the whole-vocabulary CPU path.
GATHERED_KINDS = frozenset({"logit_diff", "soft_accuracy", "token_logit"})

_GATHERED_FIELDS: Mapping[str, tuple[str, ...]] = {
    "logit_diff": ("a", "b"),
    "soft_accuracy": ("a", "b"),
    "token_logit": ("token",),
}


def metric_token_ids(
    metric: MetricSpec, rows: Sequence[Mapping[str, Any]], tokenizer: Any
) -> dict[str, list[int]]:
    """Each answer column of a :data:`GATHERED_KINDS` kind resolved to token
    ids over the rows that carry answers — the excluded rows
    (:func:`excluded_rows`) left out, as :func:`compute_metric` leaves them
    out before it resolves — under the metric's ``token_form``, exactly as
    :func:`compute_metric` resolves them. A caller whose rows never change
    (a fit's eval executor) resolves once and hands the result to
    :func:`gathered_metric` on every pass."""
    kind = str(metric.kind)
    if kind not in GATHERED_KINDS:
        raise ValueError(f"metric kind {kind!r} is not gathered at token ids")
    token_form = str(metric.token_form)
    excluded = excluded_rows(metric, rows, kind)
    kept = [row for i, row in enumerate(rows) if i not in excluded]
    out: dict[str, list[int]] = {}
    for field in _GATHERED_FIELDS[kind]:
        column = str(metric.fields[field])
        values: list[Any] = [row[column] for row in kept]
        if token_form != "id":
            values = [str(value) for value in values]
        out[field] = column_token_ids(
            tokenizer, values, token_form=token_form, where=f"metric {kind}.{field}"
        )
    return out


def gathered_metric(
    metric: MetricSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    token_ids: Mapping[str, Sequence[int]] | None = None,
    denominator_key: str | None = None,
) -> list[Any]:
    """:func:`compute_metric` for a :data:`GATHERED_KINDS` kind over a value
    wherever it sits: the excluded rows as :func:`compute_metric` excludes
    them, the answer entries gathered from the read on its own device, the
    one or two columns copied to the CPU, and the kind's per-example
    arithmetic in float exactly as :func:`compute_metric` writes it — so the
    list is the same, entry for entry, without the vocabulary ever leaving
    the device. ``token_ids`` is :func:`metric_token_ids`' result when the
    caller resolved it once. No ``vocab_axis``: validation holds a gathered
    kind's ``of`` read to a plain ``lm_head`` tap (``metric_reads_vocabulary``
    — no featurizer, no ``dims``; ``protocol/validate.py`` exempts only
    ``kl`` and ``top_k``), so the value's last axis is the vocabulary and
    a raw token id indexes it."""
    kind = str(metric.kind)
    if kind not in GATHERED_KINDS:
        raise ValueError(f"metric kind {kind!r} is not gathered at token ids")
    excluded = excluded_rows(metric, rows, denominator_key or kind)
    keep = [i for i in range(len(rows)) if i not in excluded]
    if not keep:  # every row excluded: nothing to reduce, nothing raised
        return [excluded[i] for i in range(len(rows))]
    ids = (
        token_ids
        if token_ids is not None
        else metric_token_ids(metric, rows, tokenizer)
    )
    dense = _last_pos_rows(of_value)
    if len(keep) != len(rows):
        dense = dense[torch.tensor(keep, dtype=torch.long, device=dense.device)]
    fields = _GATHERED_FIELDS[kind]
    for field in fields:
        if len(ids[field]) != len(keep):
            raise ValueError(
                f"metric {kind}.{field}: {len(ids[field])} token ids for "
                f"{len(keep)} rows"
            )
    with torch.no_grad():
        # every answer column in one gather and one copy — one host wait per
        # metric, not per column — then the CPU path's upcast: the fp32 value
        # `compute_metric` reads at `logits[i, id]`, entry for entry
        index = torch.tensor(
            [list(ids[field]) for field in fields],
            dtype=torch.long,
            device=dense.device,
        )  # (fields, keep)
        # (keep, fields) off the device contiguous, then split by column
        taken = dense.gather(1, index.t()).cpu().t().float()  # (fields, keep)
    columns = {field: taken[i] for i, field in enumerate(fields)}
    if kind == "logit_diff":
        a, b = columns["a"], columns["b"]
        scored = [float(a[i] - b[i]) for i in range(len(keep))]
    elif kind == "soft_accuracy":
        # `_compute_metric`'s line, `.float()` included (already fp32 here)
        a, b = columns["a"], columns["b"]
        scored = [
            float(torch.sigmoid(a[i].float() - b[i].float())) for i in range(len(keep))
        ]
    else:
        token = columns["token"]
        scored = [float(token[i]) for i in range(len(keep))]
    it = iter(scored)
    return [excluded[i] if i in excluded else next(it) for i in range(len(rows))]


def compute_metric(
    metric: MetricSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None = None,
    vocab_axis: bool = True,
    denominator_key: str | None = None,
) -> list[Any]:
    """One metric over one read's value, per example.

    ``vocab_axis`` says whether the read's last axis is the vocabulary — i.e.
    whether it is a plain ``lm_head`` tap, with no featurizer or ``dims``
    taking the value out of token-id space (:func:`~causalab.protocol.schema
    .metric_reads_vocabulary`). Every kind but ``top_k`` is bound to a
    vocabulary projection by validation, so the default is ``True``; ``top_k``
    is the one kind that also runs over a residual stream, an MLP activation
    or a featurizer's latents, and it needs to know because a token id is
    worth decoding and a neuron index is not.

    A row the table carries no answer for (:func:`excluded_rows`) comes back
    as the typed :class:`~causalab.protocol.resolution.Unavailable` in its
    place — an excluded measurement, keyed under ``denominator_key`` (the
    metric's cell key; its kind when the caller has no coordinates) — and
    the kind is computed over the other rows only, so no excluded row ever
    reaches a mean (§2.10 "Eligibility").
    """
    excluded = excluded_rows(metric, rows, denominator_key or str(metric.kind))
    if excluded:
        keep = [i for i in range(len(rows)) if i not in excluded]
        if not keep:  # every row excluded: nothing to reduce, nothing raised
            return [excluded[i] for i in range(len(rows))]
        index = torch.tensor(keep, dtype=torch.long, device=of_value.device)
        scored = _compute_metric(
            metric,
            _last_pos_rows(of_value)[index],
            [rows[i] for i in keep],
            tokenizer,
            target_value=(
                _last_pos_rows(target_value)[index]
                if target_value is not None
                else None
            ),
            vocab_axis=vocab_axis,
        )
        it = iter(scored)
        return [excluded[i] if i in excluded else next(it) for i in range(len(rows))]
    return _compute_metric(
        metric,
        of_value,
        rows,
        tokenizer,
        target_value=target_value,
        vocab_axis=vocab_axis,
    )


def _compute_metric(
    metric: MetricSpec,
    of_value: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_value: torch.Tensor | None,
    vocab_axis: bool,
) -> list[Any]:
    """The kinds themselves, over rows that all carry their answers."""
    # `dense` is the read's value at the addressed position, (batch, width).
    # Every kind but `top_k` is bound to an lm_head read, so for those it is
    # the vocabulary projection and reads as `logits` below.
    dense = _last_pos_rows(of_value).float()
    logits = dense
    kind = str(metric.kind)
    # §2.10: how this metric's string answers become token ids. `auto` is the
    # space-prefixed-first default every pre-token_form document gets.
    token_form = str(metric.token_form)
    vocabulary_size = len(tokenizer) if token_form == "id" else None

    def token_ids(values: Sequence[str], field: str) -> list[int]:
        return column_token_ids(
            tokenizer,
            values,
            token_form=token_form,
            where=f"metric {kind}.{field}",
            vocabulary_size=vocabulary_size,
        )

    def distinct_token_ids(
        values: Sequence[str], field: str, consequence: str
    ) -> list[int]:
        """Resolve a list of *literal* token strings, refusing when two of
        them land on one id.

        The list kinds (``class_probs``, ``token_logits``) index the
        projection by every member, so a collision is not harmless
        redundancy: ``consequence`` says what the kind would have reported.
        The ``['X', ' X']`` idiom is the usual cause and is inert anyway — a
        leading space is normalized away before ``token_form`` decides the
        form (§2.10).
        """
        by_id: dict[int, str] = {}
        for value, token in zip(values, token_ids(values, field)):
            if token in by_id:
                raise ProtocolError(
                    "P2",
                    f"metric {kind}.{field}: {value!r} and {by_id[token]!r} both "
                    f"resolve to token id {token} under token_form={token_form!r}, "
                    f"and {consequence}. The ['X', ' X'] idiom is the usual cause "
                    "and is inert anyway: a leading space is normalized away "
                    "before token_form decides the form (§2.10). List each "
                    "answer once",
                )
            by_id[token] = value
        return list(by_id)

    def raw_column(field: str) -> list[Any]:
        # a row with no value here is an excluded measurement, taken out by
        # `excluded_rows` before this runs — never a refusal, never "None"
        name = str(metric.fields[field])
        return [row[name] for row in rows]

    def column(field: str) -> list[Any]:
        values = raw_column(field)
        return values if token_form == "id" else [str(value) for value in values]

    def form_groups(field: str) -> list[list[str]]:
        """One row's expected forms: a list column is a group of equivalent
        surface forms, a scalar is a group of one (§2.10). An empty group is
        an excluded row (`excluded_rows`), so every group here has a form."""
        return [
            [str(v) for v in value] if isinstance(value, list) else [str(value)]
            for value in raw_column(field)
        ]

    if kind == "logit_diff":
        a_ids = token_ids(column("a"), "a")
        b_ids = token_ids(column("b"), "b")
        return [
            float(logits[i, a] - logits[i, b])
            for i, (a, b) in enumerate(zip(a_ids, b_ids))
        ]
    if kind == "soft_accuracy":
        # σ of the same margin: the differentiable stand-in for "a beats b"
        # (a soft-accuracy objective), bounded so a runaway margin on
        # one row cannot dominate a mean the way a raw logit_diff can. Computed
        # in float so the saved value and the objective twin agree to the bit.
        a_ids = token_ids(column("a"), "a")
        b_ids = token_ids(column("b"), "b")
        return [
            float(torch.sigmoid(logits[i, a].float() - logits[i, b].float()))
            for i, (a, b) in enumerate(zip(a_ids, b_ids))
        ]
    if kind == "token_logit":
        ids = token_ids(column("token"), "token")
        return [float(logits[i, t]) for i, t in enumerate(ids)]
    if kind == "cross_entropy":
        ids = token_ids(column("target"), "target")
        log_probs = torch.log_softmax(logits, dim=-1)
        return [float(-log_probs[i, t]) for i, t in enumerate(ids)]
    if kind == "kl":
        if target_value is None:
            raise ProtocolError("P2", "kl needs its target read's value")
        p = torch.log_softmax(logits, dim=-1)
        q = torch.log_softmax(_last_pos_rows(target_value).float(), dim=-1)
        kl = (p.exp() * (p - q)).sum(dim=-1)
        return [float(v) for v in kl]
    if kind == "js":
        if target_value is None:
            raise ProtocolError("P2", "js needs its target read's value")
        values = js_divergence(
            logits,
            _last_pos_rows(target_value).float(),
            restrict_token_ids(
                metric, rows, tokenizer, vocabulary_size=vocabulary_size
            ),
        )
        return [float(v) for v in values]
    if kind == "match":
        if token_form == "id":
            expected = raw_column("expected")
            resolved_ids = []
            for value in expected:
                forms = value if isinstance(value, list) else [value]
                if not forms:
                    raise ProtocolError(
                        "P2", "expected token ID group must not be empty"
                    )
                resolved_ids.append(set(token_ids(forms, "expected")))
            argmax = logits.argmax(dim=-1)
            return [float(int(argmax[i]) in ids) for i, ids in enumerate(resolved_ids)]
        # `mode` decides whether a form's first token counts (§2.10);
        # `token_form` decides which surface form resolves. Independent knobs.
        mode = str(metric.fields.get("mode", "exact"))
        resolve = column_first_token_id if mode == "first_token" else column_token_id
        groups = form_groups("expected")
        if token_form == "auto":
            # `match` resolves form by form, so it never reached the aggregated
            # column check — the one kind whose answers are most often bare
            # punctuation, which is where `auto` guesses worst.
            refuse_ambiguous_auto(
                tokenizer,
                [form for forms in groups for form in forms],
                where=f"metric {kind}.expected",
            )
        resolved = [
            {resolve(tokenizer, f, token_form=token_form) for f in forms}
            for forms in groups
        ]
        if mode == "first_token":
            _refuse_indistinct_first_tokens(groups, resolved, where=f"metric {kind}")
        argmax = logits.argmax(dim=-1)
        return [float(int(argmax[i]) in ids) for i, ids in enumerate(resolved)]
    if kind == "top_k":
        return _top_k(metric, dense, tokenizer, vocab_axis=vocab_axis)
    if kind == "class_probs":
        # ⚠️ `groups` is the one value field that is NOT a column name: a class
        # is a property of the answer *space*, one for the whole run, so the
        # members are literal token strings (§2.10). Every other kind's `a`,
        # `b`, `token`, `target`, `expected` name a dataset column.
        groups = metric.fields["groups"]
        if not isinstance(groups, Mapping):
            raise ProtocolError(
                "P2",
                "class_probs groups is a {name: [token strings]} mapping — "
                "literal tokens, not a dataset column name",
            )
        probs = torch.softmax(logits, dim=-1)
        group_ids = {
            name: distinct_token_ids(
                [str(v) for v in members],
                f"groups.{name}",
                "this metric sums a group's ids — so the class would count that "
                "token twice and report a 'probability' above 1",
            )
            for name, members in groups.items()
        }
        return [
            {name: float(probs[i, ids].sum()) for name, ids in group_ids.items()}
            for i in range(logits.shape[0])
        ]
    if kind == "token_logits":
        # `tokens` is literal token strings for the same reason `groups` is:
        # the answer space is a property of the run, not of a row (§2.10). The
        # result mirrors `top_k`'s fixed column identities — `indices` are the
        # ids, `tokens` the ids decoded, `values` the raw logits — so a reader
        # of either table holds the same three things under the same names.
        ids = distinct_token_ids(
            [str(v) for v in metric.fields["tokens"]],
            "tokens",
            "this metric reports one logit per listed token — so one row of the "
            "projection would appear twice under two names",
        )
        decoded = [tokenizer.decode([t]) for t in ids]
        return [
            {
                "indices": ids,
                "tokens": decoded,
                "values": [float(logits[i, t]) for t in ids],
            }
            for i in range(logits.shape[0])
        ]
    if kind == "decode":
        raise ProtocolError(
            "P2",
            "'decode' reduces the tokens a decode produced, so it binds to a "
            "read in the continuation frame (§2.3) — validation refuses it "
            "anywhere else",
        )
    raise ProtocolError("P4", f"unknown metric kind {kind!r}")


def compute_windowed_metric(
    metric: MetricSpec,
    windows: Sequence[torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_windows: Sequence[torch.Tensor] | None = None,
    generated_ids: Sequence[Sequence[int]] | None = None,
    vocab_axis: bool = True,
) -> list[list[Any]]:
    """One metric over a read that addresses **several** positions per row.

    ``windows[i]`` is example ``i``'s value at the positions it addresses,
    ``(positions_i, vocab)`` — empty when the row addressed none, which in
    the continuation frame is a result (§2.3), not a misalignment. Returns
    the same shape: one list of values per example.

    Every ``distribution`` kind reduces **per position**, and does so
    through :func:`compute_metric` on the flattened positions — one
    implementation of the kinds, not two. ``ids`` kinds never look at
    ``windows`` at all: they consume ``generated_ids``, which is why a text
    probe obliges no vocabulary projection (§8).
    """
    kind = str(metric.kind)
    if kind in WHOLE_WINDOW_METRIC_KINDS:
        if generated_ids is None:
            raise ProtocolError(
                "P2", f"metric kind {kind!r} needs the decode's token ids"
            )
        if kind == "decode":
            return [
                [
                    tokenizer.decode(
                        list(ids),
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                ]
                if len(ids)
                else []
                for ids in generated_ids
            ]
        raise ProtocolError("P4", f"unhandled whole-window metric kind {kind!r}")

    counts = [int(window.shape[0]) for window in windows]
    if not any(counts):
        return [[] for _ in windows]
    flat = torch.cat([w for w in windows if w.shape[0]], dim=0)
    flat_rows = [rows[i] for i, count in enumerate(counts) for _ in range(count)]
    flat_target = None
    if target_windows is not None:
        target_counts = [int(w.shape[0]) for w in target_windows]
        if target_counts != counts:
            raise ProtocolError(
                "P2",
                f"{kind} compares reads addressing different position counts "
                f"({counts} vs {target_counts}) — a comparison needs a "
                "position-for-position pairing",
            )
        flat_target = torch.cat([w for w in target_windows if w.shape[0]], dim=0)
    values = compute_metric(
        metric,
        flat,
        flat_rows,
        tokenizer,
        target_value=flat_target,
        vocab_axis=vocab_axis,
    )
    out: list[list[Any]] = []
    cursor = 0
    for count in counts:
        out.append(values[cursor : cursor + count])
        cursor += count
    return out
