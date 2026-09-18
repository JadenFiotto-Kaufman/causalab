"""``metrics.metric_in_ids``: a metric and its rows with every answer
resolved to token ids reduce to the numbers the authored pair gives, to the
bit, with a ``VocabularySize`` where the tokenizer stood — for the saved
reductions (``compute_metric``, ``gathered_metric``) and for the objective's
``metric_tensor``. It is what lets a fit's spec carry its metrics as plain
data."""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.metrics import (
    VocabularySize,
    compute_metric,
    metric_in_ids,
)
from causalab.neural.shared.training.objective import metric_tensor
from causalab.protocol.errors import ProtocolError
from causalab.protocol.resolution import Unavailable
from causalab.protocol.schema import MetricSpec

from tests._helpers.train_docs import TINY_LLAMA

pytestmark = pytest.mark.unit

ROWS: list[dict[str, Any]] = [
    {"a": "one", "b": "two", "forms": ["one", "five"], "set": ["one", "two", "three"]},
    {"a": "three", "b": "four", "forms": "four", "set": ["three", "four"]},
    {"a": None, "b": "one", "forms": [], "set": []},  # no answer: excluded
    {"a": "two", "b": "three", "forms": ["two"], "set": ["one", "four"]},
]
ANSWERED = [row for row in ROWS if row["a"] is not None]

METRICS = {
    "logit_diff": MetricSpec("logit_diff", "logits", {"a": "a", "b": "b"}),
    "soft_accuracy": MetricSpec("soft_accuracy", "logits", {"a": "a", "b": "b"}),
    "token_logit": MetricSpec("token_logit", "logits", {"token": "a"}),
    "cross_entropy": MetricSpec("cross_entropy", "logits", {"target": "a"}),
    "match": MetricSpec("match", "logits", {"expected": "forms"}),
    "match-first": MetricSpec(
        "match", "logits", {"expected": "forms", "mode": "first_token"}
    ),
    "kl": MetricSpec("kl", "logits", {"target": "clean"}),
    "js": MetricSpec("js", "logits", {"target": "clean"}),
    "js-column": MetricSpec("js", "logits", {"target": "clean", "restrict": "set"}),
    "js-literal": MetricSpec(
        "js", "logits", {"target": "clean", "restrict": ["one", "two", "four"]}
    ),
}


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return load_model(TINY_LLAMA).tokenizer


def _values(tokenizer: Any, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    shape = (rows, 1, len(tokenizer))
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    )


@pytest.mark.parametrize("form", ["space_prefixed", "auto"])
@pytest.mark.parametrize("name", sorted(METRICS))
def test_the_resolved_pair_scores_as_the_authored_one(tokenizer, name, form):
    metric = dataclasses.replace(METRICS[name], token_form=form)
    of_value, target = _values(tokenizer, len(ROWS))
    want = compute_metric(metric, of_value, ROWS, tokenizer, target_value=target)

    resolved, rows = metric_in_ids(metric, ROWS, tokenizer)
    resolved, rows = pickle.loads(pickle.dumps((resolved, rows)))
    got = compute_metric(
        resolved, of_value, rows, VocabularySize(len(tokenizer)), target_value=target
    )

    assert len(got) == len(want) == len(ROWS)
    for ours, theirs in zip(got, want, strict=True):
        if isinstance(theirs, Unavailable):
            assert isinstance(ours, Unavailable)
        else:
            assert ours == theirs  # to the bit
    # the rows keep the metric's own columns and nothing else
    assert all(set(row) <= {"a", "b", "forms", "set"} for row in rows)
    # and every row that is scored holds ids, not strings
    assert not any(
        isinstance(value, str)
        for row, scored in zip(rows, want, strict=True)
        if not isinstance(scored, Unavailable)
        for cell in row.values()
        for value in (cell if isinstance(cell, list) else [cell])
    )


@pytest.mark.parametrize(
    "name", ["cross_entropy", "logit_diff", "soft_accuracy", "kl", "js-column"]
)
def test_the_objective_tensor_is_the_authored_one(tokenizer, name):
    metric = dataclasses.replace(METRICS[name], token_form="space_prefixed")
    of_value, target = _values(tokenizer, len(ANSWERED))
    want = metric_tensor(metric, of_value, ANSWERED, tokenizer, target_value=target)

    resolved, rows = metric_in_ids(metric, ANSWERED, tokenizer, eligible_only=False)
    got = metric_tensor(
        resolved,
        of_value,
        rows,
        VocabularySize(len(tokenizer)),
        target_value=target,
    )
    assert torch.equal(got, want)


def test_resolution_refuses_what_the_reduction_refuses(tokenizer):
    multi = [{"a": "one two three", "b": "two"}]
    with pytest.raises(ProtocolError, match="not a single token"):
        metric_in_ids(
            dataclasses.replace(METRICS["logit_diff"], token_form="bare"),
            multi,
            tokenizer,
        )
    ids = dataclasses.replace(METRICS["cross_entropy"], token_form="id")
    with pytest.raises(ProtocolError, match="integer token ID"):
        metric_in_ids(ids, [{"a": len(tokenizer)}], tokenizer)
    same, rows = metric_in_ids(ids, [{"a": 5, "other": "dropped"}], tokenizer)
    assert same is ids and rows == [{"a": 5}]
