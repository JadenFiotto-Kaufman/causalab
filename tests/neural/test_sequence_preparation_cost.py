"""Metadata queries are batch work; added-token bounds remain exact."""

from __future__ import annotations

import copy

import pytest

from causalab.neural.sequences import prepare_sequence, prepare_sequences
from causalab.neural.shared.prepared import encode_prepared
from causalab.protocol.errors import ProtocolError
from causalab.neural.engines.pytorch_hooks.loading import load_model
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.smoke


class CountedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.length_calls = 0
        self.vocabulary_calls = 0

    def __len__(self):
        self.length_calls += 1
        return len(self.tokenizer)

    def get_vocab(self):
        self.vocabulary_calls += 1
        return self.tokenizer.get_vocab()

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


def test_batch_preparation_and_audit_query_length_once():
    tokenizer = CountedTokenizer(copy.deepcopy(load_model(TINY_LLAMA).tokenizer))
    tokenizer.tokenizer.add_tokens(["<added-behavioral-token>"])
    added = len(tokenizer.tokenizer) - 1
    examples = [
        dict(
            prompt="An answer is",
            completion=[added],
            example_id=str(i),
            split="confirmation",
            prefix_condition="baseline_generated",
        )
        for i in range(5)
    ]
    prepared = prepare_sequences(tokenizer, examples)
    assert tokenizer.length_calls == 1
    assert tokenizer.vocabulary_calls == 2  # one shared digest plus mutation check
    expected = [prepare_sequence(tokenizer, **e) for e in examples]
    assert prepared == expected
    tokenizer.length_calls = 0
    encode_prepared(
        tokenizer,
        [r["input"] for r in prepared],
        [r["input_encoding"] for r in prepared],
        device="cpu",
    )
    assert tokenizer.length_calls == 1
    bad = copy.deepcopy(prepared)
    bad[0]["input_encoding"]["input_ids"][-1] = len(tokenizer.tokenizer)
    with pytest.raises(ProtocolError, match="invalid token IDs"):
        encode_prepared(
            tokenizer,
            [r["input"] for r in bad],
            [r["input_encoding"] for r in bad],
            device="cpu",
        )
    with pytest.raises(ProtocolError, match="integer vocabulary IDs"):
        prepare_sequences(
            tokenizer, [dict(examples[0], completion=[len(tokenizer.tokenizer)])]
        )


def test_alternate_eos_rejects_padding_in_prepared_rollout():
    tokenizer = load_model(TINY_LLAMA).tokenizer
    alternate = tokenizer.encode(" Friday", add_special_tokens=False)[0]
    example = dict(
        prompt="An answer is",
        completion=[20, alternate],
        example_id="one",
        split="confirmation",
        prefix_condition="baseline_generated",
        eos_token_ids=[alternate],
    )
    row = prepare_sequence(tokenizer, **example)
    assert row["targets"][-1]["is_eos"]
    with pytest.raises(ProtocolError, match="tokens after EOS"):
        prepare_sequence(tokenizer, **dict(example, completion=[alternate, 20]))


def test_digest_tracks_rules_and_specials_but_not_padding():
    from causalab.neural.shared.prepared import tokenizer_digest
    from types import SimpleNamespace
    import json

    state = {"padding": None, "truncation": None, "normalizer": None}
    tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps(state)),
        get_vocab=lambda: {"a": 0},
        special_tokens_map={},
    )  # optional chat/BOS/EOS attributes are deliberately absent
    before = tokenizer_digest(tokenizer)
    state["padding"] = {"length": 12}
    state["truncation"] = {"max_length": 12}
    assert tokenizer_digest(tokenizer) == before
    state["normalizer"] = {"type": "Lowercase"}
    assert tokenizer_digest(tokenizer) != before
    state["normalizer"] = None
    tokenizer.special_tokens_map["eos_token"] = "a"
    assert tokenizer_digest(tokenizer) != before


def test_batch_refuses_tokenizer_mutation_during_preparation(monkeypatch):
    import causalab.neural.sequences as sequences

    tokenizer = copy.deepcopy(load_model(TINY_LLAMA).tokenizer)
    original = sequences._prepare_sequence

    def mutating(*args, **kwargs):
        row = original(*args, **kwargs)
        tokenizer.add_tokens(["<changed-during-preparation>"])
        return row

    monkeypatch.setattr(sequences, "_prepare_sequence", mutating)
    with pytest.raises(ProtocolError, match="tokenizer changed"):
        prepare_sequences(
            tokenizer,
            [
                dict(
                    prompt="An answer is",
                    completion=[20],
                    example_id="one",
                    split="eval",
                )
            ],
        )
