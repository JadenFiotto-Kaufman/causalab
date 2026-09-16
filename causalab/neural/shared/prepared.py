"""Exact tokenized inputs for fixed-sequence causal experiments.

A text field may carry a sibling ``<field>_encoding`` record. Unlike decoding
and retokenizing a generated prefix, this preserves the model's actual IDs.
Preparation is explicit and tokenizer-bound; ordinary text inputs are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from causalab.protocol.errors import ProtocolError


def tokenizer_digest(tokenizer: Any) -> str:
    """Vocabulary, tokenization rules and specials, excluding batch padding state."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    rules = json.loads(backend.to_str()) if backend is not None else None
    if rules is not None:
        rules.pop("padding", None)
        rules.pop("truncation", None)
    body = {
        "class": type(tokenizer).__name__,
        "vocabulary": tokenizer.get_vocab(),
        "rules": rules,
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "add_bos_token": getattr(tokenizer, "add_bos_token", None),
        "add_eos_token": getattr(tokenizer, "add_eos_token", None),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def encoding_field(field: str) -> str:
    """input -> input_encoding; counterfactual_inputs[0] -> its aligned sibling."""
    column, bracket, index = field.partition("[")
    return f"{column}_encoding{bracket}{index}"


def encode_prepared(
    tokenizer: Any,
    texts: Sequence[str],
    records: Sequence[Any],
    *,
    device: str,
) -> Any:
    import torch

    from causalab.neural.shared.encoding import EncodedBatch, refuse_double_bos

    if not texts or len(texts) != len(records):
        raise ProtocolError(
            "P2", "prepared texts and encoding records must be nonempty and aligned"
        )
    fingerprint = tokenizer_digest(tokenizer)
    vocabulary_size = len(tokenizer)  # Includes added tokens; immutable in this batch.
    ids, offsets = [], []
    for text, record in zip(texts, records):
        if (
            not isinstance(record, Mapping)
            or record.get("version") != 1
            or record.get("tokenizer_digest") != fingerprint
            or record.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest()
        ):
            raise ProtocolError(
                "P2",
                "prepared encoding does not match its text or tokenizer; rebuild it",
            )
        tokens = record.get("input_ids")
        spans = record.get("offset_mapping")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int or not 0 <= t < vocabulary_size for t in tokens)
            or not isinstance(spans, list)
            or len(spans) != len(tokens)
            or any(
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(type(i) is not int for i in span)
                or not 0 <= span[0] <= span[1] <= len(text)
                for span in spans
            )
        ):
            raise ProtocolError(
                "P2", "prepared encoding has invalid token IDs or character offsets"
            )
        ids.append(tokens)
        offsets.append(spans)
    width = max(map(len, ids))
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ProtocolError("P2", "prepared batches need a tokenizer pad token")
    padded, masks, padded_offsets = [], [], []
    for tokens, spans in zip(ids, offsets):
        n = width - len(tokens)
        padded.append([pad] * n + tokens)
        masks.append([0] * n + [1] * len(tokens))
        padded_offsets.append(tuple([(0, 0)] * n + [tuple(s) for s in spans]))
    batch = EncodedBatch(
        texts=tuple(texts),
        input_ids=torch.tensor(padded, dtype=torch.long, device=device),
        attention_mask=torch.tensor(masks, dtype=torch.long, device=device),
        offset_mapping=tuple(padded_offsets),
        prefix_lengths=tuple(0 for _ in texts),
    )
    refuse_double_bos(tokenizer, batch.input_ids, batch.attention_mask)
    return batch
