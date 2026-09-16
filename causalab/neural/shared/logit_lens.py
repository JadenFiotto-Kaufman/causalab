"""Logit lens over a shared residual harvest, without a transformer forward.

Use the loaded PyTorch bundle's actual final normalization and vocabulary head.
Projection is chunked over positions; the returned table holds only top-k and
optional exact target scores. Inputs may be a tensor or a stamped harvest file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.errors import ProtocolError


def logit_lens(
    bundle: Any,
    activations: Any,
    *,
    k: int = 10,
    batch_positions: int = 128,
    target_ids: Sequence[int] | None = None,
    slot: str | None = None,
    entry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One record per flattened activation, preserving its leading-axis index.

    A tensor is caller-owned data. A file must identify the matching model,
    revision, precision and block_output site. Passing another model's harvest
    must fail before projecting. This API accepts a PyTorch ModelBundle; a
    tracing envoy is not an executable decoder module outside its trace.
    """
    import itertools
    import torch

    from causalab.io.step_io import read_tensor_with_identity
    from causalab.neural.shared.sites import _projection_width, resolve_site
    from causalab.protocol.resolve import check_artifact_identity
    from causalab.protocol.schema import SiteSpec

    if (
        type(k) is not int
        or k < 1
        or type(batch_positions) is not int
        or batch_positions < 1
    ):
        raise ValueError("k and batch_positions must be positive integers")
    if isinstance(activations, (str, Path)):
        activations, identity = read_tensor_with_identity(
            Path(activations), slot=slot, entry=entry
        )
        check_artifact_identity(
            identity,
            {
                "model_key": bundle.key,
                "model_revision": bundle.revision,
                "model_dtype": bundle.dtype,
            },
            what="logit lens harvest",
        )
        quantization = getattr(bundle, "quantization", None)
        expected_quantization = (
            json.dumps(quantization, sort_keys=True)
            if quantization is not None
            else None
        )
        if identity.get("model_quantization") != expected_quantization:
            raise ProtocolError(
                "P2", "logit lens harvest quantization does not match model"
            )
        site = json.loads(identity.get("site", "{}"))
        if site.get("component") != "block_output":
            raise ProtocolError("P2", "logit lens needs a block_output harvest")
    if not isinstance(activations, torch.Tensor) or activations.ndim < 2:
        raise ValueError("activations must have shape (..., hidden_size)")
    norm = resolve_site(bundle, SiteSpec(component="ln_final")).module
    head = resolve_site(bundle, SiteSpec(component="lm_head")).module
    if not isinstance(norm, torch.nn.Module) or not isinstance(head, torch.nn.Module):
        raise ValueError("logit lens requires executable PyTorch decoder modules")
    parameter = next(head.parameters())
    rows = activations.reshape(-1, activations.shape[-1])
    if target_ids is not None:
        if len(target_ids) != len(rows):
            raise ValueError("target_ids must supply one vocabulary ID per activation")
        # a target indexes the decoder's logit axis, whose width is the head's
        # — not the tokenizer's: a padded vocabulary has more columns than
        # tokens, and added tokens without a resized head have fewer
        width = _projection_width(head) or bundle.info.vocab_size
        if any(type(t) is not int or not 0 <= t < width for t in target_ids):
            raise ValueError(
                f"target_ids must be integers in [0, {width}), the decoder vocabulary"
            )
    coordinates = list(itertools.product(*(range(n) for n in activations.shape[:-1])))
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_positions):
            values = rows[start : start + batch_positions].to(
                parameter.device, parameter.dtype
            )
            logits = head(norm(values)).float()
            if k > logits.shape[-1]:
                raise ValueError("k exceeds the decoder vocabulary")
            normalizers = logits.logsumexp(dim=-1)
            top, indices = logits.topk(k, dim=-1)
            for local in range(len(values)):
                index = start + local
                ids = indices[local].tolist()
                record = {
                    "index": list(coordinates[index]),
                    "indices": ids,
                    "tokens": [bundle.tokenizer.decode([t]) for t in ids],
                    "logits": top[local].tolist(),
                    "probabilities": (top[local] - normalizers[local]).exp().tolist(),
                    "log_normalizer": float(normalizers[local]),
                }
                if target_ids is not None:
                    token = target_ids[index]
                    logit = float(logits[local, token])
                    log_probability = logit - float(normalizers[local])
                    record.update(
                        target_id=token,
                        target_logit=logit,
                        target_log_probability=log_probability,
                    )
                records.append(record)
    return records
