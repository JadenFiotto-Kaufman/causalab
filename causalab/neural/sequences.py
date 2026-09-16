"""Prepare and analyze next-token targets with shared fixed-sequence forwards.

``prepare_sequence`` returns a JSON-serializable row. ``add_readouts`` adds
all targets to an ordinary intervention specification (protocol_version 3:
``header`` / ``model`` / ``data`` / ``method``) without adding intervened models.
Group rows by output length before authoring a document: negative indices then
align predictions even when prompt lengths differ. Targets remain separate
metrics, never a sequence-level first-token approximation.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping, Sequence

from causalab.neural.shared.prepared import tokenizer_digest
from causalab.protocol.errors import ProtocolError

PREFIX_CONDITIONS = ("correct", "baseline_generated")


def pair_sequences(
    base: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    target_alignment: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Pair frozen contexts and align the donor's expected tokens to base targets.

    Ordinal alignment is automatic only when lengths and semantic roles match.
    A changed tokenization requires an explicit one-to-one alignment. Unmatched
    targets belong in a separate cohort, not an invented target ID.
    """
    if (
        base["split"] != donor["split"]
        or base["prefix_condition"] != donor["prefix_condition"]
    ):
        raise ValueError("a pair must share its split and prefix condition")
    n, m = base["output_length"], donor["output_length"]
    if target_alignment is None:
        if n != m or [t["role"] for t in base["targets"]] != [
            t["role"] for t in donor["targets"]
        ]:
            raise ValueError(
                "different output layouts require explicit target_alignment"
            )
        target_alignment = {i: i for i in range(n)}
    if (
        set(target_alignment) != set(range(n))
        or any(type(i) is not int for i in target_alignment)
        or any(type(j) is not int or not 0 <= j < m for j in target_alignment.values())
        or len(set(target_alignment.values())) != n
    ):
        raise ValueError(
            "target_alignment must map each base target to a distinct donor target"
        )
    result = copy.deepcopy(dict(base))
    result["counterfactual_inputs"] = [donor["input"]]
    result["counterfactual_inputs_encoding"] = [copy.deepcopy(donor["input_encoding"])]
    result["counterfactual_example_id"] = donor["example_id"]
    result["counterfactual_output_length"] = m
    for i, j in target_alignment.items():
        result[f"cf_output_{i}"] = donor[f"output_{j}"]
        result["targets"][i]["donor_ordinal"] = j
    return result


def sequence_cohorts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Batch compatible output lengths and conditions, without copying per target.

    An example (including its appearances as a donor) cannot cross splits.
    Report missing semantic targets separately rather than padding a fake label.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    splits: dict[str, str] = {}
    for row in rows:
        for key in ("example_id", "counterfactual_example_id"):
            if key not in row:
                continue
            identity = str(row[key])
            split = str(row["split"])
            if identity in splits and splits[identity] != split:
                raise ValueError(f"example {identity!r} occurs in multiple splits")
            splits[identity] = split
        key = f"{row['prefix_condition']}_{row['split']}_{row['output_length']}_{row.get('counterfactual_output_length', 0)}"
        result.setdefault(key, []).append(copy.deepcopy(dict(row)))
    return result


def patching_protocol(
    model: Mapping[str, Any],
    dataset: str,
    output_length: int,
    *,
    component: str,
    bands: Sequence[Sequence[int]],
    positions: Sequence[Any],
    top_k: int = 10,
) -> dict[str, Any]:
    """One method campaign: unique bands/writes, all output targets per forward.

    Use singleton bands for residual scans and contiguous bands for attention
    or MLP scans. Positions are in the prepared sequence's frame; prompt end is
    ``-output_length-1``. Every pair must come from pair_sequences. The unchanged
    base is scored against the same donor targets as the patches (the null).
    """
    if component not in {"block_output", "attention_output", "mlp_output"}:
        raise ValueError("choose block_output, attention_output or mlp_output")
    if not bands or not positions or any(not band for band in bands):
        raise ValueError("bands and positions must be nonempty")
    if any(type(layer) is not int or layer < 0 for band in bands for layer in band):
        raise ValueError("bands must contain nonnegative layer indices")
    if any(len(set(band)) != len(band) for band in bands):
        raise ValueError("a band must not repeat a layer")
    if len({tuple(sorted(band)) for band in bands}) != len(bands):
        raise ValueError("duplicate bands repeat an identical intervention")
    method: dict[str, Any] = {
        "positions": {"tap": {"sweep": list(positions)}},
        "sites": {},
        "reads": {},
        "writes": {},
        "intervened_models": {},
    }
    for layer in sorted({i for band in bands for i in band}):
        site, read, write = f"site_{layer}", f"donor_{layer}", f"write_{layer}"
        method["sites"][site] = {"component": component, "layers": [layer]}
        method["reads"][read] = {
            "site": site,
            "pos": "tap",
            "model": "original",
            "input": "counterfactual",
        }
        method["writes"][write] = {"site": site, "pos": "tap", "do": {"swap": read}}
    doc = _specification(
        f"{component} patching over {len(bands)} band(s); every output target of "
        f"a {output_length}-token cohort is read from each forward",
        model,
        {
            "base": {"dataset": dataset, "field": "input"},
            "counterfactual": {"dataset": dataset, "field": "counterfactual_inputs[0]"},
        },
        method,
    )
    doc = add_readouts(
        doc,
        output_length,
        name="null",
        target_prefix="cf_output_",
        alternative_prefix="output_",
        top_k=top_k,
    )
    for i, band in enumerate(bands):
        name = f"band_{i}"
        doc["method"]["intervened_models"][name] = {
            "input": "base",
            "writes": [f"write_{layer}" for layer in band],
        }
        doc = add_readouts(
            doc,
            output_length,
            model=name,
            name=f"scores_{i}",
            target_prefix="cf_output_",
            alternative_prefix="output_",
            top_k=top_k,
        )
    return doc


def harvest_protocol(
    model: Mapping[str, Any],
    dataset: str,
    layers: Sequence[int],
) -> dict[str, Any]:
    """Harvest every input/output position once per layer for shared PCA/lens use.

    Ragged inputs use the protocol's flat tensor plus widths serialization.
    Keep the prepared rows beside the harvest to map tokens and semantic roles.
    """
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(type(i) is not int or i < 0 for i in layers)
    ):
        raise ValueError("layers must contain distinct nonnegative indices")
    sites, reads, save = {}, {}, []
    for layer in layers:
        site, read = f"layer_{layer}", f"residual_{layer}"
        sites[site] = {"component": "block_output", "layers": [layer]}
        reads[read] = {"site": site, "pos": "all", "model": "original", "input": "base"}
        save.append(
            {
                "value": read,
                "model": "original",
                "input": "base",
                "file_path": f"{read}.safetensors",
            }
        )
    return _specification(
        f"block_output harvest at every position of layers {list(layers)}, "
        "shared by PCA and the logit lens",
        model,
        {"base": {"dataset": dataset, "field": "input"}},
        {"sites": sites, "reads": reads, "save": save},
    )


def _specification(
    description: str,
    model: Mapping[str, Any],
    data: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, Any]:
    """A protocol_version 3 intervention specification in group order (§1)."""
    from causalab.protocol.schema import METHOD_SECTIONS, PROTOCOL_VERSION

    if set(method) - set(METHOD_SECTIONS):
        raise ValueError(
            f"unknown method sections {sorted(set(method) - set(METHOD_SECTIONS))}"
        )
    return {
        "header": {"protocol_version": PROTOCOL_VERSION, "description": description},
        "model": dict(model),
        "data": dict(data),
        "method": {k: method[k] for k in METHOD_SECTIONS if k in method},
    }


def write_sequence_workflow(
    directory: Any,
    model: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
    bands: Sequence[Sequence[int]],
    positions: Sequence[Any],
    top_k: int = 10,
) -> Any:
    """Write a runnable harvest + three patching methods for one length cohort.

    Use the directory as --data-root when running workflow.json. Analysis scripts
    and target reports consume the saved harvest and targets.json; no model work
    is duplicated per target. Location/band choices remain the researcher's.
    """
    import json
    from pathlib import Path

    from causalab.protocol.tables import write_table

    cohorts = sequence_cohorts(pairs)
    if len(cohorts) != 1 or any(
        "counterfactual_inputs_encoding" not in row for row in pairs
    ):
        raise ValueError("write one workflow per nonempty paired sequence cohort")
    directory = Path(directory)
    count = int(pairs[0]["output_length"])
    documents = {"harvest": harvest_protocol(model, "pairs", layers)}
    for name, component, selected in (
        ("residual", "block_output", [[i] for i in layers]),
        ("attention", "attention_output", bands),
        ("mlp", "mlp_output", bands),
    ):
        documents[name] = patching_protocol(
            model,
            "pairs",
            count,
            component=component,
            bands=selected,
            positions=positions,
            top_k=top_k,
        )
    workflow = {
        "version": "1",
        "output_dir": "sequence_analysis",
        "steps": {
            name: {"type": "intervention_protocol", "document": f"{name}.json"}
            for name in documents
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    write_table(directory / "pairs.json", [dict(row) for row in pairs])
    write_table(
        directory / "targets.json",
        [
            {
                "example_id": row["example_id"],
                "split": row["split"],
                **target,
            }
            for row in pairs
            for target in row["targets"]
        ],
    )
    for name, value in {**documents, "workflow": workflow}.items():
        (directory / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    return directory / "workflow.json"


def prepare_sequence(
    tokenizer: Any,
    prompt: str,
    completion: str | Sequence[int],
    *,
    example_id: str,
    split: str,
    prefix_condition: str = "correct",
    chat: bool = False,
    system: str | None = None,
    roles: Mapping[int, Any] | None = None,
    eos_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Freeze one prompt/completion; use prepare_sequences to amortize metadata."""
    return _prepare_sequence(
        tokenizer,
        prompt,
        completion,
        vocabulary_size=len(tokenizer),
        fingerprint=tokenizer_digest(tokenizer),
        example_id=example_id,
        split=split,
        prefix_condition=prefix_condition,
        chat=chat,
        system=system,
        roles=roles,
        eos_token_ids=eos_token_ids,
    )


def prepare_sequences(
    tokenizer: Any, examples: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Prepare a CPU batch with one vocabulary-size query and a shared digest.

    Each example supplies prepare_sequence's prompt, completion, example_id,
    split and optional arguments. Tokenizer mutation during preparation refuses
    the batch instead of publishing encodings with stale provenance.
    """
    vocabulary_size = len(tokenizer)
    fingerprint = tokenizer_digest(tokenizer)
    rows = [
        _prepare_sequence(
            tokenizer,
            vocabulary_size=vocabulary_size,
            fingerprint=fingerprint,
            **dict(example),
        )
        for example in examples
    ]
    if tokenizer_digest(tokenizer) != fingerprint:
        raise ProtocolError("P2", "tokenizer changed during sequence preparation")
    return rows


def _prepare_sequence(
    tokenizer: Any,
    prompt: str,
    completion: str | Sequence[int],
    *,
    example_id: str,
    split: str,
    prefix_condition: str = "correct",
    chat: bool = False,
    system: str | None = None,
    roles: Mapping[int, Any] | None = None,
    eos_token_ids: Sequence[int] | None = None,
    vocabulary_size: int,
    fingerprint: str,
) -> dict[str, Any]:
    """Freeze a prompt and output into exact IDs and per-token target records.

    Text answers are tokenized in full context and must preserve the prompt's
    token prefix. A boundary merge is refused instead of scoring a different
    prompt. For a recorded rollout, pass the emitted IDs (including EOS if it
    is a target), not a decoded string. Chat is rendered before freezing; the
    resulting protocol must not apply a second chat template or BOS.
    """
    if prefix_condition not in PREFIX_CONDITIONS:
        raise ProtocolError(
            "P2", f"fixed prefix condition must be one of {PREFIX_CONDITIONS}"
        )
    if prefix_condition == "baseline_generated" and isinstance(completion, str):
        raise ProtocolError("P2", "baseline_generated requires the recorded token IDs")
    if system is not None and not chat:
        raise ProtocolError("P2", "a system message requires chat=True")
    if chat:
        from causalab.neural.shared.framing import render_chat

        prompt, _ = render_chat(tokenizer, prompt, system=system)
    encoded_prompt = tokenizer(
        prompt, add_special_tokens=not chat, return_offsets_mapping=True
    )
    prompt_ids = list(encoded_prompt["input_ids"])
    prompt_offsets = [list(s) for s in encoded_prompt["offset_mapping"]]
    if not prompt_ids:
        raise ProtocolError("P2", "a next-token experiment needs a nonempty prompt")
    if isinstance(completion, str):
        text = prompt + completion
        encoded = tokenizer(
            text, add_special_tokens=not chat, return_offsets_mapping=True
        )
        ids = list(encoded["input_ids"])
        offsets = [list(s) for s in encoded["offset_mapping"]]
        if ids[: len(prompt_ids)] != prompt_ids:
            raise ProtocolError(
                "P2",
                "the answer merges across the prompt boundary; revise the prompt "
                "or supply the exact continuation token IDs",
            )
        output_ids = ids[len(prompt_ids) :]
    else:
        output_ids = list(completion)
        if any(type(t) is not int or not 0 <= t < vocabulary_size for t in output_ids):
            raise ProtocolError("P2", "completion must contain integer vocabulary IDs")
        # Decode in prompt context for display and character anchors. Decoding
        # a continuation alone can strip its leading space (e.g. SentencePiece).
        # Execution always keeps the recorded IDs.
        # A byte-fallback token can revise the previous decoded character.
        # Reassign the overlapping suffix's offsets in that case.
        decoded_prompt = tokenizer.decode(
            prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        suffix, output_offsets = "", []
        for i in range(len(output_ids)):
            decoded = tokenizer.decode(
                prompt_ids + output_ids[: i + 1],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not decoded.startswith(decoded_prompt):
                raise ProtocolError(
                    "P2",
                    "continuation revises decoded prompt characters; "
                    "choose a prompt boundary with stable character offsets",
                )
            grown = decoded[len(decoded_prompt) :]
            common = 0
            while (
                common < min(len(suffix), len(grown))
                and suffix[common] == grown[common]
            ):
                common += 1
            for span in output_offsets:
                if span[1] > common:
                    span[0], span[1] = min(span[0], common), len(grown)
            output_offsets.append([common, len(grown)])
            suffix = grown
        text = prompt + suffix
        offsets = prompt_offsets + [
            [len(prompt) + a, len(prompt) + b] for a, b in output_offsets
        ]
        ids = prompt_ids + output_ids
    if not output_ids:
        raise ProtocolError("P2", "a fixed sequence needs at least one output token")
    if any(
        type(i) is not int
        or not 0 <= i < len(output_ids)
        or not isinstance(role, str)
        or not role
        for i, role in (roles or {}).items()
    ):
        raise ValueError(
            "roles must map valid output ordinals to nonempty semantic names"
        )
    eos_ids = (
        tuple(eos_token_ids)
        if eos_token_ids is not None
        else (() if tokenizer.eos_token_id is None else (tokenizer.eos_token_id,))
    )
    if any(type(t) is not int or not 0 <= t < vocabulary_size for t in eos_ids):
        raise ProtocolError("P2", "EOS IDs must be integer vocabulary IDs")
    if any(token in eos_ids for token in output_ids[:-1]):
        raise ProtocolError(
            "P2", "completion contains tokens after EOS; trim the recorded rollout"
        )
    targets = []
    row: dict[str, Any] = {
        "example_id": example_id,
        "split": split,
        "prefix_condition": prefix_condition,
        "input": text,
        "input_encoding": {
            "version": 1,
            "tokenizer_digest": fingerprint,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "input_ids": ids,
            "offset_mapping": offsets,
            "prompt_length": len(prompt_ids),
        },
        "output_length": len(output_ids),
        "targets": targets,
    }
    for i, token in enumerate(output_ids):
        name = f"output_{i}"
        row[name] = token
        targets.append(
            {
                "target": name,
                "ordinal": i,
                "role": (roles or {}).get(i, name),
                "token_id": token,
                "token": tokenizer.decode([token]),
                "token_position": len(prompt_ids) + i,
                "prediction_position": len(prompt_ids) + i - 1,
                "prefix_condition": prefix_condition,
                "eligible": True,
                "is_eos": token in eos_ids,
            }
        )
    return row


def add_readouts(
    document: Mapping[str, Any],
    output_length: int,
    *,
    model: str = "original",
    input_role: str = "base",
    target_prefix: str = "output_",
    alternative_prefix: str | None = None,
    name: str = "sequence",
    top_k: int = 10,
) -> dict[str, Any]:
    """Return a document with one predictive read per target in a length cohort.

    Repeated calls may score original and patched models. ``target_prefix``
    names exact-ID columns; an alternative prefix adds logit differences.
    All rows must contain exactly ``output_length`` supplied output tokens.
    Changing the score adds reads/metrics, never another intervention.
    """
    if type(output_length) is not int or output_length < 1:
        raise ValueError("output_length must be a positive integer")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    # Sections have a normative order, even for programmatically authored docs.
    from causalab.protocol.schema import GROUP_ORDER, METHOD_SECTIONS, PROTOCOL_VERSION

    result = copy.deepcopy(dict(document))
    header = result.get("header")
    if (
        set(result) - set(GROUP_ORDER)
        or not isinstance(header, Mapping)
        or header.get("protocol_version") != PROTOCOL_VERSION
        or not isinstance(result.get("method", {}), Mapping)
    ):
        raise ValueError(
            "add_readouts requires a protocol_version "
            f"{PROTOCOL_VERSION!r} intervention specification (header / model / "
            "data / method); rewrite a v1 file with `causalab migrate <file>`"
        )
    method = result.setdefault("method", {})
    if set(method) - set(METHOD_SECTIONS):
        raise ValueError(
            f"unknown method sections {sorted(set(method) - set(METHOD_SECTIONS))}"
        )
    sites = method.setdefault("sites", {})
    head = f"{name}_head"
    if head in sites:
        raise ValueError(f"readout name {name!r} already exists")
    sites[head] = {"component": "lm_head"}
    reads, metrics, save = (
        method.setdefault(k, v)
        for k, v in (("reads", {}), ("metrics", {}), ("save", []))
    )
    for i in range(output_length):
        read = f"{name}_{i}"
        if read in reads:
            raise ValueError(f"read {read!r} already exists")
        reads[read] = {
            "site": head,
            "pos": i - output_length - 1,
            "model": model,
            "input": input_role,
        }
        target = f"{target_prefix}{i}"
        definitions = {
            "accuracy": {"kind": "match", "expected": target, "token_form": "id"},
            "nll": {"kind": "cross_entropy", "target": target, "token_form": "id"},
            "logit": {"kind": "token_logit", "token": target, "token_form": "id"},
            "top_k": {"kind": "top_k", "k": top_k, "by": "prob"},
        }
        if alternative_prefix is not None:
            definitions["logit_diff"] = {
                "kind": "logit_diff",
                "a": target,
                "b": f"{alternative_prefix}{i}",
                "token_form": "id",
            }
        for suffix, definition in definitions.items():
            metric = f"{read}_{suffix}"
            if metric in metrics:
                raise ValueError(f"metric {metric!r} already exists")
            metrics[metric] = {**definition, "of": read}
            save.append(
                {
                    "value": metric,
                    "model": model,
                    "input": input_role,
                    "file_path": f"{metric}.json",
                }
            )
    result["method"] = {k: method[k] for k in METHOD_SECTIONS if k in method}
    return {k: result[k] for k in GROUP_ORDER if k in result}


def add_rollout_readouts(
    document: Mapping[str, Any],
    output_length: int,
    *,
    model: str = "original",
    input_role: str = "base",
    target_prefix: str = "output_",
    name: str = "rollout",
    top_k: int = 10,
) -> dict[str, Any]:
    """Score all target ordinals on one greedy rollout from prompt-only inputs.

    The distribution producing output zero is at prompt -1; for output j>0 it
    is at generated j-1. The named reads share one decode, never a target sweep.
    Writes remain prefill-only. These scores must not be pooled with fixed-prefix
    scores; later missing positions retain the engine's unavailable records.
    """
    result = add_readouts(
        document,
        output_length,
        model=model,
        input_role=input_role,
        target_prefix=target_prefix,
        name=name,
        top_k=top_k,
    )
    method = result["method"]
    for i in range(output_length):
        method["reads"][f"{name}_{i}"]["pos"] = (
            -1
            if i == 0
            else {"generated": {"max_new_tokens": output_length}, "index": i - 1}
        )
    read, metric = f"{name}_continuation", f"{name}_text"
    method["reads"][read] = {
        "site": f"{name}_head",
        "model": model,
        "input": input_role,
        "pos": {"generated": {"max_new_tokens": output_length}, "all": True},
    }
    method["metrics"][metric] = {"kind": "decode", "of": read}
    method["save"].append(
        {
            "value": metric,
            "model": model,
            "input": input_role,
            "file_path": f"{metric}.json",
        }
    )
    return result
