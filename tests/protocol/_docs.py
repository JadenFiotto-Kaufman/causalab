"""Shared document builders for the protocol tests."""

from __future__ import annotations

from typing import Any

from causalab.protocol.schema import GROUP_ORDER, METHOD_SECTIONS, PROTOCOL_VERSION


def base_doc() -> dict[str, Any]:
    """A minimal valid interchange document on gpt2 (layer 3 < 12)."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {
            "base": {"dataset": "weekdays/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/data#train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "sites": {
                "tgt": {"component": "block_output", "layers": [3]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {
                    "site": "tgt",
                    "pos": -1,
                    "model": "original",
                    "input": "counterfactual",
                },
                "logits": {
                    "site": "lm_head",
                    "pos": -1,
                    "model": "patched",
                    "input": "base",
                },
            },
            "writes": {"patch": {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}},
            "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
            "metrics": {
                "ld": {
                    "kind": "logit_diff",
                    "of": "logits",
                    "a": "cf_answer",
                    "b": "base_answer",
                    "token_form": "space_prefixed",
                }
            },
            "save": [
                {
                    "value": "ld",
                    "model": "patched",
                    "input": "base",
                    "file_path": "ld.json",
                }
            ],
        },
    }


def in_order(raw: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a mutated document with its groups, and the method's sections,
    in the §1 order — test mutations append sections at the dict end, and while
    that no longer refuses the document (§5 rule 2 warns now), it does raise a
    warning the test isn't about.

    A method section spread at the top level (``{**base_doc(), "segments":
    …}``) is hoisted into ``method`` first: the spelling predates the groups,
    and what such a test is about is the section, never the shape (§1 —
    :mod:`tests.protocol.test_protocol_v2` pins the shape rule itself)."""
    method: dict[str, Any] = dict(raw.get("method") or {})
    for key in METHOD_SECTIONS:
        if key in raw:
            method[key] = raw[key]
    out = {key: raw[key] for key in GROUP_ORDER if key in raw and key != "method"}
    if method or "method" in raw:
        out["method"] = {key: method[key] for key in METHOD_SECTIONS if key in method}
    return {key: out[key] for key in GROUP_ORDER if key in out}
