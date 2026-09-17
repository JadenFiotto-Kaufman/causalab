"""Documents the module-boundary parity suites drive through two engines:
the write mechanisms, a ``block_mid`` write followed by later same-layer
reads, the mixed-precedence block writes, and a band site beside its
hand-written twin. Built on :mod:`tests._helpers.a3b_sweep`'s ``read_doc`` /
``interchange_doc`` shapes."""

from __future__ import annotations

from typing import Any

from tests._helpers import a3b_sweep as sweep

__all__ = [
    "MECHANISMS",
    "band_doc",
    "block_mid_with_later_read_doc",
    "mechanism_doc",
    "mixed_block_write_precedence_doc",
]

#: Every write mechanism, as the write set of one intervened model.
MECHANISMS: dict[str, dict[str, Any]] = {
    "swap": {"patch": {"swap": "v_cf"}},
    "add_scaled": {"patch": {"add_scaled": {"op": "v_cf", "alpha": 0.5}}},
    "lerp": {"patch": {"lerp": {"op": "v_cf", "alpha": 0.3}}},
    "gaussian": {
        "patch": {"gaussian": {"seed": 7, "scale": 0.5, "axis": "tp_duplicated"}}
    },
    "add_then_renormalize": {
        "patch": {"add_scaled": {"op": "v_cf", "alpha": 2.0}},
        "renorm": {"renormalize": True},
    },
}


def mechanism_doc(
    component: str, layer: int | None, writes: dict[str, Any]
) -> dict[str, Any]:
    """``interchange_doc``'s shape with the write set replaced: every write
    lands at the one site, at the last position, and ``v_cf`` (the site read
    on the counterfactual) is there to be an operand."""
    doc = sweep.interchange_doc(component, layer)
    doc["method"]["writes"] = {
        name: {"site": "tap", "pos": -1, "do": do} for name, do in writes.items()
    }
    doc["method"]["intervened_models"]["patched"]["writes"] = list(writes)
    # a mechanism with no operand leaves `v_cf` dead; saving it keeps the
    # document valid (§5.11) with the same forward groups
    doc["method"]["save"].append(
        {
            "value": "v_cf",
            "model": "original",
            "input": "counterfactual",
            "file_path": "v.safetensors",
        }
    )
    return doc


def block_mid_with_later_read_doc(later_component: str, layer: int) -> dict[str, Any]:
    """A mid write followed by a same-layer later read and an output read."""
    return {
        "header": {"protocol_version": "3"},
        "model": {"key": "test", "revision": "main"},
        "data": {
            "base": {"dataset": "inline", "field": "input"},
            "counterfactual": {
                "dataset": "inline",
                "field": "counterfactual_inputs[0]",
            },
        },
        "sites": {
            "mid": {"component": "block_mid", "layers": layer},
            "later": {"component": later_component, "layers": layer},
            "out": {"component": "block_output", "layers": layer},
        },
        "reads": {
            "v_mid_base": {
                "site": "mid",
                "pos": -1,
                "model": "original",
                "input": "base",
            },
            "v_mid_cf": {
                "site": "mid",
                "pos": -1,
                "model": "original",
                "input": "counterfactual",
            },
            "r_later": {
                "site": "later",
                "pos": -1,
                "model": "patched",
                "input": "base",
            },
            "r_out": {"site": "out", "pos": -1, "model": "patched", "input": "base"},
        },
        "writes": {"mid_swap": {"site": "mid", "pos": -1, "do": {"swap": "v_mid_cf"}}},
        "intervened_models": {"patched": {"input": "base", "writes": ["mid_swap"]}},
        "save": [
            {
                "value": name,
                "model": model,
                "input": role,
                "file_path": f"{name}.safetensors",
            }
            for name, model, role in (
                ("v_mid_base", "original", "base"),
                ("v_mid_cf", "original", "counterfactual"),
                ("r_later", "patched", "base"),
                ("r_out", "patched", "base"),
            )
        ],
    }


def mixed_block_write_precedence_doc(layer: int) -> dict[str, Any]:
    """Three intervened models over one block: a mid swap with an output swap
    in either authored order, and a mid swap with an additive output write —
    each read at the block's output."""
    doc = block_mid_with_later_read_doc("mlp_output", layer)
    doc["reads"]["v_out_cf"] = {
        "site": "out",
        "pos": -1,
        "model": "original",
        "input": "counterfactual",
    }
    doc["writes"]["out_swap"] = {"site": "out", "pos": -1, "do": {"swap": "v_out_cf"}}
    doc["writes"]["out_add"] = {
        "site": "out",
        "pos": -1,
        "do": {"add_scaled": {"op": "v_out_cf", "alpha": 0.25}},
    }
    doc["intervened_models"] = {
        "mid_then_out": {"input": "base", "writes": ["mid_swap", "out_swap"]},
        "out_then_mid": {"input": "base", "writes": ["out_swap", "mid_swap"]},
        "mid_plus_out": {"input": "base", "writes": ["mid_swap", "out_add"]},
    }
    doc["reads"] = {
        k: v for k, v in doc["reads"].items() if k not in ("r_later", "r_out")
    }
    del doc["sites"]["later"]
    for model in doc["intervened_models"]:
        doc["reads"][f"r_{model}"] = {
            "site": "out",
            "pos": -1,
            "model": model,
            "input": "base",
        }
    doc["save"] = [
        {
            "value": name,
            "model": read["model"],
            "input": read["input"],
            "file_path": f"{name}.safetensors",
        }
        for name, read in doc["reads"].items()
    ]
    return doc


def band_doc(one_site: bool) -> dict[str, Any]:
    """A swap at ``attention_output`` over layers 0 and 1 — as one band site,
    or as the two per-layer sites it lowers to."""
    if one_site:
        sites = {"a": {"component": "attention_output", "layers": [0, 1]}}
        reads = {
            "v": {
                "site": "a",
                "pos": -1,
                "model": "original",
                "input": "counterfactual",
            }
        }
        writes = {"w": {"site": "a", "pos": -1, "do": {"swap": "v"}}}
        in_force = ["w"]
    else:
        sites = {
            f"a{i}": {"component": "attention_output", "layers": [i]} for i in (0, 1)
        }
        reads = {
            f"v{i}": {
                "site": f"a{i}",
                "pos": -1,
                "model": "original",
                "input": "counterfactual",
            }
            for i in (0, 1)
        }
        writes = {
            f"w{i}": {"site": f"a{i}", "pos": -1, "do": {"swap": f"v{i}"}}
            for i in (0, 1)
        }
        in_force = ["w0", "w1"]
    doc = sweep.interchange_doc("attention_output", 0)
    doc["method"]["sites"] = {**sites, "head": {"component": "lm_head"}}
    doc["method"]["reads"] = {
        **reads,
        "logits": {"site": "head", "pos": -1, "model": "patched", "input": "base"},
    }
    doc["method"]["writes"] = writes
    doc["method"]["intervened_models"] = {
        "patched": {"input": "base", "writes": in_force}
    }
    return doc
