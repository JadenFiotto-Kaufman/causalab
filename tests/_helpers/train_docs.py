"""The fit documents the train suites drive.

On the tiny Llama: a DAS rotation (:func:`das_doc`), a DBM gate
(:func:`dbm_doc`) and its ``clamp`` and PID-controlled forms, the
``["rot", "gate"]`` chain and its two-phase form. On ``tiny-random/qwen3.5-moe``:
the expert-neuron DBM (:func:`expert_dbm_doc`) and its fit
(:func:`expert_dbm_fit_doc`), the tiny twin of the shipped
``dbm_expert_neuron.json``. With the rows they are fitted on and the request
a runner is handed.

One home, so every engine's train suite fits the same documents."""

from __future__ import annotations

from typing import Any

from causalab.protocol.engine import ExecutionRequest

__all__ = [
    "ANSWERS",
    "BASES",
    "COUNTERFACTUALS",
    "MOE_LAYER",
    "TINY_LLAMA",
    "TINY_QWEN35_MOE",
    "NoDatasets",
    "ROWS",
    "chain_doc",
    "clamp_dbm_doc",
    "controlled_dbm_doc",
    "das_doc",
    "dbm_doc",
    "expert_dbm_doc",
    "expert_dbm_fit_doc",
    "phased_chain_doc",
    "train_request",
]

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_QWEN35_MOE = "tiny-random/qwen3.5-moe"
#: Every fixture layer of the tiny MoE carries the sparse-MoE block.
MOE_LAYER = 0

BASES = [
    "the quick brown fox jumps over",
    "a slow green turtle sleeps deeply",
    "every shiny robot dances tonight",
    "some ancient rivers flow backwards",
]
COUNTERFACTUALS = [
    "cold silver mountains echo loudly",
    "bright yellow parrots sing early",
    "seven broken clocks tick wrongly",
    "warm quiet valleys rest gently",
]
ANSWERS = [" one", " two", " three", " four"]
#: The rows both engines' executors are handed (``a3b_sweep.make_executor``).
ROWS: list[dict[str, Any]] = [
    {"input": base, "counterfactual_inputs": [cf], "label": answer}
    for base, cf, answer in zip(BASES, COUNTERFACTUALS, ANSWERS)
]


def _data_section() -> dict[str, Any]:
    return {
        "base": {"dataset": "inline", "field": "input"},
        "counterfactual": {"dataset": "inline", "field": "counterfactual_inputs[0]"},
    }


def das_doc(*, seed: int = 0, epochs: int = 2) -> dict:
    return {
        "header": {"protocol_version": "3"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": _data_section(),
        "method": {
            "sites": {
                "tgt": {"component": "block_output", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "featurizers": {
                "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
            },
            "reads": {
                "v_cf": {
                    "site": "tgt",
                    "pos": {"index": -1},
                    "model": "original",
                    "input": "counterfactual",
                    "featurizer": "rot",
                },
                "logits": {
                    "site": "lm_head",
                    "pos": {"index": -1},
                    "model": "patched",
                    "input": "base",
                },
            },
            "writes": {
                "patch": {
                    "site": "tgt",
                    "pos": {"index": -1},
                    "featurizer": "rot",
                    "do": {"swap": "v_cf"},
                }
            },
            "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
            "metrics": {
                "ce": {
                    "kind": "cross_entropy",
                    "of": "logits",
                    "target": "label",
                    "token_form": "space_prefixed",
                }
            },
            "train": {
                "objective": [[1.0, "ce"]],
                "params": ["rot"],
                "optimizer": {"name": "adamw", "lr": 1e-2, "weight_decay": 0.0},
                "steps": {"epochs": epochs},
                "batch": {"pairs": 2},
                "seed": seed,
            },
            "save": [
                {
                    "value": "ce",
                    "model": "patched",
                    "input": "base",
                    "file_path": "ce.json",
                },
                {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"},
            ],
        },
    }


def dbm_doc() -> dict:
    doc = das_doc(seed=0, epochs=3)
    doc["method"]["featurizers"] = {"gate": {"kind": "gate"}}
    doc["method"]["reads"]["v_cf"]["featurizer"] = "gate"
    doc["method"]["writes"]["patch"]["featurizer"] = "gate"
    doc["method"]["train"]["params"] = ["gate"]
    doc["method"]["train"]["objective"] = [[1.0, "ce"], [0.01, {"l1": "gate"}]]
    doc["method"]["train"]["anneal"] = {"gate.theta.temperature": [1.0, 0.01, 0.5]}
    doc["method"]["save"] = [
        {"value": "ce", "model": "patched", "input": "base", "file_path": "ce.json"},
        {"value": "gate", "site": "tgt", "file_path": "gate.safetensors"},
    ]
    return doc


def clamp_dbm_doc(*, lr: float = 5.0) -> dict:
    """The DBM fit under `parametrization: clamp` (§2.5): no anneal, the mask
    projected onto [0, 1] after every step. A large lr so one Adam step
    reaches a pole and the projection is what keeps θ on the interval."""
    doc = dbm_doc()
    doc["method"]["featurizers"]["gate"]["parametrization"] = "clamp"
    del doc["method"]["train"]["anneal"]
    doc["method"]["train"]["optimizer"]["lr"] = lr
    return doc


def controlled_dbm_doc() -> dict:
    """The DCM sweep in one document (§2.11 `control`): a clamp gate starting
    all-patched, the sparsity weight moved by a PID so the kept-unit count
    follows a ramp from every unit to none over the run."""
    doc = clamp_dbm_doc(lr=0.1)
    doc["method"]["featurizers"]["gate"]["init"] = {"fill": 0.99}
    doc["method"]["train"]["objective"] = {
        "fit": {"weight": 1.0, "metric": "ce"},
        "sparsity": {"weight": 0.01, "l1": "gate"},
    }
    doc["method"]["train"]["steps"] = {"epochs": 10}
    doc["method"]["train"]["control"] = {
        "train.objective.sparsity.weight": {
            "kind": "pid",
            "signal": {"hard_mask_size": "gate"},
            "setpoint": {"ramp": [16, 0, 1.0]},
            "gains": {"kp": 0.5, "ki": 0.05},
        }
    }
    return doc


def expert_dbm_doc(pos: Any = -1) -> dict[str, Any]:
    """A DBM-shaped document on one MoE layer: the counterfactual routed
    interior read through an expert-keyed gate and swapped into the base run
    through the same gate, a plain gate doing the same on the shared expert,
    plus the raw reads the hand computation needs (both inputs' activations
    and routing tables)."""
    reads = {
        "routed_cf": {
            "site": "routed",
            "pos": pos,
            "model": "original",
            "input": "counterfactual",
            "featurizer": "routed_gate",
        },
        "shared_cf": {
            "site": "shared",
            "pos": pos,
            "model": "original",
            "input": "counterfactual",
            "featurizer": "shared_gate",
        },
        "pre_base": {
            "site": "routed",
            "pos": pos,
            "model": "original",
            "input": "base",
        },
        "pre_cf": {
            "site": "routed",
            "pos": pos,
            "model": "original",
            "input": "counterfactual",
        },
        "idx_base": {"site": "idx", "pos": pos, "model": "original", "input": "base"},
        "idx_cf": {
            "site": "idx",
            "pos": pos,
            "model": "original",
            "input": "counterfactual",
        },
        "post": {"site": "routed", "pos": pos, "model": "masked", "input": "base"},
        "shared_base": {
            "site": "shared",
            "pos": pos,
            "model": "original",
            "input": "base",
        },
        "shared_raw_cf": {
            "site": "shared",
            "pos": pos,
            "model": "original",
            "input": "counterfactual",
        },
        "shared_post": {
            "site": "shared",
            "pos": pos,
            "model": "masked",
            "input": "base",
        },
    }
    return {
        "header": {"protocol_version": "3"},
        "model": {"key": "test", "revision": "main"},
        "data": _data_section(),
        "method": {
            "sites": {
                "routed": {"component": "expert_activation", "layers": [MOE_LAYER]},
                "shared": {
                    "component": "shared_expert_activation",
                    "layers": [MOE_LAYER],
                },
                "idx": {"component": "expert_idx", "layers": [MOE_LAYER]},
            },
            "featurizers": {
                "routed_gate": {"kind": "gate", "group": "expert_neuron"},
                "shared_gate": {"kind": "gate"},
            },
            "reads": reads,
            "writes": {
                "mask_routed": {
                    "site": "routed",
                    "pos": pos,
                    "featurizer": "routed_gate",
                    "do": {"swap": "routed_cf"},
                },
                "mask_shared": {
                    "site": "shared",
                    "pos": pos,
                    "featurizer": "shared_gate",
                    "do": {"swap": "shared_cf"},
                },
            },
            "intervened_models": {
                "masked": {"input": "base", "writes": ["mask_routed", "mask_shared"]}
            },
            "save": [
                {
                    "value": name,
                    "model": spec["model"],
                    "input": spec["input"],
                    "file_path": f"{name}.safetensors",
                }
                for name, spec in reads.items()
            ],
        },
    }


def expert_dbm_fit_doc(
    *, pairs: int = 1, epochs: int = 1, control: bool = False
) -> dict[str, Any]:
    """The shipped ``dbm_expert_neuron.json`` fit at tiny scale
    (``tiny-random/qwen3.5-moe``): :func:`expert_dbm_doc` — a write through an
    expert-keyed gate at the routed interior and a plain gate on the shared
    expert — with an ``lm_head`` read, a cross-entropy objective and the one
    ``l1`` term over both gates. With ``control``, a PID moves the sparsity
    weight off the two gates' kept-unit counts (§2.11).

    Its read ``routed_cf`` is the whole point: ``expert_activation`` is a
    routed interior, so the value comes with a routing table and the gate is
    keyed by it, and a write through that gate joins its slots to the read's
    by expert. That is the read the nnterp engine could not finish anywhere
    but on the client, which is why this document could not be trained on
    it at all.
    """
    raw = expert_dbm_doc()
    method = raw["method"]
    method["sites"]["lm_head"] = {"component": "lm_head"}
    method["reads"]["logits"] = {
        "site": "lm_head",
        "pos": -1,
        "model": "masked",
        "input": "base",
    }
    method["metrics"] = {
        "ce": {
            "kind": "cross_entropy",
            "of": "logits",
            "target": "label",
            "token_form": "space_prefixed",
        }
    }
    method["train"] = {
        "objective": {
            "fit": {"weight": 1.0, "metric": "ce"},
            "sparsity": {"weight": 0.01, "l1": ["routed_gate", "shared_gate"]},
        },
        "params": ["routed_gate", "shared_gate"],
        "optimizer": {"name": "adamw", "lr": 0.01, "weight_decay": 0.0},
        "steps": {"epochs": epochs},
        "batch": {"pairs": pairs},
        "seed": 0,
    }
    if control:
        method["train"]["control"] = {
            "train.objective.sparsity.weight": {
                "kind": "pid",
                "signal": {"hard_mask_size": ["routed_gate", "shared_gate"]},
                "setpoint": {"ramp": [16, 0, 1.0]},
                "gains": {"kp": 0.5, "ki": 0.05},
            }
        }
    method["save"] += [
        {"value": "ce", "model": "masked", "input": "base", "file_path": "ce.json"},
        {"value": "routed_gate", "site": "routed", "file_path": "routed.safetensors"},
        {"value": "shared_gate", "site": "shared", "file_path": "shared.safetensors"},
    ]
    return raw


def chain_doc(lr) -> dict:
    """A rotation and a gate over its coordinates trained together (§2.11), the
    ``["rot", "gate"]`` chain, with ``lr`` as the document spells it."""
    doc = das_doc(seed=0, epochs=2)
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"},
        "gate": {"kind": "gate"},
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = ["rot", "gate"]
    doc["method"]["writes"]["patch"]["featurizer"] = ["rot", "gate"]
    doc["method"]["train"]["params"] = ["rot", "gate"]
    doc["method"]["train"]["objective"] = [[1.0, "ce"], [0.01, {"l1": "gate"}]]
    doc["method"]["train"]["optimizer"] = {"name": "adamw", "lr": lr}
    doc["method"]["save"] = [
        {"value": "ce", "model": "patched", "input": "base", "file_path": "ce.json"},
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"},
        {"value": "gate", "site": "tgt", "file_path": "gate.safetensors"},
    ]
    return doc


def phased_chain_doc() -> dict:
    """A rotation and a gate over its coordinates (the ``["rot", "gate"]``
    chain) fit in two phases over four updates: the gate alone, then the
    rotation alone at the gate's pinned hard mask with the sparsity weight
    annealed over that phase; every update photographed."""
    doc = chain_doc(0.05)
    doc["method"]["train"]["objective"] = {
        "fit": {"weight": 1.0, "metric": "ce"},
        "sparsity": {"weight": 0.01, "l1": "gate"},
    }
    doc["method"]["train"]["phases"] = [
        {"until": {"frac": 0.5}, "params": ["gate"]},
        {
            "until": {"frac": 1.0},
            "params": ["rot"],
            "freeze_masks": ["gate"],
            "optimizer": {"lr": 0.02},
            "anneal": {"train.objective.sparsity.weight": [1.0, 2.0, 1.0]},
        },
    ]
    doc["method"]["save"].append(
        {"kind": "trajectory", "every": {"updates": 1}, "file_path": "t.safetensors"}
    )
    return doc


class NoDatasets:
    """A dataset service for a fit whose rows are handed to the executor
    directly; ``rows`` serves an eval split when given one."""

    def __init__(self, splits: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.splits = dict(splits or {})

    def rows(self, ref: str) -> list[dict[str, Any]]:
        return self.splits[ref]

    def digest(self, ref: str) -> str:
        return "0" * 64

    def columns(self, ref: str) -> tuple[str, ...]:
        return ()


def train_request(
    splits: dict[str, list[dict[str, Any]]] | None = None,
) -> ExecutionRequest:
    """The request a train runner is handed when a test drives it directly."""
    from causalab.protocol.resolve import ResolutionEnv

    return ExecutionRequest(
        points=(),
        canonical=(),
        digests=(),
        coords=(),
        document_digest="0" * 64,
        env=ResolutionEnv(datasets=NoDatasets(splits), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
