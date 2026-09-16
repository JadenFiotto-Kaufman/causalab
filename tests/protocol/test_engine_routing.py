"""Capability derivation and engine routing (spec §8)."""

from __future__ import annotations

import pytest

from causalab.protocol.engine import (
    Engine,
    ExecutionRequest,
    RunResult,
    choose_engine,
    component_capability,
    requires,
)
from causalab.protocol.errors import ValidationError
from causalab.protocol.schema import COMPONENTS, parse_document

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


class _Stub(Engine):
    """Coarse-capability stub: serves the whole component vocabulary, so the
    capability-routing tests below vary only the §8 verbs."""

    def __init__(
        self,
        name: str,
        capabilities: frozenset[str],
        is_local: bool = False,
        components: frozenset[str] = frozenset(COMPONENTS),
        writable_components: frozenset[str] = frozenset(COMPONENTS),
    ):
        self.name = name
        self.capabilities = capabilities
        self.is_local = is_local
        self.components = components
        self.writable_components = writable_components

    def execute(self, request: ExecutionRequest) -> RunResult:  # pragma: no cover
        raise NotImplementedError


#: base_doc touches block_output (read + write) and lm_head (read).
BASE_COMPONENTS = frozenset(
    {
        component_capability("block_output"),
        component_capability("block_output", write=True),
        component_capability("lm_head"),
    }
)


def test_requires_paired_forward():
    doc = parse_document(base_doc())
    assert requires(doc) == frozenset({"paired_forward"}) | BASE_COMPONENTS


def test_requires_component_entries_split_read_from_write():
    """Every touched site contributes its component; a written site also
    contributes the :write entry — the honest routing surface once two
    engines with different site vocabularies exist (§8)."""
    doc = parse_document(base_doc())
    needed = requires(doc)
    assert component_capability("block_output", write=True) in needed
    assert component_capability("lm_head") in needed
    assert component_capability("lm_head", write=True) not in needed


def test_requires_no_coarse_verbs_for_same_input_patching():
    raw = base_doc()
    raw["method"]["reads"]["v_cf"]["input"] = "base"
    del raw["data"]["counterfactual"]
    assert requires(parse_document(raw)) == BASE_COMPONENTS


def test_requires_full_logits_when_lm_head_read_saved():
    raw = base_doc()
    raw["method"]["save"].append(
        {
            "value": "logits",
            "model": "patched",
            "input": "base",
            "file_path": "l.safetensors",
        }
    )
    assert "full_logits" in requires(parse_document(raw))


def test_requires_full_logits_for_top_k_over_an_lm_head_read():
    raw = base_doc()
    raw["method"]["metrics"]["tk"] = {
        "kind": "top_k",
        "of": "logits",
        "k": 5,
        "by": "prob",
    }
    raw["method"]["save"].append(
        {"value": "tk", "model": "patched", "input": "base", "file_path": "tk.json"}
    )
    assert "full_logits" in requires(parse_document(raw))


def test_top_k_over_a_non_vocabulary_read_needs_no_full_logits():
    """The saving that motivates any-read ``top_k``: ranking a residual stream
    obliges no vocabulary projection anywhere, so it must not route the
    document onto a full-vocab engine."""
    raw = base_doc()
    raw["method"]["metrics"]["tk"] = {
        "kind": "top_k",
        "of": "v_cf",
        "k": 5,
        "by": "abs_value",
    }
    raw["method"]["save"].append(
        {
            "value": "tk",
            "model": "original",
            "input": "counterfactual",
            "file_path": "tk.json",
        }
    )
    assert "full_logits" not in requires(parse_document(raw))


def test_top_k_over_a_featurized_lm_head_read_still_needs_full_logits():
    """Capability and axis are two different questions, split on purpose.

    A featurizer takes the read's *value* out of token-id space (so `prob`
    and token decoding are refused / withheld), but serving the read still
    means materializing the whole projection — the featurizer consumes it.
    So the document still routes onto a full-vocab engine."""
    raw = base_doc()
    raw["method"]["featurizers"] = {
        "f": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    raw["method"]["reads"]["flogits"] = {
        "site": "lm_head",
        "pos": -1,
        "model": "patched",
        "input": "base",
        "featurizer": "f",
    }
    raw["method"]["metrics"]["tk"] = {
        "kind": "top_k",
        "of": "flogits",
        "k": 2,
        "by": "value",
    }
    raw["method"]["save"].append(
        {"value": "tk", "model": "patched", "input": "base", "file_path": "tk.json"}
    )
    assert "full_logits" in requires(parse_document(in_order(raw)))


def test_top_k_over_a_dims_sliced_lm_head_read_needs_no_full_logits():
    """A `dims` slice needs only its named vocabulary rows — the same rule the
    saved-read derivation already applies."""
    raw = base_doc()
    raw["method"]["reads"]["flogits"] = {
        "site": "lm_head",
        "pos": -1,
        "model": "patched",
        "input": "base",
        "dims": [0, 1, 2],
    }
    raw["method"]["metrics"]["tk"] = {
        "kind": "top_k",
        "of": "flogits",
        "k": 2,
        "by": "value",
    }
    raw["method"]["save"].append(
        {"value": "tk", "model": "patched", "input": "base", "file_path": "tk.json"}
    )
    assert "full_logits" not in requires(parse_document(in_order(raw)))


def _fit(raw: dict) -> dict:
    """base_doc as a DAS fit: a trained subspace on the patched site."""
    raw["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    raw["method"]["writes"]["patch"]["featurizer"] = "rot"
    raw["method"]["metrics"]["ce"] = {
        "kind": "cross_entropy",
        "of": "logits",
        "target": "label",
        "token_form": "space_prefixed",
    }
    raw["method"]["train"] = {
        "objective": [[1.0, "ce"]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    raw["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    return raw


TRAIN_VERBS = frozenset(
    {"train_free_params", "train_loss_precision", "train_eval_updates"}
)


def test_a_plain_fit_requires_grad_and_no_training_verb():
    """Featurizer slots, fp32, an epoch-counted eval: `grad` alone (§8)."""
    needed = requires(parse_document(in_order(_fit(base_doc()))))
    assert "grad" in needed and not needed & TRAIN_VERBS


def test_requires_train_free_params():
    """A `train.params` entry naming a `params` entry — a free tensor (§2.6)
    — is a loop the engine must implement, so it is routed on (rule 30
    refuses the engine that lacks it)."""
    raw = _fit(base_doc())
    raw["method"]["params"] = {"w": {"shape": [768], "init": "zeros"}}
    raw["method"]["writes"]["steer"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"add_scaled": {"op": "w", "alpha": 1.0}},
    }
    raw["method"]["intervened_models"]["patched"]["writes"].append("steer")
    raw["method"]["train"]["params"] = ["rot", "w"]
    needed = requires(parse_document(in_order(raw)))
    assert needed & TRAIN_VERBS == {"train_free_params"}


def test_requires_train_loss_precision():
    """`train.precision` authored as anything but fp32 — either field."""
    raw = _fit(base_doc())
    raw["method"]["train"]["precision"] = {"feature": "fp32", "loss": "bf16"}
    assert requires(parse_document(in_order(raw))) & TRAIN_VERBS == {
        "train_loss_precision"
    }
    raw["method"]["train"]["precision"] = {"feature": "fp16", "loss": "fp32"}
    assert requires(parse_document(in_order(raw))) & TRAIN_VERBS == {
        "train_loss_precision"
    }
    raw["method"]["train"]["precision"] = {"feature": "fp32", "loss": "fp32"}
    assert not requires(parse_document(in_order(raw))) & TRAIN_VERBS


def test_requires_train_eval_updates():
    """An `eval` counted in updates rather than epochs (§2.11)."""
    raw = _fit(base_doc())
    raw["method"]["train"]["eval"] = {
        "every": {"updates": 1},
        "split": "weekdays/data#test",
        "metrics": ["ce"],
    }
    assert requires(parse_document(in_order(raw))) & TRAIN_VERBS == {
        "train_eval_updates"
    }
    raw["method"]["train"]["eval"]["every"] = {"epochs": 1}
    assert not requires(parse_document(in_order(raw))) & TRAIN_VERBS


def test_requires_writable_attention_probs():
    raw = base_doc()
    raw["method"]["sites"]["probs"] = {"component": "attention_probs", "layers": [3]}
    raw["method"]["writes"]["knock"] = {
        "site": "probs",
        "pos": -1,
        "do": {"clamp": {"lo": 0, "hi": 0}},
    }
    raw["method"]["intervened_models"]["patched"]["writes"].append("knock")
    assert "writable_attention_probs" in requires(parse_document(raw))


def test_choose_engine_first_covering():
    doc = parse_document(base_doc())
    weak = _Stub("serving", frozenset({"full_logits"}))
    strong = _Stub("hooks", frozenset({"grad", "paired_forward", "full_logits"}))
    assert choose_engine(doc, [weak, strong]) is strong


def test_refusal_names_missing_capabilities():
    doc = parse_document(base_doc())
    weak = _Stub("serving", frozenset({"full_logits"}))
    with pytest.raises(ValidationError) as err:
        choose_engine(doc, [weak])
    message = str(err.value)
    assert "paired_forward" in message and "serving" in message


def test_an_engine_without_generate_refuses_with_the_capability_named():
    """Routing is how a decode-less engine declines a continuation document,
    and the refusal names what it lacks rather than failing mid-run."""
    raw = base_doc()
    raw["method"]["positions"] = {
        "tail": {"generated": {"max_new_tokens": 8}, "index": -1}
    }
    raw["method"]["reads"]["logits"]["pos"] = "tail"
    doc = parse_document(in_order(raw))
    prefill_only = _Stub("prefill_only", frozenset({"paired_forward", "full_logits"}))
    with pytest.raises(ValidationError) as err:
        choose_engine(doc, [prefill_only])
    assert "generate" in str(err.value)
    decoder = _Stub("decoder", prefill_only.capabilities | {"generate"})
    assert choose_engine(doc, [prefill_only, decoder]) is decoder


def test_an_engine_without_the_component_refuses_by_name():
    """A document touching a component outside an engine's site vocabulary
    routes past it, and the generated refusal names the component entry —
    this is how interior components an engine cannot serve route to the one
    that can, with no hand-written case anywhere."""
    doc = parse_document(base_doc())
    verbs = frozenset({"paired_forward", "full_logits"})
    no_blocks = _Stub(
        "no_blocks",
        verbs,
        components=frozenset({"lm_head"}),
        writable_components=frozenset(),
    )
    with pytest.raises(ValidationError) as err:
        choose_engine(doc, [no_blocks])
    message = str(err.value)
    assert component_capability("block_output") in message
    assert component_capability("block_output", write=True) in message
    full = _Stub("full", verbs)
    assert choose_engine(doc, [no_blocks, full]) is full


def test_a_read_only_component_declaration_refuses_the_write():
    """components without writable_components serves reads but routes a
    write away."""
    doc = parse_document(base_doc())
    read_only = _Stub(
        "read_only",
        frozenset({"paired_forward", "full_logits"}),
        writable_components=frozenset(),
    )
    with pytest.raises(ValidationError) as err:
        choose_engine(doc, [read_only])
    assert component_capability("block_output", write=True) in str(err.value)


def test_one_definition_of_the_engine_default() -> None:
    """`--engine`'s default and every fallback are the same constant.

    There were two `getattr(args, "engine", …)` fallbacks that had drifted apart —
    `"auto"` in `protocol/cli.py`, `"pytorch_hooks"` in `workflow/cli.py` — so
    a caller that did not come through argparse got routing on one path and a
    silent pin to the reference engine on the other. Unreachable through the
    CLI, which is why nothing noticed.

    Aligning two literals would have left three uncoupled copies of `"auto"`.
    The fix was one definition; this pins it, because a *default* is exactly
    the kind of value someone changes in one place.
    """
    from causalab.cli import _build_parser
    from causalab.protocol.engine import DEFAULT_ENGINE, ENGINE_CHOICES

    assert DEFAULT_ENGINE in ENGINE_CHOICES

    subparsers = _build_parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    engine_options = {
        verb: action
        for verb, parser in subparsers.items()
        for action in parser._actions
        if "--engine" in action.option_strings
    }
    assert set(engine_options) == {"run", "explain", "dry-run"}, (
        f"--engine is on {sorted(engine_options)}; update this test if a verb "
        "gained or lost it"
    )
    assert engine_options["run"].default == DEFAULT_ENGINE, (
        "`run --engine` must default to the one constant, not a literal"
    )
    for pure in ("explain", "dry-run"):
        assert engine_options[pure].default is None, (
            f"`{pure}` must default to passing no engine at all — that is what "
            "keeps it torch-free until asked to route"
        )
    for verb, action in engine_options.items():
        assert tuple(action.choices or ()) == ENGINE_CHOICES, (
            f"{verb} --engine choices drifted from ENGINE_CHOICES"
        )
