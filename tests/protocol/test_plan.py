"""Which of a fit's forward groups the fit cannot change (spec §3, §4).

A ``train`` document re-runs its minibatch groups every optimizer step, but
only the groups a trained parameter can *reach* actually change between
steps. :func:`fit_constant_models` is the plan-level statement of that: the
models whose activations no trained featurizer, no trained free tensor, and
no operand read through either can influence. Its consumers cache those
groups across steps, epochs, eval passes and points, so an over-inclusive
answer here would serve a stale activation into a gradient step — which is
why every exclusion below is pinned, not only the happy path.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalab.protocol.plan import fit_constant_models, plan_point
from causalab.protocol.schema import parse_document
from causalab.protocol.validate import validate_document

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


def _constant(raw: dict[str, Any]) -> frozenset[str]:
    doc = parse_document(in_order(raw))
    validate_document(doc, engine_is_local=True)
    return fit_constant_models(doc)


def _train_section(params: list[str]) -> dict[str, Any]:
    return {
        "objective": [[1.0, "ce"]],
        "params": params,
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
        "seed": 0,
    }


def das_doc() -> dict[str, Any]:
    """The shipped DAS shape: a swap through a trained rotation on both the
    operand read and the write."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 8, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc["method"]["metrics"]["ce"] = {
        "kind": "cross_entropy",
        "of": "logits",
        "target": "label",
        "token_form": "space_prefixed",
    }
    doc["method"]["train"] = _train_section(["rot"])
    doc["method"]["save"].append(
        {"value": "ce", "model": "patched", "input": "base", "file_path": "ce.json"}
    )
    doc["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    return doc


def dbm_doc() -> dict[str, Any]:
    doc = das_doc()
    doc["method"]["featurizers"] = {"gate": {"kind": "gate"}}
    doc["method"]["reads"]["v_cf"]["featurizer"] = "gate"
    doc["method"]["writes"]["patch"]["featurizer"] = "gate"
    doc["method"]["train"]["params"] = ["gate"]
    doc["method"]["train"]["objective"] = [[1.0, "ce"], [0.01, {"l1": "gate"}]]
    doc["method"]["train"]["anneal"] = {"gate.theta.temperature": [1.0, 0.01, 0.5]}
    doc["method"]["save"][-1] = {
        "value": "gate",
        "site": "tgt",
        "file_path": "gate.safetensors",
    }
    return doc


def _with_second_model(doc: dict[str, Any], name: str, write: dict[str, Any]) -> None:
    """Add an intervened model ``name`` on ``base`` carrying one write, plus
    the read that makes it a sink (§5 rule 11). Sections share one namespace
    (rule 3), so the write is ``w_<name>``."""
    doc["method"]["writes"][f"w_{name}"] = write
    doc["method"]["intervened_models"][name] = {
        "input": "base",
        "writes": [f"w_{name}"],
    }
    doc["method"]["reads"][f"logits_{name}"] = {
        "site": "lm_head",
        "pos": -1,
        "model": name,
        "input": "base",
    }
    doc["method"]["metrics"][f"ld_{name}"] = {
        "kind": "logit_diff",
        "of": f"logits_{name}",
        "a": "cf_answer",
        "b": "base_answer",
        "token_form": "space_prefixed",
    }
    doc["method"]["save"].append(
        {
            "value": f"ld_{name}",
            "model": name,
            "input": "base",
            "file_path": f"{name}.json",
        }
    )


def test_das_and_dbm_fits_hold_only_the_source_forward_constant() -> None:
    assert _constant(das_doc()) == {"original"}
    assert _constant(dbm_doc()) == {"original"}


def test_an_unfeaturized_write_from_a_constant_read_is_constant() -> None:
    """A second intervened model whose swap goes through no trained
    featurizer and whose operand is read raw off ``original`` cannot move
    during the fit — it is as cacheable as the source forward."""
    doc = das_doc()
    doc["method"]["reads"]["v_cf_raw"] = {
        "site": "tgt",
        "pos": -1,
        "model": "original",
        "input": "counterfactual",
    }
    _with_second_model(
        doc, "plain", {"site": "tgt", "pos": -1, "do": {"swap": "v_cf_raw"}}
    )
    assert _constant(doc) == {"original", "plain"}


def test_a_literal_write_is_constant() -> None:
    """No operand at all: a scaled self-add with a literal coefficient."""
    doc = das_doc()
    _with_second_model(
        doc,
        "scaled",
        {"site": "tgt", "pos": -1, "do": {"add_scaled": {"op": 0.0, "alpha": 0.5}}},
    )
    assert "scaled" in _constant(doc)


def test_a_model_consuming_a_read_on_the_trained_model_is_not_constant() -> None:
    """Reachability is transitive: a write fed by a read *on* ``patched`` moves
    whenever the rotation does, even though the write itself is unfeaturized."""
    doc = das_doc()
    doc["method"]["reads"]["v_patched"] = {
        "site": "tgt",
        "pos": -1,
        "model": "patched",
        "input": "base",
    }
    _with_second_model(
        doc, "chained", {"site": "tgt", "pos": -1, "do": {"swap": "v_patched"}}
    )
    assert _constant(doc) == {"original"}


def test_a_read_featurized_by_the_trained_featurizer_is_not_constant() -> None:
    """The operand read itself carries the trained featurizer, the write does
    not — the value written still changes with every step."""
    doc = das_doc()
    _with_second_model(
        doc, "featurized_operand", {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}
    )
    assert _constant(doc) == {"original"}


def test_a_params_operand_rooted_in_train_params_is_not_constant() -> None:
    """A trainable free tensor is not a featurizer, so ``_uses_trained_featurizer``
    cannot see it — the write's param operands are checked by root name."""
    doc = das_doc()
    doc["method"]["params"] = {"bias": {"shape": [768], "init": "zeros"}}
    doc["method"]["train"]["params"] = ["rot", "bias"]
    _with_second_model(
        doc,
        "shifted",
        {"site": "tgt", "pos": -1, "do": {"add_scaled": {"op": "bias", "alpha": 1.0}}},
    )
    assert _constant(doc) == {"original"}


def test_a_loaded_params_operand_is_constant() -> None:
    """The same write with the tensor loaded from a file instead of trained."""
    doc = das_doc()
    doc["method"]["params"] = {"bias": {"file_path": "p.safetensors"}}
    _with_second_model(
        doc,
        "shifted",
        {"site": "tgt", "pos": -1, "do": {"add_scaled": {"op": "bias", "alpha": 1.0}}},
    )
    assert _constant(doc) == {"original", "shifted"}


def _shifted_by_bias(doc: dict[str, Any]) -> None:
    _with_second_model(
        doc,
        "shifted",
        {"site": "tgt", "pos": -1, "do": {"add_scaled": {"op": "bias", "alpha": 1.0}}},
    )


def _group_digest(raw: dict[str, Any], model: str) -> str:
    doc = parse_document(in_order(raw))
    validate_document(doc, engine_is_local=True)
    (group,) = [g for g in plan_point(doc).groups if g.model == model]
    return group.digest


def test_a_trained_params_operand_enters_the_group_digest() -> None:
    """Two points differing only in ``train.seed`` fit different tensors, so
    a group whose write consumes one must never intern across them — the
    rule the trained-featurizer case already had, extended to a trained free
    tensor. The un-intervened group stays shared: the seed reaches nothing
    in it."""

    def doc(seed: int) -> dict[str, Any]:
        raw = das_doc()
        raw["method"]["params"] = {"bias": {"shape": [768], "init": "zeros"}}
        raw["method"]["train"]["params"] = ["rot", "bias"]
        raw["method"]["train"]["seed"] = seed
        _shifted_by_bias(raw)
        return raw

    assert _group_digest(doc(0), "shifted") != _group_digest(doc(1), "shifted")
    assert _group_digest(doc(0), "original") == _group_digest(doc(1), "original")


def test_a_loaded_params_operand_leaves_the_group_digest_seed_free() -> None:
    """The same write with the tensor loaded rather than trained: the fit's
    seed moves nothing it consumes, so the two points share the group."""

    def doc(seed: int) -> dict[str, Any]:
        raw = das_doc()
        raw["method"]["params"] = {"bias": {"file_path": "p.safetensors"}}
        raw["method"]["train"]["seed"] = seed
        _shifted_by_bias(raw)
        return raw

    assert _group_digest(doc(0), "shifted") == _group_digest(doc(1), "shifted")


def test_unexpanded_writes_are_refused_rather_than_read_as_constant() -> None:
    """A model whose ``writes`` is still a sweep wrapper has writes the
    classifier cannot see. Reading "none visible" as "constant" would be the
    unsafe direction, so a non-point document is refused."""
    raw = das_doc()
    raw["method"]["intervened_models"]["patched"]["writes"] = {"sweep": [["patch"], []]}
    doc = parse_document(in_order(raw))  # unexpanded on purpose: no validate
    with pytest.raises(AssertionError, match="unexpanded"):
        fit_constant_models(doc)


def test_without_a_train_section_every_model_is_constant() -> None:
    """Nothing is being fitted, so nothing moves: the whole graph is constant
    and the answer names every model, ``original`` included."""
    doc = base_doc()
    assert _constant(doc) == {"original", "patched"}
