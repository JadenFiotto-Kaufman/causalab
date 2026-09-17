"""The parity suite: the same documents through the reference engine and the
nnsight + nnterp engine, asserting the answers agree.

This is the new engine's correctness proof for the module-boundary
vocabulary and its numerical oracle. Reads must agree to fp32-eager-CPU
tolerance, write effects on the logits must agree under every mechanism,
and refusals must be the *same* refusal (code and component named) where
the policy is the shared layer's. Two dense trees carry the cases — the
Llama tree and the GPT-2 tree (a tuple-returning mixer under a renamed
child, a fused ``c_attn``, absolute positions) — and the Qwen3.5-MoE hybrid
carries the MoE and ``block_mid`` ones.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.errors import ProtocolError

from tests._helpers import a3b_sweep as sweep
from tests._helpers.parity_docs import (
    MECHANISMS,
    band_doc,
    block_mid_with_later_read_doc,
    mechanism_doc,
    mixed_block_write_precedence_doc,
)
from tests.neural.engines.nnsight_nnterp.conftest import DENSE, ROWS, Family

pytestmark = pytest.mark.smoke

ATOL = sweep.ATOL

# --------------------------------------------------------------------------- #
# reads: every module-boundary component of the dense families, head slices too
# --------------------------------------------------------------------------- #

LLAMA_READS = [
    ("input_ids", None, None),
    ("embeddings", None, None),
    ("block_input", 1, None),
    ("attention_input_norm", 1, None),
    ("attention_query_pre_rope", 1, None),
    ("attention_key_pre_rope", 1, 0),
    ("attention_value_states", 1, None),
    ("attention_probs", 1, None),
    ("attention_premix", 1, None),
    ("attention_premix", 1, 1),  # per-head slice of the o-projection input
    ("attention_result", 1, None),
    ("attention_result", 1, 2),  # derived in the block, through the o_proj envoy
    ("attention_output", 1, None),
    ("block_mid", 1, None),
    ("mlp_input_norm", 1, None),
    ("mlp_input", 1, None),
    ("mlp_activation", 1, None),
    ("mlp_neuron_output", 1, None),
    ("mlp_output", 1, None),
    ("block_output", 1, None),
    ("ln_final", None, None),
    ("lm_head", None, None),
]

#: The GPT-2 tree declares no pre-RoPE or value-state tap (its ``c_attn`` is
#: one fused projection); everything else the Llama tree reads, it reads.
GPT2_READS = [
    (c, layer, head)
    for c, layer, head in LLAMA_READS
    if c
    not in (
        "attention_query_pre_rope",
        "attention_key_pre_rope",
        "attention_value_states",
    )
]


@pytest.mark.parametrize(
    "family,component,layer,head",
    [("llama", *r) for r in LLAMA_READS] + [("gpt2", *r) for r in GPT2_READS],
    indirect=["family"],
)
def test_read_parity(family, component, layer, head):
    hooked, traced = family.read_both(component, layer, head=head)
    sweep.assert_same(
        hooked,
        traced,
        f"{family.name}: read {component!r} (layer {layer}, head {head})",
    )


def test_a_whole_sequence_lm_head_read_taps_the_head_itself(hooks_llama, nnterp_llama):
    """``pos: all`` keeps the head as an ordinary tap (shared/head.py); the
    positional read projects ``ln_final`` rows — both agree with the
    reference engine and with each other at the last position."""
    llama = Family("llama", hooks_llama, nnterp_llama, ROWS)
    whole = sweep.read_doc("lm_head", None, pos="all")
    hooked = llama.hooked(whole, with_cf=False).read_value("r")
    traced = llama.traced(whole, with_cf=False).read_value("r")
    # the two base rows differ in length, so the whole-sequence read is ragged
    assert hooked.widths == traced.widths
    sweep.assert_same(hooked.flat, traced.flat, "whole-sequence logits")
    last = llama.traced(sweep.read_doc("lm_head", None), with_cf=False)
    ends = torch.tensor(traced.widths).cumsum(0) - 1
    sweep.assert_same(
        traced.flat[ends].unsqueeze(1), last.read_value("r"), "projected last logits"
    )


# --------------------------------------------------------------------------- #
# writes: every mechanism, at the residual boundaries and the sub-children
# --------------------------------------------------------------------------- #

WRITE_SITES = [
    ("embeddings", None),
    ("block_input", 1),
    ("attention_input_norm", 1),
    ("attention_premix", 1),
    ("attention_output", 1),
    ("block_mid", 1),
    ("mlp_input_norm", 1),
    ("mlp_input", 1),
    ("mlp_activation", 1),
    ("mlp_neuron_output", 1),
    ("mlp_output", 1),
    ("block_output", 1),
    ("ln_final", None),
    ("lm_head", None),
]


def _assert_write_parity(doc, component, family: Family) -> NnterpExecutor:
    hooked = family.hooked(doc, with_cf=True).dense_value("logits")
    executor = family.traced(doc, with_cf=True)
    traced = executor.dense_value("logits")
    sweep.assert_same(hooked, traced, f"patched logits after a write at {component!r}")
    # anti-vacuity: agreement must not be reachable by "neither write landed"
    assert not torch.allclose(traced, family.unpatched_logits(), atol=ATOL), (
        f"the write at {component!r} left the logits unchanged"
    )
    return executor


@pytest.mark.parametrize("family", DENSE, indirect=True)
@pytest.mark.parametrize("component,layer", WRITE_SITES)
@pytest.mark.parametrize("mechanism", sorted(MECHANISMS))
def test_write_parity(family, component, layer, mechanism):
    doc = mechanism_doc(component, layer, MECHANISMS[mechanism])
    executor = _assert_write_parity(doc, component, family)
    # every member fired exactly once in the one trace (spec §4 "Fires")
    assert executor.fires == {
        ("patched", "base"): {name: 1 for name in MECHANISMS[mechanism]}
    }


@pytest.mark.parametrize("family", DENSE, indirect=True)
def test_a_head_sliced_write_agrees(family):
    doc = mechanism_doc("attention_premix", 1, MECHANISMS["swap"])
    doc["method"]["sites"]["tap"]["head"] = 1
    _assert_write_parity(doc, "attention_premix[head=1]", family)


@pytest.mark.parametrize("family", DENSE, indirect=True)
def test_the_pattern_swap_agrees(family):
    """The attention pattern is written and read at the softmax op inside
    the attention function — the value the value-multiply consumes — and
    agrees with the reference engine's attention-interface landing. The
    bundle is loaded eager, so nothing was switched and the receipt says
    so."""
    doc = sweep.interchange_doc("attention_probs", 1, pos="all")
    executor = _assert_write_parity(doc, "attention_probs", family)
    assert executor.applied_requirements == set()


@pytest.mark.parametrize("family", DENSE, indirect=True)
def test_a_pattern_read_in_the_written_group_sees_the_write(family):
    """A pattern write and a pattern read in one group: both land on the
    softmax op, the write first, so the read sees the swapped-in
    counterfactual pattern — on both engines."""
    doc = sweep.interchange_doc("attention_probs", 1, pos="all")
    doc["method"]["reads"]["r_pattern"] = {
        "site": "tap",
        "pos": "all",
        "model": "patched",
        "input": "base",
    }
    doc["method"]["save"].append(
        {
            "value": "r_pattern",
            "model": "patched",
            "input": "base",
            "file_path": "p.safetensors",
        }
    )
    hooked = family.hooked(doc, with_cf=True)
    traced = family.traced(doc, with_cf=True)
    seen = traced.read_value("r_pattern")
    sweep.assert_same(traced.read_value("v_cf"), seen, "the pattern its own write set")
    sweep.assert_same(hooked.read_value("r_pattern"), seen, "pattern read parity")


def test_the_pattern_is_read_under_an_on_demand_eager_switch(
    hooks_llama, nnterp_llama_default_impl
):
    """A bundle loaded under the checkpoint's default (sdpa, which never
    materializes the pattern) reads and writes it by switching to eager
    around its trace, stamps the switch, and restores the default after."""
    bundle = nnterp_llama_default_impl
    assert bundle.model.config._attn_implementation == "sdpa"
    llama = Family("llama", hooks_llama, bundle, ROWS)
    doc = sweep.read_doc("attention_probs", 1, pos="all")
    executor = llama.traced(doc, with_cf=False)
    traced = executor.read_value("r")
    hooked = llama.hooked(doc, with_cf=False).read_value("r")
    sweep.assert_same(hooked, traced, "pattern read under the switch")
    assert executor.applied_requirements == {"attn_eager"}
    assert bundle.model.config._attn_implementation == "sdpa"
    doc = sweep.interchange_doc("attention_probs", 1, pos="all")
    executor = llama.traced(doc, with_cf=True)
    sweep.assert_same(
        llama.hooked(doc, with_cf=True).dense_value("logits"),
        executor.dense_value("logits"),
        "pattern write under the switch",
    )
    assert executor.applied_requirements == {"attn_eager"}
    assert bundle.model.config._attn_implementation == "sdpa"


# --------------------------------------------------------------------------- #
# block_mid: the deferred write-back, alone and composed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "family,later_component,layer",
    (
        ("llama", "mlp_output", 1),
        ("llama", "mlp_neuron_output", 1),
        ("gpt2", "mlp_output", 1),
        ("qwen", "router_scores", 0),
        ("qwen", "shared_expert_activation", 3),
    ),
    indirect=["family"],
)
def test_block_mid_write_allows_later_same_layer_reads(family, later_component, layer):
    """The deferred write-back lands at the block's output, after every
    same-layer read between the norm's input and the block's output."""
    doc = block_mid_with_later_read_doc(later_component, layer)
    hooked = family.hooked(doc, with_cf=True)
    traced = family.traced(doc, with_cf=True)
    base_mid = hooked.read_value("v_mid_base")
    cf_mid = hooked.read_value("v_mid_cf")
    assert not torch.allclose(base_mid, cf_mid, atol=ATOL)
    sweep.assert_same(base_mid, traced.read_value("v_mid_base"), "clean block_mid")
    sweep.assert_same(cf_mid, traced.read_value("v_mid_cf"), "counterfactual block_mid")
    for name in ("r_later", "r_out"):
        sweep.assert_same(
            hooked.read_value(name),
            traced.read_value(name),
            f"block_mid write followed by {name}",
        )


@pytest.mark.parametrize("family", DENSE, indirect=True)
def test_mixed_block_write_precedence_agrees(family):
    """An output swap supersedes the mid write-back; an output addition
    retains it — in either authored order."""
    doc = mixed_block_write_precedence_doc(1)
    hooked = family.hooked(doc, with_cf=True)
    traced = family.traced(doc, with_cf=True)
    expected = hooked.read_value("v_out_cf")
    for model in ("mid_then_out", "out_then_mid"):
        value = traced.read_value(f"r_{model}")
        sweep.assert_same(hooked.read_value(f"r_{model}"), value, model)
        sweep.assert_same(expected, value, f"absolute output precedence for {model}")
    sweep.assert_same(
        hooked.read_value("r_mid_plus_out"),
        traced.read_value("r_mid_plus_out"),
        "write-back followed by an additive output write",
    )


# --------------------------------------------------------------------------- #
# a band site, and several sites in one trace
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", DENSE, indirect=True)
def test_a_band_runs_as_its_hand_written_twin_and_agrees(family):
    band = family.traced(band_doc(one_site=True), with_cf=True)
    hand = family.traced(band_doc(one_site=False), with_cf=True)
    assert sorted(band.doc.sites) == ["a[layers=0]", "a[layers=1]", "head"]  # lowered
    assert torch.equal(band.read_value("logits"), hand.read_value("logits"))
    hooked = family.hooked(band_doc(one_site=True), with_cf=True)
    sweep.assert_same(
        hooked.read_value("logits"), band.read_value("logits"), "band logits"
    )


def test_many_reads_in_one_trace_are_issued_in_forward_order(hooks_llama, nnterp_llama):
    """One group reading every layer-1 boundary plus the trunk: the schedule
    must request them in the model's own order or nnsight refuses."""
    llama = Family("llama", hooks_llama, nnterp_llama, ROWS)
    components = [c for c, layer, head in LLAMA_READS if head is None]
    doc = sweep.read_doc("block_output", 1)
    doc["method"]["sites"] = {
        c: {"component": c, **({} if sweep.layerless(c) else {"layers": 1})}
        for c in components
    }
    doc["method"]["reads"] = {
        f"r_{c}": {
            "site": c,
            "pos": sweep.default_pos(c),
            "model": "original",
            "input": "base",
        }
        for c in components
    }
    doc["method"]["save"] = [
        {
            "value": f"r_{c}",
            "model": "original",
            "input": "base",
            "file_path": f"{c}.safetensors",
        }
        for c in components
    ]
    hooked = llama.hooked(doc, with_cf=False)
    traced = llama.traced(doc, with_cf=False)
    traced.run_all()
    assert len(traced._groups_run) == 1
    for c in components:
        sweep.assert_same(hooked.read_value(f"r_{c}"), traced.read_value(f"r_{c}"), c)


def test_qwen_moe_block_reads_in_one_trace_follow_the_forward(hooks_qwen, nnterp_qwen):
    """📐 The shared expert runs before the router in
    ``Qwen3_5MoeSparseMoeBlock.forward``, and the rank table says so; one
    trace reads every MoE boundary in that order."""
    qwen = Family("qwen", hooks_qwen, nnterp_qwen, ROWS)
    components = [
        "mlp_input",
        "router_logits",
        "router_scores",
        "expert_idx",
        "routed_output",
        "shared_expert_gate_proj",
        "shared_expert_up_proj",
        "shared_expert_activation",
        "shared_expert_output",
        "shared_expert_gate",
        "mlp_output",
    ]
    doc = sweep.read_doc("mlp_output", 0)
    doc["method"]["sites"] = {c: {"component": c, "layers": 0} for c in components}
    doc["method"]["reads"] = {
        f"r_{c}": {"site": c, "pos": -1, "model": "original", "input": "base"}
        for c in components
    }
    doc["method"]["save"] = [
        {
            "value": f"r_{c}",
            "model": "original",
            "input": "base",
            "file_path": f"{c}.safetensors",
        }
        for c in components
    ]
    hooked = qwen.hooked(doc, with_cf=False)
    traced = qwen.traced(doc, with_cf=False)
    traced.run_all()
    for c in components:
        sweep.assert_same(hooked.read_value(f"r_{c}"), traced.read_value(f"r_{c}"), c)


# --------------------------------------------------------------------------- #
# refusals: the same policy, the same words
# --------------------------------------------------------------------------- #


def _refusal(executor_cls, doc, bundle) -> str:
    with pytest.raises(ProtocolError) as excinfo:
        sweep.make_executor(
            executor_cls, doc, bundle, rows=ROWS, with_cf=True
        ).run_all()
    return str(excinfo.value)


def test_read_only_refusal_is_identical(hooks_qwen, nnterp_qwen):
    doc = sweep.interchange_doc("router_logits", 0)
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        NnterpExecutor, doc, nnterp_qwen
    )


def test_swap_only_refusal_is_identical(hooks_qwen, nnterp_qwen):
    doc = sweep.interchange_doc("expert_idx", 0)
    doc["method"]["writes"]["patch"]["do"] = {
        "add_scaled": {"op": "v_cf", "alpha": 2.0}
    }
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        NnterpExecutor, doc, nnterp_qwen
    )


def test_wrong_stream_refusal_is_identical(hooks_qwen, nnterp_qwen):
    qwen = Family("qwen", hooks_qwen, nnterp_qwen, ROWS)
    doc = sweep.read_doc("attention_output", 0)
    doc["method"]["sites"]["tap"]["stream"] = "full_attention"  # layer 0 is DeltaNet
    with pytest.raises(ProtocolError) as hooks_err:
        qwen.hooked(doc, with_cf=False).run_all()
    with pytest.raises(ProtocolError) as trace_err:
        qwen.traced(doc, with_cf=False).run_all()
    assert str(hooks_err.value) == str(trace_err.value)


def test_a_grad_enabled_group_runs_smoke(nnterp_llama):
    """Smoke only: a grad-enabled executor runs its trace under
    ``enable_grad`` and hands back the same numbers. Nothing here observes
    the graph — the network is frozen and the document carries no trainable
    stage — so this pins that the path runs, not that it differentiates."""
    from causalab.protocol.schema import parse_document
    from causalab.protocol.validate import validate_document

    from tests.protocol._docs import in_order

    doc = parse_document(in_order(sweep.read_doc("block_output", 1)))
    validate_document(doc, engine_is_local=True)
    kwargs = dict(
        role_rows={"base": ROWS},
        role_fields={"base": "input"},
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )
    plain = NnterpExecutor(doc, nnterp_llama, **kwargs).read_value("r")
    under_grad = NnterpExecutor(
        doc, nnterp_llama, grad_enabled=True, **kwargs
    ).read_value("r")
    assert torch.equal(plain, under_grad)
