"""The attention interior on the nnsight + nnterp engine.

The same treatment the module-boundary vocabulary got, extended to the
attention function's slots: the same documents through both engines,
agreeing to fp32-eager-CPU tolerance. Two independent implementations —
the reference engine's attention-interface wrapper vs this engine's
``.source`` address navigation — agreeing is the strongest check there is:
a wrong address (``attn_weights_0``, the pre-mask tensor, say) produces
plausible numbers of the right shape, and only the comparison catches it.
q and k are read off the interface call's arguments, so the GPT-2 tree —
no RoPE op in its forward — serves them through the same rows.

Plus what parity alone cannot pin: the identities (``softmax(scores) ==
pattern`` exactly; rows sum to 1; ``z·σ(gate) == premix``), the causal
writes (a targeted knockout moves the logits, a uniform shift is a
softmax-invariance no-op), the in-forward ordering discipline the
``.source`` interiors demand, and the on-demand implementation switch.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.errors import ProtocolError

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnsight_nnterp.conftest import (
    ROWS,
    Family,
    assert_same,
)

pytestmark = pytest.mark.smoke

#: the anti-vacuity band — a write has to move a value by more than float
#: noise to count as having landed; agreement itself is exact (conftest)
ATOL = sweep.ATOL

#: layer 3 is the qwen fixture's one full-attention layer; the dense trees
#: are full attention everywhere
LAYER = {"qwen": 3, "llama": 1, "gpt2": 1}


def _write_doc(component: str, layer: int, do: dict, *, pos: object = "all") -> dict:
    """Patch one interior site and read the last-position logits."""
    doc = sweep.interchange_doc(component, layer, pos=pos)
    doc["method"]["writes"]["patch"]["do"] = do
    return doc


def _refusal(executor_cls, doc, bundle) -> str:
    with pytest.raises(ProtocolError) as excinfo:
        sweep.make_executor(
            executor_cls, doc, bundle, rows=ROWS, with_cf=True
        ).run_all()
    return str(excinfo.value)


# --------------------------------------------------------------------------- #
# reads: the whole attention-interior surface, both engines, three trees
# --------------------------------------------------------------------------- #

#: (component, pos, head). The two pattern-shaped components have no contract
#: form, so they are read whole; everything else reads a position like any
#: other tap.
INTERIOR_READS = [
    ("attention_query", -1, None),
    ("attention_query", -1, 1),
    ("attention_key", -1, None),
    ("attention_scores", "all", None),
    ("attention_probs", "all", None),
    ("attention_z", -1, None),
    ("attention_z", -1, 1),
]

#: The gated family's own module boundaries around the interior, on qwen.
QWEN_READS = INTERIOR_READS + [
    ("attention_query_pre_rope", -1, None),
    ("attention_key_pre_rope", -1, None),
    ("attention_value_states", -1, None),
    ("attention_gate", -1, None),
    ("attention_premix", -1, 1),
    ("attention_result", -1, 1),
]


@pytest.mark.parametrize(
    "family,component,pos,head",
    [("qwen", *r) for r in QWEN_READS]
    + [("llama", *r) for r in INTERIOR_READS]
    + [("gpt2", *r) for r in INTERIOR_READS],
    indirect=["family"],
)
def test_interior_read_parity(family, component, pos, head):
    hooked, traced = family.read_both(component, LAYER[family.name], pos=pos, head=head)
    assert_same(
        hooked, traced, f"{family.name}: read {component!r} (pos {pos}, head {head})"
    )


# --------------------------------------------------------------------------- #
# identities: the pins that say each address is where it claims to be
# --------------------------------------------------------------------------- #


def _traced_read(family: Family, component: str, **kw) -> torch.Tensor:
    doc = sweep.read_doc(component, LAYER[family.name], **kw)
    return family.traced(doc, with_cf=False).read_value("r")


@pytest.mark.parametrize("family", ["qwen", "llama", "gpt2"], indirect=True)
def test_the_scores_softmax_to_the_pattern_exactly(family):
    """📐 ``softmax(attn_weights_1) == attn_weights_2`` at 0.0 — the wrong
    pick (``attn_weights_0``, pre-mask) fails this before parity even runs."""
    scores = _traced_read(family, "attention_scores", pos="all")
    probs = _traced_read(family, "attention_probs", pos="all")
    torch.testing.assert_close(
        torch.softmax(scores.float(), dim=-1), probs.float(), atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("family", ["qwen", "llama", "gpt2"], indirect=True)
def test_the_pattern_rows_sum_to_one(family):
    probs = _traced_read(family, "attention_probs", pos="all")
    torch.testing.assert_close(
        probs.float().sum(-1), torch.ones(probs.shape[:-1]), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("family", ["qwen"], indirect=True)
def test_z_times_gate_is_the_premix(family):
    """The gated family's defining identity: the o-projection's input is
    ``z · σ(gate)``. All three taps are head-major in the contract, so the
    identity is elementwise there."""
    z = _traced_read(family, "attention_z", pos=-1)
    gate = _traced_read(family, "attention_gate", pos=-1)
    premix = _traced_read(family, "attention_premix", pos=-1)
    torch.testing.assert_close(
        z.float() * torch.sigmoid(gate.float()), premix.float(), atol=0.0, rtol=0.0
    )


# --------------------------------------------------------------------------- #
# writes: interchanges agree, and the causal checks are not vacuous
# --------------------------------------------------------------------------- #

#: pos "all" only where the tap has no contract form (the two pattern-shaped
#: components, edited whole); everywhere else -1 — the fixtures' two rows
#: tokenize to different lengths, and ragged *writes* are refused (v1, the
#: same refusal on both engines).
INTERIOR_WRITES = [
    ("attention_query", -1),
    ("attention_key", -1),
    ("attention_scores", "all"),
    ("attention_probs", "all"),
    ("attention_z", -1),
]


@pytest.mark.parametrize(
    "family,component,pos",
    [("qwen", *w) for w in INTERIOR_WRITES]
    + [("llama", *w) for w in INTERIOR_WRITES]
    + [("gpt2", *w) for w in INTERIOR_WRITES],
    indirect=["family"],
)
def test_interior_write_parity(family, component, pos):
    doc = _write_doc(component, LAYER[family.name], {"swap": "v_cf"}, pos=pos)
    hooked = family.hooked(doc, with_cf=True)
    traced = family.traced(doc, with_cf=True)
    assert_same(
        hooked.dense_value("logits"),
        traced.dense_value("logits"),
        f"{family.name}: patched logits after a swap at {component!r}",
    )
    assert not torch.allclose(
        traced.dense_value("logits"), family.unpatched_logits(), atol=ATOL
    ), "the swap landed nowhere"
    assert traced.fires == {("patched", "base"): {"patch": 1}}


@pytest.mark.parametrize("family", ["qwen", "llama", "gpt2"], indirect=True)
def test_query_and_key_written_together_agree(family):
    """q and k are arguments 1 and 2 of one interface call: two writes on
    one ``inputs`` handle, each rebuilding the container around its own
    argument. Both land, and the pair's effect agrees with the reference
    engine's two slots and is neither single write's."""
    layer = LAYER[family.name]
    doc = sweep.interchange_doc("attention_query", layer)
    doc["method"]["sites"]["k"] = {"component": "attention_key", "layers": layer}
    doc["method"]["reads"]["k_cf"] = {
        "site": "k",
        "pos": -1,
        "model": "original",
        "input": "counterfactual",
    }
    doc["method"]["writes"]["patch_k"] = {
        "site": "k",
        "pos": -1,
        "do": {"swap": "k_cf"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"] = ["patch", "patch_k"]
    hooked = family.hooked(doc, with_cf=True).dense_value("logits")
    executor = family.traced(doc, with_cf=True)
    traced = executor.dense_value("logits")
    assert_same(hooked, traced, f"{family.name}: logits after q and k swapped")
    assert executor.fires == {("patched", "base"): {"patch": 1, "patch_k": 1}}
    for component in ("attention_query", "attention_key"):
        single = family.traced(sweep.interchange_doc(component, layer), with_cf=True)
        assert not torch.allclose(single.dense_value("logits"), traced, atol=ATOL), (
            component
        )


def _moved(family: Family, doc: dict, load_tensors=None) -> float:
    executor = family.traced(doc, with_cf=True)
    if load_tensors is not None:
        executor.load_tensors = load_tensors
    return float(
        (executor.dense_value("logits") - family.unpatched_logits()).abs().max()
    )


@pytest.mark.parametrize("family", ["qwen"], indirect=True)
def test_a_targeted_knockout_on_the_scores_moves_the_logits(family):
    """Attention knockout as arithmetic on the scores — the capability the
    component exists for. Head 0 blocked from attending to token 0, as a
    full-shape mask added upstream of the model's own softmax."""
    from tests.neural.engines.pytorch_hooks._drive import bundle_loader

    doc = _write_doc(
        "attention_scores", LAYER["qwen"], {"add_scaled": {"op": "knock", "alpha": 1.0}}
    )
    del doc["method"]["reads"]["v_cf"]
    doc["method"]["params"] = {"knock": {"file_path": "k.safetensors"}}
    mask = torch.zeros_like(_traced_read(family, "attention_scores", pos="all"))
    mask[:, 0, :, 0] = -1e4
    moved = _moved(
        family, doc, load_tensors=bundle_loader({"k.safetensors": {"value": mask}})
    )
    assert moved > 1e-3


@pytest.mark.parametrize("family", ["qwen"], indirect=True)
def test_a_uniform_shift_of_the_scores_is_a_no_op(family):
    """Softmax is shift-invariant along the axis it normalizes, so adding the
    same constant to every score changes nothing — pinned so a knockout
    recipe stays targeted."""
    doc = _write_doc(
        "attention_scores",
        LAYER["qwen"],
        {"add_scaled": {"op": -10000.0, "alpha": 1.0}},
    )
    del doc["method"]["reads"]["v_cf"]
    assert _moved(family, doc) < 1e-3


@pytest.mark.parametrize("family", ["qwen", "gpt2"], indirect=True)
@pytest.mark.parametrize("component", ["attention_query", "attention_probs"])
def test_a_read_of_a_written_slot_sees_the_written_value(family, component):
    """Write-before-read at one address, within one trace: the read observes
    the written value (difference 0.0), matching the reference engine's
    hook-registration order — the two engines have to agree here or the
    same document would mean different things. q lands on the call's
    arguments, the pattern on the softmax op: both handles read back."""
    pos = sweep.default_pos(component)
    doc = sweep.interchange_doc(component, LAYER[family.name], pos=pos)
    doc["method"]["reads"]["obs"] = {
        "site": "tap",
        "pos": pos,
        "model": "patched",
        "input": "base",
    }
    doc["method"]["save"].append(
        {
            "value": "obs",
            "model": "patched",
            "input": "base",
            "file_path": "o.safetensors",
        }
    )
    for executor in (
        family.traced(doc, with_cf=True),
        family.hooked(doc, with_cf=True),
    ):
        src, obs = executor.read_value("v_cf"), executor.read_value("obs")
        assert float((obs - src).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# ordering: the rank band IS the in-forward op order
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", ["qwen", "gpt2"], indirect=True)
def test_several_interior_reads_share_one_trace_in_forward_order(family):
    """`.source` ops refuse out-of-order requests (OutOfOrderError), so this
    doc — the five interior taps plus a boundary tap in one group — passes
    only if the (layer, COMPONENT_RANK) sort key already walks the forward:
    the call's inputs, then the drill into it, then its output."""
    layer = LAYER[family.name]
    components = [
        "attention_query",
        "attention_key",
        "attention_scores",
        "attention_probs",
        "attention_z",
        "attention_output",
    ]
    doc = sweep.read_doc("attention_output", layer)
    doc["method"]["sites"] = {c: {"component": c, "layers": layer} for c in components}
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
    hooked = family.hooked(doc, with_cf=False)
    traced = family.traced(doc, with_cf=False)
    traced.run_all()
    assert len(traced._groups_run) == 1
    for c in components:
        assert_same(
            hooked.read_value(f"r_{c}"), traced.read_value(f"r_{c}"), f"grouped {c!r}"
        )


# --------------------------------------------------------------------------- #
# refusals: same policy, same words
# --------------------------------------------------------------------------- #


def test_a_delta_on_the_pattern_refuses_identically(hooks_qwen, nnterp_qwen):
    doc = _write_doc(
        "attention_probs", LAYER["qwen"], {"add_scaled": {"op": -1.0, "alpha": 1.0}}
    )
    del doc["method"]["reads"]["v_cf"]
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        NnterpExecutor, doc, nnterp_qwen
    )


def test_a_positioned_read_of_the_scores_refuses_identically(hooks_qwen, nnterp_qwen):
    doc = sweep.read_doc("attention_scores", LAYER["qwen"], pos=-1)
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        NnterpExecutor, doc, nnterp_qwen
    )


def test_the_interior_at_a_deltanet_layer_refuses_architecturally(nnterp_qwen):
    doc = sweep.read_doc("attention_scores", 0, pos="all")  # DeltaNet on this fixture
    with pytest.raises(ProtocolError, match="full-attention mixer"):
        sweep.make_executor(
            NnterpExecutor, doc, nnterp_qwen, rows=ROWS, with_cf=False
        ).run_all()


# --------------------------------------------------------------------------- #
# the on-demand implementation switch
# --------------------------------------------------------------------------- #


def test_the_switch_serves_the_scores_from_an_sdpa_loaded_model(
    hooks_llama, nnterp_llama, nnterp_llama_default_impl
):
    """The engine's own loading path keeps the checkpoint default (sdpa); a
    group whose address requires eager switches around its trace, restores
    the default after, and stamps what it applied."""
    bundle = nnterp_llama_default_impl
    default = bundle.model.config._attn_implementation
    assert default != "eager"  # or this test is vacuous
    doc = sweep.read_doc("attention_scores", LAYER["llama"], pos="all")
    executor = sweep.make_executor(
        NnterpExecutor, doc, bundle, rows=ROWS, with_cf=False
    )
    switched = executor.read_value("r")
    assert bundle.model.config._attn_implementation == default  # restored
    assert executor.applied_requirements == {"attn_eager"}
    pinned = Family("llama", hooks_llama, nnterp_llama, ROWS)
    assert_same(
        pinned.traced(doc, with_cf=False).read_value("r"),
        switched,
        "scores through the runtime switch",
    )


def test_a_group_that_needs_no_switch_applies_none(nnterp_llama_default_impl):
    """`attention_z` is the interface's own return and exists under every
    implementation — a document reading only it never forces eager (the
    switch's payoff)."""
    executor = sweep.make_executor(
        NnterpExecutor,
        sweep.read_doc("attention_z", LAYER["llama"]),
        nnterp_llama_default_impl,
        rows=ROWS,
        with_cf=False,
    )
    assert executor.read_value("r").numel() > 0
    assert executor.applied_requirements == set()
    assert nnterp_llama_default_impl.model.config._attn_implementation != "eager"
