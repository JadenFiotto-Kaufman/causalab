"""The generated frame on the nnsight + nnterp engine.

One ``model.generate`` trace per group: prompt-frame taps and writes bind
occurrence 0 of their locations — the prefill — and the decode steps are
walked with ``tracer.iter``, body ``j`` being the forward that consumes
generated token ``j-1``. The reference engine hand-rolls the same decode
with hooks, which makes it the oracle here: the same documents through both,
ids and activations agreeing, on all three trees.

Plus what parity cannot say: the greedy self-consistency pin (the argmax of a
continuation ``lm_head`` read reproduces the decoded ids), the eos cut of the
frame's widths, the bridge to the DeltaNet interior — the state read per
decode step through the *recurrent* kernel's own address, continuous with
the prefill chunks — and the refusals: interiors with no decode address,
unstackable slots (the same words as the reference engine), and a write in
the continuation (rule 16, before any engine).
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_nnterp.engine import NnterpEngine
from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.shared.encoding import continuation_widths
from causalab.protocol.engine import requires
from causalab.protocol.errors import ProtocolError, ValidationError
from causalab.protocol.schema import parse_document
from causalab.protocol.validate import validate_document

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnsight_nnterp.conftest import (
    ROWS,
    Family,
    assert_same,
    single_row,
)
from tests.protocol._docs import in_order

pytestmark = pytest.mark.smoke

DEPTH = 4
#: the qwen fixture's one full-attention layer and its first DeltaNet layer
QWEN_ATTENTION_LAYER = 3
DELTANET_LAYER = 0
LAYER = {"llama": 1, "gpt2": 1, "qwen": DELTANET_LAYER}


def _window(**anchor) -> dict:
    """A continuation position: the decode budget plus one anchor."""
    return {"generated": {"max_new_tokens": DEPTH}, **(anchor or {"all": True})}


def _gen_doc(component: str, layer: int | None, *, pos: dict | None = None) -> dict:
    return sweep.read_doc(component, layer, pos=pos or _window())


def _refusal(executor_cls, doc, bundle) -> str:
    with pytest.raises(ProtocolError) as excinfo:
        sweep.make_executor(
            executor_cls, doc, bundle, rows=ROWS, with_cf=False
        ).run_all()
    return str(excinfo.value)


# --------------------------------------------------------------------------- #
# parity: the reference engine's decode is the oracle
# --------------------------------------------------------------------------- #

GEN_READS = [
    ("block_output", _window()),
    ("block_output", _window(index=-1)),
    ("block_output", _window(index=1)),
    ("block_output", _window(span=[1, 3])),
    ("attention_output", _window()),
    ("mlp_output", _window()),
    ("attention_query", _window()),
    ("attention_z", _window(index=-1)),
    ("ln_final", _window()),
    ("lm_head", _window()),
    ("lm_head", _window(index=-1)),
]


@pytest.mark.parametrize("family", ["llama", "gpt2", "qwen"], indirect=True)
@pytest.mark.parametrize("component,pos", GEN_READS)
def test_generated_read_parity(family: Family, component, pos):
    layer = None if sweep.layerless(component) else LAYER[family.name]
    if family.name == "qwen" and component in ("attention_query", "attention_z"):
        layer = QWEN_ATTENTION_LAYER
    doc = _gen_doc(component, layer, pos=pos)
    hooked = family.hooked(doc, with_cf=False).read_value("r")
    traced = family.traced(doc, with_cf=False).read_value("r")
    assert_same(hooked, traced, f"{family.name}: generated read of {component!r}")
    assert traced.shape[1] == (
        DEPTH if pos.get("all") else len(range(*pos["span"])) if "span" in pos else 1
    )


@pytest.mark.parametrize("family", ["llama", "gpt2", "qwen"], indirect=True)
def test_the_decoded_ids_agree_with_the_reference_engine(family: Family):
    """Same greedy continuation on both engines — the frame itself, not just
    the activations — and the same addressed steps."""
    doc = _gen_doc("block_output", LAYER[family.name])
    hooked = family.hooked(doc, with_cf=False)
    traced = family.traced(doc, with_cf=False)
    hooked.read_value("r"), traced.read_value("r")
    assert hooked.generated_ids("r") == traced.generated_ids("r")
    assert hooked.addressed_steps("r") == traced.addressed_steps("r")
    assert all(len(steps) == DEPTH for steps in traced.addressed_steps("r"))


@pytest.mark.parametrize("family", ["llama", "gpt2", "qwen"], indirect=True)
def test_prompt_and_continuation_reads_share_one_generate_trace(family: Family):
    """A prompt-frame read in a decoding group binds the prefill and agrees
    with the same read in a non-decoding group; the continuation read beside
    it agrees with the reference engine."""
    layer = LAYER[family.name]
    doc = _gen_doc("block_output", layer)
    doc["method"]["sites"]["prompt"] = {"component": "block_output", "layers": layer}
    doc["method"]["reads"]["p"] = {
        "site": "prompt",
        "pos": -1,
        "model": "original",
        "input": "base",
    }
    doc["method"]["save"].append(
        {
            "value": "p",
            "model": "original",
            "input": "base",
            "file_path": "p.safetensors",
        }
    )
    traced = family.traced(doc, with_cf=False)
    alone = family.traced(sweep.read_doc("block_output", layer, pos=-1), with_cf=False)
    torch.testing.assert_close(traced.read_value("p"), alone.read_value("r"))
    hooked = family.hooked(doc, with_cf=False)
    assert_same(hooked.read_value("r"), traced.read_value("r"), "continuation")


@pytest.mark.parametrize("family", ["llama", "gpt2", "qwen"], indirect=True)
def test_a_write_reaches_the_continuation_identically(family: Family):
    """Writes are prefill-only on both engines (here: everything binds
    occurrence 0 — the prefill), and reach the continuation through the
    first token and the cache. The patched continuation read must agree."""
    layer = LAYER[family.name]
    doc = _gen_doc("block_output", layer)
    doc["data"] = sweep._data(with_cf=True)
    doc["method"]["sites"]["src"] = {"component": "block_output", "layers": [layer]}
    doc["method"]["reads"]["v_cf"] = {
        "site": "src",
        "pos": -1,
        "model": "original",
        "input": "counterfactual",
    }
    doc["method"]["reads"]["r"]["model"] = "patched"
    doc["method"]["writes"] = {
        "patch": {"site": "src", "pos": -1, "do": {"swap": "v_cf"}}
    }
    doc["method"]["intervened_models"] = {
        "patched": {"input": "base", "writes": ["patch"]}
    }
    doc["method"]["save"][0]["model"] = "patched"
    hooked = family.hooked(doc, with_cf=True)
    traced = family.traced(doc, with_cf=True)
    assert_same(
        hooked.read_value("r"), traced.read_value("r"), "a patched continuation read"
    )
    assert hooked.generated_ids("r") == traced.generated_ids("r")
    assert traced.fires == {("patched", "base"): {"patch": 1}}


# --------------------------------------------------------------------------- #
# the greedy self-consistency pin and the frame's widths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", ["llama", "gpt2", "qwen"], indirect=True)
def test_the_lm_head_argmax_reproduces_the_decoded_ids(family: Family):
    """The distribution at generated position i is the one AFTER token i, so
    its argmax is token i+1 — the frame and the values must tell one story."""
    executor = family.traced(_gen_doc("lm_head", None), with_cf=False)
    value = executor.read_value("r")  # (b, steps, vocab)
    ids = executor.generated_ids("r")
    steps = executor.addressed_steps("r")
    for row, row_steps in enumerate(steps):
        for k, step in enumerate(row_steps[:-1]):
            assert int(value[row, step].argmax()) == ids[row][k + 1]


def test_the_frame_is_cut_at_each_rows_first_eos():
    """The decode runs every row to the full depth (``eos_token_id=None``);
    the frame's widths cut each row before its first eos — an eos at step 0
    is an empty row, no eos is the whole depth, and a later eos in the same
    row changes nothing."""
    eos = 7
    generated = torch.tensor(
        [[1, 2, 3, 4], [1, eos, 3, eos], [eos, 2, 3, 4], [1, 2, 3, eos]]
    )
    assert continuation_widths(generated, (eos,)) == (4, 1, 0, 3)
    assert continuation_widths(generated, (eos, 4)) == (3, 1, 0, 3)
    assert continuation_widths(generated, ()) == (4, 4, 4, 4)


# --------------------------------------------------------------------------- #
# the bridge to the DeltaNet interior: the state per decode step
# --------------------------------------------------------------------------- #


def test_the_deltanet_state_reads_per_decode_step(nnterp_qwen, hooks_qwen):
    """Served through the *recurrent* kernel's address — the decode path's own
    dispatch — with one state per generated position: continuous in shape
    with the prefill chunks (read in the same trace), advancing every step,
    and equal to the reference engine's per-token ``delta_state`` at the same
    steps (the registry's backend pair, which per step needs no chunk
    mapping)."""
    info = nnterp_qwen.info
    doc = _gen_doc("deltanet_state", DELTANET_LAYER)
    doc["method"]["sites"]["prefill"] = {
        "component": "deltanet_state",
        "layers": [DELTANET_LAYER],
    }
    doc["method"]["reads"]["last_chunk"] = {
        "site": "prefill",
        "pos": -1,
        "model": "original",
        "input": "base",
    }
    doc["method"]["save"].append(
        {
            "value": "last_chunk",
            "model": "original",
            "input": "base",
            "file_path": "c.safetensors",
        }
    )
    text = "the quick brown fox jumps"
    executor = single_row(nnterp_qwen, doc, text)
    per_step = executor.read_value("r")
    width = (
        info.linear_num_value_heads
        * info.linear_key_head_dim
        * info.linear_value_head_dim
    )
    assert tuple(per_step.shape) == (1, DEPTH, width)
    for j in range(DEPTH - 1):
        assert float((per_step[:, j + 1] - per_step[:, j]).abs().max()) > 0.0
    last_chunk = executor.read_value("last_chunk")
    drift_from_prefill = float((per_step[:, 0] - last_chunk[:, 0]).abs().max())
    fresh_scale = float(per_step[:, 0].abs().max())
    assert 0.0 < drift_from_prefill < fresh_scale * 10
    assert sweep.backend_pair("delta_state").names == {"delta_state", "deltanet_state"}
    hooked = sweep.make_executor(
        PointExecutor,
        _gen_doc("delta_state", DELTANET_LAYER),
        hooks_qwen,
        rows=[{"input": text}],
        with_cf=False,
    ).read_value("r")
    # the reference engine's state keeps its (heads, d_k, d_v) axes; the
    # per-chunk face declares one flat width
    assert_same(hooked.reshape(per_step.shape), per_step, "the state per decode step")


@pytest.mark.parametrize("component", ["expert_gate_proj", "delta_query"])
def test_an_interior_without_a_decode_address_refuses_by_name(nnterp_qwen, component):
    """No generated-frame address is verified for the grouped experts kernel
    or the chunked kernel's arguments, and decode dispatches different code
    than prefill — refused by name, not served from the prefill table."""
    doc = _gen_doc(component, DELTANET_LAYER)
    with pytest.raises(ProtocolError, match="generated-frame address"):
        sweep.make_executor(
            NnterpExecutor, doc, nnterp_qwen, rows=ROWS, with_cf=False
        ).run_all()


# --------------------------------------------------------------------------- #
# refusals stay shared: the axes rule and rule 16, the same words on both engines
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component", ["attention_key", "attention_scores", "attention_probs"]
)
def test_unstackable_generated_reads_refuse_identically(
    hooks_qwen, nnterp_qwen, component
):
    """A key-indexed tensor grows with the cache, so its steps do not stack —
    a fact about the declared axes, shared by both engines word for word."""
    doc = _gen_doc(component, QWEN_ATTENTION_LAYER)
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        NnterpExecutor, doc, nnterp_qwen
    )


def test_a_write_in_the_continuation_is_rule_16_before_any_engine():
    """Writes are prefill-only (§2.3): a write addressing the continuation
    frame is refused by validation, so no engine ever sees a decode-step
    write."""
    doc = _gen_doc("block_output", 1)
    doc["method"]["reads"]["r"]["model"] = "patched"
    doc["method"]["save"][0]["model"] = "patched"
    doc["method"]["writes"] = {
        "late": {"site": "tap", "pos": _window(index=0), "do": {"swap": 0.0}}
    }
    doc["method"]["intervened_models"] = {
        "patched": {"input": "base", "writes": ["late"]}
    }
    with pytest.raises(ValidationError, match="prefill-only") as excinfo:
        validate_document(parse_document(in_order(doc)), engine_is_local=True)
    assert excinfo.value.rule == 16


def test_routing_sends_generate_documents_here():
    doc = parse_document(in_order(_gen_doc("block_output", 1)))
    assert "generate" in NnterpEngine().capabilities
    assert "generate" in requires(doc)
    assert requires(doc) <= NnterpEngine().effective_capabilities
