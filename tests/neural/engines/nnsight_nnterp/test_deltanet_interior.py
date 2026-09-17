"""The Gated DeltaNet interior on the nnsight + nnterp engine.

30 of the 40 target layers carry this mixer, and nothing at its kernel
boundary is a module boundary. The kernel-boundary components carry one
name on both engines and are held to parity in the sweep; what this suite
pins is what parity cannot:

* **the conv split**: ``deltanet_query`` / ``key`` / ``delta_value`` are
  exactly the three column blocks of ``delta_conv`` — same tensors, two
  addresses — and ``delta_query`` is ``deltanet_query`` tiled to the
  value-head count (the kernel's argument vs the projection before it);
* **the projection chain**: the mixer's own output is
  ``out_proj(delta_premix)``, invoked through the envoy;
* **the state**: one fire per 64-token chunk (the kernel's own loop count,
  never the config), and the recurrence's causal signature — zeroing the
  state after chunk 0 leaves chunk 0's tokens bit-identical and moves later
  ones;
* **fire-axis discipline**: the state's position axis is the chunk index,
  so text anchors refuse, out-of-range fires refuse, and a write past the
  last fire refuses rather than silently never running;
* **the typed backend pairs**: the three tensors the two engines reach in
  different shapes or at different times agree after the registry's
  declared transform, and the captures are distinct tensors.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_nnterp.engine import NnterpEngine
from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.engine import choose_engine
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import parse_document

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnsight_nnterp.conftest import (
    FORMULATION_ATOL,
    ROWS,
    Family,
    assert_same,
    read_doc,
    single_row,
)
from tests.protocol._docs import in_order

pytestmark = pytest.mark.smoke

#: the anti-vacuity band — a write has to move a value by more than float
#: noise to count as having landed; agreement itself is exact (conftest)
ATOL = sweep.ATOL
LAYER = 0  # DeltaNet on the fixture (layers 0-2; layer 3 is full attention)

#: > 64 tokens, so the kernel runs more than one chunk
LONG_TEXT = "the quick brown fox jumps over the lazy dog and runs far away " * 10
SHORT_TEXT = "the quick brown fox jumps"
LONG_ROWS = [
    {
        "input": LONG_TEXT,
        "counterfactual_inputs": [
            "a slow green turtle sleeps beneath the old stone bridge at noon " * 10
        ],
    }
]


def _read_doc(component: str, *, pos: object = -1, extra: dict | None = None) -> dict:
    return read_doc(component, LAYER, pos=pos, extra=extra)


@pytest.fixture
def qwen(hooks_qwen, nnterp_qwen) -> Family:
    return Family("qwen", hooks_qwen, nnterp_qwen, ROWS)


def _read(qwen: Family, component: str, **kw) -> torch.Tensor:
    return qwen.traced(_read_doc(component, **kw), with_cf=False).read_value("r")


# --------------------------------------------------------------------------- #
# the once-fired components: widths and identities
# --------------------------------------------------------------------------- #


def test_every_deltanet_component_reads_with_its_declared_width(qwen):
    """📐 fixture dims: 4 key heads x 32, 8 value heads x 32 — so the fused
    projection is 512 wide, q/k live in the narrower key-head space and the
    kernel's tiled arguments in the value-head space."""
    info = qwen.nnterp.info
    key_dim = info.linear_num_key_heads * info.linear_key_head_dim
    value_dim = info.linear_num_value_heads * info.linear_value_head_dim
    tiled = info.linear_num_value_heads * info.linear_key_head_dim
    expected = {
        "delta_qkv": 2 * key_dim + value_dim,
        "delta_conv": 2 * key_dim + value_dim,
        "deltanet_query": key_dim,
        "deltanet_key": key_dim,
        "delta_query": tiled,
        "delta_key": tiled,
        "delta_value": value_dim,
        "delta_beta": info.linear_num_value_heads,
        "delta_decay": info.linear_num_value_heads,
        "delta_gate": value_dim,
        "delta_kernel_output": value_dim,
        "delta_premix": value_dim,
    }
    for component, width in expected.items():
        value = _read(qwen, component)
        assert tuple(value.shape) == (2, 1, width), component


def test_query_key_value_are_the_conv_splits_exactly(qwen):
    """The identity that pins the split order [key | key | value] and that the
    q/k/v reshapes really are the conv's columns — same tensors, two
    addresses, difference 0.0."""
    info = qwen.nnterp.info
    key_dim = info.linear_num_key_heads * info.linear_key_head_dim
    doc = _read_doc(
        "delta_conv",
        extra={
            "q": ({"component": "deltanet_query", "layers": [LAYER]}, -1),
            "k": ({"component": "deltanet_key", "layers": [LAYER]}, -1),
            "v": ({"component": "delta_value", "layers": [LAYER]}, -1),
        },
    )
    executor = qwen.traced(doc, with_cf=False)
    conv = executor.read_value("r")
    torch.testing.assert_close(
        conv[..., :key_dim], executor.read_value("q"), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        conv[..., key_dim : 2 * key_dim], executor.read_value("k"), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        conv[..., 2 * key_dim :], executor.read_value("v"), atol=0.0, rtol=0.0
    )


def test_the_kernel_argument_is_the_projection_tiled(qwen):
    """``delta_query`` (the kernel's argument, value-head space) is
    ``deltanet_query`` (the projection, key-head space) repeated over the
    head axis — one trace, one request on the kernel's inputs, exact."""
    info = qwen.nnterp.info
    doc = _read_doc(
        "delta_query",
        extra={"pre": ({"component": "deltanet_query", "layers": [LAYER]}, -1)},
    )
    executor = qwen.traced(doc, with_cf=False)
    tiled = executor.read_value("r")
    pre = executor.read_value("pre")
    h_k, h_v, d_k = (
        info.linear_num_key_heads,
        info.linear_num_value_heads,
        info.linear_key_head_dim,
    )
    expected = (
        pre.reshape(*pre.shape[:-1], h_k, d_k)
        .repeat_interleave(h_v // h_k, dim=-2)
        .reshape(*pre.shape[:-1], h_v * d_k)
    )
    torch.testing.assert_close(tiled, expected, atol=0.0, rtol=0.0)


def test_beta_is_a_sigmoid(nnterp_qwen):
    beta = single_row(
        nnterp_qwen, _read_doc("delta_beta", pos="all"), SHORT_TEXT
    ).read_value("r")
    assert float(beta.min()) > 0.0 and float(beta.max()) < 1.0


def test_the_mixer_output_is_the_projection_of_the_premix(qwen):
    """``attention_output`` (the module boundary, parity-proven) equals
    ``out_proj(delta_premix)`` — the envoy invokes its underlying module."""
    doc = _read_doc(
        "delta_premix",
        extra={"out": ({"component": "attention_output", "layers": [LAYER]}, -1)},
    )
    executor = qwen.traced(doc, with_cf=False)
    projected = qwen.nnterp.blocks[LAYER].linear_attn.out_proj(executor.read_value("r"))
    torch.testing.assert_close(
        projected, executor.read_value("out"), atol=0.0, rtol=0.0
    )


# --------------------------------------------------------------------------- #
# the state: per-chunk fires
# --------------------------------------------------------------------------- #


def _state_doc(pos: object) -> dict:
    return _read_doc("deltanet_state", pos=pos)


def test_the_state_fires_once_per_chunk(nnterp_qwen):
    """📐 the kernel pads to a 64 multiple and loops — the count is its own
    range op's, and on this prompt that is more than one chunk."""
    value = single_row(nnterp_qwen, _state_doc("all"), LONG_TEXT).read_value("r")
    info = nnterp_qwen.info
    tokens = len(nnterp_qwen.tokenizer(LONG_TEXT)["input_ids"])
    n_chunks = -(-tokens // 64)  # ceil
    assert n_chunks >= 2
    width = (
        info.linear_num_value_heads
        * info.linear_key_head_dim
        * info.linear_value_head_dim
    )
    assert tuple(value.shape) == (1, n_chunks, width)


def test_the_last_chunk_state_is_index_minus_one(nnterp_qwen):
    whole = single_row(nnterp_qwen, _state_doc("all"), LONG_TEXT).read_value("r")
    last = single_row(nnterp_qwen, _state_doc(-1), LONG_TEXT).read_value("r")
    torch.testing.assert_close(whole[:, -1:, :], last, atol=0.0, rtol=0.0)


def _state_write_doc(pos: object, do: dict, *, read: str = "attention_output") -> dict:
    doc = sweep.read_doc(read, LAYER, pos="all")
    doc["method"]["sites"]["state"] = {"component": "deltanet_state", "layers": [LAYER]}
    doc["method"]["reads"]["r"]["model"] = "patched"
    doc["method"]["save"][0]["model"] = "patched"
    doc["method"]["writes"] = {"zero": {"site": "state", "pos": pos, "do": do}}
    doc["method"]["intervened_models"] = {
        "patched": {"input": "base", "writes": ["zero"]}
    }
    return doc


ZERO = {"clamp": {"lo": 0.0, "hi": 0.0}}


def test_zeroing_the_state_after_chunk_0_moves_only_later_tokens(nnterp_qwen):
    """The recurrence's causal signature, as a document: a clamp-to-zero
    write at chunk 0 leaves chunk 0's own tokens bit-identical (the state is
    applied *after* the chunk that produced it) and moves later tokens; the
    write fired exactly once, at its fire."""
    clean = single_row(
        nnterp_qwen, sweep.read_doc("attention_output", LAYER, pos="all"), LONG_TEXT
    ).read_value("r")
    executor = single_row(nnterp_qwen, _state_write_doc(0, ZERO), LONG_TEXT)
    patched = executor.read_value("r")
    changed = (patched != clean).any(-1)[0]
    assert int(changed[:64].sum()) == 0, "chunk 0's own tokens must be untouched"
    assert int(changed[64:].sum()) > 0, "later tokens must move"
    assert executor.fires == {("patched", "base"): {"zero": 1}}


def test_a_write_past_the_last_fire_is_refused_not_skipped(nnterp_qwen):
    """📐 A tracer.iter body bound past the last fire never runs (nnsight
    keeps the fires it reached and drops the rest of the block), so the
    executor must turn the miss into a refusal rather than return values
    from a write that never landed."""
    executor = single_row(nnterp_qwen, _state_write_doc(99, ZERO), LONG_TEXT)
    with pytest.raises(ProtocolError, match="fire"):
        executor.read_value("r")


def test_an_anchored_fire_write_is_refused(nnterp_qwen):
    """A chunk axis has no text: a span (or a variable, a column, an anchor)
    has nothing to resolve against there."""
    doc = _state_write_doc({"span": [0, 2]}, ZERO)
    executor = single_row(nnterp_qwen, doc, LONG_TEXT)
    with pytest.raises(ProtocolError, match="fire index") as excinfo:
        executor.read_value("r")
    assert "no text" in str(excinfo.value)


@pytest.mark.parametrize("back", [1, 2])
def test_a_negative_fire_write_counts_from_the_last_fire(nnterp_qwen, back):
    """``-k`` on a write is ``count - k``, as on a read: the count is in
    hand when the write lands (the kernel's own loop range), so the two
    documents land on the same fire and read back identical. The last
    state feeds no token (it is applied after the chunk that produced it),
    so anti-vacuity is checked one chunk earlier; an index past the first
    fire refuses like one past the last."""
    tokens = len(nnterp_qwen.tokenizer(LONG_TEXT)["input_ids"])
    count = -(-tokens // 64)  # ceil
    assert count >= 3
    from_end = single_row(nnterp_qwen, _state_write_doc(-back, ZERO), LONG_TEXT)
    by_index = single_row(nnterp_qwen, _state_write_doc(count - back, ZERO), LONG_TEXT)
    torch.testing.assert_close(
        from_end.read_value("r"), by_index.read_value("r"), atol=0.0, rtol=0.0
    )
    assert from_end.fires == by_index.fires == {("patched", "base"): {"zero": 1}}
    clean = single_row(
        nnterp_qwen, sweep.read_doc("attention_output", LAYER, pos="all"), LONG_TEXT
    ).read_value("r")
    assert torch.equal(from_end.read_value("r"), clean) == (back == 1)
    too_far = single_row(nnterp_qwen, _state_write_doc(-count - 1, ZERO), LONG_TEXT)
    with pytest.raises(ProtocolError, match="fired"):
        too_far.read_value("r")


def test_an_anchored_position_on_the_state_is_refused(nnterp_qwen):
    executor = single_row(nnterp_qwen, _state_doc({"span": [0, 2]}), LONG_TEXT)
    with pytest.raises(ProtocolError, match="chunk index"):
        executor.read_value("r")


def test_an_out_of_range_chunk_read_is_refused(nnterp_qwen):
    executor = single_row(nnterp_qwen, _state_doc(99), LONG_TEXT)
    with pytest.raises(ProtocolError, match="fired"):
        executor.read_value("r")


# --------------------------------------------------------------------------- #
# writes on the once-fired components
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component",
    [
        "delta_conv",
        "delta_value",
        "delta_beta",
        "delta_decay",
        "delta_query",
        "delta_kernel_output",
    ],
)
def test_a_swap_moves_the_logits_and_a_self_swap_does_not(qwen, component):
    clean = qwen.unpatched_logits()
    doc = sweep.interchange_doc(component, LAYER)
    moved = qwen.traced(doc, with_cf=True)
    assert float((moved.dense_value("logits") - clean).abs().max()) > 1e-5, component

    doc["method"]["reads"]["v_cf"]["input"] = "base"
    same = qwen.traced(doc, with_cf=True)
    assert float((same.dense_value("logits") - clean).abs().max()) == 0.0, component


def test_two_same_shaped_interior_writes_land_as_two(qwen):
    """``deltanet_query`` and ``deltanet_key`` share a module, a kind and a
    shape — only their op differs — so a schedule keyed on the site alone
    would merge them into one landing (the read-side twin is the conv-split
    identity above). Both fire, and the pair's effect is neither single
    write's."""
    doc = sweep.interchange_doc("deltanet_query", LAYER)
    doc["method"]["sites"]["k"] = {"component": "deltanet_key", "layers": LAYER}
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
    executor = qwen.traced(doc, with_cf=True)
    logits = executor.dense_value("logits")
    assert executor.fires == {("patched", "base"): {"patch": 1, "patch_k": 1}}
    for component in ("deltanet_query", "deltanet_key"):
        single = qwen.traced(sweep.interchange_doc(component, LAYER), with_cf=True)
        assert not torch.allclose(single.dense_value("logits"), logits, atol=ATOL), (
            component
        )


def test_the_three_kernel_arguments_written_together_agree(qwen):
    """``delta_query``, ``delta_key`` and ``delta_decay`` are three
    selections of one op's ``inputs`` handle: each write rebuilds the
    container around its own argument and the next reads the rebuilt one,
    so all three land — the patched logits agree with the reference engine's
    three separate hooks, and no single write's effect is the trio's."""
    doc = sweep.interchange_doc("delta_query", LAYER)
    for name, component in (("k", "delta_key"), ("g", "delta_decay")):
        doc["method"]["sites"][name] = {"component": component, "layers": LAYER}
        doc["method"]["reads"][f"{name}_cf"] = {
            "site": name,
            "pos": -1,
            "model": "original",
            "input": "counterfactual",
        }
        doc["method"]["writes"][f"patch_{name}"] = {
            "site": name,
            "pos": -1,
            "do": {"swap": f"{name}_cf"},
        }
    doc["method"]["intervened_models"]["patched"]["writes"] = [
        "patch",
        "patch_k",
        "patch_g",
    ]
    hooked = qwen.hooked(doc, with_cf=True).dense_value("logits")
    executor = qwen.traced(doc, with_cf=True)
    traced = executor.dense_value("logits")
    assert_same(hooked, traced, "logits after q, k and g swapped together")
    assert executor.fires == {
        ("patched", "base"): {"patch": 1, "patch_k": 1, "patch_g": 1}
    }
    for component in ("delta_query", "delta_key", "delta_decay"):
        single = qwen.traced(sweep.interchange_doc(component, LAYER), with_cf=True)
        assert not torch.allclose(single.dense_value("logits"), traced, atol=ATOL), (
            component
        )


# --------------------------------------------------------------------------- #
# the typed backend pairs: two names, one tensor after the declared transform
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "hooks_component,trace_component,relation", sweep.DELTA_FAMILY_PAIRS
)
def test_delta_family_cross_engine_agreement(
    hooks_qwen, nnterp_qwen, hooks_component, trace_component, relation
):
    """The reference engine's post-tiling q/k and per-step state against
    this engine's pre-tiling q/k and per-chunk state, agreed after the
    registry's declared transform (``registry.BACKEND_PAIRS``). The relation
    is read from the registry by the alignment helper, not passed in: a
    wrong address cannot be massaged into agreement here."""
    hooked = sweep.make_executor(
        PointExecutor,
        sweep.read_doc(hooks_component, LAYER, pos="all"),
        hooks_qwen,
        rows=LONG_ROWS,
        with_cf=False,
    ).read_value("r")
    traced = sweep.make_executor(
        NnterpExecutor,
        sweep.read_doc(trace_component, LAYER, pos="all"),
        nnterp_qwen,
        rows=LONG_ROWS,
        with_cf=False,
    ).read_value("r")
    assert sweep.backend_pair(hooks_component).relation == relation
    left, right = sweep.align_delta_pair(
        hooked, traced, hooks_component, hooks_qwen.info
    )
    assert_same(
        left,
        right,
        f"{hooks_component!r} (pytorch_hooks) vs {trace_component!r} "
        f"(nnsight_nnterp), related by {relation!r}",
        # q and k are one tensor tiled, so exact; the state is two kernels'
        # arithmetic — the reference engine steps the recurrent formulation,
        # this engine reads the chunked one's running state
        atol=FORMULATION_ATOL if relation == "chunk_boundary" else 0.0,
    )


def test_the_delta_family_tensors_are_not_all_the_same_tensor(nnterp_qwen):
    """Anti-vacuity for the DeltaNet interior: the captures must be as many
    different tensors, or 'they agree' would be satisfiable by a tap that
    returns the same thing for every component."""
    seen: list[tuple[str, torch.Tensor]] = []
    for component in (
        sweep.SHARED_LINEAR_ONLY + sweep.NNSIGHT_ONLY + ("delta_query", "delta_key")
    ):
        if component == "expert_permutation":
            continue
        value = sweep.make_executor(
            NnterpExecutor,
            sweep.read_doc(component, LAYER, pos="all"),
            nnterp_qwen,
            rows=LONG_ROWS,
            with_cf=False,
        ).read_value("r")
        for name, other in seen:
            if other.shape == value.shape:
                assert not torch.allclose(other, value, atol=ATOL), (
                    f"{component!r} and {name!r} captured the same tensor"
                )
        seen.append((component, value))


# --------------------------------------------------------------------------- #
# ownership and streams
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component", ["deltanet_query", "deltanet_key", "deltanet_state"]
)
def test_the_reference_engine_refuses_the_fused_faces_by_name(hooks_qwen, component):
    doc = _read_doc(component, pos="all" if component == "deltanet_state" else -1)
    with pytest.raises(ProtocolError, match="nnsight engine"):
        sweep.make_executor(
            PointExecutor, doc, hooks_qwen, rows=ROWS, with_cf=False
        ).run_all()


def test_routing_lands_deltanet_documents_here():
    doc = parse_document(in_order(_read_doc("deltanet_state", pos="all")))
    chosen = choose_engine(doc, [PytorchHooksEngine(), NnterpEngine()])
    assert isinstance(chosen, NnterpEngine)


def test_deltanet_at_a_full_attention_layer_refuses_architecturally(qwen):
    doc = sweep.read_doc(
        "deltanet_state", 3, pos="all"
    )  # full attention on this fixture
    with pytest.raises(ProtocolError, match="Gated DeltaNet mixer"):
        qwen.traced(doc, with_cf=False).run_all()
