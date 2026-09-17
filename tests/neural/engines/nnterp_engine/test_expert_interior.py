"""The per-expert MoE interior on the nnterp engine.

The five slot components carry one name on both engines and are held to
parity in the sweep — including, now, the ragged ``expert:`` face, which
this engine serves from the routing table it captures at the experts
anchor. What this suite pins is what parity cannot:

* **two implementations of the same math**: the grouped_mm kernel this
  engine serves from, against transformers' own eager per-expert loop,
  reconstructed through ``tracer.iter``;
* **identities**: the slot-sum of ``expert_output · router_scores`` is
  ``routed_output`` exactly; the activation is ``act(gate)`` exactly; the
  permutation is the inverse of the sort the kernel ran — ``value[perm] ==
  arange`` and not ``arange`` itself, which is what the kernel's own
  ``inv_perm`` assignment reports as its output;
* **causal writes**: a swap moves the logits, a same-value swap moves
  nothing, a written slot reads back written, an ``expert:`` write lands on
  that expert's slots alone;
* **ownership**: ``expert_permutation`` is kernel bookkeeping — read-only,
  this engine's alone, and routing knows it.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnterp_engine.engine import NnterpEngine
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.shared.kernels import torch_kernel_path
from causalab.neural.shared.loading import torch_module
from causalab.protocol.engine import choose_engine
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import parse_document

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnterp_engine.conftest import (
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
LAYER = 0  # every fixture layer carries the sparse-MoE block
TEXT = "the quick brown fox jumps"

SLOT_COMPONENTS = (
    "expert_gate_proj",
    "expert_up_proj",
    "expert_activation",
    "expert_neuron_output",
    "expert_output",
)


def _read_doc(component: str, *, pos: object = -1, extra: dict | None = None) -> dict:
    return read_doc(
        component,
        LAYER,
        pos=pos,
        extra={name: (site, pos) for name, site in (extra or {}).items()},
    )


@pytest.fixture
def qwen(hooks_qwen, nnterp_qwen) -> Family:
    return Family("qwen", hooks_qwen, nnterp_qwen, ROWS)


def _read(qwen: Family, component: str, **kw) -> torch.Tensor:
    return qwen.traced(_read_doc(component, **kw), with_cf=False).read_value("r")


# --------------------------------------------------------------------------- #
# reads: shapes and identities
# --------------------------------------------------------------------------- #


def test_the_interior_reads_with_the_declared_widths(qwen):
    info = qwen.nnterp.info
    k = info.num_experts_per_tok
    for component, width in (
        ("expert_gate_proj", k * info.moe_intermediate_size),
        ("expert_up_proj", k * info.moe_intermediate_size),
        ("expert_activation", k * info.moe_intermediate_size),
        ("expert_neuron_output", k * info.moe_intermediate_size),
        ("expert_output", k * info.hidden_size),
        ("expert_permutation", k),
    ):
        value = _read(qwen, component)
        assert tuple(value.shape) == (2, 1, width), component
        if component == "expert_permutation":
            assert not value.dtype.is_floating_point


def test_the_activation_is_act_of_gate_exactly(qwen):
    """The identity that says the taps share one fused capture and its
    gate: ``expert_activation`` is ``act_fn(gate)`` alone (the registry's
    semantics, the llama ``mlp_activation`` precedent) — same rows, same
    order, before the ``· up`` multiply; and the neuron output is that
    times the up half."""
    doc = _read_doc(
        "expert_gate_proj",
        extra={
            "up": {"component": "expert_up_proj", "layers": [LAYER]},
            "act": {"component": "expert_activation", "layers": [LAYER]},
            "neuron": {"component": "expert_neuron_output", "layers": [LAYER]},
        },
    )
    executor = qwen.traced(doc, with_cf=False)
    gate, up = executor.read_value("r"), executor.read_value("up")
    act, neuron = executor.read_value("act"), executor.read_value("neuron")
    torch.testing.assert_close(torch.nn.functional.silu(gate), act, atol=0.0, rtol=0.0)
    torch.testing.assert_close(act * up, neuron, atol=0.0, rtol=0.0)


def test_expert_output_weighted_sums_to_routed_output(qwen):
    """The registry identity, on this engine: ``routed_output == Σ_slot
    expert_output · router_scores`` — ``expert_output`` is the
    down-projection output BEFORE the routing weight, so the scores re-enter
    here — pinned against the module-boundary tap."""
    info = qwen.nnterp.info
    doc = _read_doc(
        "expert_output",
        extra={
            "routed": {"component": "routed_output", "layers": [LAYER]},
            "scores": {"component": "router_scores", "layers": [LAYER]},
        },
    )
    executor = qwen.traced(doc, with_cf=False)
    per_slot = executor.read_value("r")
    weighted = per_slot.reshape(
        *per_slot.shape[:-1], info.num_experts_per_tok, info.hidden_size
    ) * executor.read_value("scores").unsqueeze(-1)
    torch.testing.assert_close(
        weighted.sum(-2), executor.read_value("routed"), atol=0.0, rtol=0.0
    )


def _kernel_sort_permutation(bundle, text: str) -> torch.Tensor:
    """The sort the grouped kernel ran on this text — ``torch_sort_0``'s
    own output, off a bare trace."""
    batch = dict(bundle.tokenizer([text], return_tensors="pt"))
    experts = bundle.blocks[LAYER].mlp.experts
    _ = experts.source
    with torch.no_grad(), torch_kernel_path(torch_module(bundle.model)):
        with bundle.model.trace(batch):
            perm = experts.source.experts_forward_1.source.torch_sort_0.output[1].save()
    return perm


def test_the_permutation_is_the_inverse_of_the_kernel_sort(nnterp_qwen):
    """Read whole: each (token, slot) row's index in the kernel's sorted
    order, so indexing it by the sort's own permutation is the identity —
    and it is *not* the identity itself. 📐 The kernel's ``inv_perm_1``
    assignment reports its right-hand side (the ``arange``) as its output,
    which the second assertion is the tripwire for."""
    value = single_row(
        nnterp_qwen, _read_doc("expert_permutation", pos="all"), TEXT
    ).read_value("r")
    flat = value.reshape(-1)
    perm = _kernel_sort_permutation(nnterp_qwen, TEXT)
    arange = torch.arange(flat.numel(), dtype=flat.dtype)
    assert torch.equal(flat[perm], arange)
    assert not torch.equal(flat, arange)


def test_grouped_and_eager_implementations_agree(nnterp_qwen):
    """The §8 cross-check: transformers' own eager per-expert loop — a wholly
    independent implementation, one Python iteration per hit expert — rebuilt
    into the same (token, slot) frame through ``tracer.iter``, against the
    grouped_mm tensor this engine serves."""
    import nnsight

    info = nnterp_qwen.info
    k, hidden = info.num_experts_per_tok, info.hidden_size
    served = single_row(
        nnterp_qwen, _read_doc("expert_output", pos="all"), TEXT
    ).read_value("r")
    rows = served.shape[0] * served.shape[1]
    served = served.reshape(rows, k, hidden)

    batch = dict(nnterp_qwen.tokenizer([TEXT], return_tensors="pt"))
    experts = nnterp_qwen.blocks[LAYER].mlp.experts
    model = nnterp_qwen.model
    model.set_experts_implementation("eager")
    try:
        with torch.no_grad(), torch_kernel_path(torch_module(model)):
            with model.trace(batch) as tracer:
                loop = experts.source.experts_forward_1.source
                n_hit = len(loop.nonzero_0.output)
                per_expert = nnsight.save([])
                for _ in tracer.iter[:n_hit]:
                    top_k_pos, token_idx = loop.torch_where_0.output
                    # `_1` is the down-projection's output, BEFORE the routing
                    # weight — the semantics `expert_output` names
                    per_expert.append(
                        (top_k_pos, token_idx, loop.current_hidden_states_1.output)
                    )
    finally:
        model.set_experts_implementation("grouped_mm")

    rebuilt = torch.zeros(rows, k, hidden)
    for top_k_pos, token_idx, unweighted in per_expert:
        rebuilt[token_idx, top_k_pos] = unweighted.to(rebuilt.dtype)
    torch.testing.assert_close(rebuilt, served, atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", SLOT_COMPONENTS)
def test_a_swap_moves_the_logits_and_a_self_swap_does_not(qwen, component):
    clean = qwen.unpatched_logits()
    doc = sweep.interchange_doc(component, LAYER)
    moved = float(
        (qwen.traced(doc, with_cf=True).dense_value("logits") - clean).abs().max()
    )
    assert moved > 1e-5, f"{component}: the swap landed nowhere"

    doc["method"]["reads"]["v_cf"]["input"] = "base"
    unmoved = float(
        (qwen.traced(doc, with_cf=True).dense_value("logits") - clean).abs().max()
    )
    assert unmoved == 0.0, f"{component}: a same-value swap must be the identity"


def test_a_written_slot_reads_back_written(qwen):
    doc = sweep.interchange_doc("expert_gate_proj", LAYER)
    doc["method"]["reads"]["obs"] = {
        "site": "tap",
        "pos": -1,
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
    executor = qwen.traced(doc, with_cf=True)
    assert (
        float((executor.read_value("obs") - executor.read_value("v_cf")).abs().max())
        == 0.0
    )


def test_the_permutation_refuses_writes_as_kernel_bookkeeping(qwen):
    with pytest.raises(ProtocolError, match="row bookkeeping"):
        qwen.traced(
            sweep.interchange_doc("expert_permutation", LAYER), with_cf=True
        ).run_all()


# --------------------------------------------------------------------------- #
# the ragged `expert:` face — parity with the reference engine's dispatch wrapper
# --------------------------------------------------------------------------- #


def _experts_hit(family: Family, pos: object) -> tuple[int, int]:
    """(an expert the router chose at ``pos``, one it never chose)."""
    idx = family.hooked(
        sweep.read_doc("expert_idx", LAYER, pos=pos), with_cf=False
    ).read_value("r")
    chosen = set(
        int(i) for i in (idx.flat if hasattr(idx, "flat") else idx).reshape(-1).tolist()
    )
    unchosen = next(e for e in range(family.nnterp.info.num_experts) if e not in chosen)
    return min(chosen), unchosen


@pytest.mark.parametrize("component", SLOT_COMPONENTS)
@pytest.mark.parametrize("pos", [-1, "all"])
def test_the_expert_face_reads_in_parity(qwen, component, pos):
    """The (position, slot) pairs the router sent to one expert, as flat rows
    with per-example widths — ragged at ``pos: all`` (the two rows differ in
    length and in how often the expert was chosen)."""
    expert, _ = _experts_hit(qwen, pos)
    doc = sweep.read_doc(component, LAYER, pos=pos)
    doc["method"]["sites"]["tap"]["expert"] = expert
    hooked = qwen.hooked(doc, with_cf=False).read_value("r")
    traced = qwen.traced(doc, with_cf=False).read_value("r")
    assert hooked.widths == traced.widths
    assert sum(traced.widths) > 0
    assert_same(hooked.flat, traced.flat, f"expert {expert} face of {component!r}")


def test_an_expert_nobody_chose_is_an_empty_selector_cell(qwen):
    _, unchosen = _experts_hit(qwen, -1)
    doc = sweep.read_doc("expert_activation", LAYER)
    doc["method"]["sites"]["tap"]["expert"] = unchosen
    executor = qwen.traced(doc, with_cf=False)
    value = executor.read_value("r")
    assert value.widths == (0, 0) and value.flat.numel() == 0
    assert executor.resolution("r").reason == "empty_selector"


def test_an_expert_face_write_lands_on_that_experts_slots_alone(qwen):
    """A swap addressed to one expert replaces only the slots the router sent
    it (the operand is the token-major form, read at a second site); the
    patched logits agree with the reference engine's masked landing and
    differ from an unmasked swap's."""
    expert, _ = _experts_hit(qwen, -1)
    doc = sweep.interchange_doc("expert_neuron_output", LAYER)
    doc["method"]["sites"]["full"] = {
        "component": "expert_neuron_output",
        "layers": LAYER,
    }
    doc["method"]["reads"]["v_cf"]["site"] = "full"
    doc["method"]["sites"]["tap"]["expert"] = expert
    hooked = qwen.hooked(doc, with_cf=True).dense_value("logits")
    traced = qwen.traced(doc, with_cf=True).dense_value("logits")
    assert_same(hooked, traced, f"logits after a swap at expert {expert}")
    assert not torch.allclose(traced, qwen.unpatched_logits(), atol=ATOL)
    whole = qwen.traced(
        sweep.interchange_doc("expert_neuron_output", LAYER), with_cf=True
    )
    assert not torch.allclose(traced, whole.dense_value("logits"), atol=ATOL)


@pytest.mark.parametrize("policy", ["exact_length_buckets", "padded_masked"])
def test_a_ragged_expert_face_write_agrees_under_each_policy(qwen, policy):
    """The same swap over every position of both rows — ragged, the rows
    differ in length — under a declared landing policy: the shared ragged
    landing runs on this engine's contract with the routing table it
    captured, and the masked result agrees with the reference engine's."""
    expert, _ = _experts_hit(qwen, "all")
    doc = sweep.interchange_doc("expert_neuron_output", LAYER, pos="all")
    doc["method"]["sites"]["full"] = {
        "component": "expert_neuron_output",
        "layers": LAYER,
    }
    doc["method"]["reads"]["v_cf"]["site"] = "full"
    doc["method"]["sites"]["tap"]["expert"] = expert
    doc["method"]["writes"]["patch"]["ragged"] = {"policy": policy}
    hooked = qwen.hooked(doc, with_cf=True)
    traced = qwen.traced(doc, with_cf=True)
    assert_same(
        hooked.dense_value("logits"),
        traced.dense_value("logits"),
        f"logits after a ragged swap at expert {expert} under {policy!r}",
    )
    assert hooked.ragged_geometry == traced.ragged_geometry
    assert len(set(traced.ragged_geometry[("patched", "patch")]["widths"])) == 2
    assert not torch.allclose(
        traced.dense_value("logits"), qwen.unpatched_logits(), atol=ATOL
    )


# --------------------------------------------------------------------------- #
# ownership: the permutation is this engine's alone, and routing knows it
# --------------------------------------------------------------------------- #


def test_the_reference_engine_refuses_the_permutation_by_name(hooks_qwen):
    with pytest.raises(ProtocolError, match="nnterp engine"):
        sweep.make_executor(
            PointExecutor,
            _read_doc("expert_permutation"),
            hooks_qwen,
            rows=ROWS,
            with_cf=False,
        ).run_all()


def test_routing_chooses_this_engine_even_listed_second():
    doc = parse_document(in_order(_read_doc("expert_permutation")))
    assert isinstance(
        choose_engine(doc, [PytorchHooksEngine(), NnterpEngine()]), NnterpEngine
    )


def test_the_generated_refusal_names_the_missing_component():
    """With the reference engine alone in the list, routing refuses at load
    and the generated capability entry names the component nobody serves."""
    from causalab.protocol.errors import ValidationError

    doc = parse_document(in_order(_read_doc("expert_permutation")))
    with pytest.raises(ValidationError, match="component:expert_permutation"):
        choose_engine(doc, [PytorchHooksEngine()])
