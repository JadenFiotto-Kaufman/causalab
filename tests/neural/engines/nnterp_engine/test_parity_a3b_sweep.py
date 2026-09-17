"""The engine-agreement sweep over the Qwen3.6-35B-A3B hookpoint surface, on
the tiny fixture.

``tests/_helpers/a3b_sweep.py`` partitions the vocabulary by which engines
the capability rows name, and the engines' ``components`` are those rows.
Every shared component must agree with the reference engine on the read and
on the intervention's downstream effect — the two tiled kernel arguments
(``delta_query`` / ``delta_key``) included, which this engine reads off the
same kernel call; the fused-forward interiors this engine alone serves read
to their declared widths (their identities are pinned in the interior
suites); and the three per-token DeltaNet faces the chunked prefill kernel
never materializes are the reference engine's alone — undeclared here, and
refused by name with the reason ``component_unavailable`` for a document
arriving unrouted. Anti-vacuity is not optional: every write case also
asserts the patched logits moved.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnterp_engine.engine import NnterpEngine
from causalab.neural.engines.nnterp_engine.executor import NnterpExecutor
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import COMPONENTS

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnterp_engine.conftest import (
    ROWS,
    assert_same,
)

pytestmark = pytest.mark.smoke

SHARED = (
    sweep.SHARED_LAYERLESS
    + sweep.SHARED_ANY_STREAM
    + sweep.SHARED_FULL_ONLY
    + sweep.SHARED_LINEAR_ONLY
)


def _hooks(doc, bundle, *, with_cf: bool):
    return sweep.make_executor(PointExecutor, doc, bundle, rows=ROWS, with_cf=with_cf)


def _trace(doc, bundle, *, with_cf: bool):
    return sweep.make_executor(NnterpExecutor, doc, bundle, rows=ROWS, with_cf=with_cf)


@pytest.fixture(scope="module")
def layers(hooks_qwen) -> tuple[int, int]:
    """(a Gated DeltaNet layer, a full-attention layer) of the fixture tower."""
    return sweep.stream_layers(hooks_qwen)


def _read_both(component, layer, hooks_bundle, trace_bundle):
    doc = sweep.read_doc(component, layer, pos=sweep.default_pos(component))
    hooked = _hooks(doc, hooks_bundle, with_cf=False).read_value("r")
    traced = _trace(doc, trace_bundle, with_cf=False).read_value("r")
    return hooked, traced


# --------------------------------------------------------------------------- #
# 1. reads — every shared component, in every block type it exists in
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", sweep.SHARED_LAYERLESS)
def test_read_parity_layerless(hooks_qwen, nnterp_qwen, component):
    hooked, traced = _read_both(component, None, hooks_qwen, nnterp_qwen)
    assert_same(hooked, traced, f"read {component!r}")


@pytest.mark.parametrize(
    "component", sweep.SHARED_ANY_STREAM + sweep.SHARED_LINEAR_ONLY
)
def test_read_parity_deltanet_layer(hooks_qwen, nnterp_qwen, layers, component):
    delta_layer, _ = layers
    hooked, traced = _read_both(component, delta_layer, hooks_qwen, nnterp_qwen)
    assert_same(hooked, traced, f"read {component!r} @ DeltaNet L{delta_layer}")


@pytest.mark.parametrize("component", sweep.SHARED_ANY_STREAM + sweep.SHARED_FULL_ONLY)
def test_read_parity_full_attention_layer(hooks_qwen, nnterp_qwen, layers, component):
    _, full_layer = layers
    hooked, traced = _read_both(component, full_layer, hooks_qwen, nnterp_qwen)
    assert_same(hooked, traced, f"read {component!r} @ full-attn L{full_layer}")


# --------------------------------------------------------------------------- #
# 2. writes — the intervention's downstream effect agrees, and it landed
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def unpatched_logits(hooks_qwen, nnterp_qwen):
    doc = sweep.read_doc("lm_head", None)
    return {
        "hooks": _hooks(doc, hooks_qwen, with_cf=False).read_value("r"),
        "trace": _trace(doc, nnterp_qwen, with_cf=False).read_value("r"),
    }


def _write_both(component, layer, hooks_bundle, trace_bundle, unpatched):
    doc = sweep.interchange_doc(component, layer, pos=sweep.default_pos(component))
    hooked = _hooks(doc, hooks_bundle, with_cf=True).dense_value("logits")
    traced = _trace(doc, trace_bundle, with_cf=True).dense_value("logits")
    where = f"{component!r}" + ("" if layer is None else f" @ L{layer}")
    assert_same(hooked, traced, f"patched logits after a swap at {where}")
    assert not torch.allclose(hooked, unpatched["hooks"], atol=sweep.ATOL), (
        f"pytorch_hooks: the interchange at {where} left the logits unchanged"
    )
    assert not torch.allclose(traced, unpatched["trace"], atol=sweep.ATOL), (
        f"nnterp: the interchange at {where} left the logits unchanged"
    )


@pytest.mark.parametrize("component", sweep.write_cases(sweep.SHARED_LAYERLESS))
def test_write_parity_layerless(hooks_qwen, nnterp_qwen, unpatched_logits, component):
    _write_both(component, None, hooks_qwen, nnterp_qwen, unpatched_logits)


@pytest.mark.parametrize(
    "component",
    sweep.write_cases(sweep.SHARED_ANY_STREAM + sweep.SHARED_LINEAR_ONLY),
)
def test_write_parity_deltanet_layer(
    hooks_qwen, nnterp_qwen, layers, unpatched_logits, component
):
    delta_layer, _ = layers
    _write_both(component, delta_layer, hooks_qwen, nnterp_qwen, unpatched_logits)


@pytest.mark.parametrize(
    "component",
    sweep.write_cases(sweep.SHARED_ANY_STREAM + sweep.SHARED_FULL_ONLY),
)
def test_write_parity_full_attention_layer(
    hooks_qwen, nnterp_qwen, layers, unpatched_logits, component
):
    _, full_layer = layers
    _write_both(component, full_layer, hooks_qwen, nnterp_qwen, unpatched_logits)


# --------------------------------------------------------------------------- #
# 3. the single-engine buckets: the fused interiors are served here, the
#    per-token faces by the reference engine — by name, on both sides
# --------------------------------------------------------------------------- #


def _layer_for(component: str, layers: tuple[int, int]) -> int | None:
    if sweep.layerless(component):
        return None
    delta_layer, full_layer = layers
    return (
        delta_layer
        if sweep.CAPABILITIES[component].stream == "linear_attention"
        else full_layer
    )


@pytest.mark.parametrize("component", sweep.NNTERP_ONLY)
def test_the_fused_interior_is_served(nnterp_qwen, layers, component):
    doc = sweep.read_doc(
        component, _layer_for(component, layers), pos=sweep.default_pos(component)
    )
    value = _trace(doc, nnterp_qwen, with_cf=False).read_value("r")
    assert value.shape[0] == len(ROWS) and value.numel() > 0


@pytest.mark.parametrize("component", sweep.NNTERP_ONLY)
def test_the_reference_engine_does_not_claim_the_fused_interior(component):
    engine = PytorchHooksEngine()
    assert component not in engine.components
    assert component not in engine.writable_components
    assert component in NnterpEngine().components


def test_the_hooks_only_bucket_is_exactly_the_per_token_deltanet_faces():
    assert set(sweep.HOOKS_ONLY) == {
        "delta_kv_mem",
        "delta_state_update",
        "delta_state",
    }


@pytest.mark.parametrize("component", sweep.HOOKS_ONLY)
def test_this_engine_does_not_claim_the_per_token_faces(component):
    engine = NnterpEngine()
    assert component not in engine.components
    assert component not in engine.writable_components
    assert component in PytorchHooksEngine().components


@pytest.mark.parametrize("component", sweep.HOOKS_ONLY)
def test_a_per_token_face_is_refused_by_name(nnterp_qwen, layers, component):
    """Routing keeps such a document away; this is the refusal one arriving
    unrouted meets, before any forward."""
    doc = sweep.read_doc(
        component, _layer_for(component, layers), pos=sweep.default_pos(component)
    )
    with pytest.raises(ProtocolError) as excinfo:
        _trace(doc, nnterp_qwen, with_cf=False).read_value("r")
    assert repr(component) in str(excinfo.value)
    assert "never materializes in prefill" in str(excinfo.value)
    assert excinfo.value.reason == "component_unavailable"


def test_the_declaration_is_the_boundaries_plus_the_address_table():
    """The rows this engine is named in are exactly what some family taps at
    a module boundary plus what the address table reaches — every kind the
    resolver knows accounted for, and every addressed component a kind the
    resolver marks as having no boundary. A row naming this engine without
    a landing, or a landing without its row, fails here."""
    from causalab.neural.engines.nnterp_engine.sources import components_addressed
    from causalab.protocol.registry import FAMILIES

    kinds = {
        c: {a.taps[c].kind for a in FAMILIES.values() if c in a.taps}
        for c in SHARED + sweep.HOOKS_ONLY + sweep.NNTERP_ONLY
    }
    for component, declared in kinds.items():
        boundary = bool(declared & {"in", "out"})
        addressed = component in components_addressed()
        assert (component in NnterpEngine.components) == (boundary or addressed), (
            component
        )
        if addressed and component != "attention_probs":
            assert not boundary, component


def test_mlp_activation_does_not_exist_on_this_architecture(nnterp_qwen, layers):
    """The one vocabulary entry the A3B has no tensor for: refused with the
    shared layer's text, which names the block's children."""
    delta_layer, _ = layers
    with pytest.raises(ProtocolError) as excinfo:
        _trace(
            sweep.read_doc("mlp_activation", delta_layer), nnterp_qwen, with_cf=False
        ).read_value("r")
    assert "mlp_activation" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 5. completeness — a new component cannot join the vocabulary unswept
# --------------------------------------------------------------------------- #


def test_every_component_is_claimed_by_exactly_one_bucket():
    """The guard that makes this file a *sweep* rather than a sample.

    Adding a component to ``schema.Component`` without deciding which engines
    serve it and which block type it lives in fails here, naming it — the same
    discipline the corpus digests apply to documents.
    """
    unclaimed = sweep.unclaimed_components()
    assert not unclaimed, (
        f"components in the vocabulary but in no sweep bucket: {list(unclaimed)}. "
        "Add each to tests/_helpers/a3b_sweep.py — SHARED_* if both engines "
        "serve it, a single-engine bucket if one does, ABSENT_ON_A3B if this "
        "architecture has no such tensor."
    )
    twice = sweep.double_claimed_components()
    assert not twice, f"components claimed by two buckets: {list(twice)}"


def test_the_buckets_match_what_the_engines_declare():
    """The partition is a restatement of the engines' own ``components`` sets;
    this is what keeps the restatement honest."""
    hooks = PytorchHooksEngine()
    nnterp = NnterpEngine()
    both_declared = set(hooks.components) & set(nnterp.components)
    # `mlp_activation` and `mlp_neuron_output` are declared by both engines
    # and exist on neither of this architecture's block types — the documented
    # subtraction.
    assert set(SHARED) == both_declared - set(sweep.ABSENT_ON_A3B)
    assert set(sweep.HOOKS_ONLY) == set(hooks.components) - set(nnterp.components)
    assert set(sweep.NNTERP_ONLY) == set(nnterp.components) - set(hooks.components)
    assert set(sweep.ABSENT_ON_A3B) <= set(COMPONENTS)
