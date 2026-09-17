"""A tree no registered family detects still runs on the standard tree: the
block-shaped taps every family shares (``registry.BLOCK_TAPS``) resolve
through nnterp's names, and a sub-child component — a family's knowledge —
refuses by name."""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnterp_engine.adapter import (
    STANDARD_MIXERS,
    standard_adapter,
)
from causalab.neural.engines.nnterp_engine.executor import NnterpExecutor
from causalab.neural.engines.nnterp_engine.loading import load_model
from causalab.neural.shared.loading import torch_module
from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import BLOCK_TAPS

from tests._helpers import a3b_sweep as sweep
from tests.neural.engines.nnterp_engine.conftest import ROWS, TINY_GPT_NEOX

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def nnterp_neox():
    return load_model(TINY_GPT_NEOX, attn_implementation="eager")


def test_the_adapter_is_the_block_taps_alone(nnterp_neox) -> None:
    adapter = standard_adapter(torch_module(nnterp_neox.model))
    assert adapter.family == "nnterp_standard"
    assert dict(adapter.taps) == dict(BLOCK_TAPS)
    assert dict(adapter.mixers) == dict(STANDARD_MIXERS)
    assert nnterp_neox.streams == ("full_attention",) * nnterp_neox.info.num_layers


@pytest.mark.parametrize(
    "component,layer",
    [
        ("embeddings", None),
        ("block_input", 1),
        ("attention_output", 1),
        ("mlp_input", 1),
        ("mlp_output", 1),
        ("block_output", 1),
        ("ln_final", None),
        ("lm_head", None),
    ],
)
def test_a_standard_boundary_resolves(nnterp_neox, component, layer) -> None:
    doc = sweep.read_doc(component, layer)
    value = sweep.make_executor(
        NnterpExecutor, doc, nnterp_neox, rows=ROWS, with_cf=False
    ).read_value("r")
    info = nnterp_neox.info
    width = info.vocab_size if component == "lm_head" else info.hidden_size
    assert value.shape == (len(ROWS), 1, width)
    assert torch.isfinite(value).all()


@pytest.mark.parametrize("component", ["attention_input_norm", "mlp_activation"])
def test_a_sub_child_component_refuses_by_name(nnterp_neox, component) -> None:
    with pytest.raises(ProtocolError) as excinfo:
        sweep.make_executor(
            NnterpExecutor,
            sweep.read_doc(component, 1),
            nnterp_neox,
            rows=ROWS,
            with_cf=False,
        ).read_value("r")
    assert repr(component) in str(excinfo.value)
    assert excinfo.value.reason == "component_unavailable"


@pytest.mark.parametrize("component", ["attention_scores", "attention_z"])
def test_an_interior_without_an_address_on_this_tree_refuses_by_name(
    nnterp_neox, component
) -> None:
    """The engine declares the interiors for every registered tree, so a
    document naming one routes here whatever the checkpoint; on a tree the
    address table has no rows for, the executor refuses by name before any
    forward."""
    with pytest.raises(ProtocolError) as excinfo:
        sweep.make_executor(
            NnterpExecutor,
            sweep.read_doc(component, 1, pos=sweep.default_pos(component)),
            nnterp_neox,
            rows=ROWS,
            with_cf=False,
        ).read_value("r")
    assert repr(component) in str(excinfo.value)
    assert "no interior address on the 'nnterp_standard' tree" in str(excinfo.value)
    assert excinfo.value.reason == "component_unavailable"
