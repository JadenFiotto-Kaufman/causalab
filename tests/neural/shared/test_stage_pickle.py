"""A stage pickles with plain :mod:`pickle`, and an optimizer pickled beside
it still steps its parameters — a ``subspace`` included, whose parametrized
module torch will not pickle whole (``Subspace.__reduce__``)."""

from __future__ import annotations

import pickle

import pytest
import torch

from causalab.neural.shared.featurizers import Gate, Stage, Subspace

pytestmark = pytest.mark.unit


def _stage(kind: str) -> Stage:
    if kind == "gate":
        return Gate(8)
    return Subspace(8, 4, kind, seed=3)


@pytest.mark.parametrize("kind", ["cayley", "matrix_exp", "stiefel", "gate"])
def test_a_stage_and_its_optimizer_round_trip_and_step_on_alike(kind):
    stage = _stage(kind).train()
    optimizer = torch.optim.AdamW(list(stage.parameters()), lr=0.1)
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(0))

    def update(a: Stage, b: torch.optim.Optimizer, power: int) -> None:
        b.zero_grad()
        a.featurize(x)[0].pow(power).sum().backward()
        b.step()

    update(stage, optimizer, 1)  # the optimizer has moments to carry
    before = torch.get_rng_state()
    copy, copied = pickle.loads(pickle.dumps((stage, optimizer)))
    # rebuilding a `matrix_exp` / `stiefel` stage draws from a forked stream
    assert torch.equal(before, torch.get_rng_state())

    assert copy.training and type(copy).__name__ == type(stage).__name__
    assert {id(p) for g in copied.param_groups for p in g["params"]} == {
        id(p) for p in copy.parameters()
    }
    for key, value in stage.state_dict().items():
        assert torch.equal(copy.state_dict()[key], value), key
    update(stage, optimizer, 2)
    update(copy, copied, 2)
    for slot, value in stage.slot_params().items():
        assert torch.equal(copy.slot_params()[slot], value), slot


def test_a_frozen_subspace_stays_frozen():
    stage = Subspace(8, 4, "cayley", seed=0).eval()
    stage.parametrizations.weight.original.requires_grad_(False)  # a §2.11 phase
    copy = pickle.loads(pickle.dumps(stage))
    assert not copy.training
    assert not copy.parametrizations.weight.original.requires_grad
