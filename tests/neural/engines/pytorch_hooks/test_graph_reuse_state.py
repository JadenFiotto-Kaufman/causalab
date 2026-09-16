"""Cross-seed capture staging must include Cayley's non-parameter base."""

import pytest
import torch

from causalab.neural.shared.featurizers import Gate, Subspace
from causalab.neural.engines.pytorch_hooks.cuda_graphs import copy_stage_state

pytestmark = pytest.mark.numerical_unit


def test_cayley_seed_state_copies_without_rebinding_storage():
    captured = Subspace(8, 2, "cayley", seed=0)
    current = Subspace(8, 2, "cayley", seed=1)
    with torch.no_grad():
        next(current.parameters()).add_(0.01)
    assert not torch.equal(captured.weight, current.weight)
    tensors = dict(captured.named_parameters()) | dict(captured.named_buffers())
    addresses = {name: tensor.data_ptr() for name, tensor in tensors.items()}
    current.eval()
    copy_stage_state(captured, current)
    assert not captured.training
    torch.testing.assert_close(captured.weight, current.weight, rtol=0, atol=0)
    after = dict(captured.named_parameters()) | dict(captured.named_buffers())
    assert addresses == {name: tensor.data_ptr() for name, tensor in after.items()}
    x = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    captured.featurize(x)[0].sum().backward()
    current.featurize(x)[0].sum().backward()
    torch.testing.assert_close(
        next(captured.parameters()).grad,
        next(current.parameters()).grad,
        rtol=0,
        atol=0,
    )


def test_gate_staging_preserves_annealed_soft_and_hard_outputs():
    captured, current = Gate(8), Gate(8)
    with torch.no_grad():
        current.theta.copy_(torch.linspace(-0.5, 0.5, 8))
    current.temperature = 0.125
    x = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    copy_stage_state(captured, current)
    a, b = captured.featurize(x)[0], current.featurize(x)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.sum().backward()
    b.sum().backward()
    torch.testing.assert_close(captured.theta.grad, current.theta.grad, rtol=0, atol=0)
    current.eval()
    current.hard_eval = False
    copy_stage_state(captured, current)
    torch.testing.assert_close(
        captured.featurize(x)[0], current.featurize(x)[0], rtol=0, atol=0
    )
    current.hard_eval = True
    copy_stage_state(captured, current)
    torch.testing.assert_close(
        captured.featurize(x)[0], current.featurize(x)[0], rtol=0, atol=0
    )
