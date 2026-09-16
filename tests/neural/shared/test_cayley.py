"""The production low-rank Cayley chart, also used by the CUDA regression."""

import copy

import pytest
import torch

from causalab.neural.shared.featurizers import Cayley, Subspace

pytestmark = pytest.mark.numerical_unit


def matched_pair(width: int, rank: int):
    actual = Subspace(width, rank, "cayley", seed=5)
    return actual, copy.deepcopy(actual)


@pytest.mark.parametrize("rank", [1, 3, 8])
def test_cayley_matches_dense_chart_and_gradients(rank):
    generator = torch.Generator().manual_seed(5)
    chart = Cayley(
        torch.linalg.qr(torch.randn(8, rank, dtype=torch.float64, generator=generator))[
            0
        ]
    )
    x = (torch.randn(8, rank, dtype=torch.float64) * 0.1).requires_grad_()
    q = chart.base
    b = q.mT @ x
    perp = x - q @ b
    skew = q @ (b - b.mT) @ q.mT + perp @ q.mT - q @ perp.mT
    eye = torch.eye(8, dtype=x.dtype)
    expected = torch.linalg.solve(eye - skew / 2, (eye + skew / 2) @ q)
    actual = chart(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    direction = torch.randn_like(q)
    actual_grad = torch.autograd.grad((actual * direction).sum(), x, retain_graph=True)[
        0
    ]
    expected_grad = torch.autograd.grad((expected * direction).sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)
