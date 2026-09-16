"""CUDA-only parity for the actual Cayley forward/backward capture path."""

from __future__ import annotations

import pytest
import torch

from tests.neural.shared.test_cayley import matched_pair

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("width,rank", [(128, 4), (128, 128), (2560, 4), (2560, 32)])
def test_cayley_capture_and_replay_match_checked_torch(width, rank):
    actual, reference = matched_pair(width, rank)
    actual.cuda()
    reference.cuda()
    parameter = next(actual.parameters())
    ref_parameter = next(reference.parameters())
    generator = torch.Generator(device="cuda").manual_seed(31)
    with torch.no_grad():
        parameter.add_(
            torch.randn(parameter.shape, device="cuda", generator=generator) * 0.02
        )
    reference.load_state_dict(actual.state_dict())
    direction = torch.randn(width, rank, device="cuda", generator=generator)

    def work(stage):
        q = stage.weight
        loss = (q * direction).sum()
        loss.backward()
        return q, loss

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            actual.zero_grad(set_to_none=True)
            work(actual)
    torch.cuda.current_stream().wait_stream(stream)
    actual.zero_grad(set_to_none=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        q, loss = work(actual)
    captured_grad = parameter.grad
    assert captured_grad is not None

    # Repeat with changed parameter/input VALUES at the same storage addresses.
    # Comparing consecutive replays also detects accidentally accumulating grads.
    for _ in range(3):
        with torch.no_grad():
            parameter.add_(
                torch.randn(parameter.shape, device="cuda", generator=generator) * 0.01
            )
            direction.copy_(
                torch.randn(direction.shape, device="cuda", generator=generator)
            )
        reference.load_state_dict(actual.state_dict())
        reference.zero_grad(set_to_none=True)
        expected_q, expected_loss = work(reference)
        graph.replay()
        torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(captured_grad, ref_parameter.grad, rtol=0, atol=0)
        assert parameter.grad is captured_grad
