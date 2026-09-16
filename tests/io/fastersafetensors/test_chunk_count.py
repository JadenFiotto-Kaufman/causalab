"""The job count a rank's slice is cut into follows a byte target."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from causalab.io.fastersafetensors import _distributed
from causalab.io.fastersafetensors.errors import PlanError

pytestmark = pytest.mark.unit

GIB = 1 << 30


def test_measured_slices_get_the_counts_that_won(monkeypatch) -> None:
    monkeypatch.setattr(_distributed, "CHUNKS_OVERRIDE", None)
    monkeypatch.setattr(_distributed, "CHUNK_TARGET_BYTES", 4 * GIB)
    assert _distributed.chunk_count(int(8.8e9)) == 4  # Llama-70B, two nodes
    assert _distributed.chunk_count(int(31.4e9)) == 8  # Nemotron-Ultra shards
    assert _distributed.chunk_count(0) == 4
    assert _distributed.chunk_count(1) == 4


@given(nbytes=st.integers(min_value=0, max_value=1 << 42))
def test_count_is_clamped_and_one_job_per_target(nbytes: int) -> None:
    # not monkeypatch: Hypothesis refuses function-scoped fixtures under @given
    override = _distributed.CHUNKS_OVERRIDE
    _distributed.CHUNKS_OVERRIDE = None
    try:
        count = _distributed.chunk_count(nbytes)
    finally:
        _distributed.CHUNKS_OVERRIDE = override
    assert _distributed.CHUNKS_MIN <= count <= _distributed.CHUNKS_MAX
    if _distributed.CHUNKS_MIN < count < _distributed.CHUNKS_MAX:
        assert (
            (count - 1) * _distributed.CHUNK_TARGET_BYTES
            < nbytes
            <= count * _distributed.CHUNK_TARGET_BYTES
        )


def test_override_wins(monkeypatch) -> None:
    monkeypatch.setattr(_distributed, "CHUNKS_OVERRIDE", 3)
    assert _distributed.chunk_count(1 << 40) == 3


def test_chunking_knobs_are_part_of_the_request_fingerprint(monkeypatch) -> None:
    # The round count comes from the knobs and every round is a collective, so
    # ranks that disagree on them must fail the signature check, not hang.
    # Pin the baseline: the ambient environment may set any of these.
    monkeypatch.setattr(_distributed, "CHUNKS_OVERRIDE", None)
    monkeypatch.setattr(_distributed, "CHUNK_TARGET_BYTES", 4 * GIB)
    monkeypatch.setattr(_distributed, "COOPERATIVE_BYTES", 4 * GIB)
    wanted: list = []
    device = torch.device("cpu")
    before = _distributed.fingerprint(wanted, device, {})
    monkeypatch.setattr(_distributed, "CHUNK_TARGET_BYTES", 8 * GIB)
    assert _distributed.fingerprint(wanted, device, {}) != before
    monkeypatch.setattr(_distributed, "CHUNK_TARGET_BYTES", 4 * GIB)
    assert _distributed.fingerprint(wanted, device, {}) == before
    monkeypatch.setattr(_distributed, "CHUNKS_OVERRIDE", 6)
    assert _distributed.fingerprint(wanted, device, {}) != before
    monkeypatch.setattr(_distributed, "CHUNKS_OVERRIDE", None)
    # COOPERATIVE_BYTES decides the delivery kind, so the collectives differ.
    monkeypatch.setattr(_distributed, "COOPERATIVE_BYTES", 2 * GIB)
    assert _distributed.fingerprint(wanted, device, {}) != before


def test_env_knobs_are_refused_naming_the_variable(monkeypatch) -> None:
    monkeypatch.delenv("FASTERSAFETENSORS_X", raising=False)
    assert _distributed._env_optional_int("FASTERSAFETENSORS_X") is None
    assert _distributed._env_int("FASTERSAFETENSORS_X", 7) == 7
    monkeypatch.setenv("FASTERSAFETENSORS_X", "")
    assert _distributed._env_int("FASTERSAFETENSORS_X", 7) == 7
    monkeypatch.setenv("FASTERSAFETENSORS_X", "3")
    assert _distributed._env_int("FASTERSAFETENSORS_X", 7, minimum=1) == 3
    assert _distributed._env_gib("FASTERSAFETENSORS_X", 7) == 3 * GIB
    monkeypatch.setenv("FASTERSAFETENSORS_X", "0")
    with pytest.raises(
        PlanError, match="FASTERSAFETENSORS_X must be at least 1, got 0"
    ):
        _distributed._env_optional_int("FASTERSAFETENSORS_X", minimum=1)
    monkeypatch.setenv("FASTERSAFETENSORS_X", "auto")
    with pytest.raises(
        PlanError, match="FASTERSAFETENSORS_X must be an integer, got 'auto'"
    ):
        _distributed._env_int("FASTERSAFETENSORS_X", 7)
