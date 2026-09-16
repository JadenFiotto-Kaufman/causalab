"""``safe_open`` matches the reference on the same file, slices included."""

from __future__ import annotations

from pathlib import Path
from types import EllipsisType

import pytest
import safetensors
import safetensors.torch as reference
import torch
from hypothesis import given
from hypothesis import strategies as st

import causalab.io.fastersafetensors.torch as fst
from causalab.io.fastersafetensors.errors import FormatError
from tests.io.fastersafetensors.strategies import (
    SETTINGS,
    REFERENCE_DTYPES,
    same_tensor,
    tensor_dicts,
    tensors,
)

pytestmark = pytest.mark.property

Item = int | slice | EllipsisType | None

ints = st.integers(-6, 6)
slices = st.builds(
    slice,
    st.none() | ints,
    st.none() | ints,
    st.none() | st.integers(1, 3),
)
items: st.SearchStrategy[Item] = st.one_of(ints, slices, st.just(Ellipsis), st.none())
indices = st.one_of(
    items,
    st.lists(items, max_size=4).map(tuple).filter(lambda t: t.count(Ellipsis) <= 1),
)


def reference_open(path: Path) -> safetensors.safe_open:
    return safetensors.safe_open(str(path), "pt")


@pytest.fixture(scope="module")
def shared_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("safe_open") / "f.safetensors"
    t = {
        "b/weight": torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(
            2, 3, 4, 5
        ),
        "a/bias": torch.arange(7, dtype=torch.int64),
        "scalar": torch.tensor(2.5, dtype=torch.float64),
        "empty": torch.empty(0, 3, dtype=torch.int8),
        "half": torch.arange(12, dtype=torch.float16).reshape(3, 4) / 3,
    }
    reference.save_file(t, str(p), {"note": "shared", "format": "pt"})
    return p


def test_keys_metadata_and_tensors(shared_file: Path) -> None:
    with (
        fst.safe_open(shared_file) as ours,
        safetensors.safe_open(str(shared_file), "pt") as theirs,
    ):
        assert ours.keys() == theirs.keys()
        assert ours.offset_keys() == theirs.offset_keys()
        assert ours.metadata() == theirs.metadata()
        for name in theirs.keys():  # noqa: SIM118 — not a dict
            assert same_tensor(ours.get_tensor(name), theirs.get_tensor(name))
            ours_slice, theirs_slice = ours.get_slice(name), theirs.get_slice(name)
            assert ours_slice.get_shape() == theirs_slice.get_shape()
            assert ours_slice.get_dtype() == theirs_slice.get_dtype()


def test_no_metadata_is_none(tmp_path: Path) -> None:
    p = tmp_path / "n.safetensors"
    fst.save_file({"a": torch.zeros(1)}, p)
    with fst.safe_open(p) as f:
        assert f.metadata() is None
    fst.save_file({"a": torch.zeros(1)}, p, {})
    with fst.safe_open(p) as f:
        assert f.metadata() == {}


def test_missing_tensor_and_framework(shared_file: Path) -> None:
    with fst.safe_open(shared_file) as f, pytest.raises(FormatError, match="nope"):
        f.get_tensor("nope")
    with pytest.raises(ValueError, match="framework"):
        fst.safe_open(shared_file, framework="np")


@pytest.mark.parametrize(
    "index",
    [
        0,
        -1,
        (0, 1),
        (0, 1, 2),
        (1, 2, 3, 4),
        slice(None),
        (slice(None), 1),
        (Ellipsis, slice(1, 3)),
        (1, slice(None, None, 2)),
        (None,),
        (None, 0, None, slice(1, 3)),
        slice(1, 0),
        (slice(None), slice(None), slice(None, None, 3)),
        (0, Ellipsis, 2),
        Ellipsis,
        slice(1, 100),
        slice(-5, None),
        (slice(None), -1),
        (slice(0, 2, 2), slice(1, None), None, slice(None, None, 4)),
    ],
)
def test_slice_indexings_match_reference(shared_file: Path, index: object) -> None:
    with (
        fst.safe_open(shared_file) as ours,
        safetensors.safe_open(str(shared_file), "pt") as theirs,
    ):
        a = ours.get_slice("b/weight")[index]
        b = theirs.get_slice("b/weight")[index]
        assert a.shape == b.shape
        assert same_tensor(a, b)


@pytest.mark.parametrize(
    "index",
    [
        (0, 0, 0, 0, 0),
        5,
        -3,
        slice(None, None, -1),
        slice(None, None, 0),
        "x",
        1.5,
    ],
)
def test_slice_errors_match_reference(shared_file: Path, index: object) -> None:
    with (
        fst.safe_open(shared_file) as ours,
        safetensors.safe_open(str(shared_file), "pt") as theirs,
    ):
        with pytest.raises((IndexError, ValueError, TypeError)) as ref_info:
            theirs.get_slice("b/weight")[index]
        with pytest.raises(type(ref_info.value)) as our_info:
            ours.get_slice("b/weight")[index]
        assert str(our_info.value) == str(ref_info.value)


def test_two_ellipses_are_refused(shared_file: Path) -> None:
    # the reference expands every ellipsis in turn and fails with an
    # unrelated message; torch's rule is one ellipsis per index
    with (
        fst.safe_open(shared_file) as f,
        pytest.raises(IndexError, match="single ellipsis"),
    ):
        f.get_slice("b/weight")[..., ...]


@SETTINGS
@given(indices)
def test_slice_property_against_full_tensor(shared_file: Path, index: object) -> None:
    """Whatever the reference accepts, we return the same tensor; whatever it
    refuses, we refuse with the same class."""
    with (
        fst.safe_open(shared_file) as ours,
        safetensors.safe_open(str(shared_file), "pt") as theirs,
    ):
        try:
            expected = theirs.get_slice("b/weight")[index]
        except (IndexError, ValueError, TypeError) as err:
            with pytest.raises(type(err)):
                ours.get_slice("b/weight")[index]
            return
        got = ours.get_slice("b/weight")[index]
        assert same_tensor(got, expected)


@SETTINGS
@given(
    tensor_dicts(min_size=1, max_size=4, dtypes=REFERENCE_DTYPES),
    tensors(max_dims=2, max_extent=6, dtypes=REFERENCE_DTYPES),
)
def test_get_tensor_and_full_slice_on_random_files(
    tmp_path_factory: pytest.TempPathFactory,
    tensors_: dict[str, torch.Tensor],
    extra: torch.Tensor,
) -> None:
    tensors_ = {**tensors_, "extra": extra}
    p = tmp_path_factory.mktemp("rand") / "r.safetensors"
    reference.save_file(tensors_, str(p))
    with fst.safe_open(p) as f:
        for name, t in tensors_.items():
            assert same_tensor(f.get_tensor(name), t)
            assert same_tensor(f.get_slice(name)[...], t)
            if t.dim() and t.shape[0] > 1:
                assert same_tensor(f.get_slice(name)[1:], t[1:])
                assert same_tensor(f.get_slice(name)[0], t[0])


@pytest.mark.cuda
def test_safe_open_on_cuda(shared_file: Path) -> None:
    with fst.safe_open(shared_file, device="cuda") as f:
        t = f.get_tensor("b/weight")
        s = f.get_slice("b/weight")[1, ..., 2:4]
    assert t.device.type == "cuda" and s.device.type == "cuda"
    assert same_tensor(t.cpu(), torch.arange(120.0).reshape(2, 3, 4, 5))
    assert same_tensor(s.cpu(), torch.arange(120.0).reshape(2, 3, 4, 5)[1, ..., 2:4])


@pytest.mark.cuda
def test_get_slice_on_cuda_equals_cpu_slice(shared_file: Path) -> None:
    with (
        fst.safe_open(shared_file, device="cuda:0") as gpu,
        fst.safe_open(shared_file) as cpu,
    ):
        for index in (
            1,
            (slice(None), 1),
            (Ellipsis, slice(1, 3)),
            (0, slice(None, None, 2)),
        ):
            a = gpu.get_slice("b/weight")[index]
            b = cpu.get_slice("b/weight")[index]
            assert a.device == torch.device("cuda", 0)
            assert same_tensor(a.cpu(), b)
        assert same_tensor(gpu.get_slice("half")[1:].cpu(), cpu.get_slice("half")[1:])
        assert same_tensor(gpu.get_tensor("empty").cpu(), cpu.get_tensor("empty"))
