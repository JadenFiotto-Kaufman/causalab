"""The probe and the plan, as Python sees them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import causalab.io.fastersafetensors.torch as fst
from causalab.io.fastersafetensors import _core
from causalab.io.fastersafetensors.errors import PlanError

pytestmark = pytest.mark.unit


def test_probe_env_keys() -> None:
    env = _core.probe_env()
    assert set(env) == {"mounts", "nvidia_fs_loaded", "cufile_library", "cpus"}
    assert env["cpus"] >= 1
    assert isinstance(env["nvidia_fs_loaded"], bool)
    assert all(
        isinstance(point, str) and isinstance(fs, str) for point, fs in env["mounts"]
    )


def test_plan_read_dict(tmp_path: Path) -> None:
    plan = _core.plan_read([str(tmp_path / "a"), str(tmp_path / "b")], [10, 20])
    assert set(plan) == {
        "files_in_flight",
        "split_bytes",
        "readers_per_file",
        "transport",
        "staging",
        "reasons",
    }
    assert 1 <= plan["files_in_flight"] <= 2
    assert plan["transport"] == "pread"
    assert plan["staging"] is None
    assert plan["reasons"]
    staged = _core.plan_read([str(tmp_path / "a")], [10], "cuda:0", 1 << 30)
    assert staged["staging"] is not None
    assert any("pinned" in r for r in staged["reasons"])


def test_plan_read_refusals(tmp_path: Path) -> None:
    with pytest.raises(PlanError):
        _core.plan_read([], [])
    with pytest.raises(PlanError, match="free"):
        _core.plan_read([str(tmp_path / "a")], [100], "cuda", 10)
    with pytest.raises(ValueError, match="device"):
        _core.plan_read([str(tmp_path / "a")], [1], "tpu")
    with pytest.raises(ValueError, match="wanted_bytes"):
        _core.plan_read([str(tmp_path / "a")], [1, 2])


def test_explain_names_class_and_files_in_flight(tmp_path: Path) -> None:
    paths = [tmp_path / f"s{i}.safetensors" for i in range(3)]
    for p in paths:
        fst.save_file({"w": torch.zeros(16)}, p)
    text = fst.explain(paths)
    classes = _core.storage_classes([str(p) for p in paths])
    assert classes[0] in text
    assert "files in flight" in text
    assert "3 file(s)" in text
    assert "host destination" in text
    single = fst.explain(paths[0])
    assert "1 file(s)" in single
    with pytest.raises(ValueError):
        fst.explain([])


def test_profile_controls_planning_and_coalescing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = json.loads(
        (
            Path(__file__).parents[3] / "docs/profiles/b200-nfs-2026-09-09.json"
        ).read_text()
    )
    # Force the measured NFS entry for this path, on any test machine.
    profile["mounts"] = {str(tmp_path): profile["storage"]["Nfs"]}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    paths = [str(tmp_path / str(i)) for i in range(16)]
    item: _core.SelectItem = (paths[0], "w", [2, 1_000_016], [(0, 2), (0, 16)], "U8")
    monkeypatch.delenv("FASTERSAFETENSORS_PROFILE", raising=False)
    _, baseline = _core.select_reads([item])
    assert baseline["reads"] == 2
    monkeypatch.setenv("FASTERSAFETENSORS_PROFILE", str(path))
    plan = _core.plan_read(paths, [32] * 16)
    assert plan["files_in_flight"] == 16
    assert plan["readers_per_file"] == min(16, _core.probe_env()["cpus"])
    assert any(
        str(path) in reason and "b200-nfs" in reason for reason in plan["reasons"]
    )
    _, selected = _core.select_reads([item])
    assert selected["reads"] == 1
    assert selected["read_bytes"] == 1_000_032
    assert selected["wanted_bytes"] == 32


def test_explicit_profile_and_invalid_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from causalab.io.fastersafetensors.errors import ProfileError

    profile = Path(__file__).parents[3] / "docs/profiles/b200-nfs-2026-09-09.json"
    monkeypatch.setenv("FASTERSAFETENSORS_PROFILE", str(profile))
    plan = _core.plan_read([str(tmp_path / "w")], [1])
    assert any("FASTERSAFETENSORS_PROFILE=" in reason for reason in plan["reasons"])
    assert any("applicability not verified" in reason for reason in plan["reasons"])
    # Provenance names have no special meaning to the runtime.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FASTERSAFETENSORS_PROFILE", "b200-nfs-2026-09-09")
    with pytest.raises(ProfileError):
        _core.plan_read(["w"], [1])
    for text in (None, "{}", '{"schema_version": 999, "source": "invalid"}'):
        path = tmp_path / "invalid.json"
        if text is not None:
            path.write_text(text)
        monkeypatch.setenv("FASTERSAFETENSORS_PROFILE", str(path))
        with pytest.raises(ProfileError):
            _core.plan_read(["w"], [1])
        with pytest.raises(ProfileError):
            _core.select_reads([])


def test_k3_inner_cut_amplification(tmp_path: Path) -> None:
    path = tmp_path / "expert.safetensors"
    fst.save_file({"w2": torch.zeros((3584, 1536), dtype=torch.uint8)}, path)
    text = fst.explain(path, select={"w2": fst.Shard(1, 0, 16)})
    assert "344064 bytes wanted, 5503584 read" in text
    assert "15.996x amplification" in text
    assert "5503584 mean bytes/read" in text


@pytest.mark.parametrize(
    ("selection", "needed", "read_bytes"),
    [
        (fst.Shard(1, 0, 16), 344_064, 5_503_584),
        ((slice(None, None, 2), slice(None)), (1792 + 3583) * 1536, 3583 * 1536),
        ((slice(0, 0, 2), slice(None)), 0, 0),
    ],
)
def test_selected_device_fit_counts_results_and_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection,
    needed: int,
    read_bytes: int,
) -> None:
    from causalab.io.fastersafetensors import _files

    path = tmp_path / "expert.safetensors"
    fst.save_file({"w2": torch.zeros((3584, 1536), dtype=torch.uint8)}, path)
    monkeypatch.setattr(_files, "device_headroom", lambda index: needed)
    # No CUDA allocations needed to test the real selection/planner boundary.
    text = fst.explain(path, device="cuda:0", select={"w2": selection})
    assert f"{read_bytes} read" in text
    assert f"tensor allocations: {needed} bytes peak" in text
    if needed:
        monkeypatch.setattr(_files, "device_headroom", lambda index: needed - 1)
        with pytest.raises(PlanError, match="free"):
            fst.explain(path, device="cuda:0", select={"w2": selection})


def test_core_plan_accepts_separate_allocation_bytes(tmp_path: Path) -> None:
    path = str(tmp_path / "w")
    # Coalesced I/O can exceed the device destination size.
    _core.plan_read([path], [1000], "cuda:0", 10, allocation_bytes=10)
    # Scratch can make peak allocation exceed the I/O volume.
    with pytest.raises(PlanError):
        _core.plan_read([path], [10], "cuda:0", 19, allocation_bytes=20)
