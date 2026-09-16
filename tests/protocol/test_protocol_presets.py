"""The shipped protocol presets load offline against the fixture environment.

`causalab/configs/protocols/` is what a user copies from, so every preset has
to survive a real `causalab validate` with no model, no network and no run
tree — except the two that *are* the second half of a workflow (their
`file_path` names a step's output directory), which the workflow smoke tests
exercise instead. Until now no test loaded the flat presets at all; the three
method applications were covered by `test_method_presets.py` and the rest by
whichever smoke test happened to pick them up.

`das_pca_init.json` gets its own checks: it is the one preset whose fit starts
from an artifact, and what makes it a different method from `das.json` is
exactly what its canonical form has to carry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.protocol.loader import load

from tests.protocol._env import PCA_FIXTURE_RELPATH

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO / "causalab/configs/protocols"

#: Presets whose `file_path` is a workflow run-tree path, so they load only
#: as a step after the one that writes it: preset name → the path it names.
RUN_TREE_ONLY = {
    "dbm_apply.json": "fit/gate.safetensors",
    "dbm_head_apply.json": "fit/gate.safetensors",
    "dbm_expert_neuron_apply.json": "fit/routed_gate.safetensors",
    "mean_ablation.json": "harvest/acts.safetensors",
}

STANDALONE = sorted(
    path.name for path in PROTOCOLS.glob("*.json") if path.name not in RUN_TREE_ONLY
)


@pytest.mark.parametrize("name", STANDALONE)
def test_a_shipped_preset_loads_offline(name, env):
    loaded = load(PROTOCOLS / name, env)
    assert loaded.expansion.points, name


def test_the_run_tree_presets_really_name_a_run_tree_path():
    """The exclusion list stays honest: each excluded preset must still point
    into a run tree, or it belongs in the parametrization above."""
    for name, run_tree_path in RUN_TREE_ONLY.items():
        raw = json.loads((PROTOCOLS / name).read_text())
        assert run_tree_path in json.dumps(raw), name


def test_das_pca_init_sweeps_rank_and_seed_from_one_basis(env):
    """Five ranks × three seeds, every fit starting from the same basis: the
    rank picks how many of its columns, the seed how the rest is completed
    and how the batches are ordered."""
    loaded = load(PROTOCOLS / "das_pca_init.json", env)
    assert len(loaded.expansion.points) == 15
    ranks = {point.coords["featurizers.rot.k"] for point in loaded.expansion.points}
    seeds = {point.coords["train.seed"] for point in loaded.expansion.points}
    assert ranks == {1, 2, 4, 8, 32}
    assert seeds == {0, 1, 2}
    for point in loaded.point_documents:
        rot = point.featurizers["rot"]
        assert rot.init == {"file_path": PCA_FIXTURE_RELPATH}
        assert rot.file_path is None
        assert point.train is not None and point.train.params == ("rot",)


def test_das_pca_init_carries_the_basis_in_its_canonical_form(env):
    """What distinguishes this from the random-start `das.json`: the basis's
    bytes are in the digest, so a fit from another basis is another document."""
    loaded = load(PROTOCOLS / "das_pca_init.json", env)
    rot = loaded.canonical_document["method"]["featurizers"]["rot"]
    assert rot["init"]["file_path"] == PCA_FIXTURE_RELPATH
    assert len(rot["init"]["content_digest"]) == 64
    for canonical in loaded.canonical_points:
        assert canonical["method"]["featurizers"]["rot"]["init"] == rot["init"]
    saved = {entry.file_path for entry in loaded.document.save}
    assert saved == {"iia.json", "logit_diff.json", "ce.json", "rot.safetensors"}
