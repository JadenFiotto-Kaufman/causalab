"""Held-out offsets and distinct position geometry survive shared PCA work."""

import pytest
import torch

from causalab.analysis import fit_pca, pca_by_position, project_pca
from causalab.io.step_io import read_tensor

pytestmark = pytest.mark.numerical_unit


def test_projection_uses_training_mean(tmp_path):
    training = torch.tensor([[8.0, 0], [12.0, 0], [10.0, 1], [10.0, -1]])
    out = {
        name: tmp_path / f"{name}.safetensors"
        for name in ("weight", "mean", "coordinates")
    }
    out["spectrum"] = tmp_path / "spectrum.json"
    fit_pca.main({"acts": training, "k": 1}, out)
    heldout = torch.tensor([[20.0, 0], [22.0, 0]])
    projected = tmp_path / "heldout.safetensors"
    project_pca.main(
        {"acts": heldout, "mean": out["mean"], "weight": out["weight"]},
        {"coordinates": projected},
    )
    torch.testing.assert_close(
        read_tensor(projected), torch.tensor([[10.0], [12.0]], dtype=torch.float64)
    )


def test_grouped_fit_keeps_positions_and_heldout_rows_separate(tmp_path):
    training = torch.tensor(
        [
            [[-2.0, 0], [0, -2.0]],
            [[2.0, 0], [0, 2.0]],
            [[0, 1.0], [1.0, 0]],
            [[0, -1.0], [-1.0, 0]],
        ]
    )
    heldout = torch.tensor([[[100.0, 0], [0, 100.0]]])
    outputs = {
        n: tmp_path / f"{n}.safetensors" for n in ("weight", "mean", "coordinates")
    }
    outputs["spectrum"] = tmp_path / "spectrum.json"
    pca_by_position.main(
        {"acts": torch.cat([training, heldout]), "train_rows": [0, 1, 2, 3], "k": 1},
        outputs,
    )
    assert torch.equal(
        read_tensor(outputs["mean"]), torch.zeros(2, 2, dtype=torch.float64)
    )
    assert torch.equal(
        read_tensor(outputs["weight"]), torch.tensor([[[1.0], [0]], [[0], [1.0]]])
    )
    assert read_tensor(outputs["coordinates"])[-1].tolist() == [[100.0], [100.0]]
