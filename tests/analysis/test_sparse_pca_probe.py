"""Sparse decoding finds a low-variance PC and never selects on evaluation."""

from typing import Any

import pytest
import torch

from causalab.analysis.sparse_pca_probe import main, probe
from causalab.io.step_io import StepError, read_table, read_values


def fixture() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(240, 2, 5, generator=generator, dtype=torch.float64)
    rows = [
        {
            "id": str(i),
            "split": (
                "train" if i < 140 else "validation" if i < 190 else "evaluation"
            ),
            "sign": "positive" if x[i, 0, 4] > 0 else "negative",
            "value": float(3 * x[i, 0, 4]),
        }
        for i in range(240)
    ]
    return {
        "coordinates": x,
        "rows": rows,
        "concepts": {"sign": "categorical", "value": "numeric"},
        "layer": 3,
        "strengths": [0.01, 0.1, 1.0],
    }


@pytest.mark.numerical_unit
def test_low_variance_pc_and_saved_outputs(tmp_path):
    inputs = fixture()
    inputs["coordinates"][:, :, 4] *= 0.001
    outputs = {
        key: tmp_path / f"{key}.json" for key in ("scores", "coefficients", "metadata")
    }
    main(inputs, outputs)
    scores = read_table(outputs["scores"])
    for row in scores[:2]:
        assert row["evaluation_score"] > 0.95
        assert 4 in row["selected_pcs"]
        assert row["evaluation_score"] > row["baseline"] + 0.4
    assert read_values(outputs["metadata"])["pc_index_base"] == 0
    assert any(
        row["pc"] == 4 and row["selected"]
        for row in read_table(outputs["coefficients"])
    )


@pytest.mark.property
def test_evaluation_labels_do_not_change_fit_or_selection():
    inputs = fixture()
    scores, coefficients, metadata = probe(inputs)
    for row in inputs["rows"][190:]:
        row["sign"] = "negative" if row["sign"] == "positive" else "positive"
        row["value"] *= -1
    changed, changed_coefficients, changed_metadata = probe(inputs)
    assert changed_coefficients == coefficients
    assert changed_metadata == metadata
    assert [r["strength"] for r in changed] == [r["strength"] for r in scores]
    assert changed[0]["evaluation_score"] < 0.1


@pytest.mark.unit
@pytest.mark.parametrize("problem", ["duplicate", "missing", "nonfinite", "constant"])
def test_invalid_labels_or_rows_refused(problem):
    inputs = fixture()
    if problem == "duplicate":
        inputs["rows"][1]["id"] = "0"
    elif problem == "missing":
        del inputs["rows"][0]["sign"]
    elif problem == "nonfinite":
        inputs["coordinates"][0, 0, 0] = float("nan")
    else:
        for row in inputs["rows"]:
            row["value"] = 1.0
    with pytest.raises(StepError):
        probe(inputs)
