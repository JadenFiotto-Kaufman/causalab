"""Project any split with a saved training mean and basis; never refit.

Workflow inputs: ``acts``, ``mean``, ``weight`` (tensors or single-entry bundle
paths). Output: ``coordinates``. Leading example/position axes are preserved.
Select one layer/position population before fitting; fit_pca pools leading axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, read_tensor, write_tensor


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    def tensor(name: str) -> Any:
        value = inputs[name]
        if isinstance(value, (str, Path)):
            value = read_tensor(Path(value), what=f"project_pca: {name}")
        return value.cpu().to(torch.float64)  # the order MPS accepts

    acts, mean, weight = (tensor(name) for name in ("acts", "mean", "weight"))
    if (
        acts.ndim < 2
        or mean.ndim != 1
        or weight.ndim != 2
        or acts.shape[-1] != mean.shape[0]
        or mean.shape[0] != weight.shape[0]
    ):
        raise StepError(
            "project_pca: acts, training mean and basis have incompatible shapes"
        )
    if not all(torch.isfinite(value).all() for value in (acts, mean, weight)):
        raise StepError("project_pca: inputs must be finite")
    write_tensor(outputs["coordinates"], (acts - mean) @ weight, slot="coordinates")
