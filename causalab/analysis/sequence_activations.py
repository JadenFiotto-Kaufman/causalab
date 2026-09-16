"""Gather aligned prediction activations from a shared all-position harvest.

Inputs: acts (a single-entry harvest path or tensor), rows (the prepared JSON
table or its rows). Output: acts (examples, targets, features). Works with the
protocol's dense and flattened ragged all-position saves. The prepared token
counts reconstruct row boundaries; padding never enters the saved all read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, read_table, read_tensor, write_tensor


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    acts, rows = inputs["acts"], inputs["rows"]
    if isinstance(acts, (str, Path)):
        acts = read_tensor(Path(acts))
    if isinstance(rows, (str, Path)):
        rows = read_table(Path(rows))
    counts = [len(row["input_encoding"]["input_ids"]) for row in rows]
    targets = [len(row["targets"]) for row in rows]
    if not counts or len(set(targets)) != 1 or targets[0] == 0:
        raise StepError("sequence_activations needs a nonempty aligned target cohort")
    if acts.ndim == 3:
        if acts.shape[0] != len(rows) or any(n != acts.shape[1] for n in counts):
            raise StepError("dense harvest shape does not match the prepared rows")
        windows = acts.unbind(0)
    elif acts.ndim == 2 and acts.shape[0] == sum(counts):
        windows = acts.split(counts)
    else:
        raise StepError("all-position harvest shape does not match the prepared rows")
    aligned = []
    for row, window in zip(rows, windows):
        indices = [target["prediction_position"] for target in row["targets"]]
        if any(type(i) is not int or not 0 <= i < len(window) for i in indices):
            raise StepError("target prediction position is outside its sequence")
        aligned.append(window[indices])
    write_tensor(outputs["acts"], torch.stack(aligned), slot="acts")
