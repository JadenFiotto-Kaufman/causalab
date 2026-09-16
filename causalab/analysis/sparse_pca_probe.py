"""Final PCA step: sparse concept probes on frozen per-position coordinates.

Inputs: coordinates (examples, positions, PCs), rows (JSON table with id, split
and concept columns), concepts ({column: categorical|numeric}), layer, and
optional strengths (positive L1 penalties). Splits are train/validation/evaluation.
PCA must already have been fitted on exactly the training examples. Outputs:
scores (table), coefficients (table), and metadata (values object). PC indices
are zero-based. All coefficients refer to training-standardized coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import (
    StepError,
    read_table,
    read_tensor,
    write_table,
    write_values,
)


def probe(inputs: Mapping[str, Any]) -> tuple[list, list, dict]:
    """Fit on train, choose on validation, and score evaluation exactly once."""
    import warnings

    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.preprocessing import StandardScaler

    coordinates = inputs["coordinates"]
    if isinstance(coordinates, (str, Path)):
        coordinates = read_tensor(Path(coordinates))
    x = coordinates.detach().cpu().double().numpy()
    rows = inputs["rows"]
    if isinstance(rows, (str, Path)):
        rows = read_table(Path(rows))
    if x.ndim != 3 or min(x.shape) < 1 or len(rows) != len(x):
        raise StepError(
            "coordinates must be (examples, positions, PCs), aligned with rows"
        )
    if not np.isfinite(x).all():
        raise StepError("coordinates must be finite")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise StepError("rows need unique nonempty string IDs")
    split_names = ("train", "validation", "evaluation")
    if any(row.get("split") not in split_names for row in rows):
        raise StepError("each row needs a train, validation, or evaluation split")
    splits = {
        name: [i for i, row in enumerate(rows) if row["split"] == name]
        for name in split_names
    }
    if any(len(indices) < 2 for indices in splits.values()):
        raise StepError("each split needs at least two examples")
    concepts = inputs["concepts"]
    if not isinstance(concepts, dict) or not concepts:
        raise StepError("concepts must map column names to categorical or numeric")
    strengths = inputs.get("strengths", [0.1, 1.0, 10.0, 100.0])
    if not strengths or any(not np.isfinite(v) or v <= 0 for v in strengths):
        raise StepError("strengths must be finite positive penalties")
    train, validation, evaluation = (splits[name] for name in split_names)
    scores, coefficients, preprocessing = [], [], []
    for position in range(x.shape[1]):
        scaler = StandardScaler().fit(x[train, position])
        features = scaler.transform(x[:, position])
        assert scaler.mean_ is not None and scaler.scale_ is not None
        preprocessing.append(
            {
                "position": position,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
        )
        for concept, kind in concepts.items():
            if kind not in ("categorical", "numeric"):
                raise StepError(f"{concept}: kind must be categorical or numeric")
            values = [row.get(concept) for row in rows]
            if any(v is None for v in values):
                raise StepError(f"{concept}: missing labels")
            if kind == "categorical":
                if any(not isinstance(v, str) for v in values):
                    raise StepError(f"{concept}: categorical labels must be strings")
                y = np.array(values)
                classes = np.unique(y[train])
                if len(classes) < 2 or any(
                    set(y[s]) != set(classes) for s in (validation, evaluation)
                ):
                    raise StepError(
                        f"{concept}: all splits must contain the same two or more classes"
                    )
                metric = balanced_accuracy_score
                metric_name = "balanced_accuracy"
                labels, counts = np.unique(y[train], return_counts=True)
                baseline_prediction = labels[counts.argmax()]
            else:
                try:
                    y = np.array(values, dtype=float)
                except (ValueError, TypeError) as exc:
                    raise StepError(
                        f"{concept}: numeric labels must be finite numbers"
                    ) from exc
                if not np.isfinite(y).all() or any(
                    np.ptp(y[s]) == 0 for s in splits.values()
                ):
                    raise StepError(
                        f"{concept}: numeric labels need finite, nonconstant values in each split"
                    )
                metric = r2_score
                metric_name = "r2"
                baseline_prediction = float(y[train].mean())
            candidates = []
            for strength in sorted(set(strengths)):
                model = (
                    LogisticRegression(
                        penalty="l1",
                        solver="saga",
                        C=1 / strength,
                        random_state=0,
                        max_iter=50000,
                        tol=1e-4,
                    )
                    if kind == "categorical"
                    else Lasso(alpha=strength, max_iter=50000, tol=1e-4)
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    try:
                        model.fit(features[train], y[train])
                    except ConvergenceWarning as exc:
                        raise StepError(
                            f"{concept}, position {position}: probe did not converge"
                        ) from exc
                score = float(
                    metric(y[validation], model.predict(features[validation]))
                )
                coef = np.atleast_2d(model.coef_)
                support = int(np.any(coef != 0, axis=0).sum())
                candidates.append((score, -support, strength, model))
            val_score, _, strength, model = max(
                candidates, key=lambda candidate: candidate[:3]
            )
            coef = np.atleast_2d(model.coef_)
            selected = np.flatnonzero(np.any(coef != 0, axis=0)).tolist()
            score = float(metric(y[evaluation], model.predict(features[evaluation])))
            baseline = float(
                metric(y[evaluation], np.full(len(evaluation), baseline_prediction))
            )
            classes = model.classes_.tolist() if kind == "categorical" else ["numeric"]
            coefficient_classes = classes[-1:] if len(coef) == 1 else classes
            common = {
                "layer": inputs["layer"],
                "position": position,
                "concept": concept,
            }
            scores.append(
                {
                    **common,
                    "metric": metric_name,
                    "validation_score": val_score,
                    "evaluation_score": score,
                    "baseline": baseline,
                    "strength": strength,
                    "selected_pcs": selected,
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "n_evaluation": len(evaluation),
                    "classes": classes,
                    "intercept": np.atleast_1d(model.intercept_).tolist(),
                }
            )
            for class_name, weights in zip(coefficient_classes, coef, strict=True):
                coefficients.extend(
                    {
                        **common,
                        "class": class_name,
                        "pc": pc,
                        "coefficient": float(weight),
                        "selected": bool(weight != 0),
                    }
                    for pc, weight in enumerate(weights)
                )
    metadata = {
        "split_ids": {
            name: [ids[i] for i in indices] for name, indices in splits.items()
        },
        "preprocessing": preprocessing,
        "strengths": strengths,
        "selection": "max validation score; ties prefer fewer PCs, then stronger penalty",
        "pca_requirement": "basis and mean fitted on the listed training IDs only",
        "seed": 0,
        "pc_index_base": 0,
    }
    return scores, coefficients, metadata


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    scores, coefficients, metadata = probe(inputs)
    write_table(outputs["scores"], scores)
    write_table(outputs["coefficients"], coefficients)
    write_values(outputs["metadata"], metadata)
