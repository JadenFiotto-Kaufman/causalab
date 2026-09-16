"""Export exact pairs and symbolic predictions before neural experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def export_hypotheses(
    models: Mapping[str, Any],
    hypotheses: Mapping[str, tuple[str, list[str]]],
    targets: Sequence[str],
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    roles: Mapping[str, Mapping[str, str]],
    output: Path,
) -> None:
    """Write a native pair table per target and predictions for every hypothesis.

    Dataset roles must declare ``family`` and ``split``. Supplied IDs survive;
    absent IDs are derived from the input values. Reuse these exact tables for
    intervention runs. A new export requires an empty destination.
    """
    from causalab.causal.causal_utils import rederive_trace
    from causalab.protocol.tables import write_table
    from causalab.tasks.serialize import serialize_examples

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"export destination is not empty: {output}")
    tables = {}
    for target in targets:
        if Path(target).name != target or target in (".", ".."):
            raise ValueError("target names must be plain directory names")
        model_name, variables = hypotheses[target]
        model = models[model_name]
        if model.scoring is None:
            raise ValueError(f"target {target} must declare ScoringSpec")
        pair_rows, prediction_rows = [], []
        endpoints = {}
        for dataset_name, examples in datasets.items():
            role = roles[dataset_name]
            family, split = role["family"], role["split"]
            for example in examples:
                base = rederive_trace(model, example["input"])
                donors = example["counterfactual_inputs"]
                if len(donors) != 1:
                    raise ValueError("each pair must have exactly one donor")
                donor = rederive_trace(model, donors[0])
                base_id = digest({key: base[key] for key in model.inputs})
                donor_id = digest({key: donor[key] for key in model.inputs})
                for endpoint in (base_id, donor_id):
                    if endpoint in endpoints and endpoints[endpoint] != split:
                        raise ValueError("pair endpoint occurs in more than one split")
                    endpoints[endpoint] = split
                pair_id = example.get("pair_id", digest([base_id, donor_id]))
                pair = {
                    **example,
                    "input": base,
                    "counterfactual_inputs": [donor],
                    "base_id": example.get("base_id", base_id),
                    "donor_id": example.get("donor_id", donor_id),
                    "pair_id": pair_id,
                    "example_id": example.get(
                        "example_id", f"{dataset_name}:{pair_id}"
                    ),
                    "family": family,
                }
                row = serialize_examples(
                    model, [pair], split=split, target_variables=variables
                ).rows[0]
                pair_rows.append(row)
                for name, (other_model_name, other_variables) in hypotheses.items():
                    other = models[other_model_name]
                    if (
                        other.scoring is None
                        or other.scoring.digest != model.scoring.digest
                    ):
                        raise ValueError(
                            "all hypotheses must share the task scoring identity"
                        )
                    other_base = rederive_trace(other, example["input"])
                    other_donor = rederive_trace(other, donors[0])
                    trace = other.run_interchange(
                        other_base, {v: other_donor for v in other_variables}
                    )
                    prediction_rows.append(
                        {
                            **{
                                key: row[key]
                                for key in (
                                    "example_id",
                                    "pair_id",
                                    "family",
                                    "split",
                                    "scoring_digest",
                                    "base_id",
                                    "donor_id",
                                )
                            },
                            "hypothesis_id": name,
                            "prediction": trace["raw_output"],
                            "answer_forms": list(
                                other.scoring.forms_of(
                                    trace[other.scoring.answer_variable]
                                )
                            ),
                        }
                    )
        ids = [row["example_id"] for row in pair_rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate example IDs for {target}")
        tables[target] = (pair_rows, prediction_rows)
    for target, (pairs, predictions) in tables.items():
        directory = output / target
        directory.mkdir(parents=True, exist_ok=True)
        write_table(directory / "pairs.json", pairs)
        write_table(directory / "predictions.json", predictions)
