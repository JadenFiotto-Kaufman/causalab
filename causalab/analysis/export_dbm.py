"""Export evaluated DBM masks, metrics, and provenance as portable JSON data.

The manifest lists experiments with ``id``, ``title``, and ``evaluations``.
Each evaluation names ``document``, ``run_dir``, ``data_root``, and
``artifacts_root``. Paths resolve from the manifest's directory.

Only apply documents are accepted. Their recorded digests must agree with the
compiled document and the frozen fit files. Masks use the same ``theta > 0``
readout as the generator's sigmoid gates. Missing evaluations are recorded
in ``omitted_points``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from causalab.protocol.bundles import entry_selection, select_entry
from causalab.protocol.examples import example_labels
from causalab.protocol.loader import load, load_text
from causalab.protocol.registry import get_model_info
from causalab.protocol.resolve import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    entry_table,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def encoded_mask(mask: Any) -> dict[str, Any]:
    """Choose a lossless range or bitset representation of a flat hard mask."""
    import numpy as np

    values = np.asarray(mask, dtype=bool).reshape(-1)
    boundaries = np.flatnonzero(np.diff(np.pad(values.astype(np.int8), (1, 1))))
    ranges = boundaries.reshape(-1, 2).tolist()
    ranged = {"encoding": "ranges", "ranges": ranges}
    packed = {
        "encoding": "bitset",
        "data": base64.b64encode(
            np.packbits(values, bitorder="little").tobytes()
        ).decode(),
        "unit_count": int(values.size),
    }
    return min((ranged, packed), key=lambda value: len(json.dumps(value)))


def metric_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Average eligible per-example records from one measured point."""
    values: list[float] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("metric") != name:
            raise ValueError(f"Metric table for {name} contains another metric")
        identity = str(row["example_id"])
        if identity in seen:
            raise ValueError(f"Metric {name} repeats example {identity}")
        seen.add(identity)
        if row.get("eligible", True) is False:
            continue
        value = row.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Metric {name} has an invalid eligible value")
        if name == "iia" and value not in (0, 1):
            raise ValueError("IIA records must be binary match outcomes")
        values.append(float(value))
    result: dict[str, Any] = {"value": None, "n": len(values), "total": len(rows)}
    if not values:
        result["reason"] = "No eligible saved evaluations"
        return result
    mean = sum(values) / len(values)
    result["value"] = mean
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        result["standard_error"] = math.sqrt(variance / len(values))
    return result


def model_manifest(model: dict[str, Any]) -> dict[str, Any]:
    info = get_model_info(model["key"])
    layers = []
    for layer in range(info.num_layers):
        delta = (
            info.layer_types is not None
            and info.layer_types[layer] == "linear_attention"
        )
        layers.append(
            {
                "index": layer,
                "type": "gated_delta_net" if delta else "normal_attention",
                "heads": info.linear_num_value_heads if delta else info.num_heads,
                "head_dim": info.linear_value_head_dim if delta else info.head_dim,
                "experts": info.num_experts or 0,
                "expert_dim": info.moe_intermediate_size or 0,
                "shared_expert_dim": info.shared_expert_intermediate_size or 0,
                "mlp_dim": info.intermediate_size or 0,
            }
        )
    return {
        **model,
        "layers": layers,
    }


def position(value: Any, method: dict[str, Any]) -> int | str:
    """Resolve a scalar token index or the shared all-token position."""
    if isinstance(value, str) and value != "all":
        value = method.get("positions", {}).get(value)
    if isinstance(value, dict):
        if set(value) == {"index"}:
            value = value["index"]
        elif value == {"all": True}:
            value = "all"
    if type(value) is int or value == "all":
        return value
    raise ValueError("DBM export requires a scalar token index or all positions")


def resolved_read(read: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    return {
        **read,
        "site": method["sites"][read["site"]],
        "pos": position(read["pos"], method),
    }


def gate_manifest(
    name: str, method: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    writes = [
        write for write in method["writes"].values() if write.get("featurizer") == name
    ]
    if len(writes) != 1:
        raise ValueError(f"Gate {name} must belong to one position and site")
    write = writes[0]
    if set(write) != {"site", "pos", "featurizer", "do"}:
        raise ValueError(f"Gate {name} requires a scalar or all-position swap")
    site = method["sites"][write["site"]]
    target_position = position(write["pos"], method)
    operation = write.get("do", {})
    if set(operation) != {"swap"} or not isinstance(operation["swap"], str):
        raise ValueError(f"Gate {name} requires a direct counterfactual swap")
    source = resolved_read(method["reads"][operation["swap"]], method)
    if source != {
        "site": site,
        "pos": target_position,
        "model": "original",
        "input": "counterfactual",
        "featurizer": name,
    }:
        raise ValueError(f"Gate {name} requires an aligned counterfactual swap")
    if len(site["layers"]) != 1 or "head" in site or "expert" in site:
        raise ValueError(f"Gate {name} requires one complete layer site")
    layer = site["layers"][0]
    architecture = model["layers"][layer]
    component = site["component"]
    gate = method["featurizers"][name]
    family = "mlp"
    if component in ("attention_premix", "delta_premix"):
        family = (
            "gated_delta_net" if component == "delta_premix" else "normal_attention"
        )
        if gate.get("group") == "head":
            kind, shape = "attention_head", [architecture["heads"]]
        else:
            kind, shape = (
                "attention_channel",
                [architecture["heads"], architecture["head_dim"]],
            )
    elif component == "expert_neuron_output" and gate.get("group") == "expert_neuron":
        kind, shape = (
            "routed_expert_neuron",
            [architecture["experts"], architecture["expert_dim"]],
        )
    elif component == "shared_expert_activation":
        kind, shape = "shared_expert_neuron", [architecture["shared_expert_dim"]]
    elif component == "mlp_neuron_output":
        kind, shape = "mlp_neuron", [architecture["mlp_dim"]]
    else:
        raise ValueError(f"Unsupported DBM export site {component!r}")
    if any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError(f"Unknown shape for {name}")
    return {
        "id": name,
        "layer": layer,
        "site_component": component,
        "component": kind,
        "family": family,
        "position": target_position,
        "shape": shape,
        "unit_count": math.prod(shape),
    }


def validate_measurement(method: dict[str, Any], gates: list[dict[str, Any]]) -> None:
    """Require both scores to measure the model with every exported gate active."""
    reads = [
        method["reads"][method["metrics"][name]["of"]] for name in ("iia", "logit_diff")
    ]
    resolved = [resolved_read(read, method) for read in reads]
    if resolved[0] != resolved[1]:
        raise ValueError("IIA and logit difference must use the same resolved read")
    measured = method["intervened_models"].get(reads[0]["model"])
    if (
        measured is None
        or measured["input"] != reads[0]["input"]
        or measured["input"] != "base"
    ):
        raise ValueError("DBM metrics must read the intervened model on its input")
    if not gates:
        raise ValueError("A DBM export requires at least one gate")
    exported = {gate["id"] for gate in gates}
    required = {
        name
        for name, write in method["writes"].items()
        if isinstance(write.get("featurizer"), str) and write["featurizer"] in exported
    }
    if not required.issubset(set(measured["writes"])):
        raise ValueError("Every exported gate must be active in the measured model")
    if set(measured["writes"]) != required:
        raise ValueError(
            "The measured model must contain only the exported gate writes"
        )


def frozen_mask(
    spec: dict[str, Any], coords: dict[str, Any], root: Path, digests: dict[Path, str]
) -> tuple[Any, dict[str, Any]]:
    """Read exactly the fitted tensor named by a recorded sigmoid apply."""
    import numpy as np
    from safetensors import safe_open

    if (
        spec.get("parametrization", "sigmoid") != "sigmoid"
        or "top_k" in spec
        or "pool" in spec
    ):
        raise ValueError("This exporter requires threshold replay of sigmoid DBM gates")
    path = (root / spec["file_path"]).resolve()
    if path not in digests:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digests[path] = digest.hexdigest()
    if digests[path] != spec["content_digest"]:
        raise ValueError(f"Frozen fit changed after evaluation: {path}")
    with safe_open(path, framework="numpy") as bundle:
        selection, implicit = entry_selection(spec.get("entry"), coords, "gate")
        key = select_entry(
            bundle.keys(),
            "theta",
            selection,
            what=str(path),
            coords_by_key=entry_table(bundle.metadata()),
            implicit=implicit,
        )
        theta = bundle.get_tensor(key)
        if not np.isfinite(theta).all():
            raise ValueError(f"Non-finite mask parameter in {path}:{key}")
        mask = theta.reshape(-1) > 0
    return mask, {"file_path": str(path), "entry": key, "sha256": digests[path]}


def evaluation(
    item: dict[str, Any],
    base: Path,
    digests: dict[Path, str],
    *,
    register_from_hf: bool = False,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    document = (base / item["document"]).resolve()
    run = (base / item["run_dir"]).resolve()
    artifact_root = (base / item["artifacts_root"]).resolve()
    source = load_text(document)
    if register_from_hf:
        from causalab.cli import register_model_key

        register_model_key(source)
    if "train" in source["method"]:
        raise ValueError(f"Use a held-out apply document: {document}")
    datasets = FileDatasets(root=(base / item["data_root"]).resolve())
    loaded = load(
        source,
        ResolutionEnv(
            datasets=datasets,
            artifacts=FileArtifacts(root=artifact_root),
        ),
    )
    receipt = read_json(run / "protocol.json")
    if receipt["document_digest"] != loaded.document_digest:
        raise ValueError(f"Evaluation receipt does not match {document}")
    first = loaded.canonical_points[0]
    if any(
        point["model"] != first["model"] or point["data"] != first["data"]
        for point in loaded.canonical_points
    ):
        raise ValueError("Model or data changes within one DBM evaluation")
    model = model_manifest(first["model"])
    method = first["method"]
    definitions = method["metrics"]
    if (
        definitions["iia"]["kind"] != "match"
        or definitions["logit_diff"]["kind"] != "logit_diff"
    ):
        raise ValueError(
            "DBM export requires distinct match IIA and logit difference metrics"
        )
    if definitions["iia"]["expected"] != definitions["logit_diff"]["a"]:
        raise ValueError("IIA and logit difference must target the same gold label")
    pair_rows = datasets.rows(first["data"]["base"]["dataset"])
    pair_ids = example_labels(pair_rows)
    changed = {
        identity
        for identity, row in zip(pair_ids, pair_rows)
        if row[definitions["logit_diff"]["a"]] != row[definitions["logit_diff"]["b"]]
    }
    gates = [
        gate_manifest(name, method, model)
        for name, spec in method["featurizers"].items()
        if spec["kind"] == "gate"
    ]
    tables: dict[str, list[dict[str, Any]]] = {}
    for metric in ("iia", "logit_diff"):
        saves = [save for save in method["save"] if save.get("value") == metric]
        if len(saves) != 1:
            raise ValueError(f"Expected one saved {metric} table")
        path = run / saves[0]["file_path"]
        tables[metric] = read_json(path) if path.exists() else []
    routing_path = run / "routing_mismatch.json"
    routing = read_json(routing_path) if routing_path.exists() else []
    points, omitted = [], []
    recorded = {point["digest"] for point in receipt["points"]}
    if any(
        row["produced_by"] not in recorded for rows in tables.values() for row in rows
    ):
        raise ValueError(f"Unrecorded point in evaluation tables: {run}")
    for record in receipt["points"]:
        index, digest = record["index"], record["digest"]
        if loaded.point_digests[index] != digest:
            raise ValueError(f"Point digest changed: {run}:{index}")
        if record["coords"] != dict(loaded.expansion.points[index].coords):
            raise ValueError(f"Point coordinates changed: {run}:{index}")
        scores = {}
        for name, rows in tables.items():
            saved = [row for row in rows if row["produced_by"] == digest]
            for row in saved:
                for axis, value in record["coords"].items():
                    serialized = (
                        value
                        if isinstance(value, (int, float, str, bool))
                        else json.dumps(value, sort_keys=True)
                    )
                    if row.get(axis) != serialized:
                        raise ValueError(
                            f"Metric coordinates changed: {run}:{index}:{name}"
                        )
            if saved and {str(row["example_id"]) for row in saved} != set(pair_ids):
                raise ValueError(
                    f"Incomplete or unknown example records: {run}:{index}:{name}"
                )
            if name == "logit_diff":
                saved = [row for row in saved if str(row["example_id"]) in changed]
            scores[name] = metric_summary(saved, name)
        if not any(score["n"] for score in scores.values()):
            omitted.append({"id": digest, "reason": "No eligible saved evaluations"})
            continue
        concrete = loaded.canonical_points[index]["method"]
        if concrete["metrics"] != definitions:
            raise ValueError("Metric definitions change within one experiment")
        validate_measurement(concrete, gates)
        masks, fits, selected = {}, {}, 0
        for gate in gates:
            current = gate_manifest(gate["id"], concrete, model)
            if current != gate:
                raise ValueError("Gate layout changes within one experiment")
            mask, fit = frozen_mask(
                concrete["featurizers"][gate["id"]],
                record["coords"],
                artifact_root,
                digests,
            )
            if len(mask) != gate["unit_count"]:
                raise ValueError(f"Mask shape mismatch: {gate['id']}")
            masks[gate["id"]] = encoded_mask(mask)
            fits[gate["id"]] = fit
            selected += int(mask.sum())
        eligible = sum(gate["unit_count"] for gate in gates)
        matching = [row for row in routing if row["point"] == digest]
        coverage: dict[str, Any] | None = None
        if matching:
            coverage = {
                "mismatched": sum(row["mismatched"] for row in matching),
                "slots": sum(row["slots"] for row in matching),
                "by_gate": {},
            }
            for name, write in concrete["writes"].items():
                rows = [row for row in matching if row["write"] == name]
                if rows:
                    slots = sum(row["slots"] for row in rows)
                    coverage["by_gate"][write["featurizer"]] = {
                        "slots": slots,
                        "matched": slots - sum(row["mismatched"] for row in rows),
                    }
        points.append(
            {
                "id": digest,
                "fit_id": hashlib.sha256(
                    json.dumps(
                        {
                            name: {"sha256": fit["sha256"], "entry": fit["entry"]}
                            for name, fit in fits.items()
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "coords": record["coords"],
                "selected_count": selected,
                "eligible_count": eligible,
                "sparsity": 1 - selected / eligible,
                "metrics": scores,
                "masks": masks,
                "routing": coverage,
                "provenance": {
                    "document": str(document),
                    "run_dir": str(run),
                    "dataset": first["data"]["base"],
                    "fits": fits,
                    "metric_definitions": definitions,
                    "comparison": {
                        "data": loaded.canonical_points[index]["data"],
                        "metrics": definitions,
                        "readouts": {
                            name: resolved_read(concrete["reads"][spec["of"]], concrete)
                            for name, spec in definitions.items()
                        },
                    },
                    "logit_difference_population": "answer-changing pairs",
                },
            }
        )
    return model, gates, points, omitted


def export(manifest_path: Path, *, register_from_hf: bool = False) -> dict[str, Any]:
    """Validate frozen DBM applies and export their measured masks and scores."""
    manifest = read_json(manifest_path)
    result: dict[str, Any] = {
        "schema_version": 1,
        "synthetic": False,
        "experiments": [],
    }
    ids: set[str] = set()
    digests: dict[Path, str] = {}
    for experiment in manifest["experiments"]:
        identifier = experiment["id"]
        if identifier in ids:
            raise ValueError(f"Repeated experiment ID: {identifier}")
        ids.add(identifier)
        combined = {
            "id": identifier,
            "title": experiment["title"],
            "points": [],
            "omitted_points": [],
        }
        seen: set[str] = set()
        comparison = None
        for item in experiment["evaluations"]:
            model, gates, points, omitted = evaluation(
                item, manifest_path.parent, digests, register_from_hf=register_from_hf
            )
            if result.setdefault("model", model) != model:
                raise ValueError("Experiments use different model realizations")
            if combined.setdefault("gates", gates) != gates:
                raise ValueError(
                    "Evaluation documents use different component universes"
                )
            for point in points:
                contract = point["provenance"]["comparison"]
                if comparison is not None and comparison != contract:
                    raise ValueError(
                        "Evaluation data, metrics, or readout differ within one curve"
                    )
                comparison = contract
                if point["id"] in seen:
                    raise ValueError(f"Repeated evaluated point: {point['id']}")
                seen.add(point["id"])
                combined["points"].append(point)
            combined["omitted_points"].extend(omitted)
        if not combined["points"]:
            raise ValueError(f"Experiment {identifier} has no saved, evaluated masks")
        combined["unit"] = (
            "heads"
            if all(gate["component"] == "attention_head" for gate in combined["gates"])
            else "neurons"
        )
        combined["position_mode"] = (
            "independent"
            if len({str(gate["position"]) for gate in combined["gates"]}) > 1
            else "shared"
        )
        result["experiments"].append(combined)
    if not result["experiments"]:
        raise ValueError("Manifest has no experiments")
    return result
