"""The chain's producer: a measurement, published as a values object and as a
one-row table — what a `decision` step's rule reads, and what `select` reads
beside the record (workflow spec §2.8)."""

import json
from pathlib import Path
from typing import Any, Mapping


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    measured = json.loads(Path(inputs["measurement"]).read_text())
    score = float(measured["score"])
    Path(outputs["values"]).write_text(json.dumps({"score": score, "k": 8}))
    Path(outputs["table"]).write_text(json.dumps([{"score": score}]))
