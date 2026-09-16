"""A gated step: writes one values object saying it ran, carrying whatever
scalar it was handed (the transitive dependent reads it by key)."""

import json
from pathlib import Path
from typing import Any, Mapping


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    Path(outputs["out"]).write_text(json.dumps({"ran": True, "k": inputs.get("k")}))
