"""Shared resolution-environment construction for the protocol tests.

One helper used by both the pytest fixtures (conftest.py) and the
digest-pin regeneration script (update_corpus_digests.py), so the pinned
digests and the asserting tests are guaranteed to resolve against identical
fixture content.

Two fixtures are generated rather than committed: ``rot_k8.safetensors``
(corpus file 09's ``file_path`` featurizer) and ``block_output_L18.safetensors``
(the PCA basis ``configs/protocols/das_pca_init.json`` starts from). Their
bytes are deterministic — sorted-key header JSON, zero-filled weight — so the
content digests inside the canonical forms that hash them are stable across
machines and sessions.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Mapping

from causalab.tasks import TASKS_ROOT
from causalab.protocol.resolve import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    build_artifact_identity,
)

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS_DIR = Path(__file__).parent.parent / "protocols"
ROT_FIXTURE_RELPATH = "artifacts/weekdays/llama31_8b/subspace/rot_k8.safetensors"
PCA_FIXTURE_RELPATH = "artifacts/weekdays/llama31_8b/pca/block_output_L18.safetensors"


def write_rot_fixture(artifacts_root: Path) -> Path:
    """A deterministic fitted-DAS bundle matching 09_das_apply_im.json:
    stamped identity for (Llama-3.1-8B @ main in bf16, block_output L18, k=8,
    cayley, fp32 params), weight zeros. Load-time checks read only the
    header.

    ``model_dtype`` is the *model's* precision and ``dtype`` the featurizer
    params', and they differ on purpose: a fit runs a bf16 backbone with fp32
    featurizers (``train.precision``), and both are stamped."""
    target = artifacts_root / ROT_FIXTURE_RELPATH
    identity = build_artifact_identity(
        produced_by="0" * 64,
        model_key="meta-llama/Llama-3.1-8B",
        model_revision="main",
        # 09 declares bf16, matching the fit it applies (weekdays_das_sweep);
        # model_dtype is part of ArtifactIdentity, so the fixture stamps it too
        model_dtype="bf16",
        tokenizer="meta-llama/Llama-3.1-8B",
        site={"component": "block_output", "layers": [18]},
        k=8,
        parametrization="cayley",
        dtype="fp32",
        trained_on="weekdays/data#train",
        trained_on_digest="0" * 64,
        engine="pytorch_hooks",
        commit="fixture",
    )
    # This fixture is 09's "previously fitted" artifact, and artifacts fitted
    # before the backend→engine rename carry the old stamp key. Keeping the old
    # key keeps 09's content_digest (hence its pinned canonical form)
    # byte-stable across the rename, and keeps the loader's tolerance of
    # pre-rename bundles under test. Flip to "engine" only with a corpus
    # re-pin.
    identity["backend"] = identity.pop("engine")
    return write_zero_bundle(target, identity, (4096, 8))


def write_pca_fixture(artifacts_root: Path) -> Path:
    """A deterministic PCA-basis bundle matching ``das_pca_init.json``'s
    ``init``: stamped the way the workflow runner stamps a
    ``causalab.analysis.fit_pca`` output over a harvest of (Llama-3.1-8B @
    main in bf16, block_output L18) — provenance, model realization and site
    inherited from the harvest, the basis's own rank (32) and fp32 dtype,
    engine ``script`` — and the single-entry ``entries`` table
    ``step_io.stamp_tensor`` writes. Weight zeros: load-time checks read only
    the header."""
    identity = build_artifact_identity(
        produced_by="1" * 64,
        model_key="meta-llama/Llama-3.1-8B",
        model_revision="main",
        model_dtype="bf16",
        site={"component": "block_output", "layers": [18]},
        k=32,
        dtype="fp32",
        engine="script",
    )
    identity["entries"] = json.dumps(
        {"weight": {"slot": "weight", "coords": {}}}, sort_keys=True
    )
    return write_zero_bundle(artifacts_root / PCA_FIXTURE_RELPATH, identity, (4096, 32))


def write_zero_bundle(
    target: Path, metadata: dict[str, str], shape: tuple[int, int]
) -> Path:
    """One fp32 ``weight`` of ``shape``, zero-filled, under a sorted-key
    header carrying ``metadata`` (none at all when empty — an unstamped
    bundle) — byte-deterministic, so the content digest a document's
    canonical form takes from it is stable across machines."""
    target.parent.mkdir(parents=True, exist_ok=True)
    n_bytes = shape[0] * shape[1] * 4
    header: dict[str, object] = {
        "weight": {"dtype": "F32", "shape": list(shape), "data_offsets": [0, n_bytes]},
    }
    if metadata:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with target.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(bytes(n_bytes))
    return target


def build_env(artifacts_root: Path) -> ResolutionEnv:
    """The test resolution environment: committed JSON fixture tables for
    datasets — with the shipped task tables behind them, as the CLI has it, so
    a shipped document and a fixture document load in the same env —
    ``artifacts_root`` (a copy of fixtures/artifacts plus the generated bundle)
    for artifacts."""
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=artifacts_root),
    )


#: The shipped weekdays table spells every weekday, and tiny-random's
#: sentencepiece tokenizer splits " Thursday" / " Wednesday" into several
#: pieces — so a `match` or `logit_diff` over it is refused at tiny scale
#: ([P2], no closed metric kind for multi-token answers). The tiny-scale smoke
#: runs therefore retarget a shipped document's dataset refs onto the 4-row
#: fixture table (whose days happen to be single tokens there), exactly the way
#: they retarget the model. Keyed by the ref a shipped document names.
FIXTURE_INPUTS = {"natural_domains_arithmetic/data/weekdays": "weekdays/data"}


def fixture_input_overrides(document: Mapping[str, Any]) -> dict[str, str]:
    """`set` entries pointing ``document``'s dataset refs at the fixture tables:
    one per data role, plus ``train.eval.split`` when the document trains."""

    def fixture(ref: str) -> str | None:
        base, _, fragment = ref.partition("#")
        if base not in FIXTURE_INPUTS:
            return None
        return (
            f"{FIXTURE_INPUTS[base]}#{fragment}" if fragment else FIXTURE_INPUTS[base]
        )

    out: dict[str, str] = {}
    # the four groups (intervention protocol spec §1): the inputs are `data`, the
    # fit is `method.train`
    for role, spec in document.get("data", {}).items():
        if (mapped := fixture(spec["dataset"])) is not None:
            out[f"data.{role}.dataset"] = mapped
    split = document.get("method", {}).get("train", {}).get("eval", {}).get("split")
    if split is not None and (mapped := fixture(split)) is not None:
        out["train.eval.split"] = mapped
    return out
