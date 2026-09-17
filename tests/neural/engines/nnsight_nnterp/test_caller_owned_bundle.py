"""The nnterp engine's ``bundle=`` entry (spec §9, the ownership
contract).

The engine takes an :class:`NnterpBundle` the way the reference engine takes
a ``ModelBundle`` and holds it to the same realization check before any
trace — the document's ``key`` / ``revision`` / ``dtype`` and the engine's
``device`` against the bundle's — and the run receipt says
``model_source: caller``. The engine is handed to ``run_protocol``
directly, which is also what proves the executor end to end: the corpus interchange document runs through ``execute_request`` and
writes the same files and receipt as a loaded run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from causalab.neural.engines.nnsight_nnterp.engine import NnterpEngine
from causalab.neural.shared.services import check_caller_bundle
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.errors import ProtocolError
from causalab.protocol.loader import load
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv

from tests.neural.engines.nnsight_nnterp.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.smoke

DOCUMENT = CORPUS_DIR / "02_interchange_im.json"
OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> ResolutionEnv:
    artifacts = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )


def test_the_bundle_satisfies_the_shared_ownership_check(nnterp_llama) -> None:
    """The bundle protocol ``check_caller_bundle`` reads: every field it
    compares is there, and a matching realization passes."""
    realization = {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"}
    check_caller_bundle(nnterp_llama, realization, device="cpu")
    with pytest.raises(ProtocolError, match="model.attn_implementation"):
        check_caller_bundle(
            nnterp_llama, {**realization, "attn_implementation": "sdpa"}, device="cpu"
        )


def test_a_dtype_disagreement_is_refused_before_any_trace(
    nnterp_llama, env, tmp_path: Path
) -> None:
    loaded = load(DOCUMENT, env, overrides={**OVERRIDES, "model.dtype": "bf16"})
    with pytest.raises(ProtocolError, match="model.dtype") as err:
        run_protocol(loaded, env, [NnterpEngine(bundle=nnterp_llama)], tmp_path)
    assert "'bf16'" in str(err.value) and "'fp32'" in str(err.value)
    assert not (tmp_path / "iia.json").exists()


def test_a_device_disagreement_is_refused(nnterp_llama, env, tmp_path: Path) -> None:
    loaded = load(DOCUMENT, env, overrides=OVERRIDES)
    with pytest.raises(ProtocolError, match="device"):
        run_protocol(
            loaded, env, [NnterpEngine(device="cuda:1", bundle=nnterp_llama)], tmp_path
        )


def test_a_matching_bundle_runs_and_the_record_says_caller(
    nnterp_llama, env, tmp_path: Path
) -> None:
    """The valid-work twin: the document's realization and the bundle agree,
    the trace runs, and the receipt is the loaded run's save for the source."""
    loaded = load(DOCUMENT, env, overrides=OVERRIDES)
    via_loader, via_caller = tmp_path / "loaded", tmp_path / "caller"
    run_protocol(loaded, env, [NnterpEngine()], via_loader)
    result = run_protocol(loaded, env, [NnterpEngine(bundle=nnterp_llama)], via_caller)
    assert result.files
    loaded_record = json.loads((via_loader / RUN_RECORD_NAME).read_text())
    caller_record = json.loads((via_caller / RUN_RECORD_NAME).read_text())
    assert loaded_record["execution"]["model_source"] == "loaded"
    assert caller_record["execution"]["model_source"] == "caller"
    # the write set's fire record reaches the receipt (spec §4 "Fires"): one
    # point, its one intervened group, its one member fired once
    assert list(caller_record["fires"].values()) == [{"patched on base": {"patch": 1}}]
    loaded_record["execution"].pop("model_source")
    caller_record["execution"].pop("model_source")
    assert loaded_record == caller_record
    assert sorted(p.name for p in via_loader.iterdir()) == sorted(
        p.name for p in via_caller.iterdir()
    )
