"""``causalab dry-run`` — everything a run decides before weights load,
resolved and reported (spec §9).

The acceptance clause, verbatim: *a document requesting an unavailable site
reports it with a reason code and a nonzero exit, on a machine with no
accelerator and no model cached.* The T-a tests run the real CLI in a
**subprocess** with the HF caches offline and assert ``torch`` was never
imported — that is "no accelerator and no model cached" — and that the
refusal carries the rule (``[V4]``), the reason (``component_unavailable``)
and the site. The T-b tests are the valid-work twins:
the shipped 64-point scan dry-runs to exit 0 with the measured counts, every
site ``available`` with its shape and width, the shard arithmetic as a
ceiling, and the ``undecided`` line naming the tokenizer facts. T-c is the
never-called-loader assertion the legality suite established, re-used with
the same monkeypatch seam. The censuses hold the report's two closed
vocabularies to the spec's tables, and the fail-closed sweep dry-runs every
shipped document.

Every docstring says how the test fails on a tree without the change: on the
base, ``dry-run`` is not a verb (argparse exit 2, ``invalid choice``) and
``causalab.protocol.dry_run`` does not import.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.protocol.compile import compile_protocol
from causalab.protocol.dry_run import (
    SITE_STATUSES,
    UNDECIDED_TOPICS,
    DryRunReport,
    dry_run,
    shard_count,
    site_report,
)
from causalab.protocol.engine import Engine, ExecutionRequest, RunResult
from causalab.protocol.errors import ValidationError
from causalab.protocol.registry import get_model_info
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import COMPONENTS

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.protocol.test_compile_protocol import _first_column, _section, _tables
from tests.protocol.test_legality_before_weights import ENGINE_DECIDED, REFUSALS
from tests.protocol.test_protocol_presets import RUN_TREE_ONLY

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PRESETS = REPO / "causalab/configs/protocols"
DEMOS = REPO / "demos"
SCAN = "07_weekdays_locate_scan_im.json"
INTERCHANGE = "02_interchange_im.json"


# --------------------------------------------------------------------------- #
# helpers: the CLI in-process and in a subprocess, and engine stubs
# --------------------------------------------------------------------------- #


def _argv(name: str, artifacts_root: Path, *extra: str) -> list[str]:
    return [
        "dry-run",
        str(CORPUS_DIR / name),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        *extra,
    ]


_PROBE = """
import contextlib, io, json, sys
from causalab.cli import main

out, err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
    code = main(json.loads(sys.argv[1]))
print(json.dumps({"code": code, "out": out.getvalue(), "err": err.getvalue(),
                  "torch": "torch" in sys.modules}))
"""


def _offline(argv: list[str]) -> dict[str, Any]:
    """The CLI in a fresh interpreter with the HF caches offline: the exit
    code, both streams, and whether ``torch`` was imported. A traceback
    (anything but a clean return from ``main``) fails here, on the
    subprocess's stderr."""
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(argv)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


class _Stub(Engine):
    """Serves the whole component vocabulary; the §8 verbs vary per test."""

    def __init__(self, name: str, capabilities: frozenset[str]) -> None:
        self.name = name
        self.capabilities = capabilities
        self.components = frozenset(COMPONENTS)
        self.writable_components = frozenset(COMPONENTS)
        self.is_local = True

    def execute(self, request: ExecutionRequest) -> RunResult:
        raise AssertionError(f"{self.name} executed under a dry run")


FULL = frozenset({"grad", "paired_forward", "full_logits", "pytorch_fn_local"})


def _install(
    monkeypatch: pytest.MonkeyPatch, module: str, attr: str, cls: type
) -> None:
    """Swap a lazily-imported engine module for a stub (the system-boundary
    mock ``docs/TESTS.md`` allows), so ``load_engines`` builds it and never
    imports torch."""
    import types

    stub = types.ModuleType(module)
    setattr(stub, attr, cls)
    monkeypatch.setitem(sys.modules, module, stub)


def _hooks_stub(capabilities: frozenset[str]) -> type:
    class _Hooks(_Stub):
        def __init__(self, *, device: str = "cpu", batch_rows: int | None = None):
            super().__init__("pytorch_hooks", capabilities)

    return _Hooks


def _nnterp_stub(capabilities: frozenset[str]) -> type:
    class _Nnterp(_Stub):
        def __init__(self, *, device: str = "cpu"):
            super().__init__("nnterp", capabilities)

    return _Nnterp


def _compile(raw: dict[str, Any], env: ResolutionEnv) -> Any:
    return compile_protocol(
        in_order(raw), None, None, env.datasets, env.artifacts, None
    )


# --------------------------------------------------------------------------- #
# T-a — the acceptance clause, verbatim, in a subprocess with nothing cached
# --------------------------------------------------------------------------- #


def test_a_an_unavailable_site_is_refused_with_its_reason_code_offline(
    artifacts_root: Path,
) -> None:
    """The acceptance clause: `routed_output` on the dense Llama entry has
    no tensor, and the dry run says so with the rule, the reason code and the
    site, exits 1, and never imports torch. On the base `dry-run` is not a
    verb (argparse exits 2, so the probe never prints); under the mutation
    that reports the site as `undecided` instead of refusing, the code is 0."""
    result = _offline(
        _argv(
            INTERCHANGE, artifacts_root, "--set", "sites.target.component=routed_output"
        )
    )
    assert result["code"] == 1
    assert "[V4]" in result["err"] and "component_unavailable" in result["err"]
    assert (
        "routed_output" in result["err"] and "sites.target.component" in result["err"]
    )
    assert "Traceback" not in result["err"]
    assert not result["torch"], "a dry run imported torch"


def test_a_the_hybrid_tower_twin_names_the_layer_kind(artifacts_root: Path) -> None:
    """`attention_premix` at a Gated DeltaNet layer of the A3B: refused
    offline from the entry's `layer_types`, naming the mixer the layer
    carries and the layers that carry the other one — no model loaded."""
    result = _offline(
        _argv(
            INTERCHANGE,
            artifacts_root,
            "--set",
            "model.key=Qwen/Qwen3.6-35B-A3B",
            "--set",
            "sites.target.component=attention_premix",
            "--set",
            "sites.target.layers=20",
        )
    )
    assert result["code"] == 1
    assert "[V4]" in result["err"] and "component_unavailable" in result["err"]
    assert "linear_attention" in result["err"] and "full_attention" in result["err"]
    assert not result["torch"]


def test_a_an_unregistered_key_is_the_registry_refusal_never_a_fetch(
    artifacts_root: Path,
) -> None:
    """No model cached, no network: an unregistered `model.key` is the
    registry's `[V4] at model.key`, exit 1, no traceback — and with
    `--register-from-hf` the flag is refused (`[P4]`) instead of inherited.
    Under the mutation that lets `dry-run` inherit the flag, the offline
    `AutoConfig.from_pretrained` raises an uncaught `OSError` (a traceback,
    so the probe exits nonzero) and imports torch on the way."""
    plain = _offline(
        _argv(INTERCHANGE, artifacts_root, "--set", "model.key=nobody/no-such-model")
    )
    assert plain["code"] == 1
    assert "[V4] at model.key" in plain["err"] and "Traceback" not in plain["err"]
    assert not plain["torch"]
    flagged = _offline(
        _argv(
            INTERCHANGE,
            artifacts_root,
            "--set",
            "model.key=nobody/no-such-model",
            "--register-from-hf",
        )
    )
    assert flagged["code"] == 1
    assert "[P4]" in flagged["err"] and "--register-from-hf" in flagged["err"]
    assert "Traceback" not in flagged["err"]
    assert not flagged["torch"]


def test_a_the_refusal_record_carries_code_path_and_reason(
    artifacts_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """In-process twin of the clause: the `refused:` line is `validate`'s,
    and the record beneath it is what the rendered text lacks — the rule's
    slug and the reason code."""
    code = main(
        _argv(
            INTERCHANGE, artifacts_root, "--set", "sites.target.component=routed_output"
        )
    )
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("refused: [V4] at sites.target.component")
    assert (
        "code V4 (references_resolve) at sites.target.component, reason "
        "component_unavailable" in err
    )


def test_a_site_report_answers_refused_for_a_site_the_compile_has_not_seen() -> None:
    """The site vocabulary's third value: asked directly about a tensor the
    entry lacks, or a `head` on a component without a head axis,
    `site_report` answers `refused` with the reason-coded record."""
    gpt2 = get_model_info("gpt2")
    absent = site_report("target", "routed_output", gpt2, layers=[3])
    assert absent.status == "refused" and absent.refusal is not None
    assert absent.refusal.code == "V4"
    assert absent.refusal.reason == "component_unavailable"
    assert absent.refusal.path == "sites.target.component"
    headed = site_report("h", "block_output", gpt2, layers=[3], head=0)
    assert headed.status == "refused" and headed.refusal is not None
    assert headed.refusal.code == "V4" and "head" in headed.refusal.message


# --------------------------------------------------------------------------- #
# T-b — the valid-work twins: the shipped scan, measured
# --------------------------------------------------------------------------- #


def test_b_the_shipped_scan_reports_the_measured_counts(
    artifacts_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Corpus 07: 64 points (2 positions × 32 layers), 2 forwards per point
    and 65 once the shared counterfactual harvest is interned, the data
    role's digest and 9 columns, every site `available` with its shape and
    width, and the `undecided` line last, naming the tokenizer facts. On the
    base `dry-run` exits 2 at argparse."""
    code = main(_argv(SCAN, artifacts_root))
    assert code == 0
    out = capsys.readouterr().out
    assert "points    64" in out
    assert "forwards  2 per point, 65 interned" in out
    assert "requires  ['component:block_output', 'component:block_output:write'" in out
    assert re.search(
        r"weekdays/data#train \(base, counterfactual\): digest [0-9a-f]{16}… 9 columns",
        out,
    )
    assert "target: block_output layers 0..31 (32): available" in out
    assert "shape (batch, position, feature), width 4096, no head axis" in out
    assert "lm_head: lm_head: available" in out and "width 128256" in out
    assert "refused" not in out
    last = out.strip().splitlines()[-1]
    assert last.startswith("undecided (decided when the run encodes its inputs): ")
    assert "tokenization" in last and "controls" in last and "pair_validity" in last
    assert "shards    64 points; pass --shard-size N to plan" in out


def test_b_shards_is_the_ceiling(
    artifacts_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--shard-size 4` over 64 points is 16 shards, `--shard-size 5` is 13
    — a ceiling, not a floor: the mutation `n_points // shard_size` prints
    12 for the second."""
    assert main(_argv(SCAN, artifacts_root, "--shard-size", "4")) == 0
    assert "shards    16 of at most 4 points (64 points)" in capsys.readouterr().out
    assert main(_argv(SCAN, artifacts_root, "--shard-size", "5")) == 0
    assert "shards    13 of at most 5 points (64 points)" in capsys.readouterr().out
    assert shard_count(64, 5) == 13 and shard_count(64, 4) == 16
    assert shard_count(64, None) is None and shard_count(1, 4) == 1
    with pytest.raises(ValueError):
        shard_count(64, 0)


def test_b_a_non_positive_shard_size_is_argparses_refusal(
    artifacts_root: Path,
) -> None:
    """Exit 2, like every malformed count flag."""
    with pytest.raises(SystemExit) as exit_info:
        main(_argv(SCAN, artifacts_root, "--shard-size", "0"))
    assert exit_info.value.code == 2


def test_b_the_api_report_answers_the_twelve_facts(
    env: ResolutionEnv, artifacts_root: Path
) -> None:
    """`dry_run` on a path: the report's fields against the census-measured
    values — composition digest equal to the compile's, one data ref with
    two roles, the Llama entry, 64 points on two axes, 2/65 forwards, 16
    shards of 4, the derived capabilities, no engines asked (so `engines`
    is undecided), every site available, the inventory undecided on an
    entry without `layer_types`, the reads with their metrics, two `save`
    entries, no refusals. On the base the module does not import."""
    path = CORPUS_DIR / SCAN
    report = dry_run(path, env, shard_size=4)
    compiled = compile_protocol(
        path, path.parent, None, env.datasets, env.artifacts, None
    )
    assert isinstance(report, DryRunReport) and report.ok
    assert report.composition.digest == compiled.digests.document
    assert report.composition.overrides == {}
    assert [(d.ref, d.roles, len(d.columns)) for d in report.data] == [
        ("weekdays/data#train", ("base", "counterfactual"), 9)
    ]
    assert report.model.key == "meta-llama/Llama-3.1-8B"
    assert report.model.num_layers == 32 and report.model.layer_pattern is None
    assert report.points.n == 64
    assert report.points.axes == (("positions.tap", 2), ("sites.target.layers", 32))
    assert (report.forwards.per_point, report.forwards.campaign) == (2, 65)
    assert (report.shards.n_points, report.shards.shard_size, report.shards.count) == (
        64,
        4,
        16,
    )
    assert report.capabilities == tuple(sorted(compiled.capabilities))
    assert report.engines == ()
    assert {site.status for site in report.sites} == {"available"}
    target = next(site for site in report.sites if site.name == "target")
    assert target.layers == tuple(range(32)) and target.width == 4096
    assert target.shape == "(batch, position, feature)" and target.head_space is None
    assert target.reads == ("nnterp", "pytorch_hooks") and target.writes is not None
    assert report.inventory is None
    assert {read.name: read.metrics for read in report.readouts} == {
        "v_cf": (),
        "logits": ("iia", "logit_diff"),
    }
    assert [out.file_path for out in report.outputs] == ["iia.json", "logit_diff.json"]
    assert report.refusals == () and report.diagnostics == ()
    assert set(report.undecided_topics) == {
        "engines",
        "inventory",
        "tokenization",
        "pair_validity",
        "controls",
    }
    # the tokenizer fact names the windowed read and write of the scan
    tokenization = next(u for u in report.undecided if u.topic == "tokenization")
    assert "reads.v_cf" in tokenization.detail and "writes.patch" in tokenization.detail


def test_b_the_hybrid_entry_reports_the_inventory_and_the_stream(
    env: ResolutionEnv,
) -> None:
    """On the A3B the entry declares `layer_types`, so the per-layer
    inventory resolves offline and the stream at the site's layer is known;
    `attention_premix` at a full-attention layer is `available` in query-head
    space."""
    raw = base_doc()
    raw["model"]["key"] = "Qwen/Qwen3.6-35B-A3B"
    raw["method"]["sites"]["tgt"] = {"component": "attention_premix", "layers": [3]}
    report = dry_run(_compile(raw, env), env)
    assert report.inventory is not None and len(report.inventory.layers) == 40
    assert report.inventory.count("full_attention") == 10
    assert "inventory" not in report.undecided_topics
    tgt = next(site for site in report.sites if site.name == "tgt")
    assert tgt.status == "available" and tgt.stream == "full_attention"
    info = get_model_info("Qwen/Qwen3.6-35B-A3B")  # 16 query heads of 256 (measured)
    assert tgt.head_space == info.num_heads == 16
    assert tgt.width == info.num_heads * info.head_dim == 4096


def test_b_a_stream_bound_site_on_a_dense_entry_is_undecided_not_green() -> None:
    """`attention_premix` on gpt2: the entry declares no `layer_types`, so
    which mixer layer 3 carries is the run's to decide against the module —
    the report says `undecided`, never `available`."""
    site = site_report("x", "attention_premix", get_model_info("gpt2"), layers=[3])
    assert site.status == "undecided"
    assert site.stream == "full_attention"  # the row's stream, known; the layer's, not
    assert any("layer_types" in why for why in site.undecided)
    assert site.width == 768 and site.head_space == 12


def test_b_a_pinned_engine_that_serves_is_reported(
    artifacts_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    """`--engine pytorch_hooks` builds the engine (a stub at the system
    boundary, no torch) and reports that it serves; exit 0."""
    _install(
        monkeypatch,
        "causalab.neural.engines.pytorch_hooks",
        "PytorchHooksEngine",
        _hooks_stub(FULL),
    )
    assert main(_argv(SCAN, artifacts_root, "--engine", "pytorch_hooks")) == 0
    assert "engine    pytorch_hooks: serves" in capsys.readouterr().out


def test_b_a_pinned_engines_shortfall_is_reported_and_exits_1(
    artifacts_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    """A stub offering nothing: `check_engine`'s refusal becomes a
    `capability_shortfall` diagnostic naming what is missing — printed with
    the rest of the report, not raised — and the pinned engine's shortfall is
    exit 1, with a `refused:` line on stderr."""
    _install(
        monkeypatch,
        "causalab.neural.engines.pytorch_hooks",
        "PytorchHooksEngine",
        _hooks_stub(frozenset()),
    )
    code = main(_argv(SCAN, artifacts_root, "--engine", "pytorch_hooks"))
    assert code == 1
    captured = capsys.readouterr()
    assert "engine    pytorch_hooks: capability_shortfall" in captured.out
    assert "lacks ['paired_forward']" in captured.out
    assert "points    64" in captured.out  # the rest of the report still prints
    assert captured.err.startswith("refused: [V13]")


def test_b_under_auto_another_candidate_serving_is_exit_0(
    artifacts_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    """`auto` is routing: the reference stub falls short and the nnterp stub
    serves, so the report carries one shortfall and one `serves`, and exits 0
    — the shortfall is information, not a refusal, until no candidate is
    left."""
    _install(
        monkeypatch,
        "causalab.neural.engines.pytorch_hooks",
        "PytorchHooksEngine",
        _hooks_stub(frozenset()),
    )
    _install(
        monkeypatch,
        "causalab.neural.engines.nnsight_nnterp",
        "NnterpEngine",
        _nnterp_stub(FULL),
    )
    assert main(_argv(SCAN, artifacts_root, "--engine", "auto")) == 0
    out = capsys.readouterr().out
    assert "engine    pytorch_hooks: capability_shortfall" in out
    assert "engine    nnterp: serves" in out


def test_b_the_api_reports_per_candidate_and_refuses_only_when_none_serves(
    env: ResolutionEnv,
) -> None:
    """`dry_run(engines=…)`: one `EngineReport` per candidate with the
    shortfall as a `capability_shortfall` `Diagnostic`; `refusals` is empty
    while one candidate serves and carries routing's rule-13 text once none
    does."""
    compiled = _compile(base_doc(), env)
    bare, full = _Stub("bare", frozenset()), _Stub("full", FULL)
    mixed = dry_run(compiled, env, engines=[bare, full])
    assert [(e.name, e.serves) for e in mixed.engines] == [
        ("bare", False),
        ("full", True),
    ]
    shortfall = mixed.engines[0].shortfall
    assert shortfall is not None and shortfall.kind == "capability_shortfall"
    assert "paired_forward" in shortfall.message and mixed.engines[0].lacks
    assert mixed.engines[1].shortfall is None and mixed.engines[1].lacks == ()
    assert mixed.ok and "engines" not in mixed.undecided_topics
    alone = dry_run(compiled, env, engines=[bare])
    assert not alone.ok and len(alone.refusals) == 1
    assert alone.refusals[0].rule == 13 and "bare lacks" in alone.refusals[0].message


def test_b_the_data_pass_is_reported_not_raised(
    env: ResolutionEnv, artifacts_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--data` runs `validate --data`'s pass: a missing column is a
    `refusals` entry (exit 1), and the valid document passes it (exit 0)."""
    assert main(_argv(INTERCHANGE, artifacts_root, "--data")) == 0
    capsys.readouterr()
    raw = base_doc()
    raw["method"]["metrics"]["ld"]["a"] = "not_a_column"
    report = dry_run(_compile(raw, env), env, check_data=True)
    assert not report.ok and "not_a_column" in report.refusals[0].message
    pair = next(u for u in report.undecided if u.topic == "pair_validity")
    assert "were checked" in pair.detail


# --------------------------------------------------------------------------- #
# T-c — never a loader, never an output
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", [*sorted(REFUSALS), "valid_twin"])
def test_c_the_model_loader_is_never_entered(
    name: str, env: ResolutionEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legality suite's seam, re-used: the reference engine's
    `load_model` — the name the engine module bound at import, the one call
    that reaches `from_pretrained` — replaced by one that raises. The
    document-decided refusals (rules 4, 29, 31) re-raise from the compile;
    the three engine-decided ones (rule 30 — a training fact the reference
    engine cannot honour) compile, and the dry run **reports** the shortfall
    against the real engine at the authored field, `ok` false; the valid twin
    returns a report with the real engine serving. Nothing enters the loader
    and nothing is written. The engine is imported here, not at module
    scope: this is a `unit` file."""
    from causalab.neural.engines.pytorch_hooks import engine as hooks_engine

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("weights loaded")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    monkeypatch.chdir(tmp_path)
    engines = [hooks_engine.PytorchHooksEngine()]
    if name == "valid_twin":
        report = dry_run(in_order(base_doc()), env, engines=engines)
        assert report.ok and report.engines[0].serves
    elif name in ENGINE_DECIDED:
        build, rule = REFUSALS[name]
        report = dry_run(in_order(build()), env, engines=engines)
        assert not report.ok and not report.engines[0].serves
        assert report.refusals[0].rule == rule and report.refusals[0].path, name
        assert ENGINE_DECIDED[name] in report.engines[0].lacks
    else:
        build, rule = REFUSALS[name]
        with pytest.raises(ValidationError) as err:
            dry_run(in_order(build()), env, engines=engines)
        assert err.value.rule == rule, str(err.value)
    assert list(tmp_path.iterdir()) == [], "a dry run wrote a file"


def test_c_dry_run_takes_no_out_and_writes_nothing(
    artifacts_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """No `--out` (argparse refuses it), and a dry run from a fresh cwd
    leaves it empty — no `protocol.json`, no receipt, no saved file."""
    monkeypatch.chdir(tmp_path)
    assert main(_argv(SCAN, artifacts_root)) == 0
    capsys.readouterr()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(SystemExit) as exit_info:
        main(_argv(SCAN, artifacts_root, "--out", str(tmp_path / "x")))
    assert exit_info.value.code == 2


def test_c_a_workflow_document_is_refused_with_one_line(
    tmp_path: Path, artifacts_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dry run is per intervention specification; a workflow is refused
    with exit 1 and one line, not dispatched to the workflow verbs."""
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"version": "1", "output_dir": "o", "steps": {}}))
    code = main(
        [
            "dry-run",
            str(wf),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts_root),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "workflow has no dry run of its own" in err


# --------------------------------------------------------------------------- #
# the censuses: the report's two closed vocabularies against the spec
# --------------------------------------------------------------------------- #


def test_the_site_statuses_are_the_spec_table() -> None:
    """§9's site-status table and `SITE_STATUSES` agree; the record refuses
    a status outside the vocabulary."""
    tables = _tables(_section("### 9.1 One compiler, four doors"))
    statuses = next(t for t in tables if t[0][0].strip("` ") == "status")
    assert set(_first_column(statuses)) == set(SITE_STATUSES)
    assert len(SITE_STATUSES) == len(set(SITE_STATUSES))
    gpt2 = get_model_info("gpt2")
    site = site_report("x", "block_output", gpt2, layers=[3])
    import dataclasses

    with pytest.raises(AssertionError):
        dataclasses.replace(site, status="maybe")  # type: ignore[arg-type]


def test_the_undecided_topics_are_the_spec_table() -> None:
    """§9's undecided-topic table and `UNDECIDED_TOPICS` agree; the record
    refuses a topic outside the vocabulary."""
    from causalab.protocol.dry_run import Undecided

    tables = _tables(_section("### 9.1 One compiler, four doors"))
    topics = next(t for t in tables if t[0][0].strip("` ") == "topic")
    assert set(_first_column(topics)) == set(UNDECIDED_TOPICS)
    assert len(UNDECIDED_TOPICS) == len(set(UNDECIDED_TOPICS))
    with pytest.raises(AssertionError):
        Undecided("weather", "x")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# fail-closed: every shipped document dry-runs to exit 0
# --------------------------------------------------------------------------- #


def _needs_run_tree(document: Path) -> bool:
    """A document whose featurizer or param `file_path` names a step's output
    (not a repo path) loads only inside its workflow's run tree — the demos
    suite's rule for the same documents."""
    raw = json.loads(document.read_text())
    method = raw.get("method") or {}
    for section in ("featurizers", "params"):
        for entry in (method.get(section) or {}).values():
            path = entry.get("file_path") if isinstance(entry, dict) else None
            if isinstance(path, str) and not (REPO / path).exists():
                return True
    return False


#: `minimal_cpu.json` names a tiny fixture model whose registry entry appears
#: only when a test loads it (`registry.py`: tests register their tiny-random
#: models), so offline and in isolation the pure verbs refuse it `[V4]` —
#: the order-dependent quirk `test_protocol_presets` already shows. A finding
#: about the fixture's registration, not about the dry run; it is asserted
#: below rather than swept here.
TINY_KEYED = {"minimal_cpu.json"}

CORPUS = sorted(p.name for p in CORPUS_DIR.glob("*_im.json"))
STANDALONE_PRESETS = sorted(
    p.name
    for p in PRESETS.glob("*.json")
    if p.name not in set(RUN_TREE_ONLY) | TINY_KEYED
)
DEMO_DOCUMENTS = sorted(
    str(p.relative_to(REPO))
    for p in DEMOS.glob("*/protocols/*.json")
    if not _needs_run_tree(p)
)


def _ok(report: DryRunReport, name: str) -> None:
    assert report.ok, (name, [r.message for r in report.refusals])
    assert all(site.status != "refused" for site in report.sites), name
    assert report.points.n >= 1 and report.forwards.campaign >= 1, name
    assert report.undecided_topics[-1] == "controls", name


@pytest.mark.parametrize("name", CORPUS)
def test_every_corpus_document_dry_runs_to_exit_0(
    name: str, env: ResolutionEnv, artifacts_root: Path, capsys
) -> None:
    """The corpus through the API and the CLI: no refusal, no refused site."""
    _ok(dry_run(CORPUS_DIR / name, env), name)
    assert main(_argv(name, artifacts_root)) == 0, name
    capsys.readouterr()


@pytest.mark.parametrize("name", STANDALONE_PRESETS)
def test_every_shipped_preset_dry_runs_to_exit_0(name: str, env: ResolutionEnv) -> None:
    """The presets a user copies from, against the fixture environment."""
    _ok(dry_run(PRESETS / name, env), name)


@pytest.mark.parametrize("name", DEMO_DOCUMENTS)
def test_every_demo_document_dry_runs_to_exit_0(name: str) -> None:
    """The demo documents, against their own tables (a demo carries its data
    under its `data/`; artifacts resolve against the repo root, as the demos
    suite has it)."""
    document = REPO / name
    env = ResolutionEnv(
        datasets=FileDatasets(root=document.parents[1] / "data"),
        artifacts=FileArtifacts(root=REPO),
    )
    _ok(dry_run(document, env), name)


def test_the_tiny_keyed_preset_is_the_registry_refusal_offline(
    env: ResolutionEnv,
) -> None:
    """Why `minimal_cpu.json` is not in the sweep: its key is registered by a
    load, so a torch-free dry run in isolation refuses it at `model.key`
    (`[V4]`) — honestly, the same answer `validate` gives. Skipped when an
    earlier test in the session has loaded the fixture."""
    raw = json.loads((PRESETS / "minimal_cpu.json").read_text())
    key = raw["model"]["key"]
    try:
        get_model_info(key)
    except ValidationError:
        pass
    else:
        pytest.skip(f"{key!r} was registered by an earlier load in this session")
    with pytest.raises(ValidationError) as err:
        dry_run(PRESETS / "minimal_cpu.json", env)
    assert err.value.rule == 4 and err.value.path == "model.key"
