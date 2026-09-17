"""Does a fitted rotation survive being saved and reloaded, exactly?

A small drift between a `subspace` fit and the same fit reloaded from its
artifact would put every replay control built on "reload the fit and score it
again" on sand — so the property is settled by executing it rather than by
reading the code, which is what this file is.

There are two round-trips, and they are not the same object:

**The artifact round-trip.** `slot_params()`
returns the *materialized* Q of an orthogonal parametrization; that tensor is
detached by `TensorFile.add`, written by `write_outputs`, and read back through
`load_tensors` and `build_stack` as a `LoadedLinear`. The round-trip here is
**the stack's own writer and reader**, not a `safetensors` stand-in: a dtype
hop is something the code around `safetensors` would introduce, never
`safetensors` itself, so a test that bypasses that code tests the one link
that was never suspect. The claim under test is that the matrix is
bit-identical and that the logits it produces are too — not merely close.

**The in-training restore path**, which is where a small drift would more
plausibly live.
`train.early_stop` snapshots the best-scoring stages and restores them at the
end, and a `Subspace`'s `weight` is *computed* from
`parametrizations.weight.original`. A snapshot taken over `slot_params()` would
save the materialized Q and restore nothing — the weight would silently stay
wherever the last update left it, which is exactly a small drift between "the
fit that was selected" and "the fit that was saved". `snapshot` uses
`state_dict()` for that reason, and
`test_a_snapshot_captures_the_parametrizations_own_parameter` is the guard that
keeps it that way — the bug is re-introducible by a one-word edit.

**What a drift of this kind would look like, and it is not fp32.**

    2^-14 = 6.1035e-5

``2^-14`` is the half-ulp of a **10-explicit-mantissa-bit** format for values in
``[0.125, 0.25)`` — float16, and also TF32, the default matmul path on Ampere
and later. A max-abs-diff sitting just *under* that ceiling is the signature of
"the largest entries in that band were rounded to 10 mantissa bits", not of a
logic bug. (bf16 has 8 mantissa bits, so its ceiling in the same band is
2.44e-4.)

Two mechanisms fit, and the conclusion reads differently under each: a dtype hop
in the real save/load path, or a **recomputation** under TF32 rather than a copy
— which would mean the number never came from the artifact at all.

So the artifact round-trip here is parametrized over **fp32, bf16 and fp16**,
through the real writer and reader. That costs nothing on CPU, and it turns
"exact in fp32" into "exact in whatever dtype a rotation is stored in". Which
dtype that is deserves saying precisely: it is the `Subspace` parameter's,
**fp32 on every engine today** — `Subspace.__init__` draws it with
`torch.randn`, `build_stack` moves a stage's *device* and not its dtype, and
nothing in the train loop casts it — so `model.dtype` never reaches it, and
``weekdays_das_sweep.json``'s bf16 fits store fp32 rotations too. The
parametrization is what keeps that from being a load-bearing coincidence: a
stage built in any of the three dtypes gets its bytes back.

Result: **the artifact round-trip is exact in all three dtypes**, and the
in-training restore path is exact in fp32.

What that does and does not settle. It rules out the *artifact* as the source at
every precision this stack stores, which is the half the replay controls rest
on. It does **not** rule out a reduced-precision *recomputation* — mechanism 2 —
because that needs a GPU: TF32 is a matmul path, not a storage format, and no
CPU test can exercise it. That check belongs on an accelerator, and the
arithmetic above is why it is worth running rather than a formality.
"""

from __future__ import annotations

import functools
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks.loading import _DTYPES
from causalab.neural.shared.featurizers import LoadedLinear, Subspace, build_stack
from causalab.neural.shared.outputs import TensorFile, write_outputs
from causalab.neural.shared.services import load_tensors
from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.resolve import (
    FileArtifacts,
    ResolutionEnv,
    build_artifact_identity,
    read_safetensors_metadata,
)
from causalab.protocol.schema import FeaturizerSpec

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES, fixture_input_overrides
from tests.tables import frame as table_frame

# One tier per test, declared per test. `docs/TESTS.md`: "every test belongs to
# exactly one tier" — and a `pytestmark` at module level *adds to* a marker on a
# test rather than being replaced by it, so a module-level `unit` plus the
# `smoke` marker on the end-to-end run put that test in two tiers at once.
# `tests/conftest.py` only catches *zero* markers, so nothing flagged it.
unit = pytest.mark.unit

#: The seed every draw in this file uses. One constant, because the non-vacuity
#: guard compares the fitted rotation against a *fresh* one at the same seed —
#: two independent literals would let a change to the document's seed silently
#: turn that guard into a comparison against an unrelated draw, which makes it
#: pass more easily.
SEED = 0

#: The half-ulp of a 10-mantissa-bit format in ``[0.125, 0.25)`` — the size of
#: drift a dtype hop would produce. Named so the assertions can say what they
#: would have caught.
HALF_ULP_DRIFT = 2.0**-14

BASES = [
    "the quick brown fox jumps over",
    "a slow green turtle sleeps deeply",
    "every shiny robot dances tonight",
    "some ancient rivers flow backwards",
]
COUNTERFACTUALS = [
    "cold silver mountains echo loudly",
    "bright yellow parrots sing early",
    "seven broken clocks tick wrongly",
    "warm quiet valleys rest gently",
]
ANSWERS = [" one", " two", " three", " four"]

K = 4


def das_doc(
    *, epochs: int = 2, early_stop_mode: str | None = None, seed: int = SEED
) -> dict:
    """A minimal DAS fit — `das.json`'s shape at fixture scale.

    ``early_stop_mode`` is the ``train.early_stop.mode`` to select the fit by
    (``"min"`` or ``"max"`` on ``ce``); ``None`` fits without selection.
    """
    doc: dict = {
        "header": {"protocol_version": "3"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "sites": {
                "tgt": {"component": "block_output", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "featurizers": {
                "rot": {"kind": "subspace", "k": K, "parametrization": "cayley"}
            },
            "reads": {
                "v_cf": {
                    "site": "tgt",
                    "pos": {"index": -1},
                    "model": "original",
                    "input": "counterfactual",
                    "featurizer": "rot",
                },
                "logits": {
                    "site": "lm_head",
                    "pos": {"index": -1},
                    "model": "patched",
                    "input": "base",
                },
            },
            "writes": {
                "patch": {
                    "site": "tgt",
                    "pos": {"index": -1},
                    "featurizer": "rot",
                    "do": {"swap": "v_cf"},
                }
            },
            "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
            "metrics": {
                "ce": {
                    "kind": "cross_entropy",
                    "of": "logits",
                    "target": "label",
                    "token_form": "space_prefixed",
                }
            },
            "train": {
                "objective": [[1.0, "ce"]],
                "params": ["rot"],
                "optimizer": {"name": "adamw", "lr": 1e-2, "weight_decay": 0.0},
                "steps": {"epochs": epochs},
                "batch": {"pairs": 2},
                "seed": seed,
            },
            "save": [
                {
                    "value": "ce",
                    "model": "patched",
                    "input": "base",
                    "file_path": "ce.json",
                },
                {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"},
            ],
        },
    }
    if early_stop_mode is not None:
        # `split: "inline"` scores the same inline rows the fit trains on. That
        # is not a held-out number and is not used as one: what this exercises
        # is the *selection* mechanism — a snapshot taken and restored — and for
        # that the split only has to resolve.
        doc["method"]["train"]["eval"] = {
            "every": {"epochs": 1},
            "split": "inline",
            "metrics": ["ce"],
        }
        doc["method"]["train"]["early_stop"] = {
            "metric": "ce",
            "patience": 5,
            "mode": early_stop_mode,
        }
    return doc


class _InlineDatasets:
    """The rows `executor_for` already holds, served under one split name.

    `train.eval` resolves its split through the dataset resolver, so an
    early-stopping fit needs one. Serving the *training* rows is deliberate:
    this test is about the selection mechanism — a snapshot taken and restored
    — not about generalization, and a held-out score would be no more
    informative on a random-weight model.
    """

    def digest(self, ref: str) -> str:
        return "0" * 64

    def columns(self, ref: str) -> tuple[str, ...]:
        return ("input", "counterfactual_inputs", "label")

    def rows(self, ref: str) -> list[dict[str, object]]:
        return [
            {
                "input": base,
                "counterfactual_inputs": [cf],
                "label": answer,
                "base_answer": answer,
                "cf_answer": answer,
            }
            for base, cf, answer in zip(BASES, COUNTERFACTUALS, ANSWERS)
        ]


def _request(artifacts_root: Path | None) -> ExecutionRequest:
    """The request the engine hands its services; ``artifacts_root`` is what
    `load_tensors` resolves a featurizer's ``file_path`` against."""
    return ExecutionRequest(
        points=(),
        canonical=(),
        digests=(),
        coords=(),
        document_digest="0" * 64,
        env=ResolutionEnv(
            datasets=_InlineDatasets(),  # type: ignore[arg-type]
            artifacts=FileArtifacts(artifacts_root) if artifacts_root else None,  # type: ignore[arg-type]
        ),
        output_dir=artifacts_root,  # type: ignore[arg-type]
    )


def _fit(doc_raw: dict):
    """Run the train loop and hand back the fitted stages."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc_raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        grad_enabled=False,
    )
    # the whole outcome, not just `.stages`: `eval_score.selected` is what says
    # the early-stop restore branch ran at all, and a test that cannot see that
    # cannot tell a restore from a no-op
    return run_training(executor.doc, executor, _request(None))


@pytest.fixture(scope="module")
def fitted() -> Subspace:
    stage = _fit(das_doc()).stages["rot"]
    assert isinstance(stage, Subspace)
    return stage


# --------------------------------------------------------------------------- #
# non-vacuity: a rotation that never moved round-trips trivially
# --------------------------------------------------------------------------- #


@unit
def test_the_fit_actually_moved_the_rotation(fitted: Subspace) -> None:
    """Without this, every assertion below could pass on an untrained basis.

    The comparison is against a fresh stage at the *same* seed, which is what
    the fit started from — so what it measures is the updates, not the draw.
    """
    initial = Subspace(fitted.weight.shape[0], K, "cayley", seed=SEED).weight
    moved = (fitted.weight - initial).abs().max().item()
    assert moved > HALF_ULP_DRIFT * 10, (
        f"the fit moved the basis by only {moved:.3e}, which is not enough "
        f"larger than the {HALF_ULP_DRIFT:.3e} drift under test for a "
        "round-trip assertion to mean anything"
    )


# --------------------------------------------------------------------------- #
# round-trip 1: the artifact, through the stack's own writer and reader
# --------------------------------------------------------------------------- #

#: The bundle a saved `rot` lands in, relative to the run tree — the
#: `file_path` both the fit's `save` entry and the apply's `featurizers.rot`
#: name.
ROT_FILE = "rot.safetensors"

#: The engine's own spelling of each storage dtype, inverted: what an
#: ArtifactIdentity stamps as ``dtype`` for a tensor of that dtype.
DTYPE_NAMES = {torch_dtype: name for name, torch_dtype in _DTYPES.items()}

#: Every dtype a rotation could be stored in. fp32 is what `Subspace` is built
#: in on every engine (module docstring); fp16 is the format whose half-ulp a
#: dtype-hop drift would match; bf16 is `weekdays_das_sweep.json`'s *model* dtype,
#: which does not reach the rotation today but is the first place it would land
#: if a stage were ever built in the model's dtype. Testing only fp32 would
#: test the one case least likely to show a drift.
STORED_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _save_through_the_stack(weight: torch.Tensor, out_dir: Path) -> Path:
    """The trained-bundle branch of `execution._execute_point`, as written
    there: `TensorFile.add` per slot with the featurizer's identity,
    `record_common`, then `write_outputs` into the run tree.

    The one-entry dict stands where ``stage.slot_params()`` does in that loop —
    a `subspace` stage's slots are exactly ``{"weight": Q}`` — so a rotation
    can be handed in at any dtype without building a `Subspace` in it.
    """
    identity = build_artifact_identity(
        produced_by="0" * 64,
        model_key=TINY_LLAMA,
        model_revision="main",
        model_dtype="fp32",
        k=K,
        parametrization="cayley",
        dtype=DTYPE_NAMES[weight.dtype],
        trained_on="inline",
    )
    bundle_file = TensorFile()
    for slot, param in {"weight": weight}.items():
        bundle_file.add(slot, param.detach(), {}, label_entry="rot", identity=identity)
    bundle_file.record_common(identity)
    written = write_outputs(
        out_dir,
        {ROT_FILE: bundle_file},
        {},
        identity_base={
            "produced_by": "0" * 64,
            "model_key": TINY_LLAMA,
            "model_revision": "main",
            "model_dtype": "fp32",
            "engine": "pytorch_hooks",
        },
    )
    return written[ROT_FILE]


def _load_through_the_stack(artifacts_root: Path, width: int) -> LoadedLinear:
    """What an apply document's ``featurizers.rot`` with a ``file_path``
    becomes: `build_stack` → `_build_stage` → `load_tensors` → `LoadedLinear`,
    with the entry's stamped ``k`` and ``parametrization`` checked against the
    spec on the way, exactly as at a run."""
    spec = FeaturizerSpec(
        kind="subspace", k=K, parametrization="cayley", file_path=ROT_FILE
    )
    stack = build_stack(
        "rot",
        {"rot": spec},
        width=width,
        load_tensors=functools.partial(load_tensors, _request(artifacts_root)),
        stage_cache={},
    )
    (stage,) = stack.stages
    assert isinstance(stage, LoadedLinear)
    return stage


def _round_trip(weight: torch.Tensor, tmp_path: Path) -> torch.Tensor:
    """Save through the stack's writer, read through its reader, and hand back
    the matrix the apply path would compute with."""
    _save_through_the_stack(weight, tmp_path)
    return _load_through_the_stack(tmp_path, weight.shape[0]).weight


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    """The tensor's bytes, for any dtype.

    ``.numpy()`` cannot represent bf16 at all, so the byte comparison — which is
    the claim as stated — goes through a uint8 view.
    """
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


@unit
@pytest.mark.parametrize("dtype", STORED_DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_the_saved_matrix_comes_back_bit_identical(
    fitted: Subspace, tmp_path: Path, dtype: torch.dtype
) -> None:
    """The round-trip claim, at the tensor level: matrix bytes, not closeness.

    Parametrized over the dtypes a rotation can be stored in, because a
    dtype-hop drift is a 10-mantissa-bit half-ulp and an fp32-only test is
    structurally blind to it. The cast happens *before* the save, so what is
    under test is the storage round-trip and not the cast.

    The dtype is checked twice, on the file and on the loaded stage, so a
    failure names its half: a hop on the way to disk lives in `TensorFile.add`
    (`outputs.py`), a hop on the way back in `_build_stage` (`featurizers.py`).
    """
    weight = fitted.weight.detach().to(dtype)
    file = _save_through_the_stack(weight, tmp_path)
    on_disk = load_file(str(file))
    assert {tensor.dtype for tensor in on_disk.values()} == {dtype}, (
        f"the writer stored a {dtype} rotation as "
        f"{sorted(str(t.dtype) for t in on_disk.values())} — a dtype hop in "
        "TensorFile.add, which is mechanism 1 for a round-trip drift"
    )
    reloaded = _load_through_the_stack(tmp_path, weight.shape[0]).weight
    assert reloaded.dtype == dtype, (
        f"the reader built a {reloaded.dtype} stage from a {dtype} file — a "
        "dtype hop in _build_stage"
    )
    assert reloaded.shape == weight.shape
    torch.testing.assert_close(reloaded, weight, atol=0.0, rtol=0.0)
    # and the bytes themselves, which is the claim as stated
    assert _raw_bytes(reloaded) == _raw_bytes(weight)


@unit
@pytest.mark.parametrize("dtype", STORED_DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_the_round_trip_does_not_drift_in_any_stored_dtype(
    fitted: Subspace, tmp_path: Path, dtype: torch.dtype
) -> None:
    """The claim under test, at every stored precision.

    A reduced-precision *cast* moves the matrix — that is arithmetic, and not
    the property. The property is a difference between a fit and **the same
    fit reloaded**, so the comparison is cast-then-save against
    cast-then-save-then-load, and it has to be zero in every dtype.
    """
    weight = fitted.weight.detach().to(dtype)
    delta = (
        (_round_trip(weight, tmp_path).to(torch.float32) - weight.to(torch.float32))
        .abs()
        .max()
        .item()
    )
    assert delta == 0.0, (
        f"the artifact round-trip drifted by {delta:.3e} in {dtype} — the "
        f"half-ulp drift of {HALF_ULP_DRIFT:.3e} a dtype hop produces would show here"
    )


@unit
def test_a_reloaded_rotation_featurizes_identically(
    fitted: Subspace, tmp_path: Path
) -> None:
    """The stage the apply path actually builds is a `LoadedLinear`, not a
    `Subspace`, so the round-trip that matters crosses a class boundary — and
    the `LoadedLinear` here is the one `build_stack` built, not one constructed
    by hand. Both halves of the featurizer contract are compared: an exact
    `featurize` with a drifting `inverse` would still corrupt a swap."""
    _save_through_the_stack(fitted.weight.detach(), tmp_path)
    loaded = _load_through_the_stack(tmp_path, fitted.weight.shape[0])
    x = torch.randn(
        8, fitted.weight.shape[0], generator=torch.Generator().manual_seed(3)
    )
    want_f, want_err = fitted.featurize(x)
    got_f, got_err = loaded.featurize(x)
    torch.testing.assert_close(got_f, want_f, atol=0.0, rtol=0.0)
    assert want_err is not None and got_err is not None
    torch.testing.assert_close(got_err, want_err, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        loaded.inverse(got_f, got_err),
        fitted.inverse(want_f, want_err),
        atol=0.0,
        rtol=0.0,
    )


# --------------------------------------------------------------------------- #
# round-trip 2: the in-training restore path
# --------------------------------------------------------------------------- #


@unit
def test_a_snapshot_captures_the_parametrizations_own_parameter(
    fitted: Subspace,
) -> None:
    """The regression guard for the snapshot path.

    A `Subspace`'s `weight` is computed by
    `torch.nn.utils.parametrizations.orthogonal`, so the tensor the optimizer
    steps is `parametrizations.weight.original`. Snapshotting `slot_params()`
    instead would capture the materialized Q and restore nothing — a silent
    drift between the fit `early_stop` selected and the fit that got saved.
    This asserts the state dict really does carry the underlying parameter, so
    the reason `snapshot` uses `state_dict()` is checked rather than
    commented.
    """
    from causalab.neural.shared.training.diagnostics import snapshot

    keys = set(snapshot({"rot": fitted})["rot"])
    assert any(
        "parametrizations" in key and key.endswith("original") for key in keys
    ), (
        f"a subspace snapshot carries {sorted(keys)} — none of which is the "
        "parametrization's own parameter, so restoring it would restore nothing"
    )


@unit
def test_restoring_a_snapshot_reproduces_the_materialized_rotation(
    fitted: Subspace,
) -> None:
    """Snapshot → perturb → restore, and the *materialized* Q must come back
    bit-identical. Perturbing is what makes this a test: without it, `restore`
    could be a no-op and pass."""
    import copy

    from causalab.neural.shared.training.diagnostics import restore, snapshot

    # A **copy**: `fitted` is module-scoped, so perturbing it in place and
    # relying on `restore` — the function under test — to undo the damage means
    # that when `restore` regresses, the real signal arrives buried in
    # unrelated failures in whichever tests run after this one. The isolation
    # should be structural, not a consequence of file order.
    stage = copy.deepcopy(fitted)
    before = stage.weight.detach().clone()
    taken = snapshot({"rot": stage})
    with torch.no_grad():
        for param in stage.parameters():
            param.add_(0.1)
    assert (stage.weight - before).abs().max().item() > HALF_ULP_DRIFT * 10
    restore({"rot": stage}, taken)
    torch.testing.assert_close(stage.weight, before, atol=0.0, rtol=0.0)
    # and the shared fixture is untouched, which is the point of the copy
    torch.testing.assert_close(fitted.weight, before, atol=0.0, rtol=0.0)


#: The parametrization's own parameter — the tensor the optimizer steps, and
#: therefore the one a restore has to move.
ORIGINAL = "parametrizations.weight.original"


@unit
def test_an_early_stopping_fit_returns_the_weights_it_selected(monkeypatch) -> None:
    """The same claim through the real loop rather than the two helpers.

    The obvious assertion here — that the returned rotation is orthonormal —
    **cannot fail**, and that is worth saying because it looks like a check.
    Under `parametrizations.orthogonal` the weight is *recomputed* from
    `parametrizations.weight.original` on every access, and the Cayley transform
    of any matrix has orthonormal columns. So the Gram identity holds before
    training, after training, after a correct restore, and after a restore that
    wrote garbage — it is a property of the map, not of the fit.

    Comparing the returned stage against the snapshot the loop took is not
    enough either, on its own. `snapshot` runs only on improvement, so if the
    last eval is the best one the snapshot *is* the live state and `restore`
    has nothing to move — and with ``mode: "min"`` on the very loss being
    minimised, scored on the training rows, the last eval is the best one
    every time. That run distinguishes `restore` from ``pass`` exactly never.

    So this fit selects with ``mode: "max"`` on ``ce``: the selection mechanism
    does not know what a metric means, and asking it for the *worst* loss makes
    epoch 1 the selected fit and every later epoch a rejected one. ``patience:
    5`` outlasts the two stale evals, so the loop runs to the end and the
    restore has to roll back two epochs of updates. A spy on `restore`
    captures the state training actually ended on, and the guard asserts that
    state differs from the snapshot — that is what makes the final equality a
    claim about `restore` rather than about `snapshot`, and it is what fails
    if the fixture's dynamics ever drift back to best-is-last.
    """
    from causalab.neural.shared.training import loop as loop_module

    taken: list[dict[str, dict[str, torch.Tensor]]] = []
    real_snapshot = loop_module.snapshot

    def snapshot_spy(stages):
        captured = real_snapshot(stages)
        taken.append({name: dict(state) for name, state in captured.items()})
        return captured

    ended_on: list[dict[str, dict[str, torch.Tensor]]] = []
    real_restore = loop_module.restore

    def restore_spy(stages, snapshot):
        ended_on.append(
            {
                name: {k: v.detach().clone() for k, v in stage.state_dict().items()}
                for name, stage in stages.items()
            }
        )
        return real_restore(stages, snapshot)

    monkeypatch.setattr(loop_module, "snapshot", snapshot_spy)
    monkeypatch.setattr(loop_module, "restore", restore_spy)
    outcome = _fit(das_doc(epochs=3, early_stop_mode="max"))

    assert taken, "no snapshot was taken — the early-stop branch never ran"
    assert len(ended_on) == 1, "the loop restores exactly once, at the end"
    assert outcome.eval_score is not None
    assert outcome.eval_score.selected == "early_stop.best", (
        "the loop did not report selecting a snapshotted fit, so there is "
        "nothing for this test to be about"
    )

    best = taken[-1]["rot"]
    assert ORIGINAL in best, (
        f"the snapshot carries {sorted(best)} and not {ORIGINAL!r} — `snapshot` "
        "is no longer taking the state dict, so a restore would restore nothing"
    )
    last = ended_on[-1]["rot"][ORIGINAL]
    rolled_back = (last - best[ORIGINAL]).abs().max().item()
    assert rolled_back > HALF_ULP_DRIFT * 10, (
        f"training ended {rolled_back:.3e} from the snapshotted state — this "
        "run never exercised a rollback, so it would pass with restore as a "
        "no-op and says nothing about it"
    )

    rot = outcome.stages["rot"]
    assert isinstance(rot, Subspace)
    # the parametrization's own parameter, which is what a restore restores
    state = dict(rot.state_dict())
    for key, want in best.items():
        torch.testing.assert_close(state[key], want, atol=0.0, rtol=0.0)
    # ...and therefore the materialized rotation recomputed from it. Implied by
    # the line above for a deterministic map — kept because "the state dict
    # alone determines Q" is the thesis of this file, and this is where it is checked.
    restored = Subspace(rot.weight.shape[0], K, "cayley", seed=SEED)
    restored.load_state_dict(dict(best))
    torch.testing.assert_close(rot.weight, restored.weight, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# round-trip 3: logits, end to end — fit then apply on the fit's own split
# --------------------------------------------------------------------------- #


REPO = Path(__file__).resolve().parents[4]
METHODS = str(REPO / "causalab/configs/protocols")
#: The tiny-random realization every CPU smoke test runs at
#: (`tests/_helpers/tiny.py`). It is *not* a claim about the fit's dtype: the
#: rotation's storage dtype is the `Subspace` parameter's, fp32 on every engine
#: whatever `model.dtype` says (module docstring), and the smoke test checks the
#: written file against its own stamp rather than against this pin.
TINY = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}


@pytest.mark.smoke
def test_an_applied_rotation_reproduces_the_fits_logits_exactly(
    tmp_path: Path,
) -> None:
    """The "and logits" half, as the strongest available form: the
    apply document scores the split the fit reported on, so any difference in
    the reloaded matrix shows up as a difference in the metric.

    Plain equality is the assertion. A tolerance here would hide precisely the
    effect under test — a half-ulp drift on a logit-scale quantity is well
    inside any tolerance one would reach for casually, which is how such a
    drift goes unnoticed.

    The preset names the shipped weekdays table, whose answers tiny-random
    cannot spell as single tokens ([P2]), so the fit's dataset refs are
    retargeted onto the 4-row fixture the way every tiny-scale run of a shipped
    document is (`fixture_input_overrides`). Two things the fit step then
    inherits from `weekdays_das_sweep.json` and does not otherwise override,
    said here so nobody has to rediscover them: its `eval` — retargeted onto
    `weekdays/data#test` — and its `early_stop` are **live**. With `epochs: 1` there
    is exactly one eval, so the snapshot is the final state and the restore is
    the identity. Bumping `epochs` would make the restore real — and the
    comparison would still be sound, because `_execute_point` runs the metrics
    and the save *after* `run_training` returns, so `fit/iia.json` and
    `fit/rot.safetensors` always describe the same, selected, weights.
    """
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    # the preset's refs map 1:1 onto the fixture's train/test splits; the apply
    # below names the train split literally, so pin what the helper resolved to
    fixture_inputs = fixture_input_overrides(
        json.loads(Path(f"{METHODS}/weekdays_das_sweep.json").read_text())
    )
    assert fixture_inputs["data.base.dataset"] == "weekdays/data#train"
    assert fixture_inputs["data.counterfactual.dataset"] == "weekdays/data#train"
    assert fixture_inputs["train.eval.split"] == "weekdays/data#test"
    workflow = {
        "version": "1",
        "description": "fit a DAS rotation, then apply it on the fit's own split",
        "output_dir": "das",
        "steps": {
            "fit": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/weekdays_das_sweep.json",
                "set": {
                    **TINY,
                    **fixture_inputs,
                    "sites.target.layers": 0,
                    "featurizers.rot.k": 2,
                    "train.seed": 0,
                    "train.steps": {"epochs": 1},
                    "train.batch": {"pairs": 2},
                },
            },
            "apply": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/weekdays_das_apply.json",
                "set": {
                    **TINY,
                    "sites.target.layers": 0,
                    "featurizers.rot.k": 2,
                    "featurizers.rot.file_path": f"fit/{ROT_FILE}",
                    # the preset's `entry` names its own k=8 fit; this apply
                    # loads the k=2 fit above, so its selector has to say so —
                    # a fit with both axes pinned writes a single un-swept
                    # entry, which resolves by slot alone, and this keeps the
                    # document consistent rather than dependent on that
                    "featurizers.rot.entry": {"k": 2, "seed": 0},
                    # the fit's own split: the two numbers are then the same
                    # question asked twice, and must agree to the bit
                    "data.base.dataset": "weekdays/data#train",
                    "data.counterfactual.dataset": "weekdays/data#train",
                },
            },
        },
    }
    path = wf_dir / "wf.json"
    path.write_text(json.dumps(workflow, indent=2))
    out = tmp_path / "run"
    code = main(
        [
            "run",
            str(path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    run = out / "das"
    # The bytes on disk are in the dtype the file's own identity claims. This is
    # `TensorFile.add` pinned against an up- or downcast at whatever dtype a
    # run declares — it keeps working if the `fp32` pin above is ever lifted.
    rot_file = run / "fit" / ROT_FILE
    stamped = read_safetensors_metadata(rot_file)
    assert stamped is not None and "dtype" in stamped
    assert {t.dtype for t in load_file(str(rot_file)).values()} == {
        _DTYPES[str(stamped["dtype"])]
    }, (
        "the writer changed the fit's dtype on the way to disk — mechanism 1 "
        "for a round-trip drift"
    )
    fitted = table_frame(run / "fit/iia.json")
    applied = table_frame(run / "apply/iia.json")
    assert len(applied) == len(fitted) > 0
    assert list(applied["value"]) == list(fitted["value"])
