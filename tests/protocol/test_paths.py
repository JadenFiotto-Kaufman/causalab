"""Path blocks (spec §3.2): the compiler into the
existing nouns, and what it refuses.

The load-bearing test is the first one. The shipped ``path_patching.json`` is
twenty hand-written entries; re-expressed as one ``method.path_patching`` block
it must compile to the **same canonical bytes and the same digests** — the
literal reading of "a compiler into the existing nouns", and cheap because the
target is already in the repo. Its twin does the same for the corpus document,
against its pinned digest, so the claim is held to a number nothing here
computes. Around it: a one-element receiver set *is* the scalar document, two
blocks differing only in restoration policy have different point identities
with the policy legible off the compiled form, no shipped or corpus document
carries a block and every one lowers nothing, the stage
sits where the order says and the eighth output sits last, and every refusal
names ``method.path_patching.<field>`` with a parser code — no rule number,
because the lowered document's checklist already owns what a block cannot say.

Every test here fails without the change: without ``paths.py`` the imports
below fail at collection; without the ``paths`` stage a block document is
refused by the gate as an unknown section ``[P3]``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import warnings
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main, register_model_key
from causalab.protocol import compile as compiler
from causalab.protocol.code import import_closure
from causalab.protocol.compile import STAGES, CompiledProtocol, compile_protocol
from causalab.protocol.errors import ParseError, ProtocolError, ValidationError
from causalab.protocol.paths import (
    BLOCK,
    BLOCK_KEYS,
    POLICIES,
    describe_paths,
    expand_paths,
    has_path_block,
)
from causalab.protocol.plan import COMPONENT_RANK
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.run import write_run_record
from causalab.protocol.schema import METHOD_SECTIONS, parse_document

from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.protocol.test_protocol_presets import RUN_TREE_ONLY
from tests.workflow.test_closure_census import SHARED

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PRESETS = REPO / "causalab/configs/protocols"
SHIPPED = PRESETS / "path_patching.json"
CORPUS_03 = CORPUS_DIR / "03_path_patching_im.json"
PINS = json.loads((Path(__file__).parent / "corpus_digests.json").read_text())

#: The names the hand-written document uses, which the byte-identical compile
#: below makes the convention.
SHIPPED_NAMES = {
    "sites": ["sender", "receiver", "a10", "a11"],
    "reads": ["v_sender", "v_a10", "v_a11", "v_receiver"],
    "writes": ["swap_sender", "freeze_10", "freeze_11", "inject"],
    "intervened_models": ["patched", "final"],
}


def _compile(
    document: Path | dict[str, Any],
    env: ResolutionEnv,
    overrides: dict[str, Any] | None = None,
) -> CompiledProtocol:
    return compile_protocol(
        document, None, overrides, env.datasets, env.artifacts, None
    )


def _block_form(
    handwritten: Path | dict[str, Any],
    *,
    restoration: str = "attention_only",
    receivers: list[dict[str, Any]] | None = None,
    **block_fields: Any,
) -> dict[str, Any]:
    """A hand-written path-patching document re-expressed as a block: the
    sender and receiver sites the block takes over, the readout (``lm_head``,
    ``logits``), the metric and the manifest stay authored. The block goes
    first — the §1 order puts generated tables after it in any case."""
    doc = copy.deepcopy(
        json.loads(handwritten.read_text())
        if isinstance(handwritten, Path)
        else handwritten
    )
    method = doc["method"]
    block: dict[str, Any] = {
        "sender": method["sites"]["sender"],
        "source": "counterfactual",
        "receivers": receivers or [method["sites"]["receiver"]],
        "pos": -1,
        "restoration": restoration,
        **block_fields,
    }
    doc["method"] = {
        BLOCK: block,
        "sites": {"lm_head": method["sites"]["lm_head"]},
        "reads": {"logits": method["reads"]["logits"]},
        "metrics": method["metrics"],
        "save": method["save"],
    }
    return doc


def _refused(
    document: dict[str, Any],
    env: ResolutionEnv,
    code: str,
    pattern: str,
    *,
    at: str = "method.path_patching",
) -> ParseError:
    """A block refusal: the parser code, the message, and the block path."""
    with pytest.raises(ParseError) as err:
        _compile(document, env)
    assert err.value.code == code, str(err.value)
    assert re.search(pattern, str(err.value)), str(err.value)
    assert err.value.path is not None and err.value.path.startswith(at), err.value.path
    return err.value


# --------------------------------------------------------------------------- #
# the compiler into the existing nouns, literally
# --------------------------------------------------------------------------- #


def test_the_shipped_preset_from_a_block_compiles_byte_identically(env) -> None:
    """Without the lowering emitting the shipped names exactly (``a10``,
    ``v_a10``, ``freeze_10``, ``receiver``, ``inject``, ``patched``,
    ``final``) the canonical forms differ; without the freeze range being
    ``range(sender + 1, receiver)`` the write sets differ."""
    handwritten = _compile(SHIPPED, env)
    block = _compile(_block_form(SHIPPED), env)
    assert block.canonical == handwritten.canonical
    assert block.digests == handwritten.digests
    assert block.points.canonical == handwritten.points.canonical
    # the hand-written document lowers nothing; the block records what it did
    assert handwritten.lowered == {}
    assert block.lowered[BLOCK]["emitted"] == SHIPPED_NAMES
    # and the explicit tree is the hand-written one, table for table
    for section in ("sites", "reads", "writes", "intervened_models"):
        assert (
            block.points.explicit["method"][section]
            == json.loads(SHIPPED.read_text())["method"][section]
        ), section


def test_the_corpus_document_from_a_block_reproduces_its_pin(env) -> None:
    """The same claim against a number this file does not compute: the block form of
    corpus 03 has the document digest ``corpus_digests.json`` pins for the
    hand-written file. Renaming any emitted entry, or shifting the freeze
    range by one layer, moves it."""
    block = _compile(_block_form(CORPUS_03), env)
    assert block.digests.document == PINS["03_path_patching_im.json"]["document"]
    assert list(block.digests.points) == PINS["03_path_patching_im.json"]["points"]


def test_the_lowered_document_raises_no_order_warning(env) -> None:
    """Generated tables land where §1 puts them, so the lowered
    document does not trip rule 2 the authored one did not."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled = _compile(_block_form(SHIPPED), env)
    keys = list(compiled.points.explicit["method"])
    assert keys == [k for k in METHOD_SECTIONS if k in keys]


def test_the_block_leaves_an_authored_section_order_alone(env) -> None:
    """The lowering inserts relative to what is there; an authored order that
    already warns still warns, with the same text — the stage neither fixes
    nor hides it."""
    doc = _block_form(SHIPPED)
    method = doc["method"]
    doc["method"] = {k: method[k] for k in (BLOCK, "save", "sites", "reads", "metrics")}
    with pytest.warns(Warning, match="not in the recommended"):
        _compile(doc, env)


# --------------------------------------------------------------------------- #
# ordered receiver sets; one element is the scalar case
# --------------------------------------------------------------------------- #


def test_a_one_element_receiver_set_is_the_scalar_document(env) -> None:
    """``receivers: [x]`` lowers to ``receiver`` / ``v_receiver`` / ``inject``
    — the scalar document's own names — so the one-element set is the scalar
    case to the byte (the byte-identical comparison above is this document). Without the
    one-element spelling it would be ``receiver_0`` and the digests would
    differ from the hand-written file."""
    one = _compile(
        _block_form(SHIPPED, receivers=[{"component": "block_input", "layers": [12]}]),
        env,
    )
    assert one.lowered[BLOCK]["receivers"] == ["receiver"]
    assert one.canonical == _compile(SHIPPED, env).canonical


def test_several_receivers_are_one_intervened_model_in_authored_order(env) -> None:
    """An ordered set executes as one joint pass — every ``inject_i``
    in the one ``final`` model — and the order is data: it numbers the members
    (``receiver_0`` is the first authored), so the same two sites in the other
    order are a different document, and the derived record repeats the order
    so no reader has to parse names. Without joint emission (one model per
    receiver) ``final`` would carry one write."""
    first = {"component": "block_input", "layers": [12]}
    second = {"component": "attention_output", "layers": [12]}
    forward = _compile(_block_form(SHIPPED, receivers=[first, second]), env)
    backward = _compile(_block_form(SHIPPED, receivers=[second, first]), env)
    final = forward.canonical["method"]["intervened_models"]["final"]
    assert final["writes"] == ["inject_0", "inject_1"]
    assert set(forward.canonical["method"]["intervened_models"]) == {
        "patched",
        "final",
    }
    assert forward.lowered[BLOCK]["receivers"] == ["receiver_0", "receiver_1"]
    lowered = forward.points.explicit["method"]
    assert lowered["sites"]["receiver_0"] == first
    assert lowered["sites"]["receiver_1"] == second
    assert lowered["reads"]["v_receiver_1"] == {
        "site": "receiver_1",
        "pos": -1,
        "model": "patched",
        "input": "base",
    }
    # the order numbers the members: reversed, the names swap addresses, so
    # the canonical form (and the digest) differ while the address set is one
    reversed_sites = backward.points.explicit["method"]["sites"]
    assert reversed_sites["receiver_0"] == second
    assert reversed_sites["receiver_1"] == first
    assert backward.canonical != forward.canonical
    assert backward.digests.document != forward.digests.document
    assert backward.lowered[BLOCK]["authored"]["receivers"] == [second, first]
    assert backward.lowered[BLOCK]["receivers"] == ["receiver_0", "receiver_1"]


# --------------------------------------------------------------------------- #
# the restoration policy is part of the point identity, and legible
# --------------------------------------------------------------------------- #


def test_the_policy_is_in_the_point_identity_and_readable(env) -> None:
    """Two blocks differing only in ``restoration`` lower to different
    write sets, so their point digests differ (§7) and a comparison across
    them fails its ``produced_by`` binding; the policy itself reads back off
    the compiled form. Without the ``freeze_m*`` emission the two identities
    collapse."""
    attention = _compile(_block_form(SHIPPED, restoration="attention_only"), env)
    both = _compile(_block_form(SHIPPED, restoration="attention_and_mlp"), env)
    assert attention.digests.points[0] != both.digests.points[0]
    assert attention.digests.document != both.digests.document
    assert attention.lowered[BLOCK]["restoration"] == "attention_only"
    assert both.lowered[BLOCK]["restoration"] == "attention_and_mlp"
    a_writes = set(attention.canonical["method"]["writes"])
    b_writes = set(both.canonical["method"]["writes"])
    assert a_writes == {"swap_sender", "freeze_10", "freeze_11", "inject"}
    assert b_writes == a_writes | {"freeze_m9", "freeze_m10", "freeze_m11"}


def test_attention_and_mlp_restores_the_senders_own_mlp_and_orders_by_rank(env) -> None:
    """The stated choice (§3.2): ``mlp_output`` is restored at the sender's
    own layer as well, because the sender's block's MLP is downstream of an
    attention sender. And the restorer boundary is ordered by
    ``(layer, COMPONENT_RANK)`` — attention before MLP within a layer."""
    both = _compile(_block_form(SHIPPED, restoration="attention_and_mlp"), env)
    record = both.lowered[BLOCK]
    assert record["restored"] == {
        "attention_output": [10, 11],
        "mlp_output": [9, 10, 11],
    }
    assert record["restorers"] == [
        [9, "mlp_output", "m9"],
        [10, "attention_output", "a10"],
        [10, "mlp_output", "m10"],
        [11, "attention_output", "a11"],
        [11, "mlp_output", "m11"],
    ]
    ranks = [
        (layer, COMPONENT_RANK[component])
        for layer, component, _ in record["restorers"]
    ]
    assert ranks == sorted(ranks)
    assert COMPONENT_RANK["attention_output"] < COMPONENT_RANK["mlp_output"]
    patched = both.canonical["method"]["intervened_models"]["patched"]["writes"]
    assert set(patched) == {
        "swap_sender",
        "freeze_10",
        "freeze_11",
        "freeze_m9",
        "freeze_m10",
        "freeze_m11",
    }
    assert both.points.explicit["method"]["sites"]["m9"] == {
        "component": "mlp_output",
        "layers": [9],
    }


def test_adjacent_layers_restore_nothing_under_attention_only(env) -> None:
    """A sender one layer below its receiver has no layer between them:
    ``attention_only`` freezes nothing and the harvest model is the sender
    swap alone; ``attention_and_mlp`` still holds the sender's own MLP."""
    doc = _block_form(SHIPPED, receivers=[{"component": "block_input", "layers": [10]}])
    none = _compile(doc, env)
    assert none.lowered[BLOCK]["restorers"] == []
    assert none.canonical["method"]["intervened_models"]["patched"]["writes"] == [
        "swap_sender"
    ]
    doc["method"][BLOCK]["restoration"] = "attention_and_mlp"
    mlp = _compile(doc, env)
    assert mlp.lowered[BLOCK]["restorers"] == [[9, "mlp_output", "m9"]]


# --------------------------------------------------------------------------- #
# the legitimate campaign: nothing shipped carries a block
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    sorted(p for p in PRESETS.glob("*.json") if p.name not in RUN_TREE_ONLY)
    + sorted(CORPUS_DIR.glob("*_im.json")),
    ids=lambda p: p.name,
)
def test_no_shipped_or_corpus_document_lowers_anything(path: Path, env) -> None:
    """The block is optional and adds no field to any existing document —
    every shipped preset and corpus file compiles with ``lowered == {}``, and
    the hand-written path-patching documents parse as they always did."""
    raw = json.loads(path.read_text())
    assert not has_path_block(raw)
    assert BLOCK not in raw["method"]
    register_model_key(raw)  # the CLI's step: a tiny-fixture key joins the registry
    compiled = _compile(path, env)
    assert compiled.lowered == {}
    assert expand_paths(raw) == raw
    assert describe_paths(raw, raw) == {}


def test_the_hand_written_document_still_parses_unchanged() -> None:
    parse_document(json.loads(SHIPPED.read_text()))
    parse_document(json.loads(CORPUS_03.read_text()))


# --------------------------------------------------------------------------- #
# the stage, the output, the closure
# --------------------------------------------------------------------------- #


def test_the_stage_sits_between_families_and_gate() -> None:
    """After families, so a wrapper inside the block is refused rather than
    expanded; before the gate, so the parser never sees the block."""
    assert STAGES.index("families") < STAGES.index("paths") < STAGES.index("gate")
    assert compiler._STAGE["paths"] is compiler._paths  # pyright: ignore[reportPrivateUsage]


def test_lowered_is_the_eighth_and_last_output() -> None:
    names = [field.name for field in dataclasses.fields(CompiledProtocol)]
    assert len(names) == 8
    assert names[-1] == "lowered"


def test_the_lowering_module_is_outside_the_hashed_closure() -> None:
    """The design's premise: ``paths.py`` is reached from ``compile.py``
    alone, so no SHARED member imports it and no workflow pin moves.
    ``test_closure_census.py`` holds the frozen table; this is the direct
    statement."""
    assert "causalab/protocol/paths.py" not in SHARED
    closure = import_closure(REPO / "causalab/io/step_io.py", root=REPO)
    assert "causalab/protocol/paths.py" not in closure
    importers = [
        path.relative_to(REPO).as_posix()
        for path in (REPO / "causalab").rglob("*.py")
        if re.search(
            r"^from causalab\.protocol\.paths import|^import causalab\.protocol\.paths",
            path.read_text(),
            re.M,
        )
    ]
    assert importers == ["causalab/protocol/compile.py"]


def test_the_public_vocabulary_is_closed() -> None:
    assert POLICIES == ("attention_only", "attention_and_mlp")
    assert BLOCK == "path_patching"
    assert BLOCK not in METHOD_SECTIONS  # not a section: never in the canonical form
    assert set(BLOCK_KEYS) == {
        "sender",
        "source",
        "receivers",
        "pos",
        "restoration",
        "harvest",
        "inject",
    }


# --------------------------------------------------------------------------- #
# the compiled form is saved, and explained
# --------------------------------------------------------------------------- #


def test_the_receipt_carries_the_derived_block_bound_by_reference(
    env, tmp_path
) -> None:
    """The run receipt writes the lowering as ``derived``, beside the
    canonical form whose keys it names and the digests over that form. A
    document that lowered nothing writes no ``derived`` key at all."""
    compiled = _compile(_block_form(SHIPPED, restoration="attention_and_mlp"), env)
    record = json.loads(
        write_run_record(compiled, tmp_path / "a", range(1)).read_text()
    )
    derived = record["derived"][BLOCK]
    assert derived["restoration"] == "attention_and_mlp"
    assert (
        derived["authored"]
        == _block_form(SHIPPED, restoration="attention_and_mlp")["method"][BLOCK]
    )
    for section, names in derived["emitted"].items():
        assert set(names) <= set(record["canonical"]["method"][section]), section
    assert record["document_digest"] == compiled.digests.document
    plain = json.loads(
        write_run_record(_compile(SHIPPED, env), tmp_path / "b", range(1)).read_text()
    )
    assert "derived" not in plain


def test_explain_prints_the_receivers_and_the_restorer_boundary(
    capsys, artifacts_root, tmp_path
) -> None:
    doc = _block_form(CORPUS_03, restoration="attention_and_mlp")
    path = tmp_path / "path_block.json"
    path.write_text(json.dumps(doc))
    argv = [
        "explain",
        str(path),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "path      attention_and_mlp: sender -> receiver" in out
    assert (
        "restorers m9=mlp_output@9 < a10=attention_output@10 < m10=mlp_output@10 "
        "< a11=attention_output@11 < m11=mlp_output@11" in out
    )
    assert "forwards  4 per point" in out  # the hand-written document's plan


# --------------------------------------------------------------------------- #
# refusals — parser codes, the block path, no rule number
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["sender", "receivers", "pos", "restoration"])
def test_a_missing_required_field_is_p2(field: str, env) -> None:
    doc = _block_form(SHIPPED)
    del doc["method"][BLOCK][field]
    _refused(doc, env, "P2", f"a path block needs {field!r}")


def test_an_unknown_key_is_p3_with_a_suggestion(env) -> None:
    doc = _block_form(SHIPPED)
    doc["method"][BLOCK]["restorations"] = "attention_only"
    err = _refused(
        doc,
        env,
        "P3",
        "unknown key 'restorations'.*did you mean 'restoration'",
        at="method.path_patching.restorations",
    )
    assert err.path == "method.path_patching.restorations"


def test_an_unknown_policy_is_p4_with_a_suggestion(env) -> None:
    doc = _block_form(SHIPPED, restoration="attention_and_mlps")
    _refused(
        doc,
        env,
        "P4",
        "unknown restoration policy 'attention_and_mlps'.*did you mean 'attention_and_mlp'",
        at="method.path_patching.restoration",
    )


def test_an_empty_receiver_list_is_p2(env) -> None:
    doc = _block_form(SHIPPED)
    doc["method"][BLOCK]["receivers"] = []
    _refused(doc, env, "P2", "non-empty list", at="method.path_patching.receivers")


def test_a_path_that_runs_upstream_is_p2(env) -> None:
    """Sender above receiver: the lowered document would pass every checklist
    rule and run as a no-freeze patch of the wrong direction — the one case
    the block has to refuse itself."""
    doc = _block_form(SHIPPED, receivers=[{"component": "block_input", "layers": [5]}])
    _refused(
        doc,
        env,
        "P2",
        r"a path runs upstream: sender layer 9 is above receiver layer 5",
        at="method.path_patching.receivers[0]",
    )


def test_a_receiver_in_the_senders_own_layer_is_p2(env) -> None:
    doc = _block_form(SHIPPED, receivers=[{"component": "block_output", "layers": [9]}])
    _refused(
        doc,
        env,
        "P2",
        r"a receiver inside the sender's own layer \(9\) is not in this version",
        at="method.path_patching.receivers[0]",
    )


def test_a_sweep_wrapper_inside_the_block_is_p2(env) -> None:
    """The freeze set depends on the layers, so a swept layer would need one
    lowering per point; the block refuses the wrapper itself, first — the
    analogue of rule 28's "no sweep anywhere in a family entry"."""
    doc = _block_form(SHIPPED)
    doc["method"][BLOCK]["sender"]["layers"] = {"sweep": [8, 9]}
    _refused(
        doc,
        env,
        "P2",
        "a 'sweep' wrapper inside a path block is not in this version",
        at="method.path_patching.sender.layers",
    )


def test_an_at_once_wrapper_inside_the_block_is_rule_28_at_the_families_stage(
    env,
) -> None:
    """As it actually falls: ``families`` runs before ``paths`` and
    walks every method key, so an ``at_once`` inside the block is refused
    there — rule 28, "sits where it has no name identity", naming the path in
    the block — before the lowering ever sees it. The block's own check
    (:func:`expand_paths` called directly) refuses the same wrapper as
    ``P2``."""
    doc = _block_form(SHIPPED)
    doc["method"][BLOCK]["sender"]["layers"] = {"at_once": [8, 9]}
    with pytest.raises(ValidationError) as err:
        _compile(doc, env)
    assert err.value.rule == 28
    assert err.value.path == "path_patching.sender.layers"
    with pytest.raises(ParseError) as direct:
        expand_paths(doc)
    assert direct.value.code == "P2"
    assert direct.value.path == "method.path_patching.sender.layers"


def test_a_duplicate_receiver_is_p2(env) -> None:
    """A receiver set is a set. Refused in the block rather than left to rule
    8: the lowered document would give the two members two *names*
    (``receiver_0``, ``receiver_1``) at one address, and rule 8 compares
    absolute writes by site name, so it would not fire. A whole-component
    site covers each of its heads, so a head-scoped member beside the whole
    one is the same repeat; two heads are two addresses (the twin)."""
    site = {"component": "block_input", "layers": [12]}
    doc = _block_form(SHIPPED, receivers=[site, dict(site)])
    _refused(
        doc,
        env,
        "P2",
        r"receivers\[1\] repeats receivers\[0\] \(block_input@12\)",
        at="method.path_patching.receivers[1]",
    )
    head = {"component": "attention_premix", "layers": [12], "head": 1}
    whole = {"component": "attention_premix", "layers": [12]}
    _refused(
        _block_form(SHIPPED, receivers=[head, whole]),
        env,
        "P2",
        r"receivers\[1\] repeats receivers\[0\] \(attention_premix@12\)",
        at="method.path_patching.receivers[1]",
    )
    two_heads = _compile(
        _block_form(SHIPPED, receivers=[head, {**head, "head": 2}]), env
    )
    assert two_heads.lowered[BLOCK]["receivers"] == ["receiver_0", "receiver_1"]


# --------------------------------------------------------------------------- #
# the address-level collision check: rule 8 compares
# absolute writes by site *name*, and the lowering hands every emitted site its
# own name, so two sites at one address would run — silently.
# --------------------------------------------------------------------------- #

#: The address list the check and the emission share (``_Path.sites``).
_MLP_INTERIOR = ["mlp_input", "mlp_activation", "mlp_output"]


@pytest.mark.parametrize("component", _MLP_INTERIOR)
def test_a_sender_inside_the_mlp_its_own_policy_freezes_is_p2(
    component: str, env
) -> None:
    """Under ``attention_and_mlp`` the sender's own layer's MLP output is held
    clean (``m9``, ``freeze_m9`` at ``mlp_output@9``). A sender at the MLP's
    input, its activation or its output at layer 9 has everything it wrote
    overwritten by that freeze: the path effect is exactly zero, and nothing
    downstream would say so — ``swap_sender`` and ``freeze_m9`` are two names
    at one address. Twins: ``attention_only`` freezes no MLP, so the same
    sender lowers; and a sender on the residual beside the MLP
    (``block_mid@9``) is not inside it."""
    doc = _block_form(SHIPPED, restoration="attention_and_mlp")
    doc["method"][BLOCK]["sender"] = {"component": component, "layers": [9]}
    _refused(
        doc,
        env,
        "P2",
        rf"the sender at {component}@9 is inside the MLP of layer 9.*"
        r"\(m9 = mlp_output@9, freeze_m9\).*would overwrite the sender's effect"
        r".*exactly zero",
        at="method.path_patching.sender",
    )
    doc["method"][BLOCK]["restoration"] = "attention_only"
    twin = _compile(doc, env)
    assert twin.lowered[BLOCK]["restorers"] == [
        [10, "attention_output", "a10"],
        [11, "attention_output", "a11"],
    ]
    assert twin.canonical["method"]["sites"]["sender"]["component"] == component
    beside = _block_form(SHIPPED, restoration="attention_and_mlp")
    beside["method"][BLOCK]["sender"] = {"component": "block_mid", "layers": [9]}
    assert _compile(beside, env).lowered[BLOCK]["restored"]["mlp_output"] == [9, 10, 11]


def test_a_receiver_at_a_restored_site_is_p2(env) -> None:
    """With the sender at 9 and the farthest receiver at 12, ``attention_only``
    freezes ``attention_output`` at 10 and 11. A second receiver at
    ``attention_output@11`` would read the frozen *clean* value in the harvest
    model and inject clean for clean — a path effect of exactly zero through
    it, with ``receiver_1`` and ``a11`` two names at one address. Twins: a
    receiver inside the attention of layer 11 (``attention_premix@11``) reads
    before the freeze and lowers; a lone receiver at ``attention_output@12``
    is the farthest, so the freeze range stops below it."""
    far = {"component": "block_input", "layers": [12]}
    _refused(
        _block_form(
            SHIPPED, receivers=[far, {"component": "attention_output", "layers": [11]}]
        ),
        env,
        "P2",
        r"receivers\[1\] \(attention_output@11\) is a site 'attention_only' holds at "
        r"its clean value \(a11, freeze_11\).*frozen clean value.*clean for clean",
        at="method.path_patching.receivers[1]",
    )
    inside = _compile(
        _block_form(
            SHIPPED,
            receivers=[
                far,
                {"component": "attention_premix", "layers": [11], "head": 3},
            ],
        ),
        env,
    )
    assert inside.lowered[BLOCK]["receivers"] == ["receiver_0", "receiver_1"]
    assert inside.lowered[BLOCK]["restored"] == {"attention_output": [10, 11]}
    farthest = _compile(
        _block_form(
            SHIPPED, receivers=[{"component": "attention_output", "layers": [12]}]
        ),
        env,
    )
    assert farthest.lowered[BLOCK]["restored"] == {"attention_output": [10, 11]}


def test_a_receiver_at_a_restored_mlp_is_p2_under_attention_and_mlp_only(env) -> None:
    """The same collision through the policy: ``mlp_output@10`` is a frozen
    site under ``attention_and_mlp`` (``m10``) and a free one under
    ``attention_only`` — the twin is the same document with the other
    policy."""
    receivers = [
        {"component": "block_input", "layers": [12]},
        {"component": "mlp_output", "layers": [10]},
    ]
    _refused(
        _block_form(SHIPPED, restoration="attention_and_mlp", receivers=receivers),
        env,
        "P2",
        r"receivers\[1\] \(mlp_output@10\) is a site 'attention_and_mlp' holds at "
        r"its clean value \(m10, freeze_m10\)",
        at="method.path_patching.receivers[1]",
    )
    free = _compile(
        _block_form(SHIPPED, restoration="attention_only", receivers=receivers), env
    )
    assert free.lowered[BLOCK]["receivers"] == ["receiver_0", "receiver_1"]


def test_one_name_for_both_intervened_models_is_p2(env) -> None:
    """``{"harvest": "final", "inject": "final"}`` would collapse the two
    intervened models into one dict key — the harvest model vanishes, and
    rule 7 would report a missing model illegibly. Refused by name at the
    block; the twin renames the harvest model and both survive."""
    _refused(
        _block_form(SHIPPED, harvest="final", inject="final"),
        env,
        "P2",
        r"'harvest' and 'inject' both name 'final'.*harvest model.*would vanish",
        at="method.path_patching.inject",
    )
    twin = _compile(_block_form(SHIPPED, harvest="before"), env)
    assert set(twin.canonical["method"]["intervened_models"]) == {"before", "final"}
    assert twin.lowered[BLOCK]["harvest"] == "before"


@pytest.mark.parametrize(("name", "key"), [("patched", "harvest"), ("final", "inject")])
def test_an_authored_intervened_model_named_like_a_generated_one_blames_the_key(
    name: str, key: str, env
) -> None:
    """The same-table collision, when the generated name is one the block's
    own field chose: the refusal names ``harvest`` / ``inject`` rather than
    the block as a whole."""
    doc = _block_form(SHIPPED)
    doc["method"]["intervened_models"] = {name: {"input": "base", "writes": []}}
    _refused(
        doc,
        env,
        "P2",
        rf"emits intervened_models.'{name}', which the document also declares",
        at=f"method.path_patching.{key}",
    )


@pytest.mark.parametrize("section", ["sites", "reads", "writes", "intervened_models"])
def test_an_authored_table_that_is_not_an_object_is_p2_not_an_assertion(
    section: str, env
) -> None:
    """An authored ``method.<section>`` that is a string where the block
    emits into it: refused ``P2`` by the table's path, saying what the block
    emits there. Without the refusal the emitted entries were dropped and
    ``describe_paths`` reached an ``AssertionError`` before the gate could
    say so by name — the mutation this test exists for."""
    doc = _block_form(SHIPPED)
    doc["method"][section] = "final"
    with pytest.raises(ParseError) as err:
        _compile(doc, env)
    assert err.value.code == "P2"
    assert err.value.path == f"method.{section}"
    assert f"'method.{section}' has to be an object" in str(err.value)
    assert "it is a str" in str(err.value)


def test_a_generated_name_the_author_also_declares_in_the_same_table_is_p2(env) -> None:
    doc = _block_form(SHIPPED)
    doc["method"]["sites"]["a10"] = {"component": "mlp_output", "layers": [10]}
    _refused(doc, env, "P2", "emits sites.'a10', which the document also declares")


def test_a_generated_name_colliding_across_sections_is_rule_3(env) -> None:
    """The lowered document's checklist owns cross-section collisions: an
    authored ``reads.a10`` and the generated ``sites.a10`` share one
    namespace, and rule 3 names both."""
    doc = _block_form(SHIPPED)
    doc["method"]["reads"]["a10"] = {
        "site": "lm_head",
        "pos": -1,
        "model": "original",
        "input": "base",
    }
    with pytest.raises(ValidationError) as err:
        _compile(doc, env)
    assert err.value.rule == 3
    assert "'a10'" in str(err.value) and "'sites'" in str(err.value)


def test_a_sender_without_an_integer_layer_is_p2(env) -> None:
    doc = _block_form(SHIPPED)
    doc["method"][BLOCK]["sender"] = {"component": "lm_head"}
    _refused(
        doc,
        env,
        "P2",
        "needs 'layers' naming one layer",
        at="method.path_patching.sender",
    )


# --------------------------------------------------------------------------- #
# protocol v3 (9-1): a site's depth is the band ``layers`` (§2.4). A path runs
# between two layers, so a sender or receiver is the one-layer band, L or [L];
# a band of several layers is refused P4 in ``_site`` — the one place to
# reverse it — and every emitted site is written in the canonical form [L].
# --------------------------------------------------------------------------- #


def test_a_bare_layer_and_the_one_member_band_are_one_document(env) -> None:
    """(i) ``layers: 3`` and ``layers: [3]`` on the sender (``12`` / ``[12]`` on
    the receiver) compile to byte-identical explicit and canonical documents
    and one digest — the §2.4 fold, made in ``_site`` before the lowering so
    every emitted site is in one spelling (the canonical form alone would
    fold the canonical bytes, not the saved explicit tree). The derived record
    keeps the block as authored. The v2 key ``layer`` is refused ``P3`` with
    the rename suggested, as the parser does for a hand-written site."""
    listed = _block_form(
        SHIPPED, receivers=[{"component": "block_input", "layers": [12]}]
    )
    listed["method"][BLOCK]["sender"] = {
        "component": "attention_premix",
        "layers": [3],
        "head": 9,
    }
    bare = copy.deepcopy(listed)
    bare["method"][BLOCK]["sender"]["layers"] = 3
    bare["method"][BLOCK]["receivers"][0]["layers"] = 12
    from_list = _compile(listed, env)
    from_bare = _compile(bare, env)
    assert from_list.points.explicit == from_bare.points.explicit
    assert from_list.canonical == from_bare.canonical
    assert from_list.digests == from_bare.digests
    sites = from_bare.points.explicit["method"]["sites"]
    assert sites["sender"] == {
        "component": "attention_premix",
        "layers": [3],
        "head": 9,
    }
    assert sites["receiver"] == {"component": "block_input", "layers": [12]}
    assert from_bare.lowered[BLOCK]["restored"] == {
        "attention_output": list(range(4, 12))
    }
    assert from_bare.lowered[BLOCK]["authored"]["sender"]["layers"] == 3
    assert from_list.lowered[BLOCK]["authored"]["sender"]["layers"] == [3]
    v2 = copy.deepcopy(listed)
    v2["method"][BLOCK]["sender"] = {
        "component": "attention_premix",
        "layer": 3,
        "head": 9,
    }
    _refused(
        v2,
        env,
        "P3",
        "unknown key 'layer'.*did you mean 'layers'",
        at="method.path_patching.sender.layer",
    )


def test_a_band_sender_is_p4_naming_the_mechanism(env) -> None:
    """(ii) ``layers: [3, 4]`` on the sender. A path runs between two layers:
    a band sender has no defined freeze range, and reading it member-wise
    would be a *set* of paths the author spells with one block per layer. So
    ``P4`` at ``sender.layers``, naming the mechanism and the band, beside the
    twin ``[3]`` that lowers. Mutation: drop the ``len(members) > 1`` refusal
    in ``_site`` — a band site is legal v3, so the document compiles (DID NOT
    RAISE) with a two-layer ``sender``."""
    doc = _block_form(SHIPPED)
    doc["method"][BLOCK]["sender"] = {
        "component": "attention_premix",
        "layers": [3, 4],
        "head": 9,
    }
    _refused(
        doc,
        env,
        "P4",
        r"a path runs between two layers; a band sender \(3\.\.4: 2 layers\) is "
        r"not defined — write one path per layer",
        at="method.path_patching.sender.layers",
    )
    doc["method"][BLOCK]["sender"]["layers"] = [4, 3]  # any two members: a band
    _refused(
        doc,
        env,
        "P4",
        r"a band sender \(4\+3: 2 layers\)",
        at="method.path_patching.sender.layers",
    )
    doc["method"][BLOCK]["sender"]["layers"] = [3]
    twin = _compile(doc, env)
    assert twin.lowered[BLOCK]["restored"] == {"attention_output": list(range(4, 12))}
    assert twin.points.explicit["method"]["sites"]["sender"]["layers"] == [3]


def test_a_band_receiver_is_p4_naming_the_mechanism(env) -> None:
    """(iii) the receiver half: ``layers: [11, 12]`` is refused ``P4`` at
    ``receivers[i].layers`` with the same sentence; a non-contiguous band is
    labelled as sweep labels are (``10+12``). The twin is one path per layer —
    one block per receiver layer — and each lowers. A malformed ``layers``
    (empty, a non-integer member, a float, a boolean) is ``P2`` as a missing
    depth is, at the site."""
    doc = _block_form(
        SHIPPED, receivers=[{"component": "block_input", "layers": [11, 12]}]
    )
    _refused(
        doc,
        env,
        "P4",
        r"a path runs between two layers; a band receiver \(11\.\.12: 2 layers\) "
        r"is not defined — write one path per layer",
        at="method.path_patching.receivers[0].layers",
    )
    second = _block_form(
        SHIPPED,
        receivers=[
            {"component": "block_input", "layers": [12]},
            {"component": "attention_output", "layers": [10, 12]},
        ],
    )
    _refused(
        second,
        env,
        "P4",
        r"a band receiver \(10\+12: 2 layers\)",
        at="method.path_patching.receivers[1].layers",
    )
    for layer in (11, 12):
        one = _compile(
            _block_form(
                SHIPPED, receivers=[{"component": "block_input", "layers": [layer]}]
            ),
            env,
        )
        assert one.lowered[BLOCK]["restored"] == {
            "attention_output": list(range(10, layer))
        }
    for bad in ([], ["12"], [True], 12.5, True, None):
        malformed = _block_form(
            SHIPPED, receivers=[{"component": "block_input", "layers": bad}]
        )
        _refused(
            malformed,
            env,
            "P2",
            r"a path's receivers needs 'layers' naming one layer, L or \[L\]",
            at="method.path_patching.receivers[0]",
        )


def test_emitted_sites_are_written_as_the_canonical_one_layer_band(env) -> None:
    """(iv) every restorer site the block emits (``a{L}``, ``m{L}``) is written
    ``{"component": …, "layers": [L]}`` — the canonical one-layer band of
    §2.4 — so the explicit tree is already canonical at its sites and equals,
    entry for entry, the canonical form of a hand-written twin: the shipped
    v3 file (``[L]``) and the same file with every site spelled bare (``L``).
    Without the list form the explicit sites would read ``layers: 10``: the
    canonical form would still fold them, the saved explicit tree would not
    be the twin's."""
    both = _compile(_block_form(SHIPPED, restoration="attention_and_mlp"), env)
    explicit = both.points.explicit["method"]["sites"]
    restorers = both.lowered[BLOCK]["restorers"]
    assert len(restorers) == 5
    for layer, component, name in restorers:
        assert explicit[name] == {"component": component, "layers": [layer]}
        assert explicit[name] == both.canonical["method"]["sites"][name]
    assert explicit["m9"] == {"component": "mlp_output", "layers": [9]}
    assert explicit["a10"] == {"component": "attention_output", "layers": [10]}
    shipped = json.loads(SHIPPED.read_text())
    bare = copy.deepcopy(shipped)
    for site in bare["method"]["sites"].values():
        if "layers" in site:
            (site["layers"],) = site["layers"]
    assert bare != shipped
    block = _compile(_block_form(SHIPPED), env)
    for twin in (shipped, bare):
        hand = _compile(twin, env)
        assert block.canonical == hand.canonical
        assert block.digests.document == hand.digests.document
        assert block.digests.points == hand.digests.points
        assert (
            block.points.explicit["method"]["sites"]
            == hand.canonical["method"]["sites"]
        )
    # `explicit` is the method group as authored — defaults not materialized —
    # so the bare-spelled hand-written twin differs there by design; the block,
    # whose sites are normalized to the list form, reads as the shipped [L] file
    assert block.digests == _compile(shipped, env).digests


def test_an_unknown_component_in_the_block_is_p4(env) -> None:
    doc = _block_form(SHIPPED)
    doc["method"][BLOCK]["sender"]["component"] = "attention_premixx"
    _refused(
        doc,
        env,
        "P4",
        "unknown component 'attention_premixx'.*did you mean 'attention_premix'",
        at="method.path_patching.sender.component",
    )


def test_a_block_that_is_not_an_object_is_p2(env) -> None:
    doc = _block_form(SHIPPED)
    doc["method"][BLOCK] = ["sender"]
    _refused(doc, env, "P2", "a path block is an object")


def test_set_cannot_address_the_block(env) -> None:
    """Deferred deliberately (design B would put the block in
    ``METHOD_SECTIONS`` and move the workflow pins): an override into the
    block is refused as a path that does not exist, as §3.2 says."""
    with pytest.raises(ProtocolError, match="does not exist in the document"):
        _compile(
            _block_form(SHIPPED),
            env,
            {"path_patching.restoration": "attention_and_mlp"},
        )


def test_a_refusal_survives_the_round_trip_through_load(env) -> None:
    """The codes reach the caller unchanged through the loader's original
    name, so ``causalab validate`` prints ``[P4]`` for a bad policy."""
    from causalab.protocol.loader import load

    with pytest.raises(ParseError) as err:
        load(_block_form(SHIPPED, restoration="mlp_only"), env)
    assert err.value.code == "P4"
    assert "[P4]" in str(err.value)


def test_the_gate_never_sees_a_block(env) -> None:
    """Without the stage, this is the refusal a block document gets: the
    parser's unknown-section ``[P3]``. That it does not is the stage."""
    with pytest.raises(ParseError) as err:
        parse_document(_block_form(SHIPPED))
    assert err.value.code == "P3" and "path_patching" in str(err.value)
    _compile(_block_form(SHIPPED), env)  # and through the compiler it loads
