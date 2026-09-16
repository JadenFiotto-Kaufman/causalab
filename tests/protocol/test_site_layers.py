"""§2.4 ``layers`` — a site's depth is a **band** of layer indices.

The one-layer band ``[n]`` is the scalar site it replaced, everywhere (T-a);
every consumer either takes a band deliberately or refuses it by name (T-b);
a one-site band plans and lowers to the hand-written N-site document (T-c —
the run half is ``tests/neural/engines/pytorch_hooks/test_band_site_run.py``);
and the rename shipped as protocol_version 3 with a migration that proves the
70 rewritten documents mechanical (T-d's document half).
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.protocol.canonical import canonicalize, digest
from causalab.protocol.errors import ParseError, ProtocolError, ValidationError
from causalab.protocol.families import expand_families
from causalab.protocol.loader import apply_overrides
from causalab.protocol.migrate import (
    format_document,
    migrate_document,
    migrate_markdown,
    needs_migration,
)
from causalab.protocol.plan import (
    band_member,
    lower_bands,
    plan_point,
    site_depth,
    site_depths,
)
from causalab.protocol.schema import (
    MIGRATABLE_PROTOCOL_VERSIONS,
    PROTOCOL_VERSION,
    SiteSpec,
    Sweep,
    parse_document,
)
from causalab.protocol.sweep import band_label, coordinate_label, label_value
from causalab.protocol.validate import validate_document

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import FIXTURES, build_env

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
ENV = build_env(FIXTURES / "artifacts")
#: The revision the rename was cut from: every document below carried
#: ``"layer"`` there, and ``causalab migrate`` wrote the committed v3 file.
BASE = "d6f27c24"
#: The documents re-described after ``BASE``: the locate scan's
#: ``description`` was reworded (it now names ``writes.<w>.ragged``) in the shipped
#: file and its corpus twin, and the DBM, multi-position-patch, random-subspace,
#: band-patch, drift and mixing-scan descriptions lost their run narrative.
#: Header prose enters no canonical form (§7), so the base-revision round trip
#: below holds these paths to their current prose and every other document byte
#: for byte.
REDESCRIBED = frozenset(
    {
        "causalab/configs/protocols/weekdays_locate_scan.json",
        "tests/protocols/07_weekdays_locate_scan_im.json",
        "causalab/configs/protocols/dbm.json",
        "causalab/configs/protocols/multi_position_patch.json",
        "causalab/configs/protocols/random_subspace_control.json",
        "tests/protocols/05_dbm_im.json",
        "tests/protocols/13_random_subspace_control_im.json",
        "tests/protocols/14_multi_position_patch_im.json",
        "tests/protocol/fixtures/band_patch_handwritten.json",
        "tests/golden/protocols/drift_interchange_im.json",
        "tests/golden/protocols/drift_locate_scan_im.json",
        "tests/golden/protocols/mixing_scan_first_im.json",
        "tests/golden/protocols/mixing_scan_last_im.json",
        "tests/golden/protocols/mixing_scan_middle_im.json",
    }
)
A3B = "Qwen/Qwen3.6-35B-A3B"


def _site(**fields: Any) -> dict[str, Any]:
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "block_output", **fields}
    return raw


def _band_doc(layers: list[int]) -> dict[str, Any]:
    """base_doc with its one write and its operand read on a band."""
    return _site(layers=layers)


# --------------------------------------------------------------------------- #
# T-b — the parser
# --------------------------------------------------------------------------- #


def test_the_field_is_a_tuple_of_layer_indices():
    doc = parse_document(_site(layers=[3]))
    assert doc.sites["tgt"].layers == (3,)
    assert parse_document(_site(layers=[3, 5, 8])).sites["tgt"].layers == (3, 5, 8)
    assert doc.sites["lm_head"].layers is None


@pytest.mark.parametrize(
    "bad, words",
    [
        ([], "at least one layer"),
        ([5, 3], "strictly increasing"),
        ([3, 3], "a layer repeats"),
        ([True], "got bool"),
        ([3, None], "got NoneType"),
        (None, "a list of layer indices"),
        ("3", "a list of layer indices"),
        (3.0, "a list of layer indices"),
    ],
)
def test_the_parser_refuses_a_malformed_band_by_name(bad, words):
    with pytest.raises(ParseError) as err:
        parse_document(_site(layers=bad))
    assert err.value.code == "P2"
    assert words in str(err.value)
    assert "sites.tgt.layers" in str(err.value)


def test_layerless_and_layered_components_are_held_to_the_field():
    raw = base_doc()
    raw["method"]["sites"]["lm_head"]["layers"] = [0]
    with pytest.raises(ParseError, match="layer-less"):
        parse_document(raw)
    raw = base_doc()
    del raw["method"]["sites"]["tgt"]["layers"]
    with pytest.raises(ParseError, match="needs 'layers'"):
        parse_document(raw)


def test_the_v2_spelling_is_refused_naming_the_rename_and_the_verb():
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "block_output", "layer": 3}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P3"
    assert "'layers'" in str(err.value)
    assert "causalab migrate" in str(err.value)


def test_a_v2_header_is_refused_by_name_and_told_how_to_migrate():
    """The rule ``test_protocol_v2`` pins for v1, applied to v2 (spec §7)."""
    for version in MIGRATABLE_PROTOCOL_VERSIONS:
        raw = base_doc()
        raw["header"]["protocol_version"] = version
        with pytest.raises(ParseError) as err:
            parse_document(raw)
        assert err.value.code == "P2"
        assert f"protocol_version {version!r} document" in str(err.value)
        assert "causalab migrate" in str(err.value)
        assert PROTOCOL_VERSION in str(err.value)
    assert PROTOCOL_VERSION == "3" and "2" in MIGRATABLE_PROTOCOL_VERSIONS


def test_a_bare_index_is_the_one_layer_band_at_parse_and_in_canonical_form():
    """An axis over ``layers`` and a workflow ``emit`` hand a point the
    index; the two spellings are one document (spec §2.4)."""
    listed, bare = _site(layers=[3]), _site(layers=3)
    import dataclasses

    assert dataclasses.replace(parse_document(listed), raw={}) == dataclasses.replace(
        parse_document(bare), raw={}
    )
    assert canonicalize(listed, ENV) == canonicalize(bare, ENV)
    assert digest(canonicalize(listed, ENV)) == digest(canonicalize(bare, ENV))
    assert canonicalize(bare, ENV)["method"]["sites"]["tgt"]["layers"] == [3]


def test_a_sweep_over_layers_is_indexed_by_layer():
    raw = _site(layers={"sweep": {"range": [1, 4]}})
    layers = parse_document(raw).sites["tgt"].layers
    assert isinstance(layers, Sweep) and layers.values == ((1,), (2,), (3,))
    from causalab.protocol.sweep import expand

    points = expand(raw).points
    assert [p.coords["sites.tgt.layers"] for p in points] == [1, 2, 3]
    assert [parse_document(p.raw).sites["tgt"].layers for p in points] == [
        (1,),
        (2,),
        (3,),
    ]


def test_a_sweep_over_bands_records_the_band_as_its_coordinate():
    raw = _site(layers={"sweep": [[1, 2], [3]]})
    from causalab.protocol.sweep import expand

    points = expand(raw).points
    assert [p.coords["sites.tgt.layers"] for p in points] == [[1, 2], [3]]
    assert parse_document(points[0].raw).sites["tgt"].layers == (1, 2)


# --------------------------------------------------------------------------- #
# T-a — the one-element band is the scalar case everywhere
# --------------------------------------------------------------------------- #


def test_one_element_band_plans_as_the_scalar_site_did():
    doc = parse_document(_site(layers=[3]))
    assert site_depth(doc, "tgt") == site_depths(doc, "tgt")[0]
    assert site_depth(doc, "tgt")[0] == 3
    plan = plan_point(doc)
    assert plan.num_forwards == 2
    assert [t.read for g in plan.groups for t in g.taps] == ["v_cf", "logits"]
    # the same document, index spelled: identical groups, identical digests
    twin = plan_point(parse_document(_site(layers=3)))
    assert [g.digest for g in twin.groups] == [g.digest for g in plan.groups]
    assert lower_bands(doc) is doc  # nothing to lower: the names are the author's


def test_coordinate_labels_and_band_labels():
    assert coordinate_label({"sites.target.layers": 3}) == "[target.layers=3]"
    assert coordinate_label({"sites.target.layers": [3]}) == "[target.layers=3]"
    assert coordinate_label({"sites.target.layers": [10, 11, 12]}) == (
        "[target.layers=10..12]"
    )
    assert label_value([10, 12, 15]) == "10+12+15"
    assert band_label((7,)) == "7"
    # neither the label syntax's comma nor its brackets appear in a band label
    assert all(c not in band_label((1, 2, 4)) for c in ",[]")


def test_the_artifact_stamp_carries_the_band_as_a_list():
    from causalab.neural.shared.services import site_identity
    from causalab.protocol.loader import _featurizer_realization

    doc = parse_document(_site(layers=[3]))
    assert site_identity(doc, "tgt") == {"component": "block_output", "layers": [3]}
    raw = _site(layers=[3])
    raw["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 2, "file_path": "rot.safetensors"}
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    raw["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc = parse_document(in_order(raw))
    expected = _featurizer_realization(doc, "rot")
    assert expected["site"] == {"component": "block_output", "layers": [3]}
    assert json.dumps(expected["site"], sort_keys=True) == json.dumps(
        site_identity(doc, "tgt"), sort_keys=True
    )


def test_the_select_to_fit_handoff_hands_the_fit_the_index():
    """``weekdays_8b``'s ``select`` emits the scan's coordinate — the layer
    index — and the fit document's ``sites.target.layers`` receives it as
    the one-layer band."""
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["layers"] = {"artifact": "best", "key": "best_layer"}
    import types

    from causalab.protocol.resolve import resolve_artifact_fields

    env = types.SimpleNamespace(
        artifacts=types.SimpleNamespace(read_value=lambda artifact, key: 18)
    )
    resolved = resolve_artifact_fields(raw, env)
    assert resolved["method"]["sites"]["tgt"]["layers"] == 18
    assert parse_document(resolved).sites["tgt"].layers == (18,)
    overridden = apply_overrides(base_doc(), {"sites.tgt.layers": 18})
    assert parse_document(overridden).sites["tgt"].layers == (18,)
    assert parse_document(
        apply_overrides(base_doc(), {"sites.tgt.layers": [4, 5]})
    ).sites["tgt"].layers == (4, 5)
    workflow = json.loads(
        (REPO / "causalab/configs/workflows/weekdays_8b.json").read_text()
    )
    assert workflow["steps"]["best"]["inputs"]["emit"]["best_layer"] == (
        "sites.target.layers"
    )
    assert workflow["steps"]["fit"]["set"]["sites.target.layers"] == {
        "artifact": "best",
        "key": "best_layer",
    }


# --------------------------------------------------------------------------- #
# T-b — every consumer, per member
# --------------------------------------------------------------------------- #


def test_rule_4_holds_every_member_of_a_band():
    raw = _site(layers=[3, 40])  # gpt2 has 12 layers
    with pytest.raises(ValidationError) as err:
        canonicalize(raw, ENV)
    assert err.value.rule == 4
    assert "layer 40 out of range" in str(err.value)
    assert "sites.tgt.layers" in str(err.value)
    canonicalize(_site(layers=[0, 11]), ENV)  # every member inside


def test_the_stream_check_holds_every_member_of_a_band():
    raw = base_doc()
    raw["model"] = {"key": A3B, "revision": "main", "dtype": "bf16"}
    raw["method"]["sites"]["tgt"] = {"component": "attention_premix", "layers": [3, 4]}
    with pytest.raises(ValidationError, match="exists only on a 'full_attention'"):
        canonicalize(in_order(raw), ENV)  # layer 4 is Gated DeltaNet
    raw["method"]["sites"]["tgt"]["layers"] = [3, 7]
    canonicalize(in_order(raw), ENV)  # both full attention


def test_rule_21_reads_a_band_operand_member_by_member():
    validate_document(parse_document(_band_doc([3, 4, 5])))  # equal depth, per member
    raw = _band_doc([3, 4, 5])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [4, 5, 6]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(raw)))
    assert err.value.rule == 21 and "layers 4..6" in str(err.value)


def test_rule_21_broadcasts_any_other_operand_to_the_bands_shallowest_member():
    raw = _band_doc([3, 4])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [5]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(raw)))
    assert err.value.rule == 21
    raw["method"]["sites"]["src"]["layers"] = [3]
    validate_document(parse_document(in_order(raw)))
    # and a band read feeding a one-layer write: its deepest member counts
    raw = _site(layers=[4])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [3, 5]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(raw)))
    assert err.value.rule == 21
    raw["method"]["sites"]["src"]["layers"] = [3, 4]
    validate_document(parse_document(in_order(raw)))


def test_site_depth_is_the_shallowest_member_and_site_depths_every_one():
    doc = parse_document(_band_doc([3, 4, 5]))
    assert site_depth(doc, "tgt")[0] == 3
    assert [d[0] for d in site_depths(doc, "tgt")] == [3, 4, 5]
    assert site_depths(doc, "lm_head") == (site_depth(doc, "lm_head"),)


def test_head_stats_refuses_a_band_cell_by_name(tmp_path):
    from causalab.analysis import head_stats
    from causalab.io.step_io import StepError

    from tests.step_scripts import put_table, run_step

    rows = [
        {"sites.target.layers": [0, 1], "sites.target.head": 0, "value": 1.0},
        {"sites.target.layers": 2, "sites.target.head": 0, "value": 1.0},
    ]
    table = put_table(tmp_path / "in.json", rows)
    with pytest.raises(StepError, match="layer band"):
        run_step(head_stats, {"table": table}, {"stats": tmp_path / "out.json"})


# --------------------------------------------------------------------------- #
# T-c — lowering: the N-site document the author would have written
# --------------------------------------------------------------------------- #


def test_lowering_fans_a_band_out_to_members_read_write_and_model():
    doc = parse_document(_band_doc([3, 4, 5]))
    low = lower_bands(doc)
    members = [band_member("tgt", n) for n in (3, 4, 5)]
    assert sorted(low.sites) == sorted([*members, "lm_head"])
    assert [low.sites[m].layers for m in members] == [(3,), (4,), (5,)]
    assert low.sites[members[0]].component == "block_output"
    assert sorted(low.reads) == sorted(
        ["logits", *(band_member("v_cf", n) for n in (3, 4, 5))]
    )
    assert low.reads["v_cf[layers=4]"].site == "tgt[layers=4]"
    assert low.writes["patch[layers=4]"].site == "tgt[layers=4]"
    assert low.writes["patch[layers=4]"].do.payload == "v_cf[layers=4]"
    assert low.intervened_models["patched"].writes == tuple(
        band_member("patch", n) for n in (3, 4, 5)
    )
    assert lower_bands(low) == low  # idempotent
    assert low.raw["method"]["sites"]["tgt[layers=5]"] == {
        "component": "block_output",
        "layers": [5],
    }
    assert low.metrics == doc.metrics and low.save == doc.save


def test_lowering_pairs_band_operands_by_index_and_broadcasts_the_rest():
    raw = _band_doc([4, 5])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [1, 2]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    low = lower_bands(parse_document(in_order(raw)))
    assert low.writes["patch[layers=4]"].do.payload == "v_cf[layers=1]"
    assert low.writes["patch[layers=5]"].do.payload == "v_cf[layers=2]"
    raw["method"]["sites"]["src"]["layers"] = [1]  # a one-layer read: broadcast
    low = lower_bands(parse_document(in_order(raw)))
    assert low.writes["patch[layers=4]"].do.payload == "v_cf"
    assert low.writes["patch[layers=5]"].do.payload == "v_cf"
    assert "v_cf" in low.reads and low.reads["v_cf"].site == "src"


def test_a_band_plans_the_forward_groups_of_its_hand_written_twin():
    """ROME's shape: one site over ten layers plans exactly what
    ``band_patch_handwritten.json``'s ten sites plan — the same groups, the
    same tap depths, the same first write, the same resume point. The three
    bands the fixture windows are three band sites here (one read, one write
    each) where the fixture shares ten writes between them."""
    from causalab.protocol.loader import load
    from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv

    from tests.protocol._env import TASKS_ROOT

    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=FIXTURES / "artifacts"),
    )
    hand = load(FIXTURES / "band_patch_handwritten.json", env).point_documents[0]
    raw = json.loads((FIXTURES / "band_patch_handwritten.json").read_text())
    method = raw["method"]
    bands = {"band5_L10": (10, 15), "band5_L15": (15, 20), "band10_L10": (10, 20)}
    method["sites"] = {
        **{
            f"a_{name}": {"component": "attention_output", "layers": list(range(*span))}
            for name, span in bands.items()
        },
        "lm_head": {"component": "lm_head"},
    }
    method["reads"] = {
        **{
            f"v_{name}": {
                "site": f"a_{name}",
                "pos": "tap",
                "model": "original",
                "input": "counterfactual",
            }
            for name in bands
        },
        **{k: v for k, v in method["reads"].items() if k.startswith("logits_")},
    }
    method["writes"] = {
        f"w_{name}": {"site": f"a_{name}", "pos": "tap", "do": {"swap": f"v_{name}"}}
        for name in bands
    }
    for name, im in method["intervened_models"].items():
        im["writes"] = [f"w_{name}"]
    band = load(raw, env, base_dir=FIXTURES).point_documents[0]
    hand_plan, band_plan = plan_point(hand), plan_point(band)
    assert band_plan.num_forwards == hand_plan.num_forwards == 4
    low = lower_bands(band)
    assert low.intervened_models["band10_L10"].writes == tuple(
        band_member("w_band10_L10", n) for n in range(10, 20)
    )
    assert len(low.writes) == 20 and len(hand.writes) == 10  # the fixture shares
    for hg, bg in zip(hand_plan.groups, band_plan.groups):
        assert (hg.model, hg.input) == (bg.model, bg.input)
        assert {t.depth for t in hg.taps} == {t.depth for t in bg.taps}
        assert hg.write_depth == bg.write_depth
        assert hg.resume_at == bg.resume_at and hg.stop_after == bg.stop_after


def test_at_once_windows_over_the_band_keep_the_hand_written_fixture():
    """The other band shape (§3.1) still expands to the fixture, now spelled
    ``layers`` — an ``at_once`` member is a layer index, a one-layer site."""
    corpus = json.loads((REPO / "tests/protocols/16_at_once_band_im.json").read_text())
    site = corpus["method"]["sites"]["a"]
    assert "layers" in site and "layer" not in site and site["names"] == "a{layers}"
    expanded = expand_families(corpus)
    assert sorted(expanded["method"]["sites"])[:3] == ["a10", "a11", "a12"]
    assert expanded["method"]["sites"]["a10"]["layers"] == [10]  # the index, a band
    assert parse_document(expanded).sites["a10"].layers == (10,)
    hand = json.loads((FIXTURES / "band_patch_handwritten.json").read_text())
    assert hand["method"]["sites"]["a10"] == {
        "component": "attention_output",
        "layers": [10],
    }
    # the corpus copy targets the fixture tables, the hand-written one the
    # shipped ones: the *method* is what the sugar must reproduce
    assert canonicalize(expanded, ENV)["method"] == canonicalize(hand, ENV)["method"]


def test_at_once_composes_with_layers_a_band_per_member():
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {
        "component": "block_output",
        "layers": {"at_once": [[3, 4], [5, 6]]},
    }
    expanded = expand_families(raw)
    names = sorted(n for n in expanded["method"]["sites"] if n != "lm_head")
    assert names == ["tgt[layers=3..4]", "tgt[layers=5..6]"]
    doc = parse_document(expanded)
    assert doc.sites["tgt[layers=3..4]"].layers == (3, 4)
    assert doc.writes["patch[layers=5..6]"].do.payload == "v_cf[layers=5..6]"
    validate_document(doc)
    low = lower_bands(doc)
    assert "tgt[layers=3..4][layers=4]" in low.sites


@pytest.mark.parametrize(
    "mutate, words",
    [
        (
            lambda m: m["save"].append(
                {
                    "value": "v_cf",
                    "model": "original",
                    "input": "counterfactual",
                    "file_path": "v.safetensors",
                }
            ),
            "save entry 'v_cf'",
        ),
        (
            lambda m: m["metrics"].__setitem__(
                "bad",
                {
                    "kind": "logit_diff",
                    "of": "v_cf",
                    "a": "cf_answer",
                    "b": "base_answer",
                    "token_form": "space_prefixed",
                },
            ),
            "metric 'bad' reduces read 'v_cf'",
        ),
        (
            lambda m: (
                m.__setitem__("featurizers", {"rot": {"kind": "subspace", "k": 2}}),
                m["reads"]["v_cf"].__setitem__("featurizer", "rot"),
                m["writes"]["patch"].__setitem__("featurizer", "rot"),
            ),
            "names a featurizer",
        ),
    ],
)
def test_what_a_band_has_no_member_for_is_refused_by_name(mutate, words):
    raw = _band_doc([3, 4])
    mutate(raw["method"])
    doc = parse_document(in_order(raw))
    with pytest.raises(ProtocolError) as err:
        lower_bands(doc)
    assert words in str(err.value) and "layers 3..4" in str(err.value)


def test_a_one_layer_write_fed_by_a_band_read_is_refused_by_name():
    raw = _site(layers=[4])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [1, 2]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    with pytest.raises(ProtocolError, match="a band read is N tensors"):
        lower_bands(parse_document(in_order(raw)))
    raw["method"]["sites"]["tgt"]["layers"] = [4, 5, 6]  # a band of another length
    with pytest.raises(ProtocolError, match="same number of layers"):
        lower_bands(parse_document(in_order(raw)))


def test_resolve_site_refuses_a_multi_layer_band_without_a_model():
    """The refusal is the resolver's own, before any module is touched."""
    from causalab.neural.shared.sites import resolve_site

    class NoBundle:
        pass

    with pytest.raises(ProtocolError, match="resolve_band"):
        resolve_site(NoBundle(), SiteSpec(component="block_output", layers=(3, 4)))


# --------------------------------------------------------------------------- #
# T-d — protocol_version 3 and the migration
# --------------------------------------------------------------------------- #


def _v2_of(v3: dict[str, Any]) -> dict[str, Any]:
    """The protocol_version 2 spelling of a v3 document — the inverse of the
    rename, so the round trip below is self-contained."""
    out = copy.deepcopy(v3)
    out["header"]["protocol_version"] = "2"
    for site in out["method"].get("sites", {}).values():
        if "layers" in site:
            value = site.pop("layers")
            site["layer"] = value[0] if isinstance(value, list) else value
            # key order as the v2 author had it: component, layer, the rest
            rest = {
                k: site.pop(k) for k in list(site) if k not in ("component", "layer")
            }
            site.update(rest)
        if isinstance(site.get("names"), str):
            site["names"] = site["names"].replace("{layers}", "{layer}")
    for table in out["method"].values():
        if isinstance(table, dict):
            for entry in table.values():
                if isinstance(entry, dict) and isinstance(entry.get("names"), str):
                    entry["names"] = entry["names"].replace("{layers}", "{layer}")
    for im in out["method"].get("intervened_models", {}).values():
        if isinstance(im.get("writes"), list):
            for item in im["writes"]:
                if isinstance(item, dict):
                    for selector in item.values():
                        if isinstance(selector, dict) and "layers" in selector:
                            selector["layer"] = selector.pop("layers")
    return json.loads(
        json.dumps(out).replace(".layers", ".layer")  # dotted ids in descriptions
    )


def _documents() -> list[Path]:
    globs = (
        "causalab/configs/protocols/*.json",
        "tests/protocols/*_im.json",
        "tests/golden/protocols/*_im.json",
        "demos/*/protocols/*.json",
        "tests/protocol/fixtures/band_patch_handwritten.json",
    )
    return sorted(p for g in globs for p in REPO.glob(g))


def _workflows() -> list[Path]:
    return sorted(
        [
            *REPO.glob("causalab/configs/workflows/*.json"),
            *REPO.glob("demos/*/workflows/*.json"),
        ]
    )


def test_every_shipped_document_is_at_version_3_and_spells_layers():
    docs = _documents()
    assert len(docs) >= 24 + 16 + 16 + 20 + 1
    layered = 0
    for path in docs:
        raw = json.loads(path.read_text())
        assert raw["header"]["protocol_version"] == PROTOCOL_VERSION, path
        assert not needs_migration(raw), path
        for name, site in raw["method"]["sites"].items():
            assert "layer" not in site, (path, name)
            layered += "layers" in site
        assert '"layer"' not in path.read_text(), path
    assert layered >= 70
    for path in _workflows():
        text = path.read_text()
        assert '.layer"' not in text and ".layer=" not in text, path
        assert not needs_migration(json.loads(text)), path


def test_the_migration_reproduces_every_committed_document_from_its_v2_spelling():
    """The 70 rewrites were written by the migrator; the inverse rename
    followed by the migration is the identity on every one of them."""
    for path in _documents():
        v3 = json.loads(path.read_text())
        migrated = migrate_document(_v2_of(v3))
        assert json.dumps(migrated, sort_keys=True) == json.dumps(v3, sort_keys=True), (
            path
        )
        assert migrate_document(v3) == v3  # idempotent


def test_the_migration_reproduces_every_committed_document_from_the_base_revision():
    """The same round trip against the bytes as they were at ``BASE`` — the
    revision the rename was cut from — for every document and workflow the
    commit rewrote, byte for byte except the ``REDESCRIBED`` documents'
    header prose. Skipped where the history is not at hand (an export)."""
    try:
        subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", f"{BASE}^{{commit}}"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError):
        pytest.skip(f"revision {BASE} is not in this checkout's history")
    checked = 0
    redescribed: set[str] = set()
    for path in [*_documents(), *_workflows()]:
        rel = path.relative_to(REPO).as_posix()
        shown = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{BASE}:{rel}"],
            capture_output=True,
            text=True,
        )
        if shown.returncode != 0:
            continue  # a document added after the base
        before = json.loads(shown.stdout)
        after = json.loads(path.read_text())
        migrated = migrate_document(before)
        if rel in REDESCRIBED:
            # the header's prose is not what the rename rewrote: `description`
            # says what a file is for and enters no canonical form (§7), and
            # these documents were re-described after BASE — so the round
            # trip is held on everything but their prose, with the current text
            # in place; the exemption is earned (the prose did change) and
            # reaches no other file
            assert migrated["header"]["description"] != after["header"]["description"]
            migrated["header"]["description"] = after["header"]["description"]
            redescribed.add(rel)
        assert json.dumps(migrated, sort_keys=True) == json.dumps(
            after, sort_keys=True
        ), rel
        if "steps" not in before:
            assert format_document(migrated) == path.read_text(), rel
        checked += 1
    assert checked >= 70 + 12
    assert redescribed == REDESCRIBED


def test_migrate_carries_windows_templates_and_dotted_ids():
    v2 = {
        "header": {"protocol_version": "2", "description": "axis sites.a.layer here"},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {"base": {"dataset": "d", "field": "input"}},
        "method": {
            "sites": {
                "a": {
                    "component": "attention_output",
                    "layer": {"at_once": {"range": [1, 3]}},
                    "names": "a{layer}",
                },
                "s": {"component": "block_output", "layer": {"sweep": [1, 2]}},
                "r": {
                    "component": "block_output",
                    "layer": {"artifact": "x", "key": "k"},
                    "head": 0,
                },
            },
            "writes": {
                "w": {"site": "a", "pos": -1, "do": {"swap": "v"}, "names": "w{layer}"}
            },
            "intervened_models": {
                "m": {
                    "input": "base",
                    "writes": [{"w": {"layer": {"at_once": [1]}}}, "other"],
                }
            },
            "save": [],
        },
    }
    v3 = migrate_document(v2)
    sites = v3["method"]["sites"]
    assert sites["a"] == {
        "component": "attention_output",
        "layers": {"at_once": {"range": [1, 3]}},
        "names": "a{layers}",
    }
    assert sites["s"] == {"component": "block_output", "layers": {"sweep": [1, 2]}}
    assert list(sites["r"]) == ["component", "layers", "head"]  # key order kept
    assert v3["method"]["writes"]["w"]["names"] == "w{layers}"
    assert v3["method"]["intervened_models"]["m"]["writes"] == [
        {"w": {"layers": {"at_once": [1]}}},
        "other",
    ]
    assert v3["header"] == {
        "protocol_version": "3",
        "description": "axis sites.a.layers here",
    }
    workflow = {
        "version": "1",
        "output_dir": "r",
        "steps": {
            "fit": {"set": {"sites.target.layer": {"artifact": "b", "key": "k"}}},
            "best": {
                "inputs": {
                    "emit": {"best_layer": "sites.target.layer"},
                    "layer_column": "sites.c.layer",
                }
            },
            "plot": {
                "inputs": {"x": "sites.target.layer", "columns": {"layer": "int64"}}
            },
        },
    }
    assert needs_migration(workflow)
    out = migrate_document(workflow)
    assert out["steps"]["fit"]["set"] == {
        "sites.target.layers": {"artifact": "b", "key": "k"}
    }
    assert out["steps"]["best"]["inputs"] == {
        "emit": {"best_layer": "sites.target.layers"},
        "layer_column": "sites.c.layers",
    }
    assert out["steps"]["plot"]["inputs"]["columns"] == {"layer": "int64"}  # not a site
    assert not needs_migration(out) and migrate_document(out) == out
    with pytest.raises(ParseError, match="unsupported protocol_version '4'"):
        migrate_document(
            {"header": {"protocol_version": "4"}, "model": {}, "data": {}, "method": {}}
        )


def test_migrate_chains_v1_through_v2_to_v3():
    v3 = base_doc()
    v1 = {
        "version": "1",
        "model": v3["model"],
        "data": v3["data"],
        **_v2_of(v3)["method"],
    }
    assert migrate_document(v1) == v3


def test_migrate_markdown_rewrites_whole_v2_documents_and_stale_workflows_only():
    v2 = _v2_of(base_doc())
    workflow = {
        "version": "1",
        "output_dir": "r",
        "steps": {"s": {"set": {"sites.t.layer": 1}}},
    }
    prose = (
        "```json\n" + json.dumps(v2, indent=2) + "\n```\n\n"
        '```json\n{"target": {"component": "block_output", "layer": 18}}\n```\n\n'
        "```json\n" + json.dumps(workflow) + "\n```\n"
    )
    out = migrate_markdown(prose)
    assert '"protocol_version": "3"' in out and '"layers": [3]' in out
    assert '{"target": {"component": "block_output", "layer": 18}}' in out  # a fragment
    assert '"sites.t.layers": 1' in out
    assert migrate_markdown(out) == out


def test_the_migrate_verb_rewrites_a_v2_file_and_check_is_quiet_on_v3(tmp_path, capsys):
    v2 = _v2_of(base_doc())
    document = tmp_path / "old.json"
    document.write_text(json.dumps(v2))
    assert main(["migrate", "--check", str(document)]) == 1
    assert "would migrate" in capsys.readouterr().out
    assert main(["migrate", str(document)]) == 0
    assert json.loads(document.read_text()) == base_doc()
    assert main(["migrate", "--check", str(document)]) == 0
    assert parse_document(json.loads(document.read_text())).sites["tgt"].layers == (3,)
