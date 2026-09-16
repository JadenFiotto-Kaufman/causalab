"""The resolver census: snapshot before, diff after.

``fixtures/resolved_sites_census.json`` records, on the base the census was
pinned on *before any edit*, what ``resolve_site`` answered for every ``(component,
layer, head)`` the census walks on the three tiny fixtures — the resolved
site (module path, io side, feature slice, slot, declared shape, derivation)
or the refusal (class, code, reason, text). Moving ``resolve_site`` onto the
family adapters' declared taps is a refactor whose acceptance is that
**nothing changes**: this test re-walks the census on the current tree and
compares every entry to the record.

Two sets of entries differ by decision, and each is named here rather than
tolerated:

* the eight ``deltanet_*`` spellings that became aliases: a document
  naming one now resolves to its canonical name's tap, so the live answer
  must equal the record **of the canonical name** — redirect, not rebind;
* GPT-2's three norm taps (``attention_input_norm``, ``block_mid``,
  ``mlp_input_norm``): the record is a bare ``AttributeError`` (the resolver
  read the llama tree's child names on every non-GPT-2 tree — and on GPT-2
  too), the family adapter declares ``ln_1`` / ``ln_2``, and the live answer
  is the resolved tap.

Everything else — every other refusal text included — is byte-identical.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.schema import DEPRECATED_COMPONENTS

from tests.neural.engines.pytorch_hooks import update_resolved_sites_census as census

pytestmark = pytest.mark.smoke

RECORD = json.loads(census.CENSUS.read_text())

GPT2 = "hf-internal-testing/tiny-random-gpt2"

#: The decided changes: (fixture, component) → what the live answer must be.
#: GPT-2's norms: the record is an AttributeError on the llama child names.
GPT2_NORMS: dict[str, tuple[str, str]] = {
    "attention_input_norm": ("ln_1", "out"),
    "block_mid": ("ln_2", "in"),
    "mlp_input_norm": ("ln_2", "out"),
}


def _live(bundle: Any, component: str, layer: int | None, head: int | None) -> dict:
    try:
        site = resolve_site(bundle, census._site(component, layer, head))
    except Exception as exc:  # noqa: BLE001 - compared, not raised
        record = census._record_refusal(exc)
    else:
        record = {"resolved": census._record_site(bundle, site)}
    # through JSON, as the record went: tuples become lists
    return json.loads(json.dumps(record))


def _recorded(entries: list[dict], component: str, layer: Any, head: Any) -> dict:
    for entry in entries:
        if (entry["component"], entry["layer"], entry["head"]) == (
            component,
            layer,
            head,
        ):
            return {k: v for k, v in entry.items() if k in ("resolved", "refusal")}
    raise AssertionError(f"no recorded entry for {component}@{layer} head {head}")


def test_the_record_is_the_bases_and_not_empty():
    assert RECORD["base"] == "7c28470e"
    assert set(RECORD["fixtures"]) == set(census.FIXTURES)
    total = sum(len(f["entries"]) for f in RECORD["fixtures"].values())
    assert total >= 600  # 644 at the capture


@pytest.mark.parametrize("key", sorted(census.FIXTURES))
def test_the_resolver_answers_as_it_did_on_the_base(key: str):
    bundle = load_model(key)
    fixture = RECORD["fixtures"][key]
    assert bundle.info.family == fixture["family"]
    assert list(bundle.streams) == fixture["streams"]
    entries = fixture["entries"]
    mismatches: list[str] = []
    aliased = decided = 0
    for entry in entries:
        component, layer, head = entry["component"], entry["layer"], entry["head"]
        live = _live(bundle, component, layer, head)
        if component in DEPRECATED_COMPONENTS:
            # the alias redirects: the live answer is the canonical name's record
            expected = _recorded(entries, DEPRECATED_COMPONENTS[component], layer, head)
            aliased += 1
        elif key == GPT2 and component in GPT2_NORMS:
            child, side = GPT2_NORMS[component]
            recorded = entry["refusal"]
            assert recorded["exc_class"] == "AttributeError", recorded
            resolved = live.get("resolved")
            assert resolved is not None, (component, live)
            assert resolved["module"] == f"transformer.h.{layer}.{child}"
            assert resolved["kind"] == side
            decided += 1
            continue
        else:
            expected = {k: v for k, v in entry.items() if k in ("resolved", "refusal")}
        if live != expected:
            mismatches.append(
                f"{component}@{layer} head {head}:\n  recorded {expected}\n  live     {live}"
            )
    assert not mismatches, f"{len(mismatches)} entries moved:\n" + "\n".join(mismatches)
    if key == GPT2:
        assert decided == 3 * 2  # three norms × two layers walked
    assert aliased == sum(1 for e in entries if e["component"] in DEPRECATED_COMPONENTS)


def test_the_aliases_are_in_the_record_under_their_old_names():
    """Vacuity floor for the redirect check: the record walked the eight
    retired spellings (the base still had them in the vocabulary)."""
    entries = RECORD["fixtures"]["tiny-random/qwen3.5-moe"]["entries"]
    walked = {e["component"] for e in entries} & set(DEPRECATED_COMPONENTS)
    assert walked == set(DEPRECATED_COMPONENTS)  # the attention-era one included
    assert len(walked) == 9
