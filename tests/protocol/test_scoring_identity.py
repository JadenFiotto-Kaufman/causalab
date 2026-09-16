"""The scoring identity at load (spec §2.2, §2.10): the ``validate --data``
pass holds a document's ``match`` ``mode`` to the ``string_mode`` the base
table records, under the translation table.

Three tables, one document shape:

* a **recorded** table built from a ``prefix`` task
  (``subject_object_relations``) — under ``mode: first_token`` it loads and
  the check says ``ok``; under ``mode: exact`` it is refused pre-forward, as
  rule 4, naming the table's mode, the metric's mode and the derivation;
* a **recorded** table built from an ``exact`` task (weekdays) — ``exact`` is
  ``ok`` and so is ``first_token`` (a strict generalization; the over-crediting
  half is the metric's own refusal, ``test_metrics.py``);
* the **unrecorded** committed fixture table — the fail-closed twin: it loads,
  every corpus document over it validates, and the check says so.

Torch-free: tables are built by the serializer and read by ``FileDatasets``;
nothing here loads a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from causalab.causal.scoring import (
    SCORING_DIGEST_COLUMN,
    STRING_MODE_COLUMN,
    check_scoring,
    declared_modes,
)
from causalab.protocol.errors import ValidationError
from causalab.protocol.loader import check_data_columns, load
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.tasks.serialize import (
    serialize_counterfactual_dataset,
    write_dataset_table,
)
from causalab.tasks.subject_object_relations.config import SubjectObjectRelationsConfig

from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.unit

MODEL = "meta-llama/Llama-3.1-8B"
PREFIX_REF = "sor/name_gender"
EXACT_REF = "weekdays/recorded"


def _document(
    ref: str, mode: str, expected: str = "base_answer_forms"
) -> dict[str, Any]:
    return {
        "header": {
            "protocol_version": "3",
            "description": "base accuracy over one table, scored by the declared forms",
        },
        "model": {"key": MODEL, "revision": "main"},
        "data": {"base": {"dataset": ref, "field": "input"}},
        "method": {
            "positions": {"answer_tok": {"index": -1}},
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {
                "logits": {
                    "site": "lm_head",
                    "pos": "answer_tok",
                    "model": "original",
                    "input": "base",
                }
            },
            "metrics": {
                "accuracy": {
                    "kind": "match",
                    "of": "logits",
                    "expected": expected,
                    "mode": mode,
                    "token_form": "space_prefixed",
                }
            },
            "save": [
                {
                    "value": "accuracy",
                    "model": "original",
                    "input": "base",
                    "file_path": "accuracy.json",
                }
            ],
        },
    }


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> tuple[ResolutionEnv, dict[str, str]]:
    """Two recorded tables in a scratch data root: a ``prefix`` task's and an
    ``exact`` task's, each carrying its spec's digest in every row."""
    root = tmp_path_factory.mktemp("scoring_data")
    digests: dict[str, str] = {}
    for ref, task, cfg in (
        (
            PREFIX_REF,
            "subject_object_relations",
            SubjectObjectRelationsConfig(relation="name_gender"),
        ),
        (
            EXACT_REF,
            "natural_domains_arithmetic",
            NaturalDomainConfig(domain_type="weekdays"),
        ),
    ):
        dataset = serialize_counterfactual_dataset(
            task, n=6, seed=0, split="all", task_cfg=cfg
        )
        write_dataset_table(dataset.rows, root / f"{ref}.json")
        spec = dataset.rows[0][SCORING_DIGEST_COLUMN]
        digests[ref] = spec
    env = ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )
    return env, digests


def test_a_recorded_table_records_one_identity(recorded):
    env, digests = recorded
    for ref, expected_mode in ((PREFIX_REF, "prefix"), (EXACT_REF, "exact")):
        rows = env.datasets.rows(ref)
        assert {row[SCORING_DIGEST_COLUMN] for row in rows} == {digests[ref]}
        assert {row[STRING_MODE_COLUMN] for row in rows} == {expected_mode}
        assert SCORING_DIGEST_COLUMN in env.datasets.columns(ref)


def test_a_prefix_table_under_first_token_validates_and_is_ok(recorded):
    env, digests = recorded
    loaded = load(_document(PREFIX_REF, "first_token"), env)
    assert "base_answer_forms" in check_data_columns(loaded, env)
    (doc,) = loaded.point_documents
    check = check_scoring(
        env.datasets.rows(PREFIX_REF),
        declared_modes(doc.metrics),
        where=PREFIX_REF,
    )
    assert check.as_record() == {
        "digest": digests[PREFIX_REF],
        "string_mode": "prefix",
        "result": "ok",
    }


def test_a_prefix_table_under_exact_is_refused_naming_both_modes_and_the_derivation(
    recorded,
):
    """The contradiction: the document says the answer is one token, the
    table says it is not. Rule 4 — a reference that does not resolve — at the
    metric's ``mode``, before any weights."""
    env, _digests = recorded
    loaded = load(_document(PREFIX_REF, "exact"), env)  # the bare load is fine
    with pytest.raises(ValidationError) as err:
        check_data_columns(loaded, env)
    message = str(err.value)
    assert "[V4]" in message
    assert "'exact'" in message and "'prefix'" in message and "'first_token'" in message
    assert "prefix → first_token" in message
    assert "metrics.accuracy.mode" in message


def test_an_exact_table_is_ok_under_either_mode(recorded):
    """``first_token`` generalizes ``exact`` on single-token answers, so an
    exact table is not contradicted by it; the multi-token half is refused
    where the ids resolve (``_refuse_indistinct_first_tokens``), not here."""
    env, digests = recorded
    for mode in ("exact", "first_token"):
        loaded = load(_document(EXACT_REF, mode), env)
        check_data_columns(loaded, env)
        (doc,) = loaded.point_documents
        check = check_scoring(
            env.datasets.rows(EXACT_REF),
            declared_modes(doc.metrics),
            where=EXACT_REF,
        )
        assert check.result == "ok" and check.digest == digests[EXACT_REF]


def test_a_table_whose_rows_disagree_on_their_identity_is_refused(recorded, tmp_path):
    env, _digests = recorded
    rows = env.datasets.rows(PREFIX_REF)
    rows[0] = {**rows[0], STRING_MODE_COLUMN: "exact"}
    root = tmp_path / "broken"
    write_dataset_table(rows, root / "sor" / "name_gender.json")
    broken = ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )
    with pytest.raises(ValidationError) as err:
        check_data_columns(load(_document(PREFIX_REF, "first_token"), broken), broken)
    assert "[V4]" in str(err.value) and "rows disagree" in str(err.value)


# --------------------------------------------------------------------------- #
# the unrecorded twin: every committed table predates the columns
# --------------------------------------------------------------------------- #


def _corpus_documents() -> list[Path]:
    return sorted(CORPUS_DIR.glob("*_im.json"))


def test_the_committed_fixture_tables_are_unrecorded():
    """The fail-closed twin's premise: no shipped or fixture table was rebuilt
    since the recorded columns were introduced, so none carries them — and
    every one of them still loads."""
    root = FIXTURES / "data"
    for table in sorted(root.rglob("*.json")):
        rows = FileDatasets(root=root)._table(
            table.relative_to(root).with_suffix("").as_posix()
        )
        assert all(SCORING_DIGEST_COLUMN not in row for row in rows), table
        assert all(STRING_MODE_COLUMN not in row for row in rows), table


@pytest.mark.parametrize("document", _corpus_documents(), ids=lambda p: p.name)
def test_every_corpus_document_over_an_unrecorded_table_validates(document, env):
    """The load half over the corpus: ``validate --data`` passes on every
    document, and for each ``match`` metric the check reports ``unrecorded``
    — nothing compared, nothing refused. ``env`` is the corpus tests' own
    (``tests/protocol/conftest.py``: the fixture tables plus the generated
    artifact bundles)."""
    loaded = load(document, env)
    check_data_columns(loaded, env)
    for doc in loaded.point_documents:
        modes = declared_modes(doc.metrics)
        if not modes:
            continue
        ref = doc.data["base"].dataset
        assert isinstance(ref, str)
        check = check_scoring(env.datasets.rows(ref), modes, where=ref)
        assert check.result == "unrecorded" and check.digest is None
