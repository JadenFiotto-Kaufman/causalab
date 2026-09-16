"""The row label (spec §2.2): ``example_id`` when authored, the index when not."""

import pytest

from causalab.protocol.examples import example_id_defect, example_labels

pytestmark = pytest.mark.unit


def test_a_table_without_the_column_is_labelled_by_index_as_strings():
    rows = [
        {"prompt": "a"},
        {"prompt": "b"},
        {"prompt": "a"},
    ]  # a repeated prompt is fine
    assert example_id_defect(rows) is None
    assert example_labels(rows) == ["0", "1", "2"]


def test_an_authored_column_labels_every_row_as_a_string():
    rows = [{"example_id": "w3", "prompt": "a"}, {"example_id": 7, "prompt": "b"}]
    assert example_id_defect(rows) is None
    assert example_labels(rows) == ["w3", "7"]


def test_an_empty_table_has_no_labels_and_no_defect():
    assert example_id_defect([]) is None
    assert example_labels([]) == []


@pytest.mark.parametrize(
    ("rows", "fragment"),
    [
        ([{"example_id": "a"}, {"prompt": "b"}], "carry no example_id"),
        ([{"example_id": "a"}, {"example_id": ""}], "row 1 has an empty example_id"),
        ([{"example_id": "a"}, {"example_id": None}], "row 1 has an empty example_id"),
        (
            [{"example_id": "a"}, {"example_id": "b"}, {"example_id": "a"}],
            "'a' labels rows 0 and 2",
        ),
        ([{"example_id": 1}, {"example_id": "1"}], "'1' labels rows 0 and 1"),
    ],
)
def test_a_column_that_cannot_label_its_rows_is_named(rows, fragment):
    defect = example_id_defect(rows)
    assert defect is not None and fragment in defect
