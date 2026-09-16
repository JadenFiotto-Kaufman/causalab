"""A row selection of an encoded frame (spec §4, "Cohorts").

A fit's minibatch used to be a fresh encode of its rows, so two minibatches
of one point — and the point's own whole-role batch — sat in three padding
frames. A cohort concatenates minibatches of several points into one forward,
which only works when every member is in the **same** frame; the minibatch is
therefore a row selection of the point's frame, and this pins the selection:
the same padded width, every per-row field sliced in step, and position
resolution unchanged.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.shared.encoding import EncodedBatch, encode

pytestmark = pytest.mark.unit


def _frame() -> EncodedBatch:
    return EncodedBatch(
        texts=("a b c", "d e", "f g h i"),
        input_ids=torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5], [6, 7, 8, 9]]),
        attention_mask=torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]),
        offset_mapping=(
            ((0, 0), (0, 1), (2, 3), (4, 5)),
            ((0, 0), (0, 0), (0, 1), (2, 3)),
            ((0, 1), (2, 3), (4, 5), (6, 7)),
        ),
        prefix_lengths=(0, 1, 0),
        segments=({"x": ((0, 1),)}, {"x": ()}, {"x": ((0, 1), (2, 3))}),
    )


def test_a_selection_keeps_the_frame_and_slices_every_row_field() -> None:
    frame = _frame()
    picked = frame.select((2, 0))
    assert picked.padded_len == frame.padded_len
    assert picked.texts == ("f g h i", "a b c")
    assert torch.equal(picked.input_ids, frame.input_ids[[2, 0]])
    assert torch.equal(picked.attention_mask, frame.attention_mask[[2, 0]])
    assert picked.offset_mapping == (frame.offset_mapping[2], frame.offset_mapping[0])
    assert picked.prefix_lengths == (0, 0)
    assert picked.segments == (frame.segments[2], frame.segments[0])
    # the per-row frame arithmetic reads the same numbers off the selection
    assert picked.first_real(1) == frame.first_real(0) == 1
    assert torch.equal(picked.position_ids(), frame.position_ids()[[2, 0]])


def test_a_selection_of_a_plain_frame_has_no_segments() -> None:
    frame = _frame()
    plain = EncodedBatch(
        texts=frame.texts,
        input_ids=frame.input_ids,
        attention_mask=frame.attention_mask,
        offset_mapping=frame.offset_mapping,
        prefix_lengths=frame.prefix_lengths,
    )
    assert plain.select((1,)).segments == ()


def test_select_covers_every_field_of_the_frame() -> None:
    """``select`` (and the cohort's ``_concat_frames``) enumerate the frame's
    fields by hand; a field added to ``EncodedBatch`` must be added there too,
    and this is what says so."""
    import dataclasses

    frame = _frame()
    picked = frame.select((1,))
    for field in dataclasses.fields(EncodedBatch):
        whole, part = getattr(frame, field.name), getattr(picked, field.name)
        if isinstance(whole, torch.Tensor):
            assert torch.equal(part, whole[[1]]), field.name
        else:
            assert part == (whole[1],), field.name


def test_an_empty_selection_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        _frame().select(())


def test_a_real_encode_selected_equals_the_wider_rows_of_the_frame() -> None:
    """Against a real tokenizer: selecting the two longer rows of a three-row
    frame keeps the frame's width, where a fresh encode of the same two rows
    would be narrower whenever the third row was the longest."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    tokenizer.padding_side = "left"
    texts = ["one two", "three", "four five six seven eight nine"]
    frame = encode(tokenizer, texts)
    picked = frame.select((0, 1))
    fresh = encode(tokenizer, texts[:2])
    assert picked.padded_len == frame.padded_len > fresh.padded_len
    # the real tokens are the same tokens, further left-padded
    for row in range(2):
        real = picked.input_ids[row][picked.attention_mask[row].bool()]
        assert torch.equal(real, fresh.input_ids[row][fresh.attention_mask[row].bool()])
