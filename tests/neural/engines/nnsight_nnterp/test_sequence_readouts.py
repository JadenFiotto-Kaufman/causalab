"""Prepared multi-token predictions agree across both execution engines."""

import pytest
import torch

from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.sequences import pair_sequences, prepare_sequence
from causalab.protocol.schema import parse_document
from tests.neural.test_sequence_analysis import document, executor

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("patch", [False, True])
def test_prepared_targets_match_hooks(hooks_llama, nnterp_llama, patch):
    rows = [
        pair_sequences(
            prepare_sequence(
                hooks_llama.tokenizer,
                prompt,
                [100, 200, 300],
                example_id=f"base-{i}",
                split="eval",
            ),
            prepare_sequence(
                hooks_llama.tokenizer,
                "the green turtle",
                [200, 100, 300],
                example_id=f"donor-{i}",
                split="eval",
            ),
        )
        for i, prompt in enumerate(["the red fox", "a very small bird"])
    ]
    raw = document(hooks_llama, patch=patch)
    roles, fields = {"base": rows}, {"base": "input"}
    if patch:
        roles["counterfactual"] = rows
        fields["counterfactual"] = "counterfactual_inputs[0]"
    hooks = executor(hooks_llama, raw, rows)
    trace = NnterpExecutor(
        parse_document(raw),
        nnterp_llama,
        role_rows=roles,
        role_fields=fields,
        load_tensors=lambda path: None,
    )
    for target in range(3):
        torch.testing.assert_close(
            trace.dense_value(f"sequence_{target}"),
            hooks.dense_value(f"sequence_{target}"),
            atol=1e-5,
            rtol=0,
        )
