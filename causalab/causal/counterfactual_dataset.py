"""Counterfactual dataset types."""

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from causalab.causal.trace import CausalTrace


class _PairIdentity(TypedDict, total=False):
    example_id: str
    pair_id: str
    family: str
    base_id: str
    donor_id: str


class CounterfactualExample(_PairIdentity):
    """
    Type for counterfactual example dictionaries.

    Each example contains:
        input: The base input as a CausalTrace
        counterfactual_inputs: List of counterfactual inputs as CausalTraces
    """

    input: "CausalTrace"
    counterfactual_inputs: list["CausalTrace"]


class LabeledCounterfactualExample(CounterfactualExample):
    """
    Counterfactual example with a ground truth label for training.

    Used by DAS/DBM training functions to compute loss and accuracy.
    The label is the expected output after intervention.
    """

    label: Any
