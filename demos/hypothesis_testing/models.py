"""Addition modulo three with two candidate intermediate variables."""

from causalab.causal.causal_model import CausalModel
from causalab.causal.scoring import ScoringSpec
from causalab.causal.trace import Mechanism, input_var

values = {
    "a": [0, 1, 2],
    "b": [0, 1, 2],
    "context": None,
    "a_value": [0, 1, 2],
    "b_value": [0, 1, 2],
    "answer": [0, 1, 2],
    "raw_input": None,
    "raw_output": None,
}
mechanisms = {
    "a": input_var([0, 1, 2]),
    "b": input_var([0, 1, 2]),
    "context": input_var([0]),
    "a_value": Mechanism(parents=["a"], compute=lambda t: t["a"]),
    "b_value": Mechanism(parents=["b"], compute=lambda t: t["b"]),
    "answer": Mechanism(
        parents=["a_value", "b_value"],
        compute=lambda t: (t["a_value"] + t["b_value"]) % 3,
    ),
    "raw_input": Mechanism(
        parents=["a", "b", "context"],
        compute=lambda t: f"Case {t['context']}: ({t['a']} + {t['b']}) mod 3 =",
    ),
    "raw_output": Mechanism(parents=["answer"], compute=lambda t: str(t["answer"])),
}
MODELS = {
    "addition": CausalModel(
        mechanisms,
        values,
        id="addition",
        scoring=ScoringSpec(
            forms={"answer": {i: [str(i)] for i in range(3)}}, answer_variable="answer"
        ),
    )
}
DEFAULT_MODEL = "addition"
HYPOTHESES = {"swap_a": ("addition", ["a_value"]), "swap_b": ("addition", ["b_value"])}
TARGETS = ["swap_a"]
