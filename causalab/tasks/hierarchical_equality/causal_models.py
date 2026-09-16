"""
Causal model definitions for the task.

DAG: (var_1, var_2) → left_equality
     (var_3, var_4) → right_equality
     (left_equality, right_equality) → result_equality → raw_output
"""

from causalab.causal.causal_model import CausalModel
from causalab.causal.scoring import ScoringSpec
from causalab.causal.trace import Mechanism, input_var

from .config import LETTERS, TASK_NAME
from .templates import TEMPLATES, fill_template

values = {
    "template": TEMPLATES,
    "var_1": LETTERS,
    "var_2": LETTERS,
    "var_3": LETTERS,
    "var_4": LETTERS,
    "left_equality": [True, False],
    "right_equality": [True, False],
    "result_equality": [True, False],
    "raw_input": None,
    "raw_output": None,
}

mechanisms = {
    "template": input_var(TEMPLATES),
    "var_1": input_var(LETTERS),
    "var_2": input_var(LETTERS),
    "var_3": input_var(LETTERS),
    "var_4": input_var(LETTERS),
    "left_equality": Mechanism(
        parents=["var_1", "var_2"],
        compute=lambda t: t["var_1"] == t["var_2"],
    ),
    "right_equality": Mechanism(
        parents=["var_3", "var_4"],
        compute=lambda t: t["var_3"] == t["var_4"],
    ),
    "result_equality": Mechanism(
        parents=["left_equality", "right_equality"],
        compute=lambda t: t["left_equality"] == t["right_equality"],
    ),
    "raw_input": Mechanism(
        parents=["template", "var_1", "var_2", "var_3", "var_4"],
        compute=lambda t: fill_template(
            t["template"], t["var_1"], t["var_2"], t["var_3"], t["var_4"]
        ),
    ),
    "raw_output": Mechanism(
        parents=["result_equality"],
        compute=lambda t: "1" if t["result_equality"] else "0",
    ),
}

# All three equality variables have boolean values, but the model emits the
# digit "1" (True) or "0" (False). Declare those surface forms once, per value,
# in the task's one ``ScoringSpec``: the probability path reads
# them, and the grader uses ``string_mode="prefix"`` — ``raw_output`` is the
# bare digit possibly followed by text, so a generation that starts with it is
# correct, exactly as the former checker.py's ``startswith`` was. The graded
# string is ``result_equality``'s digit (``answer_variable``); the other two
# variables declare the same forms for the probability path.
_EQUALITY_VARS = ("left_equality", "right_equality", "result_equality")
_EQUALITY_FORMS: dict[object, list[str]] = {True: [" 1", "1"], False: [" 0", "0"]}

CAUSAL_MODEL = CausalModel(
    mechanisms,
    values,
    id=TASK_NAME,
    scoring=ScoringSpec(
        forms={v: dict(_EQUALITY_FORMS) for v in _EQUALITY_VARS},
        answer_variable="result_equality",
        string_mode="prefix",
    ),
)


# ---------------------------------------------------------------------------
# Exports for load_task()
# ---------------------------------------------------------------------------

TARGET_VARIABLE = "result_equality"
