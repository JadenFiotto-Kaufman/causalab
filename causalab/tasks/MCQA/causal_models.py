"""Causal models for the MCQA task.

This module defines the causal model structure for multiple choice question answering,
including variables, values, parent relationships, and mechanisms.
"""

from causalab.causal.causal_model import CausalModel, build_output_tokens
from causalab.causal.scoring import ScoringSpec
from causalab.causal.trace import CausalTrace, Mechanism, input_var

# Constants
OBJECTS = [
    "ball",
    "car",
    "house",
    "shirt",
    "flower",
    "pen",
    "cup",
    "hat",
    "bag",
    "shoe",
]
COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "brown",
    "black",
    "white",
]

NUM_CHOICES = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPLATES = [
    "The {object} is {color}. What color is the {object}?"
    + "".join([f"\n{{symbol{str(i)}}}. {{choice{str(i)}}}" for i in range(NUM_CHOICES)])
    + "\nAnswer:"
]


# Causal Model Definition
values: dict[str, str | list[str] | list[int] | None] = {}
values.update({"choice" + str(x): COLORS for x in range(NUM_CHOICES)})
values.update({"symbol" + str(x): list(ALPHABET) for x in range(NUM_CHOICES)})
values.update({"answer_position": list(range(NUM_CHOICES)), "answer": list(ALPHABET)})
values.update({"template": TEMPLATES})
values.update({"object": OBJECTS, "color": COLORS})
values.update({"raw_input": None, "raw_output": None})


def _fill_template(t: CausalTrace) -> str:
    """Fill in the template with object, color, symbols, and choices."""
    template = t["template"]
    object_name = t["object"]
    color = t["color"]

    filled_template = template.replace("{object}", object_name).replace(
        "{color}", color
    )
    for i in range(NUM_CHOICES):
        filled_template = filled_template.replace(f"{{symbol{i}}}", t[f"symbol{i}"])
    for i in range(NUM_CHOICES):
        filled_template = filled_template.replace(f"{{choice{i}}}", t[f"choice{i}"])
    return filled_template


def _get_answer_position(t: CausalTrace) -> int:
    """Determine which choice position contains the correct answer."""
    color = t["color"]
    choices = [t[f"choice{i}"] for i in range(NUM_CHOICES)]
    for i in range(NUM_CHOICES):
        if choices[i] == color:
            return i
    raise ValueError(
        f"No correct answer position found for color {color} in choices {choices}"
    )


def _get_answer(t: CausalTrace) -> str:
    """Get the symbol corresponding to the correct answer position."""
    answer_position = t["answer_position"]
    return t[f"symbol{answer_position}"]


# Define mechanisms using the new Mechanism API
mechanisms = {
    # Input variables (no parents)
    "template": input_var(TEMPLATES),
    "object": input_var(OBJECTS),
    "color": input_var(COLORS),
    **{f"symbol{i}": input_var(list(ALPHABET)) for i in range(NUM_CHOICES)},
    **{f"choice{i}": input_var(COLORS) for i in range(NUM_CHOICES)},
    # Computed variables
    "raw_input": Mechanism(
        parents=(
            ["template", "object", "color"]
            + ["symbol" + str(x) for x in range(NUM_CHOICES)]
            + ["choice" + str(x) for x in range(NUM_CHOICES)]
        ),
        compute=_fill_template,
    ),
    "answer_position": Mechanism(
        parents=(["color"] + ["choice" + str(x) for x in range(NUM_CHOICES)]),
        compute=_get_answer_position,
    ),
    "answer": Mechanism(
        parents=(["answer_position"] + ["symbol" + str(x) for x in range(NUM_CHOICES)]),
        compute=_get_answer,
    ),
    "raw_output": Mechanism(
        parents=["answer"],
        compute=lambda t: " " + t["answer"],
    ),
}

# The task's definition of correct, once. Two variables declare forms:
# ``answer`` (the option *letter* — what ``raw_output`` is, ``" " + answer``)
# and ``answer_position`` (the module ``TARGET_VARIABLE``, the variable an
# interchange targets). ``answer_variable="answer"`` says which one the
# generated string is graded against: the former derived checker was keyed on
# ``answer_position``, whose forms are the digits ``" 0"`` / ``" 1"`` the model
# never emits, and only a literal-match fallback on ``raw_output`` made it
# grade letters at all — while the serialized answer forms for the same
# example were the digits. That was the disagreement the scoring differential
# (``tests/tasks/test_scoring_differential.py``) exists to refuse. The retired
# ``score_by: value`` convention (a colour word accepted in place of the
# letter) belonged to a runner that no longer exists; a colour word is now an
# undeclared value and the grader refuses it rather than crediting it.
positional_causal_model = CausalModel(
    mechanisms,
    values,
    id=f"{NUM_CHOICES}_answer_MCQA",
    scoring=ScoringSpec(
        forms={
            "answer": build_output_tokens(list(ALPHABET)),
            "answer_position": build_output_tokens(list(range(NUM_CHOICES))),
        },
        answer_variable="answer",
    ),
)


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

CAUSAL_MODEL = positional_causal_model
TARGET_VARIABLE = "answer_position"
TEMPLATE = TEMPLATES[0]
