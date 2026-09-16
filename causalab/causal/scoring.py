"""One immutable ``ScoringSpec`` per task — the task's definition of correct.

A task used to say what a correct answer *is* in eleven places across four
layers: the ``output_tokens`` forms map, the
per-variable mode map beside it (the legacy constructor kwargs, now derived
views), the plain instance attributes both were stored in, the
string checker ``derive_checker`` built from them, a bespoke ``checker.py``
that silently won over it, the probability path's ``form_groups``, the
serialized ``*_forms`` columns, an advisory copy of the mode in the
manifest, the protocol's own two-member ``MATCH_MODES``, the ``prefix`` →
``first_token`` bridge living in a parse-error string, and the token-id
resolvers with their refusals. Representations that live apart can disagree,
and two of them did: the string checker graded an undeclared expected value
by literal match while the serializer refused to write it, and the task's
``prefix`` (whole-string ``startswith``) was related to the protocol's
``first_token`` (argmax at one position) by prose alone.

This module is the one source. A :class:`ScoringSpec` holds

* ``forms`` — ``{variable: {value: (surface form, …)}}``, the word the
  ``output_tokens`` declaration always used (never "labels": that word already
  names five things in this tree);
* ``answer_variable`` — which declared variable the graded string
  (``raw_output``) is a form of. Required when several variables declare
  forms: MCQA declares ``answer`` (the letter the model emits) *and*
  ``answer_position`` (the variable an interchange targets), and grading
  against the wrong one is exactly the disagreement this object removes;
* ``string_mode`` — ``exact`` | ``prefix`` (:data:`STRING_MODES`): whether a
  generated string must equal a form or merely start with one. One mode per
  task, not per variable: no shipped task mixes them, and a per-variable map
  was one of the two sources the spec retires;
* ``protocol_mode`` — **derived**, never authored: the ``mode`` a ``match``
  metric over a table built from this spec declares, by the §2.10 translation
  table :data:`PROTOCOL_MODES` (``exact → exact``, ``prefix → first_token``).
  The bridge that used to be a diagnostic is now a field;
* ``full_string_checker`` — optional, a dotted locator
  (``package.module.function``) naming a bespoke
  ``checker(neural_output, causal_output) -> bool``. Declared *inside* the
  spec and digested (``checker_digest``, the sha256 of the module's source
  bytes — the same quantity :func:`causalab.protocol.code.source_sha256`
  hashes for a ``code`` declaration), so a bespoke grader stops being an
  unversioned override: editing it moves the spec's digest, and a grader
  whose source moved after the spec was built is refused, not run;
* ``undeclared_value`` — ``refuse`` | ``literal``
  (:data:`UNDECLARED_VALUE_POLICIES`): what :meth:`ScoringSpec.grade` does
  when the *expected* value names no declared form. ``refuse`` (the default)
  is what the serializer always did — the answer space and the declaration
  disagree, which would silently mis-score a metric; ``literal`` is what
  ``derive_checker`` did, kept as something a task states rather than
  inherits;
* ``invalid_output`` — ``incorrect`` | ``unscored``
  (:data:`INVALID_OUTPUT_POLICIES`): the grade of a *generated* string that
  matches no declared value at all. ``incorrect`` (the default) is a plain
  ``0.0`` — an off-answer-space generation is a wrong answer under a 0/1
  indicator; ``unscored`` returns ``None``, the value a metric row carries
  when "the model never said it" has to stay distinguishable from "it said it
  and scored 0". Neither is a refusal: what a model emits is data, and a run
  that refused over it would be a refusal firing on legitimate work;
* ``version`` — an integer, starting at 1, bumped when the *meaning* of a
  task's correctness changes under an unchanged declaration;
* ``digest`` — **derived**: the sha256 of the canonical JSON of every other
  field. Two specs that agree on every field share a digest; two that differ
  anywhere do not.

The spec is a frozen dataclass and its mappings are read-only views, so the
live defect — a ``CausalModel`` whose ``output_tokens`` could be reassigned
after validation with nothing re-validating or re-digesting it — cannot
recur: :class:`~causalab.causal.causal_model.CausalModel` stores the spec
and exposes ``output_tokens`` and the legacy mode map as read-only derivations
of it. Legacy interfaces compile *into* the spec; nothing stands beside it.

**Where the identity lives.** The serializer writes two constant per-row
columns into every table it builds (:data:`SCORING_DIGEST_COLUMN`,
:data:`STRING_MODE_COLUMN`), so the dataset content digest (§2.2, §7) covers
the scoring identity for free and a table rebuilt under a changed spec is a
different dataset. :func:`check_scoring` compares a document's ``match``
``mode`` against the table's recorded ``string_mode`` under the translation
table — in the ``validate --data`` pass and again before the first forward —
and refuses a contradiction. A table built before the columns existed is
*unrecorded*: it loads and runs exactly as before, and the run receipt says
so. Nothing here touches the canonical form, so no pinned digest moves.

Torch-free and stdlib-only on purpose: the loader, the serializer and the
protocol's ``validate --data`` pass all import it, and none of them may pay
for numerics.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "GRADE_RECORD_IDENTITY",
    "INVALID_OUTPUT_POLICIES",
    "PROTOCOL_MODES",
    "SCORING_DIGEST_COLUMN",
    "SCORING_FIELDS",
    "SCORING_RESULTS",
    "STRING_MODES",
    "STRING_MODE_COLUMN",
    "UNDECLARED_VALUE_POLICIES",
    "ScoringCheck",
    "ScoringError",
    "ScoringMismatch",
    "ScoringSpec",
    "check_scoring",
    "declared_modes",
    "table_scoring",
]

#: How a generated *string* is compared to a declared form (the task side of
#: §2.10's translation table): ``exact`` — the stripped string equals a form;
#: ``prefix`` — it starts with one (the continuation tokens a
#: ``max_new_tokens > 1`` task emits after the answer).
STRING_MODES: tuple[str, ...] = ("exact", "prefix")

#: §2.10's translation table, as a type: the ``mode`` a ``match`` metric
#: declares over a table built from a task with the given ``string_mode``.
#: With logits at one position, a prefix is the answer's first token. Total
#: in both directions — every ``STRING_MODES`` member maps, and the image is
#: exactly the protocol's ``MATCH_MODES`` (``schema.py``); the census in
#: ``tests/protocol/test_vocabulary_census.py`` holds all three to the spec's
#: table.
PROTOCOL_MODES: Mapping[str, str] = MappingProxyType(
    {"exact": "exact", "prefix": "first_token"}
)

#: What :meth:`ScoringSpec.grade` does with an expected value that names no
#: declared form (the module docstring).
UNDECLARED_VALUE_POLICIES: tuple[str, ...] = ("refuse", "literal")

#: The grade of a generated string that matches no declared value at all
#: (the module docstring).
INVALID_OUTPUT_POLICIES: tuple[str, ...] = ("incorrect", "unscored")

#: Every field of a :class:`ScoringSpec`, authored and derived, in the order
#: the ``causalab/tasks/README.md`` fields table lists them. The digest is
#: over every field but ``digest`` itself.
SCORING_FIELDS: tuple[str, ...] = (
    "forms",
    "answer_variable",
    "string_mode",
    "protocol_mode",
    "full_string_checker",
    "checker_digest",
    "undeclared_value",
    "invalid_output",
    "version",
    "digest",
)

#: The two constant per-row columns a serialized table carries when it was
#: built from a task with a spec (``causalab/tasks/serialize.py``): the
#: spec's digest and its ``string_mode``. Reserved column names.
SCORING_DIGEST_COLUMN = "scoring_digest"
STRING_MODE_COLUMN = "string_mode"

#: What :func:`check_scoring` reports into the run receipt: ``ok`` — the
#: table records its scoring identity and every ``match`` mode agrees with
#: it; ``unrecorded`` — the table predates the columns, nothing was compared.
#: A contradiction is a refusal, never a result.
SCORING_RESULTS: tuple[str, ...] = ("ok", "unrecorded")

#: The identity a metric row carries when a decoded string is graded through
#: a spec (``outputs.MetricTable``, spec §2.10): a 0/1 indicator in the
#: ``fraction`` unit — the same unit ``match`` produces — under its own
#: arithmetic name, because grading a decoded *string* is not the argmax
#: comparison ``match/v1`` names. This is the path a genuinely multi-token
#: answer takes without ``first_token``'s lossiness: ``decode``
#: reads the tokens the model produced, the spec grades the text.
GRADE_RECORD_IDENTITY: Mapping[str, str] = MappingProxyType(
    {"unit": "fraction", "estimand_version": "string_grade/v1"}
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ScoringError(ValueError):
    """A spec is malformed, or a grade cannot be decided honestly."""


class ScoringMismatch(ScoringError):
    """A document's ``match`` ``mode`` contradicts the table's recorded
    ``string_mode`` under the translation table. Carries the metric's name so
    the caller can point its refusal at ``metrics.<name>.mode``."""

    def __init__(self, message: str, *, metric: str) -> None:
        super().__init__(message)
        self.metric = metric


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #


def _hashable(value: Any) -> Any:
    """A declared value as a mapping key: a list becomes a tuple, the way the
    serializer always keyed a trace's list-valued setting."""
    return tuple(value) if isinstance(value, list) else value


def _freeze_forms(
    forms: Mapping[str, Mapping[Any, Sequence[str]]],
) -> Mapping[str, Mapping[Any, tuple[str, ...]]]:
    """Fail loud on a malformed forms map, then return it read-only.

    Guards against a silently-wrong-shape footgun: a
    malformed map used to score against the wrong tokens with no error. The
    declaration is the single source of truth for matching, so a malformed
    map must raise at construction, not surface as a mis-score deep in
    scoring. Two values of one variable whose string spellings coincide are
    refused too: the string grader keys the expected value by its spelling,
    so they could never be told apart.
    """
    if not isinstance(forms, Mapping) or not forms:
        raise ScoringError(
            "forms must be a non-empty mapping keyed by variable "
            "(e.g. {'weekday': {'Monday': [' Monday', 'Monday']}}), "
            f"got {forms!r}."
        )
    frozen: dict[str, Mapping[Any, tuple[str, ...]]] = {}
    for var, var_map in forms.items():
        if not isinstance(var, str) or not var:
            raise ScoringError(f"forms is keyed by variable name, got {var!r}.")
        if not isinstance(var_map, Mapping) or not var_map:
            raise ScoringError(
                f"forms[{var!r}] must be a non-empty {{value: [forms]}} mapping, "
                f"got {var_map!r}."
            )
        by_str: dict[str, Any] = {}
        var_frozen: dict[Any, tuple[str, ...]] = {}
        for value, value_forms in var_map.items():
            if isinstance(value_forms, str) or not isinstance(value_forms, Sequence):
                raise ScoringError(
                    f"forms[{var!r}][{value!r}] must be a list of surface "
                    f"forms, got {value_forms!r}."
                )
            if not all(isinstance(f, str) for f in value_forms):
                raise ScoringError(
                    f"forms[{var!r}][{value!r}] must be a list[str] of surface "
                    f"forms, got {list(value_forms)!r}."
                )
            if not value_forms:
                raise ScoringError(
                    f"forms[{var!r}][{value!r}] declares no forms; every value "
                    "needs at least one surface form."
                )
            if not all(f.strip() for f in value_forms):
                raise ScoringError(
                    f"forms[{var!r}][{value!r}] has an empty/whitespace-only "
                    f"form {list(value_forms)!r}; a blank form matches nothing "
                    "(exact) and tokenizes to no ids."
                )
            key = _hashable(value)
            spelled = str(key).strip()
            if spelled in by_str and by_str[spelled] != key:
                raise ScoringError(
                    f"forms[{var!r}] declares {by_str[spelled]!r} and {key!r}, "
                    f"which both spell {spelled!r} — the string grader could "
                    "never tell them apart."
                )
            by_str[spelled] = key
            var_frozen[key] = tuple(value_forms)
        frozen[var] = MappingProxyType(var_frozen)
    return MappingProxyType(frozen)


def _enum(value: Any, allowed: Sequence[str], field: str) -> str:
    if value not in allowed:
        raise ScoringError(f"{field} must be one of {tuple(allowed)}, got {value!r}.")
    return str(value)


def _locate_checker(locator: str) -> tuple[str, str, Path]:
    """``(module, function, source path)`` for a ``full_string_checker``
    locator, **without importing the module**.

    Only the module's *parents* are imported (``find_spec``'s documented
    behaviour), the module file itself is parsed, not executed, so a
    declaration is checked without running the grader it names. The function
    has to be a top-level ``def`` of that module: a checker reached through an
    attribute chain or a re-export is a checker whose source bytes this spec
    could not vouch for.
    """
    if not isinstance(locator, str) or "." not in locator:
        raise ScoringError(
            f"full_string_checker must be a dotted locator 'package.module.function', "
            f"got {locator!r}."
        )
    module_name, _, function = locator.rpartition(".")
    if not all(part.isidentifier() for part in (*module_name.split("."), function)):
        raise ScoringError(f"{locator!r} is not a dotted importable name.")
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError) as err:
        raise ScoringError(
            f"full_string_checker {locator!r}: {module_name!r} does not resolve: {err}"
        ) from err
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        raise ScoringError(
            f"full_string_checker {locator!r}: {module_name!r} is not a Python "
            "source module — a checker is identified by its source bytes, so "
            "there has to be some."
        )
    path = Path(spec.origin)
    tree = ast.parse(path.read_text(), filename=str(path))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if function not in defined:
        raise ScoringError(
            f"full_string_checker {locator!r}: {path} defines no top-level "
            f"function {function!r} (has {sorted(defined) or 'none'}); expected "
            "checker(neural_output, causal_output) -> bool."
        )
    return module_name, function, path


def _source_sha256(path: Path) -> str:
    """The sha256 of a Python source file's bytes — the same quantity
    :func:`causalab.protocol.code.source_sha256` hashes, computed here so this
    module stays stdlib-only."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# the spec
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ScoringSpec:
    """The task's definition of correct, once (the module docstring).

    Construct with the authored fields; ``protocol_mode``, ``checker_digest``
    and ``digest`` are derived here and cannot be passed. A frozen dataclass
    over read-only mappings: assignment to any field raises, and the derived
    views a :class:`~causalab.causal.causal_model.CausalModel` exposes are
    fresh copies, so nothing downstream can edit the declaration.
    """

    forms: Mapping[str, Mapping[Any, tuple[str, ...]]]
    answer_variable: str | None = None
    string_mode: str = "exact"
    full_string_checker: str | None = None
    undeclared_value: str = "refuse"
    invalid_output: str = "incorrect"
    version: int = 1
    protocol_mode: str = dataclasses.field(init=False)
    checker_digest: str | None = dataclasses.field(init=False)
    digest: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        put = object.__setattr__  # the one place a frozen field is written
        put(self, "forms", _freeze_forms(self.forms))
        variables = list(self.forms)
        answer = self.answer_variable
        if answer is None:
            if len(variables) != 1:
                raise ScoringError(
                    f"answer_variable is required when several variables declare "
                    f"forms ({variables}): it names the variable the graded "
                    "string (raw_output) is a form of."
                )
            answer = variables[0]
        if answer not in self.forms:
            raise ScoringError(
                f"answer_variable {answer!r} declares no forms (declared: {variables})."
            )
        put(self, "answer_variable", answer)
        put(self, "string_mode", _enum(self.string_mode, STRING_MODES, "string_mode"))
        put(self, "protocol_mode", PROTOCOL_MODES[self.string_mode])
        put(
            self,
            "undeclared_value",
            _enum(self.undeclared_value, UNDECLARED_VALUE_POLICIES, "undeclared_value"),
        )
        put(
            self,
            "invalid_output",
            _enum(self.invalid_output, INVALID_OUTPUT_POLICIES, "invalid_output"),
        )
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ScoringError(f"version must be a positive int, got {self.version!r}.")
        if self.full_string_checker is None:
            put(self, "checker_digest", None)
        else:
            _module, _function, path = _locate_checker(self.full_string_checker)
            put(self, "checker_digest", _source_sha256(path))
        put(self, "digest", _digest(self.identity()))

    # -- identity ---------------------------------------------------------- #

    def identity(self) -> dict[str, Any]:
        """Every field but ``digest``, as plain JSON — what the digest is over.
        Values are keyed by their string spelling (unique per variable by
        construction), so a tuple-keyed declaration digests like any other."""
        return {
            "forms": {
                var: {
                    str(value).strip(): list(forms) for value, forms in var_map.items()
                }
                for var, var_map in self.forms.items()
            },
            "answer_variable": self.answer_variable,
            "string_mode": self.string_mode,
            "protocol_mode": self.protocol_mode,
            "full_string_checker": self.full_string_checker,
            "checker_digest": self.checker_digest,
            "undeclared_value": self.undeclared_value,
            "invalid_output": self.invalid_output,
            "version": self.version,
        }

    # -- derivations ------------------------------------------------------- #

    def form_groups(self, variable: str | None = None) -> list[list[str]]:
        """Distinct, order-stable form groups across a variable's values — the
        probability path's groups.

        Values whose form lists are identical collapse to one group: many
        ``(entity, group)`` tuples sharing one entity's forms are one score
        token, which is the dedup ``output_token_values`` used to encode by
        hand. Formerly ``causal_utils.form_groups``; a derivation of the spec
        now, so it cannot disagree with the string grader.
        """
        groups: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for forms in self.forms[self._variable(variable)].values():
            if forms in seen:
                continue
            seen.add(forms)
            groups.append(list(forms))
        return groups

    def forms_of(self, value: Any, *, variable: str | None = None) -> tuple[str, ...]:
        """The declared surface forms of one value of ``variable`` (default:
        the ``answer_variable``) — the answer-form group a serialized row
        carries and the group the probability path scores.

        One resolution rule, shared with :meth:`grade`: by identity (the
        declared value itself), by the value's spelling, or by being one of its
        forms — so a trace's ``raw_output`` string resolves to the value it
        spells. A **list** is a list of acceptable answers (graph_walk's
        ``raw_output`` is every valid next node), and resolves to the union of
        its members' forms, in order — unless the list *is* a declared
        tuple-keyed value read back from JSON, where a tuple is a list; a tuple
        is a value (the ``(entity, group)`` keys of a grouped declaration). A
        value that names no declared
        form is decided by ``undeclared_value``: ``refuse`` raises, ``literal``
        returns the stripped string as its own one form.
        """
        var = self._variable(variable)
        found = self._resolve(var, value)
        if found is not None:
            return found
        spelled = str(value).strip()
        if self.undeclared_value == "literal":
            return (spelled,) if spelled else ()
        raise ScoringError(
            f"value {value!r} names no declared form of {var!r} — the answer "
            "space and the declaration disagree, which would silently mis-score "
            "a grade. Declare the value's forms, or declare "
            "undeclared_value='literal' to grade it as the string it is."
        )

    def grade(
        self, generated: str, expected: Any, *, variable: str | None = None
    ) -> float | None:
        """The grade of one generated string against one expected value:
        ``1.0`` correct, ``0.0`` incorrect, ``None`` unscored
        (``invalid_output: unscored`` and the string names no declared value).

        ``expected`` is a declared value of ``variable`` (default: the
        ``answer_variable``) or its string — ``raw_output``, a ``label`` — and
        is resolved to that value's forms: by identity, by the value's
        spelling, or by being one of its forms (graph_walk's coordinate tuples
        are spelled nothing like the concept the model emits; the concept is a
        form). An expected string that names no value is decided by
        ``undeclared_value``. Forms are compared stripped: leading-space forms
        exist for BPE tokenization, not for string matching, so the mechanical
        ``[" v", v]`` map collapses to the value while task-declared synonyms
        stay distinct alternatives. With a ``full_string_checker`` the
        bespoke function decides, and nothing else here applies.
        """
        var = self._variable(variable)
        if self.full_string_checker is not None:
            return float(bool(self._checker()({"string": generated}, str(expected))))
        actual = str(generated).strip()
        if self._hit(actual, self._targets(var, expected)):
            return 1.0
        declared = any(
            self._hit(actual, self._stripped(forms))
            for forms in self.forms[var].values()
        )
        if declared or self.invalid_output == "incorrect":
            return 0.0
        return None

    def grader(self, variable: str | None = None) -> Callable[[dict, str], bool]:
        """The string grader in the task-checker shape ``checker(neural_output,
        causal_output) -> bool`` — what ``Task.checker`` is, and what
        a per-example ``grade`` is computed with. ``True`` iff :meth:`grade`
        is ``1.0``."""
        var = self._variable(variable)

        def checker(neural_output: Mapping[str, Any], causal_output: Any) -> bool:
            return (
                self.grade(neural_output["string"], causal_output, variable=var) == 1.0
            )

        checker.__doc__ = f"ScoringSpec {self.digest[:12]}… grader over {var!r}"
        return checker

    # -- internals --------------------------------------------------------- #

    def _variable(self, variable: str | None) -> str:
        var = self.answer_variable if variable is None else variable
        if var not in self.forms:
            raise ScoringError(
                f"variable {var!r} declares no forms (declared: {list(self.forms)})."
            )
        assert var is not None
        return var

    @staticmethod
    def _stripped(forms: Sequence[str]) -> list[str]:
        return [f.strip() for f in forms if f.strip()]

    def _hit(self, actual: str, targets: Sequence[str]) -> bool:
        if self.string_mode == "prefix":
            return any(actual.startswith(t) for t in targets)
        return any(actual == t for t in targets)

    def _targets(self, variable: str, expected: Any) -> list[str]:
        """The stripped forms ``expected`` names under ``variable``."""
        return self._stripped(self.forms_of(expected, variable=variable))

    def _resolve(self, variable: str, value: Any) -> tuple[str, ...] | None:
        """The declared forms ``value`` names, or ``None`` when it names none
        (:meth:`forms_of` for the rule)."""
        var_map = self.forms[variable]
        if isinstance(value, list) and tuple(value) in var_map:
            # a tuple-keyed value read back from JSON, where a tuple is a list
            return var_map[tuple(value)]
        if isinstance(value, list):
            # a list of acceptable answers: every member resolves, or none does
            union: list[str] = []
            for member in value:
                forms = self._resolve(variable, member)
                if forms is None:
                    return None
                union.extend(f for f in forms if f not in union)
            return tuple(union) if union else None
        key = _hashable(value)
        try:
            if key in var_map:
                return var_map[key]
        except TypeError:
            pass  # an unhashable value has no identity lookup
        spelled = str(value).strip()
        candidates: list[tuple[str, ...]] = [
            forms
            for declared, forms in var_map.items()
            if str(declared).strip() == spelled or spelled in self._stripped(forms)
        ]
        if not candidates:
            return None
        distinct = {tuple(self._stripped(forms)) for forms in candidates}
        if len(distinct) > 1:
            raise ScoringError(
                f"value {value!r} is a form of several values of {variable!r} "
                f"with different form lists ({sorted(distinct)}) — ambiguous."
            )
        return candidates[0]

    def _checker(self) -> Callable[[Mapping[str, Any], Any], bool]:
        """Import the declared checker, refusing if its source moved since
        the spec was built — a table stamped with this spec's digest would
        otherwise be graded by code the digest does not name."""
        assert self.full_string_checker is not None
        module_name, function, path = _locate_checker(self.full_string_checker)
        now = _source_sha256(path)
        if now != self.checker_digest:
            raise ScoringError(
                f"full_string_checker {self.full_string_checker!r}: the source at "
                f"{path} digests {now[:12]}… but this spec recorded "
                f"{str(self.checker_digest)[:12]}… — the checker moved after the "
                "spec was built. Rebuild the spec (and any table stamped with it)."
            )
        return getattr(importlib.import_module(module_name), function)


def _digest(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# the table-side identity, compared at load and before the first forward
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ScoringCheck:
    """What :func:`check_scoring` found for one table: the identity it
    records (``None`` when unrecorded) and the result — the run receipt's
    ``scoring`` block, per ref."""

    digest: str | None
    string_mode: str | None
    result: str

    def as_record(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "string_mode": self.string_mode,
            "result": self.result,
        }


def table_scoring(rows: Sequence[Mapping[str, Any]]) -> tuple[str | None, str | None]:
    """``(scoring_digest, string_mode)`` a table's rows record, or
    ``(None, None)`` for an unrecorded table.

    The two columns are constant by construction, so rows that disagree — or
    a table carrying one column without the other — are a malformed table,
    refused rather than read as whichever row came first.
    """
    digests = {row.get(SCORING_DIGEST_COLUMN) for row in rows}
    modes = {row.get(STRING_MODE_COLUMN) for row in rows}
    if not rows or (digests == {None} and modes == {None}):
        return None, None
    if len(digests) != 1 or len(modes) != 1:
        raise ScoringError(
            f"the table's rows disagree on their scoring identity "
            f"({SCORING_DIGEST_COLUMN}: {sorted(map(str, digests))}, "
            f"{STRING_MODE_COLUMN}: {sorted(map(str, modes))}) — the two columns "
            "are constant per table by construction."
        )
    (digest,) = digests
    (mode,) = modes
    if digest is None or mode is None:
        raise ScoringError(
            f"the table records {SCORING_DIGEST_COLUMN}={digest!r} with "
            f"{STRING_MODE_COLUMN}={mode!r} — a recorded table carries both."
        )
    if not isinstance(digest, str) or not _HEX64.match(digest):
        raise ScoringError(
            f"{SCORING_DIGEST_COLUMN} {digest!r} is not a sha256 hex digest."
        )
    if mode not in STRING_MODES:
        raise ScoringError(
            f"{STRING_MODE_COLUMN} {mode!r} is not one of {STRING_MODES}."
        )
    return digest, str(mode)


def declared_modes(metrics: Mapping[str, Any]) -> dict[str, str]:
    """``{metric name: mode}`` for every ``match`` metric of a *point*
    document — the modes :func:`check_scoring` compares. Duck-typed on
    ``kind`` / ``fields`` so this module needs no import from the protocol
    layer; a swept ``mode`` is a scalar by the time a point exists."""
    out: dict[str, str] = {}
    for name, metric in metrics.items():
        if str(getattr(metric, "kind", "")) != "match":
            continue
        fields = getattr(metric, "fields", {})
        mode = fields.get("mode") if isinstance(fields, Mapping) else None
        if isinstance(mode, str):
            out[name] = mode
    return out


def check_scoring(
    rows: Sequence[Mapping[str, Any]],
    modes: Mapping[str, str],
    *,
    where: str,
) -> ScoringCheck:
    """Compare the ``match`` modes a document declares against the scoring
    identity the table records, under the §2.10 translation table.

    A ``prefix`` table — the task's answers are not single-token, so its
    ``string_mode`` derives to ``first_token`` — under a metric declaring
    ``mode: exact`` is a contradiction: the document says the answer is one
    token and the table says it is not. Refused, naming the table's mode, the
    metric's mode and the derivation. An ``exact`` table under
    ``first_token`` is **not** refused here: ``first_token`` is a strict
    generalization of ``exact`` on single-token answers, and the half that
    can go wrong — an exact table whose forms are multi-token, so two answers
    share a first token — is already refused where the ids are resolved
    (``metrics._refuse_indistinct_first_tokens``), which this deliberately
    does not duplicate. An unrecorded table compares nothing and says so.
    """
    digest, table_mode = table_scoring(rows)
    if table_mode is None:
        return ScoringCheck(None, None, "unrecorded")
    derived = PROTOCOL_MODES[table_mode]
    for name, declared in modes.items():
        if table_mode == "prefix" and declared != derived:
            raise ScoringMismatch(
                f"metric {name!r} declares mode {declared!r}, but the table "
                f"{where!r} records string_mode {table_mode!r}, which is "
                f"{derived!r} for a match metric (§2.10: exact → exact, prefix → "
                f"first_token; with logits at one position a prefix is the "
                f"answer's first token). Declare mode {derived!r}, or build the "
                "table from a task whose string_mode is 'exact'.",
                metric=name,
            )
    return ScoringCheck(digest, table_mode, "ok")
