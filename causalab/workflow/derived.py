"""The manifest's status words, derived from the event stream (workflow spec
§4.3, §8).

``workflow.json`` is the run's *state* and ``events.jsonl`` is its *history*.
A manifest that could disagree with the history — a step marked ``completed``
whose publish the stream never recorded — is the disagreement this module
exists to rule out. So the status a step carries in ``workflow.json`` is
not the runner's memory of what it did: it is :func:`derive_statuses` over
the lines this run appended, and the runner refuses to write a manifest whose
in-memory status differs from the derived one (a disagreement there is an
emitter bug, or an interrupt that landed between a memory assignment and its
emit; either way the manifest is withheld, and an emitter bug surfaces as a
``ProtocolError`` before the write). That comparison is a consistency
assertion over a fact still stored twice — the entry's ``status`` in memory
and the stream — not the end state: its single-writer form is a future
``_step.json`` format change (a digest mover, so not this PR's).

The derivation is one pure function over the stream's own vocabulary
(:data:`causalab.io.events.EVENTS`): ``phase_completed {step, status}`` is the
step's terminal word (``completed``, ``reused`` or ``skipped``), ``warning
{step, reason: attempt_failed}`` is ``failed``, and every step of the schedule
with no terminal line is classified by the manifest's own rule
(:func:`causalab.workflow.manifest.classify_unreached`): ``skipped`` when an
upstream step is ``skipped``, ``blocked`` when one is ``failed`` or
``blocked``, else ``pending``. A step whose
only line is ``phase_started`` is ``pending`` — §8 has no word for "started",
and a run that stopped inside a step wrote no manifest (the manifest's
``failed`` is the ``warning`` the runner appends before it re-raises).

This lives beside the manifest rather than in :mod:`causalab.io.events`
because the derivation reuses ``classify_unreached`` and ``io/`` may not
import the workflow layer (``docs/CODEBASE.md`` invariant 1); the stream
module stays stdlib-only and in no script's import closure.

``_step.json`` is **not** derived. It is written into the attempt directory
before the publish — when the stream holds only ``phase_started`` for its
step — and its ``status: completed`` is the fact that the attempt verified;
publishing it is what ``result_committed`` narrates. It is a scientific
output with the immutable lifecycle; the derived view is ``workflow.json``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from causalab.workflow.manifest import classify_unreached

__all__ = ["derive_statuses"]

#: What ``phase_completed`` may say about a workflow step (§4.3's table): the
#: three words for a step that reached its end this run — published, reused,
#: or skipped by a decision (§2.8; a skip is a terminal word, never
#: ``failed``). Anything else on the stream is refused rather than copied into
#: the manifest.
_TERMINAL_WORDS = frozenset({"completed", "reused", "skipped"})


def derive_statuses(
    records: Iterable[Mapping[str, Any]],
    *,
    order: Sequence[str],
    dependencies: Mapping[str, tuple[str, ...]],
    selective: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """``{step: status}`` for every step of ``order``, from one run's lines.
    ``selective`` names the joins declaring ``require: selected`` (spec §2.9),
    which a skipped child does not skip — handed through to
    :func:`~causalab.workflow.manifest.classify_unreached`.

    ``records`` are the lines *this run* appended (``read_events`` sliced from
    :attr:`causalab.io.events.EventLog.opened_at`); a ``--resume`` run's own
    lines say ``reused``, and the first run's ``completed`` lines before them
    are not its history. The last terminal line per step wins, so a stream
    that carries more than one run still derives the latest word.

    Pure: reads nothing but ``records``. Refuses (``ValueError``) a line whose
    ``payload`` is not an object, a terminal line (``phase_completed``, or a
    ``warning`` with ``reason: attempt_failed``) that names a step outside
    ``order``, or a ``phase_completed`` whose ``status`` is not one of the two
    terminal words — a stream this function cannot read is not one the
    manifest may be derived from.
    """
    order = tuple(order)
    known = set(order)
    reached: dict[str, dict[str, Any]] = {}
    for record in records:
        payload = record.get("payload") or {}
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"events.jsonl seq {record.get('seq')!r}: payload is "
                f"{type(payload).__name__}, not an object"
            )
        # only a step's two terminal lines are read for status;
        # `phase_started`, `progress`, `metric` and any other `warning` say
        # nothing here
        event = record.get("event")
        publish = event == "phase_completed"
        failed = event == "warning" and payload.get("reason") == "attempt_failed"
        if not (publish or failed):
            continue
        step = payload.get("step")
        if step not in known:
            raise ValueError(
                f"events.jsonl seq {record.get('seq')!r}: "
                f"{'phase_completed' if publish else 'attempt_failed'} names "
                f"step {step!r}, not one of {list(order)}"
            )
        if failed:
            reached[step] = {"status": "failed"}
            continue
        status = payload.get("status")
        if status not in _TERMINAL_WORDS:
            raise ValueError(
                f"events.jsonl seq {record.get('seq')!r}: phase_completed for "
                f"{step!r} says {status!r}, not one of {sorted(_TERMINAL_WORDS)}"
            )
        reached[step] = {"status": status}
    # `classify_unreached` skips every step already in `reached`, so the two
    # maps are disjoint and one merge is the whole schedule
    entries = {
        **reached,
        **classify_unreached(order, dependencies, reached, selective=selective),
    }
    return {name: entries[name]["status"] for name in order}
