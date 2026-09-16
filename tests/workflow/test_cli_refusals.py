"""The workflow CLI's refusals, and the docs that quote one of them verbatim.

`docs/running_experiments.md` prints the `--dtype` refusal *inside backticks*,
as the line a reader compares their own terminal against — and nothing checked
that the quote was still the message. A vocabulary sweep had to change
both sides by hand ("intervention protocol" → "intervention specification"),
and a pair kept in sync by hand is a pair that drifts; the commit message
saying both sides were changed is not a guard.

The same sweep then wrote the *tail* of the message into two demos, in italics,
and a guard that read one named file saw neither. So the quotes are **found,
not listed**: every tracked markdown file is searched for a quoted run that
carries the message's signature, and each one found has to be a substring of
what the CLI prints. Substring rather than prefix, because a demo quotes the
half a reader needs and then explains.

So this is `tests/protocol/test_vocabulary_census.py`'s guard one level down:
docs and code agreeing, checked rather than asserted. The message is read out
of the CLI's real stderr, not out of its source, so the test cannot pass
against a string the CLI never prints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from causalab.cli import main
from tests._helpers.tracked import tracked_files

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

#: The run of words a quote of the refusal is recognized by. The head of the
#: message (`docs/running_experiments.md`) and its tail (the demos) overlap on
#: exactly this, and it is asserted to be *in* the message too — so a rewording
#: that drops it fails here instead of letting the finder go quiet.
SIGNATURE = "each declare their own realization"

#: A quoted run in markdown: backticked, or italicised inside double quotes
#: (``*"…"*``), which is how the demos quote a message they are not showing as
#: a terminal line. Either may span a soft wrap but never a blank line: a
#: paragraph boundary ends a quote, so one stray backtick in a paragraph cannot
#: shift the pairing for the rest of the file.
QUOTED = re.compile(
    r"`((?:[^`\n]|\n(?![ \t]*\n))+)`" r"|" r"\*\"((?:[^\n]|\n(?![ \t]*\n))+?)\"\*"
)

#: A fenced code block. Its three backticks are an odd count, which would
#: pair the fence with the next inline run and swallow the paragraph after it —
#: and what a doc shows in a fence is a command or its output, which the
#: `docs/running_experiments.md` example quotes *outside* the fence.
FENCE = re.compile(r"```.*?```", re.S)

#: The line-lead marker of a blockquote. A wrapped quote repeats it on every
#: line, and the CLI never printed it.
BLOCKQUOTE = re.compile(r"^[ \t]*>[ \t]?", re.M)

#: How many files quoted the refusal when the finder was written:
#: `docs/running_experiments.md`, `demos/README.md`,
#: `demos/onboarding_tutorial/03_localize.md`,
#: `demos/weekdays_geometry/weekdays_geometry.md`. A floor, not a pin — a
#: finder that finds nothing passes for the wrong reason.
QUOTING_FILES_FLOOR = 4


def _refusal(tmp_path: Path, capsys) -> str:
    """The `--dtype` refusal as the CLI actually prints it.

    A workflow document is recognized by its ``steps`` section alone
    (``is_workflow``), and the refusal is the first thing ``run`` does, so an
    empty one is enough to reach it without a model or a dataset.
    """
    document = tmp_path / "workflow.json"
    document.write_text(json.dumps({"version": "1", "steps": {}}))
    code = main(
        ["run", str(document), "--out", str(tmp_path / "out"), "--dtype", "bf16"]
    )
    assert code == 1, "--dtype on a workflow is refused (§9)"
    return capsys.readouterr().err.strip()


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _quotes() -> dict[str, list[str]]:
    """Every quoted run carrying :data:`SIGNATURE`, whitespace-normalized, by
    file — from the tracked markdown tree, so a new copy is found and an
    untracked one (a worktree, a build) is not."""
    out: dict[str, list[str]] = {}
    for path in tracked_files(ROOT, "*.md"):
        text = BLOCKQUOTE.sub("", FENCE.sub("", path.read_text(errors="replace")))
        for match in QUOTED.finditer(text):
            quote = _normalized(match.group(1) or match.group(2))
            if SIGNATURE in quote:
                out.setdefault(path.relative_to(ROOT).as_posix(), []).append(quote)
    return out


def test_dtype_is_refused_on_a_workflow(tmp_path, capsys) -> None:
    """A workflow's steps each declare their own realization, so one
    `--dtype` for the whole schedule has no meaning to give it."""
    message = _refusal(tmp_path, capsys)
    assert message.startswith("refused: --dtype sets model.dtype")
    assert "intervention specification" in message, (
        "the refusal names the object §11.1 calls an intervention specification"
    )


def test_every_doc_that_quotes_the_refusal_quotes_it_verbatim(tmp_path, capsys) -> None:
    """Each quoted run is a substring of the message.

    A substring rather than the whole thing: a doc quotes as far as the reader
    needs and then explains, which is the right thing for it to do — the head
    in one place, the tail in another. What must not happen is the quoted part
    saying something the CLI does not.
    """
    message = _normalized(_refusal(tmp_path, capsys))
    assert SIGNATURE in message, (
        "the refusal no longer carries the words the finder recognizes a quote "
        f"by ({SIGNATURE!r}) — move SIGNATURE with the message"
    )
    quotes = _quotes()
    assert len(quotes) >= QUOTING_FILES_FLOOR, (
        f"only {sorted(quotes)} quote the --dtype refusal; the finder used to "
        f"see {QUOTING_FILES_FLOOR} files, so either a quote was dropped or the "
        "finder stopped seeing one"
    )
    wrong = [
        f"  {name}: {quote}"
        for name, found in sorted(quotes.items())
        for quote in found
        if quote not in message
    ]
    assert not wrong, (
        "a doc quotes a --dtype refusal the CLI does not print:\n"
        + "\n".join(wrong)
        + f"\n  cli: {message}"
    )
