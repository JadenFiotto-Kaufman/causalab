# Codebase Conventions

**Question the approach, not just the implementation.** Before improving a cache, ask if caching is the right solution. Before handling an edge case, ask if that case can actually occur. Before adding backwards compatibility, check if the old version is still used. Before fixing code to match a test, consider whether the test should change instead.

**Understand root causes.** When something breaks, dig into WHY — what led to this bug, why didn't the type checker catch it, is this a symptom of a deeper design problem?

**Simplify where possible.** Remove flags by detecting conditions automatically. Split functions that do too much. Use libraries instead of reimplementing.

**Be pragmatic.** Not everything needs to be fixed immediately. If something is a "hint for the future" rather than a required change, say so.

**Make sure code demonstrates its value.** Especially for notebooks — results should clearly show the value of the feature. If examples don't cleanly separate or outputs are confusing, consider whether there's a better setup.

**Think about implications.** What do the results mean? How does this approach compare to alternatives? What conclusions should we draw?

**Check the abstraction level.** Is there too much functionality in one place? Are we reimplementing something a library already does?

## Repository layout & conventions

See [`docs/CODEBASE.md`](docs/CODEBASE.md) for the codebase architecture, layering rules, and invariants, and [`docs/TESTS.md`](docs/TESTS.md) for the test-tier conventions.

- **Run everything through `uv run` from the repo root** — never a bare `python3`, never an inherited `VIRTUAL_ENV`.
- **The CPU gate is `uv run pytest -m "not golden"`** — what CI runs. The `golden` tier needs an accelerator and real weights.
- **Run `pre-commit` in the same command as the commit**, so a reformat never sits uncommitted and diverges your branch from what CI checks.
- **PR bodies follow the pull-request template** (`.github/pull_request_template.md` where the repository carries one) — a `Stacked on #N` line first when stacked, then Headline, Motivating example, New implementation, Risk, Scope of this PR, Notes for review. `gh pr create --body` bypasses the template, so read it and fill every section yourself. After a review round that changes the design, edit the body so it describes the branch as it stands (`gh pr edit <n> --body-file`); the round's answer goes in a comment, the body stays current.
