"""Step scripts whose purpose *is* wiring a workflow together.

Only what belongs to the workflow layer itself lives here. Numerical analysis is
:mod:`causalab.analysis`; rendering is :mod:`causalab.io.plots`; the IO helpers a
script uses are :mod:`causalab.io.step_io`. A document addresses any of them the
same way — ``{"script": {"module": "…"}}`` — so this package has no privileged
status, only a narrow subject.

| module | what it does |
|---|---|
| ``select`` | reduce a metric table to named values, which a later document's ``set`` reads |
| ``reduce`` | reduce a metric table under an authored ``reduction`` contract (workflow spec §2.6): one row per group with the value, the counts and an interval |

``select`` sits here rather than under ``analysis/`` because its output exists to
be consumed by the *next step* rather than by a reader: it is the stage-1 →
stage-2 seam expressed as data instead of a notebook. ``reduce`` sits beside it
for the same reason: the number it publishes is what the next step — or the
report — consumes, and the declaration of what that number *is* travels in the
step's identity rather than in a notebook cell.
"""

from __future__ import annotations

__all__: list[str] = []
