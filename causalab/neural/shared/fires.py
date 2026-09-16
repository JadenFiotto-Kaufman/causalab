"""Per-member fire counts of an installed write set (spec §4 "Fires").

An intervened model's writes install together, run in one forward and tear
down together — so a member that fails to resolve or mismatches its operand's
shape aborts the forward before any capture is published or any table is
written (the write set is atomic by construction; ``executor.py`` names the
three design choices that make it so). What nothing used to check is that
every member **fired**: a write installed on a module the forward never calls
— 📐 the DeltaNet ``conv1d`` module, whose forward goes through a
module-global function instead (``delta_interface.py``) — edited nothing, and
the un-intervened forward was scored as an intervention, a wrong answer with
no error. This module is the counter, the declaration of how often each kind
of write fires, and the refusal when the two disagree.

**The unit is one forward of a forward group.** A group runs as one forward,
or as several over row windows (``batch_rows``), and each of them installs
the whole set again; a forward that *resumes* from a cached prefix (§4
"Resume") still runs every block a write lands in and is one forward like
any other. Per forward, a write at a module boundary, an attention-interface
slot, the experts interior or the DeltaNet kernel boundary fires **once**;
a ``delta_state`` write fires **once per addressed step** — the stepwise
substitution hands it every step's state and it edits the ones the write's
positions name. Two members at one address share a hook and each counts its
firing. A group served from the campaign store (§3) ran in another point,
and that point's counts are its counts.

What the receipt records is layout-invariant, like everything else in it
(§8): the per-forward count a module-kind member was checked at, and for a
state write the number of distinct steps it fired at over the group's rows
— the same number whether the rows ran as one forward or six.

Torch-free on purpose: the tally counts calls, and the engine wraps its own
hooks around :meth:`FireTally.fired`.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from causalab.protocol.errors import ProtocolError

__all__ = [
    "FireTally",
    "GroupFires",
    "check_fires",
    "group_label",
]


def group_label(model: str, input_role: str) -> str:
    """The spelling one forward group has in every record the run writes:
    ``<model> on <input>`` — the location ledger's edit group (§6)."""
    return f"{model} on {input_role}"


@dataclasses.dataclass
class FireTally:
    """One forward's count of each write member's firings against what its
    kind declares.

    :meth:`declare` is called once per member while the hooks are built, with
    the count that member's kind owes this forward; :meth:`fired` is called
    by the hook each time it runs. A state write declares the distinct steps
    its rows address and fires per step, so its ``steps`` are kept as well as
    counted — the receipt's number for it is the union over the group's
    forwards (:class:`GroupFires`).
    """

    expected: dict[str, int] = dataclasses.field(default_factory=dict)
    counts: dict[str, int] = dataclasses.field(default_factory=dict)
    #: the steps a state write fired at, per member (module-kind members
    #: have none: their unit is the forward itself)
    steps: dict[str, set[int]] = dataclasses.field(default_factory=dict)

    def declare(self, members: Iterable[str], times: int) -> None:
        for member in members:
            self.expected[member] = times
            self.counts.setdefault(member, 0)

    def fired(self, members: Iterable[str], *, step: int | None = None) -> None:
        for member in members:
            self.counts[member] = self.counts.get(member, 0) + 1
            if step is not None:
                self.steps.setdefault(member, set()).add(step)


@dataclasses.dataclass
class GroupFires:
    """One forward group's fire record across the forwards it ran as.

    Each forward's checked :class:`FireTally` is folded in; :meth:`record` is
    the ``{member: count}`` the run receipt carries for the group — a
    module-kind member's per-forward count (every forward was checked at the
    same declared count, so the number does not depend on the row layout),
    a state write's number of distinct steps over every forward's rows.
    """

    per_forward: dict[str, int] = dataclasses.field(default_factory=dict)
    steps: dict[str, set[int]] = dataclasses.field(default_factory=dict)

    def fold(self, tally: FireTally) -> None:
        for member, count in tally.counts.items():
            if member in tally.steps:
                self.steps.setdefault(member, set()).update(tally.steps[member])
            else:
                self.per_forward[member] = count

    def record(self) -> dict[str, int]:
        return {
            **self.per_forward,
            **{member: len(steps) for member, steps in self.steps.items()},
        }


def check_fires(group: str, tally: FireTally) -> None:
    """Refuse the forward group when any member fired other than its
    declared count — a member that never fired named first.

    A count of zero is the measured shape: the forward never called the
    module the write was installed on, so the tensor the write addresses did
    not exist in this forward and the result would have been an un-intervened
    forward scored as an intervention. More than the declared count is a
    module the forward calls twice (a shared or looped block), where the
    write would land on a tensor the document did not name. Both carry
    ``component_unavailable``: what is missing is the tensor the member
    addresses, in this forward.
    """
    wrong = [
        (member, tally.counts.get(member, 0), expected)
        for member, expected in tally.expected.items()
        if tally.counts.get(member, 0) != expected
    ]
    if not wrong:
        return
    wrong.sort(key=lambda item: (item[1] != 0, item[0]))
    member, count, expected = wrong[0]
    if count == 0:
        what = (
            f"write {member!r} in forward group {group!r} fired 0 times in one "
            f"forward, not the {expected} its kind declares: the forward never "
            "called the module this write was installed on, so the tensor it "
            "addresses did not exist in this forward and the point would have "
            "scored an un-intervened forward as an intervention"
        )
    else:
        what = (
            f"write {member!r} in forward group {group!r} fired {count} times in "
            f"one forward, not the {expected} its kind declares: the forward "
            "calls the module this write was installed on more than once, so "
            "the write would land on a tensor the document did not name"
        )
    others = ", ".join(f"{m!r} ({c} of {e})" for m, c, e in wrong[1:])
    if others:
        what += f"; also off: {others}"
    raise ProtocolError(
        "P4",
        what + ". The write set is one transaction — the point is refused whole, and "
        "nothing of it was published or written.",
        reason="component_unavailable",
    )
