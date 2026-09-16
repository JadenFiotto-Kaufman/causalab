"""How many rows one forward of a fit may cover (spec §8, ``fit_rows``).

A cohort's optimizer step is one forward over the concatenation of its
members' minibatches (§4 "Cohorts"), and its eval pass one forward over the
members' eval rows; the rows a window may hold trade peak memory for time: a
forward's activations are close to linear in rows, the step is launch-bound,
so the largest window that fits the device is the fastest one. The grad
forwards are the ones measured — the probe is a member's forward *and
backward* — and the eval passes pack under the same bound unless
``batch_rows`` is authored (``train._advance_eval_budget``), so one number pins
both. No constant is right for every model, sequence length, ``pairs``
and device, which is why the bound is the author's (``fit_rows``) — and why,
when the author sets none, the engine **measures** it instead of guessing:

* **fixed** — an authored ``fit_rows``: members are packed under it, a
  member's own minibatch never split, and it is never probed or shrunk, so a
  pinned run stays pinned;
* **auto** — the cohort's first step runs its first member **alone** as a
  probe under CUDA peak-memory tracking; bytes per row from that one window
  and the device's free memory (with :data:`MARGIN` held back) give the
  bound the rest of that step and every later step pack under. Off CUDA
  there is nothing to read, so auto stays unbounded, as before.

Either way a window that still runs out of memory is **retried**: the
window's members have not stepped, so their gradients are zeroed, the
allocator's cache released, the bound halved (never below one member's
minibatch) and the window re-packed — the same shape as an executable
batch-size search, one halving at a time. A single member that does not fit
is re-raised: nothing smaller exists.

The bound a cohort ran under is reported (``TrainOutcome.fit_rows``, the
step receipt's ``execution.fit_rows_resolved``) so an author can read it
once and pin it: the measured number depends on what else occupied the
device at that moment, and pinning is the reproducible path.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Protocol, Sequence, TypeVar

import torch

__all__ = ["MARGIN", "Meter", "RowBudget", "cuda_meter"]

#: The share of the device's available memory the auto bound leaves unused:
#: the probe's bytes-per-row is one window's slope, and a later window with
#: more rows, more active experts or a longer eval pass beside it rounds up.
MARGIN = 0.10

_T = TypeVar("_T")


class Meter(Protocol):
    """What the auto bound needs from a device: run one window and report
    the bytes it peaked above the level it started from, and how many bytes
    the device could still give a window afterwards."""

    def measure(self, run: Callable[[], None]) -> tuple[int, int]: ...


@dataclasses.dataclass
class _CudaMeter:
    device: torch.device

    def measure(self, run: Callable[[], None]) -> tuple[int, int]:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        before = torch.cuda.memory_allocated(self.device)
        run()
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device) - before
        free, _total = torch.cuda.mem_get_info(self.device)
        # the caching allocator's reserved-but-unused bytes are ours too
        cached = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(
            self.device
        )
        return max(peak, 0), max(free + cached, 0)


def cuda_meter(model: torch.nn.Module) -> Meter | None:
    """The meter for the device ``model``'s weights are on — ``None`` off
    CUDA, where the auto bound has nothing to read."""
    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            return _CudaMeter(parameter.device)
        return None
    return None


@dataclasses.dataclass
class RowBudget:
    """The rows one forward of a fit may cover — a grad window, or an eval
    window packing under the same number — fixed or measured (module
    docstring). ``bound`` is ``None`` while unresolved or unbounded."""

    bound: int | None
    fixed: bool
    meter: Meter | None = None
    #: auto only: whether the probe has run
    resolved: bool = False
    #: what the probe measured, for the record: bytes per row, bytes available
    probe: tuple[int, int] | None = None
    #: how many windows ran out of memory and were re-packed — a measured
    #: bound that shrank was too loose; the counts of the grad budget and of
    #: the eval budget packing under the same bound reach the receipt together
    #: (``fit_rows_shrinks``)
    shrinks: int = 0

    @classmethod
    def of(cls, fit_rows: int | None, meter: Meter | None) -> "RowBudget":
        """An authored bound is fixed; none is auto — measured through
        ``meter`` when there is one, unbounded when there is not."""
        if fit_rows is not None:
            return cls(bound=fit_rows, fixed=True)
        return cls(bound=None, fixed=False, meter=meter, resolved=meter is None)

    @property
    def probing(self) -> bool:
        """Whether the next window is the auto probe: one member, measured."""
        return not self.fixed and not self.resolved

    def take(
        self, pending: Sequence[_T], size_of: Callable[[_T], int]
    ) -> tuple[list[_T], list[_T]]:
        """The next window off ``pending`` and what is left: one item while
        probing, else the greedy prefix under ``bound`` — one item always
        fits, whatever its size (a member's minibatch is never split)."""
        if not pending:
            return [], []
        if self.probing:
            return [pending[0]], list(pending[1:])
        window: list[_T] = []
        used = 0
        for item in pending:
            size = size_of(item)
            if window and self.bound is not None and used + size > self.bound:
                break
            window.append(item)
            used += size
        return window, list(pending[len(window) :])

    def run(self, rows: int, body: Callable[[], None], unit: int | None = None) -> None:
        """Run one window's ``body``. While probing — the first grad window of
        an auto budget, a member's forward and backward — run it under the
        meter and set the bound from what it peaked: the rows the available
        memory holds at that slope with :data:`MARGIN` held back, never fewer
        than the probe's own rows, and floored to a multiple of ``unit`` — the
        smallest window any member of the fit will bring, so that with equal
        minibatches the bound is whole members and small free-memory drift
        moves the packing only at member boundaries. Members' minibatches need
        not be equal (``pairs`` is per member and an epoch's last minibatch is
        a remainder); then the floor is coarser than one member and only the
        equal case is fully stable. ``available`` is read right after the
        probe, when only that member's optimizer state is resident, so it
        overstates what the other members leave by their states; the margin
        and the retry absorb that. A resolved or fixed budget just runs the
        body."""
        if not self.probing or self.meter is None:
            body()
            return
        peak, available = self.meter.measure(body)
        per_row = max(peak, 1) / max(rows, 1)
        fits = int(available * (1.0 - MARGIN) / per_row)
        member = max(unit if unit is not None else rows, 1)
        self.bound = max(rows, (fits // member) * member)
        self.probe = (int(per_row), available)
        self.resolved = True

    def can_shrink(self, window_rows: int, largest_member: int) -> bool:
        """Whether a smaller window than ``window_rows`` exists: not for a
        fixed bound, and not below one member's minibatch."""
        return not self.fixed and window_rows > largest_member

    def shrink(self, window_rows: int, largest_member: int) -> None:
        """Halve the bound after a window of ``window_rows`` ran out of
        memory, never below ``largest_member``."""
        self.bound = max(largest_member, window_rows // 2)
        self.resolved = True
        self.shrinks += 1
