"""Closed-loop schedules on a training hyperparameter (spec §2.11
``train.control``).

``train.anneal`` is the **open-loop** schedule: a hyperparameter follows a
declared ramp whatever the fit does. ``train.control`` is the **closed-loop**
one: a hyperparameter is moved every update so that a *signal the fit itself
produces* follows a declared setpoint. The one signal so far is a gate's
``hard_mask_size`` — the kept-unit count through its hard mask — and the one
controller a PID, which is what makes "sweep the sparsity from everything
patched to nothing patched, at a steady pace" a declaration: the sparsity
weight is the controlled value, the kept-head count the signal, and a linear
ramp from all units to none the setpoint.

**The control law.** With ``e_t = signal_t − setpoint_t`` (positive when too
many units are kept)::

    u_t   = kp · (e_t − e_{t−1}) + ki · e_t + kd · ((e_t − e_{t−1}) − (e_{t−1} − e_{t−2}))
    log w ← clip(log w + u_t, log bounds)          (``space: log``)
    w     ← clip(w + u_t, bounds)                  (``space: linear``)

This is a PID controller on the kept count, with the roles of the terms
shifted one derivative up: the proportional term acts on the *rate* of the
error (``e_t − e_{t−1}``), the integral term is the count error itself —
bounded by the unit count, so it cannot wind up — and the derivative term
is the change in that rate, clipped at ``d_clip`` so a plateau breaking does
not throw the weight. Updates in log space make the same gains produce the
same *relative* change in the weight whatever its magnitude. On the first
update the previous signal is taken to be the current one (no rate observed
yet), and the previous setpoint is the ramp's start — so the first rate
error is the ramp's own slope.

Pure Python on purpose: the law is unit-testable without torch, and the
train loop hands it two floats per update.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

__all__ = ["CONTROL_DEFAULTS_ENGINE", "PidController", "ramp_setpoint"]

#: The engine-side view of the controller defaults the canonical form
#: materializes (``schema.CONTROL_DEFAULTS``); read here so a spec built in
#: code without the canonicalizer still runs with the documented values.
CONTROL_DEFAULTS_ENGINE: dict[str, Any] = {
    "kd": 0.0,
    "space": "log",
    "bounds": (1e-8, 1e8),
    "d_clip": 5.0,
}


def ramp_setpoint(
    start: float, end: float, frac: float, step: int, total_steps: int
) -> float:
    """The setpoint after ``step`` updates of ``total_steps``: linear from
    ``start`` to ``end`` over the first ``frac`` of the run, then held — the
    same arithmetic ``train.anneal`` uses for a hyperparameter."""
    ramp_steps = max(1, int(frac * total_steps))
    progress = min(1.0, step / ramp_steps)
    return start + (end - start) * progress


@dataclasses.dataclass
class PidController:
    """One controlled value, moved by a PID on the signal-minus-setpoint
    error (module docstring). ``value`` is the controlled hyperparameter's
    current value; :meth:`step` takes one observation and returns the new
    value."""

    kp: float
    ki: float
    kd: float
    value: float
    space: str
    bounds: tuple[float, float]
    d_clip: float
    #: the setpoint before the first update — the ramp's start
    setpoint_before: float
    _previous_signal: float | None = dataclasses.field(default=None, repr=False)
    _previous_setpoint: float | None = dataclasses.field(default=None, repr=False)
    _previous_rate_error: float = dataclasses.field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.space not in ("log", "linear"):
            raise ValueError(f"unknown control space {self.space!r}")
        low, high = self.bounds
        if not low < high:
            raise ValueError(f"control bounds must be increasing, got {self.bounds}")
        if self.space == "log" and low <= 0.0:
            raise ValueError(
                f"log-space control needs positive bounds, got {self.bounds}"
            )
        if self.space == "log" and self.value <= 0.0:
            raise ValueError(
                f"log-space control needs a positive initial value, got {self.value}"
            )
        self.value = self._clip(self.value)

    def _clip(self, value: float) -> float:
        low, high = self.bounds
        return min(high, max(low, value))

    def step(self, signal: float, setpoint: float) -> float:
        """Observe ``signal`` against ``setpoint`` and move the value."""
        previous_signal = (
            signal if self._previous_signal is None else self._previous_signal
        )
        previous_setpoint = (
            self.setpoint_before
            if self._previous_setpoint is None
            else self._previous_setpoint
        )
        error = signal - setpoint
        rate_error = (signal - previous_signal) - (setpoint - previous_setpoint)
        derivative = rate_error - self._previous_rate_error
        derivative = max(-self.d_clip, min(self.d_clip, derivative))
        u = self.kp * rate_error + self.ki * error + self.kd * derivative
        if self.space == "log":
            # clip in log space *before* exponentiating, as run.py clips
            # log_mult: a large gain must saturate at the bound, not overflow
            low, high = (math.log(b) for b in self.bounds)
            self.value = math.exp(min(high, max(low, math.log(self.value) + u)))
        else:
            self.value = self._clip(self.value + u)
        self._previous_signal = signal
        self._previous_setpoint = setpoint
        self._previous_rate_error = rate_error
        return self.value


def build_controller(spec: Mapping[str, Any], *, initial: float) -> PidController:
    """A controller from its parsed (or canonical) ``train.control`` entry,
    defaults filled from :data:`CONTROL_DEFAULTS_ENGINE`."""
    gains = spec["gains"]
    bounds = spec.get("bounds", CONTROL_DEFAULTS_ENGINE["bounds"])
    start, _end, _frac = spec["setpoint"]["ramp"]
    return PidController(
        kp=float(gains["kp"]),
        ki=float(gains["ki"]),
        kd=float(gains.get("kd", CONTROL_DEFAULTS_ENGINE["kd"])),
        value=float(initial),
        space=str(spec.get("space", CONTROL_DEFAULTS_ENGINE["space"])),
        bounds=(float(bounds[0]), float(bounds[1])),
        d_clip=float(spec.get("d_clip", CONTROL_DEFAULTS_ENGINE["d_clip"])),
        setpoint_before=float(start),
    )
