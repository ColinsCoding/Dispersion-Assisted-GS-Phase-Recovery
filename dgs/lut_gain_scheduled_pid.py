"""lut_gain_scheduled_pid.py -- a PID controller (dgs/pid.py) whose gain
regime is selected by a synthesized boolean LUT circuit (dgs/lut_circuit.py)
instead of a continuous formula -- the real embedded-control-hardware
question of how coarse a digitized gain schedule can be before the
closed-loop response visibly degrades.

THE CONTROL LOOP, made explicit (the read/write/execute cycle any embedded
PID implementation actually runs): each step READS the plant's measured
output, quantizes the current error to an address, EXECUTES a LUT lookup
(dgs.lut_circuit.lut_read, the same gate-level decode+AND+OR this repo
already uses for LUT reads elsewhere) to pick a gain regime, computes the
PID law with that regime's gains, and WRITES the actuator command out.
Nothing here is new control theory or new digital-logic theory --
dgs/pid.py's PID class and dgs/lut_circuit.py's LUT machinery are both reused
verbatim; this module is the connective wiring between them.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple

from dgs.pid import PID
from dgs.lut_circuit import synthesize_lut, lut_read
from dgs.memory_address_decoder import address_to_bits


def quantize_error_level(error: float, error_max: float, n_bits: int) -> int:
    """Map |error| in [0, error_max] to an integer address level in
    [0, 2**n_bits - 1] -- the READ half of the control loop's digitizing
    front end (quantizing the derived error signal, the same idea as
    quantizing a raw sensor reading)."""
    if error_max <= 0:
        raise ValueError("error_max must be positive")
    if n_bits < 1:
        raise ValueError(f"n_bits={n_bits}: must be >= 1")
    n_levels = 2 ** n_bits
    level = int(np.clip(abs(error) / error_max, 0.0, 1.0) * (n_levels - 1))
    return level


def regime_lut(n_bits: int, threshold_level: int):
    """Synthesize a boolean LUT (dgs.lut_circuit.synthesize_lut) that reads
    1 ('aggressive gains') for any quantized error level >= threshold_level,
    0 ('gentle gains') otherwise -- a monotonic step function, the simplest
    real gain-scheduling law and the one an embedded PID without
    floating-point interpolation would actually implement in hardware."""
    n_levels = 2 ** n_bits
    if not (0 <= threshold_level < n_levels):
        raise ValueError(f"threshold_level={threshold_level} must be in [0, {n_levels})")

    def regime_fn(*bits):
        level = int("".join(str(b) for b in bits), 2)
        return int(level >= threshold_level)

    lut_bits, _table = synthesize_lut(n_bits, regime_fn)
    return lut_bits


class GainScheduledPID:
    """A PID controller whose gains switch between a 'gentle' and
    'aggressive' pair based on a synthesized boolean LUT read on the
    quantized error magnitude -- wraps dgs.pid.PID, does not reimplement it.
    """

    def __init__(self, gentle_gains: Tuple[float, float, float],
                 aggressive_gains: Tuple[float, float, float],
                 error_max: float, n_bits: int = 2, threshold_level: Optional[int] = None,
                 setpoint: float = 0.0, dt: float = 1.0,
                 out_min: Optional[float] = None, out_max: Optional[float] = None):
        if error_max <= 0:
            raise ValueError("error_max must be positive")
        if threshold_level is None:
            threshold_level = 2 ** (n_bits - 1)   # switch at the halfway level by default
        self.gentle_gains = gentle_gains
        self.aggressive_gains = aggressive_gains
        self.error_max = error_max
        self.n_bits = n_bits
        self.lut_bits = regime_lut(n_bits, threshold_level)
        self._pid = PID(*gentle_gains, setpoint=setpoint, dt=dt, out_min=out_min, out_max=out_max)
        self.regime_history = []

    def update(self, measurement: float) -> float:
        """One control step: READ measurement -> quantize error -> EXECUTE
        LUT lookup for gain regime -> apply the PID law -> WRITE the
        actuator command."""
        error = self._pid.setpoint - measurement
        level = quantize_error_level(error, self.error_max, self.n_bits)
        input_bits = address_to_bits(level, self.n_bits)
        aggressive = bool(lut_read(self.lut_bits, input_bits))
        self.regime_history.append(aggressive)
        kp, ki, kd = self.aggressive_gains if aggressive else self.gentle_gains
        self._pid.kp, self._pid.ki, self._pid.kd = kp, ki, kd
        return self._pid.update(measurement)


def simulate_gain_scheduled(controller: GainScheduledPID, plant, n_steps: int) -> Dict:
    """Closed-loop simulation, mirroring dgs.pid.simulate's structure but for
    a GainScheduledPID instead of a fixed-gain PID."""
    if n_steps < 1:
        raise ValueError(f"n_steps={n_steps}: must be >= 1")
    y = 0.0
    ys, us = [], []
    for _ in range(n_steps):
        u = controller.update(y)
        ys.append(y); us.append(u)
        y = float(plant(u))
    return {"y": np.array(ys), "u": np.array(us),
            "regime_history": np.array(controller.regime_history)}


if __name__ == "__main__":
    from dgs.pid import PID as FixedPID, simulate, first_order_plant

    n_steps = 40
    gentle = (0.3, 0.05, 0.02)
    aggressive = (1.2, 0.15, 0.05)

    scheduled = GainScheduledPID(gentle, aggressive, error_max=5.0, n_bits=2,
                                  setpoint=0.0, dt=1.0)
    plant_a = first_order_plant(tau=8.0, gain=1.0, dt=1.0, y0=5.0)
    result_a = simulate_gain_scheduled(scheduled, plant_a, n_steps)

    fixed = FixedPID(*gentle, setpoint=0.0, dt=1.0)
    plant_b = first_order_plant(tau=8.0, gain=1.0, dt=1.0, y0=5.0)
    _, y_fixed, u_fixed = simulate(fixed, plant_b, n_steps)

    print(f"gain-scheduled PID: final |error| = {abs(result_a['y'][-1]):.4f}, "
          f"aggressive regime used on {result_a['regime_history'].sum()}/{n_steps} steps")
    print(f"fixed gentle-gain PID: final |error| = {abs(y_fixed[-1]):.4f}")
    print(f"\nLUT bits (n_bits=2, threshold=2): {scheduled.lut_bits}  "
          f"(levels 0,1 -> gentle; levels 2,3 -> aggressive)")
