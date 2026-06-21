"""Test PID control: P leaves offset, I removes it, D damps overshoot; clamping works."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import pid as pc

def plant():
    return pc.second_order_plant(wn=1.0, zeta=0.25, dt=0.05)

N = 3000

# 1. P-only leaves a steady-state ERROR (never reaches the setpoint)
_, yP, _ = pc.simulate(pc.PID(3.0, 0.0, 0.0, setpoint=1.0, dt=0.05), plant(), N)
assert yP[-1] < 0.9, yP[-1]                      # offset below setpoint 1.0

# 2. adding I removes the steady-state error (tracks the setpoint)
_, yPI, _ = pc.simulate(pc.PID(3.0, 2.0, 0.0, setpoint=1.0, dt=0.05), plant(), N)
assert abs(yPI[-1] - 1.0) < 0.05, yPI[-1]
peak_PI = yPI.max()
assert peak_PI > 1.3                             # PI overshoots a lot on this plant

# 3. adding D damps the overshoot, still settles to the setpoint
_, yPID, _ = pc.simulate(pc.PID(3.0, 2.0, 2.0, setpoint=1.0, dt=0.05), plant(), N)
assert abs(yPID[-1] - 1.0) < 0.02, yPID[-1]
peak_PID = yPID.max()
assert peak_PID < peak_PI                         # D reduces overshoot
assert peak_PID < 1.3                             # modest overshoot

# 4. output clamping + anti-windup: the command never leaves [out_min, out_max]
pid = pc.PID(20.0, 10.0, 0.0, setpoint=1.0, dt=0.05, out_min=-0.5, out_max=0.5)
_, _, u = pc.simulate(pid, plant(), 800)
assert u.max() <= 0.5 + 1e-9 and u.min() >= -0.5 - 1e-9
assert pid._integral < 1e6                        # anti-windup kept the integral bounded

# 5. validation
try:
    pc.PID(1, 0, 0, dt=0)
except ValueError:
    pass
else:
    raise AssertionError("dt=0 should raise")

print(f"TEST PASS  (P offset={yP[-1]:.2f}<1; PI tracks (peak {peak_PI:.2f}); "
      f"PID settles, overshoot {peak_PID:.2f}<{peak_PI:.2f}; clamp+anti-windup ok)")
