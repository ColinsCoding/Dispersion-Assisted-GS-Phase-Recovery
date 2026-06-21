"""Test Griffiths Prob 7.11: the falling aluminum loop (eddy-current brake)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import electrodynamics as ed

# 1. terminal velocity formula v_t = m g R / (B^2 s^2)
m, g, R, B, s = 0.01, 9.81, 0.05, 0.5, 0.1
v_t = ed.eddy_brake_terminal_velocity(m, g, R, B, s)
assert abs(v_t - m * g * R / (B**2 * s**2)) < 1e-12

# 2. the velocity approaches v_t exponentially; at t=tau it is v_t(1 - 1/e) (from rest)
tau = m * R / (B**2 * s**2)
assert abs(ed.falling_loop_velocity(m, g, R, B, s, tau, v0=0.0) - v_t * (1 - 1/np.e)) < 1e-9
# v(0) = v0, and v(inf) -> v_t
assert abs(ed.falling_loop_velocity(m, g, R, B, s, 0.0, v0=2.0) - 2.0) < 1e-12
assert abs(ed.falling_loop_velocity(m, g, R, B, s, 50*tau, v0=0.0) - v_t) < 1e-3

# 3. numerical RK4 matches the analytic exponential
t = np.linspace(0, 8*tau, 4000)
v_num = ed.falling_loop_simulation(m, g, R, B, s, t, v0=0.0)
v_ana = ed.falling_loop_velocity(m, g, R, B, s, t, v0=0.0)
assert np.max(np.abs(v_num - v_ana)) < 1e-4 * v_t

# 4. Lenz / monotonic: from rest the loop speeds up but never overshoots v_t
assert v_num[0] == 0.0 and np.all(np.diff(v_num) >= -1e-15)   # monotonic rise
assert v_num.max() <= v_t + 1e-9                               # braked, no overshoot

# 5. a loop dropped FASTER than terminal SLOWS to v_t (brake works both ways)
v_fast = ed.falling_loop_simulation(m, g, R, B, s, t, v0=5 * v_t)
assert v_fast[0] == 5 * v_t and v_fast[-1] < v_fast[0]
assert abs(v_fast[-1] - v_t) < 1e-2 * v_t

# 6. the aluminum punchline: terminal velocity is SIZE-INDEPENDENT
v_small = ed.aluminum_loop_terminal_velocity(B=1.0)
# recompute from a concrete small and large loop -> same v_t
def vt_from_geometry(side, w, B=1.0, g=9.81, rm=2700.0, rr=2.65e-8):
    mass = rm * (4 * side) * w**2
    res = rr * (4 * side) / w**2
    return ed.eddy_brake_terminal_velocity(mass, g, res, B, side)
assert abs(vt_from_geometry(0.05, 0.002) - v_small) < 1e-9     # 5 cm loop
assert abs(vt_from_geometry(0.20, 0.005) - v_small) < 1e-9     # 20 cm loop, same v_t!

# 7. the "1 Tesla terminal" number: aluminum creeps at ~1 cm/s
assert 0.005 < ed.aluminum_loop_terminal_velocity(B=1.0) < 0.02   # ~0.011 m/s

print(f"TEST PASS  (v_t=mgR/B^2s^2; exponential approach tau={tau:.3f}s; RK4==analytic; "
      f"brakes from above and below; aluminum v_t size-independent = "
      f"{v_small*100:.2f} cm/s at 1 T)")
