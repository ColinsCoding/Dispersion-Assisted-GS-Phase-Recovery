"""Test elevator scale readings: N = m(g+a), clamping, ride kinematics."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import elevator_scale as ev

m, g = 70.0, ev.G_EARTH

# 1. the four regimes of N = m(g + a)
assert abs(float(ev.apparent_weight(m, 0.0)) - m * g) < 1e-9          # honest scale
assert abs(float(ev.apparent_weight(m, 2.0)) - m * (g + 2)) < 1e-9    # heavy
assert abs(float(ev.apparent_weight(m, -2.0)) - m * (g - 2)) < 1e-9   # light
assert float(ev.apparent_weight(m, -g)) == 0.0                        # free fall

# 2. contact force cannot pull: a < -g clamps to zero, never negative
assert float(ev.apparent_weight(m, -15.0)) == 0.0
assert np.all(ev.apparent_weight(m, np.linspace(-30, 5, 100)) >= 0.0)

# 3. g-force: 1 + a/g, so +g up doubles your weight; free fall is 0 g
assert abs(float(ev.g_force(0.0)) - 1.0) < 1e-12
assert abs(float(ev.g_force(g)) - 2.0) < 1e-12
assert float(ev.g_force(-g)) == 0.0

# 4. ride profile is antisymmetric about the midpoint -> net dv = 0
#    (an odd function's symmetric integral vanishes -- see dgs.even_odd)
t = np.linspace(0, 12, 4801)   # rest-accel(2-4)-cruise(4-8)-decel(8-10)-rest
a = ev.ride_profile(t, a_max=1.2, t_acc=2.0, t_cruise=4.0)
assert abs(np.trapezoid(a, t)) < 1e-9                               # int a dt = 0 (odd)
v, y = ev.trip_kinematics(t, a)
assert abs(v[-1]) < 1e-9                                              # ends at rest
assert abs(np.max(v) - 1.2 * 2.0) < 1e-2                              # v_cruise = a*t_acc
assert abs(y[-1] - (1.2 * 2.0) * (4.0 + 2.0)) < 0.05                  # d = v*(t_cruise+t_acc)

# 5. scale trace: rest honest, accel heavy, cruise honest, decel light
N = ev.apparent_weight(m, a)
assert abs(float(N[np.searchsorted(t, 1.0)]) - m * g) < 1e-6         # rest
assert abs(float(N[np.searchsorted(t, 3.0)]) - m * (g + 1.2)) < 1e-6  # accel: heavy
assert abs(float(N[np.searchsorted(t, 6.0)]) - m * g) < 1e-6         # cruise: honest
assert abs(float(N[np.searchsorted(t, 9.0)]) - m * (g - 1.2)) < 1e-6  # decel: light

# 6. bounds
for bad in (lambda: ev.apparent_weight(-70, 0.0),
            lambda: ev.apparent_weight(70, 0.0, g=0),
            lambda: ev.ride_profile(t, a_max=-1)):
    try:
        bad()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

print(f"TEST PASS  (N=m(g+a): rest {m*g:.0f} N, free fall 0, clamp at 0; "
      f"antisymmetric ride -> dv=0, climb {y[-1]:.1f} m; heavy/honest/light trace)")
