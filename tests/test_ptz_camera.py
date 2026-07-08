"""Test dgs.ptz_camera: the pan/tilt <-> line-of-sight kinematics, the static
holding torque m g d cos(phi), the pan-axis inertia's tilt dependence, the
Euler-Lagrange dynamics (static equilibrium under the holding torque, energy
conserved with no torque, pan/tilt decoupling only when the CG is on-axis)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import ptz_camera as ptz

m, d = 0.5, 0.05
params = {"I_p0": 0.01, "I_tilt": 0.005, "mass": m, "cg_distance": d}

# 1. pointing kinematics: unit vectors, and the cardinal directions
assert np.allclose(ptz.pointing_direction(0, 0), [1, 0, 0])
assert np.allclose(ptz.pointing_direction(np.pi/2, 0), [0, 1, 0])
assert np.allclose(ptz.pointing_direction(0, np.pi/2), [0, 0, 1])
for pan, tilt in [(0.3, 0.7), (-1.1, 0.2), (2.0, -0.4)]:
    assert np.isclose(np.linalg.norm(ptz.pointing_direction(pan, tilt)), 1.0)

# 2. aim_at is the inverse of pointing_direction (round-trip)
for pan, tilt in [(0.3, 0.7), (-1.1, 0.2), (0.9, -0.5)]:
    p2, t2 = ptz.aim_at(ptz.pointing_direction(pan, tilt))
    assert np.isclose(p2, pan) and np.isclose(t2, tilt)
assert np.allclose(ptz.aim_at([1, 1, 1]), [np.pi/4, np.arcsin(1/np.sqrt(3))])
try:
    ptz.aim_at([0, 0, 0]); assert False
except ValueError:
    pass

# 3. static holding torque = m g d cos(phi): max level, zero looking straight up
assert np.isclose(ptz.holding_torque(m, d, 0.0), m * ptz.G * d)          # level, max
assert np.isclose(ptz.holding_torque(m, d, np.pi/2), 0.0)                # up, zero
assert ptz.holding_torque(m, d, 0.3) > ptz.holding_torque(m, d, 1.0)     # decreasing

# 4. pan-axis inertia depends on tilt: max level, base value looking up
assert np.isclose(ptz.pan_inertia(0.0, 0.01, m, d), 0.01 + m * d**2)     # cos^2=1
assert np.isclose(ptz.pan_inertia(np.pi/2, 0.01, m, d), 0.01)           # cos^2=0

# 5. dynamics: the holding torque gives static equilibrium (no tilt acceleration)
phi0 = np.radians(30)
tau = ptz.holding_torque(m, d, phi0)
ta, pa = ptz.forward_dynamics([0, phi0, 0, 0], 0.0, tau, params)
assert abs(pa) < 1e-12 and abs(ta) < 1e-12                              # holds still
# released level with no torque, the tilt falls at -m g d / I_tilt
_, pa0 = ptz.forward_dynamics([0, 0, 0, 0], 0.0, 0.0, params)
assert np.isclose(pa0, -m * ptz.G * d / params["I_tilt"])

# 6. energy is conserved with no applied torque (RK4 check)
sim = ptz.simulate([0.0, np.radians(-45), 0.0, 0.0], params, t_end=3.0)
assert sim["energy_drift"] < 1e-8

# 7. decoupling: with the CG on the tilt axis (d=0) pan and tilt are independent
flat = {"I_p0": 0.01, "I_tilt": 0.005, "mass": m, "cg_distance": 0.0}
assert np.isclose(ptz.holding_torque(m, 0.0, 0.5), 0.0)                 # no gravity torque
# pan accel depends only on tau_pan, tilt accel only on tau_tilt
ta1, pa1 = ptz.forward_dynamics([0, 0.5, 3.0, 2.0], 0.1, 0.0, flat)
ta2, pa2 = ptz.forward_dynamics([0, 1.2, 3.0, 2.0], 0.1, 0.0, flat)     # different phi, tilt-rate
assert np.isclose(ta1, ta2)                                            # pan unaffected by phi
assert np.isclose(pa1, 0.0) and np.isclose(pa2, 0.0)                    # tilt only from tau_tilt

# 8. coupling: with d>0, a fast pan DOES push on the tilt (centrifugal term)
_, pa_slow = ptz.forward_dynamics([0, np.radians(45), 0.0, 0.0], 0, 0, params)
_, pa_fast = ptz.forward_dynamics([0, np.radians(45), 10.0, 0.0], 0, 0, params)
assert not np.isclose(pa_slow, pa_fast)                                # pan rate changes tilt accel

# 9. kwarg bounds
for bad in (lambda: ptz.holding_torque(-1, d, 0),
            lambda: ptz.simulate([0, 0, 0, 0], params, dt=0),
            lambda: ptz.simulate([0, 0, 0, 0], params, t_end=0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_ptz_camera: all checks passed")
