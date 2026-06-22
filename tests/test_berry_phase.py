"""Test Berry's phase: geometric, = -1/2 solid angle, gauge- and speed-invariant."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import berry_phase as bp

def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi

# 1. Berry phase = -1/2 * solid angle (mod 2 pi) for several cone angles
for deg in (20, 45, 90, 130):
    th = np.radians(deg)
    g = bp.berry_phase_spin(th)
    assert abs(wrap(g - (-0.5 * bp.solid_angle_cone(th)))) < 1e-2, deg

# 2. solid angle: 0 at the pole, 2 pi at the equator (hemisphere), 4 pi at the far pole
assert np.isclose(bp.solid_angle_cone(0.0), 0.0)
assert np.isclose(bp.solid_angle_cone(np.pi / 2), 2 * np.pi)
assert np.isclose(bp.solid_angle_cone(np.pi), 4 * np.pi)
# the equatorial loop gives exactly -pi
assert abs(wrap(bp.berry_phase_spin(np.pi / 2) - (-np.pi))) < 1e-2

# 3. GAUGE invariance: multiply each state by an arbitrary phase -> same Berry phase
states = bp.spin_loop_states(np.radians(70), n=300)
rng = np.random.default_rng(0)
regauged = [s * np.exp(1j * rng.uniform(0, 2 * np.pi)) for s in states]
assert abs(wrap(bp.berry_phase(states) - bp.berry_phase(regauged))) < 1e-9

# 4. SPEED / sampling independence: coarse vs fine loop give the same phase (geometric)
assert abs(wrap(bp.berry_phase_spin(np.radians(70), n=80) -
                bp.berry_phase_spin(np.radians(70), n=4000))) < 1e-2

# 5. a vanishing loop (tiny cone) gives ~zero Berry phase (no area enclosed)
assert abs(bp.berry_phase_spin(np.radians(2))) < 1e-2

# 6. the discrete Berry phase of a genuinely closed loop is real and gauge-invariant
g = bp.berry_phase(bp.spin_loop_states(np.radians(50)))
assert np.isreal(g)

print(f"TEST PASS  (Berry phase = -1/2 solid angle; equator -> -pi; gauge-invariant; "
      f"speed/sampling-independent; tiny loop -> 0)")
