"""Test dgs.symmetry_physics: polar vs axial vectors (E flips under a mirror, B
does not, because B is a cross product), the P/T/C field transforms, the Lorentz
force being parity- and time-reversal-consistent, and Noether's theorem -- a
central-force orbit conserving angular momentum (rotational symmetry) and energy
(time symmetry) to numerical precision."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import symmetry_physics as sym

# 1. parity: polar flips, axial does not
assert np.allclose(sym.parity([1, 2, 3]), [-1, -2, -3])          # polar (default)
assert np.allclose(sym.parity([1, 2, 3], axial=True), [1, 2, 3]) # axial unchanged
# time reversal: t-odd quantities flip
assert np.allclose(sym.time_reversal([1, 2, 3], t_odd=True), [-1, -2, -3])
assert np.allclose(sym.time_reversal([1, 2, 3], t_odd=False), [1, 2, 3])

# 2. WHY B is axial: (Pa) x (Pb) = a x b -- a cross of polar vectors is invariant
assert sym.cross_product_is_axial([1, 2, 3], [4, 5, 6])
a, b = np.array([1., 0, 0]), np.array([0., 1, 0])
assert np.allclose(np.cross(-a, -b), np.cross(a, b))            # (=+z, unflipped)

# 3. the quantity table: E is polar (P=-1), B is axial (P=+1) -- they DIFFER
assert sym.QUANTITIES["E_field"][0] == "polar" and sym.QUANTITIES["E_field"][1] == -1
assert sym.QUANTITIES["B_field"][0] == "axial" and sym.QUANTITIES["B_field"][1] == +1
assert sym.QUANTITIES["angular_momentum"][0] == "axial"
assert sym.QUANTITIES["momentum"] == ("polar", -1, -1)

# 4. field transforms under the discrete symmetries
E, B = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
Ep, Bp = sym.transform_fields(E, B, "P")
assert np.allclose(Ep, [-1, -2, -3]) and np.allclose(Bp, [4, 5, 6])   # E flips, B doesn't
Et, Bt = sym.transform_fields(E, B, "T")
assert np.allclose(Et, E) and np.allclose(Bt, [-4, -5, -6])          # B flips under T
Ec, Bc = sym.transform_fields(E, B, "C")
assert np.allclose(Ec, [-1, -2, -3]) and np.allclose(Bc, [-4, -5, -6])
try:
    sym.transform_fields(E, B, "X"); assert False
except ValueError:
    pass

# 5. Lorentz force value, and its P / T consistency for several inputs
assert np.allclose(sym.lorentz_force(2.0, [1, 0, 0], [0, 1, 0], [0, 0, 1]),
                   2.0 * (np.array([1, 0, 0]) + np.cross([0, 1, 0], [0, 0, 1])))
rng = np.random.default_rng(0)
for _ in range(20):
    q = rng.uniform(-3, 3)
    E, v, B = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    assert sym.lorentz_is_parity_consistent(q, E, v, B)
    assert sym.lorentz_is_time_reversal_consistent(q, E, v, B)

# 6. Noether: a central-force orbit conserves L (rotation) and E (time)
orb = sym.simulate_central_orbit([1.0, 0.0], [0.0, 1.2], mu=1.0, t_end=20.0, dt=1e-3)
assert orb["L_drift"] < 1e-8                    # rotational symmetry -> L conserved
assert orb["E_drift"] < 1e-8                    # time symmetry       -> E conserved
# L matches its analytic initial value throughout (x v_y - y v_x = 1*1.2)
assert np.allclose(orb["L"], 1.2, atol=1e-8)
# the orbit is bound (an ellipse), not flung to infinity
radii = np.linalg.norm(orb["trajectory"][:, :2], axis=1)
assert radii.max() < 10.0 and radii.min() > 0.1

# 7. formula spot checks
assert np.isclose(sym.angular_momentum_2d([1, 0], [0, 1.2]), 1.2)
assert np.isclose(sym.energy_2d([1, 0], [0, 1.2], mu=1.0), 0.5 * 1.2**2 - 1.0)

# 8. kwarg bounds
for bad in (lambda: sym.transform_fields(E, B, "bad"),
            lambda: sym.simulate_central_orbit([1, 0], [0, 1], mu=0),
            lambda: sym.simulate_central_orbit([1, 0], [0, 1], dt=0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_symmetry_physics: all checks passed")
