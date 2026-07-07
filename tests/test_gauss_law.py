"""Test dgs.gauss_law: the symmetry-derived fields (point 1/r^2, line 1/r, sheet
uniform, solid sphere, shell, conductor) and -- the heart of it -- a direct
numerical verification that the flux of the Coulomb field over a sphere equals
Q_enclosed/eps0, independent of where the charge sits inside, and ZERO outside."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import gauss_law as gl

Q = 1e-9
k, eps0 = gl.K_COULOMB, gl.EPS0

# 1. flux = Q/eps0
assert np.isclose(gl.gauss_flux(Q), Q / eps0)
assert np.isclose(gl.gauss_flux(3 * Q), 3 * gl.gauss_flux(Q))     # linear in Q

# 2. point charge: k Q / r^2, falling as 1/r^2
assert np.isclose(gl.point_charge_field(Q, 1.0), k * Q)
assert np.isclose(gl.point_charge_field(Q, 1.0) / gl.point_charge_field(Q, 2.0), 4.0)

# 3. line charge: 2 k lambda / r, falling as 1/r
lam = 1e-9
assert np.isclose(gl.line_charge_field(lam, 1.0), lam / (2 * np.pi * eps0))
assert np.isclose(gl.line_charge_field(lam, 1.0), 2 * k * lam)
assert np.isclose(gl.line_charge_field(lam, 1.0) / gl.line_charge_field(lam, 2.0), 2.0)

# 4. infinite sheet: sigma/(2 eps0), uniform; conductor surface is exactly 2x
sigma = 1e-9
assert np.isclose(gl.sheet_field(sigma), sigma / (2 * eps0))
assert np.isclose(gl.conductor_surface_field(sigma), sigma / eps0)
assert np.isclose(gl.conductor_surface_field(sigma), 2 * gl.sheet_field(sigma))

# 5. uniform solid sphere: linear inside, 1/r^2 outside, continuous at r=R
R = 1.0
assert gl.uniform_sphere_field(Q, R, 0.0) == 0.0                  # zero at center
assert np.isclose(gl.uniform_sphere_field(Q, R, 0.5), k * Q * 0.5 / R**3)
assert np.isclose(gl.uniform_sphere_field(Q, R, 2.0), k * Q / 4)  # like a point charge
# continuity at the surface: both branches give k Q / R^2
assert np.isclose(gl.uniform_sphere_field(Q, R, R), k * Q / R**2)
assert np.isclose(gl.uniform_sphere_field(Q, R, R * (1 + 1e-9)), k * Q / R**2, rtol=1e-6)

# 6. thin shell: zero inside (no enclosed charge), 1/r^2 outside
assert gl.spherical_shell_field(Q, R, 0.5) == 0.0
assert np.isclose(gl.spherical_shell_field(Q, R, 2.0), k * Q / 4)

# 7. VERIFY THE LAW: numerically integrate the flux over a sphere
target = gl.gauss_flux(Q)
# charge at the center -> exact (constant integrand)
assert np.isclose(gl.numerical_flux([Q], [[0, 0, 0]], [0, 0, 0], 1.0), target, rtol=1e-6)
# charge off-center but INSIDE -> still Q/eps0 (position-independent)
assert np.isclose(gl.numerical_flux([Q], [[0.4, 0.2, 0]], [0, 0, 0], 1.0), target, rtol=2e-2)
# charge OUTSIDE -> zero net flux (field lines that enter also exit)
assert abs(gl.numerical_flux([Q], [[3.0, 0, 0]], [0, 0, 0], 1.0)) < 1e-3 * target
# only the ENCLOSED charge counts: one in, one out -> flux from the inside one alone
mixed = gl.numerical_flux([Q, 2 * Q], [[0, 0, 0], [3, 0, 0]], [0, 0, 0], 1.0)
assert np.isclose(mixed, gl.gauss_flux(Q), rtol=2e-2)
# two charges inside -> total enclosed
both = gl.numerical_flux([Q, 2 * Q], [[0.2, 0, 0], [-0.3, 0, 0]], [0, 0, 0], 1.0)
assert np.isclose(both, gl.gauss_flux(3 * Q), rtol=2e-2)

# 8. enclosed_charge counts only charges strictly inside the sphere
assert np.isclose(gl.enclosed_charge([Q, 2 * Q], [[0, 0, 0], [3, 0, 0]], [0, 0, 0], 1.0), Q)
assert np.isclose(gl.enclosed_charge([Q, 2 * Q], [[0, 0, 0], [0.5, 0, 0]], [0, 0, 0], 1.0), 3 * Q)

# 9. kwarg bounds
for bad in (lambda: gl.point_charge_field(Q, 0),
            lambda: gl.line_charge_field(lam, -1),
            lambda: gl.uniform_sphere_field(Q, 0, 1),
            lambda: gl.numerical_flux([Q], [[0, 0, 0]], [0, 0, 0], 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_gauss_law: all checks passed")
