"""Test analytical mechanics: Euler-Lagrange EOMs and small-oscillation normal modes."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import lagrangian as lag

t = sp.Symbol("t")
m, l, g, k = sp.symbols("m l g k", positive=True)

# 1. free particle (V=0): Euler-Lagrange gives m x'' = 0 -> Newton's first law
x = sp.Function("x")(t)
L_free = sp.Rational(1, 2) * m * x.diff(t)**2
assert sp.simplify(lag.euler_lagrange(L_free, x, t)) == m * x.diff(t, 2)
assert sp.simplify(lag.equation_of_motion(L_free, x, t)) == 0          # x'' = 0

# 2. harmonic oscillator: m x'' + k x = 0  ->  x'' = -(k/m) x
L_osc = lag.oscillator_lagrangian(x, t, m, k)
assert sp.simplify(lag.euler_lagrange(L_osc, x, t) - (m * x.diff(t, 2) + k * x)) == 0
assert sp.simplify(lag.equation_of_motion(L_osc, x, t) - (-k * x / m)) == 0

# 3. pendulum: theta'' = -(g/l) sin(theta)
th = sp.Function("theta")(t)
eom = lag.equation_of_motion(lag.pendulum_lagrangian(th, t, m, l, g), th, t)
assert sp.simplify(eom - (-g * sp.sin(th) / l)) == 0
# small-angle linearization: omega^2 = -d(theta'')/d(theta) at theta=0 = (g/l)cos0 = g/l
restoring = -sp.diff(eom, th).subs(th, 0)               # the SHM stiffness = omega^2
assert sp.simplify(restoring - g / l) == 0             # omega = sqrt(g/l)

# 4. coupled oscillators: normal modes at sqrt(k/m) and sqrt((k+2kc)/m)
K, M = lag.coupled_oscillator_KM(m=2.0, k=8.0, k_c=3.0)
w = lag.normal_mode_frequencies(M, K)
assert np.allclose(w, [np.sqrt(8/2), np.sqrt((8 + 2*3)/2)])            # [2.0, sqrt(7)]
# the lower (in-phase) mode is below the upper (out-of-phase) mode
assert w[0] < w[1]

# 5. normal_mode_frequencies solves the generalized eigenproblem K v = omega^2 M v
Kn = np.array([[8.0+3, -3], [-3, 8.0+3]]); Mn = 2.0*np.eye(2)
vals = np.sort(np.linalg.eigvals(np.linalg.solve(Mn, Kn)).real)
assert np.allclose(w**2, vals)                          # omega^2 are the eigenvalues

print(f"TEST PASS  (free particle x''=0; oscillator x''=-(k/m)x; pendulum "
      f"th''=-(g/l)sin th, small-angle omega=sqrt(g/l); coupled modes "
      f"{np.round(w,3)} = [sqrt(k/m), sqrt((k+2kc)/m)])")

# 6. central-force motion: theta is cyclic, and its conserved momentum is
#    exactly p_theta = m r^2 theta' (the standard specific-angular-momentum
#    result), verified from the Lagrangian itself, not assumed
r, theta = sp.Function("r")(t), sp.Function("theta")(t)
mu = sp.Symbol("mu", positive=True)
V_grav = -mu * m / r
L_central = lag.central_force_lagrangian(r, theta, t, m, V_grav)
is_cyclic, p_theta = lag.angular_momentum_conservation(L_central, theta, t)
assert is_cyclic is True
assert sp.simplify(p_theta - m * r ** 2 * theta.diff(t)) == 0

# a non-central potential (explicitly depends on theta) must NOT be flagged cyclic
V_noncentral = -mu * m / r + sp.sin(theta)
L_noncentral = lag.central_force_lagrangian(r, theta, t, m, V_noncentral)
is_cyclic_bad, _ = lag.angular_momentum_conservation(L_noncentral, theta, t)
assert is_cyclic_bad is False

print("dgs.lagrangian: central-force cyclic-coordinate checks passed")

# 7. verify_radial_eom_matches_effective_potential: m r'' = -dV_eff/dr is a
#    real algebraic identity, not assumed -- this is also the regression
#    test for a caught bug where substituting the phase-space EXPRESSION for
#    p_theta (rather than a free symbol standing for its conserved value)
#    silently failed to eliminate theta', making the check always report
#    False for the wrong reason
assert lag.verify_radial_eom_matches_effective_potential(r, theta, t, m, V_grav) is True

# it must also hold for a DIFFERENT central potential (not just gravity),
# confirming this is a general reduction, not something special-cased to
# the -mu*m/r form
V_harmonic_central = sp.Rational(1, 2) * mu * m * r ** 2   # isotropic 3D-harmonic-style radial term
assert lag.verify_radial_eom_matches_effective_potential(r, theta, t, m, V_harmonic_central) is True

# a genuinely non-central potential must be correctly rejected (ValueError),
# not silently produce a wrong answer
try:
    lag.verify_radial_eom_matches_effective_potential(r, theta, t, m, V_noncentral)
    raise AssertionError("expected ValueError for a non-central (theta-dependent) potential")
except ValueError:
    pass

print("dgs.lagrangian: radial EOM / effective potential checks passed "
      "(regression test for the caught p_theta-substitution bug)")

# 8. cross-module consistency: this module's effective-potential-derived
#    circular-orbit radius formula must reproduce the SAME orbit radius
#    dgs.rocket_equation_orbital_mechanics.circular_orbit_velocity was built
#    (and independently verified) against -- two different derivations of
#    the same physics agreeing, not just each individually plausible
from dgs.rocket_equation_orbital_mechanics import MU_EARTH, R_EARTH_M
r_leo = R_EARTH_M + 400e3
cross_check = lag.verify_circular_orbit_cross_check(r_leo, MU_EARTH)
assert cross_check["matches"] is True
assert abs(cross_check["r_from_effective_potential_m"] - r_leo) / r_leo < 1e-9

for bad in [dict(r_test_m=-1.0, mu=MU_EARTH), dict(r_test_m=r_leo, mu=-1.0)]:
    try:
        lag.verify_circular_orbit_cross_check(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.lagrangian: cross-check against dgs.rocket_equation_orbital_mechanics passed")
print("all dgs.lagrangian tests passed")
