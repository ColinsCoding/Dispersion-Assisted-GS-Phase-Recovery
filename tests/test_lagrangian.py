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
