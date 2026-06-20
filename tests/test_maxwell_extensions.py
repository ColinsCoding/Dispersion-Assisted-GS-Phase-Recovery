"""Test extensions of Maxwell: magnetic monopoles + electric-magnetic duality."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from griffiths import electrodynamics as ed

# 1. symmetric Maxwell: Gauss-B gains a monopole source, Faraday a magnetic current
eqs = ed.maxwell_with_monopoles()
rho_m, Jm = sp.symbols("rho_m J_m")
divB = sp.Symbol("div_B")
curlE = sp.Symbol("curl_E")
dB_dt = sp.Symbol("dB/dt")
# div B = mu0 * rho_m  (no longer zero)
assert sp.simplify(eqs["Gauss_B"].rhs - ed.mu0 * rho_m) == 0
assert eqs["Gauss_B"].rhs != 0
# curl E = -mu0 J_m - dB/dt
assert sp.simplify(eqs["Faraday"].rhs - (-ed.mu0 * Jm - dB_dt)) == 0
# the two source-free equations match the ordinary ones when rho_m=J_m=0
assert sp.simplify(eqs["Gauss_B"].rhs.subs(rho_m, 0)) == 0
assert sp.simplify(eqs["Faraday"].rhs.subs(Jm, 0) - (-dB_dt)) == 0

# 2. duality at theta = pi/2 is the swap  E -> cB,  cB -> -E
E = np.array([1.0, 2.0, -1.0])
cB = np.array([0.5, -1.0, 3.0])
Ep, cBp = ed.duality_rotation(E, cB, np.pi / 2)
assert np.allclose(Ep, cB) and np.allclose(cBp, -E)

# 3. the energy-density invariant E^2 + (cB)^2 is preserved for ANY angle
inv0 = E @ E + cB @ cB
for th in np.linspace(0, 2 * np.pi, 12):
    e2, b2 = ed.duality_rotation(E, cB, th)
    assert abs((e2 @ e2 + b2 @ b2) - inv0) < 1e-9, th

# 4. a full turn (2*pi) returns the original fields
E2, cB2 = ed.duality_rotation(E, cB, 2 * np.pi)
assert np.allclose(E2, E) and np.allclose(cB2, cB)

# 5. duality composes by adding angles: R(a)R(b) = R(a+b)
a, b = 0.7, 1.1
e1, b1 = ed.duality_rotation(*ed.duality_rotation(E, cB, a), b)
e2, b2 = ed.duality_rotation(E, cB, a + b)
assert np.allclose(e1, e2) and np.allclose(b1, b2)

print("TEST PASS  (symmetric Maxwell: div B = mu0 rho_m; duality pi/2 swaps E<->cB; "
      "E^2+c^2B^2 invariant; rotations add)")
