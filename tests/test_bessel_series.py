"""Smoke-test the Frobenius series + recurrences + wave-equation modes in bessel.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mpmath as mp
import sympy as sp
from griffiths import bessel as bz

# 1. indicial roots of Bessel's equation are +/- m
assert bz.frobenius_indicial_roots(0) == [0, 0]
assert bz.frobenius_indicial_roots(3) == [3, -3]

# 2. the power series reproduces the closed-form J_m for several orders & arguments
for m in (0, 1, 2, 5):
    for x in (0.5, 2.4, 7.0, 13.0):
        approx = bz.bessel_J_series(m, x, terms=60)
        exact = float(mp.besselj(m, x))
        assert abs(approx - exact) < 1e-9, (m, x, approx, exact)

# 2b. works for non-integer order too (Gamma, not factorial)
assert abs(bz.bessel_J_series(0.5, 3.0, terms=60) - float(mp.besselj(0.5, 3.0))) < 1e-9

# 3. symbolic series: leading term of J_0 is 1, of J_1 is s/2
s = bz._s
assert sp.simplify(bz.bessel_J_series_terms(0, 1) - 1) == 0
assert sp.simplify(bz.bessel_J_series_terms(1, 1) - s / 2) == 0
# truncated symbolic series matches numeric at small x
f0 = sp.lambdify(s, bz.bessel_J_series_terms(0, 8), "mpmath")
assert abs(float(f0(1.0)) - float(mp.besselj(0, 1.0))) < 1e-6

# 4. recurrence + derivative ladder relations hold
for m in (1, 2, 4):
    for x in (1.3, 3.7, 9.1):
        rec, der = bz.bessel_recurrence_residual(m, x)
        assert abs(rec) < 1e-9 and abs(der) < 1e-9, (m, x, rec, der)

# 5. wave equation: drum modes from J_m zeros; fundamental uses J_0 zero 2.405
f = bz.circular_membrane_frequencies(0, 3, radius=1.0, speed=1.0)
assert abs(f[0] - 2.404825558 / (2 * 3.141592653589793)) < 1e-6
assert f[0] < f[1] < f[2]                                   # higher radial modes ring higher
# scaling: doubling the speed doubles every frequency; doubling radius halves them
f2 = bz.circular_membrane_frequencies(0, 3, radius=1.0, speed=2.0)
assert abs(f2[0] - 2 * f[0]) < 1e-9
fR = bz.circular_membrane_frequencies(0, 3, radius=2.0, speed=1.0)
assert abs(fR[0] - f[0] / 2) < 1e-9
# the (1,1) drum mode sits above the (0,1) fundamental (zero 3.83 > 2.40)
assert bz.circular_membrane_frequencies(1, 1)[0] > f[0]

print(f"SMOKE PASS  (series J_2(7.0)={bz.bessel_J_series(2, 7.0):.6f}; "
      f"drum f01/f02/f03 = {[round(v,3) for v in f]})")
