"""Smoke-test numerical complex analysis: contours, residues, Cauchy, winding."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import complex_numerics as cx

# 1. the fundamental fact: oint z^n dz = 0 for n != -1, = 2 pi i for n = -1
for nexp in (-3, 0, 2, 5):
    assert abs(cx.contour_integral(lambda z: z**nexp, 0, 1.0)) < 1e-6, nexp
assert abs(cx.contour_integral(lambda z: 1 / z, 0, 1.0) - 2j * np.pi) < 1e-6

# 2. residue theorem: oint dz/(z^2+1) around |z-i|=0.5 (encloses only z=i) = pi
f = lambda z: 1 / (z**2 + 1)
val = cx.contour_integral(f, center=1j, radius=0.5)
assert abs(val - np.pi) < 1e-4, val                    # 2 pi i * Res_i = 2 pi i /(2i) = pi
# residue at z=i is 1/(2i)
assert abs(cx.residues_sum(f, 1j, 0.5) - 1 / (2j)) < 1e-6
# a contour enclosing BOTH +/- i: residues cancel -> integral ~ 0
assert abs(cx.contour_integral(f, 0, 2.0)) < 1e-4

# 3. Cauchy integral formula recovers interior values from the boundary
assert abs(cx.cauchy_value(np.exp, 0.5, 0, 2.0) - np.e**0.5) < 1e-6
assert abs(cx.cauchy_value(lambda z: z**2 + 1, 0.3 + 0.4j, 0, 1.5)
           - ((0.3 + 0.4j)**2 + 1)) < 1e-6
try:
    cx.cauchy_value(np.exp, 5.0, 0, 2.0)               # z0 outside
except ValueError:
    pass
else:
    raise AssertionError("z0 outside contour should raise")

# 4. argument principle / winding number counts zeros minus poles
assert cx.winding_number(lambda z: z**3, 0, 1.0) == 3          # triple zero at 0
assert cx.winding_number(lambda z: z, 0, 1.0) == 1
assert cx.winding_number(lambda z: z**2 + 1, 1j, 0.5) == 1     # one zero (z=i) inside
assert cx.winding_number(lambda z: 1 / z, 0, 1.0) == -1        # a pole -> -1
assert cx.winding_number(lambda z: np.exp(z), 0, 1.0) == 0     # entire, no zeros

# 5. validation
for bad in (lambda: cx.contour_integral(lambda z: z, 0, -1.0),
            lambda: cx.contour_integral(lambda z: z, 0, 1.0, n=4)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad contour")

print(f"SMOKE PASS  (oint dz/(z^2+1)|_i = {val.real:.5f} ~ pi; "
      f"Cauchy & winding numbers exact -- complex analysis, numerically)")
