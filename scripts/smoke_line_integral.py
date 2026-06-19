"""Smoke-test line integrals: conservative=KVL (loop 0), non-conservative=EMF (Stokes)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths import fields as fld
from griffiths.vectors import CARTESIAN, curl

x, y, z = CARTESIAN
t = sp.Symbol("t", real=True)
unit_circle = [sp.cos(t), sp.sin(t), 0]          # x=cos t, y=sin t, z=0

# 1. CONSERVATIVE field F = grad(x^2+y^2) = (2x, 2y, 0): closed loop = 0  (KVL)
Fc = [2 * x, 2 * y, 0]
assert fld.is_conservative(Fc)
assert sp.simplify(fld.circulation(Fc, unit_circle, t)) == 0

# 2. NON-conservative field F = (-y, x, 0): circulation = 2*pi  (curl=2 z-hat,
#    times area pi -> 2*pi). This is the Faraday EMF that breaks naive KVL.
Fn = [-y, x, 0]
assert not fld.is_conservative(Fn)
assert sp.simplify(fld.circulation(Fn, unit_circle, t)) == 2 * sp.pi
# Stokes check: circulation == (curl . z-hat) * area = 2 * pi*r^2  (r=1)
curl_z = curl(Fn, CARTESIAN)[2]
assert sp.simplify(curl_z) == 2                  # curl = 2 z-hat
assert sp.simplify(fld.circulation(Fn, unit_circle, t) - curl_z * sp.pi) == 0

# 3. path independence for the conservative field: two different paths,
#    same endpoints (0,0)->(1,1), give the same work
straight = [t, t, 0]                             # t: 0 -> 1
parabola = [t, t**2, 0]                          # t: 0 -> 1
w1 = fld.line_integral(Fc, straight, t, 0, 1)
w2 = fld.line_integral(Fc, parabola, t, 0, 1)
assert sp.simplify(w1 - w2) == 0 and sp.simplify(w1 - 2) == 0   # = V(1,1)-V(0,0)=2

# 4. the non-conservative field is path-dependent (same endpoints, different work)
w1n = fld.line_integral(Fn, straight, t, 0, 1)
w2n = fld.line_integral(Fn, parabola, t, 0, 1)
assert sp.simplify(w1n - w2n) != 0

print(f"SMOKE PASS  (conservative loop=0 [KVL]; rotational circulation=2*pi=EMF [Stokes]; "
      f"conservative work path-indep = {w1})")
