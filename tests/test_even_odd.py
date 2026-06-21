"""Test even/odd decomposition, symmetry classification, and symmetric integrals."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import even_odd as eo

x = np.linspace(-4, 4, 2001)

# 1. classification: cos/x^2/|x| even; sin/x^3/x odd
assert eo.is_even(np.cos, x) and eo.is_even(lambda t: t**2, x) and eo.is_even(np.abs, x)
assert eo.is_odd(np.sin, x) and eo.is_odd(lambda t: t**3, x) and eo.is_odd(lambda t: t, x)
# a generic function is neither
assert not eo.is_even(np.exp, x) and not eo.is_odd(np.exp, x)

# 2. decomposition reconstructs f exactly, and the parts have the right symmetry
fe, fo = eo.decompose(np.exp, x)
assert np.max(np.abs((fe + fo) - np.exp(x))) < 1e-12        # f = f_e + f_o
assert np.max(np.abs(fe - fe[::-1])) < 1e-12               # even part is symmetric
assert np.max(np.abs(fo + fo[::-1])) < 1e-12               # odd part is antisymmetric

# 3. e^x parts ARE cosh and sinh
assert np.allclose(fe, np.cosh(x)) and np.allclose(fo, np.sinh(x))

# 4. symmetric integral of an odd function is ZERO
assert abs(eo.symmetric_integral(lambda t: t**3 + np.sin(t), 3.0)) < 1e-9

# 5. symmetric integral of an even function = 2 * the half-integral
full = eo.symmetric_integral(np.cos, 1.0)
assert abs(full - 2 * np.sin(1.0)) < 1e-4                   # int_-1^1 cos = 2 sin 1

# 6. the shortcut: int_-a^a f = 2 int_0^a f_even  (odd part drops, even for general f)
direct = eo.symmetric_integral(np.exp, 2.0)
short = eo.even_integral_shortcut(np.exp, 2.0)
assert abs(direct - short) < 1e-3
assert abs(direct - 2 * np.sinh(2.0)) < 1e-3               # = 2 sinh 2

print(f"TEST PASS  (cos/x^2 even, sin/x^3 odd; f=f_e+f_o exact; e^x->cosh+sinh; "
      f"odd integral=0; even int = 2x half = 2 sinh2 = {2*np.sinh(2):.4f})")
