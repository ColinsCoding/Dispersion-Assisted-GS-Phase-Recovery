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

# 7. parity operator P: P^2 = I, eigenvalues are exactly +/-1
n = 201
P = eo.parity_matrix(n)
assert np.allclose(P @ P, np.eye(n))
w = np.sort(np.linalg.eigvalsh(P))
assert np.allclose(np.abs(w), 1.0)
assert np.sum(w > 0) == (n + 1) // 2 and np.sum(w < 0) == (n - 1) // 2

# 8. projectors E=(I+P)/2, O=(I-P)/2: idempotent, orthogonal, complete,
#    and applying E to samples IS even_part
E, O = eo.parity_projectors(n)
assert np.allclose(E @ E, E) and np.allclose(O @ O, O)
assert np.allclose(E @ O, 0) and np.allclose(E + O, np.eye(n))
xs = np.linspace(-3, 3, n)
fx = np.exp(xs)
assert np.allclose(E @ fx, eo.even_part(np.exp, xs))
assert np.allclose(O @ fx, eo.odd_part(np.exp, xs))

# 9. the (-1,0,1) stencil anticommutes with parity on interior rows:
#    P@D = -D@P  <=>  differentiation flips parity (cosh' = sinh)
D = eo.central_diff_matrix(xs)
anti = P @ D + D @ P
assert np.allclose(anti[1:-1, :], 0.0)
dcosh = D @ np.cosh(xs)
assert np.max(np.abs(dcosh[1:-1] - np.sinh(xs)[1:-1])) < 5e-3   # even -> odd (O(h^2))
assert np.max(np.abs((dcosh + dcosh[::-1])[1:-1])) < 1e-9       # result IS odd

# 10. bounds
for bad in (lambda: eo.parity_matrix(1),
            lambda: eo.central_diff_matrix(np.array([0.0, 1.0])),
            lambda: eo.central_diff_matrix(np.array([0.0, 1.0, 3.0]))):
    try:
        bad()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

print(f"TEST PASS  (cos/x^2 even, sin/x^3 odd; f=f_e+f_o exact; e^x->cosh+sinh; "
      f"odd integral=0; even int = 2x half = 2 sinh2 = {2*np.sinh(2):.4f}; "
      f"P^2=I w/ eigenvalues +/-1; E,O projectors match parts; PD=-DP)")
