"""Smoke-test Taylor/power series: e^x, sin/cos, Euler, small-angle, dispersion beta_2."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
import taylor as t

x = sp.Symbol("x")

# 1. e^x coefficients are 1/k!
assert t.taylor_coefficients(sp.exp(x), x, 0, 5) == [sp.Rational(1, sp.factorial(k)) for k in range(6)]

# 2. sin x = x - x^3/6 + x^5/120 ; cos x = 1 - x^2/2 + x^4/24
assert sp.simplify(t.taylor_series(sp.sin(x), x, 0, 5) - (x - x**3/6 + x**5/120)) == 0
assert sp.simplify(t.taylor_series(sp.cos(x), x, 0, 4) - (1 - x**2/2 + x**4/24)) == 0

# 3. Euler's formula as power series: e^{ix} = cos x + i sin x
lhs = t.taylor_series(sp.exp(sp.I*x), x, 0, 8)
rhs = t.taylor_series(sp.cos(x), x, 0, 8) + sp.I*t.taylor_series(sp.sin(x), x, 0, 8)
assert sp.simplify(lhs - rhs) == 0

# 4. small-angle approximation: error shrinks fast near 0, grows away
#    sin x ~ x: tiny error at x=0.1, bigger at x=1
assert t.truncation_error(sp.sin(x), x, 0, 1, sp.Rational(1, 10)) < 2e-4
assert t.truncation_error(sp.sin(x), x, 0, 1, 1) > 0.1
# more terms -> smaller error at fixed x
e1 = t.truncation_error(sp.exp(x), x, 0, 2, 1)
e2 = t.truncation_error(sp.exp(x), x, 0, 6, 1)
assert e2 < e1 and e2 < 1e-3

# 5. THE REPO TIE: dispersion beta(omega) Taylor -> beta_2 is the GVD.
#    Take a model beta = b0 + b1*(w-w0) + (1/2)*b2*(w-w0)^2 ; recover beta_2.
w, w0 = sp.symbols("omega omega_0", real=True)
b0, b1, b2 = sp.symbols("b0 b1 b2", real=True)
beta = b0 + b1*(w - w0) + sp.Rational(1, 2)*b2*(w - w0)**2
coeffs = t.dispersion_taylor(beta, w, w0, n=2)
assert sp.simplify(coeffs[0] - b0) == 0          # phase constant
assert sp.simplify(coeffs[1] - b1) == 0          # 1/group velocity
assert sp.simplify(coeffs[2] - b2) == 0          # beta_2 = GVD -> H(f)=exp(i pi D f^2)

print("SMOKE PASS  (e^x=1/k!; Euler holds as series; small-angle error grows with x; "
      "dispersion Taylor recovers beta_2 = GVD)")
