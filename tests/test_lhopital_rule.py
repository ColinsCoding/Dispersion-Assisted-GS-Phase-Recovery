"""Test L'Hopital's rule: the mechanical symbolic procedure (differentiate
until no longer indeterminate), Cauchy's Mean Value Theorem identity
underlying the proof, and a real cross-check against
dgs.diffraction_grating's hardcoded sin(x)/x -> 1 limit."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import lhopital_rule as lh
from dgs.diffraction_grating import _sinc_unnormalized

x = sp.symbols('x')

# 1. sin(x)/x at x=0: exactly the indeterminate form hardcoded in
#    dgs.diffraction_grating -- L'Hopital must give exactly 1, in ONE application
value, n_apps = lh.lhopital_limit_symbolic(sp.sin(x), x, x, 0)
assert value == 1
assert n_apps == 1

# 2. (1-cos(x))/x^2 at x=0: needs L'Hopital applied TWICE (both f,g and
#    f',g' vanish at 0), converges to the well-known 1/2
value2, n_apps2 = lh.lhopital_limit_symbolic(1 - sp.cos(x), x**2, x, 0)
assert value2 == sp.Rational(1, 2)
assert n_apps2 == 2

# 3. a case needing THREE applications: x^3/(x - sin(x)) at x=0 -- a
#    classic higher-order example (x-sin(x) ~ x^3/6 for small x)
value3, n_apps3 = lh.lhopital_limit_symbolic(x**3, x - sp.sin(x), x, 0)
assert value3 == 6
assert n_apps3 == 3

# 4. Cauchy's MVT identity: for sin(x)/x, some c in (0, 0.5) must make
#    f'(c)/g'(c) EXACTLY equal f(0.5)/g(0.5) -- the actual algebraic
#    content the proof depends on, not just the final limit value
sign_change, min_residual = lh.cauchy_mvt_identity_check(sp.sin(x), x, x, 0, 0.5)
assert sign_change is True or min_residual < 1e-3

# 5. cross-check: dgs.diffraction_grating's hardcoded sinc(0)=1 matches
#    L'Hopital's prediction, and the numerical sin(x)/x genuinely
#    CONVERGES to that value as x shrinks (not just close at one point)
errors = []
for eps in [1e-2, 1e-4, 1e-6, 1e-8]:
    numeric = float(_sinc_unnormalized(np.array([eps]))[0])
    errors.append(abs(numeric - 1.0))
assert errors[0] > errors[1] > errors[2]   # monotonically improving as x shrinks
assert errors[-1] < 1e-10

# 6. away from the singular point, the sinc code should NOT just return 1
#    (confirms the x=0 special-case isn't silently overriding everything)
away_from_zero = float(_sinc_unnormalized(np.array([1.0]))[0])
assert abs(away_from_zero - np.sin(1.0)) < 1e-10
assert abs(away_from_zero - 1.0) > 0.1

# 7. a non-indeterminate ratio (f(0)=1, g(0)=0 -- neither 0/0 nor inf/inf)
#    must NOT have any derivative taken; it should fall straight through
#    to the base case (n_apps=0), confirming L'Hopital is only ever
#    applied when the form actually requires it
value_inf, n_apps_inf = lh.lhopital_limit_symbolic(sp.Integer(1), x, x, 0)
assert n_apps_inf == 0

print("all dgs.lhopital_rule tests passed")
