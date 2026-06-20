"""Smoke-test delta_composition: delta(g(x)) = sum delta(x-x_i)/|g'(x_i)|."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths.deltas import delta_composition, delta_rescale
from griffiths.vectors import x

# 1. scaling special case: delta(2x) = delta(x)/2 (matches delta_rescale)
expr, roots = delta_composition(2 * x)
assert roots == [0]
assert sp.simplify(expr - sp.DiracDelta(x) / 2) == 0
assert sp.simplify(expr - delta_rescale(2).rhs) == 0

# 2. quadratic: delta(x^2 - 4) -> [delta(x-2)+delta(x+2)] / 4  (g'=2x, |+-4|=4)
expr, roots = delta_composition(x**2 - 4)
assert set(roots) == {-2, 2}
expected = (sp.DiracDelta(x - 2) + sp.DiracDelta(x + 2)) / 4
assert sp.simplify(expr - expected) == 0

# 3. the sifting weight is right: ∫ x^2 delta(x^2-4) dx over [-3,3] = 2/4 *2 *4 = 2
val = sp.integrate(x**2 * expr, (x, -3, 3))
assert sp.simplify(val - 2) == 0, val

# 4. shifted linear: delta(3x - 6) fires at x=2 with weight 1/3
expr, roots = delta_composition(3 * x - 6)
assert roots == [2]
assert sp.simplify(expr - sp.DiracDelta(x - 2) / 3) == 0

# 5. tangent zero is rejected: delta(x^2) has g'(0)=0 -> singular
try:
    delta_composition(x**2)
except ValueError:
    pass
else:
    raise AssertionError("should reject tangent (double) zero where g'=0")

# 6. complex roots are dropped: delta(x^2 + 1) has no real zeros -> empty sum
expr, roots = delta_composition(x**2 + 1)
assert roots == [] and expr == 0

print("SMOKE PASS  (delta(x^2-4) -> [d(x-2)+d(x+2)]/4, sifted integral = 2)")
