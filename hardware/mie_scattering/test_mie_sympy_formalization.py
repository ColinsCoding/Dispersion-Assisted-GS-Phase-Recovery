"""Test mie_sympy_formalization.py: the Riccati-Bessel recurrence is
symbolically PROVEN (not numerically approximated) for several n using
sympy's own spherical Bessel function; the m=1 "invisible sphere"
identity is proven as a pure algebraic fact about the a_n formula's
structure (true for ANY function, not just Bessel functions); and that
symbolic result is cross-checked against the ACTUAL numeric
implementation in generate_mie_reference.py, which must independently
agree the sphere scatters nothing at m=1."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import sympy as sp

from mie_sympy_formalization import (
    riccati_bessel_recurrence_symbolic, m_equals_one_limit_symbolic,
    verify_m_equals_one_numerically,
)

# 1. the Riccati-Bessel recurrence is proven EXACTLY (sympy returns the
# literal integer 0, not "close to zero") for every n tested
results = riccati_bessel_recurrence_symbolic(n_values=(1, 2, 3, 4, 5))
for n, (diff, proven) in results.items():
    assert proven, f"n={n}: recurrence should simplify to exactly 0, got {diff}"
    assert diff == sp.Integer(0) or diff == 0, (n, diff)

# 2. the m=1 limit is proven as an EXACT symbolic zero for an abstract
# function -- a statement about the a_n formula's algebra, not a
# numerical coincidence
simplified, proven = m_equals_one_limit_symbolic()
assert proven, f"a_n numerator at m=1 should simplify to exactly 0, got {simplified}"
assert simplified == 0

# 3. cross-check: the ACTUAL numeric BHMIE implementation must
# independently agree that an m=1 sphere scatters (almost) nothing --
# "almost" because floating point, not exactly symbolic zero, so
# checked against a tight but nonzero tolerance
Qext, Qsca = verify_m_equals_one_numerically(x_test=5.0)
assert abs(Qext) < 1e-20, f"m=1 sphere should scatter essentially nothing, got Qext={Qext:.2e}"
assert abs(Qsca) < 1e-20, f"m=1 sphere should scatter essentially nothing, got Qsca={Qsca:.2e}"

# 4. the numerical agreement should hold at OTHER x values too, not
# just one convenient test point
for x_test in (0.5, 2.0, 8.0, 15.0):
    Qext_x, Qsca_x = verify_m_equals_one_numerically(x_test=x_test)
    assert abs(Qext_x) < 1e-15, (x_test, Qext_x)
    assert abs(Qsca_x) < 1e-15, (x_test, Qsca_x)

print(f"all mie_sympy_formalization tests passed "
      f"(Riccati-Bessel recurrence proven exactly for n=1..5 via sympy's jn; "
      f"m=1 'invisible sphere' identity proven as an exact algebraic fact "
      f"for an abstract function, not tied to the Bessel functional form; "
      f"numeric cross-check at x=5.0: Qext={Qext:.2e}, Qsca={Qsca:.2e}, "
      f"confirmed at 4 additional x values)")
