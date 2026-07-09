"""Test dgs.continued_fractions: the CF coefficients of pi/sqrt2/golden ratio, the
convergents (pi -> 22/7, 355/113; sqrt2 = [1;2,2,..]; golden = Fibonacci ratios), the
1/q^2 error bound, exact reconstruction, and best_rational = Fraction.limit_denominator."""
import sys, pathlib, math
from fractions import Fraction
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import continued_fractions as cf

# 1. continued-fraction coefficients
assert cf.continued_fraction(math.pi, 5) == [3, 7, 15, 1, 292]
assert cf.continued_fraction(math.sqrt(2), 5) == [1, 2, 2, 2, 2]
assert cf.continued_fraction((1 + math.sqrt(5)) / 2, 6) == [1, 1, 1, 1, 1, 1]
# a rational number terminates
assert cf.continued_fraction(2.5) == [2, 2]                 # 2 + 1/2

# 2. convergents (best rational approximations)
pc = cf.convergents(cf.continued_fraction(math.pi, 5))
assert (22, 7) in pc and (355, 113) in pc                   # the famous ones
assert pc[0] == (3, 1) and pc[1] == (22, 7)
assert cf.convergents(cf.continued_fraction(math.sqrt(2), 6)) == \
    [(1, 1), (3, 2), (7, 5), (17, 12), (41, 29), (99, 70)]
# golden-ratio convergents are ratios of consecutive Fibonacci numbers
assert cf.convergents(cf.continued_fraction((1 + math.sqrt(5)) / 2, 8))[:6] == \
    [(1, 1), (2, 1), (3, 2), (5, 3), (8, 5), (13, 8)]

# 3. every convergent is closer than 1/q^2, and they improve monotonically
errs = []
for p, q in pc:
    e = cf.approximation_error(math.pi, p, q)
    assert e < 1 / q**2
    errs.append(e)
assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))    # strictly better

# 4. evaluate reconstructs the exact rational value
assert cf.evaluate([3, 7, 15, 1]) == Fraction(355, 113)
assert float(cf.evaluate([3, 7, 15, 1])) == 355 / 113
assert cf.evaluate([2, 3]) == Fraction(7, 3)                 # 2 + 1/3
assert cf.evaluate([5]) == Fraction(5)

# 5. best_rational agrees with Fraction.limit_denominator
assert cf.best_rational(math.pi, 200) == (355, 113)
assert cf.best_rational(math.pi, 100) == (311, 99)          # best with q <= 100
assert cf.best_rational(math.pi, 10) == (22, 7)
assert cf.best_rational(math.sqrt(2), 30) == (41, 29)
# the approximation really is close
p, q = cf.best_rational(math.pi, 113)
assert cf.approximation_error(math.pi, p, q) < 1e-6

# 6. kwarg bounds
for bad in (lambda: cf.continued_fraction(math.pi, 0),
            lambda: cf.convergents([]),
            lambda: cf.evaluate([]),
            lambda: cf.best_rational(math.pi, 0),
            lambda: cf.approximation_error(1.0, 1, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_continued_fractions: all checks passed")
