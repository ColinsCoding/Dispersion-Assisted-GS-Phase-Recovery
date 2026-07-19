"""Test the Gaussian probability integral table: closed forms (already
verified symbolically at import), odd-n vanishing, and normalization."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import probability_integrals as pi_

# 1. module import already ran _verify_table() -- reaching here means all 6
#    closed forms matched direct symbolic integration. Spot-check a couple
#    against the textbook's literal formulas at a concrete lambda.
lam = 3.0
assert abs(pi_.I_n(0, lam) - 0.5*np.sqrt(np.pi)*lam**-0.5) < 1e-12
assert abs(pi_.I_n(5, lam) - lam**-3) < 1e-12

# 2. odd-n full-line integrals are EXACTLY zero (symmetric cancellation)
for n in (1, 3, 5):
    assert pi_.full_line_integral(n, lam) == 0.0

# 3. even-n full-line integrals are exactly 2*I_n
for n in (0, 2, 4):
    assert abs(pi_.full_line_integral(n, lam) - 2*pi_.I_n(n, lam)) < 1e-12

# 4. I_n is a decreasing function of lambda (wider/sharper Gaussian -> smaller area)
lam_small, lam_large = 1.0, 10.0
for n in range(6):
    assert pi_.I_n(n, lam_small) > pi_.I_n(n, lam_large)

# 5. gaussian_normalization_constant genuinely normalizes psi = A*exp(-lambda x^2)
for lam_test in (0.5, 2.0, 7.0):
    A = pi_.gaussian_normalization_constant(lam_test)
    integral_check = A**2 * pi_.full_line_integral(0, 2*lam_test)
    assert abs(integral_check - 1.0) < 1e-9

# 6. vectorized lambda input works
lams = np.array([1.0, 2.0, 4.0])
vals = pi_.I_n(0, lams)
assert vals.shape == lams.shape
assert np.all(np.diff(vals) < 0)   # decreasing

# 7. input validation
try:
    pi_.I_n(6, 1.0)
    assert False, "should reject n outside 0..5"
except ValueError:
    pass
try:
    pi_.I_n(0, -1.0)
    assert False, "should reject non-positive lambda"
except ValueError:
    pass

print("all dgs.probability_integrals tests passed")
