"""Test the three definitions of a derivative agree: math (exact symbolic/
limit value), physics (rate of change of sampled data), and computer
(autodiff dual numbers, exact; finite differences, approximate with a real
U-shaped error-vs-stepsize trade-off)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import dual_autodiff as ad
from dgs.numerical_methods import derivative as finite_diff_derivative, velocity

# 1. dual-number arithmetic: each operator overload reproduces the correct
#    calculus rule exactly, checked against known closed-form derivatives
x0 = 1.0

# power rule: d/dx x^3 = 3x^2
d = ad.Dual(x0, 1.0)
result = d ** 3
assert abs(result.val - x0**3) < 1e-12
assert abs(result.deriv - 3 * x0**2) < 1e-12

# product rule: d/dx [x * x^2] = d/dx x^3 = 3x^2 (via two separate Duals multiplied)
u = ad.Dual(x0, 1.0)
v = ad.Dual(x0**2, 2 * x0)   # v = x^2, v' = 2x, supplied directly
prod = u * v
assert abs(prod.val - x0**3) < 1e-12
assert abs(prod.deriv - 3 * x0**2) < 1e-9

# quotient rule: d/dx [x^3 / x] = d/dx x^2 = 2x
quot = (ad.Dual(x0**3, 3*x0**2)) / (ad.Dual(x0, 1.0))
assert abs(quot.val - x0**2) < 1e-12
assert abs(quot.deriv - 2*x0) < 1e-9

# 2. autodiff derivative of sin(x) is EXACT (0 error), matching cos(x)
for x_test in (0.0, 0.5, 1.0, 2.0, np.pi / 2):
    val, deriv = ad.autodiff_derivative(ad.dsin, x_test)
    assert abs(val - np.sin(x_test)) < 1e-14
    assert abs(deriv - np.cos(x_test)) < 1e-14

# chain rule through composition: d/dx sin(x^2) = cos(x^2)*2x
def f_composed(d):
    return ad.dsin(d ** 2)

val_c, deriv_c = ad.autodiff_derivative(f_composed, 1.3)
assert abs(val_c - np.sin(1.3 ** 2)) < 1e-12
assert abs(deriv_c - np.cos(1.3 ** 2) * 2 * 1.3) < 1e-10

# exp: d/dx exp(x) = exp(x)
val_e, deriv_e = ad.autodiff_derivative(ad.dexp, 0.7)
assert abs(deriv_e - np.exp(0.7)) < 1e-12

# 3. finite-difference error sweep: a genuine U-shape (error decreases then
#    increases as h shrinks), NOT monotonic -- and the minimum sits well
#    above zero (finite differences never reach autodiff's exactness)
hs = np.logspace(-1, -14, 30)
errors = ad.finite_difference_error_sweep(np.sin, np.cos, x0, hs)
assert errors[-1] > errors.min() * 10   # smallest h is much worse than the optimum (roundoff)
assert errors[0] > errors.min()          # largest h is worse than the optimum too (truncation)
assert errors.min() > 0                  # finite difference is never exactly zero error

# 4. the "physics definition" (rate of change from sampled data) matches
#    the analytic derivative of position for simple harmonic motion:
#    x(t) = A*cos(wt) -> v(t) = -A*w*sin(wt)
A, w = 2.0, 3.0
t = np.linspace(0, 4, 4000)
x_t = A * np.cos(w * t)
v_numeric = velocity(x_t, t)
v_analytic = -A * w * np.sin(w * t)
interior = slice(10, -10)   # avoid edge effects of np.gradient's one-sided ends
assert np.max(np.abs(v_numeric[interior] - v_analytic[interior])) < 1e-3

# 5. all three definitions agree with each other for the SAME function at
#    the SAME point: math (np.cos exact), computer/autodiff, computer/finite-diff
x_check = 0.8
math_def = np.cos(x_check)
_, computer_autodiff = ad.autodiff_derivative(ad.dsin, x_check)
computer_fd = finite_diff_derivative(np.sin, x_check, h=1e-6)
assert abs(math_def - computer_autodiff) < 1e-13
assert abs(math_def - computer_fd) < 1e-8

# 6. input validation
try:
    ad.Dual(1.0, 1.0) ** ad.Dual(2.0, 0.0)
    assert False, "should reject Dual ** Dual"
except TypeError:
    pass
try:
    ad.Dual(1.0, 1.0) / ad.Dual(0.0, 0.0)
    assert False, "should reject division by zero-valued dual"
except ZeroDivisionError:
    pass

print("all dgs.dual_autodiff tests passed")
