"""Test numerical calculus against analytic answers (power rule, FTC, flux rule)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import numerical_methods as nm

# 1. finite-difference derivative recovers the POWER RULE: d/dx x^n = n x^{n-1}
for n in (2, 3, 5):
    x0 = 1.7
    assert abs(nm.derivative(lambda x: x**n, x0) - n * x0**(n - 1)) < 1e-4, n
# second derivative of x^3 at 2 is 6x = 12
assert abs(nm.second_derivative(lambda x: x**3, 2.0) - 12.0) < 1e-3

# 2. central difference is O(h^2): halving h cuts the error ~4x
err = lambda h: abs(nm.derivative(np.sin, 1.0, h) - np.cos(1.0))
assert err(1e-2) / err(5e-3) > 3.5          # ~4x improvement -> 2nd-order accurate

# 3. kinematics: x(t)=t^2 -> v=2t, a=2 (constant)
t = np.linspace(0, 5, 500)
v = nm.velocity(t**2, t)
a = nm.acceleration(t**2, t)
assert np.max(np.abs(v[2:-2] - 2 * t[2:-2])) < 1e-2     # interior (ends are one-sided)
assert np.max(np.abs(a[3:-3] - 2.0)) < 1e-2

# 4. integration: integral_0^pi sin x dx = 2; Simpson (O(h^4)) far beats trapezoid
xs = np.linspace(0, np.pi, 101)
trap_err = abs(nm.trapezoid(np.sin(xs), xs) - 2.0)
simp_err = abs(nm.simpson(np.sin(xs), xs) - 2.0)
assert trap_err < 1e-3
assert simp_err < 1e-6                                   # Simpson ~1e-8
assert simp_err < trap_err / 1000                        # O(h^4) crushes O(h^2)
# integral_0^3 x^2 dx = 9
xs2 = np.linspace(0, 3, 201)
assert abs(nm.simpson(xs2**2, xs2) - 9.0) < 1e-6

# 5. numerical Fundamental Theorem: d/dx of the cumulative integral returns y
xs3 = np.linspace(0, 2 * np.pi, 400)
F = nm.cumulative_integral(np.cos(xs3), xs3)            # integral of cos = sin
assert np.max(np.abs(nm.gradient(F, xs3)[2:-2] - np.cos(xs3)[2:-2])) < 1e-3

# 6. Taylor approximation from numerical derivatives: e^x about 0, order 5
approx = nm.taylor_approx(np.exp, 0.0, 5, 0.5)
assert abs(approx - np.exp(0.5)) < 1e-3

# 7. THE FLUX RULE (Griffiths 7.13): x(t)=x0 - v t, Phi=B h x -> dPhi/dt = -B h v
B, h, v0, x0 = 0.5, 2.0, 3.0, 10.0
t = np.linspace(0, 1, 300)
dphi = nm.motional_flux_rate(B, h, x0 - v0 * t, t)
assert abs(np.mean(dphi) - (-B * h * v0)) < 1e-6        # dPhi/dt = -B h v = -3
emf = -np.mean(dphi)
assert abs(emf - B * h * v0) < 1e-6                     # eps = -dPhi/dt = +B h v = 3

print(f"TEST PASS  (power rule num=analytic; central diff O(h^2); v=2t,a=2; "
      f"Simpson exact to 1e-8; FTC round-trips; Taylor e^0.5 ok; "
      f"flux rule dPhi/dt=-Bhv={-B*h*v0:.0f}, emf={emf:.0f})")
