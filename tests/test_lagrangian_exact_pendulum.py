"""Test the exact (elliptic-integral) pendulum period in dgs/lagrangian.py:
the F_net=dp/dt -> energy-conservation identity, the small-angle limit,
the known 90-degree correction factor, and monotonic amplitude dependence."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import lagrangian as lg

# 1. energy conservation is verified, not assumed: dE/dt must be EXACTLY
#    proportional (ratio 1) to theta_dot times the F_net=dp/dt equation of motion
t = sp.symbols('t', real=True)
l, g = sp.symbols('l g', positive=True)
theta = sp.Function('theta')(t)
E, ratio = lg.pendulum_energy_conservation(theta, t, l, g)
assert ratio == 1

# 2. small-angle period matches the textbook T=2*pi*sqrt(L/g) formula exactly
L_val, g_val = 1.0, 9.81
T_small = lg.small_angle_period(L_val, g_val)
assert abs(T_small - 2*np.pi*np.sqrt(L_val/g_val)) < 1e-12

# 3. the exact period converges to the small-angle period as amplitude -> 0
T_exact_tiny = lg.exact_period(L_val, g_val, theta0=0.001)
assert abs(T_exact_tiny - T_small) / T_small < 1e-5

# 4. known tabulated correction factor at 90 degrees (~1.18)
factor_90 = lg.period_correction_factor(np.pi/2)
assert abs(factor_90 - 1.18) < 0.01

# 5. correction factor == 1 exactly in the theta0->0 limit, and strictly
#    increases with amplitude (approximation gets worse, never better)
amplitudes = np.array([0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
factors = np.array([lg.period_correction_factor(a) for a in amplitudes])
assert factors[0] < 1.001
assert np.all(np.diff(factors) > 0)

# 6. the ratio is independent of L and g (they cancel) -- same physical
#    pendulum shape at different L, g should have the same correction factor
f_a = lg.exact_period(1.0, 9.81, np.pi/3) / lg.small_angle_period(1.0, 9.81)
f_b = lg.exact_period(2.5, 3.71, np.pi/3) / lg.small_angle_period(2.5, 3.71)  # e.g. Mars gravity, different L
assert abs(f_a - f_b) < 1e-9

# 7. input validation
for bad_call in [
    lambda: lg.exact_period(-1.0, 9.81, 1.0),
    lambda: lg.exact_period(1.0, -9.81, 1.0),
    lambda: lg.exact_period(1.0, 9.81, 0.0),
    lambda: lg.exact_period(1.0, 9.81, np.pi),
    lambda: lg.exact_period(1.0, 9.81, -0.1),
    lambda: lg.small_angle_period(-1.0, 9.81),
    lambda: lg.period_correction_factor(0.0),
    lambda: lg.period_correction_factor(np.pi),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all exact-pendulum (elliptic integral) tests passed")
