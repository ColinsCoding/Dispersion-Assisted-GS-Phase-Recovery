"""Test that a molecular (Lorentz) optical resonance, a driven mechanical
oscillator, a series RLC circuit, and an op-amp analog computer all
solve the SAME differential equation -- verified algebraically and
numerically, with the two time-convention sign subtleties (optics
exp(-i*omega*t) vs engineering exp(+j*omega*t)) handled explicitly rather
than papered over."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import analog_computing_universality as acu

# 1. mechanical oscillator's independently-derived steady-state response
#    is algebraically IDENTICAL to dgs.causality.lorentz_susceptibility
#    (same optics exp(-i*omega*t) convention)
match_mech, X_relabeled, lorentz_expr = acu.verify_mechanical_matches_molecular()
assert match_mech is True

# 2. RLC circuit's admittance (engineering exp(+j*omega*t) convention,
#    built from dgs.ac_circuits' actual tested impedance functions) matches
#    the SAME universal resonance denominator, in ITS OWN convention
assert acu.verify_rlc_matches_universal_form() is True

# 3. op-amp analog computer (feedback-loop integrator recursion) matches
#    an independent RK4 integration of the SAME ODE, for a step force
omega0, gamma = 2.0, 0.3
F_step = lambda t: 1.0 if t > 0.1 else 0.0
t = np.linspace(0, 10, 2000)
x_rk4 = acu.solve_oscillator_rk4(omega0, gamma, F_step, t)
x_analog = acu.solve_oscillator_analog_computer(omega0, gamma, F_step, t)
assert np.max(np.abs(x_rk4 - x_analog)) < 1e-2   # first-order integrator recursion vs RK4

# refining the analog computer's own timestep should shrink the gap further
# (confirms convergence, not a coincidental match at this one resolution)
t_fine = np.linspace(0, 10, 20000)
x_rk4_fine = acu.solve_oscillator_rk4(omega0, gamma, F_step, t_fine)
x_analog_fine = acu.solve_oscillator_analog_computer(omega0, gamma, F_step, t_fine)
err_coarse = np.max(np.abs(x_rk4 - x_analog))
err_fine = np.max(np.abs(x_rk4_fine - x_analog_fine))
assert err_fine < err_coarse

# 4. a DIFFERENT forcing function (sinusoidal drive) also matches --
#    confirms the analog computer isn't just tuned to the one step-force case
F_sine = lambda t: np.sin(1.5 * t)
x_rk4_sine = acu.solve_oscillator_rk4(omega0, gamma, F_sine, t)
x_analog_sine = acu.solve_oscillator_analog_computer(omega0, gamma, F_sine, t)
assert np.max(np.abs(x_rk4_sine - x_analog_sine)) < 1e-2

# 5. sanity: with F=0, both solvers correctly report x=0 for all time
#    (zero initial conditions, zero forcing -> trivial solution)
F_zero = lambda t: 0.0
x_rk4_zero = acu.solve_oscillator_rk4(omega0, gamma, F_zero, t)
x_analog_zero = acu.solve_oscillator_analog_computer(omega0, gamma, F_zero, t)
assert np.allclose(x_rk4_zero, 0.0)
assert np.allclose(x_analog_zero, 0.0)

print("all dgs.analog_computing_universality tests passed")
