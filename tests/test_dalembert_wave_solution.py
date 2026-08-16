"""Test dgs/dalembert_wave_solution.py: the general two-term wave equation
solution psi=f(x-vt)+g(x+vt), and d'Alembert's IVP formula verified
against the wave equation and both initial conditions simultaneously, plus
the two concrete examples (displacement-only splitting into half-amplitude
pulses; velocity-only isolating the integral term, checked against
brute-force numerical quadrature)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.dalembert_wave_solution import (
    verify_general_solution_solves_wave_eq, dalembert_formula_symbolic,
    gaussian_pulse, dalembert_displacement_only,
    verify_splits_into_half_amplitude_pulses,
    dalembert_velocity_only_gaussian, verify_velocity_integral_closed_form,
)

# 1. verify_general_solution_solves_wave_eq: must pass
assert verify_general_solution_solves_wave_eq() is True

# 2. dalembert_formula_symbolic: all three residuals exactly 0
psi_symbolic, residuals = dalembert_formula_symbolic()
for name, r in residuals.items():
    assert r == 0, f"{name} residual not zero: {r}"

# 3. gaussian_pulse: known values -- peak at x0, half-max at x0+/-sigma*sqrt(2*ln2)
peak = gaussian_pulse(np.array([0.0]), amplitude=2.0, x0=0.0, sigma=1.0)[0]
assert abs(peak - 2.0) < 1e-12
try:
    gaussian_pulse(np.array([0.0]), 1.0, 0.0, -1.0)
    raise AssertionError("expected ValueError for sigma <= 0")
except ValueError:
    pass

# 4. dalembert_displacement_only: at t=0, psi(x,0) must equal phi(x) exactly
#    (the two half-pulses haven't separated yet -- they coincide)
x = np.linspace(-10, 10, 500)
psi_t0 = dalembert_displacement_only(x, 0.0, v=1.0, amplitude=1.0, x0=0.0, sigma=0.5)
phi_x = gaussian_pulse(x, 1.0, 0.0, 0.5)
assert np.max(np.abs(psi_t0 - phi_x)) < 1e-12

# 5. dalembert_displacement_only: input validation
try:
    dalembert_displacement_only(x, 1.0, v=-1.0, amplitude=1.0, x0=0.0, sigma=0.5)
    raise AssertionError("expected ValueError for v <= 0")
except ValueError:
    pass

# 6. verify_splits_into_half_amplitude_pulses: must pass, and the two
#    peaks must be symmetric about x=0 (odd/even symmetry of the setup)
result = verify_splits_into_half_amplitude_pulses()
for name, ok in result["checks"].items():
    assert ok, f"{name} failed"
assert abs(result["right_peak_x"] + result["left_peak_x"]) < 1e-6
assert abs(result["right_peak_val"] - result["left_peak_val"]) < 1e-9

# 7. energy/area conservation: total area under psi(x,t) must stay constant
#    over time (the two pulses carry away exactly the original pulse's
#    total area between them -- a real physical conservation check, not
#    just "the peaks look right")
dx = x[1] - x[0]
area_t0 = np.trapezoid(dalembert_displacement_only(x, 0.0, 1.0, 1.0, 0.0, 0.5), dx=dx)
x_late = np.linspace(-30, 30, 20000)
area_t_late = np.trapezoid(dalembert_displacement_only(x_late, 15.0, 1.0, 1.0, 0.0, 0.5),
                        dx=x_late[1] - x_late[0])
assert abs(area_t0 - area_t_late) / area_t0 < 1e-3

# 8. dalembert_velocity_only_gaussian: at t=0, the integral collapses to
#    zero (integrating over a zero-width interval [x-0, x+0])
psi_v_t0 = dalembert_velocity_only_gaussian(x, 0.0, v=1.0, amplitude=1.0, x0=0.0, sigma=0.5)
assert np.max(np.abs(psi_v_t0)) < 1e-9

# 9. verify_velocity_integral_closed_form: erf closed form vs. brute-force
#    quadrature must agree
assert verify_velocity_integral_closed_form() is True

# 10. dalembert_velocity_only_gaussian: input validation
try:
    dalembert_velocity_only_gaussian(x, 1.0, v=1.0, amplitude=1.0, x0=0.0, sigma=-0.5)
    raise AssertionError("expected ValueError for sigma <= 0")
except ValueError:
    pass

print("all dgs.dalembert_wave_solution tests passed")
