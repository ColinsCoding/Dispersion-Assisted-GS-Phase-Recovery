"""Test the parametric-phase-model fix for the TDGSA flat-phase degeneracy
found in notebooks/ece279_tdgsa_recreation.ipynb: restricting the search to
a low-order polynomial in normalized time (instead of one free phase value
per sample) recovers the true chirp/cubic coefficients on both slide
pulses, where free-form GS (2 planes, 3 planes, 2000 random restarts) got
stuck in degenerate solutions."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.tdgsa_parametric import fit_parametric_phase, evaluate_polynomial_phase

fs = 200e9
N = int(round(20e-9 * fs))
t_ns = np.linspace(-10, 10, N)
dt_ps = float((t_ns[1] - t_ns[0]) * 1000.0)
T0_ns = 2.0
tau = t_ns / T0_ns


def dispersed_intensity(E, D_ps_per_nm, dt_ps, N, lambda_nm=1550.0):
    C_LIGHT = 299_792_458.0
    lam_m = lambda_nm * 1e-9
    beta2L_ps2 = -(D_ps_per_nm * 1e-3) * lam_m ** 2 / (2 * np.pi * C_LIGHT) * 1e24
    f = np.fft.fftfreq(N, dt_ps)
    H = np.exp(1j * 0.5 * beta2L_ps2 * (2 * np.pi * f) ** 2)
    return np.abs(np.fft.ifft(np.fft.fft(E) * H)) ** 2


# 1. gas-cell pulse (pure quadratic phase, the slide's own D1/D2): recovers
# the true coefficient [0, 0, -0.5] essentially exactly -- and does so
# starting from the SAME zero-init that trapped free-form GS
envelope_gas = np.exp(-0.5 * tau ** 2)
true_theta_gas = np.array([0.0, 0.0, -0.5])
E_true_gas = envelope_gas * np.exp(1j * evaluate_polynomial_phase(true_theta_gas, tau))
I1_gas = dispersed_intensity(E_true_gas, -353.0, dt_ps, N)
I2_gas = dispersed_intensity(E_true_gas, -872.0, dt_ps, N)

theta_gas, loss_gas = fit_parametric_phase(I1_gas, I2_gas, -353.0, -872.0, dt_ps,
                                            tau, envelope_gas, degree=2)
assert loss_gas < 1e-8, f"expected near-exact fit, got loss={loss_gas}"
assert abs(theta_gas[2] - (-0.5)) < 1e-3, f"expected leading coefficient ~-0.5, got {theta_gas[2]}"

# 2. cubic-phase pulse: free-form gradient descent from zero-init lands in a
# shallow local minimum (verified separately -- wrong coefficients, loss
# ~1e-4). The grid-search step in fit_parametric_phase must find the true,
# much sharper global minimum instead.
tau5 = t_ns / (5 * T0_ns)
envelope_cubic = np.exp(-0.5 * tau5 ** 2)
true_theta_cubic = np.array([0.0, 0.0, 0.0, 0.06])
E_true_cubic = envelope_cubic * np.exp(1j * evaluate_polynomial_phase(true_theta_cubic, tau))
I1_cubic = dispersed_intensity(E_true_cubic, -600.0, dt_ps, N)
I2_cubic = dispersed_intensity(E_true_cubic, -900.0, dt_ps, N)

theta_cubic, loss_cubic = fit_parametric_phase(I1_cubic, I2_cubic, -600.0, -900.0, dt_ps,
                                                tau, envelope_cubic, degree=3)
assert loss_cubic < 1e-8, f"expected near-exact fit, got loss={loss_cubic}"
assert abs(theta_cubic[3] - 0.06) < 1e-3, f"expected leading coefficient ~0.06, got {theta_cubic[3]}"
assert abs(theta_cubic[1]) < 1e-2, "spurious linear term (the known local-minimum failure mode) should be gone"

# 3. sanity: a grid search restricted to the WRONG side of the true value
# (coef_range too small to contain it) should fail to find it -- confirms
# the grid search is actually doing the work, not gradient descent alone
theta_narrow, loss_narrow = fit_parametric_phase(I1_cubic, I2_cubic, -600.0, -900.0, dt_ps,
                                                  tau, envelope_cubic, degree=3,
                                                  coef_range=0.02, n_grid=200)
assert loss_narrow > loss_cubic * 1e3, "expected a much worse fit when the true coefficient is outside the grid range"

print("all dgs.tdgsa_parametric tests passed")
