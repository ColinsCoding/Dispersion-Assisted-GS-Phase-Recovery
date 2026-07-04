"""Test the heat equation solved as a PDE transformed into frequency space:
cross-check the FFT-based per-mode-decay solver against an independent
finite-difference solver AND the analytic self-similar Gaussian solution."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import heat_equation_fourier as heq

alpha = 0.01
sigma0 = 0.3
L = 20.0
n = 2048
x = np.linspace(-L / 2, L / 2, n, endpoint=False)
dx = x[1] - x[0]

# 1. Gaussian stays Gaussian; total heat (integral) is conserved at t=0 and later
T0 = heq.heat_kernel_gaussian(x, 0.0, alpha, sigma0)
Q0 = np.trapezoid(T0, x)
T_later = heq.heat_kernel_gaussian(x, 3.0, alpha, sigma0)
Q_later = np.trapezoid(T_later, x)
assert abs(Q0 - Q_later) / Q0 < 1e-3

# 2. variance grows linearly in time: sigma(t)^2 = sigma0^2 + 2*alpha*t
#    verified by fitting the analytic profile's second moment directly
def second_moment(T, x):
    return np.trapezoid(T * x ** 2, x) / np.trapezoid(T, x)

var0 = second_moment(T0, x)
var_later = second_moment(T_later, x)
assert abs(var0 - sigma0 ** 2) < 1e-3
assert abs(var_later - (sigma0 ** 2 + 2 * alpha * 3.0)) < 1e-3

# 3. Fourier-space solver matches the analytic solution
t_final = 5.0
T_fourier = heq.solve_heat_fourier(T0, x, alpha, t_final)
T_analytic = heq.heat_kernel_gaussian(x, t_final, alpha, sigma0)
assert np.max(np.abs(T_fourier - T_analytic)) < 1e-4

# 4. independent finite-difference solver matches the analytic solution too
dt = 0.4 * heq.ftcs_stability_limit(dx, alpha)
n_steps = int(round(t_final / dt))
T_fd = heq.solve_heat_finite_difference(T0, dx, dt, alpha, n_steps)
assert np.max(np.abs(T_fd - T_analytic)) < 5e-3

# 5. and therefore Fourier and finite-difference agree with EACH OTHER too
#    (transitively, but verify directly rather than just trusting transitivity)
assert np.max(np.abs(T_fourier - T_fd)) < 5e-3

# 6. per-mode decay: exp(-alpha*k^2*t) is 1 at k=0 (DC/total-heat mode never
#    decays from diffusion alone) and strictly decreasing in |k|
assert abs(heq.fourier_mode_decay(0.0, alpha, t_final) - 1.0) < 1e-12
ks = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
decays = heq.fourier_mode_decay(ks, alpha, t_final)
assert np.all(np.diff(decays) < 0)   # strictly decreasing as k increases

# 7. FTCS stability: exceeding the limit must raise, staying under it must not
dt_unstable = 1.5 * heq.ftcs_stability_limit(dx, alpha)
try:
    heq.solve_heat_finite_difference(T0, dx, dt_unstable, alpha, 10)
    assert False, "should have rejected an unstable dt"
except ValueError:
    pass
dt_stable = 0.4 * heq.ftcs_stability_limit(dx, alpha)
heq.solve_heat_finite_difference(T0, dx, dt_stable, alpha, 10)  # should not raise

# 8. input validation
for bad_call in [
    lambda: heq.heat_kernel_gaussian(x, -1.0, alpha, sigma0),
    lambda: heq.heat_kernel_gaussian(x, 0.0, -1.0, sigma0),
    lambda: heq.heat_kernel_gaussian(x, 0.0, alpha, -1.0),
    lambda: heq.fourier_mode_decay(1.0, -1.0, 1.0),
    lambda: heq.solve_heat_fourier(T0, x, -1.0, 1.0),
    lambda: heq.ftcs_stability_limit(-1.0, alpha),
    lambda: heq.ftcs_stability_limit(dx, -1.0),
    lambda: heq.solve_heat_finite_difference(T0, -1.0, dt_stable, alpha, 10),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.heat_equation_fourier tests passed")
