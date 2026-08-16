"""Test dgs/dispersive_fourier_torch.py: the torch port of
dgs/dispersive_fourier.py's TS-DFT physics (checked to match the numpy
version to machine precision), and the gradient-based fiber-length design
capability (checked against the closed-form answer for a Gaussian pulse).
Requires py -3.12 (torch is py-3.12 only in this repo, not 3.13)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dgs.dispersive_fourier_torch import (
    gvd_transfer_function_torch, gvd_propagate_torch, gaussian_pulse_torch,
    rms_width_torch, achieved_stretch_factor, design_fiber_length_for_stretch_factor,
)
from dgs.dispersive_fourier import gvd_propagate, gaussian_pulse

N, dt, T0 = 1024, 1e-12, 2e-12
beta2, L = -20e-27, 5000.0

# 1. gvd_propagate_torch must match dgs.dispersive_fourier.gvd_propagate
#    (the numpy reference) to machine precision -- a faithful port
pulse_np = gaussian_pulse(N, T0, dt)
res_np = gvd_propagate(pulse_np, beta2=beta2, L_m=L, dt_s=dt)
res_torch = gvd_propagate_torch(pulse_np, beta2, L, dt)
max_err = float(np.max(np.abs(res_torch["E_out"].numpy() - res_np["E_out"])))
assert max_err < 1e-10, f"torch port should match numpy to machine precision, got {max_err:.2e}"

# 2. gvd_transfer_function_torch: |H|=1 everywhere (all-pass, same physics
#    dgs/gs_verify.py already checks for the H(nu) used throughout this repo)
omega = torch.linspace(-1e12, 1e12, 500, dtype=torch.float64)
H = gvd_transfer_function_torch(omega, beta2=-20e-27, L_m=1000.0)
assert torch.allclose(torch.abs(H), torch.ones_like(omega), atol=1e-10)

# 3. gaussian_pulse_torch must match dgs.dispersive_fourier.gaussian_pulse
pulse_torch = gaussian_pulse_torch(N, T0, dt)
max_err_pulse = float(np.max(np.abs(pulse_torch.numpy() - pulse_np)))
assert max_err_pulse < 1e-10

# 4. gvd_propagate_torch bounds: dt_s<=0 and n<8 must raise
try:
    gvd_propagate_torch(pulse_np, beta2, L, dt_s=0.0)
    raise AssertionError("expected ValueError for dt_s<=0")
except ValueError:
    pass
try:
    gvd_propagate_torch(np.array([1.0, 2.0]), beta2, L, dt_s=1e-12)
    raise AssertionError("expected ValueError for n<8")
except ValueError:
    pass

# 5. rms_width_torch: a symmetric Gaussian intensity centered at 0 has a
#    known analytic RMS width equal to its 1/e half-width / sqrt(2)... use
#    a simpler direct check: scaling t doesn't change the intensity shape,
#    so rms_width should scale linearly with t
t1 = torch.linspace(-10.0, 10.0, 2001, dtype=torch.float64)
I1 = torch.exp(-t1 ** 2 / 8.0)
w1 = rms_width_torch(I1, t1)
t2 = t1 * 2.0
w2 = rms_width_torch(I1, t2)  # same intensity SHAPE, doubled time axis
assert abs(float(w2) - 2 * float(w1)) < 1e-6, "RMS width should scale linearly with the time axis"

# 6. achieved_stretch_factor: at L_m=0 (no propagation), stretch factor
#    must be ~1 (output = input)
t_full = (torch.arange(N, dtype=torch.float64) - N // 2) * dt
E_in_t = torch.exp(-t_full ** 2 / (2 * T0 ** 2)).to(torch.complex128)
M_zero = float(achieved_stretch_factor(E_in_t, 0.0, beta2, dt, t_full))
assert abs(M_zero - 1.0) < 1e-6, f"stretch factor at L_m=0 should be 1.0, got {M_zero}"

# 7. design_fiber_length_for_stretch_factor: the gradient-based answer
#    must match the closed-form answer to within a tight tolerance -- the
#    actual point of this module (autograd finds the same answer a known
#    formula predicts, giving confidence in cases with no closed form)
result = design_fiber_length_for_stretch_factor(
    T0_s=T0, dt_s=dt, beta2=beta2, target_stretch_factor=20.0,
    n_pts=1024, n_iter=1500, lr=100.0)
assert result["relative_error"] < 0.01, (
    f"gradient-based L_m should match the closed-form answer to <1%, "
    f"got {result['relative_error']*100:.2f}%")
assert result["loss_history"][-1] < result["loss_history"][0], "loss should decrease over optimization"

# 8. design_fiber_length_for_stretch_factor bounds
for bad_kwargs in [
    dict(T0_s=0.0, dt_s=dt, beta2=beta2, target_stretch_factor=10.0),
    dict(T0_s=T0, dt_s=dt, beta2=beta2, target_stretch_factor=0.5),  # <=1 must raise
    dict(T0_s=T0, dt_s=dt, beta2=beta2, target_stretch_factor=10.0, n_iter=0),
]:
    try:
        design_fiber_length_for_stretch_factor(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

print("all dgs.dispersive_fourier_torch tests passed")
