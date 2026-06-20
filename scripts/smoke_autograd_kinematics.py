"""Smoke-test autograd time derivatives vs analytic, and the noise-amplification point."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import autograd_kinematics as ak

# 1. autograd v, a, jerk of sin(2t) match the exact analytic forms
t = np.linspace(0, 2 * np.pi, 200)
x, v, a, j = ak.time_derivatives(ak.traj_sin, t, order=3)
assert np.allclose(v, 2 * np.cos(2 * t), atol=1e-9)
assert np.allclose(a, -4 * np.sin(2 * t), atol=1e-9)
assert np.allclose(j, -8 * np.cos(2 * t), atol=1e-9)        # exact to machine precision

# 2. autograd of a polynomial trajectory: d/dt[sin(2pi t)+0.5 t^2]
x2, v2 = ak.time_derivatives(ak.traj_mixed, t, order=1)
assert np.allclose(v2, 2 * np.pi * np.cos(2 * np.pi * t) + t, atol=1e-9)

# 3. SymPy ground truth lines up with the analytic forms used above
import sympy as sp
ts = sp.Symbol("t", real=True)
ders = ak.analytic_derivatives(sp.sin(2 * ts), ts, order=3)
assert sp.simplify(ders[1] - 2 * sp.cos(2 * ts)) == 0
assert sp.simplify(ders[3] + 8 * sp.cos(2 * ts)) == 0

# 4. finite difference on CLEAN samples recovers the 2nd derivative (loosely)
dt = t[1] - t[0]
a_fd_clean = ak.finite_difference(np.sin(2 * t), dt, order=2)
core = slice(2, -2)                                          # ignore edge effects
assert np.corrcoef(a_fd_clean[core], (-4 * np.sin(2 * t))[core])[0, 1] > 0.99

# 5. THE POINT: finite-differencing NOISY samples wrecks higher derivatives,
#    while autograd-on-the-function stays exact ("wrong trajectory" from noise)
rng = np.random.default_rng(0)
noisy = np.sin(2 * t) + 1e-3 * rng.standard_normal(t.size)   # 0.1% noise
a_fd_noisy = ak.finite_difference(noisy, dt, order=2)
err_clean = np.std((a_fd_clean - (-4 * np.sin(2 * t)))[core])
err_noisy = np.std((a_fd_noisy - (-4 * np.sin(2 * t)))[core])
assert err_noisy > 20 * err_clean, (err_noisy, err_clean)    # tiny input noise -> huge deriv error
assert np.std((a - (-4 * np.sin(2 * t)))[core]) < 1e-9       # autograd unaffected

# 6. validation
try:
    ak.time_derivatives(ak.traj_sin, t, order=0)
except ValueError:
    pass
else:
    raise AssertionError("order < 1 should raise")

print(f"SMOKE PASS  (autograd jerk exact to 1e-9; FD 2nd-deriv noise blowup "
      f"{err_noisy/err_clean:.0f}x from 0.1% input noise)")
