"""Test the GPU-batched Euler rigid-body integrator: agrees with
dgs.gyroscopes' independent NumPy implementation, batch results don't
depend on batch size (no cross-contamination between trajectories), and
the stability-map grid correctly identifies stable poles vs. an unstable
band."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.gyroscopes import integrate_euler_rigid_body
from dgs.gyroscopes_torch import integrate_euler_rigid_body_batch, map_stability_over_sphere

I1, I2, I3 = 1.0, 2.0, 3.0

# 1. batched GPU integrator agrees with the independent NumPy implementation
omega0 = [1e-3, 5.0, 1e-3]
final_np = integrate_euler_rigid_body(omega0, I1, I2, I3, t_max=5.0, dt=0.002)["omega"][-1]
final_batch = integrate_euler_rigid_body_batch([omega0], I1, I2, I3, t_max=5.0, dt=0.002)[0]
assert np.max(np.abs(final_np - final_batch)) < 1e-2   # float32 GPU vs float64 CPU, over an unstable regime

# 2. a STABLE axis (axis 1) should agree far more tightly -- no exponential
# sensitivity to amplify float32-vs-float64 differences there
omega0_stable = [5.0, 1e-3, 1e-3]
final_np_stable = integrate_euler_rigid_body(omega0_stable, I1, I2, I3, t_max=5.0, dt=0.002)["omega"][-1]
final_batch_stable = integrate_euler_rigid_body_batch([omega0_stable], I1, I2, I3, t_max=5.0, dt=0.002)[0]
assert np.max(np.abs(final_np_stable - final_batch_stable)) < 1e-3

# 3. batch members don't interact: running N copies of the SAME initial
# condition in one batch call must give N identical results
same_ic_batch = np.tile(omega0_stable, (20, 1))
results = integrate_euler_rigid_body_batch(same_ic_batch, I1, I2, I3, t_max=3.0, dt=0.002)
assert np.allclose(results, results[0], atol=1e-6)

# 4. running two DIFFERENT initial conditions in one batch call gives the
# same result as running each alone -- proves no cross-contamination
mixed_batch = np.array([omega0_stable, [1e-3, 5.0, 1e-3]])
mixed_result = integrate_euler_rigid_body_batch(mixed_batch, I1, I2, I3, t_max=3.0, dt=0.002)
alone_stable = integrate_euler_rigid_body_batch([omega0_stable], I1, I2, I3, t_max=3.0, dt=0.002)[0]
alone_unstable = integrate_euler_rigid_body_batch([[1e-3, 5.0, 1e-3]], I1, I2, I3, t_max=3.0, dt=0.002)[0]
assert np.allclose(mixed_result[0], alone_stable, atol=1e-5)
assert np.allclose(mixed_result[1], alone_unstable, atol=1e-5)

# 5. stability map: the pole at axis 1 (theta~0, i.e. spin aligned with z)
# should show near-zero deviation; a point near the equator's axis-2
# direction should show large deviation
Theta, Phi, max_dev = map_stability_over_sphere(I1, I2, I3, n_theta=20, n_phi=40, t_max=10.0, dt=0.002)
assert max_dev.shape == Theta.shape == Phi.shape
# theta index 0 is the northernmost ring (closest to the pole, i.e. closest
# to spinning purely about axis 3 in this parametrization) -- should be stable
assert max_dev[0, :].max() < 0.5
# somewhere in the grid there must be a genuinely large deviation (the unstable band)
assert max_dev.max() > 2.0

print("all dgs.gyroscopes_torch tests passed")
