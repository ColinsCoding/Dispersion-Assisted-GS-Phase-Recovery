"""Test that MuJoCo (an independent, real physics engine) reproduces
dgs.gyroscopes' own results for both the gravity-driven precessing top and
the torque-free asymmetric top (tennis racket theorem) -- a genuine
third-party cross-check, not just self-consistency of our own code."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.mujoco_gyroscope import (
    simulate_precessing_top, simulate_free_rigid_body, build_precessing_top_model,
    measured_nutation_frequency,
)
from dgs.gyroscopes import precession_rate, integrate_euler_rigid_body, nutation_frequency

# 1. Explicit <inertial> actually pins the disk's spin axis to local z
# (the bug this module's docstring warns about, verified NOT present)
model, I_spin, I_transverse = build_precessing_top_model(0.2, 0.1, 0.3)
assert model.body_iquat[1].tolist() == [1.0, 0.0, 0.0, 0.0]
assert abs(model.body_inertia[1][2] - I_spin) < 1e-9         # local z = spin axis
assert abs(model.body_inertia[1][0] - I_transverse) < 1e-9
assert abs(model.body_inertia[1][1] - I_transverse) < 1e-9

# 2. Precessing top: as omega_spin grows (deeper into the fast-top
# approximation), MuJoCo's measured mean precession rate converges toward
# dgs.gyroscopes.precession_rate's analytic prediction
m_disk, R_disk, r = 0.2, 0.1, 0.3
ratios = []
for omega_spin in (300.0, 1000.0, 3000.0):
    run = simulate_precessing_top(m_disk, R_disk, r, omega_spin)
    analytic = precession_rate(mass=m_disk, g=9.80665, r=r, I_spin=run["I_spin"], omega_spin=omega_spin)
    ratio = run["mean_precession_rate"] / analytic["Omega_p_rad_s"]
    ratios.append(abs(ratio - 1.0))
    # nutation stays small (bounded release-from-rest wobble, not a fall-over)
    assert run["theta"].max() - run["theta"].min() < 0.5

assert ratios[-1] < ratios[0]     # agreement improves as the fast-top approximation gets better
assert ratios[-1] < 0.06          # within 6% at omega_spin=3000

# 3. Torque-free asymmetric top: MuJoCo's own physics engine reproduces
# the exact same stable/unstable classification as dgs.gyroscopes
I1, I2, I3 = 1.0, 2.0, 3.0
for axis_idx, omega0, should_flip in [
    (0, [5.0, 1e-3, 1e-3], False),
    (1, [1e-3, 5.0, 1e-3], True),
    (2, [1e-3, 1e-3, 5.0], False),
]:
    run = simulate_free_rigid_body(omega0, I1, I2, I3, t_max=20.0, dt=0.001)
    transverse = np.delete(run["omega"], axis_idx, axis=1)
    max_transverse = float(np.max(np.abs(transverse)))
    if should_flip:
        assert max_transverse > 2.5
    else:
        assert max_transverse < 0.05

# 4. MuJoCo's free-body result agrees with dgs.gyroscopes' own independent
# NumPy RK4 integrator on the unstable case, not just the same qualitative verdict
ref = integrate_euler_rigid_body([1e-3, 5.0, 1e-3], I1, I2, I3, t_max=5.0, dt=0.001)
mujoco_run = simulate_free_rigid_body([1e-3, 5.0, 1e-3], I1, I2, I3, t_max=5.0, dt=0.001)
assert np.max(np.abs(ref["omega"][-1] - mujoco_run["omega"][-1])) < 1e-2

# 5. Nutation frequency: MuJoCo's own theta(t) wobble, measured via FFT,
# agrees with dgs.gyroscopes.nutation_frequency -- but ONLY when fed
# I_transverse about the PIVOT (I_transverse_pivot), not about the disk's
# own center of mass (I_transverse). This is a real, previously-latent
# footgun in nutation_frequency's calling convention: using the
# center-of-mass value gives ratios of ~0.02-0.03 (wrong by ~40x) because
# the parallel-axis term m*r**2 (0.2*0.3**2=0.018) dwarfs the disk's own
# I_transverse about its center (0.0005) for this geometry.
wrong_ratios, pivot_ratios = [], []
for omega_spin in (300.0, 1000.0, 3000.0):
    run = simulate_precessing_top(m_disk, R_disk, r, omega_spin)
    omega_n_measured = measured_nutation_frequency(run["t"], run["theta"])

    omega_n_wrong = nutation_frequency(run["I_spin"], run["I_transverse"], omega_spin)
    wrong_ratios.append(omega_n_measured / omega_n_wrong)

    omega_n_pivot = nutation_frequency(run["I_spin"], run["I_transverse_pivot"], omega_spin)
    pivot_ratios.append(omega_n_measured / omega_n_pivot)

assert all(r < 0.05 for r in wrong_ratios)          # confirms the footgun is real, not assumed
assert abs(pivot_ratios[-1] - 1.0) < 0.05            # pivot-corrected: within 5% at omega_spin=3000
assert abs(pivot_ratios[-1] - 1.0) < abs(pivot_ratios[0] - 1.0)   # improves with spin, like precession

print("all dgs.mujoco_gyroscope tests passed")
