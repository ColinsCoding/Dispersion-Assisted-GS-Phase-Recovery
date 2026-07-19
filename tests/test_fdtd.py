"""Test 1D FDTD (Yee grid): vacuum propagation speed, Fresnel reflection at a single
interface, and the slab transmission spectrum against the analytic Fabry-Perot (Airy)
formula -- the same physics as dgs.photonic_circuits' ring resonator, straightened out."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import fdtd

dx = 2e-8  # 20 nm cells

# 1. Courant limit: dt = S*dx/c, and S must be in (0,1]
assert abs(fdtd.courant_dt(dx, 0.5) - 0.5 * dx / fdtd.C) < 1e-30
try:
    fdtd.courant_dt(dx, 1.5)
    assert False, "should reject S > 1"
except ValueError:
    pass
try:
    fdtd.courant_dt(-1.0)
    assert False, "should reject non-positive dx"
except ValueError:
    pass

# 2. a pulse launched in vacuum arrives at a probe after distance/c (within one Courant step)
Nx = 400
source_index, probe_index = 20, 380
pulse_spread = 5e-15
t0 = 8 * pulse_spread
S = 0.99
dt = fdtd.courant_dt(dx, S)
n_steps = 1100  # comfortably past the expected arrival step (~969) so the peak isn't clipped


def src(t):
    return fdtd.gaussian_pulse(t, t0, pulse_spread)


eps_vac = np.ones(Nx)
out = fdtd.run_fdtd_1d(eps_vac, dx, n_steps, source_index, src, probe_index, S=S)
arrival_step = np.argmax(np.abs(out["Ez_probe"]))
t_arrival = arrival_step * dt
t_expected = t0 + (probe_index - source_index) * dx / fdtd.C
assert abs(t_arrival - t_expected) < 5 * dt, (t_arrival, t_expected)

# 3. single dielectric interface: normal-incidence power reflectivity matches Fresnel
n_slab = 1.5
R_formula = ((n_slab - 1) / (n_slab + 1)) ** 2
assert abs(fdtd.fabry_perot_power_reflectivity(n_slab) - R_formula) < 1e-12
assert fdtd.fabry_perot_power_reflectivity(1.0) < 1e-12   # matched index -> no reflection

# 4. slab_eps_profile places n^2 in the slab region and n_bg^2 elsewhere
eps = fdtd.slab_eps_profile(Nx=10, slab_start=3, slab_cells=4, n_slab=2.0, n_bg=1.0)
assert np.all(eps[:3] == 1.0) and np.all(eps[3:7] == 4.0) and np.all(eps[7:] == 1.0)
try:
    fdtd.slab_eps_profile(Nx=10, slab_start=8, slab_cells=5, n_slab=2.0)
    assert False, "should reject a slab region that overflows the grid"
except ValueError:
    pass

# 5. FDTD-computed slab transmission spectrum matches the analytic Fabry-Perot (Airy) formula
freqs = np.linspace(150e12, 250e12, 9)
T_fdtd, T_analytic = fdtd.slab_transmission_fdtd(n_slab=2.0, L=2e-6, freqs=freqs, dx=dx)
assert np.all(T_fdtd <= 1.0 + 1e-6) and np.all(T_fdtd >= 0.0)       # energy conservation
assert np.max(np.abs(T_fdtd - T_analytic)) < 0.02, (T_fdtd, T_analytic)

# 6. a slab with n_slab == n_bg is invisible: T ~ 1 at every frequency (no interfaces)
T_matched, _ = fdtd.slab_transmission_fdtd(n_slab=1.0, L=2e-6, freqs=freqs, dx=dx)
assert np.all(np.abs(T_matched - 1.0) < 0.02), T_matched

# 7. eps_r below vacuum is unphysical for this module and must be rejected
try:
    fdtd.run_fdtd_1d(np.full(10, 0.5), dx, 10, 5, src, 5)
    assert False, "should reject eps_r < 1"
except ValueError:
    pass

print("all dgs.fdtd tests passed")
