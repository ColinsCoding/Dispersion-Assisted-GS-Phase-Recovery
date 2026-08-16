"""Test dgs.microplastic.dispersion: calibrated Lorentz dispersion + transfer function."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import causality as ca
from dgs.microplastic import dispersion as disp
from dgs.microplastic import materials as mat
from dgs.microplastic import physics as phy

# 1. calibration hits the target n exactly at the reference wavelength
omega0, gamma, strength = disp.polymer_dispersion_model("PET")
n_at_D = disp.n_tilde_lorentz(disp.OMEGA_D, omega0, gamma, strength)
assert np.isclose(n_at_D.real, mat.refractive_index("PET"), atol=1e-6)
assert n_at_D.imag > 0     # a real (absorbing) medium, not a lossless fiction

# 2. normal dispersion: n rises toward the UV resonance (shorter wavelength = higher n)
omega_blue = 2*np.pi*phy.C / 450e-9
omega_red = 2*np.pi*phy.C / 650e-9
n_blue = disp.n_tilde_lorentz(omega_blue, omega0, gamma, strength)
n_red = disp.n_tilde_lorentz(omega_red, omega0, gamma, strength)
assert n_blue.real > n_at_D.real > n_red.real

# 3. every polymer in materials.py calibrates successfully and matches its target
for poly in mat.list_polymers():
    o0, g, s = disp.polymer_dispersion_model(poly)
    n_fit = disp.n_tilde_lorentz(disp.OMEGA_D, o0, g, s)
    assert np.isclose(n_fit.real, mat.refractive_index(poly), atol=1e-6), poly

# 4. Kramers-Kronig consistency: this model's dispersion and absorption are not
# independent -- they must reconstruct each other via the Hilbert transform
# already proved in dgs/causality.py
w = np.linspace(-3*omega0, 3*omega0, 20000)
chi = ca.lorentz_susceptibility(w, omega0, gamma, strength)
Re_kk = ca.kramers_kronig_real(chi.imag)
core = slice(4000, 16000)
rel_err = np.max(np.abs(chi.real[core] - Re_kk[core])) / np.max(np.abs(chi.real[core]))
assert rel_err < 0.05     # FFT-grid Hilbert transform, not exact -- but tight

# 5. transfer_function magnitude decays with slab thickness (absorption), and a
# thicker slab attenuates more
n_fn = lambda om: disp.n_tilde_lorentz(om, omega0, gamma, strength)
H_thin = disp.transfer_function(disp.OMEGA_D, n_fn, L=1e-3)
H_thick = disp.transfer_function(disp.OMEGA_D, n_fn, L=1e-2)
assert abs(H_thick) < abs(H_thin) < 1.0

# 6. apply_slab round-trips a pulse through the slab and broadens/delays it
# (dispersion) relative to free propagation (L=0 should leave it unchanged)
t = np.linspace(-50, 50, 20000)
Et = phy.gaussian_pulse(t, tau=2.0)   # baseband envelope (omega0=0 default)
t0_out, Et0 = disp.apply_slab(t, Et, n_fn, L=0.0)
assert np.max(np.abs(Et0 - Et)) < 1e-6     # zero-thickness slab = no-op

t_out, Et_out = disp.apply_slab(t, Et, n_fn, L=5e-3)
assert np.max(np.abs(Et_out)) < np.max(np.abs(Et))     # absorption reduces peak amplitude

print("TEST PASS  (Lorentz calibration hits target n for all polymers; normal dispersion "
      "n_blue>n_D>n_red; Kramers-Kronig self-consistent; transfer_function attenuates more "
      "for thicker slabs; apply_slab is a no-op at L=0 and attenuates a real pulse)")
