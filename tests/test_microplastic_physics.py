"""Test dgs.microplastic.physics: complex index, propagation, Fourier integrals."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.microplastic import physics as mp

# 1. complex index / Maxwell permittivity relation
n_tilde = mp.complex_index(1.33, 0.0)
assert np.isclose(mp.permittivity(n_tilde).real, 1.33 ** 2)
assert np.isclose(mp.permittivity(n_tilde).imag, 0.0)

n_tilde_lossy = mp.complex_index(1.33, 0.02)
eps = mp.permittivity(n_tilde_lossy)
assert eps.imag > 0          # absorptive medium: Im(eps) > 0 in the e^{-i*omega*t} convention

# 2. wave number reduces to the real-index case when kappa=0
omega = 2 * np.pi * 3e14   # ~1000 nm optical carrier
k = mp.wave_number(n_tilde, omega)
assert np.isclose(k.real, 1.33 * omega / mp.C)
assert np.isclose(k.imag, 0.0)

# 3. Beer-Lambert falls out of propagate_field's complex exponential
kappa = 0.01
alpha = mp.absorption_coefficient(omega, kappa)
k_lossy = mp.wave_number(mp.complex_index(1.33, kappa), omega)
z = np.linspace(0, 5e-3, 50)
E = mp.propagate_field(1.0, k_lossy, z)
intensity = np.abs(E) ** 2
assert np.allclose(intensity, mp.beer_lambert_transmittance(alpha, z), atol=1e-10)

# 4. time-averaged Poynting recovers the standard vacuum intensity formula at n=1
E0 = 1e3  # V/m
S_vac = mp.time_averaged_poynting(E0, n=1.0)
assert np.isclose(S_vac, 0.5 * mp.EPS0 * mp.C * E0 ** 2)
assert mp.time_averaged_poynting(E0, n=1.33) > S_vac   # denser medium carries more power for same E0

# 5. Fourier transform of a Gaussian pulse matches the closed-form analytic result
t = np.linspace(-50, 50, 20000)
tau = 2.0
Et = mp.gaussian_pulse(t, tau=tau, omega0=0.0)
omega_ax, Ef_num = mp.fourier_transform(t, Et)
Ef_analytic = mp.gaussian_pulse_ft_analytic(omega_ax, tau=tau, omega0=0.0, t0=0.0)
core = np.abs(omega_ax) < 3 / tau   # compare where the Gaussian has real support
rel_err = np.max(np.abs(Ef_num[core] - Ef_analytic[core])) / np.max(np.abs(Ef_analytic[core]))
assert rel_err < 1e-3

# 6. Parseval / energy conservation between time and frequency domain
t_energy, f_energy, p_err = mp.parseval_check(t, Et, omega_ax, Ef_num)
assert p_err < 1e-6

# 7. forward/inverse Fourier transform round-trips a Gaussian pulse
t2, Et_back = mp.inverse_fourier_transform(omega_ax, Ef_num, t0=t[0])
assert np.allclose(t2, t, atol=1e-6)
assert np.max(np.abs(Et_back - Et)) < 1e-8

# 8. photon rate is just counts / time
assert np.isclose(mp.photon_rate(1e9, 10), 1e8)
try:
    mp.photon_rate(1e9, 0)
    assert False, "should reject duration<=0"
except ValueError:
    pass

print("TEST PASS  (Maxwell eps_r=n~^2; Beer-Lambert from propagate_field; Poynting "
      "matches vacuum formula; Gaussian FT matches analytic to <0.1%; Parseval holds; "
      "FFT/IFFT round-trips; photon_rate correct)")
