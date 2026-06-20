"""Smoke-test quantum_statistics: occupation laws, Planck, photon noise."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import quantum_statistics as qs

# 1. Fermi-Dirac: always in [0,1], exactly 1/2 at E=mu (Pauli)
E = np.linspace(-5, 5, 200)
fd = qs.fermi_dirac(E, mu=0.0, T=1.0)
assert fd.min() >= 0 and fd.max() <= 1
assert abs(qs.fermi_dirac(0.0, 0.0, 1.0) - 0.5) < 1e-12

# 2. Bose-Einstein diverges toward E->mu+, and rejects E<=mu
assert qs.bose_einstein(0.01, 0.0, 1.0) > qs.bose_einstein(2.0, 0.0, 1.0)
try:
    qs.bose_einstein(0.0, 0.0, 1.0)
except ValueError:
    pass
else:
    raise AssertionError("BE should reject E <= mu")

# 3. classical limit: far above mu, BE ~ FD ~ MB
Ehi = 12.0
mb = qs.maxwell_boltzmann(Ehi, 0.0, 1.0)
assert abs(qs.bose_einstein(Ehi, 0, 1) - mb) / mb < 1e-3
assert abs(qs.fermi_dirac(Ehi, 0, 1) - mb) / mb < 1e-3

# 4. Planck: reduces to Rayleigh-Jeans at low frequency, Wien peak is physical
nu_lo = 1e9
assert abs(qs.planck_spectral_radiance(nu_lo, 300) /
           qs.rayleigh_jeans(nu_lo, 300) - 1) < 1e-3
# Sun ~5772 K: B(nu) frequency-peak -> c/nu_max ~883 nm (near-IR);
# B(lambda) wavelength-peak ~502 nm (visible). The two differ by the Jacobian.
lam_from_nu = qs.C / qs.wien_peak_frequency(5772.0)
assert 850e-9 < lam_from_nu < 920e-9, lam_from_nu
lam_peak = qs.wien_peak_wavelength(5772.0)
assert 480e-9 < lam_peak < 520e-9, lam_peak
# numeric peak of B(nu) matches wien_peak_frequency
nu = np.linspace(1e13, 1e15, 20000)
assert abs(nu[np.argmax(qs.planck_spectral_radiance(nu, 5772.0))] /
           qs.wien_peak_frequency(5772.0) - 1) < 0.02

# 5. Stefan-Boltzmann: integrated radiance scales as T^4
def total(T):
    nu = np.linspace(1e11, 5e15, 200000)
    return np.trapezoid(qs.planck_spectral_radiance(nu, T), nu)
assert abs(total(600) / total(300) - 16.0) / 16.0 < 0.02   # (2)^4 = 16

# 6. photon-number distributions normalize; means correct
n = np.arange(0, 200)
nbar = 6.0
pc, pt = qs.poisson_pmf(n, nbar), qs.thermal_pmf(n, nbar)
assert abs(pc.sum() - 1) < 1e-6 and abs(pt.sum() - 1) < 1e-6
assert abs((pc * n).sum() - nbar) < 1e-3
assert abs((pt * n).sum() - nbar) < 1e-3

# 7. THE POINT: coherent is Poissonian (Q=0); thermal is super-Poissonian (Q=nbar)
var_c = (pc * n**2).sum() - nbar**2
var_t = (pt * n**2).sum() - nbar**2
assert abs(qs.mandel_q(var_c, nbar) - 0.0) < 1e-2          # coherent: shot-noise limited
assert abs(qs.mandel_q(var_t, nbar) - nbar) < 0.1          # thermal: var = nbar + nbar^2
assert var_t > var_c                                       # thermal light is noisier

print(f"SMOKE PASS  (Wien lambda-peak={lam_peak*1e9:.0f} nm; "
      f"coherent Q={qs.mandel_q(var_c, nbar):.2f}, thermal Q={qs.mandel_q(var_t, nbar):.2f})")
