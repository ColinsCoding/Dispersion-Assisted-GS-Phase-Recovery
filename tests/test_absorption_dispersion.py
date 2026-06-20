"""Smoke-test the Lorentz absorption/dispersion model + low-light shot noise."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import electrodynamics as ed
from dgs import dispersion_gs_prototype as dg

# Lorentz medium: resonance at w0=1, weak damping
w = np.linspace(0.2, 2.0, 4000)
w0, gamma, wp = 1.0, 0.08, 0.4
eps = ed.lorentz_epsilon(w, w0, gamma, wp)
n_tilde = ed.complex_index(eps)
n, kappa = n_tilde.real, n_tilde.imag

# 1. absorption (kappa) peaks at the resonance and is non-negative
assert np.all(kappa >= -1e-12)
assert abs(w[np.argmax(kappa)] - w0) < 0.02, "absorption must peak at resonance"

# 2. dispersion: normal (dn/dw>0) far from resonance, anomalous (dn/dw<0) across it
anom = ed.is_anomalous(w, n)
assert anom[np.argmin(np.abs(w - w0))], "should be anomalous AT resonance"
assert not anom[np.argmin(np.abs(w - 0.4))], "should be normal well below resonance"
# the anomalous band sits inside the absorption line
assert abs(w[anom].mean() - w0) < 0.1

# 3. eps imaginary part is the absorptive part and is positive (passive medium)
assert np.all(eps.imag >= -1e-12)

# 4. Beer-Lambert: intensity decays, halves at z = ln2/alpha
alpha = ed.absorption_coefficient(w0, kappa[np.argmin(np.abs(w - w0))], c=1.0)
assert alpha > 0
I_half = ed.beer_lambert(1.0, alpha, np.log(2) / alpha)
assert abs(I_half - 0.5) < 1e-9

# 5. Kramers-Kronig: absorption (Im chi) reconstructs dispersion (Re chi) shape
chi = eps - 1.0
re_from_im = ed.kramers_kronig(chi.imag)
core = slice(len(w) // 5, 4 * len(w) // 5)            # interior, avoid FFT edges
corr = np.corrcoef(re_from_im[core], chi.real[core])[0, 1]
assert abs(corr) > 0.95, f"KK reconstruction corr too low: {corr:.3f}"

# 6. low-light Poisson shot noise: mean preserved, variance ~ 1/photons
rng = np.random.default_rng(0)
t, x, A, phi = dg.make_field(N=4096, seed=1)
I = np.abs(x) ** 2
hi = dg.photon_shot_noise(I, mean_photons=1e5, rng=rng)
lo = dg.photon_shot_noise(I, mean_photons=1e2, rng=rng)
assert abs(hi.mean() - I.mean()) / I.mean() < 0.02     # unbiased
# fewer photons -> noisier (relative residual grows)
res_hi = np.std(hi - I) / I.mean()
res_lo = np.std(lo - I) / I.mean()
assert res_lo > 5 * res_hi, (res_lo, res_hi)
try:
    dg.photon_shot_noise(I, 0, rng)
except ValueError:
    pass
else:
    raise AssertionError("should reject mean_photons <= 0")

print(f"SMOKE PASS  (KK corr={corr:.3f}, anomalous band ~[{w[anom].min():.2f},{w[anom].max():.2f}])")
