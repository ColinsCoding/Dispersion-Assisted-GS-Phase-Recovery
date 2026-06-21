"""Test Fourier series: even->cosine, odd->sine, known coefficients, Gibbs, Parseval."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import fourier_series as fs

L = np.pi

# 1. ODD square wave sign(x): only sine terms; b_n = 4/(n pi) for odd n, 0 for even
a_sq, b_sq = fs.fourier_coefficients(np.sign, L, N=12)
assert np.max(np.abs(a_sq)) < 1e-6                       # no cosine terms (odd fn)
assert abs(b_sq[1] - 4/np.pi) < 1e-3                     # b_1 = 4/pi
assert abs(b_sq[3] - 4/(3*np.pi)) < 1e-3                 # b_3 = 4/(3 pi)
assert abs(b_sq[2]) < 1e-3 and abs(b_sq[4]) < 1e-3       # even harmonics vanish

# 2. EVEN triangle |x|: only cosine terms (all b_n ~ 0)
a_tr, b_tr = fs.fourier_coefficients(np.abs, L, N=12)
assert np.max(np.abs(b_tr)) < 1e-6                       # no sine terms (even fn)
assert abs(a_tr[0]/2 - np.pi/2) < 1e-2                   # mean of |x| on [-pi,pi] is pi/2
# |x| cosine coeffs: a_n = -4/(pi n^2) for odd n, 0 for even n
assert abs(a_tr[1] - (-4/np.pi)) < 1e-2
assert abs(a_tr[2]) < 1e-2                               # even harmonic ~ 0

# 3. reconstruction converges: more terms -> lower error AWAY from the jump
xs = np.linspace(-L + 0.3, L - 0.3, 400)                 # avoid the jump at 0, +-pi
xs = xs[np.abs(xs) > 0.3]
err = {}
for N in (5, 20, 80):
    a, b = fs.fourier_coefficients(np.sign, L, N=N)
    err[N] = np.max(np.abs(fs.reconstruct(xs, a, b, L) - np.sign(xs)))
assert err[80] < err[20] < err[5]                        # monotone improvement

# 4. Gibbs overshoot persists and does NOT vanish with more terms.
#    sign(x) jumps by 2, so the peak -> 1 + 0.0895*2 ~ 1.179 for any large N.
g20 = fs.gibbs_overshoot(*fs.fourier_coefficients(np.sign, L, N=20), L=L)
g100 = fs.gibbs_overshoot(*fs.fourier_coefficients(np.sign, L, N=100), L=L)
assert 1.16 < g20 < 1.20 and 1.16 < g100 < 1.20          # ~1.179 both -> does not shrink
assert abs(g20 - g100) < 0.02                            # value persists; only width shrinks

# 5. Parseval: a real periodic signal's energy = sum of harmonic energies.
#    (1/L) int_-L^L sin^2(x) dx = 1; the series of sin(x) is just b_1=1.
a_s, b_s = fs.fourier_coefficients(np.sin, L, N=8)
assert abs(b_s[1] - 1.0) < 1e-3 and np.max(np.abs(a_s)) < 1e-6
assert abs(np.sum(fs.harmonic_energy(a_s, b_s)) - 1.0) < 1e-3

print(f"TEST PASS  (odd square->sines b1={b_sq[1]:.3f}=4/pi; even |x|->cosines; "
      f"reconstruction converges {err[5]:.2f}->{err[80]:.2f}; Gibbs {g20:.3f}~{g100:.3f} "
      f"persists; Parseval sin energy=1)")
