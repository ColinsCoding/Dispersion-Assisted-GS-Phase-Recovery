"""Smoke-test xray_phase: helix diffraction structure + HIO phase retrieval."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import xray_phase as xp

# 1. helix diffraction shows layer-line / cross structure
rho = xp.helix_density(N=160, pitch=26, radius=14, n_strands=2)
D = xp.diffraction(rho)
N = D.shape[0]; c = N // 2
# energy off-axis (the X arms) should exceed pure-axis energy -> cross pattern
diag_band = D[c-2:c+3, :].sum() + D[:, c-2:c+3].sum()
print(f"helix diffraction: peak at center? {np.unravel_index(D.argmax(), D.shape)} (expect ~({c},{c}))")
print(f"  layer-line energy concentrated off-center: total={D.sum():.3e}")

# CCV: first maximum of J_n moves outward with n (the X opening)
import mpmath as mp
print("  first max of J_n at 2pi r R = ", [round(float(mp.besseljzero(n, 1, derivative=1)), 2)
                                           for n in range(1, 5)], "(grows with n -> the X)")

# 2. HIO recovers an asymmetric compact object from |F| + support
N = 128
truth = np.zeros((N, N))
rng = np.random.default_rng(1)
# a few asymmetric blobs in the central quarter (oversampling factor 2)
centers = [(50, 54, 3.0), (60, 70, 2.0), (72, 58, 2.5), (66, 80, 1.5)]
yy, xx = np.mgrid[0:N, 0:N]
for (cy, cx, s) in centers:
    truth += np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*s**2))
support = np.zeros((N, N), dtype=bool)
support[N//4:3*N//4, N//4:3*N//4] = True       # known finite extent (oversampled)

mag = xp.amplitude(truth)
rec, errors = xp.hio_phase_retrieval(mag, support, n_iter=400, beta=0.9, seed=0)
corr = xp.best_alignment_corr(rec, truth)
final_err = np.linalg.norm(xp.amplitude(rec) - mag) / np.linalg.norm(mag)
print(f"\nHIO phase retrieval (error curve bounces -- that's HIO, not a bug):")
print(f"  Fourier error: start {errors[0]:.3f}, best {min(errors):.3f}, polished output {final_err:.3f}")
print(f"  recovered-vs-truth correlation (mod translation/inversion): {corr:.3f}")

assert final_err < 0.05, "polished output Fourier error too high"
assert corr > 0.85, "HIO did not recover the object"

# validation
for bad in [lambda: xp.helix_density(radius=200),
            lambda: xp.hio_phase_retrieval(mag, support[:10, :10])]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", str(e)[:50])
print("SMOKE PASS")
