"""Sweep D and n_iter for the dispersion-GS prototype to find a recovering regime."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import dispersion_gs_prototype as dg

t, x, A, phi = dg.make_field(N=2048, seed=0)

print("D sweep (n_iter=600), recovered phase RMS (rad), weighted by A^2:")
for D in (1500, 3000, 6000, 9000, 14000, 20000):
    I1 = np.abs(x)**2
    I2 = np.abs(dg.disperse(x, D))**2
    corr = np.corrcoef(I1, I2)[0, 1]
    xr, errs = dg.gerchberg_saxton(I1, I2, D, n_iter=600, seed=0)
    rms, _ = dg.compare_phase(np.angle(xr), phi, A**2)
    print(f"  D={D:>6}: corr(I1,I2)={corr:.3f}  amp_err {errs[-1]:.3f}  phase RMS {rms:.3f}")

# test what helps: moderate chirp, and constant-envelope
import numpy as np
def field_variant(chirp_coef, const_amp, N=2048, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, N)
    A = np.ones(N) if const_amp else (np.exp(-(t+0.05)**2/(2*0.18**2))
                                      + 0.35*np.exp(-(t-0.45)**2/(2*0.07**2)))
    phi = chirp_coef*t**2 + 0.8*np.sin(2*np.pi*3*t+0.6)
    tex = sum(rng.uniform(-1,1)*np.sin(2*np.pi*k*t+rng.uniform(0,2*np.pi)) for k in (5,7,11))
    phi = phi + 0.15*tex
    return t, A*np.exp(1j*phi), A, phi

print("\nvariant sweep at D=6000, n_iter=600 (chirp, const-envelope):")
for chirp in (3, 8, 18):
    for const in (False, True):
        t, x, A, phi = field_variant(chirp, const)
        I1 = np.abs(x)**2; I2 = np.abs(dg.disperse(x, 6000))**2
        xr, errs = dg.gerchberg_saxton(I1, I2, 6000, n_iter=600, seed=0)
        rms, _ = dg.compare_phase(np.angle(xr), phi, A**2)
        print(f"  chirp={chirp:>2} const_amp={const!s:5}: phase RMS {rms:.3f}  amp_err {errs[-1]:.3f}")

# best-effort with multiple random restarts at a good D
print("\nrandom restarts at D=9000, n_iter=800:")
D = 9000
I1 = np.abs(x)**2; I2 = np.abs(dg.disperse(x, D))**2
best = None
for seed in range(6):
    xr, errs = dg.gerchberg_saxton(I1, I2, D, n_iter=800, seed=seed)
    rms, _ = dg.compare_phase(np.angle(xr), phi, A**2)
    if best is None or rms < best[0]:
        best = (rms, seed, errs[-1])
    print(f"  seed {seed}: phase RMS {rms:.3f}  amp_err {errs[-1]:.3f}")
print("best:", best)
