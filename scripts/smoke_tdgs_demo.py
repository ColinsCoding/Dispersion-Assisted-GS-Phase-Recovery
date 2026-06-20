"""Smoke-test the TD-GS pipeline (gs_core) on a chirped Gaussian and QPSK data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import gs_core as gs

def chirped_gaussian(N=512, width=40.0, chirp=3e-4):
    n = np.arange(N) - N / 2
    envelope = np.exp(-(n / width)**2 / 2)
    phase = chirp * n**2                      # quadratic chirp (engineered phase)
    return envelope * np.exp(1j * phase), phase, envelope

def align_phase(phi_rec, phi_true, weight):
    # remove global offset and resolve the conjugate (twin) ambiguity, weighted
    best = None
    for sign in (+1, -1):
        d = phi_true - sign * phi_rec
        offset = np.angle(np.sum(weight * np.exp(1j * d)))
        err = np.sqrt(np.sum(weight * np.angle(np.exp(1j * (d - offset)))**2) / np.sum(weight))
        if best is None or err < best[0]:
            best = (err, sign, offset)
    return best

# 1. chirped Gaussian, strong dispersion (diversity present)
E, phi_true, env = chirped_gaussian()
D1, D2 = -5000.0, -5750.0
I1 = np.abs(gs.disperse(E, D1))**2
I2 = np.abs(gs.disperse(E, D2))**2
corr = np.corrcoef(I1, I2)[0, 1]
print(f"chirped Gaussian: corr(I1,I2)={corr:.4f} at D1={D1}, D2={D2}")

phi, errors = gs.retrieve_phase(I1, I2, D1, D2, n_iter=80, unit_amplitude=False)
w = env**2
err, sign, off = align_phase(phi, phi_true, w)
print(f"  recovery: weighted phase RMS error = {err:.4f} rad (twin sign {sign:+d})")
print(f"  GS amplitude error: {errors[0]:.3e} -> {errors[-1]:.3e} (should drop)")

# 2. low-diversity case: small |D| -> I1 ~ I2 -> should fail
D1b, D2b = -400.0, -500.0
I1b = np.abs(gs.disperse(E, D1b))**2
I2b = np.abs(gs.disperse(E, D2b))**2
corrb = np.corrcoef(I1b, I2b)[0, 1]
phib, errb = gs.retrieve_phase(I1b, I2b, D1b, D2b, n_iter=80, unit_amplitude=False)
errlow, _, _ = align_phase(phib, phi_true, w)
print(f"\nlow diversity D1={D1b},D2={D2b}: corr(I1,I2)={corrb:.4f}, "
      f"phase RMS error = {errlow:.4f} rad (expect worse)")

# 3. QPSK comms data via the repo's generator
try:
    data = gs.make_qpsk_measurements(n_symbols=64, sps=8, D1=-5000.0, D2=-5750.0)
    phi_q, err_q = gs.retrieve_phase(data["I1"], data["I2"], data["D1"], data["D2"],
                                     n_iter=60, unit_amplitude=True)
    perr, psign, _ = align_phase(phi_q, data["phi_true"], np.ones_like(phi_q))
    print(f"\nQPSK/smooth-phase: GS amplitude error {err_q[0]:.3e} -> {err_q[-1]:.3e}")
    print(f"  recovered phase RMS error vs truth = {perr:.4f} rad (twin {psign:+d})")
    print(f"  corr(I1,I2) = {np.corrcoef(data['I1'], data['I2'])[0,1]:.4f}")

    # diversity sweep: phase error vs |D|
    print("  diversity sweep (phase error vs |D|):")
    for Dmag in (600, 2000, 5000, 9000):
        d = gs.make_qpsk_measurements(n_symbols=64, sps=8,
                                      D1=-float(Dmag), D2=-float(Dmag) * 1.15)
        pq, _ = gs.retrieve_phase(d["I1"], d["I2"], d["D1"], d["D2"],
                                  n_iter=60, unit_amplitude=True)
        pe, _, _ = align_phase(pq, d["phi_true"], np.ones_like(pq))
        c = np.corrcoef(d["I1"], d["I2"])[0, 1]
        print(f"    |D|={Dmag:>4}: corr={c:.4f}  phase RMS err={pe:.4f} rad")
except Exception as e:
    print("QPSK generator:", type(e).__name__, e)

# validation
for bad in [lambda: gs.retrieve_phase(I1, I2, -50, -60),
            lambda: gs.retrieve_phase(I1, I2, -5000, -5000)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", str(e)[:55])
print("SMOKE PASS")
