"""Build notebooks/why_two_intensities.ipynb -- one intensity is ambiguous, two recover phase."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# Why the receiver needs TWO intensity measurements
### one intensity -> infinitely many phases; two dispersed intensities -> the unique phase

A detector measures only $|E|^2$ -- it throws the phase away (the Born rule, the same
phase loss that makes a diffraction pattern $|\\mathcal F(\\text{aperture})|^2$). So a single
intensity is consistent with **infinitely many** fields. The dispersion-assisted
Gerchberg-Saxton receiver breaks the ambiguity by measuring the intensity at **two
different dispersions**, $I_1=|\\text{disperse}(E,D_1)|^2$ and $I_2=|\\text{disperse}(E,D_2)|^2$:
two different views of the same field. With enough **diversity** (the two dispersions
sufficiently different), GS bounces between $I_1$ and $I_2$ and converges to the one
phase that fits both. Uses `dgs/gs_core.py`. Civilian education."""),

co("""import numpy as np, matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
from dgs import gs_core as gs
def rms_err(phi_true, phi_est):                       # phase RMS, removing the global offset
    off = np.angle(np.mean(np.exp(1j*(phi_true-phi_est))))
    return np.degrees(np.sqrt(np.mean(np.angle(np.exp(1j*(phi_est+off-phi_true)))**2)))
print("ready")"""),

md("""## 1. One intensity -> many phases (the ambiguity)

Take an amplitude profile and attach three completely different phases. The intensity
$|E|^2$ is **identical** in all three cases -- a detector cannot tell them apart. One
measurement does not determine the phase; that information is simply not in $|E|^2$."""),
co("""x = np.linspace(-4, 4, 400)
amp = np.exp(-x**2/4)                                  # one amplitude profile
phases = {"phi = 0 (flat)": np.zeros_like(x),
          "phi = x (linear)": x,
          "phi = 2 sin(2x)": 2*np.sin(2*x)}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for lbl, ph in phases.items():
    E = amp*np.exp(1j*ph)
    ax[0].plot(x, np.abs(E)**2, lw=2, alpha=0.6)       # all identical
    ax[1].plot(x, ph, label=lbl)
ax[0].set(xlabel="x", ylabel="|E|^2", title="intensity: IDENTICAL for all three")
ax[1].set(xlabel="x", ylabel="phase (rad)", title="phase: completely different (hidden from the detector)"); ax[1].legend()
plt.tight_layout(); plt.show()
print("one intensity is consistent with infinitely many phases -- the Born-rule ambiguity")"""),

md("""## 2. Two dispersed intensities -- two different views

Now disperse the (same) field two ways. Because dispersion mixes phase into amplitude,
$I_1$ and $I_2$ look **different** -- each carries phase information the bare intensity
lost. These two arrays are the receiver's actual measurements."""),
co("""data = gs.make_qpsk_measurements(n_symbols=128, snr_db=35.0, D1=-5000.0, D2=-5750.0, rng_seed=3)
I1, I2, D1, D2, phi_true = data['I1'], data['I2'], data['D1'], data['D2'], data['phi_true']
plt.figure(figsize=(7,3.4))
plt.plot(I1[:160], label=f"I1 (dispersion D1={D1:.0f})")
plt.plot(I2[:160], label=f"I2 (dispersion D2={D2:.0f})")
plt.xlabel("sample"); plt.ylabel("intensity"); plt.legend()
plt.title("two intensity measurements at two dispersions (different views, low correlation)")
plt.tight_layout(); plt.show()
print(f"correlation(I1, I2) = {np.corrcoef(I1,I2)[0,1]:.3f}  (well-separated -> diverse -> informative)")"""),

md("""## 3. Two intensities -> the unique phase (Gerchberg-Saxton)

Feed both into GS. It alternates -- enforce $|.|^2=I_1$ in the $D_1$ domain, propagate to
the $D_2$ domain, enforce $|.|^2=I_2$, propagate back -- and converges to the one phase
consistent with both. The recovered phase tracks the true phase (RMS ~20 deg at 35 dB
SNR)."""),
co("""phi_est, errors = gs.retrieve_phase(I1, I2, D1=D1, D2=D2, n_iter=50, unit_amplitude=True)
off = np.angle(np.mean(np.exp(1j*(phi_true-phi_est)))); phi_al = np.angle(np.exp(1j*(phi_est+off)))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(phi_true[:160], lw=2, label="true phase"); ax[0].plot(phi_al[:160], "--", label="recovered")
ax[0].set(xlabel="sample", ylabel="phase (rad)", title=f"recovered phase (RMS {rms_err(phi_true,phi_est):.1f} deg)"); ax[0].legend()
ax[1].semilogy(errors); ax[1].set(xlabel="GS iteration", ylabel="error", title="GS converges")
plt.tight_layout(); plt.show()
print(f"TWO intensities (diverse): recovered phase RMS = {rms_err(phi_true, phi_est):.2f} deg")"""),

md("""## 4. Why DIVERSITY -- two nearly-equal dispersions still fail

If $D_1$ and $D_2$ are almost the same, $I_2$ is nearly a copy of $I_1$ (correlation
~0.96) -- effectively *one* measurement, and GS cannot resolve the phase (RMS blows up).
The two intensities must be **different enough**; that is the whole reason the receiver
chooses well-separated dispersions."""),
co("""dl = gs.make_qpsk_measurements(n_symbols=128, snr_db=35.0, D1=-5000.0, D2=-5100.0, rng_seed=3)
pe_lo, _ = gs.retrieve_phase(dl['I1'], dl['I2'], D1=dl['D1'], D2=dl['D2'], n_iter=50, unit_amplitude=True)
corr_lo = np.corrcoef(dl['I1'], dl['I2'])[0,1]
print(f"GOOD diversity (D2=-5750, corr 0.20): RMS = {rms_err(phi_true, phi_est):.1f} deg  -> phase recovered")
print(f"POOR diversity (D2=-5100, corr {corr_lo:.2f}): RMS = {rms_err(dl['phi_true'], pe_lo):.1f} deg  -> fails")
plt.figure(figsize=(6.5,3.4))
plt.bar(["good diversity\\n(corr 0.20)", "poor diversity\\n(corr 0.96)"],
        [rms_err(phi_true, phi_est), rms_err(dl['phi_true'], pe_lo)], color=["C2","C3"])
plt.ylabel("phase RMS error (deg)"); plt.title("two intensities work only if they are DIFFERENT enough")
plt.tight_layout(); plt.show()"""),

md("""## Takeaway

1. **One intensity -> many phases.** $|E|^2$ discards the phase (Born rule), so a single
   measurement is ambiguous -- exactly the diffraction $|\\mathcal F|^2$ phase loss.
2. **Two dispersed intensities -> the unique phase.** $I_1$ and $I_2$ are different views;
   Gerchberg-Saxton alternates between them and converges (RMS ~20 deg at 35 dB).
3. **Diversity is essential.** Near-equal dispersions (corr ~0.96) behave like one
   measurement and fail; the receiver picks well-separated $D_1, D_2$ on purpose.

That is the entire idea of the dispersion-assisted receiver: trade a second intensity
(at a different dispersion) for the phase a detector cannot see. Civilian education."""),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/why_two_intensities.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
