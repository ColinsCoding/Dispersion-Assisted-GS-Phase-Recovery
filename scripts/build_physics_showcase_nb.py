"""Build notebooks/receiver_physics_showcase.ipynb -- paper-quality summary plots."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# The dispersion-assisted carrier-less receiver — physics in five figures

A reading/showcase notebook: the core physics and the hard limits of the
carrier-less Gerchberg-Saxton receiver, each from a verified module in this repo,
plotted at publication quality. Civilian optical metrology / education.

1. **Absorption & dispersion** — why a medium has a complex index (Lorentz).
2. **Skin depth** — EM waves dying in a conductor.
3. **Low-light limit** — phase recovery vs photon budget (shot noise).
4. **Noise budget** — shot-noise √N and the ADC's 6 dB/bit.
5. **Fourier derivative theorem** — why dispersion is a spectral multiplier."""),

co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "lines.linewidth": 1.9, "figure.facecolor": "white"})
from griffiths import electrodynamics as ed
import dispersion_gs_prototype as dg, gs_core as gs, snr, fourier_tools as ft
print("modules loaded")"""),

md("""## 1. Absorption and dispersion — the complex refractive index
A Lorentz medium (bound electrons as damped oscillators) has
$\\varepsilon_r(\\omega)=1+\\omega_p^2/(\\omega_0^2-\\omega^2-i\\gamma\\omega)$, so the index
$\\tilde n=n+i\\kappa$ is complex: $n$ is **dispersion**, $\\kappa$ is **absorption**,
and they peak together at resonance (Kramers-Kronig)."""),
co("""w = np.linspace(0.2, 2.0, 4000); w0, gamma, wp = 1.0, 0.08, 0.4
nt = ed.complex_index(ed.lorentz_epsilon(w, w0, gamma, wp))
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(w, nt.real, color="#1f77b4", label="n (dispersion)")
ax1.set_xlabel("normalized frequency ω/ω₀"); ax1.set_ylabel("refractive index n", color="#1f77b4")
ax2 = ax1.twinx(); ax2.plot(w, nt.imag, color="#d62728", label="κ (absorption)")
ax2.set_ylabel("extinction κ", color="#d62728"); ax2.grid(False)
ax1.axvline(w0, ls="--", c="grey", lw=1); ax1.set_title("Lorentz absorption & dispersion")
plt.tight_layout(); plt.show()"""),

md("""## 2. Skin depth — EM waves in a conductor
Ohm's law makes the wavenumber complex; the field dies within a skin depth
$d=\\sqrt{2/(\\omega\\mu\\sigma)}$, thinning as $1/\\sqrt{f}$ (Griffiths 9.4.1)."""),
co("""f = np.logspace(1, 10, 400); sigma_cu = 5.96e7
d = np.array([ed.skin_depth(2*np.pi*fi, sigma_cu) for fi in f])
plt.figure(figsize=(7, 4))
plt.loglog(f, d*1e3, color="#2ca02c")
for fi, lab in [(60, "60 Hz\\n(8.4 mm)"), (1e6, "1 MHz\\n(65 µm)")]:
    plt.scatter([fi], [ed.skin_depth(2*np.pi*fi, sigma_cu)*1e3], color="#d62728", zorder=5)
    plt.annotate(lab, (fi, ed.skin_depth(2*np.pi*fi, sigma_cu)*1e3), fontsize=9)
plt.xlabel("frequency [Hz]"); plt.ylabel("skin depth [mm]")
plt.title("copper skin depth ∝ 1/√f"); plt.tight_layout(); plt.show()"""),

md("""## 3. Low-light limit — phase recovery vs photon budget
The detector counts photons (Poisson). As the light fades, shot noise on
$I_1,I_2$ grows and the recovered-phase RMS climbs as $1/\\sqrt{N}$."""),
co("""rng = np.random.default_rng(7)
data = gs.make_qpsk_measurements(n_symbols=128, D1=-5000.0, D2=-5750.0, snr_db=60.0)
pt, D1, D2 = data["phi_true"], data["D1"], data["D2"]
I1c, I2c = np.maximum(data["I1"], 0), np.maximum(data["I2"], 0)
def rms(I1, I2):
    phi, _ = gs.retrieve_phase(I1, I2, D1, D2, n_iter=80); best = None
    for s in (1, -1):
        off = np.angle(np.mean(np.exp(1j*(pt - s*phi))))
        e = np.sqrt(np.mean(np.angle(np.exp(1j*(pt - (s*phi+off))))**2))
        best = e if best is None else min(best, e)
    return best
budgets = np.logspace(1, 6, 10)
r = np.array([np.mean([rms(dg.photon_shot_noise(I1c, b, rng), dg.photon_shot_noise(I2c, b, rng))
                       for _ in range(3)]) for b in budgets])
plt.figure(figsize=(7, 4)); plt.semilogx(budgets, r, "o-", color="#9467bd")
plt.xlabel("mean photons / sample"); plt.ylabel("recovered-phase RMS [rad]")
plt.title("carrier-less recovery vs photon budget"); plt.tight_layout(); plt.show()
print("RMS:", np.round(r, 3))"""),

md("""## 4. Noise budget — the two floors
Shot-noise SNR grows as $\\sqrt{N}$ (10 log₁₀N dB); the ADC adds $6.02B+1.76$ dB
per bit. Both set how clean $I_1,I_2$ can be."""),
co("""N = np.logspace(1, 8, 200); bits = np.arange(4, 17)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].semilogx(N, snr.shot_noise_snr_db(N), color="#ff7f0e")
ax[0].set_xlabel("photons N"); ax[0].set_ylabel("SNR [dB]"); ax[0].set_title("shot noise: 10 log₁₀ N (√N)")
ax[1].plot(bits, [snr.quantization_snr_db(b) for b in bits], "s-", color="#8c564b")
ax[1].set_xlabel("ADC bits B"); ax[1].set_ylabel("SQNR [dB]"); ax[1].set_title("quantization: 6.02B + 1.76 dB")
plt.tight_layout(); plt.show()"""),

md("""## 5. Fourier derivative theorem — why dispersion is a spectral multiplier
$\\mathcal F\\{d/dt\\}=i\\omega\\,\\mathcal F$, so differentiation is multiplication in
frequency — *exact* for band-limited signals, unlike finite differences on noisy
data. This is why the dispersion PDE becomes $H(f)=e^{i\\pi Df^2}$."""),
co("""t = np.linspace(0, 1, 512, endpoint=False); dt = t[1]-t[0]
f = np.sin(2*np.pi*7*t); exact = 2*np.pi*7*np.cos(2*np.pi*7*t)
spec = ft.spectral_derivative(f, dt)
noisy = f + 1e-3*rng.standard_normal(f.size)
fd = np.gradient(noisy, dt)
plt.figure(figsize=(7, 4))
plt.plot(t, exact, color="k", label="analytic d/dt", lw=2.5, alpha=0.5)
plt.plot(t, spec, "--", color="#1f77b4", label=f"spectral (err {np.max(np.abs(spec-exact)):.0e})")
plt.plot(t, fd, color="#d62728", lw=1, alpha=0.7, label="finite-diff on noisy data")
plt.xlabel("time"); plt.ylabel("derivative"); plt.legend(fontsize=9)
plt.title("spectral derivative is exact; finite differences amplify noise")
plt.tight_layout(); plt.show()"""),

md("""## Summary

| figure | result | module |
|---|---|---|
| 1 | complex index: dispersion $n$ + absorption $\\kappa$ | `griffiths.electrodynamics` |
| 2 | copper skin depth $\\propto 1/\\sqrt f$ | `griffiths.electrodynamics` |
| 3 | recovery RMS vs photons ($1/\\sqrt N$) | `gs_core` + `dispersion_gs_prototype` |
| 4 | shot-noise & quantization SNR floors | `snr` |
| 5 | spectral derivative exact (the $i\\omega$ theorem) | `fourier_tools` |

Every plot comes from a module with a passing smoke test -- the figures are
reproducible, not hand-drawn. Civilian optical metrology / education.""")
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/receiver_physics_showcase.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
