"""Generate notebooks/dispersion_gs_gradient_descent.ipynb -- torch gradient-descent
phase retrieval vs Gerchberg-Saxton, with a smoothness prior. Runs on py312.
NOTE: no triple-double-quote docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "dispersion_gs_gradient_descent.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3.12 (torch)", "language": "python", "name": "py312"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Phase Retrieval by Gradient Descent — Torch Autograd vs Gerchberg-Saxton

The previous prototype showed GS **stagnating** on the hard, low-diversity dispersion problem. Here
is the modern alternative: make the whole forward model differentiable and **descend on the phase
directly**.

$$\phi\ \longrightarrow\ x=\sqrt{I_1}\,e^{i\phi}\ \longrightarrow\ \text{disperse}(x,D)\
\longrightarrow\ |\cdot|^2,\qquad
\mathcal L=\big\||\text{disperse}|^2-I_2\big\|^2+\lambda\,\|\nabla x\|^2.$$

Two things gradient descent has that GS does not: a smoother optimisation path (it doesn't bounce
between hard projections), and the ability to add a **prior** — here a smoothness penalty, since the
true optical phase is smooth. On the same ill-posed signal where GS gave ~1.6 rad, autograd reaches
~0.5 rad. Civilian optical metrology / silicon photonics / education; runs on the **py312 (torch)**
kernel. Engine: `dispersion_gs_descent.py`.""")

code(r"""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import dispersion_gs_prototype as dg
import dispersion_gs_descent as gd

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", dev)

# the same hard signal as the prototype
t, x, A, phi = dg.make_field(2048, seed=0)
I1 = np.abs(x)**2""")

# ── §1 GD vs GS ────────────────────────────────────────────────────
md(r"""## §1 Gradient descent vs Gerchberg-Saxton, same data

A single dispersion plane and a Gaussian-dominated pulse — the low-diversity case where GS stalls.
Gradient descent fits $I_2$ to near machine precision and, crucially, finds a phase much closer to
the truth.""")

code(r"""D = 12000.0
I2 = np.abs(dg.disperse(x, D))**2
print(f"corr(I1, I2) = {np.corrcoef(I1, I2)[0,1]:.3f}  (low diversity)")

# GS baseline
xr_gs, err_gs = dg.gerchberg_saxton(I1, I2, D, n_iter=400, seed=0)
rms_gs, phi_gs = dg.compare_phase(np.angle(xr_gs), phi, A**2)

# torch gradient descent (no prior)
phi_gd_raw, hist_gd = gd.torch_phase_retrieval(I1, I2, D, reg=0.0, n_iter=2000, seed=0)
rms_gd, phi_gd = gd.compare_phase(phi_gd_raw, phi, A**2)

print(f"GS:             phase RMS = {rms_gs:.3f} rad")
print(f"gradient descent: phase RMS = {rms_gd:.3f} rad  (data-loss {hist_gd[0]:.1e} -> {hist_gd[-1]:.1e})")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(t, phi, "k", lw=2, label="true $\\phi$")
ax[0].plot(t, phi_gs, "C0--", lw=1, label=f"GS (RMS {rms_gs:.2f})")
ax[0].plot(t, phi_gd, "C3--", lw=1, label=f"gradient descent (RMS {rms_gd:.2f})")
ax[0].set_xlim(-0.5, 0.7); ax[0].set_title("phase recovery (pulse region)"); ax[0].legend(fontsize=8)
ax[1].semilogy(err_gs, "C0", label="GS amplitude mismatch")
ax[1].semilogy(hist_gd/hist_gd[0], "C3", label="GD data-loss (norm.)")
ax[1].set_xlabel("iteration"); ax[1].set_title("convergence"); ax[1].legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout(); plt.show()""")

# ── §2 the prior ───────────────────────────────────────────────────
md(r"""## §2 Adding a prior — the thing GS cannot do

Because the loss is just a differentiable function, we can *add terms*. A field-smoothness penalty
$\lambda\|\nabla x\|^2$ encodes "the true phase is smooth" and breaks some of the residual
ambiguity. Sweep $\lambda$: a little regularisation helps, too much over-smooths.""")

code(r"""lams = [0.0, 0.2, 0.5, 1.0, 3.0, 10.0]
rms_lam = []
for lam in lams:
    pr, _ = gd.torch_phase_retrieval(I1, I2, D, reg=lam, n_iter=2000, seed=0)
    rms_lam.append(gd.compare_phase(pr, phi, A**2)[0])

plt.figure(figsize=(7.5, 3.4))
plt.plot(lams, rms_lam, "o-", color="C3")
plt.axhline(rms_gs, color="C0", ls="--", label=f"GS ({rms_gs:.2f} rad)")
plt.xlabel("smoothness weight $\\lambda$"); plt.ylabel("phase RMS (rad)")
plt.title("a smoothness prior sharpens the gradient-descent recovery"); plt.legend()
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
best = lams[int(np.argmin(rms_lam))]
print(f"best lambda = {best}: RMS = {min(rms_lam):.3f} rad  (vs GS {rms_gs:.3f})")""")

# ── §3 wrap ────────────────────────────────────────────────────────
md(r"""## §3 What gradient descent buys you

- **It is not magic on ill-posed data.** With *no* prior, GD and GS are both limited by diversity
  (§1 of the prototype) — gradient descent cannot invent information that $I_2$ does not contain.
  But on this signal it navigates the landscape better and lands at ~0.5 rad where GS sticks at ~1.6.
- **Priors are the real lever.** $\lambda\|\nabla x\|^2$ says "prefer smooth fields," which the true
  optical phase satisfies — exactly the kind of physical knowledge GS has no slot for. This is
  regularised phase retrieval, and it is why differentiable optics is replacing hand-built loops.
- **Autograd through the FFT = the FNO bridge.** The forward model here is an FFT, a multiply, and a
  modulus; backprop runs straight through it on the GPU. Swap the fixed dispersion kernel for learned
  spectral weights and this is the Fourier Neural Operator in `gs_fno.py` — the same pipeline, the
  same gradients, now learning the inverse map. Differentiable phase retrieval is the stepping stone
  from `gs_core.py` to the neural version.

Civilian scope throughout. Files: `dispersion_gs_descent.py` (torch gradient-descent phase
retrieval), benchmarked in `scripts/smoke_gs_descent.py`; companion to `dispersion_gs_prototype.py`.""")

nb.cells = cells
nbf.validate(nb)
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
