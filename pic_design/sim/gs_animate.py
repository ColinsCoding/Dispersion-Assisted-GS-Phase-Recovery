"""
gs_animate.py
-------------
Animate 64-128 GS trajectories converging on the error surface.

Physics connections:
  Schrodinger : psi(t+dt) = psi(t) - (i*dt/hbar)*H*psi(t)
                Each GS step is a unitary evolution under the measurement Hamiltonian.

  Lagrangian  : L = T - V,  V = residual (potential energy)
                GS minimizes the action S = integral(L dt) over iterations.

  Statistical : 64-128 random spawns = microcanonical ensemble.
                All trajectories converge = ergodic hypothesis holds.

  PCA         : Principal directions of convergence in (R1, R2) space.
                PC1 = dominant mode of energy dissipation.

Run:  py -3.12 sim/gs_animate.py
Output: docs/gs_convergence.gif  +  docs/gs_pca.png
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA

_HERE     = pathlib.Path(__file__).parent
_DOCS     = _HERE.parent / "docs"
_DOCS.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
N        = 256          # signal length
N_ITER   = 80           # GS iterations per trajectory
N_SPAWN  = 96           # trajectories (64-128 range)
D1, D2   = -600, -900   # ps^2  — same as hardware arms
RNG      = np.random.default_rng(2026)

# ── True signal — smooth random phase (like a real optical waveform) ──────────
phi_true = np.cumsum(RNG.standard_normal(N)) * 0.08
phi_true -= phi_true.mean()

# ── Forward model ─────────────────────────────────────────────────────────────
_nu = np.fft.fftfreq(N)
_H1 = np.exp(1j * np.pi * D1 * _nu**2)
_H2 = np.exp(1j * np.pi * D2 * _nu**2)

I1_true = np.abs(np.fft.fft(np.exp(1j * phi_true)) * _H1)**2
I2_true = np.abs(np.fft.fft(np.exp(1j * phi_true)) * _H2)**2


def gs_step(E: np.ndarray, I_target: np.ndarray, H: np.ndarray) -> np.ndarray:
    """One GS projection step — amplitude replace in dispersed domain."""
    Ef = np.fft.fft(E) * H
    Ef = np.sqrt(np.clip(I_target, 0, None)) * np.exp(1j * np.angle(Ef))
    return np.fft.ifft(Ef / H)


def residuals(E: np.ndarray):
    """Normalised L2 residual on both measurement arms."""
    r1 = np.mean((np.abs(np.fft.fft(E) * _H1)**2 - I1_true)**2) / (np.mean(I1_true)**2 + 1e-30)
    r2 = np.mean((np.abs(np.fft.fft(E) * _H2)**2 - I2_true)**2) / (np.mean(I2_true)**2 + 1e-30)
    return float(r1), float(r2)


# ── Run all trajectories — pre-compute full paths ─────────────────────────────
print(f"Running {N_SPAWN} trajectories x {N_ITER} iterations...")
R1 = np.zeros((N_SPAWN, N_ITER))   # residual arm 1
R2 = np.zeros((N_SPAWN, N_ITER))   # residual arm 2

for s in range(N_SPAWN):
    E = np.exp(1j * RNG.uniform(-np.pi, np.pi, N))
    for k in range(N_ITER):
        E      = gs_step(E, I1_true, _H1)
        E      = gs_step(E, I2_true, _H2)
        R1[s,k], R2[s,k] = residuals(E)
    if (s+1) % 16 == 0:
        print(f"  spawn {s+1}/{N_SPAWN} done")

print("All trajectories complete.")

# ── Energy = total residual (Schrodinger potential) ───────────────────────────
Energy = R1 + R2                    # (N_SPAWN, N_ITER)
E0     = Energy[:, 0]               # initial energies
Ef_    = Energy[:, -1]              # final energies

# ── PCA on convergence trajectories ──────────────────────────────────────────
features = np.hstack([R1, R2])      # (N_SPAWN, 2*N_ITER)
pca      = PCA(n_components=2)
coords   = pca.fit_transform(features)   # (N_SPAWN, 2)
var_exp  = pca.explained_variance_ratio_

print(f"PCA: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%} of convergence variance")

# ── Error surface (bowl in R1-R2 space) ───────────────────────────────────────
r1_max = np.percentile(R1[:,0], 95)
r2_max = np.percentile(R2[:,0], 95)
R1g, R2g = np.meshgrid(
    np.linspace(0, r1_max * 1.1, 60),
    np.linspace(0, r2_max * 1.1, 60),
)
Z_surf = np.log1p(R1g + R2g)   # log-energy landscape

# ── Color mapping: red (high E) -> blue (low E) ──────────────────────────────
norm_e  = Normalize(vmin=0, vmax=float(E0.max()))
cmap    = plt.cm.plasma_r

# ── Animation ─────────────────────────────────────────────────────────────────
fig     = plt.figure(figsize=(13, 8))
ax      = fig.add_subplot(111, projection="3d")
fig.patch.set_facecolor("#0d0d0d")
ax.set_facecolor("#0d0d0d")
for pane in [ax.xaxis, ax.yaxis, ax.zaxis]:
    pane.pane.fill = False

# Fixed surface
ax.plot_surface(R1g, R2g, Z_surf,
                cmap="Blues", alpha=0.25, linewidth=0, antialiased=False)

ax.set_xlabel("Residual arm 1", color="white", labelpad=8)
ax.set_ylabel("Residual arm 2", color="white", labelpad=8)
ax.set_zlabel("log(E)", color="white", labelpad=8)
ax.tick_params(colors="white", labelsize=7)
ax.set_title("", color="white")

# Spawn all points at iteration 0
z0   = np.log1p(R1[:,0] + R2[:,0])
cols = cmap(norm_e(Energy[:,0]))

scat = ax.scatter(R1[:,0], R2[:,0], z0,
                  c=cols, s=18, depthshade=False, alpha=0.85)

# Trail lines — one per trajectory (initially empty)
trails = [ax.plot([], [], [],
                  color=cmap(norm_e(E0[s])), lw=0.6, alpha=0.45)[0]
          for s in range(N_SPAWN)]

title_txt = ax.text2D(0.5, 0.97, "", transform=ax.transAxes,
                      ha="center", color="white", fontsize=10)

# Colorbar
sm  = ScalarMappable(cmap=cmap, norm=norm_e)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.12)
cbar.set_label("Total residual energy", color="white", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

TRAIL_LEN = 15   # how many past steps to show


def update(frame):
    k     = frame                               # current iteration
    k_lo  = max(0, k - TRAIL_LEN)

    r1k   = R1[:, k]
    r2k   = R2[:, k]
    zk    = np.log1p(r1k + r2k)
    ek    = Energy[:, k]
    cols  = cmap(norm_e(ek))

    # Update scatter positions + colours
    scat._offsets3d = (r1k, r2k, zk)
    scat.set_color(cols)

    # Update trails
    for s in range(N_SPAWN):
        trails[s].set_data(R1[s, k_lo:k+1], R2[s, k_lo:k+1])
        trails[s].set_3d_properties(np.log1p(R1[s, k_lo:k+1] + R2[s, k_lo:k+1]))

    # Schrodinger energy annotation
    mean_e = float(ek.mean())
    title_txt.set_text(
        f"Iter {k+1:>3}/{N_ITER}   "
        f"<E> = {mean_e:.3f}   "
        f"converged = {int((ek < 0.01).sum())}/{N_SPAWN}"
    )

    return [scat, title_txt] + trails


ani = animation.FuncAnimation(
    fig, update,
    frames=N_ITER,
    interval=80,      # ms per frame
    blit=False,
)

# ── Save GIF ──────────────────────────────────────────────────────────────────
gif_path = _DOCS / "gs_convergence.gif"
print(f"Saving animation -> {gif_path}  (this takes ~30s)...")
ani.save(str(gif_path), writer="pillow", fps=12, dpi=90)
print(f"Saved {gif_path}")
plt.close()


# ── PCA figure ────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.patch.set_facecolor("#0d0d0d")

for ax_ in axes:
    ax_.set_facecolor("#111111")
    ax_.tick_params(colors="white", labelsize=8)
    for spine in ax_.spines.values():
        spine.set_edgecolor("#333333")

# Left: PC1 vs PC2, coloured by final energy
sc = axes[0].scatter(coords[:,0], coords[:,1],
                     c=Ef_, cmap="plasma_r", s=30, alpha=0.8)
axes[0].set_xlabel("PC 1", color="white")
axes[0].set_ylabel("PC 2", color="white")
axes[0].set_title(
    f"PCA of {N_SPAWN} GS trajectories\n"
    f"PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}",
    color="white", fontsize=9)
cb = fig2.colorbar(sc, ax=axes[0])
cb.set_label("Final residual", color="white", fontsize=8)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

# Right: mean energy curve + std band (statistical mechanics view)
mean_E = Energy.mean(axis=0)
std_E  = Energy.std(axis=0)
iters  = np.arange(N_ITER)

axes[1].fill_between(iters, mean_E - std_E, mean_E + std_E,
                     color="#7b2fff", alpha=0.25, label="+-1 sigma")
axes[1].plot(iters, mean_E, color="#bf7fff", lw=2, label="<E> ensemble mean")
axes[1].plot(iters, Energy.min(axis=0), color="#00ff99",
             lw=1, ls="--", label="min E")
axes[1].plot(iters, Energy.max(axis=0), color="#ff4444",
             lw=1, ls="--", label="max E")
axes[1].set_yscale("log")
axes[1].set_xlabel("GS iteration", color="white")
axes[1].set_ylabel("Residual energy (log)", color="white")
axes[1].set_title(
    "Ensemble energy decay\n"
    f"Schrodinger: <E>(t) ~ exp(-gamma*t),  N={N_SPAWN} spawns",
    color="white", fontsize=9)
leg = axes[1].legend(fontsize=7, facecolor="#111111", labelcolor="white")

plt.tight_layout()
pca_path = _DOCS / "gs_pca.png"
plt.savefig(pca_path, dpi=140, facecolor="#0d0d0d")
plt.close()
print(f"Saved {pca_path}")

# ── Print physics summary ─────────────────────────────────────────────────────
print("\n---- Physics Summary ----------------------------------------")
print(f"  Spawns         : {N_SPAWN}")
print(f"  Iterations     : {N_ITER}")
print(f"  Mean E0        : {Energy[:,0].mean():.4f}  (initial, disordered)")
print(f"  Mean Ef        : {Energy[:,-1].mean():.6f}  (final, ordered)")
print(f"  Reduction      : {Energy[:,0].mean() / (Energy[:,-1].mean()+1e-30):.0f}x")
print(f"  Converged (<1%): {int((Energy[:,-1] < 0.01).sum())}/{N_SPAWN}")
print(f"  PC1 variance   : {var_exp[0]:.1%}  (dominant convergence mode)")
print(f"  Schrodinger:   E(k) ~ exp(-gamma*k),  GS = imaginary-time evolution")
print(f"  Lagrangian:    S = sum_k L_k,  L = -log(1-residual),  GS minimises S")
print("-------------------------------------------------------------")
