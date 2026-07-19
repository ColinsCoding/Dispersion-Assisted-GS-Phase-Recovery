# notebooks/data_explore_bio_clocks.py
# %% [markdown]
# # Data Explore Bio Clocks
# 
# Interactive notebook for exploring the Dryad artifact and extracted measurements.
# 
# **What this notebook does**
# - Loads `figures/recon_artifact.npz` and `figures/recon_artifact_with_meas.npz` if present  
# - Shows time/frequency magnitude and phase diagnostics for available arrays  
# - Runs PCA / SVD denoising and quick clustering visualizations  
# - Provides a Gerchberg–Saxton reconstruction cell you can run to regenerate iterates and animate convergence  
# - Saves representative figures to `outputs/data_explore_bio_clocks/`
# 
# No external data required beyond the files in your repo. Use the toolbar to run cells.

# %% [markdown]
# ## Setup and imports

# %%
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")

# optional imports
try:
    import scipy.io as spio
except Exception:
    spio = None
try:
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
except Exception:
    PCA = TruncatedSVD = StandardScaler = KMeans = TSNE = None

OUTDIR = Path("outputs/data_explore_bio_clocks")
OUTDIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Helper functions

# %%
def load_npz_if_exists(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = np.load(str(p), allow_pickle=True)
        return {k: d[k] for k in d.files}
    except Exception as e:
        print("Failed to load", p, e)
        return None

def show_signal_summary(name, arr):
    arr = np.asarray(arr)
    print(f"{name}: shape={arr.shape} dtype={arr.dtype} min={np.nanmin(arr):.3g} max={np.nanmax(arr):.3g} mean={np.nanmean(arr):.3g}")

def plot_time_and_spectrum(x, fs=None, title=None, ax_time=None, ax_freq=None):
    x = np.asarray(x)
    t = np.arange(x.size) / fs if fs else np.arange(x.size)
    if ax_time is None:
        fig, (ax_time, ax_freq) = plt.subplots(1,2, figsize=(10,3))
    ax_time.plot(t, np.abs(x), lw=1)
    ax_time.set_title((title or "") + " time magnitude")
    ax_time.set_xlabel("time (s)" if fs else "sample")
    # spectrum
    N = x.size
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1.0/fs) if fs else np.fft.fftfreq(N)
    idx = np.argsort(freqs)
    ax_freq.plot(freqs[idx], np.abs(X)[idx], lw=1)
    ax_freq.set_title((title or "") + " spectrum magnitude")
    ax_freq.set_xlabel("freq (Hz)" if fs else "bins")
    return ax_time, ax_freq

def save_fig(fig, name):
    p = OUTDIR / name
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print("Saved", p)
    return p

# %% [markdown]
# ## Load artifact and extracted measurements

# %%
artifact = load_npz_if_exists("figures/recon_artifact.npz")
meas = load_npz_if_exists("figures/recon_artifact_with_meas.npz")

print("Artifact present:", bool(artifact))
print("Measurements present:", bool(meas))
if artifact:
    print("Artifact keys:", list(artifact.keys()))
if meas:
    print("Measurement keys:", list(meas.keys()))

# %% [markdown]
# ## Quick inspection of available arrays
# Run this cell to print summaries and view a few arrays.

# %%
if artifact:
    for k, v in artifact.items():
        show_signal_summary(f"artifact/{k}", v)
if meas:
    for k, v in meas.items():
        show_signal_summary(f"meas/{k}", v)

# %% [markdown]
# ## Visual diagnostics for the main arrays
# Pick arrays to visualize below. If `i1`/`i2` exist, they will be used; otherwise the notebook will show whatever numeric arrays were saved.

# %%
# Choose keys (edit if needed)
meas_keys = list(meas.keys()) if meas else []
artifact_keys = list(artifact.keys()) if artifact else []

print("Measurement keys:", meas_keys)
print("Artifact keys:", artifact_keys)

# Example: plot the first measurement and the artifact magnitude if present
if meas_keys:
    k = meas_keys[0]
    fig, axs = plt.subplots(1,2, figsize=(10,3))
    plot_time_and_spectrum(meas[k], title=f"meas/{k}", ax_time=axs[0], ax_freq=axs[1])
    save_fig(fig, f"meas_{k}_time_freq.png")
    plt.show()

if artifact and "mag" in artifact:
    fig, axs = plt.subplots(1,2, figsize=(10,3))
    plot_time_and_spectrum(np.abs(artifact["mag"]), title="artifact/mag", ax_time=axs[0], ax_freq=axs[1])
    save_fig(fig, "artifact_mag_time_freq.png")
    plt.show()

# %% [markdown]
# ## PCA and SVD denoising (magnitude domain)
# Convert complex arrays to magnitude before PCA/SVD. This cell computes a simple denoised reconstruction using truncated SVD.

# %%
# Build a matrix from available arrays: rows = samples, columns = features
cols = []
names = []
# prefer measurement arrays first
for k in meas_keys + artifact_keys:
    arr = np.asarray((meas or artifact)[k]) if (meas and k in meas) else np.asarray(artifact[k])
    if arr.ndim == 1:
        cols.append(np.abs(arr))
        names.append(k)
    elif arr.ndim == 2:
        # flatten or take mean across axis 0
        cols.append(np.abs(arr).mean(axis=0))
        names.append(k + "_mean")
if not cols:
    print("No 1D arrays found to build matrix.")
else:
    M = np.vstack(cols).T  # shape (n_samples, n_features)
    print("Built matrix M shape:", M.shape)
    # standardize
    scaler = StandardScaler()
    Mz = scaler.fit_transform(M)
    # truncated SVD
    ncomp = min(20, Mz.shape[1])
    if TruncatedSVD is not None:
        svd = TruncatedSVD(n_components=ncomp, random_state=0)
        Z = svd.fit_transform(Mz)
        M_approx = svd.inverse_transform(Z)
        # save denoised matrix
        np.savez_compressed(OUTDIR / "matrix_denoised.npz", M=M_approx, feature_names=np.array(names, dtype=object))
        print("Saved denoised matrix to outputs.")
    else:
        print("scikit-learn not available; skipping SVD.")

# %% [markdown]
# ## Quick clustering and embedding
# Run KMeans and t-SNE (if available) to get a sense of structure.

# %%
if 'M_approx' in locals():
    try:
        km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(M_approx)
        print("KMeans inertia:", km.inertia_)
        if TSNE is not None:
            Y = TSNE(n_components=2, random_state=0, init='pca').fit_transform(M_approx)
            fig = plt.figure(figsize=(6,5))
            plt.scatter(Y[:,0], Y[:,1], c=km.labels_, s=8, cmap='tab10')
            save_fig(fig, "tsne_clusters.png")
            plt.show()
    except Exception as e:
        print("Clustering/embedding failed:", e)

# %% [markdown]
# ## Gerchberg–Saxton reconstruction cell
# This cell runs a simple GS loop on a measured magnitude vector and stores iterates. Edit `mag_vec` to point to the measured magnitude you want to use.

# %%
def gs_reconstruct_from_mag(mag_vec, n_iter=200, seed=0):
    mag = np.asarray(mag_vec).ravel()
    N = mag.size
    rng = np.random.RandomState(seed)
    phi = rng.uniform(-np.pi, np.pi, size=N)
    f = mag * np.exp(1j * phi)
    iterates = np.zeros((n_iter, N), dtype=np.complex128)
    for k in range(n_iter):
        F = np.fft.fft(f)
        f_back = np.fft.ifft(F)
        iterates[k] = f_back
        f = mag * np.exp(1j * np.angle(f_back))
    return iterates

# Example usage: pick a measurement key or artifact mag
if meas_keys:
    mag_vec = np.abs(meas[meas_keys[0]])
elif artifact and "mag" in artifact:
    mag_vec = np.abs(artifact["mag"])
else:
    mag_vec = None

if mag_vec is not None:
    iters = gs_reconstruct_from_mag(mag_vec, n_iter=200, seed=0)
    print("Computed iterates shape:", iters.shape)
    # save iterates
    np.savez_compressed(OUTDIR / "gs_iterates.npz", iterates=iters)
    print("Saved iterates to outputs/gs_iterates.npz")
else:
    print("No magnitude vector available for GS reconstruction.")

# %% [markdown]
# ## Plot GS convergence diagnostics
# Plots magnitude at selected iterations and error curve.

# %%
if 'iters' in locals():
    n_iter = iters.shape[0]
    # compute error vs iteration (MSE between magnitude of iterate and measured mag)
    errs = np.mean((np.abs(iters) - mag_vec[None,:])**2, axis=1)
    fig, axs = plt.subplots(2,1, figsize=(8,6))
    axs[0].plot(errs, '-o', ms=3)
    axs[0].set_title("GS error vs iteration")
    # show final vs measured
    axs[1].plot(np.abs(iters[-1]), label='final recon')
    axs[1].plot(mag_vec, label='measured mag', alpha=0.7)
    axs[1].legend()
    save_fig(fig, "gs_convergence_example.png")
    plt.show()

# %% [markdown]
# ## Save a short JSON summary
# This writes a small summary file you can inspect.

# %%
summary = {
    "artifact_present": bool(artifact),
    "meas_present": bool(meas),
    "meas_keys": meas_keys,
    "artifact_keys": artifact_keys,
}
with open(OUTDIR / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("Wrote summary to", (OUTDIR / "summary.json").resolve())

# %% [markdown]
# ## Next steps and tips
# - Edit the key names at the top of the notebook to point to specific arrays you want to analyze.  
# - Use the GS cell to experiment with different `n_iter` and seeds.  
# - If you want an MP4/GIF of convergence, run the `tools/animate_gs_convergence.py` script after saving `gs_iterates.npz`.

