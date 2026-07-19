#!/usr/bin/env python3
"""
tools/resample_build_and_animate.py

- Loads figures/recon_artifact.npz and figures/recon_artifact_with_meas.npz (if present).
- Collects 1D numeric arrays, resamples each to a common target length (median length).
- Builds matrix M (samples x features), standardizes, runs TruncatedSVD denoising.
- Saves denoised matrix and summary to outputs/data_explore_bio_clocks/.
- Attempts to run tools/animate_gs_convergence.py --recompute --frames 300 using the same Python interpreter.
"""
from pathlib import Path
import numpy as np
import json
import sys
import subprocess

try:
    from scipy import interpolate
except Exception as e:
    print("scipy required but not available:", e)
    sys.exit(2)

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    print("scikit-learn required but not available:", e)
    sys.exit(2)

ROOT = Path('.').resolve()
OUT = ROOT / "outputs" / "data_explore_bio_clocks"
OUT.mkdir(parents=True, exist_ok=True)

def load_npz(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        d = np.load(str(p), allow_pickle=True)
        return {k: d[k] for k in d.files}
    except Exception as e:
        print("Failed to load", p, e)
        return {}

artifact = load_npz(ROOT / "figures" / "recon_artifact.npz")
meas = load_npz(ROOT / "figures" / "recon_artifact_with_meas.npz")

print("Loaded artifact keys:", list(artifact.keys()))
print("Loaded measurement keys:", list(meas.keys()))

# collect 1D numeric arrays
cols = []
names = []
lengths = []
sources = []

for k, v in {**meas, **artifact}.items():
    a = np.asarray(v)
    if a.ndim == 1 and a.size > 1 and np.issubdtype(a.dtype, np.number):
        cols.append(a.astype(float))
        names.append(k)
        lengths.append(a.size)
        sources.append("meas" if k in meas else "artifact")

if not cols:
    print("No 1D numeric arrays found to build matrix. Exiting.")
    sys.exit(0)

# choose target length (median)
target = int(np.median(lengths))
print("Array lengths found (unique):", sorted(set(lengths))[:10], "...")
print("Using target length:", target)

# resample each column to target length
resampled = []
for arr, nm in zip(cols, names):
    n = arr.size
    if n == target:
        resampled.append(arr)
    else:
        x_old = np.linspace(0.0, 1.0, n)
        x_new = np.linspace(0.0, 1.0, target)
        # fill NaNs with column mean
        a = np.array(arr, dtype=float)
        if np.isnan(a).any():
            a = np.where(np.isnan(a), np.nanmean(a), a)
        f = interpolate.interp1d(x_old, a, kind='linear', bounds_error=False, fill_value=(a[0], a[-1]))
        resampled.append(f(x_new))
resampled = np.vstack(resampled).T  # shape (target, n_features)
print("Built resampled matrix shape:", resampled.shape)

# standardize and truncated SVD denoise
scaler = StandardScaler()
Mz = scaler.fit_transform(resampled)
ncomp = min(20, Mz.shape[1])
svd = TruncatedSVD(n_components=ncomp, random_state=0)
Z = svd.fit_transform(Mz)
M_approx = svd.inverse_transform(Z)

# save outputs
np.savez_compressed(OUT / "matrix_resampled.npz", M=resampled, feature_names=np.array(names, dtype=object), sources=np.array(sources, dtype=object))
np.savez_compressed(OUT / "matrix_denoised.npz", M_denoised=M_approx, feature_names=np.array(names, dtype=object), sources=np.array(sources, dtype=object))

summary = {
    "resampled_shape": list(resampled.shape),
    "denoised_shape": list(M_approx.shape),
    "feature_names_sample": names[:20],
    "sources_sample": sources[:20],
    "target_length": target
}
with open(OUT / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print("Saved resampled and denoised matrices to", OUT)
print("Summary written to", OUT / "summary.json")

# If gs_iterates.npz exists in outputs (from notebook), move it to figures for animator to pick up
gs_iterates = OUT / "gs_iterates.npz"
if gs_iterates.exists():
    dest = ROOT / "figures" / "gs_iterates.npz"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        gs_iterates.replace(dest)
        print("Moved gs_iterates.npz to figures/")
    except Exception as e:
        print("Failed to move gs_iterates.npz:", e)

# Attempt to run the animator script (tools/animate_gs_convergence.py) with --recompute
animator = ROOT / "tools" / "animate_gs_convergence.py"
if animator.exists():
    cmd = [sys.executable, str(animator), "--recompute", "--frames", "300"]
    print("Running animator:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        print("Animator stdout:")
        print(proc.stdout)
        print("Animator stderr:")
        print(proc.stderr)
    except Exception as e:
        print("Failed to run animator:", e)
else:
    print("Animator script not found at", animator)
