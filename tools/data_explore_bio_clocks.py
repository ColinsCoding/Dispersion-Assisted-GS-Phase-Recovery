#!/usr/bin/env python3
"""
tools/data_explore_bio_clocks.py

Automated exploratory data transform and analysis for photonics/biological-clock style experiments.

Usage:
  python tools/data_explore_bio_clocks.py --input <path_or_dir> --out <outdir> [--torch] [--max-features N]

No stdin required. Script writes plots and a JSON summary to the output directory.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports
try:
    import scipy.io as spio
except Exception:
    spio = None
try:
    import umap
except Exception:
    umap = None
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

def find_files(root: Path):
    exts = ('.npz', '.npy', '.csv', '.mat', '.txt')
    files = []
    for p in root.rglob('*'):
        if p.suffix.lower() in exts:
            files.append(p)
    return files

def load_npz_or_npy(p: Path):
    try:
        if p.suffix.lower() == '.npz':
            d = np.load(str(p), allow_pickle=True)
            # prefer arrays named i1/i2 or numeric arrays
            arrays = {}
            for k in getattr(d, 'files', []):
                arrays[k] = np.asarray(d[k])
            return arrays
        else:
            arr = np.load(str(p), allow_pickle=True)
            return {'arr': np.asarray(arr)}
    except Exception as e:
        print("Failed to load", p, e)
        return {}

def load_csv(p: Path):
    try:
        df = pd.read_csv(str(p))
        return {'table': df}
    except Exception as e:
        print("Failed to read CSV", p, e)
        return {}

def load_mat(p: Path):
    if spio is None:
        print("scipy not available; cannot read .mat:", p)
        return {}
    try:
        m = spio.loadmat(str(p))
        arrays = {}
        for k, v in m.items():
            if not k.startswith('__'):
                arrays[k] = np.asarray(v)
        return arrays
    except Exception as e:
        print("Failed to load MAT", p, e)
        return {}

def coerce_to_matrix(collected: dict, max_features=2000):
    """
    Build a data matrix (samples x features).
    Strategy:
      - If tabular data present (pandas DataFrame), use it directly (rows = samples).
      - Else, collect 1D arrays and stack them as columns (align by length).
      - If arrays have same length, treat each array as a feature vector across samples.
      - If arrays are 2D, flatten or use as multiple features.
    """
    tables = [v for k, v in collected.items() if isinstance(v, pd.DataFrame)]
    arrays = {k: v for k, v in collected.items() if not isinstance(v, pd.DataFrame)}
    if tables:
        # merge tables by index if multiple
        df = tables[0].copy()
        for t in tables[1:]:
            df = pd.concat([df.reset_index(drop=True), t.reset_index(drop=True)], axis=1)
        # drop non-numeric columns
        df_num = df.select_dtypes(include=[np.number]).copy()
        if df_num.shape[1] == 0:
            raise RuntimeError("No numeric columns in CSV/DFs.")
        return df_num.values, list(df_num.columns)
    # else build from arrays
    # find 1D arrays and common length
    one_d = {}
    for k, v in arrays.items():
        a = np.asarray(v)
        if a.ndim == 0:
            continue
        if a.ndim == 1:
            one_d[k] = a
        elif a.ndim == 2:
            # if shape (n_iter, N) and n_iter > N, maybe treat columns as features
            n0, n1 = a.shape
            if n0 >= n1:
                # flatten rows as features (take mean across first axis)
                one_d[k + "_mean"] = np.mean(a, axis=0)
            else:
                one_d[k + "_flat"] = a.ravel()
        else:
            one_d[k + "_flat"] = a.ravel()
    if not one_d:
        raise RuntimeError("No usable 1D numeric arrays found to build matrix.")
    # choose the longest common length or pad/truncate
    lengths = [len(v) for v in one_d.values()]
    target = max(lengths)
    # build columns by padding with NaN and then drop columns with too many NaNs
    cols = []
    names = []
    for k, v in one_d.items():
        arr = np.asarray(v).ravel()
        if arr.size == target:
            cols.append(arr)
            names.append(k)
        else:
            # pad with NaN at end
            pad = np.full(target - arr.size, np.nan)
            cols.append(np.concatenate([arr, pad]))
            names.append(k)
    M = np.vstack(cols).T  # shape (target, n_features)
    # drop columns with >50% NaN
    valid = np.isnan(M).mean(axis=0) <= 0.5

    M = M[:, valid]
    names = [n for i, n in enumerate(names) if valid[i]]
    # fill remaining NaN with column mean
    col_means = np.nanmean(M, axis=0)
    inds = np.where(np.isnan(M))
    if inds[0].size:
        M[inds] = np.take(col_means, inds[1])
    # limit features
    if M.shape[1] > max_features:
        M = M[:, :max_features]
        names = names[:max_features]
    return M, names

def svd_denoise(M, n_components=10, use_torch=False):
    """Return denoised matrix using truncated SVD reconstruction."""
    if use_torch and TORCH_AVAILABLE:
        try:
            import torch
            X = torch.tensor(M, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            U = U[:, :n_components]
            S = S[:n_components]
            Vt = Vt[:n_components, :]
            Xd = (U * S) @ Vt
            return Xd.cpu().numpy()
        except Exception as e:
            print("Torch SVD failed, falling back to numpy/scikit:", e)
    # fallback: TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    Z = svd.fit_transform(M)
    M_approx = svd.inverse_transform(Z)
    return M_approx

def quick_plots(M, feature_names, outdir: Path, prefix="explore"):
    outdir.mkdir(parents=True, exist_ok=True)
    # basic heatmap of first 200 features
    plt.figure(figsize=(10,6))
    imax = min(M.shape[1], 200)
    plt.imshow(M[:, :imax].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='value')
    plt.xlabel('sample index')
    plt.ylabel('feature (first {})'.format(imax))
    plt.title('Feature heatmap (first {})'.format(imax))
    plt.tight_layout()
    p1 = outdir / f"{prefix}_heatmap.png"
    plt.savefig(p1, dpi=150); plt.close()

    # variance explained by PCA
    scaler = StandardScaler()
    Mz = scaler.fit_transform(M)
    pca = PCA(n_components=min(20, Mz.shape[1]), random_state=0)
    pca.fit(Mz)
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_)*100, '-o')
    plt.xlabel('PC index'); plt.ylabel('Cumulative explained variance (%)')
    plt.title('PCA cumulative variance')
    plt.grid(True)
    p2 = outdir / f"{prefix}_pca_cumvar.png"
    plt.savefig(p2, dpi=150); plt.close()

    # scatter of first two PCs
    pcs = pca.transform(Mz)[:, :2]
    plt.figure(figsize=(6,5))
    plt.scatter(pcs[:,0], pcs[:,1], s=10, alpha=0.8)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PC1 vs PC2')
    p3 = outdir / f"{prefix}_pc1_pc2.png"
    plt.savefig(p3, dpi=150); plt.close()

    return [p1, p2, p3]

def clustering_and_manifold(M, outdir: Path, prefix="explore"):
    outdir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    Mz = scaler.fit_transform(M)
    # KMeans with 2..5 clusters and silhouette-like quick metric (inertia)
    results = {}
    for k in (2,3,4,5):
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Mz)
        results[f'k{k}_labels'] = km.labels_.tolist()
        results[f'k{k}_inertia'] = float(km.inertia_)
    # t-SNE (slow for large N)
    tsne_path = None
    try:
        ts = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto', perplexity=30)
        Y = ts.fit_transform(Mz)
        plt.figure(figsize=(6,5))
        plt.scatter(Y[:,0], Y[:,1], s=8, alpha=0.8)
        plt.title('t-SNE embedding')
        tsne_path = outdir / f"{prefix}_tsne.png"
        plt.savefig(tsne_path, dpi=150); plt.close()
        results['tsne'] = str(tsne_path)
    except Exception as e:
        results['tsne_error'] = str(e)
    # UMAP if available
    umap_path = None
    if umap is not None:
        try:
            reducer = umap.UMAP(random_state=0)
            U = reducer.fit_transform(Mz)
            plt.figure(figsize=(6,5))
            plt.scatter(U[:,0], U[:,1], s=8, alpha=0.8)
            plt.title('UMAP embedding')
            umap_path = outdir / f"{prefix}_umap.png"
            plt.savefig(umap_path, dpi=150); plt.close()
            results['umap'] = str(umap_path)
        except Exception as e:
            results['umap_error'] = str(e)
    return results

def concentration_estimate_if_possible(M, feature_names, collected, outdir: Path):
    """
    If a column named 'concentration' or 'conc' or 'cal' exists in any loaded table, fit a linear model
    from features -> concentration and report RMSE. Save a scatter plot of predicted vs true.
    """
    out = {}
    # search for calibration in collected (pandas tables)
    calib = None
    for k, v in collected.items():
        if isinstance(v, pd.DataFrame):
            for col in v.columns:
                if col.lower() in ('concentration','conc','cal','label'):
                    calib = v[col].astype(float).values
                    break
        elif k.lower() in ('concentration','conc','cal','label'):
            try:
                calib = np.asarray(v).ravel().astype(float)
            except Exception:
                pass
        if calib is not None:
            break
    if calib is None:
        out['found_calibration'] = False
        return out
    out['found_calibration'] = True
    # align lengths
    n = M.shape[0]
    if calib.size != n:
        # try to trim or pad
        if calib.size > n:
            calib = calib[:n]
        else:
            calib = np.pad(calib, (0, n - calib.size), 'edge')
    # simple linear regression on first 10 PCs
    scaler = StandardScaler()
    Mz = scaler.fit_transform(M)
    pca = PCA(n_components=min(10, Mz.shape[1]), random_state=0)
    X = pca.fit_transform(Mz)
    model = LinearRegression().fit(X, calib)
    pred = model.predict(X)
    rmse = mean_squared_error(calib, pred, squared=False)
    out['rmse'] = float(rmse)
    out['coef'] = model.coef_.tolist()
    # save scatter
    plt.figure(figsize=(5,4))
    plt.scatter(calib, pred, s=8, alpha=0.8)
    plt.plot([calib.min(), calib.max()], [calib.min(), calib.max()], 'k--', lw=1)
    plt.xlabel('true concentration'); plt.ylabel('predicted')
    plt.title(f'Concentration fit RMSE={rmse:.3g}')
    p = outdir / "concentration_pred_vs_true.png"
    plt.savefig(p, dpi=150); plt.close()
    out['plot'] = str(p)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/raw", help="file or directory to search for data")
    parser.add_argument("--out", "-o", default="outputs/data_explore_bio_clocks", help="output directory")
    parser.add_argument("--torch", action="store_true", help="use torch (CUDA) for SVD if available")
    parser.add_argument("--max-features", type=int, default=2000, help="limit number of features")
    args = parser.parse_args()

    root = Path(args.input)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # collect data
    collected = {}
    if root.is_file():
        files = [root]
    else:
        files = find_files(root) if root.exists() else []
        # also check common repo locations
        if not files:
            for alt in (Path("figures"), Path(".")):
                files += find_files(alt)
    # load files
    for f in files:
        try:
            if f.suffix.lower() in ('.npz', '.npy'):
                arrays = load_npz_or_npy(f)
                for k, v in arrays.items():
                    # convert to 1D or 2D numeric
                    if isinstance(v, np.ndarray):
                        if v.ndim == 2 and v.shape[0] == 1:
                            v = v.ravel()
                        collected[f"{f.stem}_{k}"] = v
                    else:
                        collected[f"{f.stem}_{k}"] = np.asarray(v)
            elif f.suffix.lower() == '.csv':
                df = pd.read_csv(str(f))
                collected[f"{f.stem}_table"] = df
            elif f.suffix.lower() == '.mat':
                arrays = load_mat(f)
                for k, v in arrays.items():
                    collected[f"{f.stem}_{k}"] = v
        except Exception as e:
            print("Error loading", f, e)

    # if recon_artifact present, load its arrays too
    artifact = Path("figures/recon_artifact.npz")
    if artifact.exists():
        try:
            d = np.load(str(artifact), allow_pickle=True)
            for k in d.files:
                collected[f"artifact_{k}"] = d[k]
        except Exception:
            pass

    # build matrix
    try:
        M, feature_names = coerce_to_matrix(collected, max_features=args.max_features)
    except Exception as e:
        print("Failed to build data matrix:", e)
        sys.exit(1)

    summary = {
        "n_samples": int(M.shape[0]),
        "n_features": int(M.shape[1]),
        "feature_names_sample": feature_names[:20]
    }

    # SVD denoise and PCA
    try:
        M_denoised = svd_denoise(M, n_components=min(20, M.shape[1]), use_torch=(args.torch and TORCH_AVAILABLE))
        summary['denoised'] = True
    except Exception as e:
        print("SVD denoise failed:", e)
        M_denoised = M.copy()
        summary['denoised'] = False

    # quick plots
    plot_paths = quick_plots(M_denoised, feature_names, outdir, prefix="bio")
    summary['plots'] = [str(p.resolve()) for p in plot_paths]

    # clustering and manifold
    cluster_results = clustering_and_manifold(M_denoised, outdir, prefix="bio")
    summary['clustering'] = cluster_results

    # concentration estimate if possible
    conc_res = concentration_estimate_if_possible(M, feature_names, collected, outdir)
    summary['concentration'] = conc_res

    # save denoised matrix (compressed)
    np.savez_compressed(outdir / "matrix_denoised.npz", M=M_denoised, feature_names=np.array(feature_names, dtype=object))
    summary['matrix_saved'] = str((outdir / "matrix_denoised.npz").resolve())

    # save summary
    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Exploration complete. Summary written to", (outdir / "summary.json").resolve())
    print("Outputs in", outdir.resolve())

if __name__ == "__main__":
    main()

