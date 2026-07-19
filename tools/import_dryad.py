#!/usr/bin/env python3
"""
tools/import_dryad.py

Usage from PowerShell in repo root:
  python .\tools\import_dryad.py --zip "C:/Users/mrjel/Downloads/doi_10_5061_dryad_2h7d2__v20160420.zip" --base-dir ".\\dryad_download" --out-dir ".\\data\\raw"

Notes:
 - Use forward slashes in the example above to avoid Python unicode-escape parsing issues.
 - The script unzips the archive, attempts to extract .rar files with 7z if available,
   scans for .mat/.npy/.npz/.h5 arrays, finds two best channels (i1/i2), saves them
   normalized into out-dir/raw_traces.npz and saves calibration to fbg_group_delay.npy.
"""
import os
import sys
import json
import argparse
import shutil
import subprocess
from glob import glob
import numpy as np
from scipy import io as spio

# Optional h5py
try:
    import h5py
    H5PY = True
except Exception:
    H5PY = False

SEARCH_NAMES = ["i1","i2","I1","I2","ch1","ch2","trace1","trace2","channel1","channel2","sig1","sig2"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def unzip_archive(zip_path, dest):
    import zipfile
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)
    return True

def extract_rar_with_7z(rar_path, dest):
    for exe in ("7z", "7za", "7zr"):
        try:
            subprocess.run([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cmd = [exe, "x", rar_path, f"-o{dest}", "-y"]
            subprocess.check_call(cmd)
            return True
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            return False
    return False

def find_candidate_files(base_dir):
    exts = (".mat", ".npy", ".npz", ".csv", ".h5", ".bin", ".dat")
    out = []
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    return out

def load_mat_safe(path):
    try:
        return spio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"[warn] loadmat failed for {path}: {e}")
        return {}

def load_h5_safe(path):
    out = {}
    if not H5PY:
        return {"_h5_unavailable": path}
    try:
        with h5py.File(path, "r") as f:
            def walk(g, prefix=""):
                for k in g.keys():
                    obj = g[k]
                    p = prefix + k
                    if isinstance(obj, h5py.Dataset):
                        out[p] = obj[()]
                    else:
                        walk(obj, p + "/")
            walk(f)
    except Exception as e:
        print(f"[warn] h5 load failed for {path}: {e}")
    return out

def scan_and_collect(base_dir):
    candidates = find_candidate_files(base_dir)
    print(f"[info] Found {len(candidates)} candidate files under {base_dir}")
    arrays = []
    calibration = None
    for p in candidates:
        name = os.path.basename(p)
        print(f"[scan] {name}")
        ext = os.path.splitext(name)[1].lower()
        loaded = {}
        if ext == ".mat":
            loaded = load_mat_safe(p)
        elif ext == ".h5":
            loaded = load_h5_safe(p)
        elif ext in (".npy", ".npz"):
            try:
                data = np.load(p, allow_pickle=True)
                if isinstance(data, np.lib.npyio.NpzFile):
                    for k in data.files:
                        loaded[k] = data[k]
                else:
                    loaded[os.path.basename(p)] = data
            except Exception as e:
                print(f"[warn] np load failed for {p}: {e}")
        else:
            continue

        for k, v in loaded.items():
            try:
                arr = np.asarray(v)
            except Exception:
                continue
            kl = str(k).lower()
            if any(s in kl for s in ("group","delay","gd","fbg","calib","wavelength")) and arr.size >= 2:
                calibration = {"path": p, "key": k, "array": arr}
                print(f"[found calibration] {k} in {name} shape {arr.shape}")
            if arr.ndim >= 1 and arr.size >= 512:
                arrays.append((k, p, arr))
                print(f"[found array] {k} shape {arr.shape}")
    return arrays, calibration

def pick_channels(arrays):
    matches = []
    for k,p,a in arrays:
        kl = str(k).lower()
        if any(s in kl for s in SEARCH_NAMES):
            matches.append((k,p,a))
    if len(matches) >= 2:
        return matches[0][2], matches[1][2], matches[0][1], matches[1][1]
    flat = [(k,p,a) for (k,p,a) in arrays if a.ndim >= 1]
    if not flat:
        return None, None, None, None
    flat_sorted = sorted(flat, key=lambda t: t[2].size, reverse=True)
    if len(flat_sorted) >= 2:
        return flat_sorted[0][2], flat_sorted[1][2], flat_sorted[0][1], flat_sorted[1][1]
    return flat_sorted[0][2], None, flat_sorted[0][1], None

def normalize_array(a):
    a = np.asarray(a, dtype=np.float32)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return a

def save_outputs(i1, i2, calib, out_dir, sources):
    ensure_dir(out_dir)
    raw_npz = os.path.join(out_dir, "raw_traces.npz")
    save_dict = {}
    if i1 is not None:
        save_dict["i1"] = i1
    if i2 is not None:
        save_dict["i2"] = i2
    if save_dict:
        np.savez_compressed(raw_npz, **save_dict)
        print(f"[save] raw traces -> {raw_npz}")
    if calib is not None:
        calib_path = os.path.join(out_dir, "fbg_group_delay.npy")
        np.save(calib_path, np.asarray(calib))
        print(f"[save] calibration -> {calib_path}")
    meta = {
        "sources": sources,
        "saved": {
            "raw_traces": os.path.basename(raw_npz) if save_dict else None,
            "calibration": "fbg_group_delay.npy" if calib is not None else None
        }
    }
    meta_path = os.path.join(out_dir, "measurement_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] metadata -> {meta_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=None, help="Path to the downloaded Dryad zip archive")
    parser.add_argument("--base-dir", default="dryad_download", help="Folder to extract and scan")
    parser.add_argument("--out-dir", default=os.path.join("data","raw"), help="Output folder in repo")
    args = parser.parse_args()

    ensure_dir(args.base_dir)
    if args.zip:
        print(f"[info] Unzipping {args.zip} -> {args.base_dir}")
        try:
            unzip_archive(args.zip, args.base_dir)
        except Exception as e:
            print(f"[error] unzip failed: {e}")
            sys.exit(1)

    rar_list = glob(os.path.join(args.base_dir, "**", "*.rar"), recursive=True)
    if rar_list:
        print(f"[info] Found {len(rar_list)} .rar files. Attempting extraction with 7z if available.")
        for r in rar_list:
            print(f"  extracting {r}")
            ok = extract_rar_with_7z(r, args.base_dir)
            if not ok:
                print(f"  [warn] extraction failed for {r}. Please extract manually into {args.base_dir}")

    arrays, calibration = scan_and_collect(args.base_dir)
    if not arrays:
        print("[error] No numeric arrays found. Ensure you extracted the .rar files into the folder and re-run.")
        sys.exit(2)

    i1, i2, src1, src2 = pick_channels(arrays)
    sources = {"i1_source": src1, "i2_source": src2, "calibration_source": calibration["path"] if calibration else None}
    if i1 is None:
        print("[error] Could not identify channel arrays (i1/i2). Inspect available arrays and re-run with explicit keys.")
        for k,p,a in arrays:
            print(f"  found {k} from {p} shape {a.shape}")
        sys.exit(3)

    i1n = normalize_array(i1)
    i2n = normalize_array(i2) if i2 is not None else None
    calib_arr = calibration["array"] if calibration else None

    save_outputs(i1n, i2n, calib_arr, args.out_dir, sources)
    print("[done] Import complete. Check", args.out_dir)

if __name__ == "__main__":
    main()
