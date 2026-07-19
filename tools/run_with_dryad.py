#!/usr/bin/env python3
"""
tools/run_with_dryad.py
Loads data/raw/raw_traces.npz and fbg_group_delay.npy, writes pipeline_input.mat/.npz,
and attempts to call a common pipeline entrypoint (best-effort).
"""
import os, sys, json
import numpy as np
from scipy import io as spio

INPUT_DIR = os.path.join("data","raw")   # change if you kept files on C:

npz_path = os.path.join(INPUT_DIR, "raw_traces.npz")
calib_path = os.path.join(INPUT_DIR, "fbg_group_delay.npy")
meta_path = os.path.join(INPUT_DIR, "measurement_metadata.json")

if not os.path.exists(npz_path):
    print("ERROR: raw_traces.npz not found at", npz_path)
    sys.exit(2)

print("Loading", npz_path)
d = np.load(npz_path, allow_pickle=True)
print("Found keys:", d.files)

def pick_channels(d):
    keys = [k for k in d.files]
    for pair in (("i1","i2"),("I1","I2"),("ch1","ch2")):
        if pair[0] in d and pair[1] in d:
            return np.asarray(d[pair[0]]), np.asarray(d[pair[1]])
    arrs = [np.asarray(d[k]) for k in keys]
    if len(arrs) >= 2:
        return arrs[0], arrs[1]
    return arrs[0], None

i1, i2 = pick_channels(d)
print("i1 shape:", None if i1 is None else i1.shape)
print("i2 shape:", None if i2 is None else (None if i2 is None else i2.shape))

calib = None
if os.path.exists(calib_path):
    calib = np.load(calib_path)
    print("Loaded calibration shape:", calib.shape)
else:
    print("No calibration file found at", calib_path)

n = i1.size if i1 is not None else (i2.size if i2 is not None else 0)
time = np.arange(n)

out_mat = os.path.join(INPUT_DIR, "pipeline_input.mat")
out_npz = os.path.join(INPUT_DIR, "pipeline_input.npz")
matdict = {"i1": i1}
if i2 is not None: matdict["i2"] = i2
matdict["time"] = time
if calib is not None: matdict["group_delay"] = calib
spio.savemat(out_mat, matdict)
np.savez_compressed(out_npz, **matdict)
print("Wrote:", out_mat, out_npz)

# Print metadata if present
if os.path.exists(meta_path):
    with open(meta_path) as f:
        try:
            meta = json.load(f)
            print("Measurement metadata keys:", list(meta.keys()))
        except Exception:
            print("Measurement metadata present but could not parse JSON.")
