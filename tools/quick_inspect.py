#!/usr/bin/env python3
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)
candidates = [Path("data/raw/pipeline_input.npz"), Path("data/raw/pipeline_input.mat"), Path("data/raw/raw_traces.npz")]
found = next((p for p in candidates if p.exists()), None)
if found is None:
    print("No pipeline_input or raw_traces found in data/raw"); sys.exit(1)

data = {}
if found.suffix.lower() == ".npz":
    d = np.load(str(found), allow_pickle=True)
    for k in d.files: data[k] = np.asarray(d[k])
else:
    try:
        from scipy import io as spio
        m = spio.loadmat(str(found))
        for k in ("i1","i2","time","group_delay"):
            if k in m: data[k] = np.asarray(m[k]).squeeze()
    except Exception as e:
        print("Cannot read .mat:", e); sys.exit(1)

# optional extras
fbg = Path("data/raw/fbg_group_delay.npy")
if fbg.exists(): data["fbg_group_delay"] = np.load(str(fbg))
meta = Path("data/raw/measurement_metadata.json")
if meta.exists():
    try: data["_metadata"] = json.loads(meta.read_text())
    except Exception: data["_metadata"] = None

def to_1d_numeric(x):
    if x is None: return None
    a = np.asarray(x).ravel()
    try: return a.astype(np.float64)
    except Exception:
        try: return np.array([float(v) for v in a], dtype=np.float64)
        except Exception: return None

def pick_first(d, keys):
    for k in keys:
        if k in d and d[k] is not None: return d[k]
    return None

i1 = to_1d_numeric(pick_first(data, ["i1","I1","trace1"]))
i2 = to_1d_numeric(pick_first(data, ["i2","I2","trace2"]))
gd = to_1d_numeric(pick_first(data, ["group_delay","fbg_group_delay"]))

summary = {}
if i1 is not None:
    summary["i1"] = {"shape": i1.shape, "min": float(i1.min()), "max": float(i1.max()), "mean": float(i1.mean()), "std": float(i1.std())}
else: summary["i1"] = None
if i2 is not None:
    summary["i2"] = {"shape": i2.shape, "min": float(i2.min()), "max": float(i2.max()), "mean": float(i2.mean()), "std": float(i2.std())}
else: summary["i2"] = None
summary["group_delay"] = {"shape": gd.shape} if gd is not None else None

def snr_proxy(x):
    x = np.asarray(x); med = np.median(x); noise = x - med
    noise_rms = (np.mean((noise - np.median(noise))**2))**0.5
    sig_rms = (np.mean(x**2))**0.5
    return float(sig_rms / (noise_rms + 1e-20))

if i1 is not None: summary["snr_i1"] = snr_proxy(i1)
if i2 is not None: summary["snr_i2"] = snr_proxy(i2)

with open(OUT / "inspect_summary.json", "w") as f: json.dump(summary, f, indent=2)
print("Wrote figures/inspect_summary.json")

if i1 is not None:
    plt.figure(figsize=(10,3)); plt.plot(i1, lw=0.6); plt.title("i1 time trace"); plt.xlabel("Sample"); plt.ylabel("Amplitude"); plt.tight_layout()
    plt.savefig(OUT / "inspect_i1_time.png", dpi=150)
    print("Wrote figures/inspect_i1_time.png")
print("Done.")
