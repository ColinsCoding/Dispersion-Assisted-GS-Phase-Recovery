#!/usr/bin/env python3
"""
tools/diag_fft.py
Load pipeline_input.npz (or raw_traces.npz), coerce i1, show dtypes and first samples,
construct GS initial field f and attempt a small rfft to reproduce the error.
Prints detailed diagnostics and exits.
"""
from pathlib import Path
import numpy as np, json, sys

candidates = [Path("data/raw/pipeline_input.npz"), Path("data/raw/raw_traces.npz"), Path("data/raw/pipeline_input.mat")]
found = next((p for p in candidates if p.exists()), None)
if found is None:
    print("No input file found in data/raw. Aborting."); sys.exit(1)

print("Using:", found)
data = {}
if found.suffix.lower() == ".npz":
    d = np.load(str(found), allow_pickle=True)
    for k in d.files:
        data[k] = d[k]
else:
    try:
        from scipy import io as spio
        m = spio.loadmat(str(found))
        for k in ("i1","i2","time","group_delay"):
            if k in m: data[k] = m[k]
    except Exception as e:
        print("Cannot read .mat:", e); sys.exit(1)

def pick(d, keys):
    for k in keys:
        if k in d: return d[k]
    return None

raw_i1 = pick(data, ["i1","I1","trace1","i_1"])
print("raw_i1 type:", type(raw_i1))
if raw_i1 is None:
    print("i1 not found in file keys:", list(data.keys())); sys.exit(1)

# show numpy array info if possible
try:
    arr = np.asarray(raw_i1)
    print("raw_i1 ndarray dtype:", arr.dtype, "ndim:", arr.ndim, "shape:", arr.shape)
    # show first 20 elements and their dtypes
    sample = arr.ravel()[:20]
    print("raw_i1 sample (first 20):", sample.tolist())
    print("element types in sample:", [type(x).__name__ for x in sample.tolist()])
except Exception as e:
    print("Could not convert raw_i1 to ndarray:", e)
    sys.exit(1)

# coerce to numeric float64 robustly
def to_numeric(a):
    try:
        return np.asarray(a, dtype=np.float64).ravel()
    except Exception:
        try:
            return np.array([float(x) for x in np.asarray(a).ravel()], dtype=np.float64)
        except Exception as e:
            print("Failed coercion to float64:", e); return None

i1 = to_numeric(arr)
print("coerced i1 dtype:", None if i1 is None else i1.dtype, "shape:", None if i1 is None else i1.shape)
if i1 is None:
    print("i1 coercion failed; printing repr of raw_i1[:10]")
    print(repr(raw_i1)[:1000])
    sys.exit(1)

# build GS initial field f
N = i1.size
phi = np.random.uniform(-np.pi, np.pi, size=N).astype(np.float64)
f = (i1 * np.exp(1j * phi)).astype(np.complex128)
print("constructed f dtype:", f.dtype, "shape:", f.shape)
print("f sample (first 10):", f[:10].tolist())
print("element types in f sample:", [type(x).__name__ for x in f[:10].tolist()])

# attempt rfft with small nfft and full nfft
for nfft in (min(1024, max(1, 1<< (N-1).bit_length())), 256, 128, N):
    try:
        print("\\nTrying np.fft.rfft with nfft =", nfft)
        R = np.fft.rfft(f, n=nfft)
        print("rfft succeeded: output dtype", R.dtype, "shape", R.shape)
    except Exception as e:
        print("rfft failed for nfft", nfft, "->", repr(e))

# quick sanity test: run rfft on a known numeric array
try:
    test = np.arange(16, dtype=np.float64)
    print("\\nSanity rfft on np.arange(16):", np.fft.rfft(test).dtype, np.fft.rfft(test).shape)
except Exception as e:
    print("Sanity rfft failed:", e)

print("\\nDiagnostics complete.")
