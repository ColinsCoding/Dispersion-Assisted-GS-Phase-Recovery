# 90° optical hybrid

The I/Q mixer at the front of a coherent optical receiver. It combines the incoming
**signal** field with a **local oscillator** (LO) and produces four outputs at
relative phases 0°, 90°, 180°, 270°. Balanced photodetectors on the (0°,180°) and
(90°,270°) pairs then recover the **in-phase (I)** and **quadrature (Q)** components
— i.e. the full complex optical field.

## Files
- **`optical_hybrid_90deg.ipynb`** — teaching notebook: the transmission matrix,
  balanced detection (`D1 = 4cosφ`), a proper hybrid's (I,Q) constellation circle,
  and how phase imbalance degrades the readout.
- **`optical_hybrid_90deg.py`** — the hybrid model (cleaned port). `optical_hybrid_90deg()`
  returns the four complex outputs with optional insertion loss and phase / loss
  imbalance; `balanced_detection()` gives the two photodiode-pair difference signals.
- **`test_optical_hybrid.py`** — verifies the ideal transmission matrix, the 4×
  output power, the balanced-detection difference signals, and that sweeping the
  signal phase makes `D1` track the in-phase projection `4·cos(φ)`.
- **`matlab/`** — the original `optical_hybrid_90deg.m`, `save_optical_hybrid_outputs.m`.

## Model
Ideal transmission matrix (signal `s`, LO `l`):
```
[out_0  ]   [ 1   1 ] [s]
[out_90 ] = [ 1  -1 ] [l]
[out_180]   [ 1j  1j]
[out_270]   [ 1j -1j]
```
with diagonal phase-imbalance and insertion-loss-imbalance matrices applied for a
realistic device.

**Note on quadrature:** in this faithfully-ported matrix the `1j` on rows 3-4 is a
*global* phase, so both balanced-detector pairs yield the in-phase term
`4·Re(s·l*)` and there is no independent quadrature output. A textbook 90° hybrid
places the 90° shift on the **LO arm of the second pair** (`[1, 1j]`, `[1, −1j]`)
so the second difference gives `4·Im(s·l*)`. The function is left as the lab wrote
it; `balanced_detection()` reports what this matrix actually produces.

## Note on the source port
The original `python/optical_hybrid_90deg.py` ran an example at import time and
wrote to a hardcoded macOS path (`/Users/colincasey/...`). This version moves the
demo under `if __name__ == "__main__"` so importing the module is side-effect free.

## Run
```bash
py -3.13 projects/optical_hybrid_90deg/optical_hybrid_90deg.py   # numpy only
py -3.13 projects/optical_hybrid_90deg/test_optical_hybrid.py
```

## Connection to this repo
The hybrid recovers the complex field by mixing with a local oscillator. The
dispersion-assisted GS receiver recovers the *same* complex field **without** a
hybrid or LO — from intensity-only measurements through dispersive diversity.
