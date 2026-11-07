# Lab desk: dispersion-assisted phase recovery

A desk-style simulator of the ECE 279AS bench: 1550 nm laser, gas cell, 1×2 splitter, two
dispersive fiber spools (D1, D2), two photodiodes, an oscilloscope and a laptop. Your hand follows the
mouse with spring lag and tremor (press T to steady it). Pick up the scope probes, clip them onto test
points, and with both probes on the photodiodes the laptop runs time-domain Gerchberg-Saxton (TDGSA)
on what the scope captured. The same capture opens in MATLAB or Jupyter.

## How to open it

**Easiest:** double-click `open_lab_desk.bat` in this folder (`D:\Summer2026\Dispersion-Assisted-GS-Phase-Recovery\projects\lab_desk`).

**From a terminal:**

1. Open PowerShell or a VS Code terminal.
2. Go to the repo:
   ```
   cd D:\Summer2026\Dispersion-Assisted-GS-Phase-Recovery
   ```
3. Start the desk:
   ```
   py -3.12 projects\lab_desk\lab_desk.py
   ```
4. A window opens. Close it or press Esc to quit.

**What it needs:** Python 3.12 with `pygame` and `numpy` (already installed on Colin's machine). The
laptop's extra buttons need a bit more:

* **Open in MATLAB:** MATLAB installed (it finds `C:\Program Files\MATLAB\R2025b` on its own) and
  `scipy` for the `.mat` file.
* **Open in Jupyter:** `nbformat`. It opens JupyterLab if installed, otherwise the default app for
  `.ipynb` files. Pick the **py312** kernel; the `circuits32` env crashes on the first plot.

If a package is missing: `py -3.12 -m pip install pygame numpy scipy nbformat`

**Opening the saved files by hand:** after you press either button, everything lands in
`projects\lab_desk\out\`. In MATLAB, `cd` into that folder and run `desk_tdgsa`. In Jupyter, open
`desk_tdgsa.ipynb` from that folder.

**Checking it without a window** (saves a screenshot instead):
```
py -3.12 projects\lab_desk\lab_desk.py --shot desk.png
```

## First five minutes

1. Move the mouse: the green hand follows it. Press **T** if the wobble gets annoying.
2. Click the yellow probe tip (CH1), move it onto the gold ring at **PD1**, and click again to clip it on.
3. Do the same with the cyan probe (CH2) on **PD2**. Both traces appear on the scope.
4. Click **Run TDGSA** on the laptop (or press Space). The red dashed spectrum should land on the black one.
5. Click a fiber spool to change its dispersion and run again.

| File | What it is |
|---|---|
| `desk_physics.py` | The bench: pulse, causal (Kramers-Kronig) gas cell, ps/nm → β₂L, detector bandwidth and noise, a step-by-step TDGSA (same algorithm as `dgs/temporal_gs.py`), MATLAB/Jupyter export |
| `lab_desk.py` | The desk UI (pygame) |
| `out/` | Generated when you press "Open in MATLAB/Jupyter": `desk_capture.mat/.npz`, `desk_tdgsa.m`, `desk_tdgsa.ipynb` |
| `../../tests/test_lab_desk_physics.py` | 13 tests |

## Things to try

* **The 2024 failure:** set the spools to 353 / 872 ps/nm (click = more, right-click = less). Both
  photodiodes are in the far field and see nearly the same trace; recovery stalls near 99 % error.
  1 / 40 ps/nm (the default) recovers the field to about 7 %.
* **Causality:** click the gas cell to switch to the non-causal model (absorption with no matching
  phase). Its impulse response puts 96 % of its energy before t = 0; the causal model puts 0.09 % there.
* **Detector bandwidth:** the scope's BANDWIDTH knob (8 GHz → ideal) low-passes the photodiode
  traces. At 8 GHz the recovery more than doubles its error: the open "detector-bandwidth model"
  thread from the 2026-10-10 handoff.
* **Noise:** the NOISE knob adds 0.1 % or 1 % detector noise.

* **THz probe:** the scope's THz STRETCH knob adds a photonic time-stretch front end (×1, ×4, ×10,
  ×30). Stretching by M multiplies the scope's effective bandwidth by M: an 8 GHz scope goes from
  97 % error to 16 % at ×30, and 33 GHz × 30 ≈ 1 THz gets 9 % (ideal linear stretch, no dispersion
  penalty).
* **Topology:** the laptop's Topology tab shows the time-frequency picture (STFT) of the light in the
  D2 fiber, true and recovered, with its phase vortices circled. Each vortex is a zero where the phase
  winds once around (all the same sign, because a Gaussian-window STFT is an analytic function in
  disguise). A good recovery puts every vortex in the right place; the 2024 settings miss all of them.
  The braid panel shows how the vortices move as D changes.

## Checked

* MATLAB R2025b (`matlab -batch`) and the notebook (py312 kernel) both report field error 0.067 on
  the default capture. The desk shows 0.074 because it starts from a chirped phase guess.
* Tests: `PYTHONPATH=. python -m pytest tests/test_lab_desk_physics.py -q`.

ABET write-up: `docs/abet_writeup_lab_desk.md`. Not built yet: a visionOS export of the desk.
