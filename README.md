# Dispersion-Assisted GS Phase Recovery

ECE 279AS · UCLA · Winter 2024 · Jalali Lab

Carrier-less optical phase retrieval from two intensity-only measurements.
No local oscillator. No 90° hybrid. Just fiber, a photodetector, and math.

```
 I₁(t)  ──  D₁ = −695 ps/nm  ──┐
                                  ├──  Temporal GS  ──►  φ(t)
 I₂(t)  ──  D₂ = −800 ps/nm  ──┘

 H(ν) = exp(i π D ν²)      transfer function of dispersive fiber
```

---

## How it works

Light from a pulsed source passes through a gas cell, then splits into two paths
with different dispersive fibers (D₁, D₂). Each path maps spectrum → time
(dispersive Fourier transform), and a photodetector records the intensity.

You see brightness. Not phase. The GS algorithm recovers the missing phase by
alternating between two constraints — one per measurement arm — until both are
simultaneously satisfied.

**Grade 7 version:** run a jump rope through two different springs and film both.
From the videos alone, GS figures out how the rope was twisted.

---

## Quick start

```bash
pip install -r requirements.txt
python gs_core.py          # runs self-test, saves gs_core_test.png
```

To open the main notebook:

```bash
jupyter notebook phase_retrieval.ipynb
```

---

## Files

| File | What it is |
|---|---|
| [`gs_core.py`](gs_core.py) | Physics engine — dispersion operator, GS loop, QPSK test data generator |
| [`phase_retrieval.ipynb`](phase_retrieval.ipynb) | Main Colab notebook (project deliverable) |
| [`optical_dashboard/dsp.py`](optical_dashboard/dsp.py) | QPSK modem, DWDM simulation, digital logic |
| [`pic_design/sim/gs_surface.py`](pic_design/sim/gs_surface.py) | GS error surface 3D plot |
| [`pic_design/sim/gs_animate.py`](pic_design/sim/gs_animate.py) | Convergence animation |
| [`pic_design/sim/fdtd_dispersion.py`](pic_design/sim/fdtd_dispersion.py) | FDTD dispersion model |
| [`notebooks/seals_simulation.ipynb`](notebooks/seals_simulation.ipynb) | SEALS Mie/Rayleigh scattering (Project 1) |
| [`sample_data/`](sample_data/) | Synthetic two-arm `.mat` files for testing |
| [`references/README.md`](references/README.md) | Paper citations with DOI links |

---

## Core algorithm (`gs_core.py`)

```python
from gs_core import retrieve_phase, make_qpsk_measurements

data = make_qpsk_measurements(n_symbols=128, D1=-695.0, D2=-800.0, snr_db=30.0)
phi, errors = retrieve_phase(data["I1"], data["I2"], data["D1"], data["D2"], n_iter=20)
```

The transfer function, derived with SymPy:

```python
from gs_core import show_transfer_function
H, latex_str = show_transfer_function()
# H(ν) = exp(i π D ν²)
```

Convergence follows Fig. 3 of Solli et al. 2009 — phase and amplitude errors
decrease monotonically, reaching a noise floor around iteration 15–20.

---

## References

[1] Gerchberg & Saxton, *Optik* 35(2):237–246, 1972 — original GS algorithm  
[2] Solli, Gupta & Jalali, *Appl. Phys. Lett.* 95, 231108, 2009 — time-domain GS in the dispersive Fourier transform  
[3] Yao et al., *Optica*, 2022 — neural network time-stretch spectral regression  

DOI links: [`references/README.md`](references/README.md)

---

## Acknowledgment

The dispersive Fourier transform and time-domain GS phase retrieval concept
originates from the work of **Prof. Bahram Jalali** and his group at UCLA.
This repository is an independent student implementation for ECE 279AS.
It does not represent the lab's official code or results.
Errors are my own.

Contact for the project: Yiming Zhou and Callen MacPhee (Jalali Lab, UCLA).
