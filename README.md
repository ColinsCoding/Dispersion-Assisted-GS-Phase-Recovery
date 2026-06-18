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

## OUSD(R&E) Critical Technology Area alignment

This project maps onto **six** of the OUSD(R&E) Critical Technology Areas (the
priority-1 set, ★★). Tagging is generated programmatically — run
`python ousd_alignment.py` for the live table and JSON stamp.

| ★ | Critical Technology Area | How this repo contributes | Evidence |
|---|---|---|---|
| ★★ | **FutureG** | Optical-bandwidth sensing/comms via the dispersive Fourier transform; carrier-less phase recovery | [2] |
| ★★ | **Trusted AI and Autonomy** | Physics-grounded ML — wrapped-phase loss, FNO with a known analytic kernel, `gs_verify` checks | [3] |
| ★★ | **Advanced Computing and Software** | GPU phase retrieval (`gs_torch`), resolution-invariant FNO, SymPy analytic validation — Maxwell → the dispersion operator `H(f)=exp(iπDf²)` (`griffiths/electrodynamics.py`) | [1][3] |
| ★★ | **Integrated Sensing and Cyber** | Single-shot DFT spectroscopy + rogue-wave telemetry (`gs_monitor`, `gs_backtest`) | [2] |
| ★★ | **Directed Energy** | High-rep-rate pulsed-laser characterization and wavefront sensing from intensity only | [1][2] |
| ★★ | **Human-Machine Interfaces** | Real-time optical dashboard, 3-D phase-surface visualization, scanner control | — |
| ★ | Quantum Science · Microelectronics · Biotechnology | adjacent areas the repo touches (priority-2) | — |

Source of truth: the OUSD(R&E) Critical Technology Areas list (priority-1 = the
six arrowed items targeted here). The registry and component→CTA reverse map live
in [`ousd_alignment.py`](ousd_alignment.py); `stamp()` attaches CTA metadata to any
run's stats block for traceability.

> **Scope / honesty note.** This is a public UCLA / Jalali-Lab academic project
> (the GitHub repo is itself a course deliverable), marked **UNCLASSIFIED //
> DISTRIBUTION A — Approved for Public Release**. The CTA tags describe
> *technology-area relevance* — not DoD funding, endorsement, or controlled data.
> The Directed Energy link is **diagnostic only** (characterizing a beam from
> intensity measurements), consistent with the project's civilian optical-metrology
> scope — not a weapon or directed-energy system.

---

## References

[1] Gerchberg & Saxton, *Optik* 35(2):237–246, 1972 — original GS algorithm  
[2] Solli, Gupta & Jalali, *Appl. Phys. Lett.* 95, 231108, 2009 — time-domain GS in the dispersive Fourier transform  
[3] Yao et al., *Optica*, 2022 — neural network time-stretch spectral regression  
[4] BioXFEL, “Phase Retrieval for Coherent Diffractive Imaging: Theory and Algorithm,” YouTube, May 31, 2026, 1h 8m 45s — tutorial lecture on phase retrieval methods for coherent diffractive imaging
DOI links: [`references/README.md`](references/README.md)

---

## Acknowledgment

The dispersive Fourier transform and time-domain GS phase retrieval concept
originates from the work of **Prof. Bahram Jalali** and his group at UCLA.
This repository is an independent student implementation for ECE 279AS.
It does not represent the lab's official code or results.
Errors are my own.

Contact for the project: Yiming Zhou and Callen MacPhee (Jalali Lab, UCLA).
