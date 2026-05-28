# Dispersion-Assisted Optical Phase Recovery

**Carrier-less coherent fiber sensing via the time-domain Gerchberg–Saxton algorithm**

> Two intensity-only measurements at different dispersive fiber lengths are sufficient to
> reconstruct the full complex optical field — replacing the local-oscillator laser and
> 90° hybrid coupler required by conventional coherent detection.

UCLA EC ENGR 279AS · Jalali Lab · Completed Spring 2026

---

## OUSD(R&E) Technology Alignment

| OUSD Critical Area | This Project's Contribution |
|---|---|
| **FutureG** | Carrier-less coherent receivers enable 400G+ WDM links on DoD tactical fiber without bulky LO hardware |
| **Trusted AI and Autonomy** | INT8-quantized CNN classifier autonomously detects optical rogue waves; TD-GS runs 200 iterations in ~1.5 ms |
| **Advanced Computing and Software** | CUDA-accelerated TD-GS pipeline; embedded aarch64 firmware (RPi CM4 + 56 GSa/s ADC) |
| **Integrated Sensing and Cyber** | RogueGuard 1U appliance monitors fiber plant health, fires SNMP alerts on anomalous pulses |
| **Directed Energy (DE)** | Phase-retrieval diagnostics applicable to fiber-amplifier beam quality in high-power DE systems |
| **Human-Machine Interfaces** | Real-time spectral health dashboard; operator-facing alert via standard SNMP/MIB |

---

## System Architecture

```
  OPTICAL INPUT
  (fiber span)
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │                   SENSING FRONT-END                     │
  │  ┌──────────┐   D1 = -600 ps²   ┌──────────┐           │
  │  │  coupler │──────────────────▶│  PD + ADC│  I₁[n]   │
  │  └──────────┘                   └──────────┘           │
  │       │          D2 = -900 ps²   ┌──────────┐           │
  │       └────────────────────────▶│  PD + ADC│  I₂[n]   │
  │                                  └──────────┘           │
  └─────────────────────────────────────────────────────────┘
       │  I₁, I₂  (14-bit, 56 GSa/s)
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │              ROGUEGUARD FIRMWARE (aarch64)              │
  │                                                         │
  │  TD-GS  ──────────────────────────────────────────────▶ │
  │  200 iter × FFTW3f                      recovered φ(t)  │
  │  ~1.5 ms end-to-end                                     │
  │                                                         │
  │  INT8 CNN ────────────────────────────────────────────▶ │
  │  3-layer MLP on φ features              P(rogue) score  │
  │                                                         │
  │  ALERT ENGINE ──────────────────────────────────────── ▶│
  │  P(rogue) > 0.5  →  SNMP trap → NOC                    │
  └─────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
.
├── phase_retrieval.ipynb              # Main research notebook (§1–§55)
├── phase_retrieval_symbolic_c_numeric.ipynb  # SymPy cross-validation
├── notebooks/
│   └── seals_simulation.ipynb        # SEALS Mie/Rayleigh Python port
├── firmware/
│   ├── rogueguard_firmware.c         # TD-GS + CNN, DMA ISR (aarch64)
│   ├── rogueguard_firmware.h         # Memory map, data structures
│   ├── rogueguard_1u.scad            # 1U chassis 3D model
│   ├── rogueguard_circuit.cir        # Analog front-end SPICE netlist
│   ├── rogueguard_spice.py           # SPICE simulation driver
│   ├── rogueguard_paper.tex          # Companion paper (LaTeX)
│   └── rogueguard_build.py           # Build helper script
├── simulations/
│   ├── datacenter_rogue_wave.py      # Datacenter fiber rogue-wave sim
│   ├── ghost_imaging.py              # Ghost imaging variant
│   ├── optics_circuit_rl.py          # Reinforcement learning optical routing
│   └── optics_econ_decision.py       # Economic decision model for fiber mgmt
├── phase_cuda_pipeline/              # CUDA C++ accelerated TD-GS
├── Makefile
└── requirements.txt
```

---

## Notebook Coverage (`phase_retrieval.ipynb`)

76 cells · 31 focused sections — all mapped to OUSD critical technology areas.

| Section | Topic | OUSD Area |
|---|---|---|
| §1–6 | TD-GS core: forward model, synthetic signals, baseline recovery | Advanced Computing |
| §7 | PhyCV Phase-Stretch Transform (single-measurement baseline) | Advanced Computing |
| §8–9 | Convergence experiment (1–1600 iter); measurement diversity (D₂/D₁ sweep) | Advanced Computing |
| §10 | QPSK communication signal phase recovery | FutureG |
| §11 | Neural network phase regression (PyTorch) | Trusted AI & Autonomy |
| §12 | CUDA/GPU acceleration — PyTorch + custom kernels | Advanced Computing |
| §13 | Ghost imaging — spatial phase recovery via intensity correlations | Integrated Sensing |
| §15–17 | Wirtinger flow; Poisson noise robustness; L1 sparse retrieval | Advanced Computing |
| §19 | Parameter sweep — wavelength, fiber length, Fresnel noise | Advanced Computing |
| §22 | 2D phased array beamforming; HITRAN CO absorption data | Directed Energy |
| §23–24 | Deep unrolling (GS as trainable NN); Kalman phase tracker | Trusted AI & Autonomy |
| §31 | Heterodyne & coherent detection (traditional comparison) | FutureG |
| §32 | Final algorithm benchmark & research roadmap | Advanced Computing |
| §35 | Speckle statistics — fiber sensing noise floor | Integrated Sensing |
| §43 | Distributed fiber sensing — phase retrieval along 100 km | Integrated Sensing |
| §45 | LIGO — most sensitive phase measurement in history | Integrated Sensing |
| §46 | THz time-domain spectroscopy | Directed Energy |
| §47a–b | Qubit/Bra-ket primer (foundation for quantum sensing) | Quantum Science |
| §53 | SEALS — Mie + Rayleigh Python port, wavelength-to-angle mapping | Integrated Sensing |
| §54 | FutureG — carrier-less coherent architecture, SWaP analysis | FutureG |
| §55 | RogueGuard 1U — embedded deployment, ROC curves, pipeline latency | Trusted AI + HMI |

---

## Quick Start

```bash
make install   # pip install -r requirements.txt
make run       # open phase_retrieval.ipynb in Jupyter
make seals     # open SEALS simulation notebook
make execute   # run notebook top-to-bottom and save outputs
make firmware  # cross-compile firmware -> aarch64 ELF (requires aarch64-linux-gnu-gcc)
```

---

## Algorithm

**Time-Domain Gerchberg–Saxton (TD-GS):**

1. Initialize a random phase guess at dispersion plane D₁.
2. Propagate to D₂ via `H(ν) = exp(iαD ν²)` and replace magnitude with √I₂.
3. Back-propagate to D₁ and replace magnitude with √I₁.
4. Repeat until residual change < tolerance (typically 200 iterations → ~1.5 ms on CM4).

The dispersion transfer function uses `α = πλ₀²/c ≈ 2.515×10⁻⁵` at λ₀ = 1550 nm.
The kernel is LUT-cached — computed once per unique (D, α) pair.

---

## Key Results

| Metric | Value |
|---|---|
| Phase RMSE (QPSK, SNR=20 dB) | < 0.08 rad |
| Convergence (typical) | 50–80 iterations |
| Firmware pipeline latency | ~1.5 ms (RPi CM4, 200 iter, N=2048) |
| SEALS angle mapping range | ±20° over 1580–1600 nm |
| Rogue wave detection P(rogue) threshold | 0.5 (CNN, INT8 quantized) |

---

## References

1. R. W. Gerchberg and W. O. Saxton, "A practical algorithm for the determination of phase from image and diffraction plane pictures," *Optik*, 35(2):237–246, 1972.
2. B. Jalali et al., "Optical phase recovery in the dispersive Fourier transform," *Appl. Phys. Lett.*, 95, 231108, 2009.
3. T. Yao et al., "Neural network enabled time stretch spectral regression," *Optica*, 2022.
4. Adam et al., "Spectrally Encoded Angular Light Scattering," *Optics Express*, 2013.

---

## Related Work

- [`firmware/rogueguard_paper.tex`](firmware/rogueguard_paper.tex) — companion paper
- [`firmware/rogueguard_1u.scad`](firmware/rogueguard_1u.scad) — 1U chassis CAD
- [Jalali Lab PhyCV](https://github.com/JalaliLabUCLA/phycv) — Phase-Stretch Transform library
