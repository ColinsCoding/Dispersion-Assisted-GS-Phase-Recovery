# Dispersion-Assisted GS Phase Recovery

Intensity-only phase retrieval for carrier-less coherent fiber sensing.
Two dispersed measurements. No local oscillator. No 90° hybrid.

**OUSD(R&E)** — FutureG · Trusted AI · Advanced Computing · Integrated Sensing · Directed Energy · Quantum Science · HMI

---

```
 I₁[n] ── D₁ = −600 ps²  ──┐
                             ├── TD-GS (200 iter) ──► φ(t)  ──► CNN ──► SNMP alert
 I₂[n] ── D₂ = −900 ps²  ──┘

 H(ν) = exp(iαDν²)     α = πλ₀²/c ≈ 2.515×10⁻⁵     λ₀ = 1550 nm
```

---

```
make install   # pip install -r requirements.txt
make run       # jupyter notebook phase_retrieval.ipynb
make seals     # jupyter notebook notebooks/seals_simulation.ipynb
make execute   # run all cells in place
make firmware  # cross-compile → firmware/rogueguard_firmware.elf  (aarch64-linux-gnu-gcc)
make push      # git push origin main
```

---

## Notebook · `phase_retrieval.ipynb` · 76 cells

| § | Topic | OUSD Area |
|---|---|---|
| 1–6 | Forward model · synthetic signals · TD-GS algorithm · baseline recovery | Advanced Computing |
| 7 | PhyCV Phase-Stretch Transform — single-measurement baseline | Advanced Computing |
| 8–9 | Convergence (1–1600 iter) · measurement diversity D₂/D₁ sweep | Advanced Computing |
| 10 | QPSK phase recovery — FutureG communication waveform | FutureG |
| 11 | Neural network phase regression (PyTorch) | Trusted AI & Autonomy |
| 12 | CUDA/GPU acceleration — elementwise complex kernel | Advanced Computing |
| 13 | Ghost imaging — spatial phase via intensity correlations | Integrated Sensing |
| 15–17 | Wirtinger flow · Poisson noise · L1 sparse retrieval | Advanced Computing |
| 19 | Parameter sweep — wavelength, fiber length, Fresnel noise | Advanced Computing |
| 22 | 2D phased array beamforming · HITRAN CO absorption | Directed Energy |
| 23–24 | Deep unrolling (GS as trainable NN) · Kalman phase tracker | Trusted AI & Autonomy |
| 31 | Heterodyne & coherent detection — traditional receiver comparison | FutureG |
| 32 | Algorithm benchmark · research roadmap | Advanced Computing |
| 35 | Speckle statistics — fiber sensing noise floor | Integrated Sensing |
| 43 | Distributed fiber sensing — 100 km phase retrieval | Integrated Sensing |
| 45 | LIGO — most sensitive phase measurement in history | Integrated Sensing |
| 46 | THz time-domain spectroscopy | Directed Energy |
| 47a–b | Qubit · Bra-ket algebra (quantum sensing foundation) | Quantum Science |
| 53 | SEALS — Mie + Rayleigh Python port, wavelength-to-angle | Integrated Sensing |
| 54 | FutureG — carrier-less architecture · SWaP analysis | FutureG |
| 55 | RogueGuard 1U — embedded deployment · ROC · pipeline latency | Trusted AI · HMI |

## SEALS · `notebooks/seals_simulation.ipynb` · 10 cells

Python port of Jalali Lab MATLAB simulation (Project 1 deliverable).
Lorenz-Mie (exact) + Rayleigh-Debye (approximate) + SEALS wavelength-to-angle mapping.

---

## Hardware — RogueGuard 1U

```
RPi CM4 · 4× ARM Cortex-A72 @ 1.5 GHz · 4 GB RAM
Dual 14-bit ADC · 56 GSa/s · D₁ = −600 ps²  D₂ = −900 ps²
TD-GS (FFTW3f + NEON SIMD) ~1.5 ms · INT8 CNN (NNPACK) ~0.5 ms · Total < 3.5 ms
```

Source: `firmware/` — C11 · `make firmware` → aarch64 ELF

---

## References

```
[1] Gerchberg & Saxton, Optik 35(2):237–246, 1972
    "A practical algorithm for the determination of phase from image and diffraction plane pictures"
    references/gs1972.pdf

[2] Jalali et al., Appl. Phys. Lett. 95, 231108, 2009
    "Optical phase recovery in the dispersive Fourier transform"
    references/jalali2009.pdf

[3] Yao et al., Optica, 2022
    "Neural network enabled time stretch spectral regression"
    references/yao2022.pdf

[4] Adam et al., Opt. Express 21(4), 2013
    "Spectrally Encoded Angular Light Scattering"
```
