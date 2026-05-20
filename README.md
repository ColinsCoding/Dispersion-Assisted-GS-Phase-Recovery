# Dispersion-Assisted GS Phase Recovery

> Can two intensity-only measurements after known dispersive propagation recover
> the missing optical phase well enough to reduce or replace a local oscillator?

---

## Background

Coherent optical receivers require a local oscillator (LO) to access the full
complex field E(t) = |E(t)| e^{iφ(t)}.  The LO adds hardware cost, stability
constraints, and phase-noise coupling.  This project tests whether **two
dispersed intensity measurements** — each applying a different group-velocity
dispersion (GVD) — contain enough information to recover φ(t) algebraically.

The dispersive Fourier transform (time-stretch) maps spectrum to time:

```
E(ω, L) = E(ω, 0) · exp(−i β₂L ω² / 2)
```

Two fiber channels with D₁ = −300 ps/nm and D₂ = −1200 ps/nm (ratio 4×,
above the Solli 2009 minimum of 1.33×) provide the diversity needed for
phase retrieval.

---

## Algorithm: TD-GS

Time-Domain Gerchberg-Saxton (Solli 2009):

1. Initialise u with random phase, modulus = back-propagated √I₁
2. Forward disperse → replace modulus with √I₁ → back-disperse
3. Forward disperse → replace modulus with √I₂ → back-disperse
4. Repeat steps 2–3 for `n_iter` iterations
5. Run `n_restarts` independent restarts; keep lowest residual

Current performance (N = 2¹⁴, dt = 0.61 ps, 8 restarts × 250 iter):

| Method              | RMSE    | Time/iter  |
|---------------------|---------|-----------|
| NumPy TD-GS         | 0.1232  | 6 850 µs  |
| PyTorch TD-GS (CPU) | 0.1343  | 2 440 µs  |
| PyTorch batch B=32  | —       | 375 µs    |
| KK closed-form      | ~0.12   | 1 FFT     |

---

## Repository layout

```
simulator/
    __init__.py          — public API
    dispersion.py        — propagate(), batch_propagate(), transfer_function()
    gs.py                — td_gs()  (full TD-GS with restarts + RMSE tracking)
    kramers_kronig.py    — kk_recover(), kk_seed_gs()

prototypes/
    wirtinger_flow.py    — Adam-based Wirtinger-flow (dispersion-physics loss)
    wirtinger_flow.py    — GS vs Wirtinger benchmark

examples/
    quantum_wavepacket_demo.py   — Schrödinger wavepacket via dispersion module

tests/
    test_dispersion.py   — pytest suite (round-trip, Parseval, chirp width, batch)

phase_recovery_v10_publishable.ipynb   — main publishable notebook (ECE 279AS)
```

---

## Quick start

```bash
# Install dependencies
pip install numpy scipy matplotlib torch

# Run the pytest suite
pytest tests/ -v

# Run Wirtinger-flow phase retrieval (two-channel dispersion loss)
python prototypes/wirtinger_flow.py --N 512 --iters 2000 --beta1 -1e-22 --beta2 -4e-22

# Run quantum wavepacket demo (Schrödinger ↔ GVD analogy)
python examples/quantum_wavepacket_demo.py
```

---

## Simulator API

```python
from simulator import propagate, batch_propagate, transfer_function, td_gs, kk_recover

import numpy as np

N, dt = 4096, 1e-12
t = np.arange(N) * dt
u = np.exp(-((t - t.mean()) / 50e-12)**2).astype(complex)

# Single propagation
u_out = propagate(u, t, beta2_L=1e-22)

# Sweep 16 dispersion values simultaneously
betas = np.linspace(-2e-22, 2e-22, 16)
batch = batch_propagate(u, t, betas)   # shape (16, N)

# Transfer function (for manual spectrum manipulation)
H = transfer_function(N, dt, beta2_L=1e-22)   # shape (N,), |H|=1

# TD-GS phase retrieval
I1 = np.abs(propagate(u, t, -1e-22))**2
I2 = np.abs(propagate(u, t, -4e-22))**2
result = td_gs(I1, I2, t, -1e-22, -4e-22, n_restarts=8, n_iter=250, u_true=u)
print(f"RMSE = {result['rmse']:.4f}")

# Kramers-Kronig closed-form recovery
I_omega = np.abs(np.fft.fft(u))**2
u_kk = kk_recover(I_omega)   # time-domain field, minimum-phase signals only
```

---

## Physical constants (fiber context)

| Parameter | Value |
|-----------|-------|
| D₁ (dispersion) | −300 ps/nm |
| D₂ (dispersion) | −1200 ps/nm |
| β₂L₁ | −38.3 × 10⁻²⁴ s²/rad |
| β₂L₂ | −153 × 10⁻²⁴ s²/rad |
| Grid N | 2¹⁴ = 16 384 |
| dt | 0.61 ps |
| Window T | ≈ 10 ns |
| df | 0.1 GHz |

---

## References

- Solli, D.R. et al. (2008). "Optical rogue waves." *Nature* 450, 1054.
- Solli, D.R. et al. (2009). "Demonstration of optical fiber oscilloscope." *PRL*.
- Gerchberg, R.W. & Saxton, W.O. (1972). *Optik* 35, 237.
- Kramers, H.A. (1927); Kronig, R.d.L. (1926). Hilbert-transform phase relations.
- Dorrer, C. et al. (2003). "Spectral resolution and sampling issues in
  Fourier-transform spectral interferometry." *J. Opt. Soc. Am. B* 20, 1262.
- Candes, E.J. et al. (2015). "Phase retrieval via Wirtinger flow." *IEEE TIT*.
