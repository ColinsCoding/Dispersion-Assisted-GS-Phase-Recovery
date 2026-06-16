# Dispersion-Assisted GS Phase Recovery — 6-Step Roadmap

The carrier-less optical phase-recovery pipeline, from algorithm to hardware.
Status of each step and where it lives in the repo.

| # | Step | Status | Files |
|---|------|--------|-------|
| 1 | **Implement Gerchberg-Saxton** | ✅ done | `gs_core.py` (`retrieve_phase`, `disperse`), `dispersion_gs_prototype.py` (before/after spec), `notebooks/td_gs_phase_recovery_demo.ipynb` |
| 2 | **Add CUDA acceleration** | ✅ done | `gs_cuda.py` (`gerchberg_saxton_cuda`) — bit-identical to CPU, **6.0× faster** on GPU (250 ms → 42 ms, N=8192, 120 iters); `scripts/smoke_gs_cuda.py` |
| 3 | **PyTorch phase estimator** | ✅ done | `dispersion_gs_descent.py` (`torch_phase_retrieval`) — differentiable forward model + Adam + smoothness prior; beats GS on the ill-posed case (1.64 → 0.50 rad); `notebooks/dispersion_gs_gradient_descent.ipynb` |
| 4 | **Visualize propagation in 3D** | ✅ done | `notebooks/dispersion_gs_gallery.ipynb` (3D field helix, dispersion-stretch surface), `gallery2` (Wigner, diversity map), `gallery3` (animated GS-convergence GIF) |
| 5 | **Pygame / Blender viewer** | ✅ script (needs `pip install pygame`) | `viewer_pygame.py` — live: sweep D2, watch I1/I2 + recovered-vs-true phase update |
| 6 | **Verilog FFT core (hardware)** | ✅ done & simulated | `hardware/fft8.v` (8-pt radix-2 DIT, fixed-point) + `hardware/fft8_tb.v` — verified vs numpy (impulse → flat, cosine → spikes at k=1,7) with `iverilog` |

## The pipeline

```
fibre  ->  photodiode + amp (RC bandwidth, Ch.7)  ->  ADC (bit-depth)  ->
   I1, I2  ->  [ Gerchberg-Saxton  |  torch gradient descent  |  CUDA GS ]  ->
   recovered phase phi(t)  ->  3D viz / Pygame viewer
                                  hardware path:  FFT core (fft8.v) on FPGA
```

## Quick start

```bash
# CPU Gerchberg-Saxton + synthetic data + plots
py -3.13 dispersion_gs_prototype.py

# GPU-accelerated GS (needs torch / py 3.12 here)
py -3.12 scripts/smoke_gs_cuda.py

# Verilog FFT core: compile + simulate + verify
cd hardware && iverilog -g2012 -o fft8.vvp fft8.v fft8_tb.v && vvp fft8.vvp

# Interactive viewer
pip install pygame && py -3.13 viewer_pygame.py
```

## Engineering notes (the limits that bite)

- **Diversity (|D|).** GS needs `corr(I1, I2)` low — too little dispersion and the
  two intensities are redundant (unsolvable); see the diversity map in `gallery2`.
- **ADC bit-depth** and the analog **RC bandwidth** each cap the recovered phase
  (`notebooks/phase_binary_quantization.ipynb`, `notebooks/griffiths_ch7_to_receiver.ipynb`).
- **Ill-posed cases**: gradient descent with a smoothness prior recovers where
  plain GS stalls (step 3).
- **Civilian scope**: optical-fibre metrology, silicon photonics, detector modeling,
  and education. Not a weapon or directed-energy system.
