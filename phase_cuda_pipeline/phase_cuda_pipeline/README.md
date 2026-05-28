# Phase Recovery CUDA Pipeline

Converted from `phase_recovery_v8_final.ipynb` into a runnable project scaffold.

This project has two paths:

1. **Python visualizer**: fast exploration of time-domain Gerchberg-Saxton phase recovery.
2. **CUDA/C++ pipeline**: production-style kernels for dispersion and magnitude constraints.

## Quick Python visualizer

```bash
cd phase_cuda_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
python python/visualizer.py --waveform gas3 --iters 250 --restarts 4
```

For live updating plots:

```bash
python python/visualizer.py --waveform chirped_nrz --iters 400 --live
```

## CUDA build

Requires NVIDIA CUDA Toolkit and cuFFT.

```bash
cd phase_cuda_pipeline
cmake -S . -B build
cmake --build build -j
./build/phase_recovery_cuda --N 16384 --iters 250 --restarts 4
```

## Algorithm

The notebook's core method is two-plane time-domain Gerchberg-Saxton:

```text
measured I1 at dispersion phi2_1
measured I2 at dispersion phi2_2

start random phase at plane 1
repeat:
    propagate plane1 -> plane2
    enforce |e2| = sqrt(I2)
    propagate plane2 -> plane1
    enforce |e1| = sqrt(I1)
    optionally apply spectral support at phi2=0
```

The CUDA version uses cuFFT for propagation and custom kernels for:

- centered frequency grid
- GVD phase screen multiplication
- amplitude replacement while preserving phase
- support projection
- normalization and residual computation
