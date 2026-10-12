# TDGSA + quantum information: Python, ANSI C, C++/POSIX, CUDA

Time-domain Gerchberg-Saxton (TDGSA) phase recovery from two dispersive intensity measurements,
then the same idea on qubits. Continues the ECE 279AS project (G. Morozowsky and C. Casey, Prof. B. Jalali, UCLA, 2024),
whose stated next steps were "convert to Python, get good results, compare iterations vs accuracy".

| File | What it does |
|---|---|
| `TDGSA_Dispersion_Phase_Recovery.ipynb` | Main notebook. SymPy (dispersion operator, transport-of-intensity continuity equation), 1550 nm simulation with a Kramers-Kronig-consistent gas cell, TDGSA with random / flat / TIE initial phase, D1-D2 sweep, iterations 1 to 1e4, rogue-pulse detection, Qiskit: dispersion as a 6-qubit circuit, GS state recovery vs shots, survival probability. IEEE references. |
| `TDGSA_C_CUDA.ipynb` + `c/` | `tdgsa.c` (strict ANSI C89, own radix-2 FFT) and `tdgsa.cu` (cuFFT). Builds both, checks them against NumPy, benchmarks 1 to 1e4 iterations. |
| `qis/` | `qgs.cpp` (C++17, `std::cout`, POSIX threads) and `qgs.cu` (cuRAND + batched cuFFT): 6-qubit GS state recovery over thousands of trials. `QIS_Dispersion_Statistics.ipynb`: plain-English + math + statistics (bootstrap CIs, power-law fit, KS tests). |

Related, in the main repo: `dgs/qiskit_phase_flip.py` + `notebooks/qiskit_phase_flip.ipynb`
(unknown phase = dephasing, corrected by the 3-qubit phase-flip code).

## Running

* Kernel: a Python 3.12 environment with numpy, scipy, sympy, matplotlib, pandas, qiskit (on Colin's machine: the `py312` Jupyter kernel).
* `c\build.bat` and `qis\build.bat`: gcc/g++ (MSYS2) for C/C++, then `vcvars64.bat` (MSVC 2022) + `nvcc -arch=sm_89` (RTX 4060, CUDA 13.0). Change `-arch` for other GPUs.
* Executables and `out/` are build products (`*.exe` is gitignored).

## Key results (from the saved outputs)

* **Why the 2024 MATLAB run stalled:** with D1/D2 = 30/90 ps/nm both photodiodes sit deep in the far field and record nearly the same |spectrum|. Use D1 = 0 and choose D2 for the feature: ~0.05 ps/nm for a chirped 0.3 ps pulse, ~10 ps/nm for GHz-wide gas lines.
* **Random initial phase stalls; a transport-of-intensity (TIE) start converges:** field error 1.26 → 0.058 (chirped), 0.56 → 0.042 (cubic). Gas cell absorption + phase recovered to about 2%.
* **Noise is the open problem:** 0.1% photodiode noise → field error 0.56. Ideal detectors assumed (no bandwidth limit).
* **C / CUDA / NumPy agree** to 5e-7 / 3e-8 relative; GPU 6 to 22× faster than the C version at N = 2^15.
* **Qubits:** fidelity 0.14 / 0.54 / 0.93 / 0.988 / 0.998 / 0.99987 at 1 / 10 / 100 / 1e3 / 1e4 / 1e5 shots; infidelity ∝ S^-α with α = 0.90 ± 0.29; within ~25% of a shot-noise prediction using the participation ratio (d_eff ≈ 7 and 13).

Reference details (volumes, DOIs) were written from memory; verify before submitting anything.
