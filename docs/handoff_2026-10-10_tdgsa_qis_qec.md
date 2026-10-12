# Handoff, 2026-10-10: TDGSA in Python/C/CUDA, qubit GS, and the Qiskit phase-flip code

Work done in a Claude Code session that started in `C:\` and moved here. Everything it produced
that belongs to this project now lives in this repo. Nothing was committed (the working tree
already had ~325 uncommitted changes; commit selectively).

## What was added to this repo

| Path | Status |
|---|---|
| `projects/tdgsa_qis/` | new: 3 notebooks (executed, outputs saved), `c/` ANSI C + CUDA, `qis/` C++/POSIX + CUDA, README |
| `dgs/qiskit_phase_flip.py` | new module: phase-flip code in Qiskit (exact density matrix, deferred-measurement correction, coherent error, Wilson CIs) |
| `tests/test_qiskit_phase_flip.py` | new: 6 tests, all pass together with `tests/test_quantum_error_correction.py` (18 total) |
| `scripts/build_qiskit_phase_flip_nb.py` → `notebooks/qiskit_phase_flip.ipynb` | new: builder + executed notebook, repo conventions (Okabe-Ito, text figure descriptions) |
| `CLAUDE.md` | new: repo guidance for future Claude sessions |

The originals of `projects/tdgsa_qis/` are still in `D:\Spring2026\MATLAB\TDGSA_Python\` (absolute paths there; the copies here use relative paths).

## Thread of the work

1. Chapter 39 (Knight, 5e) reading-quiz notebooks in SymPy (in `C:\Users\mrjel\Documents`, not part of this repo).
2. Fixed `Downloads\fractional_time_stretch_prototype.ipynb` (RF aliasing past Nyquist, GPU-native interpolation, signed FFT peaks).
3. PhyCV levels 0 to 2 (`Documents\phycv_levels`): PST/VEViD, PST rebuilt in NumPy (matches PhyCV exactly), dispersion = free-particle Schrödinger = time stretch.
4. ECE 279AS TDGSA project rebuilt in Python, then ANSI C and CUDA, then on qubits with Qiskit, then C++/POSIX + CUDA trial statistics.
5. QEC: phase-flip code in Qiskit connecting "known phase (dispersion, undone by GS)" to "unknown phase (dephasing, corrected by QEC)".

## Results worth remembering

* TDGSA needs D1 ≈ 0 plus a D2 matched to the feature time scale; both-far-field pairs (30/90 ps/nm) are redundant.
* TIE initial phase fixes the "random start doesn't converge" failure from the 2024 slides.
* Phase-flip code: average fidelity 1 − 2p_L/3 with p_L = 3p² − 2p³ (matches `quantum_error_correction.logical_error` to 1e-12), break-even p = 1/2, coherent phase error suppressed θ² → θ⁴ (slopes 2.00 / 4.00).

## Open threads / good next steps

* TDGSA noise robustness (denoising, tighter window, 3+ dispersion planes) and a detector-bandwidth model.
* QEC: Shor 9-qubit or Steane 7-qubit in Qiskit (bit + phase), noisy ancillas / measurement errors (circuit-level noise), syndrome extraction with real mid-circuit measurement on a Qiskit sampler; tie to the matching decoder in `quantum_error_correction`.
* `dgs.quantum_ce.three_qubit_bit_flip_code` has inconsistent syndrome labels (noted in `notebooks/quantum_error_correction.ipynb`); there were earlier sessions on fixing it.
* C/CUDA: batch many shots with `cufftPlanMany`; drop the per-iteration host sync.
* Verify all IEEE reference details (written from memory).

## Environment notes (Colin's Windows machine)

* Use the **py312** Jupyter kernel (`C:\Users\mrjel\AppData\Local\Programs\Python\Python312\python.exe`, Qiskit 2.5.2 installed). The **circuits32** conda env's matplotlib segfaults on any draw: the kernel dies at the first plot.
* nvcc needs MSVC: `call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`; it may return nonzero from a harmless vswhere warning, so check `where cl` instead of its exit code.
* GPU: RTX 4060 (`-arch=sm_89`), CUDA 13.0, gcc 15.1 (MSYS2).
