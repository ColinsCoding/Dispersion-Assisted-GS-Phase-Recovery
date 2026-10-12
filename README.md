# Dispersion-Assisted Gerchberg-Saxton Phase Recovery

A photodetector measures only intensity, so the phase of an optical signal is lost. This project recovers it. The signal is measured after two known, different dispersions, and the Gerchberg-Saxton algorithm iterates between the two measurements until it finds a phase consistent with both.

## What is here (6 files)

| file | what it does |
|---|---|
| `dgs/gs_core.py` | simulates the two intensity measurements with noise and runs Gerchberg-Saxton phase retrieval (numpy only) |
| `dgs/materials.py` | refractive index and dispersion of fused silica, N-BK7, sapphire, CaF₂ and LiNbO₃ (Sellmeier equation) |
| `tests/test_materials.py` | checks the material library against published index values |
| `notebooks/demo_dispersion_gs.ipynb` | runs a recovery end to end and plots it (simulated data) |
| `requirements.txt` | Python packages |
| `README.md` | this file |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q
jupyter lab notebooks/demo_dispersion_gs.ipynb
```

## Conclusions

What the code in this repository shows, all on **simulated** data (no lab measurements):

1. **Two dispersions recover the phase; one does not.** With two different dispersions, Gerchberg-Saxton recovers a QPSK phase to 0.061 rad RMS (`notebooks/demo_dispersion_gs.ipynb`). An earlier single-dispersion prototype only reached 1.69 rad, which is close to a random guess.
2. **The dispersion has to be strong enough.** At 128 samples, normalized dispersions of (−200, −230) fail (1.33 rad), (−600, −690) are partial (0.73 rad), and (−1000, −1150) recover the phase essentially exactly. When the dispersions are weak, the two intensity recordings look almost the same, so there is nothing to solve from.
3. **Noise sets the accuracy.** When the intensities are sampled (photon or shot counting), the phase error falls like 1/√(number of counts): 0.070 rad at 10⁴ counts, 0.022 rad at 10⁵, 0.007 rad at 10⁶. At 10³ counts recovery fails.
4. **Some things can never be recovered from intensity:** a constant phase offset, and (in general) a sign flip of the phase. Results are reported after removing both.
5. **The material numbers are trustworthy.** `dgs/materials.py` reproduces published values: N-BK7 n_d = 1.5168 and Abbe number 64.17, fused-silica zero-dispersion wavelength 1.27 µm, and D ≈ +22 ps/(nm·km) at 1550 nm (`tests/test_materials.py`).
6. **A quantum-circuit version gives the same answer, but no speedup.** On the archive branch, the same dispersion is built in Qiskit (inverse QFT → diagonal phase → QFT on 7 qubits). It matches the FFT model to about 10⁻¹³. Loading and reading out the data cancels any quantum advantage, and the circuit needs about 350 two-qubit gates.

**Limits.** Only second-order dispersion is modeled, the input is assumed to have constant amplitude, and every result is from simulation. The next real test is measured data from a dispersive-fiber setup.

## Full history

Everything else (more notebooks, C and CUDA versions, coursework, and the complete commit history) is on the branch [`archive/full-history-2026-10-12`](../../tree/archive/full-history-2026-10-12).

Originally a UCLA ECE 279AS course project (2024).
