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

## Full history

Everything else (more notebooks, C and CUDA versions, coursework, and the complete commit history) is on the branch [`archive/full-history-2026-10-12`](../../tree/archive/full-history-2026-10-12).

Originally a UCLA ECE 279AS course project (2024).
