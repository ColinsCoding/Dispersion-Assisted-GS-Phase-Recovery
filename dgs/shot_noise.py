"""How many quantum measurements does dispersion-assisted phase retrieval need?

In plain words
--------------
``dgs/qiskit_gs.py`` builds dispersion as a quantum circuit (inverse QFT, diagonal phase, QFT) and
runs Gerchberg-Saxton (GS) on the "detector" intensities. With exact probabilities GS recovers the
phase almost perfectly. A real quantum computer only gives counts: each run of the circuit returns
one sample index, so the intensity is estimated from `shots` repetitions, with statistical
(shot) noise of relative size about 1/sqrt(counts) in each sample.

This module measures the cost of that noise: for each shot budget it repeats the full
measure-then-retrieve pipeline with different random seeds and records the RMS phase error.

What the data show (tests/test_shot_noise.py and scripts/make_shot_noise.py) is two regimes:

* **Threshold.** With too few shots, GS often stalls in a wrong solution (errors near 1 rad), so
  the useful number is the *success rate*, the fraction of runs below ``SUCCESS_RAD``.
* **Noise floor.** Once a run succeeds, its error falls like shots^(-1/2), the shot-noise law.
  ``fit_slope`` measures that exponent using successful runs only; fitting all runs together mixes
  the two regimes and gives a misleading slope.

The shot count is per detector arm (two arms are measured), so the total number of circuit
executions is 2 * shots.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Operator

from dgs import qiskit_gs as Q

N_QUBITS = 7
D1, D2 = -1000.0, -1150.0          # the same arms tests/test_qiskit_gs.py uses
SUCCESS_RAD = 0.1                  # a run "succeeds" if its RMS phase error is below this


def setup(n_qubits: int = N_QUBITS, d1: float = D1, d2: float = D2, field_seed: int = 0):
    """Test field and the two dispersion circuits with their unitaries."""
    _, E, phi = Q.smooth_phase_field(n_qubits, n_harmonics=8, seed=field_seed)
    c1, c2 = Q.dispersion_circuit(n_qubits, d1), Q.dispersion_circuit(n_qubits, d2)
    return E, phi, c1, c2, Operator(c1).data, Operator(c2).data


def trial(shots, seed: int, ctx=None, n_iter: int = 100) -> float:
    """One full pipeline: sample both arms with `shots` each (None = exact), run GS, return RMS phase error."""
    E, phi, c1, c2, U1, U2 = ctx if ctx is not None else setup()
    I1 = Q.measure_intensity(E, c1, shots=shots, seed=2 * seed + 1)
    I2 = Q.measure_intensity(E, c2, shots=shots, seed=2 * seed + 2)
    Er, _ = Q.gs_retrieve(I1, I2, U1, U2, n_iter=n_iter)
    return Q.align_rms(np.angle(Er), phi)[0]


def sweep(shot_list, n_seeds: int = 6, ctx=None, n_iter: int = 100):
    """For each shot budget: all errors, the success rate, and the median error of successful runs."""
    ctx = ctx if ctx is not None else setup()
    errs = np.array([[trial(s, k, ctx, n_iter) for k in range(n_seeds)] for s in shot_list])
    ok = errs < SUCCESS_RAD
    med_ok = np.array([np.median(e[m]) if m.any() else np.nan for e, m in zip(errs, ok)])
    return dict(shots=np.asarray(shot_list, float), errors=errs, success_rate=ok.mean(axis=1),
                median_success=med_ok, median_all=np.median(errs, axis=1), exact=trial(None, 0, ctx, n_iter))


def fit_slope(shots, rms) -> tuple[float, float]:
    """Least-squares fit of log(rms) = slope*log(shots) + b. Returns (slope, prefactor)."""
    slope, b = np.polyfit(np.log(shots), np.log(rms), 1)
    return float(slope), float(np.exp(b))


def shots_needed(target_rms: float, slope: float, prefactor: float) -> float:
    """Shots per arm to reach `target_rms`, from the fitted power law."""
    return (target_rms / prefactor) ** (1.0 / slope)
