# SEALS — Spectrally Encoded Angular Light Scattering

A dual-grating dispersive element maps **scattering angle θ → wavelength λ**: a
broadband laser illuminates a particle, and scattered light at each angle is
diffracted to a different position on a spectrometer CCD, so a single spectrum
encodes the full angular scattering pattern.

## Files
- **`seals_intro.ipynb`** — teaching notebook (numpy only, py-3.13): the grating
  λ→θ mapping, Rayleigh–Debye–Gans scattering (with the form factor's `u→0` Taylor
  limit), and the SEALS readout `I(θ(λ))`. Start here.
- **`seals_ml_inverse.ipynb`** — ML notebook (torch, py-3.12): learn the **inverse**
  problem, recovering particle diameter from a scattering pattern (R² ≈ 0.998).
- **`seals_ai_sensing.ipynb`** — AI/sensing notebook (sklearn, py-3.12), OUSD
  *Integrated Sensing* angle: detect a microplastic **size band** from the scattering
  shape — ROC/AUC baseline, multi-model bake-off (AUC ≈ 0.98), and physics-feature
  engineering. Same toolkit as `notebooks/ml_course_on_receiver.ipynb`.
- **`seals_stable.py`** / **`seals_stable.ipynb`** — numerically stable Python port
  of the original MATLAB, plus extensions (angular-momentum partial-wave spectrum,
  3D/4D spectral-angular maps, OAM/Laguerre–Gaussian decomposition). Its final
  section, "Phase Retrieval and Inverse Scattering," demonstrates `inverse/` below.
- **`matlab/`** — the original `main.m`, `SEALS.m`, `mie-2.m`, `rayleighdebye.m`.
- **`inverse/`** — phase-retrieval / inverse-scattering extensions (see below).

## Physics
- **SEALS mapping** `y(λ) = (D/6)·tan(Δ)/(1 + tan(Δ)·tan(α))`, `Δ = α − arcsin(λ/d − sin α)`.
- **Rayleigh–Debye–Gans** small-particle scattering with form factor `P(u)`.
- **Lorenz–Mie** exact sphere scattering: coefficients `a_n, b_n` from spherical
  Bessel functions, amplitudes `S₁, S₂` from the `π_n, τ_n` angular functions.

## Bug fixes vs. the original port
1. SEALS denominator `tan(Δ)·tan(α)` (was `tan(Δ)²`).
2. RDG form factor `P(θ→0) → 1` via a Taylor guard (was NaN from `0/0`).
3. Mie E-fields kept **complex** (was silent real truncation).
4. Mie angular recurrence `range(2, nmax)` (was `range(3, nmax)`, skipping `π₂`).
5. Angular loop vectorized; debug prints removed.

## Run
```bash
py -3.12 projects/seals/seals_stable.py     # needs scipy (spherical Bessel functions)
```
scipy is required (`scipy.special.spherical_jn/yn`), which is on the py-3.12
environment in this setup, not py-3.13.

## Phase retrieval / inverse scattering

The SEALS model produces wavelength-encoded angular scattering measurements.

The Mie forward model predicts a complex scattered field, while a square-law
detector measures intensity only (`inverse/measurement.py`).

This repository now contains:

- **model-based particle-parameter inversion** (`inverse/inverse_scattering.py`) —
  recovers particle diameter from a synthetic intensity spectrum via a
  derivative-free search against the validated (non-differentiable, SciPy-based)
  Mie model, then reads the corresponding phase off that same fitted model
- **complex-field measurement simulation** (`inverse/measurement.py`) — the
  explicit `I = |E|^2` boundary, and reconstruction of Mie's `E_p`, `E_s` from
  its validated `I_p`, `I_s`, `T_p`, `T_s` outputs
- **a minimal phase-retrieval experiment with measurement diversity**
  (`inverse/phase_retrieval.py`, `inverse/dispersion.py`) — PyTorch-autograd
  recovery of an arbitrary phase profile from one or more known-transform
  intensity measurements, explicitly flagged as underdetermined when only one
  measurement is used

These are research extensions and are not part of the original SEALS MATLAB
implementation. Model-based parameter inversion and generic phase retrieval are
kept as distinct concepts throughout (see `seals_stable.ipynb`'s final section
and each module's docstring) -- the former is far more constrained than the
latter, and neither is claimed to be the original SEALS paper's method.

Tests: `tests/test_seals_inverse_measurement.py`,
`tests/test_seals_phase_retrieval.py`, `tests/test_seals_inverse_scattering.py`.
