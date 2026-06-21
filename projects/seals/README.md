# SEALS — Spectrally Encoded Angular Light Scattering

A dual-grating dispersive element maps **scattering angle θ → wavelength λ**: a
broadband laser illuminates a particle, and scattered light at each angle is
diffracted to a different position on a spectrometer CCD, so a single spectrum
encodes the full angular scattering pattern.

## Files
- **`seals_stable.py`** / **`seals_stable.ipynb`** — numerically stable Python port
  of the original MATLAB, plus extensions (angular-momentum partial-wave spectrum,
  3D/4D spectral-angular maps, OAM/Laguerre–Gaussian decomposition).
- **`matlab/`** — the original `main.m`, `SEALS.m`, `mie-2.m`, `rayleighdebye.m`.

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
