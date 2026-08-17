"""spectral_fingerprint_id.py -- spectral library matching, the core
algorithm behind handheld Raman/IR analyzers used in forensic and analytical
chemistry: measure a sample's vibrational spectrum, compare it against a
LIBRARY of reference spectra via a similarity metric, and report the best
match (or 'no confident match' if nothing in the library is close enough).

This module implements and tests the GENERAL matching algorithm with
clearly SYNTHETIC example library entries ("Compound_A", "Compound_B", ...),
matching this repo's consistent pattern of synthetic test data everywhere
else (retinal_scan_imaging.py, spectral_interferometry.py, etc.). It is NOT
a populated real-world substance-identification database and should not be
treated as one -- the point is the matching algorithm, which is the same
algorithm regardless of what the library entries represent.

Physical model: a Raman/IR spectrum is a sum of Lorentzian absorption peaks,
one per vibrational mode, at that mode's characteristic wavenumber. Reuses
dgs/causality.py's lorentz_susceptibility (the same Lorentz-oscillator model
already used there for optical dispersion/absorption) rather than
re-deriving the lineshape -- a mechanical resonance and a Raman/IR
vibrational mode are the same damped-oscillator physics.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

from dgs.causality import lorentz_susceptibility


# ── 1. Forward model: synthesize a spectrum from peak parameters ───────────

def synthetic_reference_spectrum(peak_centers: List[float], peak_widths: List[float],
                                  peak_strengths: List[float], wavenumber_grid: np.ndarray) -> np.ndarray:
    """Sum of Lorentzian absorption peaks (imaginary part of
    lorentz_susceptibility, one Lorentzian per vibrational mode) at the
    given peak_centers, with the given peak_widths (damping) and
    peak_strengths -- a synthetic Raman/IR-style spectrum."""
    wavenumber_grid = np.asarray(wavenumber_grid, dtype=float)
    if not (len(peak_centers) == len(peak_widths) == len(peak_strengths)):
        raise ValueError("peak_centers, peak_widths, peak_strengths must have equal length")
    if len(peak_centers) < 1:
        raise ValueError("need at least one peak")
    spectrum = np.zeros_like(wavenumber_grid)
    for center, width, strength in zip(peak_centers, peak_widths, peak_strengths):
        chi = lorentz_susceptibility(wavenumber_grid, omega0=center, gamma=width, strength=strength)
        spectrum += chi.imag
    return spectrum


def build_synthetic_library(n_compounds: int, wavenumber_grid: np.ndarray,
                             n_peaks_range: Tuple[int, int] = (2, 4),
                             rng_seed: int = 0) -> Dict[str, np.ndarray]:
    """Generate n_compounds SYNTHETIC labeled reference spectra
    ('Compound_A', 'Compound_B', ...), each a random sum of 2-4 Lorentzian
    peaks at random wavenumbers within the grid's range -- a toy library for
    testing the matching algorithm, not a real reference database."""
    if n_compounds < 1:
        raise ValueError(f"n_compounds={n_compounds}: must be >= 1")
    wavenumber_grid = np.asarray(wavenumber_grid, dtype=float)
    rng = np.random.default_rng(rng_seed)
    lo, hi = wavenumber_grid.min(), wavenumber_grid.max()
    library = {}
    for i in range(n_compounds):
        n_peaks = rng.integers(n_peaks_range[0], n_peaks_range[1] + 1)
        centers = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), n_peaks)
        widths = rng.uniform(0.01, 0.03, n_peaks) * (hi - lo)
        strengths = rng.uniform(0.5, 2.0, n_peaks)
        label = f"Compound_{chr(ord('A') + i)}"
        library[label] = synthetic_reference_spectrum(list(centers), list(widths), list(strengths), wavenumber_grid)
    return library


# ── 2. The matching algorithm ───────────────────────────────────────────────

def spectral_similarity(measured: np.ndarray, reference: np.ndarray, metric: str = "cosine") -> float:
    """Similarity between a measured spectrum and one reference spectrum.
    'cosine': cosine similarity (scale-invariant -- robust to unknown
    measurement gain, the standard choice for library-matching instruments).
    'pearson': Pearson correlation (also scale- and offset-invariant)."""
    measured = np.asarray(measured, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if measured.shape != reference.shape:
        raise ValueError("measured and reference must have the same shape")
    if metric == "cosine":
        denom = np.linalg.norm(measured) * np.linalg.norm(reference)
        if denom == 0:
            raise ValueError("cannot compute cosine similarity against an all-zero spectrum")
        return float(np.dot(measured, reference) / denom)
    elif metric == "pearson":
        if np.std(measured) == 0 or np.std(reference) == 0:
            raise ValueError("cannot compute Pearson correlation against a constant spectrum")
        return float(np.corrcoef(measured, reference)[0, 1])
    else:
        raise ValueError(f"metric={metric!r}: must be 'cosine' or 'pearson'")


def identify_spectrum(measured: np.ndarray, library: Dict[str, np.ndarray],
                       metric: str = "cosine", confidence_threshold: float = 0.8) -> Dict:
    """Compare `measured` against every entry in `library`, return the best
    match, its similarity score, and the full ranked list. Reports
    'no confident match' (best_label=None) if the top score is below
    confidence_threshold -- an instrument that always reports SOME match
    even for an unknown sample is worse than one that says 'unidentified'."""
    if not library:
        raise ValueError("library must not be empty")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be in [0, 1]")
    scores = {label: spectral_similarity(measured, ref, metric=metric) for label, ref in library.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]
    confident = best_score >= confidence_threshold
    return {
        "best_label": best_label if confident else None,
        "best_score": best_score,
        "confident": confident,
        "ranked_matches": ranked,
    }


def add_measurement_noise(spectrum: np.ndarray, noise_std: float, rng_seed: int = 0) -> np.ndarray:
    """Additive Gaussian measurement noise, scaled to the spectrum's own
    peak height -- a real instrument's noise floor is relative to signal
    strength, not an absolute constant."""
    spectrum = np.asarray(spectrum, dtype=float)
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    rng = np.random.default_rng(rng_seed)
    scale = noise_std * np.max(np.abs(spectrum))
    return spectrum + rng.normal(0.0, scale, size=spectrum.shape)


if __name__ == "__main__":
    wavenumber_grid = np.linspace(400.0, 1800.0, 1000)  # cm^-1, typical Raman fingerprint region
    library = build_synthetic_library(n_compounds=5, wavenumber_grid=wavenumber_grid, rng_seed=1)
    print(f"synthetic library: {list(library.keys())}")

    true_label = "Compound_C"
    clean = library[true_label]

    print("\n=== identification vs. noise level ===")
    for noise_std in [0.0, 0.02, 0.05, 0.1, 0.3]:
        measured = add_measurement_noise(clean, noise_std, rng_seed=2)
        result = identify_spectrum(measured, library, confidence_threshold=0.8)
        status = f"MATCH: {result['best_label']}" if result["confident"] else "no confident match"
        print(f"  noise_std={noise_std:.2f}  best_score={result['best_score']:.4f}  {status} "
              f"(true label: {true_label})")

    print("\n=== an unrelated spectrum (should NOT confidently match anything) ===")
    unrelated = synthetic_reference_spectrum([1200.0], [15.0], [1.0], wavenumber_grid)
    result = identify_spectrum(unrelated, library, confidence_threshold=0.8)
    print(f"  best_score={result['best_score']:.4f}  confident={result['confident']}")
