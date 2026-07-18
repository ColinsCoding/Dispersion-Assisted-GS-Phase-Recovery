"""Deterministic feature extraction from 1-D optical intensity profiles / images.

Purpose:
    Map a field (complex or real) to a fixed-length, physically interpretable feature
    vector. Physics generates the field; these features are what the ML stage consumes.

Features (8):
    total_energy, centroid, rms_width, peak, kurtosis,
    spectral_centroid, spectral_bandwidth, tamura_contrast.

Assumptions:
    - Input is a 1-D array (or is raveled); intensity = |field|^2 for complex input.
Limitations:
    - Index-space statistics (not physical units); adequate for classification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FeatureVector", "extract_features", "feature_names"]

_FEATURE_NAMES: tuple[str, ...] = (
    "total_energy",
    "centroid",
    "rms_width",
    "peak",
    "kurtosis",
    "spectral_centroid",
    "spectral_bandwidth",
    "tamura_contrast",
)


def feature_names() -> tuple[str, ...]:
    """Ordered names of the extracted features."""
    return _FEATURE_NAMES


@dataclass(frozen=True)
class FeatureVector:
    """A named, ordered feature vector."""

    names: tuple[str, ...]
    values: np.ndarray

    def to_array(self) -> np.ndarray:
        """Return the underlying values as a float array."""
        return np.asarray(self.values, dtype=float)

    def as_dict(self) -> dict[str, float]:
        """Return the features as a name -> value mapping."""
        return {n: float(v) for n, v in zip(self.names, self.values)}


def _intensity(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field)
    intensity = np.abs(field) ** 2 if np.iscomplexobj(field) else field.astype(float)
    return intensity.ravel()


def extract_features(field: np.ndarray) -> FeatureVector:
    """Compute the 8-element feature vector for an intensity profile / image."""
    intensity = _intensity(field)
    n = intensity.size
    if n == 0:
        raise ValueError("Cannot extract features from an empty array.")

    idx = np.arange(n, dtype=float)
    total = float(intensity.sum())
    prob = intensity / total if total > 0 else np.full(n, 1.0 / n)

    centroid = float((idx * prob).sum())
    variance = float(((idx - centroid) ** 2 * prob).sum())
    rms_width = float(np.sqrt(variance))
    peak = float(intensity.max())
    m4 = float(((idx - centroid) ** 4 * prob).sum())
    kurtosis = float(m4 / variance**2) if variance > 0 else 0.0

    spectrum = np.abs(np.fft.rfft(intensity)) ** 2
    s_idx = np.arange(spectrum.size, dtype=float)
    s_total = float(spectrum.sum())
    s_prob = spectrum / s_total if s_total > 0 else np.full(spectrum.size, 1.0 / spectrum.size)
    spectral_centroid = float((s_idx * s_prob).sum())
    spectral_bandwidth = float(np.sqrt(((s_idx - spectral_centroid) ** 2 * s_prob).sum()))

    mean = float(intensity.mean())
    tamura_contrast = float(intensity.std() / mean) if mean > 0 else 0.0

    values = np.array(
        [total, centroid, rms_width, peak, kurtosis, spectral_centroid, spectral_bandwidth, tamura_contrast],
        dtype=float,
    )
    return FeatureVector(_FEATURE_NAMES, values)
