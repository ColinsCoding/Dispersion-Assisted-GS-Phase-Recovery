"""PyTorch dataset built from physics-derived features.

Purpose:
    Generate labeled samples where the label is the beam-divergence class and the input
    is the standardized feature vector extracted from the beam's transverse profile.
    This enforces the pipeline: physics -> features -> ML.

Assumptions:
    - Three classes distinguished by waist radius (tight / medium / wide focus).
Limitations:
    - Synthetic data; standardization statistics are computed on the full set.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from feature_extraction.features import extract_features, feature_names
from physics.gaussian_beam import GaussianBeam

__all__ = ["BeamFeatureDataset"]

# (label -> waist-radius range in um); classes are separable by beam width statistics.
_CLASS_WAIST_RANGES: tuple[tuple[float, float], ...] = ((4.0, 8.0), (12.0, 18.0), (24.0, 36.0))


class BeamFeatureDataset(Dataset):
    """Standardized (feature, label) pairs from simulated Gaussian beams."""

    def __init__(
        self,
        n_samples: int = 900,
        wavelength_um: float = 1.55,
        z_um: float = 300.0,
        half_width_um: float = 60.0,
        n_transverse: int = 256,
        seed: int = 0,
    ) -> None:
        self.feature_names = feature_names()
        rng = np.random.default_rng(seed)
        x = np.linspace(-half_width_um, half_width_um, n_transverse)
        n_classes = len(_CLASS_WAIST_RANGES)

        raw: list[np.ndarray] = []
        labels: list[int] = []
        for i in range(n_samples):
            label = i % n_classes
            lo, hi = _CLASS_WAIST_RANGES[label]
            waist = float(rng.uniform(lo, hi))
            beam = GaussianBeam(wavelength_um=wavelength_um, waist_um=waist)
            intensity = beam.intensity_1d(x, z_um)
            raw.append(extract_features(intensity).to_array())
            labels.append(label)

        features = np.asarray(raw, dtype=np.float64)
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        standardized = (features - self._mean) / self._std

        self._x = torch.tensor(standardized, dtype=torch.float32)
        self._y = torch.tensor(labels, dtype=torch.long)

    @property
    def feature_dim(self) -> int:
        """Number of features per sample."""
        return self._x.shape[1]

    @property
    def n_classes(self) -> int:
        """Number of distinct labels."""
        return int(self._y.max().item()) + 1

    @property
    def standardization(self) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std) used to standardize features (for inference-time reuse)."""
        return self._mean, self._std

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[index], self._y[index]
