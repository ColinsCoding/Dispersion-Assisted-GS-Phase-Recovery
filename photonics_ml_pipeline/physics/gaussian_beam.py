"""Numeric Gaussian-beam model.

Purpose:
    Concrete numeric realization of the symbolic beam relations, used to generate
    optical fields that the feature-extraction stage turns into ML features.

Equations:
    z_R = pi w0^2 / lambda,  w(z) = w0 sqrt(1 + (z/z_R)^2),
    R(z) = z (1 + (z_R/z)^2),  psi(z) = atan(z/z_R),
    E(x, z) = (w0/w) exp(-x^2/w^2) exp(i[k z + k x^2/(2R) - psi]).

Assumptions:
    - Paraxial, monochromatic, waist located at z = 0. Units: micrometers.
Limitations:
    - 1-D transverse profile (x); extend to 2-D by separability if needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GaussianBeam"]


@dataclass(frozen=True)
class GaussianBeam:
    """A paraxial Gaussian beam defined by wavelength and waist radius (in um)."""

    wavelength_um: float
    waist_um: float

    def __post_init__(self) -> None:
        if self.wavelength_um <= 0 or self.waist_um <= 0:
            raise ValueError("wavelength_um and waist_um must be positive.")

    @property
    def rayleigh_range_um(self) -> float:
        """z_R = pi w0^2 / lambda."""
        return float(np.pi * self.waist_um**2 / self.wavelength_um)

    @property
    def wavenumber(self) -> float:
        """k = 2 pi / lambda (rad/um)."""
        return float(2.0 * np.pi / self.wavelength_um)

    def width_um(self, z_um: np.ndarray | float) -> np.ndarray:
        """Beam radius w(z)."""
        z = np.asarray(z_um, dtype=float)
        return self.waist_um * np.sqrt(1.0 + (z / self.rayleigh_range_um) ** 2)

    def radius_of_curvature_um(self, z_um: np.ndarray | float) -> np.ndarray:
        """Wavefront radius R(z); infinite at the waist."""
        z = np.asarray(z_um, dtype=float)
        zr = self.rayleigh_range_um
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(z == 0.0, np.inf, z * (1.0 + (zr / z) ** 2))
        return r

    def gouy_phase_rad(self, z_um: np.ndarray | float) -> np.ndarray:
        """Gouy phase psi(z) = atan(z/z_R)."""
        z = np.asarray(z_um, dtype=float)
        return np.arctan(z / self.rayleigh_range_um)

    def field_1d(self, x_um: np.ndarray, z_um: float) -> np.ndarray:
        """Complex transverse field E(x, z) at a fixed axial position z."""
        x = np.asarray(x_um, dtype=float)
        w = float(self.width_um(z_um))
        r = float(self.radius_of_curvature_um(z_um))
        k = self.wavenumber
        amp = (self.waist_um / w) * np.exp(-(x**2) / w**2)
        curvature = 0.0 if np.isinf(r) else k * x**2 / (2.0 * r)
        phase = k * z_um + curvature - float(self.gouy_phase_rad(z_um))
        return amp * np.exp(1j * phase)

    def intensity_1d(self, x_um: np.ndarray, z_um: float) -> np.ndarray:
        """Transverse intensity |E(x, z)|^2."""
        return np.abs(self.field_1d(x_um, z_um)) ** 2
