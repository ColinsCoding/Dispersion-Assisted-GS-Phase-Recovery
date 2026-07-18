"""ABCD (ray-transfer) matrix optics for Gaussian beams.

Purpose:
    Propagate the complex beam parameter q through optical elements and recover the
    beam width -- an independent cross-check of `physics.gaussian_beam`.

Equations:
    q(z) = z + i z_R              (waist at z = 0)
    q' = (A q + B) / (C q + D)     (ABCD transform)
    1/q = 1/R - i lambda/(pi w^2)  =>  w = sqrt(-lambda / (pi * Im(1/q)))

References:
    - Kogelnik & Li, "Laser Beams and Resonators," Appl. Opt. 5, 1550 (1966).
Assumptions:
    - Paraxial optics; homogeneous medium of index 1. Units: micrometers.
Limitations:
    - Real ABCD elements only (no gain/loss sheets).
"""
from __future__ import annotations

import numpy as np

__all__ = ["free_space", "thin_lens", "propagate_q", "q_at_waist", "width_from_q"]


def free_space(distance_um: float) -> np.ndarray:
    """ABCD matrix for propagation over `distance_um`."""
    return np.array([[1.0, float(distance_um)], [0.0, 1.0]])


def thin_lens(focal_length_um: float) -> np.ndarray:
    """ABCD matrix for a thin lens of focal length `focal_length_um`."""
    if focal_length_um == 0:
        raise ValueError("focal_length_um must be non-zero.")
    return np.array([[1.0, 0.0], [-1.0 / float(focal_length_um), 1.0]])


def q_at_waist(rayleigh_range_um: float) -> complex:
    """Complex beam parameter at the waist: q = i z_R."""
    return complex(0.0, float(rayleigh_range_um))


def propagate_q(q: complex, matrix: np.ndarray) -> complex:
    """Apply an ABCD matrix to the complex beam parameter q."""
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    return complex((a * q + b) / (c * q + d))


def width_from_q(q: complex, wavelength_um: float) -> float:
    """Recover the beam radius w from the complex beam parameter q."""
    inv_im = np.imag(1.0 / q)
    if inv_im >= 0:
        raise ValueError("Im(1/q) must be negative for a physical Gaussian beam.")
    return float(np.sqrt(-wavelength_um / (np.pi * inv_im)))
