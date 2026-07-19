"""Phase Stretch Transform (PhyCV) -- pure-NumPy core, no ROS dependency.

Purpose:
    The image-processing heart of the ROS2 node, isolated so it can be unit-tested and
    reused without a ROS2 runtime. Implements the dispersive-phase edge detector from
    the Jalali-lab PhyCV algorithm (see physics_repo/notebooks/phycv_*).

Equations:
    C(rho) = L(rho) * exp(i phi_r(rho)),
    phi_r(rho) = S [W rho arctan(W rho) - 1/2 ln(1 + (W rho)^2)] / max,
    L(rho) = exp(-rho^2 / 2 sigma^2),  edges = |arg(IFFT2{ C * FFT2{img} })| > threshold.

Assumptions:
    - Input is a 2-D grayscale array; values are finite. Works on any size.
Limitations:
    - Isotropic PST only (PAGE orientation is a straightforward extension).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PstParams", "PstResult", "phase_stretch_transform", "apply_phase_kernel", "normalize_pedestal"]


@dataclass(frozen=True)
class PstParams:
    """Tunable PST parameters."""

    strength: float = 4.0
    warp: float = 15.0
    sigma_lpf: float = 0.2
    threshold: float = 0.3


@dataclass(frozen=True)
class PstResult:
    """PST output: the raw phase feature and a thresholded binary edge map."""

    phase: np.ndarray
    edges: np.ndarray


def _phase_kernel(shape: tuple[int, int], params: PstParams) -> np.ndarray:
    m, n = shape
    u = np.fft.fftfreq(m)[:, None]
    v = np.fft.fftfreq(n)[None, :]
    rho = np.sqrt(u**2 + v**2)
    lpf = np.exp(-0.5 * (rho / params.sigma_lpf) ** 2)
    wr = params.warp * rho
    phi = params.strength * (wr * np.arctan(wr) - 0.5 * np.log1p(wr**2))
    phi = phi / phi.max() * params.strength
    return lpf * np.exp(1j * phi)


def normalize_pedestal(gray: np.ndarray) -> np.ndarray:
    """Normalize a 2-D image to [0.2, 1.0] (a DC pedestal avoids flat-region phase noise)."""
    gray = np.asarray(gray, dtype=float)
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {gray.shape}")
    lo, hi = float(gray.min()), float(gray.max())
    norm = (gray - lo) / (hi - lo) if hi > lo else np.zeros_like(gray)
    return 0.2 + 0.8 * norm


def apply_phase_kernel(gray: np.ndarray, kernel: np.ndarray, threshold: float) -> PstResult:
    """Apply an arbitrary complex phase kernel (float or ROM-quantized) and threshold the phase.

    Sharing this step keeps the float and ROM-quantized transforms bit-identical apart from
    the kernel itself -- exactly what the FPGA does with a coefficient ROM.
    """
    norm = normalize_pedestal(gray)
    phase = np.angle(np.fft.ifft2(np.asarray(kernel) * np.fft.fft2(norm)))
    magnitude = np.abs(phase)
    peak = float(magnitude.max())
    edges = magnitude > threshold * peak if peak > 0 else np.zeros_like(magnitude, dtype=bool)
    return PstResult(phase=phase, edges=edges)


def phase_stretch_transform(gray: np.ndarray, params: PstParams | None = None) -> PstResult:
    """Apply PST with the exact (float) phase kernel and return the phase feature and edge map."""
    params = params or PstParams()
    gray = np.asarray(gray, dtype=float)
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {gray.shape}")
    kernel = _phase_kernel(gray.shape, params)
    return apply_phase_kernel(gray, kernel, params.threshold)
