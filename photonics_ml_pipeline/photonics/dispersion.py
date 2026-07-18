"""Group-velocity dispersion as an all-pass spectral filter.

Purpose:
    The dispersive time-stretch operator H_D(f) = exp(i pi D f^2) at the heart of the
    parent project. Being all-pass (|H| = 1) it is unitary and conserves energy -- a
    property the tests assert.

Equations:
    H_D(f) = exp(i pi D f^2)
    y = IFFT{ H_D(f) * FFT{x} }
    group delay  tau(f) = d(phase)/d(2 pi f) = D f

References:
    - Goda & Jalali, "Dispersive Fourier transformation," Nat. Photonics 7, 102 (2013).
Assumptions:
    - Uniform sampling; f from numpy.fft.fftfreq.
Limitations:
    - Second-order dispersion only (single parameter D).
"""
from __future__ import annotations

import numpy as np

__all__ = ["transfer_function", "apply_dispersion", "group_delay"]


def transfer_function(freq: np.ndarray, dispersion: float) -> np.ndarray:
    """All-pass dispersion transfer function H_D(f) = exp(i pi D f^2)."""
    freq = np.asarray(freq, dtype=float)
    return np.exp(1j * np.pi * float(dispersion) * freq**2)


def apply_dispersion(pulse: np.ndarray, dispersion: float) -> np.ndarray:
    """Apply second-order dispersion to a complex time-domain pulse."""
    pulse = np.asarray(pulse)
    n = pulse.shape[-1]
    freq = np.fft.fftfreq(n)
    h = transfer_function(freq, dispersion)
    return np.fft.ifft(h * np.fft.fft(pulse))


def group_delay(freq: np.ndarray, dispersion: float) -> np.ndarray:
    """Group delay tau(f) = D f imposed by the dispersion."""
    return float(dispersion) * np.asarray(freq, dtype=float)
