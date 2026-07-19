"""Split-step Fourier NLSE solver for generating dispersed-intensity pairs (I1, I2).

Used as a physically richer alternative to the linear-dispersion-only
measurement generator in gs_core.make_measurements: here the envelope
actually propagates under the nonlinear Schrodinger equation (GVD + SPM),
and two intensity-only "photodiode" traces are recorded at two different
propagation distances. These (I1, I2) pairs feed the TDGSA phase-retrieval
pipeline and, downstream, a machine-learning verifier dataset.
"""
from __future__ import annotations

import numpy as np


def _check_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def gaussian_pulse(t: np.ndarray, t0: float = 1.0, chirp: float = 0.0,
                    amplitude: float = 1.0) -> np.ndarray:
    """Chirped Gaussian envelope A(t) = amplitude * exp(-(1+i*chirp)/2 * (t/t0)^2)."""
    _check_positive(t0, "t0")
    return amplitude * np.exp(-(1 + 1j * chirp) / 2.0 * (t / t0) ** 2)


def nlse_propagate(a0: np.ndarray, t: np.ndarray, z: float, beta2: float = -1.0,
                    gamma: float = 0.0, n_steps: int = 200) -> np.ndarray:
    """Symmetric split-step Fourier propagation of the envelope NLSE.

    i dA/dz = (beta2/2) d^2A/dt^2 - gamma|A|^2 A

    Parameters
    ----------
    a0 : initial complex envelope sampled on `t`
    t : uniformly spaced time grid
    z : total propagation distance (same units as beta2's z-scale)
    beta2 : group-velocity-dispersion coefficient (negative = anomalous)
    gamma : Kerr nonlinearity coefficient (0 disables SPM -> linear dispersion)
    n_steps : number of split-step segments
    """
    _check_positive(n_steps, "n_steps")
    if z == 0:
        return a0.copy()
    dt = t[1] - t[0]
    n = len(t)
    omega = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    dz = z / n_steps
    linear_half = np.exp(1j * beta2 / 2.0 * omega ** 2 * dz / 2.0)

    a = a0.copy()
    for _ in range(n_steps):
        a = np.fft.ifft(linear_half * np.fft.fft(a))
        if gamma != 0:
            a = a * np.exp(1j * gamma * np.abs(a) ** 2 * dz)
        a = np.fft.ifft(linear_half * np.fft.fft(a))
    return a


def make_nlse_measurements(n_points: int = 256, t0: float = 1.0, chirp: float = 0.6,
                            beta2: float = -1.0, gamma: float = 0.0,
                            z1: float = 3.0, z2: float = 7.0,
                            n_steps: int = 200, noise_std: float = 0.0,
                            seed: int | None = None):
    """Generate a TDGSA-style (I1, I2) measurement pair via NLSE propagation.

    Returns
    -------
    dict with keys: t, a0, phi_true, I1, I2, z1, z2
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(-8 * t0, 8 * t0, n_points)
    a0 = gaussian_pulse(t, t0=t0, chirp=chirp)

    a1 = nlse_propagate(a0, t, z1, beta2=beta2, gamma=gamma, n_steps=n_steps)
    a2 = nlse_propagate(a0, t, z2, beta2=beta2, gamma=gamma, n_steps=n_steps)

    i1 = np.abs(a1) ** 2
    i2 = np.abs(a2) ** 2
    if noise_std > 0:
        i1 = np.clip(i1 + rng.normal(0, noise_std, size=i1.shape), 0, None)
        i2 = np.clip(i2 + rng.normal(0, noise_std, size=i2.shape), 0, None)

    phi_true = np.unwrap(np.angle(a0))
    return {"t": t, "a0": a0, "phi_true": phi_true, "I1": i1, "I2": i2,
            "z1": z1, "z2": z2}
