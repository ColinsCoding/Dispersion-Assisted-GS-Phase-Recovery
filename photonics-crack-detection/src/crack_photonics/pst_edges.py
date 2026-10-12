"""Simplified, educational approximation of the Phase Stretch Transform (PST).

WHAT THIS IS
    A short NumPy/SciPy implementation of the *idea* behind PST edge
    detection, written from the published equations for teaching purposes.

WHAT THIS IS NOT
    It is not the reference implementation, it is not tuned or validated
    against it, and it does NOT reproduce laboratory photonic hardware
    (time-stretch dispersive Fourier transform systems).  No light is
    involved: this is a digital filter whose design is *inspired by* how a
    dispersive optical element delays different frequencies by different
    amounts.

The idea in four steps
----------------------
1. Go to the spatial-frequency domain with a 2-D FFT.
2. Smooth with a Gaussian low-pass filter (suppresses pixel noise).
3. Multiply by a pure phase filter ``exp(-j * phi(r))`` where the phase
   ``phi`` grows with the radial frequency ``r``.  An all-pass phase filter
   changes no magnitudes; it only shifts frequencies relative to each other,
   which is what optical *dispersion* does.
4. Return to the image domain and keep the PHASE of the complex result.
   High spatial frequencies (edges) received the most phase, so the output
   phase is large at edges and near zero in flat regions.

The phase profile is the one proposed by Asghari and Jalali:

    phi(r) = S * [ W r * arctan(W r) - 0.5 * ln(1 + (W r)^2) ] / max(...)

with strength ``S`` and warp ``W``.  Its derivative is ``arctan(W r)``, a
group delay that rises linearly at low frequency and saturates at high
frequency.

Reference (concept only):
    M. H. Asghari and B. Jalali, "Edge detection in digital images using
    dispersive phase stretch transform," Int. J. Biomedical Imaging, 2015.

Property worth knowing: because the output is ``atan(Im / Re)`` and ``Re`` is
roughly the local brightness, the response is stronger in dark regions.
That helps with dark cracks and hurts in shadows.
"""

from __future__ import annotations

import numpy as np
from scipy import fft

from .io import require_2d_float


def pst_phase_kernel(
    shape: tuple[int, int], phase_strength: float = 0.5, warp_strength: float = 15.0
) -> np.ndarray:
    """Return the PST phase profile ``phi(u, v)`` on the FFT frequency grid.

    Frequencies are in cycles per pixel (from ``scipy.fft.fftfreq``), so the
    radial frequency ``r`` runs from 0 to about 0.707.
    """
    if phase_strength <= 0:
        raise ValueError(f"phase_strength must be > 0, got {phase_strength}.")
    if warp_strength <= 0:
        raise ValueError(f"warp_strength must be > 0, got {warp_strength}.")

    rows, cols = shape
    u = fft.fftfreq(rows)[:, None]
    v = fft.fftfreq(cols)[None, :]
    wr = warp_strength * np.hypot(u, v)

    phi = wr * np.arctan(wr) - 0.5 * np.log1p(wr**2)
    return phase_strength * phi / phi.max()


def pst_phase(
    image: np.ndarray,
    phase_strength: float = 0.5,
    warp_strength: float = 15.0,
    lowpass_sigma: float = 0.15,
    pad: int = 16,
) -> np.ndarray:
    """Apply the simplified PST and return the output phase in radians.

    Parameters
    ----------
    image : 2-D float array
    phase_strength, warp_strength : float
        ``S`` and ``W`` of the phase profile (see module docstring).
    lowpass_sigma : float
        Width of the Gaussian low-pass filter in cycles per pixel.
    pad : int
        The FFT treats the image as periodic, so the left edge "touches" the
        right edge and creates fake edges at the border.  Mirror-padding by
        ``pad`` pixels and cropping afterwards removes most of that artefact.

    Complexity
    ----------
    Two FFTs of an N-pixel image: O(N log N) time, O(N) memory.
    """
    arr = require_2d_float(image)
    if lowpass_sigma <= 0:
        raise ValueError(f"lowpass_sigma must be > 0, got {lowpass_sigma}.")
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}.")

    padded = np.pad(arr, pad, mode="reflect") if pad else arr
    rows, cols = padded.shape

    u = fft.fftfreq(rows)[:, None]
    v = fft.fftfreq(cols)[None, :]
    lowpass = np.exp(-0.5 * (np.hypot(u, v) / lowpass_sigma) ** 2)
    phase_filter = np.exp(-1j * pst_phase_kernel((rows, cols), phase_strength, warp_strength))

    spectrum = fft.fft2(padded)
    stretched = fft.ifft2(spectrum * lowpass * phase_filter)
    phase = np.angle(stretched)

    return phase[pad:rows - pad, pad:cols - pad] if pad else phase


def detect_edges(
    image: np.ndarray,
    phase_strength: float = 0.5,
    warp_strength: float = 15.0,
    lowpass_sigma: float = 0.15,
    threshold_sigma: float = 4.0,
    pad: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the simplified PST and threshold it into an edge map.

    Returns
    -------
    edge_strength : float array in [0, 1]
        ``|phase|`` divided by its maximum.
    edge_mask : bool array
        ``True`` where ``|phase|`` exceeds
        ``median + threshold_sigma * robust_std``.  The robust standard
        deviation is ``1.4826 * MAD`` (median absolute deviation), which is
        not inflated by the edges themselves the way an ordinary standard
        deviation would be.
    """
    if threshold_sigma <= 0:
        raise ValueError(f"threshold_sigma must be > 0, got {threshold_sigma}.")

    magnitude = np.abs(pst_phase(image, phase_strength, warp_strength, lowpass_sigma, pad))

    median = float(np.median(magnitude))
    robust_std = 1.4826 * float(np.median(np.abs(magnitude - median)))
    edge_mask = magnitude > median + threshold_sigma * robust_std

    peak = float(magnitude.max())
    edge_strength = magnitude / peak if peak > 0 else np.zeros_like(magnitude)
    return edge_strength, edge_mask
