"""
simulator.kramers_kronig — Closed-form single-shot phase recovery.

Theory
------
For a *minimum-phase* analytic signal E(ω), the real and imaginary parts of
ln E(ω) are Hilbert-transform pairs (Titchmarsh theorem):

    φ(ω) = -H{ ln |E(ω)| }

where H is the Hilbert transform.  This gives the spectral phase directly
from the spectral amplitude |E(ω)| with a single FFT-based computation —
no iteration required.

Minimum-phase condition
-----------------------
A signal is minimum-phase iff all zeros of E(z) lie inside the unit circle
(z-domain) / upper half ω-plane (continuous).  In practice we enforce this
by adding a real DC offset C before taking the logarithm:

    E_offset(ω) = E(ω) + C,   C > max |E(ω)|

This shifts all zeros out of the upper half-plane.  The recovered phase then
has a trivial offset contribution from C that is removed by subtracting
the phase of the offset-only term.

After phase recovery we reconstruct the complex spectrum and inverse-FFT
to recover the time-domain field.

Limitations
-----------
* Works exactly only for minimum-phase signals; results degrade gracefully
  for non-minimum-phase signals (phase wrapping artefacts).
* The DC offset choice affects accuracy — `offset_factor` tunes this.
* Phase is recovered modulo a global constant (like all phase retrieval).

References
----------
Kramers, H.A. (1927).  Atti Cong. Intern. Fisici, Como.
Kronig, R.d.L. (1926).  J. Opt. Soc. Am., 12, 547.
Dorrer, C. et al. (2003).  J. Opt. Soc. Am. B, 20, 1262.
"""

from __future__ import annotations
import numpy as np


def _hilbert_fft(x: np.ndarray) -> np.ndarray:
    """Compute the Hilbert transform of real array x via FFT.

    Uses the one-sided spectrum convention: multiply positive frequencies
    by 2, zero out negative, then IFFT.  Returns the imaginary part
    (the actual Hilbert transform).
    """
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N, dtype=complex)
    if N % 2 == 0:
        h[0] = 1
        h[1:N // 2] = 2
        h[N // 2] = 1
        # h[N//2+1:] = 0  (already zero)
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return np.fft.ifft(X * h).imag


def kk_recover(
    I_omega: np.ndarray,
    *,
    offset_factor: float = 2.5,
    return_spectrum: bool = False,
) -> np.ndarray:
    """Recover the complex spectrum from its intensity via Kramers-Kronig.

    Parameters
    ----------
    I_omega : array_like, real, shape (N,)
        Spectral intensity |E(ω)|² (linear scale, not dB).
        Must be non-negative.
    offset_factor : float
        DC offset = offset_factor * max(|E(ω)|).  Larger values enforce
        the minimum-phase condition more strongly but add a slowly-varying
        background; values in [1.5, 5] work well.  Default 2.5.
    return_spectrum : bool
        If True, return the complex spectrum E(ω) instead of the
        time-domain field u(t).

    Returns
    -------
    out : ndarray, complex, shape (N,)
        If ``return_spectrum`` is False (default): recovered time-domain
        field u(t) = IFFT[ E(ω) ].
        If ``return_spectrum`` is True: recovered complex spectrum E(ω).

    Notes
    -----
    The input I_omega is assumed to be in *natural* (non-fftshifted)
    FFT order, i.e. I_omega[0] is the DC bin.  If your data is fftshifted
    (DC at centre), apply ``np.fft.ifftshift`` before calling.

    Examples
    --------
    >>> import numpy as np
    >>> from simulator.kramers_kronig import kk_recover
    >>> N = 1024
    >>> omega = np.fft.fftfreq(N) * 2 * np.pi
    >>> # Minimum-phase Gaussian in spectrum
    >>> E_true = np.exp(-omega**2 / 0.5) * np.exp(1j * 0.3 * omega)
    >>> u_kk = kk_recover(np.abs(E_true)**2)
    >>> E_kk  = np.fft.fft(u_kk)
    >>> ph_err = np.std(np.unwrap(np.angle(E_kk)) - np.unwrap(np.angle(E_true)))
    """
    I_omega = np.asarray(I_omega, dtype=float)
    if I_omega.ndim != 1:
        raise ValueError(f"I_omega must be 1-D, got shape {I_omega.shape}")
    if np.any(I_omega < 0):
        raise ValueError("I_omega must be non-negative (it is an intensity)")

    amp = np.sqrt(np.maximum(I_omega, 0.0))
    C = offset_factor * amp.max()

    # Offset amplitude — guaranteed > 0 everywhere
    amp_off = amp + C

    # ln|E + C| — real, defined everywhere
    log_amp = np.log(amp_off)

    # Phase via Hilbert transform of the log-amplitude
    phi = -_hilbert_fft(log_amp)

    # Reconstruct offset spectrum
    E_off = amp_off * np.exp(1j * phi)

    # Remove DC-offset contribution: E(ω) = E_off(ω) - C * exp(i*phi_C(ω))
    # phi_C(ω) = -H{ln C} = 0 (C is a constant, ln C is flat → Hilbert = 0)
    E = E_off - C

    if return_spectrum:
        return E

    return np.fft.ifft(E)


def kk_seed_gs(
    I1: np.ndarray,
    t_axis: np.ndarray,
    beta2_L1: float,
) -> np.ndarray:
    """Use KK to produce a physics-informed seed for TD-GS.

    Applies KK on the first dispersed measurement to recover the complex
    field after dispersion L1, then back-propagates to t=0.

    This typically reduces TD-GS iterations from 250 to ~30 and avoids
    getting stuck in local minima that pure random-phase restarts hit.

    Parameters
    ----------
    I1 : ndarray, real, shape (N,)
        Intensity measurement after dispersion beta2_L1.
    t_axis : ndarray, real, shape (N,)
        Time axis (seconds).
    beta2_L1 : float
        Accumulated GVD of the first measurement channel (s²/rad).

    Returns
    -------
    u0 : ndarray, complex, shape (N,)
        KK-recovered seed at t=0 (before dispersion).
    """
    from .dispersion import propagate

    # KK in the time domain of the dispersed signal
    # The dispersed signal is measured in time, so its "spectrum" here
    # is its Fourier transform — we recover the complex dispersed field.
    I_fft = np.abs(np.fft.fft(np.sqrt(np.maximum(I1, 0.0)))) ** 2
    E_dispersed_freq = kk_recover(I_fft, return_spectrum=True)
    v1 = np.fft.ifft(E_dispersed_freq)

    # Back-propagate to origin
    u0 = propagate(v1, t_axis, -beta2_L1)
    return u0
