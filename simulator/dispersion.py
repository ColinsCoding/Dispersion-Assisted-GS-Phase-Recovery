"""
simulator.dispersion — GVD propagation via FFT phase multiply.

Math
----
Fibre / free-space GVD:
    E(omega, z) = E(omega, 0) * exp(-i * beta2_L * omega^2 / 2)

where omega is the *angular frequency* axis (rad/s) constructed from the
spatial or temporal axis supplied by the caller.

For the Schrödinger mapping used in quantum_wavepacket_demo.py:
    beta2_L = hbar * T / m
    t_axis  = x  (spatial coordinate in metres)
"""

import numpy as np


def _validated_axis(t_axis: np.ndarray, A_len: int) -> tuple[int, float]:
    """Return (N, dt) after checking that t_axis is uniform and matches A."""
    t_axis = np.asarray(t_axis, dtype=float)
    if t_axis.ndim != 1:
        raise ValueError(f"t_axis must be 1-D, got shape {t_axis.shape}")
    N = len(t_axis)
    if N < 2:
        raise ValueError("t_axis must have at least 2 points")
    if N != A_len:
        raise ValueError(
            f"t_axis length {N} does not match signal length {A_len}"
        )
    dts = np.diff(t_axis)
    dt = float(dts[0])
    max_rel_dev = float(np.max(np.abs(dts - dt))) / abs(dt)
    if max_rel_dev > 1e-6:
        raise ValueError(
            f"t_axis must be uniformly spaced (max relative error 1e-6); "
            f"got {max_rel_dev:.2e}"
        )
    if dt <= 0:
        raise ValueError(f"t_axis must be strictly increasing, got dt={dt}")
    return N, dt


def transfer_function(N: int, dt: float, beta2_L: float) -> np.ndarray:
    """Return the centred GVD transfer function H(omega) of length N.

    H[k] = exp(-i * beta2_L/2 * omega[k]^2)

    where omega is the centred angular-frequency axis (fftshift order).

    Parameters
    ----------
    N : int
        Number of samples.
    dt : float
        Sample spacing in seconds (or metres).
    beta2_L : float
        Accumulated GVD (s²/rad or m²).

    Returns
    -------
    H : ndarray, complex128, shape (N,)
        Transfer function in centred (fftshift) frequency order.
    """
    domega = 2.0 * np.pi / (N * dt)
    omega = (np.arange(N) - N // 2) * domega
    return np.exp(-1j * 0.5 * beta2_L * omega ** 2)


def propagate(A: np.ndarray, t_axis: np.ndarray, beta2_L: float) -> np.ndarray:
    """Propagate complex envelope A through a dispersive medium.

    Parameters
    ----------
    A : array_like, complex, shape (N,)
        Input complex field (time-domain or spatial-domain).
    t_axis : array_like, real, shape (N,)
        Uniformly-spaced coordinate axis (seconds or metres).
        Must satisfy ``np.diff(t_axis)`` ≈ constant.
    beta2_L : float
        Accumulated GVD in s²/rad  (fibre: beta2 * L;
        quantum: hbar * T / m_electron).
        Sign convention: positive beta2_L adds normal dispersion.

    Returns
    -------
    A_out : ndarray, complex, shape (N,)
        Dispersed field in the same domain as the input.

    Notes
    -----
    The transform uses the centred-FFT convention (fftshift/ifftshift)
    so that DC sits at the array centre, matching v10 notebook convention.

    Round-trip check
    ----------------
    >>> import numpy as np
    >>> from simulator.dispersion import propagate
    >>> N, dt = 4096, 1e-12
    >>> t = np.arange(N) * dt
    >>> A = np.exp(-((t - t.mean()) / (50e-12))**2).astype(complex)
    >>> A2 = propagate(propagate(A, t, 1e-22), t, -1e-22)
    >>> assert np.max(np.abs(A2 - A)) < 1e-10
    """
    A = np.asarray(A, dtype=complex)
    if A.ndim != 1:
        raise ValueError(f"A must be 1-D, got shape {A.shape}. Use batch_propagate for batches.")
    N, dt = _validated_axis(t_axis, len(A))

    H = transfer_function(N, dt, beta2_L)
    A_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))
    A_out = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A_fft * H)))
    return A_out


def batch_propagate(
    A: np.ndarray,
    t_axis: np.ndarray,
    beta2_L: float | np.ndarray,
) -> np.ndarray:
    """Propagate a batch of signals through one or more dispersive channels.

    Supports three calling modes:

    1. Many signals, one dispersion:
       A shape (B, N), beta2_L scalar → output shape (B, N)

    2. One signal, many dispersions:
       A shape (N,), beta2_L shape (K,) → output shape (K, N)

    3. Many signals, many dispersions (element-wise):
       A shape (B, N), beta2_L shape (B,) → output shape (B, N)

    Parameters
    ----------
    A : array_like, complex
        Input field(s). Last axis is the time/space axis of length N.
    t_axis : array_like, real, shape (N,)
        Uniformly-spaced coordinate axis.
    beta2_L : float or array_like of float
        Accumulated GVD value(s).

    Returns
    -------
    A_out : ndarray, complex
        Dispersed field(s) with same trailing shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> from simulator.dispersion import batch_propagate
    >>> N, dt = 512, 1e-12
    >>> t = np.arange(N) * dt
    >>> A = np.exp(-((t - t.mean()) / (50e-12))**2).astype(complex)
    >>> # Sweep 8 dispersion values in one call
    >>> betas = np.linspace(-1e-22, 1e-22, 8)
    >>> out = batch_propagate(A, t, betas)   # shape (8, 512)
    >>> assert out.shape == (8, N)
    """
    A = np.asarray(A, dtype=complex)
    beta2_L = np.asarray(beta2_L, dtype=float)
    scalar_beta = beta2_L.ndim == 0

    if A.ndim == 1:
        N, dt = _validated_axis(t_axis, len(A))
        if scalar_beta:
            return propagate(A, t_axis, float(beta2_L))
        # One signal, sweep dispersions
        A_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))  # (N,)
        domega = 2.0 * np.pi / (N * dt)
        omega = (np.arange(N) - N // 2) * domega                  # (N,)
        # H: (K, N)
        H = np.exp(-1j * 0.5 * beta2_L[:, None] * omega[None, :] ** 2)
        A_out = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(A_fft[None, :] * H, axes=-1), axis=-1),
            axes=-1,
        )
        return A_out  # (K, N)

    if A.ndim == 2:
        B, N = A.shape
        _, dt = _validated_axis(t_axis, N)
        domega = 2.0 * np.pi / (N * dt)
        omega = (np.arange(N) - N // 2) * domega  # (N,)

        A_fft = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(A, axes=-1), axis=-1), axes=-1
        )  # (B, N)

        if scalar_beta:
            H = np.exp(-1j * 0.5 * float(beta2_L) * omega ** 2)  # (N,)
            A_out = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(A_fft * H[None, :], axes=-1), axis=-1),
                axes=-1,
            )
        else:
            if beta2_L.shape != (B,):
                raise ValueError(
                    f"beta2_L shape {beta2_L.shape} incompatible with A batch size {B}"
                )
            H = np.exp(-1j * 0.5 * beta2_L[:, None] * omega[None, :] ** 2)  # (B, N)
            A_out = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(A_fft * H, axes=-1), axis=-1),
                axes=-1,
            )
        return A_out  # (B, N)

    raise ValueError(f"A must be 1-D or 2-D, got shape {A.shape}")
