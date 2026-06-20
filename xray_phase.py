"""Helical X-ray diffraction and the crystallographic phase problem.

The Fourier transform of a helix is the Cochran-Crick-Vand "X" -- layer lines
whose radial amplitude is a Bessel function J_n -- the pattern that revealed
DNA's double helix (Photo 51). Diffraction measures |F|^2 only, so the phase is
lost: the *phase problem*. It is solved by exactly the alternating-projection
idea behind this repo's time-domain Gerchberg-Saxton, here iterating between the
measured Fourier magnitude and a real-space support constraint (Fienup HIO).
Standalone (imaging/optics).
"""

import numpy as np


# ── build a (double) helix projection in real space ─────────────────
def helix_density(N=160, pitch=26, radius=14, n_strands=2, offset=0.375,
                  sigma=1.6, n_units=None):
    """Projected electron density of a helix (or double helix) in the x-z plane.

    A helix viewed perpendicular to its axis projects to a sinusoid; placing
    density along it and Fourier-transforming gives the helical diffraction. The
    double-strand offset ~3/8 reproduces B-DNA's characteristic missing 4th
    layer line. Returns an (N, N) real image.
    """
    if not 0 < radius < N / 2:
        raise ValueError("radius must be in (0, N/2)")
    img = np.zeros((N, N))
    z = np.arange(N)
    xx = np.arange(N)
    for strand in range(n_strands):
        ph = 2 * np.pi * strand * offset
        xc = N / 2 + radius * np.cos(2 * np.pi * z / pitch + ph)
        for zi in z:
            img[zi] += np.exp(-(xx - xc[zi])**2 / (2 * sigma**2))
    return img


# ── diffraction (the measurement) ───────────────────────────────────
def diffraction(rho):
    """Far-field diffraction intensity |F|^2 (centered)."""
    F = np.fft.fftshift(np.fft.fft2(rho))
    return np.abs(F)**2


def amplitude(rho):
    """Measured Fourier amplitude |F| (centered) -- what a detector records."""
    return np.abs(np.fft.fftshift(np.fft.fft2(rho)))


# ── Cochran-Crick-Vand layer lines (the Bessel story) ───────────────
def ccv_layer_intensity(n, radius, R):
    """Radial intensity on the n-th layer line of a continuous helix:
    I_n(R) proportional to J_n(2 pi radius R)^2. Uses mpmath for J_n."""
    import mpmath as mp
    R = np.atleast_1d(np.asarray(R, dtype=float))
    return np.array([float(mp.besselj(n, 2 * np.pi * radius * r))**2 for r in R])


# ── the phase problem: Fienup HIO phase retrieval ───────────────────
def hio_phase_retrieval(mag, support, n_iter=300, beta=0.9, seed=0):
    """Recover a real, non-negative object from its Fourier magnitude |F| and a
    real-space support, by the hybrid input-output (HIO) algorithm.

    mag      : measured Fourier amplitude (centered, as from `amplitude`)
    support  : boolean mask, True where the object may be nonzero (oversampling
               makes the phase solution unique)
    Returns (object_estimate, fourier_errors).
    """
    if mag.shape != support.shape:
        raise ValueError("mag and support must share shape")
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    rng = np.random.default_rng(seed)
    G = mag * np.exp(1j * rng.uniform(0, 2 * np.pi, mag.shape))
    g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    errors = []
    for _ in range(n_iter):
        G = np.fft.fftshift(np.fft.fft2(g))
        errors.append(float(np.linalg.norm(np.abs(G) - mag) / np.linalg.norm(mag)))
        Gp = mag * np.exp(1j * np.angle(G))              # enforce measured |F|
        gp = np.real(np.fft.ifft2(np.fft.ifftshift(Gp)))
        violate = (~support) | (gp < 0)                  # support + positivity
        g = np.where(violate, g - beta * gp, gp)         # HIO update
    # final error-reduction polish + clean support projection
    G = np.fft.fftshift(np.fft.fft2(g))
    Gp = mag * np.exp(1j * np.angle(G))
    g = np.real(np.fft.ifft2(np.fft.ifftshift(Gp)))
    g = np.where(support & (g > 0), g, 0.0)
    return g, errors


def best_alignment_corr(recovered, truth):
    """Correlation of recovered vs truth, maximized over the translation and
    inversion (twin-image) ambiguities of phase retrieval."""
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        c = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
        return c.max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return max(corr(recovered, truth), corr(recovered, truth[::-1, ::-1]))
