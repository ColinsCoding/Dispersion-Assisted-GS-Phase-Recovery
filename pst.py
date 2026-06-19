"""Phase Stretch Transform (PST) edge detection -- the Jalali Lab's PhyCV, from scratch.

PST is the imaging cousin of this repo's receiver. The receiver applies a
dispersive spectral phase H(f)=exp(i pi D f^2) to a pulse and then *recovers* the
phase; PST applies an engineered (warped) spectral phase to an *image* and then
*reads out* that phase as edges. Both are "multiply by a frequency-dependent
phase, then look at the phase" -- the same move, used two ways.

Pipeline (Asghari & Jalali 2015):
    image -> FFT2 -> x localization low-pass L(rho) -> x phase kernel e^{i phi(rho)}
          -> IFFT2 -> angle() = edge map (sharp = high spatial frequency = most phase).

NumPy only; civilian computer vision (the lab uses it for medical imaging,
low-light enhancement, general edge detection). Education.
"""

import numpy as np


def _radial_freq(ny, nx):
    """Normalized radial spatial-frequency grid in [0, 1] (fftshifted-aware via fftfreq)."""
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    rho = np.sqrt(FX**2 + FY**2)
    return rho / rho.max()


def pst_phase_kernel(ny, nx, warp=15.0, strength=0.48):
    """The PST warped-phase kernel phi(rho) (the integral of arctan).

    phi = S * [ W*rho*atan(W*rho) - 0.5*ln(1+(W*rho)^2) ], normalized so max=S.
    Its phase *derivative* grows with frequency, so sharp features (high rho) get
    the largest phase -- that is what makes them pop out as edges.
    """
    if warp <= 0 or strength <= 0:
        raise ValueError("warp and strength must be > 0")
    Wr = warp * _radial_freq(ny, nx)
    phi = Wr * np.arctan(Wr) - 0.5 * np.log1p(Wr**2)
    return strength * phi / phi.max()


def localization_kernel(ny, nx, sigma=0.12):
    """Gaussian low-pass L(rho)=exp(-rho^2/2sigma^2): denoise / localize before PST."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    rho = _radial_freq(ny, nx)
    return np.exp(-(rho**2) / (2 * sigma**2))


def _pst_transform(image, warp=15.0, strength=0.48, sigma=0.12):
    """The complex PST output: IFFT2( FFT2(image) * L(rho) * e^{i phi(rho)} )."""
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    F = np.fft.fft2(image)
    L = localization_kernel(ny, nx, sigma)
    K = np.exp(1j * pst_phase_kernel(ny, nx, warp, strength))
    return np.fft.ifft2(F * L * K)


def pst(image, warp=15.0, strength=0.48, sigma=0.12):
    """Apply the Phase Stretch Transform; returns the phase edge map (radians)."""
    return np.angle(_pst_transform(image, warp, strength, sigma))


def pst_edges(image, thresh=0.5, mag_floor=0.02, warp=15.0, strength=0.48, sigma=0.12):
    """Binary edge map. The phase is only meaningful where the transform has
    signal, so mask out near-zero-magnitude regions (flat zones give random
    wrapped phase) before thresholding |phase|."""
    out = _pst_transform(image, warp, strength, sigma)
    phase, mag = np.abs(np.angle(out)), np.abs(out)
    p = np.where(mag > mag_floor * mag.max(), phase, 0.0)
    return (p > thresh * p.max()).astype(int)


if __name__ == "__main__":
    yy, xx = np.mgrid[0:128, 0:128]
    disk = ((xx - 64)**2 + (yy - 64)**2 < 30**2).astype(float)
    edges = pst_edges(disk)
    print(f"disk image {disk.shape}: PST found {edges.sum()} edge pixels "
          f"(circle circumference ~ {2*np.pi*30:.0f})")
