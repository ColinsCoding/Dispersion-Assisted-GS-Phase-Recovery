"""Single-slit diffraction and resolving power -- Fourier optics.

Light through a slit of width a spreads into a Fraunhofer (far-field) pattern
    I(theta) = I0 * (sin beta / beta)^2 ,   beta = pi a sin(theta) / lambda,
a bright central lobe with dark minima where a sin(theta) = m lambda. The deep fact:
the far-field pattern IS the Fourier transform of the aperture -- a rectangular slit
transforms to a sinc, squared to an intensity. That is the same FFT that runs the
dispersion operator and the GS receiver in this repo; diffraction is Fourier optics.

A narrower slit gives a WIDER pattern (the spatial version of the uncertainty
reciprocity in dgs.uncertainty), and two point sources are just resolvable when their
separation reaches the Rayleigh limit theta_min ~ lambda/D -- the resolving power that
sets every telescope's and microscope's finest detail. NumPy. Education.
"""

import numpy as np


def single_slit_intensity(theta, a, lam, I0=1.0):
    """Fraunhofer single-slit intensity I(theta) = I0 (sin beta/beta)^2,
    beta = pi a sin(theta)/lambda. Central maximum at theta=0; minima where
    a sin(theta) = m lambda."""
    beta = np.pi * a * np.sin(np.asarray(theta, float)) / lam
    return I0 * np.sinc(beta / np.pi) ** 2      # np.sinc(x)=sin(pi x)/(pi x) -> sin(beta)/beta


def single_slit_minima(a, lam, m_max=3):
    """Angles (rad) of the diffraction minima: sin(theta) = m lambda/a, m = 1..m_max
    (only those with m lambda/a <= 1 exist)."""
    return [float(np.arcsin(m * lam / a)) for m in range(1, m_max + 1) if m * lam / a <= 1.0]


def central_lobe_halfwidth(a, lam):
    """Angular half-width of the central maximum = arcsin(lambda/a) (the first minimum).
    Narrower slit (smaller a) -> wider lobe: the diffraction-uncertainty reciprocity."""
    return float(np.arcsin(min(lam / a, 1.0)))


def rayleigh_resolution(lam, D, circular=False):
    """Rayleigh angular resolution: the smallest resolvable separation. theta_min =
    lambda/D for a slit, 1.22 lambda/D for a circular aperture of diameter D. Smaller
    wavelength or larger aperture -> finer detail (the resolving power)."""
    return (1.22 if circular else 1.0) * lam / D


def aperture_diffraction(aperture, dx, lam):
    """Far-field pattern from ANY aperture as |FFT(aperture)|^2 -- the diffraction
    pattern is the Fourier transform of the aperture. Returns (theta, intensity) with
    intensity normalized to 1 at the peak."""
    aperture = np.asarray(aperture, float)
    F = np.fft.fftshift(np.fft.fft(aperture))
    I = np.abs(F) ** 2
    I = I / I.max()
    fx = np.fft.fftshift(np.fft.fftfreq(len(aperture), dx))   # spatial frequency
    theta = np.arcsin(np.clip(lam * fx, -1, 1))
    return theta, I


if __name__ == "__main__":
    a, lam = 50e-6, 600e-9          # 50 um slit, 600 nm light
    print(f"central peak I(0) = {single_slit_intensity(0.0, a, lam):.3f} (= I0)")
    mins = single_slit_minima(a, lam)
    print(f"first minima at theta = {[round(np.degrees(t),3) for t in mins]} deg "
          f"(sin theta = m lambda/a)")
    print(f"central-lobe half-width = {np.degrees(central_lobe_halfwidth(a, lam)):.3f} deg")
    print(f"Rayleigh resolution (D=10mm): slit {np.degrees(rayleigh_resolution(lam,1e-2))*3600:.2f} arcsec, "
          f"circular {np.degrees(rayleigh_resolution(lam,1e-2,True))*3600:.2f} arcsec")
    # diffraction = FFT of the aperture: build a slit, transform, compare to the formula
    x = np.linspace(-5e-3, 5e-3, 200000); dx = x[1] - x[0]
    ap = (np.abs(x) < a / 2).astype(float)
    th, I = aperture_diffraction(ap, dx, lam)
    Iformula = single_slit_intensity(th, a, lam)
    core = np.abs(th) < np.radians(3)
    print(f"FFT pattern vs sinc^2 formula: max diff in core = {np.max(np.abs(I-Iformula)[core]):.2e}")
