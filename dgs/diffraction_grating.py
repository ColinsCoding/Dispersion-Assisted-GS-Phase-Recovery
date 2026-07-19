"""N-slit Fraunhofer diffraction grating -- the amplitude/intensity split that runs
through this whole repo (GS phase recovery, photonic_circuits, fdtd) shows up here
in its cleanest classical form: a grating is N coherent point sources, and the
FAR-FIELD AMPLITUDE is just their phasor sum. Intensity is what a detector sees
(|amplitude|^2, phase thrown away) -- the same measurement bottleneck the
dispersion-assisted GS receiver exists to work around.

    single_slit envelope:  A_slit(theta) = sinc(beta),   beta = pi a sin(theta) / lambda
    N-slit array factor:   A_array(theta) = sin(N gamma) / (N sin gamma),  gamma = pi d sin(theta)/lambda
    intensity:             I(theta) = |A_slit|^2 * |A_array|^2

A_array is a normalized Dirichlet kernel: it peaks at 1 exactly where the grating
equation d*sin(theta) = m*lambda holds (principal maxima), and is suppressed toward
0 everywhere else -- more slits (N) makes those peaks sharper, exactly like more
turns in dgs.photonic_circuits' ring resonator makes its resonance sharper (both
are "how many times does the wave interfere with itself" problems).

NumPy only. Education.
"""

import numpy as np


def _sinc_unnormalized(x):
    """sin(x)/x with the x=0 -> 1 limit filled in (NOT numpy's np.sinc, which
    uses the pi*x-normalized convention)."""
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))


def single_slit_envelope(theta, a, wavelength):
    """Single-slit diffraction envelope |sinc(beta)|^2, beta = pi*a*sin(theta)/lambda.
    Zero whenever a*sin(theta) = m*lambda (m != 0) -- the aperture's own diffraction
    minima, which the grating's sharp interference peaks live inside."""
    if a <= 0 or wavelength <= 0:
        raise ValueError("slit width a and wavelength must be positive")
    beta = np.pi * a * np.sin(theta) / wavelength
    return _sinc_unnormalized(beta) ** 2


def grating_interference_factor(theta, d, N, wavelength):
    """Normalized N-slit array factor sin(N*gamma)/(N*sin(gamma)), gamma = pi*d*sin(theta)/lambda.
    Equals +-1 (so its square is exactly 1, a principal maximum) whenever
    gamma = m*pi, i.e. the grating equation d*sin(theta) = m*lambda -- filled in
    by the sin(gamma)->0 limit rather than evaluated as 0/0."""
    if d <= 0 or wavelength <= 0:
        raise ValueError("slit spacing d and wavelength must be positive")
    if N < 1:
        raise ValueError(f"N (slit count) must be >= 1, got {N}")
    gamma = np.pi * d * np.sin(theta) / wavelength
    sin_gamma = np.sin(gamma)
    near_zero = np.abs(sin_gamma) < 1e-9
    safe_denom = np.where(near_zero, 1.0, N * sin_gamma)
    return np.where(near_zero, 1.0, np.sin(N * gamma) / safe_denom)


def grating_intensity(theta, d, a, N, wavelength):
    """Full N-slit Fraunhofer pattern I(theta) = envelope(theta) * |array_factor(theta)|^2,
    normalized so the central maximum (theta=0) equals 1."""
    return single_slit_envelope(theta, a, wavelength) * grating_interference_factor(theta, d, N, wavelength) ** 2


def principal_maxima_angles(d, wavelength, m_max):
    """Solve the grating equation d*sin(theta) = m*lambda for integer orders
    m in [-m_max, m_max] that are physically realizable (|m*lambda/d| <= 1).
    Returns (orders, angles_rad)."""
    if d <= 0 or wavelength <= 0:
        raise ValueError("d and wavelength must be positive")
    m = np.arange(-m_max, m_max + 1)
    sin_theta = m * wavelength / d
    valid = np.abs(sin_theta) <= 1.0
    return m[valid], np.arcsin(sin_theta[valid])


def angle_to_screen_position(theta, L):
    """Small-far-field-angle projection of diffraction angle theta onto a flat
    screen at distance L: x = L * tan(theta) (what a camera/eye actually sees)."""
    if L <= 0:
        raise ValueError("screen distance L must be positive")
    return L * np.tan(theta)


def wavelength_to_rgb(wavelength_nm):
    """Approximate visible-spectrum wavelength -> (R,G,B) in [0,1], the standard
    piecewise-linear CIE-adjacent mapping used across optics visualizations
    (Dan Bruton's public-domain algorithm). 380-750 nm; outside that range the
    color fades to black since it's outside human vision."""
    w = float(wavelength_nm)
    if 380 <= w < 440:
        r, g, b = -(w - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= w < 490:
        r, g, b = 0.0, (w - 440) / (490 - 440), 1.0
    elif 490 <= w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r, g, b = (w - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= w < 645:
        r, g, b = 1.0, -(w - 645) / (645 - 580), 0.0
    elif 645 <= w <= 750:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 0.0, 0.0, 0.0

    if 380 <= w < 420:
        factor = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif 420 <= w <= 700:
        factor = 1.0
    elif 700 < w <= 750:
        factor = 0.3 + 0.7 * (750 - w) / (750 - 700)
    else:
        factor = 0.0

    return tuple(max(0.0, min(1.0, c * factor)) for c in (r, g, b))


if __name__ == "__main__":
    d, a, N, wavelength = 2.0e-6, 0.4e-6, 8, 589e-9  # sodium D line, 500 nm-ish grating
    m, theta = principal_maxima_angles(d, wavelength, m_max=4)
    print("orders:", m)
    print("angles (deg):", np.round(np.degrees(theta), 2))

    theta_grid = np.linspace(-np.pi / 2, np.pi / 2, 20001)
    I = grating_intensity(theta_grid, d, a, N, wavelength)
    print(f"peak intensity = {I.max():.4f} at theta = {np.degrees(theta_grid[np.argmax(I)]):.3f} deg")
    print(f"RGB for {wavelength*1e9:.0f} nm: {wavelength_to_rgb(wavelength*1e9)}")
