"""Fourier optics: the far-field is the Fourier transform of the aperture.

In the Fraunhofer (far-field) limit, the diffracted amplitude is the FOURIER TRANSFORM of
the aperture, and a detector reads its squared magnitude. That one fact turns every wave
theorem into an optics theorem -- this module makes three of them concrete:

  1. RECIPROCAL SCALING. A narrow aperture diffracts WIDE and a wide one diffracts narrow:
     width * sin(theta_null) = lambda always, so shrinking the slit spreads the light.
     This is the Fourier scaling theorem = the diffraction "uncertainty relation".

  2. SHIFT-INVARIANT INTENSITY. Sliding the aperture sideways multiplies its transform by
     a pure phase e^{-i 2 pi f x0}, which leaves |FT|^2 -- the measured intensity --
     UNCHANGED. You cannot tell where the aperture was from its diffraction pattern. That
     lost position IS the lost phase: it is exactly why intensity-only measurement needs
     phase retrieval (dgs.gs_core), the whole point of this repo.

  3. CONVOLUTION -> MULTIPLE SLITS. A double slit is one slit CONVOLVED with two delta
     functions, so by the convolution theorem its pattern is the single-slit envelope
     (a sinc^2) TIMES an interference term cos^2(pi d sin(theta)/lambda): fine fringes at
     spacing sin(theta) = m lambda/d under the broad single-slit envelope. N slits (a
     grating) sharpen those fringes as ~1/N.

Complements dgs.diffraction (single-slit analytics + the basic FFT far-field) with the
transform theorems. Verified: the far-field of a slit matches the sinc^2 formula, the
reciprocal scaling, the exact shift-invariance of the intensity, and the double-slit
fringe spacing / grating sharpening. NumPy only; py-3.13.
"""

import numpy as np
from dgs.diffraction import single_slit_intensity


def make_grid(n, dx):
    """A centered spatial grid of n points spaced dx (units of length)."""
    if n < 8 or dx <= 0:
        raise ValueError("need n >= 8 and dx > 0")
    return (np.arange(n) - n // 2) * dx


def rect_aperture(x, width):
    """A single slit of the given width centered at 0 (1 inside, 0 outside)."""
    if width <= 0:
        raise ValueError("width must be positive")
    return (np.abs(x) <= width / 2).astype(float)


def double_slit(x, slit_width, separation):
    """Two slits of slit_width, centers +/- separation/2 -- one slit convolved with
    two deltas."""
    if slit_width <= 0 or separation <= 0:
        raise ValueError("slit_width and separation must be positive")
    left = np.abs(x + separation / 2) <= slit_width / 2
    right = np.abs(x - separation / 2) <= slit_width / 2
    return (left | right).astype(float)


def grating(x, slit_width, period, n_slits):
    """An N-slit grating: slits of slit_width on a `period` pitch, centered."""
    if slit_width <= 0 or period <= 0 or n_slits < 1:
        raise ValueError("slit_width, period > 0 and n_slits >= 1")
    centers = (np.arange(n_slits) - (n_slits - 1) / 2) * period
    ap = np.zeros_like(x)
    for c in centers:
        ap[np.abs(x - c) <= slit_width / 2] = 1.0
    return ap


def far_field(aperture, dx, lam):
    """Fraunhofer far field: complex amplitude (the FT of the aperture), the
    normalized intensity |A|^2, and the diffraction angle theta for each sample.
    Returns (theta, amplitude, intensity)."""
    aperture = np.asarray(aperture, float)
    A = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(aperture)))
    fx = np.fft.fftshift(np.fft.fftfreq(len(aperture), dx))
    theta = np.arcsin(np.clip(lam * fx, -1, 1))
    I = np.abs(A) ** 2
    return theta, A, I / I.max()


def first_null_angle(width, lam):
    """Angle of the first diffraction minimum of a slit: sin(theta) = lambda/width.
    The reciprocal-scaling law -- narrower slit, wider spread."""
    if width <= 0 or lam <= 0:
        raise ValueError("width and lam must be positive")
    r = lam / width
    if r >= 1:
        raise ValueError("lambda/width >= 1: no real first null (slit too narrow)")
    return np.arcsin(r)


def scaling_product(width, lam):
    """width * sin(theta_null) -- equals lambda for ANY slit width (Fourier scaling)."""
    return width * np.sin(first_null_angle(width, lam))


def intensity_is_shift_invariant(aperture, shift_samples):
    """Slide the aperture by an integer number of samples and confirm the far-field
    INTENSITY is unchanged (only the phase moves). Returns True -- the statement that
    a diffraction pattern cannot locate its aperture (the lost-phase / phase-retrieval
    fact)."""
    A0 = np.abs(np.fft.fft(np.asarray(aperture, float)))
    A1 = np.abs(np.fft.fft(np.roll(aperture, int(shift_samples))))
    return bool(np.allclose(A0, A1))


def double_slit_fringe_angle(separation, lam, order=1):
    """Interference maxima of a double slit: sin(theta) = m lambda / separation."""
    if separation <= 0 or lam <= 0:
        raise ValueError("separation and lam must be positive")
    r = order * lam / separation
    if abs(r) >= 1:
        raise ValueError("order lambda/separation >= 1: that maximum is evanescent")
    return np.arcsin(r)


def grating_maxima(period, lam, max_order=3):
    """Principal maxima of a grating: sin(theta) = m lambda / period, m = 0..max_order
    (only the real, propagating orders). The grating equation."""
    if period <= 0 or lam <= 0:
        raise ValueError("period and lam must be positive")
    angles = []
    for m in range(max_order + 1):
        r = m * lam / period
        if abs(r) <= 1:
            angles.append(np.arcsin(r))
    return angles


if __name__ == "__main__":
    lam = 600e-9
    x = make_grid(400000, 5e-8)      # 20 mm window, fine sampling

    print("1. reciprocal scaling (width * sin(theta_null) = lambda):")
    for w in (20e-6, 50e-6, 100e-6):
        print(f"   slit {w*1e6:.0f} um: first null at {np.degrees(first_null_angle(w, lam)):.3f} deg,"
              f"  width*sin = {scaling_product(w, lam)/lam:.4f} lambda")

    print("\n2. shift-invariant intensity (can't locate the aperture):")
    ap = rect_aperture(x, 50e-6)
    print(f"   |FT| unchanged when slit slid 5000 samples? "
          f"{intensity_is_shift_invariant(ap, 5000)}  -> the lost phase = phase retrieval")

    print("\n3. double slit = single-slit envelope x fringes:")
    d = 200e-6
    print(f"   fringe maxima (d={d*1e6:.0f} um) at "
          f"{[round(np.degrees(double_slit_fringe_angle(d, lam, m)),3) for m in range(1,4)]} deg")
    th, _, I = far_field(double_slit(x, 50e-6, d), x[1]-x[0], lam)
    a1 = double_slit_fringe_angle(d, lam, 1)             # a bright fringe
    a_dark = np.arcsin(0.5 * lam / d)                    # the dark fringe before it
    Ib = np.interp(a1, th, I); Id = np.interp(a_dark, th, I)
    print(f"   I at 1st bright fringe {Ib:.3f} >> I at dark fringe {Id:.4f}? {Ib > 10*Id}")

    print("\ngrating orders (period 2 um):",
          [round(np.degrees(a), 2) for a in grating_maxima(2e-6, lam)])
