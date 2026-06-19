"""Gaussian beams -- the wave that isn't a plane wave (real laser light).

A plane wave is an idealization: infinite, flat wavefronts, never diffracting.
A real laser beam is a *Gaussian beam* -- a paraxial solution of the wave
equation with a finite waist that spreads as it propagates. It is the spatial
twin of this repo's dispersing pulse: a beam diffracting in z is a pulse
dispersing in t (space-time duality), both governed by the same Schrodinger-like
paraxial equation.

Key quantities (waist w0 at z=0, wavelength lam):
    Rayleigh range  zR = pi w0^2 / lam        (focus depth)
    spot size       w(z) = w0 sqrt(1+(z/zR)^2)
    curvature       R(z) = z (1+(zR/z)^2)
    Gouy phase      psi(z) = arctan(z/zR)      (the extra pi/2 through a focus)
    complex param   q(z) = z + i zR,  1/q = 1/R - i lam/(pi w^2)
and ABCD ray-matrix propagation acts on q by q' = (A q + B)/(C q + D).

NumPy only. Civilian photonics / education.
"""

import numpy as np


def rayleigh_range(w0, wavelength):
    """zR = pi w0^2 / lambda: the half-length of the focal region."""
    if w0 <= 0 or wavelength <= 0:
        raise ValueError("w0 and wavelength must be > 0")
    return np.pi * w0**2 / wavelength


def beam_width(z, w0, wavelength):
    """Spot size w(z) = w0 sqrt(1 + (z/zR)^2): minimum w0 at the waist, grows away."""
    zR = rayleigh_range(w0, wavelength)
    return w0 * np.sqrt(1 + (np.asarray(z, dtype=float) / zR)**2)


def radius_of_curvature(z, w0, wavelength):
    """Wavefront radius R(z) = z(1+(zR/z)^2): infinite (flat) at the waist."""
    z = np.asarray(z, dtype=float)
    zR = rayleigh_range(w0, wavelength)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z == 0, np.inf, z * (1 + (zR / z)**2))


def gouy_phase(z, w0, wavelength):
    """Gouy phase psi(z) = arctan(z/zR): the anomalous pi phase slip through focus."""
    zR = rayleigh_range(w0, wavelength)
    return np.arctan(np.asarray(z, dtype=float) / zR)


def divergence(w0, wavelength):
    """Far-field half-angle theta = lambda/(pi w0): tighter waist -> faster spread."""
    if w0 <= 0 or wavelength <= 0:
        raise ValueError("w0 and wavelength must be > 0")
    return wavelength / (np.pi * w0)


def q_parameter(z, w0, wavelength):
    """Complex beam parameter q(z) = z + i zR (encodes width AND curvature at once)."""
    return np.asarray(z, dtype=float) + 1j * rayleigh_range(w0, wavelength)


def width_curvature_from_q(q, wavelength):
    """Recover (w, R) from q via 1/q = 1/R - i lambda/(pi w^2). R=inf if 1/q real-part 0."""
    inv = 1.0 / q
    R = np.inf if abs(inv.real) < 1e-300 else 1.0 / inv.real
    w = np.sqrt(-wavelength / (np.pi * inv.imag))
    return w, R


def abcd_propagate(q, A, B, C, D):
    """Propagate the complex beam parameter through an ABCD optical element:
    q' = (A q + B) / (C q + D). Free space of length d is [[1,d],[0,1]]; a thin
    lens of focal length f is [[1,0],[-1/f,1]]."""
    return (A * q + B) / (C * q + D)


if __name__ == "__main__":
    w0, lam = 0.5e-3, 1.55e-6          # 0.5 mm waist, 1550 nm
    zR = rayleigh_range(w0, lam)
    print(f"waist {w0*1e3:.2f} mm, lambda {lam*1e9:.0f} nm -> Rayleigh range {zR:.3f} m")
    print(f"at z=zR: w={beam_width(zR, w0, lam)*1e3:.3f} mm (= w0*sqrt2), "
          f"Gouy={np.degrees(gouy_phase(zR, w0, lam)):.0f} deg")
    print(f"far-field divergence: {divergence(w0, lam)*1e3:.3f} mrad")
