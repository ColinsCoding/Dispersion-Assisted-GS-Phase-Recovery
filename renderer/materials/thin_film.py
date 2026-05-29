"""
thin_film.py — Thin film interference for dog surface iridescence.

Physics (Griffiths EM ch9 + Michelson):
  2*n*d*cos(theta) = m*lambda   constructive
  delta_phi = 4*pi*n*d*cos(theta) / lambda

Same complex exponential as H(nu) = exp(i*pi*D*nu^2)
Just spatial instead of temporal.
"""
import numpy as np


def thin_film_rgb(theta_deg: float,
                  d_nm: float     = 200.0,
                  n_film: float   = 1.56) -> np.ndarray:
    """
    RGB color from thin film interference at viewing angle theta.
    d_nm     : film thickness in nanometres (chitin ~ 200nm)
    n_film   : refractive index of film (chitin = 1.56)
    Returns  : RGB in [0,1]
    """
    theta = np.radians(theta_deg)
    # Wavelength samples for R G B channels
    lam = np.array([650.0, 532.0, 450.0])   # nm  R G B

    # Optical path difference
    opd  = 2 * n_film * d_nm * np.cos(theta)

    # Phase difference
    delta = 2 * np.pi * opd / lam

    # Intensity: two-beam interference (air-film-air)
    r_top = (1 - n_film) / (1 + n_film)       # Fresnel r at top surface
    r_bot = (n_film - 1) / (n_film + 1)       # Fresnel r at bottom

    I = (r_top**2 + r_bot**2 + 2*r_top*r_bot*np.cos(delta))
    I = np.clip(I / I.max(), 0, 1)
    return I                                   # [R, G, B]


def iridescence_map(theta_range=(0, 80), d_nm=200, n=1.56,
                    steps=256) -> np.ndarray:
    """Precompute iridescence LUT for a range of viewing angles."""
    angles = np.linspace(*theta_range, steps)
    return np.array([thin_film_rgb(a, d_nm, n) for a in angles])
