"""SEALS hardware, as published: Adam, Mahjoubfar, Diebold, Buckley & Jalali,
"Spectrally encoded angular light scattering," Opt. Express 21(23), 28960 (2013).

This module collects the paper's hardware specifications (with section
references), derives the design equations in SymPy, and evaluates them with
the tested physics in scarlett.physics. It implements the PUBLISHED equations;
it does not copy the lab's MATLAB. Nothing here is measured data.

Design chain (paper Fig. 1):
  mode-locked laser (1590 nm, ~20 nm) -> optical equalization filter
  -> grating pair (groove spacing d, separation D, incidence alpha)   Eq. (1): lam -> y
  -> telescope (two parabolic mirrors, M = 1/6)
  -> objective (60x, NA 0.7, entrance pupil P)                         Eq. (2): y -> theta
  -> particle -> multimode fiber at a fixed angle -> OSA (or TS-DFT)
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.signal import argrelextrema

from scarlett import physics as ph

# ── published specifications ─────────────────────────────────────────
PAPER_SPECS = [
    ("source", "pulsed mode-locked laser, broadband, spatially coherent", "", "Sec. 2"),
    ("center wavelength lam0", 1590, "nm", "Sec. 2"),
    ("optical bandwidth", 20, "nm (1.3 % fractional)", "Sec. 2"),
    ("equalization", "tunable optical filter + quarter/half-wave plates", "", "Sec. 2"),
    ("grating pair", "parallel, equal groove spacing, 1st order (m = 1)", "", "Sec. 3"),
    ("grating pitch", 1100, "as printed 'd = 1100 nm' (see reconciliation)", "Sec. 3"),
    ("telescope magnification M", 1 / 6, "", "Sec. 3"),
    ("objective", "60x, NA = 0.7", "", "Sec. 2"),
    ("entrance beam waist w0", 3, "mm ('measures 3 mm')", "Sec. 3"),
    ("detection", "multimode fiber at one fixed angle -> OSA", "", "Sec. 2"),
    ("fiber core / detector aperture", "50 - 100", "um", "Sec. 3"),
    ("probe-to-target distance L", "2 - 10", "mm", "Sec. 3"),
    ("angular resolution", "~1", "deg", "Sec. 3, Fig. 4"),
    ("wavelength resolvability", 0.48, "nm -> 42 resolvable angles", "Sec. 3"),
    ("beads", "polystyrene, 7.32 um and 9.94 um, n = 1.39, dried on 140 um glass", "", "Sec. 3"),
    ("measured mapping", "~ +41 deg ... -27 deg over 1582 - 1598 nm (read from Fig. 3)", "", "Fig. 3"),
    ("measured displacement", "~ 5.3 mm over 1580 - 1600 nm (read from Fig. 2)", "", "Fig. 2"),
    ("high-throughput path", "time-stretch dispersive Fourier transform (DCF/FBG/CMD + Raman gain)", "", "Sec. 4"),
]

# Parameter set used in the lab simulation's defaults (a fitted configuration)
DEFAULTS = dict(groove_spacing=9.0909e-7, grating_distance=0.065, grating_angle=0.9023,
                dcorr=-4.2448e-4, pupil_diameter=0.0058, numerical_aperture=0.7)
LAM_MIN, LAM_MAX, LAM0 = 1580e-9, 1600e-9, 1590e-9
BEADS = {"7.32 um": 7.32e-6, "9.94 um": 9.94e-6}
N_BEAD, N_MED = 1.39, 1.0


def mapping(lam=None, **overrides):
    """(lam, y, theta[rad]) from scarlett.physics.seals_wavelength_to_angle."""
    lam = np.linspace(LAM_MIN, LAM_MAX, 2001) if lam is None else np.asarray(lam, float)
    p = {**DEFAULTS, **overrides}
    y, th = ph.seals_wavelength_to_angle(p["groove_spacing"], p["grating_distance"], p["grating_angle"],
                                         p["dcorr"], p["pupil_diameter"], p["numerical_aperture"], lam)
    return lam, y, th


# ── SymPy derivations ────────────────────────────────────────────────

def grating_symbolic():
    """Grating equation and angular dispersion d(beta)/d(lam) = m / (d cos beta)."""
    lam, d, a, m = sp.symbols("lambda d alpha m", positive=True)
    beta = sp.asin(m * lam / d - sp.sin(a))
    disp = sp.simplify(sp.diff(beta, lam))
    return beta, disp, (lam, d, a, m)


def displacement_symbolic():
    """Eq. (1) y(lam) and dy/dlam (magnitude; the sign depends on the y-axis convention)."""
    lam, d, a, m, D, M = sp.symbols("lambda d alpha m D M", positive=True)
    beta = sp.asin(m * lam / d - sp.sin(a))
    t = sp.tan(a - beta)
    y = M * D * t / (1 + t * sp.tan(a))
    return y, sp.diff(y, lam), (lam, d, a, m, D, M)


def angle_symbolic():
    """Eq. (2): theta(y) = atan(2/P (y - yc + dcorr) tan(asin NA)); dtheta/dy at center."""
    y, yc, dc, P, NA = sp.symbols("y y_c d_corr P NA", real=True)
    th = sp.atan(2 / P * (y - yc + dc) * sp.tan(sp.asin(NA)))
    return th, sp.simplify(sp.diff(th, y).subs(y, yc - dc)), (y, yc, dc, P, NA)


def min_resolvable_angle_symbolic():
    """Detector of aperture a at distance L subtends 2 atan(a / 2L)."""
    a, L = sp.symbols("a L", positive=True)
    return 2 * sp.atan(a / (2 * L)), (a, L)


def resolution_reconciliation(lam0=LAM0, bandwidth=20e-9):
    """Evaluate d(lam) = lam0 d / (2 m w0) under the possible readings of the text.

    The paper prints d = 1100 nm, w0 = 3 mm and reports 0.48 nm (42 angles).
    Only one reading reproduces 0.48 nm: pitch d = 1/(1100 lines/mm) = 909.09 nm
    (the value the lab simulation uses) AND 3 mm as the beam DIAMETER (w0 = 1.5 mm).
    """
    rows = []
    for label, d, w0 in (("as printed: d = 1100 nm, w0 = 3 mm", 1100e-9, 3e-3),
                         ("1100 lines/mm (d = 909 nm), w0 = 3 mm", 1e-3 / 1100, 3e-3),
                         ("d = 1100 nm, 3 mm = diameter (w0 = 1.5 mm)", 1100e-9, 1.5e-3),
                         ("1100 lines/mm AND 3 mm = diameter", 1e-3 / 1100, 1.5e-3)):
        dl = ph.seals_wavelength_resolution(lam0, d, w0)
        rows.append((label, dl, bandwidth / dl))
    return rows


# ── Mie on the beads ─────────────────────────────────────────────────

def bead_profile(dia, angles_rad, lam=LAM0, npar=N_BEAD, nmed=N_MED, r=0.1):
    return ph.mie_scattering(npar, nmed, dia, lam, angles_rad, r).i_unpolarized


def lobe_count(dia, lo_deg, hi_deg, order=20, n=6000):
    """Number of local maxima of the unpolarized Mie profile on [lo, hi] degrees."""
    ang = np.radians(np.linspace(lo_deg, hi_deg, n))
    I = 10 * np.log10(bead_profile(dia, ang))
    return len(argrelextrema(I, np.greater, order=order)[0])


def lobe_period_deg(dia, lam=LAM0, nmed=N_MED):
    """Rough angular lobe period 180 deg / x (diffraction-like ripple)."""
    return 180.0 / float(ph.size_parameter(dia, lam, nmed))


# ── equalization and time-stretch readout ────────────────────────────

def equalization_filter(dia, measurement_angle_deg=20.0, lam=None, max_atten_db=np.inf):
    """Filter transmission T(lam) ~ I_min / I(theta(lam)) (ideal inverse), limited to
    max_atten_db of attenuation, and the dynamic range (dB) of the detected
    spectrum before and after equalization."""
    if max_atten_db < 0:
        raise ValueError("max_atten_db must be non-negative")
    lam, _, th = mapping(lam)
    scat = np.radians(np.degrees(th) + measurement_angle_deg)
    I = bead_profile(dia, np.abs(scat), lam=LAM0)
    T = np.maximum(I.min() / I, 10 ** (-max_atten_db / 10))
    dr_before = 10 * np.log10(I.max() / I.min())
    dr_after = 10 * np.log10((I * T).max() / (I * T).min())
    return lam, T, dr_before, dr_after


def time_stretch_window(dispersion_ps_per_nm, bandwidth_nm=20.0):
    """TS-DFT maps wavelength to time: t = D * (lam - lam0). Returns the record length (ns)."""
    if dispersion_ps_per_nm <= 0 or bandwidth_nm <= 0:
        raise ValueError("dispersion and bandwidth must be positive")
    return dispersion_ps_per_nm * bandwidth_nm / 1e3


def digitizer_rate_needed(n_angles, record_ns):
    """Samples/s to resolve n_angles across the stretched record (Nyquist: 2 per angle)."""
    return 2 * n_angles / (record_ns * 1e-9)
