"""human_vs_instrument_optics.py -- human-eye optics vs. engineered-instrument
optics, built on one shared piece of vector calculus: the differential area
element on a sphere, dA_vec = r^2*sin(theta)*dtheta*dphi*rhat.

That element is derived the same way Feynman (Lectures Vol. II, flux through
a differential surface) and Griffiths (Introduction to Electrodynamics, Ch.1
spherical coordinates; Ch.11 radiated power per solid angle dP/dOmega) both
build it: as a cross product of the two coordinate tangent vectors. Once you
have dOmega = dA/r^2, "how much light does a detector collect" is always the
same integral, Phi = Integral[ (dP/dOmega) dOmega ] over the detector's
acceptance solid angle -- whether the detector is a human pupil or a lens.

This module runs that one integral twice, with the light source held fixed
(Mie scattering off a dielectric sphere, ported from the vetted
seals_stable.ipynb Sec.4/Sec.5 Mie machinery -- same pi_n/tau_n angular
recurrence, same completed/tested formulas) and only the *collector* swapped
between the human eye and a Jalali-lab-style time-stretch instrument
(aperture/NA taken from SEALS's own default lens parameters, telecom C-band).

What falls out is three concrete "human vs. instrument" numbers:
  1. Diffraction-limited angular resolution (Rayleigh criterion).
  2. Collected scattered-photon flux over each system's acceptance solid angle.
  3. Dynamic range in bits -- the eye's logarithmic (Weber-Fechner) response
     vs. a linear ADC's bit depth, which is the same log-compression math as
     mu-law/A-law companding in digital audio (a real computer-engineering
     connection, not a coincidence of vocabulary).

Honesty note: "nuclear Griffiths" doesn't map to anything concrete in this
repo or in Griffiths' *Electrodynamics* -- the closest real Griffiths
material is Ch.11's dP/dOmega radiated-power-per-solid-angle formalism,
which is what's actually used here. If a specific nuclear-physics topic was
meant, it isn't included yet.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from scipy.special import spherical_jn, spherical_yn
from typing import Dict


# ── Section A: differential area element (Feynman + Griffiths) ───────────────

def derive_spherical_area_element_symbolic() -> sp.Expr:
    """Derive |dA_vec/(dtheta*dphi)| = r^2*sin(theta) from the general
    curvilinear-coordinates cross-product construction
        dA_vec = (d(r_vec)/dtheta) x (d(r_vec)/dphi)
    -- the same construction Feynman (Vol. II) and Griffiths (Ch.1) both use
    to build the solid-angle element dOmega = dA/r^2 = sin(theta)*dtheta*dphi.
    """
    r, theta, phi = sp.symbols("r theta phi", positive=True)
    r_vec = sp.Matrix([
        r * sp.sin(theta) * sp.cos(phi),
        r * sp.sin(theta) * sp.sin(phi),
        r * sp.cos(theta),
    ])
    dr_dtheta = r_vec.diff(theta)
    dr_dphi = r_vec.diff(phi)
    cross = dr_dtheta.cross(dr_dphi)
    mag = sp.simplify(cross.norm())
    # theta ranges over (0,pi) as a polar angle, where sin(theta)>=0, so
    # Abs(sin(theta)) == sin(theta); make that explicit rather than leaving
    # the Abs() sympy's norm() introduces from not knowing that domain.
    mag = sp.refine(mag, sp.Q.positive(sp.sin(theta)))
    return sp.simplify(mag)  # -> r**2*sin(theta)


def verify_full_sphere_solid_angle() -> float:
    """Numeric sanity check: Integral[sin(theta)dtheta dphi] over the full
    sphere must equal 4*pi steradians -- the total solid angle."""
    theta = np.linspace(0, np.pi, 4001)
    phi_range = 2 * np.pi
    omega = phi_range * np.trapezoid(np.sin(theta), theta)
    return float(omega)


# ── Section B: Mie angular scattering S1(theta), S2(theta) ───────────────────
# Ported from seals_stable.ipynb Sec.4 (mie_coefficients, _angular_functions),
# already numerically verified there (PASS Sec.4/5: complex E-fields, fixed
# pi_n/tau_n recurrence range(2,nmax), no NaN/Inf). Trimmed here to just the
# angular amplitudes S1, S2 needed for dP/dOmega ~ 0.5*(|S1|^2+|S2|^2).

def _mie_bessel(n_arr: np.ndarray, x: float):
    jx = np.array([spherical_jn(int(n), x) for n in n_arr])
    yx = np.array([spherical_yn(int(n), x) for n in n_arr])
    j1x = np.empty_like(jx)
    j1x[0] = np.sin(x) / x if x > 1e-30 else 1.0
    j1x[1:] = jx[:-1]
    y1x = np.empty_like(yx)
    y1x[0] = -np.cos(x) / x if x > 1e-30 else 0.0
    y1x[1:] = yx[:-1]
    return jx, yx, j1x, y1x


def mie_coefficients(npar: float, nmed: float, dia: float, lam: float):
    if dia <= 0:
        raise ValueError(f"dia={dia}: particle diameter must be positive")
    if npar <= 0 or nmed <= 0:
        raise ValueError(f"npar={npar}, nmed={nmed}: refractive indices must be positive")
    if lam <= 0:
        raise ValueError(f"lam={lam}: wavelength must be positive")
    x = np.pi * dia / (lam / nmed)
    m = npar / nmed
    nmax = max(int(np.round(2 + x + 4 * x ** (1 / 3))), 5)
    n_arr = np.arange(1, nmax + 1, dtype=float)
    z = m * x
    jx, yx, j1x, y1x = _mie_bessel(n_arr, x)
    jz, _, j1z, _ = _mie_bessel(n_arr, z)
    hx = jx + 1j * yx
    h1x = j1x + 1j * y1x
    ax = x * j1x - n_arr * jx
    az = z * j1z - n_arr * jz
    ahx = x * h1x - n_arr * hx
    m2 = m * m
    an = (m2 * jz * ax - jx * az) / (m2 * jz * ahx - hx * az)
    bn = (jz * ax - jx * az) / (jz * ahx - hx * az)
    return n_arr, an, bn, x, nmax


def _angular_functions(nmax: int, u_arr: np.ndarray):
    N = len(u_arr)
    pi_m = np.zeros((nmax, N))
    tau_m = np.zeros((nmax, N))
    pi_m[0] = 1.0
    tau_m[0] = u_arr
    if nmax >= 2:
        pi_m[1] = 3.0 * u_arr
        tau_m[1] = 6.0 * u_arr ** 2 - 3.0
    for j in range(2, nmax):  # range(2,nmax), not range(3,nmax) -- see seals_stable.ipynb Sec.4
        n = j + 1
        pi_m[j] = ((2 * n - 1) / (n - 1)) * u_arr * pi_m[j - 1] - (n / (n - 1)) * pi_m[j - 2]
        tau_m[j] = n * u_arr * pi_m[j] - (n + 1) * pi_m[j - 1]
    return pi_m, tau_m


def mie_s1_s2(npar: float, nmed: float, dia: float, lam: float, theta_rad: np.ndarray):
    """Mie scattering amplitudes S1(theta), S2(theta). dP/dOmega ~
    0.5*(|S1|^2+|S2|^2) is the unpolarized differential scattering pattern."""
    n_arr, an, bn, x, nmax = mie_coefficients(npar, nmed, dia, lam)
    u = np.cos(theta_rad)
    pi_m, tau_m = _angular_functions(nmax, u)
    wt = (2 * n_arr + 1) / (n_arr * (n_arr + 1))
    pin = wt[:, None] * pi_m
    tin = wt[:, None] * tau_m
    S1 = np.sum(an[:, None] * pin + bn[:, None] * tin, axis=0)
    S2 = np.sum(an[:, None] * tin + bn[:, None] * pin, axis=0)
    return S1, S2


# ── Section C: eye vs. instrument optical-system metrics ─────────────────────
def _focal_length_from_NA(aperture_diameter_mm: float, NA: float) -> float:
    """Exact (non-paraxial) inversion of NA=sin(arctan((D/2)/f)) for f, so
    that a target NA is reproduced exactly by optical_system_metrics()
    below rather than only approximately (the paraxial NA~=D/(2f) shortcut
    breaks down at NA=0.70 -- that's not a small angle)."""
    return (aperture_diameter_mm / 2) / np.tan(np.arcsin(NA))


# HUMAN_EYE: typical daylight pupil (4mm), eye's effective focal length
# (~17mm), photopic peak sensitivity (555nm).
# JALALI_INSTRUMENT: SEALS's own default collection-lens parameters
# (P=5.8mm diameter, NA=0.70 exactly, via the inversion above), telecom
# C-band (1590nm) -- the same regime Jalali-lab time-stretch instruments
# operate in.
HUMAN_EYE = dict(aperture_diameter_mm=4.0, focal_length_mm=17.0, wavelength_nm=555.0)
JALALI_INSTRUMENT = dict(aperture_diameter_mm=5.8,
                          focal_length_mm=_focal_length_from_NA(5.8, 0.70),
                          wavelength_nm=1590.0)


def optical_system_metrics(aperture_diameter_mm: float, focal_length_mm: float,
                            wavelength_nm: float) -> Dict:
    """NA, f-number, Rayleigh diffraction limit, and acceptance solid angle
    for any aperture+focal-length optical collector -- the same four numbers
    for a human pupil or an instrument lens."""
    if aperture_diameter_mm <= 0:
        raise ValueError(f"aperture_diameter_mm={aperture_diameter_mm}: must be positive")
    if focal_length_mm <= 0:
        raise ValueError(f"focal_length_mm={focal_length_mm}: must be positive")
    if wavelength_nm <= 0:
        raise ValueError(f"wavelength_nm={wavelength_nm}: must be positive")

    theta_half = np.arctan((aperture_diameter_mm / 2) / focal_length_mm)  # half-angle subtended
    NA = np.sin(theta_half)
    f_number = focal_length_mm / aperture_diameter_mm
    lam_mm = wavelength_nm * 1e-6
    theta_rayleigh_rad = 1.22 * lam_mm / aperture_diameter_mm  # Rayleigh criterion
    omega_accept_sr = 2 * np.pi * (1 - np.cos(theta_half))  # dOmega = dA/r^2, integrated (Sec. A)

    return {
        "NA": float(NA),
        "f_number": float(f_number),
        "theta_rayleigh_rad": float(theta_rayleigh_rad),
        "theta_rayleigh_arcsec": float(np.degrees(theta_rayleigh_rad) * 3600),
        "acceptance_theta_half_rad": float(theta_half),
        "acceptance_solid_angle_sr": float(omega_accept_sr),
    }


# ── Section D: collected scattered flux over each system's acceptance angle ──

def collected_scattering_flux(npar: float, nmed: float, dia: float, lam_nm: float,
                               theta_max_rad: float, n_theta: int = 2000) -> float:
    """Integral[ (dP/dOmega) dOmega ] over a forward-pointed (theta=0) collector
    of half-angle theta_max, using dOmega = 2*pi*sin(theta)*dtheta from Sec. A
    (azimuthal symmetry of unpolarized Mie scattering)."""
    if not (0 < theta_max_rad < np.pi):
        raise ValueError(f"theta_max_rad={theta_max_rad}: must be in (0, pi)")
    if n_theta < 3:
        raise ValueError(f"n_theta={n_theta}: need at least 3 samples")
    theta = np.linspace(1e-6, theta_max_rad, n_theta)
    lam_mm = lam_nm * 1e-6
    S1, S2 = mie_s1_s2(npar, nmed, dia, lam_mm, theta)
    dPdOmega = 0.5 * (np.abs(S1) ** 2 + np.abs(S2) ** 2)
    return float(np.trapezoid(dPdOmega * 2 * np.pi * np.sin(theta), theta))


def compare_eye_vs_instrument_collection(npar: float = 1.39, nmed: float = 1.00,
                                          dia_nm: float = 9940.0) -> Dict:
    """Same Mie-scattering particle (SEALS's own default: 9.94 micron, n=1.39
    in water), collected by the eye's acceptance solid angle vs. the Jalali
    instrument's -- each system's own wavelength and aperture from Sec. C."""
    eye_m = optical_system_metrics(**HUMAN_EYE)
    inst_m = optical_system_metrics(**JALALI_INSTRUMENT)
    flux_eye = collected_scattering_flux(npar, nmed, dia_nm * 1e-6, HUMAN_EYE["wavelength_nm"],
                                          eye_m["acceptance_theta_half_rad"])
    flux_inst = collected_scattering_flux(npar, nmed, dia_nm * 1e-6, JALALI_INSTRUMENT["wavelength_nm"],
                                           inst_m["acceptance_theta_half_rad"])
    return {
        "eye": {**eye_m, "collected_flux": flux_eye},
        "instrument": {**inst_m, "collected_flux": flux_inst},
        "flux_ratio_instrument_over_eye": flux_inst / flux_eye if flux_eye else float("inf"),
    }


# ── Section E: dynamic range -- Weber-Fechner (eye) vs. linear ADC bits ──────

def dynamic_range_bits(intensity_ratio: float) -> float:
    """Bits of linear ADC resolution needed to span the given max/min
    intensity ratio without saturating (top) or quantizing to zero (bottom).
    The eye instead compresses intensity logarithmically (Weber-Fechner) --
    the same log-compression math as mu-law/A-law audio companding, not an
    analogy of convenience."""
    if intensity_ratio <= 1:
        raise ValueError(f"intensity_ratio={intensity_ratio}: must be > 1")
    return float(np.log2(intensity_ratio))


EYE_DYNAMIC_RANGE_RATIO = 1e9   # ~90 dB, commonly cited full adapted range (scotopic to photopic)
CAMERA_ADC_BITS = {"8-bit camera": 8, "12-bit camera": 12, "14-bit scientific CCD": 14}


# ── Section F: temporal resolution -- flicker fusion vs. time-stretch ────────

def temporal_resolution_comparison(osc_bw_ghz: float = 40.0) -> Dict:
    """Eye's flicker-fusion frame time (~60Hz, from vision science) vs. a
    Jalali-lab time-stretch instrument's temporal resolution set by its
    oscilloscope bandwidth (dispersive_fourier_teaching.py's own
    LAB_PARAMS['osc_BW_GHz']=40.0 default, ~1/(2*BW))."""
    if osc_bw_ghz <= 0:
        raise ValueError(f"osc_bw_ghz={osc_bw_ghz}: must be positive")
    eye_frame_time_s = 1.0 / 60.0
    instrument_resolution_s = 1.0 / (2 * osc_bw_ghz * 1e9)
    return {
        "eye_frame_time_s": eye_frame_time_s,
        "instrument_resolution_s": instrument_resolution_s,
        "speedup_factor": eye_frame_time_s / instrument_resolution_s,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def print_comparison() -> None:
    print("=" * 72)
    print("  HUMAN EYE vs. JALALI-LAB INSTRUMENT OPTICS")
    print("=" * 72)

    print("\n[A] Differential area element (Feynman/Griffiths):")
    print("    |dA/(dtheta dphi)| =", derive_spherical_area_element_symbolic())
    print(f"    full-sphere solid angle check: {verify_full_sphere_solid_angle():.6f} sr (expect 4*pi={4*np.pi:.6f})")

    print("\n[C] Optical-system metrics:")
    for name, params in [("Human eye", HUMAN_EYE), ("Jalali instrument", JALALI_INSTRUMENT)]:
        m = optical_system_metrics(**params)
        print(f"    {name:18s}  NA={m['NA']:.3f}  f/#={m['f_number']:.2f}  "
              f"Rayleigh={m['theta_rayleigh_arcsec']:.2f} arcsec  "
              f"Omega_accept={m['acceptance_solid_angle_sr']:.4e} sr")

    print("\n[D] Collected Mie-scattering flux (same particle, each system's own aperture):")
    c = compare_eye_vs_instrument_collection()
    print(f"    eye flux={c['eye']['collected_flux']:.4e}  "
          f"instrument flux={c['instrument']['collected_flux']:.4e}  "
          f"ratio(instrument/eye)={c['flux_ratio_instrument_over_eye']:.2e}")

    print("\n[E] Dynamic range:")
    eye_bits = dynamic_range_bits(EYE_DYNAMIC_RANGE_RATIO)
    print(f"    eye (Weber-Fechner, ~{EYE_DYNAMIC_RANGE_RATIO:.0e} ratio) needs {eye_bits:.1f} linear bits to match")
    for cam, bits in CAMERA_ADC_BITS.items():
        print(f"    {cam:20s}: {bits} bits linear -> {2**bits:.0f}:1 ratio")

    print("\n[F] Temporal resolution:")
    t = temporal_resolution_comparison()
    print(f"    eye frame time = {t['eye_frame_time_s']*1e3:.2f} ms")
    print(f"    instrument resolution = {t['instrument_resolution_s']*1e12:.3f} ps")
    print(f"    instrument is {t['speedup_factor']:.2e}x faster")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    print_comparison()
