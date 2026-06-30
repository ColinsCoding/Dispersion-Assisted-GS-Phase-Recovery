"""Thermal radiation, Planck distribution, and Gaussian integrals.

AP PHYSICS C + ENGINEERING CONNECTION:
  Thermal radiation is what every hot object emits. The sun, a soldering iron,
  a human body, and a laser diode all radiate according to the same Planck law.
  The Gaussian integral is the mathematical backbone: the area under a Gaussian
  to two sigma is exactly 95.44% -- the "2-sigma rule" -- and it falls out of
  the same integral that gives the Stefan-Boltzmann law.

KEY RESULTS:
  Planck: B(nu,T) = (2*h*nu^3/c^2) / (exp(h*nu/k_B*T) - 1)
  Wien: lambda_max * T = 2.898e-3 m*K  (color of a star tells you its temperature)
  Stefan-Boltzmann: P = sigma * A * T^4  (total power radiated)
  Gaussian 2-sigma: integral_{-2s}^{2s} Gauss = erf(2/sqrt(2)) = 0.9544

THERMAL IMAGING CONNECTION (to this repo):
  A thermal camera measures I(x,y) = integral B(nu,T(x,y)) d(nu) -- the
  same intensity-only measurement problem as the GS receiver. Temperature
  maps directly to intensity via T^4. Phase (= T distribution) must be
  inferred from I(x,y) alone -- the same algorithmic challenge.

PLASMA CONNECTION:
  A plasma is a gas of free electrons. Electrons radiate when accelerated
  (Larmor formula). Thermal plasma emits bremsstrahlung (braking radiation):
  P = (q^2 * a^2) / (6*pi*eps0*c^3) -- the Larmor power, same formula as
  the radiation from an accelerating charge in antenna theory.
"""
import numpy as np
import sympy as sp
from scipy.special import erf


# ── physical constants ────────────────────────────────────────────────

h_J_s = 6.62607015e-34    # Planck constant (J*s)
k_B_J_K = 1.380649e-23    # Boltzmann constant (J/K)
c_m_s = 2.99792458e8      # speed of light (m/s)
sigma_SB = 5.670374419e-8  # Stefan-Boltzmann constant (W/m^2/K^4)


# ── Planck distribution ───────────────────────────────────────────────

def planck_spectral_radiance(wavelength_nm, T_K):
    """Planck spectral radiance B(lambda, T) in W/m^2/sr/nm.

    B(lambda, T) = (2*h*c^2/lambda^5) / (exp(h*c/(lambda*k_B*T)) - 1)

    Parameters
    ----------
    wavelength_nm : float or array -- wavelength in nanometres
    T_K : float                    -- temperature in Kelvin

    Returns spectral radiance in W/(m^2 * sr * nm).
    """
    if np.any(np.asarray(T_K) <= 0):
        raise ValueError("temperature T_K must be positive")
    if np.any(np.asarray(wavelength_nm) <= 0):
        raise ValueError("wavelength must be positive")
    lam = np.asarray(wavelength_nm, dtype=float) * 1e-9  # convert nm -> m
    exponent = h_J_s * c_m_s / (lam * k_B_J_K * T_K)
    B = (2 * h_J_s * c_m_s**2 / lam**5) / (np.exp(exponent) - 1)
    return B * 1e-9  # convert W/m^2/sr/m -> W/m^2/sr/nm


def wien_peak_wavelength(T_K):
    """Wien displacement law: lambda_max * T = 2.898e-3 m*K.

    Tells you the color (wavelength of peak emission) of a blackbody.
    Sun (5778 K) -> 502 nm (green). Human body (310 K) -> 9.4 um (mid-IR).
    Thermal camera detects 8-14 um (T ~ 207-362 K range for humans).
    """
    if T_K <= 0:
        raise ValueError("T_K must be positive")
    b_wien = 2.897771955e-3  # m*K
    lam_peak_m = b_wien / T_K
    return {
        "lambda_peak_m": lam_peak_m,
        "lambda_peak_nm": lam_peak_m * 1e9,
        "lambda_peak_um": lam_peak_m * 1e6,
        "T_K": T_K,
        "band": ("visible" if 380e-9 <= lam_peak_m <= 700e-9 else
                 "near-IR" if lam_peak_m < 2.5e-6 else
                 "mid-IR" if lam_peak_m < 15e-6 else "far-IR"),
    }


def stefan_boltzmann_power(T_K, area_m2=1.0, emissivity=1.0):
    """Total radiated power: P = epsilon * sigma * A * T^4.

    Parameters
    ----------
    T_K : float       -- temperature (K)
    area_m2 : float   -- surface area (m^2)
    emissivity : float -- 0 (perfect mirror) to 1 (perfect blackbody)
    """
    if T_K <= 0:
        raise ValueError("T_K must be positive")
    if not (0 <= emissivity <= 1):
        raise ValueError("emissivity must be in [0,1]")
    P = emissivity * sigma_SB * area_m2 * T_K**4
    return {"power_W": P, "T_K": T_K, "area_m2": area_m2,
            "emissivity": emissivity, "sigma_SB": sigma_SB}


def temperature_from_intensity(intensity_W_m2, emissivity=1.0):
    """Invert Stefan-Boltzmann: T = (I / (epsilon * sigma))^(1/4).

    This is what a thermal camera does: measures intensity I -> infers T.
    Same intensity-only inversion as the GS phase retrieval problem.
    """
    if intensity_W_m2 <= 0:
        raise ValueError("intensity must be positive")
    T = (intensity_W_m2 / (emissivity * sigma_SB)) ** 0.25
    return {"T_K": T, "T_C": T - 273.15}


# ── Gaussian integral (the 2-sigma rule) ─────────────────────────────

def gaussian_integral_fraction(n_sigma):
    """Fraction of a Gaussian distribution within +/- n_sigma of the mean.

    Result = erf(n_sigma / sqrt(2))

    1-sigma: 68.27%  (one standard deviation)
    2-sigma: 95.45%  (the '2-sigma rule', 1-in-22 outside)
    3-sigma: 99.73%  (3 sigma -- quality control 'three nines')
    6-sigma: 99.99966% (Six Sigma manufacturing)

    Arises in: noise statistics, quantum tunneling, thermal fluctuations,
    GS convergence rate (error falls as ~exp(-n^2/2) per iteration near optimum).
    """
    if n_sigma < 0:
        raise ValueError("n_sigma must be non-negative")
    fraction = float(erf(n_sigma / np.sqrt(2)))
    return {
        "n_sigma": n_sigma,
        "fraction_inside": fraction,
        "fraction_outside": 1 - fraction,
        "percent_inside": fraction * 100,
        "one_in_N_outside": 1.0 / (1 - fraction) if fraction < 1 else float("inf"),
    }


def gaussian_integral_sympy():
    """The Gaussian integral and its error function form in SymPy."""
    x, sigma, mu = sp.symbols('x sigma mu', real=True)
    sigma_pos = sp.Symbol('sigma', positive=True)
    # Standard Gaussian (unnormalized): int_{-inf}^{inf} exp(-x^2) dx = sqrt(pi)
    integral_exact = sp.sqrt(sp.pi)
    # Normalized: int_{-inf}^{inf} (1/sqrt(2*pi*sigma^2)) * exp(-x^2/(2*sigma^2)) dx = 1
    n_sigma_sym = sp.Symbol('n_sigma', positive=True)
    erf_form = sp.erf(n_sigma_sym / sp.sqrt(2))
    return {
        "Gaussian_integral":
            sp.Eq(sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo)), integral_exact),
        "Fraction_within_n_sigma":
            sp.Eq(sp.Symbol('P(|X|<n*sigma)'), erf_form),
        "Two_sigma_rule":
            sp.Eq(sp.Symbol('P(|X|<2*sigma)'),
                  sp.erf(sp.sqrt(2)).evalf()),
    }


# ── plasma connection: Debye length and plasma frequency ─────────────

def debye_length(n_e_per_m3, T_e_K):
    """Debye screening length in a plasma: lambda_D = sqrt(eps0*k_B*T / (n*q^2)).

    The Debye length is the distance over which charge imbalances are screened.
    Below lambda_D: you see individual charges. Above: plasma behaves as neutral.
    Earth ionosphere: lambda_D ~ 3 mm at n_e ~ 1e10/m^3, T_e ~ 1000 K.
    """
    eps0 = 8.854187817e-12
    q_e = 1.602176634e-19
    if n_e_per_m3 <= 0 or T_e_K <= 0:
        raise ValueError("n_e and T_e must be positive")
    lam_D = np.sqrt(eps0 * k_B_J_K * T_e_K / (n_e_per_m3 * q_e**2))
    return {"debye_length_m": lam_D, "debye_length_mm": lam_D * 1e3,
            "n_e": n_e_per_m3, "T_e_K": T_e_K}


def plasma_frequency(n_e_per_m3):
    """Electron plasma frequency: omega_p = sqrt(n*q^2 / (eps0*m_e)).

    Frequencies below omega_p are reflected by the plasma (no propagation).
    This is why AM radio (low frequency) bounces off the ionosphere but
    FM radio and GPS (high frequency) pass through.
    """
    eps0 = 8.854187817e-12
    q_e = 1.602176634e-19
    m_e = 9.1093837015e-31
    if n_e_per_m3 <= 0:
        raise ValueError("n_e must be positive")
    omega_p = np.sqrt(n_e_per_m3 * q_e**2 / (eps0 * m_e))
    f_p = omega_p / (2 * np.pi)
    return {"omega_p_rad_s": omega_p, "f_p_Hz": f_p, "f_p_MHz": f_p / 1e6,
            "n_e": n_e_per_m3}


# ── SymPy 5 ──────────────────────────────────────────────────────────

def thermal_radiation_sympy_5():
    """Five key thermal radiation equations."""
    lam, T_s, h_s, c_s, k_s = sp.symbols('lambda T h c k_B', positive=True)
    eps, sigma_s, A_s = sp.symbols('epsilon sigma A', positive=True)
    x_s, n_s = sp.symbols('x n', real=True)
    b_s = sp.Symbol('b')   # Wien constant

    return {
        "Planck_law":
            sp.Eq(sp.Symbol('B(lambda,T)'),
                  2*h_s*c_s**2/lam**5 / (sp.exp(h_s*c_s/(lam*k_s*T_s)) - 1)),
        "Wien_law":
            sp.Eq(lam * T_s, b_s),
        "Stefan_Boltzmann":
            sp.Eq(sp.Symbol('P'), eps*sigma_s*A_s*T_s**4),
        "Gaussian_integral":
            sp.Eq(sp.Integral(sp.exp(-x_s**2), (x_s, -sp.oo, sp.oo)), sp.sqrt(sp.pi)),
        "Debye_length":
            sp.Eq(sp.Symbol('lambda_D'),
                  sp.sqrt(sp.Symbol('epsilon_0')*k_s*T_s /
                          (sp.Symbol('n_e')*sp.Symbol('q')**2))),
    }


if __name__ == "__main__":
    print("=== Wien peak wavelength ===")
    for T, label in [(5778, "Sun"), (3000, "Incandescent bulb"),
                     (310, "Human body"), (77, "Liquid nitrogen")]:
        w = wien_peak_wavelength(T)
        print(f"  {label} ({T} K): peak at {w['lambda_peak_um']:.2f} um ({w['band']})")

    print("\n=== Stefan-Boltzmann: human body at 37 C ===")
    sb = stefan_boltzmann_power(310, area_m2=1.8, emissivity=0.97)
    print(f"  P = {sb['power_W']:.0f} W  (that is why you heat a room)")

    print("\n=== Gaussian integral -- sigma rule ===")
    for ns in [1, 2, 3, 6]:
        g = gaussian_integral_fraction(ns)
        print(f"  {ns}-sigma: {g['percent_inside']:.4f}% inside "
              f"(1 in {g['one_in_N_outside']:.0f} outside)")

    print("\n=== Plasma: Earth ionosphere ===")
    d = debye_length(1e10, 1000)
    fp = plasma_frequency(1e10)
    print(f"  Debye length: {d['debye_length_mm']:.2f} mm")
    print(f"  Plasma freq:  {fp['f_p_MHz']:.2f} MHz  (AM radio below this is reflected)")

    print("\n=== SymPy 5 ===")
    for k, eq in thermal_radiation_sympy_5().items():
        print(f"  {k}: {eq}")
