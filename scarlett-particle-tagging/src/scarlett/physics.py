"""Physics model: Rayleigh, Rayleigh-Debye and Mie scattering by a sphere,
the SEALS wavelength-to-angle map, and Brownian motion.

This is a Python port of the MATLAB simulation (main.m, mie.m,
rayleighdebye.m, SEALS.m). The equations are preserved; every place where
this port deliberately differs from the original is controlled by a
``legacy=True`` switch that reproduces the original behaviour, and is listed
in docs/CODE_REVIEW.md. Classical electrodynamics throughout.

Units: SI everywhere. Lengths in metres, angles in RADIANS, irradiance in
W/m^2, cross sections in m^2. (The original SEALS.m returned degrees.)

Symbols (same names as the MATLAB code where they exist)
    dia      particle diameter [m]            radius = dia / 2
    lam      vacuum wavelength [m]            ("lambda" is reserved in Python)
    npar     particle refractive index [-]
    nmed     medium refractive index [-]
    theta    scattering angle [rad], 0 = forward
    r        distance from particle to detector [m]
    x        size parameter  x = pi * dia / (lam / nmed)  [-]
    m        relative index  m = npar / nmed  [-]
    k        wavenumber in the medium  k = 2 pi nmed / lam  [rad/m]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

C_LIGHT = 299_792_458.0          # m/s, exact
H_PLANCK = 6.626_070_15e-34      # J s, exact
K_BOLTZMANN = 1.380_649e-23      # J/K, exact
E_CHARGE = 1.602_176_634e-19     # C, exact


# ----------------------------------------------------------------------
# Basic quantities
# ----------------------------------------------------------------------
def _positive(value, name):
    if not np.all(np.asarray(value) > 0):
        raise ValueError(f"{name} must be positive, got {value!r}")


def size_parameter(dia, lam, nmed=1.0):
    """x = pi * dia / (lam / nmed): circumference over wavelength in the medium."""
    _positive(dia, "dia"); _positive(lam, "lam"); _positive(nmed, "nmed")
    return np.pi * np.asarray(dia, dtype=float) / (np.asarray(lam, dtype=float) / nmed)


def wavenumber(lam, nmed=1.0):
    """k = 2 pi nmed / lam, the wavenumber inside the medium [rad/m]."""
    _positive(lam, "lam"); _positive(nmed, "nmed")
    return 2 * np.pi * nmed / np.asarray(lam, dtype=float)


def lorentz_lorenz(npar, nmed=1.0):
    """K = (m^2 - 1) / (m^2 + 2), the polarizability factor of a small sphere."""
    m = npar / nmed
    return (m ** 2 - 1) / (m ** 2 + 2)


def photon_energy(lam):
    """E = h f = h c / lam [J]. The only quantum relation the model needs."""
    _positive(lam, "lam")
    return H_PLANCK * C_LIGHT / np.asarray(lam, dtype=float)


def regime(dia, lam, npar, nmed=1.0):
    """Which approximation is trustworthy: 'rayleigh', 'rayleigh-debye' or 'mie'.

    Rayleigh        x << 1 and |m| x << 1        (thresholds used: 0.3)
    Rayleigh-Debye  |m - 1| << 1 and 2 x |m - 1| << 1   (thresholds: 0.1, 0.3)
    Mie             always valid for a homogeneous sphere; needed otherwise.
    The thresholds are conventional rules of thumb, not sharp boundaries.
    """
    x = float(size_parameter(dia, lam, nmed))
    m = abs(npar / nmed)
    if x < 0.3 and m * x < 0.3:
        return "rayleigh"
    if abs(m - 1) < 0.1 and 2 * x * abs(m - 1) < 0.3:
        return "rayleigh-debye"
    return "mie"


# ----------------------------------------------------------------------
# Mie theory (same formulation as mie.m, after C. Maetzler's MATLAB routines)
# ----------------------------------------------------------------------
def mie_nmax(x):
    """Number of multipole terms kept: round(2 + x + 4 x^(1/3)) (Wiscombe)."""
    return int(round(2 + x + 4 * x ** (1 / 3)))


def mie_coefficients(m, x):
    """Mie coefficients a_n, b_n for n = 1 .. nmax.

    With psi_n(x) = x j_n(x), xi_n(x) = x h_n^(1)(x) and
    [x j_n(x)]' = x j_{n-1}(x) - n j_n(x):

        a_n = (m^2 j_n(z) [x j_n(x)]' - j_n(x) [z j_n(z)]')
              / (m^2 j_n(z) [x h_n(x)]' - h_n(x) [z j_n(z)]')
        b_n = (    j_n(z) [x j_n(x)]' - j_n(x) [z j_n(z)]')
              / (    j_n(z) [x h_n(x)]' - h_n(x) [z j_n(z)]'),   z = m x.
    """
    _positive(x, "size parameter x")
    n = np.arange(1, mie_nmax(x) + 1)
    z = m * x
    jx, jz = special.spherical_jn(n, x), special.spherical_jn(n, z)
    yx = special.spherical_yn(n, x)
    hx = jx + 1j * yx
    j1x, j1z = special.spherical_jn(n - 1, x), special.spherical_jn(n - 1, z)
    h1x = j1x + 1j * special.spherical_yn(n - 1, x)
    ax = x * j1x - n * jx          # [x j_n(x)]'
    az = z * j1z - n * jz          # [z j_n(z)]'
    ahx = x * h1x - n * hx         # [x h_n(x)]'
    m2 = m * m
    an = (m2 * jz * ax - jx * az) / (m2 * jz * ahx - hx * az)
    bn = (jz * ax - jx * az) / (jz * ahx - hx * az)
    return an, bn


def mie_efficiencies(m, x):
    """Dimensionless efficiencies Q = (cross section) / (pi radius^2).

    qsca = (2/x^2) sum (2n+1)(|a_n|^2 + |b_n|^2)      scattering
    qext = (2/x^2) sum (2n+1) Re(a_n + b_n)            extinction
    qabs = qext - qsca                                 absorption (0 for real m)
    qback = (1/x^2) |sum (2n+1)(-1)^n (a_n - b_n)|^2   backscatter
    """
    an, bn = mie_coefficients(m, x)
    n = np.arange(1, an.size + 1)
    qsca = 2 / x ** 2 * np.sum((2 * n + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2))
    qext = 2 / x ** 2 * np.sum((2 * n + 1) * np.real(an + bn))
    qback = np.abs(np.sum((2 * n + 1) * (-1.0) ** n * (an - bn))) ** 2 / x ** 2
    return {"qsca": float(qsca), "qext": float(qext),
            "qabs": float(qext - qsca), "qback": float(qback)}


def mie_amplitudes(m, x, theta):
    """Scattering amplitudes S1 (perpendicular) and S2 (parallel) at angles theta.

        S1 = sum (2n+1)/(n(n+1)) (a_n pi_n + b_n tau_n)
        S2 = sum (2n+1)/(n(n+1)) (a_n tau_n + b_n pi_n)

    pi_n, tau_n by the upward recurrences used in mie.m:
        pi_1 = 1, pi_2 = 3u,  pi_j = (2j-1)/(j-1) u pi_{j-1} - j/(j-1) pi_{j-2}
        tau_1 = u, tau_2 = 3 cos(2 theta),  tau_j = j u pi_j - (j+1) pi_{j-1},
    with u = cos(theta).
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    an, bn = mie_coefficients(m, x)
    nmax = an.size
    u = np.cos(theta)
    pi_n = np.zeros((nmax, theta.size))
    tau_n = np.zeros((nmax, theta.size))
    pi_n[0], tau_n[0] = 1.0, u
    if nmax > 1:
        pi_n[1], tau_n[1] = 3 * u, 3 * np.cos(2 * theta)
    for j in range(3, nmax + 1):
        pi_n[j - 1] = (2 * j - 1) / (j - 1) * u * pi_n[j - 2] - j / (j - 1) * pi_n[j - 3]
        tau_n[j - 1] = j * u * pi_n[j - 1] - (j + 1) * pi_n[j - 2]
    n = np.arange(1, nmax + 1)
    weight = ((2 * n + 1) / (n * (n + 1)))[:, None]
    s1 = np.sum(weight * (an[:, None] * pi_n + bn[:, None] * tau_n), axis=0)
    s2 = np.sum(weight * (an[:, None] * tau_n + bn[:, None] * pi_n), axis=0)
    return s1, s2


@dataclass(frozen=True)
class MieResult:
    """Everything mie.m returned, with units, plus the unpolarized intensity."""
    x: float                    # size parameter [-]
    sigma_sca: float            # scattering cross section [m^2]
    s1: np.ndarray              # amplitude, perpendicular (s) [-]
    s2: np.ndarray              # amplitude, parallel (p) [-]
    e_par: np.ndarray           # scattered field for unit parallel-polarized input [-]
    e_perp: np.ndarray          # scattered field for unit perpendicular-polarized input [-]
    i_par: np.ndarray           # irradiance, incident light polarized parallel [W/m^2]
    i_perp: np.ndarray          # irradiance, incident light polarized perpendicular [W/m^2]
    i_unpolarized: np.ndarray   # irradiance, unpolarized incident light [W/m^2]
    phase_par: np.ndarray       # wrapped phase of e_par [rad]
    phase_perp: np.ndarray      # wrapped phase of e_perp [rad]
    an: np.ndarray
    bn: np.ndarray

    @property
    def i_legacy_total(self):
        """I_p + I_s as plotted by the original main.m.

        This is the SUM of two separate experiments (parallel input, then
        perpendicular input), each with the full incident irradiance. It is
        exactly 2 x i_unpolarized and is kept only to reproduce old figures.
        """
        return self.i_par + self.i_perp


def mie_scattering(npar, nmed, dia, lam, theta, r, incident_irradiance=1.0):
    """Far-field Mie scattering of a plane wave by a homogeneous sphere.

    Fields follow mie.m (scattering plane phi = pi):
        E_par  = exp(i k r) / (-i k r) * cos(phi) * S2
        E_perp = exp(i k r) / ( i k r) * cos(phi) * S1
    in units of the incident field amplitude, so
        I_par  = I0 |S2|^2 / (k r)^2,   I_perp = I0 |S1|^2 / (k r)^2,
        I_unpolarized = (I_par + I_perp) / 2.
    The far-field form requires k r >> 1 and r >> dia^2 / lam.
    """
    _positive(r, "r"); _positive(incident_irradiance, "incident_irradiance")
    x = float(size_parameter(dia, lam, nmed))
    k = float(wavenumber(lam, nmed))
    m = npar / nmed
    an, bn = mie_coefficients(m, x)
    s1, s2 = mie_amplitudes(m, x, theta)
    phi = np.pi
    spherical = np.exp(1j * k * r) / (1j * k * r)
    e_par = -spherical * np.cos(phi) * s2
    e_perp = spherical * np.cos(phi) * s1
    i_par = incident_irradiance * np.abs(e_par) ** 2
    i_perp = incident_irradiance * np.abs(e_perp) ** 2
    qsca = mie_efficiencies(m, x)["qsca"]
    return MieResult(
        x=x, sigma_sca=qsca * np.pi * dia ** 2 / 4, s1=s1, s2=s2,
        e_par=e_par, e_perp=e_perp, i_par=i_par, i_perp=i_perp,
        i_unpolarized=0.5 * (i_par + i_perp),
        phase_par=np.angle(e_par), phase_perp=np.angle(e_perp), an=an, bn=bn)


def mie_differential_cross_section(npar, nmed, dia, lam, theta):
    """d(sigma)/d(Omega) for unpolarized light [m^2/sr] = (|S1|^2 + |S2|^2) / (2 k^2)."""
    x = float(size_parameter(dia, lam, nmed))
    s1, s2 = mie_amplitudes(npar / nmed, x, theta)
    return (np.abs(s1) ** 2 + np.abs(s2) ** 2) / (2 * float(wavenumber(lam, nmed)) ** 2)


def mie_cross_section(npar, nmed, dia, lam):
    """Total scattering cross section sigma_sca = Q_sca * pi * dia^2 / 4 [m^2]."""
    x = float(size_parameter(dia, lam, nmed))
    return mie_efficiencies(npar / nmed, x)["qsca"] * np.pi * dia ** 2 / 4


# ----------------------------------------------------------------------
# Rayleigh and Rayleigh-Debye (rayleighdebye.m)
# ----------------------------------------------------------------------
def sphere_form_amplitude(u):
    """3 (sin u - u cos u) / u^3, the Fourier transform of a uniform sphere.

    Tends to 1 as u -> 0 (handled explicitly; the original returned NaN at
    theta = 0). Its SQUARE is the Rayleigh-Debye form factor P(theta).
    """
    u = np.asarray(u, dtype=float)
    out = np.ones_like(u)
    big = np.abs(u) > 1e-4
    ub = u[big]
    out[big] = 3 * (np.sin(ub) - ub * np.cos(ub)) / ub ** 3
    out[~big] = 1 - u[~big] ** 2 / 10
    return out


def rayleigh_debye_intensity(dia, lam, nmed, npar, theta, r,
                             incident_irradiance=1.0, legacy=False):
    """Scattered irradiance [W/m^2] for unpolarized light, Rayleigh-Debye model.

        I = I0 * (1 + cos^2 theta) / (2 r^2) * k^4 * radius^6 * |K|^2 * P(theta)
        P(theta) = [3 (sin u - u cos u) / u^3]^2,   u = 2 k radius sin(theta/2)

    legacy=True reproduces rayleighdebye.m exactly, which differs in two ways
    (see docs/CODE_REVIEW.md): it uses the form AMPLITUDE to the first power
    instead of its square, and it uses the vacuum wavenumber 2 pi / lam in the
    k^4 prefactor even when nmed != 1.
    """
    _positive(r, "r")
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    radius = dia / 2
    k = float(wavenumber(lam, nmed))
    u = 2 * k * radius * np.sin(theta / 2)
    dipole = 1 + np.cos(theta) ** 2
    k_factor = abs(lorentz_lorenz(npar, nmed)) ** 2
    if legacy:
        with np.errstate(invalid="ignore", divide="ignore"):
            amplitude = 3 * (np.sin(u) - u * np.cos(u)) / u ** 3
        return np.abs(amplitude * dipole * k_factor / (2 * r ** 2)
                      * (2 * np.pi / lam) ** 4 * radius ** 6)
    return (incident_irradiance * dipole / (2 * r ** 2) * k ** 4 * radius ** 6
            * k_factor * sphere_form_amplitude(u) ** 2)


def rayleigh_intensity(dia, lam, nmed, npar, theta, r, incident_irradiance=1.0):
    """Rayleigh (point-dipole) limit: the Rayleigh-Debye formula with P = 1."""
    _positive(r, "r")
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    k = float(wavenumber(lam, nmed))
    return (incident_irradiance * (1 + np.cos(theta) ** 2) / (2 * r ** 2)
            * k ** 4 * (dia / 2) ** 6 * abs(lorentz_lorenz(npar, nmed)) ** 2)


def rayleigh_cross_section(dia, lam, nmed, npar):
    """sigma = (8 pi / 3) k^4 radius^6 |K|^2  [m^2]; scales as dia^6 / lam^4."""
    k = float(wavenumber(lam, nmed))
    return 8 * np.pi / 3 * k ** 4 * (dia / 2) ** 6 * abs(lorentz_lorenz(npar, nmed)) ** 2


def rayleigh_debye_cross_section(dia, lam, nmed, npar, n_angles=4001):
    """Total Rayleigh-Debye cross section by integrating over the sphere [m^2]."""
    theta = np.linspace(0.0, np.pi, n_angles)
    dsigma = rayleigh_debye_intensity(dia, lam, nmed, npar, theta, r=1.0)   # r^2 I / I0
    return float(2 * np.pi * np.trapezoid(dsigma * np.sin(theta), theta))


# ----------------------------------------------------------------------
# SEALS wavelength-to-angle map (SEALS.m; Adam et al., Opt. Express 2013, Eqs. 1-2)
# ----------------------------------------------------------------------
def seals_wavelength_to_angle(groove_spacing, grating_distance, grating_angle,
                              dcorr, pupil_diameter, numerical_aperture, lam,
                              magnification=1 / 6, order=1):
    """Map wavelength to beam displacement y [m] and illumination angle [rad].

        beta  = asin(order * lam / d - sin(alpha))
        y     = M * D * tan(alpha - beta) / (1 + tan(alpha - beta) tan(alpha))
        y     = y - y(lam_max)                       (displacement from the red edge)
        y_c   = (y(lam_min) - y(lam_max)) / 2
        theta = atan( 2/P * (y - y_c + dcorr) * tan(asin(NA)) )

    `lam` must be increasing. SEALS.m hard-coded M = 1/6 and wrote +M where
    the paper's Eq. (1) has -M; after subtracting y(lam_max) the code's sign
    is the one that reproduces the paper's Fig. 2, so it is kept.
    Returns (y, theta) with theta in radians (SEALS.m returned degrees).
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    if lam.size < 2 or np.any(np.diff(lam) <= 0):
        raise ValueError("lam must be an increasing array with at least 2 points")
    if not 0 < numerical_aperture < 1:
        raise ValueError("numerical_aperture must be in (0, 1) for a lens in air")
    _positive(groove_spacing, "groove_spacing"); _positive(pupil_diameter, "pupil_diameter")
    s = order * lam / groove_spacing - np.sin(grating_angle)
    if np.any(np.abs(s) > 1):
        raise ValueError(
            "grating equation has no solution: |order*lam/d - sin(alpha)| > 1 for some "
            "wavelength (the diffracted order is evanescent). Check d, alpha and lam.")
    t = np.tan(grating_angle - np.arcsin(s))
    y = magnification * grating_distance * t / (1 + t * np.tan(grating_angle))
    y = y - y[-1]
    y_center = (y[0] - y[-1]) / 2
    theta = np.arctan(2 / pupil_diameter * (y - y_center + dcorr)
                      * np.tan(np.arcsin(numerical_aperture)))
    return y, theta


def laser_lineshape(lam, lam_center, bandwidth, legacy=False):
    """Lorentzian source spectrum, peak-normalized, evaluated at each wavelength.

        nu = c / lam,  nu0 = c / lam_center,  dnu = c * bandwidth / lam_center^2
        L(nu) = dnu / (2 pi ((nu - nu0)^2 + (dnu/2)^2)),  then L / max(L)

    legacy=True reproduces main.m, which evaluated L on a frequency grid
    linspace(c/lam_max, c/lam_min) and multiplied it index-by-index with data
    ordered by increasing WAVELENGTH, so the spectrum was applied reversed
    (and on a slightly different grid). main.m also used c = 3e8.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    _positive(bandwidth, "bandwidth")
    c = 3e8 if legacy else C_LIGHT
    nu0 = c / lam_center
    dnu = c / lam_center ** 2 * bandwidth
    nu = np.linspace(c / lam[-1], c / lam[0], lam.size) if legacy else c / lam
    shape = dnu / (2 * np.pi * ((nu - nu0) ** 2 + (dnu / 2) ** 2))
    return shape / shape.max()


def seals_min_resolvable_angle(detector_aperture, target_distance):
    """Paper Sec. 3: delta_theta_min = 2 atan(d / (2 L))  [rad]."""
    _positive(detector_aperture, "detector_aperture"); _positive(target_distance, "target_distance")
    return 2 * np.arctan(detector_aperture / (2 * target_distance))


def seals_wavelength_resolution(lam_center, groove_spacing, beam_waist, order=1):
    """Paper Sec. 3: d(lam) = lam0 * d / (2 * order * w0)  [m]."""
    return lam_center * groove_spacing / (2 * order * beam_waist)


# ----------------------------------------------------------------------
# Motion: Brownian diffusion seen by dynamic light scattering
# ----------------------------------------------------------------------
def stokes_einstein_diffusion(dia, temperature=293.15, viscosity=1.0016e-3):
    """D = k_B T / (3 pi eta dia)  [m^2/s] for a sphere in a viscous liquid.

    Default viscosity is water at 20 C. Not meaningful for particles dried on
    a surface or suspended in air without a different drag model.
    """
    _positive(dia, "dia"); _positive(temperature, "temperature"); _positive(viscosity, "viscosity")
    return K_BOLTZMANN * temperature / (3 * np.pi * viscosity * dia)


def diameter_from_diffusion(diffusion, temperature=293.15, viscosity=1.0016e-3):
    """Invert Stokes-Einstein: hydrodynamic diameter from a measured D [m]."""
    _positive(diffusion, "diffusion")
    return K_BOLTZMANN * temperature / (3 * np.pi * viscosity * diffusion)


def scattering_vector(lam, nmed, theta):
    """q = (4 pi nmed / lam) sin(theta / 2)  [rad/m], the momentum transfer."""
    return 2 * wavenumber(lam, nmed) * np.sin(np.asarray(theta, dtype=float) / 2)


def dls_decay_rate(dia, lam, nmed, theta, temperature=293.15, viscosity=1.0016e-3):
    """Gamma = D q^2 [1/s]. Field autocorrelation g1(tau) = exp(-Gamma tau);
    intensity autocorrelation g2(tau) = 1 + beta exp(-2 Gamma tau)."""
    return (stokes_einstein_diffusion(dia, temperature, viscosity)
            * scattering_vector(lam, nmed, theta) ** 2)


# Default parameters of main.m (the "d" branch), in SI units and radians.
SEALS_DEFAULTS = {
    "dia": 9940e-9, "npar": 1.39, "nmed": 1.0, "r": 0.1,
    "groove_spacing": 9.0909e-7, "grating_distance": 0.065, "grating_angle": 0.9023,
    "dcorr": -4.2448e-4, "pupil_diameter": 0.0058, "numerical_aperture": 0.7,
    "measurement_angle": np.deg2rad(20.0),
    "lam_min": 1580e-9, "lam_max": 1600e-9, "n_wavelengths": 500,
    "source_bandwidth": 20e-9,
}
