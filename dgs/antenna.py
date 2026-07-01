"""Antenna theory: half-wave dipole, radiation pattern, arrays, beam steering.

THE HALF-WAVE DIPOLE is the most important antenna in engineering:
  - Length L = lambda/2 (physical wire cut to half the operating wavelength)
  - Radiation resistance R_rad = 73.1 Ohm (calculated from integrating Poynting vector)
  - Directivity D = 1.64 (2.15 dBi)
  - Radiation pattern: donut shape, null on axis, max broadside

Every phone, WiFi router, and radio telescope starts here. The physics is the
Larmor formula applied to a sinusoidal current distribution on a wire:
  I(z) = I_0 * cos(k*z)  for -L/2 <= z <= L/2

The resulting far-field electric field:
  E_theta ~ [cos(pi/2 * cos(theta)) / sin(theta)]  (the dipole pattern function)

RADIATION PATTERN vs DIRECTIVITY:
  Pattern: F(theta, phi) -- normalized angular power distribution
  Directivity: D = 4*pi * F_max / integral(F d_Omega)
  Gain: G = eta * D  where eta is the efficiency (0-1)
  EIRP: Effective Isotropic Radiated Power = P_tx * G_tx

ANTENNA ARRAYS:
  N dipoles fed with phase shifts delta = k*d*sin(theta_0) steer the beam
  to angle theta_0. Array Factor AF(theta) = sum_{n=0}^{N-1} exp(i*n*psi)
  where psi = k*d*sin(theta) + delta. This is a DFT -- the same FFT the
  GS algorithm runs, applied spatially instead of temporally.

NEAR FIELD vs FAR FIELD:
  Boundary: R_ff = 2*D^2 / lambda  (Fraunhofer distance)
  Near field: reactive energy dominates (inductor/capacitor character)
  Far field: radiation dominates, fields fall as 1/R (Fraunhofer zone)

IMPEDANCE MATCHING (quarter-wave transformer):
  Z_in = Z_0^2 / Z_L  for a quarter-wave (L=lambda/4) transmission line section.
  To match 50 Ohm coax to 73 Ohm dipole: Z_0 = sqrt(50*73) = 60.4 Ohm.
"""
import numpy as np
import sympy as sp


# ── half-wave dipole radiation pattern ───────────────────────────────

def dipole_pattern(theta_rad):
    """Half-wave dipole radiation pattern function F(theta).

    F(theta) = [cos(pi/2 * cos(theta)) / sin(theta)]^2

    theta is the polar angle from the dipole axis (z-axis).
    Pattern is zero on-axis (theta=0, pi) and maximum broadside (theta=pi/2).
    """
    theta = np.asarray(theta_rad, dtype=float)
    # handle singularities at theta=0 and theta=pi
    sin_th = np.sin(theta)
    safe_sin = np.where(np.abs(sin_th) < 1e-10, 1e-10, sin_th)
    numerator = np.cos(np.pi / 2 * np.cos(theta))
    F = (numerator / safe_sin) ** 2
    # zero out the true singularity points
    F = np.where(np.abs(sin_th) < 1e-10, 0.0, F)
    return F


def dipole_pattern_grid(n_theta=360, n_phi=1):
    """Evaluate dipole pattern on a theta grid [0, pi].

    Returns theta_rad array and F(theta) array (normalized to max=1).
    """
    theta = np.linspace(0, np.pi, n_theta)
    F = dipole_pattern(theta)
    F_norm = F / np.max(F) if np.max(F) > 0 else F
    return {"theta_rad": theta, "F": F, "F_norm": F_norm,
            "F_dB": 10 * np.log10(np.maximum(F_norm, 1e-10))}


def dipole_directivity():
    """Numerical directivity of the half-wave dipole.

    D = 4*pi * F_max / integral(F(theta) * sin(theta) d_theta d_phi)
      = 1.6409  (exact numerical value)
      = 2.15 dBi
    """
    theta = np.linspace(1e-6, np.pi - 1e-6, 10000)
    F = dipole_pattern(theta)
    # integrate over solid angle: d_Omega = sin(theta) d_theta d_phi
    integral = 2 * np.pi * np.trapezoid(F * np.sin(theta), theta)
    D = 4 * np.pi * np.max(F) / integral
    return {"directivity_linear": D,
            "directivity_dBi": 10 * np.log10(D),
            "gain_dBi_lossless": 10 * np.log10(D)}


# ── half-wave dipole parameters ───────────────────────────────────────

def half_wave_dipole(freq_Hz):
    """Complete half-wave dipole specification at given frequency.

    Returns length, radiation resistance, reactance (near resonance ~ 0),
    directivity, and far-field boundary.
    """
    if freq_Hz <= 0:
        raise ValueError("freq_Hz must be positive")
    c = 2.99792458e8
    lam = c / freq_Hz
    L = lam / 2
    R_rad = 73.1     # Ohm (numerical, Griffiths Eq 11.22 evaluated)
    X_in = 42.5      # Ohm (input reactance at exact L=lambda/2; ~ 0 at resonance)
    R_ff = 2 * L**2 / lam   # Fraunhofer distance
    D = dipole_directivity()
    return {
        "freq_Hz": freq_Hz,
        "wavelength_m": lam,
        "length_m": L,
        "length_cm": L * 100,
        "R_rad_ohm": R_rad,
        "X_in_ohm": X_in,
        "Z_in_ohm": complex(R_rad, X_in),
        "directivity_dBi": D["directivity_dBi"],
        "far_field_boundary_m": R_ff,
    }


# ── quarter-wave impedance matching ──────────────────────────────────

def quarter_wave_transformer(Z_source, Z_load, freq_Hz):
    """Quarter-wave transmission line transformer: Z_0 = sqrt(Z_S * Z_L).

    Inserts a lambda/4 section of transmission line between source and load.
    The characteristic impedance Z_0 of that section transforms Z_L -> Z_S.
    Used to match 50 Ohm coax to 73 Ohm dipole.

    Returns: Z_0 of the matching section, and VSWR before/after matching.
    """
    Z_S = float(Z_source)
    Z_L = float(Z_load)
    if Z_S <= 0 or Z_L <= 0:
        raise ValueError("impedances must be positive reals")
    Z_0_match = np.sqrt(Z_S * Z_L)
    c = 2.99792458e8
    lam = c / freq_Hz
    L_match = lam / 4

    # VSWR before matching (direct connection)
    gamma_before = (Z_L - Z_S) / (Z_L + Z_S)
    vswr_before = (1 + abs(gamma_before)) / (1 - abs(gamma_before))

    # After matching: Z_in = Z_0_match^2 / Z_L = Z_S (perfect at center freq)
    Z_in_matched = Z_0_match**2 / Z_L
    gamma_after = (Z_in_matched - Z_S) / (Z_in_matched + Z_S)
    vswr_after = (1 + abs(gamma_after)) / (1 - abs(gamma_after))

    return {
        "Z_0_match_ohm": Z_0_match,
        "length_m": L_match,
        "length_cm": L_match * 100,
        "VSWR_before": vswr_before,
        "VSWR_after": vswr_after,
        "Z_in_matched_ohm": Z_in_matched,
    }


# ── antenna array: N-element uniform linear array ────────────────────

def array_factor(N, d_over_lambda, theta_rad, steering_angle_rad=0.0):
    """Array Factor for N isotropic elements in a uniform linear array (ULA).

    AF(theta) = sum_{n=0}^{N-1} exp(i * n * psi)
    where psi = 2*pi * (d/lambda) * (sin(theta) - sin(theta_0))

    This is a DISCRETE FOURIER TRANSFORM evaluated on the unit circle --
    the same FFT that drives the GS phase retrieval algorithm, applied
    spatially. Steering the beam is equivalent to applying a linear phase
    ramp across the array elements (just like a frequency shift in DSP).

    Parameters
    ----------
    N : int               -- number of elements
    d_over_lambda : float -- element spacing in wavelengths (usually 0.5)
    theta_rad : array     -- observation angles
    steering_angle_rad : float -- desired beam direction

    Returns normalized |AF|^2.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    theta = np.asarray(theta_rad, dtype=float)
    psi = 2 * np.pi * d_over_lambda * (np.sin(theta) - np.sin(steering_angle_rad))
    # sum of geometric series (avoid division by zero when psi~0)
    n_arr = np.arange(N)
    # efficient: AF = sum exp(i*n*psi) for each theta
    AF = np.sum(np.exp(1j * np.outer(n_arr, psi)), axis=0)
    AF_power = np.abs(AF)**2 / N**2   # normalize to 1
    return {"theta_rad": theta, "AF_power": AF_power,
            "AF_dB": 10 * np.log10(np.maximum(AF_power, 1e-10)),
            "N": N, "d_over_lambda": d_over_lambda,
            "steering_deg": np.degrees(steering_angle_rad)}


def beam_width_3dB(N, d_over_lambda, steering_angle_rad=0.0, n_pts=3600):
    """Approximate 3 dB beam width of an N-element ULA (degrees).

    BWFN ~ 0.886 * lambda / (N * d)  (radians, broadside)
    Converted to degrees.
    """
    BWFN_rad = 0.886 / (N * d_over_lambda)   # half-power beam width in sin(theta) space
    BWFN_deg = np.degrees(np.arcsin(min(BWFN_rad, 1.0))) * 2
    return {"BWFN_3dB_deg": BWFN_deg, "N": N, "d_over_lambda": d_over_lambda}


# ── near field vs far field ───────────────────────────────────────────

def fraunhofer_distance(aperture_m, wavelength_m):
    """Far-field (Fraunhofer) distance: R_ff = 2 * D^2 / lambda.

    Beyond R_ff: radiation pattern is stable (far field, 1/R fields).
    Within R_ff: near field -- reactive energy, pattern changes with distance.

    Aperture D here is the maximum physical dimension (not directivity).
    """
    if aperture_m <= 0 or wavelength_m <= 0:
        raise ValueError("aperture and wavelength must be positive")
    R_ff = 2 * aperture_m**2 / wavelength_m
    R_reactive = wavelength_m / (2 * np.pi)   # reactive near-field boundary
    return {"R_ff_m": R_ff, "R_reactive_m": R_reactive,
            "aperture_m": aperture_m, "wavelength_m": wavelength_m}


# ── EIRP and link budget ──────────────────────────────────────────────

def link_budget(P_tx_W, G_tx_dBi, G_rx_dBi, freq_Hz, distance_m):
    """Simple Friis link budget.

    P_rx = P_tx * G_tx * G_rx * (lambda / (4*pi*R))^2

    Returns received power, path loss, and link margin.
    Used for: satellite comm, WiFi planning, optical free-space links.
    """
    if P_tx_W <= 0 or freq_Hz <= 0 or distance_m <= 0:
        raise ValueError("P_tx, freq_Hz, and distance_m must be positive")
    c = 2.99792458e8
    lam = c / freq_Hz
    G_tx = 10 ** (G_tx_dBi / 10)
    G_rx = 10 ** (G_rx_dBi / 10)
    P_rx = P_tx_W * G_tx * G_rx * (lam / (4 * np.pi * distance_m))**2
    FSPL_dB = 20 * np.log10(4 * np.pi * distance_m / lam)
    EIRP_dBm = 10 * np.log10(P_tx_W * G_tx * 1000)
    return {
        "P_rx_W": P_rx,
        "P_rx_dBm": 10 * np.log10(P_rx * 1000),
        "FSPL_dB": FSPL_dB,
        "EIRP_dBm": EIRP_dBm,
        "path_loss_dB": FSPL_dB,
    }


# ── SymPy formalism ───────────────────────────────────────────────────

def antenna_sympy_5():
    """Five key antenna equations in SymPy."""
    theta, phi = sp.symbols('theta phi', real=True)
    Z_S, Z_L, Z_0 = sp.symbols('Z_S Z_L Z_0', positive=True)
    N, d, lam = sp.symbols('N d lambda', positive=True)
    n_sym = sp.Symbol('n')
    psi = sp.Symbol('psi')
    R, D_sym, lam2 = sp.symbols('R D lambda', positive=True)

    return {
        "Dipole_pattern":
            sp.Eq(sp.Symbol('F(theta)'),
                  (sp.cos(sp.pi/2 * sp.cos(theta)) / sp.sin(theta))**2),
        "Array_factor":
            sp.Eq(sp.Symbol('AF'),
                  sp.Sum(sp.exp(sp.I * n_sym * psi), (n_sym, 0, N-1))),
        "Quarter_wave_match":
            sp.Eq(Z_0, sp.sqrt(Z_S * Z_L)),
        "Fraunhofer_distance":
            sp.Eq(sp.Symbol('R_ff'), 2 * D_sym**2 / lam2),
        "Friis_equation":
            sp.Eq(sp.Symbol('P_rx'),
                  sp.Symbol('P_tx') * sp.Symbol('G_tx') * sp.Symbol('G_rx')
                  * (lam2 / (4 * sp.pi * R))**2),
    }


if __name__ == "__main__":
    print("=== Half-wave dipole at 2.4 GHz (WiFi) ===")
    d = half_wave_dipole(2.4e9)
    print(f"  Length:        {d['length_cm']:.1f} cm")
    print(f"  R_rad:         {d['R_rad_ohm']:.1f} Ohm")
    print(f"  Directivity:   {d['directivity_dBi']:.2f} dBi")
    print(f"  Far field > :  {d['far_field_boundary_m']*100:.1f} cm")

    print("\n=== Quarter-wave transformer: 50 Ohm -> 73 Ohm ===")
    qt = quarter_wave_transformer(50, 73, 2.4e9)
    print(f"  Z_0 of matching section: {qt['Z_0_match_ohm']:.1f} Ohm")
    print(f"  Matching section length: {qt['length_cm']:.1f} cm")
    print(f"  VSWR before: {qt['VSWR_before']:.2f}")
    print(f"  VSWR after:  {qt['VSWR_after']:.4f}")

    print("\n=== 4-element ULA at 0.5 lambda spacing, steered to 30 deg ===")
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    af = array_factor(4, 0.5, theta, steering_angle_rad=np.radians(30))
    peak_idx = np.argmax(af["AF_power"])
    print(f"  Beam peak at: {np.degrees(theta[peak_idx]):.1f} deg")
    bw = beam_width_3dB(4, 0.5)
    print(f"  3dB beam width: {bw['BWFN_3dB_deg']:.1f} deg")

    print("\n=== Friis: WiFi at 10 m, 20 dBm TX, dipole antennas ===")
    lb = link_budget(P_tx_W=0.1, G_tx_dBi=2.15, G_rx_dBi=2.15,
                     freq_Hz=2.4e9, distance_m=10)
    print(f"  P_rx = {lb['P_rx_dBm']:.1f} dBm  (path loss {lb['FSPL_dB']:.1f} dB)")

    print("\n=== SymPy 5 ===")
    for k, eq in antenna_sympy_5().items():
        print(f"  {k}: {eq}")
