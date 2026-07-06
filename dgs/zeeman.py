"""Zeeman effect in photonics units -- from mu_B*B to what the spectrometer sees.

A magnetic field splits an atomic level (L, S, J) into 2J+1 sublevels:

    delta_E = g_J * mu_B * B * m_J,     m_J = -J, -J+1, ..., +J

with the Lande g-factor

    g_J = 1 + [J(J+1) + S(S+1) - L(L+1)] / (2 J(J+1)).

S = 0 gives g_J = 1 for every term -> all transition lines shift by
(delta m_J)*mu_B*B/h and only 3 lines appear (the NORMAL Zeeman triplet).
With spin, upper and lower terms have different g -> more lines (ANOMALOUS).

THE PHOTONICS ANGLE (why this module exists next to a time-stretch receiver):
lab instruments do not read joules. The chain is

    energy (J)  ->  frequency (Hz):   delta_f = delta_E / h
                ->  wavelength (nm):  delta_lambda = -lambda^2 * delta_f / c

mu_B/h = 13.996 GHz/T sets the scale: at 1 T a g=1 line moves ~14 GHz,
which at 589 nm (sodium D) is only ~16 pm -- exactly why Zeeman needed a
high-resolution spectrograph in 1896, and why in the telecom C-band the
same GHz-scale shifts are routine DWDM channel spacings (100 GHz = 0.8 nm
at 1550 nm). Same delta_f, different lambda^2 lever arm.

NumPy only (py -3.13 safe). Education.
"""

import numpy as np

# CODATA-ish constants (SI)
H = 6.62607015e-34          # J s, Planck
C = 2.99792458e8            # m/s
MU_B = 9.2740100783e-24     # J/T, Bohr magneton
MU_B_HZ_PER_T = MU_B / H    # ~1.3996e10 Hz/T: the Zeeman frequency scale


# ----------------------------------------------------------------------
# THz <-> wavelength: the conversion every photonics datasheet assumes
# ----------------------------------------------------------------------

def thz_to_wavelength_nm(f_thz):
    """lambda = c/f. Note c in nm*THz is 299792.458, so the SAME formula
    converts both ways -- the map is its own inverse (an involution)."""
    f_thz = np.asarray(f_thz, float)
    if np.any(f_thz <= 0):
        raise ValueError("frequency must be positive (THz)")
    return 299792.458 / f_thz


def wavelength_nm_to_thz(lambda_nm):
    """f = c/lambda; numerically identical to thz_to_wavelength_nm."""
    lambda_nm = np.asarray(lambda_nm, float)
    if np.any(lambda_nm <= 0):
        raise ValueError("wavelength must be positive (nm)")
    return 299792.458 / lambda_nm


def wavelength_shift_nm(lambda_nm, delta_f_hz):
    """First-order lever arm: delta_lambda = -lambda^2 * delta_f / c.
    Negative sign: higher frequency = shorter wavelength. The lambda^2
    factor is why 100 GHz is 0.80 nm at 1550 nm but only 0.012 nm at 589 nm/GHz-scale.
    """
    lambda_nm = np.asarray(lambda_nm, float)
    if np.any(lambda_nm <= 0):
        raise ValueError("wavelength must be positive (nm)")
    lam_m = lambda_nm * 1e-9
    return -(lam_m ** 2) * np.asarray(delta_f_hz, float) / C * 1e9


def common_bands():
    """Reference cases: wavelength (nm) -> frequency (THz), rounded to 0.01."""
    lines = {
        "telecom C-band": 1550.0,
        "O-band": 1310.0,
        "Nd:YAG": 1064.0,
        "VCSEL/datacom": 850.0,
        "HeNe red": 632.8,
        "sodium D2": 589.0,
        "green DPSS": 532.0,
    }
    return {name: (lam, round(float(wavelength_nm_to_thz(lam)), 2))
            for name, lam in lines.items()}


# ----------------------------------------------------------------------
# Atomic structure: Lande g and sublevel splitting
# ----------------------------------------------------------------------

def _check_term(L, S, J):
    if L < 0 or S < 0 or J < 0:
        raise ValueError("L, S, J must be non-negative")
    if abs(round(2 * J) - 2 * J) > 1e-9:
        raise ValueError("J must be integer or half-integer")
    if J < abs(L - S) - 1e-9 or J > L + S + 1e-9:
        raise ValueError(f"J={J} outside |L-S|..L+S for L={L}, S={S}")


def lande_g(L, S, J):
    """g_J = 1 + [J(J+1)+S(S+1)-L(L+1)] / (2J(J+1)).
    S=0 -> g=1 (pure orbital, normal Zeeman); L=0 -> g=2 (pure spin, ESR)."""
    _check_term(L, S, J)
    if J == 0:
        return 0.0  # single m_J=0 sublevel: no splitting, g is moot
    return 1.0 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))


def zeeman_sublevels(g_J, J, B_tesla):
    """(m_J array, energy shifts in J): delta_E = g_J*mu_B*B*m_J.
    2J+1 equally spaced levels, symmetric about the unshifted line."""
    if B_tesla < 0:
        raise ValueError("B must be >= 0 tesla")
    if J < 0 or abs(round(2 * J) - 2 * J) > 1e-9:
        raise ValueError("J must be a non-negative (half-)integer")
    m_J = np.arange(-J, J + 0.5, 1.0)
    return m_J, g_J * MU_B * B_tesla * m_J


def splitting_frequency_hz(g_J, B_tesla):
    """Spacing between adjacent m_J sublevels, in Hz: g_J * mu_B * B / h.
    For g=1 this is 13.996 GHz per tesla -- the number to memorize."""
    if B_tesla < 0:
        raise ValueError("B must be >= 0 tesla")
    return g_J * MU_B_HZ_PER_T * B_tesla


# ----------------------------------------------------------------------
# Transition lines: what actually shows up on the spectrograph
# ----------------------------------------------------------------------

def transition_shifts_hz(L_u, S_u, J_u, L_l, S_l, J_l, B_tesla):
    """Frequency shifts (Hz, sorted, distinct) of all electric-dipole-allowed
    Zeeman components: delta m_J in {-1, 0, +1}, each line shifted by
    (g_u*m_u - g_l*m_l) * mu_B * B / h relative to the zero-field line.

    S=0 on both terms -> exactly 3 shifts (normal triplet).
    Different g_u, g_l -> up to 3*(2*min(J_u,J_l)+1)-ish distinct lines (anomalous).
    """
    _check_term(L_u, S_u, J_u)
    _check_term(L_l, S_l, J_l)
    if B_tesla < 0:
        raise ValueError("B must be >= 0 tesla")
    g_u, g_l = lande_g(L_u, S_u, J_u), lande_g(L_l, S_l, J_l)
    shifts = []
    for m_u in np.arange(-J_u, J_u + 0.5, 1.0):
        for m_l in np.arange(-J_l, J_l + 0.5, 1.0):
            if abs(m_u - m_l) <= 1 + 1e-9:  # selection rule delta m = 0, +/-1
                shifts.append((g_u * m_u - g_l * m_l) * MU_B_HZ_PER_T * B_tesla)
    return np.unique(np.round(np.array(shifts), 6))


def zeeman_wavelength_report(lambda0_nm, L_u, S_u, J_u, L_l, S_l, J_l, B_tesla):
    """Full instrument-facing picture of one Zeeman-split line:
    zero-field wavelength/frequency, each component's shift in GHz and pm,
    and the total spread -- the resolution your spectrometer must beat."""
    f0_thz = float(wavelength_nm_to_thz(lambda0_nm))
    shifts = transition_shifts_hz(L_u, S_u, J_u, L_l, S_l, J_l, B_tesla)
    dlam = wavelength_shift_nm(lambda0_nm, shifts)
    return {
        "lambda0_nm": float(lambda0_nm),
        "f0_thz": f0_thz,
        "n_lines": int(len(shifts)),
        "shifts_ghz": (shifts / 1e9).tolist(),
        "shifts_pm": (dlam * 1e3).tolist(),
        "spread_pm": float((dlam.max() - dlam.min()) * 1e3),
    }


if __name__ == "__main__":
    print("=== THz <-> wavelength cases ===")
    for name, (lam, f) in common_bands().items():
        print(f"  {name:15s} {lam:7.1f} nm  <->  {f:7.2f} THz")

    print("\n=== Zeeman scale: mu_B/h = %.4f GHz/T ===" % (MU_B_HZ_PER_T / 1e9))
    print("DWDM sanity check: 100 GHz at 1550 nm =",
          f"{abs(float(wavelength_shift_nm(1550.0, 100e9))):.3f} nm")

    print("\n=== Sodium D2 (589 nm), 2P_3/2 -> 2S_1/2, B = 1 T (anomalous) ===")
    rep = zeeman_wavelength_report(589.0, 1, 0.5, 1.5, 0, 0.5, 0.5, 1.0)
    print(f"  {rep['n_lines']} lines; shifts (GHz): "
          + ", ".join(f"{s:+.2f}" for s in rep["shifts_ghz"]))
    print(f"  wavelength spread: {rep['spread_pm']:.1f} pm "
          "(needs a spectrograph resolving ~1 in 40,000 -- Zeeman 1896)")

    print("\n=== Same terms with S=0 (hypothetical): normal triplet ===")
    rep3 = zeeman_wavelength_report(589.0, 1, 0, 1, 0, 0, 0, 1.0)
    print(f"  {rep3['n_lines']} lines; shifts (GHz): "
          + ", ".join(f"{s:+.2f}" for s in rep3["shifts_ghz"]))
