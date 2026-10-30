"""
Laser eye safety: exposure, maximum permissible exposure (MPE), hazard distance, eyewear OD.

EDUCATIONAL ONLY. This is not a laser-safety determination. Real decisions use the current
ANSI Z136.1 (or IEC 60825-1) edition in full and a Laser Safety Officer.

In plain words
--------------
A laser is dangerous to the eye when the power per area reaching the cornea exceeds the
**MPE**. Four numbers answer most public-safety questions:

* **Irradiance at distance r.** The beam starts with diameter a and spreads at full-angle
  divergence phi, so D(r) = a + r phi (small angles), and E(r) = 4P / (pi D(r)^2).
* **MPE.** For visible light (400-700 nm) and exposures 1.8e-5 s <= t <= 10 s, the
  radiant-exposure limit is H = 1.8 t^0.75 mJ/cm^2, i.e. irradiance MPE = H/t. The blink
  reflex (t = 0.25 s) gives about 2.5 mW/cm^2.
* **NOHD (nominal ocular hazard distance)**, where E(r) falls to the MPE:
  NOHD = (sqrt(4P / (pi MPE)) - a) / phi. Inside it the beam can injure an eye. This is why
  pointing a "laser pointer" at an aircraft or a driver is a public-safety crime.
* **Optical density** of protective eyewear: OD = log10(E / MPE). Transmission is 10^-OD, so
  OD 3 passes 1/1000 of the light.

Modern-physics link: the same beam is a stream of photons, N/t = P lambda / (h c).

Units: power W, lengths cm, divergence rad, irradiance W/cm^2, time s, wavelength nm.

Sources (IEEE style)
--------------------
[1] American National Standard for Safe Use of Lasers, ANSI Z136.1-2007, Laser Institute of
    America, 2007, Table 5a (visible MPE 1.8 t^0.75 mJ/cm^2) and Sec. 8 / App. B (NOHD, OD).
    Later editions (2014, 2022) revised parts of the tables; check the current one.
[2] IEC 60825-1:2014, Safety of laser products, Part 1 (same visible limit, 18 t^0.75 J/m^2).
[3] D. Sliney and M. Wolbarsht, Safety with Lasers and Other Optical Sources. Plenum, 1980.
"""
import math

import numpy as np

HC_EV_NM = 1239.84198
EV_J = 1.602176634e-19
BLINK_S = 0.25


def _pos(name, val):
    if not np.all(np.asarray(val) > 0):
        raise ValueError(f"{name} must be > 0, got {val}")


def photon_rate(power_w, wavelength_nm):
    """Photons per second, P / (hc / lambda)."""
    _pos("power_w", power_w)
    _pos("wavelength_nm", wavelength_nm)
    return power_w / (HC_EV_NM / wavelength_nm * EV_J)


def beam_diameter(r_cm, a_cm, phi_rad):
    """D(r) = a + r phi (small-angle, ANSI convention)."""
    _pos("a_cm", a_cm)
    if phi_rad < 0 or np.any(np.asarray(r_cm) < 0):
        raise ValueError("phi_rad and r_cm must be >= 0")
    return a_cm + np.asarray(r_cm, dtype=float) * phi_rad


def irradiance(power_w, r_cm, a_cm, phi_rad):
    """E(r) = 4 P / (pi D(r)^2) in W/cm^2."""
    _pos("power_w", power_w)
    return 4 * power_w / (math.pi * beam_diameter(r_cm, a_cm, phi_rad) ** 2)


def mpe_visible(t_s, wavelength_nm=532.0):
    """Irradiance MPE (W/cm^2), 400-700 nm, 1.8e-5 s <= t <= 10 s: 1.8e-3 t^0.75 / t."""
    if not 400 <= wavelength_nm <= 700:
        raise ValueError(f"this simplified MPE only covers 400-700 nm, got {wavelength_nm} nm")
    if not 1.8e-5 <= t_s <= 10:
        raise ValueError(f"this simplified MPE only covers 1.8e-5 s <= t <= 10 s, got {t_s} s")
    return 1.8e-3 * t_s ** 0.75 / t_s


def nohd(power_w, mpe_w_cm2, a_cm, phi_rad):
    """Nominal ocular hazard distance in cm (0 if the beam is below the MPE at the exit)."""
    _pos("power_w", power_w)
    _pos("mpe_w_cm2", mpe_w_cm2)
    _pos("a_cm", a_cm)
    _pos("phi_rad", phi_rad)
    return max(0.0, (math.sqrt(4 * power_w / (math.pi * mpe_w_cm2)) - a_cm) / phi_rad)


def required_od(exposure_w_cm2, mpe_w_cm2):
    """Minimum eyewear optical density log10(E / MPE); 0 if already below the MPE."""
    _pos("exposure_w_cm2", exposure_w_cm2)
    _pos("mpe_w_cm2", mpe_w_cm2)
    return max(0.0, math.log10(exposure_w_cm2 / mpe_w_cm2))


def transmission(od):
    """Fraction of light an OD filter passes, 10^-OD."""
    if od < 0:
        raise ValueError(f"OD must be >= 0, got {od}")
    return 10.0 ** -od


def eyewear_check(power_w, wavelength_nm, r_cm, a_cm, phi_rad, od, t_s=BLINK_S):
    """Educational scenario check: exposure behind eyewear vs the visible MPE.

    Returns a dict with the bare-eye exposure, MPE, required OD, exposure behind the
    eyewear, and margin = MPE / exposure_behind (> 1 means below the MPE).
    """
    E_bare = float(irradiance(power_w, r_cm, a_cm, phi_rad))
    mpe = mpe_visible(t_s, wavelength_nm)
    behind = E_bare * transmission(od)
    return {"E_bare_W_cm2": E_bare, "MPE_W_cm2": mpe, "OD_required": required_od(E_bare, mpe),
            "E_behind_W_cm2": behind, "margin": mpe / behind, "below_MPE": behind <= mpe}
