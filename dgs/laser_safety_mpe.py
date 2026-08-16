"""Laser eye safety: Maximum Permissible Exposure (MPE), laser
classification, and Nominal Ocular Hazard Distance (NOHD) -- resolving a
caveat dgs.retinal_scan_imaging's own docstring explicitly left open:
"ANSI Z136.1 eye-safe exposure limits are wavelength- and duration-
dependent and would need to be looked up per configuration before this
became a real imaging-dose claim, not assumed here." This module
implements the well-documented STRUCTURE of that calculation (the
time/wavelength-dependent power-law form used throughout laser-safety
education), not a verbatim reproduction of the current standard's tables.

****************************************************************************
THIS MODULE IS EDUCATIONAL, NOT A SUBSTITUTE FOR A CERTIFIED LASER SAFETY
OFFICER OR THE ACTUAL CURRENT ANSI Z136.1 STANDARD. The formulas below are
simplified/illustrative versions of the widely-published GENERAL FORM of
MPE calculations (commonly presented this way in laser-safety textbooks
and coursework), not a verbatim transcription of the current standard's
exact tables, footnotes, and edge cases -- real laser system safety
classification and hazard-distance determination MUST be done against the
actual current standard by a qualified Laser Safety Officer. Do not use
this module's outputs as the basis for an actual safety decision.
****************************************************************************

THE STRUCTURE (illustrative, thermal-hazard regime, 18us-10s exposures,
400-1050nm retinal hazard region):

    MPE(t) = 1.8 * C_A(wavelength) * t^0.75   mJ/cm^2

  C_A = 1                              for 400-700nm (visible)
  C_A = 10^(0.002*(wavelength-700))    for 700-1050nm (near-IR; the
                                        retina tolerates progressively
                                        more NIR exposure than visible,
                                        the physical basis for the rising
                                        correction factor)

NOHD (Nominal Ocular Hazard Distance): the far-field distance at which a
divergent beam's irradiance drops to the CW long-duration MPE --

    NOHD = (2/phi) * sqrt(P / (pi * MPE))

phi = full beam divergence angle (rad), P = beam power (W), MPE in W/m^2.
"""

from __future__ import annotations
import numpy as np

MODULE_SAFETY_DISCLAIMER = (
    "Educational illustration only -- NOT a substitute for the actual current "
    "ANSI Z136.1 standard or a certified Laser Safety Officer's review. Do not "
    "use these outputs as the basis for a real safety decision."
)


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ── 1. Wavelength correction factor C_A ─────────────────────────────────────

def wavelength_correction_CA(wavelength_nm: float) -> float:
    """C_A(wavelength): 1.0 for 400-700nm (visible), rising as
    10^(0.002*(wavelength-700)) for 700-1050nm (near-IR) -- the
    illustrative form used throughout laser-safety education, NOT a
    verbatim transcription of the current standard's exact tables (see
    module docstring)."""
    if 400 <= wavelength_nm <= 700:
        return 1.0
    elif 700 < wavelength_nm <= 1050:
        return 10 ** (0.002 * (wavelength_nm - 700))
    else:
        raise ValueError(f"wavelength_nm={wavelength_nm}: this illustrative model only "
                         f"covers 400-1050nm (the retinal hazard region); a real system "
                         f"outside this range needs the actual standard's other bands.")


# ── 2. MPE, thermal regime (18us-10s) ────────────────────────────────────────

def mpe_thermal_regime(exposure_s: float, wavelength_nm: float) -> float:
    """MPE(t) = 1.8 * C_A * t^0.75 mJ/cm^2, valid (illustratively) for
    18us <= t <= 10s -- returns J/cm^2 (SI-consistent units, not mJ)."""
    _validate_positive(exposure_s=exposure_s)
    if not (18e-6 <= exposure_s <= 10.0):
        raise ValueError(f"exposure_s={exposure_s}: this illustrative thermal-regime "
                         f"formula only covers 18us-10s; outside that range the actual "
                         f"standard uses different (photochemical- or single-pulse-)regime formulas.")
    C_A = wavelength_correction_CA(wavelength_nm)
    mpe_mJ_cm2 = 1.8 * C_A * exposure_s ** 0.75
    return mpe_mJ_cm2 * 1e-3   # mJ -> J


def mpe_cw_long_duration(wavelength_nm: float, exposure_s: float = 10.0,
                         baseline_mpe_W_cm2: float = 2.5e-3) -> float:
    """CW long-duration MPE (irradiance, W/cm^2), for exposures at or
    beyond the thermal regime's 10s upper bound, where MPE saturates to a
    roughly constant irradiance rather than continuing the t^0.75 growth
    -- baseline_mpe_W_cm2 is the illustrative visible-CW figure commonly
    cited in laser-safety education (an order-of-magnitude teaching
    value, not the exact current standard's number)."""
    _validate_positive(baseline_mpe_W_cm2=baseline_mpe_W_cm2)
    if exposure_s < 10.0:
        raise ValueError(f"exposure_s={exposure_s}: use mpe_thermal_regime for exposures under 10s")
    C_A = wavelength_correction_CA(wavelength_nm)
    return baseline_mpe_W_cm2 * C_A


# ── 3. Nominal Ocular Hazard Distance ───────────────────────────────────────

def nohd(power_W: float, mpe_W_cm2: float, divergence_rad: float) -> float:
    """NOHD = (2/phi)*sqrt(P/(pi*MPE)) -- the far-field distance at which
    a divergent beam's irradiance drops to the MPE. Uses a far-field
    approximation (ignores the initial beam waist diameter, valid once
    the beam has expanded well past its starting size)."""
    _validate_positive(power_W=power_W, mpe_W_cm2=mpe_W_cm2, divergence_rad=divergence_rad)
    mpe_W_m2 = mpe_W_cm2 * 1e4
    return (2.0 / divergence_rad) * np.sqrt(power_W / (np.pi * mpe_W_m2))


def verify_nohd_scaling(power_W: float = 5e-3, mpe_W_cm2: float = 2.5e-3,
                        divergence_rad: float = 1e-3) -> dict:
    """CHECKED, not assumed: NOHD must increase with power (more power
    stays hazardous farther out) and with TIGHTER divergence (a more
    collimated beam stays dangerous over a longer range) -- both
    directions of the physical intuition, verified numerically."""
    baseline = nohd(power_W, mpe_W_cm2, divergence_rad)
    higher_power = nohd(power_W * 2, mpe_W_cm2, divergence_rad)
    tighter_divergence = nohd(power_W, mpe_W_cm2, divergence_rad / 2)
    return {"baseline_NOHD_m": baseline, "double_power_NOHD_m": higher_power,
            "half_divergence_NOHD_m": tighter_divergence,
            "power_increases_NOHD": bool(higher_power > baseline),
            "tighter_divergence_increases_NOHD": bool(tighter_divergence > baseline)}


# ── 4. Simplified classification check ──────────────────────────────────────

def exceeds_class1_illustrative(power_W: float, beam_area_cm2: float, wavelength_nm: float,
                                exposure_s: float = 10.0) -> dict:
    """Illustrative Class 1 (inherently eye-safe, no engineering
    controls needed) check: compares the beam's irradiance/radiant
    exposure against this module's illustrative MPE. A real Class 1
    determination uses the standard's actual Accessible Emission Limit
    (AEL) tables, not a direct MPE comparison at the source -- this is a
    simplified stand-in for teaching the CONCEPT, not the real
    classification procedure."""
    _validate_positive(power_W=power_W, beam_area_cm2=beam_area_cm2)
    irradiance_W_cm2 = power_W / beam_area_cm2
    if exposure_s >= 10.0:
        mpe = mpe_cw_long_duration(wavelength_nm, exposure_s)
    else:
        mpe_energy = mpe_thermal_regime(exposure_s, wavelength_nm)
        mpe = mpe_energy / exposure_s   # J/cm^2 / s -> W/cm^2, average irradiance
    return {"irradiance_W_cm2": irradiance_W_cm2, "mpe_W_cm2": mpe,
            "exceeds_mpe": bool(irradiance_W_cm2 > mpe),
            "margin_factor": irradiance_W_cm2 / mpe}


# ── 5. Applying this to dgs.retinal_scan_imaging's flagged gap ─────────────

def check_retinal_scan_exposure_illustrative(power_W: float, beam_diameter_um: float,
                                             wavelength_nm: float = 850.0,
                                             exposure_s: float = 1.0) -> dict:
    """The check dgs.retinal_scan_imaging's own docstring flagged as
    needed but not implemented -- run here ILLUSTRATIVELY (see module
    disclaimer), for a STEAM-style retinal line-scan source, using this
    module's simplified thermal-regime MPE. Default wavelength (850nm) is
    a realistic near-IR value WITHIN this model's 400-1050nm scope --
    dgs.retinal_scan_imaging's own default (1550nm) falls in ANSI
    Z136.1's eye-safe IR band (corneal, not retinal, absorption
    dominates), a genuinely different hazard category this illustrative
    model does not implement; callers targeting that wavelength must
    consult the actual standard, not force this function to answer
    outside its documented range."""
    beam_radius_cm = (beam_diameter_um * 1e-4) / 2
    beam_area_cm2 = np.pi * beam_radius_cm ** 2
    irradiance_W_cm2 = power_W / beam_area_cm2
    mpe_energy_J_cm2 = mpe_thermal_regime(exposure_s, wavelength_nm)
    exposure_energy_J_cm2 = irradiance_W_cm2 * exposure_s
    return {"beam_area_cm2": beam_area_cm2, "irradiance_W_cm2": irradiance_W_cm2,
            "exposure_energy_J_cm2": exposure_energy_J_cm2, "mpe_J_cm2": mpe_energy_J_cm2,
            "exceeds_mpe": bool(exposure_energy_J_cm2 > mpe_energy_J_cm2),
            "margin_factor": exposure_energy_J_cm2 / mpe_energy_J_cm2,
            "disclaimer": MODULE_SAFETY_DISCLAIMER}


if __name__ == "__main__":
    print(f"*** {MODULE_SAFETY_DISCLAIMER} ***\n")

    print("=== 1. MPE, thermal regime (18us-10s), across wavelength and exposure time ===")
    for wl in (532, 633, 850, 1050):
        print(f"  wavelength={wl}nm, C_A={wavelength_correction_CA(wl):.3f}")
        for t in (1e-3, 0.01, 0.1, 1.0):
            mpe = mpe_thermal_regime(t, wl)
            print(f"    t={t:>6.3f}s: MPE={mpe:.2e} J/cm^2")

    print("\n=== 2. CW long-duration MPE ===")
    for wl in (633, 850, 1050):
        mpe = mpe_cw_long_duration(wl)
        print(f"  wavelength={wl}nm: CW MPE = {mpe*1000:.2f} mW/cm^2")

    print("\n=== 3. Nominal Ocular Hazard Distance ===")
    check = verify_nohd_scaling()
    print(f"  5mW pointer, 1mrad divergence: NOHD = {check['baseline_NOHD_m']:.2f} m")
    print(f"  double power: NOHD = {check['double_power_NOHD_m']:.2f} m "
          f"(increased: {check['power_increases_NOHD']})")
    print(f"  half divergence: NOHD = {check['half_divergence_NOHD_m']:.2f} m "
          f"(increased: {check['tighter_divergence_increases_NOHD']})")

    print("\n=== 4. Illustrative Class 1 check ===")
    result = exceeds_class1_illustrative(power_W=1e-3, beam_area_cm2=0.01, wavelength_nm=633.0)
    print(f"  1mW visible, 0.01cm^2 beam: irradiance={result['irradiance_W_cm2']*1000:.3f} mW/cm^2, "
          f"MPE={result['mpe_W_cm2']*1000:.3f} mW/cm^2, exceeds: {result['exceeds_mpe']}")

    print("\n=== 5. Resolving dgs.retinal_scan_imaging's flagged gap (illustratively) ===")
    print("  NOTE: dgs.retinal_scan_imaging's own default wavelength (1550nm) falls OUTSIDE")
    print("  this module's 400-1050nm illustrative model -- 1550nm sits in ANSI Z136.1's")
    print("  eye-safe IR band, where CORNEAL (not retinal) absorption dominates, a genuinely")
    print("  different hazard category this module does not implement. Demonstrated here at")
    print("  850nm (a realistic near-IR wavelength within this model's actual scope) instead")
    print("  of silently forcing an answer for a wavelength outside it.")
    scan_check = check_retinal_scan_exposure_illustrative(power_W=1e-6, beam_diameter_um=20.0,
                                                           wavelength_nm=850.0)
    print(f"\n  1uW, 20um-diameter 850nm beam, 1s exposure:")
    print(f"    irradiance = {scan_check['irradiance_W_cm2']*1000:.3f} mW/cm^2")
    print(f"    exposure energy = {scan_check['exposure_energy_J_cm2']:.2e} J/cm^2")
    print(f"    MPE = {scan_check['mpe_J_cm2']:.2e} J/cm^2")
    print(f"    exceeds MPE: {scan_check['exceeds_mpe']} (margin: {scan_check['margin_factor']:.3f}x)")

    print(f"\n*** {MODULE_SAFETY_DISCLAIMER} ***")
