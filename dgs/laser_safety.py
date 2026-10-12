"""ANSI/IEC laser safety fundamentals: classification, MPE, NOHD, optical density.

SCOPE, STATED HONESTLY: the actual ANSI Z136.1 / IEC 60825-1 Maximum
Permissible Exposure (MPE) tables are piecewise functions of wavelength AND
exposure duration, published inside those (purchased, copyrighted) standards
documents. This module does NOT reproduce those tables and is not a
substitute for a real Laser Safety Officer's sign-off. MPE is always an
INPUT here -- either a widely-cited rule-of-thumb figure (documented per
call site below) or a value the caller supplies after looking up the
current standard for their actual wavelength/exposure regime.

What IS exact, and what this module actually verifies: the standard
ENGINEERING RELATIONSHIPS that turn a given MPE into a go/no-go answer for
a real beam --

  irradiance:  E = P / A,  A = pi/4 * d^2                      (beam area)
  beam growth: d(z) = d0 + z * theta                            (far-field, small-angle divergence)
  NOHD:        the distance z at which E(z) drops to MPE        (Nominal Ocular Hazard Distance)
  eyewear OD:  OD = log10(E / MPE), attenuation factor = 10^OD  (required optical density)

Laser hazard classes (IEC 60825-1 / ANSI Z136.1, general safety knowledge,
not the copyrighted numeric tables):
  Class 1  - safe under all reasonably foreseeable conditions, incl. optics
  Class 1M - safe unaccessed by eye alone; hazardous through magnifying optics
  Class 2  - visible only; the 0.25s blink/aversion reflex is assumed protective
  Class 2M - like 2, but hazardous through magnifying optics
  Class 3R - low direct-viewing hazard; up to 5x the Class 1/2 limit
  Class 3B - hazardous for direct/specular viewing; diffuse reflection is safe
  Class 4  - hazardous even diffusely reflected; fire and skin hazard

Ties to this repo's photonic hardware (steam_imaging, RogueGuard, the HeNe-
wavelength conventions used throughout dgs.laser_physics): every optical
bench setup in this repo emits real, non-zero power at a real wavelength,
and "what class is this and how far away is it still dangerous" is the
actual first question a lab safety review asks, before any physics.
"""
import math

C_LIGHT = 2.99792458e8   # m/s


# ----------------------------------------------------------------------
# 1. Irradiance and beam geometry
# ----------------------------------------------------------------------

def beam_area(diameter_m):
    """A = pi/4 * d^2, the circular beam cross-section."""
    if diameter_m <= 0:
        raise ValueError("diameter must be positive")
    return math.pi / 4.0 * diameter_m ** 2


def beam_irradiance(power_W, diameter_m):
    """E = P / A [W/m^2]. The quantity MPE limits are actually stated in."""
    if power_W < 0:
        raise ValueError("power must be non-negative")
    return power_W / beam_area(diameter_m)


def beam_diameter_at_distance(initial_diameter_m, divergence_rad, distance_m):
    """d(z) = d0 + z*theta, the small-angle far-field beam growth from
    divergence half-angle... here theta is the FULL divergence angle, matching
    how laser datasheets quote it (total angular spread, not half-angle)."""
    if initial_diameter_m < 0 or divergence_rad < 0 or distance_m < 0:
        raise ValueError("diameter, divergence, and distance must be non-negative")
    return initial_diameter_m + distance_m * divergence_rad


# ----------------------------------------------------------------------
# 2. Nominal Ocular Hazard Distance
# ----------------------------------------------------------------------

def nominal_ocular_hazard_distance(power_W, divergence_rad, mpe_W_per_m2, initial_diameter_m=0.0):
    """NOHD: solve E(z) = MPE for z, using E(z) = P / (pi/4 * (d0 + z*theta)^2).

    Rearranged:  d0 + z*theta = sqrt(4P / (pi*MPE))
                 z = (sqrt(4P/(pi*MPE)) - d0) / theta

    Beyond this distance, direct intrabeam viewing is below the supplied
    MPE. theta=0 (perfectly collimated) is degenerate -- irradiance never
    drops with distance -- and is rejected below since no finite NOHD
    exists in that idealization.
    """
    if power_W <= 0 or mpe_W_per_m2 <= 0 or initial_diameter_m < 0:
        raise ValueError("power, MPE must be positive; diameter non-negative")
    if divergence_rad <= 0:
        raise ValueError("divergence must be positive (a perfectly collimated beam has no finite NOHD)")

    d_at_mpe = math.sqrt(4.0 * power_W / (math.pi * mpe_W_per_m2))
    z = (d_at_mpe - initial_diameter_m) / divergence_rad
    return max(z, 0.0)


# ----------------------------------------------------------------------
# 3. Eyewear optical density
# ----------------------------------------------------------------------

def optical_density_required(irradiance_W_per_m2, mpe_W_per_m2):
    """OD = log10(E/MPE): the eyewear attenuation (as a power of 10) needed
    to bring a beam's irradiance down to the MPE. OD<=0 means no attenuation
    is needed (already at or below MPE)."""
    if irradiance_W_per_m2 <= 0 or mpe_W_per_m2 <= 0:
        raise ValueError("irradiance and MPE must be positive")
    return max(math.log10(irradiance_W_per_m2 / mpe_W_per_m2), 0.0)


def attenuation_factor(optical_density):
    """The transmission-reduction factor an OD actually buys: T = 10^-OD,
    so attenuation = 1/T = 10^OD."""
    if optical_density < 0:
        raise ValueError("optical density must be non-negative")
    return 10.0 ** optical_density


# ----------------------------------------------------------------------
# 4. Classification (descriptive; numeric class BOUNDARIES are wavelength/
#    duration-dependent and belong in the actual standard, not hardcoded here)
# ----------------------------------------------------------------------

CLASS_DESCRIPTIONS = {
    '1':  'Safe under all reasonably foreseeable conditions, including with optical instruments.',
    '1M': 'Safe for the naked eye; hazardous when viewed through magnifying/collecting optics.',
    '2':  'Visible (400-700nm) only; the ~0.25s blink/aversion reflex is assumed protective for CW.',
    '2M': 'Like Class 2, but hazardous through magnifying optics.',
    '3R': 'Low direct-viewing hazard; accessible emission up to 5x the Class 1/2 limit.',
    '3B': 'Hazardous for direct or specular-reflection viewing; diffuse reflection is normally safe.',
    '4':  'Hazardous even from a diffuse reflection; fire and skin hazard. Requires interlocked enclosure.',
}


def is_extended_source_safe_via_blink(exposure_duration_s):
    """Class 2's entire safety argument: is the exposure short enough that
    the human blink/aversion reflex (~0.25s) is expected to end it before a
    hazardous dose accumulates? This is the ONE universally-cited numeric
    constant in visible-laser classification that isn't wavelength-dependent
    -- the 0.25s aversion response time."""
    if exposure_duration_s <= 0:
        raise ValueError("exposure duration must be positive")
    AVERSION_RESPONSE_TIME_S = 0.25
    return exposure_duration_s <= AVERSION_RESPONSE_TIME_S


if __name__ == "__main__":
    print("=== LASER SAFETY: NOHD AND EYEWEAR OD FOR A REAL BENCH SETUP ===\n")

    # A 5 mW HeNe (632.8nm, the wavelength convention used throughout this
    # repo's laser_physics.py demo), 1 mrad divergence, 1mm waist.
    # MPE: a widely-cited rule-of-thumb figure for CW visible intrabeam
    # viewing, extended (>10s) exposure -- ~2.5 mW/cm^2 = 25 W/m^2
    # (see e.g. Rockwell Laser Industries / LIA laser-safety training
    # references). NOT a substitute for the actual ANSI Z136.1 table lookup.
    power_W = 5e-3
    divergence_rad = 1e-3
    d0 = 1e-3
    mpe_extended = 25.0   # W/m^2, rule-of-thumb, see docstring above

    nohd = nominal_ocular_hazard_distance(power_W, divergence_rad, mpe_extended, d0)
    print(f"5 mW HeNe, 1 mrad divergence, MPE={mpe_extended} W/m^2 (rule-of-thumb, extended CW exposure):")
    print(f"  NOHD = {nohd:.2f} m -- beyond this, direct intrabeam viewing is below the MPE used here.")

    # eyewear OD needed to view this beam safely at the source (z=0, worst case)
    E_at_source = beam_irradiance(power_W, d0)
    od = optical_density_required(E_at_source, mpe_extended)
    print(f"\n  irradiance at the source: {E_at_source:.1f} W/m^2")
    print(f"  eyewear OD required at the source: {od:.2f}  (attenuation factor 10^{od:.2f} = {attenuation_factor(od):.1f}x)")

    print(f"\n  Class 2 (visible, blink-protected) applies only if exposure <= 0.25s: "
          f"{is_extended_source_safe_via_blink(0.1)} for a 0.1s glance, "
          f"{is_extended_source_safe_via_blink(1.0)} for a 1s stare.")

    print("\n=== DONE ===")
