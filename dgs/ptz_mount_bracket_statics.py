"""Structural statics of the bracket that physically bolts dgs.ptz_camera's
gimbal to a helicopter airframe -- the half of "mount a PTZ camera on a
helicopter" that dgs.ptz_helicopter_stabilization deliberately does NOT
cover. That module asks "can the CONTROLLER hold the line of sight steady
against airframe vibration"; this one asks the question that has to be
answered before that one is even relevant: "does the physical bracket
survive the vibration, and does it avoid making things worse by resonating
with it?"

MODEL: the bracket is a cantilever beam -- fixed at the airframe, free at
the tip where the gimbal's mass sits. Two classical (Euler-Bernoulli)
results, both standard mechanics-of-materials formulas, not derived here:

  BENDING STRESS under the gimbal's own weight (times a load factor for
  vibration/maneuvering -- see g_load below):
      M = F * L                  (root bending moment, tip-loaded cantilever)
      sigma = M * c / I          (flexure formula, c = distance to outer fiber)

  TIP-MASS NATURAL FREQUENCY (a cantilever whose own mass is negligible next
  to the concentrated tip mass -- the right regime for a light bracket
  holding a much heavier gimbal, which is why this is NOT the usual
  "beam's own distributed mass" vibration formula):
      f_n = (1 / 2*pi) * sqrt(3 * E * I / (L^3 * m_tip))

RESONANCE AVOIDANCE: standard aerospace/mechanical design practice is to
keep a structure's natural frequency at least ~20% away from any
significant excitation frequency (here, dgs.ptz_helicopter_stabilization's
BLADE_PASS_HZ) in EITHER direction -- a narrow "keep-out band" around the
excitation, not just "don't match it exactly". This module checks that
margin, not just equality.
"""
import math

from dgs.ptz_helicopter_stabilization import BLADE_PASS_HZ, default_gimbal_params

RESONANCE_MARGIN_FRACTION = 0.20   # keep f_n at least 20% away from BLADE_PASS_HZ


def _check_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def rectangular_tube_second_moment(outer_width, outer_height, wall_thickness):
    """Second moment of area I [m^4] for a hollow rectangular tube (bending
    about the horizontal axis), the common shape for a lightweight aluminum
    mounting bracket: I = (b_o h_o^3 - b_i h_i^3) / 12, inner dims = outer
    dims minus 2*wall_thickness on each side."""
    _check_positive(outer_width, "outer_width")
    _check_positive(outer_height, "outer_height")
    _check_positive(wall_thickness, "wall_thickness")
    inner_w = outer_width - 2 * wall_thickness
    inner_h = outer_height - 2 * wall_thickness
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("wall_thickness too large for the given outer dimensions")
    I = (outer_width * outer_height ** 3 - inner_w * inner_h ** 3) / 12.0
    return I


def root_bending_stress(weight_N, length_m, second_moment_I, outer_height_m, g_load=1.0):
    """Max bending stress [Pa] at the bracket's fixed (airframe) end, for a
    tip load of weight_N * g_load (g_load lets the caller apply a vibration/
    maneuver load factor on top of static weight -- helicopter external
    mounts are commonly specified against 3-6g equivalent, not just 1g)."""
    _check_positive(weight_N, "weight_N")
    _check_positive(length_m, "length_m")
    _check_positive(second_moment_I, "second_moment_I")
    _check_positive(outer_height_m, "outer_height_m")
    if g_load <= 0:
        raise ValueError(f"g_load must be positive, got {g_load}")
    F = weight_N * g_load
    M = F * length_m
    c = outer_height_m / 2.0
    return M * c / second_moment_I


def tip_deflection(weight_N, length_m, second_moment_I, E_modulus, g_load=1.0):
    """Static tip deflection [m] of a cantilever under a tip load:
    delta = F L^3 / (3 E I)."""
    _check_positive(weight_N, "weight_N")
    _check_positive(length_m, "length_m")
    _check_positive(second_moment_I, "second_moment_I")
    _check_positive(E_modulus, "E_modulus")
    if g_load <= 0:
        raise ValueError(f"g_load must be positive, got {g_load}")
    F = weight_N * g_load
    return F * length_m ** 3 / (3 * E_modulus * second_moment_I)


def tip_mass_natural_frequency(E_modulus, second_moment_I, length_m, tip_mass_kg):
    """Natural frequency [Hz] of a massless cantilever with a concentrated
    tip mass: f_n = (1/2pi) sqrt(3 E I / (L^3 m))."""
    _check_positive(E_modulus, "E_modulus")
    _check_positive(second_moment_I, "second_moment_I")
    _check_positive(length_m, "length_m")
    _check_positive(tip_mass_kg, "tip_mass_kg")
    k_effective = 3 * E_modulus * second_moment_I / length_m ** 3
    return (1.0 / (2 * math.pi)) * math.sqrt(k_effective / tip_mass_kg)


def check_resonance_margin(natural_freq_hz, excitation_freq_hz=BLADE_PASS_HZ,
                            margin_fraction=RESONANCE_MARGIN_FRACTION):
    """A structure's natural frequency should sit outside a keep-out band
    around the excitation frequency, not just avoid exact equality. Returns
    the fractional separation and whether it clears the required margin on
    EITHER side (below the band or above it)."""
    _check_positive(natural_freq_hz, "natural_freq_hz")
    _check_positive(excitation_freq_hz, "excitation_freq_hz")
    if not (0 < margin_fraction < 1):
        raise ValueError(f"margin_fraction must be in (0,1), got {margin_fraction}")
    lower_bound = excitation_freq_hz * (1 - margin_fraction)
    upper_bound = excitation_freq_hz * (1 + margin_fraction)
    clear = bool(natural_freq_hz < lower_bound or natural_freq_hz > upper_bound)
    fractional_separation = abs(natural_freq_hz - excitation_freq_hz) / excitation_freq_hz
    return {
        "natural_freq_hz": natural_freq_hz,
        "excitation_freq_hz": excitation_freq_hz,
        "keep_out_band_hz": (lower_bound, upper_bound),
        "fractional_separation": fractional_separation,
        "clears_margin": clear,
    }


def evaluate_bracket(length_m, outer_width_m, outer_height_m, wall_thickness_m,
                      E_modulus=69e9, yield_stress_Pa=270e6, density_kg_m3=2700.0,
                      g_load=4.0, gimbal_params=None, g_earth=9.81):
    """One-stop evaluation of a candidate aluminum (default properties:
    6061-T6-ish E=69 GPa, yield~270 MPa, rho=2700 kg/m^3) rectangular-tube
    bracket carrying dgs.ptz_helicopter_stabilization's default gimbal:
    bending stress + factor of safety, tip deflection, natural frequency,
    and the resonance-margin check against BLADE_PASS_HZ -- everything
    needed to say whether a specific bracket design is acceptable."""
    if gimbal_params is None:
        gimbal_params = default_gimbal_params()
    tip_mass_kg = gimbal_params["mass"]
    weight_N = tip_mass_kg * g_earth

    I = rectangular_tube_second_moment(outer_width_m, outer_height_m, wall_thickness_m)
    stress_Pa = root_bending_stress(weight_N, length_m, I, outer_height_m, g_load=g_load)
    deflection_m = tip_deflection(weight_N, length_m, I, E_modulus, g_load=g_load)
    f_n_hz = tip_mass_natural_frequency(E_modulus, I, length_m, tip_mass_kg)
    resonance = check_resonance_margin(f_n_hz)

    bracket_volume = outer_width_m * outer_height_m * length_m - \
        (outer_width_m - 2 * wall_thickness_m) * (outer_height_m - 2 * wall_thickness_m) * length_m
    bracket_mass_kg = bracket_volume * density_kg_m3

    return {
        "second_moment_I_m4": I,
        "root_bending_stress_Pa": stress_Pa,
        "factor_of_safety": yield_stress_Pa / stress_Pa,
        "tip_deflection_m": deflection_m,
        "natural_freq_hz": f_n_hz,
        "resonance_check": resonance,
        "bracket_mass_kg": bracket_mass_kg,
        "acceptable": bool(stress_Pa < yield_stress_Pa and resonance["clears_margin"]),
    }


if __name__ == "__main__":
    print(f"Blade-pass excitation frequency (from dgs.ptz_helicopter_stabilization): "
          f"{BLADE_PASS_HZ:.1f} Hz\n")

    print("=== Candidate bracket: 25mm x 15mm x 2mm-wall aluminum tube, 0.30 m long ===")
    result = evaluate_bracket(length_m=0.30, outer_width_m=0.025, outer_height_m=0.015,
                               wall_thickness_m=0.002)
    print(f"bracket mass: {result['bracket_mass_kg']*1000:.1f} g")
    print(f"root bending stress: {result['root_bending_stress_Pa']/1e6:.1f} MPa  "
          f"(factor of safety: {result['factor_of_safety']:.1f})")
    print(f"tip deflection: {result['tip_deflection_m']*1000:.2f} mm")
    print(f"natural frequency: {result['natural_freq_hz']:.1f} Hz")
    lo, hi = result['resonance_check']['keep_out_band_hz']
    print(f"blade-pass keep-out band: {lo:.1f}-{hi:.1f} Hz -> "
          f"{'CLEAR' if result['resonance_check']['clears_margin'] else 'INSIDE BAND -- resonance risk'}")
    print(f"\noverall: {'ACCEPTABLE' if result['acceptable'] else 'NOT ACCEPTABLE'}")

    print("\n=== A thinner-wall bracket: stress alone looks fine, resonance check catches it ===")
    bad = evaluate_bracket(length_m=0.30, outer_width_m=0.025, outer_height_m=0.015,
                            wall_thickness_m=0.0001)
    print(f"root bending stress: {bad['root_bending_stress_Pa']/1e6:.1f} MPa  "
          f"(factor of safety: {bad['factor_of_safety']:.1f} -- looks fine by stress alone)")
    print(f"natural frequency: {bad['natural_freq_hz']:.1f} Hz  "
          f"({'CLEAR' if bad['resonance_check']['clears_margin'] else 'INSIDE BLADE-PASS KEEP-OUT BAND'})")
    print(f"overall: {'ACCEPTABLE' if bad['acceptable'] else 'NOT ACCEPTABLE'}  "
          "-- stress passing is not enough; resonance is a separate failure mode")
