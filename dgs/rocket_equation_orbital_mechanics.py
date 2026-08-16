"""Rocket propulsion and orbital mechanics, "Griffiths solution style":
each result DERIVED from first principles via SymPy (not looked up and
plugged in), then checked numerically against publicly documented
engineering figures -- the same "prove it, then verify it" posture as
this session's Griffiths modules, applied to the physics behind real,
publicly cited launch-vehicle engineering (Falcon 9's Merlin engine
specific impulse, LEO/GEO orbital velocities) rather than claims about
any person or company's internal decisions.

1. THE TSIOLKOVSKY ROCKET EQUATION, derived from momentum conservation
   for a variable-mass system (a rocket expelling propellant at exhaust
   velocity v_e), not stated and trusted:

       m dv = -v_e dm   =>   dv/dm = -v_e/m   =>   Delta_v = v_e*ln(m0/mf)

   -- integrated symbolically below, exactly, via SymPy.

2. MULTI-STAGE OPTIMIZATION: for N stages with EQUAL exhaust velocity and
   EQUAL structural mass fraction, minimizing total initial mass for a
   fixed total Delta-v budget is a genuine calculus optimization problem
   (Lagrange-multiplier-flavored), solved here symbolically for 2 stages
   -- the answer, an EQUAL Delta-v split across stages, comes out of
   SymPy's own critical-point solve, not asserted from "textbooks say so."

3. THE HOHMANN TRANSFER: the standard two-impulse orbital maneuver
   between circular orbits, derived from the vis-viva equation (energy
   conservation on an elliptical transfer orbit), applied to a realistic
   LEO-to-GEO transfer and checked against widely-cited orbital-mechanics
   reference values (~7.8 km/s LEO circular velocity, ~3.9 km/s total
   Hohmann Delta-v for LEO-GEO -- standard orbital-mechanics textbook
   figures, not specific to any one program).

Public, widely-cited figures used for numeric sanity checks (Merlin 1D
vacuum Isp ~311s) are ENGINEERING SPECIFICATIONS commonly reported in
open literature (e.g. SpaceX's own published payload user guides), used
here purely as realistic numbers to sanity-check the PHYSICS, not as
claims about any individual's role in developing them.
"""

from __future__ import annotations
import numpy as np
import sympy as sp

G0 = 9.80665           # m/s^2, standard gravity (defines specific impulse's units)
MU_EARTH = 3.986004418e14   # m^3/s^2, Earth's standard gravitational parameter
R_EARTH_M = 6378137.0       # m


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ── 1. Tsiolkovsky rocket equation, derived from first principles ──────────

def derive_rocket_equation_symbolic():
    """Integrates dv/dm = -v_e/m from m0 to mf via SymPy -- the actual
    derivation (momentum conservation for a variable-mass system), not a
    memorized formula. Returns the derived expression and confirms it
    matches the textbook v_e*ln(m0/mf) form exactly."""
    m, m0, mf, ve = sp.symbols('m m0 mf v_e', positive=True)
    dv_dm = -ve / m
    delta_v = sp.integrate(dv_dm, (m, m0, mf))
    delta_v = sp.simplify(delta_v)
    textbook_form = ve * sp.log(m0 / mf)
    matches = sp.simplify(delta_v - textbook_form) == 0
    if not matches:
        raise AssertionError(f"derived delta_v={delta_v} does not match ve*ln(m0/mf)")
    return {"derived_delta_v": delta_v, "matches_textbook_form": matches}


def delta_v_tsiolkovsky(exhaust_velocity_m_s: float, m0_kg: float, mf_kg: float) -> float:
    """Delta_v = v_e * ln(m0/mf) -- the closed form derive_rocket_equation_symbolic proved."""
    _validate_positive(exhaust_velocity_m_s=exhaust_velocity_m_s, m0_kg=m0_kg, mf_kg=mf_kg)
    if mf_kg >= m0_kg:
        raise ValueError(f"mf_kg={mf_kg} must be < m0_kg={m0_kg} (can't burn negative propellant)")
    return exhaust_velocity_m_s * np.log(m0_kg / mf_kg)


def exhaust_velocity_from_isp(specific_impulse_s: float, g0: float = G0) -> float:
    """v_e = Isp * g0 -- specific impulse (seconds) to exhaust velocity (m/s)."""
    _validate_positive(specific_impulse_s=specific_impulse_s, g0=g0)
    return specific_impulse_s * g0


# ── 2. Multi-stage optimization: symbolic proof of the equal-split result ──

def verify_two_stage_optimal_split_symbolic():
    """CHECKED, not assumed: for two stages with EQUAL exhaust velocity
    and EQUAL structural mass fraction, minimizing total initial mass
    (for fixed payload, fixed total Delta-v) has its critical point at
    dv1 = dv2 = DV/2 -- solved via SymPy's differentiate-and-solve, AND
    confirmed to be a genuine MINIMUM (not a maximum or saddle) via the
    second derivative, since a critical point alone doesn't prove which
    kind it is.

    lambda = R*(1-eps)/(1-eps*R) is the growth factor (m0/m_payload) for
    ONE stage -- re-derived here independently via SymPy's own equation
    solver (not the earlier hand-algebra that had it upside down; see
    stage_growth_factor's docstring for that mistake)."""
    dv1, DV, ve, eps = sp.symbols('dv1 DV v_e epsilon', positive=True)
    R1 = sp.exp(dv1 / ve)
    R2 = sp.exp((DV - dv1) / ve)
    growth1 = R1 * (1 - eps) / (1 - eps * R1)
    growth2 = R2 * (1 - eps) / (1 - eps * R2)
    total_growth = growth1 * growth2

    d_total = sp.diff(total_growth, dv1)
    critical_points = sp.solve(sp.Eq(d_total, 0), dv1)
    equal_split_is_critical = sp.simplify(d_total.subs(dv1, DV / 2)) == 0

    d2_total = sp.diff(total_growth, dv1, 2)
    d2_at_equal_split = sp.simplify(d2_total.subs(dv1, DV / 2))
    # positive second derivative (for any valid ve, eps, DV) confirms a MINIMUM;
    # checked at representative numeric values since the symbolic sign isn't
    # obviously positive by inspection
    is_minimum = float(d2_at_equal_split.subs({ve: 3000.0, eps: 0.08, DV: 9000.0})) > 0

    return {"critical_points": critical_points, "equal_split_is_critical_point": equal_split_is_critical,
            "second_derivative_at_equal_split": d2_at_equal_split, "confirmed_minimum": is_minimum}


def max_single_stage_delta_v(exhaust_velocity: float, structural_fraction: float) -> float:
    """v_e*ln(1/eps): the HARD physical ceiling on a single stage's
    Delta-v for a given structural mass fraction -- beyond this, R would
    exceed 1/eps and the growth factor's denominator (1-eps*R) would go
    to zero and then negative, i.e. the stage's own structural mass alone
    would exceed what's achievable even with zero payload. Not a
    numerical curiosity: this is a real rocket-design constraint (why
    multi-staging exists at all -- a single stage physically cannot
    deliver unlimited Delta-v no matter how little payload it carries)."""
    _validate_positive(exhaust_velocity=exhaust_velocity)
    if not (0 < structural_fraction < 1):
        raise ValueError(f"structural_fraction must be in (0,1), got {structural_fraction}")
    return exhaust_velocity * np.log(1.0 / structural_fraction)


def stage_growth_factor(delta_v: float, exhaust_velocity: float, structural_fraction: float) -> float:
    """lambda = R*(1-eps) / (1-eps*R), R=exp(dv/ve) -- one stage's
    "growth factor" (initial mass / payload-carried-by-this-stage mass),
    re-derived from m0=ms+mpr+mp, mf=ms+mp, eps=ms/(ms+mpr), and R=m0/mf
    by direct algebraic elimination (cross-checked independently via
    SymPy's own equation solver -- see verify_two_stage_optimal_split_symbolic).

    THIS MODULE'S FIRST VERSION HAD THIS FORMULA INVERTED
    ((1-eps*R)/(R*(1-eps)), the reciprocal of the correct one) -- caught
    because the "equal split is optimal" numerical verification failed:
    an unequal split appeared to give a SMALLER value near the physical
    boundary, which is exactly what an inverted (payload-fraction-like)
    quantity would do while a true growth factor diverges there instead.
    Re-deriving from scratch (rather than patching the sign) is what
    caught and fixed the actual error.

    Raises if delta_v exceeds max_single_stage_delta_v (eps*R >= 1) --
    that regime has no physical solution."""
    _validate_positive(exhaust_velocity=exhaust_velocity, delta_v=delta_v)
    if not (0 < structural_fraction < 1):
        raise ValueError(f"structural_fraction must be in (0,1), got {structural_fraction}")
    max_dv = max_single_stage_delta_v(exhaust_velocity, structural_fraction)
    if delta_v >= max_dv:
        raise ValueError(f"delta_v={delta_v:.1f} m/s exceeds this stage's physical maximum "
                         f"({max_dv:.1f} m/s for structural_fraction={structural_fraction}) -- "
                         f"no single stage can deliver this much Delta-v regardless of payload")
    R = np.exp(delta_v / exhaust_velocity)
    return R * (1 - structural_fraction) / (1 - structural_fraction * R)


def verify_equal_split_beats_unequal(total_delta_v: float, exhaust_velocity: float,
                                     structural_fraction: float, n_offsets: int = 20) -> dict:
    """CHECKED numerically: the equal Delta-v split's total growth factor
    (product of both stages' growth factors) is <= every unequal split
    tested -- a genuine numerical confirmation of the symbolic critical-
    point result, not just trusting that a critical point is a minimum.

    The offset sweep is bounded to stay within EACH stage's own physical
    Delta-v ceiling (max_single_stage_delta_v) -- an earlier version of
    this function swept past that ceiling, where stage_growth_factor's
    formula has no physical solution, and got a spurious NEGATIVE growth
    factor that looked (wrongly) better than the equal split. That bug is
    the reason this bound is explicit and asserted, not just "seemed to
    work" on one example."""
    _validate_positive(total_delta_v=total_delta_v)
    equal_growth = (stage_growth_factor(total_delta_v / 2, exhaust_velocity, structural_fraction) *
                    stage_growth_factor(total_delta_v / 2, exhaust_velocity, structural_fraction))

    max_dv = max_single_stage_delta_v(exhaust_velocity, structural_fraction)
    max_offset = min(total_delta_v * 0.4, max_dv - total_delta_v / 2 - 1.0)
    if max_offset <= 0:
        raise ValueError(f"total_delta_v={total_delta_v} leaves no room for an unequal-split "
                         f"comparison under this stage's physical ceiling ({max_dv:.1f} m/s)")

    offsets = np.linspace(-max_offset, max_offset, n_offsets)
    offsets = offsets[np.abs(offsets) > 1e-6]
    unequal_growths = []
    for off in offsets:
        dv1 = total_delta_v / 2 + off
        dv2 = total_delta_v / 2 - off
        g = (stage_growth_factor(dv1, exhaust_velocity, structural_fraction) *
             stage_growth_factor(dv2, exhaust_velocity, structural_fraction))
        assert g > 0, f"unexpected non-physical growth factor {g} at dv1={dv1}, dv2={dv2}"
        unequal_growths.append(g)
    return {"equal_split_growth_factor": equal_growth,
            "min_unequal_growth_factor": min(unequal_growths),
            "max_offset_tested_m_s": max_offset,
            "equal_split_is_best": bool(equal_growth <= min(unequal_growths))}


# ── 3. Hohmann transfer, from the vis-viva equation ─────────────────────────

def circular_orbit_velocity(radius_m: float, mu: float = MU_EARTH) -> float:
    """v = sqrt(mu/r) -- circular orbit velocity, from setting
    gravitational force equal to centripetal force."""
    _validate_positive(radius_m=radius_m, mu=mu)
    return np.sqrt(mu / radius_m)


def vis_viva_velocity(radius_m: float, semi_major_axis_m: float, mu: float = MU_EARTH) -> float:
    """v = sqrt(mu*(2/r - 1/a)) -- the vis-viva equation (energy
    conservation on any conic-section orbit)."""
    _validate_positive(radius_m=radius_m, semi_major_axis_m=semi_major_axis_m, mu=mu)
    return np.sqrt(mu * (2 / radius_m - 1 / semi_major_axis_m))


def hohmann_transfer_delta_v(r1_m: float, r2_m: float, mu: float = MU_EARTH) -> dict:
    """The two burns of a Hohmann transfer between circular orbits of
    radius r1 (start) and r2 (target): a first burn onto an elliptical
    transfer orbit (semi-major axis (r1+r2)/2), a second burn to
    circularize at r2."""
    _validate_positive(r1_m=r1_m, r2_m=r2_m, mu=mu)
    a_transfer = (r1_m + r2_m) / 2
    v1_circular = circular_orbit_velocity(r1_m, mu)
    v2_circular = circular_orbit_velocity(r2_m, mu)
    v_transfer_at_r1 = vis_viva_velocity(r1_m, a_transfer, mu)
    v_transfer_at_r2 = vis_viva_velocity(r2_m, a_transfer, mu)
    dv1 = abs(v_transfer_at_r1 - v1_circular)
    dv2 = abs(v2_circular - v_transfer_at_r2)
    return {"v1_circular_m_s": v1_circular, "v2_circular_m_s": v2_circular,
            "dv1_m_s": dv1, "dv2_m_s": dv2, "total_delta_v_m_s": dv1 + dv2}


if __name__ == "__main__":
    print("=== 1. Tsiolkovsky rocket equation, derived from first principles ===")
    derivation = derive_rocket_equation_symbolic()
    print(f"  derived: Delta_v = {derivation['derived_delta_v']}")
    print(f"  matches v_e*ln(m0/mf): {derivation['matches_textbook_form']}")

    print("\n  Numeric check with a publicly-cited Merlin 1D vacuum Isp (~311s):")
    ve = exhaust_velocity_from_isp(311.0)
    print(f"  exhaust velocity = {ve:.1f} m/s")
    dv = delta_v_tsiolkovsky(ve, m0_kg=111500.0, mf_kg=4000.0)
    print(f"  illustrative single-stage Delta_v (m0=111500kg, mf=4000kg) = {dv/1000:.2f} km/s")

    print("\n=== 2. Multi-stage optimization: equal Delta-v split is optimal ===")
    sym_check = verify_two_stage_optimal_split_symbolic()
    print(f"  critical point(s) of total growth factor vs. dv1: {sym_check['critical_points']}")
    print(f"  DV/2 (equal split) is a critical point: {sym_check['equal_split_is_critical_point']}")
    print(f"  second derivative there is positive (confirms MINIMUM, not max/saddle): "
          f"{sym_check['confirmed_minimum']}")

    num_check = verify_equal_split_beats_unequal(total_delta_v=9400.0, exhaust_velocity=ve,
                                                 structural_fraction=0.08)
    print(f"  equal-split growth factor: {num_check['equal_split_growth_factor']:.4f}")
    print(f"  smallest unequal-split growth factor found: {num_check['min_unequal_growth_factor']:.4f}")
    print(f"  (searched dv1 offsets up to +/-{num_check['max_offset_tested_m_s']:.0f} m/s, "
          f"bounded by each stage's physical Delta-v ceiling)")
    print(f"  equal split is best: {num_check['equal_split_is_best']}")

    print("\n=== 3. Hohmann transfer: LEO to GEO ===")
    r_leo = R_EARTH_M + 300e3
    r_geo = R_EARTH_M + 35786e3
    hohmann = hohmann_transfer_delta_v(r_leo, r_geo)
    print(f"  LEO circular velocity: {hohmann['v1_circular_m_s']/1000:.3f} km/s "
          f"(widely-cited reference value: ~7.8 km/s)")
    print(f"  GEO circular velocity: {hohmann['v2_circular_m_s']/1000:.3f} km/s")
    print(f"  dv1 (transfer injection): {hohmann['dv1_m_s']/1000:.3f} km/s")
    print(f"  dv2 (GEO circularization): {hohmann['dv2_m_s']/1000:.3f} km/s")
    print(f"  total Hohmann Delta_v: {hohmann['total_delta_v_m_s']/1000:.3f} km/s "
          f"(widely-cited reference value: ~3.9 km/s)")

    print("\nEach number above is derived, not looked up: the rocket equation from momentum")
    print("conservation, the optimal stage split from a critical-point solve, the transfer")
    print("delta-v from vis-viva -- then checked against real, publicly documented figures.")
