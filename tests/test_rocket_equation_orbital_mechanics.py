"""Test dgs/rocket_equation_orbital_mechanics.py: the Tsiolkovsky rocket
equation derived from first principles, the two-stage optimal-split
proof (including a regression test for a real inverted-formula bug this
module's own development caught), the physical Delta-v ceiling, and the
Hohmann transfer checked against widely-cited reference values."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
import sympy as sp
from dgs.rocket_equation_orbital_mechanics import (
    derive_rocket_equation_symbolic, delta_v_tsiolkovsky, exhaust_velocity_from_isp,
    verify_two_stage_optimal_split_symbolic, max_single_stage_delta_v, stage_growth_factor,
    verify_equal_split_beats_unequal, circular_orbit_velocity, vis_viva_velocity,
    hohmann_transfer_delta_v, R_EARTH_M, MU_EARTH,
)

# 1. derive_rocket_equation_symbolic: the actual symbolic derivation
#    matches the textbook closed form exactly
derivation = derive_rocket_equation_symbolic()
assert derivation["matches_textbook_form"] is True

# 2. delta_v_tsiolkovsky / exhaust_velocity_from_isp: known values
ve = exhaust_velocity_from_isp(311.0)
assert abs(ve - 311.0 * 9.80665) < 1e-6

dv = delta_v_tsiolkovsky(ve, m0_kg=100.0, mf_kg=100.0 / math.e)
assert abs(dv - ve) < 1e-6   # mass ratio e -> delta_v = ve*ln(e) = ve exactly

for bad in [dict(exhaust_velocity_m_s=-1, m0_kg=100, mf_kg=50),
            dict(exhaust_velocity_m_s=1000, m0_kg=50, mf_kg=100)]:   # mf >= m0
    try:
        delta_v_tsiolkovsky(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.rocket_equation_orbital_mechanics: rocket equation checks passed")

# 3. verify_two_stage_optimal_split_symbolic: equal split is a critical
#    point AND confirmed to be a genuine minimum (not just a critical point)
sym_check = verify_two_stage_optimal_split_symbolic()
assert sym_check["equal_split_is_critical_point"] is True
assert sym_check["confirmed_minimum"] is True
DV = sp.Symbol('DV', positive=True)
assert sym_check["critical_points"] == [DV / 2]

print("dgs.rocket_equation_orbital_mechanics: symbolic optimal-split checks passed")

# 4. max_single_stage_delta_v / stage_growth_factor: the physical ceiling
#    is enforced -- a delta_v at or past it must raise, not silently
#    return a nonsensical (e.g. negative) growth factor
ve_test, eps_test = 3000.0, 0.08
ceiling = max_single_stage_delta_v(ve_test, eps_test)
assert abs(ceiling - ve_test * math.log(1 / eps_test)) < 1e-6

just_under = stage_growth_factor(ceiling - 1.0, ve_test, eps_test)
assert just_under > 0   # a real, finite, positive growth factor

try:
    stage_growth_factor(ceiling, ve_test, eps_test)
    raise AssertionError("expected ValueError at the exact ceiling")
except ValueError:
    pass
try:
    stage_growth_factor(ceiling + 100.0, ve_test, eps_test)
    raise AssertionError("expected ValueError past the ceiling")
except ValueError:
    pass

# growth factor must be monotonically INCREASING with delta_v (more
# Delta-v demanded from one stage costs more initial mass per unit
# payload) -- the property whose violation (via the inverted formula)
# is exactly what this module's development caught as a bug
low_dv_growth = stage_growth_factor(1000.0, ve_test, eps_test)
high_dv_growth = stage_growth_factor(5000.0, ve_test, eps_test)
assert high_dv_growth > low_dv_growth, (
    f"growth factor should increase with delta_v: {low_dv_growth} vs {high_dv_growth}")
# and it should diverge as delta_v approaches the ceiling, not shrink toward 0
near_ceiling_growth = stage_growth_factor(ceiling - 0.01, ve_test, eps_test)
assert near_ceiling_growth > high_dv_growth * 10, (
    "growth factor should diverge approaching the physical ceiling, not stay small "
    "(this is the exact regression check for the earlier inverted-formula bug)")

print("dgs.rocket_equation_orbital_mechanics: growth-factor direction/ceiling checks passed "
      "(regression test for the caught inverted-formula bug)")

# 5. verify_equal_split_beats_unequal: the equal split must be the
#    minimum among every tested split, and the search stays within the
#    physical ceiling (no ValueError from an out-of-range offset)
num_check = verify_equal_split_beats_unequal(total_delta_v=9400.0, exhaust_velocity=ve,
                                             structural_fraction=0.08)
assert num_check["equal_split_is_best"] is True
assert num_check["equal_split_growth_factor"] <= num_check["min_unequal_growth_factor"]
assert num_check["max_offset_tested_m_s"] > 0

print("dgs.rocket_equation_orbital_mechanics: equal-split-is-optimal numeric checks passed")

# 6. circular_orbit_velocity / vis_viva_velocity: known relationship --
#    vis-viva AT the circular radius, with semi-major axis EQUAL to that
#    radius, must reduce exactly to the circular velocity formula
r_test = R_EARTH_M + 400e3
v_circ = circular_orbit_velocity(r_test)
v_visviva_circular_case = vis_viva_velocity(r_test, r_test)
assert abs(v_circ - v_visviva_circular_case) / v_circ < 1e-9

print("dgs.rocket_equation_orbital_mechanics: vis-viva/circular-velocity checks passed")

# 7. hohmann_transfer_delta_v: checked against widely-cited reference
#    values for LEO circular velocity (~7.8 km/s) and total LEO-GEO
#    Hohmann delta-v (~3.9 km/s)
r_leo = R_EARTH_M + 300e3
r_geo = R_EARTH_M + 35786e3
hohmann = hohmann_transfer_delta_v(r_leo, r_geo)
assert abs(hohmann["v1_circular_m_s"] / 1000 - 7.73) < 0.1
assert abs(hohmann["total_delta_v_m_s"] / 1000 - 3.89) < 0.1

# a transfer to a HIGHER orbit should need dv1 to SPEED UP and dv2 to
# SLOW DOWN relative to the respective circular velocities -- checked via
# the vis-viva values directly, not just the final delta-v magnitudes
a_transfer = (r_leo + r_geo) / 2
v_transfer_at_leo = vis_viva_velocity(r_leo, a_transfer)
v_transfer_at_geo = vis_viva_velocity(r_geo, a_transfer)
assert v_transfer_at_leo > hohmann["v1_circular_m_s"]   # speeds up to leave LEO
assert hohmann["v2_circular_m_s"] > v_transfer_at_geo   # speeds up again to circularize at GEO

for bad in [dict(r1_m=-1.0, r2_m=r_geo), dict(r1_m=r_leo, r2_m=-1.0)]:
    try:
        hohmann_transfer_delta_v(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.rocket_equation_orbital_mechanics: Hohmann transfer checks passed")
print("all dgs.rocket_equation_orbital_mechanics tests passed")
