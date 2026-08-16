"""Test dgs/thz_waveguide_dispersion_relation.py: the exact algebraic
identity between waveguide dispersion (omega^2=c^2k^2+omega_c^2) and the
relativistic dispersion relation (E^2=(pc)^2+(mc^2)^2) from
dgs/compton_scattering.py, the resulting v_phase*v_group=c^2 identity, and
THz pulse broadening from group velocity dispersion."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.thz_waveguide_dispersion_relation import (
    verify_waveguide_matches_relativistic_dispersion, effective_photon_mass,
    waveguide_wavenumber, phase_velocity, group_velocity,
    verify_phase_group_velocity_product, group_velocity_dispersion,
    thz_pulse_broadening, HBAR, C_LIGHT,
    verify_gvd_sign_is_fixed, rank_modes_by_dispersion, thermal_broadening_shift,
    verify_dispersion_relation_is_geometry_independent,
    rectangular_waveguide_cutoff_frequency,
)

# 1. verify_waveguide_matches_relativistic_dispersion: must pass
assert verify_waveguide_matches_relativistic_dispersion() is True

# 2. effective_photon_mass: known formula, and validation
omega_c = 2 * np.pi * 0.3e12   # 0.3 THz cutoff
m_eff = effective_photon_mass(omega_c)
assert abs(m_eff - HBAR * omega_c / C_LIGHT**2) < 1e-50
try:
    effective_photon_mass(-1.0)
    raise AssertionError("expected ValueError for omega_c <= 0")
except ValueError:
    pass

# 3. waveguide_wavenumber: below cutoff must raise (no real propagating
#    wavenumber -- the evanescent regime)
try:
    waveguide_wavenumber(0.5 * omega_c, omega_c)
    raise AssertionError("expected ValueError for omega below cutoff")
except ValueError:
    pass
# at omega well above cutoff, k -> omega/c (the "relativistic" limit,
# mass term negligible, same as a highly relativistic particle p~E/c)
omega_high = 1000 * omega_c
k_high = waveguide_wavenumber(omega_high, omega_c)
assert abs(k_high - omega_high / C_LIGHT) / (omega_high / C_LIGHT) < 1e-5

# 4. phase_velocity: always > c (for any omega > omega_c)
for factor in (1.001, 1.1, 2.0, 10.0, 1000.0):
    v_p = phase_velocity(factor * omega_c, omega_c)
    assert v_p > C_LIGHT, f"phase velocity {v_p} should exceed c at factor={factor}"

# 5. group_velocity: always < c, and -> c as omega >> omega_c (far from
#    cutoff, the waveguide looks like free space)
for factor in (1.001, 1.1, 2.0, 10.0, 1000.0):
    v_g = group_velocity(factor * omega_c, omega_c)
    assert v_g < C_LIGHT, f"group velocity {v_g} should be below c at factor={factor}"
v_g_far = group_velocity(1e6 * omega_c, omega_c)
assert abs(v_g_far - C_LIGHT) / C_LIGHT < 1e-6

# 6. group_velocity: right AT cutoff must raise (v_g -> 0 there, but the
#    function requires omega strictly > omega_c, matching
#    waveguide_wavenumber's validation)
try:
    group_velocity(omega_c, omega_c)
    raise AssertionError("expected ValueError for omega == omega_c")
except ValueError:
    pass

# 7. verify_phase_group_velocity_product: the central identity, checked
#    across several operating points, not just one
for factor in (1.01, 1.5, 3.0, 50.0):
    assert verify_phase_group_velocity_product(factor * omega_c, omega_c) is True

# 8. group_velocity_dispersion: sign must be consistent (normal dispersion
#    near cutoff -- beta_2 negative here, matching the analytic formula
#    -omega_c^2/(c*(omega^2-omega_c^2)^1.5), and must grow in MAGNITUDE
#    as omega approaches cutoff (dispersion worsens near cutoff, a known
#    waveguide fact)
beta2_near = group_velocity_dispersion(1.01 * omega_c, omega_c)
beta2_far = group_velocity_dispersion(10.0 * omega_c, omega_c)
assert beta2_near < 0 and beta2_far < 0
assert abs(beta2_near) > abs(beta2_far), "GVD magnitude should be larger near cutoff"

# 9. thz_pulse_broadening: must be positive, and scale linearly with L
#    and with bandwidth (both explicit in the formula, checked directly)
L1, L2 = 1.0, 2.0
bw = 2 * np.pi * 10e9
broadening_L1 = thz_pulse_broadening(L1, bw, 1.5 * omega_c, omega_c)
broadening_L2 = thz_pulse_broadening(L2, bw, 1.5 * omega_c, omega_c)
assert broadening_L1 > 0
assert abs(broadening_L2 - 2 * broadening_L1) / broadening_L1 < 1e-9

bw2 = 2 * bw
broadening_bw2 = thz_pulse_broadening(L1, bw2, 1.5 * omega_c, omega_c)
assert abs(broadening_bw2 - 2 * broadening_L1) / broadening_L1 < 1e-9

# 10. thz_pulse_broadening: input validation
try:
    thz_pulse_broadening(-1.0, bw, 1.5 * omega_c, omega_c)
    raise AssertionError("expected ValueError for L_m <= 0")
except ValueError:
    pass

# 11. verify_gvd_sign_is_fixed: Problems 1 & 2 (CSUS deliverable) -- GVD
#     has no zero-dispersion point and can't self-cancel across a
#     two-segment link of the same mechanism
assert verify_gvd_sign_is_fixed() is True

# 12. rank_modes_by_dispersion: Problem 3 -- TE11 (already the dominant
#     mode by cutoff) must ALSO come out as the least-dispersive mode
a_test = 0.3e-3
ranking = rank_modes_by_dispersion(a_test)
assert ranking["ranked_best_to_worst"][0] == "TE11"
# broadening must strictly increase along the ranking (that's what "ranked" means)
broadenings = [ranking["modes"][name]["broadening_ps"] for name in ranking["ranked_best_to_worst"]]
assert broadenings == sorted(broadenings)
try:
    rank_modes_by_dispersion(a_test, omega_headroom=1.0)
    raise AssertionError("expected ValueError for omega_headroom <= 1")
except ValueError:
    pass

# 13. thermal_broadening_shift: Problem 4 -- a positive delta_T must
#     shrink the cutoff-defining radius's effect consistently (larger a ->
#     lower omega_c), and the shift must be small (<5%) for a realistic
#     60 K swing -- the quantitative "is it negligible" answer
thermal = thermal_broadening_shift(a_test, 1.5 * 2 * np.pi * 0.3e12)
assert thermal["omega_c_frac_shift"] < 0   # bigger (hotter) guide -> lower cutoff
assert abs(thermal["broadening_frac_shift"]) < 0.05

# 14. verify_dispersion_relation_is_geometry_independent: Problem 5 --
#     general proof plus the concrete rectangular-guide check, both inside
assert verify_dispersion_relation_is_geometry_independent() is True

# 15. rectangular_waveguide_cutoff_frequency: matches the well-known WR-90
#     TE10 cutoff (6.557 GHz), a real published value, not just internal
#     consistency
f_c_wr90 = rectangular_waveguide_cutoff_frequency(1, 0, 22.86e-3, 10.16e-3)
assert abs(f_c_wr90 - 6.5571e9) / 6.5571e9 < 1e-3
try:
    rectangular_waveguide_cutoff_frequency(0, 0, 22.86e-3, 10.16e-3)
    raise AssertionError("expected ValueError for m=n=0")
except ValueError:
    pass

print("all dgs.thz_waveguide_dispersion_relation tests passed")
