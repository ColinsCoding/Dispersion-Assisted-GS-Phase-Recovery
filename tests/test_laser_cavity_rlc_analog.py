"""Test dgs/laser_cavity_rlc_analog.py: Fabry-Perot cavity Q/finesse/
linewidth, the RLC electrical analog (verified by real ODE simulation of
the free series-RLC circuit, at a computationally tractable demo
frequency since real optical Q spans too many oscillation periods to
literally simulate), and the laser-threshold <-> net-resistance-zero
analogy."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
from dgs.laser_cavity_rlc_analog import (
    cavity_round_trip_time, cavity_free_spectral_range,
    cavity_round_trip_power_survival, cavity_photon_lifetime, cavity_Q_factor,
    linewidth_from_Q, cavity_finesse, linewidth_from_finesse,
    verify_linewidth_formulas_agree, rlc_equivalent_from_Q,
    verify_rlc_matches_cavity_decay, laser_threshold_gain,
    verify_threshold_condition, electrical_threshold_analog, C_LIGHT,
)

L, R1, R2, alpha, n = 0.30, 1.0, 0.98, 0.0, 1.0
f0 = C_LIGHT / 633e-9

# 1. cavity_round_trip_time / FSR: known relationship, FSR = 1/T_rt exactly
T_rt = cavity_round_trip_time(L, n)
FSR = cavity_free_spectral_range(L, n)
assert abs(FSR - 1.0 / T_rt) < 1e-15
assert abs(T_rt - 2 * L / C_LIGHT) < 1e-15   # n=1: T_rt = 2L/c

# 2. cavity_round_trip_power_survival: R1=1 (lossless HR mirror), alpha=0
#    -> R_rt = R2 exactly
R_rt = cavity_round_trip_power_survival(1.0, 0.98, 0.0, L)
assert abs(R_rt - 0.98) < 1e-15

# 3. input validation across the reflectivity/loss functions
for bad_R1, bad_R2, bad_alpha, bad_L in [(0.0, 0.98, 0.0, L), (1.5, 0.98, 0.0, L),
                                          (1.0, 0.98, -1.0, L), (1.0, 0.98, 0.0, -1.0)]:
    try:
        cavity_round_trip_power_survival(bad_R1, bad_R2, bad_alpha, bad_L)
        raise AssertionError(f"expected ValueError for {(bad_R1, bad_R2, bad_alpha, bad_L)}")
    except ValueError:
        pass

# 4. cavity_photon_lifetime / cavity_Q_factor: known HeNe-scale numbers,
#    sanity-checked against the textbook order of magnitude (tau_c ~ 1e-7 s,
#    Q ~ 1e8 for a visible-wavelength cavity with ~2% output coupling)
tau_c = cavity_photon_lifetime(L, R1, R2, n, alpha)
assert 1e-8 < tau_c < 1e-6, f"tau_c={tau_c} outside expected HeNe-scale order of magnitude"
Q = cavity_Q_factor(f0, tau_c)
assert 1e7 < Q < 1e10, f"Q={Q} outside expected order of magnitude"

# 5. linewidth_from_Q vs linewidth_from_finesse: must agree closely at
#    realistic (high) finesse, and DISAGREE beyond 1% at deliberately low
#    finesse -- both outcomes checked, not just the "they agree" case
high_finesse_check = verify_linewidth_formulas_agree(L, R1, R2, f0, n, alpha)
assert high_finesse_check["agree_within_rtol"] is True
assert high_finesse_check["relative_difference"] < 1e-4

low_finesse_check = verify_linewidth_formulas_agree(L, 0.5, 0.5, f0, n, alpha)
assert low_finesse_check["agree_within_rtol"] is False
assert low_finesse_check["relative_difference"] > 0.01

# 6. cavity_finesse: reduces to the textbook pi*sqrt(R)/(1-R) for equal
#    mirrors, no loss
R = 0.9
F = cavity_finesse(R, R, alpha=0.0, L=L)
expected_F = math.pi * math.sqrt(R) / (1 - R)
assert abs(F - expected_F) / expected_F < 1e-9

# 7. rlc_equivalent_from_Q: the derived L, C reproduce the SAME omega0 and
#    Q by construction -- verified directly, not just trusted algebra
rlc = rlc_equivalent_from_Q(f0, Q, R=50.0)
omega0_check = 1.0 / math.sqrt(rlc["L_H"] * rlc["C_F"])
Q_check = omega0_check * rlc["L_H"] / rlc["R_ohm"]
assert abs(omega0_check - rlc["omega0_rad_s"]) / rlc["omega0_rad_s"] < 1e-9
assert abs(Q_check - Q) / Q < 1e-9

# 8. verify_rlc_matches_cavity_decay: at a computationally tractable demo
#    frequency (real optical Q spans too many oscillation periods to
#    literally ODE-simulate -- this session's own finding), the real ODE
#    simulation of the free RLC circuit must match tau_energy=L/R
rlc_demo = rlc_equivalent_from_Q(f0=1e6, Q=50.0, R=50.0)
decay_check = verify_rlc_matches_cavity_decay(rlc_demo["L_H"], rlc_demo["C_F"], rlc_demo["R_ohm"], n_cycles=60)
assert decay_check["matches"] is True
assert decay_check["relative_error"] < 0.02

# 9. verify_rlc_matches_cavity_decay: input validation
for bad in [(-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)]:
    try:
        verify_rlc_matches_cavity_decay(*bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

# 10. laser_threshold_gain / verify_threshold_condition: the closed-form
#     threshold gain must satisfy the round-trip unity-gain condition
#     exactly when checked directly (not just trusted from the formula)
g_th = laser_threshold_gain(R1, R2, alpha, L)
assert g_th > 0
assert verify_threshold_condition(R1, R2, alpha, L) is True

# more loss (lower R2) must require MORE threshold gain
g_th_lossier = laser_threshold_gain(R1, 0.80, alpha, L)
assert g_th_lossier > g_th

# 11. electrical_threshold_analog: net resistance is EXACTLY zero at
#     threshold, for any R_loss
for R_loss in (10.0, 50.0, 1000.0):
    analog = electrical_threshold_analog(R_loss)
    assert analog["net_R_ohm"] == 0.0
    assert analog["at_threshold"] is True
try:
    electrical_threshold_analog(-5.0)
    raise AssertionError("expected ValueError for R_loss <= 0")
except ValueError:
    pass

print("all dgs.laser_cavity_rlc_analog tests passed")
