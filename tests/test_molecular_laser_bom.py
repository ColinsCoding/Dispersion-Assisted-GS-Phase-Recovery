"""Test dgs.molecular_laser_bom: the self-termination window must be a
genuine crossing point of the population kinetics (not a hardcoded
number), gain-window/energy/power arithmetic must be internally
consistent, the single-pass gain reuse of dgs.laser_physics must match
that module directly, and the BOM must actually total to a real, checkable
number via dgs.hardware_bom's own cost function (reused, not
reimplemented)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.molecular_laser_bom import (
    photon_energy, population_kinetics, gain_window_duration, discharge_energy,
    peak_electrical_power, optical_output, amplification_over_gain_length,
    mirrorless_vs_cavity_threshold, BOM_DIY_N2_LASER, budget_feasibility,
    N2_UPPER_LIFETIME_NS, N2_LOWER_LIFETIME_NS, N2_TYPICAL_GAIN_PER_M,
)
from dgs.laser_physics import intensity_after_gain
from dgs.hardware_bom import bom_total_cost

# 1. photon energy at 337.1 nm is the well-known ~3.68 eV UV photon
E = photon_energy()
assert 3.6 < E["E_eV"] < 3.8, f"N2 laser photon energy should be ~3.68 eV, got {E['E_eV']}"

# 2. population kinetics: N2 starts at N0 and decays monotonically;
# N1 starts at 0, must eventually exceed N2 (that's the whole point)
t = np.linspace(0, 100, 2000)
N2_t, N1_t = population_kinetics(t, N2_UPPER_LIFETIME_NS, N2_LOWER_LIFETIME_NS, N0=1.0)
assert np.isclose(N2_t[0], 1.0, atol=1e-3)
assert np.isclose(N1_t[0], 0.0, atol=1e-6)
assert np.all(np.diff(N2_t) <= 1e-12), "N2 should be monotonically decaying"
assert N1_t.max() > N2_t[-1] or (N1_t > N2_t).any(), "N1 should eventually overtake N2 (self-termination)"

# 3. gain_window_duration finds an ACTUAL crossing: N2 ~ N1 there, and it's
# in a physically sane nanosecond range for these lifetimes
t_window = gain_window_duration(N2_UPPER_LIFETIME_NS, N2_LOWER_LIFETIME_NS)
assert 1.0 < t_window < 100.0, f"expected a few-to-tens-of-ns gain window, got {t_window} ns"
N2_at_cross, N1_at_cross = population_kinetics(t_window, N2_UPPER_LIFETIME_NS, N2_LOWER_LIFETIME_NS)
assert abs(N2_at_cross - N1_at_cross) / max(N2_at_cross, N1_at_cross, 1e-9) < 0.05, \
    "at the reported crossing time, N2 and N1 should actually be approximately equal"

# 4. rejects the non-self-terminating regime (tau_lower <= tau_upper)
try:
    gain_window_duration(tau_upper_ns=100.0, tau_lower_ns=50.0)
    assert False, "should reject tau_lower <= tau_upper"
except ValueError:
    pass

# 5. discharge energetics: basic arithmetic, and MW-scale peak power from mJ-scale energy on ns timescales
C, V, dt = 1e-9, 20000.0, 3e-9
E_elec = discharge_energy(C, V)
assert abs(E_elec - 0.5 * C * V ** 2) < 1e-15
P_peak = peak_electrical_power(E_elec, dt)
assert P_peak > 1e6, f"expected MW-scale peak power from a ns-scale discharge, got {P_peak:.3e} W"

opt = optical_output(E_elec, efficiency=0.005, pulse_duration_s=dt)
assert abs(opt["optical_energy_J"] - E_elec * 0.005) < 1e-15
assert opt["optical_energy_J"] < E_elec, "optical output can't exceed electrical input"

# 6. single-pass gain matches dgs.laser_physics.intensity_after_gain directly (real reuse, not reimplemented)
amp = amplification_over_gain_length(N2_TYPICAL_GAIN_PER_M, 0.4)
expected_ratio = intensity_after_gain(1.0, N2_TYPICAL_GAIN_PER_M, 0.4)
assert abs(amp["I_over_I0"] - expected_ratio) < 1e-9
assert amp["I_over_I0"] > 1.0, "gain should amplify (ratio > 1)"

# 7. N2's typical gain exceeds a bare-window cavity's threshold -- the mirrorless/ASE claim
cmp = mirrorless_vs_cavity_threshold(N2_TYPICAL_GAIN_PER_M, 0.4)
assert cmp["exceeds_threshold"] is True
assert cmp["actual_gain_per_m"] > cmp["threshold_gain_per_m"]

# 8. the BOM totals to a real, checkable number via dgs.hardware_bom's own function
total = bom_total_cost(BOM_DIY_N2_LASER)
assert total == sum(item["qty"] * item["approx_usd"] for item in BOM_DIY_N2_LASER)
assert 500 < total < 1500, f"BOM total ${total} should be in a realistic hobbyist-project range"

feas = budget_feasibility(BOM_DIY_N2_LASER, budget_usd=1000.0)
assert feas["total_usd"] == total
assert feas["within_budget"] == (total <= 1000.0)

# 9. input validation
try:
    photon_energy(-1.0)
    assert False, "should reject negative wavelength"
except ValueError:
    pass
try:
    discharge_energy(-1e-9, 1000.0)
    assert False, "should reject negative capacitance"
except ValueError:
    pass
try:
    optical_output(1.0, efficiency=1.5, pulse_duration_s=1e-9)
    assert False, "should reject efficiency outside (0,1)"
except ValueError:
    pass

print(f"all dgs.molecular_laser_bom tests passed  (BOM total=${total}, gain window={t_window:.2f} ns)")
