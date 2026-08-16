"""Test dgs/measurement_safety_margins.py: guard-banded acceptance
limits, the false-accept-probability closed form (checked by direct
Monte Carlo, not just quoted), the coverage-factor inverse, and
guard-banding a derived quantity via dgs.error_propagation's existing
machinery."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.measurement_safety_margins import (
    guard_band, false_accept_probability_at_limit, verify_false_accept_rate_by_monte_carlo,
    coverage_factor_for_target_risk, guard_band_for_derived_quantity,
    verify_derived_quantity_guard_band_by_monte_carlo,
)

# 1. guard_band: upper limit tightens DOWN, lower limit tightens UP
upper = guard_band(spec_limit=100.0, sigma=5.0, coverage_factor=2.0, limit_type="upper")
assert upper == 90.0
lower = guard_band(spec_limit=100.0, sigma=5.0, coverage_factor=2.0, limit_type="lower")
assert lower == 110.0

for bad in [dict(spec_limit=100.0, sigma=-1.0), dict(spec_limit=100.0, sigma=5.0, coverage_factor=-1.0),
            dict(spec_limit=100.0, sigma=5.0, limit_type="sideways")]:
    try:
        guard_band(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.measurement_safety_margins: guard_band checks passed")

# 2. false_accept_probability_at_limit: known values (standard normal CDF)
assert abs(false_accept_probability_at_limit(1.0) - 0.158655) < 1e-5
assert abs(false_accept_probability_at_limit(2.0) - 0.022750) < 1e-5
assert abs(false_accept_probability_at_limit(3.0) - 0.001350) < 1e-5
# larger k -> strictly smaller risk
assert false_accept_probability_at_limit(3.0) < false_accept_probability_at_limit(2.0) < false_accept_probability_at_limit(1.0)

# 3. verify_false_accept_rate_by_monte_carlo: simulated rate must match
#    theory closely, for both upper and lower limit types
for k in (1.0, 2.0, 3.0):
    check = verify_false_accept_rate_by_monte_carlo(100.0, 5.0, coverage_factor=k, n_trials=1_000_000)
    assert check["abs_diff"] < 5e-3, f"k={k}: {check}"

lower_check = verify_false_accept_rate_by_monte_carlo(100.0, 5.0, coverage_factor=2.0,
                                                       limit_type="lower", n_trials=1_000_000)
assert lower_check["abs_diff"] < 5e-3
assert abs(lower_check["theoretical_false_accept_rate"] - 0.022750) < 1e-5

print("dgs.measurement_safety_margins: false-accept-rate Monte Carlo checks passed")

# 4. coverage_factor_for_target_risk: round-trips back through
#    false_accept_probability_at_limit to the target risk
for target in (0.05, 0.01, 0.001, 0.1):
    k = coverage_factor_for_target_risk(target)
    achieved = false_accept_probability_at_limit(k)
    assert abs(achieved - target) < 1e-6, f"target={target}: k={k}, achieved={achieved}"

for bad_target in (0.0, 0.5, 0.6, -0.1):
    try:
        coverage_factor_for_target_risk(bad_target)
        raise AssertionError(f"expected ValueError for target={bad_target}")
    except ValueError:
        pass

print("dgs.measurement_safety_margins: coverage-factor inversion checks passed")

# 5. guard_band_for_derived_quantity: stress = force/area, propagated
#    sigma must be positive and the pass/fail decision must be internally
#    consistent with the computed guard-banded limit
def stress(vals):
    force, area = vals
    return force / area

nominal = [1000.0, 0.02]
sigmas = [20.0, 0.0005]
result = guard_band_for_derived_quantity(stress, nominal, sigmas, spec_limit=55000.0)
assert result["nominal_value"] == 1000.0 / 0.02
assert result["propagated_sigma"] > 0
if result["passes_guard_banded_test"]:
    assert result["nominal_value"] <= result["guard_banded_limit"]
else:
    assert result["nominal_value"] > result["guard_banded_limit"]

# a spec limit set impossibly low should fail the test
fail_result = guard_band_for_derived_quantity(stress, nominal, sigmas, spec_limit=1000.0)
assert fail_result["passes_guard_banded_test"] is False

print("dgs.measurement_safety_margins: derived-quantity guard band checks passed")

# 6. verify_derived_quantity_guard_band_by_monte_carlo: linearized sigma
#    (dgs.error_propagation.propagate) and Monte Carlo sigma
#    (dgs.error_propagation.propagate_mc) must agree closely for this
#    mild-nonlinearity (division) case
cross_check = verify_derived_quantity_guard_band_by_monte_carlo(stress, nominal, sigmas)
assert cross_check["methods_agree"] is True
assert cross_check["relative_difference"] < 0.05
assert cross_check["sigma_linearized"] > 0 and cross_check["sigma_monte_carlo"] > 0

print("dgs.measurement_safety_margins: linearized-vs-Monte-Carlo cross-check passed")
print("all dgs.measurement_safety_margins tests passed")
