"""Test dgs/drug_discovery_statistics.py: Lipinski's Rule of Five, Hill-
equation dose-response fitting (recovers known parameters from noisy
data), QSAR regression, bootstrap IC50 confidence intervals, and
compound comparison (reusing dgs.statistics.t_test_two_sample)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.drug_discovery_statistics import (
    lipinski_rule_of_five, hill_equation, fit_dose_response,
    bootstrap_ic50_ci, qsar_linear_regression, compare_compounds,
)

# 1. Lipinski's Rule of Five: a clean drug-like molecule (aspirin-like) has
#    0 violations; a molecule violating everything is correctly flagged
r1 = lipinski_rule_of_five(MW=180.0, logP=1.2, h_bond_donors=1, h_bond_acceptors=4)
assert r1["violations"] == 0
assert r1["drug_like"] is True

r2 = lipinski_rule_of_five(MW=950.0, logP=6.5, h_bond_donors=8, h_bond_acceptors=14)
assert r2["violations"] == 4
assert r2["drug_like"] is False

# 2. Lipinski bounds
for bad_kwargs in [dict(MW=0.0, logP=1.0, h_bond_donors=1, h_bond_acceptors=1),
                    dict(MW=100.0, logP=1.0, h_bond_donors=-1, h_bond_acceptors=1)]:
    try:
        lipinski_rule_of_five(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

# 3. hill_equation: at dose=IC50, response must be exactly the midpoint
#    (bottom+top)/2, by construction of the formula
mid = hill_equation(np.array([15.0]), IC50=15.0, hill_coefficient=1.3, top=100.0, bottom=0.0)
assert abs(mid[0] - 50.0) < 1e-9

# 4. hill_equation bounds
try:
    hill_equation(np.array([1.0]), IC50=0.0, hill_coefficient=1.0)
    raise AssertionError("expected ValueError for IC50<=0")
except ValueError:
    pass
try:
    hill_equation(np.array([-1.0]), IC50=1.0, hill_coefficient=1.0)
    raise AssertionError("expected ValueError for negative dose")
except ValueError:
    pass

# 5. fit_dose_response: must recover known IC50/hill_coefficient from
#    noisy synthetic data to within a few standard errors
rng = np.random.default_rng(0)
true_IC50, true_hill = 15.0, 1.3
doses = np.logspace(-1, 3, 30)
responses = hill_equation(doses, true_IC50, true_hill) + rng.normal(0, 3.0, len(doses))
fit = fit_dose_response(doses, responses)
assert abs(fit["IC50"] - true_IC50) < 5 * fit["IC50_stderr"], (
    f"fitted IC50={fit['IC50']:.2f} should be within 5 stderr of true {true_IC50}")
assert abs(fit["hill_coefficient"] - true_hill) < 5 * fit["hill_coefficient_stderr"]
assert fit["r_squared"] > 0.9, f"expected a good fit (R^2>0.9), got {fit['r_squared']:.3f}"

# 6. fit_dose_response bounds
try:
    fit_dose_response(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    raise AssertionError("expected ValueError for n<4")
except ValueError:
    pass
try:
    fit_dose_response(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
    raise AssertionError("expected ValueError for mismatched lengths")
except ValueError:
    pass

# 7. bootstrap_ic50_ci: the true IC50 should fall inside (or very close
#    to) the reported CI, and the point estimate should match
#    fit_dose_response's own answer on the full data
boot = bootstrap_ic50_ci(doses, responses, n_boot=300, seed=1)
assert boot["ci_lo"] < boot["ci_hi"]
assert boot["n_successful_boots"] > 0.8 * boot["n_requested_boots"], (
    "most bootstrap resamples should converge")
assert abs(boot["IC50_point_estimate"] - fit["IC50"]) < 1e-9

# 8. qsar_linear_regression: must recover a known slope from synthetic
#    correlated data, and correctly flag significance
logP = rng.uniform(-1, 5, 25)
true_slope = 0.6
pIC50 = 4.0 + true_slope * logP + rng.normal(0, 0.4, 25)
qsar = qsar_linear_regression(logP, pIC50)
assert abs(qsar["slope"] - true_slope) < 5 * qsar["stderr"]
assert qsar["significant_at_5pct"] is True
assert 0.0 <= qsar["r_squared"] <= 1.0

# 9. qsar_linear_regression: uncorrelated data should NOT show significance
noise_only = rng.normal(0, 1, 25)
qsar_null = qsar_linear_regression(rng.normal(0, 1, 25), noise_only)
# (not asserting significant_at_5pct is always False -- 5% of random draws
#  will spuriously appear significant; just check the function runs and
#  returns a sane p-value in [0,1])
assert 0.0 <= qsar_null["p_value"] <= 1.0

# 10. qsar_linear_regression bounds
try:
    qsar_linear_regression(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    raise AssertionError("expected ValueError for n<3")
except ValueError:
    pass

# 11. compare_compounds: reuses dgs.statistics.t_test_two_sample directly
#     -- a clearly different-potency pair should be flagged significant
compound_a = np.array([9.0, 9.1, 8.9, 9.05, 9.0])
compound_b = np.array([5.0, 5.1, 4.9, 5.05, 5.0])
comparison = compare_compounds(compound_a, compound_b)
assert comparison["significant_at_5pct"] is True
assert comparison["mean_a"] > comparison["mean_b"]

print("all dgs.drug_discovery_statistics tests passed")
