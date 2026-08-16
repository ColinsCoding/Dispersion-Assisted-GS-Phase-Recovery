"""drug_discovery_statistics.py -- the actual statistics behind early-stage
computational drug discovery: drug-likeness screening (Lipinski's Rule of
Five), dose-response curve fitting (the Hill equation -> IC50), QSAR
(quantitative structure-activity relationship) regression, and comparing
candidate compounds -- reusing dgs/statistics.py's t_test_two_sample and
the same bootstrap-CI idea as dgs/hypothesis.py's bootstrap_ci, rather than
reimplementing hypothesis testing from scratch.

FOUR REAL, STANDARD PHARMACOLOGY/CHEMINFORMATICS TOOLS:
  1. Lipinski's Rule of Five (Lipinski et al., Adv. Drug Deliv. Rev. 1997) --
     an empirical drug-likeness filter: molecular weight <= 500, logP <= 5,
     H-bond donors <= 5, H-bond acceptors <= 10. More than one violation
     historically correlates with poor oral bioavailability -- a real,
     widely-used (if approximate) screening heuristic, not a hard law.
  2. The Hill equation (dose-response pharmacology): response =
     bottom + (top-bottom)/(1+(dose/IC50)^hill_coefficient) -- the standard
     sigmoidal model fit to real dose-response assay data to extract IC50
     (half-maximal inhibitory concentration) and the Hill coefficient
     (cooperativity).
  3. QSAR linear regression: a molecular descriptor (e.g. logP) regressed
     against measured potency (pIC50 = -log10(IC50 in M)) -- the simplest
     real QSAR model, via scipy.stats.linregress (slope, R^2, p-value all
     computed by scipy's tested implementation, not reimplemented here).
  4. Bootstrap confidence interval on a fitted IC50 -- because curve_fit's
     covariance-based standard error assumes the fit residuals are
     Gaussian and the model is locally linear near the optimum, which is
     often a poor approximation for a nonlinear 2-parameter fit on noisy
     assay data; resampling gives a distribution-free alternative.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats as spstats
from typing import Dict, Optional


# ── 1. Lipinski's Rule of Five (drug-likeness screening) ────────────────────

def lipinski_rule_of_five(MW: float, logP: float, h_bond_donors: int,
                           h_bond_acceptors: int) -> Dict:
    """Lipinski et al. 1997's four rules. `violations` counts how many are
    broken; the historical rule of thumb is that MORE THAN ONE violation
    correlates with poor oral bioavailability -- an empirical screening
    heuristic (widely used, well-known exceptions exist), not a
    first-principles physical law.
    """
    if MW <= 0:
        raise ValueError(f"MW={MW}: molecular weight must be positive")
    if h_bond_donors < 0 or h_bond_acceptors < 0:
        raise ValueError("h_bond_donors and h_bond_acceptors must be non-negative")

    rules = {
        "MW <= 500": MW <= 500,
        "logP <= 5": logP <= 5,
        "H-bond donors <= 5": h_bond_donors <= 5,
        "H-bond acceptors <= 10": h_bond_acceptors <= 10,
    }
    violations = sum(1 for passed in rules.values() if not passed)
    return {"rules": rules, "violations": violations,
            "drug_like": violations <= 1,
            "inputs": {"MW": MW, "logP": logP, "h_bond_donors": h_bond_donors,
                       "h_bond_acceptors": h_bond_acceptors}}


# ── 2. Dose-response: the Hill equation ──────────────────────────────────────

def hill_equation(dose, IC50: float, hill_coefficient: float,
                   top: float = 100.0, bottom: float = 0.0):
    """response = bottom + (top-bottom)/(1+(dose/IC50)^hill_coefficient) --
    the standard sigmoidal dose-response model. At dose=IC50, response is
    exactly halfway between bottom and top, by construction."""
    if IC50 <= 0:
        raise ValueError(f"IC50={IC50}: must be positive")
    dose = np.asarray(dose, dtype=float)
    if np.any(dose < 0):
        raise ValueError("dose values must be non-negative")
    return bottom + (top - bottom) / (1 + (dose / IC50) ** hill_coefficient)


def fit_dose_response(doses: np.ndarray, responses: np.ndarray,
                       top: float = 100.0, bottom: float = 0.0,
                       p0: Optional[tuple] = None) -> Dict:
    """Fit IC50 and hill_coefficient to measured (doses, responses) data via
    scipy.optimize.curve_fit (Levenberg-Marquardt). Returns the fitted
    parameters, their covariance-based standard errors, and R^2 of the fit.

    top/bottom are held FIXED at their assumed assay plateau values (the
    standard 2-parameter Hill fit) -- a real assay would determine these
    from actual plateau measurements, not fit them simultaneously with
    IC50/hill_coefficient unless the data actually constrains all 4.
    """
    doses = np.asarray(doses, dtype=float)
    responses = np.asarray(responses, dtype=float)
    if len(doses) != len(responses):
        raise ValueError("doses and responses must have the same length")
    if len(doses) < 4:
        raise ValueError(f"n={len(doses)}: need at least 4 points to fit 2 parameters")
    p0 = p0 or (float(np.median(doses)), 1.0)

    def model(d, IC50, hill):
        return hill_equation(d, IC50, hill, top=top, bottom=bottom)

    # Constrain IC50 > 0 (physically required -- hill_equation itself
    # raises otherwise) so the optimizer can't wander into invalid
    # territory mid-search and crash instead of just fitting poorly;
    # hill_coefficient is left essentially unconstrained (a real fit can
    # have either sign depending on activation vs. inhibition).
    popt, pcov = curve_fit(model, doses, responses, p0=p0, maxfev=10000,
                            bounds=([1e-9, -20.0], [np.inf, 20.0]))
    perr = np.sqrt(np.diag(pcov))
    residuals = responses - model(doses, *popt)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((responses - responses.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"IC50": float(popt[0]), "IC50_stderr": float(perr[0]),
            "hill_coefficient": float(popt[1]), "hill_coefficient_stderr": float(perr[1]),
            "r_squared": r_squared, "top": top, "bottom": bottom}


def bootstrap_ic50_ci(doses: np.ndarray, responses: np.ndarray, n_boot: int = 1000,
                       ci: float = 0.95, top: float = 100.0, bottom: float = 0.0,
                       seed: int = 0) -> Dict:
    """Nonparametric bootstrap CI for the fitted IC50: resample (dose,
    response) PAIRS with replacement, refit, repeat n_boot times, and take
    percentiles of the resulting IC50 distribution -- the same
    resample-and-repeat idea as dgs/hypothesis.py's bootstrap_ci, applied
    to a nonlinear curve fit rather than a simple statistic, since a
    curve fit can't be passed directly as bootstrap_ci's `statistic`
    argument (it needs BOTH doses and responses resampled together, not a
    single 1-D array)."""
    doses = np.asarray(doses, dtype=float)
    responses = np.asarray(responses, dtype=float)
    n = len(doses)
    if n < 4:
        raise ValueError(f"n={n}: need at least 4 points")
    rng = np.random.default_rng(seed)
    point_estimate = fit_dose_response(doses, responses, top=top, bottom=bottom)["IC50"]

    ic50_boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            fit = fit_dose_response(doses[idx], responses[idx], top=top, bottom=bottom)
            ic50_boots.append(fit["IC50"])
        except (RuntimeError, ValueError):
            continue  # curve_fit failed to converge on this resample -- skip it, don't crash
    ic50_boots = np.array(ic50_boots)
    lo, hi = np.percentile(ic50_boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"IC50_point_estimate": point_estimate, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_successful_boots": len(ic50_boots), "n_requested_boots": n_boot}


# ── 3. QSAR: descriptor vs. potency linear regression ────────────────────────

def qsar_linear_regression(descriptor: np.ndarray, pIC50: np.ndarray) -> Dict:
    """Simple QSAR: regress a molecular descriptor (e.g. logP) against
    pIC50 = -log10(IC50 in M) (higher pIC50 = more potent). Uses
    scipy.stats.linregress directly (tested, not reimplemented) for the
    slope, intercept, R^2, and the p-value that the slope is nonzero."""
    descriptor = np.asarray(descriptor, dtype=float)
    pIC50 = np.asarray(pIC50, dtype=float)
    if len(descriptor) != len(pIC50):
        raise ValueError("descriptor and pIC50 must have the same length")
    if len(descriptor) < 3:
        raise ValueError(f"n={len(descriptor)}: need at least 3 points for a regression")
    res = spstats.linregress(descriptor, pIC50)
    return {"slope": float(res.slope), "intercept": float(res.intercept),
            "r_squared": float(res.rvalue ** 2), "p_value": float(res.pvalue),
            "stderr": float(res.stderr), "significant_at_5pct": bool(res.pvalue < 0.05)}


# ── 4. Comparing two candidate compounds ─────────────────────────────────────

def compare_compounds(potency_a: np.ndarray, potency_b: np.ndarray) -> Dict:
    """Is compound A significantly more/less potent than compound B, given
    replicate potency measurements for each? Direct reuse of
    dgs.statistics.t_test_two_sample (Welch's t-test), not reimplemented."""
    from dgs.statistics import t_test_two_sample
    return t_test_two_sample(potency_a, potency_b)


if __name__ == "__main__":
    print("=== 1. Lipinski's Rule of Five ===")
    for name, params in [
        ("aspirin-like", dict(MW=180.0, logP=1.2, h_bond_donors=1, h_bond_acceptors=4)),
        ("large peptide-like", dict(MW=950.0, logP=6.5, h_bond_donors=8, h_bond_acceptors=14)),
    ]:
        result = lipinski_rule_of_five(**params)
        print(f"  {name}: violations={result['violations']}  drug_like={result['drug_like']}")

    print("\n=== 2. Dose-response fitting (Hill equation) ===")
    rng = np.random.default_rng(0)
    true_IC50, true_hill = 15.0, 1.3
    doses = np.logspace(-1, 3, 30)
    responses = hill_equation(doses, true_IC50, true_hill) + rng.normal(0, 3.0, len(doses))
    fit = fit_dose_response(doses, responses)
    print(f"  true IC50={true_IC50}, fitted={fit['IC50']:.2f} +/- {fit['IC50_stderr']:.2f}")
    print(f"  true hill={true_hill}, fitted={fit['hill_coefficient']:.2f} +/- {fit['hill_coefficient_stderr']:.2f}")
    print(f"  R^2 = {fit['r_squared']:.4f}")

    boot = bootstrap_ic50_ci(doses, responses, n_boot=500)
    print(f"  Bootstrap 95% CI on IC50: [{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}] "
          f"({boot['n_successful_boots']}/{boot['n_requested_boots']} resamples converged)")

    print("\n=== 3. QSAR: logP vs. pIC50 ===")
    logP = rng.uniform(-1, 5, 25)
    pIC50 = 4.0 + 0.6 * logP + rng.normal(0, 0.4, 25)
    qsar = qsar_linear_regression(logP, pIC50)
    print(f"  slope={qsar['slope']:.3f}  R^2={qsar['r_squared']:.3f}  "
          f"p={qsar['p_value']:.2e}  significant={qsar['significant_at_5pct']}")

    print("\n=== 4. Comparing two candidate compounds ===")
    compound_a = rng.normal(7.5, 0.3, 6)   # pIC50 replicates
    compound_b = rng.normal(7.0, 0.3, 6)
    comparison = compare_compounds(compound_a, compound_b)
    print(f"  mean pIC50: A={comparison['mean_a']:.2f}  B={comparison['mean_b']:.2f}  "
          f"p={comparison['p_value']:.4f}  significant={comparison['significant_at_5pct']}")
