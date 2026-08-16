"""Measurement uncertainty guard-banding: how much margin a safety-
critical accept/reject decision needs, given a KNOWN measurement
uncertainty, to bound the probability of accepting a non-compliant item
-- the real metrology practice (ISO/IEC 17025, ILAC-G8 informed) behind
"quantify uncertainty, measure safely." Extends dgs.error_propagation's
existing uncertainty machinery (Measurement, propagate, propagate_mc)
directly, rather than building a parallel uncertainty model.

THE PROBLEM: a measured value x_measured = x_true + noise (noise ~
N(0, sigma)) is compared against a specification limit L. Comparing the
RAW measurement against L directly is unsafe: a truly non-compliant item
(x_true > L) can still measure BELOW L purely from noise, and get
accepted. Guard-banding tightens the acceptance limit to
L' = L - k*sigma (k = "coverage factor"), so an item is only accepted if
its MEASURED value clears L' -- a real margin against measurement noise,
not just the raw spec.

THE GUARANTEE, verified below by direct Monte Carlo (not just quoted from
a formula): for the WORST-CASE truly-non-compliant item (x_true exactly
at the spec limit L), the probability it still gets accepted under the
guard-banded rule is EXACTLY Phi(-k) (the standard normal CDF evaluated
at -k) -- k=2 bounds false-accept risk at ~2.28%, k=3 at ~0.135%. Any item
with x_true further past the limit has an even LOWER false-accept
probability, so this is the worst case, not a typical case.

WHEN THE MEASURED QUANTITY IS DERIVED (e.g. stress = force/area, not read
directly off one sensor): the sigma used for guard-banding must be the
PROPAGATED uncertainty, not one raw sensor's spec -- computed here via
dgs.error_propagation.propagate, reusing that module's numerical-Jacobian
machinery unmodified.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm

from dgs.error_propagation import propagate, propagate_mc


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


def _validate_limit_type(limit_type: str) -> None:
    if limit_type not in ("upper", "lower"):
        raise ValueError(f"limit_type must be 'upper' or 'lower', got {limit_type!r}")


# ── 1. Guard-banded acceptance limit ────────────────────────────────────────

def guard_band(spec_limit: float, sigma: float, coverage_factor: float = 2.0,
               limit_type: str = "upper") -> float:
    """The guard-banded acceptance limit L' = L - k*sigma (upper limit,
    e.g. a maximum allowed stress) or L' = L + k*sigma (lower limit, e.g.
    a minimum allowed strength) -- tightened relative to the raw spec
    limit L by the coverage factor k times the measurement uncertainty."""
    _validate_positive(sigma=sigma, coverage_factor=coverage_factor)
    _validate_limit_type(limit_type)
    margin = coverage_factor * sigma
    return spec_limit - margin if limit_type == "upper" else spec_limit + margin


def false_accept_probability_at_limit(coverage_factor: float = 2.0) -> float:
    """Phi(-k): the theoretical worst-case false-accept probability for
    an item whose TRUE value sits exactly at the spec limit, under the
    guard-banded rule -- the exact closed form the Monte Carlo check
    below is verified against."""
    _validate_positive(coverage_factor=coverage_factor)
    return float(norm.cdf(-coverage_factor))


def verify_false_accept_rate_by_monte_carlo(spec_limit: float, sigma: float,
                                            coverage_factor: float = 2.0,
                                            limit_type: str = "upper",
                                            n_trials: int = 2_000_000, seed: int = 0) -> dict:
    """CHECKED, not assumed: simulates n_trials noisy measurements of an
    item whose TRUE value sits exactly at the spec limit (the documented
    worst case), applies the guard-banded accept/reject rule to the
    MEASURED value, and compares the simulated false-accept rate against
    the theoretical Phi(-k) -- an independent numerical confirmation of
    the closed-form guarantee, not a restatement of it."""
    _validate_positive(sigma=sigma, coverage_factor=coverage_factor, n_trials=n_trials)
    _validate_limit_type(limit_type)
    L_guard = guard_band(spec_limit, sigma, coverage_factor, limit_type)
    rng = np.random.default_rng(seed)
    measured = spec_limit + rng.normal(0.0, sigma, n_trials)
    if limit_type == "upper":
        accepted = measured <= L_guard
    else:
        accepted = measured >= L_guard
    simulated_rate = float(accepted.mean())
    theoretical_rate = false_accept_probability_at_limit(coverage_factor)
    return {"guard_banded_limit": L_guard, "simulated_false_accept_rate": simulated_rate,
            "theoretical_false_accept_rate": theoretical_rate,
            "abs_diff": abs(simulated_rate - theoretical_rate)}


# ── 2. Coverage factor needed for a target risk level ───────────────────────

def coverage_factor_for_target_risk(target_false_accept_prob: float) -> float:
    """Inverse of false_accept_probability_at_limit: the coverage factor
    k needed so the worst-case false-accept probability does not exceed a
    target risk level (e.g. 0.01 for 1% risk)."""
    if not (0 < target_false_accept_prob < 0.5):
        raise ValueError(f"target_false_accept_prob must be in (0, 0.5), got {target_false_accept_prob}")
    return float(-norm.ppf(target_false_accept_prob))


# ── 3. Guard-banding a DERIVED quantity, via dgs.error_propagation ─────────

def guard_band_for_derived_quantity(f, nominal_values, sigmas, spec_limit: float,
                                    coverage_factor: float = 2.0, limit_type: str = "upper") -> dict:
    """Guard-bands a quantity that is COMPUTED from several raw
    measurements (e.g. stress = force/area), not read directly off one
    sensor -- the sigma used is the PROPAGATED uncertainty from
    dgs.error_propagation.propagate (that module's own numerical-Jacobian
    machinery, called here unmodified), not a single raw sensor spec."""
    nominal_value, sigma_propagated = propagate(f, nominal_values, sigmas)
    L_guard = guard_band(spec_limit, sigma_propagated, coverage_factor, limit_type)
    if limit_type == "upper":
        passes = nominal_value <= L_guard
    else:
        passes = nominal_value >= L_guard
    return {"nominal_value": nominal_value, "propagated_sigma": sigma_propagated,
            "guard_banded_limit": L_guard, "passes_guard_banded_test": bool(passes)}


def verify_derived_quantity_guard_band_by_monte_carlo(f, nominal_values, sigmas,
                                                       coverage_factor: float = 2.0,
                                                       n_trials: int = 200_000, seed: int = 0) -> dict:
    """CHECKED: the propagated sigma from dgs.error_propagation.propagate
    (linearized, first-order) is cross-checked against
    dgs.error_propagation.propagate_mc's own Monte Carlo propagation
    (that module's independent, non-linearized method) -- confirms the
    guard band computed above rests on a sigma that's actually correct,
    not just internally self-consistent."""
    _, sigma_linear = propagate(f, nominal_values, sigmas)
    _, sigma_mc = propagate_mc(f, nominal_values, sigmas, n=n_trials, seed=seed)
    rel_diff = abs(sigma_linear - sigma_mc) / sigma_mc
    return {"sigma_linearized": sigma_linear, "sigma_monte_carlo": sigma_mc,
            "relative_difference": rel_diff, "methods_agree": bool(rel_diff < 0.05)}


if __name__ == "__main__":
    print("=== 1. Guard-banded acceptance limit ===")
    L, sigma, k = 100.0, 5.0, 2.0
    L_guard = guard_band(L, sigma, k)
    print(f"  spec limit L={L}, sigma={sigma}, k={k}: guard-banded limit = {L_guard}")

    print("\n=== 2. False-accept risk, closed form vs. Monte Carlo ===")
    for k_test in (1.0, 2.0, 3.0):
        theory = false_accept_probability_at_limit(k_test)
        check = verify_false_accept_rate_by_monte_carlo(L, sigma, k_test)
        print(f"  k={k_test}: theory={theory:.5f}, simulated={check['simulated_false_accept_rate']:.5f}, "
              f"diff={check['abs_diff']:.2e}")

    print("\n=== 3. Coverage factor needed for a target risk level ===")
    for target in (0.05, 0.01, 0.001):
        k_needed = coverage_factor_for_target_risk(target)
        achieved = false_accept_probability_at_limit(k_needed)
        print(f"  target risk={target:.3%}: k={k_needed:.4f}, achieved risk={achieved:.5f}")

    print("\n=== 4. Guard-banding a derived quantity (stress = force/area) ===")
    def stress(vals):
        force, area = vals
        return force / area

    nominal = [1000.0, 0.02]   # N, m^2
    sigmas_raw = [20.0, 0.0005]   # N, m^2 sensor uncertainties
    spec_limit_stress = 55000.0   # Pa, e.g. a material yield-stress-derived limit

    result = guard_band_for_derived_quantity(stress, nominal, sigmas_raw, spec_limit_stress)
    print(f"  nominal stress = {result['nominal_value']:.1f} Pa, propagated sigma = {result['propagated_sigma']:.1f} Pa")
    print(f"  guard-banded limit = {result['guard_banded_limit']:.1f} Pa")
    print(f"  passes guard-banded test: {result['passes_guard_banded_test']}")

    cross_check = verify_derived_quantity_guard_band_by_monte_carlo(stress, nominal, sigmas_raw)
    print(f"\n  sigma (linearized): {cross_check['sigma_linearized']:.3f} Pa")
    print(f"  sigma (Monte Carlo): {cross_check['sigma_monte_carlo']:.3f} Pa")
    print(f"  relative difference: {cross_check['relative_difference']:.2%}, "
          f"methods agree: {cross_check['methods_agree']}")

    print("\nA guard band isn't a rule of thumb here -- its false-accept probability is")
    print("an exact number (Phi(-k)), verified by direct simulation, and the sigma feeding")
    print("it into a derived quantity is dgs.error_propagation's own machinery, not a new model.")
