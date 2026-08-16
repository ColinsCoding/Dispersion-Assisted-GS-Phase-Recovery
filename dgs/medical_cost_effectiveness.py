"""medical_cost_effectiveness.py -- cost per successful diagnostic outcome for
this repo's two medical TD-GSA use cases: CTC blood screening
(sbir_portfolio.py's P5_BAYES) and retinal depth scanning
(retinal_scan_imaging.py), tied together with dgs/bayes_inference.py's
detection_posterior base-rate machinery.

Two DIFFERENT cost structures, kept honestly separate rather than collapsed
into one misleading number:

1. CTC BLOOD SCREENING (P5_BAYES) is a RARE-EVENT BINARY DETECTOR: prevalence
   ~1e-6, sensitivity/false-alarm targets are established in
   sbir_portfolio.py's P5 milestones (sensitivity 0.999, false-alarm <1e-6).
   Bayes' theorem (base rate) directly applies via
   dgs/bayes_inference.py's detection_posterior -- the SAME function
   RogueGuard uses for detection confidence, reused here for a cost
   question. The base-rate fallacy that makes rare-event detection hard also
   makes it expensive: most flagged cells are false positives, and each one
   still costs a confirmatory workup.

2. RETINAL DEPTH SCANNING (retinal_scan_imaging.py) has NO disease-prevalence
   model in this repo -- retinal_depth_phase_recovery is a PROPOSED
   depth-recovery technique (see that module's docstring), not a validated
   binary classifier. There is no honest P(disease), sensitivity, or
   specificity number to run through Bayes' theorem. This module instead
   reports cost PER DIAGNOSTIC-QUALITY SCAN (one whose recovered-depth RMS
   phase error clears a quality threshold) -- the cost unit that's actually
   supportable at this stage.

Both share the same cost primitives (instrument amortization + per-test
compute/consumable cost), so the two ARE comparable on a $/successful-outcome
basis -- but "successful outcome" means something structurally different in
each, and compare_use_cases() reports that caveat explicitly rather than
letting the two numbers get read as apples-to-apples.

All dollar figures are function ARGUMENTS with illustrative defaults, hedged
here as such -- not sourced vendor quotes, same posture as
sbir_portfolio.py's photonics_manufacturing_funding_landscape. Verify real
costs before citing any of this in an actual proposal.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

from dgs.bayes_inference import detection_posterior
from dgs.retinal_scan_imaging import (
    retinal_depth_phase_recovery, synthetic_vessel_reflectance_with_depth,
)


# ── shared cost primitives ──────────────────────────────────────────────────

def amortized_instrument_cost_per_test(capex: float, lifetime_tests: float,
                                        annual_maintenance: float = 0.0,
                                        tests_per_year: float = 1.0) -> float:
    """Instrument capex + maintenance spread over its useful-life test count.
    capex: one-time hardware cost ($). lifetime_tests: total tests run before
    replacement/end-of-life. annual_maintenance: $/year upkeep, amortized
    over tests_per_year to get a per-test maintenance share."""
    if capex < 0 or lifetime_tests <= 0:
        raise ValueError("capex must be >= 0 and lifetime_tests must be > 0")
    if annual_maintenance < 0 or tests_per_year <= 0:
        raise ValueError("annual_maintenance must be >= 0 and tests_per_year must be > 0")
    return capex / lifetime_tests + annual_maintenance / tests_per_year


def gs_compute_cost_per_test(n_iter: int, n_samples: int, gpu_cost_per_hour: float) -> float:
    """$/test for running GS phase recovery, scaled from dgs/sbir_portfolio.py's
    P4 CUDA benchmark (50-iteration GS on N=1024 in ~40us on an A100).
    Linear scaling in n_iter*n_samples is the conservative/simple estimate
    (true FFT cost is ~N log N), used here to make the point that GS compute
    is essentially free next to instrument/consumable cost, not to model
    scaling precisely."""
    if n_iter <= 0 or n_samples <= 0 or gpu_cost_per_hour <= 0:
        raise ValueError("n_iter, n_samples, and gpu_cost_per_hour must be positive")
    reference_seconds = 40e-6 * (n_iter / 50) * (n_samples / 1024)
    return reference_seconds / 3600.0 * gpu_cost_per_hour


def cost_per_test(instrument_cost: float, consumable_cost: float, compute_cost: float = 0.0) -> float:
    """Total $/test: instrument amortization + consumables (reagents, blood
    draw kit, etc.) + compute (see gs_compute_cost_per_test -- negligible)."""
    for name, v in [("instrument_cost", instrument_cost),
                    ("consumable_cost", consumable_cost),
                    ("compute_cost", compute_cost)]:
        if v < 0:
            raise ValueError(f"{name} must be non-negative")
    return instrument_cost + consumable_cost + compute_cost


# ── 1. CTC blood screening: rare-event Bayes cost ───────────────────────────

def ctc_screening_cost_report(cost_per_screen: float, prevalence: float = 1e-6,
                               sensitivity: float = 0.999, false_alarm: float = 1e-6,
                               confirmatory_test_cost: float = 0.0) -> Dict:
    """Cost per true positive found for CTC screening, using
    sbir_portfolio.py's P5_BAYES numbers as defaults (prevalence=1e-6,
    sensitivity target 0.999, false-alarm target <1e-6 from milestone M3).
    Reuses dgs/bayes_inference.py's detection_posterior for PPV."""
    if cost_per_screen < 0 or confirmatory_test_cost < 0:
        raise ValueError("costs must be non-negative")
    ppv = detection_posterior(prevalence, sensitivity, false_alarm, observed="alarm")
    number_needed_to_screen = 1.0 / (prevalence * sensitivity)
    cost_per_true_positive = cost_per_screen * number_needed_to_screen
    false_positives_per_true_positive = (1.0 - ppv) / ppv
    confirmatory_cost_per_true_positive = false_positives_per_true_positive * confirmatory_test_cost
    return {
        "ppv": ppv,
        "number_needed_to_screen": number_needed_to_screen,
        "cost_per_screen": cost_per_screen,
        "cost_per_true_positive": cost_per_true_positive,
        "false_positives_per_true_positive": false_positives_per_true_positive,
        "confirmatory_cost_per_true_positive": confirmatory_cost_per_true_positive,
        "total_cost_per_true_positive": cost_per_true_positive + confirmatory_cost_per_true_positive,
    }


# ── 2. Retinal depth scanning: no prevalence model, cost per usable scan ───

def sample_retinal_rms_errors(n_trials: int = 20, rng_seed: int = 0,
                               D1: float = -5000.0, D2: float = -5750.0, n_iter: int = 50,
                               n_vessels: int = 4, depth_amplitude_rad: float = 1.5) -> np.ndarray:
    """Run n_trials synthetic retinal depth-phase reconstructions
    (retinal_scan_imaging.py's retinal_depth_phase_recovery on a fresh
    synthetic_vessel_reflectance_with_depth profile each time) and return the
    RMS depth-phase recovery error (degrees, after global-phase-offset
    alignment) for each trial -- the same error metric that module's own
    __main__ demo reports. D1/D2/n_iter are exposed because they're the
    knobs that actually change recovery quality (and, via
    gs_compute_cost_per_test, are nearly free to turn up)."""
    if n_trials < 1:
        raise ValueError(f"n_trials={n_trials}: must be >= 1")
    rng = np.random.default_rng(rng_seed)
    rms_errors = []
    for _ in range(n_trials):
        seed = int(rng.integers(0, 1_000_000))
        profile = synthetic_vessel_reflectance_with_depth(
            n=256, n_vessels=n_vessels, depth_amplitude_rad=depth_amplitude_rad, rng_seed=seed)
        result = retinal_depth_phase_recovery(profile, D1=D1, D2=D2, n_iter=n_iter)
        offset = np.angle(np.mean(np.exp(1j * (result["phi_true"] - result["phi_est"]))))
        aligned_err = np.angle(np.exp(1j * (result["phi_est"] + offset - result["phi_true"])))
        rms_errors.append(float(np.degrees(np.sqrt(np.mean(aligned_err ** 2)))))
    return np.array(rms_errors)


def retinal_scan_cost_report(cost_per_scan: float, rms_error_threshold_deg: float = 5.0,
                              n_trials: int = 20, rng_seed: int = 0, **recovery_kwargs) -> Dict:
    """Cost per DIAGNOSTIC-QUALITY scan (recovered-depth RMS phase error below
    rms_error_threshold_deg) for the proposed retinal depth-recovery
    technique. There is no disease-prevalence number to run through Bayes'
    theorem here (see module docstring), so 'successful outcome' means a
    usable scan, not a true-positive diagnosis. recovery_kwargs pass through
    to sample_retinal_rms_errors (D1, D2, n_iter, n_vessels, depth_amplitude_rad).
    """
    if cost_per_scan < 0:
        raise ValueError("cost_per_scan must be non-negative")
    rms_errors = sample_retinal_rms_errors(n_trials=n_trials, rng_seed=rng_seed, **recovery_kwargs)
    usable = rms_errors < rms_error_threshold_deg
    yield_fraction = float(usable.mean())
    cost_per_usable_scan = cost_per_scan / yield_fraction if yield_fraction > 0 else float("inf")
    return {
        "rms_errors_deg": rms_errors,
        "yield_fraction": yield_fraction,
        "cost_per_scan": cost_per_scan,
        "cost_per_usable_scan": cost_per_usable_scan,
        "mean_rms_error_deg": float(rms_errors.mean()),
    }


def retinal_cost_vs_threshold(cost_per_scan: float, thresholds_deg, n_trials: int = 200,
                               rng_seed: int = 0, **recovery_kwargs) -> Dict:
    """Sweep rms_error_threshold_deg and report yield fraction + cost per
    usable scan at each threshold, from ONE shared sample of RMS errors
    (cheaper than re-running retinal_scan_cost_report per threshold, and
    apples-to-apples since every threshold sees the same trials). Answers
    'where does the cost turn finite': the smallest threshold in
    thresholds_deg with cost_per_usable_scan < inf, i.e. the first one at
    least one sampled trial clears -- reported as
    finite_cost_onset_threshold_deg, alongside min_rms_error_deg (the true
    floor of the sample, independent of the requested threshold grid)."""
    if cost_per_scan < 0:
        raise ValueError("cost_per_scan must be non-negative")
    thresholds_deg = np.asarray(thresholds_deg, dtype=float)
    if thresholds_deg.size < 1:
        raise ValueError("thresholds_deg must be non-empty")
    rms_errors = sample_retinal_rms_errors(n_trials=n_trials, rng_seed=rng_seed, **recovery_kwargs)
    yield_fractions = np.array([(rms_errors < t).mean() for t in thresholds_deg])
    cost_per_usable_scan = np.array([
        cost_per_scan / y if y > 0 else float("inf") for y in yield_fractions])
    finite_mask = np.isfinite(cost_per_usable_scan)
    onset = float(thresholds_deg[finite_mask].min()) if finite_mask.any() else None
    return {
        "thresholds_deg": thresholds_deg,
        "yield_fractions": yield_fractions,
        "cost_per_usable_scan": cost_per_usable_scan,
        "rms_errors_deg": rms_errors,
        "min_rms_error_deg": float(rms_errors.min()),
        "finite_cost_onset_threshold_deg": onset,
    }


# ── 3. Side-by-side comparison, caveat included ─────────────────────────────

def compare_use_cases(ctc_cost_per_screen: float = 25.0, ctc_confirmatory_cost: float = 500.0,
                       retinal_cost_per_scan: float = 15.0) -> Dict:
    """Side-by-side $/successful-outcome for both use cases. See module
    docstring: the two 'success' units are NOT the same and should not be
    compared as if they were -- the returned 'caveat' string says so
    explicitly for any downstream consumer of this dict."""
    ctc = ctc_screening_cost_report(ctc_cost_per_screen, confirmatory_test_cost=ctc_confirmatory_cost)
    retinal = retinal_scan_cost_report(retinal_cost_per_scan)
    return {
        "ctc_blood_screening": ctc,
        "retinal_depth_scanning": retinal,
        "caveat": ("CTC 'success' = one true positive found (Bayes PPV-driven, "
                   "P5_BAYES numbers); retinal 'success' = one scan clearing an "
                   "RMS-error quality bar (no prevalence model exists for retinal "
                   "yet, see module docstring) -- these are NOT the same unit."),
    }


if __name__ == "__main__":
    print("=== GS compute cost per test (both use cases: negligible) ===")
    compute_cost = gs_compute_cost_per_test(n_iter=50, n_samples=1024, gpu_cost_per_hour=2.0)
    print(f"  ${compute_cost:.8f}/test (50 iters, N=1024, $2/hr GPU)")

    print("\n=== 1. CTC blood screening cost (P5_BAYES numbers) ===")
    ctc = ctc_screening_cost_report(cost_per_screen=25.0, confirmatory_test_cost=500.0)
    print(f"  PPV = {ctc['ppv']:.4f}  (base-rate fallacy: still well under 1)")
    print(f"  number needed to screen per true positive: {ctc['number_needed_to_screen']:,.0f}")
    print(f"  false positives per true positive: {ctc['false_positives_per_true_positive']:.2f}")
    print(f"  cost per true positive found: ${ctc['cost_per_true_positive']:,.0f}")
    print(f"  + confirmatory workup cost: ${ctc['confirmatory_cost_per_true_positive']:,.0f}")
    print(f"  TOTAL cost per true positive: ${ctc['total_cost_per_true_positive']:,.0f}")

    print("\n=== 2. Retinal depth scan cost (proposed technique, no prevalence model) ===")
    retinal = retinal_scan_cost_report(cost_per_scan=15.0)
    print(f"  yield fraction (usable scans): {retinal['yield_fraction']:.2f}")
    print(f"  mean RMS depth-phase error: {retinal['mean_rms_error_deg']:.2f} deg")
    print(f"  cost per usable scan: ${retinal['cost_per_usable_scan']:.2f}")

    print("\n=== 3. Side-by-side (caveat: NOT the same success unit) ===")
    cmp = compare_use_cases()
    print(f"  {cmp['caveat']}")
