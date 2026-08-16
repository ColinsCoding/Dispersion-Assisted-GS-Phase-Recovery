import numpy as np
import pytest
from dgs.medical_cost_effectiveness import (
    amortized_instrument_cost_per_test, gs_compute_cost_per_test, cost_per_test,
    ctc_screening_cost_report, retinal_scan_cost_report, retinal_cost_vs_threshold,
    sample_retinal_rms_errors, compare_use_cases,
)


def test_amortized_instrument_cost_per_test_basic():
    cost = amortized_instrument_cost_per_test(capex=100_000, lifetime_tests=10_000)
    assert cost == pytest.approx(10.0)


def test_amortized_instrument_cost_per_test_with_maintenance():
    cost = amortized_instrument_cost_per_test(capex=100_000, lifetime_tests=10_000,
                                               annual_maintenance=5_000, tests_per_year=1_000)
    assert cost == pytest.approx(10.0 + 5.0)


def test_amortized_instrument_cost_per_test_rejects_bad_input():
    with pytest.raises(ValueError):
        amortized_instrument_cost_per_test(capex=-1, lifetime_tests=10)
    with pytest.raises(ValueError):
        amortized_instrument_cost_per_test(capex=1, lifetime_tests=0)


def test_gs_compute_cost_per_test_scales_linearly_with_iters():
    base = gs_compute_cost_per_test(n_iter=50, n_samples=1024, gpu_cost_per_hour=3.6)
    doubled = gs_compute_cost_per_test(n_iter=100, n_samples=1024, gpu_cost_per_hour=3.6)
    assert doubled == pytest.approx(2 * base)


def test_gs_compute_cost_per_test_is_tiny():
    cost = gs_compute_cost_per_test(n_iter=50, n_samples=1024, gpu_cost_per_hour=5.0)
    assert cost < 1e-6


def test_gs_compute_cost_per_test_rejects_nonpositive():
    with pytest.raises(ValueError):
        gs_compute_cost_per_test(n_iter=0, n_samples=1024, gpu_cost_per_hour=1.0)


def test_cost_per_test_sums_components():
    total = cost_per_test(instrument_cost=10.0, consumable_cost=5.0, compute_cost=0.1)
    assert total == pytest.approx(15.1)


def test_cost_per_test_rejects_negative():
    with pytest.raises(ValueError):
        cost_per_test(instrument_cost=-1.0, consumable_cost=5.0)


def test_ctc_screening_cost_report_ppv_matches_bayes_inference():
    from dgs.bayes_inference import detection_posterior
    report = ctc_screening_cost_report(cost_per_screen=25.0)
    expected_ppv = detection_posterior(1e-6, 0.999, 1e-6, observed="alarm")
    assert report["ppv"] == pytest.approx(expected_ppv)


def test_ctc_screening_cost_report_base_rate_fallacy_holds():
    # 1e-6 prevalence with 1e-6 false-alarm rate: PPV should still be well under 1
    report = ctc_screening_cost_report(cost_per_screen=25.0)
    assert 0 < report["ppv"] < 1
    assert report["false_positives_per_true_positive"] > 0


def test_ctc_screening_cost_report_number_needed_to_screen():
    report = ctc_screening_cost_report(cost_per_screen=25.0, prevalence=1e-6, sensitivity=0.999)
    assert report["number_needed_to_screen"] == pytest.approx(1.0 / (1e-6 * 0.999))


def test_ctc_screening_cost_report_total_includes_confirmatory():
    report = ctc_screening_cost_report(cost_per_screen=25.0, confirmatory_test_cost=500.0)
    assert report["total_cost_per_true_positive"] == pytest.approx(
        report["cost_per_true_positive"] + report["confirmatory_cost_per_true_positive"])


def test_ctc_screening_cost_report_rejects_negative_cost():
    with pytest.raises(ValueError):
        ctc_screening_cost_report(cost_per_screen=-1.0)


def test_retinal_scan_cost_report_yield_in_unit_interval():
    report = retinal_scan_cost_report(cost_per_scan=15.0, n_trials=10)
    assert 0.0 <= report["yield_fraction"] <= 1.0
    assert len(report["rms_errors_deg"]) == 10


def test_retinal_scan_cost_report_cost_scales_with_inverse_yield():
    report = retinal_scan_cost_report(cost_per_scan=15.0, n_trials=10)
    if report["yield_fraction"] > 0:
        assert report["cost_per_usable_scan"] == pytest.approx(
            15.0 / report["yield_fraction"])


def test_retinal_scan_cost_report_deterministic_with_seed():
    r1 = retinal_scan_cost_report(cost_per_scan=15.0, n_trials=5, rng_seed=42)
    r2 = retinal_scan_cost_report(cost_per_scan=15.0, n_trials=5, rng_seed=42)
    np.testing.assert_array_equal(r1["rms_errors_deg"], r2["rms_errors_deg"])


def test_retinal_scan_cost_report_rejects_negative_cost():
    with pytest.raises(ValueError):
        retinal_scan_cost_report(cost_per_scan=-1.0)


def test_retinal_cost_vs_threshold_monotonic_yield():
    result = retinal_cost_vs_threshold(15.0, thresholds_deg=[10, 30, 60, 90], n_trials=40, rng_seed=1)
    yields = result["yield_fractions"]
    assert all(y2 >= y1 for y1, y2 in zip(yields, yields[1:]))


def test_retinal_cost_vs_threshold_onset_is_first_finite_threshold():
    result = retinal_cost_vs_threshold(15.0, thresholds_deg=[5, 200], n_trials=40, rng_seed=1)
    assert result["finite_cost_onset_threshold_deg"] == pytest.approx(200.0)
    assert np.isinf(result["cost_per_usable_scan"][0])
    assert np.isfinite(result["cost_per_usable_scan"][1])


def test_retinal_cost_vs_threshold_none_finite_returns_none_onset():
    result = retinal_cost_vs_threshold(15.0, thresholds_deg=[0.001], n_trials=5, rng_seed=1)
    assert result["finite_cost_onset_threshold_deg"] is None


def test_retinal_cost_vs_threshold_min_rms_error_is_sample_floor():
    result = retinal_cost_vs_threshold(15.0, thresholds_deg=[50], n_trials=30, rng_seed=1)
    assert result["min_rms_error_deg"] == pytest.approx(result["rms_errors_deg"].min())


def test_sample_retinal_rms_errors_more_iters_does_not_hurt():
    # more GS iterations should not make the recovery systematically worse
    few = sample_retinal_rms_errors(n_trials=15, rng_seed=2, n_iter=50)
    many = sample_retinal_rms_errors(n_trials=15, rng_seed=2, n_iter=500)
    assert many.mean() <= few.mean() + 5.0  # allow noise, but no big regression


def test_compare_use_cases_has_both_and_caveat():
    result = compare_use_cases()
    assert "ctc_blood_screening" in result
    assert "retinal_depth_scanning" in result
    assert "not the same unit" in result["caveat"].lower()
