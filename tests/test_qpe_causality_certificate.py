"""Tests for dgs/qpe_causality_certificate.py"""
import pytest

from dgs.qpe_causality_certificate import (
    two_scale_phase_unwrap, monte_carlo_unwrap_success_rates, wilson_ci,
)


class TestTwoScalePhaseUnwrap:
    def test_recovers_true_theta_within_qpe_precision(self):
        result = two_scale_phase_unwrap(2.63, K=5, n_ancilla=5)
        assert abs(result["certified_estimate"] - 2.63) < 1.0 / (2 ** 5)

    def test_naive_estimate_is_wrap_blind(self):
        # naive is just theta mod 1 -- it CANNOT be near 2.63 for true theta=2.63
        result = two_scale_phase_unwrap(2.63, K=5, n_ancilla=5)
        assert abs(result["naive_estimate"] - 2.63) > 1.0

    def test_rejects_theta_outside_range(self):
        with pytest.raises(ValueError):
            two_scale_phase_unwrap(5.5, K=5, n_ancilla=5)
        with pytest.raises(ValueError):
            two_scale_phase_unwrap(-0.1, K=5, n_ancilla=5)

    def test_rejects_non_integer_K(self):
        with pytest.raises(ValueError):
            two_scale_phase_unwrap(1.0, K=2.5, n_ancilla=5)

    def test_works_for_k_equals_one_trivial_case(self):
        # K=1: no wrap ambiguity at all -- naive and certified must agree
        result = two_scale_phase_unwrap(0.4, K=1, n_ancilla=5)
        assert result["certified_estimate"] == pytest.approx(result["naive_estimate"])

    def test_selected_bucket_matches_floor_of_true_theta(self):
        result = two_scale_phase_unwrap(3.2, K=5, n_ancilla=6)
        assert result["selected_bucket"] == 3


class TestDegenerateSchemeIsActuallyDegenerate:
    """Documents the FIRST (rejected) approach: naively comparing QPE(U)
    to QPE(U^K). Kept here as a cautionary check, matching this repo's
    style of keeping wrong-first-attempts visible (see dgs/critical_points.py,
    dgs/ehrenfest_theorem.py). K*m mod 1 == 0 for ALL integers K, m, so
    every candidate integer offset predicts the identical U^K phase --
    zero discriminating power. Verified directly, not just asserted."""

    def test_uK_scheme_gives_identical_prediction_for_every_offset(self):
        K = 5
        phase_fine = 0.63
        predictions = set()
        for m in range(K):
            theta_candidate = phase_fine + m
            predicted_uK_phase = (K * theta_candidate) % 1.0
            predictions.add(round(predicted_uK_phase, 9))
        # all K candidates predict the SAME U^K phase -- no way to discriminate
        assert len(predictions) == 1


class TestMonteCarloUnwrapSuccessRates:
    def test_certified_beats_naive_by_a_wide_margin(self):
        mc = monte_carlo_unwrap_success_rates(K=5, n_ancilla=5, n_trials=200, seed=0)
        assert mc["certified_success_rate"] > 0.9
        assert mc["naive_success_rate"] < 0.3

    def test_naive_success_rate_matches_theoretical_one_over_k(self):
        mc = monte_carlo_unwrap_success_rates(K=5, n_ancilla=5, n_trials=200, seed=0)
        assert mc["naive_success_rate"] == pytest.approx(1.0 / 5, abs=0.1)

    def test_failures_cluster_near_the_k_boundary(self):
        mc = monte_carlo_unwrap_success_rates(K=5, n_ancilla=5, n_trials=200, seed=0)
        for f in mc["failures"]:
            assert (5 - f["theta_true"]) < 0.1


class TestWilsonCI:
    def test_contains_the_point_estimate(self):
        lo, hi = wilson_ci(197, 200)
        assert lo <= 197 / 200 <= hi

    def test_zero_trials_returns_zero_interval(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)
