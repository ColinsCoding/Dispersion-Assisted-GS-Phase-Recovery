"""Tests for dgs/shor_discrete_log.py"""
import numpy as np
import pytest

from dgs.shor_discrete_log import (
    discrete_log_brute_force, multiplicative_order, find_prime_order_generator,
    discrete_log_period_grid, shor_discrete_log_classical, secp256k1_quantum_threat_summary,
)


class TestBruteForceAndOrder:
    def test_brute_force_finds_known_log(self):
        # 2^7 mod 23 = 128 mod 23 = 13
        assert discrete_log_brute_force(2, 13, 23, r=11) == 7

    def test_multiplicative_order_matches_known_value(self):
        assert multiplicative_order(2, 23) == 11

    def test_find_prime_order_generator_has_correct_order(self):
        g = find_prime_order_generator(23, 11)
        assert multiplicative_order(g, 23) == 11

    def test_find_prime_order_generator_rejects_non_divisor(self):
        with pytest.raises(ValueError):
            find_prime_order_generator(23, 5)   # 5 does not divide 22


class TestPeriodGrid:
    def test_grid_matches_closed_form_exactly(self):
        # f(a,b) must equal g^((a - x*b) mod r) mod p for every (a,b) -- the
        # structural claim the whole algorithm depends on, checked directly
        p, r, g, x = 23, 11, 2, 4
        h = pow(g, x, p)
        F = discrete_log_period_grid(g, h, p, r)
        for a in range(r):
            for b in range(r):
                expected = pow(g, (a - x * b) % r, p)
                assert F[a, b] == expected


class TestShorDiscreteLogClassical:
    def test_recovers_known_discrete_log(self):
        p, r, g = 23, 11, 2
        x_true = 7
        h = pow(g, x_true, p)
        result = shor_discrete_log_classical(g, h, p, r, seed=1)
        assert result["unique"]
        assert result["x_recovered"] == x_true

    def test_robust_across_many_random_secrets_and_measurement_samples(self):
        p, r, g = 23, 11, 2
        rng = np.random.default_rng(0)
        successes = 0
        n_trials = 20
        for _ in range(n_trials):
            x_true = int(rng.integers(1, r))
            h = pow(g, x_true, p)
            a0, b0 = int(rng.integers(0, r)), int(rng.integers(0, r))
            result = shor_discrete_log_classical(g, h, p, r, a0=a0, b0=b0)
            if result["unique"] and result["x_recovered"] == x_true:
                successes += 1
        assert successes == n_trials, f"only {successes}/{n_trials} succeeded"

    def test_agrees_with_brute_force_on_a_different_prime(self):
        # a second, independent (p, r, g) triple -- not just the one example
        # tuned to work, matching this session's cross-check discipline
        p, r = 47, 23   # 47-1=46=2*23, r=23 is prime
        g = find_prime_order_generator(p, r)
        x_true = 15
        h = pow(g, x_true, p)
        assert discrete_log_brute_force(g, h, p, r) == x_true
        result = shor_discrete_log_classical(g, h, p, r, seed=2)
        assert result["x_recovered"] == x_true

    def test_rejects_composite_order(self):
        with pytest.raises(ValueError):
            shor_discrete_log_classical(2, 4, 23, r=22)   # r=22=2*11 is composite


class TestThreatSummary:
    def test_returns_sources_and_caveat(self):
        summary = secp256k1_quantum_threat_summary()
        assert "sources" in summary and len(summary["sources"]) >= 2
        assert "caveat" in summary
        assert "secp256k1" in summary["curve"]
