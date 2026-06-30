import math
import pytest
import sympy as sp
from dgs.probability_combinatorics import (
    prob_to_odds, odds_to_prob, parity_probability, is_odd,
    geometric_pmf, geometric_cdf, first_success_stats, prob_success_within_n,
    order_statistic_expected, min_max_expected,
    combinations, permutations, binomial_pmf, binomial_mean_var,
    birthday_collision_prob, probability_sympy_5,
)


def test_prob_to_odds_basic():
    r = prob_to_odds(0.5)
    assert r["ratio"] == pytest.approx(1.0)


def test_prob_to_odds_quarter():
    r = prob_to_odds(0.25)
    assert r["ratio"] == pytest.approx(1/3)


def test_prob_to_odds_invalid():
    with pytest.raises(ValueError):
        prob_to_odds(0.0)
    with pytest.raises(ValueError):
        prob_to_odds(1.0)


def test_odds_to_prob_roundtrip():
    p = 0.3
    r = prob_to_odds(p)
    assert odds_to_prob(r["odds_for"], r["odds_against"]) == pytest.approx(p)


def test_parity_1_to_10():
    r = parity_probability(1, 10)
    assert r["P_odd"] == pytest.approx(0.5)
    assert r["P_even"] == pytest.approx(0.5)


def test_parity_1_to_9():
    r = parity_probability(1, 9)
    assert r["n_odd"] == 5
    assert r["n_even"] == 4


def test_is_odd():
    assert is_odd(3) is True
    assert is_odd(4) is False
    assert is_odd(0) is False


def test_geometric_pmf_k1():
    assert geometric_pmf(0.5, 1) == pytest.approx(0.5)


def test_geometric_pmf_sums_to_1():
    p = 0.3
    total = sum(geometric_pmf(p, k) for k in range(1, 200))
    assert total == pytest.approx(1.0, abs=1e-4)


def test_geometric_pmf_invalid():
    with pytest.raises(ValueError):
        geometric_pmf(0.0, 1)
    with pytest.raises(ValueError):
        geometric_pmf(0.5, 0)


def test_geometric_cdf_certain():
    assert geometric_cdf(1.0, 1) == pytest.approx(1.0)


def test_geometric_cdf_monotone():
    p = 0.2
    vals = [geometric_cdf(p, k) for k in range(1, 20)]
    assert all(vals[i] < vals[i+1] for i in range(len(vals)-1))


def test_first_success_mean():
    r = first_success_stats(0.1)
    assert r["mean_trials"] == pytest.approx(10.0)


def test_prob_success_within_n():
    # p=1 -> always succeeds on trial 1
    assert prob_success_within_n(1.0, 1) == pytest.approx(1.0)


def test_order_stat_minimum():
    r = order_statistic_expected(1, 9)
    assert r["expected_value"] == pytest.approx(0.1)  # 1/(9+1)


def test_order_stat_maximum():
    r = order_statistic_expected(9, 9)
    assert r["expected_value"] == pytest.approx(0.9)  # 9/10


def test_min_max_expected_symmetry():
    mm = min_max_expected(4)
    assert mm["expected_min"] + mm["expected_max"] == pytest.approx(1.0)


def test_order_stat_invalid():
    with pytest.raises(ValueError):
        order_statistic_expected(5, 3)


def test_combinations_basic():
    assert combinations(5, 2) == 10


def test_combinations_edge():
    assert combinations(5, 0) == 1
    assert combinations(5, 5) == 1
    assert combinations(5, 6) == 0


def test_permutations_basic():
    assert permutations(5, 2) == 20


def test_binomial_pmf_sums_to_1():
    n, p = 10, 0.4
    total = sum(binomial_pmf(n, k, p) for k in range(n+1))
    assert total == pytest.approx(1.0, abs=1e-10)


def test_binomial_mean():
    r = binomial_mean_var(10, 0.3)
    assert r["mean"] == pytest.approx(3.0)


def test_birthday_23():
    # Classic result: 23 people -> P > 0.5
    assert birthday_collision_prob(23) > 0.5


def test_birthday_2_low():
    assert birthday_collision_prob(2) == pytest.approx(1/365, rel=0.01)


def test_birthday_overflow():
    assert birthday_collision_prob(366) == 1.0


def test_probability_sympy_5_count():
    eqs = probability_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
