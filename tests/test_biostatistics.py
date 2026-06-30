"""Tests for dgs/biostatistics.py."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.biostatistics import (
    exponential_survival, weibull_survival,
    kaplan_meier, log_rank_test,
    species_extinction_model, biostatistics_sympy_5,
)


def test_exp_survival_at_mean():
    t = np.array([10.0])
    res = exponential_survival(t, lam=0.1)
    assert abs(res["S"][0] - np.exp(-1)) < 1e-10


def test_exp_survival_starts_at_one():
    t = np.array([0.0])
    res = exponential_survival(t, lam=0.5)
    assert abs(res["S"][0] - 1.0) < 1e-10


def test_exp_survival_decreasing():
    t = np.linspace(0, 20, 100)
    res = exponential_survival(t, lam=0.2)
    assert np.all(np.diff(res["S"]) <= 0)


def test_exp_constant_hazard():
    t = np.linspace(0.1, 10, 50)
    res = exponential_survival(t, lam=0.3)
    np.testing.assert_allclose(res["h"], 0.3, rtol=1e-10)


def test_exp_median():
    res = exponential_survival(np.array([0.0]), lam=1.0)
    assert abs(res["median"] - np.log(2)) < 1e-10


def test_weibull_beta1_is_exponential():
    t = np.linspace(0.1, 10, 100)
    lam = 0.5
    exp_s = exponential_survival(t, lam)["S"]
    wb_s  = weibull_survival(t, eta=1/lam, beta=1.0)["S"]
    np.testing.assert_allclose(wb_s, exp_s, rtol=1e-5)


def test_weibull_increasing_hazard():
    t = np.linspace(0.1, 10, 100)
    wb = weibull_survival(t, eta=10.0, beta=2.0)
    assert wb["regime"] == "aging_wearout"
    # Hazard should be increasing
    assert np.all(np.diff(wb["h"]) >= 0)


def test_weibull_decreasing_hazard():
    t = np.linspace(0.1, 10, 100)
    wb = weibull_survival(t, eta=10.0, beta=0.5)
    assert wb["regime"] == "infant_mortality"
    assert np.all(np.diff(wb["h"]) <= 0)


def test_km_no_censoring():
    # All events observed: KM should reach 0
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    km = kaplan_meier(t)
    assert km["n_censored"] == 0
    assert km["S"][-1] == 0.0


def test_km_starts_at_one():
    t = np.array([2.0, 5.0, 8.0])
    km = kaplan_meier(t)
    assert km["S"][0] == 1.0


def test_km_decreasing():
    rng = np.random.default_rng(0)
    t = rng.exponential(5, 20)
    km = kaplan_meier(t)
    assert np.all(np.diff(km["S"]) <= 0)


def test_km_ci_contains_estimate():
    rng = np.random.default_rng(1)
    t = rng.exponential(8, 30)
    c = rng.random(30) < 0.2
    km = kaplan_meier(t, c)
    # CI should contain the KM estimate
    assert np.all(km["CI_lower"] <= km["S"] + 1e-6)
    assert np.all(km["CI_upper"] >= km["S"] - 1e-6)


def test_km_median():
    # Exponential lam=0.1: median = ln(2)/0.1 = 6.93
    rng = np.random.default_rng(42)
    t = rng.exponential(10, 500)   # large sample for accuracy
    km = kaplan_meier(t)
    assert abs(km["median_t"] - np.log(2)*10) < 1.5


def test_log_rank_different_groups():
    rng = np.random.default_rng(5)
    t1 = rng.exponential(10, 50);  c1 = np.zeros(50, bool)
    t2 = rng.exponential(3, 50);   c2 = np.zeros(50, bool)
    lr = log_rank_test(t1, c1, t2, c2)
    # Very different groups -> should reject H0
    assert bool(lr["reject_H0"]) is True
    assert lr["chi2"] > 3.84   # 95th percentile of chi2(1)


def test_log_rank_same_group():
    rng = np.random.default_rng(7)
    t1 = rng.exponential(5, 40);  c1 = np.zeros(40, bool)
    t2 = rng.exponential(5, 40);  c2 = np.zeros(40, bool)
    lr = log_rank_test(t1, c1, t2, c2)
    # Same distribution -> should not reject (p > 0.05 usually)
    assert lr["p_value"] > 0.01   # very unlikely to reject with same lambda


def test_extinction_event():
    ext = species_extinction_model(n_species=300, extinction_multiplier=20,
                                    seed=0)
    # With 20x hazard: most should go extinct
    assert ext["pct_extinct_total"] > 50.0
    assert ext["S_at_event"] > ext["S_true"][-1]


def test_extinction_survival_monotone():
    ext = species_extinction_model()
    assert np.all(np.diff(ext["S_true"]) <= 0)


def test_biostatistics_sympy_5():
    eqs = biostatistics_sympy_5()
    assert len(eqs) == 5
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic)
