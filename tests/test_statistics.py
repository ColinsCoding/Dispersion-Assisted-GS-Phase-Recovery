"""Tests for dgs/statistics.py."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.statistics import (
    gaussian_pdf, exponential_pdf, poisson_pmf, binomial_pmf,
    clt_demo, z_test, t_test_one_sample,
    bayesian_update_beta, bayesian_gaussian_update,
    attention_span_model, shannon_information,
    triangle_inequality_demo, statistics_sympy_5,
)


def test_gaussian_peak():
    x = np.array([0.0])
    res = gaussian_pdf(x, mu=0, sigma=1)
    assert abs(res["pdf"][0] - 1/np.sqrt(2*np.pi)) < 1e-8


def test_gaussian_cdf_at_mean():
    x = np.array([0.0])
    res = gaussian_pdf(x, mu=0, sigma=1)
    assert abs(res["cdf"][0] - 0.5) < 1e-4


def test_gaussian_integrates_to_1():
    x = np.linspace(-6, 6, 2000)
    res = gaussian_pdf(x)
    assert abs(np.trapezoid(res["pdf"], x) - 1.0) < 1e-4


def test_exponential_mean():
    t = np.linspace(0, 100, 5000)
    res = exponential_pdf(t, lam=0.5)
    computed_mean = np.trapezoid(t * res["pdf"], t)
    assert abs(computed_mean - 2.0) < 0.05   # mean = 1/lambda = 2


def test_exponential_memoryless():
    # P(T > 10) = exp(-lambda*10)
    t = np.array([10.0])
    res = exponential_pdf(t, lam=0.1)
    assert abs(res["survival"][0] - np.exp(-1.0)) < 1e-10


def test_poisson_sum_to_1():
    k = np.arange(0, 40)
    res = poisson_pmf(k, lam=5.0)
    assert abs(res["pmf"].sum() - 1.0) < 1e-6


def test_poisson_mode():
    k = np.arange(0, 30)
    res = poisson_pmf(k, lam=8.0)
    # Mode is at floor(lambda) = 8
    assert res["k"][np.argmax(res["pmf"])] == 8


def test_binomial_sum():
    k = np.arange(0, 11)
    res = binomial_pmf(k, n=10, p=0.5)
    assert abs(res["pmf"].sum() - 1.0) < 1e-8


def test_clt_convergence():
    res = clt_demo("exponential", lam=1.0, sample_sizes=[100])
    assert res["results"][100]["converged"] is True
    assert res["results"][100]["frac_2sigma"] > 0.93


def test_z_test_reject():
    res = z_test(x_bar=3.0, mu0=0.0, sigma=1.0, n=1)
    assert bool(res["reject_H0_05"]) is True


def test_z_test_fail_to_reject():
    res = z_test(x_bar=0.5, mu0=0.0, sigma=1.0, n=1)
    assert bool(res["reject_H0_05"]) is False


def test_z_test_ci():
    res = z_test(x_bar=0.0, mu0=0.0, sigma=1.0, n=100)
    assert res["CI_95_lower"] < 0.0 < res["CI_95_upper"]


def test_t_test_basic():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(50) + 3.0   # mean = 3
    res = t_test_one_sample(data, mu0=0.0)
    assert bool(res["reject_H0"]) is True   # mean=3 >> 0


def test_bayesian_beta_update():
    res = bayesian_update_beta(7, 3, prior_alpha=1, prior_beta=1)
    # Posterior: Beta(8, 4) -> mean = 8/12 = 0.667
    assert abs(res["posterior_mean"] - 8/12) < 1e-6


def test_bayesian_beta_ci_contains_mean():
    res = bayesian_update_beta(5, 5)
    assert res["CI_95_lower"] < res["posterior_mean"] < res["CI_95_upper"]


def test_bayesian_gaussian_update():
    rng = np.random.default_rng(1)
    data = rng.standard_normal(100) + 5.0
    res = bayesian_gaussian_update(data, prior_mu=0.0, prior_sigma=10.0,
                                    likelihood_sigma=1.0)
    # Posterior mean should be close to 5.0
    assert abs(res["post_mu"] - 5.0) < 0.5


def test_attention_span_half_life():
    res = attention_span_model(tau_s=20.0)
    assert abs(res["t_half_s"] - 20*np.log(2)) < 1e-6


def test_attention_span_survival():
    res = attention_span_model(tau_s=10.0, t_max_s=50.0)
    # At t=tau: survival = 1/e
    idx = np.argmin(np.abs(res["t_s"] - 10.0))
    assert abs(res["survival"][idx] - np.exp(-1)) < 0.01


def test_shannon_entropy_uniform():
    p = np.ones(4) / 4
    res = shannon_information(p, base=2)
    assert abs(res["entropy"] - 2.0) < 1e-10   # log2(4) = 2 bits


def test_shannon_entropy_pure():
    p = np.array([1.0, 0.0, 0.0])
    res = shannon_information(p)
    assert res["entropy"] < 1e-6   # pure state: entropy = 0


def test_triangle_inequality_L2():
    x = np.linspace(0, 2*np.pi, 500)
    res = triangle_inequality_demo(np.sin(x), np.cos(x), x)
    assert res["triangle_satisfied"] is True
    assert res["cauchy_schwarz"] is True


def test_triangle_inequality_norms():
    x = np.linspace(0, np.pi, 300)
    f, g = np.sin(x), np.ones_like(x)
    res = triangle_inequality_demo(f, g, x)
    assert abs(res["norm_f"] - np.sqrt(np.trapezoid(f**2, x))) < 1e-6


def test_statistics_sympy_5():
    eqs = statistics_sympy_5()
    assert len(eqs) == 5
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic)
