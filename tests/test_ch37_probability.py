"""Tests for dgs/ch37_probability.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.ch37_probability import (
    theta, bmax, rutherford_theta_pdf, rutherford_from_cross_section, theta_min,
    tail_probability, sample_theta, alphas_needed, poisson_at_least_one_symbolic,
    poisson_mean_var_symbolic, millikan_relative_sigma_symbolic, millikan_relative_sigma,
    millikan_mc_relative_sigma, misassignment_probability, misassignment_symbolic,
)


def test_change_of_variables_equals_rutherford():
    assert sp.simplify(rutherford_theta_pdf() - rutherford_from_cross_section()) == 0


def test_pdf_normalised():
    B = 20.0
    total = sp.integrate(rutherford_theta_pdf().subs(bmax, B), (theta, theta_min(B), sp.pi))
    assert float(total) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("T_deg", [30, 90, 150])
def test_tail_probability_monte_carlo(T_deg):
    B, n = 20.0, 1_000_000
    th = sample_theta(n, B, seed=T_deg)
    p = float(tail_probability(np.radians(T_deg), B))
    assert abs(np.mean(th > np.radians(T_deg)) - p) < 5 * np.sqrt(p / n)


def test_poisson_symbolic_and_rule_of_three():
    lam = sp.Symbol("lambda", positive=True)
    assert sp.simplify(poisson_at_least_one_symbolic() - (1 - sp.exp(-lam))) == 0
    assert poisson_mean_var_symbolic() == (lam, lam)
    p = 1 / 8000
    n = alphas_needed(p)
    assert 1 - np.exp(-n * p) >= 0.95 > 1 - np.exp(-(n - 1) * p)
    assert n == pytest.approx(3 / p, rel=0.01)


def test_millikan_propagation():
    rel = millikan_relative_sigma_symbolic()
    v, V, sv, sV = sp.symbols("v V sigma_v sigma_V", positive=True)
    val = float(rel.subs({v: 1, V: 1, sv: 0.02, sV: 0.005}))
    assert val == pytest.approx(millikan_relative_sigma(0.02, 0.005))
    assert millikan_mc_relative_sigma(0.02, 0.005) == pytest.approx(val, rel=0.02)


def test_misassignment():
    p = misassignment_probability(np.arange(1, 20), 0.03)
    assert np.all(np.diff(p) > 0)
    s = sp.Symbol("s", positive=True)
    assert float(misassignment_symbolic().subs(s, 0.1)) == pytest.approx(
        float(misassignment_probability(1, 0.1)), rel=1e-12)
    with pytest.raises(ValueError):
        misassignment_probability(0, 0.03)
