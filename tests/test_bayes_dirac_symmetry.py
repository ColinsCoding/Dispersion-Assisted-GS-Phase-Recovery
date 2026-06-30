import numpy as np
import pytest
from dgs.bayes_dirac_symmetry import (
    bayes_theorem, bayes_two_hypothesis, bayes_sympy,
    dirac_delta_as_gaussian_limit, sifting_property_numeric,
    dirac_delta_sympy_sifting, dirac_comb,
    classify_symmetry, symmetry_integral_shortcut, symmetry_sympy_5,
)


def test_bayes_theorem_basic():
    res = bayes_theorem(prior=0.5, likelihood=0.8, evidence=0.5)
    assert res["posterior"] == pytest.approx(0.8)


def test_bayes_theorem_invalid_prior_raises():
    with pytest.raises(ValueError):
        bayes_theorem(prior=1.5, likelihood=0.5, evidence=0.5)


def test_bayes_theorem_zero_evidence_raises():
    with pytest.raises(ValueError):
        bayes_theorem(prior=0.5, likelihood=0.5, evidence=0.0)


def test_bayes_two_hypothesis_sums_to_one():
    res = bayes_two_hypothesis(0.5, 0.9, 0.2)
    assert res["posterior_H1"] + res["posterior_H2"] == pytest.approx(1.0)


def test_bayes_two_hypothesis_favors_better_likelihood():
    res = bayes_two_hypothesis(0.5, 0.9, 0.1)
    assert res["posterior_H1"] > res["posterior_H2"]


def test_bayes_sympy_is_equation():
    import sympy as sp
    eq = bayes_sympy()
    assert isinstance(eq, sp.Eq)


def test_dirac_delta_gaussian_peak_at_a():
    x = np.array([0.0, 1.0, 2.0])
    vals = dirac_delta_as_gaussian_limit(x, a=1.0, sigma=0.1)
    assert np.argmax(vals) == 1


def test_dirac_delta_gaussian_integrates_to_one():
    x = np.linspace(-10, 10, 200001)
    vals = dirac_delta_as_gaussian_limit(x, a=0.0, sigma=0.5)
    integral = np.trapezoid(vals, x)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_sifting_property_numeric_matches_f_at_a():
    res = sifting_property_numeric(lambda x: x**2 + 1, a=2.0, x_range=(-5, 5))
    assert res["abs_error"] < 1e-3


def test_dirac_delta_sympy_sifting_structure():
    import sympy as sp
    eq = dirac_delta_sympy_sifting()
    assert isinstance(eq, sp.Eq)


def test_dirac_comb_has_n_peaks():
    x = np.linspace(-5, 5, 100001)
    comb = dirac_comb(x, spacing=1.0, n_teeth=3, sigma=0.02)
    # 7 teeth total (k = -3..3); count local peaks above half-max
    threshold = comb.max() * 0.5
    above = comb > threshold
    crossings = np.sum(np.diff(above.astype(int)) == 1)
    assert crossings == 7


def test_classify_symmetry_even():
    assert classify_symmetry(lambda x: x**2)["kind"] == "even"


def test_classify_symmetry_odd():
    assert classify_symmetry(lambda x: x**3)["kind"] == "odd"


def test_classify_symmetry_neither():
    assert classify_symmetry(lambda x: x**2 + x)["kind"] == "neither"


def test_symmetry_shortcut_odd_is_zero():
    res = symmetry_integral_shortcut(lambda x: x**3, L=4.0)
    assert res["integral"] == 0.0
    assert res["kind"] == "odd"


def test_symmetry_shortcut_even_doubles_half():
    res = symmetry_integral_shortcut(lambda x: x**2, L=2.0)
    expected = 2 * (2**3 / 3)  # integral of x^2 from 0 to 2 = 8/3
    assert res["integral"] == pytest.approx(expected, rel=1e-3)


def test_symmetry_sympy_5_count_and_type():
    import sympy as sp
    eqs = symmetry_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
