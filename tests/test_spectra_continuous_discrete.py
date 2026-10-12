"""Tests for dgs/spectra_continuous_discrete.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.spectra_continuous_discrete import (
    bose_integral_by_series, bose_integral_numeric, stefan_boltzmann_symbolic, sigma_numeric,
    planck_lambda, hydrogen_level_eV, hydrogen_line_nm, series_limit_symbolic,
    duane_hunt_cutoff_nm, kramers_spectrum, moseley_kalpha_keV, OPTIONS, check_option, R_INF,
)


def test_bose_integral_three_ways():
    term, total = bose_integral_by_series()
    n = sp.Symbol("n", integer=True, positive=True)
    assert sp.simplify(term - 6 / n ** 4) == 0 and total == sp.pi ** 4 / 15
    assert bose_integral_numeric() == pytest.approx(np.pi ** 4 / 15, rel=1e-10)


def test_stefan_boltzmann_constant():
    assert sigma_numeric() == pytest.approx(5.670374419e-8, rel=1e-9)
    T = 5780.0
    lam = np.linspace(1, 200_000, 2_000_001)
    total = np.pi * np.trapezoid(planck_lambda(lam, T), lam * 1e-9)       # pi * int B_lambda = sigma T^4
    assert total == pytest.approx(sigma_numeric() * T ** 4, rel=1e-3)


def test_hydrogen_discrete():
    assert hydrogen_level_eV(1) == pytest.approx(-13.6057, abs=1e-3)
    assert [round(hydrogen_line_nm(n, 2), 1) for n in (3, 4)] == [656.1, 486.0]
    lim, _ = series_limit_symbolic(2)
    R = sp.Symbol("R", positive=True)
    assert lim == 4 / R and hydrogen_line_nm(10_000, 2) == pytest.approx(4 / R_INF * 1e9, rel=1e-6)
    with pytest.raises(ValueError):
        hydrogen_line_nm(2, 3)


def test_xray_tube_has_both():
    lmin = duane_hunt_cutoff_nm(30e3)
    assert lmin == pytest.approx(0.04133, rel=1e-3)
    lam = np.linspace(0.01, 0.3, 3000)
    I = kramers_spectrum(lam, 30e3)
    assert np.all(I[lam <= lmin] == 0) and np.count_nonzero(I) > 2500       # continuous above cutoff
    assert moseley_kalpha_keV(29) == pytest.approx(8.0, abs=0.1)            # Cu K-alpha (meas. 8.05)


def test_only_fourth_option_correct():
    verdicts = [check_option(o)[0] for o in OPTIONS]
    assert verdicts == [False, False, False, True]
    assert any("electrons" in r for r in check_option(OPTIONS[0])[1])
    assert any("half true" in r for r in check_option(OPTIONS[1])[1])
