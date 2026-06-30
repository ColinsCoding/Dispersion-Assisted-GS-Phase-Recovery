"""Tests for dgs/integration.py — Riemann, FTC, quadrature, tolerance, physics."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.integration import (
    riemann_sum, riemann_convergence,
    trapezoid_rule, simpsons_rule, gaussian_quadrature,
    ftc_sympy, compare_top_down_bottom_up,
    integral_tolerance, tsdft_fiber_length_tolerance,
    planck_radiation_integral, gaussian_normalization_integral,
    tsdft_power_integral, integration_sympy_5,
)


# ── Riemann sums ──────────────────────────────────────────────────────────────

def test_riemann_left_x2():
    # int_0^1 x^2 dx = 1/3
    res = riemann_sum(lambda x: x**2, 0, 1, 10000, "left")
    assert abs(res["approx"] - 1/3) < 1e-3


def test_riemann_midpoint_x2():
    res = riemann_sum(lambda x: x**2, 0, 1, 100, "midpoint")
    assert abs(res["approx"] - 1/3) < 1e-4


def test_riemann_midpoint_better_than_left():
    n = 50
    left = riemann_sum(lambda x: x**2, 0, 1, n, "left")
    mid  = riemann_sum(lambda x: x**2, 0, 1, n, "midpoint")
    assert abs(mid["approx"] - 1/3) < abs(left["approx"] - 1/3)


def test_riemann_sin():
    # int_0^pi sin(x) dx = 2
    res = riemann_sum(np.sin, 0, np.pi, 10000, "midpoint")
    assert abs(res["approx"] - 2.0) < 1e-4


def test_riemann_convergence_improves():
    res = riemann_convergence(lambda x: x**2, 0, 1, 1/3, [10, 100, 1000])
    errors = [res["results"][f"midpoint_n{n}"]["error"] for n in [10, 100, 1000]]
    assert errors[0] > errors[1] > errors[2]


# ── Higher-order quadrature ───────────────────────────────────────────────────

def test_trapezoid_x2():
    res = trapezoid_rule(lambda x: x**2, 0, 1, 1000)
    assert abs(res["approx"] - 1/3) < 1e-6


def test_simpsons_x3():
    # int_0^1 x^3 dx = 1/4; Simpson's is exact for cubics
    res = simpsons_rule(lambda x: x**3, 0, 1, 4)
    assert abs(res["approx"] - 1/4) < 1e-10


def test_simpsons_better_than_trapezoid():
    f = lambda x: np.sin(x)
    exact = 2.0
    trap = trapezoid_rule(f, 0, np.pi, 10)
    simp = simpsons_rule(f, 0, np.pi, 10)
    assert abs(simp["approx"] - exact) < abs(trap["approx"] - exact)


def test_gauss_legendre_polynomial():
    # int_0^1 x^5 dx = 1/6; GL n=3 exact for deg 5
    res = gaussian_quadrature(lambda x: x**5, 0, 1, n_points=3)
    assert abs(res["approx"] - 1/6) < 1e-10


def test_gauss_legendre_exp():
    # int_0^1 exp(x) dx = e-1
    res = gaussian_quadrature(np.exp, 0, 1, n_points=5)
    assert abs(res["approx"] - (np.e - 1)) < 1e-8


# ── FTC (top-down) ───────────────────────────────────────────────────────────

def test_ftc_sin():
    x = sp.Symbol("x")
    res = ftc_sympy(sp.sin(x), x, 0, sp.pi)
    assert abs(res["exact"] - 2.0) < 1e-10
    assert res["agreement"] is True


def test_ftc_polynomial():
    x = sp.Symbol("x")
    res = ftc_sympy(x**3, x, 0, 1)
    assert abs(res["exact"] - 0.25) < 1e-10


def test_ftc_antiderivative_sin():
    x = sp.Symbol("x")
    res = ftc_sympy(sp.sin(x), x, 0, sp.pi)
    assert res["F"] == -sp.cos(x)


def test_compare_top_down_bottom_up_agreement():
    x = sp.Symbol("x")
    cmp = compare_top_down_bottom_up(x**2, x, 0, 1)
    assert abs(cmp["exact"] - 1/3) < 1e-10
    # Simpson's with n=1000 should be essentially exact
    row = cmp["rows"][-1]   # largest n
    assert row["simp_err"] < 1e-10


# ── Tolerance ────────────────────────────────────────────────────────────────

def test_integral_tolerance_statistical():
    x  = np.linspace(0, 1, 1000)
    f  = np.ones_like(x)
    df = 0.01 * np.ones_like(x)
    res = integral_tolerance(f, x, df, mode="statistical")
    assert res["I"] > 0
    assert res["delta_I"] > 0
    assert res["SNR"] > 1


def test_integral_tolerance_worst_case_larger():
    x  = np.linspace(0, 1, 1000)
    f  = np.ones_like(x)
    df = 0.01 * np.ones_like(x)
    stat = integral_tolerance(f, x, df, mode="statistical")
    worst = integral_tolerance(f, x, df, mode="worst_case")
    # Worst case always >= statistical
    assert worst["delta_I"] >= stat["delta_I"]


def test_fiber_tolerance_wavelength_error():
    res = tsdft_fiber_length_tolerance(L_km=10.0, delta_L_m=0.5)
    # 0.5m / 10000m = 5e-5 relative; at 1550nm -> 0.0775 nm
    assert abs(res["delta_lambda_nm"] - 1550 * 5e-5) < 0.001


def test_fiber_tolerance_rel_error_ppm():
    res = tsdft_fiber_length_tolerance(L_km=10.0, delta_L_m=0.5)
    assert abs(res["rel_error_ppm"] - 50.0) < 0.1   # 50 ppm


# ── Physics integrals ─────────────────────────────────────────────────────────

def test_planck_stefan_boltzmann():
    # Test that Stefan-Boltzmann check passes (total power agrees to <5%)
    pl = planck_radiation_integral(5778)
    assert pl["error_pct"] < 5.0


def test_planck_visible_fraction():
    pl = planck_radiation_integral(5778)
    assert 0.2 < pl["vis_fraction"] < 0.6   # sun: ~37% visible


def test_gaussian_normalization_1d():
    g = gaussian_normalization_integral()
    assert abs(g["I_1d_numerical"] - np.sqrt(np.pi)) < 1e-5
    assert g["error_1d_pct"] < 0.001


def test_gaussian_normalization_2d():
    g = gaussian_normalization_integral()
    assert abs(g["I_2d_numerical"] - np.pi) < 0.01
    assert g["error_2d_pct"] < 0.5


def test_tsdft_power_error():
    ts = tsdft_power_integral(n_pts=2048)
    assert ts["error_pct"] < 1.0


def test_tsdft_power_adc_counts_positive():
    ts = tsdft_power_integral()
    assert ts["P_counts"] > 0
    assert ts["ADC_bits"] == 12


# ── SymPy ─────────────────────────────────────────────────────────────────────

def test_sympy_5_count():
    eqs = integration_sympy_5()
    assert len(eqs) == 5


def test_sympy_ftc_form():
    eqs = integration_sympy_5()
    eq = eqs["FTC_top_down"]
    assert isinstance(eq, sp.Eq)
    # FTC RHS should be F(b) - F(a)
    assert isinstance(eq.rhs, sp.Add) or eq.rhs != eq.lhs


def test_sympy_gaussian_exact():
    eqs = integration_sympy_5()
    eq = eqs["Gaussian_integral"]
    assert sp.simplify(eq.rhs - sp.sqrt(sp.pi)) == 0
