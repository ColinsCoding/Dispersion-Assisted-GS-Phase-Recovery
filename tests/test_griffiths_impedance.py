"""Tests for griffiths/impedance.py."""
import numpy as np
import pytest
import sympy as sp

from griffiths.impedance import (
    w, R, L, C, CONVENTIONS, Z0_VACUUM,
    derive_element_impedances, reactance, series_rlc, series_rlc_numeric,
    time_domain_check, complex_power, reactive_power_energy_check,
    wave_impedance_symbolic, griffiths_reflection, conductor_impedance,
    skin_depth, surface_impedance_good_conductor, input_impedance_symbolic,
    film_reflectance_impedance, film_reflectance_matrix,
)


def test_element_impedances_both_conventions():
    ee = derive_element_impedances("ee")
    ph = derive_element_impedances("physics")
    assert ee == {"R": R, "L": sp.I * w * L, "C": -sp.I / (w * C)}
    for k in "RLC":
        assert sp.simplify(ph[k] - sp.conjugate(ee[k])) == 0


@pytest.mark.parametrize("conv", CONVENTIONS)
def test_reactance_sign_is_convention_independent(conv):
    Z = derive_element_impedances(conv)
    assert reactance(Z["L"], conv) == w * L
    assert sp.simplify(reactance(Z["C"], conv) + 1 / (w * C)) == 0
    assert reactance(Z["R"], conv) == 0


def test_bad_convention():
    with pytest.raises(ValueError):
        derive_element_impedances("engineering")


def test_series_rlc_symbolic():
    s = series_rlc()
    assert sp.simplify(s["w0"] - 1 / sp.sqrt(L * C)) == 0
    assert sp.simplify(s["X"].subs(w, s["w0"])) == 0
    assert sp.simplify(s["Q"] - sp.sqrt(L / C) / R) == 0


@pytest.mark.parametrize("omega", [1.0e4, 3.16227766e4, 1.0e5])
def test_phasor_matches_ode(omega):
    r = time_domain_check(10.0, 1e-3, 1e-6, 1.0, omega)
    assert r["ode_amp"] == pytest.approx(r["phasor_amp"], rel=1e-6)
    assert r["ode_phase"] == pytest.approx(r["phasor_phase"], abs=1e-6)


def test_reactive_power_is_stored_energy():
    for omega in (2e4, 4e4):
        Q, Q_e, P, P_r = reactive_power_energy_check(10.0, 1e-3, 1e-6, 1.0, omega)
        assert Q_e == pytest.approx(Q, rel=1e-6)
        assert P_r == pytest.approx(P, rel=1e-6)


def test_complex_power_resistor():
    S = complex_power(10.0, 2.0)
    assert S == pytest.approx(10.0)


def test_numeric_validation():
    with pytest.raises(ValueError):
        series_rlc_numeric(0, 1, 1, 1.0)
    with pytest.raises(ValueError):
        series_rlc_numeric(1, 1, 1, -1.0)


def test_wave_impedance_and_reflection():
    Z, muv = wave_impedance_symbolic()
    assert sp.simplify(Z - muv) == 0
    g = griffiths_reflection()
    assert g["R+T"] == 1 and g["matches_line_formula"]
    assert Z0_VACUUM == pytest.approx(376.73, abs=0.01)


def test_conductor_limits():
    sigma, om = 5.8e7, 2 * np.pi * 1e9                     # copper, 1 GHz
    Z = conductor_impedance(sigma, om)
    Zs = surface_impedance_good_conductor(sigma, om)
    assert Z == pytest.approx(Zs, rel=1e-6)
    assert Zs.real == pytest.approx(Zs.imag)               # 45 deg, inductive
    assert skin_depth(sigma, om) == pytest.approx(2.09e-6, rel=0.01)
    assert conductor_impedance(0.0, om) == pytest.approx(Z0_VACUUM)


def test_quarter_wave_transformer():
    closed, quarter = input_impedance_symbolic()
    Z0, ZL = sp.symbols("Z_0 Z_L", positive=True)
    assert sp.simplify(quarter - Z0 ** 2 / ZL) == 0
    th = sp.Symbol("theta", positive=True)
    assert sp.simplify(closed.subs(th, 0) - ZL) == 0


def test_film_two_methods_agree_and_ar_coating():
    lam = np.linspace(400e-9, 900e-9, 101)
    n = np.sqrt(1.5)
    d = 550e-9 / (4 * n)
    a = film_reflectance_impedance(lam, n, d)
    b = film_reflectance_matrix(lam, n, d)
    assert np.allclose(a, b, atol=1e-12)
    assert float(film_reflectance_impedance(550e-9, n, d)) < 1e-20
    bare = ((1 - 1.5) / (1 + 1.5)) ** 2
    assert np.all(a <= bare + 1e-12)
    # zero-thickness film == bare substrate
    assert float(film_reflectance_matrix(550e-9, 1.38, 0.0)) == pytest.approx(bare)
