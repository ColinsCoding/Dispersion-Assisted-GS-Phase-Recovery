"""Tests for dgs/wien_wave_502.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.wien_wave_502 import (
    wien_root_symbolic, wien_condition_symbolic, wien_constant, scale_peak_symbolic, scale_peak,
    peak_numpy, peak_torch, options_table, wave_parameters, field, polarizer_projection,
    unpolarized_through_polarizer, ellipse_polygon, shoelace_area, ellipse_area_symbolic,
    IEEE_CITATION,
)


def test_lambert_w_root_and_constant():
    root, x = wien_root_symbolic()
    assert float(sp.N(root)) == pytest.approx(4.965114, abs=1e-6)
    cond, xs = wien_condition_symbolic()
    assert abs(float(cond.subs(xs, sp.N(root, 30)))) < 1e-20
    assert wien_constant() == pytest.approx(2.897772e-3, rel=1e-6)


def test_scaling_answer():
    l2, (l1, T1, T2) = scale_peak_symbolic()
    assert sp.simplify(l2 - l1 * T1 / T2) == 0
    assert scale_peak(502, 5780, 7000) == pytest.approx(414.51, abs=0.01)
    with pytest.raises(ValueError):
        scale_peak(502, 0, 7000)


def test_numeric_peaks_agree():
    for T in (5780.0, 7000.0):
        exact = wien_constant() / T * 1e9
        assert peak_numpy(T) == pytest.approx(exact, abs=0.02)
        assert peak_torch(T, device="cpu")[0] == pytest.approx(exact, abs=1e-6)


def test_options_table_single_correct():
    df, target = options_table()
    assert list(df.loc[df["verdict"] == "correct", "option (nm)"]) == [414]
    assert df.loc[df["option (nm)"] == 414, "bluer than 502 nm?"].item()


def test_wave_numbers():
    w = wave_parameters()
    assert w["frequency (THz)"] == pytest.approx(597.2, abs=0.1)
    assert w["photon energy (eV)"] == pytest.approx(2.4698, abs=1e-3)
    assert w["E0 (V/m)"] == pytest.approx(1012.6, abs=0.5)
    lam = 502e-9
    assert field(lam, 0.0) == pytest.approx(field(0.0, 0.0))                 # spatial period
    assert field(0.0, lam / 299792458.0) == pytest.approx(field(0.0, 0.0))   # temporal period


def test_malus_and_unpolarized():
    for th in (0, 30, 45, 60, 90):
        amp, I = polarizer_projection(0, th)
        assert amp == pytest.approx(abs(np.cos(np.radians(th))), abs=1e-12)
        assert I == pytest.approx(np.cos(np.radians(th)) ** 2, abs=1e-12)
    assert unpolarized_through_polarizer() == pytest.approx(0.5, abs=0.005)


def test_polygon_area_converges_to_ellipse():
    a, b, d = 2.0, 1.0, np.pi / 3
    exact = float(ellipse_area_symbolic().subs({sp.Symbol("E_x0", positive=True): a,
                                                sp.Symbol("E_y0", positive=True): b,
                                                sp.Symbol("delta", positive=True): d}))
    assert exact == pytest.approx(np.pi * a * b * np.sin(d))
    errs = [abs(shoelace_area(ellipse_polygon(a, b, d, n)) - exact) for n in (8, 32, 128)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-2
    assert shoelace_area(ellipse_polygon(a, b, 0.0, 64)) == pytest.approx(0.0, abs=1e-12)   # linear
    with pytest.raises(ValueError):
        ellipse_polygon(1, 1, 0, 2)


def test_citation_mentions_book():
    assert "Knight" in IEEE_CITATION and "5th ed." in IEEE_CITATION
