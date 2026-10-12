"""Tests for dgs.frequency_comb."""
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from dgs import frequency_comb as fc


def test_geometric_series_and_peaks():
    S, residual = fc.comb_geometric_sum_symbolic()
    assert residual == 0
    f_n = fc.comb_peak_frequencies_symbolic()
    T, dphi = sp.symbols("T Delta_phi", real=True)
    n = sp.symbols("n", integer=True)
    assert sp.simplify(f_n - (n / T + dphi / (2 * sp.pi * T))) == 0
    assert fc.dirichlet_kernel(np.array([0.0, 2 * np.pi]), 10) == pytest.approx([100, 100])


def test_spectrum_of_simulated_train_has_teeth_at_n_frep_plus_fceo():
    f_rep, dphi, fs = 1e6, 2 * np.pi * 0.3, 200e6
    t, s = fc.simulate_pulse_train(64, f_rep, dphi, fs, f_carrier=20e6, pulse_fwhm_s=50e-9)
    spec = np.abs(np.fft.rfft(s)) ** 2
    f = np.fft.rfftfreq(len(s), 1 / fs)
    peak = f[np.argmax(spec)]
    f_ceo = fc.ceo_from_slip(dphi, f_rep)
    assert f_ceo == pytest.approx(0.3e6)
    n = round((peak - f_ceo) / f_rep)
    assert peak == pytest.approx(n * f_rep + f_ceo, abs=f_rep / 64 * 1.5)


def test_ceo_and_f2f():
    dphi, fceo = fc.ceo_slip_symbolic()
    vg, vp = sp.symbols("v_g v_p", positive=True)
    assert sp.simplify(dphi.subs(vg, vp)) == 0                    # no dispersion -> no slip
    assert fc.f2f_symbolic() == sp.Symbol("f_ceo", positive=True)
    assert fc.ceo_from_slip(2 * np.pi * 1.25, 100.0) == pytest.approx(25.0)


def test_two_repetition_rates_resolve_ambiguity():
    nu = 194.1e12 + 12345.678e6
    survivors, n_single = fc.resolve_with_two_reps(nu, 250e6, 250.01e6, 35e6, nu + 2e9, 5e9)
    assert n_single > 40
    assert survivors == [pytest.approx(nu, abs=1.0)]


def test_exact_clockwork_beats_float():
    nu = "429228004229873.13"
    n = 1716912
    fr_exact = fc.frep_from_optical(nu, n, 35_000_000, Fraction("-30770126.87"))
    assert fr_exact == Fraction(250_000_000)
    # the double nearest to nu is off by 5 mHz: ~1e-17 fractional, the size of an optical clock's uncertainty
    stored_err = Fraction(float(nu)) - Fraction(nu)
    assert stored_err == Fraction(-5, 1000)
    assert abs(float(stored_err) / float(nu)) == pytest.approx(1.165e-17, rel=1e-3)


def test_orthonormal_teeth_and_dimension():
    T = sp.symbols("T", positive=True)
    assert fc.tooth_orthogonality_symbolic(4, 4) == T and fc.tooth_orthogonality_symbolic(4, 7) == 0
    assert np.allclose(fc.tooth_gram_matrix([1, 2, 5, 9], 250e6), np.eye(4), atol=1e-10)
    assert fc.comb_dimension(40e12, 250e6) == 160000
    with pytest.raises(ValueError):
        fc.comb_dimension(0, 1)


def test_minkowski_classes():
    assert fc.minkowski_interval(1e-9, fc.C * 1e-9)[1] == "light-like"
    assert fc.minkowski_interval(1.0, 1.0)[1] == "time-like"
    assert fc.minkowski_interval(1e-9, 10.0)[1] == "space-like"


def test_safety_arithmetic():
    assert fc.pulse_energy(0.1, 250e6) == pytest.approx(4e-10)
    assert fc.peak_power(0.1, 250e6, 100e-15) == pytest.approx(0.8814 * 4e-10 / 100e-15)
    s = fc.safety_summary(0.1, 250e6, 100e-15, 2e-3, 10.0)
    assert s["od_for_given_mpe"] == pytest.approx(np.log10(s["avg_irradiance_W_m2"] / 10.0))
    with pytest.raises(ValueError):
        fc.pulse_energy(0, 1)
