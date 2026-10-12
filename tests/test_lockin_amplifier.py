"""Tests for dgs.lockin_amplifier."""
import numpy as np
import pytest
import sympy as sp

from dgs import lockin_amplifier as L


def test_demodulation_symbolic():
    X, Y, R, th = L.demod_symbolic()
    A, phi = sp.symbols("A phi", positive=True)
    assert sp.simplify(X - A * sp.cos(phi)) == 0 and sp.simplify(Y - A * sp.sin(phi)) == 0
    assert sp.simplify(R - A) == 0
    assert L.mixer_products_symbolic()[1] == 0
    assert L.off_frequency_rejection_symbolic()[1] == 0


def test_square_wave_reference_odd_harmonics():
    c = dict(L.square_wave_reference_symbolic(3))
    assert c == {1: 4 / sp.pi, 3: 4 / (3 * sp.pi), 5: 4 / (5 * sp.pi)}


def test_enbw_symbolic_matches_closed_form():
    tau = sp.symbols("tau", positive=True)
    for n, expect in zip((1, 2, 3, 4), (1 / (4 * tau), 1 / (8 * tau), 3 / (32 * tau), 5 / (64 * tau))):
        assert sp.simplify(L.enbw_symbolic(n) - expect) == 0
        assert L.enbw(n, 1.0) == pytest.approx(float(expect.subs(tau, 1)))
    assert L.settle_time(1, 1.0) == pytest.approx(np.log(100), rel=1e-6)
    with pytest.raises(ValueError):
        L.enbw(0, 1.0)


def test_digital_lockin_recovers_amplitude_and_phase():
    fs, f0 = 50000.0, 1234.0
    t = np.arange(int(2 * fs)) / fs
    sig = 0.3 * np.cos(2 * np.pi * f0 * t + np.deg2rad(40)) + 0.5 * np.random.default_rng(0).normal(size=len(t))
    X, Y, R, th = L.lockin(sig, t, f0, tau=0.05, order=4)
    k = int(0.5 * fs)
    assert R[k:].mean() == pytest.approx(0.3, rel=0.03)
    assert th[k:].mean() == pytest.approx(40, abs=3)


def test_filter_implementations_agree():
    x = np.random.default_rng(1).normal(size=2000)
    assert np.allclose(L.lowpass(x, 1e-3, 0.01, 2), L.lowpass_fast(x, 1e-3, 0.01, 2))


def test_lockin_beats_dc_under_flicker():
    r = L.dc_vs_lockin_demo()
    assert r["lockin_std"] < r["dc_std"] / 10
    assert r["lockin_mean"] == pytest.approx(r["true"], abs=3 * r["lockin_std"])


def test_fm_first_harmonic_and_optimal_depth():
    d = sp.symbols("delta", real=True)
    fc = sp.symbols("f_c", real=True)
    S = sp.Function("S")
    expect = d * sp.diff(S(fc), fc) + d**3 * sp.diff(S(fc), fc, 3) / 8
    assert sp.simplify(L.fm_first_harmonic_symbolic() - expect) == 0
    C, G = 0.03, 8e6
    dopt = L.optimal_modulation_depth(C, G)
    assert dopt / G == pytest.approx(1 / (2 * np.sqrt(2)), rel=1e-3)
    assert abs(L.zero_crossing_slope(dopt, C, G)) == pytest.approx(8 * C / (3 * np.sqrt(3) * G), rel=1e-4)
    # small depth: slope -> delta * S''(0)
    small = 1e-3 * G
    assert L.zero_crossing_slope(small, C, G) == pytest.approx(small * 8 * C / G**2, rel=1e-3)
    # error signal is odd in detuning with a zero at resonance
    e = L.fm_error_signal([-1e6, 0.0, 1e6], dopt, C, G)
    assert abs(e[1]) < 1e-12 and e[0] == pytest.approx(-e[2])


def test_tracking_noise_matches_prediction():
    t = np.arange(0, 0.5, 1e-4)
    B = np.full_like(t, 1e-4)
    for g in (0.3, 0.03):
        est = L.track_field(B, t, photon_rate=1e13, gain=g)
        assert np.std(est[2000:] - B[2000:]) == pytest.approx(
            L.predicted_tracking_noise(photon_rate=1e13, gain=g), rel=0.1)
