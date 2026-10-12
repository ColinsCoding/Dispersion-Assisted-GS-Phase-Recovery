"""Smoke tests for dgs/laser_doppler_vibrometer.py."""
import numpy as np
import pytest
import sympy as sp

from dgs import laser_doppler_vibrometer as L


def test_symbolic():
    ratio, series = L.mirror_doppler_symbolic()
    beta = sp.Symbol("beta", real=True)
    assert sp.expand(series - (1 + 2 * beta + 2 * beta**2)) == 0
    assert L.fringe_rate_equals_doppler() == 0
    assert L.airy_symbolic()[1] == 0


def test_mirror_shift_first_order():
    assert L.mirror_doppler_shift_hz(1.0) == pytest.approx(2 / 632.8e-9, rel=1e-6)


def test_interference_extremes():
    assert L.michelson_intensity(0.0, 1, 4) == pytest.approx(9.0)
    assert L.michelson_intensity(np.pi, 1, 4) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        L.michelson_intensity(0.0, -1, 1)


@pytest.mark.parametrize("phi0", [0.0, np.pi / 2, 1.0])
def test_harmonics_match_fft(phi0):
    a = L.detector_harmonics(1.7, phi0)[1]
    b = L.fft_harmonics(1.7, phi0)[1]
    assert np.max(np.abs(a - b)) < 1e-12


def test_jacobi_anger_and_null():
    assert L.jacobi_anger_check(2.0) < 1e-12
    z, a = L.bessel_null_amplitudes(1)
    assert z[0] == pytest.approx(2.404825557695773)
    assert a[0] == pytest.approx(121.1e-9, rel=1e-3)


def test_heterodyne_recovers_velocity():
    r = L.heterodyne_ldv(lambda t: 0.5 * np.sin(2 * np.pi * 5e3 * t), snr_db=30)
    assert r["rms_error"] < 0.05
    with pytest.raises(ValueError):
        L.heterodyne_ldv(lambda t: 0 * t, fs_hz=50e6)


def test_airy():
    x, P, A = L.airy_numeric_check()
    assert np.max(np.abs(P - A)) < 0.01
    assert L.rayleigh_factor() == pytest.approx(1.2197, abs=1e-4)


def test_cic_demodulator():
    t, s, v = L.vibration_signal(100000, 0.5)
    errs = []
    for order in (1, 3):
        vr, d = L.cic_demodulate(s, order=order)
        e = vr[int(d):] - v[: len(v) - int(d)]
        errs.append(np.sqrt(np.nanmean(e**2)))
    assert errs[1] < 1e-4 < errs[0]
    with pytest.raises(ValueError):
        L.cic_demodulate(s, order=0)
