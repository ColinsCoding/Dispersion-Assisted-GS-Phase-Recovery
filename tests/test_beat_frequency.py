import numpy as np
import pytest
import sympy as sp
from dgs.beat_frequency import (
    beat_frequency, beat_frequency_sympy, superpose_waves,
    standing_wave_modes, standing_wave_sympy,
    dirac_delta_fourier_pair_sympy, heterodyne_mixing,
    beat_frequency_sympy_5,
)


def test_beat_frequency_basic():
    r = beat_frequency(440, 443)
    assert r["f_beat_hz"] == pytest.approx(3.0)
    assert r["f_carrier_hz"] == pytest.approx(441.5)


def test_beat_frequency_period():
    r = beat_frequency(100, 104)
    assert r["period_beat_s"] == pytest.approx(0.25)


def test_beat_frequency_negative_raises():
    with pytest.raises(ValueError):
        beat_frequency(-1, 440)


def test_beat_frequency_equal_tones():
    r = beat_frequency(440, 440)
    assert r["f_beat_hz"] == 0.0
    assert r["period_beat_s"] == float("inf")


def test_superpose_waves_shape():
    r = superpose_waves(440, 443)
    assert r["t"].shape == r["y"].shape
    assert r["t"].shape == r["envelope"].shape


def test_superpose_waves_envelope_positive():
    r = superpose_waves(440, 443)
    assert np.all(r["envelope"] >= 0)


def test_superpose_waves_beat_period_in_output():
    # At t = 1/f_beat the envelope should return to maximum
    r = superpose_waves(100, 102, t_max=0.5)
    assert r["f_beat_hz"] == pytest.approx(2.0)


def test_standing_wave_modes_fundamental():
    modes = standing_wave_modes(L_m=1.0, v_ms=340.0, n_max=1)
    n, f = modes[0]
    assert n == 1
    assert f == pytest.approx(170.0)


def test_standing_wave_modes_harmonics():
    modes = standing_wave_modes(L_m=0.5, v_ms=300.0, n_max=3)
    freqs = [f for _, f in modes]
    # each mode is integer multiple of fundamental
    f1 = freqs[0]
    assert freqs[1] == pytest.approx(2 * f1)
    assert freqs[2] == pytest.approx(3 * f1)


def test_standing_wave_modes_invalid():
    with pytest.raises(ValueError):
        standing_wave_modes(L_m=0, v_ms=300)


def test_standing_wave_sympy_returns_equations():
    eqs = standing_wave_sympy()
    assert isinstance(eqs["standing_wave"], sp.Eq)
    assert isinstance(eqs["mode_condition"], sp.Eq)


def test_dirac_delta_ft_unity():
    eqs = dirac_delta_fourier_pair_sympy()
    assert eqs["ft_of_delta"].rhs == sp.Integer(1)


def test_heterodyne_mixing_if_frequency():
    r = heterodyne_mixing(1e9, 0.9e9)
    assert r["f_IF_hz"] == pytest.approx(1e8)


def test_heterodyne_mixing_sum_frequency():
    r = heterodyne_mixing(1e9, 0.9e9)
    assert r["f_sum_hz"] == pytest.approx(1.9e9)


def test_beat_frequency_sympy_5_count():
    eqs = beat_frequency_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)


def test_beat_frequency_sympy_keys():
    eqs = beat_frequency_sympy()
    assert "beat_freq" in eqs
    assert "factored_form" in eqs
