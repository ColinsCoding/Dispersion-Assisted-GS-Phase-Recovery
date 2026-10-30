"""Tests for dgs/laser_safety.py: MPE formula, NOHD inversion, OD bookkeeping."""
import math

import pytest

from dgs import laser_safety as S


def test_blink_mpe_is_about_2p5_mW_cm2():
    assert S.mpe_visible(0.25) == pytest.approx(1.8e-3 * 0.25 ** -0.25)
    assert 2.5e-3 < S.mpe_visible(0.25) < 2.6e-3


def test_mpe_radiant_exposure_scaling():
    # H = MPE * t grows as t^0.75
    H = lambda t: S.mpe_visible(t) * t
    assert H(10) / H(0.1) == pytest.approx(100 ** 0.75)


def test_photon_rate():
    # 1 mW at 532 nm
    assert S.photon_rate(1e-3, 532) == pytest.approx(1e-3 / (1239.84198 / 532 * 1.602176634e-19))
    assert 2.6e15 < S.photon_rate(1e-3, 532) < 2.7e15


def test_irradiance_and_nohd_inverse():
    P, a, phi = 5e-3, 0.1, 1e-3
    mpe = S.mpe_visible(0.25)
    r = S.nohd(P, mpe, a, phi)
    assert S.irradiance(P, r, a, phi) == pytest.approx(mpe)
    assert S.irradiance(P, 0.5 * r, a, phi) > mpe > S.irradiance(P, 2 * r, a, phi)
    assert 1400 < r < 1550                                       # ~15 m for a 5 mW pointer
    assert S.nohd(1.0, mpe, a, phi) / r == pytest.approx(
        (math.sqrt(4 / (math.pi * mpe)) - a) / (math.sqrt(4 * P / (math.pi * mpe)) - a))
    assert S.nohd(1e-7, mpe, 0.5, 1e-3) == 0.0                   # safe at the aperture


def test_od():
    assert S.required_od(25.0, 2.5e-3) == pytest.approx(4.0)
    assert S.required_od(1e-4, 2.5e-3) == 0.0
    assert S.transmission(3) == pytest.approx(1e-3)
    c = S.eyewear_check(1.0, 532, 100, 0.2, 1e-3, od=4)
    assert c["below_MPE"] == (c["OD_required"] <= 4)
    assert c["margin"] == pytest.approx(10 ** (4 - c["OD_required"]))


@pytest.mark.parametrize("call", [
    lambda: S.mpe_visible(0.25, wavelength_nm=1064),
    lambda: S.mpe_visible(100),
    lambda: S.nohd(-1, 1e-3, 0.1, 1e-3),
    lambda: S.beam_diameter(10, 0.1, -1e-3),
    lambda: S.transmission(-1),
    lambda: S.photon_rate(1e-3, 0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
