import numpy as np
import pytest
import sympy as sp
from dgs.thermal_radiation import (
    planck_spectral_radiance, wien_peak_wavelength, stefan_boltzmann_power,
    temperature_from_intensity, gaussian_integral_fraction,
    gaussian_integral_sympy, debye_length, plasma_frequency,
    thermal_radiation_sympy_5,
)


def test_wien_sun():
    w = wien_peak_wavelength(5778)
    assert 450 < w["lambda_peak_nm"] < 560   # green-yellow


def test_wien_human_body_midir():
    w = wien_peak_wavelength(310)
    assert w["band"] == "mid-IR"


def test_wien_positive_only():
    with pytest.raises(ValueError):
        wien_peak_wavelength(0)


def test_stefan_boltzmann_scales_as_T4():
    p1 = stefan_boltzmann_power(300)["power_W"]
    p2 = stefan_boltzmann_power(600)["power_W"]
    assert p2 / p1 == pytest.approx(16.0, rel=1e-4)


def test_stefan_boltzmann_emissivity_zero():
    p = stefan_boltzmann_power(1000, emissivity=0)["power_W"]
    assert p == pytest.approx(0.0)


def test_stefan_boltzmann_invalid():
    with pytest.raises(ValueError):
        stefan_boltzmann_power(-1)
    with pytest.raises(ValueError):
        stefan_boltzmann_power(300, emissivity=1.5)


def test_temperature_from_intensity_roundtrip():
    T = 500.0
    sb = stefan_boltzmann_power(T, area_m2=1.0)
    T_rec = temperature_from_intensity(sb["power_W"])["T_K"]
    assert T_rec == pytest.approx(T, rel=1e-4)


def test_gaussian_one_sigma():
    r = gaussian_integral_fraction(1.0)
    assert r["percent_inside"] == pytest.approx(68.27, abs=0.1)


def test_gaussian_two_sigma():
    r = gaussian_integral_fraction(2.0)
    assert r["percent_inside"] == pytest.approx(95.45, abs=0.1)


def test_gaussian_three_sigma():
    r = gaussian_integral_fraction(3.0)
    assert r["percent_inside"] == pytest.approx(99.73, abs=0.1)


def test_gaussian_zero_sigma():
    r = gaussian_integral_fraction(0.0)
    assert r["fraction_inside"] == pytest.approx(0.0)


def test_gaussian_invalid():
    with pytest.raises(ValueError):
        gaussian_integral_fraction(-1)


def test_planck_radiance_positive():
    B = planck_spectral_radiance(500, 5778)
    assert B > 0


def test_planck_radiance_peak_near_500nm_for_sun():
    wavelengths = np.arange(300, 1000, 10)
    B = planck_spectral_radiance(wavelengths, 5778)
    peak_idx = np.argmax(B)
    peak_nm = wavelengths[peak_idx]
    assert 450 < peak_nm < 560


def test_debye_length_positive():
    r = debye_length(1e10, 1000)
    assert r["debye_length_m"] > 0


def test_debye_length_scales_correctly():
    r1 = debye_length(1e10, 1000)
    r2 = debye_length(4e10, 1000)
    # lambda_D ~ n^(-1/2)
    assert r1["debye_length_m"] / r2["debye_length_m"] == pytest.approx(2.0, rel=0.01)


def test_plasma_frequency_ionosphere():
    r = plasma_frequency(1e10)
    # Earth ionosphere ~0.9 MHz
    assert 0.5 < r["f_p_MHz"] < 5.0


def test_plasma_frequency_positive():
    with pytest.raises(ValueError):
        plasma_frequency(0)


def test_thermal_sympy_5_count():
    eqs = thermal_radiation_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
