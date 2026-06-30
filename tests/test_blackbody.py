"""Tests for dgs/blackbody.py"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.blackbody import (
    photon_energy_eV, wavelength_from_energy_eV,
    planck_radiance, wien_peak_nm, stefan_boltzmann_power,
    planck_integral_numerical, spectral_modulation_sequence,
    am_modulated_photon_flux, numerical_product_rule, verify_product_rule,
    blackbody_sympy_5, hydrogen_series_wavelengths,
    HC_EV_NM, SIGMA_SB,
)
import sympy as sp


def test_hc_constant():
    # hc = 1240 eV*nm to within 0.02%
    assert abs(HC_EV_NM - 1240.0) / 1240.0 < 2e-4


def test_photon_energy_visible():
    # 550 nm green -> ~2.25 eV
    E = photon_energy_eV(550.0)
    assert abs(E - 2.254) < 0.01


def test_photon_energy_xray():
    # 0.1 nm hard X-ray -> ~12.4 keV
    E = photon_energy_eV(0.1)
    assert 12000 < E < 13000


def test_wavelength_roundtrip():
    lam = 632.8   # HeNe laser
    assert abs(wavelength_from_energy_eV(photon_energy_eV(lam)) - lam) < 1e-6


def test_planck_peaks_near_wien():
    # Planck spectrum should peak near Wien prediction
    T = 5778.0
    lam_arr = np.linspace(200, 3000, 5000)
    B = planck_radiance(lam_arr, T)
    lam_peak_numerical = float(lam_arr[np.argmax(B)])
    lam_peak_wien = wien_peak_nm(T)
    assert abs(lam_peak_numerical - lam_peak_wien) / lam_peak_wien < 0.01


def test_planck_short_wavelength_suppressed():
    # B(200 nm, 300 K) ~ 0 (Boltzmann suppression); Sun peak is ~3700 W/m2/sr/nm
    # at 502 nm, room-temp UV is astronomically smaller
    assert planck_radiance(200.0, 300.0) < 1e-50


def test_wien_sun():
    # Sun peak ~502 nm
    assert abs(wien_peak_nm(5778) - 502) < 5


def test_wien_room_temp_infrared():
    # Room temperature peaks in infrared (~10000 nm)
    assert wien_peak_nm(300) > 5000


def test_stefan_boltzmann_sun():
    # Sun surface: ~6.3e7 W/m^2
    P = stefan_boltzmann_power(5778)
    assert 5e7 < P < 8e7


def test_stefan_boltzmann_quadruple():
    # Doubling T -> 16x power
    P1 = stefan_boltzmann_power(1000)
    P2 = stefan_boltzmann_power(2000)
    assert abs(P2 / P1 - 16.0) < 0.01


def test_planck_integral_positive():
    integral = planck_integral_numerical(5778, 100, 5000)
    assert integral > 0


def test_spectral_modulation_sequence():
    seq = spectral_modulation_sequence(5000, 500, 10, 550.0)
    assert len(seq["T_K"]) == 10
    assert seq["E_probe_eV"] > 2.0
    assert seq["modulation_depth"] > 0


def test_am_modulated_flux():
    result = am_modulated_photon_flux(5000, 1000.0, 0.5, 550.0, n_cycles=2)
    assert len(result["t"]) > 0
    assert np.all(result["flux_t"] > 0)


def test_numerical_product_rule():
    x = np.linspace(1.0, 5.0, 1000)
    dx = float(x[1] - x[0])
    f, g = x**2, np.sin(x)
    deriv = numerical_product_rule(f, g, dx)
    # d/dx[x^2 sin x] = 2x sin x + x^2 cos x
    analytical = 2 * x * np.sin(x) + x**2 * np.cos(x)
    rel_err = np.abs(deriv[2:-2] - analytical[2:-2]).max() / np.abs(analytical[2:-2]).max()
    assert rel_err < 1e-4


def test_verify_product_rule_planck():
    lam = np.linspace(300, 2000, 2000)
    result = verify_product_rule(lam, 5778)
    assert result["passes"]


def test_blackbody_sympy_5():
    eqs = blackbody_sympy_5()
    assert len(eqs) == 5
    for name, eq in eqs.items():
        assert isinstance(eq, sp.Eq)


def test_balmer_series():
    rows = hydrogen_series_wavelengths([3, 4, 5], n_lower=2)
    # H-alpha at 656 nm
    halpha = rows[0]
    assert abs(halpha["wavelength_nm"] - 656.1) < 1.0
    # Energy = hc/lambda ~ 1.89 eV
    assert abs(halpha["energy_eV"] - 1.89) < 0.01
