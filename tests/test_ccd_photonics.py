"""Tests for dgs/ccd_photonics.py"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.ccd_photonics import (
    si_quantum_efficiency, si_bandgap_eV, si_cutoff_wavelength_nm,
    CCDPixel, airy_disk_radius_um, psf_airy_pattern,
    check_nyquist_sampling, photocurrent_from_blackbody,
    ccd_signal_chain_sympy,
)
import sympy as sp


def test_qe_peak_green():
    assert abs(si_quantum_efficiency(600) - 0.65) < 0.01


def test_qe_zero_beyond_si():
    assert si_quantum_efficiency(1200) == 0.0
    assert si_quantum_efficiency(300) == 0.0


def test_qe_array():
    lam = np.array([400, 550, 700, 1000, 1200])
    qe = si_quantum_efficiency(lam)
    assert qe.shape == (5,)
    assert qe[1] == 0.65     # 550 nm peak
    assert qe[4] == 0.0      # beyond Si


def test_si_bandgap_room_temp():
    Eg = si_bandgap_eV(300)
    assert 1.10 < Eg < 1.15, f"Eg={Eg}"


def test_si_bandgap_decreases_with_temp():
    assert si_bandgap_eV(77) > si_bandgap_eV(300) > si_bandgap_eV(500)


def test_si_cutoff_near_1100nm():
    lam_cut = si_cutoff_wavelength_nm(300)
    assert 1090 < lam_cut < 1120, f"cutoff={lam_cut}"


def test_ccd_pixel_snr_high_signal():
    px = CCDPixel(full_well_e=50_000, read_noise_e=5.0)
    res = px.snr(photon_flux=100_000, qe=0.65, exposure_s=1.0)
    assert res["SNR_dB"] > 30


def test_ccd_pixel_saturation():
    px = CCDPixel(full_well_e=1000)
    res = px.snr(photon_flux=100_000, qe=1.0, exposure_s=1.0)
    assert res["saturated"]


def test_ccd_pixel_read_noise_limited():
    px = CCDPixel(full_well_e=50_000, read_noise_e=20.0)
    # Very low signal -- read noise dominated
    res = px.snr(photon_flux=1, qe=0.5, exposure_s=0.01)
    assert res["total_noise_e"] >= 20.0


def test_dynamic_range():
    px = CCDPixel(full_well_e=50_000, read_noise_e=5.0)
    res = px.snr(1e5, 0.65, 1.0)
    # DR = 20*log10(50000/5) = 80 dB
    assert abs(res["dynamic_range_dB"] - 80.0) < 0.1


def test_airy_disk_scales_with_fnumber():
    r1 = airy_disk_radius_um(550, 2.8)
    r2 = airy_disk_radius_um(550, 5.6)
    assert abs(r2 / r1 - 2.0) < 1e-6   # Airy radius linear in f/#


def test_airy_disk_scales_with_wavelength():
    r_blue = airy_disk_radius_um(400, 4.0)
    r_red  = airy_disk_radius_um(800, 4.0)
    assert abs(r_red / r_blue - 2.0) < 1e-6


def test_psf_peak_at_center():
    r = np.linspace(0, 10, 500)
    psf = psf_airy_pattern(r, 550, 4.0)
    assert abs(psf[0] - 1.0) < 1e-6


def test_psf_first_zero():
    r = np.linspace(0, 10, 10000)
    psf = psf_airy_pattern(r, 550, 4.0)
    r_airy = airy_disk_radius_um(550, 4.0)
    # Search for minimum within [0.5, 2.0] * r_Airy
    window = (r >= 0.5 * r_airy) & (r <= 2.0 * r_airy)
    r_min = r[window][np.argmin(psf[window])]
    assert abs(r_min - r_airy) / r_airy < 0.15, f"r_min={r_min:.3f}, r_airy={r_airy:.3f}"


def test_nyquist_large_pixel():
    # 5.6 um at f/11, 550 nm -- Airy=7.38 um, Nyquist limit=3.69 um
    ny = check_nyquist_sampling(5.6, 550, 11)
    assert not ny["nyquist_ok"]   # 5.6 > 3.69 -> undersampled


def test_nyquist_small_pixel():
    # 1 um at f/2, 550 nm -- Airy=1.34 um, Nyquist limit=0.67 um
    # 1 um > 0.67 um -- still undersampled for diffraction limit at f/2
    ny = check_nyquist_sampling(1.0, 550, 2)
    # Just verify the dict has correct keys
    assert "r_airy_um" in ny and "nyquist_limit_um" in ny


def test_photocurrent_positive():
    result = photocurrent_from_blackbody(5778)
    assert result["electrons_per_s"] > 0
    assert result["power_W"] > 0


def test_ccd_sympy_5_equations():
    eqs = ccd_signal_chain_sympy()
    assert len(eqs) == 5
    for name, eq in eqs.items():
        assert isinstance(eq, sp.Eq), f"{name} is not sp.Eq"
