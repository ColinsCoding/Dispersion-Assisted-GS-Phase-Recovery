"""Tests for dgs/photons_in_semiconductors.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.photons_in_semiconductors import (
    MATERIALS, effective_density_cm3, boltzmann_limit_symbolic, fermi_dirac_half,
    fermi_dirac_half_mapped, electron_density, intrinsic_density, fermi_level,
    led_spectrum, led_peak_symbolic, led_fwhm_over_kT, gain_spectrum,
    bernard_duraffourg_symbolic, photon_counts, beat_photocurrent, coherent_wdm_batch,
    K_B_EV,
)


def test_boltzmann_limit_exact():
    assert boltzmann_limit_symbolic() == sp.exp(sp.Symbol("eta", real=True))


def test_fermi_dirac_integral_values_and_two_methods():
    assert float(fermi_dirac_half(0.0)) == pytest.approx(0.765147, abs=1e-6)
    e = np.array([-10, -3, 0, 3, 10.0])
    assert np.allclose(fermi_dirac_half(e), fermi_dirac_half_mapped(e), rtol=1e-6)
    # Boltzmann limit for eta << 0, Sommerfeld leading term for eta >> 0
    assert float(fermi_dirac_half(-15)) == pytest.approx(np.exp(-15), rel=1e-6)
    big = 60.0
    assert float(fermi_dirac_half(big)) == pytest.approx(4 / (3 * np.sqrt(np.pi)) * big ** 1.5, rel=1e-3)


def test_boltzmann_overestimates_when_degenerate():
    assert electron_density(3.0, boltzmann=True) > 4 * electron_density(3.0)
    assert electron_density(-5.0, boltzmann=True) == pytest.approx(electron_density(-5.0), rel=0.01)


def test_effective_densities_and_ni():
    assert effective_density_cm3(MATERIALS["Si"]["me"]) == pytest.approx(2.8e19, rel=0.02)
    assert effective_density_cm3(MATERIALS["GaAs"]["me"]) == pytest.approx(4.4e17, rel=0.05)
    assert 5e9 < intrinsic_density("Si") < 1.5e10
    assert 1e6 < intrinsic_density("GaAs") < 3e6
    assert intrinsic_density("Si", T=350) > 10 * intrinsic_density("Si", T=300)


def test_fermi_level_doping():
    r0 = fermi_level(material="GaAs")
    assert r0["n"] == pytest.approx(r0["p"], rel=1e-8)
    assert r0["n"] == pytest.approx(intrinsic_density("GaAs"), rel=1e-6)
    rn = fermi_level(Nd=1e16, material="Si")
    assert rn["n"] == pytest.approx(1e16, rel=1e-6)
    assert rn["n"] * rn["p"] == pytest.approx(intrinsic_density("Si") ** 2, rel=1e-3)
    rp = fermi_level(Na=1e17, material="Si")
    assert rp["E_F"] < r0["E_F"] and rn["E_F"] > fermi_level(material="Si")["E_F"]
    with pytest.raises(ValueError):
        fermi_level(Nd=-1)


def test_led_peak_and_width():
    kT = sp.Symbol("kT", positive=True)
    assert led_peak_symbolic() == kT / 2
    assert led_fwhm_over_kT() == pytest.approx(1.795, abs=0.002)
    hv = np.linspace(1.40, 1.60, 20001)
    S = led_spectrum(hv, 1.42)
    assert hv[np.argmax(S)] == pytest.approx(1.42 + K_B_EV * 300 / 2, abs=2e-5)


def test_bernard_duraffourg():
    num = bernard_duraffourg_symbolic()
    E1, E2, Fc, Fv, kT = sp.symbols("E1 E2 F_c F_v kT", real=True)
    expect = (sp.exp((E1 + Fc) / kT) - sp.exp((E2 + Fv) / kT)) * sp.exp(-(Fc + Fv) / kT)
    assert sp.simplify(num - expect) == 0
    Eg = MATERIALS["GaAs"]["Eg"]
    EFc, EFv = Eg + 0.05, -0.02
    hv = np.linspace(Eg + 1e-4, Eg + 0.2, 4000)
    g = gain_spectrum(hv, EFc, EFv)
    cross = hv[np.argmax(g < 0)]
    assert cross == pytest.approx(EFc - EFv, abs=1e-3)
    # no inversion -> no gain anywhere
    assert np.all(gain_spectrum(hv, Eg - 0.3, 0.3) <= 0)


def test_photon_counts_are_poisson():
    c, mean = photon_counts(1e-9, 1550, 1e-9, n=200000)
    assert c.mean() == pytest.approx(mean, rel=0.01)
    assert c.var() == pytest.approx(mean, rel=0.02)


def test_beat_photocurrent_extrema():
    t = np.linspace(0, 1e-9, 10001)
    i = beat_photocurrent(1e-3, 1e-3, 5e9, t, eta=1.0)
    from dgs.transimpedance_amplifier import responsivity
    R = responsivity(1550, 1.0)
    assert i.max() == pytest.approx(4e-3 * R, rel=1e-6)          # (sqrt P1 + sqrt P2)^2
    assert i.min() == pytest.approx(0.0, abs=1e-9)


def test_coherent_wdm_recovers_channels():
    P = np.array([1e-6, 2e-6, 0.5e-6])
    r = coherent_wdm_batch(P, [1e9, 2.5e9, 4e9], trials=32, seed=3, device="cpu")
    assert np.allclose(r["recovered"].mean(0), P, rtol=5e-3)
    assert np.allclose(r["tone_measured"].mean(0), r["tone_expected"], rtol=5e-3)
    with pytest.raises(ValueError):
        coherent_wdm_batch([1e-6], [15e9])


def test_validation():
    with pytest.raises(ValueError):
        effective_density_cm3(0)
    with pytest.raises(ValueError):
        intrinsic_density("Unobtainium")


def test_strong_lo_reaches_shot_noise_limit():
    std = lambda Plo, RL: coherent_wdm_batch([1e-8], [2e9], P_lo=Plo, trials=200,
                                             R_load=RL, seed=5, device="cpu")["recovered"].std()
    shot = std(1e-2, None)
    assert std(1e-6, 50.0) > 10 * shot          # thermal-noise limited at weak LO
    assert std(1e-2, 50.0) < 1.2 * shot         # strong LO: shot-noise limited
    with pytest.raises(ValueError):
        coherent_wdm_batch([1e-8], [2e9], R_load=0.0)
