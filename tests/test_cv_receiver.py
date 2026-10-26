"""Tests for dgs.cv_receiver."""
import numpy as np
import pytest

from dgs import cv_receiver as cv

T0 = 40.0
DS = [-5000.0, -12000.0]


def _t(n=1024):
    return np.arange(n) - n / 2


def test_loss_and_db():
    assert np.isclose(cv.db(cv.VACUUM), 0)
    assert np.isclose(cv.measured_squeezing_db(-10, 1.0), -10)
    assert np.isclose(cv.after_loss(0.05, 0.9), 0.9 * 0.05 + 0.1 * 0.5)
    assert np.all(np.diff(cv.measured_squeezing_db(-10, np.array([1.0, 0.9, 0.5]))) > 0)


def test_phase_insensitive_amp_destroys_squeezing_psa_does_not():
    V = cv.variance_from_db(-10)
    G = np.array([1.0, 2.0, 10.0, 1000.0])
    referred = cv.after_pia(V, G) / G
    assert np.isclose(cv.db(referred[0]), -10)
    assert np.all(cv.db(referred[2:]) > -0.1)                       # >= ~vacuum: squeezing gone
    Vx, Vp = cv.after_psa(V, 1 / V * cv.VACUUM ** 2, 100.0)
    assert np.isclose(Vx / 100.0, V)                                 # noiseless on the amplified quadrature


def test_homodyne_samples_variance():
    s = cv.homodyne_samples(0.05, 200000)
    assert np.isclose(s.var(), 0.05, rtol=0.02)


def test_chirp_overlap_closed_form():
    t = _t()
    for C in (0.0, 2.0, 4.0, 8.0):
        assert np.isclose(cv.mode_overlap(cv.chirped_gaussian(t, T0, 0), cv.chirped_gaussian(t, T0, C)),
                          cv.gaussian_chirp_overlap(C), rtol=1e-9)


def test_two_planes_are_ambiguous_three_are_not():
    t = _t()
    E = cv.chirped_gaussian(t, T0, 3.0)
    I = cv.calibration_intensities(E, DS)
    g2, e2 = cv.gs_multiplane(I[0], I[1:2], DS[:1], np.sqrt(I[0]).astype(complex), 500)
    assert e2 < 1e-20 and cv.mode_overlap(g2, E) < 0.9               # fits the data, wrong mode
    g3, e3, _ = cv.retrieve_mode(I[0], I[1:], DS, T0)
    assert cv.mode_overlap(g3, E) > 0.9999


def test_multistart_handles_large_chirp():
    t = _t()
    for C in (-8.0, 10.0):
        E = cv.chirped_gaussian(t, T0, C)
        I = cv.calibration_intensities(E, DS)
        assert cv.mode_overlap(cv.retrieve_mode(I[0], I[1:], DS, T0)[0], E) > 0.9999


def test_denoising_rescues_noisy_calibration():
    t = _t()
    E = cv.chirped_gaussian(t, T0, 4.0)
    I = cv.calibration_intensities(E, DS, snr_db=20, seed=1)
    raw = cv.mode_overlap(cv.retrieve_mode(I[0], I[1:], DS, T0)[0], E)
    J = [cv.denoise_intensity(x, 0.02, 20) for x in I]
    den = cv.mode_overlap(cv.retrieve_mode(J[0], J[1:], DS, T0)[0], E)
    assert den > 0.95 and den > raw


def test_gkp_error_exact_approx_and_monte_carlo():
    sig = np.sqrt(cv.gkp_noise_variance(-8))
    assert np.isclose(cv.gkp_error_exact(sig), cv.gkp_error_approx(sig), rtol=1e-6)
    assert np.isclose(cv.gkp_monte_carlo(-8, n=400000), cv.gkp_error_exact(sig), rtol=0.1)
    assert cv.gkp_error_exact(np.sqrt(cv.gkp_noise_variance(-8, eta=0.9))) > cv.gkp_error_exact(sig)


def test_gkp_adc_needs_enough_bits():
    assert cv.gkp_monte_carlo(-10, adc_bits=3, n=20000) > 0.3        # LSB ~ sqrt(pi): cells unresolved
    assert cv.gkp_monte_carlo(-10, adc_bits=8, n=20000) < 0.01


@pytest.mark.parametrize("call", [
    lambda: cv.after_loss(0.5, 0.0),
    lambda: cv.after_pia(0.5, 0.5),
    lambda: cv.chirped_gaussian(_t(), 0.0, 1.0),
    lambda: cv.gkp_error_exact(0.0),
    lambda: cv.denoise_intensity(np.ones(8), cutoff=0.0),
    lambda: cv.retrieve_mode(np.ones(8), [np.ones(8)], [], 1.0),
    lambda: cv.homodyne_samples(0.5, 0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
