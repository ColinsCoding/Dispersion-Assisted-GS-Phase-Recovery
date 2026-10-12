"""Tests for dgs/laplace_transform.py"""
import numpy as np
import pytest

try:
    from dgs.laplace_transform import (
        laplace_transform_common_pairs, analog_transfer_function_poles_zeros,
        dispersion_kernel_frequency_response, fit_rational_transfer_function_levy,
    )
except ImportError:
    from laplace_transform import (
        laplace_transform_common_pairs, analog_transfer_function_poles_zeros,
        dispersion_kernel_frequency_response, fit_rational_transfer_function_levy,
    )


class TestLaplacePairs:
    def test_all_pairs_verified_symbolically(self):
        r = laplace_transform_common_pairs()
        assert r['all_verified']

    def test_six_pairs_present(self):
        r = laplace_transform_common_pairs()
        assert len(r['pairs']) == 6

    def test_step_pair(self):
        r = laplace_transform_common_pairs()
        assert r['pairs']['unit_step']['matches_textbook']


class TestAnalogTransferFunction:
    def test_stable_first_order_lowpass(self):
        # H(s) = 1/(s+1): pole at s=-1, left-half-plane -> stable
        r = analog_transfer_function_poles_zeros([1.0], [1.0, 1.0])
        assert r['stable']
        assert r['stability_check'] == 'STABLE'

    def test_unstable_right_half_plane_pole(self):
        # H(s) = 1/(s-1): pole at s=+1, right-half-plane -> unstable
        r = analog_transfer_function_poles_zeros([1.0], [1.0, -1.0])
        assert not r['stable']

    def test_dc_gain(self):
        # H(s) = 1/(s+1): at omega=0 (DC), |H|=1
        r = analog_transfer_function_poles_zeros([1.0], [1.0, 1.0], f_max=1.0)
        dc_idx = np.argmin(np.abs(r['omega']))
        assert r['magnitude'][dc_idx] == pytest.approx(1.0, rel=1e-3)


class TestDispersionKernel:
    def test_magnitude_is_unity(self):
        r = dispersion_kernel_frequency_response(D=5000.0)
        assert np.allclose(r['magnitude'], 1.0)

    def test_phase_is_quadratic(self):
        D = 100.0
        r = dispersion_kernel_frequency_response(D=D, f_max=0.4, n_freq=200)
        expected_phase = np.angle(np.exp(1j*np.pi*D*r['f']**2))
        assert np.allclose(np.angle(r['H_complex']), expected_phase)


class TestLevyFit:
    def test_recovers_known_rational_exactly(self):
        # H(s) = 1/((s+1)(s+2)) = 1/(s^2+3s+2) -- Levy's method should recover
        # this essentially exactly since the data genuinely IS rational
        omega = np.linspace(0.01, 10, 500)
        s = 1j * omega
        H_known = 1.0 / ((s + 1) * (s + 2))
        fit = fit_rational_transfer_function_levy(omega, H_known, n_poles=2, n_zeros=0)
        assert fit['max_rel_error'] < 1e-8

    def test_dispersion_kernel_fit_is_genuinely_poor(self):
        # the honest negative result this module exists to demonstrate:
        # H(f)=exp(j*pi*D*f^2) is NOT well-approximated by a low-order
        # rational function -- it is an entire function (no poles), unlike
        # a resonance. A generous 6-pole/6-zero fit should still fail badly.
        r = dispersion_kernel_frequency_response(D=5000.0, f_max=0.3, n_freq=512)
        fit = fit_rational_transfer_function_levy(r['omega'], r['H_complex'], n_poles=6, n_zeros=6)
        assert fit['mean_rel_error'] > 0.1

    def test_condition_number_finite(self):
        omega = np.linspace(0.01, 10, 200)
        H = 1.0 / (1j*omega + 1)
        fit = fit_rational_transfer_function_levy(omega, H, n_poles=1, n_zeros=0)
        assert np.isfinite(fit['condition_number'])
