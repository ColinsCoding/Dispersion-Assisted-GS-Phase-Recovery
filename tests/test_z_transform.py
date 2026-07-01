"""Tests for dgs/z_transform.py"""
import numpy as np
import pytest

try:
    from dgs.z_transform import (
        z_transform_common_pairs, transfer_function_poles_zeros,
        digital_filter_examples, bilinear_transform,
        first_order_iir_c_implementation, dtft_vs_dft_vs_fft,
    )
except ImportError:
    from z_transform import (
        z_transform_common_pairs, transfer_function_poles_zeros,
        digital_filter_examples, bilinear_transform,
        first_order_iir_c_implementation, dtft_vs_dft_vs_fft,
    )


class TestZTransformPairs:
    def test_six_pairs(self):
        r = z_transform_common_pairs()
        assert len(r['pairs']) >= 5

    def test_unit_impulse_pair(self):
        r = z_transform_common_pairs()
        assert r['pairs']['unit_impulse']['X_z'] == '1'

    def test_theorems_present(self):
        r = z_transform_common_pairs()
        assert 'convolution' in r['theorems']
        assert 'time_shift' in r['theorems']

    def test_convolution_theorem(self):
        r = z_transform_common_pairs()
        thm = r['theorems']['convolution']
        assert 'X(z)' in thm and 'H(z)' in thm


class TestTransferFunction:
    def test_stable_ma_filter(self):
        N = 8
        b = np.ones(N) / N
        a = [1.0]
        r = transfer_function_poles_zeros(b, a)
        assert r['stable']
        assert r['stability_check'] == 'STABLE'

    def test_unstable_pole_outside_circle(self):
        # H(z) = 1/(1 - 1.5*z^-1): pole at z=1.5 (unstable)
        b = [1.0]
        a = [1.0, -1.5]
        r = transfer_function_poles_zeros(b, a)
        assert not r['stable']
        assert 'UNSTABLE' in r['stability_check']

    def test_magnitude_shape(self):
        # Low-pass: DC gain (omega=0) should be high for MA
        b = np.ones(4) / 4
        a = [1.0]
        r = transfer_function_poles_zeros(b, a)
        # DC (first point) should be 0 dB, Nyquist should be lower
        assert r['magnitude_dB'][0] == pytest.approx(0.0, abs=0.5)
        assert r['magnitude_dB'][-1] < r['magnitude_dB'][0]

    def test_output_shapes(self):
        b = [1.0, 0.5]
        a = [1.0, -0.8]
        r = transfer_function_poles_zeros(b, a)
        assert len(r['omega']) == len(r['magnitude_dB'])
        assert len(r['omega']) == len(r['phase_deg'])


class TestDigitalFilterExamples:
    def test_ma_stable(self):
        r = digital_filter_examples()
        assert r['moving_average_N8']['stability'] == 'STABLE'

    def test_iir_stable(self):
        r = digital_filter_examples()
        assert r['iir_lowpass_alpha01']['stability'] == 'STABLE'
        assert r['iir_lowpass_alpha01']['pole_magnitude'] == pytest.approx(0.9, rel=1e-3)

    def test_notch_stable(self):
        r = digital_filter_examples()
        assert r['notch_60Hz']['stability'] == 'STABLE'
        assert r['notch_60Hz']['notch_freq_hz'] == 60.0

    def test_dispersion_magnitude_flat(self):
        r = digital_filter_examples()
        assert r['dispersion_allpass']['magnitude_flat'] is True

    def test_notch_attenuation_at_60hz(self):
        r = digital_filter_examples()
        freq = r['notch_60Hz']['freq_hz']
        mag  = r['notch_60Hz']['mag_dB']
        # Find magnitude near 60 Hz
        idx = np.argmin(np.abs(freq - 60.0))
        assert mag[idx] < -20.0   # deep notch at 60 Hz


class TestBilinearTransform:
    def test_stable_analog_to_stable_digital(self):
        fs = 44100.0
        fc = 1000.0
        analog_pole = -2*np.pi*fc   # left-half-plane: stable
        r = bilinear_transform([analog_pole], [], 2*np.pi*fc, fs)
        assert r['stable']
        assert all(abs(p) < 1.0 for p in r['digital_poles'])

    def test_pole_count_preserved(self):
        fs = 1000.0
        r = bilinear_transform([-100.0, -200.0], [], 1.0, fs)
        assert len(r['digital_poles']) == 2

    def test_extra_zeros_added(self):
        fs = 1000.0
        # 2 analog poles, 0 zeros -> 2 digital zeros at z=-1
        r = bilinear_transform([-100.0, -200.0], [], 1.0, fs)
        assert len(r['digital_zeros']) == 2
        assert all(abs(z + 1) < 1e-6 for z in r['digital_zeros'])


class TestIIRCCode:
    def test_c_code_present(self):
        r = first_order_iir_c_implementation()
        assert 'c_code' in r
        assert 'iir_lowpass_f32' in r['c_code']
        assert 'iir_lowpass_q15' in r['c_code']

    def test_transfer_function_string(self):
        r = first_order_iir_c_implementation()
        assert 'H(z)' in r['transfer_function']
        assert 'alpha' in r['transfer_function']

    def test_rc_analogy(self):
        r = first_order_iir_c_implementation()
        assert 'RC' in r['rc_analogy']

    def test_rogue_guard_use(self):
        r = first_order_iir_c_implementation()
        assert 'GS' in r['rogue_guard_use'] or 'phase' in r['rogue_guard_use']


class TestDTFTvsDFTvsFFT:
    def test_dft_matches_fft(self):
        r = dtft_vs_dft_vs_fft()
        assert r['DFT_matches_FFT'] is True

    def test_z_vs_laplace_keys(self):
        r = dtft_vs_dft_vs_fft()
        zl = r['Z_vs_Laplace']
        assert 'Laplace' in zl
        assert 'Z_transform' in zl
        assert 'mapping' in zl

    def test_mapping_contains_exp(self):
        r = dtft_vs_dft_vs_fft()
        assert 'exp' in r['Z_vs_Laplace']['mapping']

    def test_repo_connection(self):
        r = dtft_vs_dft_vs_fft()
        conn = r['repo_connection']
        assert 'H(f)' in conn or 'H_disp' in conn

    def test_complexity_nlogn(self):
        r = dtft_vs_dft_vs_fft()
        assert 'N log N' in r['FFT'] or 'log' in r['FFT']
