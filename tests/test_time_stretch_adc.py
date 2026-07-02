"""Tests for dgs/time_stretch_adc.py -- Photonic Time-Stretch ADC"""
import math
import numpy as np
import pytest

try:
    from dgs.time_stretch_adc import (
        derive_stretch_factor,
        time_stretch_adc_system,
        steam_camera,
        random_forest_adc,
    )
except ImportError:
    from time_stretch_adc import (
        derive_stretch_factor,
        time_stretch_adc_system,
        steam_camera,
        random_forest_adc,
    )


class TestStretchFactorDerivation:
    def test_M_formula_matches_1_plus_ratio(self):
        sf = derive_stretch_factor(D1_ps_nm_km=1000, L1_km=1,
                                   D2_ps_nm_km=1000, L2_km=4)
        assert sf['M'] == pytest.approx(5.0, rel=1e-6)

    def test_M_check_equals_M(self):
        sf = derive_stretch_factor(D1_ps_nm_km=500, L1_km=2,
                                   D2_ps_nm_km=500, L2_km=8)
        assert sf['M'] == pytest.approx(sf['M_check'], rel=1e-8)

    def test_M_equals_1_when_L2_is_zero(self):
        # limit: L2=0 -> M=1
        sf = derive_stretch_factor(D1_ps_nm_km=1000, L1_km=1,
                                   D2_ps_nm_km=1000, L2_km=0.001)
        assert sf['M'] == pytest.approx(1.001, rel=0.01)

    def test_M_large_for_long_L2(self):
        sf = derive_stretch_factor(D1_ps_nm_km=1000, L1_km=1,
                                   D2_ps_nm_km=1000, L2_km=99)
        assert sf['M'] == pytest.approx(100.0, rel=1e-6)

    def test_B_RF_equals_M_times_fs_over_2(self):
        sf = derive_stretch_factor(D1_ps_nm_km=1000, L1_km=1,
                                   D2_ps_nm_km=1000, L2_km=9,
                                   f_s_GHz=1.0)
        assert sf['B_RF_GHz'] == pytest.approx(sf['M'] * 1.0 / 2, rel=1e-6)

    def test_T_total_equals_M_times_T1(self):
        sf = derive_stretch_factor(D1_ps_nm_km=1000, L1_km=1,
                                   D2_ps_nm_km=1000, L2_km=4,
                                   Delta_lambda_nm=10)
        T1 = sf['derivation']['T1_ps']
        T_tot = sf['derivation']['T_total_ps']
        assert T_tot == pytest.approx(sf['M'] * T1, rel=1e-6)

    def test_beta2_negative_for_anomalous_D(self):
        # D > 0 (anomalous dispersion) -> beta2 < 0
        sf = derive_stretch_factor(D1_ps_nm_km=17, L1_km=1,
                                   D2_ps_nm_km=17, L2_km=1)
        assert sf['derivation']['beta2_1_s2_m'] < 0
        assert sf['derivation']['beta2_2_s2_m'] < 0

    def test_chirp_rate_positive(self):
        sf = derive_stretch_factor()
        assert sf['derivation']['alpha_chirp_nm_per_ps'] > 0

    def test_shot_noise_SNR_reasonable(self):
        sf = derive_stretch_factor()
        assert 10 < sf['noise']['SNR_shot_dB'] < 80

    def test_H_total_is_allpass(self):
        sf = derive_stretch_factor()
        H_mag = np.array(sf['H_total']['H_mag'])
        assert np.allclose(H_mag, 1.0, atol=1e-10)

    def test_H_total_has_quadratic_phase(self):
        sf = derive_stretch_factor()
        phi = np.array(sf['H_total']['H_phase_rad'])
        f = np.array(sf['H_total']['f_GHz'])
        # phi should be proportional to f^2: check second differences are ~constant
        # (second derivative of quadratic is constant)
        dphi = np.diff(phi)
        d2phi = np.diff(dphi)
        # Should be approximately constant (quadratic)
        assert np.std(d2phi) < 0.1 * (np.abs(np.mean(d2phi)) + 1e-30)

    def test_invalid_D1_raises(self):
        with pytest.raises(ValueError):
            derive_stretch_factor(D1_ps_nm_km=-1)

    def test_invalid_L1_raises(self):
        with pytest.raises(ValueError):
            derive_stretch_factor(L1_km=-1)

    def test_invalid_fs_raises(self):
        with pytest.raises(ValueError):
            derive_stretch_factor(f_s_GHz=0)

    def test_GS_connection_string_present(self):
        sf = derive_stretch_factor()
        assert 'GS' in sf['GS_connection'] or 'phase' in sf['GS_connection']

    def test_N_samples_per_pulse_positive(self):
        sf = derive_stretch_factor()
        assert sf['N_samples_per_pulse'] > 0

    def test_large_bandwidth_captured(self):
        # M=100, f_s=1 Gsample/s -> 50 GHz captured
        sf = derive_stretch_factor(D1_ps_nm_km=1000, L1_km=1,
                                   D2_ps_nm_km=1000, L2_km=99,
                                   f_s_GHz=1.0)
        assert sf['B_RF_GHz'] == pytest.approx(50.0, rel=1e-6)


class TestTSADCSystem:
    def test_M_stored_correctly(self):
        sys = time_stretch_adc_system(M=5.0, f_s_GHz=1.0, N_pts=256)
        assert sys['system']['M'] == pytest.approx(5.0)

    def test_effective_bandwidth_equals_M_fs_half(self):
        sys = time_stretch_adc_system(M=10.0, f_s_GHz=1.0, N_pts=256)
        assert sys['ADC']['B_RF_GHz'] == pytest.approx(5.0)

    def test_detected_intensity_nonnegative(self):
        sys = time_stretch_adc_system(M=5.0, N_pts=256)
        assert np.all(np.array(sys['signal']['I_detected']) >= 0)

    def test_ENOB_positive(self):
        sys = time_stretch_adc_system(M=5.0, SNR_dB=40, N_pts=256)
        assert sys['ADC']['ENOB'] > 0

    def test_ENOB_bounded_by_n_bits(self):
        s = time_stretch_adc_system(M=5.0, SNR_dB=30, N_pts=256)
        assert 0 < s['ADC']['ENOB'] <= 8

    def test_transfer_function_all_pass(self):
        sys = time_stretch_adc_system(M=5.0, N_pts=256)
        assert sys['transfer_function']['all_pass'] is True

    def test_poles_none(self):
        sys = time_stretch_adc_system(M=5.0, N_pts=256)
        assert 'None' in sys['transfer_function']['poles']

    def test_group_delay_linear(self):
        sys = time_stretch_adc_system(M=5.0, N_pts=256)
        gd = np.array(sys['transfer_function']['group_delay_ps'])
        omega = np.array(sys['transfer_function']['omega_rad_s'])
        # Group delay should be linear in omega: d(tau)/d(omega) = beta2*L = const
        # Check linearity via R^2
        coeffs = np.polyfit(omega, gd, 1)
        gd_fit = np.polyval(coeffs, omega)
        ss_res = np.sum((gd - gd_fit)**2)
        ss_tot = np.sum((gd - gd.mean())**2)
        R2 = 1 - ss_res / (ss_tot + 1e-30)
        assert R2 > 0.99

    def test_spectrum_has_positive_frequencies(self):
        sys = time_stretch_adc_system(M=5.0, N_pts=512)
        f = np.array(sys['spectrum']['f_MHz'])
        assert np.all(f >= 0)

    def test_invalid_M_raises(self):
        with pytest.raises(ValueError):
            time_stretch_adc_system(M=0.5)

    def test_invalid_fs_raises(self):
        with pytest.raises(ValueError):
            time_stretch_adc_system(f_s_GHz=-1)

    def test_PID_view_present(self):
        sys = time_stretch_adc_system(M=5.0, N_pts=256)
        assert len(sys['transfer_function']['PID_view']) > 0

    def test_t_original_shorter_than_t_stretched(self):
        M = 5.0
        sys = time_stretch_adc_system(M=M, N_pts=256, f_s_GHz=1.0)
        t_o = np.array(sys['signal']['t_original_ns'])
        t_s = np.array(sys['signal']['t_stretched_ns'])
        # Original time axis spans 1/M of the stretched
        assert t_o[-1] == pytest.approx(t_s[-1] / M, rel=0.05)


class TestSTEAMCamera:
    def test_total_pixels_correct(self):
        s = steam_camera(n_pixels_x=100, n_pixels_y=50)
        assert s['geometry']['total_pixels'] == 5000

    def test_frame_rate_equals_rep_rate(self):
        s = steam_camera(f_rep_MHz=100)
        assert s['geometry']['frame_rate_Mfps'] == pytest.approx(100.0)

    def test_tau_total_positive(self):
        s = steam_camera()
        assert s['time_stretch']['tau_total_ps'] > 0

    def test_f_ADC_needed_positive(self):
        s = steam_camera()
        assert s['time_stretch']['f_ADC_needed_GHz'] > 0

    def test_image_shape_correct(self):
        s = steam_camera(n_pixels_x=50, n_pixels_y=25)
        img = np.array(s['image_2d'])
        assert img.shape == (25, 50)

    def test_image_range_0_to_1(self):
        s = steam_camera(n_pixels_x=50, n_pixels_y=25)
        img = np.array(s['image_2d'])
        assert np.all(img >= 0) and np.max(img) <= 1.05   # slight noise ok

    def test_serialized_length_correct(self):
        s = steam_camera(n_pixels_x=50, n_pixels_y=25)
        assert len(s['image_serialized']) == 50*25

    def test_duty_cycle_between_0_and_100(self):
        s = steam_camera()
        dc = s['performance']['duty_cycle_pct']
        assert 0 < dc <= 100

    def test_spatial_extent_positive(self):
        s = steam_camera()
        assert s['optics']['total_spatial_extent_mm'] > 0

    def test_dt_per_pixel_positive(self):
        s = steam_camera()
        assert s['time_stretch']['dt_per_pixel_ps'] > 0

    def test_grating_density_stored(self):
        s = steam_camera()
        assert s['optics']['grating_density_lpmm'] in (600, 1200)

    def test_hyperspectral_channels_4(self):
        s = steam_camera()
        n_ch = sum(1 for k in s['hyperspectral_channels'] if k.endswith('_nm'))
        assert n_ch == 4

    def test_GS_connection_present(self):
        s = steam_camera()
        assert 'GS' in s['GS_connection']

    def test_duty_cycle_less_when_rep_rate_higher(self):
        s1 = steam_camera(f_rep_MHz=10)
        s2 = steam_camera(f_rep_MHz=100)
        assert s2['performance']['duty_cycle_pct'] > s1['performance']['duty_cycle_pct']


class TestRandomForestADC:
    def test_n_classes_correct(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        assert rf['dataset']['n_classes'] == 4

    def test_class_names_correct(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        assert rf['dataset']['class_names'] == ['OOK', 'BPSK', 'QPSK', '16-QAM']

    def test_train_acc_above_threshold(self):
        rf = random_forest_adc(n_samples=400, n_trees=30)
        assert rf['forest']['acc_train'] > 0.5

    def test_test_acc_above_random(self):
        rf = random_forest_adc(n_samples=400, n_trees=30)
        # Random = 0.25 for 4 classes; forest should beat it
        assert rf['forest']['acc_test'] > 0.25

    def test_confusion_matrix_shape(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        cm = np.array(rf['confusion_matrix'])
        assert cm.shape == (4, 4)

    def test_confusion_matrix_nonneg(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        cm = np.array(rf['confusion_matrix'])
        assert np.all(cm >= 0)

    def test_confusion_matrix_sum_correct(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        cm = np.array(rf['confusion_matrix'])
        n_test = int(0.2 * 200)
        assert cm.sum() == pytest.approx(n_test, abs=2)

    def test_feature_importance_sums_to_1(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        total = sum(rf['feature_importance'].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_feature_importance_nonneg(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        assert all(v >= 0 for v in rf['feature_importance'].values())

    def test_expected_feature_names(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        for name in ['mean', 'rms', 'PAPR', 'centroid', 'bandwidth',
                     'skewness', 'kurtosis', 'ZCR', 'autocorr']:
            assert name in rf['feature_importance']

    def test_GS_connection_modulation_recognition(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        gs_c = rf['GS_connection']
        assert 'modulation' in gs_c['modulation_recognition'].lower()

    def test_OUSD_HMI_key(self):
        rf = random_forest_adc(n_samples=200, n_trees=20)
        assert 'OUSD_HMI' in rf['GS_connection']

    def test_n_trees_stored(self):
        rf = random_forest_adc(n_samples=200, n_trees=25)
        assert rf['forest']['n_trees'] == 25

    def test_more_trees_does_not_crash(self):
        rf = random_forest_adc(n_samples=200, n_trees=100)
        assert rf['forest']['acc_test'] >= 0
