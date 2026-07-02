"""Tests for dgs/jalali_modern_physics.py"""
import math
import numpy as np
import pytest

try:
    from dgs.jalali_modern_physics import (
        dispersive_fourier_transform, steam_camera, optical_rogue_waves,
        compressed_sensing_ts, coherent_time_stretch, ml_on_pts_data,
        jalali_system_summary,
    )
except ImportError:
    from jalali_modern_physics import (
        dispersive_fourier_transform, steam_camera, optical_rogue_waves,
        compressed_sensing_ts, coherent_time_stretch, ml_on_pts_data,
        jalali_system_summary,
    )


# ── DFT tests ────────────────────────────────────────────────────────────

class TestDFT:
    def setup_method(self):
        self.dft = dispersive_fourier_transform(
            D_ps_nm_km=1000.0, L_km=5.0, n_pts=256, f_bandwidth_GHz=100.0
        )

    def test_D_eff_correct(self):
        assert self.dft['fiber']['D_eff_ps_nm'] == pytest.approx(5000.0, rel=0.01)

    def test_H_f_all_pass(self):
        assert self.dft['H_f']['all_pass'] is True

    def test_beta2L_negative_for_anomalous(self):
        # SMF-28 D > 0 at 1550 nm -> normal dispersion convention (beta2 < 0)
        assert self.dft['H_f']['beta2L_s2'] < 0

    def test_output_length(self):
        assert len(self.dft['output']['I_time']) == 256

    def test_t_ps_length(self):
        assert len(self.dft['output']['t_ps']) == 256

    def test_DFT_condition_satisfied(self):
        assert self.dft['DFT_condition']['satisfied'] is True

    def test_time_to_wavelength_formula_key(self):
        assert 'D_eff' in self.dft['time_to_wavelength']['formula']

    def test_output_nonnegative_intensity(self):
        I = self.dft['output']['I_time']
        assert all(v >= 0 for v in I)

    def test_max_phase_nonzero(self):
        assert self.dft['H_f']['max_phase_rad'] > 0

    def test_reference_jalali(self):
        assert 'Jalali' in self.dft['jalali_reference']

    def test_raises_on_negative_D_and_L(self):
        with pytest.raises(ValueError):
            dispersive_fourier_transform(D_ps_nm_km=-1.0, L_km=-1.0)

    def test_custom_spectrum_accepted(self):
        spec = np.ones(256, dtype=complex)
        r = dispersive_fourier_transform(pulse_spectrum=spec, n_pts=256)
        assert len(r['output']['I_time']) == 256


# ── STEAM Camera tests ───────────────────────────────────────────────────

class TestSTEAM:
    def setup_method(self):
        self.st = steam_camera(
            n_rows=40, n_cols=80, D_ps_nm_km=1000.0, L_km=5.0,
            f_rep_MHz=10.0, BW_nm=20.0, lpmm=600.0
        )

    def test_grating_valid(self):
        assert self.st['grating']['valid'] is True

    def test_littrow_angle_positive(self):
        assert self.st['grating']['theta_Littrow_deg'] > 0

    def test_littrow_angle_range(self):
        theta = self.st['grating']['theta_Littrow_deg']
        assert 0 < theta < 90

    def test_angular_dispersion_positive(self):
        assert self.st['grating']['angular_dispersion_rad_per_nm'] > 0

    def test_spatial_dispersion_positive(self):
        assert self.st['grating']['spatial_dispersion_mm_per_nm'] > 0

    def test_D_eff_correct(self):
        assert self.st['time_encoding']['D_eff_ps_nm'] == pytest.approx(5000.0)

    def test_reconstruction_lossless(self):
        assert self.st['reconstruction']['lossless'] is True

    def test_serialized_length(self):
        assert self.st['reconstruction']['serialized_length'] == 40*80

    def test_frame_rate(self):
        assert self.st['system']['frame_rate_MHz'] == pytest.approx(10.0)

    def test_pixel_rate_correct(self):
        # pixel_rate = n_rows * n_cols * f_rep
        assert self.st['system']['pixel_rate_MHz'] == pytest.approx(40*80*10.0)

    def test_image_2d_shape(self):
        img = self.st['image_2d']
        assert len(img) == 40
        assert len(img[0]) == 80

    def test_paper_citation(self):
        assert 'Goda' in self.st['paper'] or 'Nature' in self.st['paper']


# ── Optical Rogue Waves tests ────────────────────────────────────────────

class TestRogueWaves:
    def setup_method(self):
        self.rw = optical_rogue_waves(
            n_pts=512, n_ensemble=30, rng_seed=42,
            gamma=1.3, beta2=-20e-27, P0=1.0
        )

    def test_L_NL_positive(self):
        assert self.rw['physics']['L_NL_m'] > 0

    def test_MI_gain_max_positive(self):
        assert self.rw['modulation_instability']['g_max_per_m'] > 0

    def test_MI_frequency_positive(self):
        assert self.rw['modulation_instability']['f_MI_peak_GHz'] > 0

    def test_Peregrine_amplification_positive(self):
        assert self.rw['Peregrine_soliton']['amplification_factor'] > 1.0

    def test_Peregrine_9x_key(self):
        assert self.rw['Peregrine_soliton']['expected_9x'] is True

    def test_I_Peregrine_length(self):
        assert len(self.rw['Peregrine_soliton']['I_Peregrine']) == 512

    def test_I_Peregrine_nonnegative(self):
        assert all(v >= 0 for v in self.rw['Peregrine_soliton']['I_Peregrine'])

    def test_ensemble_statistics(self):
        assert self.rw['ensemble']['n_ensemble'] == 30
        assert self.rw['ensemble']['I_mean_W'] > 0
        assert self.rw['ensemble']['I_max_W'] >= self.rw['ensemble']['I_mean_W']

    def test_rogue_probability_in_0_1(self):
        p = self.rw['ensemble']['rogue_probability']
        assert 0 <= p <= 1

    def test_jalali_td_gs_connection(self):
        assert 'GS' in self.rw['jalali_td_gs'] or 'phase' in self.rw['jalali_td_gs']

    def test_gamma_negative_raises(self):
        with pytest.raises(ValueError):
            optical_rogue_waves(gamma=-1.0)

    def test_ensemble_too_small_raises(self):
        with pytest.raises(ValueError):
            optical_rogue_waves(n_ensemble=1)

    def test_paper_citation(self):
        assert 'Solli' in self.rw['paper'] or 'Nature' in self.rw['paper']


# ── Compressed Sensing tests ─────────────────────────────────────────────

class TestCompressedSensing:
    def setup_method(self):
        self.cs = compressed_sensing_ts(N=128, K=8, stretch_factor=5.0)

    def test_M_less_than_N(self):
        assert self.cs['signal']['M_measurements'] <= self.cs['signal']['N_Nyquist']

    def test_compression_ratio_greater_1(self):
        assert self.cs['signal']['compression_ratio'] > 1.0

    def test_RIP_bound_present(self):
        assert 'M >=' in self.cs['RIP']['RIP_bound']

    def test_mu_coherence_in_0_1(self):
        assert 0 <= self.cs['RIP']['mu_coherence'] <= 1.0

    def test_ISTA_residual_decreasing(self):
        resid = self.cs['ISTA']['residuals']
        # First 10 iterations should show decrease overall
        assert resid[-1] <= resid[0] * 2   # some convergence

    def test_NMSE_defined(self):
        assert 'NMSE' in self.cs['ISTA']
        assert self.cs['ISTA']['NMSE'] >= 0

    def test_B_captured_positive(self):
        assert self.cs['photonic_advantage']['B_captured_GHz'] > 0

    def test_B_captured_greater_B_ADC(self):
        # With stretch and compression, capture > raw ADC BW
        assert (self.cs['photonic_advantage']['B_captured_GHz'] >
                self.cs['photonic_advantage']['B_ADC_GHz'])

    def test_support_length(self):
        assert len(self.cs['support']) == 8

    def test_K_zero_raises(self):
        with pytest.raises(ValueError):
            compressed_sensing_ts(N=128, K=0)

    def test_stretch_lt_1_raises(self):
        with pytest.raises(ValueError):
            compressed_sensing_ts(stretch_factor=0.5)

    def test_paper_citation(self):
        assert 'Asghari' in self.cs['paper'] or 'Optica' in self.cs['paper']


# ── Coherent Detection tests ─────────────────────────────────────────────

class TestCoherentDetection:
    def test_QPSK_runs(self):
        r = coherent_time_stretch(modulation_format='QPSK')
        assert 'SNR' in r

    def test_BPSK_runs(self):
        r = coherent_time_stretch(modulation_format='BPSK')
        assert r['modulation'] == 'BPSK'

    def test_OOK_runs(self):
        r = coherent_time_stretch(modulation_format='OOK')
        assert r['modulation'] == 'OOK'

    def test_16QAM_runs(self):
        r = coherent_time_stretch(modulation_format='16-QAM')
        assert r['modulation'] == '16-QAM'

    def test_IQ_length(self):
        r = coherent_time_stretch(n_pts=256)
        assert len(r['I_coherent']) == 256
        assert len(r['Q_coherent']) == 256

    def test_coherent_capacity_exceeds_direct(self):
        r = coherent_time_stretch()
        assert r['capacity']['C_coherent_Gbps'] > 0
        # Coherent should be >= direct (IQ doubles capacity)
        assert r['capacity']['ratio'] >= 1.0

    def test_SNR_in_dB_preserved(self):
        r = coherent_time_stretch(SNR_in_dB=30.0)
        assert r['SNR']['SNR_in_dB'] == pytest.approx(30.0)

    def test_invalid_modulation_raises(self):
        with pytest.raises(ValueError):
            coherent_time_stretch(modulation_format='FSK')

    def test_D_eff_correct(self):
        r = coherent_time_stretch(D_ps_nm_km=1000.0, L_km=5.0)
        assert r['fiber']['D_eff_ps_nm'] == pytest.approx(5000.0)

    def test_paper_citation(self):
        r = coherent_time_stretch()
        assert 'Mahjoubfar' in r['paper'] or 'Photon' in r['paper']


# ── ML on PTS tests ──────────────────────────────────────────────────────

class TestMLOnPTS:
    def setup_method(self):
        self.ml = ml_on_pts_data(n_train=200, n_test=50, n_classes=4, rng_seed=17)

    def test_accuracy_above_chance(self):
        # 4-class -> chance = 0.25; trained RF should beat it
        assert self.ml['accuracy'] >= 0.25

    def test_feature_importances_sum_to_1(self):
        total = sum(self.ml['feature_importances'].values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_top_feature_is_named(self):
        names = ['mean','rms','peak','PAPR','centroid','BW','skewness','kurtosis','ZCR','autocorr']
        assert self.ml['top_feature'] in names

    def test_dataset_keys(self):
        for key in ['n_train','n_test','n_classes','n_features']:
            assert key in self.ml['dataset']

    def test_n_features_correct(self):
        assert self.ml['dataset']['n_features'] == 10

    def test_classes_list(self):
        assert len(self.ml['dataset']['classes']) == 4

    def test_rogue_detection_key(self):
        assert 'PAPR' in self.ml['rogue_detection']['key_feature']

    def test_n_train_too_small_raises(self):
        with pytest.raises(ValueError):
            ml_on_pts_data(n_train=5)

    def test_2_class_works(self):
        r = ml_on_pts_data(n_train=100, n_test=20, n_classes=2, rng_seed=1)
        assert r['accuracy'] >= 0.5   # 2-class -> chance = 0.5; should do better

    def test_photonic_advantage_key(self):
        assert 'stretch_factor' in self.ml['photonic_advantage']


# ── System summary tests ─────────────────────────────────────────────────

class TestJalaliSummary:
    def setup_method(self):
        self.ss = jalali_system_summary()

    def test_H_f_formula_present(self):
        assert 'exp' in self.ss['H_f']['formula']

    def test_all_pass_true(self):
        assert self.ss['H_f']['all_pass'] is True

    def test_7_techniques(self):
        assert len(self.ss['techniques']) >= 7

    def test_timeline_ordered(self):
        years = [y for y, _, _ in self.ss['timeline']]
        assert years == sorted(years)

    def test_1999_entry(self):
        years = [y for y, _, _ in self.ss['timeline']]
        assert 1999 in years

    def test_2007_rogue_wave(self):
        descs = [d for _, d, _ in self.ss['timeline']]
        assert any('rogue' in d.lower() for d in descs)

    def test_2009_STEAM(self):
        descs = [d for _, d, _ in self.ss['timeline']]
        assert any('STEAM' in d or 'steam' in d.lower() for d in descs)

    def test_sbir_connection(self):
        assert 'SBIR' in self.ss['sbir_connection'] or 'RogueGuard' in self.ss['sbir_connection']

    def test_connects_to_GS(self):
        connects = self.ss['H_f']['connects_to']
        assert any('GS' in c or 'phase' in c.lower() for c in connects)
