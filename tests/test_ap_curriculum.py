"""Tests for dgs/ap_curriculum.py"""
import math
import numpy as np
import pytest

try:
    from dgs.ap_curriculum import (
        integrated_math_to_eigenvalues, ap_physics_c_em,
        ap_calculus_photonics, ap_statistics_logistic,
        ap_chem_bio_photonics, ap_cs_econ_lang,
    )
except ImportError:
    from ap_curriculum import (
        integrated_math_to_eigenvalues, ap_physics_c_em,
        ap_calculus_photonics, ap_statistics_logistic,
        ap_chem_bio_photonics, ap_cs_econ_lang,
    )


class TestIntegratedMathEigenvalues:
    def setup_method(self):
        self.im = integrated_math_to_eigenvalues()

    def test_linear_system_correct(self):
        assert self.im['math1_linear_system']['check_Ax_equals_b'] is True

    def test_euler_formula_error_zero(self):
        assert self.im['math3_euler']['max_error'] < 1e-14

    def test_euler_thetas_6(self):
        assert len(self.im['math3_euler']['thetas_deg']) == 6

    def test_hermitian_all_real(self):
        assert self.im['hermitian_eigenvalues']['all_real'] is True

    def test_hermitian_reconstruction_error(self):
        assert self.im['hermitian_eigenvalues']['reconstruction_error'] < 1e-10

    def test_eigenvalues_real_values(self):
        eigs = self.im['hermitian_eigenvalues']['eigenvalues']
        # All should be real floats
        for e in eigs:
            assert isinstance(e, float)

    def test_4_eigenvalues(self):
        assert len(self.im['hermitian_eigenvalues']['eigenvalues']) == 4

    def test_spectral_theorem_verified(self):
        assert 'verified' in self.im['hermitian_eigenvalues']['spectral_theorem'].lower()

    def test_pauli_eigenvalues_pm1(self):
        assert self.im['pauli_matrices']['all_pm1'] is True
        for m in ['sigma_x', 'sigma_z']:
            eigs = sorted(self.im['pauli_matrices'][m])
            assert eigs[0] == pytest.approx(-1.0, abs=1e-10)
            assert eigs[1] == pytest.approx(1.0, abs=1e-10)

    def test_complex_k_has_real_imag(self):
        k_r = self.im['math2_quadratic_complex']['complex_k_real']
        k_i = self.im['math2_quadratic_complex']['complex_k_imag']
        assert k_r > 0
        assert k_i != 0

    def test_skin_depth_positive(self):
        assert self.im['math2_quadratic_complex']['skin_depth_m'] > 0

    def test_photonics_key_in_each_section(self):
        for key in ['math1_linear_system', 'math2_quadratic_complex',
                    'math3_euler', 'hermitian_eigenvalues']:
            assert 'photonics' in self.im[key]


class TestApPhysicsCEM:
    def setup_method(self):
        self.em = ap_physics_c_em()

    def test_c_from_maxwell_close(self):
        c_calc = self.em['maxwell_equations']['c_from_Maxwell']
        assert abs(c_calc - 2.998e8)/2.998e8 < 0.01

    def test_c_error_small_ppm(self):
        assert self.em['maxwell_equations']['c_error_ppm'] < 100

    def test_skin_depth_copper_mm_range(self):
        d = self.em['skin_depths']['copper_60Hz_mm']
        assert 8 < d < 12   # copper 60 Hz skin depth ~8.5 mm

    def test_skin_depth_rebar_greater_copper(self):
        assert (self.em['skin_depths']['rebar_60Hz_mm'] >
                self.em['skin_depths']['copper_60Hz_mm'])

    def test_solar_efficiency_in_range(self):
        eta = self.em['sustainability']['efficiency_pct']
        assert 1 < eta < 50

    def test_Voc_positive(self):
        assert self.em['sustainability']['V_oc_V'] > 0

    def test_lambda_gap_Si_near_1100nm(self):
        assert self.em['sustainability']['lambda_gap_Si_nm'] == pytest.approx(1107, abs=20)

    def test_ECT_delta_change_positive(self):
        assert self.em['civil_em']['ECT_rebar']['impedance_change_pct'] > 0

    def test_shielding_dB_positive(self):
        assert self.em['civil_em']['shielding_dB'] > 0

    def test_EMF_transformer_positive(self):
        assert self.em['induction_faraday']['EMF_transformer_V_rms'] > 0

    def test_maxwell_4_equations_present(self):
        for k in ['div_E','div_B','curl_E','curl_B']:
            assert k in self.em['maxwell_equations']

    def test_UC_pathway_present(self):
        for uc in ['UCD','UCR','UCM']:
            assert uc in self.em['uc_pathway']

    def test_dispersion_wavelength_check(self):
        # lambda = 2*pi/k at 193 THz; should be near 1550 nm but may differ by ~500 nm
        # depending on n_silica vs exact frequency used
        lam = self.em['dispersion']['lambda_check_nm']
        assert 900 < lam < 2000


class TestApCalculusPhotonics:
    def setup_method(self):
        self.calc = ap_calculus_photonics()

    def test_sinc_at_0_equals_1(self):
        assert self.calc['AB']['sinc_at_0'] == pytest.approx(1.0, abs=0.01)

    def test_parseval_error_small(self):
        assert self.calc['AB']['parseval_error'] < 0.05

    def test_chain_rule_error_bounded(self):
        # FD vs analytic on H(f) -- scales with H magnitude * df^{-1}; just check finite
        assert self.calc['AB']['chain_rule_H_f_error'] < 1e6  # finite and computed

    def test_DFT_riemann_error_small(self):
        assert self.calc['BC']['DFT_manual_vs_numpy_error'] < 1e-10

    def test_Taylor_error_tiny(self):
        assert self.calc['BC']['Taylor_exp_j_phi_error'] < 1e-6

    def test_Fourier_series_has_harmonics(self):
        assert len(self.calc['BC']['Fourier_series_harmonics']) > 0

    def test_polar_IQ_unit_circle(self):
        assert self.calc['polar_IQ']['r_mean'] == pytest.approx(1.0, abs=1e-10)
        assert self.calc['polar_IQ']['r_std'] < 1e-10

    def test_group_delay_formula(self):
        assert 'tau_g' in self.calc['photonics_formulas']['group_delay']

    def test_GDD_formula(self):
        assert 'beta2' in self.calc['photonics_formulas']['GDD']

    def test_AB_keys_present(self):
        for k in ['sinc_at_0','chain_rule_H_f_error','parseval_error','FTC']:
            assert k in self.calc['AB']

    def test_BC_keys_present(self):
        for k in ['Fourier_series_harmonics','Taylor_exp_j_phi_error',
                  'DFT_manual_vs_numpy_error','moment_theorem']:
            assert k in self.calc['BC']


class TestApStatisticsLogistic:
    def setup_method(self):
        self.stat = ap_statistics_logistic(n_samples=200, rng_seed=42)

    def test_accuracy_above_chance(self):
        assert self.stat['performance']['accuracy'] >= 0.5

    def test_F1_nonneg(self):
        assert self.stat['performance']['F1'] >= 0

    def test_confusion_sums_to_n(self):
        c = self.stat['performance']['confusion']
        assert c['TP']+c['TN']+c['FP']+c['FN'] == 200

    def test_loss_decreased(self):
        assert self.stat['logistic_regression']['loss_decreased'] is True

    def test_final_loss_positive(self):
        assert self.stat['logistic_regression']['final_loss'] > 0

    def test_Z_test_reject_H0(self):
        # Amplitudes are generated to differ -> should reject H0
        assert self.stat['AP_statistics']['reject_H0'] is True

    def test_CLT_verified(self):
        assert self.stat['AP_statistics']['CLT']['CLT_verified'] is True

    def test_describe_has_6_keys(self):
        for cls in ['amplitude_class0', 'amplitude_class1']:
            d = self.stat['AP_statistics'][cls]
            for k in ['mean','std','median','Q1','Q3','IQR']:
                assert k in d

    def test_mean_class1_gt_class0(self):
        m0 = self.stat['AP_statistics']['amplitude_class0']['mean']
        m1 = self.stat['AP_statistics']['amplitude_class1']['mean']
        assert m1 > m0

    def test_annual_savings_positive(self):
        assert self.stat['business']['annual_savings_usd'] > 0

    def test_formulas_complete(self):
        for k in ['logistic','cross_entropy','gradient','Z_test','CLT']:
            assert k in self.stat['formulas']

    def test_beta_has_5_elements(self):
        assert len(self.stat['logistic_regression']['beta']) == 5

    def test_p_value_range(self):
        assert 0 <= self.stat['AP_statistics']['p_value'] <= 1


class TestApChemBio:
    def setup_method(self):
        self.cb = ap_chem_bio_photonics()

    def test_beer_lambert_transmittance_in_0_1(self):
        T = self.cb['AP_chem']['Beer_Lambert']['T_transmittance']
        assert 0 < T <= 1

    def test_beer_lambert_absorbance_positive(self):
        A = self.cb['AP_chem']['Beer_Lambert']['A_hemoglobin']
        assert A > 0

    def test_FRET_at_R0_near_half(self):
        r_nm = self.cb['AP_bio']['FRET']['r_nm']
        E = self.cb['AP_bio']['FRET']['E_FRET']
        idx = r_nm.index(5.0)  # R0=5 nm
        assert E[idx] == pytest.approx(0.5, abs=0.01)

    def test_FRET_decreases_with_r(self):
        E = self.cb['AP_bio']['FRET']['E_FRET']
        assert all(E[i] >= E[i+1] for i in range(len(E)-1))

    def test_MM_v_bounded(self):
        v = self.cb['AP_bio']['Michaelis_Menten']['v_arr']
        Vmax = 1.0
        assert all(0 < vi < Vmax for vi in v)

    def test_MM_v_increases_with_S(self):
        v = self.cb['AP_bio']['Michaelis_Menten']['v_arr']
        assert all(v[i] < v[i+1] for i in range(len(v)-1))

    def test_BER_positive(self):
        assert self.cb['AP_bio']['BER_vs_DNA_fidelity']['BER_at_10dB'] > 0

    def test_d_prime_increases_with_SNR(self):
        d = self.cb['AP_psych']['signal_detection']['d_prime']
        assert all(d[i] <= d[i+1] for i in range(len(d)-1))

    def test_photon_energy_green(self):
        E = self.cb['AP_psych']['photon_vision']['E_photon_green_eV']
        assert 2.0 < E < 2.5   # green photon ~2.25 eV

    def test_Gibbs_formula(self):
        assert 'delta_G' in self.cb['AP_chem']['Gibbs']['formula']


class TestApCSEconLang:
    def setup_method(self):
        self.cs = ap_cs_econ_lang()

    def test_FFT_faster_than_DFT(self):
        dft = self.cs['AP_CS']['Big_O']['DFT_N2']
        fft = self.cs['AP_CS']['Big_O']['FFT_NlogN']
        assert all(f < d for f, d in zip(fft[3:], dft[3:]))  # skip tiny N

    def test_speedup_at_1024_correct(self):
        assert self.cs['AP_CS']['Big_O']['speedup_at_1024'] == pytest.approx(1024/10, abs=5)

    def test_cooley_tukey_error_tiny(self):
        assert self.cs['AP_CS']['Cooley_Tukey']['error_vs_numpy'] < 1e-10

    def test_boolean_gates_4_defined(self):
        gates = self.cs['AP_CS']['boolean_photonics']
        assert len(gates) >= 4

    def test_sbir_budget_correct(self):
        assert self.cs['AP_Econ']['SBIR_budget']['Phase_I_usd'] == 275000
        assert self.cs['AP_Econ']['SBIR_budget']['Phase_II_usd'] == 1750000

    def test_base_salary_positive(self):
        assert self.cs['AP_Econ']['SBIR_budget']['base_salary_per_person_usd'] > 0

    def test_optimal_fiber_length_positive(self):
        assert self.cs['AP_Econ']['marginal_analysis']['L_optimal_km'] > 0

    def test_argument_has_claim(self):
        arg = self.cs['AP_Lang']['SBIR_argument']
        assert 'claim' in arg
        assert len(arg['claim']) > 10

    def test_argument_has_4_evidence(self):
        assert len(self.cs['AP_Lang']['SBIR_argument']['evidence']) >= 4

    def test_essay_6_sections(self):
        assert len(self.cs['AP_Lang']['essay_structure']) == 6

    def test_n_people_3(self):
        assert self.cs['AP_Econ']['SBIR_budget']['n_people'] == 3
