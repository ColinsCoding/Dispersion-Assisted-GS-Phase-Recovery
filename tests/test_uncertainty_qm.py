"""Tests for dgs/uncertainty_qm.py"""
import math
import numpy as np
import pytest

try:
    from dgs.uncertainty_qm import (
        uncertainty_principle, photonic_vector_space,
        multiplexed_pulsed_laser, math_operations_grammar,
        statistical_resolution,
    )
except ImportError:
    from uncertainty_qm import (
        uncertainty_principle, photonic_vector_space,
        multiplexed_pulsed_laser, math_operations_grammar,
        statistical_resolution,
    )

hbar = 1.0546e-34


class TestUncertaintyPrinciple:
    def setup_method(self):
        self.up = uncertainty_principle(delta_x_nm=0.1, delta_t_ps=1.0,
                                        BW_GHz=100.0, M_stretch=10.0)

    def test_delta_p_min_correct(self):
        dx = 0.1e-9
        assert self.up['position_momentum']['delta_p_min_kg_m_s'] == pytest.approx(
            hbar/(2*dx), rel=0.01)

    def test_TBP_above_minimum(self):
        assert self.up['time_frequency']['TBP_SI'] >= self.up['time_frequency']['TBP_min'] - 1e-10

    def test_TBP_satisfied(self):
        assert self.up['time_frequency']['TBP_satisfied'] is True

    def test_TBP_conserved_after_stretch(self):
        assert self.up['time_stretch']['TBP_conserved'] is True

    def test_T_out_scaled_by_M(self):
        assert self.up['time_stretch']['T_out_ps'] == pytest.approx(
            self.up['time_stretch']['T_in_ps'] * 10.0, rel=0.01)

    def test_BW_out_divided_by_M(self):
        assert self.up['time_stretch']['BW_out_GHz'] == pytest.approx(
            self.up['time_stretch']['BW_in_GHz'] / 10.0, rel=0.01)

    def test_photon_energy_quantum_regime(self):
        assert self.up['energy_scales']['quantum_regime'] is True

    def test_E_photon_near_0_8eV(self):
        assert self.up['energy_scales']['E_photon_1550nm_eV'] == pytest.approx(0.80, abs=0.02)

    def test_wavefunction_normalized(self):
        assert self.up['wavefunction']['psi_x_norm'] == pytest.approx(1.0, abs=0.01)

    def test_wigner_is_2d(self):
        W = self.up['wigner']['W']
        assert len(W) > 0 and len(W[0]) > 0

    def test_M_stretch_lt_1_raises(self):
        with pytest.raises(ValueError):
            uncertainty_principle(M_stretch=0.5)

    def test_delta_t_negative_raises(self):
        with pytest.raises(ValueError):
            uncertainty_principle(delta_t_ps=-1.0)

    def test_stretch_improvement_equals_M(self):
        assert self.up['resolution']['stretch_improvement'] == pytest.approx(10.0)


class TestPhotonicVectorSpace:
    def setup_method(self):
        self.vs = photonic_vector_space(n_modes=8)

    def test_modes_orthonormal(self):
        assert self.vs['spatial_modes']['orthonormal'] is True

    def test_orthonormality_error_tiny(self):
        assert self.vs['spatial_modes']['orthonormality_error'] < 1e-9

    def test_H_f_unitary(self):
        assert self.vs['operators']['H_is_unitary'] is True

    def test_singular_values_all_1(self):
        assert self.vs['operators']['all_singular_values_1'] is True

    def test_p_is_hermitian(self):
        assert self.vs['operators']['p_is_hermitian'] is True

    def test_MZI_unitary(self):
        assert self.vs['MZI']['unitary'] is True

    def test_MZI_theta_45deg(self):
        assert self.vs['MZI']['theta_deg'] == pytest.approx(45.0, abs=0.01)

    def test_all_senior_vars_present(self):
        vars_ = self.vs['senior_project_variables']
        for key in ['T(omega)', 'G(t,f)', 'B(f)', 'A(Q,K,V)', 'H(f)', 'S(omega)']:
            assert key in vars_

    def test_H_f_in_all_topics(self):
        assert 'H(f)' in self.vs['senior_project_variables']

    def test_hilbert_map_complete(self):
        h_map = self.vs['hilbert_space_map']
        assert '|psi>' in h_map
        assert 'unitary U' in h_map


class TestMultiplexedPulsedLaser:
    def setup_method(self):
        self.wdm = multiplexed_pulsed_laser(
            n_channels=4, T_pulse_ps=1.0, ch_spacing_nm=10.0,
            D_ps_nm_km=1000.0, L_km=5.0
        )

    def test_n_channels_correct(self):
        assert self.wdm['system']['n_channels'] == 4

    def test_channel_wavelengths_centered(self):
        lams = self.wdm['system']['lambda_channels_nm']
        center = sum(lams) / len(lams)
        # n_channels=4 -> channels at -20,-10,0,+10 nm -> mean = 1547.5
        assert abs(center - 1550.0) < 20.0

    def test_arrival_times_ordered(self):
        t = self.wdm['time_encoding']['t_arrival_ps']
        assert all(t[i] < t[i+1] for i in range(len(t)-1))

    def test_T_out_greater_T_in(self):
        assert self.wdm['time_encoding']['T_out_ps'] > self.wdm['system']['T_pulse_ps']

    def test_TBP_conserved(self):
        assert self.wdm['TBP']['conserved'] is True

    def test_throughput_positive(self):
        assert self.wdm['throughput']['B_total_Gbps'] > 0

    def test_time_domain_length(self):
        assert len(self.wdm['time_domain']['I_total']) == 1024

    def test_D_eff_correct(self):
        assert self.wdm['system']['D_eff_ps_nm'] == pytest.approx(5000.0)


class TestMathOperationsGrammar:
    def setup_method(self):
        self.mg = math_operations_grammar(x=2.5, n=4)

    def test_addition(self):
        assert self.mg['addition'] == pytest.approx(6.5)

    def test_subtraction(self):
        assert self.mg['subtraction'] == pytest.approx(-1.5)

    def test_multiplication(self):
        assert self.mg['multiplication'] == pytest.approx(10.0)

    def test_division(self):
        assert self.mg['division'] == pytest.approx(0.625)

    def test_exponent(self):
        assert self.mg['exponent'] == pytest.approx(2.5**4)

    def test_modulo(self):
        assert self.mg['modulo'] == pytest.approx(2.5 % 4)

    def test_sqrt(self):
        assert self.mg['sqrt'] == pytest.approx(math.sqrt(2.5))

    def test_log_e(self):
        assert self.mg['log_e'] == pytest.approx(math.log(2.5))

    def test_log_2(self):
        assert self.mg['log_2'] == pytest.approx(math.log2(2.5))

    def test_exp(self):
        assert self.mg['exp'] == pytest.approx(math.exp(2.5))

    def test_sin_cos_pythagorean(self):
        s = self.mg['sin']; c = self.mg['cos']
        assert s**2 + c**2 == pytest.approx(1.0, abs=1e-12)

    def test_to_dB(self):
        assert self.mg['to_dB'] == pytest.approx(10*math.log10(2.5))

    def test_from_dB(self):
        assert self.mg['from_dB'] == pytest.approx(10**(2.5/10))

    def test_group_delay_nonzero(self):
        assert self.mg['group_delay_ps'] != 0.0

    def test_Shannon_entropy_positive(self):
        assert self.mg['Shannon_entropy_bits'] > 0

    def test_Shannon_entropy_at_most_2(self):
        # 4 symbols -> max entropy = log2(4) = 2
        assert self.mg['Shannon_entropy_bits'] <= 2.0 + 1e-9

    def test_integral_gaussian_energy_positive(self):
        assert self.mg['integral_gaussian_energy'] > 0

    def test_convolution_peak_positive(self):
        assert self.mg['convolution_peak'] > 0

    def test_n_zero_raises(self):
        with pytest.raises(ValueError):
            math_operations_grammar(x=1.0, n=0)

    def test_phase_wrapped_in_0_2pi(self):
        assert 0 <= self.mg['phase_wrapped_rad'] < 2*math.pi


class TestStatisticalResolution:
    def setup_method(self):
        self.sr = statistical_resolution(N_photons=1000, n_measurements=200,
                                          phi_true=0.75, SNR_dB=20.0, rng_seed=5)

    def test_shot_noise_limit_correct(self):
        assert self.sr['limits']['shot_noise_limit'] == pytest.approx(1/math.sqrt(1000), rel=0.01)

    def test_heisenberg_limit_correct(self):
        assert self.sr['limits']['Heisenberg_limit'] == pytest.approx(1/1000.0, rel=0.01)

    def test_heisenberg_beats_shot(self):
        assert self.sr['limits']['Heisenberg_limit'] < self.sr['limits']['shot_noise_limit']

    def test_MLE_error_small(self):
        assert self.sr['MLE']['error_rad'] < 0.5

    def test_posterior_normalized(self):
        phi_grid = np.array(self.sr['Bayesian']['phi_grid_rad'])
        posterior = np.array(self.sr['Bayesian']['posterior'])
        norm = float(np.trapezoid(posterior, phi_grid))
        assert norm == pytest.approx(1.0, abs=0.05)

    def test_posterior_std_positive(self):
        assert self.sr['Bayesian']['posterior_std_rad'] > 0

    def test_CRB_positive(self):
        assert self.sr['limits']['CRB'] > 0

    def test_resolution_table_has_7_entries(self):
        assert len(self.sr['resolution_table']) == 7

    def test_GS_connection_mentioned(self):
        assert 'GS' in self.sr['GS_connection']

    def test_speedup_gt_1(self):
        assert self.sr['limits']['speedup'] > 1
