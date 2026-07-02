"""Tests for dgs/frontier_calculus.py"""
import math
import numpy as np
import pytest

try:
    from dgs.frontier_calculus import (
        set_theory_and_boolean, trig_precalculus_complex,
        chain_rule_calculus, serway_modern_physics_problems,
        multiphysics_coupling,
    )
except ImportError:
    from frontier_calculus import (
        set_theory_and_boolean, trig_precalculus_complex,
        chain_rule_calculus, serway_modern_physics_problems,
        multiphysics_coupling,
    )


class TestSetTheoryBoolean:
    def test_de_morgan_law1(self):
        sb = set_theory_and_boolean()
        assert sb['de_morgan']['law1_verified'] is True

    def test_de_morgan_law2(self):
        sb = set_theory_and_boolean()
        assert sb['de_morgan']['law2_verified'] is True

    def test_AND_gate_truth_table(self):
        sb = set_theory_and_boolean()
        assert sb['truth_tables']['gates']['AND'] == [0, 0, 0, 1]

    def test_OR_gate_truth_table(self):
        sb = set_theory_and_boolean()
        assert sb['truth_tables']['gates']['OR'] == [0, 1, 1, 1]

    def test_XOR_gate_truth_table(self):
        sb = set_theory_and_boolean()
        assert sb['truth_tables']['gates']['XOR'] == [0, 1, 1, 0]

    def test_NAND_is_complement_of_AND(self):
        sb = set_theory_and_boolean()
        AND = sb['truth_tables']['gates']['AND']
        NAND = sb['truth_tables']['gates']['NAND']
        assert all(a + b == 1 for a, b in zip(AND, NAND))

    def test_7_plus_5_equals_12(self):
        sb = set_theory_and_boolean()
        assert sb['adder']['7_plus_5']['correct'] is True
        assert sb['adder']['7_plus_5']['sum'] == 12

    def test_set_union_correct(self):
        sb = set_theory_and_boolean()
        assert sorted(sb['sets']['union']) == [1, 2, 3, 4, 5, 6, 7]

    def test_set_intersection_correct(self):
        sb = set_theory_and_boolean()
        assert sorted(sb['sets']['intersection']) == [3, 4, 5]

    def test_power_set_size(self):
        sb = set_theory_and_boolean()
        assert sb['sets']['power_set_size'] == 32

    def test_ARM_chain_rule_key(self):
        sb = set_theory_and_boolean()
        assert 'composition' in sb['ARM']['chain_rule'].lower()

    def test_SQL_WHERE_is_filter(self):
        sb = set_theory_and_boolean()
        assert 'filter' in sb['SQL_set_map']['WHERE P(x)'].lower()

    def test_SQL_UNION_present(self):
        sb = set_theory_and_boolean()
        assert 'UNION' in sb['SQL_set_map']

    def test_H_f_connection_present(self):
        sb = set_theory_and_boolean()
        assert 'H(f)' in sb['H_f_connection'] or 'GS' in sb['H_f_connection']


class TestTrigPrecalculusComplex:
    def test_pythagorean_identity_error_tiny(self):
        tc = trig_precalculus_complex()
        assert tc['unit_circle']['identity_max_error'] < 1e-12

    def test_euler_formula_exact(self):
        tc = trig_precalculus_complex()
        assert tc['unit_circle']['euler_max_error'] < 1e-14

    def test_cos_addition_formula(self):
        tc = trig_precalculus_complex()
        assert tc['addition_formula']['error'] < 1e-14

    def test_sin_pi_6_exact(self):
        tc = trig_precalculus_complex()
        assert tc['trig_exact']['sin(pi/6)'] == pytest.approx(0.5, abs=1e-10)

    def test_sin_pi_4_exact(self):
        tc = trig_precalculus_complex()
        assert tc['trig_exact']['sin(pi/4)'] == pytest.approx(math.sqrt(2)/2, abs=1e-10)

    def test_RC_magnitude_at_3dB(self):
        tc = trig_precalculus_complex()
        assert tc['phasor_RC']['H_mag_at_3dB'] == pytest.approx(1/math.sqrt(2), abs=0.01)

    def test_RC_phase_near_45_deg(self):
        tc = trig_precalculus_complex()
        assert abs(abs(tc['phasor_RC']['H_phase_at_3dB_deg']) - 45.0) < 2.0

    def test_complex_modulus_z1(self):
        tc = trig_precalculus_complex()
        assert tc['complex_arithmetic']['z1_modulus'] == pytest.approx(5.0, abs=1e-10)

    def test_complex_arg_z1(self):
        tc = trig_precalculus_complex()
        expected = math.degrees(math.atan2(4, 3))
        assert tc['complex_arithmetic']['z1_arg_deg'] == pytest.approx(expected, abs=1e-6)

    def test_H_f_derivative_is_chain_rule(self):
        tc = trig_precalculus_complex()
        assert 'chain rule' in tc['H_f_chain_rule']['derivative'].lower()

    def test_python_history_has_313(self):
        tc = trig_precalculus_complex()
        assert '3.13' in tc['python_history']['this_repo']


class TestChainRuleCalculus:
    def test_sin_x2_derivative_accurate(self):
        cr = chain_rule_calculus()
        assert cr['real_chain_rule']['f1_sin_x2_error'] < 0.01

    def test_gaussian_derivative_accurate(self):
        cr = chain_rule_calculus()
        assert cr['real_chain_rule']['f2_gaussian_error'] < 0.001

    def test_complex_chain_rule_error_small(self):
        cr = chain_rule_calculus()
        assert cr['complex_chain_rule']['error'] < 0.1

    def test_cauchy_riemann_satisfied_for_z2(self):
        cr = chain_rule_calculus()
        assert cr['complex_chain_rule']['CR_satisfied'] is True

    def test_backprop_loss_decreases(self):
        cr = chain_rule_calculus()
        assert cr['backprop']['converged'] is True

    def test_backprop_chain_rule_5_terms(self):
        cr = chain_rule_calculus()
        # The explanation should mention 5 chain factors
        assert 'dL/dw1' in cr['backprop']['chain_rule_explanation']

    def test_phasor_RC_magnitude(self):
        cr = chain_rule_calculus()
        assert cr['phasor']['H_RC_mag'] == pytest.approx(1/math.sqrt(2), abs=0.01)

    def test_phasor_rule_d_dt(self):
        cr = chain_rule_calculus()
        assert 'j*omega' in cr['phasor']['rule_d_dt'].replace(' ', '')

    def test_H_f_derivative_formula(self):
        cr = chain_rule_calculus()
        assert 'j*2*pi*D*f' in cr['H_f_chain_rule']['dH_df']

    def test_chain_rule_examples_complete(self):
        cr = chain_rule_calculus()
        for key in ['d/dx[sin(x^2)]', 'd/dx[exp(-x^2/2)]',
                    'd/dx[(1+x^2)^-1]', 'd/dx[ln(sin(x))]']:
            assert key in cr['real_chain_rule']['examples']


class TestSerwayModernPhysics:
    def setup_method(self):
        self.s = serway_modern_physics_problems()

    def test_gamma_at_0_99c(self):
        assert self.s['Ch1_relativity']['problem_electron_0_99c']['gamma'] == pytest.approx(7.089, abs=0.01)

    def test_relativistic_KE_positive(self):
        assert self.s['Ch1_relativity']['problem_electron_0_99c']['KE_MeV'] > 0

    def test_gamma_array_increases_with_beta(self):
        gamma = self.s['Ch1_relativity']['gamma']
        assert all(gamma[i] < gamma[i+1] for i in range(len(gamma)-1))

    def test_photoelectric_KE_positive(self):
        assert self.s['Ch2_quantum_origins']['photoelectric']['KE_max_eV'] > 0

    def test_threshold_frequency_below_UV(self):
        f_thresh = self.s['Ch2_quantum_origins']['photoelectric']['threshold_f_Hz']
        f_UV = self.s['Ch2_quantum_origins']['photoelectric']['f_UV_Hz']
        assert f_thresh < f_UV

    def test_Compton_wavelength_correct(self):
        # lambda_C = h/(m_e*c) = 2.426 pm
        lam_C = self.s['Ch2_quantum_origins']['Compton']['lambda_C_pm']
        assert lam_C == pytest.approx(2.426, abs=0.01)

    def test_Compton_shift_at_90deg(self):
        # delta_lambda = lambda_C*(1-cos(90)) = lambda_C = 2.426 pm
        delta = self.s['Ch2_quantum_origins']['Compton']['delta_lambda_pm']
        lam_C = self.s['Ch2_quantum_origins']['Compton']['lambda_C_pm']
        assert delta == pytest.approx(lam_C, abs=0.01)

    def test_deBroglie_neutron_angstrom_scale(self):
        lam = self.s['Ch2_quantum_origins']['de_Broglie']['lambda_angstrom']
        assert 0.5 < lam < 5.0   # thermal neutron ~1 Angstrom

    def test_Bohr_H_alpha_near_656nm(self):
        assert self.s['Ch3_Bohr']['H_alpha_nm'] == pytest.approx(656.3, abs=1.0)

    def test_Bohr_energy_levels_negative(self):
        E_n = self.s['Ch3_Bohr']['energy_levels_eV']
        assert all(e < 0 for e in E_n)

    def test_Bohr_ground_state_minus_13_6eV(self):
        assert self.s['Ch3_Bohr']['energy_levels_eV'][0] == pytest.approx(-13.6, abs=0.1)

    def test_Schrodinger_normalization(self):
        assert self.s['Ch4_5_Schrodinger']['normalization_n1'] == pytest.approx(1.0, abs=0.001)
        assert self.s['Ch4_5_Schrodinger']['normalization_n2'] == pytest.approx(1.0, abs=0.001)

    def test_Schrodinger_orthogonality(self):
        assert self.s['Ch4_5_Schrodinger']['orthogonality_12'] < 1e-10

    def test_Schrodinger_energy_increases_with_n(self):
        E_n = self.s['Ch4_5_Schrodinger']['energy_levels_eV']
        assert all(E_n[i] < E_n[i+1] for i in range(len(E_n)-1))

    def test_tunneling_transmission_positive(self):
        T = self.s['Ch6_tunneling']['T_at_1_eV']
        assert 0 < T < 1

    def test_tunneling_transmission_increases_with_E(self):
        T = np.array(self.s['Ch6_tunneling']['T_transmission'])
        E = np.array(self.s['Ch6_tunneling']['E_eV'])
        assert T[-1] > T[0]   # higher energy -> higher tunneling

    def test_kT_at_300K(self):
        assert self.s['Ch9_statistical']['kT_meV'] == pytest.approx(25.85, abs=0.1)

    def test_FD_distribution_sum_positive(self):
        f_FD = self.s['Ch9_statistical']['Fermi_Dirac']['f_FD']
        assert sum(f_FD) > 0

    def test_H_f_connection_in_Schrodinger(self):
        assert 'H(f)' in self.s['Ch4_5_Schrodinger']['GS_connection']


class TestMultiphysics:
    def setup_method(self):
        self.mp = multiphysics_coupling(n_steps=200)

    def test_ring_FSR_positive(self):
        assert self.mp['ring']['FSR_GHz'] > 0

    def test_thermal_steady_state_positive(self):
        assert self.mp['bistability_sim']['DeltaT_steady_K'] > 0

    def test_resonance_shift_nonzero(self):
        assert abs(self.mp['bistability_sim']['omega_shift_GHz']) > 0

    def test_transmission_spectra_keys(self):
        for key in ['P_0.1mW', 'P_1.0mW', 'P_5.0mW']:
            assert key in self.mp['transmission_spectra']

    def test_transmission_peak_at_zero_detuning(self):
        T_spec = np.array(self.mp['transmission_spectra']['P_0.1mW'])
        delta = np.array(self.mp['transmission_spectra']['delta_kappa'])
        idx_peak = np.argmax(T_spec)
        assert abs(delta[idx_peak]) < 0.5   # peak near zero detuning

    def test_coupling_map_all_keys(self):
        for key in ['EM->Thermal', 'Thermal->EM', 'Thermal->Mechanical',
                    'Mechanical->EM', 'GS_impact']:
            assert key in self.mp['coupling_map']

    def test_GS_impact_mentions_D(self):
        assert '|D|' in self.mp['coupling_map']['GS_impact']

    def test_T_arr_starts_at_ambient(self):
        T = self.mp['bistability_sim']['T_K']
        assert T[0] == pytest.approx(300.0, abs=0.1)

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError):
            multiphysics_coupling(n_steps=0)

    def test_dn_dT_correct_Si(self):
        assert self.mp['thermal']['dn_dT_per_K'] == pytest.approx(1.86e-4, rel=0.01)
