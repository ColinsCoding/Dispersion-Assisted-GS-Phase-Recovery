"""Tests for dgs/rf_microwave.py"""
import math
import numpy as np
import pytest

try:
    from dgs.rf_microwave import (
        wire_cross_section, transmission_line,
        pid_controller, gradient_descent_four_ways,
        nuclear_macroscopic_cross_sections,
    )
except ImportError:
    from rf_microwave import (
        wire_cross_section, transmission_line,
        pid_controller, gradient_descent_four_ways,
        nuclear_macroscopic_cross_sections,
    )


class TestWireCrossSection:
    def test_J_DC_from_geometry(self):
        w = wire_cross_section(radius_mm=1.0, current_A=1.0)
        a = 1e-3
        expected_J = 1.0 / (np.pi*a**2)
        assert w['geometry']['J_DC_A_m2'] == pytest.approx(expected_J, rel=1e-4)

    def test_H_at_surface_correct(self):
        w = wire_cross_section(radius_mm=1.0, current_A=1.0)
        a = 1e-3; I = 1.0
        expected_H = I/(2*np.pi*a)
        assert w['H_field']['H_at_surface'] == pytest.approx(expected_H, rel=1e-4)

    def test_I_from_area_integral(self):
        # Area integral of J over cross section should return I
        w = wire_cross_section(radius_mm=1.0, current_A=1.0)
        assert w['area_integrals']['I_from_area_integral_A'] == pytest.approx(1.0, rel=0.05)

    def test_skin_depth_decreases_with_freq(self):
        w = wire_cross_section()
        delta = np.array(w['skin_effect']['delta_mm'])
        assert delta[-1] < delta[0]

    def test_R_AC_increases_at_high_freq(self):
        w = wire_cross_section()
        R = np.array(w['skin_effect']['R_AC_per_m'])
        assert R[-1] > R[0]

    def test_L_int_per_m_formula(self):
        # L_int = mu0/(8*pi) per unit length
        w = wire_cross_section()
        expected = 4*np.pi*1e-7 / (8*np.pi)
        assert w['area_integrals']['L_int_per_m_H'] == pytest.approx(expected, rel=1e-4)

    def test_GS_connection_present(self):
        w = wire_cross_section()
        assert 'GS' in w['GS_connection']['total_power'] or 'integral' in w['GS_connection']['total_power']


class TestTransmissionLine:
    def test_matched_load_zero_reflection(self):
        tl = transmission_line(Z0_line=50, Z_load=50, f_GHz=1.0)
        assert tl['reflection']['Gamma_mag'] == pytest.approx(0.0, abs=1e-10)

    def test_VSWR_one_for_matched(self):
        tl = transmission_line(Z0_line=50, Z_load=50, f_GHz=1.0)
        assert tl['reflection']['VSWR'] == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_VSWR_greater_than_1(self):
        tl = transmission_line(Z0_line=50, Z_load=75, f_GHz=2.4)
        assert tl['reflection']['VSWR'] > 1.0

    def test_Gamma_formula(self):
        tl = transmission_line(Z0_line=50, Z_load=75, f_GHz=1.0)
        expected = abs((75-50)/(75+50))
        assert tl['reflection']['Gamma_mag'] == pytest.approx(expected, rel=1e-6)

    def test_QWT_impedance(self):
        tl = transmission_line(Z0_line=50, Z_load=75, f_GHz=1.0)
        expected_Z0_QWT = math.sqrt(50*75)
        assert tl['QWT']['Z0_QWT_ohm'] == pytest.approx(expected_Z0_QWT, rel=1e-4)

    def test_S21_gain_correct(self):
        tl = transmission_line()
        assert tl['S_params']['S21_dB'] == pytest.approx(15.0, abs=0.1)

    def test_stability_Rollett(self):
        tl = transmission_line()
        assert tl['S_params']['unconditionally_stable'] is True

    def test_cavity_resonance_at_5GHz(self):
        tl = transmission_line()
        f = np.array(tl['resonant_cavity']['f_GHz'])
        S21 = np.array(tl['resonant_cavity']['S21_dB'])
        idx_peak = np.argmax(S21)
        assert f[idx_peak] == pytest.approx(5.0, abs=0.1)


class TestPIDController:
    def test_PI_eliminates_SS_error(self):
        pid = pid_controller()
        assert abs(pid['step_response']['ss_error_PI']) < 0.01

    def test_P_has_SS_error(self):
        pid = pid_controller()
        assert abs(pid['step_response']['ss_error_P']) > 0.01

    def test_GD_converges(self):
        pid = pid_controller()
        assert pid['GD_as_P_controller']['converged'] is True

    def test_GD_final_near_target(self):
        pid = pid_controller()
        assert pid['GD_as_P_controller']['final_theta'] == pytest.approx(3.0, abs=0.01)

    def test_phase_margin_positive(self):
        pid = pid_controller()
        assert pid['stability']['phase_margin_deg'] > 0

    def test_GS_is_feedback_loop(self):
        pid = pid_controller()
        assert pid['topology']['GS_is_feedback_loop'] is True

    def test_optimizer_analogy_keys(self):
        pid = pid_controller()
        oa = pid['optimizer_PID_analogy']
        assert 'SGD' in oa and 'Momentum' in oa and 'Adam' in oa and 'GS' in oa

    def test_topology_has_cycle(self):
        pid = pid_controller()
        assert pid['topology']['cycles'] == 1


class TestGradientDescentFourWays:
    def test_python_converges(self):
        gd = gradient_descent_four_ways()
        assert gd['python_pure']['error_vs_x_true'] < 0.1

    def test_numpy_converges(self):
        gd = gradient_descent_four_ways()
        assert gd['numpy']['error_vs_x_true'] < 0.1

    def test_all_same_result(self):
        gd = gradient_descent_four_ways()
        assert gd['all_same_result'] is True

    def test_exact_solution_smallest_error(self):
        gd = gradient_descent_four_ways()
        assert gd['exact']['error_vs_x_true'] < gd['numpy']['error_vs_x_true'] + 0.05

    def test_C_pseudocode_is_string(self):
        gd = gradient_descent_four_ways()
        assert isinstance(gd['C_pseudocode'], str)
        assert 'for' in gd['C_pseudocode']
        assert 'grad' in gd['C_pseudocode']

    def test_loss_decreases(self):
        gd = gradient_descent_four_ways()
        loss = gd['numpy']['loss_history']
        assert loss[-1] < loss[0]

    def test_GS_connection_present(self):
        gd = gradient_descent_four_ways()
        assert 'I_meas' in gd['GS_connection']['GS_loss'] or 'GS' in gd['GS_connection']['GS_loss']

    def test_alpha_optimal_positive(self):
        gd = gradient_descent_four_ways()
        assert gd['problem']['alpha_optimal'] > 0

    def test_problem_shape_correct(self):
        gd = gradient_descent_four_ways()
        assert gd['problem']['A_shape'] == [20, 5]


class TestNuclearCrossSections:
    def test_U235_thermal_cross_section(self):
        nc = nuclear_macroscopic_cross_sections()
        # U-235 thermal fission cross section ~582 barns
        assert nc['U235']['sigma_thermal_barns'] == pytest.approx(582, rel=0.01)

    def test_macroscopic_positive(self):
        nc = nuclear_macroscopic_cross_sections()
        assert nc['UO2_fuel']['Sigma_a_thermal_per_cm'] > 0

    def test_Gaussian_beam_integral_accurate(self):
        nc = nuclear_macroscopic_cross_sections()
        assert nc['Gaussian_beam_integral']['relative_error'] < 0.02

    def test_Gaussian_numerical_near_exact(self):
        nc = nuclear_macroscopic_cross_sections()
        num = nc['Gaussian_beam_integral']['Phi_numerical']
        exact = nc['Gaussian_beam_integral']['Phi_exact']
        assert num == pytest.approx(exact, rel=0.02)

    def test_Breit_Wigner_peak_at_resonance(self):
        nc = nuclear_macroscopic_cross_sections()
        E = np.array(nc['U235']['E_eV'])
        sigma_BW = np.array(nc['Breit_Wigner_resonance']['sigma_BW_barns'])
        E_res = nc['Breit_Wigner_resonance']['E_res_eV']
        idx_peak = np.argmax(sigma_BW)
        assert E[idx_peak] == pytest.approx(E_res, rel=0.1)

    def test_analogy_resonator(self):
        nc = nuclear_macroscopic_cross_sections()
        assert 'resonator' in nc['Breit_Wigner_resonance']['analogy'].lower()

    def test_connection_keys(self):
        nc = nuclear_macroscopic_cross_sections()
        for key in ['neutron_transport', 'optical', 'QM', 'GS', 'area_integral']:
            assert key in nc['connections']

    def test_flux_profile_positive(self):
        nc = nuclear_macroscopic_cross_sections()
        phi = np.array(nc['flux_profile']['phi_r'])
        assert np.all(phi >= 0)
