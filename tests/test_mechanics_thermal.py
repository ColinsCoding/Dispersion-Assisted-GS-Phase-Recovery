"""Tests for dgs/mechanics_thermal.py"""
import math
import numpy as np
import pytest

try:
    from dgs.mechanics_thermal import (
        terminal_velocity, stiffness_matrix_6x6,
        qm_operators_classical, continuous_rotation_SO3,
        symmetry_integration, lagrangian_mechanics, thermal_physics,
    )
except ImportError:
    from mechanics_thermal import (
        terminal_velocity, stiffness_matrix_6x6,
        qm_operators_classical, continuous_rotation_SO3,
        symmetry_integration, lagrangian_mechanics, thermal_physics,
    )


class TestTerminalVelocity:
    def test_skydiver_terminal_positive(self):
        tv = terminal_velocity()
        assert tv['skydiver']['v_terminal_ms'] > 0

    def test_skydiver_terminal_reasonable(self):
        # Terminal velocity ~50-60 m/s for standard skydiver
        tv = terminal_velocity()
        assert 40 < tv['skydiver']['v_terminal_ms'] < 80

    def test_baumgartner_supersonic(self):
        tv = terminal_velocity()
        assert tv['Baumgartner_39km']['supersonic'] is True

    def test_baumgartner_Ma_above_1(self):
        tv = terminal_velocity()
        assert tv['Baumgartner_39km']['Ma'] > 1.0

    def test_velocity_starts_zero(self):
        tv = terminal_velocity()
        assert tv['velocity_vs_time']['v_ms'][0] == pytest.approx(0.0, abs=1e-6)

    def test_velocity_approaches_terminal(self):
        tv = terminal_velocity()
        v = np.array(tv['velocity_vs_time']['v_ms'])
        v_t = tv['velocity_vs_time']['v_terminal_ms']
        assert v[-1] > 0.9 * v_t

    def test_ice_friction_less_than_wheel(self):
        tv = terminal_velocity()
        assert tv['skating']['F_friction_ice_N'] < tv['skating']['F_friction_wheels_N']

    def test_Reynolds_keys(self):
        tv = terminal_velocity()
        assert 'Stokes' in tv['Reynolds'] and 'Quadratic' in tv['Reynolds']

    def test_Re_terminal_turbulent(self):
        tv = terminal_velocity()
        assert tv['skydiver']['Re_terminal'] > 1000


class TestStiffnessMatrix:
    def test_K_symmetric(self):
        K6 = stiffness_matrix_6x6()
        assert K6['is_symmetric'] is True

    def test_K_shape_6x6(self):
        K6 = stiffness_matrix_6x6()
        assert np.array(K6['K']).shape == (6, 6)

    def test_all_eigenvalues_positive(self):
        K6 = stiffness_matrix_6x6()
        evals = K6['eigenvalues']
        assert all(v > 0 for v in evals)

    def test_eigenvalues_sorted_ascending(self):
        K6 = stiffness_matrix_6x6()
        evals = K6['eigenvalues']
        assert evals == sorted(evals)

    def test_condition_number_large(self):
        # Stiffness matrix should be ill-conditioned (axial >> bending)
        K6 = stiffness_matrix_6x6()
        assert K6['condition_number'] > 100

    def test_feature_discovery_keys(self):
        K6 = stiffness_matrix_6x6()
        fd = K6['feature_discovery']
        assert 'most_compliant' in fd and 'most_stiff' in fd

    def test_natural_frequencies_positive(self):
        K6 = stiffness_matrix_6x6()
        freqs = K6['natural_frequencies_Hz']
        assert all(f >= 0 for f in freqs)

    def test_PCA_connection_mentioned(self):
        K6 = stiffness_matrix_6x6()
        assert 'PCA' in K6['connection']['PCA'] or 'principal' in K6['connection']['PCA'].lower()


class TestQMOperators:
    def test_X_Hermitian(self):
        qm = qm_operators_classical()
        assert qm['operators']['X_Hermitian'] is True

    def test_P_Hermitian(self):
        qm = qm_operators_classical()
        assert qm['operators']['P_Hermitian'] is True

    def test_commutator_XP_equals_jhbar(self):
        qm = qm_operators_classical()
        assert qm['commutator']['[X,P]_equals_jhbar_I'] is True

    def test_Heisenberg_satisfied(self):
        qm = qm_operators_classical()
        assert qm['commutator']['Heisenberg_satisfied'] is True

    def test_uncertainty_product_near_half(self):
        qm = qm_operators_classical()
        # Ground state of HO saturates uncertainty: Delta_x*Delta_p = hbar/2
        product = qm['commutator']['product_hbar_units']
        assert product == pytest.approx(0.5, abs=1e-4)

    def test_GS_iteration_convergence(self):
        qm = qm_operators_classical()
        assert qm['GS_iteration']['converges'] is True

    def test_GS_spectral_radius_less_than_1(self):
        qm = qm_operators_classical()
        assert qm['GS_iteration']['spectral_radius'] < 1.0

    def test_expectation_x_zero_ground_state(self):
        qm = qm_operators_classical()
        assert qm['expectation_values']['<x>_ground'] == pytest.approx(0.0, abs=1e-10)

    def test_expectation_p_zero_ground_state(self):
        qm = qm_operators_classical()
        assert qm['expectation_values']['<p>_ground'] == pytest.approx(0.0, abs=1e-10)


class TestContinuousRotationSO3:
    def test_commutation_correct(self):
        rot = continuous_rotation_SO3()
        assert rot['SO3_generators']['commutation_correct'] is True

    def test_Rodrigues_is_SO3(self):
        rot = continuous_rotation_SO3()
        assert rot['rodrigues_rotation']['is_SO3'] is True

    def test_trace_at_0_is_3(self):
        rot = continuous_rotation_SO3()
        trace = np.array(rot['rotation_Rz']['trace'])
        assert trace[0] == pytest.approx(3.0, abs=0.01)

    def test_trace_at_pi_is_minus1(self):
        rot = continuous_rotation_SO3()
        theta = np.array(rot['rotation_Rz']['theta_rad'])
        trace = np.array(rot['rotation_Rz']['trace'])
        idx = np.argmin(np.abs(theta - np.pi))
        assert trace[idx] == pytest.approx(-1.0, abs=0.05)

    def test_skater_spins_faster_arms_in(self):
        rot = continuous_rotation_SO3()
        assert rot['skater_spin']['omega_arms_in_rps'] > rot['skater_spin']['omega_initial_rps']

    def test_orthogonality_n_ne_m(self):
        rot = continuous_rotation_SO3()
        assert abs(rot['orthogonality']['integral_sin_n_sin_m_n_ne_m']) < 0.01

    def test_orthogonality_n_eq_m(self):
        rot = continuous_rotation_SO3()
        assert rot['orthogonality']['integral_sin_n_sin_n'] == pytest.approx(np.pi, abs=0.01)

    def test_Lie_group_connection(self):
        rot = continuous_rotation_SO3()
        assert 'exp' in rot['Lie_group_connection']['SO3']
        assert 'H(f)' in rot['Lie_group_connection']['H_f']

    def test_sinc_at_zero_is_one(self):
        rot = continuous_rotation_SO3()
        assert rot['piecewise_functions']['sinc_at_0'] == pytest.approx(1.0, abs=1e-10)


class TestSymmetryIntegration:
    def test_integral_sin_zero(self):
        sym = symmetry_integration()
        assert abs(sym['odd_functions']['integrals']['int_sin_-pi_pi']) < 1e-6

    def test_integral_x3_zero(self):
        sym = symmetry_integration()
        assert abs(sym['odd_functions']['integrals']['int_x_-pi_pi']) < 0.01

    def test_integral_cos_small(self):
        sym = symmetry_integration()
        # int cos over [-pi, pi] = 2*sin(pi) = 0
        assert abs(sym['even_functions']['integrals']['int_cos_-pi_pi']) < 0.01

    def test_integral_sinc_near_pi(self):
        sym = symmetry_integration()
        # int_{-pi}^{pi} sinc(x) dx = int_{-pi}^{pi} sin(x)/x dx ~ pi (converges)
        assert sym['even_functions']['integrals']['int_sinc_-pi_pi'] > 2.0

    def test_statics_net_moment_zero(self):
        sym = symmetry_integration()
        assert sym['statics_symmetry']['is_zero'] is True

    def test_GS_intensity_is_even(self):
        sym = symmetry_integration()
        assert sym['GS_connection']['intensity_is_even'] is True

    def test_odd_phase_invisible(self):
        sym = symmetry_integration()
        assert sym['GS_connection']['phase_odd_part_invisible'] is True

    def test_Dirac_delta_integral(self):
        sym = symmetry_integration()
        assert sym['piecewise_integration']['Dirac_delta_integral'] == pytest.approx(1.0)

    def test_Dirac_delta_even(self):
        sym = symmetry_integration()
        assert sym['piecewise_integration']['Dirac_delta_even'] is True


class TestLagrangianMechanics:
    def test_pendulum_omega_correct(self):
        lag = lagrangian_mechanics()
        g_val = 9.81; L_pend = 1.0
        expected = math.sqrt(g_val/L_pend)
        assert lag['pendulum']['omega_0_rad_s'] == pytest.approx(expected, rel=1e-6)

    def test_energy_conserved_RK4(self):
        lag = lagrangian_mechanics()
        assert lag['pendulum']['energy_conserved'] is True

    def test_equilibria_found(self):
        lag = lagrangian_mechanics()
        assert len(lag['statics_equilibria']['equilibrium_x']) >= 2

    def test_stable_unstable_mix(self):
        lag = lagrangian_mechanics()
        stable = lag['statics_equilibria']['stable']
        assert True in stable and False in stable

    def test_separatrix_energy_positive(self):
        lag = lagrangian_mechanics()
        assert lag['Hamiltonian']['separatrix_energy'] > 0

    def test_EL_equation_string_present(self):
        lag = lagrangian_mechanics()
        assert 'dL/dq' in lag['EL_equations']['general']

    def test_statics_limit_dV_zero(self):
        lag = lagrangian_mechanics()
        assert 'dV/dq' in lag['EL_equations']['statics_limit'] or 'equilibrium' in lag['EL_equations']['statics_limit']

    def test_Hamiltonian_grid_shape(self):
        lag = lagrangian_mechanics()
        H_grid = np.array(lag['Hamiltonian']['H_pendulum'])
        assert H_grid.shape[0] > 1 and H_grid.shape[1] > 1


class TestThermalPhysics:
    def test_Carnot_efficiency_range(self):
        th = thermal_physics()
        eta = th['thermodynamics']['Carnot']['eta_at_1000K']
        assert 0 < eta < 1

    def test_Carnot_eta_at_1000K_near_0p7(self):
        th = thermal_physics()
        assert th['thermodynamics']['Carnot']['eta_at_1000K'] == pytest.approx(0.7, abs=0.01)

    def test_sun_peak_visible(self):
        th = thermal_physics()
        lam = th['blackbody']['lambda_peak_sun_nm']
        assert 450 < lam < 600

    def test_kT_room_near_25meV(self):
        th = thermal_physics()
        assert th['stat_mech']['kT_at_room_eV'] == pytest.approx(0.0257, abs=0.002)

    def test_1550nm_quantum_limited(self):
        th = thermal_physics()
        assert th['photon_noise']['quantum_limited'] is True

    def test_hf_kT_1550_large(self):
        th = thermal_physics()
        assert th['photon_noise']['hf_over_kT_at_1550nm'] > 30

    def test_Si_thermal_phase_positive(self):
        th = thermal_physics()
        assert th['Si_photonics_thermal']['Delta_phi_per_K_rad'] > 0

    def test_Boltzmann_Shannon_connection(self):
        th = thermal_physics()
        assert 'Shannon' in th['building_future']['Boltzmann_to_Shannon']

    def test_4_thermodynamic_laws(self):
        th = thermal_physics()
        laws = th['thermodynamics']['laws']
        for k in ['0th', '1st', '2nd', '3rd']:
            assert k in laws
