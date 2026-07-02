"""Tests for dgs/photon_interactions.py"""
import math
import numpy as np
import pytest

try:
    from dgs.photon_interactions import (
        einstein_coefficients,
        nonlinear_photon_interactions,
        photon_propagator_in_medium,
        feynman_diagram_tdgsa,
        capacitance_voltage,
        complex_line_integrals_to_electrodynamics,
    )
except ImportError:
    from photon_interactions import (
        einstein_coefficients,
        nonlinear_photon_interactions,
        photon_propagator_in_medium,
        feynman_diagram_tdgsa,
        capacitance_voltage,
        complex_line_integrals_to_electrodynamics,
    )


class TestEinsteinCoefficients:
    def test_A21_hydrogen_positive(self):
        r = einstein_coefficients()
        assert r['hydrogen']['A21_per_s'] > 0

    def test_tau_sp_hydrogen_ns_range(self):
        r = einstein_coefficients()
        # Hydrogen 2p->1s: ~1.6 ns
        assert 0.5 < r['hydrogen']['tau_sp_ns'] < 5.0

    def test_EDFA_gain_positive(self):
        r = einstein_coefficients()
        assert r['EDFA']['gain_dB_10m'] > 0

    def test_EDFA_wavelengths(self):
        r = einstein_coefficients()
        assert r['EDFA']['lambda_signal_nm'] == 1550
        assert r['EDFA']['lambda_pump_nm'] == 980

    def test_rate_eq_inversion(self):
        r = einstein_coefficients()
        inv = r['rate_eq']['steady_state_inversion']
        assert 0 < inv < 1.0

    def test_vertex_count_keys(self):
        r = einstein_coefficients()
        vc = r['vertex_count']
        assert 'absorption' in vc and 'stimulated' in vc and 'spontaneous' in vc

    def test_stimulated_same_phase(self):
        r = einstein_coefficients()
        assert 'phase' in r['vertex_count']['stimulated'].lower() or 'gain' in r['vertex_count']['stimulated'].lower()


class TestNonlinearInteractions:
    def test_gamma_fiber_positive(self):
        r = nonlinear_photon_interactions()
        assert r['SPM']['gamma_fiber_per_W_km'] > 0

    def test_SPM_phi_increases_with_power(self):
        r = nonlinear_photon_interactions()
        phi = np.array(r['SPM']['phi_NL_rad']['L=100km'])
        assert phi[-1] > phi[0]

    def test_SPDC_energy_conserved(self):
        r = nonlinear_photon_interactions()
        assert r['SPDC']['energy_conserved'] is True

    def test_SPDC_entanglement_string(self):
        r = nonlinear_photon_interactions()
        assert 'entangle' in r['SPDC']['entanglement'].lower()

    def test_Kerr_vertex_count_4(self):
        r = nonlinear_photon_interactions()
        assert r['Kerr']['vertex_count'] == 4

    def test_SHG_vertex_count_3(self):
        r = nonlinear_photon_interactions()
        assert r['SHG']['vertex_count'] == 3

    def test_Kerr_phi_increases_with_I(self):
        r = nonlinear_photon_interactions()
        phi = np.array(r['Kerr']['phi_Kerr_rad'])
        assert phi[-1] > phi[0]


class TestPhotonPropagator:
    def test_ZDW_silica_near_1300nm(self):
        r = photon_propagator_in_medium()
        # ZDW of fused silica ~1270 nm
        assert 1200 < r['Sellmeier']['ZDW_nm'] < 1400

    def test_n_at_1550nm_glass(self):
        r = photon_propagator_in_medium()
        # n(silica, 1550 nm) ~ 1.444
        assert 1.40 < r['Sellmeier']['n_at_1550nm'] < 1.50

    def test_H_f_magnitude_is_1(self):
        r = photon_propagator_in_medium()
        H_mag = np.array(r['H_f']['H_mag'])
        # Dispersive propagator is lossless: |H(f)| = 1
        assert np.allclose(H_mag, 1.0, atol=1e-6)

    def test_Taylor_syntax_keys(self):
        r = photon_propagator_in_medium()
        ts = r['Taylor_syntax']['vertex_count']
        assert '0' in ts and '1' in ts and '2' in ts

    def test_2nd_order_is_this_repo(self):
        r = photon_propagator_in_medium()
        assert 'GVD' in r['Taylor_syntax']['2nd_order'] or 'REPO' in r['Taylor_syntax']['2nd_order'].upper()

    def test_beta2_negative_SMF28(self):
        r = photon_propagator_in_medium()
        # SMF-28 has anomalous dispersion at 1550 nm (D=-17 ps/nm/km -> beta2<0)
        assert r['H_f']['beta2_s2_m'] < 0


class TestFeynmanDiagramTDGSA:
    def test_returns_base64_or_path(self):
        result = feynman_diagram_tdgsa()
        # Should return base64 string
        assert isinstance(result, str) and len(result) > 100

    def test_save_to_file(self, tmp_path):
        path = str(tmp_path / 'test_feynman.png')
        result = feynman_diagram_tdgsa(save_path=path)
        import os
        assert os.path.exists(path)


class TestCapacitanceVoltage:
    def test_V_pi_positive(self):
        r = capacitance_voltage()
        assert r['MZI_modulator']['V_pi_volts'] > 0

    def test_null_at_V_pi(self):
        r = capacitance_voltage()
        assert r['MZI_modulator']['null_at_V_pi'] is True

    def test_quadrature_at_half_Vpi(self):
        r = capacitance_voltage()
        assert r['MZI_modulator']['quadrature_at_Vpi_over_2'] is True

    def test_P_out_max_at_zero_voltage(self):
        r = capacitance_voltage()
        V = np.array(r['MZI_modulator']['V_mod'])
        P = np.array(r['MZI_modulator']['P_out_normalized'])
        idx_zero = np.argmin(np.abs(V))
        assert P[idx_zero] == pytest.approx(1.0, abs=0.05)

    def test_impedance_angle_negative90(self):
        r = capacitance_voltage()
        phases = np.array(r['impedance_phasor']['Z_phase_deg'])
        # Capacitor: angle = -90 degrees
        assert np.allclose(phases, -90.0, atol=0.01)

    def test_syntax_0_1_2(self):
        r = capacitance_voltage()
        assert '0' in r['syntax'] and '1' in r['syntax'] and '2' in r['syntax']

    def test_varactor_tuning_positive(self):
        r = capacitance_voltage()
        assert r['varactor']['tuning_range_GHz'] > 0


class TestComplexLineIntegrals:
    def test_Stokes_line_equals_surface(self):
        r = complex_line_integrals_to_electrodynamics()
        lhs = r['Stokes_check']['line_integral']
        rhs = r['Stokes_check']['surface_integral']
        assert abs(lhs - rhs) < 0.05

    def test_Stokes_match_flag(self):
        r = complex_line_integrals_to_electrodynamics()
        assert r['Stokes_check']['match'] is True

    def test_Faraday_satisfied(self):
        r = complex_line_integrals_to_electrodynamics()
        assert r['plane_wave']['Faraday_satisfied'] is True

    def test_Poynting_real_positive(self):
        r = complex_line_integrals_to_electrodynamics()
        S_real = np.array(r['plane_wave']['S_real_W_m2'])
        assert np.all(S_real > 0)

    def test_Maxwell_4_equations_present(self):
        r = complex_line_integrals_to_electrodynamics()
        mp = r['Maxwell_phasor']
        for key in ['Faraday', 'Ampere', 'Gauss_E', 'Gauss_B']:
            assert key in mp

    def test_j_omega_replaces_d_dt(self):
        r = complex_line_integrals_to_electrodynamics()
        assert 'j*omega' in r['Maxwell_phasor']['key'] or 'jω' in r['Maxwell_phasor']['key']

    def test_math_maturity_string(self):
        r = complex_line_integrals_to_electrodynamics()
        mm = r['mathematical_maturity']
        assert 'GS' in mm or 'Stokes' in mm
