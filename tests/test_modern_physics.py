"""Tests for dgs/modern_physics.py"""
import numpy as np
import pytest

try:
    from dgs.modern_physics import (
        special_relativity_I, special_relativity_II,
        quantum_theory_of_light, particle_nature_of_matter,
        matter_waves_wave_packets, qm_1d, tunneling_phenomena,
        qm_3d_hydrogen, atomic_structure, statistical_physics,
        molecular_structure, solid_state, nuclear_structure,
        nuclear_applications,
    )
except ImportError:
    from modern_physics import (
        special_relativity_I, special_relativity_II,
        quantum_theory_of_light, particle_nature_of_matter,
        matter_waves_wave_packets, qm_1d, tunneling_phenomena,
        qm_3d_hydrogen, atomic_structure, statistical_physics,
        molecular_structure, solid_state, nuclear_structure,
        nuclear_applications,
    )


class TestSpecialRelativityI:
    def test_gamma_at_rest(self):
        r = special_relativity_I()
        # gamma at beta=0 should be 1
        assert r['gamma'][0] == pytest.approx(1.0, abs=1e-6)

    def test_gamma_increases_with_beta(self):
        r = special_relativity_I()
        assert r['gamma'][-1] > r['gamma'][0]

    def test_muon_gamma_positive(self):
        r = special_relativity_I()
        assert r['muon']['gamma'] > 1

    def test_muon_lab_lifetime_longer(self):
        r = special_relativity_I()
        m = r['muon']
        assert m['lifetime_lab_us'] > m['lifetime_rest_us']

    def test_causality_is_spacelike(self):
        r = special_relativity_I()
        assert r['causality']['spacelike'] is True

    def test_causality_lesson_mentions_c(self):
        r = special_relativity_I()
        assert 'c' in r['causality']['lesson']

    def test_lorentz_transforms_present(self):
        r = special_relativity_I()
        lt = r['lorentz_transforms']
        assert 'x_prime' in lt and 't_prime' in lt and 'velocity_addition' in lt


class TestSpecialRelativityII:
    def test_pair_production_above_MeV(self):
        r = special_relativity_II()
        assert r['pair_production_threshold_MeV'] == pytest.approx(1.022, rel=0.02)

    def test_E_mc2_positive(self):
        r = special_relativity_II()
        assert r['E_mc2']['1g_in_joules'] > 1e13

    def test_photon_energy_visible(self):
        r = special_relativity_II()
        # 550 nm photon should be ~2.25 eV
        assert 2.0 < r['photon']['energy_eV'] < 2.5

    def test_PET_scanner_key_present(self):
        r = special_relativity_II()
        assert 'PET_scanner' in r
        assert '511' in r['PET_scanner']['mechanism']


class TestQuantumLight:
    def test_wien_sun_visible(self):
        r = quantum_theory_of_light()
        lam = r['planck']['wien_sun_nm']
        assert 400 < lam < 700   # visible range

    def test_photoelectric_threshold_positive(self):
        r = quantum_theory_of_light()
        assert r['photoelectric']['threshold_freq_Hz'] > 0

    def test_photoelectric_kmax_positive_above_threshold(self):
        r = quantum_theory_of_light()
        assert r['photoelectric']['K_max_eV'] > 0

    def test_compton_wavelength_pm(self):
        r = quantum_theory_of_light()
        # Compton wavelength = h/(m_e*c) = 2.426 pm
        assert r['compton']['lambda_C_pm'] == pytest.approx(2.426, rel=0.01)

    def test_compton_delta_90_positive(self):
        r = quantum_theory_of_light()
        assert r['compton']['delta_lambda_90_pm'] > 0

    def test_key_insight_in_photoelectric(self):
        r = quantum_theory_of_light()
        assert 'frequency' in r['photoelectric']['key_insight'].lower()


class TestParticleNature:
    def test_de_broglie_decreases_with_energy(self):
        r = particle_nature_of_matter()
        lams = r['de_broglie']['lambda_nm']
        assert lams[0] > lams[-1]

    def test_uncertainty_position_momentum(self):
        r = particle_nature_of_matter()
        assert 'hbar' in r['uncertainty']['position_momentum'] or '>=' in r['uncertainty']['position_momentum']

    def test_K_min_atom_positive(self):
        r = particle_nature_of_matter()
        assert r['uncertainty']['K_min_atom_eV'] > 0

    def test_natural_linewidth_positive(self):
        r = particle_nature_of_matter()
        assert r['uncertainty']['natural_linewidth_Hz'] > 0

    def test_repo_connection_mentioned(self):
        r = particle_nature_of_matter()
        assert 'STEAM' in r['uncertainty']['repo_connection'] or 'bandwidth' in r['uncertainty']['repo_connection'].lower()


class TestMatterWaves:
    def test_v_group_twice_v_phase(self):
        r = matter_waves_wave_packets()
        assert r['v_group_m_per_s'] == pytest.approx(2 * r['v_phase_m_per_s'], rel=1e-6)

    def test_packet_spreads_over_time(self):
        r = matter_waves_wave_packets()
        sigma = r['sigma_x_pm']
        assert sigma[-1] > sigma[0]

    def test_optical_pulse_spreads_in_fiber(self):
        r = matter_waves_wave_packets()
        tau = r['optical_analog']['tau_ps']
        assert tau[-1] > tau[0]

    def test_identical_math_string(self):
        r = matter_waves_wave_packets()
        assert 'GS' in r['identical_math'] or 'phase retrieval' in r['identical_math'].lower()


class TestQM1D:
    def test_energy_levels_quadratic(self):
        r = qm_1d()
        En = r['box']['E_n_eV']
        # E_n = n^2 * E_1: ratio E_4/E_1 should be 16
        assert En[3]/En[0] == pytest.approx(16.0, rel=1e-4)

    def test_ground_state_nonzero(self):
        r = qm_1d()
        assert r['box']['E_1_eV'] > 0

    def test_ho_zero_point_nonzero(self):
        r = qm_1d()
        assert r['harmonic_osc']['zero_point_eV'] > 0

    def test_ho_levels_equally_spaced(self):
        r = qm_1d()
        en = r['harmonic_osc']['E_n_eV']
        gaps = np.diff(en)
        assert np.allclose(gaps, gaps[0], rtol=1e-4)

    def test_wavefunction_arrays_nonzero(self):
        r = qm_1d()
        wf = np.array(r['wavefunctions']['psi_1'])
        assert np.max(np.abs(wf)) > 0

    def test_gaussian_ground_state(self):
        r = qm_1d()
        assert r['harmonic_osc']['psi0_Gaussian'] is True


class TestTunneling:
    def test_T_decays_with_d(self):
        r = tunneling_phenomena()
        T = r['T_arr']
        assert T[0] > T[-1]

    def test_T_at_zero_is_1(self):
        r = tunneling_phenomena()
        assert r['T_arr'][0] == pytest.approx(1.0, rel=0.01)

    def test_kappa_positive(self):
        r = tunneling_phenomena()
        assert r['kappa_per_nm'] > 0

    def test_stm_ratio_less_than_1(self):
        r = tunneling_phenomena()
        assert 0 < r['stm']['T_ratio_1A_increase'] < 1

    def test_gamow_factor_positive(self):
        r = tunneling_phenomena()
        assert r['alpha_decay_U238']['Gamow_factor'] > 0

    def test_evanescent_lesson_mentions_dark_side(self):
        r = tunneling_phenomena()
        assert 'Im[k]' in r['evanescent_connection'] or 'imaginary' in r['evanescent_connection'].lower()


class TestQM3DHydrogen:
    def test_energy_levels_negative(self):
        r = qm_3d_hydrogen()
        for n, E in r['E_n_eV'].items():
            assert E < 0

    def test_ionization_energy_136(self):
        r = qm_3d_hydrogen()
        assert r['ionization_eV'] == pytest.approx(13.6, rel=0.01)

    def test_r_peak_1s_near_a0(self):
        r = qm_3d_hydrogen()
        # peak of 1s at a0 = 0.529 Angstrom
        assert r['r_peak_1s_ang'] == pytest.approx(0.529, abs=0.1)

    def test_Balmer_Halpha_656nm(self):
        r = qm_3d_hydrogen()
        # H-alpha: n=3 -> n=2, 656 nm
        found = False
        for key, data in r['Balmer_series'].items():
            if 'n3' in key:
                assert data['lambda_nm'] == pytest.approx(656, abs=5)
                found = True
        assert found

    def test_quantum_numbers_keys(self):
        r = qm_3d_hydrogen()
        qn = r['quantum_numbers']
        for key in ['n', 'l', 'm', 's']:
            assert key in qn


class TestAtomicStructure:
    def test_aufbau_starts_with_1s(self):
        r = atomic_structure()
        assert r['aufbau_order'][0] == '1s'

    def test_selection_rules_present(self):
        r = atomic_structure()
        assert 'Delta_l' in r['selection_rules'] or 'l' in r['selection_rules']

    def test_Er_for_fiber_present(self):
        r = atomic_structure()
        assert 'Er' in r['elements']
        assert '1550' in r['elements']['Er']['laser_relevance']

    def test_pauli_exclusion_string(self):
        r = atomic_structure()
        assert 'Pauli' in r['pauli'] or 'two' in r['pauli'].lower()

    def test_causality_chain_mentioned(self):
        r = atomic_structure()
        assert 'causal' in r['periodic_table_causality'].lower()


class TestStatisticalPhysics:
    def test_fermi_step_at_T0(self):
        r = statistical_physics()
        f0 = np.array(r['fermi_dirac']['f_0K'])
        E = np.array(r['fermi_dirac']['E_eV'])
        E_F = r['fermi_dirac']['E_F_Cu_eV']
        # Below E_F: f=1, above E_F: f=0
        assert np.all(f0[E < E_F - 0.5] == 1.0)
        assert np.all(f0[E > E_F + 0.5] == 0.0)

    def test_T_fermi_large(self):
        r = statistical_physics()
        assert r['fermi_dirac']['T_Fermi_Cu_K'] > 50000

    def test_photon_occupancy_small_visible(self):
        r = statistical_physics()
        # Room-temp photons at 2eV: BE occupancy very small
        assert r['BE_photon_occupancy_room_T'] < 1e-30

    def test_three_distributions_present(self):
        r = statistical_physics()
        d = r['distributions']
        assert 'Maxwell-Boltzmann' in d and 'Fermi-Dirac' in d and 'Bose-Einstein' in d

    def test_photon_stats_keys(self):
        r = statistical_physics()
        ps = r['photon_stats']
        assert 'thermal' in ps and 'laser' in ps and 'squeezed' in ps


class TestMolecularStructure:
    def test_overlap_S12_positive(self):
        r = molecular_structure()
        assert 0 < r['H2_plus']['overlap_S12'] < 1

    def test_all_bond_types_present(self):
        r = molecular_structure()
        bt = r['bond_types']
        for btype in ['ionic', 'covalent', 'metallic', 'molecular', 'amorphous']:
            assert btype in bt

    def test_life_science_in_amorphous(self):
        r = molecular_structure()
        assert 'protein' in r['bond_types']['amorphous']['life_science'].lower()

    def test_GaAs_photonics_mentioned(self):
        r = molecular_structure()
        assert 'GaAs' in r['bond_types']['covalent']['photonics']

    def test_band_theory_preview(self):
        r = molecular_structure()
        assert 'band' in r['band_theory_preview'].lower()


class TestSolidState:
    def test_Cu_fermi_energy_near_7eV(self):
        r = solid_state()
        assert r['metals']['Cu']['E_F_eV'] == pytest.approx(7.04, rel=0.05)

    def test_Si_indirect_gap(self):
        r = solid_state()
        assert r['semiconductors']['Si']['type'] == 'indirect'

    def test_GaAs_direct_gap(self):
        r = solid_state()
        assert r['semiconductors']['GaAs']['type'] == 'direct'

    def test_InGaAsP_1550nm(self):
        r = solid_state()
        assert r['semiconductors']['InGaAsP']['lambda_onset_nm'] == pytest.approx(1550, abs=10)

    def test_PDMS_amorphous(self):
        r = solid_state()
        assert 'PDMS' in r['amorphous']

    def test_causality_life_science(self):
        r = solid_state()
        assert 'causal' in r['causality_in_life_science'].lower()


class TestNuclearStructure:
    def test_peak_near_Fe56(self):
        r = nuclear_structure()
        # Max B/A near A=56 (iron)
        assert 50 <= r['peak_A'] <= 65

    def test_peak_B_per_A_near_88(self):
        r = nuclear_structure()
        assert r['peak_B_per_A'] == pytest.approx(8.8, abs=0.5)

    def test_magic_numbers_include_2_and_8(self):
        r = nuclear_structure()
        assert 2 in r['magic_numbers'] and 8 in r['magic_numbers']

    def test_Fe56_binding_per_A_positive(self):
        r = nuclear_structure()
        assert r['nuclei']['Fe56']['B_per_A_MeV'] > 8.0

    def test_U238_present(self):
        r = nuclear_structure()
        assert 'U238' in r['nuclei']


class TestNuclearApplications:
    def test_F18_beta_plus(self):
        r = nuclear_applications()
        assert r['isotopes']['F18_PET']['decay'] == 'beta+'

    def test_F18_decay_curve_starts_at_1(self):
        r = nuclear_applications()
        assert r['N_F18_fraction'][0] == pytest.approx(1.0, abs=1e-6)

    def test_F18_decays_to_half(self):
        r = nuclear_applications()
        # At t=T_{1/2}=109.8 min, N should be ~0.5
        t = np.array(r['t_F18_hours']) * 3600
        N = np.array(r['N_F18_fraction'])
        T_half = 109.8*60
        idx = np.argmin(np.abs(t - T_half))
        assert N[idx] == pytest.approx(0.5, abs=0.02)

    def test_fission_much_more_energy_than_coal(self):
        r = nuclear_applications()
        assert r['fission']['ratio'] > 1e5

    def test_PET_511keV_mentioned(self):
        r = nuclear_applications()
        assert '511' in r['PET_scanner']['mechanism']

    def test_causality_individual_vs_ensemble(self):
        r = nuclear_applications()
        c = r['causality']
        assert 'random' in c['individual_atom'].lower()
        assert 'causal' in c['ensemble'].lower()

    def test_Bragg_peak_key(self):
        r = nuclear_applications()
        assert 'Bragg_peak' in r
        assert 'proton' in r['Bragg_peak']['lesson'].lower()
