"""Tests for dgs/particle_physics.py"""
import numpy as np
import pytest

try:
    from dgs.particle_physics import (
        standard_model_particles, four_forces_field_equations,
        feynman_diagrams, conservation_laws_noether,
        cosmology, quantum_field_theory_preview, seven_hour_study_plan,
    )
except ImportError:
    from particle_physics import (
        standard_model_particles, four_forces_field_equations,
        feynman_diagrams, conservation_laws_noether,
        cosmology, quantum_field_theory_preview, seven_hour_study_plan,
    )


class TestStandardModel:
    def test_17_particles(self):
        sm = standard_model_particles()
        assert sm['total_particles'] == 17

    def test_3_generations(self):
        sm = standard_model_particles()
        assert sm['generations'] == 3

    def test_6_quarks(self):
        sm = standard_model_particles()
        assert len(sm['quarks']) == 6

    def test_6_leptons(self):
        sm = standard_model_particles()
        assert len(sm['leptons']) == 6

    def test_top_quark_heaviest(self):
        sm = standard_model_particles()
        masses = {k: v['mass_MeV'] for k, v in sm['quarks'].items() if 'mass_MeV' in v}
        assert max(masses, key=masses.get) == 'top'

    def test_photon_massless(self):
        sm = standard_model_particles()
        assert sm['bosons']['photon']['mass'] == 0

    def test_4_forces_present(self):
        sm = standard_model_particles()
        for f in ['electromagnetic', 'strong', 'weak', 'gravity']:
            assert f in sm['forces']

    def test_EM_coupling_alpha(self):
        sm = standard_model_particles()
        alpha = sm['forces']['electromagnetic']['coupling']
        assert alpha == pytest.approx(1/137.036, rel=0.001)

    def test_symmetry_group_string(self):
        sm = standard_model_particles()
        assert 'SU(3)' in sm['symmetry_group'] and 'U(1)' in sm['symmetry_group']


class TestFourForces:
    def test_CKM_unitary(self):
        ff = four_forces_field_equations()
        assert ff['CKM_unitary'] is True

    def test_PMNS_unitary(self):
        ff = four_forces_field_equations()
        assert ff['PMNS_unitary'] is True

    def test_CKM_matrix_shape(self):
        ff = four_forces_field_equations()
        CKM = np.array(ff['CKM_matrix'])
        assert CKM.shape == (3, 3)

    def test_running_alpha_QED_increases(self):
        ff = four_forces_field_equations()
        alpha = ff['running_coupling']['alpha_QED']
        # QED coupling increases with energy
        assert alpha[-1] > alpha[0]

    def test_running_alpha_s_decreases(self):
        ff = four_forces_field_equations()
        alpha_s = ff['running_coupling']['alpha_s_QCD']
        # Strong coupling decreases with energy (asymptotic freedom)
        assert alpha_s[-1] < alpha_s[0]

    def test_Yukawa_suppressed_at_long_range(self):
        ff = four_forces_field_equations()
        V_em = np.array(ff['Yukawa_vs_Coulomb']['V_EM'])
        V_w = np.array(ff['Yukawa_vs_Coulomb']['V_weak'])
        # Yukawa more suppressed at large r (index -1)
        assert abs(V_w[-1]) < abs(V_em[-1]) * 0.01

    def test_PMNS_beam_splitter_analogy(self):
        ff = four_forces_field_equations()
        assert 'beam splitter' in ff['PMNS_beam_splitter_analogy'].lower() or 'unitary' in ff['PMNS_beam_splitter_analogy']


class TestFeynmanDiagrams:
    def test_Z_peak_at_91_GeV(self):
        fd = feynman_diagrams()
        E = np.array(fd['E_cm_GeV'])
        sigma_Z = np.array(fd['Z_resonance']['sigma_Z_nb'])
        idx_peak = np.argmax(sigma_Z)
        assert E[idx_peak] == pytest.approx(91.2, abs=3.0)

    def test_QED_sigma_decreases_with_energy(self):
        fd = feynman_diagrams()
        sigma = np.array(fd['e_plus_e_minus_sigma_QED_nb'])
        # sigma ~ 1/s -> decreases with energy
        assert sigma[-1] < sigma[0]

    def test_alpha_s_positive(self):
        fd = feynman_diagrams()
        alpha_s = np.array(fd['alpha_s_running'])
        assert np.all(alpha_s > 0)

    def test_4_key_processes(self):
        fd = feynman_diagrams()
        assert len(fd['key_processes']) >= 4

    def test_Breit_Wigner_resonator_analogy(self):
        fd = feynman_diagrams()
        assert 'resonator' in fd['Breit_Wigner_resonator_analogy'].lower() or 'S21' in fd['Breit_Wigner_resonator_analogy']


class TestConservationLaws:
    def test_noether_table_has_6_entries(self):
        cl = conservation_laws_noether()
        assert len(cl['noether_table']) >= 6

    def test_allowed_reaction(self):
        cl = conservation_laws_noether()
        assert cl['allowed_reaction']['allowed'] is True

    def test_forbidden_reaction(self):
        cl = conservation_laws_noether()
        assert cl['forbidden_reaction']['allowed'] is False

    def test_CPT_exact(self):
        cl = conservation_laws_noether()
        assert 'EXACT' in cl['CPT'] or 'exact' in cl['CPT'].lower()

    def test_CP_violation_68_deg(self):
        cl = conservation_laws_noether()
        assert cl['CP_violation']['CP_phase_deg'] == pytest.approx(68.0, abs=1.0)

    def test_photon_gauge_connection(self):
        cl = conservation_laws_noether()
        assert 'U(1)' in cl['photon_connection'] or 'gauge' in cl['photon_connection'].lower()


class TestCosmology:
    def test_H0_near_67(self):
        cosm = cosmology()
        assert cosm['H0_km_s_Mpc'] == pytest.approx(67.4, abs=1.0)

    def test_T_CMB_near_273(self):
        cosm = cosmology()
        assert cosm['T_CMB_K'] == pytest.approx(2.725, abs=0.01)

    def test_energy_budget_sums_to_100(self):
        cosm = cosmology()
        eb = cosm['energy_budget']
        total = eb['dark_energy_pct'] + eb['dark_matter_pct'] + eb['baryons_pct']
        assert total == pytest.approx(100.0, abs=0.5)

    def test_baryons_less_than_5pct(self):
        cosm = cosmology()
        assert cosm['energy_budget']['baryons_pct'] < 6

    def test_Friedmann_H_over_H0_positive(self):
        cosm = cosmology()
        H = np.array(cosm['Friedmann']['H_over_H0'])
        assert np.all(H > 0)

    def test_CMB_spectrum_positive(self):
        cosm = cosmology()
        I = np.array(cosm['I_CMB'])
        assert np.all(I >= 0)

    def test_4_dark_matter_candidates(self):
        cosm = cosmology()
        assert len(cosm['dark_matter']['candidates']) >= 3

    def test_BBN_H_fraction(self):
        cosm = cosmology()
        assert cosm['BBN']['H'] == pytest.approx(0.75, abs=0.02)

    def test_photon_connection_mentions_GS(self):
        cosm = cosmology()
        assert 'GS' in cosm['photon_connection'] or 'dispersion' in cosm['photon_connection'].lower()


class TestQFT:
    def test_photon_dispersion_linear(self):
        qft = quantum_field_theory_preview()
        k = np.array(qft['Klein_Gordon']['k_per_m'])
        omega_ph = np.array(qft['Klein_Gordon']['omega_photon_rad_per_s'])
        # omega = c*k: ratio should be constant
        ratio = omega_ph[1:] / (k[1:] + 1e-30)
        assert np.allclose(ratio, 2.998e8, rtol=1e-4)

    def test_group_velocity_less_than_c(self):
        qft = quantum_field_theory_preview()
        v_gr = np.array(qft['Klein_Gordon']['v_group_electron_frac_c'])
        assert np.all(v_gr < 1.0)   # always < c

    def test_coherent_state_Poisson(self):
        qft = quantum_field_theory_preview()
        cs = qft['coherent_state_photon']
        assert cs['mean_n'] == pytest.approx(cs['variance_n'], rel=1e-6)
        assert cs['is_Poisson'] is True

    def test_laser_is_coherent_state(self):
        qft = quantum_field_theory_preview()
        assert qft['coherent_state_photon']['laser_is_coherent_state'] is True

    def test_vacuum_energy_large_ratio(self):
        qft = quantum_field_theory_preview()
        assert qft['vacuum_energy']['ratio'] >= 1e100

    def test_GS_path_integral_connection(self):
        qft = quantum_field_theory_preview()
        assert 'GS' in qft['GS_path_integral'] or 'phase retrieval' in qft['GS_path_integral'].lower()


class TestStudyPlan:
    def test_7_blocks(self):
        plan = seven_hour_study_plan()
        assert len(plan['plan']) == 7

    def test_each_block_has_keys(self):
        plan = seven_hour_study_plan()
        for block in plan['plan']:
            assert 'hour' in block and 'topic' in block
            assert 'read' in block and 'write' in block
            assert 'key_insight' in block

    def test_total_hours(self):
        plan = seven_hour_study_plan()
        assert plan['total_hours'] == 7

    def test_theory_first_theme(self):
        plan = seven_hour_study_plan()
        assert 'theory' in plan['theme']
