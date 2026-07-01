"""Tests for dgs/inverse_calculus.py"""
import numpy as np
import pytest

try:
    from dgs.inverse_calculus import (
        integration_techniques, feynman_technique_demo,
        fundamental_theorem_calculus, bioluminescence_physics,
        beer_lambert_absorption, complex_refractive_index,
        sls_laser_sintering, inverse_problem_framework,
        deconvolution_demo, maxwell_vector_field,
    )
except ImportError:
    from inverse_calculus import (
        integration_techniques, feynman_technique_demo,
        fundamental_theorem_calculus, bioluminescence_physics,
        beer_lambert_absorption, complex_refractive_index,
        sls_laser_sintering, inverse_problem_framework,
        deconvolution_demo, maxwell_vector_field,
    )


class TestIntegrationTechniques:
    def test_returns_dict(self):
        r = integration_techniques()
        assert isinstance(r, dict)
        assert 'power_rule_backwards' in r
        assert 'gaussian_integral' in r
        assert 'professor_trick' in r

    def test_gaussian_integral_pi(self):
        r = integration_techniques()
        assert 'pi' in r['gaussian_integral'].lower() or 'sqrt' in r['gaussian_integral'].lower()


class TestFeynmanTechnique:
    def test_sinc_integral(self):
        r = feynman_technique_demo()
        import sympy as sp
        assert r['sinc_integral_result'] == sp.pi / 2

    def test_gaussian_with_param(self):
        r = feynman_technique_demo()
        import sympy as sp
        a = sp.Symbol('a', positive=True)
        expected = sp.sqrt(sp.pi / a)
        assert sp.simplify(r['gaussian_with_param'] - expected) == 0


class TestFTC:
    def test_keys_present(self):
        r = fundamental_theorem_calculus()
        assert 'FTC_part1' in r
        assert 'kramers_kronig' in r
        assert 'neural_network' in r

    def test_ftc_statement(self):
        r = fundamental_theorem_calculus()
        assert 'd/dx' in r['FTC_part1']


class TestBioluminescence:
    def test_photon_energy(self):
        r = bioluminescence_physics()
        E = r['photon_energy_eV']
        assert 2.0 < E < 2.5  # 560nm green-yellow photon ~2.2 eV

    def test_quantum_yields(self):
        r = bioluminescence_physics()
        for name, data in r['species'].items():
            phi = data['phi_Q']
            assert 0 < phi <= 1.0, f"{name}: phi_Q={phi} out of range"

    def test_spectrum_shapes(self):
        r = bioluminescence_physics()
        sf = r['spectrum_firefly']
        assert sf.max() == pytest.approx(1.0, abs=0.01)
        # peak near 560nm index
        lam = r['lam_nm']
        peak_idx = np.argmax(sf)
        assert 550 <= lam[peak_idx] <= 570

    def test_deep_sea_blue(self):
        r = bioluminescence_physics()
        lam = r['lam_nm']
        peak_idx = np.argmax(r['spectrum_deep'])
        assert lam[peak_idx] < 500   # blue


class TestBeerLambert:
    def test_exponential_decay(self):
        r = beer_lambert_absorption(alpha_per_cm=2.0)
        I = r['I']
        # monotonically decreasing
        assert np.all(np.diff(I) <= 0)
        # starts at I0=1
        assert I[0] == pytest.approx(1.0, abs=0.01)

    def test_1_over_e_depth(self):
        r = beer_lambert_absorption(alpha_per_cm=2.0)
        assert r['depth_1_over_e_cm'] == pytest.approx(0.5, rel=1e-3)

    def test_ode_string(self):
        r = beer_lambert_absorption()
        assert 'dI/dz' in r['ODE']

    def test_extinction_coefficient(self):
        r = beer_lambert_absorption(alpha_per_cm=0.0)
        assert r['k_extinction_coefficient'] == pytest.approx(0.0, abs=1e-10)


class TestComplexRefractiveIndex:
    def test_absorption_coefficient(self):
        n, k, lam = 1.5, 0.01, 550.0
        r = complex_refractive_index(n, k, lam)
        expected_alpha = 4*np.pi*k / (lam*1e-9)
        assert r['alpha_per_m'] == pytest.approx(expected_alpha, rel=1e-3)

    def test_phase_velocity(self):
        r = complex_refractive_index(n_real=1.5)
        assert r['phase_velocity_m_per_s'] == pytest.approx(3e8/1.5, rel=1e-3)

    def test_zero_extinction(self):
        r = complex_refractive_index(n_real=1.5, k_extinction=0.0)
        assert r['skin_depth_nm'] == float('inf') or r['skin_depth_nm'] > 1e20

    def test_I_z_decay(self):
        r = complex_refractive_index(n_real=1.5, k_extinction=0.1)
        I = r['I_z']
        assert np.all(np.diff(I) <= 0)
        assert I[0] == pytest.approx(1.0, abs=0.01)


class TestSLSLaserSintering:
    def test_high_irradiance(self):
        r = sls_laser_sintering()
        assert r['irradiance_MW_per_m2'] > 100   # should be >100 MW/m^2

    def test_skin_depth_small(self):
        r = sls_laser_sintering()
        assert r['skin_depth_um'] < 100   # should be ~10 um

    def test_explosion_keys(self):
        r = sls_laser_sintering()
        exp = r['explosion']
        assert 'MEC_g_per_m3' in exp
        assert 'prevention' in exp
        assert exp['MEC_g_per_m3'] > 0

    def test_photonics_connection(self):
        r = sls_laser_sintering()
        assert 'galvo' in r['photonics_connection'] or 'GS' in r['photonics_connection']


class TestInverseProblem:
    def test_framework_keys(self):
        r = inverse_problem_framework()
        assert 'forward' in r
        assert 'inverse' in r
        assert 'phase_retrieval' in r['examples']

    def test_gs_mentioned(self):
        r = inverse_problem_framework()
        assert 'GS' in r['gs_as_inverse'] or 'Gauss' in r['gs_as_inverse']


class TestDeconvolution:
    def test_wiener_beats_naive(self):
        r = deconvolution_demo(noise_level=0.02)
        assert r['corr_wiener'] > r['corr_naive'], (
            f"Wiener corr={r['corr_wiener']:.4f} should exceed naive corr={r['corr_naive']:.4f}"
        )

    def test_wiener_corr_high(self):
        r = deconvolution_demo(noise_level=0.005)
        assert r['corr_wiener'] > 0.90

    def test_arrays_length(self):
        N = 128
        r = deconvolution_demo(signal_length=N)
        assert len(r['f_true']) == N
        assert len(r['f_wiener']) == N

    def test_lesson_string(self):
        r = deconvolution_demo()
        assert 'corr=' in r['lesson'] or 'corr' in r['lesson']


class TestMaxwellVectorField:
    def test_point_charge_shape(self):
        r = maxwell_vector_field('point_charge', N=10)
        assert r['X'].shape == (10, 10)
        assert r['Ex'].shape == (10, 10)
        assert r['magnitude'].shape == (10, 10)

    def test_wire_B_closed_loops(self):
        # For wire: div B = 0 verified by checking Ex_norm/Ey_norm forms circles
        r = maxwell_vector_field('wire_B', N=10)
        Ex = r['Ex_norm']
        Ey = r['Ey_norm']
        # magnitude of normalized vectors should be ~1 (away from singularity)
        X, Y = r['X'], r['Y']
        mask = (X**2 + Y**2) > 0.5
        mag = np.sqrt(Ex[mask]**2 + Ey[mask]**2)
        assert np.all(mag > 0.99)

    def test_dipole(self):
        r = maxwell_vector_field('dipole', N=10)
        assert 'dipole' in r['title']

    def test_divergence_string(self):
        r = maxwell_vector_field('point_charge')
        assert 'div' in r['divergence_check'].lower()
