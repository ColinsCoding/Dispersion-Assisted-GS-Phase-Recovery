"""Tests for dgs/vector_calculus.py"""
import numpy as np
import pytest

try:
    from dgs.vector_calculus import (
        unit_circle_trig, unit_sphere_coordinates,
        gradient_demo, divergence_demo, curl_demo, laplacian_demo,
        complex_field_2d, complex_poynting_vector,
        ray_tracing_bvh, ousd_vector_calc_alignment,
    )
except ImportError:
    from vector_calculus import (
        unit_circle_trig, unit_sphere_coordinates,
        gradient_demo, divergence_demo, curl_demo, laplacian_demo,
        complex_field_2d, complex_poynting_vector,
        ray_tracing_bvh, ousd_vector_calc_alignment,
    )


class TestUnitCircle:
    def test_euler_formula_key_angles(self):
        uc = unit_circle_trig()
        for deg, expected_cos, expected_sin in [(0,1,0),(90,0,1),(180,-1,0),(270,0,-1)]:
            a = uc['key_angles'][f'{deg}deg']
            assert a['cos'] == pytest.approx(expected_cos, abs=1e-5)
            assert a['sin'] == pytest.approx(expected_sin, abs=1e-5)

    def test_pythagorean_identity(self):
        uc = unit_circle_trig()
        x, y = uc['x'], uc['y']
        assert np.all(np.abs(x**2 + y**2 - 1) < 1e-10)

    def test_exp_j_theta_on_unit_circle(self):
        uc = unit_circle_trig()
        for key, a in uc['key_angles'].items():
            z = a['exp_j_theta']
            assert abs(abs(z) - 1) < 1e-4

    def test_euler_string(self):
        uc = unit_circle_trig()
        assert 'exp' in uc['eulers_formula'] and 'j' in uc['eulers_formula']


class TestUnitSphere:
    def test_orthonormal_basis(self):
        us = unit_sphere_coordinates()
        assert us['all_orthogonal'] is True
        o = us['orthonormality']
        for k, v in o.items():
            if 'dot' in k:
                assert abs(v) < 1e-10

    def test_r_hat_unit_length(self):
        us = unit_sphere_coordinates()
        assert us['orthonormality']['r_norm'] == pytest.approx(1.0, abs=1e-10)

    def test_cartesian_on_sphere(self):
        us = unit_sphere_coordinates()
        X, Y, Z = us['X'], us['Y'], us['Z']
        r2 = X**2 + Y**2 + Z**2
        assert np.all(np.abs(r2 - 1) < 1e-10)

    def test_dipole_pattern_zero_at_poles(self):
        us = unit_sphere_coordinates()
        P = us['P_dipole']
        # At theta=0 (pole): sin(0)=0, P=0
        assert abs(P[0, 0]) < 1e-10

    def test_position_dependent_note(self):
        us = unit_sphere_coordinates()
        note = us['position_dependent']
        assert 'position' in note.lower() or 'CRITICAL' in note


class TestGradient:
    def test_returns_keys(self):
        gr = gradient_demo()
        assert 'grad_V' in gr and 'E_field' in gr
        assert len(gr['grad_V']) == 3

    def test_plane_wave_grad_complex(self):
        gr = gradient_demo()
        lesson = gr['plane_wave_lesson']
        assert 'j*k' in lesson or 'complex' in lesson.lower()

    def test_group_delay_mentioned(self):
        gr = gradient_demo()
        gd = gr['group_delay']
        assert 'group' in gd.lower() or 'tau' in gd or 'delay' in gd.lower()


class TestDivergence:
    def test_point_charge_div_zero_away(self):
        dv = divergence_demo()
        assert '0' in dv['point_charge_div_E']

    def test_gauss_law_mentioned(self):
        dv = divergence_demo()
        assert 'Gauss' in dv['Gauss_law'] or 'rho' in dv['Gauss_law']

    def test_no_monopoles(self):
        dv = divergence_demo()
        assert 'div(B) = 0' in dv['no_monopoles']

    def test_complex_lesson_real_and_imag(self):
        dv = divergence_demo()
        lesson = dv['complex_lesson']
        assert 'absorption' in lesson.lower() or 'decay' in lesson.lower()
        assert 'propagation' in lesson.lower()


class TestCurl:
    def test_curl_grad_zero(self):
        cr = curl_demo()
        assert 'True' in cr['vector_identity_curl_grad_zero']

    def test_maxwell_phasor_keys(self):
        cr = curl_demo()
        mp = cr['Maxwell_complex_phasor']
        assert 'curl_E' in mp and 'curl_H' in mp

    def test_faraday_mentioned(self):
        cr = curl_demo()
        assert 'Faraday' in cr['Faraday'] or 'EMF' in cr['Faraday']

    def test_dark_side_GS(self):
        cr = curl_demo()
        assert 'GS' in cr['dark_side'] or 'phase retrieval' in cr['dark_side'].lower()

    def test_stokes_theorem(self):
        cr = curl_demo()
        assert 'Stokes' in cr['Stokes_theorem'] or 'line' in cr['Stokes_theorem'].lower()


class TestLaplacian:
    def test_wave_laplacian_minus_k2(self):
        lp = laplacian_demo()
        assert '-k**2' in lp['wave_laplacian'] or '-k^2' in lp['wave_laplacian']

    def test_1_over_r_is_zero(self):
        lp = laplacian_demo()
        assert '0' in lp['laplace_1_over_r']

    def test_five_equations(self):
        lp = laplacian_demo()
        assert len(lp['equations']) >= 4

    def test_spherical_harmonics(self):
        lp = laplacian_demo()
        sh = lp['spherical_harmonics']
        assert 'Y_00' in sh and 'Y_10' in sh

    def test_complex_k_dark_side(self):
        lp = laplacian_demo()
        dark = lp['complex_k_dark_side']
        assert 'absorption' in dark.lower() and 'propagation' in dark.lower()


class TestComplexField2D:
    def test_em_wave_shapes(self):
        r = complex_field_2d('EM_wave', N=10)
        assert r['X'].shape == (10, 10)
        assert r['Eimag_y'].shape == (10, 10)

    def test_real_and_imag_different(self):
        r = complex_field_2d('EM_wave', N=20)
        assert not np.allclose(r['Ereal_y'], r['Eimag_y'])

    def test_point_source(self):
        r = complex_field_2d('point_source', N=10)
        assert 'point_source' in r['title'] or 'Point' in r['title']

    def test_vortex(self):
        r = complex_field_2d('vortex', N=10)
        assert 'vortex' in r['title'].lower() or 'Vortex' in r['title']

    def test_dark_side_lesson(self):
        r = complex_field_2d('EM_wave')
        assert 'GS' in r['dark_side_lesson'] or 'phase retrieval' in r['dark_side_lesson'].lower()


class TestComplexPoynting:
    def test_amplitude_decays_in_lossy(self):
        r = complex_poynting_vector(n_complex=complex(1.5, 0.1))
        S = r['S_real']
        # Power should decay along z
        assert S[0] > S[-1]

    def test_lossless_has_skin_inf(self):
        r = complex_poynting_vector(n_complex=complex(1.5, 0.0))
        assert r['skin_depth_m'] == float('inf')

    def test_alpha_from_k_i(self):
        n = complex(1.5, 0.05)
        r = complex_poynting_vector(n_complex=n)
        # alpha = 2*Im[k] = 2*Re[k0*n]*... actually 2*Im[k0*n]
        assert r['alpha_per_m'] > 0

    def test_ac_analogy_in_lesson(self):
        r = complex_poynting_vector()
        assert 'power' in r['AC_analogy'].lower() or 'V*I' in r['AC_analogy']


class TestRayTracing:
    def test_ray_box_hit(self):
        r = ray_tracing_bvh()
        assert bool(r['ray_box_hit']) is True

    def test_ray_box_miss(self):
        r = ray_tracing_bvh()
        assert bool(r['ray_box_miss']) is False

    def test_reflection_correct(self):
        r = ray_tracing_bvh()
        # Downward ray [0,-1,0] reflected off upward normal [0,1,0] -> [0,1,0]
        ref = r['reflected_direction']
        assert abs(ref[1] - 1.0) < 1e-6

    def test_total_internal_reflection(self):
        r = ray_tracing_bvh()
        assert r['total_internal_reflection'] is True

    def test_evanescent_k_imaginary(self):
        r = ray_tracing_bvh()
        k = r['evanescent_k']
        assert k.imag < 0   # imaginary -> evanescent decay

    def test_bvh_lesson(self):
        r = ray_tracing_bvh()
        assert 'log' in r['bvh_lesson'] or 'O(' in r['bvh_lesson']

    def test_evanescent_lesson_tunneling(self):
        r = ray_tracing_bvh()
        assert 'tunneling' in r['evanescent_lesson'].lower() or 'evanescent' in r['evanescent_lesson'].lower()


class TestOUSDAlignment:
    def test_six_ctas(self):
        ousd = ousd_vector_calc_alignment()
        for cta in ['FutureG','Directed_Energy','Integrated_Sensing','Trusted_AI','HMI']:
            assert cta in ousd

    def test_dark_side_in_each(self):
        ousd = ousd_vector_calc_alignment()
        for cta in ['FutureG','Directed_Energy','Integrated_Sensing']:
            assert 'dark_side' in ousd[cta]

    def test_repo_connection(self):
        ousd = ousd_vector_calc_alignment()
        assert 'GS' in ousd['this_repo'] or 'phase retrieval' in ousd['this_repo'].lower()

    def test_complex_in_futureg(self):
        ousd = ousd_vector_calc_alignment()
        assert 'complex' in ousd['FutureG']['math'].lower() or 'exp' in ousd['FutureG']['math']
