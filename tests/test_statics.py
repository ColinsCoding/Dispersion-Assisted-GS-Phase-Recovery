"""Tests for dgs/statics.py"""
import numpy as np
import pytest

try:
    from dgs.statics import (
        equilibrium_2d, beam_reactions, shear_moment_diagram,
        truss_method_of_joints, three_bar_truss, moment_of_inertia_shapes,
        distribution_sifting, importance_sampling_demo,
        jane_street_tech_stack, autograd_truss_sensitivity,
    )
except ImportError:
    from statics import (
        equilibrium_2d, beam_reactions, shear_moment_diagram,
        truss_method_of_joints, three_bar_truss, moment_of_inertia_shapes,
        distribution_sifting, importance_sampling_demo,
        jane_street_tech_stack, autograd_truss_sensitivity,
    )


class TestEquilibrium2D:
    def test_balanced(self):
        forces = [{'Fx':10,'Fy':5,'x':0,'y':0}, {'Fx':-10,'Fy':-5,'x':0,'y':0}]
        r = equilibrium_2d(forces)
        assert r['in_equilibrium']
        assert r['check'] == 'PASS'

    def test_unbalanced(self):
        forces = [{'Fx':5,'Fy':0,'x':0,'y':0}]
        r = equilibrium_2d(forces)
        assert not r['in_equilibrium']

    def test_moment_check(self):
        forces = [{'Fx':0,'Fy':100,'x':0,'y':0}, {'Fx':0,'Fy':-100,'x':1,'y':0}]
        r = equilibrium_2d(forces)
        assert not r['in_equilibrium']


class TestBeamReactions:
    def test_center_load(self):
        r = beam_reactions(L=4.0, loads=[{'P':1000.0,'x':2.0}])
        assert r['Ay_N'] == pytest.approx(500.0, rel=1e-6)
        assert r['By_N'] == pytest.approx(500.0, rel=1e-6)

    def test_load_sum(self):
        loads = [{'P':3000,'x':1.0},{'P':2000,'x':3.0}]
        r = beam_reactions(L=6.0, loads=loads)
        assert r['Ay_N'] + r['By_N'] == pytest.approx(5000.0, rel=1e-6)

    def test_distributed_load(self):
        r = beam_reactions(L=5.0, loads=[{'w':200.0,'x_start':0,'x_end':5.0}])
        assert r['Ay_N'] == pytest.approx(500.0, rel=1e-4)
        assert r['By_N'] == pytest.approx(500.0, rel=1e-4)

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            beam_reactions(L=0.0, loads=[])


class TestShearMomentDiagram:
    def test_boundary_M0(self):
        smd = shear_moment_diagram(6.0, [{'P':5000,'x':3.0}])
        assert abs(smd['M'][0]) < 1.0

    def test_max_moment_near_center(self):
        smd = shear_moment_diagram(4.0, [{'P':4000,'x':2.0}])
        assert 1.5 < smd['x_max_moment'] < 2.5

    def test_lesson_has_FTC(self):
        smd = shear_moment_diagram(4.0, [{'P':1000,'x':2.0}])
        assert 'FTC' in smd['lesson'] or 'dM' in smd['lesson']


class TestTruss:
    def test_three_bar_no_error(self):
        t = three_bar_truss()
        assert 'error' not in t

    def test_reaction_sum(self):
        t = three_bar_truss()
        rxns = t['reactions_N']
        Ay = rxns.get('R_A_y', 0)
        By = rxns.get('R_B_y', 0)
        assert Ay + By == pytest.approx(10000.0, rel=1e-3)

    def test_member_count(self):
        t = three_bar_truss()
        assert t['n_members'] == 3
        assert t['n_reactions'] == 3

    def test_tension_positive(self):
        assert three_bar_truss()['tension_positive'] is True


class TestMomentOfInertia:
    def test_rectangle(self):
        import sympy as sp
        shapes = moment_of_inertia_shapes()
        b, h = sp.symbols('b h', positive=True)
        expected = sp.Rational(1,12)*b*h**3
        assert sp.simplify(shapes['rectangle']['I_x'] - expected) == 0

    def test_parallel_axis(self):
        shapes = moment_of_inertia_shapes()
        formula_str = str(shapes['parallel_axis_theorem']['formula'])
        assert 'd' in formula_str

    def test_statistics_connection(self):
        shapes = moment_of_inertia_shapes()
        conn = shapes['statistics_connection']
        assert 'MSE' in conn or 'variance' in conn.lower()


class TestDistributionSifting:
    def test_keys(self):
        ds = distribution_sifting()
        for k in ['sifting_property','attention','finance','importance_sampling','greens_function']:
            assert k in ds

    def test_softmax_in_attention(self):
        ds = distribution_sifting()
        assert 'softmax' in ds['attention']['soft']

    def test_photonics_greens(self):
        ds = distribution_sifting()
        gf = ds['greens_function']['photonics']
        assert 'H(f)' in gf or 'dispersive' in gf

    def test_cost_connection_three_domains(self):
        ds = distribution_sifting()
        conn = ds['cost_connection']
        assert 'Statics' in conn and 'Finance' in conn and 'ML' in conn


class TestImportanceSampling:
    def test_true_prob_order(self):
        r = importance_sampling_demo(N=1000)
        assert 1e-4 < r['true_prob'] < 1e-3

    def test_is_fewer_samples(self):
        r = importance_sampling_demo(N=2000)
        assert r['is_N'] < r['naive_N']

    def test_is_se_better(self):
        r = importance_sampling_demo(N=5000)
        # IS should estimate more accurately than naive (which often gets 0 hits)
        assert r['is_se'] < r['true_prob'] * 0.5  # IS SE < 50% of true value

    def test_lesson_has_sift(self):
        r = importance_sampling_demo(N=1000)
        assert 'sift' in r['lesson'].lower() or 'efficient' in r['lesson'].lower()


class TestJaneStreetStack:
    def test_three_languages(self):
        js = jane_street_tech_stack()
        for lang in ['OCaml', 'C', 'Python_torch_JAX']:
            assert lang in js

    def test_levels_0_1_2(self):
        js = jane_street_tech_stack()
        for lvl in [0, 1, 2]:
            assert lvl in js['levels']

    def test_autodiff_in_python(self):
        js = jane_street_tech_stack()
        ad = js['Python_torch_JAX']['autodiff']
        assert 'grad' in ad.lower()

    def test_physics_connection(self):
        js = jane_street_tech_stack()
        assert 'H(f)' in js['physics_connection'] or 'dispersion' in js['physics_connection'].lower()


class TestAdjointSensitivity:
    def test_positive_strain_energy(self):
        r = autograd_truss_sensitivity()
        assert 'error' not in r
        assert r['strain_energy_J'] > 0

    def test_sensitivities_negative(self):
        r = autograd_truss_sensitivity()
        for m, s in r['dU_dA'].items():
            assert s < 0, f"{m}: dU/dA={s} should be < 0"

    def test_lesson_backprop(self):
        r = autograd_truss_sensitivity()
        assert 'backprop' in r['lesson'] or 'backward' in r['lesson']
