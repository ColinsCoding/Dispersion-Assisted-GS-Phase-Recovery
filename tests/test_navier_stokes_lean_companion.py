"""Smoke tests for dgs/navier_stokes_lean_companion.py."""
import numpy as np
import pytest
import sympy as sp

from dgs import navier_stokes_lean_companion as m


def test_divergence_lemmas_all_zero():
    assert all(sp.simplify(v) == 0 for v in m.divergence_lemmas().values())


def test_trace_and_convective_identities():
    assert m.trace_is_coordinate_free() == 0
    assert m.convective_matches_textbook() == sp.zeros(3, 1)


@pytest.mark.parametrize("flow", [m.taylor_green, m.abc_flow])
def test_exact_solutions(flow):
    v, p, f = flow()
    assert m.divergence(v) == 0
    assert m.is_one_periodic(v)
    assert m.ns_residual(v, p, f) == sp.zeros(3, 1)


def test_scaling_keeps_solution():
    assert m.ns_residual(*m.scaled_solution(*m.taylor_green())) == sp.zeros(3, 1)
    assert m.energy_scaling_exponent(3) == -1
    assert m.energy_scaling_exponent(2) == 0


def test_energy_identity():
    lhs, rhs = m.energy_balance(m.taylor_green()[0])
    assert sp.simplify(lhs - rhs) == 0
    v, _, f = m.abc_flow()
    lhs, rhs = m.energy_balance(v, f)
    assert sp.simplify(lhs - rhs) == 0


def test_2d_has_no_vortex_stretching():
    assert m.vortex_stretching_2d() == sp.zeros(3, 1)


def test_decay_condition():
    assert m.divergence(m.gaussian_swirl()) == 0
    r, prof = m.decay_profile(m.gaussian_swirl(), K=6)
    assert prof[-1] < 1e-6
    r, prof = m.decay_profile(m.algebraic_swirl(), K=4)
    assert prof[-1] > prof[len(prof) // 2]           # grows: fails for K = 4


def test_tracking_recovers_flow():
    r = m.track_and_compare(noise=0.05, max_area=18, stride=4)
    assert r["n_tracked"] >= 20
    assert r["swap_fraction"] == 0.0
    assert r["rel_rms_error"] < 0.10


def test_leray_projection():
    d = m.projection_denoise_demo()
    assert d["signal_change"] < 1e-12
    assert d["div_projected_rms"] < 1e-10
    assert abs(d["ratio"] - np.sqrt(0.5)) < 0.03


def test_bounds():
    with pytest.raises(ValueError):
        m.channel_reynolds(-1.0)
    with pytest.raises(ValueError):
        m.advect(np.zeros((1, 2)), 0.0, 0.01, 0)
    assert m.channel_reynolds(1.0, 50e-6) == pytest.approx(50.0)


def test_lean_reader_optional():
    if m.lean_repo() is None:
        pytest.skip("NavierStokesAndEuler clone not present")
    s = m.lean_statement("divergence")
    assert s is not None and "trace" in s
    assert m.repo_inventory()["challenge_sorry_tactics"] >= 1


def test_definitions_match_challenge():
    d = m.definitions_match()
    if d is None:
        pytest.skip("clone with ComparatorDefinitions.lean not present")
    n_a, n_b, diff = d
    assert n_a == n_b and diff == []


def test_built_proofs_use_standard_axioms_only():
    if m.lean_build_dir() is None:
        pytest.skip("no completed lake build found")
    ax = m.lean_axioms()
    assert ax and all(set(v) <= m.STANDARD_AXIOMS for v in ax.values())
