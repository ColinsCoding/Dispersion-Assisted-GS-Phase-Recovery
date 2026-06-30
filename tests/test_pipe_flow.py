"""Tests for dgs/pipe_flow.py."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.pipe_flow import (
    poiseuille_profile, shell_method_demo, entry_length,
    annular_flow, complex_potential_flow,
    real_vs_complex_numbers_summary, pipe_flow_sympy_5,
)


def test_poiseuille_parabolic_max_at_center():
    p = poiseuille_profile(R=0.01, mu=1e-3, dP_dz=-100.0)
    assert p["v"][0] == p["v_max"]   # max at r=0


def test_poiseuille_zero_at_wall():
    p = poiseuille_profile(R=0.01, mu=1e-3, dP_dz=-100.0)
    assert abs(p["v"][-1]) < 1e-10   # v=0 at r=R (no-slip)


def test_poiseuille_avg_half_max():
    p = poiseuille_profile(R=0.01, mu=1e-3, dP_dz=-100.0)
    assert abs(p["v_avg"] - p["v_max"]/2) < 1e-10


def test_poiseuille_shell_method_accuracy():
    p = poiseuille_profile(R=0.01, mu=1e-3, dP_dz=-100.0)
    assert p["Q_error_pct"] < 0.01   # < 0.01% error with 300 shells


def test_poiseuille_hagen_formula():
    R, mu, G = 0.005, 1e-3, 200.0
    p = poiseuille_profile(R=R, mu=mu, dP_dz=-G)
    Q_formula = np.pi * G * R**4 / (8 * mu)
    assert abs(p["Q_exact"] - Q_formula) < 1e-15


def test_poiseuille_r4_scaling():
    # Q scales as R^4: double R -> 16x flow
    p1 = poiseuille_profile(R=0.005, mu=1e-3, dP_dz=-100.0)
    p2 = poiseuille_profile(R=0.010, mu=1e-3, dP_dz=-100.0)
    ratio = p2["Q_exact"] / p1["Q_exact"]
    assert abs(ratio - 16.0) < 1e-6


def test_shell_method_convergence():
    s = shell_method_demo()
    # Error should decrease with more shells
    errors = [s["results"][n]["error_pct"] for n in s["n_shells"]]
    for i in range(len(errors)-1):
        assert errors[i] > errors[i+1]


def test_shell_method_exact():
    s = shell_method_demo()
    assert s["results"][1000]["error_pct"] < 0.001


def test_entry_length_laminar():
    res = entry_length(R=0.005, v_avg=0.01, mu=1e-3)
    assert res["laminar"] is True
    assert res["Re"] < 2300


def test_entry_length_turbulent():
    res = entry_length(R=0.01, v_avg=5.0, mu=1e-3)
    assert bool(res["turbulent"]) is True
    assert res["L_entry_turbulent"] is not None


def test_annular_flow_zero_at_walls():
    ann = annular_flow(R_inner=0.003, R_outer=0.010, mu=1e-3, dP_dz=-100.0)
    assert abs(ann["v"][0]) < 1e-8    # inner wall
    assert abs(ann["v"][-1]) < 1e-8   # outer wall


def test_annular_flow_accuracy():
    ann = annular_flow(R_inner=0.003, R_outer=0.010, mu=1e-3, dP_dz=-100.0)
    assert ann["Q_error_pct"] < 0.01


def test_complex_potential_stream_shapes():
    res = complex_potential_flow(n_pts=20)
    assert "psi_uniform" in res
    assert "psi_circle" in res
    assert res["X"].shape == res["psi_uniform"].shape


def test_real_vs_complex_keys():
    rv = real_vs_complex_numbers_summary()
    assert "real_observables" in rv
    assert "complex_fields" in rv
    assert "GS_bridge" in rv


def test_sympy_5_count():
    eqs = pipe_flow_sympy_5()
    assert len(eqs) == 5


def test_sympy_5_hagen_poiseuille():
    eqs = pipe_flow_sympy_5()
    eq = eqs["Hagen_Poiseuille"]
    assert isinstance(eq, sp.Eq)
    # RHS should contain R (as free symbol, regardless of assumptions)
    syms = {str(s) for s in eq.rhs.free_symbols}
    assert "R" in syms


def test_sympy_shell_method_derived():
    # Shell method integral Q = int v(r)*2*pi*r dr should match Hagen-Poiseuille
    eqs = pipe_flow_sympy_5()
    # Both Shell_method_Q and Hagen_Poiseuille should have pi*G*R^4/(8*mu) form
    rhs_shell = eqs["Shell_method_Q"].rhs
    rhs_hp    = eqs["Hagen_Poiseuille"].rhs
    assert sp.simplify(rhs_shell - rhs_hp) == 0
