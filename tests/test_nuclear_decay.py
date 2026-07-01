import numpy as np
import pytest
import sympy as sp
from dgs.nuclear_decay import (
    half_life, decay_constant, activity, n_atoms_from_mass,
    single_decay, bateman_two_step, secular_equilibrium_time,
    bateman_chain_numerical, q_value_alpha, q_value_beta_minus,
    gamow_factor, u238_chain_overview, secular_equilibrium_activities,
    physics_field_tree, topological_sort_physics, nuclear_decay_sympy_5,
    LN2,
)


# ── half-life / decay constant ────────────────────────────────────────

def test_half_life_C14():
    t = half_life(LN2 / (5730 * 3.156e7))
    assert t == pytest.approx(5730 * 3.156e7, rel=1e-6)

def test_decay_constant_inverse():
    lam = decay_constant(100.0)
    assert half_life(lam) == pytest.approx(100.0, rel=1e-6)

def test_half_life_invalid():
    with pytest.raises(ValueError):
        half_life(0)

def test_decay_constant_invalid():
    with pytest.raises(ValueError):
        decay_constant(-1)


# ── activity ──────────────────────────────────────────────────────────

def test_activity_1gram_Ra226():
    # 1 gram of Ra-226: N = (1/226) * 6.022e23, lam = ln2/(1600*3.156e7)
    N = n_atoms_from_mass(1.0, 226)
    lam = LN2 / (1600 * 3.156e7)
    r = activity(N, lam)
    assert r["A_Ci"] == pytest.approx(1.0, rel=0.05)   # definition of 1 Curie

def test_activity_zero_N():
    r = activity(0, 1e-3)
    assert r["A_Bq"] == 0

def test_activity_invalid():
    with pytest.raises(ValueError):
        activity(-1, 1e-3)


# ── single decay ──────────────────────────────────────────────────────

def test_single_decay_at_one_half_life():
    lam = LN2 / 100.0
    r = single_decay(1e10, lam, 100.0)
    assert float(r["fraction_remaining"]) == pytest.approx(0.5, rel=1e-6)

def test_single_decay_at_t0():
    lam = LN2 / 100.0
    r = single_decay(1e10, lam, 0.0)
    assert float(r["N_t"]) == pytest.approx(1e10, rel=1e-6)

def test_single_decay_array():
    lam = LN2 / 100.0
    t = np.array([0.0, 100.0, 200.0])
    r = single_decay(1e12, lam, t)
    assert r["N_t"].shape == (3,)
    assert r["N_t"][1] == pytest.approx(0.5e12, rel=1e-6)
    assert r["N_t"][2] == pytest.approx(0.25e12, rel=1e-6)

def test_single_decay_stable():
    r = single_decay(1e10, 0.0, np.array([0.0, 100.0, 1000.0]))
    assert np.allclose(r["N_t"], 1e10)


# ── Bateman two-step ──────────────────────────────────────────────────

def test_bateman_conservation():
    lam_A = LN2 / 10.0
    lam_B = LN2 / 2.0
    t = np.linspace(0, 50, 100)
    b = bateman_two_step(1e10, 0, lam_A, lam_B, t)
    total = b["N_A"] + b["N_B"] + b["N_C"]
    assert np.allclose(total, 1e10, rtol=1e-4)

def test_bateman_at_t0():
    lam_A = LN2 / 10.0
    lam_B = LN2 / 2.0
    b = bateman_two_step(1e10, 0, lam_A, lam_B, np.array([0.0]))
    assert float(b["N_A"][0]) == pytest.approx(1e10, rel=1e-6)
    assert float(b["N_B"][0]) == pytest.approx(0.0, abs=1.0)

def test_bateman_secular_equilibrium():
    # Ra-226 -> Rn-222: after many Rn half-lives, activities equalize
    lam_Ra = LN2 / (1600 * 3.156e7)
    lam_Rn = LN2 / (3.82 * 86400)
    t = np.array([50 * 86400.0])   # 50 days >> 7 * 3.82 days
    b = bateman_two_step(1e20, 0, lam_Ra, lam_Rn, t)
    ratio = float(b["A_B_Bq"][0] / b["A_A_Bq"][0])
    assert ratio == pytest.approx(1.0, rel=0.01)


# ── secular equilibrium time ──────────────────────────────────────────

def test_secular_eq_time_positive():
    r = secular_equilibrium_time(1e-10, LN2 / 10.0)
    assert r["t_equilibrium_s"] > 0

def test_secular_eq_time_is_7_halflives():
    lam_B = LN2 / 10.0
    r = secular_equilibrium_time(1e-10, lam_B)
    assert r["t_equilibrium_s"] == pytest.approx(70.0, rel=1e-6)

def test_secular_eq_invalid():
    with pytest.raises(ValueError):
        secular_equilibrium_time(1e-10, 0)


# ── Bateman chain numerical ───────────────────────────────────────────

def test_bateman_numerical_3step():
    lam = [LN2 / 10.0, LN2 / 2.0, 0.0]
    N0 = [1e10, 0, 0]
    t = np.linspace(0, 100, 1000)
    r = bateman_chain_numerical(lam, N0, t)
    total = r["N"].sum(axis=1)
    assert np.allclose(total, 1e10, rtol=1e-3)

def test_bateman_numerical_matches_analytic():
    lam_A = LN2 / 10.0
    lam_B = LN2 / 2.0
    t = np.linspace(0, 30, 300)
    analytic = bateman_two_step(1e10, 0, lam_A, lam_B, t)
    numerical = bateman_chain_numerical([lam_A, lam_B, 0.0], [1e10, 0, 0], t)
    assert np.allclose(numerical["N"][:, 0], analytic["N_A"], rtol=1e-3)
    assert np.allclose(numerical["N"][:, 1], analytic["N_B"], rtol=1e-3)

def test_bateman_numerical_invalid():
    with pytest.raises(ValueError):
        bateman_chain_numerical([0.1, 0.05], [1e10, 0, 0], np.linspace(0, 10, 100))


# ── Q values ─────────────────────────────────────────────────────────

def test_q_alpha_polonium210():
    # 210Po -> 206Pb + 4He: Q ~ 5.41 MeV
    # Masses: 210Po = 209.982874, 206Pb = 205.974465, 4He = 4.002602
    r = q_value_alpha(209.982874, 205.974465)
    assert r["Q_MeV"] == pytest.approx(5.41, abs=0.05)
    assert bool(r["spontaneous"]) is True

def test_q_beta_minus_tritium():
    # 3H -> 3He + e- + nu: Q ~ 0.0186 MeV (18.6 keV)
    # Masses: 3H = 3.016049, 3He = 3.016029
    r = q_value_beta_minus(3.016049, 3.016029)
    assert r["Q_MeV"] > 0
    assert bool(r["spontaneous"]) is True


# ── Gamow factor ──────────────────────────────────────────────────────

def test_gamow_positive():
    g = gamow_factor(82, E_MeV=5.0)
    assert g["G_gamow"] > 0
    assert 0 < g["P_tunnel"] < 1

def test_gamow_higher_Z_lower_tunneling():
    g_low  = gamow_factor(50, E_MeV=5.0)
    g_high = gamow_factor(82, E_MeV=5.0)
    assert g_high["P_tunnel"] < g_low["P_tunnel"]

def test_gamow_higher_energy_higher_tunneling():
    g_slow = gamow_factor(82, E_MeV=4.0)
    g_fast = gamow_factor(82, E_MeV=6.0)
    assert g_fast["P_tunnel"] > g_slow["P_tunnel"]


# ── U-238 chain ───────────────────────────────────────────────────────

def test_u238_chain_length():
    chain = u238_chain_overview()
    assert len(chain) == 15   # 14 radioactive + 1 stable Pb-206

def test_u238_chain_starts_with_U238():
    chain = u238_chain_overview()
    assert chain[0]["symbol"] == "238U"

def test_u238_chain_ends_stable():
    chain = u238_chain_overview()
    assert chain[-1]["decay_mode"] == "stable"
    assert chain[-1]["symbol"] == "206Pb"

def test_u238_chain_activities_all_equal_secular():
    N_U = 1e20
    r = secular_equilibrium_activities(N_U)
    # all daughters should have same activity as parent
    A_parent = r["A_parent_Bq"]
    for d in r["daughters"][1:]:
        if d["A_Bq"] > 0:
            assert d["A_Bq"] == pytest.approx(A_parent, rel=1e-6)


# ── physics field tree ────────────────────────────────────────────────

def test_field_tree_has_qm():
    tree = physics_field_tree()
    assert "Quantum Mechanics" in tree

def test_field_tree_topic_lookup():
    node = physics_field_tree("Phase Retrieval / GS")
    assert "Optics" in node["prereqs"]
    assert "Dispersion-Assisted GS" in node["enables"]

def test_field_tree_invalid():
    with pytest.raises(ValueError):
        physics_field_tree("Astrology")

def test_topological_sort_linear_algebra_before_qm():
    order = topological_sort_physics()
    la_idx = order.index("Linear Algebra")
    qm_idx = order.index("Quantum Mechanics")
    assert la_idx < qm_idx

def test_topological_sort_classical_mech_first():
    order = topological_sort_physics()
    # Classical Mechanics has no prereqs -> should be near front
    cm_idx = order.index("Classical Mechanics")
    assert cm_idx < 5


# ── SymPy 5 ──────────────────────────────────────────────────────────

def test_nuclear_sympy_5_count():
    eqs = nuclear_decay_sympy_5()
    assert len(eqs) == 5

def test_nuclear_sympy_5_types():
    for k, eq in nuclear_decay_sympy_5().items():
        assert isinstance(eq, sp.Basic), f"{k} not SymPy"
