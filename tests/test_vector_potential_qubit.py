import json
import math

import numpy as np
import pytest
import sympy as sp

from dgs import vector_potential_qubit as v


def test_gauge_transform_sympy():
    assert v.derive_gauge_transform()["residual"] == 0


def test_symbolic_matrix_elements_match_numeric_closed_forms():
    sym = v.box_matrix_elements_symbolic()
    m, n = sp.symbols("m n", positive=True, integer=True)
    a = sp.Symbol("a", positive=True)
    E, X, P = v.box_matrices(6)
    for i in range(1, 7):
        for j in range(1, 7):
            if i == j:
                continue
            x_ij = float(sym["x_mn"].subs({m: i, n: j, a: 1}))
            d_ij = float(sym["ddx_mn"].subs({m: i, n: j, a: 1}))
            assert X[i - 1, j - 1] == pytest.approx(x_ij, abs=1e-14)
            assert (1j * P[i - 1, j - 1]).real == pytest.approx(d_ij, abs=1e-14)    # P = -i d/dx


def test_box_commutator_p_equals_i_omega_x():
    E, X, P = v.box_matrices(8)
    W = E[:, None] - E[None, :]
    off = ~np.eye(8, dtype=bool)
    assert np.allclose(P[off], (1j * W * X)[off], atol=1e-12)          # hbar = m = 1


def test_numerical_gauge_invariance_converges():
    rows = v.gauge_table((2, 8, 16))
    assert all(r["P2_length"] > 0.99 and r["P2_velocity"] > 0.99 for r in rows)
    assert rows[-1]["diff"] < 1e-5
    assert rows[-1]["diff"] < rows[0]["diff"]
    with pytest.raises(ValueError):
        v.simulate_box(4, "coulomb")


def test_pulse_vector_potential_vanishes_at_ends():
    A, E = v.pulse(np.array([0.0, 3.0]), 1.0, 5.0, 3.0)
    assert abs(A[0]) < 1e-15 and abs(A[1]) < 1e-15
    t = np.linspace(0.1, 2.9, 7); dt = 1e-6
    Ap, _ = v.pulse(t + dt, 1.0, 5.0, 3.0); Am, _ = v.pulse(t - dt, 1.0, 5.0, 3.0)
    _, E = v.pulse(t, 1.0, 5.0, 3.0)
    assert np.allclose(E, -(Ap - Am) / (2 * dt), atol=1e-6)


def test_derived_values():
    d = v.derived()
    assert d["transmon"]["f_transition_Hz"] == pytest.approx(4.75e9, rel=1e-12)    # 19 EC (Lean proves this)
    assert d["box_atom"]["Omega_velocity_gauge_rad_s"] == pytest.approx(d["box_atom"]["Omega_rad_s"], rel=1e-12)
    for p in d.values():
        assert p["t_pi_s"] == pytest.approx(math.pi / p["Omega_rad_s"])
    assert d["transmon"]["n_thermal"] < 1e-5 < 1e6 < d["csus_lc_analog"]["n_thermal"]     # quantum vs classical regime


def test_json_matches_bench_module():
    from dgs import qubit_analog_bench as qb
    p = v.load_platforms()["platforms"]["csus_lc_analog"]
    cp = qb.coil_pair_coupling(0.02, 30, 0.02, 0.255e-3, p["gap_m"])
    assert p["L_H"] == pytest.approx(cp["L"], rel=1e-8) and p["k"] == pytest.approx(cp["k"], rel=1e-8)


def test_sympy_table_agrees_with_derived():
    d = v.derived()
    t = v.sympy_table()
    tm = dict((q, val) for q, _, val in t["transmon"])
    assert tm["f01 (Hz)"] == pytest.approx(d["transmon"]["f_transition_Hz"], rel=1e-12)
    assert tm["Omega (rad/s)"] == pytest.approx(d["transmon"]["Omega_rad_s"], rel=1e-12)
    cs = dict((q, val) for q, _, val in t["csus_lc_analog"])
    assert cs["swap time pi/Omega (s)"] == pytest.approx(d["csus_lc_analog"]["t_pi_s"], rel=1e-12)
    bx = dict((q, val) for q, _, val in t["box_atom"])
    assert bx["Omega, velocity gauge (rad/s)"] == pytest.approx(bx["Omega, length gauge (rad/s)"], rel=1e-12)
    assert bx["A0 = E0/omega (V s/m)"] == pytest.approx(d["box_atom"]["vector_potential_amp"], rel=1e-12)


def test_lean_file_is_in_sync_with_json():
    assert v.LEAN_PATH.read_text(encoding="utf-8") == v.lean_source(), \
        "regenerate: python -m dgs.vector_potential_qubit --write-lean"


def test_lean_literals_are_exact():
    assert v._q(250000000) == "(250000000 : ℝ)"
    assert v._q(0.0806173509) == "((806173509 : ℝ) / 10000000000)"


def test_table_compare_catches_drift():
    a = v.parse_table("\n".join(v.table_lines()))
    b = dict(a); b["transmon.f_Rabi_Hz"] *= 1 + 1e-6; del b["box_atom.x12_m"]
    bad = {k for k, *_ in v.compare_tables(a, b)}
    assert bad == {"transmon.f_Rabi_Hz", "box_atom.x12_m"}
    assert v.compare_tables(a, dict(a)) == []


def test_bad_json(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"constants": {}}))
    with pytest.raises(ValueError, match="platforms"):
        v.load_platforms(f)
