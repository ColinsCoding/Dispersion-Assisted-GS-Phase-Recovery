"""Tests for dgs.dephasing_phase_retrieval and dgs.qis_pathway_planning."""
import numpy as np
import pytest
import sympy as sp

from dgs import dephasing_phase_retrieval as dp
from dgs import qis_pathway_planning as q


# ── quantum half ─────────────────────────────────────────────────────

def test_dephasing_kills_coherence_not_population():
    rho12, drho11, sx = dp.dephasing_symbolic()
    t, g = sp.symbols("t gamma", positive=True)
    a, b = sp.symbols("a b", positive=True)
    phi = sp.symbols("phi", real=True)
    assert drho11 == 0
    assert sp.simplify(rho12 - a * b * sp.exp(-g * t - sp.I * phi)) == 0
    assert sp.simplify(sx - 2 * a * b * sp.exp(-g * t) * sp.cos(phi)) == 0


def test_rotate_then_measure_z_reads_each_bloch_axis():
    x, y, z = sp.symbols("x y z", real=True)
    P = dp.measurement_probabilities_symbolic()
    assert list(P.values()) == [(1 + z) / 2, (1 + x) / 2, (1 + y) / 2]


def test_tomography_recovers_state_and_z_only_cannot():
    r = dp.bloch_from_angles(1.0, 0.7)
    assert dp.state_fidelity(r, dp.tomography(r, 100000, rng=0)) > 0.995
    z_only = dp.state_fidelity(r, dp.tomography(r, 100000, "Z", rng=0))
    assert z_only == pytest.approx((1 + r[2] ** 2) / 2, abs=0.01)
    with pytest.raises(ValueError):
        dp.tomography(r, 0)


def test_dephasing_lowers_tomographic_fidelity_monotonically():
    r = dp.bloch_from_angles(np.pi / 2, 0.3)
    f = [dp.state_fidelity(r, dp.dephase_bloch(r, gt)) for gt in (0, 0.5, 1, 3)]
    assert f[0] == pytest.approx(1) and all(np.diff(f) < 0)
    assert f[-1] == pytest.approx((1 + np.exp(-3)) / 2)


# ── classical half ──────────────────────────────────────────────────

def test_coherence_factor_gaussian():
    assert dp.coherence_factor(0) == 1
    assert dp.coherence_factor(1.0) == pytest.approx(np.exp(-0.5))
    rng = np.random.default_rng(0)
    s = rng.normal(0, 0.8, 200000)
    assert abs(np.mean(np.exp(1j * s))) == pytest.approx(dp.coherence_factor(0.8), abs=5e-3)


def test_phase_fidelity_ignores_global_phase():
    phi = np.linspace(0, 6, 100)
    assert dp.phase_fidelity(phi + 1.3, phi) == pytest.approx(1)


def test_retrieval_degrades_with_classical_dephasing():
    out = dp.retrieval_vs_dephasing([0.0, 0.6, 2.0], n_realizations=20, n_iter=60)
    assert out["fidelity"][0] > 0.99
    assert out["fidelity"][0] > out["fidelity"][1] > out["fidelity"][2] - 0.05
    assert np.all(np.diff(out["contrast"]) < 0)
    assert out["fidelity"][-1] < 0.7 and 0 <= out["chance"] < 0.7


def test_flow_diagram_has_alt_text_and_matching_stages():
    import matplotlib
    matplotlib.use("Agg")
    fig, alt = dp.flow_diagram()
    assert len(dp.CLASSICAL_CHAIN) == len(dp.QUANTUM_CHAIN) == 6
    assert "photodiode" in alt.lower() and "tomography" in alt.lower()
    assert len(fig.axes[0].patches) == 12


# ── planning ────────────────────────────────────────────────────────

def test_pathway_is_a_valid_dag_and_modules_exist():
    order = q.topological_order(q.PATHWAY)
    pos = {k: i for i, k in enumerate(order)}
    for k, (pre, _) in q.PATHWAY.items():
        assert all(pos[p] < pos[k] for p in pre)
    assert all(q.module_available(m) for _, m in q.PATHWAY.values())
    L = q.layers(q.PATHWAY)
    assert L[-1] == ["Capstone: bench I1/I2 measurement"]


def test_cycle_and_unknown_prereq_rejected():
    with pytest.raises(ValueError):
        q.topological_order({"a": (("b",), None), "b": (("a",), None)})
    with pytest.raises(ValueError):
        q.topological_order({"a": (("ghost",), None)})


def test_roadmap_critical_path_and_slack():
    total, path, finish = q.critical_path()
    assert total == 45 and path[0] == "Theory & SymPy derivations"
    assert "Bench I1/I2 measurement" in path
    s = q.slack()
    assert all(s[k] == 0 for k in path)
    assert s["Customer-discovery interviews"] > 0
    assert q.trl_reached()[0] == 7
