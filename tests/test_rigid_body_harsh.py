"""Tests for dgs.rigid_body_core, dgs.rigid_body_harsh and the micro:bit script."""
import importlib.util
import pathlib

import numpy as np
import pytest

from dgs import rigid_body_core as core
from dgs import rigid_body_harsh as r

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "projects" / "microbit_statics" / "main.py"


def _block(text):
    a = text.index("# --- BEGIN SHARED CORE")
    b = text.index("# --- END SHARED CORE ---")
    return text[a:b]


def test_solve3_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = rng.standard_normal((3, 3)) + 3 * np.eye(3)
        b = rng.standard_normal(3)
        assert np.allclose(core.solve3(A.tolist(), b.tolist()), np.linalg.solve(A, b))


def test_solve3_pivots_and_rejects_singular():
    assert np.allclose(core.solve3([[0, 1, 0], [1, 0, 0], [0, 0, 2]], [2, 3, 4]), [3, 2, 2])
    with pytest.raises(ValueError):
        core.solve3([[1, 2, 3], [2, 4, 6], [0, 0, 1]], [1, 2, 3])
    with pytest.raises(ValueError):
        core.solve3([[1, 0, 0], [0, 1, 0], [0, 0, float("nan")]], [1, 1, 1])


def test_mast_matches_free_body_diagram_and_balances():
    for v, t_ice in [(0, 0), (10, 0), (35, 0), (35, 0.025)]:
        assert np.allclose(r.solve_mast(v, t_ice), r.solve_mast_numpy(v, t_ice))
        assert r.equilibrium_residual(v, t_ice) < 1e-9


def test_calm_mast_hand_solution():
    Ax, Ay, T = r.solve_mast(0.0)
    assert abs(Ax) < 1e-12 and abs(T) < 1e-12
    assert np.isclose(Ay, (r.MAST["pole_mass"] + r.MAST["head_mass"]) * r.G)


def test_wind_moment_hand_check():
    # sum M_A = 0: T * (d H / L) = F_pole H/2 + F_head H  (cable moment arm about the pin)
    v = 30.0
    m = r.MAST
    L = np.hypot(m["d"], m["H"])
    F_pole = r.drag(v, m["pole_D"] * m["H"], m["cd_pole"])
    F_head = r.drag(v, m["head_area"], m["cd_head"])
    T_hand = (F_pole * m["H"] / 2 + F_head * m["H"]) / (m["d"] * m["H"] / L)
    assert np.isclose(r.solve_mast(v)[2], T_hand)


def test_tension_is_exactly_quadratic_in_wind():
    vs = np.linspace(0, 40, 17)
    T0, c, r2 = r.regress_quadratic(vs, [r.solve_mast(v)[2] for v in vs])
    assert abs(T0) < 1e-9 and r2 > 1 - 1e-12 and c > 0


def test_ice_raises_tension_and_light_is_negligible():
    assert r.solve_mast(35, 0.025)[2] > r.solve_mast(35, 0.0)[2]
    ratio = r.drag(1.0, 0.04, 1.1) / r.radiation_force(1000, 0.04)
    assert 1e5 < ratio < 1e6                                       # ~2e5 for a 1 m/s breeze
    assert np.isclose(r.radiation_force(1000, 1.0, 1.0), 2000 / r.C_LIGHT)


def test_ops_count_and_line_regression():
    assert r.ops_per_solve() == 61
    ov, sl, r2 = r.regress_line([1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5])
    assert np.isclose(sl, 1.0) and np.isclose(ov, 0.5) and np.isclose(r2, 1.0)


def test_microbit_script_shares_the_core_verbatim():
    assert _block(MAIN.read_text(encoding="utf-8")) == _block((ROOT / "dgs" / "rigid_body_core.py").read_text(encoding="utf-8"))


def test_microbit_script_runs_on_cpython_and_agrees():
    spec = importlib.util.spec_from_file_location("microbit_main", MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not mod.ON_DEVICE
    T, us_op = mod.run(Ns=(50, 100, 200))
    assert np.isclose(T, r.solve_mast(35.0, 0.025)[2])
    assert us_op > 0


@pytest.mark.parametrize("call", [
    lambda: core.guyed_mast_reactions(0.0, 1.0, []),
    lambda: core.moment_rows([], [(1, 0, 0, 0)]),
    lambda: r.drag(1.0, -1.0, 1.0),
    lambda: r.radiation_force(1000, 1.0, 1.5),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
