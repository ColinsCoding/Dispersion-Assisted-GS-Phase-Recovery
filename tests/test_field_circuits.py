"""Tests for dgs/field_circuits.py: field energy = circuit energy, Poynting, cavity, nodal analysis."""
import pytest
import sympy as sp

from dgs import field_circuits as F
from dgs.field_circuits import Circuit, s


@pytest.mark.parametrize("fn", [F.capacitor_field_energy, F.solenoid_field_energy,
                                F.wire_poynting, F.charging_capacitor_poynting])
def test_field_energy_matches_circuit_energy(fn):
    assert fn()["match"]


def test_poynting_points_inward_and_equals_joule_heat():
    p = F.wire_poynting()
    assert sp.simplify(p["P_in"] - F.I**2 * F.R) == 0
    assert p["S"].is_positive                        # E x B / mu0 radially inward, magnitude > 0


def test_capacitor_series_and_cavity():
    ser = F.capacitor_field_series(2)
    x = F.w * F.r / F.c
    assert sp.simplify(ser - (1 - x**2 / 4 + x**4 / 64)) == 0
    assert F.first_bessel_zero() == pytest.approx(2.404826, abs=1e-6)
    assert F.cavity_resonance_hz(0.01) == pytest.approx(11.47e9, rel=1e-3)     # 1 cm plates
    with pytest.raises(ValueError):
        F.cavity_resonance_hz(0)


def test_lumped_table_complete():
    assert len(F.LUMPED_ASSUMPTIONS) == 5
    for row in F.LUMPED_ASSUMPTIONS:
        assert all(row[k] for k in ("assumption", "maxwell", "dropped", "gives", "fails_when"))


def test_examples_match_textbook_results():
    R, R1, R2, R3, R4, C, L = sp.symbols("R R_1 R_2 R_3 R_4 C L", positive=True)
    H = {name: ckt.transfer(out, src) for name, ckt, out, src in F.example_circuits()}
    assert sp.simplify(H["voltage divider"] - R2 / (R1 + R2)) == 0
    assert sp.simplify(H["RC low-pass"] - 1 / (1 + s * R * C)) == 0
    assert sp.simplify(H["series RLC band-pass"] - s * R * C / (s**2 * L * C + s * R * C + 1)) == 0
    # balanced bridge: R1 R4 = R2 R3 -> no current in the galvanometer, V_a = V_b
    bridge = F.example_circuits()[3][1]
    sol = bridge.solve()
    bal = {R4: R2 * R3 / R1}
    assert sp.simplify((sol[sp.Symbol("V_a")] - sol[sp.Symbol("V_b")]).subs(bal)) == 0


def test_numeric_circuit_and_current_source():
    ckt = Circuit().I("Is", "0", "a", 2).R("R1", "a", "0", 5).R("R2", "a", "b", 10).R("R3", "b", "0", 10)
    sol = ckt.solve()
    # 2 A into node a; R1 || (R2 + R3) = 5 || 20 = 4 ohm -> V_a = 8 V, V_b = 4 V
    assert sol[sp.Symbol("V_a")] == 8 and sol[sp.Symbol("V_b")] == 4
    A, b = ckt.matrix()
    assert A.shape == (2, 2)
    assert "KCL" in ckt.paper()


def test_bad_circuits_rejected():
    with pytest.raises(ValueError):
        Circuit().R("R1", "a", "a", 1)
    with pytest.raises(ValueError):
        Circuit().R("R1", "a", "b", -1)
    with pytest.raises(ValueError):
        Circuit().R("R1", "a", "b", 1).R("R1", "b", "0", 1)
    with pytest.raises(ValueError):
        Circuit().R("R1", "a", "b", 1).nodes
    with pytest.raises(KeyError):
        Circuit().V("Vs", "a", "0", 1).R("R1", "a", "0", 1).transfer("a", "nope")
