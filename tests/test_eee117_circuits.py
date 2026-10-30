"""Tests for dgs/eee117_circuits.py: textbook results, node-voltage vs mesh agreement, op-amps, DAC."""
from itertools import product

import pytest
import sympy as sp

from dgs import eee117_circuits as E
from dgs.field_circuits import Circuit

R1, R2, R3, R4, Rf, Rin, Vs, Is = sp.symbols("R_1 R_2 R_3 R_4 R_f R_in V_s I_s", positive=True)


def test_ohm_and_power():
    r = E.ohm(V=12, R=4)
    assert r["I"] == 3 and r["P"] == 36
    assert E.ohm(I=2, R=5)["V"] == 10 and E.ohm(V=9, I=3)["R"] == 3
    pf = E.power_forms(12, 3, 4)
    assert pf["VI"] == pf["I2R"] == pf["V2/R"] == 36
    with pytest.raises(ValueError):
        E.ohm(V=1)
    with pytest.raises(ValueError):
        E.ohm(V=1, R=-2)


def test_series_parallel_dividers():
    assert E.series(1, 2, 3) == 6
    assert E.parallel(6, 3) == 2
    assert sp.simplify(E.parallel(R1, R2) - R1 * R2 / (R1 + R2)) == 0
    assert E.voltage_divider(10, 3, 2) == 6                       # 10 V * 3/(3+2)
    assert E.current_divider(9, 3, 6) == 6                        # 9 A: 3 ohm branch takes 6 A
    assert sp.simplify(E.current_divider(Is, R1, R2) + E.current_divider(Is, R2, R1) - Is) == 0
    with pytest.raises(ValueError):
        E.parallel()


def test_wheatstone():
    assert E.wheatstone_output(10, 100, 200, 300, 600) == 0       # 100*600 = 200*300
    assert E.wheatstone_unknown(100, 200, 300) == 600
    assert E.wheatstone_output(10, 100, 100, 100, 110) != 0
    # matches the full node-voltage bridge with no galvanometer
    ckt = Circuit().V("Vs", "t", "0", Vs).R("R1", "t", "a", R1).R("R2", "a", "0", R2) \
        .R("R3", "t", "b", R3).R("R4", "b", "0", R4)
    sol = ckt.solve()
    assert sp.simplify(sol[sp.Symbol("V_a")] - sol[sp.Symbol("V_b")] - E.wheatstone_output(Vs, R1, R2, R3, R4)) == 0


def test_delta_wye_round_trip_and_equivalence():
    y = E.delta_to_wye(R1, R2, R3)
    back = E.wye_to_delta(*y)
    assert all(sp.simplify(a - b) == 0 for a, b in zip(back, (R1, R2, R3)))
    assert E.delta_to_wye(30, 30, 30) == (10, 10, 10)
    # terminal-pair resistance: wye R_a + R_b == delta R12 || (R23 + R31)
    assert sp.simplify(y[0] + y[1] - E.parallel(R1, R2 + R3)) == 0


def test_mesh_matches_node_voltage():
    # two meshes: 10 V source in mesh A, R1 on A only, R2 shared, R3 on B only
    m = E.Mesh().V("Vs", 10, "A").R("R1", 2, "A").R("R2", 4, "A", "B").R("R3", 4, "B")
    sol = m.solve()
    # node-voltage of the same circuit
    ckt = Circuit().V("Vs", "a", "0", 10).R("R1", "a", "b", 2).R("R2", "b", "0", 4).R("R3", "b", "0", 4)
    nv = ckt.solve()
    assert sol[sp.Symbol("I_A")] == 10 / sp.Integer(4)            # 10 / (2 + 4||4)
    assert m.branch_current("R2") * 4 == nv[sp.Symbol("V_b")]
    A, b = m.matrix()
    assert A == sp.Matrix([[6, -4], [-4, 8]])                    # symmetric resistance matrix
    assert "KVL" in m.paper()


@pytest.mark.parametrize("fn, args, expected", [
    (E.inverting_amp, (Rin, Rf), -Rf / Rin),
    (E.noninverting_amp, (Rin, Rf), 1 + Rf / Rin),
])
def test_opamp_gains(fn, args, expected):
    ckt, vout = fn(*args)
    assert sp.simplify(vout / sp.Symbol("V_s") - expected) == 0
    assert ckt.tellegen() == 0                                    # op-amp rails balance the books


def test_source_symbol_clashing_with_node_voltage_is_rejected():
    # a source valued V_in on node "in" would turn the constraint into V_in = V_in (silently True)
    ckt = Circuit().V("Vin", "in", "0", sp.Symbol("V_in")).R("R1", "in", "0", 1)
    with pytest.raises(ValueError, match="node-voltage names"):
        ckt.solve()


def test_summing_amp():
    _, vout = E.summing_amp([1000, 2000], 4000, [1, 2])
    assert vout == -(4 * 1 + 2 * 2)


@pytest.mark.parametrize("bits", list(product((0, 1), repeat=3)))
def test_r2r_dac_is_binary(bits):
    _, vout = E.r2r_dac(bits, Vref=8)
    code = int("".join(map(str, bits)), 2)
    assert vout == code                                            # Vref * code / 2^3 with Vref = 8
