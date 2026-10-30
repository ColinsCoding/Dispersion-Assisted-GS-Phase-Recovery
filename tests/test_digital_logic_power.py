"""Tests for dgs/digital_logic_power.py: gates, NAND universality, MUX, K-maps, power/energy."""
from itertools import product

import pytest
import sympy as sp

from dgs import digital_logic_power as D


def test_greek_alphabet_alpha_to_omega():
    g = D.greek_alphabet()
    assert len(g) == 24 and g[0][0] == "alpha" and g[-1][0] == "omega"
    assert sp.pretty(g[10][1], use_unicode=True) == "λ"
    assert sp.pretty(g[-1][2], use_unicode=True) == "Ω"


@pytest.mark.parametrize("gate, outputs", [
    ("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("NAND", [1, 1, 1, 0]),
    ("NOR", [1, 0, 0, 0]), ("XOR", [0, 1, 1, 0]), ("XNOR", [1, 0, 0, 1]),
])
def test_truth_tables(gate, outputs):
    assert [row[-1] for row in D.truth_table(gate)] == outputs


def test_truth_tables_match_sympy_expressions():
    for g, expr in D.GATE_EXPR.items():
        for row in D.truth_table(g):
            env = {D.A: bool(row[0])} if g == "NOT" else {D.A: bool(row[0]), D.B: bool(row[1])}
            assert bool(expr.subs(env)) == bool(row[-1]), g


def test_nand_universality_and_bad_input():
    assert D.check_nand_universality()
    assert D.truth_table("NOT") == [(0, 1), (1, 0)]
    with pytest.raises(ValueError):
        D.AND(2, 1)
    with pytest.raises(KeyError):
        D.truth_table("BUFFERZ")


def test_mux_selects_and_expression_agrees():
    data = [0, 1, 1, 0, 1, 0, 0, 1]
    expr, S, Dsym = D.mux_expression(3)
    for i, sel in enumerate(product((0, 1), repeat=3)):
        assert D.mux(sel, data) == data[i]
        env = {**{s: bool(b) for s, b in zip(S, sel)}, **{d: bool(v) for d, v in zip(Dsym, data)}}
        assert bool(expr.subs(env)) == bool(data[i])
    with pytest.raises(ValueError):
        D.mux((0, 1), [0, 1, 1])


def test_function_via_mux_is_xor3():
    data = D.function_via_mux([1, 2, 4, 7], 3)     # odd parity
    for sel in product((0, 1), repeat=3):
        assert D.mux(sel, data) == sum(sel) % 2


def test_gray_code_neighbours_differ_by_one_bit():
    for n in (1, 2, 3):
        g = D.gray(n)
        for a, b in zip(g, g[1:] + g[:1]):
            assert sum(x != y for x, y in zip(a, b)) == 1


@pytest.mark.parametrize("minterms, dc, vars_, literals", [
    ([0, 2, 5, 7, 8, 10, 13, 15], [], "ABCD", 4),       # B'D' + BD
    ([1, 3, 5, 7], [], "ABC", 1),                        # C
    ([0, 1, 2, 5, 8, 9, 10], [], "ABCD", 7),             # B'C' + B'D' + A'C'D
    ([1, 3, 7, 11, 15], [0, 2, 5], "ABCD", 4),           # CD + A'B'
])
def test_kmap_minimal_and_correct(minterms, dc, vars_, literals):
    grid, rows, cols, sop = D.kmap(minterms, list(vars_), dc)
    assert D.verify_sop(sop, minterms, list(vars_), dc)
    assert D.literal_count(sop) == literals
    assert sum(cell == 1 for line in grid for cell in line) == len(minterms)


def test_kmap_rejects_bad_input():
    with pytest.raises(ValueError):
        D.kmap([16], list("ABCD"))
    with pytest.raises(ValueError):
        D.kmap([1], list("ABCDE"))
    with pytest.raises(ValueError):
        D.kmap([1], list("AB"), dontcares=[1])


def test_power_three_ways_agree():
    # 2 A through a 6 ohm resistor: 12 V, 24 W
    assert D.power_electrical(I=2, R=6) == D.power_electrical(I=2, V=12) == D.power_electrical(V=12, R=6) == 24
    # lifting 10 kg at constant 1.2 m/s: F v equals (m g h)/t
    F, v, t = 10 * D.G_EARTH, 1.2, 5.0
    assert D.power_force_velocity(F, v) == pytest.approx(D.power_rate_of_work(D.energy_from_mass(10, h=v * t), t))
    assert all(D.check_power_units().values())
    with pytest.raises(ValueError):
        D.power_electrical(I=1)


def test_energy_power_time_and_sources():
    assert D.energy_power_time(P=60, t=3600)["E_J"] == 216000
    assert D.energy_power_time(E=100, P=25)["t_s"] == 4
    with pytest.raises(ValueError):
        D.energy_power_time(E=1)
    # 1 cm^2 plates, 1 mm gap, 10 V: C = 0.885 pF, U = 44.3 pJ
    assert D.energy_capacitor_geometry(1e-4, 1e-3, 10) == pytest.approx(4.427e-11, rel=1e-3)
    assert D.energy_from_mass(1e-3, rest=True) == pytest.approx(8.988e13, rel=1e-3)
    assert D.energy_from_charge(1.602176634e-19, V=1) == pytest.approx(1.602e-19, rel=1e-3)   # 1 eV
    assert D.energy_from_charge(1e-6, q2_C=1e-6, r_m=0.1) == pytest.approx(0.0899, rel=1e-3)
    with pytest.raises(ValueError):
        D.energy_capacitor_geometry(1e-4, 0, 10)
