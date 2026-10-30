"""Tests for dgs/sympy_to_c.py and the Circuit polarity/power table in dgs/field_circuits.py."""
import pytest
import sympy as sp

from dgs import sympy_to_c as S
from dgs.field_circuits import Circuit
from dgs.sympy_to_c import x


def test_operator_table_c_text():
    c_text = [row[2] for row in S.operator_table()]
    assert c_text[:6] == ["a == b", "a != b", "a < b", "a <= b", "a > b", "a >= b"]
    assert c_text[6] == "y = a + b;"


@pytest.mark.parametrize("f, kind", [
    (x**2, "even"), (sp.cos(x), "even"), (x**3, "odd"), (sp.sin(x), "odd"),
    (sp.exp(x), "neither"), (x**2 + x, "neither"), (sp.Integer(0), "zero"), (sp.Abs(x), "even"),
])
def test_parity(f, kind):
    assert S.parity(f) == kind


def test_even_odd_parts_of_exp_are_cosh_sinh():
    ev, od = S.even_odd_parts(sp.exp(x))
    assert sp.simplify(ev - sp.cosh(x)) == 0 and sp.simplify(od - sp.sinh(x)) == 0
    assert sp.simplify(ev + od - sp.exp(x)) == 0


def test_puck_inertia_and_energy():
    I = S.puck_inertia()
    assert sp.simplify(I["I_axis"] - S.M * S.R**2 / 2) == 0                 # height drops out
    assert sp.simplify(I["I_diameter"] - S.M * (S.R**2 / 4 + S.h**2 / 12)) == 0
    E = S.puck_energy()
    # a spinning puck with rim speed = slide speed (w R = v) has 1/3 of its energy in the spin
    assert sp.simplify(E["spin_fraction"].subs(S.w, S.v / S.R)) == sp.Rational(1, 3)


def test_wall_bounce_logic():
    f = S.wall_bounce()
    vals = {S.R: 0.05, S.xmin: 0, S.xmax: 1, S.e: 0.9}
    assert f.subs({**vals, S.px: 0.04, S.vx: -2}) == pytest.approx(1.8)      # hits left wall
    assert f.subs({**vals, S.px: 0.04, S.vx: 2}) == 2                       # already leaving
    assert f.subs({**vals, S.px: 0.5, S.vx: -2}) == -2                      # mid-table
    assert f.subs({**vals, S.px: 0.96, S.vx: 2}) == pytest.approx(-1.8)     # right wall
    code = sp.ccode(f)
    assert "<=" in code and ">=" in code and "&&" in code and "?" in code


def test_generated_c_matches_sympy():
    E = S.puck_energy()["K"]
    fns = [S.c_function("puck_KE", E, [S.M, S.R, S.v, S.w]),
           S.c_function("wall_vx", S.wall_bounce(), [S.px, S.vx, S.R, S.xmin, S.xmax, S.e])]
    calls = ["puck_KE(0.17, 0.05, 2.0, 30.0)", "wall_vx(0.04, -2.0, 0.05, 0.0, 1.0, 0.9)",
             "wall_vx(0.5, -2.0, 0.05, 0.0, 1.0, 0.9)"]
    try:
        got, src = S.compile_and_run(fns, calls)
    except S.ToolUnavailable:
        pytest.skip("gcc not installed")
    want = float(E.subs({S.M: 0.17, S.R: 0.05, S.v: 2.0, S.w: 30.0}))
    assert got == pytest.approx([want, 1.8, -2.0])
    assert "else" in src and "<=" in src and "&&" in src      # Piecewise became if / else


def test_c_function_validation():
    with pytest.raises(ValueError):
        S.c_function("2bad", S.M, [S.M])
    with pytest.raises(ValueError):
        S.c_function("f", S.M * S.v, [S.M])


# ---- netlist polarity and power ---------------------------------------------------------------
def test_power_table_resistive():
    # 12 V source, 2 ohm and 4 ohm in series: I = 2 A
    ckt = Circuit().V("Vs", "a", "0", 12).R("R1", "a", "b", 2).R("R2", "b", "0", 4)
    rows = {r["element"]: r for r in ckt.element_table()}
    assert rows["R1"]["p = v i (absorbed)"] == 8 and rows["R2"]["p = v i (absorbed)"] == 16
    assert rows["Vs"]["p = v i (absorbed)"] == -24 and rows["Vs"]["role"] == "delivers"
    assert rows["Vs"]["i (+ to - through it)"] == -2         # current leaves the + terminal
    assert ckt.tellegen() == 0


def test_tellegen_symbolic_and_absorbing_source():
    R1, R2, V1, V2 = sp.symbols("R_1 R_2 V_1 V_2", positive=True)
    # two sources fighting through a resistor: the weaker one absorbs (it is being charged)
    ckt = Circuit().V("V1", "a", "0", 10).R("R1", "a", "b", 1).V("V2", "b", "0", 4)
    rows = {r["element"]: r for r in ckt.element_table()}
    assert rows["V2"]["role"] == "absorbs" and rows["V1"]["role"] == "delivers"
    sym = Circuit().V("V1", "a", "0", V1).R("R1", "a", "b", R1).R("R2", "b", "0", R2)
    assert sym.tellegen() == 0
