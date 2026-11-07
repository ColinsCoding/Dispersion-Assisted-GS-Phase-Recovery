import math

import pytest
import sympy as sp

from dgs import circuit_rules as cr
from dgs.circuit_rules import Circuit, s


@pytest.mark.parametrize("r", cr.RULES, ids=[r.name for r in cr.RULES])
def test_every_rule_derives_from_the_axioms(r):
    assert r.verify(), r.name


def test_rulebook_size_and_categories():
    assert len(cr.RULES) >= 50
    assert {r.category for r in cr.RULES} == set(cr.CATEGORIES)
    assert len({r.name for r in cr.RULES}) == len(cr.RULES)


def test_checks_can_fail():
    # the classic student mistake: capacitors in series do NOT add
    c = Circuit().C("C1", "a", "m", cr.C1).C("C2", "m", "0", cr.C2)
    assert not cr._same(cr.impedance(c, "a"), 1 / (s * (cr.C1 + cr.C2)))
    # and a 1 % wrong divider is caught
    sol = Circuit().V("Vs", "i", "0", 1).R("R1", "i", "o", 1000).R("R2", "o", "0", 1000).solve()
    assert not cr._same(sol.V("o"), sp.Rational(101, 200))


def test_numeric_dc_operating_point():
    sol = Circuit().V("Vs", "i", "0", 12).R("R1", "i", "o", 1000).R("R2", "o", "0", 2000) \
        .L("L1", "o", "x", 1e-3).R("R3", "x", "0", 2000).C("C1", "o", "0", 1e-6).solve(s_value=0)
    assert float(sol.V("o")) == pytest.approx(6.0)          # 1k over (2k || 2k), inductor shorts, cap open
    assert float(sol.I("L1")) == pytest.approx(3e-3)


def test_input_validation():
    with pytest.raises(ValueError, match="positive"):
        Circuit().R("R1", "a", "0", -5)
    with pytest.raises(ValueError, match="duplicate"):
        Circuit().R("R1", "a", "0", 5).R("R1", "b", "0", 5)
    with pytest.raises(ValueError, match="0 <= k <= 1"):
        Circuit().L("L1", "a", "0", 1).L("L2", "b", "0", 1).K("K", "L1", "L2", 1.2)
    with pytest.raises(ValueError, match="no inductor"):
        Circuit().L("L1", "a", "0", 1).K("K", "L1", "Lx", 0.5)
    with pytest.raises(ValueError, match="shorted"):
        Circuit().R("R1", "a", "a", 5)
    with pytest.raises(ValueError, match="floats|no solution"):
        Circuit().V("Vs", "a", "0", 1).C("C1", "a", "b", 1e-6).R("R1", "b", "c", 10).solve(s_value=0)


def test_shock_bands():
    a = cr.shock_assessment(12, ac=True)
    assert a["within_elv"] and a["band"] == "perceptible, startling, usually harmless"
    b = cr.shock_assessment(120, ac=True)                       # ~76 mA through ~1.6 kOhm
    assert not b["within_elv"] and "DANGEROUS" in b["band"]
    assert cr.shock_assessment(48, ac=False)["within_elv"]
    assert not cr.shock_assessment(48, ac=True, wet=True)["within_elv"]


def test_bleeder_and_kickback():
    assert cr.bleeder_time(100e-6, 10e3, 400, 60) == pytest.approx(1.0 * math.log(400 / 60))
    assert cr.bleeder_time(1e-6, 1e3, 10) == 0.0
    assert cr.inductor_kickback(10e-3, 1.0, 1e-6) == pytest.approx(1e4)


def test_safety_check_flags_problems():
    c = Circuit().V("Vs", "i", "0", 24).R("R1", "i", "o", 100).L("L1", "o", "0", 10e-3)
    report = {(f["part"], f["quantity"]): f for f in cr.safety_check(c)}
    assert report[("R1", "power (W)")]["status"] == "WARN"           # 5.76 W in a 1/4 W part
    assert report[("L1", "switch-off kickback (V)")]["status"] == "WARN"   # 240 mA * 10 mH / 1 us = 2.4 kV
    assert report[("Vs", "source voltage (V)")]["status"] == "OK"
    ok = Circuit().V("Vs", "i", "0", 5).R("R1", "i", "o", 10e3).C("C1", "o", "0", 10e-6).R("R2", "o", "0", 10e3)
    assert all(f["status"] == "OK" for f in cr.safety_check(ok, ratings={"C1": 16}))
