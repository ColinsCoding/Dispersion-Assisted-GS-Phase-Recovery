"""Tests for dgs/charge_to_mass.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.charge_to_mass import (
    PARTICLES, textbook_ratio, precise_ratio, ratio_si, odd_one_out, binding_energy_MeV,
    thomson_symbolic, thomson_deflection, millikan_symbolic, millikan_charge, is_physical_charge,
    M_ALPHA_U, M_DEUTERON_U,
)


def test_textbook_answer_is_proton():
    t = {k: textbook_ratio(k) for k in PARTICLES}
    assert odd_one_out(t) == "proton"
    assert t["proton"] == 1 and all(v == sp.Rational(1, 2) for k, v in t.items() if k != "proton")


def test_precise_ratios_split_by_binding():
    r = {k: precise_ratio(k) for k in PARTICLES}
    halves = [r[k] for k in PARTICLES if k != "proton"]
    assert max(halves) - min(halves) > 1e-3                       # not exactly equal
    assert r["alpha particle"] == max(halves)                     # most tightly bound -> lightest per A
    assert ratio_si("proton") == pytest.approx(9.5788e7, rel=1e-4)


def test_binding_energies():
    assert binding_energy_MeV(2, 2, M_ALPHA_U) == pytest.approx(28.296, abs=0.01)
    assert binding_energy_MeV(1, 1, M_DEUTERON_U) == pytest.approx(2.2246, abs=1e-3)


def test_thomson():
    v_sel, y_expr, qm = thomson_symbolic()
    E, B, L, y = sp.symbols("E B L y", positive=True)
    assert sp.simplify(qm - 2 * y * E / (B ** 2 * L ** 2)) == 0
    qm_p = ratio_si("proton")
    ydef, v = thomson_deflection(qm_p, 1.5e4, 0.01, 0.05)
    assert v == pytest.approx(1.5e6)
    assert 2 * ydef * 1.5e4 / (0.01 ** 2 * 0.05 ** 2) == pytest.approx(qm_p)   # recovers q/m


def test_millikan_round2():
    expr, (m, g, V, d) = millikan_symbolic()
    assert sp.simplify(expr - m * g * d / V) == 0
    _, n = millikan_charge(1.00e-17, 40.8, 1.0)
    assert n == pytest.approx(15.0, abs=0.02)
    assert [is_physical_charge(x) for x in (20, 10, 1.5, 5, 15)] == [True, True, False, True, True]
    with pytest.raises(ValueError):
        millikan_charge(1e-17, 40.8, 0)
