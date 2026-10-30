"""Tests for dgs/engineering_economics.py against factor identities and hand-worked examples."""
import math

import pytest
import sympy as sp

from dgs import engineering_economics as E


def test_factor_table_values():
    # standard interest-table entries (i = 10%, n = 5)
    assert E.p_given_f(0.10, 5) == pytest.approx(0.6209, abs=1e-4)
    assert E.p_given_a(0.10, 5) == pytest.approx(3.7908, abs=1e-4)
    assert E.a_given_p(0.10, 5) == pytest.approx(0.26380, abs=1e-5)
    assert E.f_given_a(0.10, 5) == pytest.approx(6.1051, abs=1e-4)
    assert E.a_given_f(0.10, 5) == pytest.approx(0.16380, abs=1e-5)


@pytest.mark.parametrize("i, n", [(0.05, 1), (0.08, 10), (0.12, 30), (0.0, 7)])
def test_factor_identities(i, n):
    assert E.a_given_p(i, n) * E.p_given_a(i, n) == pytest.approx(1)
    if i > 0:
        assert E.a_given_p(i, n) == pytest.approx(E.a_given_f(i, n) + i)   # A/P = A/F + i
    assert E.f_given_a(i, n) == pytest.approx(E.p_given_a(i, n) * E.f_given_p(i, n))


def test_annuity_formula_matches_geometric_sum_symbolically():
    i, n, t = sp.symbols("i n t", positive=True)
    s = sp.summation((1 + i) ** -t, (t, 1, 5)).subs(i, sp.Rational(1, 10))
    closed = ((1 + i) ** 5 - 1) / (i * (1 + i) ** 5)
    assert sp.nsimplify(s) == sp.nsimplify(closed.subs(i, sp.Rational(1, 10)))


def test_npv_irr_hand_example():
    flows = [-1000, 500, 600]
    assert E.npv(flows, 0.10) == pytest.approx(-1000 + 500 / 1.1 + 600 / 1.21)
    r = E.irr(flows)
    assert E.npv(flows, r) == pytest.approx(0, abs=1e-8)
    # x = 1/(1+r) solves 600 x^2 + 500 x - 1000 = 0
    x = (-500 + math.sqrt(500**2 + 4 * 600 * 1000)) / (2 * 600)
    assert r == pytest.approx(1 / x - 1, abs=1e-10)


def test_eac_and_breakeven():
    eac = E.equivalent_annual_cost(10000, 500, 1000, 0.10, 5)
    assert eac == pytest.approx(10000 * 0.263797 + 500 - 1000 * 0.163797, rel=1e-5)
    # present worth of the EAC stream equals present worth of owning
    pw_own = 10000 + 500 * E.p_given_a(0.10, 5) - 1000 * E.p_given_f(0.10, 5)
    assert eac * E.p_given_a(0.10, 5) == pytest.approx(pw_own)
    assert E.breakeven_usage(eac, 100.0) == pytest.approx(eac / 100)
    assert E.breakeven_usage(1000, 50, own_cost_per_unit=10) == pytest.approx(25)


def test_depreciation():
    sl = E.straight_line(10000, 1000, 5)
    assert sl[-1][2] == pytest.approx(1000) and sum(d for _, d, _ in sl) == pytest.approx(9000)
    ddb = E.declining_balance(10000, 5)
    assert ddb[0][1] == pytest.approx(4000) and ddb[1][1] == pytest.approx(2400)
    assert all(b >= 0 for _, _, b in ddb)
    assert E.declining_balance(10000, 5, salvage=2000)[-1][2] == pytest.approx(2000)


def test_payback():
    assert E.payback_period([-1000, 400, 400, 400]) == pytest.approx(2.5)
    assert E.payback_period([-1000, 100]) == math.inf


@pytest.mark.parametrize("call", [
    lambda: E.p_given_f(-1.5, 3),
    lambda: E.p_given_a(0.1, 2.5),
    lambda: E.a_given_p(0.1, 0),
    lambda: E.irr([100, 100]),
    lambda: E.equivalent_annual_cost(-1, 0, 0, 0.1, 5),
    lambda: E.straight_line(100, 200, 5),
    lambda: E.breakeven_usage(1000, 5, own_cost_per_unit=10),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
