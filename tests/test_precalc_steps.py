"""Tests for dgs/precalc_steps.py: correct derivations pass, classic precalc mistakes are rejected."""
import math

import pytest
import sympy as sp

from dgs import precalc_steps as P
from dgs.precalc_steps import Steps, StepError, a, b, x, t, lam, N0, L, z, s, th


@pytest.mark.parametrize("fn, expected", [
    (P.example_complete_square, sp.sqrt(sp.pi / a) * sp.exp(b**2 / (4 * a))),
    (P.example_box_mean_x, L / 2),
    (P.example_stefan_boltzmann, sp.pi**4 / 15),
    (P.example_half_life, sp.log(2) / lam),
    (P.example_trig_sub, 1 / (z**2 + s**2) ** sp.Rational(3, 2)),
])
def test_examples_reach_known_results(fn, expected):
    S = fn()
    assert sp.simplify(S.result - expected) == 0
    assert len(S.steps) >= 3


def test_all_examples_build():
    for fn in P.EXAMPLES:
        assert fn().result is not None


# --- classic mistakes must be caught -------------------------------------------------------
def test_wrong_completed_square_rejected():
    S = Steps("t").start(-a * x**2 + b * x)
    with pytest.raises(StepError):
        S.eq("wrong constant", -a * (x - b / (2 * a))**2 + b**2 / (2 * a))      # should be b^2/(4a)


def test_wrong_power_reduction_rejected():
    S = Steps("t").start(sp.sin(x)**2)
    with pytest.raises(StepError):
        S.eq("sign error", (1 + sp.cos(2 * x)) / 2)                          # that is cos^2
    S.eq("correct", (1 - sp.cos(2 * x)) / 2)


def test_wrong_log_rule_rejected():
    S = Steps("t", unknown=t).start(sp.Eq(sp.exp(-lam * t), sp.Rational(1, 2)))
    with pytest.raises(StepError):
        S.eq("ln(1/2) = +ln 2 (wrong sign)", sp.Eq(-lam * t, sp.log(2)))
    S.eq("correct", sp.Eq(-lam * t, -sp.log(2)))


def test_distributing_a_square_rejected():
    S = Steps("t").start((x + b)**2)
    with pytest.raises(StepError):
        S.eq("freshman's dream", x**2 + b**2)


def test_equivalent_equations_need_unknown():
    S = Steps("t").start(sp.Eq(N0 * sp.exp(-lam * t), N0 / 2))
    with pytest.raises(StepError):
        S.eq("divide", sp.Eq(sp.exp(-lam * t), sp.Rational(1, 2)))


def test_op_needs_a_check():
    S = Steps("t").start(x)
    with pytest.raises(ValueError):
        S.op("integrate", x**2 / 2, "no check given")
    with pytest.raises(StepError):
        S.op("integrate", x**2, "wrong antiderivative", check=lambda e: sp.integrate(e, x))


def test_trig_sub_states_its_domain():
    S = P.example_trig_sub()
    assert "cos theta > 0" in S.steps[1].why


def test_table():
    tab = P.example_half_life().table()
    assert list(tab.columns) == ["step", "skill", "expression", "why", "verified"]
    assert len(tab) == 5


# --- independent numerical check with torch autograd ---------------------------------------
def test_torch_autograd_checks_antiderivatives():
    torch = pytest.importorskip("torch")
    zs = torch.linspace(-3, 3, 41, dtype=torch.float64, requires_grad=True)
    s0 = 0.7
    F = zs / (s0**2 * torch.sqrt(zs**2 + s0**2))            # trig-substitution result
    (dF,) = torch.autograd.grad(F.sum(), zs)
    assert torch.allclose(dF, 1 / (zs.detach()**2 + s0**2) ** 1.5, atol=1e-12)
    k = 2.3
    G = zs * torch.sin(k * zs) / k + torch.cos(k * zs) / k**2   # by-parts result
    (dG,) = torch.autograd.grad(G.sum(), zs)
    assert torch.allclose(dG, zs.detach() * torch.cos(k * zs.detach()), atol=1e-12)
