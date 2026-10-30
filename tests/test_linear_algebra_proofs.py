"""Tests for dgs/linear_algebra_proofs.py: symbolic identities, projections, market (SymPy + C), attention."""
import pytest
import sympy as sp

from dgs import linear_algebra_proofs as L


@pytest.mark.parametrize("n", [2, 3])
def test_identities_hold(n):
    assert all(L.prove_identities(n).values())


def test_bad_size_rejected():
    with pytest.raises(ValueError):
        L.symbolic_matrix("a", 0)
    with pytest.raises(ValueError):
        L.prove_identities(9)


@pytest.mark.parametrize("n", [2, 3])
def test_cauchy_schwarz_is_sum_of_squares(n):
    gap, sos = L.cauchy_schwarz(n)
    assert sp.expand(gap - sos) == 0


def test_projection_and_kernel():
    P, props = L.projection([1, 2, 2])
    assert all(props.values()) and P.trace() == 1                     # rank 1
    with pytest.raises(ValueError):
        L.projection([0, 0])
    basis, ok = L.kernel_midpoint([[1, 1, 1], [1, -1, 0]])
    assert len(basis) == 1 and ok


def test_market_symbolic_and_numbers():
    a, b, c, d = sp.symbols("a b c d", positive=True)
    m = L.market(a, b, c, d)
    assert sp.simplify(m["P"] - (a - c) / (b + d)) == 0 and sp.simplify(m["det"] + b + d) == 0
    m = L.market(**L.PHOTONICS_MARKET)
    assert m["P"] == pytest.approx(100) and m["Q"] == pytest.approx(80)
    s = L.market_shift(**L.PHOTONICS_MARKET, demand_shift=10)
    assert s["dP"] == pytest.approx(10) and s["dQ"] > 0              # dP = delta/(b+d) = 10/1
    with pytest.raises(ValueError):
        L.market(10, -1, 0, 1)
    with pytest.raises(ValueError):
        L.market(10, 1, 50, 1)                                        # supply starts above demand


def test_market_c_matches_sympy():
    try:
        r = L.market_c(**L.PHOTONICS_MARKET, demand_shift=10, supply_shift=-5)
    except RuntimeError as e:
        pytest.skip(str(e))
    s = L.market_shift(**L.PHOTONICS_MARKET, demand_shift=10, supply_shift=-5)
    assert r["P2"] == pytest.approx(float(s["after"][0])) and r["Q2"] == pytest.approx(float(s["after"][1]))
    with pytest.raises(ValueError):
        L.market_c(10, 0, 0, 1)


def test_softmax_symbolic():
    w, total, shift = L.softmax_symbolic(3)
    assert total == 1 and all(s == 0 for s in shift)


def test_attention_matches_torch():
    pytest.importorskip("torch")
    r = L.check_attention()
    assert r["max |ours - torch|"] < 1e-12 and r["causal max |ours - torch|"] < 1e-12
    assert all(v for k, v in r.items() if not k.startswith(("max", "causal max")))
