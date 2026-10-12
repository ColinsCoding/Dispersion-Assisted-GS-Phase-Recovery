"""Tests for dgs/optics_decision_tree.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.decision_tree import accuracy
from dgs.optics_decision_tree import (
    V_CUTOFF, U_CUTOFF, v_number, single_mode, fiber_dataset, engineered_feature, fit_tree,
    count_nodes_recursive, count_nodes_iterative, depth_recursive, tree_to_piecewise, tree_to_logic,
    leaf_paths, evaluate_piecewise, full_tree_bitstrings, hadamard_uniform_state,
)
from scipy.special import jn_zeros


def test_cutoff_is_bessel_zero():
    assert V_CUTOFF == pytest.approx(jn_zeros(0, 1)[0], abs=1e-12)
    assert single_mode(4.1, 0.12, 1.55) and not single_mode(25.0, 0.2, 1.55)   # SMF-28-like vs multimode
    with pytest.raises(ValueError):
        v_number(0, 0.1, 1.0)


@pytest.fixture(scope="module")
def data():
    return fiber_dataset(1500, seed=0), fiber_dataset(1500, seed=1)


def test_engineered_feature_one_split(data):
    (X, y), (Xt, yt) = data
    t = fit_tree(engineered_feature(X), y, max_depth=3)
    assert count_nodes_recursive(t.root) == 3
    assert t.root.thresh == pytest.approx(U_CUTOFF, rel=0.01)
    assert accuracy(yt, t.predict(engineered_feature(Xt))) > 0.99


def test_raw_features_need_many_nodes(data):
    (X, y), (Xt, yt) = data
    t = fit_tree(X, y)
    assert count_nodes_recursive(t.root) > 30
    assert accuracy(yt, t.predict(Xt)) < 0.99


def test_recursion_equals_loop_and_depth(data):
    (X, y), _ = data
    t = fit_tree(X, y, max_depth=5)
    assert count_nodes_recursive(t.root) == count_nodes_iterative(t.root)
    assert depth_recursive(t.root) <= 5
    paths = leaf_paths(t.root)
    assert len(paths) == (count_nodes_recursive(t.root) + 1) // 2        # full binary tree: leaves = (n+1)/2
    assert all(len(p[0]) == len(p[1]) for p in paths)


def test_piecewise_and_logic_reproduce_tree(data):
    (X, y), (Xt, yt) = data
    syms = sp.symbols("a NA lam", positive=True)
    t = fit_tree(X, y, max_depth=4)
    pw = tree_to_piecewise(t.root, syms)
    assert np.array_equal(evaluate_piecewise(pw, syms, Xt), t.predict(Xt))
    logic = tree_to_logic(t.root, syms)
    f = sp.lambdify(syms, logic, "numpy")
    assert np.array_equal(np.asarray(f(*Xt.T), dtype=int), t.predict(Xt))


def test_bitstrings_and_qubits():
    b = full_tree_bitstrings(10)
    assert len(b) == 1024 and len(set(b)) == 1024 and b[0] == "0" * 10
    psi = hadamard_uniform_state(10)
    assert psi.size == 1024 and np.allclose(psi, 2 ** -5) and np.sum(psi ** 2) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        full_tree_bitstrings(-1)
