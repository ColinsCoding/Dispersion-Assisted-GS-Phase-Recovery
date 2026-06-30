"""Tests for dgs/decision_tree.py"""
import numpy as np
from dgs.decision_tree import (entropy, info_gain, best_split,
                                DecisionTree, RandomForest, accuracy)


def test_entropy_pure():
    assert entropy(np.array([0,0,0])) == 0.0

def test_entropy_balanced():
    H = entropy(np.array([0,1]))
    assert abs(H - 1.0) < 1e-9   # 1 bit

def test_entropy_four_class():
    H = entropy(np.array([0,1,2,3]))
    assert abs(H - 2.0) < 1e-9   # 2 bits

def test_info_gain_perfect():
    # split that perfectly separates classes
    y = np.array([0,0,1,1])
    y_l = np.array([0,0])
    y_r = np.array([1,1])
    assert abs(info_gain(y, y_l, y_r) - 1.0) < 1e-9

def test_info_gain_useless():
    y = np.array([0,1,0,1])
    y_l = np.array([0,1])
    y_r = np.array([0,1])
    assert abs(info_gain(y, y_l, y_r)) < 1e-9

def test_best_split_linearly_separable():
    X = np.array([[1.0],[2.0],[3.0],[10.0],[11.0],[12.0]])
    y = np.array([0,0,0,1,1,1])
    feat, thresh, ig, _ = best_split(X, y)
    assert feat == 0
    assert 3.0 < thresh < 10.0
    assert abs(ig - 1.0) < 1e-9

def test_tree_memorises_training():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 2))
    y = (X[:,0] + X[:,1] > 0).astype(int)
    tree = DecisionTree(max_depth=None, min_samples_split=1)
    tree.fit(X, y)
    assert accuracy(y, tree.predict(X)) == 1.0

def test_tree_max_depth():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 2))
    y = (X[:,0] > 0).astype(int)
    tree = DecisionTree(max_depth=1)
    tree.fit(X, y)
    assert tree.depth() <= 1

def test_tree_xor():
    # XOR is not linearly separable — needs depth ≥ 2
    X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    y = np.array([0,1,1,0])
    tree = DecisionTree(max_depth=None, min_samples_split=1)
    tree.fit(X, y)
    assert accuracy(y, tree.predict(X)) == 1.0

def test_random_forest_accuracy():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 4))
    y = (X[:,0]**2 + X[:,1]**2 < 1.5).astype(int)
    split = 160
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    rf = RandomForest(n_estimators=30, max_depth=5, random_state=7)
    rf.fit(X_tr, y_tr)
    acc = accuracy(y_te, rf.predict(X_te))
    assert acc > 0.80, f"RF accuracy {acc:.2f} < 0.80"

def test_random_forest_feature_importance():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((300, 4))
    # only feature 0 matters
    y = (X[:,0] > 0).astype(int)
    rf = RandomForest(n_estimators=20, max_depth=4, random_state=5)
    rf.fit(X, y)
    imp = rf.feature_importances(X, y)
    assert imp[0] == imp.max(), f"feature 0 not most important: {imp}"


if __name__ == "__main__":
    test_entropy_pure()
    test_entropy_balanced()
    test_entropy_four_class()
    test_info_gain_perfect()
    test_info_gain_useless()
    test_best_split_linearly_separable()
    test_tree_memorises_training()
    test_tree_max_depth()
    test_tree_xor()
    test_random_forest_feature_importance()
    print("all tests passed")
