"""Tests for dgs/decision_tree.py"""
import numpy as np
from dgs.decision_tree import (entropy, gini, impurity, info_gain, best_split,
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


# ── Gini impurity (the CART default alternative to entropy) ────────────────
def test_gini_pure():
    assert gini(np.array([1,1,1,1])) == 0.0

def test_gini_balanced():
    assert abs(gini(np.array([0,1])) - 0.5) < 1e-12        # 1 - 2*(1/2)^2

def test_gini_four_class():
    assert abs(gini(np.array([0,1,2,3])) - 0.75) < 1e-12   # 1 - 4*(1/4)^2

def test_impurity_dispatch():
    y = np.array([0,0,1,1])
    assert impurity(y, "gini") == gini(y)
    assert impurity(y, "entropy") == entropy(y)

def test_info_gain_gini_perfect():
    # perfect split: parent gini 0.5 -> children pure -> gain 0.5
    y, y_l, y_r = np.array([0,0,1,1]), np.array([0,0]), np.array([1,1])
    assert abs(info_gain(y, y_l, y_r, criterion="gini") - 0.5) < 1e-12

def test_tree_gini_learns_xor():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    y = np.array([0,1,1,0])
    tree = DecisionTree(max_depth=None, min_samples_split=1, criterion="gini")
    tree.fit(X, y)
    assert accuracy(y, tree.predict(X)) == 1.0

def test_gini_and_entropy_agree_on_clean_data():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((120, 2))
    y = (X[:,0] + X[:,1] > 0).astype(int)
    tg = DecisionTree(min_samples_split=1, criterion="gini").fit(X, y)
    te = DecisionTree(min_samples_split=1, criterion="entropy").fit(X, y)
    assert accuracy(y, tg.predict(X)) == 1.0
    assert accuracy(y, te.predict(X)) == 1.0


# ── out-of-bag error: the random forest's free validation ─────────────────
def test_oob_score_set_and_reasonable():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((300, 4))
    y = (X[:,0]**2 + X[:,1]**2 < 1.5).astype(int)
    rf = RandomForest(n_estimators=60, max_depth=6, random_state=1, oob_score=True)
    rf.fit(X, y)
    assert rf.oob_score_ is not None
    assert 0.75 < rf.oob_score_ <= 1.0, f"oob_score {rf.oob_score_}"
    # each bootstrap leaves out roughly 1/e ~ 37% of the samples
    frac_oob = np.mean([len(o) for o in rf.oob_indices_]) / len(y)
    assert 0.30 < frac_oob < 0.42, f"oob fraction {frac_oob:.3f}"

def test_oob_tracks_held_out_accuracy():
    rng = np.random.default_rng(21)
    X = rng.standard_normal((260, 3))
    y = (X[:,0] > X[:,1]).astype(int)
    Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]
    rf = RandomForest(n_estimators=80, max_depth=6, random_state=2, oob_score=True)
    rf.fit(Xtr, ytr)
    test_acc = accuracy(yte, rf.predict(Xte))
    # OOB score is an honest estimate of generalisation: close to true test accuracy
    assert abs(rf.oob_score_ - test_acc) < 0.12, f"oob {rf.oob_score_} vs test {test_acc}"

def test_oob_default_off():
    rf = RandomForest(n_estimators=5, random_state=0).fit(
        np.random.default_rng(0).standard_normal((40, 2)),
        np.random.default_rng(1).integers(0, 2, 40))
    assert rf.oob_score_ is None


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
    test_random_forest_accuracy()
    test_random_forest_feature_importance()
    test_gini_pure()
    test_gini_balanced()
    test_gini_four_class()
    test_impurity_dispatch()
    test_info_gain_gini_perfect()
    test_tree_gini_learns_xor()
    test_gini_and_entropy_agree_on_clean_data()
    test_oob_score_set_and_reasonable()
    test_oob_tracks_held_out_accuracy()
    test_oob_default_off()
    print("all tests passed")
