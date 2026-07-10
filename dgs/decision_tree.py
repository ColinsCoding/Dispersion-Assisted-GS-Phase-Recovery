"""
Decision tree and random forest — built from scratch with numpy only.

Key math:  entropy  H(S) = -sum_k p_k * log2(p_k)      (impurity, in bits)
           gini     G(S) = 1 - sum_k p_k^2             (impurity, the CART default)
           info gain  IG = I(parent) - sum_child (|child|/|parent|) * I(child)

The tree recursively finds the split (feature, threshold) that maximises IG for the
chosen impurity I (entropy or gini) — the "folding" of 2-D space into axis-aligned
half-planes. A random forest bags many such trees on bootstrap samples and, because
each bootstrap leaves out ~1/e ~ 37% of the data, it gets a free validation score on
those OUT-OF-BAG samples (oob_score_) with no held-out set needed.
"""
import numpy as np


# ── impurity measures and information gain ────────────────────────────────
def entropy(y):
    """Shannon entropy in bits: H = -sum p_k log2(p_k). Zero for a pure node, 1 bit
    for a balanced two-class node."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    # log2(0) = -inf; filter with p>0 (always true here since counts>=1)
    return float(-np.sum(p * np.log2(p + 1e-300)))


def gini(y):
    """Gini impurity G = 1 - sum p_k^2 (the CART default). Zero for a pure node, 0.5
    for a balanced two-class node; cheaper than entropy (no logs)."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def impurity(y, criterion="entropy"):
    """Dispatch to the requested impurity measure."""
    return gini(y) if criterion == "gini" else entropy(y)


def info_gain(y_parent, y_left, y_right, criterion="entropy"):
    n = len(y_parent)
    if n == 0:
        return 0.0
    return (impurity(y_parent, criterion)
            - len(y_left)  / n * impurity(y_left,  criterion)
            - len(y_right) / n * impurity(y_right, criterion))


# ── single split search ───────────────────────────────────────────────────
def best_split(X, y, feature_indices=None, n_thresholds=None, rng=None,
               criterion="entropy"):
    """
    Find (feature_idx, threshold) that maximises information gain.

    Parameters
    ----------
    feature_indices : array-like or None — subset of features to consider
    n_thresholds    : int or None — subsample candidate thresholds (speed)
    rng             : np.random.Generator or None
    criterion       : 'entropy' | 'gini' — impurity measure to maximise gain of

    Returns
    -------
    best_feat, best_thresh, best_ig, split_history
    split_history : list of (feat, thresh, ig) for all candidates examined
    """
    n_samples, n_features = X.shape
    if feature_indices is None:
        feature_indices = np.arange(n_features)

    best_feat, best_thresh, best_ig = None, None, -np.inf
    history = []

    for feat in feature_indices:
        vals = np.sort(np.unique(X[:, feat]))
        if len(vals) <= 1:
            continue
        # midpoints as candidate thresholds
        thresholds = (vals[:-1] + vals[1:]) / 2
        if n_thresholds is not None and len(thresholds) > n_thresholds:
            rng = rng or np.random.default_rng()
            thresholds = rng.choice(thresholds, n_thresholds, replace=False)

        for thresh in thresholds:
            mask = X[:, feat] <= thresh
            y_l, y_r = y[mask], y[~mask]
            if len(y_l) == 0 or len(y_r) == 0:
                continue
            ig = info_gain(y, y_l, y_r, criterion)
            history.append((int(feat), float(thresh), float(ig)))
            if ig > best_ig:
                best_ig = ig
                best_feat = int(feat)
                best_thresh = float(thresh)

    return best_feat, best_thresh, best_ig, history


# ── tree node ─────────────────────────────────────────────────────────────
class Node:
    __slots__ = ('feat', 'thresh', 'left', 'right', 'value',
                 'depth', 'n_samples', 'entropy')

    def __init__(self, feat=None, thresh=None, left=None, right=None,
                 value=None, depth=0, n_samples=0, entropy=0.0):
        self.feat      = feat
        self.thresh    = thresh
        self.left      = left
        self.right     = right
        self.value     = value        # majority class at this node (leaf)
        self.depth     = depth
        self.n_samples = n_samples
        self.entropy   = entropy

    @property
    def is_leaf(self):
        return self.value is not None


# ── decision tree ─────────────────────────────────────────────────────────
class DecisionTree:
    """
    Axis-aligned binary decision tree using entropy / information gain.

    Parameters
    ----------
    max_depth       : int — maximum tree depth (None = grow fully)
    min_samples_split : int — don't split a node with fewer samples
    max_features    : int, float ('sqrt','log2'), or None
    n_thresholds    : int or None — limit threshold candidates per feature
    """
    def __init__(self, max_depth=None, min_samples_split=2,
                 max_features=None, n_thresholds=None, random_state=None,
                 criterion="entropy"):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.n_thresholds      = n_thresholds
        self.criterion         = criterion
        self.rng               = np.random.default_rng(random_state)
        self.root              = None
        self.classes_          = None
        self.split_log         = []   # ordered list of splits (for animation)

    def _n_features_to_try(self, n_features):
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if mf == 'log2':
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        return int(mf)

    def _build(self, X, y, depth):
        n_samples = len(y)
        node_entropy = impurity(y, self.criterion)
        classes, counts = np.unique(y, return_counts=True)
        majority = classes[np.argmax(counts)]

        node = Node(depth=depth, n_samples=n_samples, entropy=node_entropy,
                    value=majority)

        # stopping conditions
        pure      = len(classes) == 1
        too_small = n_samples < self.min_samples_split
        too_deep  = self.max_depth is not None and depth >= self.max_depth

        if pure or too_small or too_deep:
            return node

        # feature subsampling
        n_try = self._n_features_to_try(X.shape[1])
        feat_idx = self.rng.choice(X.shape[1], n_try, replace=False)

        feat, thresh, ig, _ = best_split(X, y, feat_idx,
                                          self.n_thresholds, self.rng,
                                          self.criterion)
        if feat is None:
            return node   # no valid split exists (all features constant)

        mask = X[:, feat] <= thresh
        node.feat   = feat
        node.thresh = thresh
        node.value  = None   # internal node

        self.split_log.append({'feat': feat, 'thresh': thresh,
                                'ig': ig, 'depth': depth,
                                'n_samples': n_samples})

        node.left  = self._build(X[mask],  y[mask],  depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.split_log = []
        self.root = self._build(np.asarray(X, float), np.asarray(y), 0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feat] <= node.thresh:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X):
        # leaf value is majority class — return hard 0/1 proba
        preds = self.predict(X)
        n_cls = len(self.classes_)
        P = np.zeros((len(X), n_cls))
        for i, p in enumerate(preds):
            P[i, list(self.classes_).index(p)] = 1.0
        return P

    def depth(self):
        def _d(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_d(node.left), _d(node.right))
        return _d(self.root)


# ── random forest ─────────────────────────────────────────────────────────
class RandomForest:
    """
    Ensemble of DecisionTrees with bootstrap sampling and feature subsampling.

    Parameters
    ----------
    n_estimators   : int
    max_depth      : int or None
    max_features   : 'sqrt' | 'log2' | float | int | None
    min_samples_split : int
    random_state   : int or None
    """
    def __init__(self, n_estimators=100, max_depth=None,
                 max_features='sqrt', min_samples_split=2,
                 random_state=None, criterion="entropy", oob_score=False):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.max_features      = max_features
        self.min_samples_split = min_samples_split
        self.criterion         = criterion
        self.oob_score         = oob_score
        self.rng               = np.random.default_rng(random_state)
        self.trees_            = []
        self.oob_indices_      = []   # out-of-bag sample indices per tree
        self.oob_score_        = None
        self.classes_          = None

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        n = len(y)
        self.trees_ = []
        self.oob_indices_ = []
        all_idx = np.arange(n)
        for i in range(self.n_estimators):
            seed = int(self.rng.integers(0, 2**31))
            idx  = self.rng.integers(0, n, size=n)   # bootstrap (with replacement)
            tree = DecisionTree(max_depth=self.max_depth,
                                max_features=self.max_features,
                                min_samples_split=self.min_samples_split,
                                n_thresholds=20,
                                random_state=seed,
                                criterion=self.criterion)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            self.oob_indices_.append(np.setdiff1d(all_idx, idx))   # ~37% left out
        if self.oob_score:
            self.oob_score_ = self._compute_oob_score(X, y)
        return self

    def _compute_oob_score(self, X, y):
        """Free validation: each sample is scored only by the trees that did NOT train on
        it (its out-of-bag trees), then compared to its true label."""
        n, n_cls = len(y), len(self.classes_)
        votes = np.zeros((n, n_cls))
        counts = np.zeros(n)
        cls_index = {c: i for i, c in enumerate(self.classes_)}
        for tree, oob in zip(self.trees_, self.oob_indices_):
            if len(oob) == 0:
                continue
            P = tree.predict_proba(X[oob])
            for ci, c in enumerate(tree.classes_):
                votes[oob, cls_index[c]] += P[:, ci]
            counts[oob] += 1
        scored = counts > 0
        pred = self.classes_[np.argmax(votes[scored], axis=1)]
        return float(np.mean(pred == y[scored]))

    def predict_proba(self, X):
        n_cls = len(self.classes_)
        proba = np.zeros((len(X), n_cls))
        for tree in self.trees_:
            P = tree.predict_proba(X)
            # align columns
            for ci, c in enumerate(tree.classes_):
                gi = list(self.classes_).index(c)
                proba[:, gi] += P[:, ci]
        return proba / self.n_estimators

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def feature_importances(self, X, y):
        """Mean decrease in impurity across all trees."""
        n_features = X.shape[1]
        imp = np.zeros(n_features)
        for tree in self.trees_:
            for split in tree.split_log:
                imp[split['feat']] += split['ig'] * split['n_samples']
        total = imp.sum()
        return imp / total if total > 0 else imp


# ── accuracy helper ───────────────────────────────────────────────────────
def accuracy(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))
