"""Supervised learning from scratch -- regression and classification.

The labelled-data half of the repo's ML: fit a model to (X, y) pairs and predict
on new X. Two foundations, both NumPy:

  * linear regression  -- least squares (normal equations); the workhorse, and
    the tool that recovers a physical scaling law from data (e.g. the receiver's
    shot-noise RMS ~ N^{-1/2}: a log-log fit returns the exponent -0.5).
  * logistic regression -- gradient descent on the sigmoid; binary classification
    (the linear cousin of the RogueGuard rare-event detector).

Plus the plumbing: standardize, train/test split, and the metrics (R^2, MSE,
accuracy). Civilian ML / education.
"""

import numpy as np


def standardize(X, mean=None, std=None):
    """Zero-mean, unit-variance columns. Returns (Xs, mean, std) to reuse on test data."""
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def train_test_split(X, y, test_frac=0.25, seed=0):
    """Shuffle and split into train/test."""
    X, y = np.asarray(X), np.asarray(y)
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


# ── linear regression (least squares) ───────────────────────────────
def _as_design(X):
    """Coerce X to a 2-D (n_samples, n_features) array (1-D -> column)."""
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def linear_regression(X, y):
    """Fit y = X w + b by least squares. Returns weights (last entry is bias)."""
    X = _as_design(X)
    A = np.hstack([X, np.ones((len(X), 1))])         # augment with bias column
    w, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
    return w


def predict_linear(X, w):
    X = _as_design(X)
    return np.hstack([X, np.ones((len(X), 1))]) @ w


# ── logistic regression (gradient descent) ──────────────────────────
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression(X, y, lr=0.1, n_iter=2000):
    """Binary classifier via gradient descent on the cross-entropy. Returns weights
    (last entry bias). Expects standardized X and labels in {0, 1}."""
    X = _as_design(X)
    y = np.asarray(y, dtype=float)
    A = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(A.shape[1])
    for _ in range(n_iter):
        grad = A.T @ (_sigmoid(A @ w) - y) / len(y)
        w -= lr * grad
    return w


def predict_proba(X, w):
    X = _as_design(X)
    return _sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w)


def predict(X, w, thresh=0.5):
    return (predict_proba(X, w) >= thresh).astype(int)


# ── metrics ─────────────────────────────────────────────────────────
def mse(y, yhat):
    return float(np.mean((np.asarray(y) - np.asarray(yhat))**2))


def r2(y, yhat):
    y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - np.asarray(yhat))**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def accuracy(y, yhat):
    return float(np.mean(np.asarray(y) == np.asarray(yhat)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # recover the shot-noise exponent from synthetic low-light data
    N = np.logspace(1, 6, 40)
    rms = 2.0 * N**-0.5 * (1 + 0.05 * rng.standard_normal(N.size))
    w = linear_regression(np.log10(N), np.log10(rms))
    print(f"shot-noise law from data: RMS ~ N^{w[0]:.3f}  (true exponent -0.5)")
