"""Smoke-test supervised learning: linear/logistic regression, metrics, the shot-noise fit."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import supervised as sl

rng = np.random.default_rng(0)

# 1. linear regression recovers known slope & intercept (y = 3x + 2)
x = np.linspace(0, 10, 200)
y = 3 * x + 2 + 0.1 * rng.standard_normal(x.size)
w = sl.linear_regression(x, y)
assert abs(w[0] - 3) < 0.05 and abs(w[1] - 2) < 0.1, w
assert sl.r2(y, sl.predict_linear(x, w)) > 0.99

# 2. THE PHYSICS: a log-log fit of RMS = c N^{-1/2} recovers the exponent -0.5
N = np.logspace(1, 6, 60)
rms = 2.0 * N**-0.5 * (1 + 0.05 * rng.standard_normal(N.size))
wln = sl.linear_regression(np.log10(N), np.log10(rms))
assert abs(wln[0] - (-0.5)) < 0.03, wln[0]              # supervised learning finds the shot-noise law

# 3. logistic regression separates two blobs with high accuracy
n = 300
c0 = rng.standard_normal((n, 2)) + [-2, -2]
c1 = rng.standard_normal((n, 2)) + [2, 2]
X = np.vstack([c0, c1]); yc = np.r_[np.zeros(n), np.ones(n)]
Xs, mean, std = sl.standardize(X)
Xtr, Xte, ytr, yte = sl.train_test_split(Xs, yc, test_frac=0.3, seed=1)
wlog = sl.logistic_regression(Xtr, ytr, lr=0.2, n_iter=3000)
assert sl.accuracy(yte, sl.predict(Xte, wlog)) > 0.95
# probabilities are in [0,1]
p = sl.predict_proba(Xte, wlog)
assert p.min() >= 0 and p.max() <= 1

# 4. standardize gives mean ~0, std ~1; reused stats transform test data consistently
assert np.allclose(Xs.mean(axis=0), 0, atol=1e-9) and np.allclose(Xs.std(axis=0), 1, atol=1e-9)

# 5. train/test split sizes and disjointness
Xtr2, Xte2, ytr2, yte2 = sl.train_test_split(X, yc, test_frac=0.25, seed=2)
assert len(Xtr2) + len(Xte2) == len(X)
assert abs(len(Xte2) / len(X) - 0.25) < 0.01

# 6. metrics sanity
assert sl.mse([1, 2, 3], [1, 2, 3]) == 0
assert abs(sl.r2([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
assert sl.accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75

print(f"SMOKE PASS  (linear R2>0.99; shot-noise exponent from data = {wln[0]:.3f} ~ -0.5; "
      f"logistic test acc = {sl.accuracy(yte, sl.predict(Xte, wlog)):.2f})")
