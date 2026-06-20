"""Test hypothesis-testing tools: permutation, bootstrap, proportion z, AUC permutation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import hypothesis as h

rng = np.random.default_rng(0)

# 1. permutation test: clearly different means -> small p; same distribution -> large p
a, b = rng.normal(1.0, 1.0, 80), rng.normal(0.0, 1.0, 80)
_, p_diff = h.permutation_test(a, b, n_perm=3000)
_, p_same = h.permutation_test(rng.normal(0, 1, 80), rng.normal(0, 1, 80), n_perm=3000)
assert p_diff < 0.01, p_diff
assert p_same > 0.05, p_same

# 2. bootstrap CI brackets the true mean and shrinks with more data
est, lo, hi = h.bootstrap_ci(rng.normal(5.0, 2.0, 400), n_boot=2000)
assert lo < 5.0 < hi and (hi - lo) < 0.7
_, lo_small, hi_small = h.bootstrap_ci(rng.normal(5.0, 2.0, 25), n_boot=2000)
assert (hi_small - lo_small) > (hi - lo)                 # less data -> wider interval

# 3. two-proportion z-test: 79/100 vs 65/100 differs; 50/100 vs 51/100 doesn't
_, _, p_big = h.proportion_z_test(79, 100, 65, 100)
_, _, p_null = h.proportion_z_test(50, 100, 51, 100)
assert p_big < 0.05 and p_null > 0.5
# identical proportions -> p = 1
assert abs(h.proportion_z_test(40, 100, 40, 100)[2] - 1.0) < 1e-9
try:
    h.proportion_z_test(5, 0, 1, 10)
except ValueError:
    pass
else:
    raise AssertionError("n=0 should raise")

# 4. AUC permutation: a near-perfect score is significant; random scores are not
labels = np.r_[np.zeros(60), np.ones(60)]
good = np.r_[rng.normal(0, 1, 60), rng.normal(3, 1, 60)]   # well separated
auc_g, p_g = h.auc_permutation_p(good, labels, n_perm=1000)
assert auc_g > 0.9 and p_g < 0.01
rand = rng.normal(0, 1, 120)
auc_r, p_r = h.auc_permutation_p(rand, labels, n_perm=1000)
assert p_r > 0.05

# 5. the repo result: is AUC 0.79 on a 45%-positive set significantly above chance?
#    (synthesize scores that yield ~0.79 AUC and confirm significance)
n = 500; y = (rng.random(n) < 0.45).astype(int)
s = y * 0.9 + rng.normal(0, 1.0, n)                       # signal + noise -> AUC ~0.7-0.8
auc, p = h.auc_permutation_p(s, y, n_perm=1000)
assert auc > 0.6 and p < 0.01                             # clearly real, not luck

print(f"TEST PASS  (perm p_diff={p_diff:.4f}; prop 79vs65 p={p_big:.4f}; "
      f"AUC {auc:.2f} significant p={p:.4f})")
