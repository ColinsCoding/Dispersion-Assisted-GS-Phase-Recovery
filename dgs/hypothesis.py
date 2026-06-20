"""Hypothesis testing -- is a result real, or could it be luck?

Feature engineering gave a jump (AUC 0.65 -> 0.79); statistics asks whether that
jump is significant. These are the nonparametric workhorses -- no distribution
assumptions, just resampling -- plus the classic two-proportion z-test:

  * permutation_test  -- shuffle the labels; how often does chance beat the result?
  * bootstrap_ci      -- resample with replacement; a confidence interval for a stat
  * proportion_z_test -- do two success rates (e.g. two models' accuracies) differ?
  * auc_permutation_p -- is a classifier's AUC significantly above chance (0.5)?

NumPy + math only. Civilian statistics / education.
"""

import math

import numpy as np


def _normal_cdf(z):
    """Standard-normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _auc(scores, labels):
    """ROC AUC (Mann-Whitney form): P(a random positive scores above a random negative)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def permutation_test(a, b, n_perm=10000, seed=0):
    """Two-sided permutation test for a difference in means between samples a, b.

    Pool the data, reshuffle into two groups n_perm times, and see how often the
    shuffled mean-difference is at least as extreme as observed. Returns
    (observed_diff, p_value). No normality assumption."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= abs(obs) - 1e-12:
            count += 1
    return float(obs), (count + 1) / (n_perm + 1)          # +1: never report p=0


def bootstrap_ci(data, statistic=np.mean, n_boot=10000, ci=0.95, seed=0):
    """Bootstrap confidence interval for any statistic of `data` (default the mean).
    Resample with replacement n_boot times; return (point_estimate, lo, hi)."""
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([statistic(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(statistic(data)), float(lo), float(hi)


def proportion_z_test(k1, n1, k2, n2):
    """Two-proportion z-test: do success rates k1/n1 and k2/n2 differ? (e.g. two
    classifiers' accuracies). Returns (p1-p2, z, two_sided_p). Pooled-variance form."""
    if not (0 <= k1 <= n1 and 0 <= k2 <= n2) or n1 == 0 or n2 == 0:
        raise ValueError("need 0 <= k <= n and n > 0")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, 0.0, 1.0
    z = (p1 - p2) / se
    return p1 - p2, z, 2 * (1 - _normal_cdf(abs(z)))


def auc_permutation_p(scores, labels, n_perm=2000, seed=0):
    """Is the classifier's AUC significantly above chance? Permute the labels and
    count how often a random labelling reaches the observed AUC. Returns (auc, p)."""
    labels = np.asarray(labels)
    obs = _auc(scores, labels)
    rng = np.random.default_rng(seed)
    perm = labels.copy()
    count = sum(1 for _ in range(n_perm)
                if (rng.shuffle(perm) or True) and _auc(scores, perm) >= obs - 1e-12)
    return obs, (count + 1) / (n_perm + 1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 1, 60), rng.normal(0.3, 1, 60)
    obs, p = permutation_test(a, b)
    print(f"permutation test: mean diff={obs:.3f}, p={p:.4f}")
    print(f"bootstrap 95% CI for mean(a): {bootstrap_ci(a)[1:]}")
    print(f"accuracy 79/100 vs 65/100:    p={proportion_z_test(79,100,65,100)[2]:.4f}")
