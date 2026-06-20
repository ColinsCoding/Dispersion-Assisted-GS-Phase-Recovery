"""Smoke-test bayes_inference: discrete update, conjugate posteriors, detection."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import bayes_inference as bi

# 1. discrete Bayes normalizes; uniform prior -> posterior proportional to likelihood
post = bi.bayes_posterior([0.5, 0.5], [0.2, 0.8])
assert abs(post.sum() - 1) < 1e-12
assert np.allclose(post, [0.2, 0.8])
# impossible data is rejected
try:
    bi.bayes_posterior([1, 0], [0, 1])
except ValueError:
    pass
else:
    raise AssertionError("zero-evidence data should raise")

# 2. Poisson rate posterior -> true rate as data grows
rng = np.random.default_rng(1)
counts = rng.poisson(3.0, size=5000)
post = bi.poisson_rate_posterior(counts)
assert abs(post["mean"] - 3.0) < 0.1, post["mean"]
assert post["var"] > 0

# 2b. SUFFICIENT STATISTIC: full data and the (sum, n) summary give the SAME posterior
#     -- the "compressed engineered feature" loses nothing
import types
total, n = counts.sum(), counts.size
post_summary = bi.poisson_rate_posterior(np.concatenate([[total], np.zeros(n - 1)]))
assert abs(post["a"] - post_summary["a"]) < 1e-9    # a = a0 + sum k is identical
assert abs(post["b"] - post_summary["b"]) < 1e-9    # b = b0 + n is identical
assert abs(post["mean"] - post_summary["mean"]) < 1e-12

# 3. Normal-Normal: posterior mean lies between prior and sample mean, tighter than both
data = rng.normal(5.0, 2.0, size=50)
g = bi.gaussian_mean_posterior(data, mu0=0.0, sigma0=10.0, sigma=2.0)
assert 0.0 < g["mean"] < data.mean() + 1e-9 or data.mean() < g["mean"] < 0.0
assert g["std"] < 2.0 / np.sqrt(50) + 1e-9          # at least as tight as the data alone
# a near-flat prior -> posterior mean ~ sample mean
g_flat = bi.gaussian_mean_posterior(data, mu0=0.0, sigma0=1e6, sigma=2.0)
assert abs(g_flat["mean"] - data.mean()) < 1e-3

# 4. base-rate / rare-event detection (the RogueGuard lesson)
p = bi.detection_posterior(0.001, 0.99, 0.05, "alarm")
expected = (0.001 * 0.99) / (0.001 * 0.99 + 0.999 * 0.05)
assert abs(p - expected) < 1e-12
assert p < 0.02                                     # high sensitivity, still tiny posterior
# a clean (no-alarm) observation drives the posterior far below the base rate
p_clear = bi.detection_posterior(0.001, 0.99, 0.05, "no_alarm")
assert p_clear < 0.001
# validation
for bad in (-0.1, 1.5):
    try:
        bi.detection_posterior(bad, 0.9, 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range probability should raise")

print(f"SMOKE PASS  (Poisson mean={post['mean']:.3f}; P(rogue|alarm)={p:.4f}; "
      f"sufficient-stat posterior identical)")
