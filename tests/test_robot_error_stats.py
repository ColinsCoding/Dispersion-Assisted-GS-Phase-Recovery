"""Robot-arm error stats: analytic Jacobian vs finite diff, pose cov vs Monte
Carlo, Stirling vs exact log(N!), rank tests vs brute-force permutation p."""
import sys, pathlib, math, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import robot_error_stats as r

links = [1.0, 0.8, 0.5]
thetas = [math.radians(30), math.radians(45), math.radians(-20)]

# 1. FORWARD KINEMATICS: 2-link closed form by hand
x, y = r.planar_arm_fk([math.radians(30), math.radians(45)], [1.0, 0.8])
hx = math.cos(math.radians(30)) + 0.8 * math.cos(math.radians(75))
hy = math.sin(math.radians(30)) + 0.8 * math.sin(math.radians(75))
assert abs(x - hx) < 1e-12 and abs(y - hy) < 1e-12

# 2. JACOBIAN: analytic vs central-difference numerical Jacobian
J = r.arm_jacobian(thetas, links)
h = 1e-7
Jn = np.zeros_like(J)
for k in range(len(thetas)):
    tp, tm = list(thetas), list(thetas)
    tp[k] += h; tm[k] -= h
    xp, yp = r.planar_arm_fk(tp, links)
    xm, ym = r.planar_arm_fk(tm, links)
    Jn[0, k] = (xp - xm) / (2 * h)
    Jn[1, k] = (yp - ym) / (2 * h)
assert np.allclose(J, Jn, atol=1e-6), (J, Jn)

# 3. POSE COVARIANCE: Sigma = J St J^T vs Monte Carlo through the true FK
js = [math.radians(0.4), math.radians(0.4), math.radians(0.4)]
Sp = r.pose_covariance(thetas, links, js)
rng = np.random.default_rng(1)
samples = rng.normal(thetas, js, size=(200_000, len(thetas)))
tips = np.array([r.planar_arm_fk(s, links) for s in samples[:20_000]])  # subset for speed
Smc = np.cov(tips.T)
assert np.allclose(Sp, Smc, rtol=0.06, atol=1e-8), (Sp, Smc)
# error ellipse: RMS radius = sqrt(trace)
maj, mnr, ang, rms = r.error_ellipse(Sp)
assert abs(rms - math.sqrt(np.trace(Sp))) < 1e-15
assert maj >= mnr >= 0

# 4. sqrt(N): std of the mean of N draws ~ sigma/sqrt(N)
sig = 0.5
emp = np.std([rng.normal(0, sig, 50).mean() for _ in range(4000)])
assert abs(emp - r.sqrt_n_error(sig, 50)) / r.sqrt_n_error(sig, 50) < 0.05
assert r.reaches_needed(1.0, 0.1) == 100        # (1/0.1)^2

# 5. log(N!): Stirling series vs exact math.lgamma
for nn in (2, 5, 10, 50, 200):
    exact = math.lgamma(nn + 1)
    assert abs(r.log_factorial(nn) - exact) < 1e-3 / nn + 1e-6, nn
assert r.log_factorial(0) == 0.0 and r.log_factorial(1) == 0.0
# more terms -> not worse (monotone-ish improvement at n=5)
e5 = math.lgamma(6)
assert abs(r.log_factorial(5, terms=4) - e5) <= abs(r.log_factorial(5, terms=1) - e5)

# 6. MANN-WHITNEY U: normal-approx p vs exact permutation p (small samples)
a = [1.0, 3.0, 5.0, 7.0]
b = [2.0, 4.0, 6.0, 8.0, 10.0]
U, z, p = r.mann_whitney_u(a, b)
# brute force: exact null over all ways to choose which pooled ranks are group a
pooled = np.concatenate([a, b]); na = len(a)
ranks = r._ranks(pooled)
Uobs = min(sum(c) - na * (na + 1) / 2
           for c in [ranks[:na]])  # observed via ranks of a
# enumerate all C(n,na) label assignments, compute U for each
allU = []
n = len(pooled)
for combo in itertools.combinations(range(n), na):
    Ra = ranks[list(combo)].sum()
    Ua = Ra - na * (na + 1) / 2
    allU.append(min(Ua, na * len(b) - Ua))
allU = np.array(allU)
exact_p = np.mean(allU <= U + 1e-9)             # one-sided lower-tail proportion
# two-sided normal p should be in the right ballpark of 2x the one-sided exact
assert 0.0 < p <= 1.0
assert abs(p - min(1.0, 2 * exact_p)) < 0.25, (p, exact_p)
# a clean separation is highly significant
_, _, psep = r.mann_whitney_u([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
assert psep < 0.02

# 7. WILCOXON signed-rank: normal-approx vs exact sign-enumeration
d = np.array([0.5, -0.2, 0.8, 1.1, -0.1, 0.9, 0.3, 0.7])
W, zc, pw = r.wilcoxon_signed_rank(d)
rk = r._ranks(np.abs(d))
# exact null: every sign pattern equally likely -> enumerate 2^n W+ values
Wnull = [sum(rk[i] for i in range(len(d)) if bits & (1 << i))
         for bits in range(1 << len(d))]
Wnull = np.array(Wnull)
exact_pw = np.mean(Wnull >= W) + np.mean(Wnull <= (len(d) * (len(d) + 1) / 2 - W))
assert abs(pw - min(1.0, exact_pw)) < 0.15, (pw, exact_pw)

# 8. decision logic
assert r.decide(0.01)["reject"] is True
assert r.decide(0.20)["reject"] is False
assert r.decide(0.05, alpha=0.05)["reject"] is True   # boundary: p <= alpha

print(f"TEST PASS  (FK matches hand calc; analytic J == finite-diff J; "
      f"pose cov J St J^T == Monte Carlo; RMS radial {rms*1e3:.3f} mm; "
      f"sqrt(N) mean scatter matches; Stirling==lgamma to 1e-3; "
      f"MW U={U:.0f} p={p:.3f}~2x exact; Wilcoxon W+={W:.0f} p={pw:.3f})")
