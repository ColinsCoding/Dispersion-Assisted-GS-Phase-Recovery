"""Where does a robot arm actually end up, and how sure are you? -- the four
pieces of that one question, cross-checked against each other.

A planar arm reaches for a target. Its joint encoders each carry a little
uncertainty; you want to know (1) how that uncertainty lands at the fingertip,
(2) how averaging repeated reaches beats it down, and (3) whether a new
calibration is *really* better than the old one or just lucky. Those are the
kernels this module ties together:

  * ROBOTICS ERROR PROPAGATION -- the manipulator Jacobian J = d(x,y)/d(theta)
    maps joint-angle covariance to a 2x2 end-effector covariance:
        Sigma_pos = J Sigma_theta J^T          (the "propagation of errors")
    Same first-order rule as dgs.error_propagation, but J here is the *geometric*
    Jacobian of a kinematic chain (closed form), and the output is a 2-D error
    ellipse, not a single sigma.

  * sqrt(N) -- average N independent reaches and the standard error of the mean
    pose shrinks as sigma/sqrt(N). Halving the error costs 4x the trials. This is
    the same random-walk law that sets shot-noise limits and Monte-Carlo error.

  * log(N!) via STIRLING -- log(N!) ~ N ln N - N + (1/2)ln(2 pi N). It is the log
    of the number of orderings of N things, i.e. the log-size of the sample space
    a RANK test lives in. That combinatorial size is exactly why, for moderate N,
    the discrete rank-statistic null distribution is well approximated by a
    Gaussian -- so p-values come from the normal CDF, no permutation enumeration.

  * RANK NULL-HYPOTHESIS TESTS -- Mann-Whitney U (rank-sum, two independent
    groups) and Wilcoxon signed-rank (paired). Distribution-free: they replace the
    data by their ranks, so one wild outlier reach can't swing the verdict. The
    "design logic" of the test is explicit in decide(): state H0, pick a
    statistic, get its null distribution, convert to a p-value, compare to alpha.

NumPy + math only (no scipy). Education.
"""

import math

import numpy as np


# ── robotics: forward kinematics, the manipulator Jacobian, pose covariance ──
def planar_arm_fk(thetas, links):
    """End-effector (x, y) of a planar chain of `len(links)` revolute joints.

    `thetas[k]` is the RELATIVE angle of link k w.r.t. the previous link; the
    absolute orientation of link k is the running sum phi_k = theta_0+...+theta_k.
        x = sum_i l_i cos(phi_i),   y = sum_i l_i sin(phi_i)
    Angles in radians, link lengths > 0."""
    thetas = np.asarray(thetas, float)
    links = np.asarray(links, float)
    if thetas.shape != links.shape:
        raise ValueError("thetas and links must have the same length")
    if np.any(links <= 0):
        raise ValueError("link lengths must be positive")
    phi = np.cumsum(thetas)
    return float(links @ np.cos(phi)), float(links @ np.sin(phi))


def arm_jacobian(thetas, links):
    """Analytic 2xN geometric Jacobian J = d(x, y)/d(theta) of the planar arm.

    Row 0 = dx/dtheta_k = -sum_{i>=k} l_i sin(phi_i)
    Row 1 = dy/dtheta_k = +sum_{i>=k} l_i cos(phi_i)
    (each joint rotates the whole outboard chain, hence the tail sum)."""
    thetas = np.asarray(thetas, float)
    links = np.asarray(links, float)
    phi = np.cumsum(thetas)
    dx = -links * np.sin(phi)          # per-link contribution
    dy = links * np.cos(phi)
    # column k gets the sum over links i >= k -> reverse-cumsum of the tail
    jx = np.cumsum(dx[::-1])[::-1]
    jy = np.cumsum(dy[::-1])[::-1]
    return np.vstack([jx, jy])


def pose_covariance(thetas, links, joint_sigmas=None, cov=None):
    """2x2 end-effector position covariance from joint-angle uncertainty.

    Sigma_pos = J Sigma_theta J^T, with J the manipulator Jacobian. Pass either
    per-joint `joint_sigmas` (independent encoders -> diagonal Sigma_theta) or a
    full NxN `cov`. First-order/linear; exact for small angle errors."""
    J = arm_jacobian(thetas, links)
    n = J.shape[1]
    if cov is not None:
        St = np.asarray(cov, float)
    elif joint_sigmas is not None:
        St = np.diag(np.asarray(joint_sigmas, float) ** 2)
    else:
        raise ValueError("give joint_sigmas (independent) or a full cov matrix")
    return J @ St @ J.T


def error_ellipse(sigma_pos):
    """Turn a 2x2 position covariance into a 1-sigma error ellipse.
    Returns (semi_major, semi_minor, angle_rad, rms_radius) where the semi-axes
    are sqrt of the covariance eigenvalues, angle is the major-axis direction, and
    rms_radius = sqrt(trace) = sqrt(sigma_x^2 + sigma_y^2) is the RMS radial error."""
    vals, vecs = np.linalg.eigh(np.asarray(sigma_pos, float))
    vals = np.clip(vals, 0.0, None)
    semis = np.sqrt(vals)
    major_vec = vecs[:, int(np.argmax(vals))]
    angle = float(np.arctan2(major_vec[1], major_vec[0]))
    return (float(semis.max()), float(semis.min()), angle,
            float(np.sqrt(np.trace(np.asarray(sigma_pos, float)))))


# ── sqrt(N): the standard error of the mean of repeated reaches ──────
def sqrt_n_error(sigma, n):
    """Standard error of the mean of N independent measurements: sigma / sqrt(N).
    Averaging N reaches shrinks the pose scatter by sqrt(N) -- 4x trials halves it."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return float(sigma) / math.sqrt(n)


def reaches_needed(sigma, target_se):
    """Smallest N with sigma/sqrt(N) <= target_se, i.e. N = ceil((sigma/target_se)^2)."""
    if target_se <= 0:
        raise ValueError("target_se must be positive")
    return int(math.ceil((sigma / target_se) ** 2))


# ── log(N!): Stirling's series (the log-size of the rank sample space) ──
def log_factorial(n, terms=3):
    """log(n!) via Stirling's asymptotic series (natural log).

        ln(n!) ~ n ln n - n + (1/2)ln(2 pi n) + 1/(12 n) - 1/(360 n^3) + ...

    `terms` selects how many are kept (1: n ln n - n; 2: + the (1/2)ln term;
    3: + 1/(12n); 4: - 1/(360 n^3)). For n=0,1 returns 0 exactly. This is the
    log-count of the N! orderings a rank test's null distribution ranges over."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if n < 2:
        return 0.0
    s = n * math.log(n) - n
    if terms >= 2:
        s += 0.5 * math.log(2 * math.pi * n)
    if terms >= 3:
        s += 1.0 / (12 * n)
    if terms >= 4:
        s -= 1.0 / (360 * n ** 3)
    return float(s)


# ── rank-based null-hypothesis tests (distribution-free) ─────────────
def _normal_sf(z):
    """Upper-tail standard-normal survival P(Z > z) via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _ranks(x):
    """Ranks 1..n of x, ties assigned the average (fractional) rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    # average tied ranks
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, r)
        r = (sums / counts)[inv]
    return r


def mann_whitney_u(a, b):
    """Mann-Whitney U rank-sum test: do independent samples a, b come from the
    same distribution? Returns (U, z, two_sided_p).

    Pool and rank; R_a = sum of a's ranks; U_a = R_a - n_a(n_a+1)/2 counts the
    a-beats-b pairs; U = min(U_a, U_b). Under H0, E[U] = n_a n_b / 2 and (with the
    standard tie correction) U is ~Normal for moderate n -- that Gaussian is the
    Stirling/log(N!) shadow of the discrete permutation null. p from the normal."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        raise ValueError("both samples must be non-empty")
    n = na + nb
    r = _ranks(np.concatenate([a, b]))
    Ra = r[:na].sum()
    Ua = Ra - na * (na + 1) / 2.0
    Ub = na * nb - Ua
    U = min(Ua, Ub)
    mu = na * nb / 2.0
    # tie-corrected variance
    _, counts = np.unique(np.concatenate([a, b]), return_counts=True)
    tie = np.sum(counts ** 3 - counts)
    var = na * nb / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0:
        return float(U), 0.0, 1.0
    z = (U - mu) / math.sqrt(var)          # <= 0 since U <= mu
    return float(U), float(z), float(2 * _normal_sf(abs(z)))


def wilcoxon_signed_rank(diffs):
    """Wilcoxon signed-rank test for paired data (e.g. same targets, old vs new
    calibration): is the median difference zero? Returns (W_plus, z, two_sided_p).

    Drop zero differences; rank |diff|; W+ = sum of ranks with diff > 0. Under H0,
    E[W+] = n(n+1)/4, Var = n(n+1)(2n+1)/24, and W+ is ~Normal for moderate n."""
    d = np.asarray(diffs, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        raise ValueError("all differences are zero -- nothing to test")
    r = _ranks(np.abs(d))
    W_plus = r[d > 0].sum()
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return float(W_plus), 0.0, 1.0
    z = (W_plus - mu) / math.sqrt(var)
    return float(W_plus), float(z), float(2 * _normal_sf(abs(z)))


def decide(p_value, alpha=0.05):
    """The decision logic of a hypothesis test, stated plainly. Returns a dict
    {reject, alpha, p_value, verdict}: reject H0 iff p <= alpha. A small p means
    the observed effect would be surprising if H0 (no difference) were true."""
    reject = p_value <= alpha
    return {
        "reject": bool(reject),
        "alpha": float(alpha),
        "p_value": float(p_value),
        "verdict": ("reject H0: the effect is significant at alpha=%.2g" % alpha)
        if reject else
        ("fail to reject H0: consistent with chance at alpha=%.2g" % alpha),
    }


if __name__ == "__main__":
    # a 2-link planar arm reaching out; each encoder has 0.5 deg of jitter
    links = [1.0, 0.8]
    thetas = [math.radians(30), math.radians(45)]
    js = [math.radians(0.5), math.radians(0.5)]
    x, y = planar_arm_fk(thetas, links)
    Sp = pose_covariance(thetas, links, js)
    a_maj, a_min, ang, rms = error_ellipse(Sp)
    print(f"tip at ({x:.3f}, {y:.3f}) m")
    print(f"1-sigma error ellipse: {a_maj*1e3:.2f} x {a_min*1e3:.2f} mm "
          f"at {math.degrees(ang):.0f} deg; RMS radial {rms*1e3:.2f} mm")
    print(f"average 25 reaches -> RMS shrinks to {sqrt_n_error(rms, 25)*1e3:.2f} mm "
          f"(sqrt(25)=5x); need {reaches_needed(rms, 0.1e-3)} reaches for 0.1 mm")

    # log(N!) Stirling vs exact
    for nn in (5, 20):
        exact = math.log(math.factorial(nn))
        print(f"ln({nn}!) exact={exact:.5f}  Stirling(3 terms)={log_factorial(nn):.5f}")

    # rank test: is the new calibration's radial error lower? (paired, per target)
    rng = np.random.default_rng(0)
    old = rng.normal(1.0, 0.3, 12)
    new = old - rng.normal(0.25, 0.1, 12)     # new is genuinely a bit better
    W, z, p = wilcoxon_signed_rank(old - new)
    print(f"Wilcoxon signed-rank: W+={W:.0f}, z={z:.2f}, p={p:.4f} -> "
          f"{decide(p)['verdict']}")
