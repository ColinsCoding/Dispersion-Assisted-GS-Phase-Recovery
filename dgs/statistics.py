"""Probability and statistics — distributions, CLT, Bayesian, attention models.

ATTENTION SPAN CONNECTION:
  Human attention follows an exponential decay model:
    P(still attending at time t) = exp(-t/tau)
  This is the survival function of an exponential distribution.
  Mean attention span tau ~ 8-20 s (mobile) to 45 min (lecture) depending on
  novelty and task complexity. The 'attention head' in transformers computes a
  softmax-weighted average — same Boltzmann distribution as thermal physics.

HILBERT SPACE + PROBABILITY:
  L²[a,b] is a Hilbert space. A probability density p(x) lives in this space:
    ||p||² = int p(x)^2 dx  (square-integrable)
  The inner product <f,g> = int f(x)*g(x) dx (or with weight function).
  Orthogonal polynomials (Hermite, Laguerre) are the basis functions for
  Gaussian and Poisson distributions on this Hilbert space.

TRIANGLE INEQUALITY IN STATISTICS:
  The triangle inequality ||f+g|| <= ||f|| + ||g|| in L² gives:
    Var(X+Y) <= (sigma_X + sigma_Y)^2  (with equality only if perfectly correlated)
  Cauchy-Schwarz: |Cov(X,Y)| <= sigma_X * sigma_Y -> |rho| <= 1
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════════════════════════
# §1  PROBABILITY DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════

def gaussian_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> Dict:
    """Gaussian N(mu, sigma²): pdf, cdf, moments.

    Connection to Hilbert space: N(0,1) is the ground state of the quantum
    harmonic oscillator — Hermite polynomials H_n(x)*exp(-x²/2) are
    orthonormal basis of L²(R, exp(-x²)).
    """
    pdf = np.exp(-0.5 * ((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
    # CDF via error function: Phi(x) = 0.5*(1 + erf((x-mu)/(sigma*sqrt(2))))
    from numpy import sqrt as _sqrt
    z = (x - mu) / (sigma * np.sqrt(2))
    cdf = 0.5 * (1 + _erf(z))
    return {
        "pdf":      pdf,
        "cdf":      cdf,
        "mean":     mu,
        "variance": sigma**2,
        "std":      sigma,
        "skewness": 0.0,
        "kurtosis": 0.0,   # excess kurtosis
        "entropy_nats": 0.5 * np.log(2 * np.pi * np.e * sigma**2),
    }


def _erf(z: np.ndarray) -> np.ndarray:
    """Error function via Horner-form approximation (Abramowitz & Stegun 7.1.26)."""
    t = 1 / (1 + 0.3275911 * np.abs(z))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (
           1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf_abs = 1 - poly * np.exp(-z**2)
    return np.where(z >= 0, erf_abs, -erf_abs)


def exponential_pdf(x: np.ndarray, lam: float = 1.0) -> Dict:
    """Exponential Exp(lambda): attention span, radioactive decay, Poisson inter-arrivals.

    Memoryless: P(T > s+t | T > s) = P(T > t).
    Mean = 1/lambda, Variance = 1/lambda^2.

    Attention span model: lambda = 1/tau where tau is mean attention time.
    """
    pdf = np.where(x >= 0, lam * np.exp(-lam * x), 0.0)
    cdf = np.where(x >= 0, 1 - np.exp(-lam * x), 0.0)
    return {
        "pdf":          pdf,
        "cdf":          cdf,
        "survival":     1 - cdf,   # P(T > x) = exp(-lambda*x)
        "mean":         1 / lam,
        "variance":     1 / lam**2,
        "median":       np.log(2) / lam,
        "hazard_rate":  lam,       # constant hazard = memoryless
        "entropy_nats": 1 - np.log(lam),
    }


def poisson_pmf(k: np.ndarray, lam: float) -> Dict:
    """Poisson PMF: P(K=k) = lambda^k * exp(-lambda) / k!

    Photon counting: at low light levels, photon arrivals follow Poisson.
    SNR = sqrt(N) where N is mean photon count.
    """
    k = np.asarray(k, dtype=int)
    # Use log-space to avoid overflow
    log_pmf = k * np.log(lam) - lam - _log_factorial(k)
    pmf = np.exp(log_pmf)
    return {
        "pmf":      pmf,
        "k":        k,
        "mean":     lam,
        "variance": lam,      # mean = variance for Poisson
        "std":      np.sqrt(lam),
        "SNR":      np.sqrt(lam),   # shot-noise limited SNR
    }


def _log_factorial(k: np.ndarray) -> np.ndarray:
    """log(k!) via Stirling for large k; exact for small k."""
    from math import lgamma
    return np.array([float(lgamma(ki + 1)) for ki in k.flat]).reshape(k.shape)


def binomial_pmf(k: np.ndarray, n: int, p: float) -> Dict:
    """Binomial B(n,p): n Bernoulli trials with success probability p."""
    k = np.asarray(k, dtype=int)
    from math import comb
    pmf = np.array([comb(n, ki) * p**ki * (1-p)**(n-ki) for ki in k.flat])
    pmf = pmf.reshape(k.shape)
    return {
        "pmf":      pmf,
        "mean":     n * p,
        "variance": n * p * (1-p),
        "std":      np.sqrt(n * p * (1-p)),
    }


# ════════════════════════════════════════════════════════════════════════════
# §2  CENTRAL LIMIT THEOREM
# ════════════════════════════════════════════════════════════════════════════

def clt_demo(dist: str = "exponential", lam: float = 1.0,
             sample_sizes: List[int] = None,
             n_experiments: int = 2000,
             seed: int = 42) -> Dict:
    """Demonstrate CLT: sample mean of iid rvs -> Gaussian regardless of dist.

    For X_i ~ Exp(lambda):
      E[X_i] = 1/lambda, Var[X_i] = 1/lambda^2
      Xbar_n = (X1+...+Xn)/n -> N(1/lambda, 1/(n*lambda^2))
      Standardized: sqrt(n)*(Xbar - mu)/sigma -> N(0,1)
    """
    if sample_sizes is None:
        sample_sizes = [1, 5, 30, 100]
    rng = np.random.default_rng(seed)

    if dist == "exponential":
        mu_true = 1 / lam
        sigma_true = 1 / lam
        sampler = lambda size: rng.exponential(1/lam, size=size)
    elif dist == "uniform":
        mu_true = 0.5
        sigma_true = 1 / np.sqrt(12)
        sampler = lambda size: rng.uniform(0, 1, size=size)
    else:
        raise ValueError(f"Unknown dist: {dist}")

    results = {}
    for n in sample_sizes:
        samples = sampler((n_experiments, n))   # (n_experiments, n)
        means   = samples.mean(axis=1)          # sample means
        # Standardize
        std_err = sigma_true / np.sqrt(n)
        z_scores = (means - mu_true) / std_err
        # KS-like normality: fraction within 1,2,3 sigma
        frac_1s = float(np.mean(np.abs(z_scores) <= 1))
        frac_2s = float(np.mean(np.abs(z_scores) <= 2))
        frac_3s = float(np.mean(np.abs(z_scores) <= 3))
        results[n] = {
            "means":    means,
            "z_scores": z_scores,
            "sample_mean_mean": float(means.mean()),
            "sample_mean_std":  float(means.std()),
            "theoretical_std":  std_err,
            "frac_1sigma":      frac_1s,   # ~0.683
            "frac_2sigma":      frac_2s,   # ~0.954
            "frac_3sigma":      frac_3s,   # ~0.997
            "converged":        frac_2s > 0.93,
        }
    return {
        "dist":        dist,
        "sample_sizes": sample_sizes,
        "mu_true":     mu_true,
        "sigma_true":  sigma_true,
        "results":     results,
    }


# ════════════════════════════════════════════════════════════════════════════
# §3  HYPOTHESIS TESTING
# ════════════════════════════════════════════════════════════════════════════

def z_test(x_bar: float, mu0: float, sigma: float, n: int,
           alternative: str = "two-sided") -> Dict:
    """Z-test: test H0: mu = mu0 when sigma is known.

    Test statistic: z = (x_bar - mu0) / (sigma / sqrt(n))
    p-value from standard normal.
    """
    z = (x_bar - mu0) / (sigma / np.sqrt(n))
    # p-value via erf
    phi_z = 0.5 * (1 + _erf(np.array([z / np.sqrt(2)]))[0])
    if alternative == "two-sided":
        p_val = 2 * min(phi_z, 1 - phi_z)
    elif alternative == "greater":
        p_val = 1 - phi_z
    else:   # less
        p_val = phi_z
    return {
        "z_stat":     float(z),
        "p_value":    float(p_val),
        "reject_H0_05": p_val < 0.05,
        "reject_H0_01": p_val < 0.01,
        "effect_size":  float((x_bar - mu0) / sigma),  # Cohen's d (sigma known)
        "CI_95_lower":  float(x_bar - 1.96 * sigma / np.sqrt(n)),
        "CI_95_upper":  float(x_bar + 1.96 * sigma / np.sqrt(n)),
    }


def t_test_one_sample(data: np.ndarray, mu0: float = 0.0,
                      alternative: str = "two-sided") -> Dict:
    """One-sample t-test: H0: mu = mu0 (sigma unknown, estimated from data).

    t = (x_bar - mu0) / (s / sqrt(n))  with n-1 degrees of freedom.
    p-value via t-distribution (approximated via normal for large n).
    """
    n = len(data)
    x_bar = float(data.mean())
    s = float(data.std(ddof=1))
    t_stat = (x_bar - mu0) / (s / np.sqrt(n))
    # Approximate p via normal (good for n >= 30)
    phi = 0.5 * (1 + _erf(np.array([t_stat / np.sqrt(2)]))[0])
    if alternative == "two-sided":
        p_val = 2 * min(phi, 1 - phi)
    elif alternative == "greater":
        p_val = 1 - phi
    else:
        p_val = phi
    return {
        "t_stat":     float(t_stat),
        "n":          n,
        "df":         n - 1,
        "x_bar":      x_bar,
        "s":          s,
        "p_value":    float(p_val),
        "reject_H0":  p_val < 0.05,
        "CI_95_lower": x_bar - 1.96 * s / np.sqrt(n),
        "CI_95_upper": x_bar + 1.96 * s / np.sqrt(n),
    }


# ════════════════════════════════════════════════════════════════════════════
# §4  BAYESIAN INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def bayesian_update_beta(successes: int, failures: int,
                          prior_alpha: float = 1.0,
                          prior_beta: float = 1.0) -> Dict:
    """Bayesian update for Bernoulli parameter p using Beta-Binomial model.

    Prior:    p ~ Beta(alpha, beta)   (alpha=beta=1 = uniform prior)
    Likelihood: x | p ~ Binomial(n, p)
    Posterior: p | x ~ Beta(alpha + successes, beta + failures)

    Conjugate prior: posterior is same family as prior -> closed form.

    Application: estimate THC% from n test measurements, k positives.
    """
    post_alpha = prior_alpha + successes
    post_beta  = prior_beta  + failures
    # Beta mean = alpha/(alpha+beta), variance = alpha*beta/((a+b)^2*(a+b+1))
    mean = post_alpha / (post_alpha + post_beta)
    var  = (post_alpha * post_beta /
            ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1)))
    mode = ((post_alpha - 1) / (post_alpha + post_beta - 2)
            if post_alpha > 1 and post_beta > 1 else 0.0)
    # 95% credible interval (approximate via normal)
    std = np.sqrt(var)
    return {
        "prior_alpha":  prior_alpha,
        "prior_beta":   prior_beta,
        "post_alpha":   post_alpha,
        "post_beta":    post_beta,
        "posterior_mean": mean,
        "posterior_mode": mode,
        "posterior_std":  std,
        "CI_95_lower":    max(0, mean - 1.96*std),
        "CI_95_upper":    min(1, mean + 1.96*std),
        "n_obs":   successes + failures,
        "successes": successes,
    }


def bayesian_gaussian_update(data: np.ndarray,
                              prior_mu: float = 0.0,
                              prior_sigma: float = 10.0,
                              likelihood_sigma: float = 1.0) -> Dict:
    """Bayesian update for Gaussian mean: prior N(mu0,sigma0) + data -> posterior.

    With known likelihood variance sigma_L:
      posterior_mu = (mu0/sigma0^2 + n*x_bar/sigma_L^2) / (1/sigma0^2 + n/sigma_L^2)
      posterior_var = 1 / (1/sigma0^2 + n/sigma_L^2)
    """
    n = len(data)
    x_bar = float(data.mean())
    prec_prior = 1 / prior_sigma**2
    prec_like  = n / likelihood_sigma**2
    prec_post  = prec_prior + prec_like
    mu_post    = (prec_prior * prior_mu + prec_like * x_bar) / prec_post
    sigma_post = np.sqrt(1 / prec_post)
    return {
        "n":          n,
        "x_bar":      x_bar,
        "prior_mu":   prior_mu,
        "prior_sigma": prior_sigma,
        "post_mu":    float(mu_post),
        "post_sigma": float(sigma_post),
        "CI_95_lower": float(mu_post - 1.96*sigma_post),
        "CI_95_upper": float(mu_post + 1.96*sigma_post),
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  ATTENTION MODEL + INFORMATION THEORY
# ════════════════════════════════════════════════════════════════════════════

def attention_span_model(tau_s: float = 20.0,
                          t_max_s: float = 90.0,
                          n_pts: int = 500) -> Dict:
    """Exponential decay model of sustained attention.

    P(attending at time t) = exp(-t/tau)
    Information received by time T: I(T) = int_0^T r * exp(-t/tau) dt
                                         = r*tau*(1 - exp(-T/tau))
    where r = information rate (bits/s).

    The 'attention head' in transformers: softmax(QK^T/sqrt(d)) * V
    Each attention weight = Boltzmann probability exp(-E/kT)/Z
    with E = -QK^T (negative dot product = energy), kT = sqrt(d).
    """
    t = np.linspace(0, t_max_s, n_pts)
    survival = np.exp(-t / tau_s)
    # Cumulative information (normalized)
    cum_info = 1 - np.exp(-t / tau_s)
    # Half-attention time: t such that survival = 0.5
    t_half = tau_s * np.log(2)
    return {
        "t_s":        t,
        "survival":   survival,
        "cum_info":   cum_info,
        "tau_s":      tau_s,
        "t_half_s":   t_half,
        "efficiency": float(1 - np.exp(-t_max_s / tau_s)),
        "optimal_chunk_s": tau_s,   # chunk sessions to tau for max efficiency
    }


def shannon_information(probs: np.ndarray,
                         base: float = 2.0) -> Dict:
    """Shannon entropy H = -sum p_i * log_b(p_i).

    Triangle inequality in information space:
      H(X,Y) <= H(X) + H(Y)  (subadditivity)
      H(X|Y) >= 0             (conditioning reduces entropy)
      I(X;Y) = H(X) - H(X|Y) >= 0  (mutual information non-negative)
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()   # normalize
    log_base = np.log(base)
    H = float(-np.sum(probs * np.log(probs + 1e-300) / log_base))
    # Max entropy = log_b(n) for uniform
    n = len(probs)
    H_max = np.log(n) / log_base
    return {
        "entropy":      H,
        "H_max":        H_max,
        "efficiency":   H / H_max if H_max > 0 else 1.0,
        "n_symbols":    n,
        "base":         base,
        "surprisal":    -np.log(probs + 1e-300) / log_base,  # per-symbol
    }


# ════════════════════════════════════════════════════════════════════════════
# §6  TRIANGLE INEQUALITY + CAUCHY-SCHWARZ
# ════════════════════════════════════════════════════════════════════════════

def triangle_inequality_demo(f: np.ndarray, g: np.ndarray,
                               x: np.ndarray) -> Dict:
    """Verify triangle inequality ||f+g|| <= ||f|| + ||g|| in L²[x].

    Also: Cauchy-Schwarz |<f,g>| <= ||f|| * ||g||
    -> |Corr(f,g)| = |<f,g>| / (||f|| * ||g||) <= 1

    Triangle approximation:
      Any function can be approximated by piecewise linear (triangle) basis:
      f(x) approx sum_i f(x_i) * phi_i(x)
      where phi_i = hat/tent function (triangle centered at x_i).
      Error: ||f - f_approx||² <= (h²/8) * ||f''||²  (O(h²) convergence)
    """
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    norm_f   = np.sqrt(np.trapezoid(f**2, x))
    norm_g   = np.sqrt(np.trapezoid(g**2, x))
    norm_fpg = np.sqrt(np.trapezoid((f+g)**2, x))
    inner_fg = np.trapezoid(f * g, x)
    corr = inner_fg / (norm_f * norm_g + 1e-300)

    # Triangle approximation (piecewise linear interpolation)
    # Sample f at coarse grid and interpolate with hat functions
    n_coarse = max(5, len(x) // 10)
    idx_coarse = np.linspace(0, len(x)-1, n_coarse, dtype=int)
    x_coarse = x[idx_coarse]
    f_coarse = f[idx_coarse]
    f_triangle = np.interp(x, x_coarse, f_coarse)
    h = x_coarse[1] - x_coarse[0] if len(x_coarse) > 1 else 1.0
    triangle_error = np.sqrt(np.trapezoid((f - f_triangle)**2, x))

    return {
        "norm_f":       float(norm_f),
        "norm_g":       float(norm_g),
        "norm_f_plus_g": float(norm_fpg),
        "triangle_satisfied": bool(norm_fpg <= norm_f + norm_g + 1e-10),
        "inner_product":  float(inner_fg),
        "correlation":    float(corr),
        "cauchy_schwarz": bool(abs(inner_fg) <= norm_f * norm_g + 1e-10),
        "triangle_approx_error": float(triangle_error),
        "n_coarse_pts":  n_coarse,
    }


# ════════════════════════════════════════════════════════════════════════════
# §7  SYMPY: 5 EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def statistics_sympy_5() -> Dict:
    """5 core equations: Gaussian, CLT, Bayes, triangle inequality, attention."""
    x, mu, sigma, n, t, lam = sp.symbols("x mu sigma n t lambda", real=True, positive=True)
    # 1. Gaussian PDF
    eq1 = sp.Eq(sp.Symbol("f(x)"),
                sp.exp(-(x-mu)**2 / (2*sigma**2)) / (sigma*sp.sqrt(2*sp.pi)))
    # 2. CLT: Xbar -> N(mu, sigma^2/n)
    eq2 = sp.Eq(sp.Symbol("SE"),
                sigma / sp.sqrt(n))
    # 3. Bayes' theorem
    P_A, P_B, P_BA = sp.symbols("P_A P_B P(B|A)", positive=True)
    eq3 = sp.Eq(sp.Symbol("P(A|B)"),
                P_BA * P_A / P_B)
    # 4. Triangle inequality in L²
    f_sym, g_sym = sp.symbols("||f|| ||g||", positive=True)
    eq4 = sp.Eq(sp.Symbol("||f+g||"),
                sp.Le(sp.Symbol("||f+g||_val"), f_sym + g_sym))
    # 5. Attention decay
    eq5 = sp.Eq(sp.Symbol("P(attending at t)"),
                sp.exp(-lam * t))
    return {
        "Gaussian_pdf":        eq1,
        "CLT_standard_error":  eq2,
        "Bayes_theorem":       eq3,
        "Triangle_inequality": eq4,
        "Attention_decay":     eq5,
    }


if __name__ == "__main__":
    print("=== Gaussian PDF ===")
    x = np.linspace(-4, 4, 200)
    g = gaussian_pdf(x, mu=0, sigma=1)
    print(f"  peak pdf = {g['pdf'].max():.4f}  (theory: {1/np.sqrt(2*np.pi):.4f})")
    print(f"  entropy  = {g['entropy_nats']:.4f} nats")

    print("\n=== Exponential (attention span tau=20s) ===")
    t = np.linspace(0, 60, 300)
    e = exponential_pdf(t, lam=1/20)
    print(f"  P(T>20s) = {e['survival'][100]:.4f}  (theory: {np.exp(-1):.4f})")
    print(f"  median   = {e['median']:.1f} s  (theory: {20*np.log(2):.1f} s)")

    print("\n=== CLT Demo: Exp(1) -> Gaussian ===")
    res = clt_demo("exponential", lam=1.0, sample_sizes=[1, 30, 100])
    for n, r in res["results"].items():
        print(f"  n={n:3d}: frac_2sigma={r['frac_2sigma']:.3f}  (converged={r['converged']})")

    print("\n=== Bayesian Update: THC test, 7/10 positive ===")
    b = bayesian_update_beta(7, 3, prior_alpha=2, prior_beta=2)
    print(f"  Posterior mean = {b['posterior_mean']:.3f}")
    print(f"  95% CI = [{b['CI_95_lower']:.3f}, {b['CI_95_upper']:.3f}]")

    print("\n=== Triangle Inequality in L² ===")
    x = np.linspace(0, 2*np.pi, 500)
    res = triangle_inequality_demo(np.sin(x), np.cos(x), x)
    print(f"  ||sin||   = {res['norm_f']:.4f}  (theory: {np.sqrt(np.pi):.4f})")
    print(f"  ||cos||   = {res['norm_g']:.4f}")
    print(f"  ||sin+cos|| = {res['norm_f_plus_g']:.4f} <= {res['norm_f']+res['norm_g']:.4f}? {res['triangle_satisfied']}")
    print(f"  Cauchy-Schwarz: {res['cauchy_schwarz']}")
    print(f"  Triangle approx error: {res['triangle_approx_error']:.4f}")

    print("\n=== Attention Span Model ===")
    att = attention_span_model(tau_s=20.0, t_max_s=90.0)
    print(f"  Half-attention time: {att['t_half_s']:.1f} s")
    print(f"  90s session efficiency: {att['efficiency']:.1%}")


# ════════════════════════════════════════════════════════════════════════════
# §8  BINARY SEARCH TREE
# ════════════════════════════════════════════════════════════════════════════

class BSTNode:
    __slots__ = ("val", "left", "right")
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class BST:
    """Binary Search Tree: insert O(log n) avg, search O(log n) avg.

    Physics connection: BST ~ divide-and-conquer frequency search;
    same idea as FFT splitting even/odd indices at each level.
    """
    def __init__(self):
        self.root = None

    def insert(self, val):
        def _ins(node, v):
            if node is None:
                return BSTNode(v)
            if v < node.val:
                node.left = _ins(node.left, v)
            elif v > node.val:
                node.right = _ins(node.right, v)
            return node
        self.root = _ins(self.root, val)

    def search(self, val):
        node = self.root
        while node:
            if val == node.val:
                return True
            node = node.left if val < node.val else node.right
        return False

    def inorder(self):
        result = []
        def _dfs(n):
            if n:
                _dfs(n.left)
                result.append(n.val)
                _dfs(n.right)
        _dfs(self.root)
        return result

    def height(self):
        def _h(n):
            return 0 if n is None else 1 + max(_h(n.left), _h(n.right))
        return _h(self.root)

    def ascii_lines(self):
        """Return list of strings showing the tree structure."""
        if self.root is None:
            return ["(empty)"]
        lines = []
        def _draw(node, prefix, is_left):
            if node is None:
                return
            _draw(node.right, prefix + ("    " if is_left else "|   "), False)
            lines.append(prefix + ("+-- " if is_left else "\\-- ") + str(node.val))
            _draw(node.left,  prefix + ("|   " if is_left else "    "), True)
        lines.append(str(self.root.val))
        _draw(self.root.right, "    ", False)
        _draw(self.root.left,  "    ", True)
        return lines


# ════════════════════════════════════════════════════════════════════════════
# §9  PCA via SVD
# ════════════════════════════════════════════════════════════════════════════

def pca(X, n_components=None):
    """Principal Component Analysis via numpy SVD (no scipy).

    Physics connection: PCA eigenvectors ARE the normal modes of the
    data covariance matrix -- same math as solving coupled oscillators.
    The Hermitian operator C = X^T X / (n-1) has eigenvalues = variances.

    Returns dict: components, explained_var, explained_ratio, scores, mean
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    n, p = X.shape
    if n_components is None:
        n_components = min(n, p)
    n_components = int(n_components)
    if not (1 <= n_components <= min(n, p)):
        raise ValueError(f"n_components must be in [1, {min(n,p)}]")
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    explained_var = (S ** 2) / max(n - 1, 1)
    total_var = explained_var.sum() + 1e-300
    return {
        "components":      Vt[:n_components],
        "explained_var":   explained_var[:n_components],
        "explained_ratio": explained_var[:n_components] / total_var,
        "scores":          (Xc @ Vt.T)[:, :n_components],
        "mean":            mean,
        "singular_values": S[:n_components],
    }


# ════════════════════════════════════════════════════════════════════════════
# §10  CHI-SQUARED GOODNESS OF FIT
# ════════════════════════════════════════════════════════════════════════════

def chi_squared_goodness_of_fit(observed, expected):
    """Pearson chi-squared test for goodness of fit.

    Physics use: "Do my counting experiment results match the Poisson model?"
    Rule: expected counts >= 5 per bin (merge bins if not).

    Returns dict: chi2, p_value, df, reduced_chi2, good_fit
    """
    import math
    obs = np.asarray(observed, float)
    exp = np.asarray(expected, float)
    if obs.shape != exp.shape:
        raise ValueError("observed and expected must be the same shape")
    if np.any(exp < 1):
        raise ValueError("expected counts must be >= 1; merge low-count bins")
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    df = len(obs) - 1
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, df))
    except ImportError:
        z = (chi2 - df) / math.sqrt(2 * df + 1e-12)
        p = float(0.5 * math.erfc(z / math.sqrt(2)))
    return {
        "chi2": chi2,
        "p_value": p,
        "df": df,
        "reduced_chi2": chi2 / df if df > 0 else float("nan"),
        "good_fit": bool(0.05 < p < 0.95),
    }


def poisson_counting_experiment(true_rate, n_measurements, rng=None):
    """Simulate a Poisson counting experiment (radioactive decay, photon counting).

    Returns observed and expected bin counts ready for chi_squared_goodness_of_fit.
    """
    import math
    if rng is None:
        rng = np.random.default_rng(0)
    if true_rate <= 0:
        raise ValueError("true_rate must be positive")
    if n_measurements < 10:
        raise ValueError("n_measurements must be >= 10")
    observed = rng.poisson(true_rate, size=n_measurements)
    max_k = int(true_rate * 3) + 1
    bins = np.arange(max_k + 1)
    counts, _ = np.histogram(observed, bins=bins)
    k_vals = bins[:-1].astype(float)
    log_rate = math.log(true_rate) if true_rate > 0 else 0.0
    expected = n_measurements * np.array([
        math.exp(-true_rate + k * log_rate - sum(math.log(i+1) for i in range(int(k))))
        for k in k_vals
    ])
    mask = (expected >= 5) & (counts > 0)
    if mask.sum() < 2:
        mask = counts > 0
    return {
        "observed": counts[mask],
        "expected": expected[mask],
        "k_vals":   k_vals[mask],
        "true_rate": true_rate,
    }


def t_test_two_sample(data_a, data_b):
    """Welch two-sample t-test (unequal variances).

    Physics use: "Do two experimental conditions give different results?"
    Returns dict: t_stat, p_value, mean_a, mean_b, df, significant_at_5pct
    """
    import math
    a, b = np.asarray(data_a, float), np.asarray(data_b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("Need at least 2 points per group")
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb + 1e-300)
    t_stat = (ma - mb) / se
    num = (va/na + vb/nb)**2
    den = (va/na)**2/(na-1) + (vb/nb)**2/(nb-1)
    df = num / (den + 1e-300)
    try:
        from scipy.stats import t as t_dist
        p = float(t_dist.sf(abs(t_stat), df) * 2)
    except ImportError:
        import math
        p = float(math.erfc(abs(t_stat) / math.sqrt(2)))
    return {
        "t_stat": float(t_stat),
        "p_value": float(p),
        "mean_a": float(ma),
        "mean_b": float(mb),
        "df": float(df),
        "significant_at_5pct": bool(p < 0.05),
    }
    print(f"  Optimal chunk duration: {att['optimal_chunk_s']:.0f} s")
