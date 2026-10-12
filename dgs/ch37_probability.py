"""Knight Ch. 37 (Foundations of Modern Physics) through probability and
statistics -- formal random variables with sympy.stats, checked by simulation.

1. RUTHERFORD SCATTERING AS A CHANGE OF RANDOM VARIABLE
   An alpha hits a random point of a disc of radius b_max around a nucleus,
   so the impact parameter b has pdf 2b / b_max^2. Deflection (lengths in
   units of the closest-approach distance d):  theta = 2 atan(1 / (2b)).
   The pdf of theta follows from  f_theta = f_b(b(theta)) |db/dtheta|:
       f_theta = cos(theta/2) / (4 b_max^2 sin^3(theta/2))
   which IS Rutherford's 1/sin^4(theta/2) cross-section times the solid-angle
   element 2 pi sin(theta) dtheta. Tail:  P(theta > T) = cot^2(T/2) / (4 b_max^2).

2. RARE EVENTS ARE POISSON
   If each alpha back-scatters with probability p, the count in n alphas is
   ~Poisson(n p). Alphas needed to see >= 1 event with confidence c:
   n = -ln(1 - c) / p  ("rule of three" for c = 95 %).

3. MILLIKAN ERROR PROPAGATION BY LOG-DIFFERENTIATION
   q = m g d / V with m from Stokes' law: q is proportional to v^{3/2} / V, so
       sigma_q / q = sqrt((3/2 sigma_v/v)^2 + (sigma_V/V)^2).
   A drop with n charges has absolute noise n * sigma_rel * e; once that
   approaches e/2 the integer n can no longer be read off -- why Millikan
   preferred lightly charged drops.
"""
import numpy as np
import sympy as sp
from scipy.special import erfc
from sympy.stats import Normal, Poisson, P as Prob, E as Expect


# ── 1. Rutherford ────────────────────────────────────────────────────

theta, b, bmax = sp.symbols("theta b b_max", positive=True)


def rutherford_theta_pdf():
    """Derive f_theta symbolically by the change-of-variables rule."""
    f_b = 2 * b / bmax ** 2
    b_of_theta = sp.cot(theta / 2) / 2
    f = f_b.subs(b, b_of_theta) * sp.Abs(sp.diff(b_of_theta, theta))
    return sp.simplify(sp.trigsimp(f))


def rutherford_from_cross_section():
    """(1/(pi b_max^2)) dsigma/dOmega * 2 pi sin(theta), dsigma/dOmega = (1/4)^2/sin^4(theta/2)
    (lengths in units of d). Should equal rutherford_theta_pdf()."""
    dsig = sp.Rational(1, 16) / sp.sin(theta / 2) ** 4
    return sp.simplify(dsig * 2 * sp.pi * sp.sin(theta) / (sp.pi * bmax ** 2))


def theta_min(b_max):
    """Smallest deflection, reached at b = b_max."""
    return 2 * np.arctan(1 / (2 * b_max))


def tail_probability(T, b_max):
    """P(theta > T) = cot^2(T/2) / (4 b_max^2), valid for T >= theta_min."""
    T = np.asarray(T, dtype=float)
    if b_max <= 0:
        raise ValueError("b_max must be positive")
    return np.minimum(1.0, 1 / np.tan(T / 2) ** 2 / (4 * b_max ** 2))


def sample_theta(n, b_max, seed=0):
    """Monte Carlo: uniform points on the disc -> deflection angles."""
    if n <= 0 or b_max <= 0:
        raise ValueError("n and b_max must be positive")
    rng = np.random.default_rng(seed)
    bb = b_max * np.sqrt(rng.uniform(size=n))
    return 2 * np.arctan(1 / (2 * bb))


# ── 2. Poisson counting ──────────────────────────────────────────────

def alphas_needed(p, confidence=0.95):
    """Smallest n with P(at least one event) >= confidence, Poisson(n p)."""
    if not 0 < p < 1 or not 0 < confidence < 1:
        raise ValueError("0 < p < 1 and 0 < confidence < 1")
    return int(np.ceil(-np.log(1 - confidence) / p))


def poisson_at_least_one_symbolic():
    """sympy.stats: P(K >= 1) for K ~ Poisson(lam) = 1 - e^{-lam}."""
    lam = sp.Symbol("lambda", positive=True)
    K = Poisson("K", lam)
    return sp.simplify(Prob(K >= 1))


def poisson_mean_var_symbolic():
    lam = sp.Symbol("lambda", positive=True)
    K = Poisson("K", lam)
    return sp.simplify(Expect(K)), sp.simplify(Expect(K ** 2) - Expect(K) ** 2)


# ── 3. Millikan error propagation ────────────────────────────────────

def millikan_relative_sigma_symbolic():
    """Log-differentiate q = C v^{3/2} / V: d ln q = (3/2) d ln v - d ln V."""
    v, V, C, sv, sV = sp.symbols("v V C sigma_v sigma_V", positive=True)
    q = C * v ** sp.Rational(3, 2) / V
    dlnq_dv = sp.simplify(sp.diff(sp.log(q), v))
    dlnq_dV = sp.simplify(sp.diff(sp.log(q), V))
    rel = sp.sqrt((dlnq_dv * sv) ** 2 + (dlnq_dV * sV) ** 2)
    return sp.simplify(rel)


def millikan_relative_sigma(rel_v, rel_V):
    if rel_v < 0 or rel_V < 0:
        raise ValueError("relative errors must be non-negative")
    return float(np.hypot(1.5 * rel_v, rel_V))


def millikan_mc_relative_sigma(rel_v, rel_V, n=400_000, seed=0):
    rng = np.random.default_rng(seed)
    v = 1 + rel_v * rng.standard_normal(n)
    V = 1 + rel_V * rng.standard_normal(n)
    q = v ** 1.5 / V
    return float(q.std() / q.mean())


def misassignment_probability(n_charges, rel_sigma):
    """P(rounding q/e gives the wrong integer) = P(|noise| > e/2), noise ~ N(0, n rel e)."""
    n_charges = np.asarray(n_charges, dtype=float)
    if np.any(n_charges < 1) or rel_sigma <= 0:
        raise ValueError("n_charges >= 1 and rel_sigma > 0")
    return erfc(0.5 / (np.sqrt(2) * n_charges * rel_sigma))


def misassignment_symbolic():
    """sympy.stats version: P(|X| > 1/2) for X ~ Normal(0, s)."""
    s = sp.Symbol("s", positive=True)
    X = Normal("X", 0, s)
    return sp.simplify(Prob(X > sp.Rational(1, 2)) + Prob(X < -sp.Rational(1, 2)))
