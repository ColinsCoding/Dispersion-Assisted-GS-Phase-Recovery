"""Probability and combinatorics -- odds, parity, order statistics, luck.

CENTRAL IDEA: "luck" in engineering is just a geometric distribution --
the number of trials before the first success. The math of luck, first/last
occurrence, and odd/even parity all live in the same probability framework.

Key functions:
  odds()               -- convert probability to odds ratio and back
  geometric_pmf()      -- P(first success on trial k) = (1-p)^(k-1) * p
  first_success_stats()-- mean/variance of the geometric distribution
  order_statistic()    -- expected value of the k-th smallest in n samples
  parity_probability() -- P(random integer in range is odd or even)
  combinations()       -- C(n,k) = n! / (k! * (n-k)!)
  permutations()       -- P(n,k) = n! / (n-k)!
  birthday_problem()   -- P(collision) in n draws from m bins
  binomial_pmf()       -- P(X=k) for X~Binomial(n,p)

Connection to this repo:
  The GS algorithm is a random restart optimization. The probability that a
  random initial phase gives a converged solution after n_iter iterations is
  modeled by a geometric distribution: each restart is a Bernoulli trial.
  The expected number of restarts to find the global minimum is 1/p_success.
"""
import math
import numpy as np
import sympy as sp


# ── basic probability / odds conversion ──────────────────────────────

def prob_to_odds(p):
    """Convert probability p to odds ratio (for : against).

    Odds = p / (1-p). Example: p=0.25 -> 1:3 (1 for, 3 against).
    """
    if not (0 < p < 1):
        raise ValueError("p must be strictly between 0 and 1")
    return {"odds_for": p, "odds_against": 1 - p,
            "ratio": p / (1 - p),
            "display": f"{p/(1-p):.4f} : 1"}


def odds_to_prob(odds_for, odds_against):
    """Convert odds (for : against) back to probability."""
    if odds_for <= 0 or odds_against <= 0:
        raise ValueError("odds must be positive")
    return odds_for / (odds_for + odds_against)


# ── parity: odd and even ─────────────────────────────────────────────

def parity_probability(n_min, n_max):
    """Probability that a uniformly chosen integer in [n_min, n_max] is odd or even.

    Parity in computing: bit 0 of an integer. Parity in physics: symmetry
    of a wavefunction under reflection (Griffiths Ch 2 parity operator).
    Same word, same math: even/odd split of the integers.
    """
    if n_min > n_max:
        raise ValueError("n_min must be <= n_max")
    total = n_max - n_min + 1
    odds_count = sum(1 for x in range(n_min, n_max + 1) if x % 2 != 0)
    evens_count = total - odds_count
    return {
        "P_odd": odds_count / total,
        "P_even": evens_count / total,
        "n_odd": odds_count,
        "n_even": evens_count,
        "total": total,
    }


def is_odd(n):
    """True if n is odd. Equivalent to n & 1 in hardware (lowest bit)."""
    return bool(n & 1)


# ── geometric distribution (first success / "luck") ──────────────────

def geometric_pmf(p, k):
    """P(first success on exactly trial k) = (1-p)^(k-1) * p.

    This is the LUCK distribution: each trial is a Bernoulli(p) coin flip.
    k is how many tries until you win for the first time.
    """
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    if k < 1:
        raise ValueError("k must be >= 1")
    return (1 - p) ** (k - 1) * p


def geometric_cdf(p, k):
    """P(first success by trial k) = 1 - (1-p)^k."""
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    return 1 - (1 - p) ** k


def first_success_stats(p):
    """Mean and variance of the geometric distribution.

    Mean = 1/p (expected number of trials to first success).
    Variance = (1-p) / p^2.
    'Luck' means low k -- getting first success quickly.
    """
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    mean = 1.0 / p
    var = (1 - p) / p ** 2
    return {"mean_trials": mean, "variance": var, "std_dev": math.sqrt(var),
            "prob_success": p}


def prob_success_within_n(p, n):
    """P(at least one success in n independent trials) = 1 - (1-p)^n.

    Same as geometric CDF. The complement of 'no luck after n tries'.
    """
    return geometric_cdf(p, n)


# ── order statistics (first / last occurrence) ────────────────────────

def order_statistic_expected(k, n, dist_min=0.0, dist_max=1.0):
    """Expected value of the k-th order statistic from n Uniform(a,b) samples.

    E[X_(k)] = a + (b-a) * k / (n+1)

    The FIRST (minimum, k=1) and LAST (maximum, k=n) are special cases.
    This is how you size a net: if you want the expected MINIMUM of n
    measurements to be near a, use large n. In GS: the MINIMUM error over
    n random restarts decreases as 1/(n+1) of the range.
    """
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= n")
    if dist_min >= dist_max:
        raise ValueError("dist_min must be < dist_max")
    expected = dist_min + (dist_max - dist_min) * k / (n + 1)
    return {
        "expected_value": expected,
        "k": k,
        "n": n,
        "label": ("minimum (first)" if k == 1 else
                  "maximum (last)" if k == n else f"order stat {k} of {n}"),
    }


def min_max_expected(n, dist_min=0.0, dist_max=1.0):
    """Expected minimum and maximum of n Uniform(a,b) samples."""
    e_min = order_statistic_expected(1, n, dist_min, dist_max)["expected_value"]
    e_max = order_statistic_expected(n, n, dist_min, dist_max)["expected_value"]
    return {"expected_min": e_min, "expected_max": e_max, "n": n}


# ── combinations and permutations ─────────────────────────────────────

def combinations(n, k):
    """C(n,k) = n! / (k! * (n-k)!). Number of ways to choose k from n (order doesn't matter)."""
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        return 0
    return math.comb(n, k)


def permutations(n, k):
    """P(n,k) = n! / (n-k)!. Number of ordered arrangements of k from n."""
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        return 0
    return math.factorial(n) // math.factorial(n - k)


# ── binomial distribution ─────────────────────────────────────────────

def binomial_pmf(n, k, p):
    """P(X=k) for X ~ Binomial(n,p): C(n,k) * p^k * (1-p)^(n-k)."""
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0,1]")
    if k > n or k < 0:
        return 0.0
    return combinations(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binomial_mean_var(n, p):
    """Mean = n*p, Variance = n*p*(1-p) for Binomial(n,p)."""
    return {"mean": n * p, "variance": n * p * (1 - p),
            "std_dev": math.sqrt(n * p * (1 - p))}


# ── birthday problem ──────────────────────────────────────────────────

def birthday_collision_prob(n_people, n_days=365):
    """P(at least two people share a birthday) among n_people.

    P(collision) = 1 - prod_{k=0}^{n-1} (m-k)/m
    where m = number of equally likely outcomes (n_days = 365 for birthdays).
    The surprise: only 23 people needed for P > 0.5.
    """
    if n_people < 0 or n_days < 1:
        raise ValueError("n_people >= 0 and n_days >= 1 required")
    if n_people > n_days:
        return 1.0
    p_no_collision = 1.0
    for k in range(n_people):
        p_no_collision *= (n_days - k) / n_days
    return 1 - p_no_collision


# ── SymPy formalism ───────────────────────────────────────────────────

def probability_sympy_5():
    """Five key probability equations as SymPy Eq objects."""
    p, k_sym, n_sym = sp.symbols('p k n', positive=True)

    return {
        "Geometric_PMF":
            sp.Eq(sp.Symbol('P(X=k)'),
                  (1 - p)**(k_sym - 1) * p),
        "Expected_first_success":
            sp.Eq(sp.Symbol('E[X]'), 1 / p),
        "Prob_success_within_n":
            sp.Eq(sp.Symbol('P(X<=n)'), 1 - (1 - p)**n_sym),
        "Combinations":
            sp.Eq(sp.Symbol('C(n,k)'),
                  sp.factorial(n_sym) / (sp.factorial(k_sym) * sp.factorial(n_sym - k_sym))),
        "Order_statistic_uniform":
            sp.Eq(sp.Symbol('E[X_(k)]'),
                  (k_sym) / (n_sym + 1)),
    }


if __name__ == "__main__":
    print("=== Odds: P=0.25 (roll a 6-sided die, win on 1 or 2) ===")
    o = prob_to_odds(2/6)
    print(f"  odds ratio = {o['ratio']:.3f} : 1  ({o['display']})")

    print("\n=== Parity: integers 1 to 100 ===")
    par = parity_probability(1, 100)
    print(f"  P(odd)={par['P_odd']:.2f}  P(even)={par['P_even']:.2f}")

    print("\n=== Luck: geometric distribution (p=0.1) ===")
    stats = first_success_stats(0.1)
    print(f"  Mean trials to first success: {stats['mean_trials']:.1f}")
    print(f"  P(success within 10 tries): {prob_success_within_n(0.1, 10):.3f}")

    print("\n=== Order statistics: expected min/max of 10 Uniform[0,1] samples ===")
    mm = min_max_expected(10)
    print(f"  E[min] = {mm['expected_min']:.3f}  E[max] = {mm['expected_max']:.3f}")

    print("\n=== Combinations: C(52,5) = number of 5-card poker hands ===")
    print(f"  C(52,5) = {combinations(52,5):,}")

    print("\n=== Birthday problem ===")
    for n in [10, 23, 50]:
        p = birthday_collision_prob(n)
        print(f"  {n} people: P(shared birthday) = {p:.3f}")

    print("\n=== SymPy 5 ===")
    for k, eq in probability_sympy_5().items():
        print(f"  {k}: {eq}")
