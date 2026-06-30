"""Login/sign-in attempt probability, and the product rule that shows up
when you differentiate it.

If each login attempt independently fails with probability q_i (attempt i
succeeds with probability p_i = 1-q_i), the probability NONE of N attempts
succeed is the PRODUCT of individual failure probabilities:

    P(all fail) = q_1 * q_2 * ... * q_N

so P(at least one succeeds) = 1 - product(q_i). When q_i depends on a
parameter (e.g. attempt rate, account lockout backoff, password-strength
estimate that changes between attempts), differentiating that product
requires the PRODUCT RULE -- and because it's an N-term product, you need
the GENERALIZED (Leibniz) product rule:

    d/dx [f1*f2*...*fN] = sum_i ( (df_i/dx) * product_{j != i} f_j )

This module implements both the discrete brute-force-probability model and
the symbolic product-rule differentiation of it.
"""
import numpy as np
import sympy as sp


# -- Discrete brute-force / login attempt probability --------------------------

def single_attempt_success_prob(keyspace_size):
    """p = 1 / keyspace_size -- probability a single random guess is correct."""
    if keyspace_size <= 0:
        raise ValueError("keyspace_size must be positive")
    return 1.0 / keyspace_size


def prob_crack_within_n_attempts(p, n_attempts):
    """P(at least one success in n_attempts) = 1 - (1-p)^n_attempts, the
    constant-probability special case of the product rule above
    (q_i = (1-p) for every i, so the product collapses to a power)."""
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    q = 1 - p
    return 1 - q ** n_attempts


def expected_attempts_to_crack(p):
    """Geometric distribution: E[N] = 1/p attempts until first success."""
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    return 1.0 / p


def prob_crack_with_lockout(p, n_attempts, lockout_after, lockout_duration_s, attempt_interval_s):
    """Account-lockout defense: after `lockout_after` failed attempts, the
    attacker is blocked for `lockout_duration_s`. Returns the effective
    attempts-per-unit-time and the resulting crack probability over a fixed
    wall-clock window."""
    if lockout_after <= 0:
        raise ValueError("lockout_after must be positive")
    cycle_time = lockout_after * attempt_interval_s + lockout_duration_s
    cycles = n_attempts / lockout_after  # number of lockout cycles needed
    wall_clock_s = cycles * cycle_time
    crack_prob = prob_crack_within_n_attempts(p, n_attempts)
    return {"crack_prob": crack_prob, "wall_clock_s": wall_clock_s,
            "wall_clock_days": wall_clock_s / 86400}


def prob_crack_varying_attempts(p_list):
    """Product rule's natural habitat: each attempt has a DIFFERENT success
    probability p_i (e.g. an attacker narrowing the keyspace attempt to
    attempt). P(all fail) = product(1-p_i); P(at least one succeeds) =
    1 - that product."""
    p = np.asarray(p_list, float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("all probabilities must be in [0,1]")
    q = 1 - p
    prob_all_fail = np.prod(q)
    return {"prob_all_fail": float(prob_all_fail), "prob_at_least_one_success": float(1 - prob_all_fail)}


# -- Product rule, symbolically, on exactly this quantity ----------------------

def product_rule_two_factors():
    """The basic product rule: d/dx[f*g] = f'*g + f*g'."""
    x = sp.Symbol('x')
    f = sp.Function('f')(x)
    g = sp.Function('g')(x)
    lhs = sp.diff(f * g, x)
    return sp.Eq(sp.Symbol("d_dx[f*g]"), lhs)


def leibniz_product_rule_n_factors(n):
    """Generalized product rule for n factors: d/dx[prod f_i] =
    sum_i ( f_i' * prod_{j!=i} f_j ). Returns the symbolic expression for
    given n, built explicitly term by term (not just the summation notation)."""
    x = sp.Symbol('x')
    fs = [sp.Function(f'f{i+1}')(x) for i in range(n)]
    product = sp.prod(fs)
    derivative = sp.diff(product, x)
    return {"product": product, "derivative_expanded": sp.expand(derivative)}


def attempt_probability_product_rule_demo(n_attempts):
    """Apply the Leibniz product rule to the EXACT quantity from
    prob_crack_varying_attempts: q_i(x) = 1 - p_i(x), each depending on a
    shared parameter x (e.g. attacker's compute budget). Differentiate
    P(all fail) = product(q_i(x)) symbolically."""
    x = sp.Symbol('x', positive=True)
    qs = [sp.Function(f'q{i+1}')(x) for i in range(n_attempts)]
    P_all_fail = sp.prod(qs)
    dP_dx = sp.diff(P_all_fail, x)
    return {"P_all_fail": P_all_fail, "dP_dx_via_product_rule": sp.expand(dP_dx)}


def login_sympy_5():
    """Five symbolic equations tying login probability to the product rule."""
    p, n, x = sp.symbols('p n x', positive=True)
    q = 1 - p
    return {
        "Single_attempt_crack_prob":
            sp.Eq(sp.Symbol('P_crack'), 1 - q**n),
        "Expected_attempts_geometric":
            sp.Eq(sp.Symbol('E_N'), 1/p),
        "Two_factor_product_rule":
            product_rule_two_factors(),
        "N_factor_product_collapses_to_power":
            sp.Eq(sp.prod([q]*3), q**3),  # constant q_i case -> product = power
        "Generalized_leibniz_3_factors":
            sp.Eq(sp.Symbol("d_dx[f1*f2*f3]"),
                  leibniz_product_rule_n_factors(3)["derivative_expanded"]),
    }


if __name__ == "__main__":
    print("=== Single attempt success prob: 4-digit PIN ===")
    p = single_attempt_success_prob(10000)
    print(f"  p = {p}")

    print("\n=== Crack probability within N attempts ===")
    for n in [1, 100, 1000, 10000]:
        prob = prob_crack_within_n_attempts(p, n)
        print(f"  n={n:5d} attempts -> P(crack) = {prob:.4%}")

    print(f"\n=== Expected attempts to crack ===")
    print(f"  E[N] = {expected_attempts_to_crack(p):.0f}")

    print("\n=== With account lockout: 5 attempts, then 15 min lockout ===")
    res = prob_crack_with_lockout(p, n_attempts=10000, lockout_after=5,
                                   lockout_duration_s=900, attempt_interval_s=1)
    print(f"  crack_prob = {res['crack_prob']:.4%}, wall_clock = {res['wall_clock_days']:.2f} days")

    print("\n=== Varying per-attempt probability (attacker narrowing keyspace) ===")
    res2 = prob_crack_varying_attempts([0.0001, 0.001, 0.01, 0.05])
    print(f"  P(at least one success) = {res2['prob_at_least_one_success']:.4%}")

    print("\n=== Product rule, two factors ===")
    print(f"  {product_rule_two_factors()}")

    print("\n=== Leibniz product rule, 3 factors ===")
    res3 = leibniz_product_rule_n_factors(3)
    print(f"  d/dx[f1*f2*f3] = {res3['derivative_expanded']}")

    print("\n=== Product rule applied to login-attempt probability, 3 attempts ===")
    res4 = attempt_probability_product_rule_demo(3)
    print(f"  P_all_fail = {res4['P_all_fail']}")
    print(f"  dP/dx = {res4['dP_dx_via_product_rule']}")

    print("\n=== SymPy 5 ===")
    for k, eq in login_sympy_5().items():
        print(f"  {k}: {eq}")
