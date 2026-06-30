"""Bayes' theorem, the Dirac delta function, and symmetry -- three ideas that
share one structure: each is a way of LOCALIZING information.

  * Bayes' theorem localizes belief: P(H|D) concentrates probability mass
    onto the hypotheses consistent with the data.
  * The Dirac delta delta(x-a) localizes a function: it picks out (sifts)
    the value of a function at exactly one point, integral f(x)delta(x-a)dx=f(a).
  * Symmetry localizes WORK: an even/odd or rotationally-symmetric integrand
    lets you evaluate (or kill) an integral without doing the integration --
    you only need to know which symmetry class the integrand is in.

All three are "don't compute everything, exploit structure" tools.
"""
import numpy as np
import sympy as sp


# -- Bayes' theorem -------------------------------------------------------------

def bayes_theorem(prior, likelihood, evidence):
    """P(H|D) = P(D|H) * P(H) / P(D).
    prior=P(H), likelihood=P(D|H), evidence=P(D)."""
    if not (0 <= prior <= 1):
        raise ValueError("prior must be a probability in [0,1]")
    if evidence <= 0:
        raise ValueError("evidence P(D) must be positive")
    posterior = likelihood * prior / evidence
    return {"posterior": posterior, "prior": prior, "likelihood": likelihood, "evidence": evidence}


def bayes_two_hypothesis(prior_H1, likelihood_D_given_H1, likelihood_D_given_H2):
    """Binary-hypothesis Bayes update where evidence is computed by total
    probability: P(D) = P(D|H1)P(H1) + P(D|H2)P(H2)."""
    prior_H2 = 1.0 - prior_H1
    evidence = likelihood_D_given_H1 * prior_H1 + likelihood_D_given_H2 * prior_H2
    posterior_H1 = likelihood_D_given_H1 * prior_H1 / evidence
    posterior_H2 = likelihood_D_given_H2 * prior_H2 / evidence
    return {"posterior_H1": posterior_H1, "posterior_H2": posterior_H2, "evidence": evidence}


def bayes_sympy():
    """Symbolic Bayes' theorem: P(H|D) = P(D|H)*P(H) / P(D)."""
    P_H, P_D, P_D_given_H, P_H_given_D = sp.symbols('P_H P_D P_D_given_H P_H_given_D', positive=True)
    return sp.Eq(P_H_given_D, P_D_given_H * P_H / P_D)


# -- Dirac delta ------------------------------------------------------------

def dirac_delta_as_gaussian_limit(x, a, sigma):
    """delta(x-a) = lim_{sigma->0} (1/(sigma*sqrt(2*pi))) * exp(-(x-a)^2/(2*sigma^2)).
    Returns the finite-sigma Gaussian approximation -- a real, integrable
    stand-in for the idealized delta, useful numerically."""
    x = np.asarray(x, float)
    return np.exp(-(x - a) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def sifting_property_numeric(f, a, x_range, sigma=1e-3, n=200001):
    """Numerically verify integral f(x)*delta(x-a) dx ~= f(a) using the
    Gaussian-limit approximation of delta on a fine grid."""
    x = np.linspace(x_range[0], x_range[1], n)
    delta_approx = dirac_delta_as_gaussian_limit(x, a, sigma)
    integrand = f(x) * delta_approx
    integral = np.trapezoid(integrand, x)
    return {"integral": integral, "f_at_a": f(a), "abs_error": abs(integral - f(a))}


def dirac_delta_sympy_sifting():
    """Symbolic sifting property via SymPy's DiracDelta + integrate."""
    x, a = sp.symbols('x a', real=True)
    f = sp.Function('f')
    expr = sp.Integral(f(x) * sp.DiracDelta(x - a), (x, -sp.oo, sp.oo))
    return sp.Eq(expr, f(a))


def dirac_comb(x, spacing, n_teeth, sigma=0.02):
    """Sum of Gaussian-approximated deltas at integer multiples of `spacing` --
    the Dirac comb, which underlies sampling theory (each tooth = one sample)."""
    x = np.asarray(x, float)
    total = np.zeros_like(x)
    for k in range(-n_teeth, n_teeth + 1):
        total += dirac_delta_as_gaussian_limit(x, k * spacing, sigma)
    return total


# -- Symmetry --------------------------------------------------------------

def classify_symmetry(f, x_range=(-5, 5), n=2001, tol=1e-8):
    """Classify f as even, odd, or neither by sampling f(x) vs f(-x) on a
    symmetric grid. Even: f(-x)=f(x). Odd: f(-x)=-f(x)."""
    x = np.linspace(0, x_range[1], n)  # only need x>=0 by symmetry of the test
    f_pos = f(x)
    f_neg = f(-x)
    even_err = np.max(np.abs(f_neg - f_pos))
    odd_err = np.max(np.abs(f_neg + f_pos))
    if even_err < tol:
        kind = "even"
    elif odd_err < tol:
        kind = "odd"
    else:
        kind = "neither"
    return {"kind": kind, "even_error": float(even_err), "odd_error": float(odd_err)}


def symmetry_integral_shortcut(f, L, kind=None):
    """Exploit symmetry to avoid computing the full integral:
      odd  function on [-L, L]  -> integral is exactly 0, no computation needed
      even function on [-L, L]  -> integral = 2 * integral over [0, L]
    If kind is None, classify_symmetry() determines it first."""
    if kind is None:
        kind = classify_symmetry(f, x_range=(-L, L))["kind"]
    if kind == "odd":
        return {"integral": 0.0, "kind": kind, "shortcut": "odd integrand -> exactly zero, no integration done"}
    x_half = np.linspace(0, L, 20001)
    half_integral = np.trapezoid(f(x_half), x_half)
    if kind == "even":
        return {"integral": 2 * half_integral, "kind": kind, "shortcut": "even -> 2x the [0,L] half-integral"}
    x_full = np.linspace(-L, L, 40001)
    full_integral = np.trapezoid(f(x_full), x_full)
    return {"integral": full_integral, "kind": kind, "shortcut": "no symmetry -> full integral computed"}


def symmetry_sympy_5():
    """Five symbolic equations linking Bayes, Dirac delta, and symmetry."""
    x, a, L = sp.symbols('x a L', real=True)
    P_H, P_D, P_D_given_H = sp.symbols('P_H P_D P_D_given_H', positive=True)
    f = sp.Function('f')

    return {
        "Bayes_theorem":
            sp.Eq(sp.Symbol('P_H_given_D'), P_D_given_H * P_H / P_D),
        "Dirac_sifting":
            sp.Eq(sp.Integral(f(x) * sp.DiracDelta(x - a), (x, -sp.oo, sp.oo)), f(a)),
        "Dirac_as_gaussian_limit":
            sp.Eq(sp.DiracDelta(x),
                  sp.Limit(sp.exp(-x**2 / (2*sp.Symbol('sigma', positive=True)**2)) /
                            (sp.Symbol('sigma', positive=True) * sp.sqrt(2*sp.pi)),
                            sp.Symbol('sigma', positive=True), 0, dir='+')),
        "Odd_integral_vanishes":
            sp.Eq(sp.Integral(f(x), (x, -L, L)), 0),  # holds when f is odd
        "Even_integral_halved":
            sp.Eq(sp.Integral(f(x), (x, -L, L)), 2*sp.Integral(f(x), (x, 0, L))),  # holds when f is even
    }


if __name__ == "__main__":
    print("=== Bayes: medical test example ===")
    res = bayes_theorem(prior=0.01, likelihood=0.95, evidence=0.0594)
    print(f"  P(disease|positive) = {res['posterior']:.4f}")

    print("\n=== Bayes two-hypothesis ===")
    res2 = bayes_two_hypothesis(prior_H1=0.5, likelihood_D_given_H1=0.9, likelihood_D_given_H2=0.2)
    print(f"  posterior_H1 = {res2['posterior_H1']:.4f}, posterior_H2 = {res2['posterior_H2']:.4f}")

    print("\n=== Dirac delta sifting property (numeric) ===")
    sift = sifting_property_numeric(lambda x: x**2 + 1, a=2.0, x_range=(-5, 5))
    print(f"  integral ~= {sift['integral']:.6f}, f(a) = {sift['f_at_a']:.6f}, error = {sift['abs_error']:.2e}")

    print("\n=== Symmetry classification ===")
    print("  x^2:", classify_symmetry(lambda x: x**2)["kind"])
    print("  x^3:", classify_symmetry(lambda x: x**3)["kind"])
    print("  x^2+x:", classify_symmetry(lambda x: x**2 + x)["kind"])

    print("\n=== Symmetry shortcut: integral of x^3 on [-3,3] ===")
    sc = symmetry_integral_shortcut(lambda x: x**3, L=3.0)
    print(f"  integral = {sc['integral']}, {sc['shortcut']}")

    print("\n=== SymPy 5 ===")
    for k, eq in symmetry_sympy_5().items():
        print(f"  {k}: {eq}")
