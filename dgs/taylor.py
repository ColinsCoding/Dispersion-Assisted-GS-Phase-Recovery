"""Taylor / power series -- and where they hide in this repo.

A Taylor series rebuilds a function near a point from its derivatives:

    f(x) = sum_{k>=0}  f^{(k)}(x0)/k!  (x - x0)^k.

This one idea is everywhere here:
  * Euler's formula  e^{ix} = cos x + i sin x  is two Taylor series interleaved,
  * the small-angle / paraxial approximation is a truncated Taylor series,
  * and the **dispersion** the receiver inverts is a Taylor expansion of the
    propagation constant beta(omega): the quadratic term beta_2 (group-velocity
    dispersion) is what makes H(f)=exp(i pi D f^2).

SymPy for exact coefficients; truncation error shown numerically. Education.
"""

import sympy as sp


def taylor_coefficients(expr, var, x0, n):
    """The coefficients a_k = f^{(k)}(x0)/k! for k = 0..n (the power-series weights)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return [sp.diff(expr, var, k).subs(var, x0) / sp.factorial(k) for k in range(n + 1)]


def taylor_series(expr, var, x0, n):
    """The degree-n Taylor polynomial sum a_k (x-x0)^k, built from the coefficients."""
    coeffs = taylor_coefficients(expr, var, x0, n)
    return sum(c * (var - x0)**k for k, c in enumerate(coeffs))


def truncation_error(expr, var, x0, n, x_val):
    """|f(x_val) - (degree-n Taylor polynomial)(x_val)|: how wrong the truncation is."""
    approx = taylor_series(expr, var, x0, n).subs(var, x_val)
    exact = expr.subs(var, x_val)
    return float(sp.Abs(exact - approx))


def dispersion_taylor(beta_expr, omega, omega0, n=3):
    """GVD expansion: the physical coefficients beta_k = d^k beta/d omega^k at omega0.

    beta_0 = phase constant, beta_1 = 1/group velocity, beta_2 = group-velocity
    dispersion (the term that reshapes the pulse and defines H(f)=exp(i pi D f^2)).
    Note these are NOT divided by k! -- they are the named beta_k of fibre optics.
    """
    return [sp.diff(beta_expr, omega, k).subs(omega, omega0) for k in range(n + 1)]


if __name__ == "__main__":
    sp.init_printing()
    x = sp.Symbol("x")
    print("e^x  coefficients:", taylor_coefficients(sp.exp(x), x, 0, 5))
    print("sin x to order 5 :", taylor_series(sp.sin(x), x, 0, 5))
    # Euler: e^{ix} = cos x + i sin x, as power series
    lhs = taylor_series(sp.exp(sp.I * x), x, 0, 6)
    rhs = taylor_series(sp.cos(x), x, 0, 6) + sp.I * taylor_series(sp.sin(x), x, 0, 6)
    print("Euler holds as series:", sp.simplify(lhs - rhs) == 0)
