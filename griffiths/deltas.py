"""Dirac delta and step-function machinery for Griffiths Problems 1.45-1.46."""

import sympy as sp

from .vectors import x


def delta_integral(f, a, b, c=0, var=x):
    """Evaluate integral over [a, b] of f(var) * delta(var - c).

    Returns f(c) if c lies strictly inside (a, b), else 0 -- the
    "is the spike inside the window?" logic of Problem 1.45.
    Infinite limits are fine (use sympy.oo).
    """
    a, b = sp.sympify(a), sp.sympify(b)
    if sp.simplify(b - a).is_nonpositive:
        raise ValueError(f"need a < b, got a={a}, b={b}")
    f = sp.sympify(f)
    return sp.integrate(f * sp.DiracDelta(var - c), (var, a, b))


def delta_rescale(k, var=x):
    """The scaling identity delta(k*x) = delta(x)/|k| as a sympy Eq.

    Griffiths Problem 1.43; needed for 1.45(a) and (c).
    """
    k = sp.sympify(k)
    if k == 0:
        raise ValueError("k must be nonzero; delta(0*x) is not a distribution")
    return sp.Eq(sp.DiracDelta(k * var), sp.DiracDelta(var) / sp.Abs(k))


def step(var=x):
    """Griffiths' theta(x): 1 for x > 0, 0 for x <= 0 (Eq. 1.95)."""
    return sp.Heaviside(var, 0)


def d_step_dx(var=x):
    """d(theta)/dx -- equals delta(x), Problem 1.46(b)."""
    return sp.diff(step(var), var)


def x_ddx_delta(test=None, var=x):
    """Problem 1.46(a): show x * d/dx delta(x) = -delta(x).

    Both sides are distributions, so equality means: integrated against any
    test function f they give the same number.  Returns
    (lhs_integral, rhs_integral) for the supplied test function
    (a generic f(x) by default); the two must match.
    """
    f = sp.Function("f")(var) if test is None else sp.sympify(test)
    lhs = sp.integrate(f * var * sp.DiracDelta(var, 1), (var, -sp.oo, sp.oo))
    rhs = sp.integrate(f * (-sp.DiracDelta(var)), (var, -sp.oo, sp.oo))
    return sp.simplify(lhs), sp.simplify(rhs)
