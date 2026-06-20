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


def delta_composition(g, var=x):
    """The composition rule  delta(g(x)) = sum_i delta(x - x_i) / |g'(x_i)|.

    The delta of a function fires at every simple zero x_i of g, each weighted by
    one over the *absolute value* of the slope there -- a steep crossing gives a
    small spike, a shallow one a big spike. delta_rescale (delta(kx)=delta(x)/|k|)
    is the special case g = k*x. The rule needs g'(x_i) != 0 at each zero.

    Returns (expr, real_roots). In physics this is how delta(energy) collapses in
    Fermi's golden rule / density of states: the delta selects the states where a
    process is allowed and the 1/|g'| is the Jacobian onto that surface.
    """
    g = sp.sympify(g)
    gp = sp.diff(g, var)
    terms, real_roots = [], []
    for r in sp.solve(sp.Eq(g, 0), var):
        if sp.im(r) != 0:                       # keep real crossings only
            continue
        slope = sp.simplify(gp.subs(var, r))
        if slope == 0:
            raise ValueError(f"g'({r}) = 0: delta(g) is singular at a tangent zero")
        terms.append(sp.DiracDelta(var - r) / sp.Abs(slope))
        real_roots.append(r)
    return sp.Add(*terms), real_roots


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
