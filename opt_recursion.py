"""Optimization as a deterministic recursion: the symbolic side.

Gradient descent is a fixed-point iteration x_{n+1} = g(x_n) with
g(x) = x - eta f'(x). For a quadratic objective the recurrence is linear and
SymPy solves it in closed form; in general the fixed points and their stability
(|g'(x*)| < 1) decide whether the optimizer converges, oscillates, or diverges
into chaos. Standalone (the dynamical-systems view of optimization).
"""

import numpy as np
import sympy as sp


def gd_closed_form(curvature, x0, eta):
    """Closed form of gradient descent on f = (a/2) x^2, i.e. the linear recurrence
    x_{n+1} = (1 - eta*a) x_n, solved symbolically via rsolve.

    Returns x_n as a function of the symbolic index n (and the contraction factor).
    """
    a, e, X0 = sp.sympify(curvature), sp.sympify(eta), sp.sympify(x0)
    n = sp.Symbol("n", integer=True, nonnegative=True)
    x = sp.Function("x")
    rec = sp.Eq(x(n + 1), x(n) - e * a * x(n))
    sol = sp.rsolve(rec, x(n), {x(0): X0})
    return sp.simplify(sol), sp.simplify(1 - e * a)


def fixed_point_stability(f, x, eta):
    """Fixed points of GD on f and the multiplier g'(x*) = 1 - eta f''(x*).
    A fixed point is stable (locally convergent) iff |g'(x*)| < 1.

    Returns list of (x_star, multiplier, stable_condition).
    """
    f = sp.sympify(f)
    g = x - eta * sp.diff(f, x)                  # the iteration map
    stars = sp.solve(sp.Eq(g, x), x)             # g(x*)=x*  <=>  f'(x*)=0
    out = []
    for xs in stars:
        mult = sp.simplify(sp.diff(g, x).subs(x, xs))
        out.append((xs, mult, sp.simplify(sp.Abs(mult) < 1)))
    return out


def iterate_map(g_func, x0, n_steps):
    """Iterate x_{n+1} = g_func(x_n) deterministically; return the trajectory."""
    xs = np.empty(n_steps + 1)
    xs[0] = x0
    for i in range(n_steps):
        xs[i + 1] = g_func(xs[i])
    return xs


def gd_step(f_prime, eta):
    """Build the gradient-descent map g(x) = x - eta f'(x) as a plain callable."""
    if eta <= 0:
        raise ValueError("eta must be > 0")
    return lambda x: x - eta * f_prime(x)


def bifurcation(f_prime, etas, x0=0.7, n_warmup=400, n_keep=200):
    """For each step size eta, iterate GD and keep the attractor (last n_keep
    iterates). Returns (eta_column, x_column) suitable for a scatter bifurcation
    plot -- the optimizer's route to chaos as the step grows."""
    E, X = [], []
    for eta in etas:
        g = gd_step(f_prime, eta)
        x = x0
        for _ in range(n_warmup):
            x = g(x)
            if not np.isfinite(x):
                break
        if np.isfinite(x):
            for _ in range(n_keep):
                x = g(x)
                if not np.isfinite(x):
                    break
                E.append(eta)
                X.append(x)
    return np.array(E), np.array(X)
