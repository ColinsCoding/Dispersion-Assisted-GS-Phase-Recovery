"""Time derivatives of a trajectory two ways: torch autograd vs finite differences.

A trajectory x(t) hides its velocity, acceleration, and jerk as successive time
derivatives. Two ways to get them:

  * **autograd** -- if you have x(t) as code, torch differentiates it *exactly*
    (chain rule, not approximation), to any order, via create_graph=True. This
    is the same machinery the fiber-ray tracer used for Hamilton's equations.
  * **finite differences** -- if you only have sampled data, you divide
    differences. Fine for clean data, but each derivative multiplies the noise
    spectrum by ~omega, so the 2nd/3rd derivative of noisy samples is garbage --
    the "wrong trajectory". (Causality/filtering is the cure: smooth first.)

SymPy (with init_printing in a notebook) supplies the analytic ground truth.
torch (py 3.12 here). Civilian education.
"""

import numpy as np


# example trajectories (torch-friendly: elementwise in t)
def traj_sin(t):
    import torch
    return torch.sin(2 * t)                      # v=2cos2t, a=-4sin2t, jerk=-8cos2t


def traj_mixed(t):
    import torch
    return torch.sin(2 * np.pi * t) + 0.5 * t**2


def time_derivatives(x_func, t, order=3):
    """Successive time derivatives [x, v, a, jerk, ...] of x_func at points t,
    by repeated torch autograd (exact). t is a 1-D array-like; returns numpy arrays."""
    import torch
    if order < 1:
        raise ValueError("order must be >= 1")
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float64).requires_grad_(True)
    x = x_func(tt)
    out = [x]
    cur = x
    for _ in range(order):
        (g,) = torch.autograd.grad(cur.sum(), tt, create_graph=True)
        out.append(g)
        cur = g
    return [d.detach().numpy() for d in out]


def analytic_derivatives(expr, t_sym, order=3):
    """Symbolic [x, x', x'', ...] via SymPy (the ground truth; init_printing to view)."""
    import sympy as sp
    return [sp.diff(expr, t_sym, k) for k in range(order + 1)]


def finite_difference(y, dt, order=1):
    """Repeated np.gradient: the data-only derivative. Amplifies noise each pass."""
    d = np.asarray(y, dtype=float)
    for _ in range(order):
        d = np.gradient(d, dt)
    return d


if __name__ == "__main__":
    import sympy as sp
    sp.init_printing()
    ts = sp.Symbol("t", real=True)
    print("analytic derivatives of sin(2t):")
    for k, e in enumerate(analytic_derivatives(sp.sin(2 * ts), ts)):
        print(f"  d^{k}/dt^{k}:", e)

    t = np.linspace(0, 2, 9)
    x, v, a, j = time_derivatives(traj_sin, t)
    print("\nautograd at t=1.0:  v=%.4f  a=%.4f  jerk=%.4f"
          % (np.interp(1.0, t, v), np.interp(1.0, t, a), np.interp(1.0, t, j)))
    print("analytic  at t=1.0:  v=%.4f  a=%.4f  jerk=%.4f"
          % (2*np.cos(2), -4*np.sin(2), -8*np.cos(2)))
