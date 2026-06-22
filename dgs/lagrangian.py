"""Analytical mechanics -- the Lagrangian, and small oscillations as eigenvalues.

Instead of summing forces, write one scalar: the Lagrangian L = T - V (kinetic minus
potential energy). The principle of least action then gives the equation of motion
through the Euler-Lagrange equation, for every generalized coordinate q:

    d/dt ( dL/dq' ) - dL/dq = 0.

That single recipe produces the pendulum, the orbit, the coupled chain -- no
free-body diagrams. And near a stable equilibrium, expanding T = 1/2 q'^T M q' and
V = 1/2 q^T K q turns the Euler-Lagrange equations into the generalized eigenvalue
problem K v = omega^2 M v: the NORMAL MODES (the same eigenproblem as
dgs.eigen_modes). SymPy for the symbolic EOM, NumPy for the modes. Education.
"""

import numpy as np
import sympy as sp


def euler_lagrange(L, q, t):
    """The Euler-Lagrange equation for coordinate q(t) given Lagrangian L:
        d/dt(dL/dq') - dL/dq  ( = 0 ).
    Returns the left-hand expression (set it to zero for the equation of motion)."""
    qd = q.diff(t)
    return sp.simplify(sp.diff(L.diff(qd), t) - L.diff(q))


def equation_of_motion(L, q, t):
    """Solve the Euler-Lagrange equation for the acceleration q'' -- the EOM in the
    form q'' = f(q, q', t)."""
    eom = euler_lagrange(L, q, t)
    qdd = q.diff(t, 2)
    return sp.simplify(sp.solve(eom, qdd)[0])


def pendulum_lagrangian(theta, t, m, l, g):
    """Lagrangian of a simple pendulum: L = 1/2 m l^2 theta'^2 - m g l (1 - cos theta)."""
    return sp.Rational(1, 2) * m * l**2 * theta.diff(t)**2 - m * g * l * (1 - sp.cos(theta))


def oscillator_lagrangian(x, t, m, k):
    """Lagrangian of a mass on a spring: L = 1/2 m x'^2 - 1/2 k x^2."""
    return sp.Rational(1, 2) * m * x.diff(t)**2 - sp.Rational(1, 2) * k * x**2


def normal_mode_frequencies(M, K):
    """Small-oscillation normal-mode angular frequencies: sqrt of the eigenvalues of
    M^{-1} K, where T = 1/2 q'^T M q' and V = 1/2 q^T K q. This is the generalized
    eigenproblem K v = omega^2 M v -- the same one dgs.eigen_modes solves with eigh."""
    M = np.asarray(M, float)
    K = np.asarray(K, float)
    eig = np.linalg.eigvals(np.linalg.solve(M, K))
    return np.sqrt(np.sort(eig.real))


def coupled_oscillator_KM(m, k, k_c):
    """Two equal masses m, each tied to a wall by spring k and to each other by k_c.
    Returns (K, M): V = 1/2 q^T K q, T = 1/2 q'^T M q'. Normal modes come out at
    omega = sqrt(k/m) (in-phase) and sqrt((k+2 k_c)/m) (out-of-phase)."""
    K = np.array([[k + k_c, -k_c], [-k_c, k + k_c]])
    M = m * np.eye(2)
    return K, M


if __name__ == "__main__":
    t = sp.Symbol("t")
    m, l, g, k = sp.symbols("m l g k", positive=True)

    x = sp.Function("x")(t)
    print("oscillator EOM  x'' =", equation_of_motion(oscillator_lagrangian(x, t, m, k), x, t))

    th = sp.Function("theta")(t)
    print("pendulum   EOM  th'' =", equation_of_motion(pendulum_lagrangian(th, t, m, l, g), th, t))
    print("small-angle frequency omega =", sp.sqrt(g / l), " (period 2 pi sqrt(l/g))")

    K, M = coupled_oscillator_KM(m=1.0, k=4.0, k_c=1.5)
    w = normal_mode_frequencies(M, K)
    print(f"coupled modes: omega = {np.round(w,4)}  (sqrt(k/m)={np.sqrt(4):.3f}, "
          f"sqrt((k+2kc)/m)={np.sqrt(4+3):.3f})")
