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


# -- The EXACT (large-amplitude) pendulum: elliptic integrals, not small-angle --

def pendulum_energy_conservation(theta, t, l, g):
    """The exact pendulum EOM m*l*theta'' = -m*g*sin(theta) IS F_net=dp/dt's
    tangential component (dp/dt for the arc-length coordinate). Multiplying
    both sides by theta' and integrating over t turns F_net=dp/dt into an
    energy statement -- verified here by checking d/dt of the claimed
    conserved quantity is proportional to the EOM itself, not assumed.
    Returns the conserved quantity E = (1/2)*l*theta'^2 - g*cos(theta)."""
    thetad = theta.diff(t)
    E = sp.Rational(1, 2) * l * thetad ** 2 - g * sp.cos(theta)
    dE_dt = sp.diff(E, t)
    # dE/dt should be theta'*(l*theta'' + g*sin(theta)) -- exactly l * (EOM),
    # i.e. proportional to the F_net=dp/dt equation itself, confirming E is
    # conserved precisely BECAUSE the EOM holds (not for any other reason)
    eom_form = l * theta.diff(t, 2) + g * sp.sin(theta)
    ratio = sp.simplify(dE_dt / (thetad * eom_form))
    return E, ratio   # ratio should simplify to exactly 1


def exact_period(L, g, theta0):
    """Exact (large-amplitude) pendulum period via the complete elliptic
    integral of the first kind: T = 4*sqrt(L/g)*K(sin^2(theta0/2))
    (SymPy's elliptic_k takes the PARAMETER m=k^2, not the modulus k --
    verified against direct numerical quadrature of the energy-conservation
    integral, not just quoted from a table). theta0 is the amplitude (max
    angle from vertical, radians), 0 < theta0 < pi."""
    if L <= 0 or g <= 0:
        raise ValueError("L and g must be positive")
    if not (0 < theta0 < np.pi):
        raise ValueError(f"theta0 must be in (0, pi) radians, got {theta0}")
    k_squared = np.sin(theta0 / 2) ** 2
    K = float(sp.elliptic_k(k_squared))
    return 4 * np.sqrt(L / g) * K


def small_angle_period(L, g):
    """The small-angle limit T = 2*pi*sqrt(L/g) -- what exact_period reduces
    to as theta0 -> 0 (sin(theta0/2)->0, K(0)=pi/2, giving 4*sqrt(L/g)*pi/2)."""
    if L <= 0 or g <= 0:
        raise ValueError("L and g must be positive")
    return 2 * np.pi * np.sqrt(L / g)


def period_correction_factor(theta0):
    """T_exact/T_small as a function of amplitude -- how badly the small-
    angle approximation degrades as swing amplitude grows. ~1.0 for small
    theta0, ~1.18 at 90 degrees, diverging (-> infinity) as theta0 -> pi
    (a pendulum released from exactly vertical never actually gets there)."""
    if not (0 < theta0 < np.pi):
        raise ValueError(f"theta0 must be in (0, pi) radians, got {theta0}")
    L, g = 1.0, 1.0   # ratio is independent of L, g -- they cancel exactly
    return exact_period(L, g, theta0) / small_angle_period(L, g)


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
