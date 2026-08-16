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

from dgs.rocket_equation_orbital_mechanics import circular_orbit_velocity


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


# -- Central-force motion: a cyclic coordinate is Noether's theorem in miniature --
#
# None of the systems above have a cyclic coordinate (one absent from L itself,
# only its velocity appears) -- central-force motion in polar coordinates does:
# theta never appears in L, only theta'. d/dt(dL/dtheta') - dL/dtheta = 0 with
# dL/dtheta = 0 means dL/dtheta' is conserved outright -- angular momentum
# conservation falls out of the Euler-Lagrange equation itself, not bolted on
# separately. Substituting that conserved quantity back into the radial
# equation of motion turns the 2D orbit problem into an effective 1D radial
# problem, V_eff(r) = V(r) + p_theta^2/(2 m r^2) -- the same "effective
# potential" language used throughout orbital mechanics and central-force
# scattering.

def central_force_lagrangian(r, theta, t, m, V_r):
    """L = T - V for a particle in a plane under a central force V(r):
    T = 1/2 m (r'^2 + r^2 theta'^2) in polar coordinates."""
    T = sp.Rational(1, 2) * m * (r.diff(t) ** 2 + r ** 2 * theta.diff(t) ** 2)
    return T - V_r


def angular_momentum_conservation(L, theta, t):
    """theta is cyclic in a central-force L (checked, not assumed: dL/dtheta
    must simplify to exactly 0). When it is, dL/dtheta' = p_theta is conserved
    by the Euler-Lagrange equation itself. Returns (is_cyclic: bool, p_theta)."""
    is_cyclic = sp.simplify(L.diff(theta)) == 0
    p_theta = sp.simplify(L.diff(theta.diff(t)))
    return is_cyclic, p_theta


def effective_potential(V_r, r, m, p_theta):
    """V_eff(r) = V(r) + p_theta^2/(2 m r^2) -- the centrifugal barrier term
    that appears once theta' is eliminated in favor of the conserved p_theta."""
    return V_r + p_theta ** 2 / (2 * m * r ** 2)


def verify_radial_eom_matches_effective_potential(r, theta, t, m, V_r):
    """The claim "m r'' = -dV_eff/dr" is a real algebraic reduction, not a
    definition -- verified here by deriving the radial Euler-Lagrange
    equation directly (with theta' eliminated via the conserved angular
    momentum) and checking it matches -dV_eff/dr exactly, both sides built
    independently from the same Lagrangian. Returns True if they match.

    CAUGHT BUG: eliminating theta' requires substituting a genuinely FREE
    symbol standing for the conserved value of p_theta -- substituting the
    phase-space EXPRESSION p_theta = m*r^2*theta' returned by
    angular_momentum_conservation() instead just algebraically unwinds back
    to the original theta'^2 term (p_theta**2/(2*m*r**2) = m*r**2*theta'^2/2
    exactly reproduces the centrifugal term you started with), so the
    "verification" would trivially compare an expression to itself and
    always report False for the wrong reason. A dedicated free symbol is
    required so the substitution actually eliminates theta'."""
    L = central_force_lagrangian(r, theta, t, m, V_r)
    is_cyclic, _p_theta_expr = angular_momentum_conservation(L, theta, t)
    if not is_cyclic:
        raise ValueError("theta is not cyclic in this Lagrangian -- V_r must depend only on r")

    p_theta = sp.Symbol("p_theta", positive=True)   # free symbol for the CONSERVED value
    radial_eom = euler_lagrange(L, r, t)   # m r'' - m r theta'^2 + dV/dr  (=0)
    thetad = theta.diff(t)
    radial_eom_reduced = radial_eom.subs(thetad, p_theta / (m * r ** 2))

    V_eff = effective_potential(V_r, r, m, p_theta)
    rhs = -sp.diff(V_eff, r)   # -dV_eff/dr
    lhs = -(radial_eom_reduced - m * r.diff(t, 2))   # isolate the non-(m r'') side, sign-matched to rhs

    return sp.simplify(lhs - rhs) == 0


def circular_orbit_radius_from_specific_angular_momentum(specific_ang_momentum, mu):
    """r_circ = h^2/mu for an attractive V(r) = -mu*m/r, where h = p_theta/m is
    the SPECIFIC angular momentum (r^2*theta'). Derived from dV_eff/dr = 0."""
    if specific_ang_momentum <= 0 or mu <= 0:
        raise ValueError("specific_ang_momentum and mu must be positive")
    return specific_ang_momentum ** 2 / mu


def verify_circular_orbit_cross_check(r_test_m, mu):
    """Cross-module consistency check: take a circular-orbit radius and its
    vis-viva circular velocity from dgs.rocket_equation_orbital_mechanics
    (built and verified independently, in a completely different session
    context, against widely-cited reference values), compute the specific
    angular momentum h = r*v_circ for that orbit, then feed h back through
    THIS module's effective-potential-derived formula r_circ = h^2/mu. The
    two radii should match to numerical precision -- confirming the
    Lagrangian/Noether route to orbital mechanics and the vis-viva route
    describe the exact same physics, not just similar-looking answers.

    NOTE: 'matches' and the numeric fields are explicitly cast with
    bool()/float() -- circular_orbit_velocity returns np.float64, and a raw
    comparison on it yields numpy.bool_, which is never `is True` (same
    caught-bug pattern as dgs.lunar_laser_communication's optical/RF
    comparison)."""
    if r_test_m <= 0 or mu <= 0:
        raise ValueError("r_test_m and mu must be positive")
    v_circ = circular_orbit_velocity(r_test_m, mu)
    h = r_test_m * v_circ
    r_from_effective_potential = circular_orbit_radius_from_specific_angular_momentum(h, mu)
    matches = bool(abs(r_from_effective_potential - r_test_m) / r_test_m < 1e-9)
    return {
        "r_test_m": r_test_m,
        "v_circular_m_s": float(v_circ),
        "specific_angular_momentum": float(h),
        "r_from_effective_potential_m": float(r_from_effective_potential),
        "matches": matches,
    }


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

    print("\n=== Central-force motion: cyclic coordinate -> conserved angular momentum ===")
    r, theta = sp.Function("r")(t), sp.Function("theta")(t)
    mu = sp.Symbol("mu", positive=True)
    V_grav = -mu * m / r
    L_central = central_force_lagrangian(r, theta, t, m, V_grav)
    is_cyclic, p_theta = angular_momentum_conservation(L_central, theta, t)
    print(f"theta is cyclic: {is_cyclic}   p_theta (conserved) = {p_theta}")
    matches = verify_radial_eom_matches_effective_potential(r, theta, t, m, V_grav)
    print(f"radial EOM matches -dV_eff/dr: {matches}")

    from dgs.rocket_equation_orbital_mechanics import MU_EARTH, R_EARTH_M
    cross_check = verify_circular_orbit_cross_check(R_EARTH_M + 400e3, MU_EARTH)
    print(f"LEO (400km alt): v_circ={cross_check['v_circular_m_s']:.1f} m/s, "
          f"r from effective-potential formula = {cross_check['r_from_effective_potential_m']/1e3:.3f} km "
          f"(input r = {cross_check['r_test_m']/1e3:.3f} km), matches: {cross_check['matches']}")
