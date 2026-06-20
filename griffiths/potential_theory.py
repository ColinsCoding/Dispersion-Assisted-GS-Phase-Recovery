"""Electric-potential engine for Griffiths Ch. 2.3-3.4 (a 'potential' problem set).

Direct potential calculations (sphere, disk), electrostatic energy, the dipole
potential/field, multipole moments, the Laplace mean-value property, and
separation of variables in Cartesian (Fourier) and spherical (Legendre)
geometry. Builds on griffiths.electrostatics; epsilon_0 symbolic, KE = 1/4 pi eps0.
"""

import sympy as sp

from .electrostatics import KE, eps0
from .vectors import CARTESIAN

# ── direct potentials ───────────────────────────────────────────────
def sphere_potential(Q, r, R):
    """G2.21: V(r) of a uniformly charged solid sphere (charge Q, radius R).

    Outside: k Q / r. Inside: (k Q / 2R)(3 - r^2/R^2). Continuous at r = R.
    """
    outside = KE * Q / r
    inside = KE * Q / (2 * R) * (3 - r**2 / R**2)
    return sp.Piecewise((inside, r < R), (outside, True))


def disk_potential(sigma, R, Zc):
    """G2.27: on-axis potential a height Zc>0 above a uniformly charged disk.

    V = (sigma / 2 eps0)(sqrt(R^2 + Zc^2) - Zc), by integrating rings.
    """
    s = sp.Symbol("s", positive=True)
    integrand = KE * sigma * 2 * sp.pi * s / sp.sqrt(s**2 + Zc**2)
    return sp.simplify(sp.integrate(integrand, (s, 0, R)))


def energy_uniform_sphere(Q, R):
    """G2.33: electrostatic energy of a uniform solid sphere, W = (3/5) k Q^2 / R,

    from W = (eps0/2) int E^2 dtau over all space (inside + outside fields).
    """
    r = sp.Symbol("r", positive=True)
    E_in = KE * Q * r / R**3
    E_out = KE * Q / r**2
    W = (eps0 / 2) * (sp.integrate(E_in**2 * 4 * sp.pi * r**2, (r, 0, R))
                      + sp.integrate(E_out**2 * 4 * sp.pi * r**2, (r, R, sp.oo)))
    return sp.simplify(W)


# ── dipole ──────────────────────────────────────────────────────────
def grad_spherical(V, r, theta, phi=None):
    """Gradient of V(r, theta[, phi]) in spherical coordinates, as (E_r, E_th, E_ph)
    with the electrostatic sign E = -grad V."""
    E_r = -sp.diff(V, r)
    E_th = -sp.diff(V, theta) / r
    E_ph = 0 if phi is None else -sp.diff(V, phi) / (r * sp.sin(theta))
    return sp.simplify(E_r), sp.simplify(E_th), sp.simplify(E_ph)


def dipole_potential(p, r, theta):
    """Pure dipole potential V = k p cos(theta) / r^2."""
    return sp.simplify(KE * p * sp.cos(theta) / r**2)


def dipole_field(p, r, theta):
    """Dipole field (E_r, E_theta) = (2 k p cos/r^3, k p sin/r^3), from -grad V."""
    E_r, E_th, _ = grad_spherical(dipole_potential(p, r, theta), r, theta)
    return E_r, E_th


# ── multipole moments of a discrete charge distribution ─────────────
def multipole_moments(charges, positions):
    """Monopole (scalar), dipole (3-vector), and quadrupole (3x3, traceless)
    moments of point charges q_i at positions r_i.

    Q_ij = sum q_i (3 r_i r_j - |r_i|^2 delta_ij).
    """
    if len(charges) != len(positions):
        raise ValueError("charges and positions must have equal length")
    charges = [sp.sympify(q) for q in charges]
    pos = [sp.Matrix(p) for p in positions]
    monopole = sp.simplify(sum(charges))
    dipole = sp.zeros(3, 1)
    for q, rv in zip(charges, pos):
        dipole += q * rv
    quad = sp.zeros(3, 3)
    for q, rv in zip(charges, pos):
        r2 = (rv.T * rv)[0]
        for i in range(3):
            for j in range(3):
                quad[i, j] += q * (3 * rv[i] * rv[j] - (r2 if i == j else 0))
    return sp.simplify(monopole), sp.simplify(dipole), sp.simplify(quad)


# ── Laplace's equation ──────────────────────────────────────────────
def laplacian_cartesian(V, vars=CARTESIAN):
    """Cartesian Laplacian of V."""
    return sp.simplify(sum(sp.diff(V, v, 2) for v in vars))


def is_harmonic(V, vars=CARTESIAN):
    """True iff V satisfies Laplace's equation (Laplacian = 0)."""
    return laplacian_cartesian(V, vars) == 0


def mean_value_check(V, center, radius, vars=CARTESIAN):
    """G3.3 mean-value property: for a harmonic V, the average over a sphere equals
    the centre value. Returns (V_center, surface_average); they match iff V is harmonic.
    Done on a circle (2-D) for tractable symbolic averaging.
    """
    x, y = vars[0], vars[1]
    x0, y0 = center
    th = sp.Symbol("vartheta", real=True)
    on_circle = V.subs({x: x0 + radius * sp.cos(th), y: y0 + radius * sp.sin(th)})
    avg = sp.simplify(sp.integrate(on_circle, (th, 0, 2 * sp.pi)) / (2 * sp.pi))
    return sp.simplify(V.subs({x: x0, y: y0})), avg


# ── separation of variables: Cartesian (Fourier) ────────────────────
def cartesian_slot(V0, a, n_terms, x, y):
    """Grounded slot/pipe (Griffiths Ex. 3.3): plates at y=0 and y=a grounded, the
    strip at x=0 held at V0, V -> 0 as x -> +inf.

    V(x,y) = (4 V0 / pi) sum_{n odd} (1/n) e^{-n pi x / a} sin(n pi y / a).
    Returns the truncated sum over the first `n_terms` odd harmonics.
    """
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    V = sp.Integer(0)
    for k in range(n_terms):
        n = 2 * k + 1
        V += (sp.Rational(1, n)) * sp.exp(-n * sp.pi * x / a) * sp.sin(n * sp.pi * y / a)
    # do NOT sp.simplify a Fourier series -- it tries to factor the whole sum
    return 4 * V0 / sp.pi * V


# ── separation of variables: spherical (Legendre) ───────────────────
def legendre_coefficients(surface_potential, theta, n_max):
    """Project a surface potential V(theta) onto Legendre polynomials:
    c_l = (2l+1)/2 int_0^pi V(theta) P_l(cos theta) sin(theta) dtheta."""
    coeffs = []
    for l in range(n_max + 1):
        Pl = sp.legendre(l, sp.cos(theta))
        cl = sp.Rational(2 * l + 1, 2) * sp.integrate(
            surface_potential * Pl * sp.sin(theta), (theta, 0, sp.pi))
        coeffs.append(sp.simplify(cl))
    return coeffs


def legendre_sphere(surface_potential, R, r, theta, n_max, inside=True):
    """Solve Laplace inside/outside a sphere whose surface is held at
    V(R, theta) = surface_potential(theta), via spherical separation of variables.

    Inside:  V = sum c_l (r/R)^l P_l(cos theta)
    Outside: V = sum c_l (R/r)^(l+1) P_l(cos theta)
    Both reduce to the surface potential at r = R.
    """
    coeffs = legendre_coefficients(surface_potential, theta, n_max)
    V = sp.Integer(0)
    for l, cl in enumerate(coeffs):
        Pl = sp.legendre(l, sp.cos(theta))
        radial = (r / R)**l if inside else (R / r)**(l + 1)
        V += cl * radial * Pl
    return sp.simplify(V)
