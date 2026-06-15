"""Magnetic fields in matter -- Griffiths Ch. 6 (the dual of Ch. 4 dielectrics).

Magnetization M, bound currents (surface K_b = M x n-hat, volume J_b = curl M),
the auxiliary field H = B/mu0 - M with curl H = J_free, linear media
mu_r = 1 + chi_m, and the two boundary-value problems (uniformly magnetized
sphere; linear sphere in a uniform field). SymPy throughout; mu0 symbolic.
Mirror of griffiths.dielectrics with P<->M, E<->H, eps<->mu.
"""

import sympy as sp

from .magnetostatics import mu0
from .vectors import CARTESIAN, _as_vec3, curl


# ── magnetization and bound currents ────────────────────────────────
def bound_surface_current(M, theta):
    """|K_b| = |M x n-hat|. For M along z on a sphere, the normal makes angle
    theta with z, so |K_b| = M sin(theta) (circulating azimuthally)."""
    return sp.simplify(M * sp.sin(theta))


def bound_volume_current(M_field, vars=CARTESIAN):
    """J_b = curl M (zero for uniform magnetization)."""
    return sp.simplify(curl(_as_vec3(M_field, "M"), vars))


def auxiliary_field_H(B, M):
    """H = B/mu0 - M (the field whose curl is the *free* current only)."""
    return sp.simplify(sp.Matrix(_as_vec3(B, "B")) / mu0 - sp.Matrix(_as_vec3(M, "M")))


# ── linear media ────────────────────────────────────────────────────
def permeability_constant(chi_m):
    """Relative permeability mu_r = 1 + chi_m (so mu = mu0 mu_r, B = mu H)."""
    return sp.simplify(1 + chi_m)


def classify_magnetic(chi_m):
    """Diamagnet (chi_m < 0), paramagnet (small chi_m > 0), or ferromagnet
    (chi_m >> 1, nonlinear/hysteretic)."""
    chi_m = sp.sympify(chi_m)
    if not chi_m.is_number:
        raise ValueError("chi_m must be numeric to classify")
    if chi_m < 0:
        return "diamagnet"
    if chi_m < 1:
        return "paramagnet"
    return "ferromagnet (linear model only approximate -- real ones hysteresis)"


def curie_law(C, T):
    """Curie law for a paramagnet: chi_m = C / T (susceptibility falls with
    temperature as thermal agitation fights alignment)."""
    if hasattr(T, "is_number") and T.is_number and T <= 0:
        raise ValueError("temperature T must be > 0")
    return sp.simplify(C / T)


# ── BVP 1: uniformly magnetized sphere ──────────────────────────────
def uniformly_magnetized_sphere(M):
    """Sphere with uniform magnetization M (along z). Interior fields are uniform:

        B_in = (2/3) mu0 M,   H_in = B_in/mu0 - M = -M/3.

    Exterior is a pure magnetic dipole, m = (4/3) pi R^3 M. (Dual of the
    uniformly polarized sphere: H plays E's role, H_in = -M/3 <-> E_in = -P/3eps0.)
    """
    B_in = sp.simplify(sp.Rational(2, 3) * mu0 * M)
    H_in = sp.simplify(B_in / mu0 - M)
    return {"B_in": B_in, "H_in": H_in}


# ── BVP 2: linear magnetic sphere in a uniform field ────────────────
def magnetizable_sphere_in_field(mu_r, B0, R=None):
    """Linear sphere (relative permeability mu_r) in a uniform field B0 z-hat.

    Solve Laplace for the magnetic scalar potential W (H = -grad W) with W
    continuous (tangential H) and B_perp continuous. The H field is reduced
    exactly like a dielectric's E; the B field is *enhanced* by mu_r:

        H_in/H0 = 3/(mu_r + 2),    B_in/B0 = 3 mu_r/(mu_r + 2).

    mu_r -> inf concentrates flux (B_in -> 3 B0) while expelling H (H_in -> 0).
    """
    A, Bc = sp.symbols("A B")
    Rs = sp.Symbol("R", positive=True) if R is None else R
    H0 = sp.Symbol("H0", positive=True)               # B0 = mu0 H0 outside
    bc1 = sp.Eq(-H0 * Rs + Bc / Rs**2, A * Rs)        # W continuous (tangential H)
    bc2 = sp.Eq(-H0 - 2 * Bc / Rs**3, mu_r * A)       # B_perp continuous
    sol = sp.solve([bc1, bc2], [A, Bc], dict=True)[0]
    H_in = sp.simplify(-sol[A])                       # H_in = -dW_in/dz = -A
    H_in_over_H0 = sp.simplify(H_in / H0)
    B_in_over_B0 = sp.simplify(mu_r * H_in_over_H0)   # B_in = mu_r mu0 H_in, B0 = mu0 H0
    return {
        "H_in_over_H0": H_in_over_H0,
        "B_in_over_B0": B_in_over_B0,
        "B_in": sp.simplify(B_in_over_B0 * B0),
    }
