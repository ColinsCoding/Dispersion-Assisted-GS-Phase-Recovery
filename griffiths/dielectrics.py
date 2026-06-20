"""Electric fields in matter -- Griffiths Ch. 4.

Polarization and bound charge, the displacement field D, linear dielectrics,
atomic polarizability, the Clausius-Mossotti relation, and the two canonical
boundary-value problems (uniformly polarized sphere; dielectric sphere in a
uniform field) -- the latter solved with the same Legendre separation of
variables as Ch. 3. SymPy throughout; epsilon_0 symbolic.
"""

import sympy as sp

from .electrostatics import KE, eps0
from .vectors import CARTESIAN, _as_vec3, div


# ── polarization and bound charge ───────────────────────────────────
def bound_surface_charge(P, theta):
    """sigma_b = P . n-hat. For polarization P along z on a sphere, the outward
    normal makes angle theta with z, so sigma_b = P cos(theta)."""
    return sp.simplify(P * sp.cos(theta))


def bound_volume_charge(P_field, vars=CARTESIAN):
    """rho_b = -div P (zero for uniform polarization)."""
    return sp.simplify(-div(_as_vec3(P_field, "P"), vars))


def displacement_field(E, P):
    """D = eps0 E + P (vectors or scalars)."""
    return sp.simplify(eps0 * sp.Matrix(_as_vec3(E, "E")) + sp.Matrix(_as_vec3(P, "P")))


# ── atomic polarizability ───────────────────────────────────────────
def polarizability_sphere(a):
    """Griffiths 4.1: a conducting/uniform sphere of radius a has atomic
    polarizability alpha = 4 pi eps0 a^3 (so induced dipole p = alpha E)."""
    return sp.simplify(4 * sp.pi * eps0 * a**3)


def induced_dipole(alpha, E):
    """p = alpha E."""
    return sp.simplify(alpha * E)


# ── linear dielectrics ──────────────────────────────────────────────
def dielectric_constant(chi_e):
    """Relative permittivity eps_r = 1 + chi_e (so eps = eps0 eps_r)."""
    return sp.simplify(1 + chi_e)


def clausius_mossotti(N, alpha):
    """Solve (eps_r - 1)/(eps_r + 2) = N alpha / (3 eps0) for eps_r -- the link
    between microscopic polarizability alpha and the macroscopic constant."""
    eps_r = sp.Symbol("epsilon_r", positive=True)
    eqn = sp.Eq((eps_r - 1) / (eps_r + 2), N * alpha / (3 * eps0))
    return sp.simplify(sp.solve(eqn, eps_r)[0])


def capacitor_with_dielectric(eps_r, C0):
    """A dielectric slab filling a capacitor raises its capacitance: C = eps_r C0."""
    if hasattr(eps_r, "is_number") and eps_r.is_number and eps_r < 1:
        raise ValueError("eps_r must be >= 1 for a passive dielectric")
    return sp.simplify(eps_r * C0)


# ── boundary-value problem 1: uniformly polarized sphere ────────────
def uniformly_polarized_sphere(P):
    """A sphere with uniform polarization P (along z).

    Returns (E_inside, dipole_moment): the interior field is uniform,
    E_in = -P/(3 eps0); the exterior is a pure dipole with p = (4/3) pi R^3 P
    (returned per unit R^3 as the coefficient). The surface charge sigma_b =
    P cos(theta) is a pure l=1 (Legendre P_1) distribution.
    """
    E_in = sp.simplify(-P / (3 * eps0))
    return E_in


# ── boundary-value problem 2: dielectric sphere in a uniform field ──
def dielectric_sphere_in_field(eps_r, E0, R=None):
    """Linear dielectric sphere (relative permittivity eps_r) in a uniform field
    E0 z-hat. Solve Laplace with Legendre l=1 terms and the dielectric boundary
    conditions (V continuous, D_perp continuous).

    Returns dict: A (interior slope), B (exterior dipole coeff / R^3 if R given),
    E_in (uniform interior field), E_in_over_E0 (the screening factor 3/(eps_r+2)).
    """
    A, B = sp.symbols("A B")
    Rs = sp.Symbol("R", positive=True) if R is None else R
    th = sp.Symbol("theta")
    # V_out = (-E0 r + B/r^2) cos th ; V_in = A r cos th
    # BC1: V continuous at R
    bc1 = sp.Eq(-E0 * Rs + B / Rs**2, A * Rs)
    # BC2: eps0 dV_out/dr = eps0 eps_r dV_in/dr  at R  (no free surface charge)
    dVout = -E0 - 2 * B / Rs**3
    dVin = A
    bc2 = sp.Eq(dVout, eps_r * dVin)
    sol = sp.solve([bc1, bc2], [A, B], dict=True)[0]
    A_val, B_val = sp.simplify(sol[A]), sp.simplify(sol[B])
    E_in = sp.simplify(-A_val)                      # E_in = -dV_in/dz = -A
    return {
        "A": A_val,
        "B": B_val,
        "E_in": E_in,
        "E_in_over_E0": sp.simplify(E_in / E0),     # = 3/(eps_r + 2)
    }
