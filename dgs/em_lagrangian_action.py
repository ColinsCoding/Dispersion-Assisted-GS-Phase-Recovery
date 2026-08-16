"""em_lagrangian_action.py -- derive Maxwell's equations FROM the
electromagnetic action integral S = INT L d^4x with SymPy, decomposing the
Lagrangian density into recognizable pieces at each step instead of quoting
the tensor formalism's end results. Four separate algebraic claims are
CHECKED, not assumed:

  1. F_{mu nu} = d_mu A_nu - d_nu A_mu, built purely from the four-potential,
     reproduces the textbook E = -grad(V) - dA/dt, B = curl(A) component by
     component (F_0i = E_i/c, F_ij = -eps_ijk B_k).
  2. L_free = -1/(4 mu0) F_{mu nu} F^{mu nu} algebraically decomposes into
     the familiar field-energy form (eps0/2)E^2 - B^2/(2 mu0).
  3. The canonical momentum density pi^{mu nu} = dL_free/d(d_mu A_nu) --
     computed by treating each of the 16 partial derivatives as an
     INDEPENDENT variable, the field-theory analogue of dL/d(dq/dt) in
     point mechanics -- equals -F^{mu nu}/mu0.
  4. The resulting Euler-Lagrange field equation d_mu F^{mu nu} = mu0 J^nu
     reduces, component by component, to Gauss's law (nu=0) and the
     Ampere-Maxwell law (nu=1,2,3); separately, the Bianchi identity
     d_lam F_{mu nu} + d_mu F_{nu lam} + d_nu F_{lam mu} = 0 holds
     IDENTICALLY for F = dA (no field equation needed), which is what
     gives Faraday's law and "no magnetic monopoles" for free.

Metric convention: eta = diag(1,-1,-1,-1) (Griffiths' mostly-minus
signature), coordinates (x0,x1,x2,x3) = (ct,x,y,z).
"""

from __future__ import annotations
from itertools import combinations
import sympy as sp

# ── 0. Spacetime setup, shared by every function below ──────────────────────

X0, X1, X2, X3 = sp.symbols("x0 x1 x2 x3", real=True)   # (ct, x, y, z)
COORDS = [X0, X1, X2, X3]
ETA = sp.diag(1, -1, -1, -1)

EPS0, MU0, C = sp.symbols("epsilon_0 mu_0 c", positive=True)
MU0_FROM_EPS0 = 1 / (EPS0 * C**2)   # mu0 = 1/(eps0 c^2)


def four_potential_symbols():
    """Generic scalar/vector potential V(x), A_x(x), A_y(x), A_z(x) as SymPy
    Functions of all four coordinates -- the dynamical field this module
    derives Maxwell's equations FROM, not one it assumes."""
    V = sp.Function("V")(*COORDS)
    Ax = sp.Function("A_x")(*COORDS)
    Ay = sp.Function("A_y")(*COORDS)
    Az = sp.Function("A_z")(*COORDS)
    return V, Ax, Ay, Az


def four_current_symbols():
    """Generic charge/current density rho(x), J_x(x), J_y(x), J_z(x)."""
    rho = sp.Function("rho")(*COORDS)
    Jx = sp.Function("J_x")(*COORDS)
    Jy = sp.Function("J_y")(*COORDS)
    Jz = sp.Function("J_z")(*COORDS)
    return rho, Jx, Jy, Jz


def contravariant_and_covariant_potential(V, Ax, Ay, Az):
    """A^mu = (V/c, Ax, Ay, Az);  A_mu = eta_{mu mu} A^mu (diagonal metric)."""
    A_up = [V / C, Ax, Ay, Az]
    A_lo = [ETA[mu, mu] * A_up[mu] for mu in range(4)]
    return A_up, A_lo


# ── 1. Field strength tensor F_{mu nu} = d_mu A_nu - d_nu A_mu ──────────────

def field_strength_tensor(A_lo):
    """Build F_{mu nu} (covariant) and F^{mu nu} (contravariant, via the
    diagonal metric) directly from the four-potential's covariant
    components -- this IS the field tensor, not a separate postulate."""
    F_lo = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            F_lo[mu, nu] = sp.diff(A_lo[nu], COORDS[mu]) - sp.diff(A_lo[mu], COORDS[nu])
    F_up = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            F_up[mu, nu] = ETA[mu, mu] * ETA[nu, nu] * F_lo[mu, nu]
    return F_lo, F_up


def E_and_B_from_potentials(V, Ax, Ay, Az):
    """The textbook E = -grad(V) - dA/dt, B = curl(A). d/dt = c*d/dx0 since
    x0 = c*t."""
    Ex = -sp.diff(V, X1) - C * sp.diff(Ax, X0)
    Ey = -sp.diff(V, X2) - C * sp.diff(Ay, X0)
    Ez = -sp.diff(V, X3) - C * sp.diff(Az, X0)
    Bx = sp.diff(Az, X2) - sp.diff(Ay, X3)
    By = sp.diff(Ax, X3) - sp.diff(Az, X1)
    Bz = sp.diff(Ay, X1) - sp.diff(Ax, X2)
    return (Ex, Ey, Ez), (Bx, By, Bz)


def verify_field_tensor_matches_E_B(F_lo, E, B) -> bool:
    """CHECKED, not assumed: F_{0i} = E_i/c and F_{ij} = -eps_{ijk} B_k must
    fall out of the SAME derivative definition used to build F_lo, matched
    component by component against the independently-defined E, B above.
    Raises AssertionError naming the first mismatched component."""
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    checks = {
        "F_01 - Ex/c": F_lo[0, 1] - Ex / C,
        "F_02 - Ey/c": F_lo[0, 2] - Ey / C,
        "F_03 - Ez/c": F_lo[0, 3] - Ez / C,
        "F_12 - (-Bz)": F_lo[1, 2] - (-Bz),
        "F_23 - (-Bx)": F_lo[2, 3] - (-Bx),
        "F_31 - (-By)": F_lo[3, 1] - (-By),
    }
    for name, expr in checks.items():
        if sp.simplify(expr) != 0:
            raise AssertionError(f"{name} did not simplify to 0: {expr}")
    return True


# ── 2. Lagrangian density: decomposing the action integral's integrand ──────

def free_lagrangian_density(F_lo, F_up):
    """L_free = -1/(4 mu0) F_{mu nu} F^{mu nu} -- the field part of the
    action integral S = INT L d^4x, before any simplification."""
    FF = sum(F_lo[mu, nu] * F_up[mu, nu] for mu in range(4) for nu in range(4))
    return -sp.expand(FF) / (4 * MU0)


def verify_lagrangian_reduces_to_field_energy(L_free, E, B) -> bool:
    """CHECKED: -1/(4 mu0) F_munu F^munu must algebraically decompose into
    the familiar field-energy Lagrangian (eps0/2)E^2 - B^2/(2 mu0), using
    mu0 = 1/(eps0 c^2) -- the identity that recovers the usual 'electric
    minus magnetic energy density' form from the tensor formalism."""
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    E2 = Ex**2 + Ey**2 + Ez**2
    B2 = Bx**2 + By**2 + Bz**2
    L_free_eps = L_free.subs(MU0, MU0_FROM_EPS0)
    target = EPS0 / 2 * E2 - B2 / (2 * MU0_FROM_EPS0)
    diff = sp.simplify(sp.expand(L_free_eps) - sp.expand(target))
    if diff != 0:
        raise AssertionError(f"L_free did not reduce to (eps0/2)E^2 - B^2/(2 mu0): leftover {diff}")
    return True


# ── 3. Euler-Lagrange field equation: canonical momentum density ────────────

def canonical_momentum_density():
    """pi^{mu nu} = dL_free/d(d_mu A_nu), treating each of the 16 partial
    derivatives d_mu A_nu as an INDEPENDENT variable -- the field-theory
    analogue of dL/d(dq/dt) in point-particle mechanics -- NOT found by
    differentiating the already-substituted, concrete L_free from section 2.
    Returns (pi_generic, F_up_generic): the 4x4 canonical-momentum Matrix
    and the matching generic F^{mu nu}, both still in terms of abstract
    derivative symbols so verify_canonical_momentum_matches_F can compare
    them before any concrete potential is substituted in."""
    d = [[sp.Symbol(f"d{mu}{nu}") for nu in range(4)] for mu in range(4)]
    F_lo_generic = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            F_lo_generic[mu, nu] = d[mu][nu] - d[nu][mu]
    F_up_generic = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            F_up_generic[mu, nu] = ETA[mu, mu] * ETA[nu, nu] * F_lo_generic[mu, nu]
    FF_generic = sp.expand(sum(F_lo_generic[mu, nu] * F_up_generic[mu, nu]
                                for mu in range(4) for nu in range(4)))
    L_free_generic = -FF_generic / (4 * MU0)
    pi_generic = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            pi_generic[mu, nu] = sp.diff(L_free_generic, d[mu][nu])
    return pi_generic, F_up_generic


def verify_canonical_momentum_matches_F(pi_generic, F_up_generic) -> bool:
    """CHECKED: pi^{mu nu} = -F^{mu nu}/mu0 -- this is the step that makes
    the Euler-Lagrange equation collapse to d_mu F^{mu nu} = mu0 J^nu
    instead of some more complicated expression."""
    for mu in range(4):
        for nu in range(4):
            diff = sp.simplify(pi_generic[mu, nu] - (-F_up_generic[mu, nu] / MU0))
            if diff != 0:
                raise AssertionError(f"pi^({mu}{nu}) != -F^({mu}{nu})/mu0: leftover {diff}")
    return True


def euler_lagrange_maxwell_equation(F_up, J_up):
    """d_mu F^{mu nu} for nu=0..3, from varying S = INT (L_free - J^mu A_mu)
    d^4x: d L_int/d A_nu = -J^nu, so Euler-Lagrange gives
    d_mu pi^{mu nu} - (-J^nu) = 0, i.e. d_mu F^{mu nu} = mu0 J^nu. Returns
    the raw d_mu F^{mu nu} list (for verify_gauss_and_ampere_maxwell) and
    the full equation-of-motion residual d_mu F^{mu nu} - mu0 J^nu."""
    div_F = [sp.simplify(sum(sp.diff(F_up[mu, nu], COORDS[mu]) for mu in range(4)))
             for nu in range(4)]
    residual = [sp.simplify(div_F[nu] - MU0 * J_up[nu]) for nu in range(4)]
    return div_F, residual


def verify_gauss_and_ampere_maxwell(div_F, E, B) -> bool:
    """CHECKED: the nu=0 component of d_mu F^{mu nu} must equal div(E)/c
    (so d_mu F^{mu 0} = mu0 J^0 = mu0 c rho becomes div(E) = rho/eps0,
    Gauss's law, using mu0 c^2 = 1/eps0). The nu=1,2,3 components must
    equal (curl B)_i - (1/c^2) dE_i/dt (so = mu0 J_i becomes the
    Ampere-Maxwell law)."""
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    divE = sp.diff(Ex, X1) + sp.diff(Ey, X2) + sp.diff(Ez, X3)
    curlB = (sp.diff(Bz, X2) - sp.diff(By, X3),
             sp.diff(Bx, X3) - sp.diff(Bz, X1),
             sp.diff(By, X1) - sp.diff(Bx, X2))
    dE_dt = tuple(C * sp.diff(Ei, X0) for Ei in E)

    checks = [("nu=0 (Gauss)", div_F[0], divE / C)]
    for i, name in enumerate(("nu=1 (Ampere-Maxwell x)",
                               "nu=2 (Ampere-Maxwell y)",
                               "nu=3 (Ampere-Maxwell z)")):
        checks.append((name, div_F[i + 1], curlB[i] - dE_dt[i] / C**2))

    for name, lhs, rhs in checks:
        diff = sp.simplify(lhs - rhs)
        if diff != 0:
            raise AssertionError(f"{name}: d_mu F^(mu nu) did not match its familiar EM form: leftover {diff}")
    return True


# ── 4. Bianchi identity: the homogeneous Maxwell equations for free ─────────

def verify_bianchi_identity(F_lo) -> bool:
    """d_lam F_{mu nu} + d_mu F_{nu lam} + d_nu F_{lam mu} = 0 for every
    (lambda, mu, nu) -- must hold IDENTICALLY for F = dA (mixed partials of
    A commute), with NO field equation needed. This is what gives Faraday's
    law and 'no magnetic monopoles' for free, algebraically, rather than as
    separately-imposed Maxwell equations."""
    for lam, mu, nu in combinations(range(4), 3):
        cyclic = (sp.diff(F_lo[mu, nu], COORDS[lam])
                  + sp.diff(F_lo[nu, lam], COORDS[mu])
                  + sp.diff(F_lo[lam, mu], COORDS[nu]))
        if sp.simplify(cyclic) != 0:
            raise AssertionError(f"Bianchi identity failed for (lambda,mu,nu)=({lam},{mu},{nu}): {cyclic}")
    return True


# ── 5. Orchestration: run every step, verify every claim ────────────────────

def derive_maxwell_from_lagrangian() -> dict:
    """Run the full chain -- field tensor, Lagrangian decomposition,
    canonical momentum, Euler-Lagrange field equation, Bianchi identity --
    and verify every algebraic claim along the way. Returns a dict of the
    intermediate symbolic objects plus every verification's True/Exception
    outcome, so a caller (or test) can inspect anything, not just the
    final pass/fail."""
    V, Ax, Ay, Az = four_potential_symbols()
    rho, Jx, Jy, Jz = four_current_symbols()
    A_up, A_lo = contravariant_and_covariant_potential(V, Ax, Ay, Az)
    J_up = [C * rho, Jx, Jy, Jz]

    F_lo, F_up = field_strength_tensor(A_lo)
    E, B = E_and_B_from_potentials(V, Ax, Ay, Az)
    field_tensor_ok = verify_field_tensor_matches_E_B(F_lo, E, B)

    L_free = free_lagrangian_density(F_lo, F_up)
    lagrangian_ok = verify_lagrangian_reduces_to_field_energy(L_free, E, B)

    pi_generic, F_up_generic = canonical_momentum_density()
    momentum_ok = verify_canonical_momentum_matches_F(pi_generic, F_up_generic)

    div_F, residual = euler_lagrange_maxwell_equation(F_up, J_up)
    maxwell_inhomogeneous_ok = verify_gauss_and_ampere_maxwell(div_F, E, B)

    bianchi_ok = verify_bianchi_identity(F_lo)

    return {
        "F_lo": F_lo, "F_up": F_up, "E": E, "B": B, "L_free": L_free,
        "div_F": div_F, "field_tensor_matches_E_B": field_tensor_ok,
        "lagrangian_reduces_to_field_energy": lagrangian_ok,
        "canonical_momentum_matches_F": momentum_ok,
        "maxwell_inhomogeneous_verified": maxwell_inhomogeneous_ok,
        "bianchi_identity_verified": bianchi_ok,
    }


if __name__ == "__main__":
    result = derive_maxwell_from_lagrangian()

    print("1. F_{mu nu} built from A_mu matches E, B component-by-component:",
          result["field_tensor_matches_E_B"])
    print("\n2. L_free = -1/(4 mu0) F_munu F^munu decomposes to (eps0/2)E^2 - B^2/(2 mu0):",
          result["lagrangian_reduces_to_field_energy"])
    print("\n3. Canonical momentum pi^(mu nu) = dL_free/d(d_mu A_nu) equals -F^(mu nu)/mu0:",
          result["canonical_momentum_matches_F"])
    print("\n4. Euler-Lagrange field equation d_mu F^(mu nu) = mu0 J^nu reduces to")
    print("   Gauss's law (nu=0) and the Ampere-Maxwell law (nu=1,2,3):",
          result["maxwell_inhomogeneous_verified"])
    print("\n5. Bianchi identity (Faraday's law + no monopoles) holds identically for F=dA:",
          result["bianchi_identity_verified"])

    print("\nAll four inhomogeneous Maxwell equations (Gauss + Ampere-Maxwell x,y,z),")
    print("plus the homogeneous pair (Faraday + no monopoles), have been DERIVED from")
    print("one Lagrangian density and CHECKED against their familiar vector-calculus")
    print("form -- not assumed, not quoted from a textbook table.")
