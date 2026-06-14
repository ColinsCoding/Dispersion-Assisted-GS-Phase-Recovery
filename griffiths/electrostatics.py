"""Electrostatics engine for Griffiths Ch. 2 (Regan PS#2: 2.5, 2.7, 2.8, 2.11-2.13, 2.18, 2.20, 2.22).

On-axis fields of rings/disks/shells by direct Coulomb integration, Gauss-law
fields for the three symmetries, the curl test for "impossible" electrostatic
fields, and line-charge potentials. SymPy throughout; epsilon_0 is symbolic.
"""

import sympy as sp

from .vectors import CARTESIAN, _as_vec3, curl, grad

eps0 = sp.Symbol("epsilon_0", positive=True)
KE = 1 / (4 * sp.pi * eps0)              # Coulomb constant 1/(4 pi eps0)


# ── on-axis fields by direct Coulomb integration ────────────────────
def ring_field_axis(lam, R, Z):
    """G2.5: on-axis E_z a height Z above the centre of a ring (radius R, line
    charge lambda). Horizontal components cancel; cos(theta)=Z/script_r and the
    arc length integrates to 2 pi R."""
    return sp.simplify(KE * lam * 2 * sp.pi * R * Z / (R**2 + Z**2)**sp.Rational(3, 2))


def disk_field_axis(sigma, R, Z):
    """On-axis E_z above a uniformly charged disk (radius R, surface charge
    sigma), built from rings dq = sigma * 2 pi s ds. (Griffiths 2.6 machinery.)"""
    s = sp.Symbol("s", positive=True)
    integrand = KE * sigma * 2 * sp.pi * s * Z / (s**2 + Z**2)**sp.Rational(3, 2)
    return sp.simplify(sp.integrate(integrand, (s, 0, R)))


def shell_field_coulomb(sigma, R, Z, region="outside"):
    """G2.7 the hard way: E_z on axis at distance Z from the centre of a spherical
    shell (radius R, surface charge sigma), by direct integration over the shell.

    Uses the law of cosines script_r^2 = R^2 + Z^2 - 2 R Z cos(theta'); the
    substitution u = cos(theta') turns it into an elementary integral. region =
    'outside' (Z>R) gives k*q/Z^2, 'inside' (Z<R) gives 0.
    """
    if region not in ("outside", "inside"):
        raise ValueError("region must be 'outside' or 'inside'")
    u = sp.Symbol("u")
    a = R**2 + Z**2
    b = 2 * R * Z
    integrand = (Z - R * u) / (a - b * u)**sp.Rational(3, 2)
    val = sp.simplify(sp.integrate(integrand, (u, -1, 1)))
    # sympy leaves sqrt((Z-R)^2) and sqrt((Z+R)^2) uncollapsed; resolve by sign:
    sqrt_minus = sp.sqrt(R**2 - 2 * R * Z + Z**2)      # = |Z - R|
    sqrt_plus = sp.sqrt(R**2 + 2 * R * Z + Z**2)       # = Z + R  (always > 0)
    val = val.subs(sqrt_plus, Z + R)
    val = val.subs(sqrt_minus, Z - R if region == "outside" else R - Z)
    E = KE * 2 * sp.pi * sigma * R**2 * sp.simplify(val)
    return sp.simplify(E)


# ── Gauss's law for the three symmetries ────────────────────────────
def gauss_sphere(Q, r, R=None, uniform=False):
    """E(r) for a spherical charge distribution by Gauss's law.

    Shell (uniform=False): k Q / r^2 outside, 0 inside (need R for inside).
    Solid uniform sphere (uniform=True, radius R): k Q r / R^3 inside, k Q / r^2 outside.
    Here r is the field radius and the return is the radial component.
    """
    outside = KE * Q / r**2
    if R is None:
        return outside
    inside = KE * Q * r / R**3 if uniform else sp.Integer(0)
    return sp.Piecewise((inside, r < R), (outside, True))


def gauss_line(lam, s):
    """G2.13: infinite line charge lambda, radial field E_s = lambda/(2 pi eps0 s)."""
    return sp.simplify(lam / (2 * sp.pi * eps0 * s))


def gauss_plane(sigma):
    """Infinite sheet sigma: field magnitude sigma/(2 eps0), independent of distance."""
    return sigma / (2 * eps0)


# ── potentials ──────────────────────────────────────────────────────
def line_charge_potential(lam, s, s_ref):
    """G2.22: V(s) = (lambda / 2 pi eps0) ln(s_ref / s); the reference cannot be at
    infinity for an infinite line (the field falls off too slowly)."""
    return sp.simplify(lam / (2 * sp.pi * eps0) * sp.log(s_ref / s))


def potential_from_field_radial(E_r, r, r_ref):
    """V(r) = -int_{r_ref}^{r} E_r dr' for a radial field."""
    rp = sp.Symbol("r'", positive=True)
    return sp.simplify(-sp.integrate(E_r.subs(r, rp), (rp, r_ref, r)))


# ── G2.20: which fields can be electrostatic? ───────────────────────
def is_electrostatic(E, vars=CARTESIAN):
    """True iff curl E = 0 (a necessary condition for E = -grad V). Returns
    (ok: bool, curl_E)."""
    c = sp.simplify(curl(E, vars))
    return (c == sp.zeros(3, 1)), c


def potential_of(E, vars=CARTESIAN, ref=(0, 0, 0)):
    """Scalar potential V with E = -grad V, by line integration from `ref`.
    Raises if E is not curl-free."""
    ok, c = is_electrostatic(E, vars)
    if not ok:
        raise ValueError(f"field is not electrostatic: curl E = {c.T} != 0")
    E = _as_vec3(E, "E")
    t = sp.Symbol("t", real=True)
    path = [ref[i] + t * (vars[i] - ref[i]) for i in range(3)]
    dpath = [vars[i] - ref[i] for i in range(3)]
    integrand = sum(E[i].subs(dict(zip(vars, path))) * dpath[i] for i in range(3))
    return sp.simplify(-sp.integrate(integrand, (t, 0, 1)))


# ── G2.18: overlapping uniformly charged spheres ────────────────────
def overlapping_spheres_field(rho, d):
    """Field in the overlap of two uniform spheres (+rho, -rho) whose centres are
    offset by the vector d. Superposition gives the uniform field E = rho*d/(3 eps0).
    Returns the (constant) field vector."""
    d = _as_vec3(d, "d")
    return sp.simplify(rho * d / (3 * eps0))
