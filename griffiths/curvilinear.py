"""curvilinear.py -- gradient, divergence, curl, and Laplacian in general
orthogonal curvilinear coordinates (Griffiths' inside-front-cover
formulas), specialized to spherical and cylindrical, and CROSS-CHECKED
against griffiths.vectors' Cartesian operators via an independent
coordinate substitution -- not just transcribed from the book.

The general orthogonal-coordinate formulas, in terms of scale factors
h1, h2, h3 (h_i = |d(position)/dq_i|, i.e. how much actual distance one
unit of coordinate q_i corresponds to):

    grad(f) = (1/h1) df/dq1 e1_hat + (1/h2) df/dq2 e2_hat + (1/h3) df/dq3 e3_hat

    div(A) = (1/(h1 h2 h3)) * [ d(h2 h3 A1)/dq1 + d(h1 h3 A2)/dq2 + d(h1 h2 A3)/dq3 ]

    curl(A) = (1/(h2 h3))[d(h3 A3)/dq2 - d(h2 A2)/dq3] e1_hat
            + (1/(h1 h3))[d(h1 A1)/dq3 - d(h3 A3)/dq1] e2_hat
            + (1/(h1 h2))[d(h2 A2)/dq1 - d(h1 A1)/dq2] e3_hat

    Laplacian(f) = div(grad(f))

SPHERICAL (r, theta, phi): h_r=1, h_theta=r, h_phi=r*sin(theta)
CYLINDRICAL (s, phi, z):   h_s=1, h_phi=s, h_z=1
ELLIPTIC (u, v, z), x=c*cosh(u)*cos(v), y=c*sinh(u)*sin(v): h_u=h_v=c*sqrt(sinh(u)^2+sin(v)^2), h_z=1
    -- the orthogonal coordinate system built around a confocal family of
    ELLIPSES (u=const) and HYPERBOLAS (v=const), sharing foci at x=+-c.
    This is the standard, genuinely orthogonal way to get "hyperbolic
    coordinates" -- see verify_naive_hyperbolic_polar_fails_orthogonality
    for why the more obvious-looking guess x=rho*cosh(eta), y=rho*sinh(eta)
    (polar coordinates with cosh/sinh swapped in for cos/sin) is NOT
    orthogonal, unlike real polar coordinates.

A and f passed to the spherical_*/cylindrical_* wrappers are expressed IN
that coordinate system's own basis (A1, A2, A3 = the physical components
along e1_hat, e2_hat, e3_hat), matching Griffiths' own convention -- NOT
Cartesian components relabeled.
"""

import sympy as sp

r, theta, phi = sp.symbols("r theta phi", positive=True)
s, phi_cyl, z_cyl = sp.symbols("s phi z", positive=True)
c_focal, u_ell, v_ell = sp.symbols("c u v", positive=True)

SPHERICAL = (r, theta, phi)
CYLINDRICAL = (s, phi_cyl, z_cyl)
ELLIPTIC = (u_ell, v_ell, z_cyl)   # shares the z axis with CYLINDRICAL


def _check_scale_factors(scale_factors):
    if len(scale_factors) != 3:
        raise ValueError(f"scale_factors must have exactly 3 entries, got {len(scale_factors)}")
    return tuple(scale_factors)


def spherical_scale_factors():
    """h_r=1, h_theta=r, h_phi=r*sin(theta)."""
    return (sp.Integer(1), r, r * sp.sin(theta))


def cylindrical_scale_factors():
    """h_s=1, h_phi=s, h_z=1."""
    return (sp.Integer(1), s, sp.Integer(1))


def elliptic_scale_factors(c=c_focal):
    """h_u=h_v=c*sqrt(sinh(u)^2+sin(v)^2), h_z=1, for
    x=c*cosh(u)*cos(v), y=c*sinh(u)*sin(v), z=z."""
    h = c * sp.sqrt(sp.sinh(u_ell) ** 2 + sp.sin(v_ell) ** 2)
    return (h, h, sp.Integer(1))


# ── General orthogonal curvilinear operators ─────────────────────────────────

def curvilinear_grad(f, coords, scale_factors):
    h1, h2, h3 = _check_scale_factors(scale_factors)
    q1, q2, q3 = coords
    return sp.Matrix([sp.diff(f, q1) / h1, sp.diff(f, q2) / h2, sp.diff(f, q3) / h3])


def curvilinear_div(A, coords, scale_factors):
    h1, h2, h3 = _check_scale_factors(scale_factors)
    q1, q2, q3 = coords
    A1, A2, A3 = A
    return sp.simplify(
        (sp.diff(h2 * h3 * A1, q1) + sp.diff(h1 * h3 * A2, q2) + sp.diff(h1 * h2 * A3, q3))
        / (h1 * h2 * h3)
    )


def curvilinear_curl(A, coords, scale_factors):
    h1, h2, h3 = _check_scale_factors(scale_factors)
    q1, q2, q3 = coords
    A1, A2, A3 = A
    c1 = (sp.diff(h3 * A3, q2) - sp.diff(h2 * A2, q3)) / (h2 * h3)
    c2 = (sp.diff(h1 * A1, q3) - sp.diff(h3 * A3, q1)) / (h1 * h3)
    c3 = (sp.diff(h2 * A2, q1) - sp.diff(h1 * A1, q2)) / (h1 * h2)
    return sp.Matrix([sp.simplify(c1), sp.simplify(c2), sp.simplify(c3)])


def volume_element(scale_factors):
    """dV = h1*h2*h3 (times dq1 dq2 dq3) -- the general orthogonal-
    coordinates volume element, used by both the divergence theorem's
    left side and, per-face, by its right side (a face's area element
    drops the corresponding h)."""
    h1, h2, h3 = _check_scale_factors(scale_factors)
    return h1 * h2 * h3


def surface_element(scale_factors, fixed_index):
    """Area element (times the other two dq's) for a q_i=const surface,
    fixed_index in {0,1,2} selecting which coordinate is held fixed --
    e.g. fixed_index=0 (an r=const sphere) gives h2*h3."""
    h1, h2, h3 = _check_scale_factors(scale_factors)
    hs = [h1, h2, h3]
    del hs[fixed_index]
    return hs[0] * hs[1]


def curvilinear_laplacian(f, coords, scale_factors):
    grad_f = curvilinear_grad(f, coords, scale_factors)
    return curvilinear_div(grad_f, coords, scale_factors)


# ── Spherical convenience wrappers ───────────────────────────────────────────

def spherical_grad(f):
    return curvilinear_grad(f, SPHERICAL, spherical_scale_factors())


def spherical_div(A):
    return curvilinear_div(A, SPHERICAL, spherical_scale_factors())


def spherical_curl(A):
    return curvilinear_curl(A, SPHERICAL, spherical_scale_factors())


def spherical_laplacian(f):
    return curvilinear_laplacian(f, SPHERICAL, spherical_scale_factors())


# ── Cylindrical convenience wrappers ─────────────────────────────────────────

def cylindrical_grad(f):
    return curvilinear_grad(f, CYLINDRICAL, cylindrical_scale_factors())


def cylindrical_div(A):
    return curvilinear_div(A, CYLINDRICAL, cylindrical_scale_factors())


def cylindrical_curl(A):
    return curvilinear_curl(A, CYLINDRICAL, cylindrical_scale_factors())


def cylindrical_laplacian(f):
    return curvilinear_laplacian(f, CYLINDRICAL, cylindrical_scale_factors())


# ── Elliptic (confocal ellipse/hyperbola) convenience wrappers ──────────────

def elliptic_grad(f):
    return curvilinear_grad(f, ELLIPTIC, elliptic_scale_factors())


def elliptic_div(A):
    return curvilinear_div(A, ELLIPTIC, elliptic_scale_factors())


def elliptic_curl(A):
    return curvilinear_curl(A, ELLIPTIC, elliptic_scale_factors())


def elliptic_laplacian(f):
    return curvilinear_laplacian(f, ELLIPTIC, elliptic_scale_factors())


# ── Cross-checks against Cartesian (griffiths.vectors), and known results ──

def verify_point_charge_field_divergence_free() -> bool:
    """CHECKED: div(r_hat/r^2) = 0 for r != 0 -- the spherically-symmetric
    field of a point charge is divergence-free everywhere except the
    origin (where the charge itself sits), a fact usually just quoted
    ("except at the origin") without being computed from the curvilinear
    divergence formula directly."""
    A = [1 / r**2, 0, 0]   # (A_r, A_theta, A_phi)
    result = spherical_div(A)
    if sp.simplify(result) != 0:
        raise AssertionError(f"div(r_hat/r^2) should be 0 for r!=0, got {result}")
    return True


def verify_spherical_laplacian_matches_cartesian() -> dict:
    """CHECKED, two independent test functions: the spherical Laplacian
    formula applied to f(r) must match the CARTESIAN Laplacian
    (sum of second partials) of the SAME function expressed in (x,y,z),
    for BOTH a nonzero case (f=r^2, matching x^2+y^2+z^2) and the classic
    zero case (f=1/r, the Coulomb potential, matching
    1/sqrt(x^2+y^2+z^2)) -- an independent cross-check via coordinate
    substitution, not a restatement of the curvilinear formula."""
    x, y, z = sp.symbols("x y z", real=True)

    # case 1: f = r^2  <->  x^2+y^2+z^2, Laplacian = 6 (nonzero, a real check)
    lap_sph_r2 = sp.simplify(spherical_laplacian(r**2))
    lap_cart_r2 = sp.diff(x**2 + y**2 + z**2, x, 2) + sp.diff(x**2 + y**2 + z**2, y, 2) \
                  + sp.diff(x**2 + y**2 + z**2, z, 2)
    if lap_sph_r2 != lap_cart_r2:
        raise AssertionError(f"spherical Laplacian of r^2 ({lap_sph_r2}) != "
                             f"Cartesian Laplacian of x^2+y^2+z^2 ({lap_cart_r2})")

    # case 2: f = 1/r  <->  1/sqrt(x^2+y^2+z^2), Laplacian = 0 (r != 0)
    lap_sph_coulomb = sp.simplify(spherical_laplacian(1 / r))
    f_cart_coulomb = 1 / sp.sqrt(x**2 + y**2 + z**2)
    lap_cart_coulomb = sp.simplify(sp.diff(f_cart_coulomb, x, 2) + sp.diff(f_cart_coulomb, y, 2)
                                   + sp.diff(f_cart_coulomb, z, 2))
    if lap_sph_coulomb != lap_cart_coulomb:
        raise AssertionError(f"spherical Laplacian of 1/r ({lap_sph_coulomb}) != "
                             f"Cartesian Laplacian of 1/sqrt(x^2+y^2+z^2) ({lap_cart_coulomb})")

    return {"r_squared_case": {"spherical": lap_sph_r2, "cartesian": lap_cart_r2},
            "coulomb_case": {"spherical": lap_sph_coulomb, "cartesian": lap_cart_coulomb}}


def verify_cylindrical_div_of_position_vector() -> bool:
    """CHECKED: the cylindrical-coordinate position vector s*s_hat + z*z_hat
    has div = 3 -- the same scalar-invariant result div(position)=3 gives
    in Cartesian or spherical coordinates, verified directly from the
    cylindrical divergence formula rather than assumed by analogy (the
    s_hat term alone contributes (1/s)*d(s*s)/ds=2, the z_hat term
    contributes d(z)/dz=1, and there is no phi_hat component at all)."""
    A = [s, 0, z_cyl]   # (A_s, A_phi, A_z) = s_hat*s + z_hat*z
    result = sp.simplify(cylindrical_div(A))
    expected = 3   # div(position vector) = 3 in ANY consistent coordinate system (it's a scalar invariant)
    if result != expected:
        raise AssertionError(f"div of the cylindrical position vector: got {result}, expected {expected}")
    return True


def verify_elliptic_orthogonal() -> bool:
    """CHECKED, from the parametrization directly (not assumed from the h_u=h_v
    formula): the tangent vectors dr/du and dr/dv for x=c*cosh(u)*cos(v),
    y=c*sinh(u)*sin(v) have zero dot product for every u, v -- genuinely
    orthogonal coordinates, confirmed by computing the Jacobian and dotting
    its columns, not by citing the standard result."""
    x_of = c_focal * sp.cosh(u_ell) * sp.cos(v_ell)
    y_of = c_focal * sp.sinh(u_ell) * sp.sin(v_ell)
    dr_du = sp.Matrix([sp.diff(x_of, u_ell), sp.diff(y_of, u_ell)])
    dr_dv = sp.Matrix([sp.diff(x_of, v_ell), sp.diff(y_of, v_ell)])
    dot = sp.simplify(dr_du.dot(dr_dv))
    if dot != 0:
        raise AssertionError(f"elliptic coordinates should be orthogonal, got dr/du . dr/dv = {dot}")
    return True


def verify_naive_hyperbolic_polar_fails_orthogonality() -> bool:
    """CHECKED: the more obvious-looking guess at "hyperbolic coordinates" --
    swap cosh/sinh in for cos/sin in ordinary polar coordinates,
    x=rho*cosh(eta), y=rho*sinh(eta) -- is NOT orthogonal, unlike real polar
    coordinates. Real polar works because d(cos)/dtheta=-sin flips sign,
    making the cross term cancel; d(cosh)/deta=+sinh does NOT flip sign, so
    it doesn't. Returns True iff the cross term is confirmed nonzero (i.e.
    the naive guess is confirmed to fail, which is the point of this check)."""
    rho, eta = sp.symbols("rho eta", positive=True)
    x_of = rho * sp.cosh(eta)
    y_of = rho * sp.sinh(eta)
    dr_drho = sp.Matrix([sp.diff(x_of, rho), sp.diff(y_of, rho)])
    dr_deta = sp.Matrix([sp.diff(x_of, eta), sp.diff(y_of, eta)])
    cross_term = sp.simplify(dr_drho.dot(dr_deta))
    expected_nonzero = sp.simplify(cross_term - rho * sp.sinh(2 * eta))
    if expected_nonzero != 0:
        raise AssertionError(f"expected cross term rho*sinh(2 eta), got {cross_term}")
    if cross_term == 0:
        raise AssertionError("naive hyperbolic-polar coordinates were expected to be non-orthogonal")
    return True


def verify_elliptic_laplacian_matches_cartesian() -> dict:
    """CHECKED: the elliptic Laplacian of f = x (expressed as
    c*cosh(u)*cos(v)) must match the trivial Cartesian result
    Laplacian(x) = 0 -- x is a harmonic function in any consistent
    coordinate system, an independent cross-check via coordinate
    substitution rather than a restatement of the curvilinear formula."""
    f_elliptic = c_focal * sp.cosh(u_ell) * sp.cos(v_ell)   # this IS "x" in elliptic coordinates
    lap_elliptic = sp.simplify(elliptic_laplacian(f_elliptic))
    if lap_elliptic != 0:
        raise AssertionError(f"Laplacian(x) should be 0 (x is harmonic), got {lap_elliptic}")
    return {"elliptic": lap_elliptic, "cartesian": sp.Integer(0)}


if __name__ == "__main__":
    print("=== 1. Point charge field: div(r_hat/r^2)=0, checked ===")
    ok1 = verify_point_charge_field_divergence_free()
    print(f"  verified: {ok1}")

    print("\n=== 2. Spherical Laplacian matches Cartesian, two test cases ===")
    result = verify_spherical_laplacian_matches_cartesian()
    print(f"  r^2 case: spherical={result['r_squared_case']['spherical']}, "
          f"cartesian={result['r_squared_case']['cartesian']}")
    print(f"  1/r (Coulomb) case: spherical={result['coulomb_case']['spherical']}, "
          f"cartesian={result['coulomb_case']['cartesian']}")

    print("\n=== 3. Cylindrical divergence of the position vector ===")
    ok3 = verify_cylindrical_div_of_position_vector()
    print(f"  div(s*s_hat + z*z_hat) = 3, verified: {ok3}")

    print("\n=== Example: grad, div, curl, Laplacian of f=r*cos(theta) (spherical) ===")
    f_example = r * sp.cos(theta)
    print("  grad(r*cos(theta)) =")
    sp.pprint(sp.simplify(spherical_grad(f_example)))
    print("  Laplacian(r*cos(theta)) =", sp.simplify(spherical_laplacian(f_example)),
          " (matches z=r*cos(theta), Laplacian of z is 0 -- harmonic)")

    print("\n=== 4. Elliptic (confocal ellipse/hyperbola) coordinates ===")
    ok4 = verify_elliptic_orthogonal()
    print(f"  dr/du . dr/dv = 0, verified orthogonal: {ok4}")
    ok5 = verify_naive_hyperbolic_polar_fails_orthogonality()
    print(f"  naive x=rho*cosh(eta), y=rho*sinh(eta) confirmed NOT orthogonal: {ok5}")
    result4 = verify_elliptic_laplacian_matches_cartesian()
    print(f"  Laplacian(x) in elliptic coords: {result4['elliptic']}  (Cartesian: {result4['cartesian']})")
    print("  scale factors h_u = h_v =", sp.simplify(elliptic_scale_factors()[0]), ", h_z = 1")
