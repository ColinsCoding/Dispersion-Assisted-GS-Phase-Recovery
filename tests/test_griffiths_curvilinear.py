"""Test griffiths/curvilinear.py: gradient, divergence, curl, and
Laplacian in spherical and cylindrical coordinates, cross-checked against
Cartesian results (griffiths.vectors) via coordinate substitution, not
just transcribed from Griffiths' inside-front-cover formulas."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths.curvilinear import (
    r, theta, phi, s, phi_cyl, z_cyl, c_focal, u_ell, v_ell,
    spherical_scale_factors, cylindrical_scale_factors, elliptic_scale_factors,
    spherical_grad, spherical_div, spherical_curl, spherical_laplacian,
    cylindrical_grad, cylindrical_div, cylindrical_curl, cylindrical_laplacian,
    elliptic_grad, elliptic_div, elliptic_curl, elliptic_laplacian,
    verify_point_charge_field_divergence_free,
    verify_spherical_laplacian_matches_cartesian,
    verify_cylindrical_div_of_position_vector,
    verify_elliptic_orthogonal,
    verify_naive_hyperbolic_polar_fails_orthogonality,
    verify_elliptic_laplacian_matches_cartesian,
    volume_element, surface_element,
)

# 1. scale factors: known values
h_r, h_theta, h_phi = spherical_scale_factors()
assert h_r == 1
assert h_theta == r
assert sp.simplify(h_phi - r * sp.sin(theta)) == 0

h_s, h_phi_c, h_z = cylindrical_scale_factors()
assert h_s == 1
assert h_phi_c == s
assert h_z == 1

# 2. verify_point_charge_field_divergence_free: must pass
assert verify_point_charge_field_divergence_free() is True

# 3. verify_spherical_laplacian_matches_cartesian: must pass, both cases
#    nonzero
result = verify_spherical_laplacian_matches_cartesian()
assert result["r_squared_case"]["spherical"] == result["r_squared_case"]["cartesian"] == 6
assert result["coulomb_case"]["spherical"] == result["coulomb_case"]["cartesian"] == 0

# 4. verify_cylindrical_div_of_position_vector: must pass
assert verify_cylindrical_div_of_position_vector() is True

# 5. spherical_grad: grad(r*cos(theta)) must equal the KNOWN spherical-
#    basis expression for z_hat (since r*cos(theta)=z): (cos(theta),
#    -sin(theta), 0) -- an independent, well-known identity check
grad_result = sp.simplify(spherical_grad(r * sp.cos(theta)))
expected = sp.Matrix([sp.cos(theta), -sp.sin(theta), 0])
assert sp.simplify(grad_result - expected) == sp.zeros(3, 1)

# 6. spherical_curl: curl of a pure radial field (any function of r times
#    r_hat) must be EXACTLY zero -- a purely radial field is always
#    curl-free (a classic, checkable vector-calculus fact)
radial_field = [r**2 * sp.exp(r), 0, 0]
curl_result = sp.simplify(spherical_curl(radial_field))
assert curl_result == sp.zeros(3, 1)

# 7. cylindrical_curl: curl of phi_hat/s (the classic infinite-wire
#    B-field pattern outside the wire) must be zero for s != 0 -- the
#    magnetic field of a straight wire is curl-free everywhere except
#    on the wire itself
wire_field = [0, 1 / s, 0]
curl_wire = sp.simplify(cylindrical_curl(wire_field))
assert curl_wire == sp.zeros(3, 1)

# 8. cylindrical_laplacian matches Cartesian for a simple test case:
#    f = s^2 (= x^2+y^2 in Cartesian), Laplacian should be 4 in both
x, y = sp.symbols("x y", real=True)
lap_cyl = sp.simplify(cylindrical_laplacian(s**2))
f_cart = x**2 + y**2
lap_cart = sp.diff(f_cart, x, 2) + sp.diff(f_cart, y, 2)
assert lap_cyl == lap_cart == 4

# 9. spherical_div / cylindrical_div: input format sanity -- passing a
#    2-component list must raise a clear error (via SymPy's own unpacking,
#    or the wrapper), not silently misbehave
try:
    spherical_div([r, 0])
    raise AssertionError("expected an error for a 2-component field")
except (ValueError, TypeError):
    pass

# 10. elliptic scale factors: known closed form h_u=h_v=c*sqrt(sinh(u)^2+sin(v)^2), h_z=1
h_u, h_v, h_z_ell = elliptic_scale_factors()
assert h_u == h_v
assert sp.simplify(h_u - c_focal * sp.sqrt(sp.sinh(u_ell) ** 2 + sp.sin(v_ell) ** 2)) == 0
assert h_z_ell == 1

# 11. verify_elliptic_orthogonal: the parametrization is genuinely orthogonal
assert verify_elliptic_orthogonal() is True

# 12. verify_naive_hyperbolic_polar_fails_orthogonality: the naive
#     rho*cosh(eta)/rho*sinh(eta) guess is confirmed NOT orthogonal
assert verify_naive_hyperbolic_polar_fails_orthogonality() is True

# 13. verify_elliptic_laplacian_matches_cartesian: Laplacian(x)=0, both ways
result_ell = verify_elliptic_laplacian_matches_cartesian()
assert result_ell["elliptic"] == result_ell["cartesian"] == 0

# 14. elliptic_curl of a pure gradient field must vanish (curl(grad(f))=0,
#     a coordinate-independent vector identity -- checked here in the new
#     coordinate system specifically, not assumed by analogy)
f_test = c_focal * sp.cosh(u_ell) * sp.sin(v_ell)   # this is "y" in elliptic coordinates
grad_f = elliptic_grad(f_test)
curl_of_grad = sp.simplify(elliptic_curl(list(grad_f)))
assert curl_of_grad == sp.zeros(3, 1)

# 15. volume_element / surface_element: known closed forms
assert volume_element(spherical_scale_factors()) == r**2 * sp.sin(theta)
assert surface_element(spherical_scale_factors(), 0) == r**2 * sp.sin(theta)   # r=const sphere
assert sp.simplify(surface_element(spherical_scale_factors(), 1) - r*sp.sin(theta)) == 0   # theta=const cone: h_r*h_phi
assert volume_element(cylindrical_scale_factors()) == s
assert sp.simplify(volume_element(elliptic_scale_factors())
                    - c_focal**2*(sp.sin(v_ell)**2 + sp.sinh(u_ell)**2)) == 0

print("all griffiths.curvilinear tests passed")
