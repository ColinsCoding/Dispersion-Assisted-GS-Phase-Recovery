"""Test the geometric meaning of div/curl/Laplacian, measured directly
(flux, circulation, neighbor-average) rather than assumed equal to the
textbook formula -- and a real photonics curl check via Faraday's law."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import vector_calculus_geometric as vcg

# 1. divergence via flux matches the analytic dFx/dx+dFy/dy for a field with
#    BOTH nonzero divergence and nonzero curl (so the two don't get confused)
Fx = lambda x, y: x + 0.3 * y
Fy = lambda x, y: y - 0.3 * x
x0, y0 = 0.7, -0.4
analytic_div = 2.0
analytic_curl = -0.6
for eps in [0.1, 0.01, 0.001]:
    est_div = vcg.divergence_via_flux_2d(Fx, Fy, x0, y0, eps)
    assert abs(est_div - analytic_div) < 1e-8
    est_curl = vcg.curl_via_circulation_2d(Fx, Fy, x0, y0, eps)
    assert abs(est_curl - analytic_curl) < 1e-8

# 2. a PURELY rotational field (rigid rotation) has zero divergence,
#    nonzero curl -- confirms the two measurements aren't secretly the same
Fx_rot = lambda x, y: -y
Fy_rot = lambda x, y: x
assert abs(vcg.divergence_via_flux_2d(Fx_rot, Fy_rot, 0.3, 0.5, 0.01)) < 1e-10
assert abs(vcg.curl_via_circulation_2d(Fx_rot, Fy_rot, 0.3, 0.5, 0.01) - 2.0) < 1e-8

# 3. a PURELY radial (divergent, non-rotational) field has zero curl,
#    nonzero divergence
Fx_rad = lambda x, y: x
Fy_rad = lambda x, y: y
assert abs(vcg.curl_via_circulation_2d(Fx_rad, Fy_rad, 0.3, 0.5, 0.01)) < 1e-10
assert abs(vcg.divergence_via_flux_2d(Fx_rad, Fy_rad, 0.3, 0.5, 0.01) - 2.0) < 1e-8

# 4. Laplacian via 5-point stencil matches for f=x^2+y^2 (analytic Laplacian=4)
f_quad = lambda x, y: x ** 2 + y ** 2
for h in [0.1, 0.01, 0.001]:
    est = vcg.laplacian_5point_stencil(f_quad, 0.5, -0.2, h)
    assert abs(est - 4.0) < 1e-6

# a HARMONIC function (satisfies Laplace's equation) has Laplacian exactly 0 --
# e.g. f=x^2-y^2 is harmonic (classic example, real part of z^2)
f_harmonic = lambda x, y: x ** 2 - y ** 2
est_harmonic = vcg.laplacian_5point_stencil(f_harmonic, 0.3, 0.7, 0.01)
assert abs(est_harmonic) < 1e-8

# 5. point-source flow field: divergence away from the source converges to
#    0 at the O(eps^2) rate (quadratic convergence, not just "small")
grid = np.linspace(-2, 2, 9)
errs = []
for eps in [0.05, 0.01]:
    div_field = vcg.divergence_grid_from_source(0.0, 0.0, 1.0, grid, grid, eps)
    away = div_field[~np.isnan(div_field)]
    errs.append(np.max(np.abs(away)))
ratio = errs[0] / errs[1]
assert 20 < ratio < 30   # (0.05/0.01)^2 = 25, confirming O(eps^2) convergence

# 6. Faraday's law forces B0 = E0/c for a plane wave -- verified symbolically
result = vcg.faraday_plane_wave_curl_check()
assert result["matches_E0_over_c"] is True

# 7. input validation
for bad_call in [
    lambda: vcg.divergence_via_flux_2d(Fx, Fy, x0, y0, -1.0),
    lambda: vcg.curl_via_circulation_2d(Fx, Fy, x0, y0, -1.0),
    lambda: vcg.laplacian_5point_stencil(f_quad, x0, y0, -1.0),
    lambda: vcg.point_source_flow_field(0.0, 0.0, strength=0.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.vector_calculus_geometric tests passed")
