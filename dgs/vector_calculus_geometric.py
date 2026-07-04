"""The geometric MEANING of grad/div/curl/Laplacian, verified by measuring
it directly, not by reciting the formula:

  * DIVERGENCE = net outward flux per unit area, in the limit of a
    shrinking box. Verified by literally integrating flux through the four
    sides of a small square and dividing by its area -- not by evaluating
    dFx/dx + dFy/dy and calling that "divergence" by definition.
  * CURL = circulation per unit area, in the limit of a shrinking loop.
    Verified by literally summing F.dl around a small square's perimeter.
  * LAPLACIAN = how different a point is from the AVERAGE of its
    neighbors (the 5-point stencil is not a numerical trick bolted onto
    the Laplacian -- it converges to the Laplacian's own limit
    definition as the neighbor spacing shrinks).
  * A "drop": a discrete point source. Its flow field's divergence should
    be concentrated entirely at the source, zero everywhere else --
    literally "divergence measures where fluid is being added."
  * Curl in photonics: Faraday's law curl(E) = -dB/dt, checked against a
    real plane-wave solution (not assumed), which is what forces the
    B0 = E0/c amplitude relation between a plane wave's E and B fields.
"""

import numpy as np
import sympy as sp


def divergence_via_flux_2d(Fx, Fy, x0, y0, eps):
    """Numerically estimate divergence at (x0,y0) as (net outward flux
    through a small square of half-width eps) / (its area) -- the
    divergence theorem's defining limit, computed directly rather than
    assumed equal to dFx/dx+dFy/dy."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    # midpoint-rule flux through each of the 4 sides, outward normal each time
    flux_right = Fx(x0 + eps, y0) * (2 * eps)     # normal = +x, side length 2*eps
    flux_left = -Fx(x0 - eps, y0) * (2 * eps)     # normal = -x
    flux_top = Fy(x0, y0 + eps) * (2 * eps)       # normal = +y
    flux_bottom = -Fy(x0, y0 - eps) * (2 * eps)   # normal = -y
    total_flux = flux_right + flux_left + flux_top + flux_bottom
    area = (2 * eps) ** 2
    return total_flux / area


def curl_via_circulation_2d(Fx, Fy, x0, y0, eps):
    """Numerically estimate the (scalar, z-component) curl at (x0,y0) as
    (circulation F.dl around a small square) / (its area) -- Green's/
    Stokes' theorem's defining limit, computed directly."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    # counterclockwise loop: bottom (+x), right (+y), top (-x), left (-y)
    circ_bottom = Fx(x0, y0 - eps) * (2 * eps)
    circ_right = Fy(x0 + eps, y0) * (2 * eps)
    circ_top = -Fx(x0, y0 + eps) * (2 * eps)
    circ_left = -Fy(x0 - eps, y0) * (2 * eps)
    total_circulation = circ_bottom + circ_right + circ_top + circ_left
    area = (2 * eps) ** 2
    return total_circulation / area


def laplacian_5point_stencil(f, x0, y0, h):
    """The discrete Laplacian: (average of 4 neighbors - center) scaled by
    4/h^2 -- literally "how far this point is from its neighborhood's
    average," converging to the true Laplacian as h shrinks."""
    if h <= 0:
        raise ValueError("h must be positive")
    neighbor_avg = (f(x0 + h, y0) + f(x0 - h, y0) + f(x0, y0 + h) + f(x0, y0 - h)) / 4.0
    return 4.0 * (neighbor_avg - f(x0, y0)) / h ** 2


def point_source_flow_field(source_x, source_y, strength=1.0):
    """The 2D flow field of a point source ("a drop" adding fluid at a
    single point): v(r) = strength * (r - r_source) / |r - r_source|^2,
    radially outward, falling off as 1/distance (the 2D analog of a 3D
    point source's 1/r^2 falloff). Returns (Fx, Fy) callables."""
    if strength == 0:
        raise ValueError("strength must be nonzero")

    def Fx(x, y):
        dx, dy = x - source_x, y - source_y
        r2 = dx ** 2 + dy ** 2
        return strength * dx / r2 if r2 > 1e-12 else 0.0

    def Fy(x, y):
        dx, dy = x - source_x, y - source_y
        r2 = dx ** 2 + dy ** 2
        return strength * dy / r2 if r2 > 1e-12 else 0.0

    return Fx, Fy


def divergence_grid_from_source(source_x, source_y, strength, grid_x, grid_y, eps):
    """Divergence of the point-source flow field, evaluated at every point
    of a grid EXCEPT points too close to the source (where the field is
    singular). Confirms divergence is ~0 away from the source -- "a drop
    adds fluid only where it lands, not everywhere downstream of it"."""
    Fx, Fy = point_source_flow_field(source_x, source_y, strength)
    div = np.zeros((len(grid_y), len(grid_x)))
    for j, y in enumerate(grid_y):
        for i, x in enumerate(grid_x):
            if (x - source_x) ** 2 + (y - source_y) ** 2 < (3 * eps) ** 2:
                div[j, i] = np.nan   # too close to the singularity to evaluate meaningfully
            else:
                div[j, i] = divergence_via_flux_2d(Fx, Fy, x, y, eps)
    return div


def faraday_plane_wave_curl_check():
    """Curl in photonics: for a plane wave E=(Ex,0,0), Ex=E0*cos(kz-wt),
    B=(0,By,0), By=B0*cos(kz-wt), verify Faraday's law
    curl(E) = -dB/dt SYMBOLICALLY, which is what FORCES the amplitude
    relation B0=E0*k/w=E0/c between a light wave's E and B fields --
    not an independently assumed fact."""
    z, t, k, w, E0, B0 = sp.symbols('z t k omega E0 B0', real=True, positive=True)
    Ex = E0 * sp.cos(k * z - w * t)
    By = B0 * sp.cos(k * z - w * t)

    curl_E_y = sp.diff(Ex, z)          # (curl E)_y = dEx/dz for E=(Ex,0,0), no x,y dependence
    minus_dB_dt = -sp.diff(By, t)

    # Faraday's law: curl(E)_y = -dB/dt must hold for ALL z,t -- solve for B0/E0
    eq = sp.Eq(curl_E_y, minus_dB_dt)
    ratio = sp.solve(eq, B0)
    B0_solution = ratio[0] if ratio else None
    c_expected = w / k   # the wave's phase speed
    matches_c = sp.simplify(B0_solution - E0 / c_expected) == 0 if B0_solution is not None else False
    return {
        "curl_E_y": curl_E_y, "minus_dB_dt": minus_dB_dt,
        "B0_solution": B0_solution, "matches_E0_over_c": matches_c,
    }


if __name__ == "__main__":
    print("=== Divergence: net outward flux / area, for a swirling+diverging field ===")
    Fx = lambda x, y: x + 0.3 * y
    Fy = lambda x, y: y - 0.3 * x
    x0, y0 = 0.7, -0.4
    analytic_div = 1.0 + 1.0   # dFx/dx=1, dFy/dy=1
    for eps in [0.1, 0.01, 0.001, 1e-4]:
        est = divergence_via_flux_2d(Fx, Fy, x0, y0, eps)
        print(f"  eps={eps:.0e}: flux-based divergence estimate = {est:.6f} "
              f"(analytic dFx/dx+dFy/dy = {analytic_div:.6f}, error={abs(est-analytic_div):.2e})")

    print("\n=== Curl: circulation / area, same field ===")
    analytic_curl = -0.3 - 0.3   # dFy/dx=-0.3, dFx/dy=0.3 -> curl_z = dFy/dx - dFx/dy = -0.6
    for eps in [0.1, 0.01, 0.001, 1e-4]:
        est = curl_via_circulation_2d(Fx, Fy, x0, y0, eps)
        print(f"  eps={eps:.0e}: circulation-based curl estimate = {est:.6f} "
              f"(analytic dFy/dx-dFx/dy = {analytic_curl:.6f}, error={abs(est-analytic_curl):.2e})")

    print("\n=== Laplacian: neighbor-average deviation, for f=x^2+y^2 (analytic Laplacian=4) ===")
    f = lambda x, y: x ** 2 + y ** 2
    for h in [0.1, 0.01, 0.001]:
        est = laplacian_5point_stencil(f, 0.5, -0.2, h)
        print(f"  h={h}: 5-point Laplacian estimate = {est:.6f} (analytic = 4.0, error={abs(est-4.0):.2e})")

    print("\n=== A 'drop': divergence of a point-source flow field ===")
    grid = np.linspace(-2, 2, 9)
    print("  (residual is genuine O(eps^2) discretization error, not a real nonzero")
    print("   divergence -- shown converging to 0 as eps shrinks, confirming that, not asserting it)")
    for eps in [0.05, 0.01, 0.001]:
        div_field = divergence_grid_from_source(0.0, 0.0, strength=1.0, grid_x=grid, grid_y=grid, eps=eps)
        away_from_source = div_field[~np.isnan(div_field)]
        print(f"  eps={eps:.3f}: max |divergence| away from the source = "
              f"{np.max(np.abs(away_from_source)):.2e}")

    print("\n=== Curl in photonics: Faraday's law forces B0 = E0/c for a plane wave ===")
    result = faraday_plane_wave_curl_check()
    print(f"  (curl E)_y = dEx/dz = {result['curl_E_y']}")
    print(f"  -dB/dt              = {result['minus_dB_dt']}")
    print(f"  Faraday's law solved for B0: B0 = {result['B0_solution']}")
    print(f"  matches the wave-impedance relation B0 = E0/c (c=omega/k)? {result['matches_E0_over_c']}")
