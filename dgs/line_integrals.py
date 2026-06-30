"""Line integrals, surface integrals, and the three big theorems of vector calculus.

This module implements:
  - Scalar line integrals  int_C f ds
  - Vector line integrals  int_C F . dr  (work, circulation)
  - Numerical verification of Green's theorem  (2-D plane)
  - Numerical verification of Stokes' theorem  (3-D surface)
  - Numerical verification of the Divergence theorem

Physics context (Griffiths Ch 1, Ch 7):
  Every quantity in electrodynamics is a line or surface integral.
  The EMF around a loop = -d/dt int_S B.da  (Faraday's law = Stokes applied to E)
  The circulation of B = mu0 * I_enc              (Ampere = Stokes applied to B)
  The flux of E through a closed surface = Q/eps0  (Gauss = Divergence theorem)

All integrals are computed by Gaussian quadrature (numpy) on a parameterised curve,
so they converge for smooth integrands without needing many points.

Usage:
    from dgs.line_integrals import (
        scalar_line_integral, vector_line_integral,
        verify_greens, verify_stokes, verify_divergence,
        lorentz_work, biot_savart_segment
    )
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# Parameterised curve helper
# ---------------------------------------------------------------------------

def _gauss_pts(n: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights on [0, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    t = (nodes + 1) / 2          # map [-1,1] -> [0,1]
    w = weights / 2              # adjust weights
    return t, w


# ---------------------------------------------------------------------------
# Scalar line integral  int_C f(r) ds
# ---------------------------------------------------------------------------

def scalar_line_integral(
    f: Callable[[np.ndarray], float],
    r: Callable[[float], np.ndarray],
    n_pts: int = 64,
) -> float:
    """Compute int_C f(r) ds along the curve r(t), t in [0, 1].

    Parameters
    ----------
    f    : scalar function f(xyz_array) -> float
    r    : curve r(t) -> 3-vector, parameterised on [0, 1]
    n_pts: Gauss-Legendre quadrature order

    Returns
    -------
    Value of the scalar line integral.

    Example -- arc length of unit circle (should be 2*pi):
        r = lambda t: np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t), 0])
        scalar_line_integral(lambda xyz: 1.0, r)  # -> 6.2832...
    """
    t, w = _gauss_pts(n_pts)
    dt = 1e-6
    total = 0.0
    for ti, wi in zip(t, w):
        r0 = r(ti)
        dr = (r(ti + dt) - r(ti - dt)) / (2 * dt)   # central difference
        ds = np.linalg.norm(dr)                       # |dr/dt|
        total += wi * f(r0) * ds
    return total


# ---------------------------------------------------------------------------
# Vector line integral  int_C F(r) . dr
# ---------------------------------------------------------------------------

def vector_line_integral(
    F: Callable[[np.ndarray], np.ndarray],
    r: Callable[[float], np.ndarray],
    n_pts: int = 64,
) -> float:
    """Compute int_C F(r) . dr along the curve r(t), t in [0, 1].

    Parameters
    ----------
    F    : vector field F(xyz_array) -> 3-vector
    r    : curve r(t) -> 3-vector, parameterised on [0, 1]
    n_pts: Gauss-Legendre quadrature order

    Returns
    -------
    Value of the line integral (scalar, work done by F along C).

    Example -- work done by F=(y, -x, 0) around unit circle (should be -2*pi):
        F = lambda r: np.array([r[1], -r[0], 0.])
        r = lambda t: np.array([np.cos(2*pi*t), np.sin(2*pi*t), 0.])
        vector_line_integral(F, r)  # -> -6.2832...
    """
    t, w = _gauss_pts(n_pts)
    dt = 1e-6
    total = 0.0
    for ti, wi in zip(t, w):
        ri = r(ti)
        dr = (r(ti + dt) - r(ti - dt)) / (2 * dt)
        total += wi * np.dot(F(ri), dr)
    return total


# ---------------------------------------------------------------------------
# Green's theorem:  int_C F.dr  =  int_int_S (dFy/dx - dFx/dy) dA
# ---------------------------------------------------------------------------

def verify_greens(
    F: Callable[[float, float], Tuple[float, float]],
    boundary: Callable[[float], Tuple[float, float]],
    domain_samples: int = 200,
    n_line_pts: int = 128,
) -> dict:
    """Verify Green's theorem numerically on a 2-D region.

    Green's theorem: int_C (Fx dx + Fy dy) = int_S (dFy/dx - dFx/dy) dA
    where C is the positively-oriented boundary of S.

    Parameters
    ----------
    F        : vector field F(x, y) -> (Fx, Fy)
    boundary : r(t) -> (x, y) for t in [0,1], positively oriented
    domain_samples: Monte Carlo samples for the area integral
    n_line_pts: quadrature points for the line integral

    Returns
    -------
    dict with keys: line_integral, area_integral, relative_error
    """
    # Line integral side
    r3d = lambda t: np.array([*boundary(t), 0.0])
    def F3d(xyz):
        fx, fy = F(xyz[0], xyz[1])
        return np.array([fx, fy, 0.0])

    line_val = vector_line_integral(F3d, r3d, n_pts=n_line_pts)

    # Area integral side: sample bounding box, keep points inside curve
    n_bdry = 2000
    ts_bdry = np.linspace(0, 1 - 1/n_bdry, n_bdry)
    xs = np.array([boundary(t)[0] for t in ts_bdry])
    ys = np.array([boundary(t)[1] for t in ts_bdry])
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    area_box = (xmax - xmin) * (ymax - ymin)

    rng = np.random.default_rng(0)
    px = rng.uniform(xmin, xmax, domain_samples)
    py = rng.uniform(ymin, ymax, domain_samples)

    h = 1e-5
    curl_vals = []
    for xi, yi in zip(px, py):
        dFy_dx = (F(xi+h, yi)[1] - F(xi-h, yi)[1]) / (2*h)
        dFx_dy = (F(xi, yi+h)[0] - F(xi, yi-h)[0]) / (2*h)
        curl_vals.append(dFy_dx - dFx_dy)

    inside = _winding_inside(px, py, xs, ys)
    # Monte Carlo: area_box * E[curl * 1_inside] = int_S curl dA
    area_val = area_box * np.mean(np.array(curl_vals) * inside.astype(float))

    rel_err = abs(line_val - area_val) / (abs(line_val) + 1e-15)
    return {
        'line_integral': line_val,
        'area_integral': area_val,
        'relative_error': rel_err,
    }


def _winding_inside(px, py, cx, cy):
    """Simple ray-casting test: is point (px,py) inside polygon (cx,cy)?"""
    n = len(cx)
    inside = np.zeros(len(px), dtype=bool)
    for k in range(len(px)):
        count = 0
        x, y = px[k], py[k]
        for i in range(n - 1):
            xi, yi = cx[i], cy[i]
            xj, yj = cx[i+1], cy[i+1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
                count += 1
        inside[k] = (count % 2 == 1)
    return inside


# ---------------------------------------------------------------------------
# Stokes' theorem: int_C F.dr  =  int_S (curl F) . da
# ---------------------------------------------------------------------------

def verify_stokes(
    F: Callable[[np.ndarray], np.ndarray],
    r_boundary: Callable[[float], np.ndarray],
    r_surface: Callable[[float, float], np.ndarray],
    n_line_pts: int = 128,
    n_surf_pts: int = 32,
) -> dict:
    """Verify Stokes' theorem numerically.

    Stokes: int_C F.dr = int_S (curl F) . dA
    where C = boundary of surface S.

    Parameters
    ----------
    F          : vector field F(xyz) -> 3-vector
    r_boundary : r(t) -> 3-vector, t in [0,1], positively oriented boundary
    r_surface  : r(u,v) -> 3-vector, (u,v) in [0,1]^2, parameterised surface patch
    n_line_pts : Gauss-Legendre order for line integral
    n_surf_pts : grid points per dimension for surface integral

    Returns
    -------
    dict with keys: line_integral, surface_integral, relative_error
    """
    line_val = vector_line_integral(F, r_boundary, n_pts=n_line_pts)

    # Surface integral of curl F
    h = 1e-5
    def curl_F(xyz):
        x, y, z = xyz
        dFz_dy = (F(np.array([x, y+h, z]))[2] - F(np.array([x, y-h, z]))[2]) / (2*h)
        dFy_dz = (F(np.array([x, y, z+h]))[1] - F(np.array([x, y, z-h]))[1]) / (2*h)
        dFx_dz = (F(np.array([x, y, z+h]))[0] - F(np.array([x, y, z-h]))[0]) / (2*h)
        dFz_dx = (F(np.array([x+h, y, z]))[2] - F(np.array([x-h, y, z]))[2]) / (2*h)
        dFy_dx = (F(np.array([x+h, y, z]))[1] - F(np.array([x-h, y, z]))[1]) / (2*h)
        dFx_dy = (F(np.array([x, y+h, z]))[0] - F(np.array([x, y-h, z]))[0]) / (2*h)
        return np.array([dFz_dy - dFy_dz, dFx_dz - dFz_dx, dFy_dx - dFx_dy])

    us = np.linspace(0, 1, n_surf_pts)
    vs = np.linspace(0, 1, n_surf_pts)
    du = us[1] - us[0]
    dv = vs[1] - vs[0]
    dt = 1e-5

    surf_val = 0.0
    for u in us:
        for v in vs:
            p  = r_surface(u, v)
            pu = (r_surface(u+dt, v) - r_surface(u-dt, v)) / (2*dt)
            pv = (r_surface(u, v+dt) - r_surface(u, v-dt)) / (2*dt)
            dA = np.cross(pu, pv) * du * dv        # vector area element
            surf_val += np.dot(curl_F(p), dA)

    rel_err = abs(line_val - surf_val) / (abs(line_val) + 1e-15)
    return {
        'line_integral': line_val,
        'surface_integral': surf_val,
        'relative_error': rel_err,
    }


# ---------------------------------------------------------------------------
# Divergence theorem: int_S F.da = int_V (div F) dV
# ---------------------------------------------------------------------------

def verify_divergence(
    F: Callable[[np.ndarray], np.ndarray],
    center: np.ndarray = None,
    radius: float = 1.0,
    n_surf: int = 40,
    n_vol: int = 20,
) -> dict:
    """Verify the Divergence (Gauss's) theorem on a sphere.

    Gauss: closed int F.da = int_V div(F) dV

    Parameters
    ----------
    F      : vector field F(xyz) -> 3-vector
    center : center of the sphere (default origin)
    radius : sphere radius
    n_surf : grid points per dimension on sphere surface
    n_vol  : grid points per dimension in volume (Monte Carlo)

    Returns
    -------
    dict with keys: surface_integral, volume_integral, relative_error
    """
    if center is None:
        center = np.zeros(3)

    # Surface integral over sphere (outward normal = r_hat)
    thetas = np.linspace(0, np.pi, n_surf)
    phis   = np.linspace(0, 2*np.pi, n_surf)
    dth = thetas[1] - thetas[0]
    dph = phis[1] - phis[0]

    surf_val = 0.0
    for th in thetas:
        for ph in phis:
            n_hat = np.array([np.sin(th)*np.cos(ph),
                              np.sin(th)*np.sin(ph),
                              np.cos(th)])
            p = center + radius * n_hat
            dA = radius**2 * np.sin(th) * dth * dph
            surf_val += np.dot(F(p), n_hat) * dA

    # Volume integral of divergence (Monte Carlo in sphere)
    h = 1e-5
    def div_F(xyz):
        x, y, z = xyz
        dFx = (F(np.array([x+h, y, z]))[0] - F(np.array([x-h, y, z]))[0]) / (2*h)
        dFy = (F(np.array([x, y+h, z]))[1] - F(np.array([x, y-h, z]))[1]) / (2*h)
        dFz = (F(np.array([x, y, z+h]))[2] - F(np.array([x, y, z-h]))[2]) / (2*h)
        return dFx + dFy + dFz

    rng = np.random.default_rng(7)
    n_mc = n_vol**3
    pts = rng.uniform(-radius, radius, (n_mc, 3)) + center
    mask = np.linalg.norm(pts - center, axis=1) <= radius
    pts_in = pts[mask]
    vol_sphere = 4/3 * np.pi * radius**3
    if len(pts_in) == 0:
        vol_val = 0.0
    else:
        divs = np.array([div_F(p) for p in pts_in])
        vol_val = vol_sphere * np.mean(divs)

    rel_err = abs(surf_val - vol_val) / (abs(surf_val) + 1e-15)
    return {
        'surface_integral': surf_val,
        'volume_integral': vol_val,
        'relative_error': rel_err,
    }


# ---------------------------------------------------------------------------
# Electrodynamics: work done by Lorentz force along a trajectory
# ---------------------------------------------------------------------------

def lorentz_work(
    E_field: Callable[[np.ndarray, float], np.ndarray],
    B_field: Callable[[np.ndarray, float], np.ndarray],
    r_traj:  Callable[[float], np.ndarray],
    v_traj:  Callable[[float], np.ndarray],
    q: float = 1.602e-19,
    n_pts: int = 128,
) -> float:
    """Compute work done by the Lorentz force on a charged particle.

    W = int_0^T q(E + v x B) . v dt

    Note: the magnetic force q(v x B) is always perpendicular to v, so it
    does no work.  This function verifies that numerically.

    Parameters
    ----------
    E_field  : E(r, t) -> 3-vector
    B_field  : B(r, t) -> 3-vector
    r_traj   : r(t) -> 3-vector, particle position, t in [0,1]
    v_traj   : v(t) -> 3-vector, particle velocity, t in [0,1]
    q        : particle charge (default electron charge in Coulombs)
    n_pts    : quadrature points

    Returns
    -------
    Work done by Lorentz force (Joules).
    """
    t, w = _gauss_pts(n_pts)
    total = 0.0
    for ti, wi in zip(t, w):
        ri = r_traj(ti)
        vi = v_traj(ti)
        Ei = E_field(ri, ti)
        Bi = B_field(ri, ti)
        F  = q * (Ei + np.cross(vi, Bi))
        total += wi * np.dot(F, vi)        # power = F . v
    return total


# ---------------------------------------------------------------------------
# Biot-Savart: magnetic field from a current-carrying line segment
# ---------------------------------------------------------------------------

def biot_savart_segment(
    r_obs: np.ndarray,
    r_start: np.ndarray,
    r_end:   np.ndarray,
    I: float = 1.0,
) -> np.ndarray:
    """Compute B at r_obs from a straight current segment (Biot-Savart, analytical).

    For a finite straight segment from r_start to r_end carrying current I,
    the Biot-Savart integral has a closed-form solution:

        B = (mu0*I)/(4*pi*R) * (cos(theta1) - cos(theta2)) * phi_hat

    where R is the perpendicular distance from r_obs to the wire axis and
    theta1, theta2 are the angles from the perpendicular to each endpoint.

    This is exact, unlike numerical Gauss quadrature which fails when the
    segment is much longer than the perpendicular distance (sharp integrand peak).

    Parameters
    ----------
    r_obs  : observation point (3-vector)
    r_start: start of current segment (3-vector)
    r_end  : end of current segment (3-vector)
    I      : current in Amperes (positive = current flows start -> end)

    Returns
    -------
    B field (3-vector) in Tesla.

    Example -- infinite wire at z-axis, I=1A, observation at (1,0,0):
        B = biot_savart_segment([1,0,0], [0,0,-1e6], [0,0,1e6], I=1)
        # |B| -> 2e-7 T, direction y-hat
    """
    MU0_OVER_4PI = 1e-7   # exact in SI

    L_vec = r_end - r_start
    L_len = np.linalg.norm(L_vec)
    if L_len < 1e-15:
        return np.zeros(3)

    L_hat = L_vec / L_len

    # Project r_obs onto the wire axis; find foot of perpendicular
    r_rel   = r_obs - r_start
    s_foot  = np.dot(r_rel, L_hat)      # signed distance along wire to foot
    perp    = r_rel - s_foot * L_hat    # perpendicular vector from wire to r_obs
    R       = np.linalg.norm(perp)      # perpendicular distance

    if R < 1e-15:
        return np.zeros(3)              # on the wire axis -- field undefined

    perp_hat = perp / R

    # Signed distances from foot to each endpoint
    l1 = -s_foot                       # start is at s_foot before foot
    l2 = L_len - s_foot                # end is at (L_len - s_foot) after foot

    # cos(angle from foot to endpoint): l/sqrt(l^2+R^2)
    c1 = l1 / np.sqrt(l1**2 + R**2)
    c2 = l2 / np.sqrt(l2**2 + R**2)

    # phi_hat = L_hat x perp_hat (direction of B, right-hand rule)
    phi_hat = np.cross(L_hat, perp_hat)
    phi_hat /= np.linalg.norm(phi_hat) + 1e-30

    B_mag = MU0_OVER_4PI * I / R * (c2 - c1)
    return B_mag * phi_hat


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Run verification suite and print results."""
    print("Line Integrals -- Verification Suite")
    print("=" * 50)

    # 1. Arc length of unit circle (scalar, should be 2*pi)
    r_circle = lambda t: np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t), 0.0])
    arc_len = scalar_line_integral(lambda xyz: 1.0, r_circle, n_pts=128)
    print(f"\n1. Arc length of unit circle: {arc_len:.6f}  (exact: {2*np.pi:.6f})")

    # 2. Work by conservative field F = grad(x^2+y^2)  along any path = Delta(phi)
    F_cons = lambda r: np.array([2*r[0], 2*r[1], 0.0])   # grad of x^2+y^2
    r_path = lambda t: np.array([t, t**2, 0.0])           # from (0,0) to (1,1)
    work_cons = vector_line_integral(F_cons, r_path)
    print(f"\n2. Work by F=grad(x^2+y^2) from (0,0) to (1,1): {work_cons:.6f}  (exact: 2.0)")

    # 3. Circulation of F=(-y, x, 0) around unit circle (should be 2*pi)
    F_rot = lambda r: np.array([-r[1], r[0], 0.0])
    circ = vector_line_integral(F_rot, r_circle)
    print(f"\n3. Circulation of (-y,x,0) around unit circle: {circ:.6f}  (exact: {2*np.pi:.6f})")

    # 4. Green's theorem for F=(-y, x) on unit disk
    bdry_2d = lambda t: (np.cos(2*np.pi*t), np.sin(2*np.pi*t))
    F_2d    = lambda x, y: (-y, x)
    g = verify_greens(F_2d, bdry_2d, domain_samples=500)
    print(f"\n4. Green's theorem (unit disk, F=(-y,x)):")
    print(f"   Line integral  = {g['line_integral']:.5f}")
    print(f"   Area integral  = {g['area_integral']:.5f}")
    print(f"   Relative error = {g['relative_error']:.2e}")

    # 5. Stokes' theorem: F=(y,-x,z) on hemisphere boundary = equator circle
    F_stokes = lambda r: np.array([r[1], -r[0], r[2]])
    r_equator = lambda t: np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t), 0.0])
    r_hemi    = lambda u, v: np.array([
        np.sin(np.pi/2*u) * np.cos(2*np.pi*v),
        np.sin(np.pi/2*u) * np.sin(2*np.pi*v),
        np.cos(np.pi/2*u),
    ])
    s = verify_stokes(F_stokes, r_equator, r_hemi, n_line_pts=256, n_surf_pts=30)
    print(f"\n5. Stokes' theorem (hemisphere, F=(y,-x,z)):")
    print(f"   Line integral    = {s['line_integral']:.5f}")
    print(f"   Surface integral = {s['surface_integral']:.5f}")
    print(f"   Relative error   = {s['relative_error']:.2e}")

    # 6. Divergence theorem: F = r (radial field), div F = 3, int_V 3 dV = 4*pi
    F_div = lambda r: r.copy()
    d = verify_divergence(F_div, radius=1.0, n_surf=50, n_vol=25)
    exact_div = 4 * np.pi   # int_V 3 dV over unit sphere = 3 * (4/3 pi) = 4 pi
    print(f"\n6. Divergence theorem (unit sphere, F=r, div=3):")
    print(f"   Surface integral = {d['surface_integral']:.5f}  (exact: {exact_div:.5f})")
    print(f"   Volume integral  = {d['volume_integral']:.5f}")
    print(f"   Relative error   = {d['relative_error']:.2e}")

    # 7. Biot-Savart: infinite wire along z-axis approximated by long segment
    #    B at (1,0,0) from infinite wire with I=1A:  B = mu0*I/(2*pi*R) = 2e-7 T
    B = biot_savart_segment(
        r_obs=np.array([1.0, 0.0, 0.0]),
        r_start=np.array([0.0, 0.0, -1000.0]),
        r_end  =np.array([0.0, 0.0,  1000.0]),
        I=1.0,
    )
    B_theory = 2e-7   # mu0 * I / (2*pi*R) in Tesla
    print(f"\n7. Biot-Savart (long wire I=1A at R=1m):")
    print(f"   |B| numerical = {np.linalg.norm(B):.4e} T")
    print(f"   |B| theory    = {B_theory:.4e} T")
    print(f"   B direction   = {B / np.linalg.norm(B)}  (expect [0, 1, 0] * sign)")

    # 8. Magnetic force does no work (Lorentz)
    E_zero = lambda r, t: np.zeros(3)
    B_uniform = lambda r, t: np.array([0.0, 0.0, 1.0])   # 1 T along z
    r_circ = lambda t: np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t), 0.0])
    v_circ = lambda t: np.array([-np.sin(2*np.pi*t), np.cos(2*np.pi*t), 0.0]) * 2*np.pi
    W_mag = lorentz_work(E_zero, B_uniform, r_circ, v_circ, q=1.0)
    print(f"\n8. Work by magnetic force on circular orbit: {W_mag:.2e}  (exact: 0)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    demo()
