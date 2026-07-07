"""Gauss's law and its applications: symmetry turns an integral into an answer.

Gauss's law is one of Maxwell's equations in integral form:
        closed-surface integral of E . dA  =  Q_enclosed / eps0.
The flux of the electric field through ANY closed surface depends only on the charge
INSIDE -- not on where that charge sits, not on charges outside. That is a strong
statement, and its power is practical: when the charge has enough SYMMETRY, you can
pick a surface on which E is constant and perpendicular, pull it out of the integral,
and read off the field with no calculus at all.

The classic results, each a Gaussian surface matched to the symmetry:

  point charge / sphere (spherical surface)   E = k Q / r^2
  infinite line charge  (coaxial cylinder)    E = lambda / (2 pi eps0 r) = 2 k lambda / r
  infinite sheet        (pillbox)             E = sigma / (2 eps0)          [uniform!]
  uniform solid sphere  (inside)              E = k Q r / R^3   (grows with r)
                        (outside)             E = k Q / r^2     (like a point charge)
  thin spherical shell  (inside)              E = 0             (no charge enclosed)
  conductor surface     (pillbox)             E = sigma / eps0  (twice the sheet)

This module gives those fields AND verifies Gauss's law directly: it numerically
integrates the exact Coulomb field (dgs.charge_configurations.coulomb_field) over a
sphere and recovers Q_enclosed/eps0 -- and gets ZERO when the charge is outside, the
whole content of the law. Ties to dgs.charge_configurations (the 1/r^2 monopole IS
the point-charge Gaussian result). NumPy only; py-3.13.
"""

import numpy as np
from dgs.charge_configurations import coulomb_field

EPS0 = 8.8541878128e-12                 # F/m, vacuum permittivity
K_COULOMB = 8.9875517873681764e9        # 1/(4 pi eps0)


def gauss_flux(Q_enclosed):
    """The flux through ANY closed surface: Phi = Q_enclosed / eps0. Independent
    of the surface's shape and of every charge outside it."""
    return Q_enclosed / EPS0


# ----------------------------------------------------------------------
# Symmetry-derived fields
# ----------------------------------------------------------------------

def point_charge_field(Q, r):
    """Spherical Gaussian surface: E = k Q / r^2 (Coulomb, recovered)."""
    if r <= 0:
        raise ValueError("r must be positive")
    return K_COULOMB * Q / r ** 2


def line_charge_field(lambda_lin, r):
    """Coaxial cylinder around an infinite line of charge density lambda:
    E = lambda/(2 pi eps0 r) = 2 k lambda / r -- falls as 1/r, not 1/r^2."""
    if r <= 0:
        raise ValueError("r must be positive")
    return lambda_lin / (2 * np.pi * EPS0 * r)


def sheet_field(sigma):
    """Pillbox through an infinite sheet of surface charge sigma:
    E = sigma/(2 eps0) -- UNIFORM, independent of distance."""
    return sigma / (2 * EPS0)


def uniform_sphere_field(Q, R, r):
    """A ball of charge Q spread through radius R. Inside, only the charge within
    r contributes: E = k Q r / R^3 (rises linearly from 0). Outside, it looks
    like a point charge: E = k Q / r^2. Continuous at r = R."""
    if R <= 0 or r < 0:
        raise ValueError("need R > 0 and r >= 0")
    if r <= R:
        return K_COULOMB * Q * r / R ** 3
    return K_COULOMB * Q / r ** 2


def spherical_shell_field(Q, R, r):
    """A thin shell of charge at radius R. Inside there is NO enclosed charge, so
    E = 0 (Faraday cage in miniature); outside, E = k Q / r^2."""
    if R <= 0 or r < 0:
        raise ValueError("need R > 0 and r >= 0")
    return 0.0 if r < R else K_COULOMB * Q / r ** 2


def conductor_surface_field(sigma):
    """Just outside a conductor, all flux exits one face of the pillbox:
    E = sigma / eps0 -- exactly TWICE the isolated-sheet field."""
    return sigma / EPS0


# ----------------------------------------------------------------------
# Verify the law itself: numerically integrate the flux over a sphere
# ----------------------------------------------------------------------

def _fibonacci_sphere(n):
    """n near-uniform points and outward normals on the unit sphere, so each
    represents an equal area 4 pi / n -- a clean quadrature for a surface flux."""
    i = np.arange(n)
    y = 1 - 2 * (i + 0.5) / n
    r_xy = np.sqrt(np.clip(1 - y ** 2, 0, None))
    phi = np.pi * (3 - np.sqrt(5)) * i        # golden-angle spiral
    pts = np.column_stack([r_xy * np.cos(phi), y, r_xy * np.sin(phi)])
    return pts                                  # points == outward unit normals


def numerical_flux(charges, positions, center, radius, n=6000):
    """Numerically integrate E . dA of the exact Coulomb field over a sphere of
    the given center and radius. By Gauss's law this equals Q_enclosed/eps0
    (charges strictly inside) and ZERO for charges outside -- computed here from
    the field, with no symmetry assumed, as an independent check of the law."""
    if radius <= 0 or n < 100:
        raise ValueError("need radius > 0 and n >= 100")
    normals = _fibonacci_sphere(n)
    dA = 4 * np.pi * radius ** 2 / n
    center = np.asarray(center, float)
    flux = 0.0
    for nrm in normals:
        point = center + radius * nrm
        E = coulomb_field(charges, positions, point)
        flux += np.dot(E, nrm) * dA
    return flux


def enclosed_charge(charges, positions, center, radius):
    """Total charge strictly inside the sphere -- what Gauss's law says the flux
    reports, and nothing else."""
    charges = np.asarray(charges, float)
    positions = np.asarray(positions, float)
    inside = np.linalg.norm(positions - np.asarray(center, float), axis=1) < radius
    return float(np.sum(charges[inside]))


if __name__ == "__main__":
    Q = 1e-9
    print("Gauss's law: flux = Q/eps0")
    print(f"  Q=1 nC -> flux = {gauss_flux(Q):.2f} V.m (through ANY closed surface)")

    print("\nsymmetry-derived fields (Q=1 nC, r=1 m):")
    print(f"  point charge : {point_charge_field(Q, 1.0):.3f} V/m  (~1/r^2)")
    print(f"  line (lambda=1 nC/m): {line_charge_field(1e-9, 1.0):.3f} V/m  (~1/r)")
    print(f"  sheet (sigma=1 nC/m^2): {sheet_field(1e-9):.3f} V/m  (uniform)")
    print(f"  conductor surface: {conductor_surface_field(1e-9):.3f} V/m  (2x sheet)")
    print(f"  solid sphere R=1: inside r=0.5 -> {uniform_sphere_field(Q,1,0.5):.3f}, "
          f"outside r=2 -> {uniform_sphere_field(Q,1,2.0):.3f}")
    print(f"  shell R=1: inside r=0.5 -> {spherical_shell_field(Q,1,0.5):.3f} (zero)")

    print("\nVERIFY the law -- numerically integrate the flux over a sphere:")
    exact = gauss_flux(Q)
    fc = numerical_flux([Q], [[0, 0, 0]], [0, 0, 0], 1.0)
    print(f"  charge at center:   flux = {fc:.2f}  (Q/eps0 = {exact:.2f})")
    fo = numerical_flux([Q], [[0.3, 0, 0]], [0, 0, 0], 1.0)
    print(f"  charge off-center:  flux = {fo:.2f}  (still Q/eps0 -- position-independent)")
    fx = numerical_flux([Q], [[2.0, 0, 0]], [0, 0, 0], 1.0)
    print(f"  charge OUTSIDE:     flux = {fx:.2e}  (~0 -- what goes in comes out)")
