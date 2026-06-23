"""Griffiths Problem 1.16 -- the divergence of r-hat/r^2, and the 3-D Dirac delta.

The famous paradox. The field v = r-hat/r^2 = (x,y,z)/r^3 obviously fans OUT from the
origin, so it looks like it has positive divergence everywhere. But compute it and for
r != 0 you get ZERO:
    div v = d/dx[x/r^3] + ... = (1/r^3 - 3x^2/r^5) + ... = 3/r^3 - 3 r^2/r^5 = 0.
Yet the flux through ANY sphere is
    flux = integral (r-hat/r^2).(r-hat R^2 dOmega) = integral dOmega = 4 pi,
independent of R. Nonzero flux with zero divergence everywhere except one point can
only mean the divergence is a SPIKE at the origin:
    div(r-hat/r^2) = 4 pi delta^3(r),
zero everywhere, infinite at r=0, integrating to 4 pi. (Equivalently the Laplacian
Green's function laplacian(1/r) = -4 pi delta^3(r).) This is exactly why Gauss's law
gives a point charge's field -- the whole divergence sits on the charge. SymPy + NumPy.
"""

import numpy as np
import sympy as sp


def divergence_inverse_square():
    """Symbolic div of v = r-hat/r^2 = (x,y,z)/r^3 in Cartesian. Returns the simplified
    result, which is 0 for r != 0 (the surprising half of the paradox)."""
    x, y, z = sp.symbols("x y z", real=True)
    r3 = (x**2 + y**2 + z**2) ** sp.Rational(3, 2)
    v = [x / r3, y / r3, z / r3]
    div = sum(sp.diff(v[i], var) for i, var in enumerate((x, y, z)))
    return sp.simplify(div)


def flux_through_sphere(R=1.0, n=400):
    """Numerical surface integral of v = r-hat/r^2 over a sphere of radius R:
        flux = integral v . da = integral (1/R^2)(R^2 sin th) d th d phi = integral sin th = 4 pi,
    INDEPENDENT of R. (v . da reduces to sin(theta) d theta d phi on the sphere.)"""
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n)
    integrand = np.outer(np.ones_like(ph), np.sin(th))      # sin(theta), the v.da element
    return float(np.trapezoid(np.trapezoid(integrand, th, axis=1), ph))


def point_source_delta_strength():
    """The resolution: div(r-hat/r^2) integrated over any volume enclosing the origin
    equals the flux through its surface (divergence theorem) = 4 pi. So
        div(r-hat/r^2) = 4 pi delta^3(r).
    Returns the coefficient 4 pi (the delta's strength)."""
    return 4 * np.pi


if __name__ == "__main__":
    print("div(r-hat/r^2) for r != 0  :", divergence_inverse_square(), " (zero -- the surprise)")
    for R in (0.5, 1.0, 5.0):
        print(f"  flux through sphere R={R}: {flux_through_sphere(R):.5f}  (= 4 pi independent of R)")
    print(f"4 pi = {point_source_delta_strength():.5f}")
    print("\nResolution: div(r-hat/r^2) = 4 pi delta^3(r) -- zero everywhere, all the")
    print("divergence concentrated at the origin where r=0 makes the formula blow up.")
