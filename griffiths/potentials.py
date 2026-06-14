"""Potential formulation of electrodynamics (Griffiths Ch. 10, PS#1 problem 11).

E and B defined from (V, A) via Eqs. 10.2-10.3, the d'Alembertian, the Lorenz
gauge, and the residual algebra that turns the inhomogeneous wave equations
back into Maxwell's equations.
"""

import sympy as sp

from .vectors import CARTESIAN, _as_vec3, _check_vars, curl, div, grad

t = sp.Symbol("t", real=True)
c = sp.Symbol("c", positive=True)


def E_from_potentials(V, A, vars=CARTESIAN, time=t):
    """Eq. 10.3:  E = -grad V - dA/dt."""
    A = _as_vec3(A, "A")
    return -grad(V, vars) - sp.diff(A, time)


def B_from_potentials(A, vars=CARTESIAN):
    """Eq. 10.2 (back cover):  B = curl A."""
    return curl(A, vars)


def laplacian(f, vars=CARTESIAN):
    vars = _check_vars(vars)
    return sum(sp.diff(f, v, 2) for v in vars)


def dalembertian(f, vars=CARTESIAN, time=t, cc=c):
    """Griffiths box^2:  del^2 f - (1/c^2) d^2 f/dt^2  (mu0 eps0 = 1/c^2)."""
    return laplacian(f, vars) - sp.diff(f, time, 2) / cc**2


def lorenz_gauge_residual(V, A, vars=CARTESIAN, time=t, cc=c):
    """G = div A + (1/c^2) dV/dt; the Lorenz gauge is G = 0 (Eq. 10.12)."""
    A = _as_vec3(A, "A")
    return div(A, vars) + sp.diff(V, time) / cc**2


def generic_potentials(vars=CARTESIAN, time=t):
    """Generic V(x,y,z,t) and A(x,y,z,t) with no structure assumed."""
    args = (*vars, time)
    V = sp.Function("V")(*args)
    A = sp.Matrix([sp.Function(f"A_{s}")(*args) for s in "xyz"])
    return V, A


def generic_sources(vars=CARTESIAN, time=t):
    """Generic charge density rho(x,y,z,t) and current density J(x,y,z,t)."""
    args = (*vars, time)
    rho = sp.Function("rho")(*args)
    J = sp.Matrix([sp.Function(f"J_{s}")(*args) for s in "xyz"])
    return rho, J
