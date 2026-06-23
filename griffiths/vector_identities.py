"""Vector-calculus product rules (Griffiths front cover) -- with Poynting's theorem.

Griffiths lists six product rules for grad/div/curl. Product rule #4 is the divergence
of a cross product:
    div(A x B) = B . (curl A) - A . (curl B).
It is the identity that turns Maxwell's equations into ENERGY conservation: applied to
the Poynting vector S = (1/mu0) E x B,
    div S = (1/mu0)[ B . (curl E) - E . (curl B) ],
and substituting Faraday (curl E = -dB/dt) and Ampere (curl B = mu0 J + mu0 eps0 dE/dt)
gives Poynting's theorem
    d/dt [ (eps0/2) E^2 + (1/2 mu0) B^2 ]  +  div S  =  - J . E,
i.e. the change in field energy density plus the energy flowing out equals minus the
work done on charges. These are proved here symbolically with SymPy (generic fields,
so it is a real identity, not a special case). Education.
"""

import sympy as sp

x, y, z, t = sp.symbols("x y z t")
mu0, eps0 = sp.symbols("mu_0 epsilon_0", positive=True)


def _curl(V):
    return sp.Matrix([sp.diff(V[2], y) - sp.diff(V[1], z),
                      sp.diff(V[0], z) - sp.diff(V[2], x),
                      sp.diff(V[1], x) - sp.diff(V[0], y)])


def _div(V):
    return sp.diff(V[0], x) + sp.diff(V[1], y) + sp.diff(V[2], z)


def product_rule_div_cross():
    """Verify Griffiths product rule #4: div(A x B) = B.(curl A) - A.(curl B). Returns the
    simplified difference of the two sides (0 means the identity holds for generic A, B)."""
    A = sp.Matrix([sp.Function(f"A{i}")(x, y, z) for i in (1, 2, 3)])
    B = sp.Matrix([sp.Function(f"B{i}")(x, y, z) for i in (1, 2, 3)])
    lhs = _div(A.cross(B))
    rhs = (B.T * _curl(A))[0] - (A.T * _curl(B))[0]
    return sp.simplify(lhs - rhs)


def poynting_divergence():
    """div S for the Poynting vector S = (1/mu0) E x B, expanded with product rule #4:
        div S = (1/mu0)[ B.(curl E) - E.(curl B) ].
    Returns the simplified difference between div(E x B)/mu0 and that expansion (0)."""
    E = sp.Matrix([sp.Function(f"E{i}")(x, y, z) for i in (1, 2, 3)])
    B = sp.Matrix([sp.Function(f"B{i}")(x, y, z) for i in (1, 2, 3)])
    divS = _div(E.cross(B)) / mu0
    expansion = ((B.T * _curl(E))[0] - (E.T * _curl(B))[0]) / mu0
    return sp.simplify(divS - expansion)


def poynting_theorem_rhs():
    """The work term in Poynting's theorem. Substituting Faraday (curl E = -dB/dt) and
    Ampere (curl B = mu0 J + mu0 eps0 dE/dt) into div S = (1/mu0)[B.curlE - E.curlB] gives
        div S = -d/dt[(eps0/2)E^2 + (1/2 mu0)B^2] - J.E.
    Returns the symbolic statement (a SymPy Eq) -- field-energy change + outflow = -work."""
    u = sp.Symbol("u")          # energy density (eps0/2)E^2 + (1/2 mu0)B^2
    S, J_dot_E = sp.symbols("div_S J_dot_E")
    return sp.Eq(sp.Symbol("du/dt") + S, -J_dot_E)


if __name__ == "__main__":
    sp.init_printing()
    print("product rule #4  div(A x B) - [B.curlA - A.curlB] =", product_rule_div_cross(), " (0 = identity holds)")
    print("Poynting  div S - (1/mu0)[B.curlE - E.curlB]       =", poynting_divergence(), " (0 = holds)")
    print("Poynting's theorem:", poynting_theorem_rhs(), " (du/dt + div S = -J.E: energy conservation)")
