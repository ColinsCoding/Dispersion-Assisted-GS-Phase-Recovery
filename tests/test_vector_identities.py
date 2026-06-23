"""Test Griffiths product rule #4 and its Poynting-vector application."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths import vector_identities as vi

# 1. product rule #4: div(A x B) = B.(curl A) - A.(curl B), for GENERIC fields
assert vi.product_rule_div_cross() == 0

# 2. Poynting: div(E x B)/mu0 = (1/mu0)[B.(curl E) - E.(curl B)]  (same identity on E,B)
assert vi.poynting_divergence() == 0

# 3. concrete check on specific fields: A = (y, 0, 0), B = (0, x, 0) -> A x B = (0,0,xy)
x, y, z = sp.symbols("x y z")
A = sp.Matrix([y, 0, 0]); B = sp.Matrix([0, x, 0])
lhs = vi._div(A.cross(B))                                  # div(0,0,xy) = 0
rhs = (B.T * vi._curl(A))[0] - (A.T * vi._curl(B))[0]
assert sp.simplify(lhs - rhs) == 0 and lhs == 0

# 4. the structural statement of Poynting's theorem: du/dt + div S = -J.E
eq = vi.poynting_theorem_rhs()
assert eq.rhs == -sp.Symbol("J_dot_E")                     # energy out = minus work on charges

print("TEST PASS  (product rule #4 div(AxB)=B.curlA-A.curlB holds for generic fields; "
      "Poynting div S = (1/mu0)[B.curlE - E.curlB]; theorem du/dt + div S = -J.E)")
