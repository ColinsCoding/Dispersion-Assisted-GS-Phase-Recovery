"""Test dgs.truss_symbolic: cross-validated against dgs.truss.solve_truss's
independent NumPy solve (fully numeric case), and checked that leaving the
load symbolic and substituting a number back in reproduces the same answer
(the symbolic and numeric routes must agree, not just each look plausible)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp

from dgs.truss import solve_truss
from dgs.truss_symbolic import solve_truss_symbolic

NODES = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (1.0, 1.5)}
MEMBERS = [("A", "C"), ("B", "C"), ("A", "B")]
SUPPORTS = {"A": (True, True), "B": (False, True)}

# 1. fully numeric: must match dgs.truss.solve_truss to machine precision
loads_numeric = {"C": (0.0, -10000.0)}
mf_ref, rx_ref = solve_truss(NODES, MEMBERS, SUPPORTS, loads_numeric)
mf_sym, rx_sym = solve_truss_symbolic(NODES, MEMBERS, SUPPORTS, loads_numeric)
for k in mf_ref:
    assert abs(mf_ref[k] - float(mf_sym[k])) < 1e-9, (k, mf_ref[k], mf_sym[k])
for n in rx_ref:
    for a in range(2):
        assert abs(rx_ref[n][a] - float(rx_sym[n][a])) < 1e-9

# 2. genuinely symbolic: leave the load as a free parameter P, get closed-form
# expressions, then substituting P back in must reproduce the numeric answer
P = sp.Symbol("P", positive=True)
loads_symbolic = {"C": (0, -P)}
mf_param, rx_param = solve_truss_symbolic(NODES, MEMBERS, SUPPORTS, loads_symbolic)
for k, expr in mf_param.items():
    assert expr.free_symbols == {P}, f"{k} should depend on P and nothing else, got {expr}"
    recovered = float(expr.subs(P, 10000.0))
    assert abs(recovered - mf_ref[k]) < 1e-9, (k, recovered, mf_ref[k])

# 3. member forces scale LINEARLY with the load (a real property of linear
# statics, checkable directly from the symbolic expression, not assumed)
for k, expr in mf_param.items():
    assert sp.degree(sp.Poly(expr, P)) == 1, f"{k} should be linear in P"

# 4. a genuinely symbolic COORDINATE too: leave the apex height h free,
# member forces should still solve, and a numeric substitution should
# reproduce a fresh dgs.truss.solve_truss call at that same height
h = sp.Symbol("h", positive=True)
nodes_param = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (1.0, h)}
mf_h, rx_h = solve_truss_symbolic(nodes_param, MEMBERS, SUPPORTS, loads_numeric)
mf_h2, rx_h2 = solve_truss(NODES, MEMBERS, SUPPORTS, loads_numeric)   # h=1.5 case, independently
for k in mf_h:
    recovered_h = float(mf_h[k].subs(h, 1.5))
    assert abs(recovered_h - mf_h2[k]) < 1e-6, (k, recovered_h, mf_h2[k])

# 5. an indeterminate/mechanism truss (wrong DOF count) must raise, not silently
# return a wrong or partial answer
try:
    solve_truss_symbolic(
        {"A": (0, 0), "B": (2, 0), "C": (1, 1.5), "D": (1, 3)},
        [("A", "C"), ("B", "C"), ("A", "B")],   # too few members for 4 nodes
        SUPPORTS, loads_numeric,
    )
    assert False, "expected a ValueError for a non-determinate truss"
except ValueError:
    pass

print("test_truss_symbolic: all checks passed")
