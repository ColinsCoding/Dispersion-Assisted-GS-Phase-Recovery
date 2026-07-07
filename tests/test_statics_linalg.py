"""Test dgs.statics_linalg: a truss solved as A x = b. Member forces and
reactions match a hand-solved symmetric two-bar truss, the solution passes an
INDEPENDENT global-equilibrium check, and determinacy maps onto the counts and
the matrix RANK (determinate / indeterminate / mechanism, including a geometric
instability the count test alone would miss)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import statics_linalg as st

# 1. hand-solved symmetric two-bar truss: A(-3,0) pin, B(3,0) pin, C(0,-4), 100 N down
nodes = {"A": (-3, 0), "B": (3, 0), "C": (0, -4)}
members = [("C", "A"), ("C", "B")]
supports = {"A": (True, True), "B": (True, True)}
loads = {"C": (0.0, -100.0)}
sol = st.solve_truss(nodes, members, supports, loads)
assert sol["classification"] == "determinate" and sol["degree"] == 0
assert sol["rank"] == sol["n_equations"] == sol["n_unknowns"] == 6
# T = P*L/(2h) = 100*5/8 = 62.5, both in tension (positive)
assert np.isclose(sol["member_forces"][("C", "A")], 62.5)
assert np.isclose(sol["member_forces"][("C", "B")], 62.5)
assert np.isclose(sol["reactions"][("A", "x")], -37.5)
assert np.isclose(sol["reactions"][("A", "y")], 50.0)
assert np.isclose(sol["reactions"][("B", "x")], 37.5)
assert np.isclose(sol["reactions"][("B", "y")], 50.0)
assert sol["residual"] < 1e-9

# 2. INDEPENDENT global-equilibrium check: reactions balance the load in F and M
Fx, Fy, M = st.global_equilibrium_residual(nodes, sol["reactions"], loads)
assert abs(Fx) < 1e-9 and abs(Fy) < 1e-9 and abs(M) < 1e-9

# 3. compression comes out negative: push the load the other way
sol_up = st.solve_truss(nodes, members, supports, {"C": (0.0, 100.0)})
assert sol_up["member_forces"][("C", "A")] < 0      # now compression

# 4. assemble shape: 2j rows, (m + r) columns
A, b, cols = st.assemble(nodes, members, supports, loads)
assert A.shape == (2 * 3, 2 + 4) and len(cols) == 6

# 5. determinacy by counts
assert st.static_determinacy(3, 3, 3) == ("determinate", 0)
assert st.static_determinacy(3, 4, 3) == ("indeterminate", 1)
assert st.static_determinacy(3, 2, 2) == ("mechanism", -2)

# 6. statically INDETERMINATE truss (redundant member): counts say degree 1,
#    the system is still consistent so global equilibrium holds
red_members = [("C", "A"), ("C", "B"), ("A", "B")]
sol_i = st.solve_truss(nodes, red_members, supports, loads)
assert sol_i["classification"] == "indeterminate" and sol_i["degree"] == 1
assert sol_i["residual"] < 1e-9                     # consistent (min-norm) solution
Fx, Fy, M = st.global_equilibrium_residual(nodes, sol_i["reactions"], loads)
assert abs(Fx) < 1e-8 and abs(Fy) < 1e-8 and abs(M) < 1e-8

# 7. MECHANISM the count test misses: matched counts but all reactions vertical
#    (three vertical rollers) -> no horizontal restraint -> rank-deficient
tri = {"A": (0, 0), "B": (2, 0), "C": (1, 1)}
tri_members = [("A", "B"), ("B", "C"), ("C", "A")]
vert_rollers = {"A": (False, True), "B": (False, True), "C": (False, True)}
sol_m = st.solve_truss(tri, tri_members, vert_rollers, {"C": (0.0, -10.0)})
assert sol_m["n_unknowns"] == sol_m["n_equations"] == 6   # counts MATCH (degree 0)
assert sol_m["rank"] < 6                                  # but the matrix is deficient
assert sol_m["classification"] == "mechanism"            # rank catches the instability

# 8. kwarg bounds
for bad in (lambda: st.assemble(nodes, [("A", "A")], supports, loads),   # zero length
            lambda: st.assemble(nodes, [("A", "Z")], supports, loads),   # unknown node
            lambda: st.static_determinacy(0, 1, 1)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_statics_linalg: all checks passed")
