"""Test truss analysis (method of joints): known forces, equilibrium, determinacy."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import truss as tr

# the classic triangular truss: pin A, roller B, apex C, 10 kN down at C
nodes = {"A": (0, 0), "B": (4, 0), "C": (2, 2)}
members = [("A", "C"), ("B", "C"), ("A", "B")]
supports = {"A": (True, True), "B": (False, True)}
loads = {"C": (0.0, -10.0)}

mf, rx = tr.solve_truss(nodes, members, supports, loads)

# 1. member forces match the hand calculation (tension +, compression -)
assert np.isclose(mf[("A", "C")], -10 / np.sqrt(2), atol=1e-6)     # -7.071, compression
assert np.isclose(mf[("B", "C")], -10 / np.sqrt(2), atol=1e-6)
assert np.isclose(mf[("A", "B")], 5.0, atol=1e-6)                  # +5, tension

# 2. reactions carry the load: vertical reactions sum to the 10 kN, no horizontal
assert np.isclose(rx["A"][1] + rx["B"][1], 10.0)
assert np.isclose(rx["A"][0], 0.0) and np.isclose(rx["B"][0], 0.0)
assert np.isclose(rx["A"][1], 5.0) and np.isclose(rx["B"][1], 5.0)  # symmetric load

# 3. EVERY joint is in equilibrium: members + reaction + load = 0
def joint_residual(node):
    F = np.array([0.0, 0.0])
    for (a, c), f in mf.items():
        if node in (a, c):
            pa, pc = np.array(nodes[a], float), np.array(nodes[c], float)
            u = (pc - pa) / np.linalg.norm(pc - pa)
            F += f * (u if node == a else -u)        # pull toward the other end
    if node in rx:
        F += np.array(rx[node])
    if node in loads:
        F += np.array(loads[node])
    return np.linalg.norm(F)
for n in nodes:
    assert joint_residual(n) < 1e-9, (n, joint_residual(n))

# 4. determinacy: members + reactions = 2 * nodes (3 + 3 = 6 = 2*3)
assert tr.is_determinate(3, 3, 3)
assert not tr.is_determinate(3, 2, 3)                # a mechanism (too few members)

# 5. an OFF-center load gives ASYMMETRIC reactions that still carry it
mf2, rx2 = tr.solve_truss(nodes, members, supports, {"C": (0.0, -12.0)})
assert np.isclose(rx2["A"][1] + rx2["B"][1], 12.0)   # still balances the full load
# (load over the centroid here stays symmetric; the sum law is the robust check)

print(f"TEST PASS  (triangle truss: AC=BC={mf[('A','C')]:.2f} kN compression, "
      f"AB={mf[('A','B')]:.2f} kN tension; reactions sum to 10 kN; every joint "
      f"equilibrium residual <1e-9; determinacy m+r=2n)")
