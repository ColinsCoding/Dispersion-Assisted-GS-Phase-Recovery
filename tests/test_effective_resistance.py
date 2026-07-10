"""Test dgs.effective_resistance: series/parallel, cycle & complete-graph formulas, the
resistance-distance metric, Foster's theorem, Matrix-Tree spanning counts, nodal voltages,
and the commute-time = 2m R_eff random-walk identity."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import effective_resistance as er

# 1. series: resistors in a line add
assert math.isclose(er.effective_resistance(4, [(0,1),(1,2),(2,3)], 0, 3), 3.0, rel_tol=1e-9)
assert math.isclose(er.effective_resistance(3, [(0,1),(1,2)], 0, 2, [2.0,3.0]), 5.0, rel_tol=1e-9)

# 2. parallel: repeated edges combine as conductances
assert math.isclose(er.effective_resistance(2, [(0,1),(0,1)], 0, 1, [2.0,3.0]), 1.2, rel_tol=1e-9)
# n identical unit resistors in parallel -> 1/n
assert math.isclose(er.effective_resistance(2, [(0,1)]*5, 0, 1), 1/5, rel_tol=1e-9)

# 3. n-cycle of unit resistors: R_eff between nodes d apart = d(n-d)/n
n = 8
ring = [(k, (k+1) % n) for k in range(n)]
for d in range(1, n):
    assert math.isclose(er.effective_resistance(n, ring, 0, d), d*(n-d)/n, rel_tol=1e-9)

# 4. complete graph K_n (unit resistors): R_eff = 2/n between any pair
for nn in (3, 4, 5):
    Kn = [(i, j) for i in range(nn) for j in range(i+1, nn)]
    assert math.isclose(er.effective_resistance(nn, Kn, 0, 1), 2/nn, rel_tol=1e-9)

# 5. resistance distance is a metric: zero diagonal, symmetric, triangle inequality
L = er.conductance_laplacian(n, ring)
R = er.effective_resistance_matrix(L)
assert np.allclose(np.diag(R), 0)
assert np.allclose(R, R.T)
for a in range(n):
    for b in range(n):
        for c in range(n):
            assert R[a, c] <= R[a, b] + R[b, c] + 1e-9      # triangle inequality

# 6. Foster's theorem: sum over edges of g_ij R_eff(i,j) = n - 1
assert math.isclose(er.foster_sum(n, ring), n - 1, rel_tol=1e-9)
K5 = [(i, j) for i in range(5) for j in range(i+1, 5)]
assert math.isclose(er.foster_sum(5, K5), 5 - 1, rel_tol=1e-9)
# weighted network too
we = [(0,1),(1,2),(2,0)]
assert math.isclose(er.foster_sum(3, we, [2.0, 3.0, 4.0]), 3 - 1, rel_tol=1e-9)

# 7. Matrix-Tree: spanning-tree counts (tree->1, cycle->n, Cayley K_n -> n^{n-2})
tree = [(0,1),(1,2),(1,3)]
assert round(er.spanning_tree_count(er.conductance_laplacian(4, tree))) == 1
assert round(er.spanning_tree_count(er.conductance_laplacian(n, ring))) == n
for nn in (3, 4, 5):
    Kn = [(i, j) for i in range(nn) for j in range(i+1, nn)]
    assert round(er.spanning_tree_count(er.conductance_laplacian(nn, Kn))) == nn ** (nn - 2)

# 8. nodal voltages: pushing I amps across gives V = I * R_eff, and Ohm's law holds
edges = [(0,1),(1,2),(2,3)]                     # series 1-ohm chain
Ls = er.conductance_laplacian(4, edges)
v = er.nodal_voltages(Ls, current_in=2.0, source=0, sink=3)
assert math.isclose(v[3], 0.0, abs_tol=1e-9)                       # sink grounded
assert math.isclose(v[0] - v[3], 2.0 * 3.0, rel_tol=1e-9)         # V = I * R_eff = 2*3
# voltage drops evenly across three equal resistors
assert math.isclose(v[0]-v[1], v[1]-v[2], rel_tol=1e-9)

# 9. resistor network == random walker: commute time ~ 2m R_eff (MC, loose tol)
m = len(ring)
Reff01 = er.effective_resistance(n, ring, 0, 1)
mc = er.commute_time_mc(n, ring, 0, 1, trials=30000, seed=1)
assert math.isclose(mc, 2 * m * Reff01, rel_tol=0.08)

# 10. kwarg bounds
for bad in (lambda: er.conductance_laplacian(1, []),
            lambda: er.conductance_laplacian(2, [(0,1)], [-5.0]),
            lambda: er.effective_resistance(2, [(0,1)], 0, 0)):   # i==j returns 0, not error
    pass
assert er.effective_resistance(2, [(0,1)], 0, 0) == 0.0
for bad in (lambda: er.conductance_laplacian(1, []),
            lambda: er.conductance_laplacian(2, [(0,1)], [-5.0])):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_effective_resistance: all checks passed")
