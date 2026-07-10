"""Effective resistance: the voltage a network topology dictates, from the graph Laplacian.

Wire up resistors between nodes and the whole thing is fixed by ONE object: the conductance-
weighted graph Laplacian L, with L_ii = sum of conductances at node i and L_ij = -g_ij. Node
voltages obey L v = i (inject current i, read voltages v) -- the electrical face of the same
Laplacian that dgs.graph_theory and dgs.spectral_clustering use for communities and diffusion.

Once the hardware is built, the resistance you would MEASURE between any two nodes -- the
EFFECTIVE RESISTANCE, a.k.a. resistance distance -- depends only on the topology, not on any
source. Push 1 A in at node i and out at node j; the voltage that develops is
        R_eff(i,j) = (e_i - e_j)^T L^+ (e_i - e_j) = L^+_ii + L^+_jj - 2 L^+_ij,
where L^+ is the Moore-Penrose pseudoinverse of L (it inverts L on everything but the constant
null vector). This one formula collapses series (R_eff = sum), parallel (conductances add), and
every bridge/mesh in between, and it is a genuine METRIC on the graph.

It also reaches past circuits:
  * FOSTER'S THEOREM: sum over edges of g_ij R_eff(i,j) = n - 1, always.
  * MATRIX-TREE: the number of spanning trees is (1/n) times the product of the nonzero
    Laplacian eigenvalues.
  * RANDOM WALKS: the commute time between i and j is exactly 2m * R_eff(i,j) -- a resistor
    network and a random walker measure the same distance.

Complements dgs.dc_circuits / dgs.spice (which solve for specific node voltages) with the
topology-only invariant. Verified against series/parallel, cycles, complete graphs, Cayley's
tree count, Foster's theorem, and a Monte-Carlo random walk. NumPy only; py-3.13.
"""

import numpy as np


def conductance_laplacian(n_nodes, edges, resistances=None):
    """Conductance-weighted graph Laplacian of a resistor network. `edges` is a list of (i,j)
    node pairs; `resistances` their ohms (default 1 each). Parallel resistors = repeated edges
    (their conductances add)."""
    if n_nodes < 2:
        raise ValueError("need at least 2 nodes")
    Rs = [1.0] * len(edges) if resistances is None else resistances
    if any(R <= 0 for R in Rs):
        raise ValueError("resistances must be > 0")
    L = np.zeros((n_nodes, n_nodes), float)
    for (i, j), R in zip(edges, Rs):
        g = 1.0 / R
        L[i, i] += g; L[j, j] += g
        L[i, j] -= g; L[j, i] -= g
    return L


def laplacian_pseudoinverse(L):
    """Moore-Penrose pseudoinverse L^+ (inverts L off its constant null space)."""
    return np.linalg.pinv(L)


def effective_resistance_matrix(L):
    """All-pairs effective resistance R_eff(i,j) = L^+_ii + L^+_jj - 2 L^+_ij."""
    Lp = laplacian_pseudoinverse(L)
    d = np.diag(Lp)
    return d[:, None] + d[None, :] - 2 * Lp


def effective_resistance(n_nodes, edges, i, j, resistances=None):
    """Effective resistance between nodes i and j of the given resistor network."""
    if i == j:
        return 0.0
    L = conductance_laplacian(n_nodes, edges, resistances)
    return float(effective_resistance_matrix(L)[i, j])


def nodal_voltages(L, current_in, source, sink):
    """Solve L v = i for the node voltages when `current_in` amperes are pushed in at `source`
    and out at `sink`, with the sink grounded (v[sink] = 0). The voltage across source-sink is
    current_in * R_eff(source, sink)."""
    n = L.shape[0]
    b = np.zeros(n)
    b[source] += current_in
    b[sink] -= current_in
    Lp = laplacian_pseudoinverse(L)
    v = Lp @ b
    return v - v[sink]                         # reference the sink to 0 V


def spanning_tree_count(L):
    """Number of spanning trees (Matrix-Tree theorem): (1/n) * product of nonzero eigenvalues
    of the (unit-weight) Laplacian."""
    n = L.shape[0]
    vals = np.linalg.eigvalsh(L)
    nz = vals[vals > 1e-9]
    return float(np.prod(nz) / n)


def foster_sum(n_nodes, edges, resistances=None):
    """Left side of Foster's theorem: sum over edges of g_ij * R_eff(i,j). Equals n - 1 for any
    connected network."""
    L = conductance_laplacian(n_nodes, edges, resistances)
    Reff = effective_resistance_matrix(L)
    Rs = [1.0] * len(edges) if resistances is None else resistances
    return float(sum((1.0 / R) * Reff[i, j] for (i, j), R in zip(edges, Rs)))


def commute_time_mc(n_nodes, edges, i, j, trials=20000, seed=0):
    """Monte-Carlo estimate of the random-walk commute time between i and j (steps to go
    i -> j and back). For an unweighted graph this should equal 2m * R_eff(i,j)."""
    rng = np.random.default_rng(seed)
    nbrs = [[] for _ in range(n_nodes)]
    for a, b in edges:
        nbrs[a].append(b); nbrs[b].append(a)

    def hit(start, target):
        node, steps = start, 0
        while node != target:
            node = nbrs[node][rng.integers(len(nbrs[node]))]
            steps += 1
        return steps

    total = sum(hit(i, j) + hit(j, i) for _ in range(trials))
    return total / trials


if __name__ == "__main__":
    print("=== series and parallel from one formula ===")
    print(f"  three 1-ohm resistors in a line 0-1-2-3: R_eff(0,3) = "
          f"{effective_resistance(4, [(0,1),(1,2),(2,3)], 0, 3):.3f} ohm (=3, series)")
    print(f"  two resistors 2 and 3 ohm in parallel:   R_eff(0,1) = "
          f"{effective_resistance(2, [(0,1),(0,1)], 0, 1, [2.0,3.0]):.3f} ohm (=1.2, parallel)")

    print("\n=== a 6-node ring of 1-ohm resistors (R_eff = d(n-d)/n) ===")
    ring = [(k, (k+1) % 6) for k in range(6)]
    for d in (1, 2, 3):
        print(f"  nodes {d} apart: R_eff = {effective_resistance(6, ring, 0, d):.4f} "
              f"(formula {d*(6-d)/6:.4f})")

    print("\n=== global identities ===")
    print(f"  Foster's theorem (ring):   sum g*R_eff = {foster_sum(6, ring):.3f}  (= n-1 = 5)")
    K4 = [(i, j) for i in range(4) for j in range(i+1, 4)]
    print(f"  spanning trees of K4:      {spanning_tree_count(conductance_laplacian(4, K4)):.0f}"
          f"  (Cayley 4^2 = 16)")

    print("\n=== resistor network == random walker (commute time = 2m R_eff) ===")
    m = len(ring); Reff01 = effective_resistance(6, ring, 0, 1)
    print(f"  ring: 2m*R_eff(0,1) = {2*m*Reff01:.1f} steps  vs  MC commute "
          f"{commute_time_mc(6, ring, 0, 1):.1f}")
