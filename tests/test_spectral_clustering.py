"""Test dgs.spectral_clustering: L=D-A properties, zero-eigenvalue = components, Fiedler
bipartition, recovery of planted communities on a stochastic block model, and the
normalized-cut / modularity quality metrics."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import spectral_clustering as sc

# 1. Laplacian basics: symmetric, PSD, rows sum to zero, smallest eigenvalue 0 (const vector)
edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
A = sc.adjacency_from_edges(4, edges)
L = sc.laplacian(A)
assert np.allclose(L, L.T)                              # symmetric
assert np.allclose(L.sum(axis=1), 0)                   # rows sum to zero
vals, vecs = sc.laplacian_spectrum(A)
assert vals[0] > -1e-9 and abs(vals[0]) < 1e-9         # smallest eigenvalue ~ 0
assert np.all(vals > -1e-9)                            # positive semidefinite
# constant vector is the zero-eigenvalue eigenvector
assert np.allclose(np.abs(vecs[:, 0]), abs(vecs[0, 0]))

# 2. number of zero eigenvalues = number of connected components
A_conn = sc.adjacency_from_edges(5, [(0, 1), (1, 2), (2, 0), (3, 4)])   # 2 components
assert sc.count_components(A_conn) == 2
A_one = sc.adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3)])            # 1 component (path)
assert sc.count_components(A_one) == 1

# 3. Fiedler bipartition of a two-clique barbell splits the cliques
be = [(i, j) for i in range(5) for j in range(i + 1, 5)]               # clique 0-4
be += [(i, j) for i in range(5, 10) for j in range(i + 1, 10)]        # clique 5-9
be += [(4, 5)]                                                        # single bridge
Ab = sc.adjacency_from_edges(10, be)
lab = sc.spectral_bipartition(Ab)
# nodes 0-4 all one label, 5-9 all the other
assert len(set(lab[:5])) == 1 and len(set(lab[5:])) == 1 and lab[0] != lab[5]

# 4. recover planted communities on a stochastic block model
A_sbm, true = sc.stochastic_block_model([30, 30, 30], p_in=0.45, p_out=0.03, seed=1)
pred = sc.spectral_clustering(A_sbm, k=3, seed=0)
acc = sc.clustering_accuracy(true, pred)
assert acc > 0.9, f"recovery accuracy only {acc}"
# a 2-community network too
A2, true2 = sc.stochastic_block_model([40, 40], p_in=0.4, p_out=0.03, seed=2)
assert sc.clustering_accuracy(true2, sc.spectral_clustering(A2, k=2, seed=0)) > 0.9

# 5. quality metrics: the true partition beats a random one
rng = np.random.default_rng(0)
rand_labels = rng.integers(3, size=len(true))
assert sc.modularity(A_sbm, true) > sc.modularity(A_sbm, rand_labels)
assert sc.modularity(A_sbm, true) > 0                  # real community structure
assert sc.normalized_cut(A_sbm, true) < sc.normalized_cut(A_sbm, rand_labels)

# 6. normalized Laplacian: symmetric, eigenvalues in [0, 2]
Ln = sc.normalized_laplacian(A_sbm)
assert np.allclose(Ln, Ln.T)
vn, _ = np.linalg.eigh(Ln)
assert vn.min() > -1e-9 and vn.max() < 2 + 1e-6

# 7. clustering_accuracy is permutation-invariant
assert sc.clustering_accuracy([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
assert sc.clustering_accuracy([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
assert math.isclose(sc.clustering_accuracy([0, 0, 1, 1], [0, 1, 0, 1]), 0.5)

# 8. kwarg bounds
for bad in (lambda: sc.spectral_clustering(A, 0),
            lambda: sc.spectral_clustering(A, 99),
            lambda: sc.adjacency_from_edges(0, []),
            lambda: sc.stochastic_block_model([10, 10], p_in=0.1, p_out=0.5)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_spectral_clustering: all checks passed")
