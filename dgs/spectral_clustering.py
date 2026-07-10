"""Spectral clustering: find the communities in a social network from the graph Laplacian.

A social network is a graph -- people are nodes, friendships are edges. Its structure is
encoded in the LAPLACIAN L = D - A (degree matrix minus adjacency), the same operator
dgs.graph_theory builds. This module adds what that one leaves out: an actual
COMMUNITY-DETECTION algorithm that turns the Laplacian's eigenvectors into cluster labels.

WHY EIGENVECTORS FIND COMMUNITIES.  L is symmetric positive-semidefinite, so it has real
eigenvalues 0 = lambda_1 <= lambda_2 <= ... The number of ZERO eigenvalues equals the number of
connected components, and their eigenvectors are the indicator vectors of those components. A
social network isn't usually fully disconnected -- communities are only *weakly* linked -- so
those exact-zero modes soften into the SMALLEST NONZERO ones: the Fiedler vector (2nd eigenvector)
splits the graph along its weakest cut, and the first k eigenvectors embed every node into R^k so
that community members land close together. Run k-means in that embedding and the communities fall
out. This is exactly the eigenvectors->PCA/clustering bridge from the Griffiths->ML axis: the
spectral decomposition of a Hermitian operator organizes the data.

The pipeline (Ng-Jordan-Weiss): normalized Laplacian -> first k eigenvectors -> row-normalize ->
k-means -> labels. Verified on a stochastic-block-model network with planted communities (it
recovers them), against the component/Fiedler theorems, and by normalized-cut / modularity scores.
Ties to dgs.graph_theory (the Laplacian, GFT, graph diffusion). NumPy-only; py-3.13.
"""

import numpy as np


def adjacency_from_edges(n_nodes, edges, weights=None):
    """Symmetric adjacency matrix A from an undirected edge list."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    A = np.zeros((n_nodes, n_nodes), float)
    w = [1.0] * len(edges) if weights is None else weights
    for (i, j), wij in zip(edges, w):
        A[i, j] = A[j, i] = wij
    return A


def laplacian(A):
    """Combinatorial graph Laplacian L = D - A."""
    A = np.asarray(A, float)
    return np.diag(A.sum(axis=1)) - A


def normalized_laplacian(A):
    """Symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}."""
    A = np.asarray(A, float)
    deg = A.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return np.eye(A.shape[0]) - (dinv[:, None] * A * dinv[None, :])


def laplacian_spectrum(A, normalized=False):
    """Eigenvalues (ascending) and eigenvectors (columns) of the graph Laplacian."""
    L = normalized_laplacian(A) if normalized else laplacian(A)
    vals, vecs = np.linalg.eigh(L)          # L is symmetric
    return vals, vecs


def count_components(A, tol=1e-8):
    """Number of connected components = multiplicity of the zero Laplacian eigenvalue."""
    vals, _ = laplacian_spectrum(A)
    return int(np.sum(vals < tol))


def fiedler_vector(A):
    """The Fiedler vector: eigenvector of the 2nd-smallest Laplacian eigenvalue (the
    algebraic connectivity). Its sign pattern is the graph's weakest cut."""
    vals, vecs = laplacian_spectrum(A)
    return vecs[:, 1]


def spectral_bipartition(A):
    """Split the graph in two by the sign of the Fiedler vector. Returns 0/1 labels."""
    return (fiedler_vector(A) > 0).astype(int)


def _kmeans(X, k, seed=0, n_init=8, max_iter=100):
    """Small Lloyd's k-means with k-means++ init; returns the best labels over n_init runs."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        # k-means++ seeding
        centers = [X[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
            centers.append(X[rng.choice(n, p=probs)])
        C = np.array(centers)
        labels = np.zeros(n, int)
        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
            new = dists.argmin(axis=1)
            if np.array_equal(new, labels) and _ > 0:
                break
            labels = new
            for c in range(k):
                if np.any(labels == c):
                    C[c] = X[labels == c].mean(axis=0)
        inertia = np.sum((X - C[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def spectral_clustering(A, k, seed=0):
    """Partition a graph into k communities (Ng-Jordan-Weiss): embed each node in R^k via the
    first k eigenvectors of the normalized Laplacian, row-normalize, then k-means. Returns an
    integer label per node."""
    if k < 1:
        raise ValueError("k must be >= 1")
    n = np.asarray(A).shape[0]
    if k > n:
        raise ValueError("k cannot exceed the number of nodes")
    _, vecs = laplacian_spectrum(A, normalized=True)
    U = vecs[:, :k]                                     # k smallest eigenvalues
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.where(norms > 0, norms, 1.0)             # project onto the unit sphere
    return _kmeans(U, k, seed=seed)


def normalized_cut(A, labels):
    """Normalized cut of a partition: sum over clusters of (edges leaving)/(cluster volume).
    Lower is a cleaner community split."""
    A = np.asarray(A, float)
    deg = A.sum(axis=1)
    total = 0.0
    for c in np.unique(labels):
        inside = labels == c
        vol = deg[inside].sum()
        cut = A[np.ix_(inside, ~inside)].sum()
        if vol > 0:
            total += cut / vol
    return total


def modularity(A, labels):
    """Newman modularity Q of a partition: fraction of within-community edges minus its
    expected value in a degree-matched random graph. Q > 0 means real community structure."""
    A = np.asarray(A, float)
    m2 = A.sum()                                        # = 2 * (number of edges)
    if m2 == 0:
        return 0.0
    deg = A.sum(axis=1)
    Q = 0.0
    for c in np.unique(labels):
        idx = labels == c
        Q += A[np.ix_(idx, idx)].sum() - deg[idx].sum() ** 2 / m2
    return Q / m2


def stochastic_block_model(sizes, p_in, p_out, seed=0):
    """Generate a social network with planted communities: nodes in the same block connect with
    probability p_in, across blocks with p_out. Returns (A, true_labels)."""
    if not (0 <= p_out <= p_in <= 1):
        raise ValueError("require 0 <= p_out <= p_in <= 1")
    rng = np.random.default_rng(seed)
    labels = np.concatenate([[b] * s for b, s in enumerate(sizes)])
    n = len(labels)
    A = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    return A, labels


def clustering_accuracy(true_labels, pred_labels):
    """Fraction correct under the best matching of predicted labels to true labels
    (clustering is invariant to how the groups are numbered). Tries every relabeling of
    the true classes onto the predicted ones and keeps the best agreement."""
    from itertools import permutations
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    classes = list(np.unique(true))
    best = 0.0
    for perm in permutations(range(len(classes))):
        remap = {classes[i]: perm[i] for i in range(len(classes))}
        mapped_true = np.array([remap[t] for t in true])
        best = max(best, float(np.mean(mapped_true == pred)))
    return best


if __name__ == "__main__":
    print("=== a social network with three planted communities ===")
    A, true = stochastic_block_model([30, 30, 30], p_in=0.45, p_out=0.03, seed=1)
    print(f"  {A.shape[0]} people, {int(A.sum()/2)} friendships, "
          f"{count_components(A)} connected component(s)")

    pred = spectral_clustering(A, k=3, seed=0)
    acc = clustering_accuracy(true, pred)
    print(f"  spectral clustering accuracy = {acc*100:.1f}% (recovered the communities)")
    print(f"  modularity   Q(true)={modularity(A, true):.3f}  Q(pred)={modularity(A, pred):.3f}")
    print(f"  normalized cut(true)={normalized_cut(A, true):.3f}  "
          f"vs random={normalized_cut(A, np.random.default_rng(0).integers(3, size=90)):.3f}")

    print("\n=== Fiedler bipartition of a two-clique barbell ===")
    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]          # clique A: 0-4
    edges += [(i, j) for i in range(5, 10) for j in range(i + 1, 10)]    # clique B: 5-9
    edges += [(4, 5)]                                                    # single bridge
    Ab = adjacency_from_edges(10, edges)
    print(f"  bipartition labels: {spectral_bipartition(Ab)}  (splits the two cliques)")
