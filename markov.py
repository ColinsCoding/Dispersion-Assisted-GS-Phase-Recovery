"""Markov chains -- the probabilistic state machine (linear algebra at work).

Three kinds of state machine, one idea (a matrix acting on a state vector):
  * deterministic FSM (digital_logic): state -> state, one arrow out.
  * **Markov chain (here): a probability *distribution* -> distribution, via a
    row-stochastic transition matrix P (rows sum to 1).**
  * unitary quantum circuit (qubits): amplitudes -> amplitudes, via U.

A Markov chain forgets where it started: iterate P and the distribution settles
onto the **stationary distribution** pi (pi P = pi), the left eigenvector of P
with eigenvalue 1 (Perron-Frobenius). That's PageRank, equilibrium statistical
mechanics, and the noise model behind a lot of ML. NumPy only. Education.
"""

import numpy as np


def is_stochastic(P, tol=1e-9):
    """True if P is a valid row-stochastic matrix: non-negative, rows sum to 1."""
    P = np.asarray(P, dtype=float)
    return (P.ndim == 2 and P.shape[0] == P.shape[1]
            and np.all(P >= -tol) and np.allclose(P.sum(axis=1), 1.0, atol=tol))


def step(dist, P):
    """One time step: new distribution = dist @ P (row vector times matrix)."""
    dist, P = np.asarray(dist, dtype=float), np.asarray(P, dtype=float)
    if not is_stochastic(P):
        raise ValueError("P must be row-stochastic")
    if abs(dist.sum() - 1.0) > 1e-6 or np.any(dist < -1e-9):
        raise ValueError("dist must be a probability vector (non-negative, sums to 1)")
    return dist @ P


def evolve(dist, P, n):
    """Apply n steps of the chain; returns the distribution after n steps."""
    if n < 0:
        raise ValueError("n must be >= 0")
    d = np.asarray(dist, dtype=float)
    for _ in range(n):
        d = step(d, P)
    return d


def stationary_distribution(P):
    """The stationary distribution pi with pi P = pi: the left eigenvector of P
    for eigenvalue 1 (equivalently, the eigenvector of P^T), normalized to sum 1."""
    P = np.asarray(P, dtype=float)
    if not is_stochastic(P):
        raise ValueError("P must be row-stochastic")
    vals, vecs = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(vals - 1.0)))           # eigenvalue closest to 1
    pi = np.real(vecs[:, i])
    pi = np.abs(pi)                                  # Perron eigenvector is sign-definite
    return pi / pi.sum()


def hitting_time_matrix(P):
    """Expected number of steps to first reach state j starting from i (i!=j).

    Solves H[i,j] = 1 + sum_{k!=j} P[i,k] H[k,j], with H[j,j]=0, per target j."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    H = np.zeros((n, n))
    for j in range(n):
        idx = [i for i in range(n) if i != j]
        A = np.eye(n - 1) - P[np.ix_(idx, idx)]
        h = np.linalg.solve(A, np.ones(n - 1))
        for k, i in enumerate(idx):
            H[i, j] = h[k]
    return H


if __name__ == "__main__":
    # sunny/rainy weather chain
    P = np.array([[0.9, 0.1],
                  [0.5, 0.5]])
    pi = stationary_distribution(P)
    print("transition P =\n", P)
    print("stationary distribution pi =", np.round(pi, 4), "(analytic [5/6, 1/6])")
    print("from 'rainy' [0,1] after 20 steps:", np.round(evolve([0, 1], P, 20), 4))
