"""Linear-algebra helpers reused from chapter 06 (linear algebra) onward.

The physics motivation is uniform: an observable is a Hermitian operator, its matrix is symmetric (or
Hermitian), and its eigenpairs are the measurable values and stationary states. These routines are
the tools chapters 07, 13, 17, and 18 call to turn a Hamiltonian matrix into a spectrum.
"""

import numpy as np


def is_hermitian(A, tol=1e-10):
    """True if A equals its conjugate transpose within tolerance."""
    A = np.asarray(A)
    return np.allclose(A, A.conj().T, atol=tol)


def gram_schmidt(vectors):
    """Orthonormalize the columns of `vectors` (modified Gram-Schmidt). Returns an orthonormal
    matrix Q with the same column span."""
    A = np.asarray(vectors, dtype=complex)
    n, k = A.shape
    Q = np.zeros((n, k), dtype=complex)
    for j in range(k):
        v = A[:, j].copy()
        for i in range(j):
            v -= np.vdot(Q[:, i], v) * Q[:, i]
        norm = np.linalg.norm(v)
        if norm < 1e-14:
            raise ValueError("vectors are linearly dependent")
        Q[:, j] = v / norm
    return Q


def rayleigh_quotient(A, x):
    """The Rayleigh quotient x^H A x / x^H x -- the expectation value of operator A in state x."""
    x = np.asarray(x, dtype=complex)
    return complex(np.vdot(x, np.asarray(A) @ x) / np.vdot(x, x))


def eigosystem(A):
    """Eigenvalues (ascending) and eigenvectors (columns) of a Hermitian matrix A.

    Wraps numpy.linalg.eigh but sorts ascending and returns real eigenvalues, matching the physics
    convention that energy levels are listed from the ground state up.
    """
    A = np.asarray(A)
    if not is_hermitian(A):
        raise ValueError("matrix must be Hermitian for a real spectrum")
    w, V = np.linalg.eigh(A)
    idx = np.argsort(w.real)
    return w[idx].real, V[:, idx]


def power_iteration(A, iters=1000, tol=1e-12, seed=0):
    """Dominant eigenvalue and eigenvector of A by power iteration -- the bare mechanism behind
    many large-scale eigensolvers. Returns (eigenvalue, eigenvector)."""
    A = np.asarray(A, dtype=complex)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[0]) + 0j
    x /= np.linalg.norm(x)
    lam_old = 0.0
    for _ in range(iters):
        y = A @ x
        x = y / np.linalg.norm(y)
        lam = rayleigh_quotient(A, x)
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam
    return lam, x


def finite_difference_laplacian(n, dx):
    """The 1-D second-derivative operator on n interior points with spacing dx (Dirichlet ends):
    tridiagonal (1, -2, 1)/dx^2. This is the discrete kinetic-energy operator of chapters 13-18."""
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    return (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / dx ** 2
