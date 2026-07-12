"""Tests for physkit.linalg: orthogonalization and the eigenproblem."""
import numpy as np
from physkit import linalg as la


def test_is_hermitian():
    A = np.array([[2.0, 1j], [-1j, 3.0]])
    assert la.is_hermitian(A)
    assert not la.is_hermitian(np.array([[0.0, 1.0], [0.0, 0.0]]))


def test_gram_schmidt_orthonormal():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 3))
    Q = la.gram_schmidt(A)
    assert np.allclose(Q.conj().T @ Q, np.eye(3), atol=1e-10)


def test_eigosystem_sorted_and_correct():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])          # eigenvalues 1 and 3
    w, V = la.eigosystem(A)
    assert np.allclose(w, [1.0, 3.0])
    for i in range(2):
        assert np.allclose(A @ V[:, i], w[i] * V[:, i])


def test_rayleigh_quotient_is_eigenvalue_on_eigenvector():
    A = np.diag([1.0, 5.0, 9.0])
    e = np.array([0.0, 1.0, 0.0])
    assert np.isclose(la.rayleigh_quotient(A, e).real, 5.0)


def test_power_iteration_finds_dominant():
    A = np.diag([1.0, 2.0, 7.0]).astype(complex)
    lam, x = la.power_iteration(A)
    assert np.isclose(lam.real, 7.0, atol=1e-6)


def test_finite_difference_laplacian_matches_second_derivative():
    # -d^2/dx^2 of sin(kx) is k^2 sin(kx); the discrete operator approximates it
    n, L = 200, np.pi
    dx = L / (n + 1)
    x = np.arange(1, n + 1) * dx
    f = np.sin(x)                                    # k = 1
    D2 = la.finite_difference_laplacian(n, dx)
    approx = -(D2 @ f)
    assert np.allclose(approx, f, atol=2e-3)
