import numpy as np
import pytest
from laplacian import laplacian_2d, helmholtz_operator


def test_laplacian_2d_rejects_small_grid():
    with pytest.raises(ValueError):
        laplacian_2d(2, 5, 0.1, 0.1)


def test_laplacian_2d_rejects_bad_spacing():
    with pytest.raises(ValueError):
        laplacian_2d(5, 5, -0.1, 0.1)


def test_laplacian_2d_is_symmetric():
    L = laplacian_2d(8, 8, 0.1, 0.1).toarray()
    np.testing.assert_allclose(L, L.T)


def test_laplacian_2d_exact_on_quadratic_in_interior():
    # 2nd-order central differences are EXACT for polynomials up to degree 3,
    # so d^2/dx^2(x^2) = 2 should come back exactly (interior points only --
    # the Dirichlet-truncated stencil at the boundary rows is not tested here)
    nx, ny, dx, dy = 21, 21, 0.5, 0.5
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    psi = (X ** 2).reshape(-1, order="C")  # depends only on x
    L = laplacian_2d(nx, ny, dx, dy)
    lap_psi = (L @ psi).reshape(nx, ny, order="C")
    interior = lap_psi[2:-2, 2:-2]
    np.testing.assert_allclose(interior, 2.0, atol=1e-10)


def test_helmholtz_operator_rejects_nonpositive_wavelength():
    n_grid = np.full((5, 5), 1.44)
    with pytest.raises(ValueError):
        helmholtz_operator(n_grid, 0.1, 0.1, wavelength_um=0.0)


def test_helmholtz_operator_rejects_non_2d_grid():
    with pytest.raises(ValueError):
        helmholtz_operator(np.ones(5), 0.1, 0.1, wavelength_um=1.55)


def test_helmholtz_operator_is_symmetric():
    n_grid = np.full((6, 6), 1.44)
    A = helmholtz_operator(n_grid, 0.1, 0.1, wavelength_um=1.55).toarray()
    np.testing.assert_allclose(A, A.T)


def test_helmholtz_operator_diagonal_increases_with_index():
    n_low = np.full((6, 6), 1.44)
    n_high = np.full((6, 6), 3.4)
    A_low = helmholtz_operator(n_low, 0.1, 0.1, wavelength_um=1.55)
    A_high = helmholtz_operator(n_high, 0.1, 0.1, wavelength_um=1.55)
    # higher index -> larger k0^2*n^2 diagonal contribution everywhere
    assert np.all(A_high.diagonal() > A_low.diagonal())


def test_helmholtz_operator_equals_laplacian_plus_index_term():
    n_grid = np.array([[1.44, 3.4, 1.44], [1.44, 3.4, 1.44], [1.44, 3.4, 1.44]])
    dx = dy = 0.1
    wavelength_um = 1.55
    A = helmholtz_operator(n_grid, dx, dy, wavelength_um).toarray()
    L = laplacian_2d(3, 3, dx, dy).toarray()
    k0 = 2 * np.pi / wavelength_um
    expected = L + np.diag((k0 ** 2 * n_grid ** 2).reshape(-1, order="C"))
    np.testing.assert_allclose(A, expected)
