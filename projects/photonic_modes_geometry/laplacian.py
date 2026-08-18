"""laplacian.py -- finite-difference discretization of the scalar Helmholtz
eigenproblem  grad^2(psi) + k0^2 n(x,y)^2 psi = beta^2 psi  into an explicit
sparse matrix A(p) such that A(p) E = beta^2 E (the bookkeeping fixed in
notebook Section 2: p=geometry/material params, E=flattened field vector,
lambda=beta^2).

Dirichlet boundary conditions (psi=0 at the domain edge) -- the simplest
choice, only valid when the mode has decayed to ~0 well before the edge.
Flattening is row-major ('C' order): field[i,j] -> vector[i*ny+j].
"""
import numpy as np
import scipy.sparse as sp


def laplacian_2d(nx: int, ny: int, dx: float, dy: float):
    """Sparse 2D Laplacian (5-point stencil, Dirichlet BC) via a Kronecker
    sum of the 1D second-difference matrices. Returns an (nx*ny, nx*ny)
    scipy.sparse.csr_matrix acting on a row-major-flattened field vector."""
    if nx < 3 or ny < 3:
        raise ValueError(f"nx={nx}, ny={ny}: need >= 3 points per axis for a Laplacian stencil")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")
    Dxx = sp.diags([np.ones(nx - 1), -2 * np.ones(nx), np.ones(nx - 1)], [-1, 0, 1]) / dx ** 2
    Dyy = sp.diags([np.ones(ny - 1), -2 * np.ones(ny), np.ones(ny - 1)], [-1, 0, 1]) / dy ** 2
    L = sp.kron(Dxx, sp.identity(ny)) + sp.kron(sp.identity(nx), Dyy)
    return L.tocsr()


def helmholtz_operator(n_grid: np.ndarray, dx: float, dy: float, wavelength_um: float):
    """Assemble A(p) = Laplacian + k0^2 * diag(n_grid^2), the discretized
    scalar Helmholtz operator such that A(p) E = beta^2 E. n_grid is a 2D
    refractive-index array from geometry.py. Returns a sparse csr_matrix."""
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive")
    if n_grid.ndim != 2:
        raise ValueError("n_grid must be a 2D array")
    nx, ny = n_grid.shape
    L = laplacian_2d(nx, ny, dx, dy)
    k0 = 2 * np.pi / wavelength_um
    n_flat = n_grid.reshape(-1, order="C")
    A = L + sp.diags(k0 ** 2 * n_flat ** 2)
    return A.tocsr()
