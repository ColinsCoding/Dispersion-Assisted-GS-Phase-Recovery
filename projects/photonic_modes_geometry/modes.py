"""modes.py -- solve A(p) E = beta^2 E (laplacian.py's discretized scalar
Helmholtz operator) for its largest-eigenvalue modes, the physically
guided ones (see notebook Section 6 for the derivation of why "largest
beta^2" means "most confined mode").
"""
import numpy as np
from scipy.sparse.linalg import eigsh
from laplacian import helmholtz_operator


def solve_modes(n_grid: np.ndarray, dx: float, dy: float, wavelength_um: float, n_modes: int = 6):
    """Solve for the n_modes largest-beta^2 eigenmodes of the scalar
    Helmholtz operator built from n_grid. Returns a list of dicts, sorted
    from most- to least-confined, each with:
      'beta_sq' -- the eigenvalue (propagation constant squared, 1/um^2)
      'n_eff'   -- effective index beta/k0 (nan if beta_sq <= 0)
      'psi'     -- the mode field, reshaped to n_grid's (nx, ny) shape and
                   normalized so sum(psi^2)*dx*dy = 1 (a SIMPLIFIED NUMERICAL
                   METRIC -- see notebook Section 9 for a rigorous power
                   normalization)
    """
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if n_grid.ndim != 2:
        raise ValueError("n_grid must be a 2D array")
    nx, ny = n_grid.shape
    if n_modes >= nx * ny - 1:
        raise ValueError(f"n_modes={n_modes} too large for a {nx}x{ny} grid (ARPACK needs k < N-1)")

    A = helmholtz_operator(n_grid, dx, dy, wavelength_um)
    k0 = 2 * np.pi / wavelength_um

    vals, vecs = eigsh(A, k=n_modes, which="LA")
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]

    modes = []
    for i in range(n_modes):
        psi = vecs[:, i].reshape(nx, ny, order="C")
        norm = np.sqrt(np.sum(psi ** 2) * dx * dy)
        psi = psi / norm if norm > 0 else psi
        beta_sq = vals[i]
        n_eff = np.sqrt(beta_sq) / k0 if beta_sq > 0 else np.nan
        modes.append({"beta_sq": beta_sq, "n_eff": n_eff, "psi": psi})
    return modes
