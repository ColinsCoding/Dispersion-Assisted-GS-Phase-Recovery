"""The 2D Poisson/Laplace equation on a grid: PDE -> finite differences ->
sparse matrix -> linear solve.

Every module elsewhere in this repo that touches V(r) (griffiths.potentials,
dgs.pde_em, dgs.electrostatics_multipoles) solves Laplace's equation
ANALYTICALLY, by separation of variables or a multipole series -- powerful,
but only where the geometry has enough symmetry for that to work. Two
electrode strips held at different voltages inside a grounded box has no
such symmetry. This module is the general-purpose fallback: discretize

    eps0 * Laplacian(V) = -rho

on a regular (nx, ny) grid with the standard 5-point stencil,

    (V[i,j+1] + V[i,j-1] + V[i+1,j] + V[i-1,j] - 4 V[i,j]) / h^2 = -rho[i,j]/eps0,

which is one linear equation per grid point, i.e. a big sparse matrix
equation A v = b (v = V flattened row-major, index = i*nx+j). Dirichlet
boundary conditions (electrodes, grounded walls) are enforced by overwriting
those rows of A with an identity row and b with the prescribed voltage --
the standard "pin the boundary" trick, same pattern as
dgs.grill_heat_equation's sparse FEM boundary handling.

scipy.sparse.linalg.spsolve does the linear solve; E = -grad(V) via
np.gradient recovers the field once V is known. py-3.13 (scipy 1.18 has
scipy.sparse; confirmed working, see memory note on scipy availability).
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

EPS0 = 8.8541878128e-12   # F/m, vacuum permittivity


def _index(i, j, nx):
    """Flatten a (row i, col j) grid point to v's 1D index, row-major."""
    return i * nx + j


def laplacian_2d_sparse(nx, ny, dx, dy):
    """Assemble the 5-point-stencil discrete Laplacian on an (ny, nx) grid as a
    sparse (nx*ny, nx*ny) matrix acting on the flattened potential -- no
    boundary conditions baked in yet (that's solve_poisson's job, which
    overwrites the relevant rows). Points on the physical edge simply have
    fewer neighbors wired in (a one-sided stencil); solve_poisson always pins
    the true domain boundary to a Dirichlet value, so those rows get
    overwritten anyway and this edge case never matters in practice."""
    if nx < 3 or ny < 3:
        raise ValueError("need at least a 3x3 grid for a 5-point stencil to make sense")
    n = nx * ny
    A = lil_matrix((n, n))
    inv_dx2, inv_dy2 = 1.0 / dx ** 2, 1.0 / dy ** 2
    for i in range(ny):
        for j in range(nx):
            k = _index(i, j, nx)
            A[k, k] = -2 * inv_dx2 - 2 * inv_dy2
            if j > 0:
                A[k, _index(i, j - 1, nx)] = inv_dx2
            if j < nx - 1:
                A[k, _index(i, j + 1, nx)] = inv_dx2
            if i > 0:
                A[k, _index(i - 1, j, nx)] = inv_dy2
            if i < ny - 1:
                A[k, _index(i + 1, j, nx)] = inv_dy2
    return A.tocsr()


def solve_poisson(rho, nx, ny, dx, dy, boundary_mask, boundary_values, eps0=EPS0):
    """Solve eps0 * Laplacian(V) = -rho on an (ny, nx) grid, i.e. build
    A v = b from laplacian_2d_sparse and then pin every grid point where
    boundary_mask is True to boundary_values there (electrodes, grounded
    walls -- any Dirichlet condition). rho=0 everywhere recovers the pure
    Laplace (charge-free) case. Returns V reshaped back to (ny, nx)."""
    rho = np.asarray(rho, float)
    boundary_mask = np.asarray(boundary_mask, bool)
    boundary_values = np.asarray(boundary_values, float)
    if rho.shape != (ny, nx) or boundary_mask.shape != (ny, nx) or boundary_values.shape != (ny, nx):
        raise ValueError("rho, boundary_mask, boundary_values must all be shape (ny, nx)")

    A = laplacian_2d_sparse(nx, ny, dx, dy).tolil()
    b = (-rho / eps0).reshape(-1)
    mask_flat = boundary_mask.reshape(-1)
    val_flat = boundary_values.reshape(-1)

    pinned = np.where(mask_flat)[0]
    for k in pinned:
        A.rows[k] = [k]
        A.data[k] = [1.0]
        b[k] = val_flat[k]

    V = spsolve(A.tocsr(), b)
    return V.reshape(ny, nx)


def field_from_potential(V, dx, dy):
    """E = -grad(V) by central differences. Returns (Ex, Ey), each shape
    matching V. np.gradient(V, dy, dx) differentiates axis 0 (rows, y) by dy
    and axis 1 (cols, x) by dx, matching the (ny, nx) row-major convention
    used throughout this module."""
    dVdy, dVdx = np.gradient(np.asarray(V, float), dy, dx)
    return -dVdx, -dVdy


def parallel_plate_boundary(nx, ny, V0=1.0):
    """A textbook two-electrode geometry: the left column of the grid held at
    +V0, the right column at -V0, top/bottom rows held at the LINEAR
    interpolation between them (i.e. the exact 1D parallel-plate solution
    V(x) = V0*(1 - 2x/(nx-1))) so the whole boundary is consistent with the
    known-exact uniform-field solution. Returns (rho, boundary_mask,
    boundary_values) ready for solve_poisson."""
    rho = np.zeros((ny, nx))
    boundary_mask = np.zeros((ny, nx), bool)
    boundary_values = np.zeros((ny, nx))

    x_frac = np.linspace(0, 1, nx)
    v_exact_row = V0 * (1 - 2 * x_frac)   # +V0 at j=0, -V0 at j=nx-1, linear between

    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_values[:, :] = v_exact_row[np.newaxis, :]
    return rho, boundary_mask, boundary_values


def manufactured_solution(nx, ny, Lx, Ly):
    """A Poisson problem with a KNOWN exact solution, for a genuine numerical
    convergence study (not just a sanity check): pick
        V_exact(x,y) = sin(pi x / Lx) * sin(pi y / Ly),
    which vanishes on the whole boundary (simple Dirichlet BC = 0 everywhere)
    and satisfies Laplacian(V_exact) = -((pi/Lx)^2 + (pi/Ly)^2) * V_exact
    exactly, so the matching charge density is
        rho = eps0 * ((pi/Lx)^2 + (pi/Ly)^2) * V_exact.
    Returns (rho, boundary_mask, boundary_values, V_exact) all shape (ny,nx);
    solve_poisson's output should converge to V_exact at the standard 2nd-
    order rate as the grid is refined (5-point stencil is O(h^2))."""
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)   # shape (ny, nx)
    V_exact = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    k2 = (np.pi / Lx) ** 2 + (np.pi / Ly) ** 2
    rho = EPS0 * k2 * V_exact

    boundary_mask = np.zeros((ny, nx), bool)
    boundary_mask[:, 0] = boundary_mask[:, -1] = True
    boundary_mask[0, :] = boundary_mask[-1, :] = True
    boundary_values = np.zeros((ny, nx))   # V_exact is 0 on this boundary by construction
    return rho, boundary_mask, boundary_values, V_exact


if __name__ == "__main__":
    print("=== parallel-plate capacitor (exact linear solution) ===")
    nx, ny, L = 41, 41, 1.0
    dx = dy = L / (nx - 1)
    rho, mask, vals = parallel_plate_boundary(nx, ny, V0=1.0)
    V = solve_poisson(rho, nx, ny, dx, dy, mask, vals)
    Ex, Ey = field_from_potential(V, dx, dy)
    print(f"  grid {nx}x{ny}: interior V range [{V.min():.4f}, {V.max():.4f}]")
    print(f"  Ex (should be uniform, ~2*V0/L = {2*1.0/L:.4f}): "
          f"mean={Ex.mean():.4f}, std={Ex.std():.2e}")
    print(f"  Ey (should be ~0): max|Ey| = {np.abs(Ey).max():.2e}")

    print("\n=== manufactured-solution convergence study ===")
    for n in (11, 21, 41, 81):
        rho, mask, vals, V_exact = manufactured_solution(n, n, 1.0, 1.0)
        h = 1.0 / (n - 1)
        V_num = solve_poisson(rho, n, n, h, h, mask, vals)
        err = np.abs(V_num - V_exact).max()
        print(f"  n={n:3d}  h={h:.4f}  max error = {err:.3e}")
