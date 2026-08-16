"""Test dgs.poisson_2d: the sparse 5-point-stencil finite-difference solver
must (1) reproduce the EXACT uniform field of a parallel-plate capacitor
to near machine precision at any resolution (finite differences are exact
on linear functions, so this isn't a convergence claim, just correctness),
and (2) show genuine O(h^2) convergence toward a curved manufactured
solution as the grid is refined -- the actual claim a PDE solver needs to
back up."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.poisson_2d import (
    laplacian_2d_sparse, solve_poisson, field_from_potential,
    parallel_plate_boundary, manufactured_solution, EPS0,
)

# 1. the assembled operator is the right shape and symmetric (a real discrete
# Laplacian is self-adjoint under Dirichlet/free boundary conditions)
nx, ny = 7, 5
A = laplacian_2d_sparse(nx, ny, dx=0.1, dy=0.1)
assert A.shape == (nx * ny, nx * ny)
assert np.abs((A - A.T)).max() < 1e-12

# 2. parallel-plate capacitor: interior field must be uniform and point from
# +V0 toward -V0 (Ex constant, Ey ~ 0), matching the textbook E = 2V0/L result
nx, ny, L, V0 = 41, 41, 1.0, 1.0
dx = dy = L / (nx - 1)
rho, mask, vals = parallel_plate_boundary(nx, ny, V0=V0)
V = solve_poisson(rho, nx, ny, dx, dy, mask, vals)
Ex, Ey = field_from_potential(V, dx, dy)

interior = (slice(2, -2), slice(2, -2))
E_expected = 2 * V0 / L
assert np.abs(Ex[interior].mean() - E_expected) < 1e-6
assert Ex[interior].std() < 1e-8            # uniform to near machine precision
assert np.abs(Ey[interior]).max() < 1e-8    # no y-component anywhere interior

# boundary conditions were actually respected (not silently ignored)
assert np.allclose(V[:, 0], V0, atol=1e-10)
assert np.allclose(V[:, -1], -V0, atol=1e-10)

# 3. rho=0 (pure Laplace) with these BCs solves the same problem as an
# explicit rho array -- solve_poisson must not implicitly assume anything
# about rho beyond what's passed in
rho_zero = np.zeros((ny, nx))
V_laplace = solve_poisson(rho_zero, nx, ny, dx, dy, mask, vals)
assert np.allclose(V, V_laplace, atol=1e-12)

# 4. manufactured-solution convergence study: max error must shrink with
# grid refinement, and roughly at the 2nd-order rate expected of a 5-point
# stencil (halving h should cut the error by ~4x; allow slack for a coarse,
# non-asymptotic first step)
resolutions = (11, 21, 41, 81)
errors = []
for n in resolutions:
    h = 1.0 / (n - 1)
    rho_m, mask_m, vals_m, V_exact = manufactured_solution(n, n, 1.0, 1.0)
    V_num = solve_poisson(rho_m, n, n, h, h, mask_m, vals_m)
    errors.append(np.abs(V_num - V_exact).max())

assert errors[-1] < errors[0]                      # error shrank overall
assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))  # monotonic

# check convergence ORDER on the two finest (most asymptotic) grids:
# error(h) ~ C h^2  =>  log(err1/err2) / log(h1/h2) ~ 2
h_coarse, h_fine = 1.0 / (resolutions[-2] - 1), 1.0 / (resolutions[-1] - 1)
order = np.log(errors[-2] / errors[-1]) / np.log(h_coarse / h_fine)
assert 1.7 < order < 2.3, f"expected ~2nd-order convergence, got order={order:.2f}"

# 5. rejects malformed input rather than silently reshaping garbage
try:
    solve_poisson(np.zeros((3, 3)), nx, ny, dx, dy, mask, vals)
    assert False, "should have rejected mismatched rho shape"
except ValueError:
    pass

try:
    laplacian_2d_sparse(2, 5, 0.1, 0.1)
    assert False, "should reject a grid too small for a 5-point stencil"
except ValueError:
    pass

print("all dgs.poisson_2d tests passed  (convergence order ~%.2f)" % order)
