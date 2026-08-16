"""Test the three independent heat-equation solvers agree with each
other, not just that each runs without crashing: the Bessel series is
exact for the axisymmetric problem, the radial finite-difference solver
must match it closely, and the triangulated FEM solver (which does NOT
assume radial symmetry) must independently arrive at essentially the
same answer despite using a completely different discretization."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.grill_heat_equation import (
    BesselHeatSolution, RadialFiniteDifferenceSolver, TriangulatedFEMSolver,
    searzone_initial_condition, run_c_radial_solver,
)

R, alpha = 0.15, 4.0e-6
f_initial = searzone_initial_condition(peak_temp=250.0, sear_radius=0.05)

# 1. Bessel series reconstructs the initial condition at t=0 in the
# interior (boundary point is expected to mismatch -- f_initial(R) != 0
# exactly, while every basis function IS exactly zero there by
# construction; this is the standard Gibbs-phenomenon-adjacent behavior
# of a Dirichlet eigenfunction expansion, not a bug)
bessel = BesselHeatSolution(R, alpha, f_initial, n_modes=40)
r_interior = np.linspace(0, 0.8 * R, 10)
reconstructed = bessel.temperature(r_interior, 0.0)
exact = f_initial(r_interior)
assert np.max(np.abs(reconstructed - exact)) < 0.5

# the boundary point genuinely does NOT match at t=0 (confirms the
# reconstruction check above isn't vacuously passing near the edge too)
assert bessel.temperature(R, 0.0)[0] < 1e-6          # basis forces exactly 0
assert f_initial(R) > 1e-3                             # but f_initial(R) isn't exactly 0

# 2. the temperature decays over time (energy leaves through the Dirichlet
# boundary) and stays radially decreasing (hottest at center, always)
T_center_early = bessel.temperature(0.0, 10.0)[0]
T_center_late = bessel.temperature(0.0, 200.0)[0]
assert T_center_late < T_center_early
r_profile = np.linspace(0, R, 20)
T_profile = bessel.temperature(r_profile, 100.0)
assert np.all(np.diff(T_profile) <= 1e-9)   # monotonically non-increasing outward

# 3. radial finite-difference matches the exact Bessel solution closely
fd = RadialFiniteDifferenceSolver(R, alpha, f_initial, n_points=200)
T_fd = fd.advance_to(100.0)
T_bessel_at_fd_t = bessel.temperature(fd.r, fd.t)
max_diff_fd = np.max(np.abs(T_fd[:-2] - T_bessel_at_fd_t[:-2]))   # exclude last 2 pts near the boundary
assert max_diff_fd < 0.5

# 4. the triangulated FEM solver -- genuinely 2D, no radial-symmetry
# assumption baked into the solver itself -- still lands close to the
# exact axisymmetric answer at the center
fem = TriangulatedFEMSolver(R, alpha, f_initial, n_r=30, n_theta=60, dt=1.0)
assert fem.n_triangles > 1000   # a real "high number of triangles" mesh
T_fem = fem.advance_to(100.0)
bessel_center_100 = bessel.temperature(0.0, 100.0)[0]
assert abs(T_fem[0] - bessel_center_100) < 2.0   # ~1% of the ~150-200 degree scale

# 5. FEM temperature field is still radially symmetric-ish despite not
# assuming it: points on the SAME polar-grid ring (same discrete radius,
# different angle) should have close to the same temperature. Must pick
# points from one exact ring, not "N nearest by |r-target|" -- that can
# straddle two adjacent rings and pull in a point that's genuinely at a
# different radius, which is a real difference, not a symmetry violation.
angles_r = fem.r_of_points
candidate_r = 0.075
closest_ring_r = angles_r[np.argmin(np.abs(angles_r - candidate_r))]
same_ring_mask = np.abs(angles_r - closest_ring_r) < 1e-9
temps_on_ring = T_fem[same_ring_mask]
assert same_ring_mask.sum() >= 8   # a real ring of points, not a fluke single match
assert np.std(temps_on_ring) < 0.05 * np.mean(temps_on_ring)

# 6. input validation carries through naturally -- R and alpha must be usable
assert BesselHeatSolution(R, alpha, f_initial, n_modes=5).coefficients.shape == (5,)

# 7. the C implementation (dgs/c/grill_heat/grill_heat_fd.c, same Makefile
# project) is a genuinely independent language/toolchain -- it should
# match the Python radial finite-difference solver to near machine
# precision, since both implement the exact same FTCS scheme
T_c = run_c_radial_solver()
assert len(T_c) == len(T_fd)
assert np.max(np.abs(T_c - T_fd)) < 1e-8

print("all dgs.grill_heat_equation tests passed")
