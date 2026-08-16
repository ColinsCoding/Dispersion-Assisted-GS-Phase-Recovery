"""Test the potential-field voxel sandbox: the analytic scalar force
actually matches -grad(V) computed numerically (not just asserted to be a
gradient), the vector (Lorentz) force is genuinely perpendicular to
velocity, the terrain generates real block geoms, and the full simulation
stays stable while producing real curved trajectories (not just orbs
falling straight into a potential well)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dgs.potential_field_sandbox import (
    build_voxel_terrain_xml, build_sandbox_model, make_beacons,
    scalar_potential, scalar_force, vector_potential_force, simulate_sandbox, DEVICE,
)

# 1. terrain generation produces a nonempty set of block geoms with sensible heights
terrain_xml, heights = build_voxel_terrain_xml(grid_n=6, seed=0)
assert "<geom" in terrain_xml
assert heights.shape == (6, 6)
assert heights.min() >= 1 and heights.max() <= 4

# 2. the model loads with the requested number of free-flying orbs
model, _ = build_sandbox_model(n_orbs=10, grid_n=6, seed=0)
n_orb_freejoints = sum(1 for j in range(model.njnt) if model.jnt_type[j] == 0)  # mjJNT_FREE == 0
assert n_orb_freejoints == 10

# 3. the ANALYTIC scalar force matches a NUMERICAL finite-difference
# gradient of the same scalar potential -- the actual point of calling it
# a "scalar potential force" rather than an arbitrary force function
beacon_pos, beacon_strength = make_beacons(n=3, seed=5)
test_points = torch.tensor([[0.3, -0.2, 3.0], [1.1, 0.5, 2.5], [-0.8, 0.9, 4.0]],
                            dtype=torch.float32, device=DEVICE)

analytic = scalar_force(test_points, beacon_pos, beacon_strength).cpu().numpy()

h = 1e-3
numeric = np.zeros((3, 3))
for axis in range(3):
    dx = torch.zeros_like(test_points)
    dx[:, axis] = h
    V_plus = scalar_potential(test_points + dx, beacon_pos, beacon_strength).cpu().numpy()
    V_minus = scalar_potential(test_points - dx, beacon_pos, beacon_strength).cpu().numpy()
    numeric[:, axis] = -(V_plus - V_minus) / (2 * h)

assert np.max(np.abs(analytic - numeric)) < 1e-2, \
    f"analytic scalar force should match -grad(V) by finite difference, max diff={np.max(np.abs(analytic-numeric))}"

# 4. the vector (Lorentz) force is genuinely perpendicular to velocity for
# every test case -- the defining property of a magnetic-style force,
# checked directly rather than assumed
velocities = torch.tensor([[1.0, 0.0, 0.0], [0.3, -0.7, 0.2], [0.0, 0.0, 2.0]],
                           dtype=torch.float32, device=DEVICE)
B_field = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float32, device=DEVICE)
F_vec = vector_potential_force(velocities, B_field, charge=1.0)
dot_products = (F_vec * velocities).sum(-1).abs().cpu().numpy()
assert np.max(dot_products) < 1e-4, "Lorentz force F=q(v x B) must be perpendicular to v"

# 5. the full simulation stays numerically stable
result = simulate_sandbox(t_max=2.0, n_orbs=20, grid_n=6, seed=0)
assert result["any_nan"] is False

# 6. orbs show real curved trajectories -- path length well beyond the
# straight-line distance between start and end, the checkable signature of
# the vector-potential swirl term actually doing something (not just a
# scalar force pulling orbs straight to a potential well)
curl_ratio = result["path_length"] / np.maximum(result["straight_dist"], 1e-6)
assert np.median(curl_ratio) > 2.0, f"expected clearly curved trajectories, median ratio={np.median(curl_ratio):.2f}"

print("all dgs.potential_field_sandbox tests passed")
