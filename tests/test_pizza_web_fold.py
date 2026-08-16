"""Test the deformable spider-web pizza fold: the mesh loads correctly,
pinned (fold-edge) points track their driving anchor's rotation exactly,
and -- the actual point of building this instead of reusing the rigid
two-panel model -- a free (unpinned) rim point genuinely DEFORMS/LAGS
relative to what a rigid rotation of the same anchor angle would predict,
by an amount much larger than could be numerical noise, and the whole
simulation stays numerically stable throughout."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
from dgs.pizza_web_fold import (
    _half_disk_web_points, _half_disk_web_elements, _edge_point_indices,
    build_web_pizza_model, fold_progress, simulate_web_fold,
)

# 1. mesh generation: point/element/edge-index counts are internally consistent
n_rings, n_spokes = 4, 10
points = _half_disk_web_points(radius=0.12, n_rings=n_rings, n_spokes=n_spokes)
elems = _half_disk_web_elements(n_rings, n_spokes)
edge_idx = _edge_point_indices(n_rings, n_spokes)
assert len(points) == 1 + n_rings * (n_spokes + 1)
assert len(elems) == n_spokes + 2 * n_spokes * (n_rings - 1)
assert len(edge_idx) == 1 + 2 * n_rings   # center + (theta=0, theta=pi) per ring
assert max(max(e) for e in elems) < len(points)   # no out-of-range face index (the exact bug
                                                    # class caught earlier in pizza_to_calzone.py)

# 2. the model actually loads with two flexes
model, edge_idx2 = build_web_pizza_model()
assert model.nflex == 2

# 3. a PINNED (fold-edge) point tracks the anchor's rigid rotation EXACTLY
# -- a point at theta=0 sits ON the hinge's own rotation axis (x-axis), so
# rotating the anchor must not move it at all, regardless of angle
data = mujoco.MjData(model)
hinge_a_adr = model.joint("hinge_a").qposadr[0]
pinned_idx = edge_idx2[1]   # first ring's theta=0 point
r_ring1 = 0.12 / 4
expected_pinned_pos = np.array([r_ring1, 0.0, 0.4])
for angle_deg in (0.0, 45.0, 90.0, 150.0):
    data.qpos[hinge_a_adr] = np.deg2rad(angle_deg)
    mujoco.mj_forward(model, data)
    assert np.allclose(data.flexvert_xpos[pinned_idx], expected_pinned_pos, atol=1e-9)

# 4. fold_progress sweeps from 0 to 150 degrees, monotonically
ts = np.linspace(0, 3, 100)
angles = np.array([fold_progress(t) for t in ts])
assert angles[0] < 5.0
assert angles[-1] > 145.0
assert np.all(np.diff(angles) >= -1e-9)

# 5. the full driven simulation stays numerically stable (no NaN) --
# a real risk for a mass-spring mesh being yanked through a fast fold
result = simulate_web_fold(t_max=2.5)
assert result["any_nan"] is False

# 6. the actual point of this module: a FREE rim point (theta=pi/2,
# maximally far from the pinned fold edge) deviates substantially from
# what a rigid rotation of the anchor would predict -- real drape/lag,
# not just a rigid body wearing a mesh costume
r = 0.12
anchor_z = 0.4
idx_1s = np.argmin(np.abs(result["t"] - 1.0))
theta = np.deg2rad(result["anchor_hinge_deg"][idx_1s])
rigid_predicted = np.array([0.0, r * np.cos(theta), anchor_z + r * np.sin(theta)])
actual = result["rim_a_pos"][idx_1s]
deviation = np.linalg.norm(actual - rigid_predicted)
assert deviation > 0.05   # much larger than could be float/timestep noise (pizza radius is 0.12)

print("all dgs.pizza_web_fold tests passed")
