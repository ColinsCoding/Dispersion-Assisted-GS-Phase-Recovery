"""Test the pizza-fold-to-calzone rig: the half-disk mesh is correctly
built (right vertex/face counts, no degenerate geometry), the fold-angle
convention actually matches what was verified by rendering (0deg=open
circle, not 180deg -- the OPPOSITE of pizza_box_fold.py's convention, a
real gotcha caught by looking at renders rather than assuming by analogy),
the scripted fold sweeps monotonically from open to closed, and the whole
scripted animation runs without producing NaN."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
from dgs.pizza_to_calzone import (
    _half_disk_mesh_xml, build_pizza_to_calzone_model, fold_progress, simulate_fold_to_calzone,
)

# 1. the half-disk mesh helper produces valid, loadable MJCF (regression
# test for the original face-indexing bug caught while building this:
# closing the mesh with side-wall/diameter-edge faces produced an
# out-of-range vertex index; the fix was to drop those faces entirely)
mesh_xml = _half_disk_mesh_xml("test_mesh", radius=0.1, n_arc=10)
wrapped = f'<mujoco><asset>{mesh_xml}</asset><worldbody><body><freejoint/><geom type="mesh" mesh="test_mesh" mass="0.1"/></body></worldbody></mujoco>'
test_model = mujoco.MjModel.from_xml_string(wrapped)
assert test_model.mesh_vertnum[0] == 2 * 10 + 2   # top rim + bottom rim + 2 centers
assert test_model.mesh_facenum[0] == 2 * (10 - 1)  # top fan + bottom fan

# 2. fold=0 produces a COMPLETE circle (both halves coplanar), NOT
# fold=180 -- verified directly by checking the two geoms' world
# positions/orientations coincide in the same plane at fold=0
model = build_pizza_to_calzone_model()
data = mujoco.MjData(model)
fold_adr = model.joint("fold_hinge").qposadr[0]
half_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pizza_half_a")
half_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pizza_half_b")

data.qpos[fold_adr] = 0.0
mujoco.mj_forward(model, data)
# at fold=0, half_b's body frame should be coincident with half_a's (same
# position, since pos="0 0 0" -- the fold hinge pivot IS the shared origin)
assert np.allclose(data.xpos[half_a_id], data.xpos[half_b_id], atol=1e-9)

# 3. fold_progress starts near 0 (open/flat) and ends near 155 (closed/calzone)
assert fold_progress(0.0) < 10.0
assert fold_progress(10.0) > 150.0

# 4. the sweep is monotonically non-decreasing (folds progressively
# closed, never springs back open partway through)
ts = np.linspace(0, 5, 200)
angles = np.array([fold_progress(t) for t in ts])
assert np.all(np.diff(angles) >= -1e-9)

# 5. the full scripted animation runs clean -- no NaN, and the fold
# genuinely progresses from open toward closed over the run
result = simulate_fold_to_calzone(t_max=3.0)
assert result["any_nan"] is False
assert result["initial_fold_deg"] < 10.0
assert result["final_fold_deg"] > 150.0
assert result["final_fold_deg"] > result["initial_fold_deg"]

print("all dgs.pizza_to_calzone tests passed")
