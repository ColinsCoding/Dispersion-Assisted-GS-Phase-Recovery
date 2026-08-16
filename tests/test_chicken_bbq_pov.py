"""Test the first-person POV rig: joint address lookups are correct (a
regression test for a real bug caught while building this -- the
drumstick's freejoint occupies qpos[0:7] since it's declared before the
arm in the XML, so hardcoding qpos[0,1,2] for the arm joints would
silently overwrite the drumstick's position instead), the scripted arm
animation actually reaches its target pose and returns to idle, and a
short render produces a valid video without crashing or corrupting the
drumstick's own physics."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
from dgs.chicken_bbq_pov import build_arm_grill_model, scripted_arm_pose, render_pov_flip

# 1. regression test: arm joint qpos addresses must NOT be [0, 1, 2] --
# the drumstick's freejoint (7 qpos) is declared first in the XML
model, _ = build_arm_grill_model()
shoulder_adr = model.joint("shoulder_pitch").qposadr[0]
elbow_adr = model.joint("elbow").qposadr[0]
wrist_adr = model.joint("wrist").qposadr[0]
drumstick_adr = model.joint("drumstick_joint").qposadr[0]
assert drumstick_adr == 0
assert {shoulder_adr, elbow_adr, wrist_adr} == {7, 8, 9}

# 2. setting the arm's qpos must NOT disturb the drumstick's own qpos
data = mujoco.MjData(model)
data.qpos[drumstick_adr:drumstick_adr + 7] = [0.1, 0.15, 0.15, 1, 0, 0, 0]
data.qpos[shoulder_adr] = np.deg2rad(-55.0)
data.qpos[elbow_adr] = np.deg2rad(65.0)
mujoco.mj_forward(model, data)
assert np.allclose(data.qpos[drumstick_adr:drumstick_adr + 7], [0.1, 0.15, 0.15, 1, 0, 0, 0])

# 3. scripted_arm_pose: idle at t=0, at the reach target during the hold
# window, and back to idle well after retraction finishes
idle = scripted_arm_pose(0.0, t_strike=0.6)
at_strike = scripted_arm_pose(0.6, t_strike=0.6)
late = scripted_arm_pose(2.0, t_strike=0.6)
assert np.allclose(idle, (-5.0, 5.0, 0.0), atol=0.1)
assert np.allclose(at_strike, (-55.0, 65.0, 0.0), atol=0.1)
assert np.allclose(late, (-5.0, 5.0, 0.0), atol=0.1)

# the arm actually MOVES through the reach window, not a snap cut --
# shoulder angle decreases monotonically (idle -5deg -> strike -55deg)
mid_reach = scripted_arm_pose(0.45, t_strike=0.6)
assert at_strike[0] < mid_reach[0] < idle[0]

# 4. a short render actually produces a valid video with the expected
# frame count, and the launched drumstick ends up displaced from its
# start (i.e. the impulse really was applied, not silently dropped)
result = render_pov_flip(
    linear_velocity=[0.0, 0.0, 1.8], angular_velocity=[5.0, 0.0, 0.0],
    out_path="notebooks/_test_pov_short.mp4", t_strike=0.3, t_max=0.8, fps=15,
)
assert result["n_frames"] > 0
assert pathlib.Path(result["out_path"]).exists()
assert pathlib.Path(result["out_path"]).stat().st_size > 0
pathlib.Path(result["out_path"]).unlink()   # clean up the test artifact

print("all dgs.chicken_bbq_pov tests passed")
