"""Test the hand-rolled camera projection math and the F=dp/dt vs F=ma
equivalence: a point at the look-at target must project to screen center,
an offset point must project to the correct side, the two force-integration
methods must agree EXACTLY for constant mass and genuinely DIVERGE for
variable mass, and the spoon's rigid-body rotation conserves the same
invariants dgs.gyroscopes already established."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.kitchen_3d_projection import (
    Camera, Spoon, KitchenVisualizer, integrate_via_force_and_acceleration,
    integrate_via_momentum, integrate_via_momentum_variable_mass,
)
from dgs.laser_tag_raycaster import mouse_look_update

# 1. a point exactly at the look-at target projects to screen center
cam = Camera(eye=[0, -2.0, 0.5], target=[0, 0, 0.0], screen_size=(640, 480))
xy, depth, visible = cam.project(np.array([[0.0, 0.0, 0.0]]))
assert visible[0]
assert abs(xy[0, 0] - 320) < 1e-6
assert abs(xy[0, 1] - 240) < 1e-6

# 2. a point offset in +x (camera's right, given this eye/target/up)
# projects to screen x > center
xy2, _, vis2 = cam.project(np.array([[0.5, 0.0, 0.0]]))
assert vis2[0]
assert xy2[0, 0] > 320

# 3. a point directly BEHIND the camera is correctly marked not visible
behind = cam.eye - (cam.target - cam.eye)
xy3, _, vis3 = cam.project(np.array([behind]))
assert vis3[0] == False

# 4. F=dp/dt and F=ma are IDENTICAL for constant mass -- exactly, not
# approximately, since they're the same equation algebraically
mass = 0.5
force = np.array([0.0, 0.0, -9.80665]) * mass
v0 = np.array([2.0, 0.0, 3.0])
v_a = integrate_via_force_and_acceleration(mass, force, v0, dt=0.01, n_steps=200)
v_p = integrate_via_momentum(mass, force, v0, dt=0.01, n_steps=200)
assert np.max(np.abs(v_a - v_p)) == 0.0

# 5. for VARIABLE mass, naive F=ma (with a fixed mass) and correct F=dp/dt
# genuinely diverge -- the whole point of stating F=dp/dt as the more
# fundamental law
def shrinking_mass(t):
    return max(0.1, mass - 0.15 * t)

v_p_var = integrate_via_momentum_variable_mass(shrinking_mass, force, v0, dt=0.01, n_steps=200)
v_a_naive = integrate_via_force_and_acceleration(mass, force, v0, dt=0.01, n_steps=200)
assert np.max(np.abs(v_p_var - v_a_naive)) > 1.0

# 6. the spoon's rigid-body rotation conserves kinetic energy and |L|^2
# (torque-free case), same invariant dgs.gyroscopes already established --
# an independent re-implementation of Euler's equations should still
# respect the same conservation laws
spoon = Spoon()
spoon.omega_body = np.array([0.3, 12.0, 0.4])
I1, I2, I3 = spoon.I1, spoon.I2, spoon.I3
Ts, L2s = [], []
for _ in range(2000):
    w1, w2, w3 = spoon.omega_body
    Ts.append(0.5 * (I1 * w1 ** 2 + I2 * w2 ** 2 + I3 * w3 ** 2))
    L2s.append((I1 * w1) ** 2 + (I2 * w2) ** 2 + (I3 * w3) ** 2)
    # rotate only (no gravity call here -- isolate the rotational integrator)
    k1 = spoon._euler_rhs(spoon.omega_body)
    k2 = spoon._euler_rhs(spoon.omega_body + 0.001 / 2 * k1)
    k3 = spoon._euler_rhs(spoon.omega_body + 0.001 / 2 * k2)
    k4 = spoon._euler_rhs(spoon.omega_body + 0.001 * k3)
    spoon.omega_body = spoon.omega_body + 0.001 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
Ts, L2s = np.array(Ts), np.array(L2s)
assert (Ts.max() - Ts.min()) / Ts.mean() < 1e-6
assert (L2s.max() - L2s.min()) / L2s.mean() < 1e-6

# 7. Spoon.world_vertices() correctly rotates+translates: at identity
# orientation, world vertices should equal local vertices + position
spoon2 = Spoon(position=(1.0, 2.0, 3.0))
world = spoon2.world_vertices()
assert np.allclose(world, spoon2.local_vertices + np.array([1.0, 2.0, 3.0]))

# 8. free-look mode: cycling through the 3 modes lands back on "first"
# after 3 presses, and the free camera's look direction actually responds
# to mouse_look_update (reused directly from dgs.laser_tag_raycaster, not
# reimplemented)
viz = KitchenVisualizer()
assert viz.MODES[viz.mode_idx] == "first"
viz.cycle_mode(); assert viz.MODES[viz.mode_idx] == "third"
viz.cycle_mode(); assert viz.MODES[viz.mode_idx] == "free"
viz.cycle_mode(); assert viz.MODES[viz.mode_idx] == "first"

viz2 = KitchenVisualizer()
cam_before = viz2._free_camera()
viz2.yaw, viz2.pitch, _ = mouse_look_update(200.0, 0.0, viz2.mouse_sensitivity, viz2.yaw, viz2.pitch, viz2.pitch_limit)
cam_after = viz2._free_camera()
assert not np.allclose(cam_before.target, cam_after.target)   # look direction genuinely changed
assert np.array_equal(cam_before.eye, cam_after.eye)           # eye position is fixed in free-look (mouse-look only, no WASD)

# 9. pitch is correctly clamped -- a huge dy shouldn't produce an
# out-of-range pitch
viz3 = KitchenVisualizer()
viz3.yaw, viz3.pitch, _ = mouse_look_update(0.0, 100000.0, viz3.mouse_sensitivity, viz3.yaw, viz3.pitch, viz3.pitch_limit)
assert abs(viz3.pitch) <= viz3.pitch_limit + 1e-9

# 10. WASD movement: W moves along the horizontal look direction (yaw
# only, pitch ignored -- looking down shouldn't walk you into the floor),
# and the movement is clamped to stay inside the room's floor bounds
viz4 = KitchenVisualizer()
viz4.yaw = 0.0   # facing +x
start = viz4.free_eye.copy()


class FakeKeysW:
    def __getitem__(self, k):
        import pygame as pg
        return k == pg.K_w


viz4.move_free_camera(FakeKeysW(), dt=0.1, speed=1.0)
assert viz4.free_eye[0] > start[0]           # moved in +x (facing +x)
assert abs(viz4.free_eye[1] - start[1]) < 1e-9   # no sideways drift from pure W
assert viz4.free_eye[2] == start[2]           # W/S never changes height

# 11. movement is clamped to the room -- can't walk through a wall
viz5 = KitchenVisualizer()
viz5.free_eye = np.array([1.49, 0.0, 1.2])   # right at the wall
viz5.yaw = 0.0   # facing +x, straight into the wall


class FakeKeysW2:
    def __getitem__(self, k):
        import pygame as pg
        return k == pg.K_w


for _ in range(50):
    viz5.move_free_camera(FakeKeysW2(), dt=0.1, speed=2.0)
x_max = viz5.scene.floor[:, 0].max()
assert viz5.free_eye[0] <= x_max - 0.15 + 1e-9

print("all dgs.kitchen_3d_projection tests passed")
