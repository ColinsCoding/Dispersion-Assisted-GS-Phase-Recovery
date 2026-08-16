"""First-person POV: real 3D arm+tongs geometry (MuJoCo bodies/joints), a
first-person camera looking down at the grill, layered on top of the same
validated drumstick physics as dgs/chicken_bbq_3d.py.

DIVISION OF LABOR (deliberate, not an oversight): the arm is a purely
KINEMATIC/scripted puppet -- its joint angles are set directly every frame
from scripted_arm_pose(), not driven by actuators or physics. Real
human-arm dynamics (muscle models, PD control) is a much bigger
biomechanics project than "does the tongs animation look right," and
nothing about the actual physics this repo cares about depends on the arm
having correct dynamics. The drumstick, by contrast, is a genuine free
rigid body: once the scripted tongs "release" it (apply_flip_impulse_3d,
timed to the arm's swing), it tumbles under real torque-free rotation
(Euler's equations, exactly as validated in dgs.gyroscopes and
dgs.chicken_bbq_3d) and lands via real MuJoCo contact physics with the
grill. Two different levels of rigor for two different questions.

Camera/geometry note found empirically while building this: creating a
NEW mujoco.Renderer object inside a per-frame loop occasionally produced
fully black frames even at a valid, non-occluding camera pose -- a
renderer lifecycle issue, not a geometry/occlusion bug (confirmed by
re-rendering the identical pose with a single, reused Renderer instance,
which worked correctly). Always create one Renderer and reuse it.
"""

import numpy as np
import mujoco

from dgs.chicken_bbq_3d import drumstick_inertia


def build_arm_grill_model(mass=0.12, half_sizes=(0.02, 0.03, 0.09), dt=0.0005):
    """Grill (with contact) + a free drumstick body + a 3-joint kinematic
    arm (shoulder pitch, elbow, wrist) ending in two tong prongs, and TWO
    cameras: 'pov' (first-person, positioned and oriented -- validated
    against black-frame/occlusion issues empirically -- to look down at
    the grill with the arm entering from the bottom of the frame), and
    'third_person' (an external over-the-shoulder view watching the whole
    arm+grill+drumstick scene from behind and above, same empirical
    validate-by-screenshot process used for the first-person camera)."""
    Ix, Iy, Iz = drumstick_inertia(mass, half_sizes)
    sx, sy, sz = half_sizes
    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.80665" timestep="{dt}" integrator="RK4"/>
      <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
      <worldbody>
        <light pos="0 -0.1 2" dir="0 0.2 -1" diffuse="0.8 0.8 0.7"/>
        <light pos="0.5 -0.6 1.2" dir="-0.4 0.5 -1" diffuse="0.5 0.5 0.5"/>
        <camera name="pov" pos="0 -0.55 0.65" xyaxes="1 0 0  0 0.75 0.55"/>
        <camera name="third_person" pos="0.75 -0.85 1.05" xyaxes="0.75 0.66 0  -0.35 0.4 0.85"/>

        <geom name="grill" type="plane" size="0.5 0.5 0.05" pos="0 0.15 0"
              rgba="0.35 0.25 0.2 1" friction="0.6 0.005 0.0001"/>

        <body name="drumstick" pos="0.1 0.15 0.15">
          <freejoint name="drumstick_joint"/>
          <inertial pos="0 0 0" mass="{mass}" diaginertia="{Ix} {Iy} {Iz}"/>
          <geom type="box" size="{sx} {sy} {sz}" mass="{mass}" rgba="0.75 0.35 0.25 1"
                friction="0.6 0.005 0.0001"/>
        </body>

        <body name="shoulder" pos="0.1 -0.35 0.55">
          <joint name="shoulder_pitch" type="hinge" axis="1 0 0" range="-90 20"/>
          <geom type="capsule" fromto="0 0 0  0 0.18 -0.02" size="0.026" rgba="0.9 0.75 0.6 1" mass="1.5"
                contype="0" conaffinity="0"/>
          <body name="forearm" pos="0 0.18 -0.02">
            <joint name="elbow" type="hinge" axis="1 0 0" range="-30 110"/>
            <geom type="capsule" fromto="0 0 0  0 0.17 -0.06" size="0.022" rgba="0.9 0.75 0.6 1" mass="1.0"
                  contype="0" conaffinity="0"/>
            <body name="tongs" pos="0 0.17 -0.06">
              <joint name="wrist" type="hinge" axis="1 0 0" range="-60 60"/>
              <geom type="box" size="0.015 0.02 0.005" rgba="0.6 0.6 0.65 1" mass="0.1"
                    contype="0" conaffinity="0"/>
              <geom type="capsule" fromto="0.012 0.02 0  0.02 0.09 -0.01" size="0.007" rgba="0.8 0.8 0.85 1"
                    mass="0.05" contype="0" conaffinity="0"/>
              <geom type="capsule" fromto="-0.012 0.02 0  -0.02 0.09 -0.01" size="0.007" rgba="0.8 0.8 0.85 1"
                    mass="0.05" contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml), (Ix, Iy, Iz)


def scripted_arm_pose(t, t_strike=0.6, reach_duration=0.35, idle_deg=(-5.0, 5.0, 0.0),
                       reach_deg=(-55.0, 65.0, 0.0)):
    """(shoulder, elbow, wrist) angles in degrees at time t: ease from
    idle to reach as t approaches t_strike, hold briefly, then ease back
    to idle -- a real scripted keyframe animation, not just a snap cut."""
    if t < t_strike - reach_duration:
        frac = 0.0
    elif t < t_strike:
        frac = (t - (t_strike - reach_duration)) / reach_duration
    elif t < t_strike + 0.15:
        frac = 1.0
    elif t < t_strike + 0.15 + reach_duration:
        frac = 1.0 - (t - (t_strike + 0.15)) / reach_duration
    else:
        frac = 0.0
    ease = 0.5 - 0.5 * np.cos(np.clip(frac, 0.0, 1.0) * np.pi)   # smootherstep-ish easing
    return tuple(idle_deg[i] + ease * (reach_deg[i] - idle_deg[i]) for i in range(3))


def render_pov_flip(linear_velocity, angular_velocity, out_path, t_strike=0.6,
                     t_max=2.5, fps=30, mass=0.12, half_sizes=(0.02, 0.03, 0.09),
                     camera="pov"):
    """Simulate + render a full POV flip: the arm scripts its reach toward
    the drumstick, at t_strike the drumstick is given real launch velocity
    (same convention as dgs.chicken_bbq_3d: linear_velocity is WORLD
    frame, angular_velocity is BODY-LOCAL frame), and both the arm's
    retraction and the drumstick's real tumble-and-land physics play out
    together.

    camera: "pov" (first-person, default), "third_person" (external
    over-the-shoulder view), or "both" (side-by-side, same simulation
    rendered from both cameras every frame -- identical physics, just two
    camera angles on it)."""
    import imageio.v3 as iio

    if camera not in ("pov", "third_person", "both"):
        raise ValueError(f"camera must be 'pov', 'third_person', or 'both', got {camera!r}")

    model, (I1, I2, I3) = build_arm_grill_model(mass, half_sizes)
    data = mujoco.MjData(model)

    # Address lookups by NAME, not hardcoded indices -- the drumstick's
    # freejoint (7 qpos: 3 pos + 4 quat) is declared BEFORE the arm's hinge
    # joints in the XML, so the arm joints do NOT start at qpos[0] the way
    # a naive guess would assume (verified directly: qposadr came back
    # [7, 8, 9] for shoulder/elbow/wrist, not [0, 1, 2]).
    drumstick_qvel_adr = model.joint("drumstick_joint").dofadr[0]
    shoulder_qpos_adr = model.joint("shoulder_pitch").qposadr[0]
    elbow_qpos_adr = model.joint("elbow").qposadr[0]
    wrist_qpos_adr = model.joint("wrist").qposadr[0]
    shoulder_dof_adr = model.joint("shoulder_pitch").dofadr[0]

    launched = {"done": False}
    dt = model.opt.timestep
    n_steps = int(t_max / dt)
    steps_per_frame = max(1, int(1 / (fps * dt)))

    renderer = mujoco.Renderer(model, height=480, width=640)   # ONE renderer, reused every frame
    pov_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "pov")
    third_person_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "third_person")
    frames = []

    mujoco.mj_forward(model, data)
    for step in range(n_steps):
        t = step * dt
        shoulder_deg, elbow_deg, wrist_deg = scripted_arm_pose(t, t_strike=t_strike)
        data.qpos[shoulder_qpos_adr] = np.deg2rad(shoulder_deg)
        data.qpos[elbow_qpos_adr] = np.deg2rad(elbow_deg)
        data.qpos[wrist_qpos_adr] = np.deg2rad(wrist_deg)
        data.qvel[shoulder_dof_adr:shoulder_dof_adr + 3] = 0.0   # kinematic puppet -- no residual arm dynamics

        if not launched["done"] and t >= t_strike:
            data.qvel[drumstick_qvel_adr:drumstick_qvel_adr + 3] = linear_velocity
            data.qvel[drumstick_qvel_adr + 3:drumstick_qvel_adr + 6] = angular_velocity
            launched["done"] = True

        mujoco.mj_step(model, data)

        if step % steps_per_frame == 0:
            if camera == "pov":
                renderer.update_scene(data, camera=pov_id)
                frames.append(renderer.render())
            elif camera == "third_person":
                renderer.update_scene(data, camera=third_person_id)
                frames.append(renderer.render())
            else:   # "both" -- side by side, same frame, two camera angles
                renderer.update_scene(data, camera=pov_id)
                left = renderer.render()
                renderer.update_scene(data, camera=third_person_id)
                right = renderer.render()
                frames.append(np.hstack([left, right]))

    iio.imwrite(out_path, np.stack(frames), fps=fps, codec="libx264")
    return {"n_frames": len(frames), "I1": I1, "I2": I2, "I3": I3, "out_path": out_path}


if __name__ == "__main__":
    print("Rendering a first-person flip POV...")
    result = render_pov_flip(
        linear_velocity=[0.0, 0.0, 1.8],
        angular_velocity=[5.0, 0.0, 0.0],
        out_path="notebooks/chicken_bbq_pov_flip.mp4",
        t_strike=0.6,
    )
    print(f"wrote {result['out_path']}, {result['n_frames']} frames, "
          f"I1={result['I1']:.6f} I2={result['I2']:.6f} I3={result['I3']:.6f}")
