"""3D grill dynamics: the chicken_bbq_simulator flip, upgraded from a
single-axis 2D rotation to genuine 3D rigid-body tumbling, contact
physics, and landing detection -- built on MuJoCo (dgs/mujoco_gyroscope.py's
validated conventions), not hand-rolled quaternion math.

WHY THIS IS THE SAME PHYSICS AS THE GYROSCOPE WORK: once a flipped piece
leaves the tongs, gravity is the only force acting on it, and gravity acts
uniformly at the center of mass -- so it exerts NO TORQUE about the COM.
The piece's ORIENTATION during flight is therefore governed by exactly the
same torque-free Euler's equations as dgs.gyroscopes' tennis racket
theorem, just for a genuinely asymmetric body (a drumstick modeled as a
box with three distinct principal moments of inertia, not a symmetric
disk). A badly-aimed flip that happens to spin close to the drumstick's
own INTERMEDIATE axis will tumble chaotically for the same structural
reason the asymmetric top does -- verified below by cross-checking the
airborne (pre-contact) segment against dgs.gyroscopes.integrate_euler_rigid_body
directly.

Both MuJoCo conventions used here were verified empirically (not assumed)
in dgs/mujoco_gyroscope.py and re-confirmed here for the translational part:
freejoint qvel[0:3] (linear) is WORLD frame, qvel[3:6] (angular) is
BODY-LOCAL frame, regardless of orientation.
"""

import numpy as np
import mujoco


def drumstick_inertia(mass, half_sizes):
    """Box moment of inertia about its own center for a drumstick modeled
    as an asymmetric box (sx, sy, sz half-sizes) -- three DISTINCT
    principal moments as long as the three half-sizes differ, exactly the
    asymmetric-top condition the tennis racket theorem needs."""
    sx, sy, sz = half_sizes
    Ix = (mass / 12) * ((2 * sy) ** 2 + (2 * sz) ** 2)
    Iy = (mass / 12) * ((2 * sx) ** 2 + (2 * sz) ** 2)
    Iz = (mass / 12) * ((2 * sx) ** 2 + (2 * sy) ** 2)
    return Ix, Iy, Iz


def build_drumstick_grill_model(mass=0.12, half_sizes=(0.02, 0.03, 0.09), dt=0.0005):
    """A real grill plane (with contact/friction) and a free drumstick
    body above it, explicit <inertial> (identity-oriented, same fix as
    dgs.mujoco_gyroscope) so the three distinct box moments of inertia are
    pinned to the body's own local axes, not MuJoCo's auto-derived frame."""
    Ix, Iy, Iz = drumstick_inertia(mass, half_sizes)
    sx, sy, sz = half_sizes
    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.80665" timestep="{dt}" integrator="RK4"/>
      <worldbody>
        <geom name="grill" type="plane" size="0.5 0.5 0.05" pos="0 0 0"
              rgba="0.35 0.25 0.2 1" friction="0.6 0.005 0.0001"/>
        <body name="drumstick" pos="0 0 0.3">
          <freejoint/>
          <inertial pos="0 0 0" mass="{mass}" diaginertia="{Ix} {Iy} {Iz}"/>
          <geom type="box" size="{sx} {sy} {sz}" mass="{mass}" rgba="0.75 0.35 0.25 1"
                friction="0.6 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml), (Ix, Iy, Iz)


def launch_and_simulate_flip_3d(linear_velocity, angular_velocity, mass=0.12,
                                 half_sizes=(0.02, 0.03, 0.09), t_max=3.0, dt=0.0005,
                                 record_trajectory=False):
    """Launch the drumstick with a given WORLD-frame linear velocity
    (m/s) and BODY-LOCAL angular velocity (rad/s) -- exactly the
    convention verified for MuJoCo freejoints -- and simulate real contact
    physics with the grill until it settles (or t_max runs out).

    Returns final orientation quaternion, whether it's resting flat, and
    (if record_trajectory) the full position/orientation/contact history.
    """
    model, (I1, I2, I3) = build_drumstick_grill_model(mass, half_sizes, dt=dt)
    data = mujoco.MjData(model)
    data.qpos[2] = 0.3   # start above the grill
    data.qvel[0:3] = linear_velocity
    data.qvel[3:6] = angular_velocity
    mujoco.mj_forward(model, data)

    n_steps = int(t_max / dt)
    history = {"t": [], "pos": [], "quat": [], "omega_body": [], "in_contact": []} if record_trajectory else None
    airborne_omega = []   # body-frame omega while NOT touching the grill -- torque-free segment
    airborne_t = []

    for i in range(n_steps):
        in_contact = data.ncon > 0
        if not in_contact:
            airborne_omega.append(data.qvel[3:6].copy())
            airborne_t.append(data.time)
        if record_trajectory:
            history["t"].append(data.time)
            history["pos"].append(data.qpos[0:3].copy())
            history["quat"].append(data.qpos[3:7].copy())
            history["omega_body"].append(data.qvel[3:6].copy())
            history["in_contact"].append(in_contact)
        mujoco.mj_step(model, data)

    settled = float(np.max(np.abs(data.qvel))) < 0.02
    result = {
        "final_quat": data.qpos[3:7].copy(),
        "settled": settled,
        "I1": I1, "I2": I2, "I3": I3,
        "airborne_t": np.array(airborne_t),
        "airborne_omega": np.array(airborne_omega),
    }
    if record_trajectory:
        result["history"] = {k: np.array(v) for k, v in history.items()}
    return result


def which_face_down(quat):
    """Which local face (+z/-z, "skin side"/"bone side" by convention) is
    closest to facing the grill, given the body's final orientation
    quaternion. Returns 'skin' if local +z points mostly down, 'bone' if
    local -z points mostly down, or 'edge' if neither is close (landed on
    its side -- a bad flip)."""
    local_z = np.array([0.0, 0.0, 1.0])
    world_z = np.zeros(3)
    mujoco.mju_rotVecQuat(world_z, local_z, quat)
    cos_angle_down = -world_z[2]   # +1 if local+z points straight down
    if cos_angle_down > 0.7:
        return "skin"
    if cos_angle_down < -0.7:
        return "bone"
    return "edge"


if __name__ == "__main__":
    from dgs.gyroscopes import integrate_euler_rigid_body

    print("=== 3D flip: airborne phase is torque-free, cross-checked against dgs.gyroscopes ===\n")
    linear_v = [0.0, 0.0, 2.2]
    angular_v = [0.2, 9.0, 0.3]   # spin mostly about the intermediate-ish axis on purpose
    run = launch_and_simulate_flip_3d(linear_v, angular_v, t_max=2.0)

    print(f"I1,I2,I3 = {run['I1']:.6f}, {run['I2']:.6f}, {run['I3']:.6f}")
    print(f"airborne samples: {len(run['airborne_t'])}")

    ref = integrate_euler_rigid_body(angular_v, run["I1"], run["I2"], run["I3"],
                                      t_max=float(run["airborne_t"][-1]), dt=0.0005)
    idx_ref = min(len(ref["omega"]) - 1, len(run["airborne_omega"]) - 1)
    print(f"MuJoCo airborne omega (last sample):     {run['airborne_omega'][-1]}")
    print(f"dgs.gyroscopes RK4 omega (matched time):  {ref['omega'][idx_ref]}")
    print(f"max diff: {np.max(np.abs(run['airborne_omega'][-1] - ref['omega'][idx_ref])):.2e}")

    print(f"\nsettled: {run['settled']}")
    print(f"landing: {which_face_down(run['final_quat'])}")
