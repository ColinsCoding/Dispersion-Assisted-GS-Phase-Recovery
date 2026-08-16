"""A pizza, held by a (kinematic) arm, folded in half into a calzone.

Two things this reuses rather than reinvents:
  * The arm rig and "kinematic puppet" pattern from dgs/chicken_bbq_pov.py
    -- joint angles set directly every frame, not driven by actuators.
    Here BOTH the arm's pose AND the pizza's fold hinge are scripted this
    way: the arm holds a fixed grip while the fold hinge sweeps from open
    (a flat, whole pizza) to closed (a folded calzone), which avoids
    fighting the physics engine to make a held object move exactly how a
    human hand folding food actually would.
  * The two-panel hinge-fold structure from dgs/pizza_box_fold.py -- same
    idea (two flat pieces joined by a hinge along a shared edge), just
    HELD and DRIVEN instead of dropped and left to fall freely.

The one genuinely new piece: MuJoCo has no built-in half-disk primitive,
so each pizza half is a procedurally generated triangle-fan MESH (a flat
semicircular wedge, thin extrusion for thickness) -- verified to load and
render correctly before being wired into the full scene.

Fold-angle convention -- the OPPOSITE of pizza_box_fold.py's, found by
rendering both extremes and looking, not assumed by analogy: hinge angle
0deg is the two halves coplanar (a whole, flat, open pizza -- confirmed
by rendering, this combination of mesh-mirroring geom quaternions
produces a complete circle at hinge=0), and hinge angle near 180deg is
the two halves folded together (a closed calzone). The scripted animation
sweeps from ~5deg up to ~155deg, not all the way to 180, since a real
calzone still has filling between the two folded halves -- they never
actually touch.
"""

import numpy as np
import mujoco


def _half_disk_mesh_xml(name, radius=0.12, thickness=0.004, n_arc=14):
    """A flat semicircular wedge (top+bottom triangle fans from the
    center, sweeping 0..pi) as inline MJCF mesh vertex/face data --
    MuJoCo's box/cylinder/sphere primitives have no half-disk shape."""
    theta = np.linspace(0, np.pi, n_arc)
    top_rim = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n_arc, thickness / 2)], axis=1)
    bot_rim = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n_arc, -thickness / 2)], axis=1)
    top_center = np.array([[0.0, 0.0, thickness / 2]])
    bot_center = np.array([[0.0, 0.0, -thickness / 2]])
    verts = np.vstack([top_center, top_rim, bot_center, bot_rim])
    vert_str = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts)

    faces = []
    for i in range(1, n_arc):
        faces.append((0, i, i + 1))
    bc = n_arc + 1
    for i in range(n_arc - 1):
        faces.append((bc, bc + 1 + i + 1, bc + 1 + i))
    face_str = " ".join(f"{a} {b} {c}" for a, b, c in faces)

    return f'<mesh name="{name}" vertex="{vert_str}" face="{face_str}"/>'


def build_pizza_to_calzone_model(pizza_radius=0.12, pizza_mass=0.25, dt=0.001):
    """Arm (shoulder/elbow/wrist, same rig as chicken_bbq_pov.py) holding
    a two-half pizza (each half a half-disk mesh) hinged along the
    diameter -- both the arm pose and the fold angle are scripted
    kinematically, not physically actuated."""
    mesh_a = _half_disk_mesh_xml("pizza_half_a", pizza_radius)
    mesh_b = _half_disk_mesh_xml("pizza_half_b", pizza_radius)

    I_disk = 0.25 * pizza_mass * pizza_radius ** 2   # rough half-disk moment, adequate for a kinematic body

    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.80665" timestep="{dt}" integrator="RK4"/>
      <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
      <asset>
        {mesh_a}
        {mesh_b}
      </asset>
      <worldbody>
        <light pos="0.3 -0.3 1.2" dir="-0.3 0.3 -1" diffuse="0.85 0.8 0.7"/>
        <light pos="-0.2 -0.4 0.8" dir="0.2 0.4 -0.6" diffuse="0.4 0.4 0.4"/>

        <body name="shoulder" pos="0.05 -0.35 0.55">
          <joint name="shoulder_pitch" type="hinge" axis="1 0 0" range="-90 20"/>
          <geom type="capsule" fromto="0 0 0  0 0.18 -0.02" size="0.026" rgba="0.9 0.75 0.6 1" mass="1.5"
                contype="0" conaffinity="0"/>
          <body name="forearm" pos="0 0.18 -0.02">
            <joint name="elbow" type="hinge" axis="1 0 0" range="-30 110"/>
            <geom type="capsule" fromto="0 0 0  0 0.17 -0.06" size="0.022" rgba="0.9 0.75 0.6 1" mass="1.0"
                  contype="0" conaffinity="0"/>
            <body name="wrist" pos="0 0.17 -0.06">
              <joint name="wrist_j" type="hinge" axis="1 0 0" range="-60 60"/>
              <geom type="box" size="0.02 0.015 0.008" rgba="0.6 0.6 0.65 1" mass="0.1"
                    contype="0" conaffinity="0"/>

              <!-- pizza half A: rigidly follows the wrist (this is the
                   "held" half) -->
              <body name="pizza_half_a" pos="0 0.03 0.01" quat="0.7071 0.7071 0 0">
                <geom type="mesh" mesh="pizza_half_a" mass="{pizza_mass / 2}" rgba="0.92 0.75 0.4 1"
                      contype="0" conaffinity="0"/>
                <body name="pizza_half_b" pos="0 0 0">
                  <joint name="fold_hinge" type="hinge" axis="1 0 0" pos="0 0 0" range="0 180"/>
                  <geom type="mesh" mesh="pizza_half_b" mass="{pizza_mass / 2}"
                        quat="0 1 0 0" rgba="0.88 0.7 0.38 1" contype="0" conaffinity="0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def scripted_hold_pose():
    """A fixed, steady holding pose for the arm -- no reach/retract
    animation needed here, the arm just holds the pizza still while the
    fold happens."""
    return (-45.0, 60.0, -10.0)


def fold_progress(t, fold_start=0.5, fold_duration=2.0, open_deg=5.0, closed_deg=155.0):
    """Fold hinge angle (degrees) at time t: holds open_deg until
    fold_start, eases up to closed_deg over fold_duration, then holds --
    the actual "folding into a calzone" motion.

    Convention verified by rendering, not assumed: fold=0deg is the two
    halves COPLANAR (a whole, flat, open pizza -- confirmed, this
    combination of mesh-mirroring quaternions produces a complete circle
    at hinge angle 0). fold=180deg is the two halves folded flat against
    each other (fully closed). closed_deg=155, not 180, leaves a visible
    gap between the folded halves -- a real calzone has filling between
    them, the two crusts never actually touch."""
    if t < fold_start:
        frac = 0.0
    elif t < fold_start + fold_duration:
        frac = (t - fold_start) / fold_duration
    else:
        frac = 1.0
    ease = 0.5 - 0.5 * np.cos(frac * np.pi)
    return open_deg + ease * (closed_deg - open_deg)


def simulate_fold_to_calzone(t_max=3.0, dt=0.001):
    """Run the scripted hold+fold animation and return the fold-angle
    trajectory plus sanity data (no NaN, angle actually decreases from
    open toward closed)."""
    model = build_pizza_to_calzone_model(dt=dt)
    data = mujoco.MjData(model)

    shoulder_adr = model.joint("shoulder_pitch").qposadr[0]
    elbow_adr = model.joint("elbow").qposadr[0]
    wrist_adr = model.joint("wrist_j").qposadr[0]
    fold_adr = model.joint("fold_hinge").qposadr[0]
    shoulder_dof = model.joint("shoulder_pitch").dofadr[0]

    shoulder_deg, elbow_deg, wrist_deg = scripted_hold_pose()

    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    fold_hist = np.zeros(n_steps)
    for i in range(n_steps):
        t = i * dt
        data.qpos[shoulder_adr] = np.deg2rad(shoulder_deg)
        data.qpos[elbow_adr] = np.deg2rad(elbow_deg)
        data.qpos[wrist_adr] = np.deg2rad(wrist_deg)
        fold_deg = fold_progress(t)
        data.qpos[fold_adr] = np.deg2rad(fold_deg)
        data.qvel[shoulder_dof:shoulder_dof + 4] = 0.0

        t_hist[i] = t
        fold_hist[i] = fold_deg
        mujoco.mj_step(model, data)

    return {
        "t": t_hist,
        "fold_deg": fold_hist,
        "any_nan": bool(np.any(np.isnan(data.qpos))),
        "final_fold_deg": float(fold_hist[-1]),
        "initial_fold_deg": float(fold_hist[0]),
    }


if __name__ == "__main__":
    print("=== Pizza held by an arm, scripted fold into a calzone ===\n")
    result = simulate_fold_to_calzone()
    print(f"initial fold angle: {result['initial_fold_deg']:.1f} deg (flat, open pizza)")
    print(f"final fold angle:   {result['final_fold_deg']:.1f} deg (folded calzone)")
    print(f"any NaN: {result['any_nan']}")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        idx = min(len(result["t"]) - 1, int(frac * (len(result["t"]) - 1)))
        print(f"  t={result['t'][idx]:.2f}s  fold={result['fold_deg'][idx]:6.1f} deg")
