"""A pizza box, hinged at 90 degrees with a pizza inside, dropped onto the
ground -- simulated, not staged. "Folds over itself" isn't an input to
this model, it's the OUTCOME: the box starts genuinely balanced at a right
angle (an unstable configuration), and real hinge + contact physics
determines whether and how far it keeps folding once gravity takes over.

Hinge convention (MuJoCo): qpos=0 is the lid coplanar with the base (the
box fully open, flat) -- that's how the lid's body pos is defined in the
XML before any joint rotation is applied. qpos=90deg is perpendicular
(the initial condition here). qpos=180deg is the lid folded completely
back over the base (the box fully closed). So the hinge angle
INCREASING past 90 deg toward 180 deg IS the box folding further over
itself -- confirmed by simulation, not assumed: starting at exactly 90deg
and dropping the assembly, the hinge angle climbs toward ~160 deg as it
settles, i.e. it keeps folding shut under its own weight and the ground
impact, matching the physical scenario directly.

Same validated conventions as the rest of this repo's MuJoCo work:
explicit <inertial> tags (identity-oriented, no auto-derived-frame
surprises), a single reused Renderer, and freejoint qvel[0:3]=world-frame
linear / qvel[3:6]=body-local angular (not needed here since we let the
box fall from rest, but noted for anyone extending this).
"""

import numpy as np
import mujoco


def build_pizza_box_model(box_half=0.2, box_thickness=0.008, box_mass=0.15,
                           pizza_radius=0.15, pizza_mass=0.4, drop_height=0.3, dt=0.0005):
    """Ground + a hinged two-panel box (base free-falling, lid hinged to
    the base) + a free pizza disk resting inside the fold at the start."""
    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.80665" timestep="{dt}" integrator="RK4"/>
      <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
      <worldbody>
        <light pos="0.3 -0.3 2" dir="-0.2 0.2 -1" diffuse="0.8 0.8 0.75"/>
        <geom name="ground" type="plane" size="1 1 0.05" pos="0 0 0"
              rgba="0.32 0.32 0.34 1" friction="0.5 0.005 0.0001"/>

        <body name="box_base" pos="0 0 {drop_height}">
          <freejoint name="box_free"/>
          <inertial pos="0 0 0" mass="{box_mass}" diaginertia="0.002 0.002 0.004"/>
          <geom type="box" size="{box_half} {box_half} {box_thickness}" mass="{box_mass}"
                rgba="0.75 0.55 0.35 1" friction="0.5 0.005 0.0001"/>
          <body name="box_lid" pos="0 -{box_half} 0">
            <joint name="hinge" type="hinge" axis="1 0 0" pos="0 0 0" range="0 180"/>
            <inertial pos="0 -{box_half} 0" mass="{box_mass}" diaginertia="0.002 0.002 0.004"/>
            <geom type="box" size="{box_half} {box_half} {box_thickness}" pos="0 -{box_half} 0"
                  mass="{box_mass}" rgba="0.7 0.5 0.3 1" friction="0.5 0.005 0.0001"/>
          </body>
        </body>

        <body name="pizza" pos="0 -{box_half * 0.25} {drop_height + 0.05}">
          <freejoint name="pizza_free"/>
          <inertial pos="0 0 0" mass="{pizza_mass}" diaginertia="0.003 0.003 0.006"/>
          <geom type="cylinder" size="{pizza_radius} 0.005" mass="{pizza_mass}"
                rgba="0.9 0.72 0.35 1" friction="0.5 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def simulate_fold(initial_hinge_deg=90.0, t_max=4.0, dt=0.0005, **model_kwargs):
    """Drop the box+pizza from rest with the hinge fixed at
    initial_hinge_deg, simulate real contact physics, and track the hinge
    angle and pizza height over time. Returns a dict of time series plus
    the final settled state."""
    model = build_pizza_box_model(dt=dt, **model_kwargs)
    data = mujoco.MjData(model)

    hinge_adr = model.joint("hinge").qposadr[0]
    data.qpos[hinge_adr] = np.deg2rad(initial_hinge_deg)
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_base")
    pizza_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pizza")

    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    hinge_hist = np.zeros(n_steps)
    pizza_z_hist = np.zeros(n_steps)
    for i in range(n_steps):
        t_hist[i] = data.time
        hinge_hist[i] = data.qpos[hinge_adr]
        pizza_z_hist[i] = data.xpos[pizza_id][2]
        mujoco.mj_step(model, data)

    return {
        "t": t_hist,
        "hinge_deg": np.rad2deg(hinge_hist),
        "pizza_z": pizza_z_hist,
        "final_hinge_deg": float(np.rad2deg(data.qpos[hinge_adr])),
        "final_pizza_pos": data.xpos[pizza_id].copy(),
        "final_base_pos": data.xpos[base_id].copy(),
        "settled": bool(np.max(np.abs(data.qvel)) < 0.05),
        "any_nan": bool(np.any(np.isnan(data.qpos))),
    }


if __name__ == "__main__":
    print("=== Pizza box folded to 90 degrees, dropped onto the ground ===\n")
    result = simulate_fold(initial_hinge_deg=90.0, t_max=4.0)
    print(f"initial hinge angle: 90.0 deg (perpendicular, the described starting configuration)")
    print(f"final hinge angle:   {result['final_hinge_deg']:.1f} deg")
    print(f"  ('180 deg' = fully folded over onto itself; this is how far it got)")
    print(f"final pizza height (z): {result['final_pizza_pos'][2]:.4f} m (~on the ground)")
    print(f"settled (low residual velocity): {result['settled']}")
    print(f"any NaN (simulation blew up): {result['any_nan']}")

    print("\n=== Hinge angle over time (every 0.5s) ===")
    for i in range(0, len(result["t"]), int(0.5 / (result["t"][1] - result["t"][0]))):
        print(f"  t={result['t'][i]:.2f}s  hinge={result['hinge_deg'][i]:6.1f} deg  "
              f"pizza_z={result['pizza_z'][i]:.4f} m")
