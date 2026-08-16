"""A pizza box that starts flat and stable (lid coplanar with the base,
0 degrees), is driven up to 90 degrees -- the lid sweeping through the
resting deformable pizza mesh IS what folds it, not a passive drop onto
an already-standing wall like dgs.pizza_in_box_90deg -- held there
briefly, then driven back down to 0 (the reverse rotation), ending flat
on the ground.

Same box-hinge convention as dgs.pizza_box_fold (0deg=flat/open,
90deg=perpendicular, verified there, reused here, not re-derived): axis
"1 0 0" hinge, positive rotation lifts the lid's +y-offset geometry up
into +z. Same full-circle deformable mesh as dgs.pizza_in_box_90deg
(imported, not duplicated).

Unlike dgs.pizza_in_box_90deg, the pizza is NOT pinned to anything and
the base is NOT oversized-relative-to-floor -- here the fold is forced by
the lid's own sweeping motion dragging/pressing the mesh through contact
as it rotates, closer to how an actual pizza box lid folds the pizza
inside it than a static too-small floor is."""

import numpy as np
import mujoco

from dgs.pizza_in_box_90deg import _full_circle_web_points, _full_circle_web_elements


def hinge_profile(t, rotate_start=0.3, rotate_duration=1.0, hold_duration=0.5,
                   reverse_duration=1.0, peak_deg=90.0):
    """Hinge angle (degrees) at time t: hold at 0 (let the pizza settle
    onto the flat, open box), ease up to peak_deg, hold, ease back down to
    0 (the reverse rotation), then hold flat -- 'put it on the ground'."""
    t1 = rotate_start
    t2 = t1 + rotate_duration
    t3 = t2 + hold_duration
    t4 = t3 + reverse_duration
    if t < t1:
        return 0.0
    if t < t2:
        frac = (t - t1) / rotate_duration
        ease = 0.5 - 0.5 * np.cos(frac * np.pi)
        return ease * peak_deg
    if t < t3:
        return peak_deg
    if t < t4:
        frac = (t - t3) / reverse_duration
        ease = 0.5 - 0.5 * np.cos(frac * np.pi)
        return peak_deg * (1.0 - ease)
    return 0.0


def build_pizza_box_reverse_model(panel_size=0.15, box_thickness=0.008, pizza_radius=0.12,
                                   pizza_mass=0.25, pizza_thickness=0.012,
                                   n_rings=4, n_spokes=16, dt=0.0005):
    """Static base panel (y in [-panel_size, 0]) + hinged lid panel (y in
    [0, panel_size] at hinge=0, sweeping up as the hinge is driven) + a
    full-circle flexible pizza mesh resting flat, centered on the hinge
    line, comfortably within the combined flat surface (radius <
    panel_size, so nothing dangles off the edges before the fold starts --
    unlike dgs.pizza_in_box_90deg, oversizing isn't the forcing mechanism
    here, the lid's sweep is)."""
    points = _full_circle_web_points(pizza_radius, n_rings, n_spokes)
    elems = _full_circle_web_elements(n_rings, n_spokes)
    point_str = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in points)
    elem_str = " ".join(f"{a} {b} {c}" for a, b, c in elems)

    drop_z = 2 * box_thickness + 0.015   # small drop so it settles onto the flat panels quickly

    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.80665" timestep="{dt}"/>
      <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
      <worldbody>
        <light pos="0.3 -0.3 1.2" dir="-0.3 0.3 -1" diffuse="0.85 0.8 0.7"/>
        <light pos="-0.2 -0.4 0.8" dir="0.2 0.4 -0.6" diffuse="0.4 0.4 0.4"/>
        <geom name="ground" type="plane" size="1 1 0.05" pos="0 0 0" rgba="0.32 0.32 0.34 1"
              friction="0.5 0.005 0.0001"/>

        <geom name="box_base" type="box" size="{panel_size} {panel_size / 2} {box_thickness}"
              pos="0 {-panel_size / 2} {box_thickness}" rgba="0.75 0.55 0.35 1"
              friction="0.5 0.005 0.0001"/>

        <body name="lid" pos="0 0 {box_thickness}">
          <joint name="hinge" type="hinge" axis="1 0 0" pos="0 0 0" range="-5 185"/>
          <inertial pos="0 {panel_size / 2} 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
          <geom name="box_lid" type="box" size="{panel_size} {panel_size / 2} {box_thickness}"
                pos="0 {panel_size / 2} 0" rgba="0.7 0.5 0.3 1" friction="0.5 0.005 0.0001"/>
        </body>

        <body name="pizza" pos="0 0 {drop_z}">
          <flexcomp name="web" type="direct" dim="2" pos="0 0 0" radius="{pizza_thickness / 2}"
                    point="{point_str}" element="{elem_str}" mass="{pizza_mass}"
                    rgba="0.92 0.75 0.4 1">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def simulate_fold_and_reverse(t_max=4.5, dt=0.0005, **model_kwargs):
    """Drive the hinge through hinge_profile (0 -> 90 -> hold -> 0) while
    the resting pizza mesh deforms freely under contact with the sweeping
    lid, and track hinge angle + mesh height stats throughout."""
    model = build_pizza_box_reverse_model(dt=dt, **model_kwargs)
    data = mujoco.MjData(model)

    hinge_adr = model.joint("hinge").qposadr[0]
    data.qpos[hinge_adr] = np.deg2rad(hinge_profile(0.0))
    mujoco.mj_forward(model, data)

    n_verts = model.flex_vertnum[0]
    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    hinge_hist = np.zeros(n_steps)
    mean_z_hist = np.zeros(n_steps)
    max_z_hist = np.zeros(n_steps)

    for i in range(n_steps):
        t = i * dt
        deg = hinge_profile(t)
        data.qpos[hinge_adr] = np.deg2rad(deg)

        t_hist[i] = t
        hinge_hist[i] = deg
        z = data.flexvert_xpos[:n_verts, 2]
        mean_z_hist[i] = z.mean()
        max_z_hist[i] = z.max()
        mujoco.mj_step(model, data)

    return {
        "t": t_hist,
        "hinge_deg": hinge_hist,
        "mean_z": mean_z_hist,
        "max_z": max_z_hist,
        "peak_fold_max_z": float(max_z_hist.max()),
        "final_mean_z": float(mean_z_hist[-1]),
        "any_nan": bool(np.any(np.isnan(data.flexvert_xpos))),
    }


if __name__ == "__main__":
    print("=== Pizza box: flat -> rotate to 90 (fold) -> reverse to 0 (unfold) -> on the ground ===\n")
    result = simulate_fold_and_reverse()
    print(f"any NaN: {result['any_nan']}")
    print(f"peak mesh height reached during fold: {result['peak_fold_max_z']:.4f} m")
    print(f"final mean mesh height (back on the ground, flat): {result['final_mean_z']:.4f} m")
    for frac in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
        idx = min(len(result["t"]) - 1, int(frac * (len(result["t"]) - 1)))
        print(f"  t={result['t'][idx]:.2f}s  hinge={result['hinge_deg'][idx]:6.1f} deg  "
              f"mean_z={result['mean_z'][idx]:.4f}  max_z={result['max_z'][idx]:.4f}")
