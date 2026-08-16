"""Best attempt: the real deformable spider-web pizza mesh (dgs/pizza_web_fold.py's
approach, generalized from a half-disk to a full circle) settling under
gravity inside a pizza box whose fold is fixed at 90 degrees (the same
box geometry as dgs/pizza_box_fold.py, but held static here rather than
left to tumble -- the point of this module is what happens to the PIZZA's
own topology as it conforms to the box, not re-simulating the box itself).

Labeled "best attempt" deliberately: contact between a deformable flex
mesh and rigid box panels is a genuinely harder physics problem than
anything else built this session (rigid-rigid contact, or a flex mesh
with nothing external to press against), and MuJoCo's flex-vs-rigid
contact defaults were not assumed to be tuned correctly for this shape --
verified by checking for penetration/instability after settling, not by
assuming the first working config was right.
"""

import numpy as np
import mujoco


def _full_circle_web_points(radius=0.12, n_rings=4, n_spokes=16):
    """Center + n_rings concentric rings, n_spokes points per ring,
    sweeping the FULL 2*pi -- the whole-pizza generalization of
    pizza_web_fold.py's half-disk layout (which sweeps 0..pi for a single
    fold-able half)."""
    points = [[0.0, 0.0, 0.0]]
    for ring in range(1, n_rings + 1):
        r = radius * ring / n_rings
        theta = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False)
        for th in theta:
            points.append([r * np.cos(th), r * np.sin(th), 0.0])
    return np.array(points)


def _full_circle_web_elements(n_rings=4, n_spokes=16):
    elems = []
    for k in range(n_spokes):
        a, b = 1 + k, 1 + (k + 1) % n_spokes
        elems.append((0, a, b))
    for ring in range(1, n_rings):
        base_in = 1 + (ring - 1) * n_spokes
        base_out = 1 + ring * n_spokes
        for k in range(n_spokes):
            a, b = base_in + k, base_in + (k + 1) % n_spokes
            c, d = base_out + k, base_out + (k + 1) % n_spokes
            elems.append((a, b, d))
            elems.append((a, d, c))
    return elems


def build_pizza_in_box_model(floor_depth=0.1, box_thickness=0.008, pizza_radius=0.12,
                              pizza_mass=0.25, pizza_thickness=0.012,
                              n_rings=4, n_spokes=16, dt=0.0005):
    """A STATIC box corner (floor panel + a vertical wall panel meeting
    it AT the crease line y=0, fixed at 90deg -- no free/hinge joint, this
    module is about the pizza, not the box) plus a full-circle flexible
    pizza mesh CENTERED ON THE CREASE.

    Sizing is the actual forcing mechanism, not decoration: floor_depth
    (0.1m) is deliberately smaller than pizza_radius (0.12m). A flat disk
    bigger than the floor it's dropped onto, centered on the crease, has
    nowhere to go on one side except drape down against the vertical wall
    -- that overhang is what makes this a genuine 90-degree conforming
    test rather than the first version's mistake (wall positioned 0.2m
    from a 0.12m-radius pizza, so nothing ever reached it and the pizza
    just landed flat -- caught by checking final vertex heights, not
    assumed correct)."""
    points = _full_circle_web_points(pizza_radius, n_rings, n_spokes)
    elems = _full_circle_web_elements(n_rings, n_spokes)
    point_str = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in points)
    elem_str = " ".join(f"{a} {b} {c}" for a, b, c in elems)

    drop_z = 0.15   # start the pizza flat, above the crease

    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.80665" timestep="{dt}"/>
      <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
      <worldbody>
        <light pos="0.3 -0.3 1.2" dir="-0.3 0.3 -1" diffuse="0.85 0.8 0.7"/>
        <light pos="-0.2 -0.4 0.8" dir="0.2 0.4 -0.6" diffuse="0.4 0.4 0.4"/>
        <geom name="ground" type="plane" size="1 1 0.05" pos="0 0 0" rgba="0.32 0.32 0.34 1"
              friction="0.5 0.005 0.0001"/>

        <!-- floor panel: spans y in [-floor_depth, 0], crease at y=0 -->
        <geom name="box_floor" type="box" size="{floor_depth} {floor_depth / 2} {box_thickness}"
              pos="0 {-floor_depth / 2} {box_thickness}" rgba="0.75 0.55 0.35 1"
              friction="0.5 0.005 0.0001"/>
        <!-- wall panel: vertical, rooted right at the crease (y=0), rising in +z --
             matches pizza_box_fold.py's convention that 90deg is perpendicular -->
        <geom name="box_wall" type="box" size="{floor_depth} {floor_depth} {box_thickness}"
              pos="0 0 {box_thickness + floor_depth}" euler="90 0 0"
              rgba="0.7 0.5 0.3 1" friction="0.5 0.005 0.0001"/>

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


def simulate_pizza_settling(t_max=5.0, dt=0.0005, pizza_radius=0.12,
                             n_rings=4, n_spokes=16, **model_kwargs):
    """Drop the flexible pizza onto the static 90-degree corner and let it
    settle under gravity + contact. Returns per-vertex height trajectory
    (to check it actually descends and then stops moving -- settles, not
    just falls forever or explodes), a penetration sanity check, and the
    actual "did it conform to 90 degrees" metric: the overhang vertices
    (rest position on the far side of the crease, y>0, with nothing under
    them) should end up climbing well above floor height, pressed against
    the wall -- not lying flat, which would mean the sizing failed to
    force any real bend."""
    model = build_pizza_in_box_model(dt=dt, pizza_radius=pizza_radius,
                                      n_rings=n_rings, n_spokes=n_spokes, **model_kwargs)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rest_points = _full_circle_web_points(pizza_radius, n_rings, n_spokes)
    overhang_idx = np.where(rest_points[:, 1] > 0)[0]

    n_verts = model.flex_vertnum[0]
    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    mean_z_hist = np.zeros(n_steps)
    min_z_hist = np.zeros(n_steps)
    overhang_max_z_hist = np.zeros(n_steps)

    for i in range(n_steps):
        t_hist[i] = i * dt
        z = data.flexvert_xpos[:n_verts, 2]
        mean_z_hist[i] = z.mean()
        min_z_hist[i] = z.min()
        overhang_max_z_hist[i] = z[overhang_idx].max()
        mujoco.mj_step(model, data)

    final_qvel_flex = float(np.max(np.abs(data.qvel))) if model.nv > 0 else 0.0
    return {
        "t": t_hist,
        "mean_z": mean_z_hist,
        "min_z": min_z_hist,
        "overhang_max_z": overhang_max_z_hist,
        "final_min_z": float(min_z_hist[-1]),
        "final_overhang_max_z": float(overhang_max_z_hist[-1]),
        "settled": bool(final_qvel_flex < 0.1),
        "any_nan": bool(np.any(np.isnan(data.flexvert_xpos))),
    }


if __name__ == "__main__":
    print("=== Best attempt: real deformable pizza settling inside a static 90-degree box ===\n")
    result = simulate_pizza_settling()
    print(f"any NaN: {result['any_nan']}")
    print(f"settled (low residual velocity): {result['settled']}")
    print(f"final minimum vertex height: {result['final_min_z']:.4f} m (should be >= 0, i.e. not through the ground)")
    print(f"final overhang max height: {result['final_overhang_max_z']:.4f} m (>0.1 m = genuinely climbed the wall)")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        idx = min(len(result["t"]) - 1, int(frac * (len(result["t"]) - 1)))
        print(f"  t={result['t'][idx]:.2f}s  mean_z={result['mean_z'][idx]:.4f}  min_z={result['min_z'][idx]:.4f}  "
              f"overhang_max_z={result['overhang_max_z'][idx]:.4f}")
