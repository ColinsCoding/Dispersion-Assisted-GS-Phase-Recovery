"""A pizza that folds because it's made of a real deformable material --
a spider-web-style mass-spring mesh (MuJoCo's <flexcomp>, dim=2), not two
rigid half-disks swinging on a single hinge (dgs/pizza_to_calzone.py's
approximation, kept for what it's good at: a fast, stable "shape of the
fold" demo -- but it can't sag, drape, or wrinkle, because both halves are
perfectly flat rigid bodies. This module is the actual physics-engine
version the earlier one approximated.

WHY TWO SEPARATE FLEXCOMPS, NOT ONE: MuJoCo's <pin> fixes flex vertices to
the flexcomp's OWN single parent body -- verified directly, not assumed
(moving one parent anchor moved every pinned vertex in that flexcomp by
the same rigid offset, even ones on "the other side"). A single flexcomp
therefore can't have two independently-moving grip edges. Two half-disk
webs, each pinned along its own straight (fold) edge to its own small
hinged anchor body, DO move independently -- and each half's outer rim,
being unpinned, is free to sag/lag/deform according to its own
mass-spring dynamics as the two anchors swing together. That lag and sag
is the actual point: it's what a rigid-hinge approximation cannot show.

Mesh topology: the same radial "spider web" point layout as a rigid
half-disk (center + concentric rings, sweeping 0..pi) -- reused, not
redesigned, from the same idea behind pizza_to_calzone.py's mesh, just
triangulated into flex elements instead of a single rigid mesh geom.
"""

import numpy as np
import mujoco


def _half_disk_web_points(radius=0.12, n_rings=3, n_spokes=8):
    """Center + n_rings concentric arcs (sweeping 0..pi), n_spokes+1
    points per arc -- the vertex layout for one flexible pizza half."""
    points = [[0.0, 0.0, 0.0]]
    for ring in range(1, n_rings + 1):
        r = radius * ring / n_rings
        theta = np.linspace(0, np.pi, n_spokes + 1)
        for th in theta:
            points.append([r * np.cos(th), r * np.sin(th), 0.0])
    return np.array(points)


def _half_disk_web_elements(n_rings=3, n_spokes=8):
    """Triangulate the point layout above into flex (triangle) elements:
    center fan for the first ring, then quad-strips (as 2 triangles each)
    between successive rings."""
    per_ring = n_spokes + 1
    elems = []
    # center fan to ring 1
    for k in range(n_spokes):
        a = 1 + k
        b = 1 + k + 1
        elems.append((0, a, b))
    # ring i to ring i+1
    for ring in range(1, n_rings):
        base_in = 1 + (ring - 1) * per_ring
        base_out = 1 + ring * per_ring
        for k in range(n_spokes):
            a, b = base_in + k, base_in + k + 1
            c, d = base_out + k, base_out + k + 1
            elems.append((a, b, d))
            elems.append((a, d, c))
    return elems


def _edge_point_indices(n_rings=3, n_spokes=8):
    """Indices of the points lying ON the straight (fold) edge -- theta=0
    and theta=pi, i.e. k=0 and k=n_spokes in every ring, plus the center
    point (which is ON the edge by construction, r=0). These are the ONLY
    points pinned to the hinge anchor; everything else is free."""
    per_ring = n_spokes + 1
    idx = [0]   # center
    for ring in range(1, n_rings + 1):
        base = 1 + (ring - 1) * per_ring
        idx.append(base)              # theta=0
        idx.append(base + n_spokes)   # theta=pi
    return idx


def build_web_pizza_model(radius=0.12, n_rings=4, n_spokes=10, mass_per_half=0.1, dt=0.0005):
    """Two mirrored half-disk spider-web meshes, each pinned along its
    straight edge to its own hinged anchor body. Both anchors start folded
    flat (hinge=0, matching pizza_to_calzone.py's verified convention:
    hinge=0 is coplanar/open); driving the hinges is what folds it."""
    points = _half_disk_web_points(radius, n_rings, n_spokes)
    elems = _half_disk_web_elements(n_rings, n_spokes)
    edge_idx = _edge_point_indices(n_rings, n_spokes)

    point_str = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in points)
    elem_str = " ".join(f"{a} {b} {c}" for a, b, c in elems)
    pin_str = " ".join(str(i) for i in edge_idx)

    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.80665" timestep="{dt}"/>
      <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
      <worldbody>
        <light pos="0.3 -0.3 1.2" dir="-0.3 0.3 -1" diffuse="0.85 0.8 0.7"/>

        <body name="anchor_a" pos="0 0 0.4">
          <joint name="hinge_a" type="hinge" axis="1 0 0" pos="0 0 0"/>
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
          <flexcomp name="web_a" type="direct" dim="2" pos="0 0 0" radius="0.0015"
                    point="{point_str}" element="{elem_str}" mass="{mass_per_half}"
                    rgba="0.92 0.75 0.4 1">
            <edge equality="true"/>
            <pin id="{pin_str}"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </body>

        <body name="anchor_b" pos="0 0 0.4">
          <joint name="hinge_b" type="hinge" axis="1 0 0" pos="0 0 0"/>
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
          <flexcomp name="web_b" type="direct" dim="2" pos="0 0 0" radius="0.0015"
                    point="{point_str}" element="{elem_str}" mass="{mass_per_half}"
                    rgba="0.88 0.7 0.38 1" euler="0 0 180">
            <edge equality="true"/>
            <pin id="{pin_str}"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml), edge_idx


def fold_progress(t, fold_start=0.3, fold_duration=1.5, open_deg=0.0, closed_deg=150.0):
    """Anchor hinge angle (degrees) at time t -- same easing shape as
    pizza_to_calzone.py's, driving the RIGID ANCHORS only; the flexible
    web's outer rim is never directly commanded, it responds to the
    anchors' motion through the mesh's own spring dynamics."""
    if t < fold_start:
        frac = 0.0
    elif t < fold_start + fold_duration:
        frac = (t - fold_start) / fold_duration
    else:
        frac = 1.0
    ease = 0.5 - 0.5 * np.cos(frac * np.pi)
    return open_deg + ease * (closed_deg - open_deg)


def simulate_web_fold(t_max=3.0, dt=0.0005, **model_kwargs):
    """Drive both anchor hinges through the fold while the two flexible
    half-webs deform freely, and return rim-point trajectories (to check
    they genuinely LAG the anchor -- the signature of real drape/inertia,
    absent from a rigid-hinge model) plus a stability check."""
    model, edge_idx = build_web_pizza_model(dt=dt, **model_kwargs)
    data = mujoco.MjData(model)

    hinge_a_adr = model.joint("hinge_a").qposadr[0]
    hinge_b_adr = model.joint("hinge_b").qposadr[0]

    # the outer ring's MIDDLE point (theta ~ pi/2) -- maximally far from
    # the fold (hinge) axis, so it actually moves as the hinge rotates.
    # (A real bug caught here: the LAST-generated point is theta=pi,
    # which lies exactly ON the hinge's rotation axis and therefore
    # never moves under that rotation no matter how far the hinge turns
    # -- tracking it looked like "the web isn't responding" when the
    # actual problem was tracking a point rotation can't move at all.)
    n_rings_used = model_kwargs.get("n_rings", 4)
    n_spokes_used = model_kwargs.get("n_spokes", 10)
    rim_a = 1 + (n_rings_used - 1) * (n_spokes_used + 1) + n_spokes_used // 2
    rim_b = model.flex_vertnum[0] + model.flex_vertnum[1] - 1

    fold_deg0 = fold_progress(0.0)
    data.qpos[hinge_a_adr] = np.deg2rad(fold_deg0)
    data.qpos[hinge_b_adr] = np.deg2rad(fold_deg0)
    mujoco.mj_forward(model, data)   # populate flexvert_xpos before the first recorded sample

    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    anchor_hinge_deg = np.zeros(n_steps)
    rim_a_pos = np.zeros((n_steps, 3))

    for i in range(n_steps):
        t = i * dt
        fold_deg = fold_progress(t)
        data.qpos[hinge_a_adr] = np.deg2rad(fold_deg)
        data.qpos[hinge_b_adr] = np.deg2rad(fold_deg)

        t_hist[i] = t
        anchor_hinge_deg[i] = fold_deg
        rim_a_pos[i] = data.flexvert_xpos[rim_a]
        mujoco.mj_step(model, data)

    return {
        "t": t_hist,
        "anchor_hinge_deg": anchor_hinge_deg,
        "rim_a_pos": rim_a_pos,
        "any_nan": bool(np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.flexvert_xpos))),
        "final_rim_a_pos": rim_a_pos[-1].copy(),
    }


if __name__ == "__main__":
    print("=== Pizza as a real deformable spider-web mesh, folded via two hinged edge anchors ===\n")
    result = simulate_web_fold()
    print(f"any NaN: {result['any_nan']}")
    print(f"final anchor hinge angle: {result['anchor_hinge_deg'][-1]:.1f} deg")
    print(f"final outer-rim point position: {result['final_rim_a_pos']}")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        idx = min(len(result["t"]) - 1, int(frac * (len(result["t"]) - 1)))
        print(f"  t={result['t'][idx]:.2f}s  anchor={result['anchor_hinge_deg'][idx]:6.1f} deg  "
              f"rim_a={result['rim_a_pos'][idx]}")
