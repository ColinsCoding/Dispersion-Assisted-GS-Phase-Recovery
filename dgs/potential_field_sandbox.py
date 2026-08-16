"""A Minecraft/Fortnite-flavored voxel-block sandbox where many free-flying
orbs move under REAL scalar + vector potential forces, PyTorch-batched
(GPU) for the "macro simulation" scale (all orbs' forces computed in one
batched pass, not a Python loop per orb), rendered as a real 3D MuJoCo
scene (VR-plausible: a true 3D world, not a 2D projection).

SCALAR POTENTIAL force: a handful of point "beacons" each create an
electrostatic-style potential V(r) = k/r; the force on an orb is
F = -grad(V) = k*(r_hat)/r^2, computed ANALYTICALLY here but verified in
the test suite against a numerical finite-difference gradient of the same
V, so "scalar potential force" isn't just asserted.

VECTOR POTENTIAL force: a uniform magnetic-style field B exerts a Lorentz
force F = q*(v x B) on each moving orb -- always perpendicular to velocity
(verified directly), producing the curved/swirling trajectories that make
this look like more than particles just falling into potential wells.

Terrain is a static heightmap of stacked cube "voxels" (the actual
Minecraft/Fortnite-style building block), generated once and left rigid;
only the orbs are dynamic (MuJoCo free bodies), driven by externally
applied forces computed each step rather than any built-in MuJoCo force
field, since we need PyTorch doing the (batched, GPU) force math.
"""

import numpy as np
import torch
import mujoco

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _heightmap(grid_n, seed=0):
    """A simple, deterministic bumpy terrain height (in block units) --
    a couple of sine hills plus noise, not a flat Minecraft superflat world."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(grid_n), np.arange(grid_n), indexing="ij")
    h = (1.5 + 1.2 * np.sin(xs / 2.3) * np.cos(ys / 2.7)
         + 0.6 * np.sin(xs / 1.1 + ys / 1.7))
    h += rng.uniform(-0.3, 0.3, size=h.shape)
    return np.clip(np.round(h), 1, 4).astype(int)


def build_voxel_terrain_xml(grid_n=10, block_size=0.5, seed=0):
    heights = _heightmap(grid_n, seed)
    geoms = []
    half = block_size / 2
    for i in range(grid_n):
        for j in range(grid_n):
            h = heights[i, j]
            for k in range(h):
                x = (i - grid_n / 2) * block_size
                y = (j - grid_n / 2) * block_size
                z = half + k * block_size
                shade = 0.35 + 0.35 * (k / 4)
                geoms.append(
                    f'<geom type="box" size="{half:.3f} {half:.3f} {half:.3f}" '
                    f'pos="{x:.3f} {y:.3f} {z:.3f}" '
                    f'rgba="{0.25:.2f} {shade:.2f} {0.20:.2f} 1"/>')
    return "\n        ".join(geoms), heights


def build_sandbox_model(n_orbs=80, grid_n=10, block_size=0.5, orb_radius=0.06,
                         orb_mass=0.05, spawn_height=6.0, seed=0):
    terrain_xml, heights = build_voxel_terrain_xml(grid_n, block_size, seed)
    rng = np.random.default_rng(seed + 1)
    world_half = grid_n * block_size / 2

    orb_bodies = []
    for n in range(n_orbs):
        x = rng.uniform(-world_half * 0.8, world_half * 0.8)
        y = rng.uniform(-world_half * 0.8, world_half * 0.8)
        z = spawn_height + rng.uniform(0, 2.0)
        orb_bodies.append(f"""
        <body name="orb_{n}" pos="{x:.4f} {y:.4f} {z:.4f}">
          <freejoint/>
          <inertial pos="0 0 0" mass="{orb_mass}" diaginertia="1e-5 1e-5 1e-5"/>
          <geom type="sphere" size="{orb_radius}" mass="{orb_mass}" rgba="0.9 0.75 0.2 1"
                contype="0" conaffinity="0"/>
        </body>""")

    xml = f"""
    <mujoco>
      <option gravity="0 0 -2.0" timestep="0.004" integrator="RK4"/>
      <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
      <worldbody>
        <light pos="2 -2 6" dir="-0.3 0.3 -1" diffuse="0.85 0.8 0.7"/>
        <light pos="-2 -3 5" dir="0.3 0.4 -0.7" diffuse="0.4 0.4 0.4"/>
        {terrain_xml}
        {''.join(orb_bodies)}
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml), heights


# ── scalar + vector potential forces, PyTorch-batched ───────────────────

def make_beacons(n=4, world_half=2.5, seed=2):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-world_half * 0.7, world_half * 0.7, size=(n, 3))
    positions[:, 2] = rng.uniform(2.0, 5.0, size=n)
    strengths = rng.choice([-1.0, 1.0], size=n) * rng.uniform(1.5, 3.0, size=n)
    return torch.tensor(positions, dtype=torch.float32, device=DEVICE), \
           torch.tensor(strengths, dtype=torch.float32, device=DEVICE)


def scalar_potential(positions, beacon_pos, beacon_strength, softening=0.15):
    """V(r) = sum_i k_i / |r - r_i| (electrostatic-style point-source
    potential, softened to avoid a singularity at the source)."""
    diff = positions.unsqueeze(1) - beacon_pos.unsqueeze(0)          # (N, B, 3)
    dist = torch.sqrt((diff ** 2).sum(-1) + softening ** 2)           # (N, B)
    return (beacon_strength.unsqueeze(0) / dist).sum(-1)              # (N,)


def scalar_force(positions, beacon_pos, beacon_strength, softening=0.15):
    """F = -grad(V), computed analytically: F_i = sum_j k_j*(r_i-r_j)/|r_i-r_j|_soft^3."""
    diff = positions.unsqueeze(1) - beacon_pos.unsqueeze(0)           # (N, B, 3)
    dist2 = (diff ** 2).sum(-1) + softening ** 2                       # (N, B)
    dist3 = dist2 * torch.sqrt(dist2)
    return (beacon_strength.unsqueeze(0).unsqueeze(-1) * diff / dist3.unsqueeze(-1)).sum(1)  # (N, 3)


def vector_potential_force(velocities, B_field, charge=1.0):
    """Lorentz force from a uniform magnetic-style field: F = q*(v x B)."""
    B = B_field.unsqueeze(0).expand_as(velocities)
    return charge * torch.linalg.cross(velocities, B)


def simulate_sandbox(t_max=6.0, n_orbs=80, charge=2.0, B_strength=1.2,
                      beacon_strength_scale=1.0, force_clip=25.0, **model_kwargs):
    model, heights = build_sandbox_model(n_orbs=n_orbs, **model_kwargs)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    beacon_pos, beacon_strength = make_beacons()
    beacon_strength = beacon_strength * beacon_strength_scale
    B_field = torch.tensor([0.0, 0.0, B_strength], dtype=torch.float32, device=DEVICE)

    orb_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"orb_{n}") for n in range(n_orbs)]

    dt = model.opt.timestep
    n_steps = int(t_max / dt)
    t_hist = np.zeros(n_steps)
    pos_hist = np.zeros((n_steps, n_orbs, 3))

    for i in range(n_steps):
        t_hist[i] = i * dt
        positions_np = data.xpos[orb_body_ids]
        velocities_np = data.cvel[orb_body_ids, 3:6]     # linear velocity, world frame
        pos_hist[i] = positions_np

        positions = torch.tensor(positions_np, dtype=torch.float32, device=DEVICE)
        velocities = torch.tensor(velocities_np, dtype=torch.float32, device=DEVICE)

        F_scalar = scalar_force(positions, beacon_pos, beacon_strength)
        F_vector = vector_potential_force(velocities, B_field, charge=charge)
        F_total = torch.clamp(F_scalar + F_vector, -force_clip, force_clip)
        F_total_np = F_total.detach().cpu().numpy()

        for k, body_id in enumerate(orb_body_ids):
            data.xfrc_applied[body_id, 0:3] = F_total_np[k]

        mujoco.mj_step(model, data)

    path_length = np.sum(np.linalg.norm(np.diff(pos_hist, axis=0), axis=-1), axis=0)     # (n_orbs,)
    straight_dist = np.linalg.norm(pos_hist[-1] - pos_hist[0], axis=-1)                   # (n_orbs,)

    return {
        "t": t_hist,
        "pos": pos_hist,
        "path_length": path_length,
        "straight_dist": straight_dist,
        "beacon_pos": beacon_pos.cpu().numpy(),
        "any_nan": bool(np.any(np.isnan(pos_hist))),
    }


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    print("=== Potential-field voxel sandbox: scalar + vector (Lorentz) forces on 80 orbs ===\n")
    result = simulate_sandbox()
    print(f"any NaN: {result['any_nan']}")
    curl_ratio = result["path_length"] / np.maximum(result["straight_dist"], 1e-6)
    print(f"path-length / straight-line-distance ratio: min={curl_ratio.min():.2f}  "
          f"mean={curl_ratio.mean():.2f}  max={curl_ratio.max():.2f}")
    print("(ratio >> 1 means real curved/swirling trajectories, not straight lines to a potential minimum)")
