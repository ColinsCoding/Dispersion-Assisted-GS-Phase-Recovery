"""The Lennard-Jones potential and a small molecular-dynamics engine.

V(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]: the steep r^-12 wall (Pauli
repulsion of overlapping electron clouds) plus the r^-6 attraction (van der
Waals). Minimum at r = 2^(1/6) sigma, depth -eps. The MD integrator is plain
velocity-Verlet with an explicit O(N^2) pairwise force loop -- the heart of
every molecular simulator. Standalone (computational physics).
"""

import numpy as np


# ── the potential and its force ─────────────────────────────────────
def lj_potential(r, eps=1.0, sigma=1.0):
    """V(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]."""
    sr6 = (sigma / r) ** 6
    return 4 * eps * (sr6 * sr6 - sr6)


def lj_force_magnitude(r, eps=1.0, sigma=1.0):
    """Radial force F = -dV/dr (positive = repulsive). Zero at the minimum."""
    sr6 = (sigma / r) ** 6
    return 24 * eps / r * (2 * sr6 * sr6 - sr6)


def equilibrium_distance(sigma=1.0):
    """r_min = 2^(1/6) sigma, where V is deepest and the force vanishes."""
    return 2 ** (1 / 6) * sigma


# ── pairwise forces (the loop at the heart of MD) ───────────────────
def pair_forces(pos, eps=1.0, sigma=1.0):
    """Total LJ force on every atom and the potential energy, by explicit loop
    over all i<j pairs. Returns (forces[N,dim], potential_energy)."""
    pos = np.asarray(pos, dtype=float)
    N = len(pos)
    F = np.zeros_like(pos)
    U = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = pos[i] - pos[j]
            r = np.linalg.norm(d)
            if r < 1e-9:
                raise ValueError("two atoms coincide (r ~ 0)")
            fij = lj_force_magnitude(r, eps, sigma) * d / r
            F[i] += fij
            F[j] -= fij
            U += lj_potential(r, eps, sigma)
    return F, U


# ── velocity-Verlet molecular dynamics ──────────────────────────────
def simulate(pos, vel, eps=1.0, sigma=1.0, dt=0.005, steps=2000, store_every=20):
    """Velocity-Verlet MD with LJ forces (unit mass). Returns dict with the
    stored trajectory and the kinetic/potential/total energy histories."""
    if steps < 1 or store_every < 1:
        raise ValueError("steps and store_every must be >= 1")
    pos = np.asarray(pos, dtype=float).copy()
    vel = np.asarray(vel, dtype=float).copy()
    F, U = pair_forces(pos, eps, sigma)
    traj, KE, PE = [pos.copy()], [], []
    for s in range(1, steps + 1):
        vel += 0.5 * F * dt                 # half kick
        pos += vel * dt                     # drift
        F, U = pair_forces(pos, eps, sigma)  # new forces
        vel += 0.5 * F * dt                 # half kick
        if s % store_every == 0:
            traj.append(pos.copy())
            KE.append(0.5 * np.sum(vel ** 2))
            PE.append(U)
    KE, PE = np.array(KE), np.array(PE)
    return {"traj": np.array(traj), "KE": KE, "PE": PE, "E": KE + PE,
            "vel": vel}


def hex_cluster(n_rings=2, spacing=None, sigma=1.0):
    """A 2-D hexagonal cluster of atoms at near-equilibrium spacing (a little
    crystal that stays bound under LJ cohesion)."""
    a = spacing if spacing is not None else equilibrium_distance(sigma)
    pts = [(0.0, 0.0)]
    for ring in range(1, n_rings + 1):
        for k in range(6 * ring):
            ang = 2 * np.pi * k / (6 * ring)
            pts.append((ring * a * np.cos(ang), ring * a * np.sin(ang)))
    return np.array(pts)
