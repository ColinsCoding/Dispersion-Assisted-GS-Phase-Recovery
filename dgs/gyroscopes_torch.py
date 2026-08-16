"""GPU-batched Euler rigid-body integration (torch) -- same physics as
dgs.gyroscopes.integrate_euler_rigid_body, but thousands of initial
conditions at once instead of one at a time.

HONEST PERFORMANCE NOTE, benchmarked on this machine before writing this
docstring: RK4 with a fixed step count is a genuinely SEQUENTIAL workload
(step i+1 needs step i's result), so each timestep still pays full
Python-loop + CUDA-kernel-launch overhead regardless of batch size. At
batch=5 the GPU path was ~13x SLOWER per-trajectory than a plain NumPy
loop (302 ms vs 23 ms). The crossover is somewhere between batch=5 and
batch=500; by batch=5000 the GPU path is ~75x FASTER per-trajectory, and
by batch=20000 it's ~300x faster. The lesson: batch this only when you
actually have thousands of initial conditions to run (e.g. mapping a
stability boundary), not for a handful of trajectories.
"""

import numpy as np
import torch


def euler_rigid_body_rhs_batch(omega, I1, I2, I3):
    """Same RHS as dgs.gyroscopes.euler_rigid_body_rhs, vectorized over a
    (batch, 3) tensor instead of a single 3-vector."""
    w1, w2, w3 = omega[:, 0], omega[:, 1], omega[:, 2]
    dw1 = (I2 - I3) * w2 * w3 / I1
    dw2 = (I3 - I1) * w3 * w1 / I2
    dw3 = (I1 - I2) * w1 * w2 / I3
    return torch.stack([dw1, dw2, dw3], dim=1)


def integrate_euler_rigid_body_batch(omega0_batch, I1, I2, I3, t_max=20.0, dt=0.001,
                                      device=None, return_trajectory=False):
    """RK4-integrate a whole BATCH of initial conditions at once.

    omega0_batch : (N, 3) array-like -- N initial angular velocity vectors
    return_trajectory : if False (default), only the final state is kept
        (much less GPU memory for large N); if True, returns the full
        (n_steps, N, 3) trajectory.

    Returns the final (N, 3) state, or (t, trajectory) if return_trajectory.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.as_tensor(np.asarray(omega0_batch, dtype=np.float32), device=dev)
    n_steps = int(t_max / dt)

    if return_trajectory:
        traj = torch.zeros((n_steps, state.shape[0], 3), device=dev)

    for i in range(n_steps):
        if return_trajectory:
            traj[i] = state
        k1 = euler_rigid_body_rhs_batch(state, I1, I2, I3)
        k2 = euler_rigid_body_rhs_batch(state + dt / 2 * k1, I1, I2, I3)
        k3 = euler_rigid_body_rhs_batch(state + dt / 2 * k2, I1, I2, I3)
        k4 = euler_rigid_body_rhs_batch(state + dt * k3, I1, I2, I3)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    if return_trajectory:
        t = np.arange(n_steps) * dt
        return t, traj.cpu().numpy()
    return state.cpu().numpy()


def map_stability_over_sphere(I1, I2, I3, speed=5.0, n_theta=60, n_phi=120,
                               t_max=15.0, dt=0.002, device=None):
    """Batch-simulate a whole GRID of initial spin directions on the unit
    sphere (all with the same |omega|=speed) and report, for each one, how
    far the spin direction wandered from where it started -- the actual
    payoff of batching: this classifies thousands of initial conditions at
    once instead of one simulation at a time, mapping the stability
    structure (Poinsot's construction) empirically rather than analytically.

    Returns theta_grid, phi_grid (both (n_theta, n_phi)) and
    max_deviation (n_theta, n_phi): the maximum angle (radians) between the
    spin direction at time t and its own starting direction, over the run.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    theta = np.linspace(1e-3, np.pi - 1e-3, n_theta)  # avoid exact poles (singular direction)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

    # initial spin direction on the unit sphere -> omega0 = speed * direction
    x = np.sin(Theta) * np.cos(Phi)
    y = np.sin(Theta) * np.sin(Phi)
    z = np.cos(Theta)
    omega0 = speed * np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    t_arr, traj = integrate_euler_rigid_body_batch(
        omega0, I1, I2, I3, t_max=t_max, dt=dt, device=dev, return_trajectory=True
    )

    # angle between the direction at each timestep and the INITIAL direction, per trajectory
    initial_dir = torch.as_tensor(omega0 / speed, dtype=torch.float32, device=dev)
    traj_t = torch.as_tensor(traj, device=dev)
    traj_dir = traj_t / traj_t.norm(dim=2, keepdim=True).clamp_min(1e-8)
    cos_angle = (traj_dir * initial_dir.unsqueeze(0)).sum(dim=2).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    max_deviation = angle.max(dim=0).values.cpu().numpy().reshape(n_theta, n_phi)

    return Theta, Phi, max_deviation


if __name__ == "__main__":
    import time

    I1, I2, I3 = 1.0, 2.0, 3.0
    print("=== Cross-check: batched GPU integrator vs dgs.gyroscopes numpy RK4 ===\n")
    from dgs.gyroscopes import integrate_euler_rigid_body

    omega0 = [1e-3, 5.0, 1e-3]
    run_np = integrate_euler_rigid_body(omega0, I1, I2, I3, t_max=5.0, dt=0.002)
    final_np = run_np["omega"][-1]

    final_batch = integrate_euler_rigid_body_batch([omega0], I1, I2, I3, t_max=5.0, dt=0.002)[0]
    print(f"numpy final:  {final_np}")
    print(f"torch final:  {final_batch}")
    print(f"max diff:     {np.max(np.abs(final_np - final_batch)):.2e}")

    print("\n=== Batch-size scaling (t_max=5s, dt=0.002) ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    for batch_size in (5, 500, 5000, 20000):
        batch = np.zeros((batch_size, 3), dtype=np.float32)
        batch[:, 1] = 5.0
        batch[:, 0] = rng.uniform(-1e-3, 1e-3, batch_size)
        batch[:, 2] = rng.uniform(-1e-3, 1e-3, batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        integrate_euler_rigid_body_batch(batch, I1, I2, I3, t_max=5.0, dt=0.002, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        print(f"batch={batch_size:6d}: {wall:6.3f}s total, {wall / batch_size * 1e3:.4f} ms/trajectory-equivalent")

    print("\n=== Mapping the stability boundary over the sphere of initial spin directions ===\n")
    Theta, Phi, max_dev = map_stability_over_sphere(I1, I2, I3, n_theta=30, n_phi=60, t_max=10.0, dt=0.002)
    print(f"grid shape: {max_dev.shape}")
    print(f"max deviation range: [{max_dev.min():.4f}, {max_dev.max():.4f}] rad")
