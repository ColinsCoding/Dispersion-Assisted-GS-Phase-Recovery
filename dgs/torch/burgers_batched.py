"""Viscous Burgers, batched over viscosities, on whatever torch device exists.

The same scheme as dgs.lagrangian_fluids.solve_burgers (central differences,
conservative flux, RK4, periodic), but B problems with different nu advance
together as one (B, N) tensor. Device order: CUDA (NVIDIA), then MPS (Apple
Metal), then CPU. Only CUDA and CPU have been run on this machine; the MPS
branch is the standard torch device string and is untested here.

Checked against the exact Cole-Hopf solution (dgs.lagrangian_fluids.cole_hopf_exact)
and against the NumPy reference solver.
"""
import time

import numpy as np
import torch


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rhs(u, dx, nus):
    up, um = torch.roll(u, -1, dims=-1), torch.roll(u, 1, dims=-1)
    return -(up * up - um * um) / (4 * dx) + nus * (up - 2 * u + um) / (dx * dx)


def solve_batched(viscosities, n=512, t_end=0.5, device=None, dtype=torch.float64, safety=0.4):
    """Returns (x as numpy, u as numpy (B, n), steps, seconds, device name).
    One common dt, set by the smallest viscosity's advective limit and the
    largest viscosity's diffusive limit."""
    nus_np = np.asarray(viscosities, float)
    if nus_np.ndim != 1 or np.any(nus_np <= 0) or n < 16 or t_end <= 0:
        raise ValueError("need a 1D list of positive viscosities, n >= 16, t_end > 0")
    if device is None:
        device = pick_device()
    if device.type == "mps":
        dtype = torch.float32                      # MPS has no float64
    dx = 1.0 / n
    dt = safety * min(dx, dx * dx / (2 * nus_np.max()))
    steps = int(np.ceil(t_end / dt))
    dt = t_end / steps
    xg = torch.arange(n, dtype=dtype, device=device) * dx
    u = torch.sin(2 * np.pi * xg).repeat(len(nus_np), 1)
    nus = torch.tensor(nus_np, dtype=dtype, device=device)[:, None]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        k1 = _rhs(u, dx, nus)
        k2 = _rhs(u + dt / 2 * k1, dx, nus)
        k3 = _rhs(u + dt / 2 * k2, dx, nus)
        k4 = _rhs(u + dt * k3, dx, nus)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    return xg.cpu().numpy(), u.cpu().numpy(), steps, sec, str(device)


def throughput(batch, n=512, n_steps=200, device=None, dtype=torch.float64, viscosity=0.01):
    """Seconds for n_steps RK4 steps of `batch` independent problems, and
    grid-point updates per second. Small batches are launch-overhead bound on
    a GPU (each step is ~40 tiny kernels); large batches are bandwidth bound."""
    device = pick_device() if device is None else device
    if device.type == "mps":
        dtype = torch.float32
    dx = 1.0 / n
    dt = 0.4 * min(dx, dx * dx / (2 * viscosity))
    u = torch.sin(2 * np.pi * torch.arange(n, dtype=dtype, device=device) * dx).repeat(batch, 1)
    nus = torch.full((batch, 1), viscosity, dtype=dtype, device=device)
    for _ in range(3):                                   # warm-up
        u = u + dt * _rhs(u, dx, nus)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        k1 = _rhs(u, dx, nus)
        k2 = _rhs(u + dt / 2 * k1, dx, nus)
        k3 = _rhs(u + dt / 2 * k2, dx, nus)
        k4 = _rhs(u + dt * k3, dx, nus)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    return sec, batch * n * n_steps / sec
