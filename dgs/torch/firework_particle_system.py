"""A GPU-accelerated (CUDA via torch, falls back to CPU honestly) firework
particle system: real projectile-motion physics (gravity, optional exact
linear-drag solution), vectorized across (burst, particle, frame) as one
batched tensor computation -- the actual reason a particle-heavy light
show is a natural CUDA workload: thousands of independent trajectories
updated every frame is embarrassingly parallel.

PHYSICS:
  No drag: x(t) = x0 + vx0*t,  y(t) = y0 + vy0*t - (1/2)*g*t^2
  Linear drag (dv/dt = -g*yhat - k*v), exact solution (not numerically
  integrated -- this ODE is analytically solvable):
    v_x(t) = vx0 * exp(-k*t)
    x(t)   = x0 + (vx0/k) * (1 - exp(-k*t))
    v_y(t) = (vy0 + g/k) * exp(-k*t) - g/k
    y(t)   = y0 - (g/k)*t + ((vy0 + g/k)/k) * (1 - exp(-k*t))
  Brightness: simple exponential afterglow decay exp(-t/tau) -- a
  simplified model of spark burn-down, not exact pyrotechnic chemistry.

py-3.12 ONLY (torch), per this repo's convention.
"""

import numpy as np
import torch

G_EARTH = 9.8   # m/s^2


def get_device():
    """Honest device selection: reports whether CUDA is actually
    available rather than assuming it. A flaky/unstable CUDA driver can
    make even torch.cuda.is_available() itself raise (seen in practice
    under system memory pressure) -- caught here so a driver hiccup
    degrades to CPU instead of crashing the whole script."""
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except RuntimeError:
        return torch.device("cpu")


def make_burst(n_particles, origin_xy, t_start, speed_mean, speed_std,
                color_rgb, lifetime_s, device=None):
    """Sample an isotropic (2D, viewed-from-the-ground) explosion: random
    launch angles and speeds drawn from a normal distribution (clipped
    non-negative -- a real spark can't have negative speed)."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if t_start < 0:
        raise ValueError("t_start must be non-negative")
    if speed_mean <= 0 or speed_std < 0:
        raise ValueError("speed_mean must be positive, speed_std must be non-negative")
    if lifetime_s <= 0:
        raise ValueError("lifetime_s must be positive")
    if len(color_rgb) != 3:
        raise ValueError("color_rgb must be an (r, g, b) triple")
    device = device or get_device()
    angles = torch.rand(n_particles, device=device) * 2 * np.pi
    speeds = torch.clamp(
        torch.normal(speed_mean, speed_std, (n_particles,), device=device), min=0.0)
    vx0 = speeds * torch.cos(angles)
    vy0 = speeds * torch.sin(angles)
    return {
        "origin_xy": origin_xy, "t_start": t_start, "vx0": vx0, "vy0": vy0,
        "color_rgb": color_rgb, "lifetime_s": lifetime_s, "n_particles": n_particles,
        "device": device,
    }


def simulate_burst(burst, t_array, g=G_EARTH, drag_k=0.0):
    """Vectorized (all particles, all frames at once) trajectory + brightness.
    Returns x, y, brightness as (n_particles, n_frames) torch tensors."""
    if g <= 0:
        raise ValueError("g must be positive")
    if drag_k < 0:
        raise ValueError("drag_k must be non-negative")
    device = burst["device"]
    t = torch.as_tensor(t_array, dtype=torch.float32, device=device)
    local_t = torch.clamp(t - burst["t_start"], min=0.0)   # (n_frames,)
    vx0 = burst["vx0"].unsqueeze(1)   # (n_particles, 1)
    vy0 = burst["vy0"].unsqueeze(1)
    x0, y0 = burst["origin_xy"]
    lt = local_t.unsqueeze(0)   # (1, n_frames)

    if drag_k == 0.0:
        x = x0 + vx0 * lt
        y = y0 + vy0 * lt - 0.5 * g * lt**2
    else:
        k = drag_k
        x = x0 + (vx0 / k) * (1 - torch.exp(-k * lt))
        y = y0 - (g / k) * lt + ((vy0 + g / k) / k) * (1 - torch.exp(-k * lt))

    # brightness peaks (exp(0)=1) at the instant the burst starts (lt=0),
    # then decays; `active` separately zeroes frames strictly BEFORE the
    # burst has started (lt alone can't distinguish "not started yet" from
    # "just started", since clamp(min=0) maps both to lt=0)
    brightness = torch.exp(-local_t.unsqueeze(0) / burst["lifetime_s"])
    active = (t.unsqueeze(0) >= burst["t_start"]).float()
    brightness = brightness * active
    return x, y, brightness


def simulate_show(bursts, t_array, g=G_EARTH, drag_k=0.0):
    """Run simulate_burst for every burst in the show; returns a list of
    (x, y, brightness, color_rgb) tuples, one per burst, plus which
    device actually ran the computation (honest CUDA-vs-CPU report)."""
    if not bursts:
        raise ValueError("bursts must be a non-empty list")
    results = []
    for burst in bursts:
        x, y, brightness = simulate_burst(burst, t_array, g, drag_k)
        results.append((x, y, brightness, burst["color_rgb"]))
    device_used = bursts[0]["device"]
    return results, device_used


if __name__ == "__main__":
    device = get_device()
    print(f"=== Firework particle system: device = {device} ===\n")

    t_array = np.linspace(0, 6, 180)
    bursts = [
        make_burst(2000, (0.0, 0.0), t_start=0.0, speed_mean=12.0, speed_std=2.0,
                   color_rgb=(1.0, 0.3, 0.2), lifetime_s=1.5, device=device),
        make_burst(1500, (-3.0, 1.0), t_start=1.5, speed_mean=9.0, speed_std=1.5,
                   color_rgb=(0.3, 0.7, 1.0), lifetime_s=1.2, device=device),
        make_burst(1800, (3.0, 0.5), t_start=3.0, speed_mean=10.0, speed_std=2.0,
                   color_rgb=(0.5, 1.0, 0.3), lifetime_s=1.8, device=device),
    ]

    results, device_used = simulate_show(bursts, t_array, drag_k=0.05)
    print(f"simulated {len(bursts)} bursts, {sum(b['n_particles'] for b in bursts)} "
          f"total particles, {len(t_array)} frames, on device: {device_used}\n")

    for i, (x, y, brightness, color) in enumerate(results):
        peak_height = float(y.max())
        print(f"burst {i}: color={color}, peak height={peak_height:.2f} m, "
              f"shape={tuple(x.shape)}")
