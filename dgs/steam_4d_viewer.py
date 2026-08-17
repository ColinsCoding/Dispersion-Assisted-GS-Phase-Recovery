"""steam_4d_viewer.py -- a 4D (x, y, z, t) time-series viewer for STEAM
microscopy, built on vector-valued calculus: a tracked feature's position is
a parametric vector function of time,

    r(t) = (x(t), y(t), z(t)),

exactly the Calc III object (a space curve), and its velocity/acceleration
are its first/second derivatives, its path length the integral of its speed.
This module reuses dgs/numerical_methods.py's velocity/acceleration/
cumulative_integral (already-tested finite-difference and trapezoid-rule
primitives) applied per-component, rather than re-deriving calculus this
repo already has.

WHERE THE "4D" DATA COMES FROM: dgs/steam_3d_depth_encoding.py already adds a
depth (z) channel to STEAM's native 2D (x, y) imaging (P9 in
dgs/sbir_portfolio.py) -- a PROPOSED, not-yet-built combination, per that
module's own docstring. A sequence of such depth-resolved frames over pulses
(t) is the natural 4th axis: this module's synthetic generator builds one
flowing-particle trajectory and its corresponding frame stack, reusing
dgs/steam_3d_depth_encoding.py's depth_range_um to keep z physically bounded
to what that encoding scheme can actually resolve, and viewer functions to
look at both the raw frames and the extracted trajectory calculus together.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

from dgs.numerical_methods import velocity, acceleration, cumulative_integral
from dgs.steam_3d_depth_encoding import depth_range_um


# ── 1. Synthetic 4D (x, y, z, t) STEAM dataset ──────────────────────────────

def synthetic_cell_trajectory(n_frames: int, xy_extent_um: float = 40.0,
                               z_band_bandwidth_nm: float = 20.0,
                               axial_dispersion_nm_per_um: float = 5.0,
                               flow_speed_um_per_frame: float = 2.0,
                               wobble_amplitude_um: float = 3.0,
                               rng_seed: int = 0) -> Dict:
    """A synthetic flowing-cell trajectory r(t) = (x(t), y(t), z(t)) through a
    STEAM field of view: steady flow along x, a sinusoidal wobble in y (e.g.
    microfluidic channel drift), and a bounded z oscillation whose amplitude
    is set by depth_range_um (steam_3d_depth_encoding.py's own depth-encoding
    limit, not an arbitrary choice)."""
    if n_frames < 4:
        raise ValueError(f"n_frames={n_frames}: must be >= 4")
    if xy_extent_um <= 0:
        raise ValueError("xy_extent_um must be positive")
    rng = np.random.default_rng(rng_seed)
    t = np.arange(n_frames, dtype=float)

    z_range = depth_range_um(z_band_bandwidth_nm, axial_dispersion_nm_per_um)
    z_amp = z_range / 2.0

    x0 = -xy_extent_um / 2.0 + rng.uniform(-2.0, 2.0)
    y0 = rng.uniform(-wobble_amplitude_um, wobble_amplitude_um)
    wobble_freq = rng.uniform(0.05, 0.15)
    z_freq = rng.uniform(0.03, 0.08)
    z_phase = rng.uniform(0, 2 * np.pi)

    x_t = x0 + flow_speed_um_per_frame * t
    y_t = y0 + wobble_amplitude_um * np.sin(2 * np.pi * wobble_freq * t)
    z_t = z_amp * np.sin(2 * np.pi * z_freq * t + z_phase)

    return {"t": t, "x_um": x_t, "y_um": y_t, "z_um": z_t,
            "xy_extent_um": xy_extent_um, "z_range_um": z_range}


def synthetic_steam_frame_stack(trajectory: Dict, frame_size: int = 64,
                                 spot_sigma_um: float = 3.0) -> np.ndarray:
    """Render one 2D Gaussian-blob STEAM frame per time point, positioned at
    (x(t), y(t)) with brightness set by how in-focus z(t) is (a defocus
    penalty -- the further from z=0, the dimmer/wider the blob, the same
    qualitative behavior a real chromatic-confocal depth channel has).
    Returns array shape (n_frames, frame_size, frame_size)."""
    if frame_size < 8:
        raise ValueError(f"frame_size={frame_size}: must be >= 8")
    if spot_sigma_um <= 0:
        raise ValueError("spot_sigma_um must be positive")
    x_t, y_t, z_t = trajectory["x_um"], trajectory["y_um"], trajectory["z_um"]
    extent = trajectory["xy_extent_um"]
    z_range = trajectory["z_range_um"]
    n_frames = len(x_t)

    grid = np.linspace(-extent, extent, frame_size)
    X, Y = np.meshgrid(grid, grid)
    frames = np.zeros((n_frames, frame_size, frame_size))
    for i in range(n_frames):
        defocus = abs(z_t[i]) / (z_range / 2.0 + 1e-12)
        sigma_eff = spot_sigma_um * (1.0 + defocus)
        amplitude = 1.0 / (1.0 + defocus)
        frames[i] = amplitude * np.exp(-((X - x_t[i]) ** 2 + (Y - y_t[i]) ** 2) / (2 * sigma_eff ** 2))
    return frames


# ── 2. Vector calculus on the tracked trajectory r(t) ───────────────────────

def trajectory_velocity(trajectory: Dict) -> Dict:
    """v(t) = dr/dt, per component -- reuses numerical_methods.velocity
    (finite-difference derivative) on each of x(t), y(t), z(t)."""
    t = trajectory["t"]
    return {
        "vx": velocity(trajectory["x_um"], t),
        "vy": velocity(trajectory["y_um"], t),
        "vz": velocity(trajectory["z_um"], t),
    }


def trajectory_acceleration(trajectory: Dict) -> Dict:
    """a(t) = d^2r/dt^2, per component -- reuses numerical_methods.acceleration."""
    t = trajectory["t"]
    return {
        "ax": acceleration(trajectory["x_um"], t),
        "ay": acceleration(trajectory["y_um"], t),
        "az": acceleration(trajectory["z_um"], t),
    }


def trajectory_speed(trajectory: Dict) -> np.ndarray:
    """|v(t)| = sqrt(vx^2+vy^2+vz^2), the scalar speed along the space curve."""
    v = trajectory_velocity(trajectory)
    return np.sqrt(v["vx"] ** 2 + v["vy"] ** 2 + v["vz"] ** 2)


def trajectory_arc_length(trajectory: Dict) -> Dict:
    """Path length s(t) = integral_0^t |v(tau)| dtau, via
    numerical_methods.cumulative_integral (trapezoid rule) of the speed --
    the numerical Fundamental Theorem of Calculus applied to a space curve.
    Returns the running arc length s(t) and the total path length s(t_end)."""
    speed = trajectory_speed(trajectory)
    t = trajectory["t"]
    s_t = cumulative_integral(speed, t)
    return {"speed": speed, "s_t": s_t, "total_length": float(s_t[-1])}


# ── 3. Viewer ─────────────────────────────────────────────────────────────

def plot_4d_time_series(trajectory: Dict, frames: np.ndarray, n_preview_frames: int = 5):
    """One figure showing the 4D dataset three ways: (a) small-multiples of
    the raw 2D STEAM frames across time, (b) the 3D trajectory r(t) colored
    by time with velocity vectors, (c) speed |v(t)| vs t with the total arc
    length annotated. Returns the matplotlib Figure (caller saves/shows it)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    n_frames = len(trajectory["t"])
    if n_preview_frames < 2:
        raise ValueError(f"n_preview_frames={n_preview_frames}: must be >= 2")
    preview_idx = np.linspace(0, n_frames - 1, min(n_preview_frames, n_frames)).astype(int)

    v = trajectory_velocity(trajectory)
    arc = trajectory_arc_length(trajectory)

    fig = plt.figure(figsize=(13, 8))

    for j, i in enumerate(preview_idx):
        ax = fig.add_subplot(3, len(preview_idx), j + 1)
        ax.imshow(frames[i], cmap="inferno", origin="lower")
        ax.set_title(f"t={int(trajectory['t'][i])}\nz={trajectory['z_um'][i]:.1f}um", fontsize=8)
        ax.axis("off")

    ax3d = fig.add_subplot(3, 1, 2, projection="3d")
    x_t, y_t, z_t, t = trajectory["x_um"], trajectory["y_um"], trajectory["z_um"], trajectory["t"]
    p = ax3d.scatter(x_t, y_t, z_t, c=t, cmap="viridis", s=20)
    skip = max(1, n_frames // 12)
    ax3d.quiver(x_t[::skip], y_t[::skip], z_t[::skip],
                v["vx"][::skip], v["vy"][::skip], v["vz"][::skip],
                length=1.5, normalize=True, color="crimson")
    ax3d.set_xlabel("x (um)"); ax3d.set_ylabel("y (um)"); ax3d.set_zlabel("z (um)")
    ax3d.set_title("r(t) = (x(t), y(t), z(t))  -- trajectory colored by t, velocity vectors in red")
    fig.colorbar(p, ax=ax3d, shrink=0.5, label="t (frame index)")

    ax_speed = fig.add_subplot(3, 1, 3)
    ax_speed.plot(t, arc["speed"], "o-", color="steelblue")
    ax_speed.set_xlabel("t (frame index)"); ax_speed.set_ylabel("|v(t)| (um/frame)")
    ax_speed.set_title(f"speed |v(t)|  --  total arc length = {arc['total_length']:.1f} um")
    ax_speed.grid(alpha=0.3)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    traj = synthetic_cell_trajectory(n_frames=40)
    frames = synthetic_steam_frame_stack(traj)
    arc = trajectory_arc_length(traj)
    print(f"trajectory: {len(traj['t'])} frames, z_range={traj['z_range_um']:.2f} um")
    print(f"total arc length: {arc['total_length']:.2f} um")
    print(f"mean speed: {arc['speed'].mean():.2f} um/frame")

    fig = plot_4d_time_series(traj, frames)
    fig.savefig("steam_4d_demo.png", dpi=110)
    print("saved steam_4d_demo.png")
