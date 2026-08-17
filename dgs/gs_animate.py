"""gs_animate.py -- matplotlib animations of GS phase-retrieval convergence.

Fills the "gs_animate" entry `dgs/ousd_alignment.py`'s Human_Machine_Interfaces
CTA table references but which did not exist in `dgs/` as of this session
(found and flagged in `notebooks/phase_retrieval_connections.ipynb`'s Part 4).

`dgs/trajectory_viewer.py`'s `make_gs_convergence_viewer` already does something
similar -- an interactive Plotly 3D (Re, Im, iteration) viewer -- but Plotly is
NOT installed in this environment (`ModuleNotFoundError` on import, checked
directly, not assumed). This module is a deliberately dependency-light
alternative: matplotlib only (already a hard dependency everywhere else in this
repo), so it actually runs and is tested here, at the cost of a simpler 2D
(not interactive-3D) view.

Input throughout: `E_history`, shape `(n_frames, N)` complex -- exactly
`dgs.gs_core.retrieve_phase_with_history`'s third return value, index 0 the
pre-iteration initial guess.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional


def _validate_E_history(E_history: np.ndarray) -> np.ndarray:
    E_history = np.asarray(E_history, dtype=complex)
    if E_history.ndim != 2:
        raise ValueError("E_history must be 2D: (n_frames, N)")
    if E_history.shape[0] < 1:
        raise ValueError("E_history must have at least one frame")
    return E_history


def gs_convergence_animation(E_history: np.ndarray, t: Optional[np.ndarray] = None,
                              phi_true: Optional[np.ndarray] = None,
                              interval_ms: int = 80) -> animation.FuncAnimation:
    """Two-panel animation over GS iteration: left = the complex plane
    (Re(E) vs Im(E), colored by |E|), right = recovered phase vs t (with the
    true phase overlaid as a static reference, if given). Returns a
    matplotlib FuncAnimation -- call `.save(path)` or display with
    `HTML(anim.to_jshtml())` in a notebook."""
    E_history = _validate_E_history(E_history)
    n_frames, N = E_history.shape
    if interval_ms <= 0:
        raise ValueError(f"interval_ms={interval_ms}: must be positive")

    if t is None:
        t = np.arange(N)
    else:
        t = np.asarray(t, dtype=float)
        if t.shape[0] != N:
            raise ValueError("t must have the same length as E_history's sample axis")

    if phi_true is not None:
        phi_true = np.asarray(phi_true, dtype=float)
        if phi_true.shape[0] != N:
            raise ValueError("phi_true must have the same length as E_history's sample axis")

    amp_max = float(np.abs(E_history).max()) or 1.0
    phase_all = np.angle(E_history)

    fig, (ax_complex, ax_phase) = plt.subplots(1, 2, figsize=(10, 4))

    scat = ax_complex.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=amp_max, s=20)
    ax_complex.set_xlim(np.real(E_history).min() - 0.1, np.real(E_history).max() + 0.1)
    ax_complex.set_ylim(np.imag(E_history).min() - 0.1, np.imag(E_history).max() + 0.1)
    ax_complex.set_xlabel("Re(E)"); ax_complex.set_ylabel("Im(E)")
    ax_complex.set_title("complex-plane estimate")

    (line_phase,) = ax_phase.plot([], [], "b-", lw=1.3, label="recovered")
    if phi_true is not None:
        ax_phase.plot(t, phi_true, "k--", lw=1.0, label="true")
    ax_phase.set_xlim(t.min(), t.max())
    ax_phase.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    ax_phase.set_xlabel("sample"); ax_phase.set_ylabel("phase (rad)")
    ax_phase.legend(fontsize=8)
    title = ax_phase.set_title("GS iteration 0")

    def update(k):
        E_k = E_history[k]
        scat.set_offsets(np.column_stack([np.real(E_k), np.imag(E_k)]))
        scat.set_array(np.abs(E_k))
        line_phase.set_data(t, phase_all[k])
        title.set_text(f"GS iteration {k}/{n_frames - 1}")
        return scat, line_phase, title

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)
    plt.tight_layout()
    return anim


def save_gs_convergence_gif(E_history: np.ndarray, path: str, t: Optional[np.ndarray] = None,
                             phi_true: Optional[np.ndarray] = None, fps: int = 12) -> str:
    """Render gs_convergence_animation and save it as a .gif via Pillow
    (already a dependency of matplotlib's own image I/O). Returns path."""
    if fps <= 0:
        raise ValueError(f"fps={fps}: must be positive")
    anim = gs_convergence_animation(E_history, t=t, phi_true=phi_true, interval_ms=int(1000 / fps))
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(anim._fig)
    return path


def convergence_checkpoint_grid(E_history: np.ndarray, phi_true: Optional[np.ndarray] = None,
                                 checkpoints=(0.0, 0.25, 0.5, 1.0)):
    """A STATIC grid of snapshots at the given fractional checkpoints
    (0.0=initial guess, 1.0=final iteration) -- for embedding in a report or
    PDF where an actual animation can't be viewed. Returns
    (fig, checkpoint_iteration_indices)."""
    E_history = _validate_E_history(E_history)
    n_frames, N = E_history.shape
    checkpoints = list(checkpoints)
    if not checkpoints:
        raise ValueError("checkpoints must be non-empty")
    if any(c < 0.0 or c > 1.0 for c in checkpoints):
        raise ValueError("all checkpoints must be in [0, 1]")

    if phi_true is not None:
        phi_true = np.asarray(phi_true, dtype=float)
        if phi_true.shape[0] != N:
            raise ValueError("phi_true must have the same length as E_history's sample axis")

    indices = [int(round(c * (n_frames - 1))) for c in checkpoints]
    fig, axs = plt.subplots(1, len(indices), figsize=(3.2 * len(indices), 3.2), squeeze=False)
    axs = axs[0]
    for ax, k, c in zip(axs, indices, checkpoints):
        phase_k = np.angle(E_history[k])
        ax.plot(phase_k, "b-", lw=1.2, label="recovered")
        if phi_true is not None:
            ax.plot(phi_true, "k--", lw=1.0, label="true")
        ax.set_title(f"iter {k} ({c:.0%})")
        ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    axs[0].legend(fontsize=7)
    plt.tight_layout()
    return fig, indices


if __name__ == "__main__":
    from dgs.gs_core import make_measurements, retrieve_phase_with_history

    m = make_measurements('QPSK', n_symbols=32, sps=8, D1=-5000, D2=-5750, snr_db=30, rng_seed=1)
    phi, errors, E_history = retrieve_phase_with_history(
        m['I1'], m['I2'], m['D1'], m['D2'], n_iter=50, unit_amplitude=True)

    print(f"E_history shape: {E_history.shape}")
    fig, indices = convergence_checkpoint_grid(E_history, phi_true=m['phi_true'])
    print(f"checkpoint iterations shown: {indices}")

    out_path = "gs_convergence_demo.gif"
    save_gs_convergence_gif(E_history[::5], out_path, t=m['t'], phi_true=m['phi_true'], fps=8)
    print(f"saved animation (every 5th iteration) to {out_path}")
