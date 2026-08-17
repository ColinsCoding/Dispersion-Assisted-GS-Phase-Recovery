import os
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from dgs.gs_animate import (
    gs_convergence_animation, save_gs_convergence_gif, convergence_checkpoint_grid,
)


def _toy_E_history(n_frames=6, N=16, seed=0):
    rng = np.random.default_rng(seed)
    phi = np.cumsum(rng.normal(0, 0.05, N))
    base = np.exp(1j * phi)
    # fake "convergence": interpolate from a random start to `base`
    start = rng.normal(size=N) + 1j * rng.normal(size=N)
    frac = np.linspace(0, 1, n_frames)[:, None]
    return (1 - frac) * start[None, :] + frac * base[None, :], phi


def test_gs_convergence_animation_returns_funcanimation():
    E_history, phi = _toy_E_history()
    anim = gs_convergence_animation(E_history, phi_true=phi)
    assert isinstance(anim, animation.FuncAnimation)
    plt.close("all")


def test_gs_convergence_animation_rejects_1d_history():
    with pytest.raises(ValueError):
        gs_convergence_animation(np.array([1.0, 2.0, 3.0]))


def test_gs_convergence_animation_rejects_mismatched_t():
    E_history, phi = _toy_E_history()
    with pytest.raises(ValueError):
        gs_convergence_animation(E_history, t=np.arange(3))
    plt.close("all")


def test_gs_convergence_animation_rejects_mismatched_phi_true():
    E_history, _ = _toy_E_history()
    with pytest.raises(ValueError):
        gs_convergence_animation(E_history, phi_true=np.zeros(3))
    plt.close("all")


def test_gs_convergence_animation_rejects_nonpositive_interval():
    E_history, _ = _toy_E_history()
    with pytest.raises(ValueError):
        gs_convergence_animation(E_history, interval_ms=0)
    plt.close("all")


def test_save_gs_convergence_gif_writes_file(tmp_path):
    E_history, phi = _toy_E_history(n_frames=4, N=8)
    out = str(tmp_path / "conv.gif")
    path = save_gs_convergence_gif(E_history, out, phi_true=phi, fps=10)
    assert path == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    plt.close("all")


def test_save_gs_convergence_gif_rejects_nonpositive_fps(tmp_path):
    E_history, _ = _toy_E_history(n_frames=4, N=8)
    with pytest.raises(ValueError):
        save_gs_convergence_gif(E_history, str(tmp_path / "x.gif"), fps=0)


def test_convergence_checkpoint_grid_picks_correct_indices():
    E_history, phi = _toy_E_history(n_frames=9, N=10)  # indices 0..8
    fig, indices = convergence_checkpoint_grid(E_history, phi_true=phi, checkpoints=(0.0, 0.5, 1.0))
    assert indices == [0, 4, 8]
    assert len(fig.axes) == 3
    plt.close(fig)


def test_convergence_checkpoint_grid_rejects_out_of_range_checkpoint():
    E_history, _ = _toy_E_history()
    with pytest.raises(ValueError):
        convergence_checkpoint_grid(E_history, checkpoints=(0.0, 1.5))


def test_convergence_checkpoint_grid_rejects_empty_checkpoints():
    E_history, _ = _toy_E_history()
    with pytest.raises(ValueError):
        convergence_checkpoint_grid(E_history, checkpoints=())


def test_convergence_checkpoint_grid_rejects_too_few_frames():
    with pytest.raises(ValueError):
        convergence_checkpoint_grid(np.zeros((0, 8), dtype=complex))
