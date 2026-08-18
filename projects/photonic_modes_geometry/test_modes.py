import numpy as np
import pytest
from geometry import make_rectangle
from modes import solve_modes


@pytest.fixture
def rect_grid():
    n_grid, _ = make_rectangle(48, 48, 0.1, 0.1, width=2.0, height=1.0, n_core=3.4, n_clad=1.44)
    return n_grid


def test_solve_modes_rejects_bad_n_modes(rect_grid):
    with pytest.raises(ValueError):
        solve_modes(rect_grid, 0.1, 0.1, 1.55, n_modes=0)


def test_solve_modes_rejects_non_2d_grid():
    with pytest.raises(ValueError):
        solve_modes(np.ones(10), 0.1, 0.1, 1.55, n_modes=2)


def test_solve_modes_rejects_n_modes_too_large_for_grid(rect_grid):
    with pytest.raises(ValueError):
        solve_modes(rect_grid, 0.1, 0.1, 1.55, n_modes=rect_grid.size)


def test_solve_modes_returns_requested_count(rect_grid):
    modes = solve_modes(rect_grid, 0.1, 0.1, 1.55, n_modes=4)
    assert len(modes) == 4


def test_solve_modes_sorted_by_decreasing_beta_sq(rect_grid):
    modes = solve_modes(rect_grid, 0.1, 0.1, 1.55, n_modes=4)
    beta_sqs = [m["beta_sq"] for m in modes]
    assert beta_sqs == sorted(beta_sqs, reverse=True)


def test_solve_modes_fundamental_is_guided(rect_grid):
    # n_eff of the most-confined mode must lie strictly between n_clad and n_core
    # for it to be a physically guided mode, not a boundary/discretization artifact
    modes = solve_modes(rect_grid, 0.1, 0.1, 1.55, n_modes=1)
    n_eff = modes[0]["n_eff"]
    assert 1.44 < n_eff < 3.4


def test_solve_modes_field_shape_matches_grid(rect_grid):
    modes = solve_modes(rect_grid, 0.1, 0.1, 1.55, n_modes=1)
    assert modes[0]["psi"].shape == rect_grid.shape


def test_solve_modes_field_is_normalized(rect_grid):
    modes = solve_modes(rect_grid, 0.1, 0.1, 1.55, n_modes=1)
    psi = modes[0]["psi"]
    integral = np.sum(psi ** 2) * 0.1 * 0.1
    assert integral == pytest.approx(1.0, rel=1e-6)


def test_solve_modes_fundamental_peaks_inside_core(rect_grid):
    modes = solve_modes(rect_grid, 0.1, 0.1, 1.55, n_modes=1)
    psi = modes[0]["psi"]
    peak_idx = np.unravel_index(np.argmax(np.abs(psi)), psi.shape)
    assert rect_grid[peak_idx] == pytest.approx(3.4)  # peak field sits in the high-index core
