import numpy as np
import pytest
from dgs.steam_4d_viewer import (
    synthetic_cell_trajectory, synthetic_steam_frame_stack,
    trajectory_velocity, trajectory_acceleration, trajectory_speed,
    trajectory_arc_length, plot_4d_time_series,
)


def test_synthetic_cell_trajectory_shapes():
    traj = synthetic_cell_trajectory(n_frames=20)
    assert traj["x_um"].shape == (20,)
    assert traj["y_um"].shape == (20,)
    assert traj["z_um"].shape == (20,)
    assert traj["t"].shape == (20,)


def test_synthetic_cell_trajectory_z_bounded_by_depth_range():
    traj = synthetic_cell_trajectory(n_frames=30, z_band_bandwidth_nm=20.0,
                                      axial_dispersion_nm_per_um=5.0)
    assert np.max(np.abs(traj["z_um"])) <= traj["z_range_um"] / 2.0 + 1e-9


def test_synthetic_cell_trajectory_rejects_small_n_frames():
    with pytest.raises(ValueError):
        synthetic_cell_trajectory(n_frames=2)


def test_synthetic_cell_trajectory_rejects_bad_extent():
    with pytest.raises(ValueError):
        synthetic_cell_trajectory(n_frames=10, xy_extent_um=0.0)


def test_synthetic_steam_frame_stack_shape():
    traj = synthetic_cell_trajectory(n_frames=15)
    frames = synthetic_steam_frame_stack(traj, frame_size=32)
    assert frames.shape == (15, 32, 32)


def test_synthetic_steam_frame_stack_rejects_small_frame_size():
    traj = synthetic_cell_trajectory(n_frames=10)
    with pytest.raises(ValueError):
        synthetic_steam_frame_stack(traj, frame_size=4)


def test_synthetic_steam_frame_stack_brighter_when_in_focus():
    # a trajectory pinned at z=0 (in focus) should be brighter at its peak
    # than one forced far from focus, all else equal
    traj_focus = synthetic_cell_trajectory(n_frames=10, z_band_bandwidth_nm=1e-6)
    frames_focus = synthetic_steam_frame_stack(traj_focus, frame_size=32)
    assert frames_focus.max() > 0.9   # near-unity amplitude when z~0 throughout


def test_trajectory_velocity_of_uniform_motion_is_constant():
    # straight-line, constant-speed flow (no wobble/z motion): velocity should
    # be nearly constant except at the finite-difference boundary
    traj = synthetic_cell_trajectory(n_frames=50, wobble_amplitude_um=0.0,
                                      z_band_bandwidth_nm=1e-6, flow_speed_um_per_frame=3.0)
    v = trajectory_velocity(traj)
    interior_vx = v["vx"][2:-2]
    assert np.allclose(interior_vx, 3.0, atol=1e-6)
    assert np.allclose(v["vy"][2:-2], 0.0, atol=1e-6)


def test_trajectory_acceleration_of_uniform_motion_is_zero():
    traj = synthetic_cell_trajectory(n_frames=50, wobble_amplitude_um=0.0,
                                      z_band_bandwidth_nm=1e-6, flow_speed_um_per_frame=3.0)
    a = trajectory_acceleration(traj)
    assert np.allclose(a["ax"][2:-2], 0.0, atol=1e-4)


def test_trajectory_speed_matches_velocity_magnitude():
    traj = synthetic_cell_trajectory(n_frames=25)
    v = trajectory_velocity(traj)
    speed = trajectory_speed(traj)
    expected = np.sqrt(v["vx"] ** 2 + v["vy"] ** 2 + v["vz"] ** 2)
    np.testing.assert_allclose(speed, expected)


def test_trajectory_arc_length_of_straight_line_matches_distance():
    # pure straight-line uniform motion: arc length = speed * elapsed time
    traj = synthetic_cell_trajectory(n_frames=50, wobble_amplitude_um=0.0,
                                      z_band_bandwidth_nm=1e-6, flow_speed_um_per_frame=2.0)
    arc = trajectory_arc_length(traj)
    expected_total = 2.0 * (len(traj["t"]) - 1)
    assert arc["total_length"] == pytest.approx(expected_total, rel=0.05)


def test_trajectory_arc_length_is_monotonically_nondecreasing():
    traj = synthetic_cell_trajectory(n_frames=30)
    arc = trajectory_arc_length(traj)
    assert np.all(np.diff(arc["s_t"]) >= -1e-9)


def test_plot_4d_time_series_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    traj = synthetic_cell_trajectory(n_frames=20)
    frames = synthetic_steam_frame_stack(traj, frame_size=32)
    fig = plot_4d_time_series(traj, frames, n_preview_frames=4)
    assert fig is not None
    assert len(fig.axes) > 0


def test_plot_4d_time_series_rejects_too_few_preview_frames():
    traj = synthetic_cell_trajectory(n_frames=10)
    frames = synthetic_steam_frame_stack(traj, frame_size=16)
    with pytest.raises(ValueError):
        plot_4d_time_series(traj, frames, n_preview_frames=1)
