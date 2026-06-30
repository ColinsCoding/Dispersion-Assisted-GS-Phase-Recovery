import numpy as np
import pytest
from dgs.spatial_computing import (
    perspective_divide, homogeneous_coords, dehomogenize,
    rot_x, rot_y, rot_z, translation_matrix, scale_matrix,
    quaternion_from_axis_angle, quaternion_multiply, quaternion_to_matrix,
    quaternion_slerp, perspective_matrix, project_points,
    stereo_depth, depth_uncertainty, phase_retrieval_vs_stereo_analogy,
    spatial_computing_sympy_5,
)


def test_perspective_divide_basic():
    p = perspective_divide([3, 2, 5], focal_length=2.0)
    assert p["x_screen"] == pytest.approx(1.2)
    assert p["y_screen"] == pytest.approx(0.8)
    assert p["depth"] == 5


def test_perspective_divide_z_zero_raises():
    with pytest.raises(ValueError):
        perspective_divide([1, 1, 0])


def test_homogeneous_round_trip():
    pt = [1.0, -2.0, 3.5]
    h = homogeneous_coords(pt)
    assert h[3] == 1.0
    back = dehomogenize(h)
    assert np.allclose(back, pt)


def test_dehomogenize_scaled():
    h = np.array([2.0, 4.0, 6.0, 2.0])
    assert np.allclose(dehomogenize(h), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("rot", [rot_x, rot_y, rot_z])
def test_rotation_orthogonal_det1(rot):
    R = rot(0.7)
    R3 = R[:3, :3]
    assert np.allclose(R3 @ R3.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R3) == pytest.approx(1.0)


def test_rotation_identity_at_zero():
    assert np.allclose(rot_z(0.0), np.eye(4))


def test_translation_matrix():
    T = translation_matrix(1, 2, 3)
    pt = np.array([0, 0, 0, 1.0])
    assert np.allclose(T @ pt, [1, 2, 3, 1])


def test_scale_matrix():
    S = scale_matrix(2, 3, 4)
    pt = np.array([1, 1, 1, 1.0])
    assert np.allclose(S @ pt, [2, 3, 4, 1])


def test_quaternion_identity_rotation():
    q = quaternion_from_axis_angle([0, 0, 1], 0.0)
    assert np.allclose(q, [1, 0, 0, 0])


def test_quaternion_90deg_z_rotates_x_to_y():
    q = quaternion_from_axis_angle([0, 0, 1], np.pi / 2)
    M = quaternion_to_matrix(q)
    pt = M[:3, :3] @ np.array([1, 0, 0])
    assert np.allclose(pt, [0, 1, 0], atol=1e-10)


def test_quaternion_180deg_x_flips_yz():
    q = quaternion_from_axis_angle([1, 0, 0], np.pi)
    M = quaternion_to_matrix(q)
    pt = M[:3, :3] @ np.array([0, 1, 0])
    assert np.allclose(pt, [0, -1, 0], atol=1e-10)


def test_quaternion_multiply_identity():
    q_id = np.array([1.0, 0, 0, 0])
    q = quaternion_from_axis_angle([0, 1, 0], 0.4)
    assert np.allclose(quaternion_multiply(q_id, q), q)


def test_quaternion_slerp_endpoints():
    q0 = quaternion_from_axis_angle([0, 0, 1], 0.0)
    q1 = quaternion_from_axis_angle([0, 0, 1], np.pi)
    assert np.allclose(quaternion_slerp(q0, q1, 0.0), q0, atol=1e-8)
    s1 = quaternion_slerp(q0, q1, 1.0)
    assert np.allclose(s1, q1, atol=1e-8) or np.allclose(s1, -q1, atol=1e-8)


def test_quaternion_slerp_midpoint_unit_norm():
    q0 = quaternion_from_axis_angle([0, 0, 1], 0.0)
    q1 = quaternion_from_axis_angle([0, 0, 1], np.pi / 2)
    qm = quaternion_slerp(q0, q1, 0.5)
    assert np.linalg.norm(qm) == pytest.approx(1.0)


def test_perspective_matrix_shape():
    P = perspective_matrix(60, 1.0, 0.1, 100)
    assert P.shape == (4, 4)


def test_project_points_basic():
    P = perspective_matrix(90, 1.0, 0.1, 100)
    pts = np.array([[0, 0, -1], [1, 0, -1]])
    res = project_points(pts, P)
    assert res["screen"].shape == (2, 2)
    assert res["depth"].shape == (2,)


def test_stereo_depth_numeric():
    sd = stereo_depth(np.array([10]), baseline_m=0.06, focal_px=800)
    expected = 0.06 * 800 / 10
    assert sd["depth_m"][0] == pytest.approx(expected)


def test_stereo_depth_zero_disparity_is_zero():
    # d=0 -> denominator set to inf in stereo_depth, so depth collapses to 0
    sd = stereo_depth(np.array([0]), baseline_m=0.06, focal_px=800)
    assert sd["depth_m"][0] == 0.0


def test_depth_uncertainty_positive():
    dz = depth_uncertainty(Z=5.0, baseline_m=0.06, focal_px=800, sigma_d=0.5)
    assert dz > 0
    assert dz == pytest.approx(5.0**2 * 0.5 / (0.06 * 800))


def test_phase_retrieval_vs_stereo_analogy_structure():
    a = phase_retrieval_vs_stereo_analogy()
    assert "Stereo imaging" in a
    assert "GS phase retrieval" in a
    assert "Remote sensing" in a
    assert "common_math" in a
    for key in ["Stereo imaging", "GS phase retrieval", "Remote sensing"]:
        assert "known" in a[key] and "unknown" in a[key]


def test_spatial_computing_sympy_5_count_and_type():
    import sympy as sp
    eqs = spatial_computing_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
