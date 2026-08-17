import numpy as np
import pytest
from dgs.rotation_vs_boost import (
    rotation_matrix, lorentz_boost_matrix, euclidean_metric, minkowski_metric_2d,
    preserves_metric, rotation_invariant, boost_invariant,
    rotation_composition_check, boost_composition_check,
)


def test_rotation_matrix_is_orthogonal():
    R = rotation_matrix(0.9)
    np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-10)


def test_rotation_matrix_determinant_is_one():
    R = rotation_matrix(1.234)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_lorentz_boost_matrix_rejects_superluminal_beta():
    with pytest.raises(ValueError):
        lorentz_boost_matrix(1.0)
    with pytest.raises(ValueError):
        lorentz_boost_matrix(1.5)


def test_lorentz_boost_matrix_zero_beta_is_identity():
    Lambda = lorentz_boost_matrix(0.0)
    np.testing.assert_allclose(Lambda, np.eye(2), atol=1e-10)


def test_preserves_metric_rotation_preserves_euclidean():
    R = rotation_matrix(0.4)
    assert preserves_metric(R, euclidean_metric())


def test_preserves_metric_rotation_does_not_preserve_minkowski():
    R = rotation_matrix(0.4)
    assert not preserves_metric(R, minkowski_metric_2d())


def test_preserves_metric_boost_preserves_minkowski():
    Lambda = lorentz_boost_matrix(0.5)
    assert preserves_metric(Lambda, minkowski_metric_2d())


def test_preserves_metric_boost_does_not_preserve_euclidean():
    Lambda = lorentz_boost_matrix(0.5)
    assert not preserves_metric(Lambda, euclidean_metric())


def test_rotation_invariant_preserves_length():
    result = rotation_invariant(0.7, np.array([3.0, 4.0]))
    assert result["length_after"] == pytest.approx(result["length_before"])
    assert result["preserves_euclidean_metric"] is True


def test_rotation_invariant_rejects_bad_vector_shape():
    with pytest.raises(ValueError):
        rotation_invariant(0.5, np.array([1.0, 2.0, 3.0]))


def test_boost_invariant_preserves_minkowski_not_euclidean():
    result = boost_invariant(beta=0.6, ct=5.0, x=3.0)
    assert result["minkowski_after"] == pytest.approx(result["minkowski_before"])
    assert result["euclidean_after"] != pytest.approx(result["euclidean_before"])
    assert result["preserves_minkowski_metric"] is True


def test_boost_invariant_matches_four_vector_boost_cross_check():
    result = boost_invariant(beta=0.6, ct=5.0, x=3.0)
    assert result["cross_check_invariant_orig"] == pytest.approx(result["minkowski_before"])
    assert result["cross_check_invariant_prime"] == pytest.approx(result["minkowski_after"])


def test_rotation_composition_angles_add():
    result = rotation_composition_check(0.3, 0.5)
    assert result["matches"] is True


def test_boost_composition_rapidities_add_not_velocities():
    result = boost_composition_check(0.3, 0.5)
    assert result["matches"] is True
    assert result["rapidities_add"] is True
    # velocities do NOT add linearly: combined beta should be < 0.3+0.5
    assert result["beta_combined"] < 0.8


def test_boost_composition_low_velocity_limit_is_nearly_additive():
    # at low beta, relativistic addition ~= Galilean addition (Correspondence principle)
    result = boost_composition_check(0.001, 0.002)
    assert result["beta_combined"] == pytest.approx(0.003, abs=1e-6)
