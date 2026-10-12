"""Tests for reflection, even/odd decomposition and the symmetry score."""

import numpy as np
import pytest

from crack_photonics import io, symmetry

AXES = ["horizontal", "vertical"]


# ---------------------------------------------------------------- symmetric
@pytest.mark.parametrize("axis", AXES)
def test_symmetric_synthetic_image_scores_one(axis):
    image = io.generate_symmetric_image(seed=0, axis=axis)

    assert symmetry.symmetry_score(image, axis) == pytest.approx(1.0, abs=1e-9)
    assert symmetry.odd_energy_fraction(image, axis) == pytest.approx(0.0, abs=1e-12)

    _, odd = symmetry.even_odd_decompose(image, axis)
    assert np.max(np.abs(odd)) < 1e-12


def test_symmetry_survives_preprocessing():
    """The mirrored-border blur must not break an exact mirror symmetry."""
    image = io.preprocess(io.generate_symmetric_image(seed=0))
    assert symmetry.symmetry_score(image, "horizontal") == pytest.approx(1.0, abs=1e-9)


def test_symmetric_about_one_axis_is_not_symmetric_about_the_other():
    image = io.generate_symmetric_image(seed=0, axis="horizontal")
    assert symmetry.symmetry_score(image, "vertical", remove_mean=True) < 0.5


# --------------------------------------------------------------- asymmetric
@pytest.mark.parametrize("axis", AXES)
def test_asymmetric_crack_image_scores_below_symmetric(axis):
    crack = io.preprocess(io.generate_synthetic_crack(seed=0).image)
    flat = io.preprocess(io.generate_symmetric_image(seed=0, axis=axis))

    crack_score = symmetry.symmetry_score(crack, axis)
    assert crack_score < 0.95
    assert crack_score < symmetry.symmetry_score(flat, axis)
    assert symmetry.odd_energy_fraction(crack, axis) > 1e-3


def test_odd_part_highlights_the_crack():
    """Crack pixels carry more odd-part magnitude than the average pixel."""
    sample = io.generate_synthetic_crack(seed=0, noise_std=0.0)
    _, odd = symmetry.even_odd_decompose(sample.image, "horizontal")
    assert np.abs(odd)[sample.mask].mean() > 3 * np.abs(odd).mean()


# ----------------------------------------------------------- reconstruction
@pytest.mark.parametrize("axis", AXES)
def test_even_plus_odd_reconstructs_image(axis):
    image = io.generate_synthetic_crack(seed=3).image
    even, odd = symmetry.even_odd_decompose(image, axis)

    np.testing.assert_allclose(even + odd, image, rtol=0, atol=1e-15)
    # Parity eigenvalues: +1 for the even part, -1 for the odd part.
    np.testing.assert_allclose(symmetry.reflect(even, axis), even, atol=1e-15)
    np.testing.assert_allclose(symmetry.reflect(odd, axis), -odd, atol=1e-15)


@pytest.mark.parametrize("axis", AXES)
def test_even_and_odd_energies_add(axis):
    """Orthogonality: ||I||^2 = ||I_even||^2 + ||I_odd||^2."""
    image = io.generate_synthetic_crack(seed=1).image
    even, odd = symmetry.even_odd_decompose(image, axis)
    assert np.sum(even * odd) == pytest.approx(0.0, abs=1e-9)
    assert np.sum(even**2) + np.sum(odd**2) == pytest.approx(np.sum(image**2), rel=1e-12)


def test_reflecting_twice_is_identity():
    image = io.generate_synthetic_crack(seed=2).image
    for axis in AXES:
        np.testing.assert_array_equal(
            symmetry.reflect(symmetry.reflect(image, axis), axis), image
        )


# ------------------------------------------------------------- score limits
def test_score_lower_bound_for_non_overlapping_image():
    """A non-negative image that misses its own mirror gives S = 1 - sqrt(2)."""
    image = np.zeros((8, 8))
    image[:, :4] = 1.0
    assert symmetry.symmetry_score(image, "horizontal") == pytest.approx(1 - np.sqrt(2), abs=1e-9)


def test_all_zero_image_does_not_divide_by_zero():
    assert symmetry.symmetry_score(np.zeros((4, 4)), "horizontal") == 1.0


def test_analyze_symmetry_report_structure():
    report = symmetry.analyze_symmetry(io.generate_synthetic_crack(seed=0).image)
    assert set(report) == {"horizontal", "vertical"}
    for axis_report in report.values():
        assert set(axis_report) == {"score", "score_mean_removed", "odd_energy_fraction"}
        assert 0.0 <= axis_report["odd_energy_fraction"] <= 1.0


# ---------------------------------------------------------- reproducibility
def test_synthetic_generator_is_deterministic():
    first = io.generate_synthetic_crack(seed=7)
    second = io.generate_synthetic_crack(seed=7)
    other = io.generate_synthetic_crack(seed=8)

    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert not np.array_equal(first.image, other.image)


# ------------------------------------------------------------------- errors
def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="axis must be one of"):
        symmetry.reflect(np.ones((4, 4)), axis="diagonal")


def test_colour_array_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        symmetry.symmetry_score(np.ones((4, 4, 3)))


def test_nan_is_rejected():
    image = np.ones((4, 4))
    image[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        symmetry.even_odd_decompose(image)
