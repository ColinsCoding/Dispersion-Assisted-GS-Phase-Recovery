"""Tests for segmentation, crack metrics and the simplified PST edge detector."""

import numpy as np
import pytest

from crack_photonics import crack_metrics as cm
from crack_photonics import io, pst_edges
from crack_photonics.cli import analyze


def bar_mask(shape=(100, 200), rows=slice(48, 53), cols=slice(50, 150)):
    """A filled rectangle: by default 5 px thick and 100 px long, horizontal."""
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    return mask


def metrics_of(mask):
    """compute_metrics with a neutral image and no edges (geometry only)."""
    image = np.where(mask, 0.2, 0.8)
    return cm.compute_metrics(mask, image, np.zeros_like(mask))


# ------------------------------------------------------------- percentages
def test_area_percent_known_fraction():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :5] = True  # 25 of 100 pixels
    assert cm.crack_area_percent(mask) == pytest.approx(25.0)


def test_area_percent_empty_and_full():
    assert cm.crack_area_percent(np.zeros((7, 9), dtype=bool)) == 0.0
    assert cm.crack_area_percent(np.ones((7, 9), dtype=bool)) == 100.0


def test_area_percent_non_square_image():
    mask = np.zeros((20, 50), dtype=bool)
    mask[3, :10] = True  # 10 of 1000 pixels
    assert cm.crack_area_percent(mask) == pytest.approx(1.0)


def test_area_percent_matches_compute_metrics():
    mask = bar_mask()
    result = metrics_of(mask)
    assert result["crack_pixel_count"] == 500
    assert result["crack_area_percent"] == pytest.approx(100.0 * 500 / 20000)


def test_area_percent_rejects_non_boolean_mask():
    with pytest.raises(TypeError, match="dtype bool"):
        cm.crack_area_percent(np.ones((4, 4)))


# ---------------------------------------------------------------- geometry
def test_horizontal_bar_length_width_orientation():
    result = metrics_of(bar_mask())
    assert result["crack_length_px"] == pytest.approx(100, abs=6)
    assert result["mean_width_px"] == pytest.approx(5, abs=0.5)
    assert result["max_width_px"] == pytest.approx(5, abs=0.01)
    assert result["orientation_deg"] == pytest.approx(0.0, abs=1e-6)
    assert result["orientation_anisotropy"] > 0.99
    assert result["connected_components"] == 1


def test_vertical_bar_orientation_is_90_degrees():
    result = metrics_of(bar_mask(shape=(200, 100), rows=slice(50, 150), cols=slice(48, 53)))
    assert result["orientation_deg"] == pytest.approx(90.0, abs=1e-6)


def test_rising_diagonal_is_45_degrees_and_length_uses_sqrt2():
    """A line towards the upper right: 45 deg, length n * sqrt(2)."""
    n = 60
    mask = np.zeros((n, n), dtype=bool)
    mask[np.arange(n)[::-1], np.arange(n)] = True
    result = metrics_of(mask)
    assert result["orientation_deg"] == pytest.approx(45.0, abs=1e-6)
    assert result["crack_length_px"] == pytest.approx((n - 1) * np.sqrt(2) + 1, rel=1e-9)


def test_falling_diagonal_is_135_degrees():
    mask = np.eye(60, dtype=bool)
    assert metrics_of(mask)["orientation_deg"] == pytest.approx(135.0, abs=1e-6)


def test_connected_components_counts_separate_pieces():
    mask = np.zeros((60, 60), dtype=bool)
    mask[10, 5:30] = True
    mask[40, 5:30] = True
    mask[25, 50] = True
    assert cm.count_components(mask) == 3


def test_diagonal_touch_counts_as_connected():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = mask[2, 2] = True
    assert cm.count_components(mask) == 1


def test_skeleton_is_thin_and_inside_mask():
    mask = bar_mask()
    skeleton = cm.skeletonize(mask)
    assert skeleton.any()
    assert not (skeleton & ~mask).any()
    assert skeleton.sum(axis=0).max() == 1  # one pixel per column for a horizontal bar


def test_empty_mask_gives_zero_metrics_not_an_error():
    result = metrics_of(np.zeros((32, 32), dtype=bool))
    assert result["crack_area_percent"] == 0.0
    assert result["crack_length_px"] == 0.0
    assert result["mean_width_px"] == 0.0
    assert result["orientation_deg"] is None
    assert result["connected_components"] == 0
    assert result["confidence"] == 0.0


# --------------------------------------------------------------------- units
def test_millimetre_conversion_requires_explicit_scale():
    mask = bar_mask()
    image = np.where(mask, 0.2, 0.8)
    edges = np.zeros_like(mask)

    unscaled = cm.compute_metrics(mask, image, edges)
    assert unscaled["crack_length_mm"] is None and unscaled["mean_width_mm"] is None

    scaled = cm.compute_metrics(mask, image, edges, pixel_size_mm=0.1)
    assert scaled["crack_length_mm"] == pytest.approx(0.1 * scaled["crack_length_px"])
    assert scaled["max_width_mm"] == pytest.approx(0.5, abs=0.001)


def test_non_positive_pixel_size_is_rejected():
    mask = bar_mask()
    with pytest.raises(ValueError, match="pixel_size_mm"):
        cm.compute_metrics(mask, mask.astype(float), mask, pixel_size_mm=0.0)


# ------------------------------------------------- full pipeline, synthetic
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pipeline_recovers_synthetic_crack(seed):
    sample = io.generate_synthetic_crack(seed=seed)
    result = analyze(sample.image)
    mask, crack = result["mask"], result["crack"]

    iou = (mask & sample.mask).sum() / (mask | sample.mask).sum()
    assert iou > 0.7

    truth_percent = cm.crack_area_percent(sample.mask)
    assert crack["crack_area_percent"] == pytest.approx(truth_percent, rel=0.25)
    assert crack["crack_length_px"] == pytest.approx(sample.true_length_px, rel=0.15)
    assert crack["mean_width_px"] == pytest.approx(sample.true_width_px, rel=0.35)
    assert crack["connected_components"] == 1
    assert 0.0 <= crack["confidence"] <= 1.0
    assert crack["confidence"] > 0.7
    # The synthetic crack runs from upper left to lower right.
    assert 150.0 < crack["orientation_deg"] < 175.0


def test_pipeline_finds_nothing_on_crack_free_surface():
    result = analyze(io.generate_symmetric_image(seed=0))
    assert result["crack"]["crack_area_percent"] < 0.1
    assert result["crack"]["confidence"] < 0.3


def test_pipeline_is_deterministic():
    image = io.generate_synthetic_crack(seed=5).image
    first, second = analyze(image), analyze(image)
    np.testing.assert_array_equal(first["mask"], second["mask"])
    np.testing.assert_array_equal(first["edge_strength"], second["edge_strength"])
    assert first["crack"] == second["crack"]
    assert first["symmetry"] == second["symmetry"]


# ----------------------------------------------------- simplified PST edges
def test_pst_marks_a_step_edge_and_not_the_flat_regions():
    image = np.full((64, 64), 0.3)
    image[:, 32:] = 0.8
    strength, edges = pst_edges.detect_edges(image)

    assert strength.shape == image.shape
    assert 0.0 <= strength.min() and strength.max() == pytest.approx(1.0)
    assert strength[:, 29:35].mean() > 10 * strength[:, :20].mean()
    assert edges[:, 29:35].any()
    assert not edges[:, :20].any() and not edges[:, 44:].any()


def test_pst_phase_kernel_is_radially_increasing():
    kernel = pst_edges.pst_phase_kernel((64, 64), phase_strength=0.5, warp_strength=15.0)
    assert kernel[0, 0] == 0.0
    assert kernel.max() == pytest.approx(0.5)
    assert np.all(np.diff(kernel[0, :32]) > 0)  # grows with frequency


def test_pst_rejects_bad_parameters():
    image = np.random.default_rng(0).random((32, 32))
    with pytest.raises(ValueError, match="phase_strength"):
        pst_edges.pst_phase(image, phase_strength=0.0)
    with pytest.raises(ValueError, match="lowpass_sigma"):
        pst_edges.pst_phase(image, lowpass_sigma=-1.0)


# ------------------------------------------------------------------- errors
def test_constant_image_is_rejected_by_preprocess():
    with pytest.raises(ValueError, match="constant"):
        io.preprocess(np.full((32, 32), 0.5))


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        io.load_image(tmp_path / "does_not_exist.png")


def test_undecodable_file_raises_value_error(tmp_path):
    bad = tmp_path / "not_an_image.png"
    bad.write_text("this is not a PNG")
    with pytest.raises(ValueError, match="Could not decode"):
        io.load_image(bad)


def test_image_round_trip_through_png(tmp_path):
    image = io.generate_synthetic_crack(seed=0).image
    loaded = io.load_image(io.save_image(tmp_path / "round_trip.png", image))
    assert loaded.shape == image.shape
    assert np.max(np.abs(loaded - image)) <= 0.5 / 255 + 1e-12  # 8-bit quantisation
