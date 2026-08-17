import numpy as np
import pytest
from dgs.spectral_fingerprint_id import (
    synthetic_reference_spectrum, build_synthetic_library, spectral_similarity,
    identify_spectrum, add_measurement_noise,
)


def test_synthetic_reference_spectrum_shape():
    grid = np.linspace(400, 1800, 500)
    spectrum = synthetic_reference_spectrum([800.0, 1200.0], [10.0, 15.0], [1.0, 0.5], grid)
    assert spectrum.shape == grid.shape


def test_synthetic_reference_spectrum_peaks_near_centers():
    grid = np.linspace(400, 1800, 2000)
    spectrum = synthetic_reference_spectrum([1000.0], [10.0], [1.0], grid)
    peak_wavenumber = grid[np.argmax(spectrum)]
    assert abs(peak_wavenumber - 1000.0) < 5.0


def test_synthetic_reference_spectrum_rejects_mismatched_lengths():
    grid = np.linspace(400, 1800, 500)
    with pytest.raises(ValueError):
        synthetic_reference_spectrum([800.0, 1200.0], [10.0], [1.0], grid)


def test_synthetic_reference_spectrum_rejects_empty_peaks():
    grid = np.linspace(400, 1800, 500)
    with pytest.raises(ValueError):
        synthetic_reference_spectrum([], [], [], grid)


def test_build_synthetic_library_has_requested_count_and_labels():
    grid = np.linspace(400, 1800, 500)
    library = build_synthetic_library(4, grid, rng_seed=0)
    assert len(library) == 4
    assert set(library.keys()) == {"Compound_A", "Compound_B", "Compound_C", "Compound_D"}


def test_build_synthetic_library_rejects_nonpositive_count():
    grid = np.linspace(400, 1800, 500)
    with pytest.raises(ValueError):
        build_synthetic_library(0, grid)


def test_spectral_similarity_identical_spectra_is_one():
    grid = np.linspace(400, 1800, 500)
    spectrum = synthetic_reference_spectrum([900.0], [10.0], [1.0], grid)
    assert spectral_similarity(spectrum, spectrum, metric="cosine") == pytest.approx(1.0, abs=1e-9)


def test_spectral_similarity_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        spectral_similarity(np.zeros(10), np.zeros(20))


def test_spectral_similarity_rejects_unknown_metric():
    a = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        spectral_similarity(a, a, metric="not_a_metric")


def test_spectral_similarity_rejects_zero_reference_for_cosine():
    a = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        spectral_similarity(a, np.zeros(3), metric="cosine")


def test_identify_spectrum_finds_exact_match():
    grid = np.linspace(400, 1800, 500)
    library = build_synthetic_library(5, grid, rng_seed=0)
    target = library["Compound_C"]
    result = identify_spectrum(target, library)
    assert result["best_label"] == "Compound_C"
    assert result["confident"] is True
    assert result["best_score"] == pytest.approx(1.0, abs=1e-9)


def test_identify_spectrum_rejects_empty_library():
    with pytest.raises(ValueError):
        identify_spectrum(np.array([1.0, 2.0]), {})


def test_identify_spectrum_rejects_bad_threshold():
    grid = np.linspace(400, 1800, 500)
    library = build_synthetic_library(2, grid)
    with pytest.raises(ValueError):
        identify_spectrum(library["Compound_A"], library, confidence_threshold=1.5)


def test_identify_spectrum_reports_no_match_for_unrelated_spectrum():
    grid = np.linspace(400, 1800, 500)
    library = build_synthetic_library(5, grid, rng_seed=1)
    unrelated = synthetic_reference_spectrum([1500.0], [5.0], [1.0], grid)
    result = identify_spectrum(unrelated, library, confidence_threshold=0.8)
    assert result["confident"] is False
    assert result["best_label"] is None


def test_identify_spectrum_ranked_matches_sorted_descending():
    grid = np.linspace(400, 1800, 500)
    library = build_synthetic_library(5, grid, rng_seed=0)
    result = identify_spectrum(library["Compound_B"], library)
    scores = [s for _, s in result["ranked_matches"]]
    assert scores == sorted(scores, reverse=True)


def test_add_measurement_noise_degrades_similarity():
    grid = np.linspace(400, 1800, 500)
    library = build_synthetic_library(3, grid, rng_seed=0)
    clean = library["Compound_A"]
    noisy = add_measurement_noise(clean, noise_std=0.3, rng_seed=5)
    sim_clean = spectral_similarity(clean, clean)
    sim_noisy = spectral_similarity(noisy, clean)
    assert sim_noisy < sim_clean


def test_add_measurement_noise_rejects_negative_std():
    with pytest.raises(ValueError):
        add_measurement_noise(np.array([1.0, 2.0]), noise_std=-0.1)


def test_add_measurement_noise_zero_std_is_unchanged():
    grid = np.linspace(400, 1800, 500)
    library = build_synthetic_library(1, grid, rng_seed=0)
    clean = library["Compound_A"]
    result = add_measurement_noise(clean, noise_std=0.0)
    np.testing.assert_allclose(result, clean)
