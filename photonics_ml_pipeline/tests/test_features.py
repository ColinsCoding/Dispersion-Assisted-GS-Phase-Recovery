"""Feature-extraction tests: shapes, determinism, known values."""
from __future__ import annotations

import numpy as np

from feature_extraction.features import extract_features, feature_names


def test_feature_vector_length_and_names() -> None:
    field = np.ones(64)
    fv = extract_features(field)
    assert fv.to_array().shape == (len(feature_names()),)
    assert fv.names == feature_names()


def test_centroid_of_symmetric_profile_is_center() -> None:
    n = 129
    x = np.linspace(-1, 1, n)
    intensity = np.exp(-(x**2) / 0.1)  # symmetric about the middle index
    fv = extract_features(intensity).as_dict()
    assert np.isclose(fv["centroid"], (n - 1) / 2, atol=1e-6)


def test_extraction_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    field = rng.standard_normal(100) + 1j * rng.standard_normal(100)
    a = extract_features(field).to_array()
    b = extract_features(field).to_array()
    assert np.array_equal(a, b)


def test_wider_beam_has_larger_rms_width() -> None:
    x = np.linspace(-5, 5, 256)
    narrow = extract_features(np.exp(-(x**2) / 0.2)).as_dict()["rms_width"]
    wide = extract_features(np.exp(-(x**2) / 5.0)).as_dict()["rms_width"]
    assert wide > narrow


def test_empty_input_raises() -> None:
    try:
        extract_features(np.array([]))
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty input")
