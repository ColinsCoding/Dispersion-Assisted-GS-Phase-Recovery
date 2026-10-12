"""Synthetic-data generator: reproducible, labelled synthetic, physically sensible."""
import numpy as np
import pytest

from scarlett import sensor as se
from scarlett import synthetic as sy


def test_dataset_is_reproducible_and_labelled_synthetic():
    a = sy.generate_dataset(seed=3, dls_samples=2000)
    b = sy.generate_dataset(seed=3, dls_samples=2000)
    c = sy.generate_dataset(seed=4, dls_samples=2000)
    assert a["synthetic"] is True
    np.testing.assert_array_equal(a["scan"]["code_sums"], b["scan"]["code_sums"])
    np.testing.assert_array_equal(a["dls_counts"], b["dls_counts"])
    assert not np.array_equal(a["scan"]["code_sums"], c["scan"]["code_sums"])


def test_scan_stays_inside_the_adc_range():
    data = sy.generate_dataset(seed=0, dls_samples=2000)
    scan = data["scan"]
    assert not scan["clipped"].any()
    assert scan["code_sums"].dtype == np.int64
    assert 0 < scan["codes"].min() and scan["codes"].max() < 4095
    assert scan["n_particles_true"] == pytest.approx(20.0)


def test_noiseless_scan_has_no_randomness():
    bench, sensor, truth = sy.default_bench(), se.SensorConfig(), sy.SampleTruth()
    a = sy.generate_angular_measurement(truth, bench, sensor)
    b = sy.generate_angular_measurement(truth, bench, sensor)
    np.testing.assert_array_equal(a["code_sums"], b["code_sums"])


def test_poisson_particle_count_varies():
    bench, sensor, truth = sy.default_bench(), se.SensorConfig(), sy.SampleTruth()
    rng = np.random.default_rng(0)
    counts = [sy.generate_angular_measurement(truth, bench, sensor, rng=rng, gain=32,
                                              fixed_count=False)["n_particles_true"] for _ in range(200)]
    assert np.mean(counts) == pytest.approx(20.0, rel=0.1)
    assert np.var(counts) == pytest.approx(20.0, rel=0.4)        # Poisson: variance = mean


def test_dls_trace_statistics():
    bench, truth = sy.default_bench(), sy.SampleTruth()
    counts, gamma = sy.generate_dls_trace(truth, bench, 50_000, 5e-3, rng=np.random.default_rng(1))
    assert counts.dtype == np.int64 and counts.min() >= 0
    assert gamma == pytest.approx(4.737, rel=1e-3)
    mean = counts.mean()
    # single-speckle intensity is exponentially distributed: variance ~ mean^2 (+ shot noise)
    assert counts.var() == pytest.approx(mean ** 2 + mean, rel=0.25)


def test_channel_ripple_is_fixed_for_a_seed():
    np.testing.assert_array_equal(sy.channel_ripple(42), sy.channel_ripple(42))
    assert np.std(sy.channel_ripple(10_000, rel=0.02)) == pytest.approx(0.02, rel=0.05)
    np.testing.assert_array_equal(sy.channel_ripple(5, rel=0.0), np.ones(5))


def test_invalid_truth_is_rejected():
    with pytest.raises(ValueError):
        sy.SampleTruth(dia=-1e-6)
    with pytest.raises(ValueError):
        sy.generate_dls_trace(sy.SampleTruth(), sy.default_bench(), 4, 1e-3)
