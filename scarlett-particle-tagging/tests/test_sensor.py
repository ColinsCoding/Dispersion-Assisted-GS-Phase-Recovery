"""Sensor-chain tests: photodiode, TIA, PGA, anti-alias filter, ADC, noise budget."""
from dataclasses import replace

import numpy as np
import pytest

from scarlett import physics as ph
from scarlett import sensor as se
from scarlett import synthetic as sy


def test_hand_calculated_operating_point(sensor):
    """1 nW -> 1 nA signal + 1 nA dark -> 2 mV at 1 Mohm -> gain 128 -> code 1049."""
    out = se.acquire(np.array([1e-9]), sensor)
    assert se.expected_volts(1e-9, sensor) == pytest.approx(2e-3)
    assert out["gain"] == 128
    assert out["codes"][0] == round(2e-3 * 128 / sensor.lsb) == 1049
    assert out["volts"][0] == pytest.approx(2e-3, abs=sensor.lsb / 128)
    assert not out["clipped"].any()


def test_noise_budget_formulas(sensor):
    b = sensor.noise_bandwidth
    assert b == pytest.approx(np.pi / 2 * 40e3)
    n = se.noise_current(1e-9, sensor)
    assert n["shot"] == pytest.approx(np.sqrt(2 * ph.E_CHARGE * 2e-9 * b))
    assert n["thermal"] == pytest.approx(np.sqrt(4 * ph.K_BOLTZMANN * 293.15 * b / 1e6))
    assert n["amplifier"] == pytest.approx(1e-13 * np.sqrt(b))
    assert n["total"] == pytest.approx(np.sqrt(n["shot"] ** 2 + n["thermal"] ** 2 + n["amplifier"] ** 2))
    assert se.noise_floor_power(sensor) == pytest.approx(4.08e-11, rel=0.01)


def test_shot_noise_grows_as_square_root_of_signal(sensor):
    quiet = replace(sensor, dark_current=1e-15)
    low, high = se.noise_current(1e-6, quiet)["shot"], se.noise_current(4e-6, quiet)["shot"]
    assert high / low == pytest.approx(2.0, rel=1e-6)


def test_measured_noise_matches_budget(sensor):
    """Scatter of repeated acquisitions agrees with sigma_volts."""
    rng = np.random.default_rng(0)
    power = np.full(8, 2e-9)
    volts = np.array([se.acquire(power, sensor, rng=rng)["volts"] for _ in range(3000)])
    predicted = se.acquire(power, sensor)["sigma_volts"]
    np.testing.assert_allclose(volts.std(axis=0), predicted, rtol=0.08)
    np.testing.assert_allclose(volts.mean(axis=0), se.expected_volts(power, sensor),
                               atol=5 * predicted.max() / np.sqrt(3000))


def test_averaging_reduces_noise_by_sqrt_n(sensor):
    one = se.acquire(np.array([2e-9]), replace(sensor, samples_per_channel=1), gain=64)["sigma_volts"]
    sixteen = se.acquire(np.array([2e-9]), sensor, gain=64)["sigma_volts"]
    assert one[0] / sixteen[0] == pytest.approx(4.0)


def test_pga_gain_selection_and_clipping(sensor):
    assert se.choose_pga_gain(0.5, sensor) == 1
    assert se.choose_pga_gain(0.05, sensor) == 16
    assert se.choose_pga_gain(1e-6, sensor) == 128
    assert se.choose_pga_gain(5.0, sensor) == 1                 # too big for any gain
    out = se.acquire(np.array([1e-9, 5e-6]), sensor)            # 5 uW -> 5 V at the TIA
    assert out["clipped"].tolist() == [False, True]
    assert out["codes"][1] == sensor.max_code


def test_invalid_sensor_settings_are_rejected(sensor):
    with pytest.raises(ValueError, match="Nyquist"):
        replace(sensor, antialias_cutoff=60e3)
    with pytest.raises(ValueError, match="power of two"):
        replace(sensor, samples_per_channel=10)
    with pytest.raises(ValueError, match="adc_bits"):
        replace(sensor, adc_bits=2)
    with pytest.raises(ValueError, match="non-negative"):
        se.acquire(np.array([-1e-9]), sensor)
    with pytest.raises(ValueError, match="not one of the PGA settings"):
        se.acquire(np.array([1e-9]), sensor, gain=3)


def test_bench_geometry(bench):
    assert bench.n_channels == 42
    assert bench.solid_angle == pytest.approx(1e-6 / 0.05 ** 2)
    theta = np.degrees(bench.scattering_angles())
    assert theta[0] == pytest.approx(56.5, abs=0.1) and theta[-1] == pytest.approx(-25.8, abs=0.1)
    usable = bench.usable_channels()
    assert usable.sum() == 38 and np.all(np.abs(theta[~usable]) < 5.0)


def test_equalization_flattens_the_reference_pattern(bench):
    plain = se.OpticalBench()
    usable = plain.usable_channels()
    raw = se.scattered_power_per_particle(sy.REFERENCE_DIA, sy.REFERENCE_NPAR, plain)[usable]
    flat = se.scattered_power_per_particle(sy.REFERENCE_DIA, sy.REFERENCE_NPAR, bench)[usable]
    assert raw.max() / raw.min() > 100                    # the dynamic-range problem
    assert flat.max() / flat.min() < raw.max() / raw.min() / 20
    eq = np.array(bench.equalization)
    assert eq.max() == pytest.approx(1.0) and eq.min() >= 1e-3 - 1e-12   # 30 dB limit
    with pytest.raises(ValueError, match="equalization"):
        replace(plain, equalization=(1.0, 0.5))


def test_power_is_proportional_to_concentration(bench):
    one = se.scattered_power(5e-6, 1.57, 1e13, bench)
    three = se.scattered_power(5e-6, 1.57, 3e13, bench)
    np.testing.assert_allclose(three, 3 * one, rtol=1e-12)
    np.testing.assert_allclose(one, 10 * se.scattered_power_per_particle(5e-6, 1.57, bench), rtol=1e-12)
    with pytest.raises(ValueError, match="non-negative"):
        se.scattered_power(5e-6, 1.57, -1.0, bench)


def test_photon_rate():
    """1 nW at 1590 nm is about 8e9 photons per second."""
    assert se.photon_rate(1e-9, 1590e-9) == pytest.approx(8.0e9, rel=0.01)
