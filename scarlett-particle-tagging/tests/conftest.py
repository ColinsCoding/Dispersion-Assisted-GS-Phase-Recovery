"""Shared fixtures. Built once per test session because Mie lookup tables cost time."""
import numpy as np
import pytest

from scarlett import estimator as es
from scarlett import sensor as se
from scarlett import synthetic as sy


@pytest.fixture(scope="session")
def bench():
    return sy.default_bench()


@pytest.fixture(scope="session")
def sensor():
    return se.SensorConfig()


@pytest.fixture(scope="session")
def lut(bench, sensor):
    """Diameters 4.6 .. 5.4 um in 20 nm steps at the true refractive index."""
    return es.build_lookup_table(np.arange(4.6e-6, 5.4e-6 + 1e-12, 20e-9), [1.57], bench, sensor)


def scan_volts(truth, bench, sensor, seed, rel_uncertainty=0.02, ripple=None):
    """One noisy synthetic scan, dark-subtracted, with its uncertainty."""
    rng = np.random.default_rng(seed)
    scan = sy.generate_angular_measurement(truth, bench, sensor, rng=rng, ripple=ripple)
    dark = sy.generate_dark_measurement(bench, sensor, scan["gain"], rng=rng)
    volts = scan["volts"] - dark["volts"]
    noise = np.sqrt(scan["sigma_volts"] ** 2 + dark["sigma_volts"] ** 2)
    return scan, dark, volts, es.combined_sigma(volts, noise, rel_uncertainty)
