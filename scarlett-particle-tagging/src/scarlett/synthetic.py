"""Synthetic-data generator. EVERYTHING produced here is simulated.

No number from this module is an experimental result. The "truth" values are
choices made for the simulation; the "measurements" are that truth passed
through the sensor model with pseudo-random noise from a fixed seed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import physics as ph
from .sensor import (OpticalBench, SensorConfig, acquire, design_equalization,
                     scattered_power)

DEFAULT_SEED = 0

# Reference bead the equalization filter and the gain calibration are designed
# around (a simulated calibration standard, not a real product).
REFERENCE_DIA = 4.0e-6
REFERENCE_NPAR = 1.57


@dataclass(frozen=True)
class SampleTruth:
    """Ground truth of the synthetic sample (a dilute bead suspension in water)."""
    dia: float = 5.0e-6                 # [m]
    npar: float = 1.57                  # nominal polymer bead index
    concentration: float = 2.0e13       # [particles/m^3] -> 20 particles in a 1 nL probe volume
    temperature: float = 293.15         # [K]
    viscosity: float = 1.0016e-3        # water at 20 C [Pa s]
    tag_id: int = 0x5CA41E77            # label of the (simulated) sample container

    def __post_init__(self):
        if self.dia <= 0 or self.npar <= 0 or self.concentration < 0:
            raise ValueError("dia and npar must be positive, concentration non-negative")


def default_bench():
    """SEALS bench with the equalization filter designed for the reference bead."""
    return design_equalization(OpticalBench(), REFERENCE_DIA, REFERENCE_NPAR)


def channel_ripple(n_channels, rel=0.02, seed=1234):
    """Fixed multiplicative error per channel, 1 + rel * N(0, 1).

    Stands for everything the model does not know exactly (filter ripple,
    fibre coupling versus wavelength). It is the same on every scan, so
    averaging does not remove it; the estimator must budget for it.
    """
    if rel < 0:
        raise ValueError("rel must be non-negative")
    return 1.0 + rel * np.random.default_rng(seed).normal(size=n_channels)


def generate_angular_measurement(truth: SampleTruth, bench: OpticalBench, sensor: SensorConfig,
                                 rng=None, gain=None, gain_error=1.0, ripple=None,
                                 fixed_count=True):
    """One SEALS angular scan: ADC codes per channel.

    gain_error   multiplies the true optical power (an uncalibrated system
                 gain; calibration is supposed to find it).
    ripple       optional (J,) multiplicative error per channel (channel_ripple).
    fixed_count  True: exactly concentration * volume particles. False: the
                 count is Poisson distributed (particles drift in and out).
    """
    mean_count = truth.concentration * bench.probe_volume
    count = mean_count if (fixed_count or rng is None) else rng.poisson(mean_count)
    power = gain_error * scattered_power(truth.dia, truth.npar, count / bench.probe_volume, bench)
    if ripple is not None:
        power = power * np.asarray(ripple, dtype=float)
    out = acquire(power, sensor, rng=rng, gain=gain)
    out["n_particles_true"] = float(count)
    out["power_w"] = power
    return out


def generate_dark_measurement(bench: OpticalBench, sensor: SensorConfig, gain, rng=None):
    """Scan with the light blocked: dark current and offsets only."""
    return acquire(np.zeros(bench.n_channels), sensor, rng=rng, gain=gain)


def generate_dls_trace(truth: SampleTruth, bench: OpticalBench, n_samples, dt,
                       theta=np.pi / 2, mean_counts=400.0, rng=None):
    """Intensity time series from Brownian particles at one fixed angle.

    The scattered field is modelled as a complex Gaussian process with
    g1(tau) = exp(-Gamma tau) (an AR(1) recursion); intensity = |field|^2;
    photon counting adds Poisson noise. Returns (integer counts, Gamma).
    Assumes many independent scatterers, one coherence area on the detector,
    continuous-wave light at the centre wavelength, and a second detector at
    `theta` (default 90 degrees, where the decay is fast enough to measure).
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    if n_samples < 16 or dt <= 0 or mean_counts <= 0:
        raise ValueError("need n_samples >= 16, dt > 0, mean_counts > 0")
    gamma = float(ph.dls_decay_rate(truth.dia, bench.lam_center, bench.nmed, theta,
                                    truth.temperature, truth.viscosity))
    rho = np.exp(-gamma * dt)
    innovations = (rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples)) / np.sqrt(2)
    field = np.empty(n_samples, dtype=complex)
    field[0] = innovations[0]
    drive = np.sqrt(1 - rho ** 2)
    for i in range(1, n_samples):
        field[i] = rho * field[i - 1] + drive * innovations[i]
    counts = rng.poisson(mean_counts * np.abs(field) ** 2)
    return counts.astype(np.int64), gamma


def generate_dataset(seed=DEFAULT_SEED, truth: SampleTruth = SampleTruth(),
                     bench: OpticalBench | None = None,
                     sensor: SensorConfig = SensorConfig(),
                     ripple_rel=0.02, dls_samples=120_000, dls_dt=5e-3, dls_theta=np.pi / 2):
    """Complete synthetic dataset: angular scan, dark scan and a DLS trace."""
    bench = default_bench() if bench is None else bench
    rng = np.random.default_rng(seed)
    ripple = channel_ripple(bench.n_channels, ripple_rel)
    scan = generate_angular_measurement(truth, bench, sensor, rng=rng, ripple=ripple)
    dark = generate_dark_measurement(bench, sensor, scan["gain"], rng=rng)
    dls, gamma = generate_dls_trace(truth, bench, dls_samples, dls_dt, theta=dls_theta, rng=rng)
    return {
        "synthetic": True, "seed": seed, "truth": asdict(truth), "bench": bench,
        "sensor": sensor, "ripple_rel": ripple_rel,
        "wavelengths_m": bench.wavelengths, "scattering_angles_rad": bench.scattering_angles(),
        "usable_channels": bench.usable_channels(),
        "scan": scan, "dark": dark,
        "dls_counts": dls, "dls_dt_s": dls_dt, "dls_theta_rad": dls_theta,
        "dls_decay_rate_true_per_s": gamma,
    }
