"""Synthetic optical and RF/microwave measurements -- ALL SYNTHETIC.

Four classes, each a spectrum sampled on a normalized axis u in [0, 1) with
N points, returned as amplitude (linear, arbitrary units) and phase (rad):

  optical-continuous    smooth thermal-like (Planck-shaped) spectrum, small chirp phase
  optical-line-spectrum a few narrow emission lines on a weak baseline
  RF-wideband           broad flat band with roll-off; linear phase (a pure delay)
  microwave-resonant    |S11| of a series-RLC resonator against 50 ohm: a
                        resonance dip in reflection, with the phase swing

Measurement model: additive white noise on amplitude (sigma_amp) and phase
(sigma_phase). Every generator takes a numpy Generator, so a fixed seed makes
the whole dataset reproducible.
"""
import numpy as np
from scipy.signal import find_peaks

from src.physics_models import reflection_coefficient, series_rlc_impedance

CLASSES = ("optical-continuous", "optical-line-spectrum", "RF-wideband", "microwave-resonant")
N_POINTS = 512


def _axis(n=N_POINTS):
    return np.linspace(0.0, 1.0, n, endpoint=False)


def optical_continuous(rng, n=N_POINTS):
    u = _axis(n)
    x = 0.15 + 0.85 * u                                    # normalized photon energy
    T = rng.uniform(0.12, 0.25)                            # normalized temperature
    amp = x ** 3 / np.expm1(x / T)
    amp /= amp.max()
    phase = rng.uniform(1.0, 4.0) * (u - 0.5) ** 2         # weak chirp
    return amp, phase


def optical_line_spectrum(rng, n=N_POINTS):
    u = _axis(n)
    amp = 0.05 * np.ones(n)
    for _ in range(rng.integers(3, 7)):
        c, w, h = rng.uniform(0.05, 0.95), rng.uniform(0.003, 0.008), rng.uniform(0.4, 1.0)
        amp += h * np.exp(-0.5 * ((u - c) / w) ** 2)
    amp /= amp.max()
    phase = rng.uniform(1.0, 4.0) * (u - 0.5) ** 2
    return amp, phase


def rf_wideband(rng, n=N_POINTS):
    u = _axis(n)
    lo, hi = rng.uniform(0.05, 0.2), rng.uniform(0.7, 0.95)
    edge = rng.uniform(0.02, 0.05)
    amp = 1 / (1 + np.exp(-(u - lo) / edge)) / (1 + np.exp((u - hi) / edge))
    amp *= 1 + 0.15 * np.sin(2 * np.pi * rng.uniform(2, 6) * u + rng.uniform(0, 2 * np.pi))
    amp /= amp.max()
    phase = -2 * np.pi * rng.uniform(0.3, 15) * u          # linear phase: group delay (short delays overlap the resonator)
    return amp, phase


def microwave_resonant(rng, n=N_POINTS, f_lo=1e9, f_hi=3e9):
    f = f_lo + (f_hi - f_lo) * _axis(n)
    f0 = rng.uniform(1.4e9, 2.6e9)
    Q = rng.uniform(15, 60)
    R = rng.uniform(40, 60)                                # near-matched at resonance
    L = Q * R / (2 * np.pi * f0)
    C = 1 / ((2 * np.pi * f0) ** 2 * L)
    gamma = reflection_coefficient(series_rlc_impedance(f, R, L, C))
    return np.abs(gamma), np.unwrap(np.angle(gamma))


GENERATORS = dict(zip(CLASSES, (optical_continuous, optical_line_spectrum, rf_wideband, microwave_resonant)))


def make_measurement(label, rng, sigma_amp=None, sigma_phase=None, n=N_POINTS):
    """One noisy measurement. sigma_amp / sigma_phase = None draws a random noise
    level per measurement (amplitude 0.02-0.25, phase 0.05-0.6 rad), so the SNR
    varies from shot to shot as it does on a real bench."""
    if label not in GENERATORS:
        raise ValueError(f"unknown class {label!r}; choose from {CLASSES}")
    amp, phase = GENERATORS[label](rng, n)
    if sigma_amp is None:
        sigma_amp = rng.uniform(0.02, 0.25)
    if sigma_phase is None:
        sigma_phase = rng.uniform(0.05, 0.6)
    if sigma_amp < 0 or sigma_phase < 0:
        raise ValueError("noise levels must be non-negative")
    return (amp + sigma_amp * rng.standard_normal(n),
            phase + sigma_phase * rng.standard_normal(n))


# ── feature extraction ───────────────────────────────────────────────

FEATURES = ("mean_intensity", "peak_count", "bandwidth", "phase_slope", "amplitude",
            "snr_db", "return_loss_like_db")


def noise_sigma_estimate(y):
    """Robust noise estimate from first differences: MAD(diff) / (0.6745 sqrt 2)."""
    d = np.diff(np.asarray(y, float))
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2))


def extract_features(amp, phase):
    """Seven interpretable features of one measurement (all on the normalized axis)."""
    amp = np.asarray(amp, float)
    phase = np.asarray(phase, float)
    u = _axis(amp.size)
    intensity = np.clip(amp, 0, None) ** 2
    sigma = max(noise_sigma_estimate(amp), 1e-9)
    peaks, _ = find_peaks(amp, prominence=max(5 * sigma, 0.05))
    sm = np.convolve(amp, np.ones(9) / 9, mode="same")
    above = u[sm >= 0.5 * sm.max()]                        # -6 dB amplitude (half-power ~ -3 dB) span, smoothed
    bandwidth = float(above.max() - above.min()) if above.size else 0.0
    slope = float(np.polyfit(u, np.unwrap(phase), 1)[0])
    signal_power = float(np.mean(amp ** 2))
    snr_db = 10 * np.log10(signal_power / sigma ** 2)
    floor = 3 * sigma                                      # cannot see a dip below the noise floor
    smooth = np.convolve(amp, np.ones(9) / 9, mode="same")
    depth = float(max(smooth[4:-4].min(), floor) / max(np.median(amp), floor))
    return {
        "mean_intensity": float(intensity.mean()),
        "peak_count": int(len(peaks)),
        "bandwidth": bandwidth,
        "phase_slope": slope,
        "amplitude": float(amp.max()),
        "snr_db": float(snr_db),
        "return_loss_like_db": float(-20 * np.log10(depth)),
    }


def make_dataset(n_per_class=120, seed=0, **noise):
    """(features DataFrame, labels Series, raw dict) -- deterministic for a seed."""
    import pandas as pd
    if n_per_class < 1:
        raise ValueError("n_per_class must be >= 1")
    rng = np.random.default_rng(seed)
    rows, labels, raw = [], [], {c: [] for c in CLASSES}
    for label in CLASSES:
        for _ in range(n_per_class):
            amp, phase = make_measurement(label, rng, **noise)
            rows.append(extract_features(amp, phase))
            labels.append(label)
            raw[label].append((amp, phase))
    return pd.DataFrame(rows, columns=FEATURES), pd.Series(labels, name="label"), raw


# ── optional: physics-informed features ──────────────────────────────

PHYSICS_FEATURES = ("spectral_flatness", "group_delay", "line_fraction")


def physics_informed_features(amp, phase):
    """Three features chosen from the physics of the four sources (a classical model):
      spectral_flatness  geometric / arithmetic mean of smoothed |amp| (1 = flat band)
      group_delay        -d(phase)/d(omega) on the normalized axis (a pure delay is linear phase)
      line_fraction      fraction of samples above half maximum (narrow lines -> small)
    """
    a = np.clip(np.convolve(np.abs(amp), np.ones(9) / 9, mode="same"), 1e-6, None)
    u = _axis(len(amp))
    flat = float(np.exp(np.mean(np.log(a))) / np.mean(a))
    delay = float(-np.polyfit(u, np.unwrap(phase), 1)[0] / (2 * np.pi))
    frac = float(np.mean(a >= 0.5 * a.max()))
    return {"spectral_flatness": flat, "group_delay": delay, "line_fraction": frac}


def physics_dataset(raw):
    import pandas as pd
    rows, labels = [], []
    for label, items in raw.items():
        for amp, phase in items:
            rows.append(physics_informed_features(amp, phase))
            labels.append(label)
    return pd.DataFrame(rows, columns=PHYSICS_FEATURES), pd.Series(labels, name="label")
