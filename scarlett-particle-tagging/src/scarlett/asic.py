"""ASIC behavioural model: an ARCHITECTURE PROPOSAL, not a fabricated chip.

Nothing here has been synthesized, laid out, taped out or measured. This
module is a bit-accurate behavioural simulation of the DIGITAL part of a
proposed mixed-signal chip, written so the arithmetic an ASIC would do
(integers, fixed word lengths, no division in the inner loop) can be checked
against the floating-point reference in scarlett.estimator. The analog blocks
are modelled in scarlett.sensor.

Proposed signal path
    photodiode input -> transimpedance amplifier -> programmable gain
    -> anti-alias filter -> ADC -> [averager] -> [dark subtraction]
    -> [correlator bank against calibration memory] -> [digital estimator]
    -> [autocorrelator for motion] -> low-power RF telemetry interface

Why a correlator bank
    For a template T_k, least squares over the particle count gives
        chi^2_k = |v|^2 - <v, T_k>^2 / |T_k|^2,
    so the best template maximizes  <v, T_k>^2 / |T_k|^2.  The inner loop is
    one multiply-accumulate per channel per template, the same structure as
    a matched-filter bank; each template then needs one multiplication by a
    stored reciprocal energy. This assumes equal noise on all channels (the
    reference estimator allows unequal noise).

    Templates are stored at (nearly) unit energy in 16 bits. "Nearly" matters:
    rounding to 16 bits changes each template's energy by about 1 part in
    10^4, and in the default configuration that shifts chi-square by a
    sizeable fraction (tens of percent) of the 95 % confidence threshold.
    The exact integer energy of each stored template is therefore used
    instead of assuming it is exactly one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Q15 = 32767   # full scale of a signed 16-bit template sample


@dataclass(frozen=True)
class AsicConfig:
    """Word lengths of the proposed digital datapath."""
    template_bits: int = 16        # signed, unit-energy templates in Q15
    measurement_bits: int = 20     # dark-subtracted averaged ADC sum (12-bit ADC x 16 samples + sign)
    accumulator_bits: int = 48     # correlator accumulator
    autocorr_lags: int = 64


# Block list used for documentation and for the README block diagram.
ASIC_BLOCKS = (
    ("photodiode input", "analog", "current input pad, ESD, guard ring", "sensor.acquire"),
    ("transimpedance amplifier", "analog", "current to voltage, gain R_f", "sensor.acquire"),
    ("programmable gain", "analog", "x1 .. x128 in powers of two", "sensor.choose_pga_gain"),
    ("anti-alias filter", "analog", "low-pass below fs/2", "sensor.SensorConfig.noise_bandwidth"),
    ("ADC", "mixed", "12-bit, 100 kS/s (nominal)", "sensor.acquire"),
    ("averager + dark subtraction", "digital", "sum 2^n samples, subtract stored dark", "asic.subtract_dark"),
    ("autocorrelation block", "digital", "integer lag products for motion", "asic.integer_autocorrelation"),
    ("calibration memory", "digital", "dark codes, unit-energy Mie templates, norms", "asic.CalibrationMemory"),
    ("digital estimator", "digital", "correlator bank, argmax, confidence set", "asic.fixed_point_estimate"),
    ("RF telemetry interface", "digital", "packet framing + CRC to a low-power radio", "telemetry.encode_packet"),
)


@dataclass(frozen=True)
class CalibrationMemory:
    """What the chip would hold in non-volatile memory after calibration."""
    templates_q15: np.ndarray    # (K, J_used) int16, each row unit energy * Q15
    norm_counts: np.ndarray      # (K,) template norm in code-sum units per particle
    dark_code_sums: np.ndarray   # (J,) int64, dark level per channel (same gain, same n)
    mask: np.ndarray             # (J,) bool, channels used
    gain: int                    # PGA gain the table was built for
    samples_per_channel: int


def build_calibration_memory(lut, dark_code_sums, gain, sensor):
    """Quantize a floating-point lookup table into ASIC calibration memory.

    lut.templates are volts per particle; the chip works in integer code
    sums, so each template is converted (volts * gain * n / LSB), normalized
    to unit energy and stored as Q15.
    """
    counts_per_volt = gain * sensor.samples_per_channel / sensor.lsb
    t = lut.templates[:, lut.mask] * counts_per_volt
    norms = np.linalg.norm(t, axis=1)
    if np.any(norms <= 0):
        raise ValueError("a template has zero energy on the usable channels")
    q = np.round(t / norms[:, None] * Q15).astype(np.int16)
    return CalibrationMemory(q, norms, np.asarray(dark_code_sums, dtype=np.int64),
                             lut.mask.copy(), int(gain), int(sensor.samples_per_channel))


def subtract_dark(code_sums, memory: CalibrationMemory):
    """Dark-subtracted integer measurement on the usable channels (int64)."""
    code_sums = np.asarray(code_sums)
    if not np.issubdtype(code_sums.dtype, np.integer):
        raise TypeError("code_sums must be integers (the ADC sample sums)")
    if code_sums.shape != memory.dark_code_sums.shape:
        raise ValueError("code_sums has the wrong number of channels")
    return (code_sums.astype(np.int64) - memory.dark_code_sums)[memory.mask]


def fixed_point_estimate(meas, templates_q15, noise_sigma_counts, delta_chi2=3.841458820694124,
                         cfg: AsicConfig = AsicConfig()):
    """Correlator-bank estimator in integer arithmetic.

    meas               (J_used,) int64 dark-subtracted code sums
    templates_q15      (K, J_used) int16 unit-energy templates
    noise_sigma_counts RMS noise of one element of `meas` [counts]
    delta_chi2         chi-square threshold of the confidence set (3.84 = 95 %)

    For each template k:  stat_k = corr_k^2 / template_energy_k  [counts^2],
    with corr_k = <meas, T_k> and template_energy_k = sum(T_k^2), both exact
    integers. chi^2_k = energy - stat_k. Template k is accepted when
    corr_k > 0 and stat_k >= stat_best - delta_chi2 * sigma^2.

    Returns dict: best_index, corr (K,) int, template_energy (K,) int, energy,
    chi2_min [counts^2], accepted (K,) bool, ci_lo_index, ci_hi_index,
    ambiguous. Checks that the accumulators fit the proposed word length;
    mirrors c/scarlett_estimator.c exactly.
    """
    meas = np.asarray(meas)
    tq = np.asarray(templates_q15)
    if not np.issubdtype(meas.dtype, np.integer) or tq.dtype != np.int16:
        raise TypeError("meas must be integer and templates_q15 int16")
    if tq.ndim != 2 or meas.shape != (tq.shape[1],):
        raise ValueError("shape mismatch between measurement and templates")
    if noise_sigma_counts <= 0:
        raise ValueError("noise_sigma_counts must be positive")
    if np.max(np.abs(meas)) >= 2 ** (cfg.measurement_bits - 1):
        raise OverflowError("measurement exceeds the proposed measurement word length")

    corr = tq.astype(np.int64) @ meas.astype(np.int64)
    limit = 2 ** (cfg.accumulator_bits - 1)
    if np.max(np.abs(corr)) >= limit:
        raise OverflowError("correlator accumulator would overflow")
    energy = int(np.sum(meas.astype(np.int64) ** 2))

    template_energy = np.sum(tq.astype(np.int64) ** 2, axis=1)
    if np.any(template_energy == 0):
        raise ValueError("a template is all zeros")
    positive = corr > 0
    stat = np.where(positive, corr.astype(float) * corr.astype(float) / template_energy.astype(float), -1.0)
    best = int(np.argmax(stat))
    if not positive[best]:
        raise ValueError("no template correlates positively with the measurement")
    threshold = stat[best] - delta_chi2 * noise_sigma_counts * noise_sigma_counts
    accepted = positive & (stat >= threshold)
    idx = np.flatnonzero(accepted)
    return {
        "best_index": best, "corr": corr, "template_energy": template_energy, "energy": energy,
        "chi2_min": float(energy) - float(stat[best]),
        "accepted": accepted, "ci_lo_index": int(idx[0]), "ci_hi_index": int(idx[-1]),
        "ambiguous": bool(idx[-1] - idx[0] + 1 != idx.size),
    }


def particle_count_from_correlation(corr_best, template_energy_best, norm_counts_best):
    """N = <v, T_stored> / (|T_stored| * |T|): undo the template normalization.

    T_stored is the Q15 row, |T| (norm_counts) the norm of the unquantized
    template in code-sum units per particle."""
    return corr_best / np.sqrt(float(template_energy_best)) / norm_counts_best


def integer_autocorrelation(samples, max_lag, cfg: AsicConfig = AsicConfig()):
    """R[l] = sum_i x[i] x[i+l] for l = 0 .. max_lag, as integers.

    The autocorrelation block of the proposal. Normalizing to g2 and fitting
    the decay need a division and a logarithm, which are left to the digital
    estimator / host (scarlett.estimator.fit_decay_rate).
    """
    x = np.asarray(samples)
    if not np.issubdtype(x.dtype, np.integer) or x.ndim != 1:
        raise TypeError("samples must be a 1-D integer array")
    if not 0 <= max_lag < x.size:
        raise ValueError("max_lag must be in [0, len(samples))")
    x = x.astype(np.int64)
    return np.array([int(np.dot(x[: x.size - lag], x[lag:])) for lag in range(max_lag + 1)],
                    dtype=object)


def g2_from_integer_autocorrelation(acf, samples):
    """Convert integer lag sums to normalized g2: (R[l] / (n - l)) / mean^2."""
    x = np.asarray(samples, dtype=float)
    lags = np.arange(len(acf))
    return np.array([float(a) for a in acf]) / (x.size - lags) / x.mean() ** 2


# ----------------------------------------------------------------------
# Test vectors for the C implementation (c/main.c reads these files)
# ----------------------------------------------------------------------
def write_estimate_vectors(path, meas, templates_q15, noise_sigma_counts,
                           delta_chi2=3.841458820694124):
    """Write a measurement and template table in the text format of c/main.c."""
    tq = np.asarray(templates_q15)
    meas = np.asarray(meas)
    if tq.ndim != 2 or meas.shape != (tq.shape[1],):
        raise ValueError("shape mismatch between measurement and templates")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"{tq.shape[1]} {tq.shape[0]} {float(noise_sigma_counts)!r} {float(delta_chi2)!r}\n")
        f.write(" ".join(str(int(v)) for v in meas) + "\n")
        for row in tq:
            f.write(" ".join(str(int(v)) for v in row) + "\n")


def write_acf_vectors(path, samples, max_lag):
    """Write an integer time series in the text format of c/main.c (acf mode)."""
    samples = np.asarray(samples)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"{samples.size} {int(max_lag)}\n")
        f.write(" ".join(str(int(v)) for v in samples) + "\n")
