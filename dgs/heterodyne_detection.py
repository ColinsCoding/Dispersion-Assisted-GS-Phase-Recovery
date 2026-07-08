"""Heterodyne detection: measure a frequency by beating it against a known one.

A photodiode (or any square-law detector) is far too slow to follow an optical carrier
at ~10^14 Hz. So to measure an optical signal you MIX it with a strong local oscillator
(LO) of known frequency and detect the intensity. Square-law detection of the sum
        |E_sig e^{i w_s t} + E_LO e^{i w_LO t}|^2
       = |E_sig|^2 + |E_LO|^2 + 2 E_sig E_LO cos((w_s - w_LO) t + dphi)
throws away the fast carriers and leaves a BEAT at the DIFFERENCE frequency
f_IF = |f_s - f_LO| -- slow enough to digitize. This is heterodyne detection, the
basis of coherent optical receivers (dgs.iq_quadrature), radio, and FMCW LIDAR.

Two things make it powerful:

  * FREQUENCY MEASUREMENT. The beat note sits at f_IF, so measuring the beat (an FFT
    peak, refined by parabolic interpolation) recovers an unknown f_s from a known
    f_LO -- how laser linewidth, Doppler shifts, and gas lines are measured.

  * COHERENT GAIN. The beat amplitude is 2 E_sig E_LO -- LINEAR in the (weak) signal
    and multiplied by the (strong) LO. Direct detection of the signal alone gives only
    |E_sig|^2, which is vanishingly small; heterodyne lifts a weak signal far above the
    noise by beating it against a bright LO. A single photon's worth of signal becomes
    a macroscopic beat.

  * VELOCITY (Doppler / LIDAR). A moving target shifts the return by the Doppler
    f_D = 2 f0 v/c (round trip), so the beat frequency IS the velocity readout.

Verified: the detected beat sits at f_IF and is recovered to sub-bin accuracy, the
coherent gain scales as LO/signal, and the Doppler velocity round-trips. Complements
dgs.beat_frequency (superposition, standing waves). NumPy only; py-3.13.
"""

import numpy as np

C_LIGHT = 2.99792458e8


def intermediate_frequency(f_sig, f_lo):
    """The beat / intermediate frequency f_IF = |f_sig - f_LO| -- what the detector
    actually outputs after the fast carriers cancel."""
    if f_sig < 0 or f_lo < 0:
        raise ValueError("frequencies must be non-negative")
    return abs(f_sig - f_lo)


def photodetector_beat(f_sig, f_lo, fs, duration, A_sig=1.0, A_lo=1.0, phase=0.0):
    """Simulate the square-law detector output: DC term A_sig^2 + A_lo^2 plus the
    beat 2 A_sig A_lo cos(2 pi f_IF t + phase). Returns (t, detected). The optical
    carriers are gone -- only the beat at f_IF = |f_sig - f_lo| survives, so f_IF
    must be below Nyquist fs/2."""
    f_if = intermediate_frequency(f_sig, f_lo)
    if fs <= 0 or duration <= 0:
        raise ValueError("fs and duration must be positive")
    if f_if >= fs / 2:
        raise ValueError("beat frequency exceeds Nyquist -- raise fs")
    t = np.arange(int(fs * duration)) / fs
    dc = A_sig ** 2 + A_lo ** 2
    beat = 2 * A_sig * A_lo * np.cos(2 * np.pi * f_if * t + phase)
    return t, dc + beat


def estimate_frequency(signal, fs):
    """Recover the beat frequency from the detected signal: FFT, take the AC peak
    (DC removed), and refine to sub-bin accuracy with parabolic interpolation on
    the spectral magnitude. This is the actual frequency MEASUREMENT."""
    x = np.asarray(signal, float)
    n = len(x)
    if n < 8 or fs <= 0:
        raise ValueError("need >= 8 samples and fs > 0")
    X = np.abs(np.fft.rfft(x - x.mean()))
    k = int(np.argmax(X))
    if k == 0:
        return 0.0
    if 1 <= k < len(X) - 1:                       # parabolic peak interpolation
        a, b, c = X[k - 1], X[k], X[k + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return (k + delta) * fs / n


def coherent_gain(A_sig, A_lo):
    """Compare heterodyne to direct detection of a weak signal. Returns dict with
    the beat amplitude 2 A_sig A_lo, the direct-detection intensity A_sig^2, and
    the gain ratio 2 A_lo / A_sig -- huge when the LO is much stronger than the
    signal, which is why heterodyne sees signals direct detection cannot."""
    if A_sig <= 0 or A_lo <= 0:
        raise ValueError("amplitudes must be positive")
    return {
        "beat_amplitude": 2 * A_sig * A_lo,
        "direct_intensity": A_sig ** 2,
        "gain_ratio": 2 * A_lo / A_sig,
    }


def doppler_beat(velocity, f0, round_trip=True):
    """Beat frequency from a moving target: f_D = 2 f0 v/c (round trip, radar/LIDAR)
    or f0 v/c (one way). Positive v = approaching (higher beat)."""
    if f0 <= 0:
        raise ValueError("f0 must be positive")
    factor = 2 if round_trip else 1
    return factor * f0 * velocity / C_LIGHT


def velocity_from_beat(f_beat, f0, round_trip=True):
    """Invert the Doppler beat to a target velocity -- the LIDAR/radar readout."""
    if f0 <= 0:
        raise ValueError("f0 must be positive")
    factor = 2 if round_trip else 1
    return f_beat * C_LIGHT / (factor * f0)


if __name__ == "__main__":
    # two lasers ~1 MHz apart beat at 50 kHz -- easily measured, though neither
    # carrier (~1 MHz here, ~200 THz in reality) can be sampled directly
    f_sig, f_lo, fs = 1_000_000.0, 1_050_000.0, 2_000_000.0
    t, det = photodetector_beat(f_sig, f_lo, fs, duration=1e-3)
    f_est = estimate_frequency(det, fs)
    print(f"beat f_IF = {intermediate_frequency(f_sig, f_lo)/1e3:.1f} kHz, "
          f"measured {f_est/1e3:.3f} kHz")

    print("\ncoherent gain -- a weak signal beaten against a bright LO:")
    g = coherent_gain(A_sig=1e-3, A_lo=1.0)
    print(f"  beat amplitude {g['beat_amplitude']:.1e} vs direct intensity "
          f"{g['direct_intensity']:.1e}  -> gain x{g['gain_ratio']:.0f}")

    print("\nLIDAR: velocity from the Doppler beat (1550 nm, target at 30 m/s):")
    f0 = C_LIGHT / 1550e-9
    fb = doppler_beat(30.0, f0)
    print(f"  beat = {fb/1e6:.2f} MHz  ->  velocity {velocity_from_beat(fb, f0):.2f} m/s")
