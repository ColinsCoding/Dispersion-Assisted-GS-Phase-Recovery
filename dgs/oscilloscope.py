"""Oscilloscope theory, taught through the bad ideas that break a measurement.

A scope does three things -- SAMPLE, BANDLIMIT, and PROBE -- and each has a way to
lie to you. Learn the theory by learning the trap:

  BAD IDEA #1: "sample as fast as the signal." Sampling below the Nyquist rate
  (2x the highest frequency present) FOLDS high frequencies down to fake low ones
  -- ALIASING. A 7 kHz tone sampled at 10 kS/s shows up as a rock-steady 3 kHz
  wave that isn't there. The fix is 2x the *bandwidth* (not the frequency you care
  about), plus an anti-alias filter. Same fold as decimation in dgs.strided_slicing.

  BAD IDEA #2: "any scope that shows the waveform is fine." A scope of finite
  bandwidth BW has its own rise time t_scope = 0.35/BW, and it adds in quadrature:
        t_measured = sqrt(t_signal^2 + t_scope^2).
  Measure a 1 ns edge on a 350 MHz scope (t_scope = 1 ns) and you read 1.4 ns -- a
  41% error that is entirely the instrument. The rule "scope BW >= 5x signal
  bandwidth" is exactly the bandwidth that holds this error near 2% (derived below).
  Same 0.35/BW rise-time law as the RC gates in dgs.interconnect_delay.

  BAD IDEA #3: "the probe doesn't matter." A probe's input capacitance C_probe
  forms a low-pass with the source resistance R_source, cutoff f = 1/(2*pi R C).
  Put a 100 pF probe on a 10 kohm node and you have a 159 kHz filter -- you are
  measuring the PROBE, not the circuit. This is why 10x probes (low C, high R) exist.

Everything is a short closed-form check, plus a numerical aliasing demonstration
(sample a tone, FFT it, watch the peak land at the aliased frequency). NumPy only;
py-3.13.
"""

import numpy as np

RISE_TIME_K = 0.35      # t_r * BW = 0.35 for a single-pole (Gaussian-ish) response


# ----------------------------------------------------------------------
# BAD IDEA #1: undersampling -> aliasing
# ----------------------------------------------------------------------

def nyquist_rate(f_max):
    """The minimum sample rate to capture content up to f_max: 2*f_max.
    Sampling any slower folds frequencies above f_s/2 back down."""
    if f_max <= 0:
        raise ValueError("f_max must be positive")
    return 2 * f_max


def is_aliased(f_signal, f_sample):
    """True if f_signal sits above the Nyquist frequency f_sample/2 and will
    therefore be folded to a false lower frequency."""
    if f_signal < 0 or f_sample <= 0:
        raise ValueError("need f_signal >= 0 and f_sample > 0")
    return f_signal > f_sample / 2


def alias_frequency(f_signal, f_sample):
    """The APPARENT frequency a scope displays for f_signal sampled at f_sample:
    fold f_signal into [0, f_sample/2]. Below Nyquist it returns f_signal itself;
    above, it returns the (wrong) frequency you actually see."""
    if f_signal < 0 or f_sample <= 0:
        raise ValueError("need f_signal >= 0 and f_sample > 0")
    f = f_signal % f_sample
    return f_sample - f if f > f_sample / 2 else f


def observed_frequency_numeric(f_signal, f_sample, n_periods=200):
    """Actually sample a sine at f_sample and read back the dominant frequency
    from its FFT -- an independent check of alias_frequency, not the same
    formula restated. Returns the measured apparent frequency."""
    if f_signal <= 0 or f_sample <= 0:
        raise ValueError("frequencies must be positive")
    duration = n_periods / f_signal
    n = int(round(duration * f_sample))
    t = np.arange(n) / f_sample
    x = np.sin(2 * np.pi * f_signal * t)
    spec = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(n, 1 / f_sample)
    return float(freqs[np.argmax(spec)])


# ----------------------------------------------------------------------
# BAD IDEA #2: too little bandwidth -> inflated rise time
# ----------------------------------------------------------------------

def rise_time_from_bandwidth(bw):
    """A scope of bandwidth BW has rise time t_r = 0.35/BW (10-90%)."""
    if bw <= 0:
        raise ValueError("bandwidth must be positive")
    return RISE_TIME_K / bw


def bandwidth_from_rise_time(t_r):
    """Invert it: the bandwidth implied by a rise time, BW = 0.35/t_r."""
    if t_r <= 0:
        raise ValueError("rise time must be positive")
    return RISE_TIME_K / t_r


def measured_rise_time(signal_rise, scope_bw):
    """What the scope actually displays: the signal's edge convolved with the
    scope's own response adds in quadrature,
        t_measured = sqrt(t_signal^2 + (0.35/BW)^2)."""
    if signal_rise <= 0:
        raise ValueError("signal rise time must be positive")
    t_scope = rise_time_from_bandwidth(scope_bw)
    return np.hypot(signal_rise, t_scope)


def rise_time_error(signal_rise, scope_bw):
    """Fractional error the scope's bandwidth adds to the rise-time reading,
    (t_measured - t_signal)/t_signal. ~0 for a fast scope, large for a slow one."""
    return measured_rise_time(signal_rise, scope_bw) / signal_rise - 1.0


def required_bandwidth_for_error(signal_rise, max_error=0.02):
    """The scope bandwidth needed to keep the rise-time error at or below
    max_error. Solving sqrt(1+(t_scope/t_sig)^2) = 1+max_error gives a bandwidth
    that, for 2%, comes out to ~5x the signal's own bandwidth -- the origin of
    the '5x' rule of thumb."""
    if signal_rise <= 0 or max_error <= 0:
        raise ValueError("signal_rise and max_error must be positive")
    ratio = np.sqrt((1 + max_error) ** 2 - 1)      # t_scope / t_signal
    t_scope = ratio * signal_rise
    return RISE_TIME_K / t_scope


# ----------------------------------------------------------------------
# BAD IDEA #3: the probe loads the circuit
# ----------------------------------------------------------------------

def probe_cutoff(R_source, C_probe):
    """The low-pass corner formed by the source resistance and the probe's input
    capacitance: f_3dB = 1/(2*pi*R_source*C_probe). Above it, you measure the
    probe, not the circuit."""
    if R_source <= 0 or C_probe <= 0:
        raise ValueError("R_source and C_probe must be positive")
    return 1.0 / (2 * np.pi * R_source * C_probe)


def probe_amplitude_ratio(f, R_source, C_probe):
    """Fraction of the true amplitude the probe passes at frequency f:
    1/sqrt(1+(f/f_3dB)^2). At the cutoff it is 0.707 (-3 dB)."""
    if f < 0:
        raise ValueError("frequency must be non-negative")
    fc = probe_cutoff(R_source, C_probe)
    return 1.0 / np.sqrt(1 + (f / fc) ** 2)


if __name__ == "__main__":
    print("BAD IDEA #1 -- undersampling aliases a 7 kHz tone at 10 kS/s:")
    print(f"  Nyquist needs {nyquist_rate(7e3)/1e3:.0f} kS/s; at 10 kS/s aliased? {is_aliased(7e3, 10e3)}")
    print(f"  formula apparent freq = {alias_frequency(7e3, 10e3)/1e3:.2f} kHz, "
          f"measured from FFT = {observed_frequency_numeric(7e3, 10e3)/1e3:.2f} kHz")

    print("\nBAD IDEA #2 -- a 1 ns edge on scopes of different bandwidth:")
    for bw in (350e6, 1e9, 2e9):
        tm = measured_rise_time(1e-9, bw)
        print(f"  BW={bw/1e9:4.2f} GHz (t_scope={rise_time_from_bandwidth(bw)*1e9:.2f} ns): "
              f"measured {tm*1e9:.2f} ns, error {rise_time_error(1e-9, bw)*100:4.1f}%")
    bw5 = required_bandwidth_for_error(1e-9, 0.02)
    print(f"  for <=2% error: BW >= {bw5/1e9:.2f} GHz "
          f"(= {bw5/bandwidth_from_rise_time(1e-9):.1f}x the signal's own {bandwidth_from_rise_time(1e-9)/1e9:.2f} GHz -> the '5x rule')")

    print("\nBAD IDEA #3 -- a 100 pF probe on a 10 kohm node:")
    fc = probe_cutoff(10e3, 100e-12)
    print(f"  cutoff f_3dB = {fc/1e3:.1f} kHz; a 10 MHz signal reads "
          f"{probe_amplitude_ratio(10e6, 10e3, 100e-12)*100:.2f}% of its true amplitude")
