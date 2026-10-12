"""Radio-frequency-tagged fluorescence imaging: a signal-processing model.

Inspired by FIRE (Fluorescence Imaging using Radiofrequency-tagged Emission;
Diebold, Buckley, Gossett and Jalali, Nature Photonics 2013). SCOPE, STATED
HONESTLY: this is NOT a model of that instrument's optics (acousto-optic
frequency comb, interferometer, scan mirror, photomultiplier electronics) and
its numbers are illustrative, not the paper's. It models the IDEA, which is
frequency-division multiplexing -- the same thing FM radio does:

  1. TAG.    Pixel k of a row is excited by light whose intensity beats at
             its own radio frequency  f_k = f_start + k * spacing.
             (Two beams offset by f_k interfere: |E1 + E2|^2 contains
             cos(2 pi f_k t); the optical frequency drops out.)
  2. EMIT.   The fluorophore at pixel k re-emits with the same beat, but a
             molecule with lifetime tau is a first-order low-pass filter:
                 modulation  M(f) = 1 / sqrt(1 + (2 pi f tau)^2)
                 phase lag   phi(f) = atan(2 pi f tau)
  3. DETECT. ONE detector sums every pixel. Photon counting adds shot noise.
  4. SAMPLE. An ADC samples at fs; every tag must sit below fs/2 (Nyquist).
  5. DEMOD.  A digital lock-in (multiply by the reference, accumulate) reads
             each pixel back out. On an FFT-bin grid this is exactly the FFT.

The three engineering results the module verifies numerically:

  throughput        Tones spaced by `spacing` are orthogonal only over a line
                    time T = 1/spacing (or a multiple). So line rate = spacing
                    and pixel rate = n_pixels * spacing = the RF bandwidth used.
  lifetime limit    M(f) rolls off at f ~ 1/(2 pi tau): nanosecond lifetimes
                    cap useful tag frequencies near 100 MHz, and the phase lag
                    measures tau (frequency-domain lifetime imaging).
  multiplex penalty All pixels share one detector, so every pixel's estimate
                    carries the shot noise of ALL the light:
                        sigma_k = sqrt(2 * N_total) / (n_samples * m_k)
                    A dim pixel beside bright ones is noisier than it would
                    be alone.

Units: "brightness" is the mean number of detected photons per ADC sample
from one pixel. NumPy only; py -3.13 safe.
"""
import numpy as np


# ----------------------------------------------------------------------
# 1. Tag frequencies, throughput, and the fluorophore's low-pass response
# ----------------------------------------------------------------------

def tag_frequencies(n_pixels, f_start_hz, spacing_hz):
    """f_k = f_start + k * spacing for k = 0 .. n_pixels - 1 [Hz]."""
    if int(n_pixels) != n_pixels or n_pixels < 1:
        raise ValueError("n_pixels must be a positive integer")
    if f_start_hz <= 0 or spacing_hz <= 0:
        raise ValueError("f_start_hz and spacing_hz must be positive")
    return f_start_hz + spacing_hz * np.arange(int(n_pixels))


def throughput(n_pixels, spacing_hz):
    """Speed limits set by the tag spacing alone.

    Returns dict with line_time_s = 1/spacing (shortest record that separates
    neighbouring tags), line_rate_hz = spacing, rf_bandwidth_hz =
    n_pixels * spacing, and pixel_rate_hz = line_rate * n_pixels (equal to the
    RF bandwidth: one pixel per second per hertz).
    """
    if int(n_pixels) != n_pixels or n_pixels < 1:
        raise ValueError("n_pixels must be a positive integer")
    if spacing_hz <= 0:
        raise ValueError("spacing_hz must be positive")
    return {
        "line_time_s": 1.0 / spacing_hz,
        "line_rate_hz": float(spacing_hz),
        "rf_bandwidth_hz": float(n_pixels * spacing_hz),
        "pixel_rate_hz": float(n_pixels * spacing_hz),
    }


def lifetime_response(f_hz, tau_s):
    """(modulation, phase_lag_rad) of a single-exponential fluorophore at f.

    The emission is the excitation convolved with exp(-t/tau)/tau, a
    first-order low-pass filter: the same transfer function as an RC circuit
    with RC = tau.
    """
    if tau_s < 0:
        raise ValueError("tau_s must be non-negative")
    x = 2 * np.pi * np.asarray(f_hz, dtype=float) * tau_s
    return 1.0 / np.sqrt(1.0 + x ** 2), np.arctan(x)


def lifetime_from_phase(phase_lag_rad, f_hz):
    """Invert phi = atan(2 pi f tau):  tau = tan(phi) / (2 pi f)."""
    phase = np.asarray(phase_lag_rad, dtype=float)
    if np.any(phase < 0) or np.any(phase >= np.pi / 2):
        raise ValueError("phase lag must be in [0, pi/2) for a single-exponential decay")
    return np.tan(phase) / (2 * np.pi * np.asarray(f_hz, dtype=float))


def design_line(n_pixels, f_start_hz, spacing_hz, oversample=2.5, periods=1):
    """ADC settings for one line with every tag exactly on an FFT bin.

    line time = periods / spacing; the sample count is the smallest integer
    giving fs >= oversample * f_max. f_start must be a multiple of spacing so
    that all tags complete a whole number of cycles in the line.

    Returns dict: freqs_hz, fs_hz, n_samples, line_time_s.
    """
    if oversample <= 2:
        raise ValueError("oversample must exceed 2 (Nyquist)")
    if int(periods) != periods or periods < 1:
        raise ValueError("periods must be a positive integer")
    freqs = tag_frequencies(n_pixels, f_start_hz, spacing_hz)
    ratio = f_start_hz / spacing_hz
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("f_start_hz must be an integer multiple of spacing_hz "
                         "for an orthogonal (leak-free) design")
    line_time = periods / spacing_hz
    n_samples = int(np.ceil(oversample * freqs[-1] * line_time))
    return {"freqs_hz": freqs, "fs_hz": n_samples / line_time,
            "n_samples": n_samples, "line_time_s": line_time}


# ----------------------------------------------------------------------
# 2. Forward model: one detector, many tagged pixels
# ----------------------------------------------------------------------

def _validate(brightness, freqs_hz, fs_hz, n_samples, mod_depth, tau_s):
    b = np.asarray(brightness, dtype=float)
    f = np.asarray(freqs_hz, dtype=float)
    if b.ndim != 1 or f.ndim != 1 or b.shape != f.shape:
        raise ValueError("brightness and freqs_hz must be 1-D arrays of equal length")
    if np.any(b < 0):
        raise ValueError("brightness (photons per sample) must be non-negative")
    if fs_hz <= 0 or int(n_samples) != n_samples or n_samples < 2:
        raise ValueError("fs_hz must be positive and n_samples an integer >= 2")
    if np.any(f <= 0) or np.any(f >= fs_hz / 2):
        raise ValueError(
            f"every tag frequency must lie in (0, fs/2 = {fs_hz/2:.4g} Hz); "
            f"got up to {f.max():.4g} Hz -- the ADC would alias it")
    if not 0 < mod_depth <= 1:
        raise ValueError("mod_depth must be in (0, 1]")
    if tau_s < 0:
        raise ValueError("tau_s must be non-negative")
    return b, f


def simulate_line(brightness, freqs_hz, fs_hz, n_samples, mod_depth=1.0,
                  tau_s=0.0, rng=None):
    """Detector samples for one image row.

    Expected photons in sample j:
        sum_k  b_k * [1 + m * M(f_k) * cos(2 pi f_k t_j - phi(f_k))]
    With rng=None the noiseless expectation is returned. Pass
    numpy.random.default_rng(seed) to draw Poisson photon counts instead.
    """
    b, f = _validate(brightness, freqs_hz, fs_hz, n_samples, mod_depth, tau_s)
    t = np.arange(int(n_samples)) / fs_hz
    M, lag = lifetime_response(f, tau_s)
    carriers = 1.0 + mod_depth * M[:, None] * np.cos(2 * np.pi * f[:, None] * t - lag[:, None])
    expected = b @ carriers
    if rng is None:
        return expected
    return rng.poisson(expected).astype(float)


# ----------------------------------------------------------------------
# 3. Demodulation: the digital lock-in
# ----------------------------------------------------------------------

def tone_amplitudes(signal, freqs_hz, fs_hz):
    """Complex amplitude of each tag: (2/n) * sum_j x_j * exp(-i 2 pi f t_j).

    This is one multiply-accumulate per sample per pixel, i.e. a digital
    lock-in. When f lands on an FFT bin it equals 2/n times that FFT bin.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("signal must be a 1-D array with at least 2 samples")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    t = np.arange(x.size) / fs_hz
    return (2.0 / x.size) * (np.exp(-2j * np.pi * f[:, None] * t) @ x)


def demodulate_line(signal, freqs_hz, fs_hz, mod_depth=1.0, tau_s=0.0, method="lockin"):
    """Recover each pixel's brightness (photons per sample) from one line.

    method="lockin"     project onto the known reference phase (including the
                        lifetime lag for the tau_s supplied). Linear and
                        unbiased; needs tau_s to be right.
    method="magnitude"  use |amplitude|. Needs no phase knowledge, but is
                        biased upward when the signal-to-noise ratio is low.

    Both divide by m * M(f_k), so the lifetime roll-off is corrected for the
    tau_s you pass. Passing tau_s=0 for a sample with a real lifetime leaves
    the high-frequency pixels too dim -- see the notebook.
    """
    if method not in ("lockin", "magnitude"):
        raise ValueError("method must be 'lockin' or 'magnitude'")
    if not 0 < mod_depth <= 1:
        raise ValueError("mod_depth must be in (0, 1]")
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    if np.any(f <= 0) or np.any(f >= fs_hz / 2):
        raise ValueError("every tag frequency must lie in (0, fs/2)")
    amp = tone_amplitudes(signal, f, fs_hz)
    M, lag = lifetime_response(f, tau_s)
    if method == "lockin":
        raw = np.real(amp * np.exp(1j * lag))
    else:
        raw = np.abs(amp)
    return raw / (mod_depth * M)


def measure_phase_lag(signal, freqs_hz, fs_hz):
    """Phase lag of each tag relative to the excitation reference [rad]."""
    return -np.angle(tone_amplitudes(signal, freqs_hz, fs_hz))


# ----------------------------------------------------------------------
# 4. Shot noise: the multiplex penalty
# ----------------------------------------------------------------------

def predicted_sigma(total_brightness, n_samples, mod_depth=1.0, modulation=1.0):
    """Shot-noise standard deviation of a lock-in brightness estimate.

    sigma = sqrt(2 * N_total) / (n_samples * m * M),  N_total = n_samples *
    total_brightness = every photon the detector collected in the line, from
    ALL pixels. It does not depend on which pixel is being estimated.
    """
    if total_brightness < 0 or n_samples < 2:
        raise ValueError("total_brightness must be >= 0 and n_samples >= 2")
    m_eff = mod_depth * np.asarray(modulation, dtype=float)
    if np.any(m_eff <= 0) or np.any(m_eff > 1):
        raise ValueError("effective modulation m * M must be in (0, 1]")
    return np.sqrt(2.0 * n_samples * total_brightness) / (n_samples * m_eff)


def predicted_snr(pixel_brightness, total_brightness, n_samples, mod_depth=1.0, modulation=1.0):
    """Signal-to-noise ratio of one pixel: brightness / predicted_sigma."""
    return pixel_brightness / predicted_sigma(total_brightness, n_samples, mod_depth, modulation)


# ----------------------------------------------------------------------
# 5. A whole image, one row at a time
# ----------------------------------------------------------------------

def simulate_image(image, f_start_hz, spacing_hz, peak_photons_per_sample,
                   mod_depth=1.0, tau_s=0.0, correct_lifetime=True,
                   oversample=2.5, periods=1, seed=None):
    """Image a 2-D array row by row and return (recovered, info).

    Each row is one line: its pixels are tagged, summed on one detector with
    Poisson noise (noiseless if seed is None) and demodulated. `image` is
    scaled so its maximum is `peak_photons_per_sample`; `recovered` is in the
    same photon units. info holds the line design and frame timing.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2-D array")
    if np.any(img < 0) or img.max() <= 0:
        raise ValueError("image must be non-negative with a positive maximum")
    if peak_photons_per_sample <= 0:
        raise ValueError("peak_photons_per_sample must be positive")

    rows, cols = img.shape
    design = design_line(cols, f_start_hz, spacing_hz, oversample, periods)
    truth = img / img.max() * peak_photons_per_sample
    rng = None if seed is None else np.random.default_rng(seed)

    recovered = np.empty_like(truth)
    for r in range(rows):
        signal = simulate_line(truth[r], design["freqs_hz"], design["fs_hz"],
                               design["n_samples"], mod_depth, tau_s, rng)
        recovered[r] = demodulate_line(signal, design["freqs_hz"], design["fs_hz"], mod_depth,
                                       tau_s if correct_lifetime else 0.0)
    info = dict(design, truth=truth, frame_time_s=rows * design["line_time_s"],
                frame_rate_hz=1.0 / (rows * design["line_time_s"]),
                **throughput(cols, spacing_hz))
    return recovered, info


if __name__ == "__main__":
    print("=== fire_imaging: radio-frequency-tagged fluorescence, signal model ===\n")

    n_pix, f0, df = 16, 20e6, 1e6
    d = design_line(n_pix, f0, df)
    tp = throughput(n_pix, df)
    print(f"{n_pix} pixels tagged {f0/1e6:.0f}-{d['freqs_hz'][-1]/1e6:.0f} MHz, spacing {df/1e6:.0f} MHz")
    print(f"  ADC: fs = {d['fs_hz']/1e6:.1f} MHz, {d['n_samples']} samples per line")
    print(f"  line rate {tp['line_rate_hz']/1e6:.1f} MHz, pixel rate {tp['pixel_rate_hz']/1e6:.0f} Mpixel/s")

    truth = np.linspace(5.0, 50.0, n_pix)
    clean = demodulate_line(simulate_line(truth, d["freqs_hz"], d["fs_hz"], d["n_samples"]),
                            d["freqs_hz"], d["fs_hz"])
    print(f"\nnoiseless round trip, max error: {np.max(np.abs(clean - truth)):.2e} photons/sample")

    tau = 4e-9
    M, lag = lifetime_response(d["freqs_hz"], tau)
    print(f"lifetime {tau*1e9:.0f} ns: modulation falls {M[0]:.2f} -> {M[-1]:.2f} across the tags, "
          f"phase lag {np.degrees(lag[0]):.0f} -> {np.degrees(lag[-1]):.0f} deg")

    sigma = predicted_sigma(truth.sum(), d["n_samples"])
    rng = np.random.default_rng(0)
    trials = np.array([demodulate_line(simulate_line(truth, d["freqs_hz"], d["fs_hz"],
                                                     d["n_samples"], rng=rng),
                                       d["freqs_hz"], d["fs_hz"]) for _ in range(2000)])
    print(f"shot noise per pixel: predicted {sigma:.3f}, Monte Carlo {trials.std(axis=0).mean():.3f} photons/sample")
    print("\n=== DONE ===")
