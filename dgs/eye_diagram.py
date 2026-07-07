"""Eye diagrams -- the receiver's folded oscilloscope truth.

Trigger a scope on the recovered clock and overlay every bit period on top
of the previous one: the result looks like an eye. An OPEN eye means the
receiver can slice 1s from 0s reliably; noise closes it vertically, jitter
and dispersion close it horizontally. Every serial link datasheet (PCIe,
Ethernet, optical transceivers) specs its "eye mask" from exactly this plot.

The three numbers that matter:
  * Q-FACTOR. Sample the eye at its center; the 1s cluster at mu1 with
    spread sigma1, the 0s at mu0 with sigma0. Q = (mu1-mu0)/(sigma1+sigma0)
    is "how many combined standard deviations separate the rails."
  * BER. For Gaussian noise the slicer errs when noise crosses half the eye:
    BER = 0.5*erfc(Q/sqrt(2)). Q=6 is the classic "error-free" 1e-9 line;
    Q=7 gives ~1e-12. This is the same erfc as dgs.statistics' z-tests --
    a bit error IS a hypothesis test the receiver runs every bit period.
  * EYE OPENING. Height (mu0+3*sigma0 .. mu1-3*sigma1) and width (the run
    of sampling phases where 1-traces stay above 0-traces): the margin an
    ADC + clock with finite accuracy actually needs (dgs.adc_snr_bits).

DISPERSION closes the eye too: an intensity-modulated optical field through
a dispersive fiber picks up the all-pass phase H(f) = exp(j*pi*D*f^2) (the
same transfer function as dgs.coppinger_jalali_1999 time-stretch and the
TD-GS pipeline), and the photodiode's |.|^2 turns that pure phase into
amplitude ISI. apply_fiber_dispersion() reproduces the classic dispersion
power penalty -- feeds the BELLA_optical_ai_receiver_prototype notebook.

FOURIER PARITY (the dgs.even_odd connection, now in spectrum land):
split any real signal into circular even + odd parts. Under the DFT the
even part's spectrum is PURELY REAL and the odd part's is PURELY IMAGINARY
-- the odd one out is the one that comes back imaginary. Together they are
Hermitian symmetry F(-f) = conj(F(f)), the reason an N-point real FFT only
needs N/2 bins, and the symmetry the parity projectors in dgs.even_odd
diagonalize. Time axis is normalized to the bit period (T_bit = 1); the
dispersion parameter D is therefore in units of T_bit^2. NumPy only.
"""

import numpy as np
from math import erfc, sqrt


# ----------------------------------------------------------------------
# Waveform generation: PRBS bits -> band-limited NRZ / PAM levels
# ----------------------------------------------------------------------

def prbs_bits(n_bits, seed=0x5A):
    """PRBS7 bit pattern from the x^7 + x^6 + 1 LFSR -- the same generator
    BER testers use, so every 7-bit subsequence (except all-zeros) appears
    once per 127-bit period: a worst-case-ish ISI workout for the link."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    state = seed & 0x7F
    if state == 0:
        raise ValueError("seed must be nonzero in its low 7 bits")
    bits = np.empty(n_bits, dtype=int)
    for i in range(n_bits):
        bits[i] = state & 1
        newbit = ((state >> 6) ^ (state >> 5)) & 1
        state = ((state << 1) | newbit) & 0x7F
    return bits


def pam_waveform(symbols, sps=32, levels=2, bandwidth=0.75):
    """Band-limited PAM waveform: map symbols {0..levels-1} onto equally
    spaced amplitudes in [0, 1], hold each for one bit period (sps samples),
    then low-pass with a Gaussian filter of -3 dB (power) bandwidth
    `bandwidth` in units of the bit rate. bandwidth ~0.75/T is the classic
    scope "Bt = 0.75" setting; smaller values smear symbols into ISI.
    Returns the waveform sampled at dt = 1/sps (T_bit = 1)."""
    symbols = np.asarray(symbols)
    if symbols.ndim != 1 or len(symbols) < 4:
        raise ValueError("need a 1-D array of at least 4 symbols")
    if levels < 2:
        raise ValueError("levels must be >= 2 (2 = NRZ, 4 = PAM4)")
    if symbols.min() < 0 or symbols.max() > levels - 1:
        raise ValueError(f"symbols must lie in 0..{levels - 1}")
    if sps < 8:
        raise ValueError("sps must be >= 8 to resolve the eye")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive (units of bit rate)")
    amplitudes = symbols / (levels - 1)
    wave = np.repeat(amplitudes.astype(float), sps)
    # Gaussian low-pass: |H|^2 = 1/2 at f = bandwidth
    f = np.fft.fftfreq(len(wave), d=1.0 / sps)
    H = np.exp(-np.log(2) / 2 * (f / bandwidth) ** 2)
    return np.real(np.fft.ifft(np.fft.fft(wave) * H))


def add_noise(wave, sigma, rng_seed=0):
    """Additive white Gaussian noise -- the receiver's thermal/shot floor."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng(rng_seed)
    return wave + rng.normal(0.0, sigma, len(wave))


# ----------------------------------------------------------------------
# Dispersion: the Jalali all-pass phase, then the photodiode's |.|^2
# ----------------------------------------------------------------------

def apply_fiber_dispersion(wave, sps=32, D=0.0):
    """Send an intensity waveform through dispersive fiber + photodiode:
    field = sqrt(P), multiply its spectrum by the ALL-PASS quadratic phase
    H(f) = exp(j*pi*D*f^2) (identical to the dgs.coppinger_jalali_1999
    time-stretch kernel; D in units of T_bit^2), detect |field|^2.
    H changes no field amplitude -- only phase -- yet the detected eye
    closes, because square-law detection folds phase into amplitude: the
    dispersion power penalty, and the very phase the TD-GS pipeline
    recovers. D=0 returns the waveform unchanged."""
    wave = np.asarray(wave, float)
    if sps < 8:
        raise ValueError("sps must be >= 8")
    field = np.sqrt(np.clip(wave, 0.0, None))
    f = np.fft.fftfreq(len(wave), d=1.0 / sps)
    H = np.exp(1j * np.pi * D * f ** 2)
    out = np.fft.ifft(np.fft.fft(field) * H)
    return np.abs(out) ** 2


# ----------------------------------------------------------------------
# Folding and measuring the eye
# ----------------------------------------------------------------------

def fold_eye(wave, sps=32, n_ui=2):
    """Fold a waveform into overlaid eye traces: rows are successive
    bit-period-aligned slices, each n_ui unit intervals long. Plotting all
    rows against one 0..n_ui time axis IS the oscilloscope eye diagram."""
    wave = np.asarray(wave, float)
    if sps < 8:
        raise ValueError("sps must be >= 8")
    if n_ui < 1:
        raise ValueError("n_ui must be >= 1")
    n_traces = len(wave) // sps - n_ui
    if n_traces < 8:
        raise ValueError("waveform too short: need >= 8 full traces")
    return np.array([wave[i * sps: (i + n_ui) * sps] for i in range(n_traces)])


def eye_metrics(traces, sps=32):
    """Measure a binary (NRZ) eye from folded traces. Samples each trace at
    the center of its first unit interval, splits 1s from 0s at the midpoint
    threshold, and returns:
      q_factor    (mu1-mu0)/(sigma1+sigma0),
      ber         0.5*erfc(Q/sqrt(2)),
      eye_height  (mu1-3*sigma1) - (mu0+3*sigma0)  [3-sigma inner opening],
      eye_width   longest run of sampling phases (in UI) around center where
                  every 1-trace stays above every 0-trace,
      mu0/mu1/sigma0/sigma1 for the level histograms."""
    traces = np.asarray(traces, float)
    if traces.ndim != 2:
        raise ValueError("traces must be 2-D (use fold_eye)")
    if traces.shape[1] % sps != 0:
        raise ValueError("trace length must be a whole number of UIs (n*sps)")
    center = sps // 2
    samples = traces[:, center]
    threshold = (samples.min() + samples.max()) / 2
    ones, zeros = samples[samples > threshold], samples[samples <= threshold]
    if len(ones) < 2 or len(zeros) < 2:
        raise ValueError("need both 1s and 0s at the sampling instant "
                         "(is this really a binary eye?)")
    mu1, s1 = float(ones.mean()), float(ones.std())
    mu0, s0 = float(zeros.mean()), float(zeros.std())
    q = float("inf") if s1 + s0 == 0 else (mu1 - mu0) / (s1 + s0)
    # per-column inner opening: worst 1-trace minus best 0-trace
    is_one = samples > threshold
    opening = traces[is_one].min(axis=0) - traces[~is_one].max(axis=0)
    open_cols = opening > 0
    width = 0
    if open_cols[center]:
        lo = hi = center
        while lo > 0 and open_cols[lo - 1]:
            lo -= 1
        while hi < len(open_cols) - 1 and open_cols[hi + 1]:
            hi += 1
        width = hi - lo + 1
    return {
        "q_factor": q,
        "ber": ber_from_q(q),
        "eye_height": (mu1 - 3 * s1) - (mu0 + 3 * s0),
        "eye_width": width / sps,
        "mu0": mu0, "mu1": mu1, "sigma0": s0, "sigma1": s1,
    }


def ber_from_q(q):
    """BER = 0.5*erfc(Q/sqrt(2)): the Gaussian tail past the slicer.
    Q=0 -> 0.5 (coin flip), Q=6 -> ~1e-9, Q=7 -> ~1.3e-12."""
    if q == float("inf"):
        return 0.0
    if q < 0:
        raise ValueError("Q must be non-negative")
    return 0.5 * erfc(q / sqrt(2))


def q_for_target_ber(ber):
    """Invert ber_from_q by bisection: the Q-factor a link must deliver to
    hit a target BER (e.g. 1e-12 -> Q ~= 7.03). Companion to
    dgs.adc_snr_bits.bits_for_target_snr -- spec in, requirement out."""
    if not 0 < ber < 0.5:
        raise ValueError("target BER must be in (0, 0.5)")
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if ber_from_q(mid) > ber:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ----------------------------------------------------------------------
# Fourier parity: the even part stays real, the odd one turns imaginary
# ----------------------------------------------------------------------

def circular_reverse(x):
    """x[(-n) mod N]: time reversal on the DFT's circle (index 0 fixed).
    This -- not plain x[::-1] -- is the reversal the DFT's symmetry theorems
    refer to, exactly as parity_matrix in dgs.even_odd is reversal on a
    grid symmetric about 0."""
    x = np.asarray(x)
    return np.roll(x[::-1], 1)


def circular_even_part(x):
    """Even part on the DFT circle: (x[n] + x[-n]) / 2."""
    x = np.asarray(x, float)
    return (x + circular_reverse(x)) / 2


def circular_odd_part(x):
    """Odd part on the DFT circle: (x[n] - x[-n]) / 2."""
    x = np.asarray(x, float)
    return (x - circular_reverse(x)) / 2


def fourier_parity_check(x):
    """The DFT symmetry theorem, verified numerically on any real signal:
      * DFT(even part) is PURELY REAL   (cosines only),
      * DFT(odd part)  is PURELY IMAGINARY (sines only),
      * their sum obeys Hermitian symmetry F[-k] = conj(F[k]) -- why the
        FFT of N real samples carries only N/2 independent complex bins.
    Returns the residuals (should be ~1e-15, i.e. machine epsilon) and the
    spectra themselves for plotting."""
    x = np.asarray(x, float)
    if x.ndim != 1 or len(x) < 4:
        raise ValueError("need a 1-D real signal of at least 4 samples")
    xe, xo = circular_even_part(x), circular_odd_part(x)
    Fe, Fo = np.fft.fft(xe), np.fft.fft(xo)
    F = np.fft.fft(x)
    hermitian_residual = np.max(np.abs(F - np.conj(circular_reverse(F))))
    return {
        "even_spectrum_max_imag": float(np.max(np.abs(Fe.imag))),
        "odd_spectrum_max_real": float(np.max(np.abs(Fo.real))),
        "hermitian_residual": float(hermitian_residual),
        "even_spectrum": Fe, "odd_spectrum": Fo,
    }


if __name__ == "__main__":
    sps = 32
    bits = prbs_bits(600)
    clean = pam_waveform(bits, sps=sps)
    m = eye_metrics(fold_eye(clean, sps=sps), sps=sps)
    print(f"clean NRZ eye:      Q = {m['q_factor']:7.2f}   BER = {m['ber']:.2e}   "
          f"height = {m['eye_height']:.3f}  width = {m['eye_width']:.2f} UI")
    noisy = add_noise(clean, 0.05)
    m = eye_metrics(fold_eye(noisy, sps=sps), sps=sps)
    print(f"sigma=0.05 noise:   Q = {m['q_factor']:7.2f}   BER = {m['ber']:.2e}   "
          f"height = {m['eye_height']:.3f}  width = {m['eye_width']:.2f} UI")
    dispersed = add_noise(apply_fiber_dispersion(clean, sps=sps, D=0.4), 0.05)
    m = eye_metrics(fold_eye(dispersed, sps=sps), sps=sps)
    print(f"+ dispersion D=0.4: Q = {m['q_factor']:7.2f}   BER = {m['ber']:.2e}   "
          f"height = {m['eye_height']:.3f}  width = {m['eye_width']:.2f} UI")
    print(f"Q needed for BER 1e-12: {q_for_target_ber(1e-12):.3f}")
    chk = fourier_parity_check(clean)
    print(f"DFT parity: even part's spectrum imag <= {chk['even_spectrum_max_imag']:.1e}, "
          f"odd part's real <= {chk['odd_spectrum_max_real']:.1e} -- "
          f"the odd one comes back imaginary.")
