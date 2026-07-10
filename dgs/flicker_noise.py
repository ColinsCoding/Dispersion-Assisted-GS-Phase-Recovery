"""1/f (flicker / pink) noise: the symbolic reason it has equal power per octave, and how
to synthesize it on a computer.

FLICKER NOISE has a power spectral density that falls as S(f) = A / f^alpha. The famous case
alpha=1 -- "pink" or "1/f" noise -- is everywhere: the flicker floor of a transistor or op-amp,
the close-in PHASE NOISE of every oscillator and clock, the drift of a resistor, even loudness
in music and the flow of the Nile. It sits between "white" noise (alpha=0, flat, equal power per
hertz) and "brown"/random-walk noise (alpha=2).

WHY 1/f IS SPECIAL (the symbolic part).  Integrate the PSD over a band [f1, f2]:
        P = integral of A/f df from f1 to f2 = A * ln(f2 / f1).
The power depends only on the RATIO f2/f1 -- so every octave [f, 2f] carries the same power
A*ln 2, and every decade the same A*ln 10, no matter where you look. Equal power per octave is
the defining signature of pink noise, and it is exactly why 1/f noise looks self-similar across
scales. This module proves that in closed form with SymPy (a Mathematica-style CAS).

HOW TO MAKE IT (the computer part).  Spectral shaping (the Timmer-Koenig method): take white
noise, Fourier transform it, scale each amplitude by f^(-alpha/2) so the *power* scales as
f^(-alpha), drop the DC term, and inverse-transform. The resulting periodogram has slope
-alpha in log-log, which we verify by fitting. Ties to dgs.snr / dgs.adc_snr_bits (the noise
floor that sets a receiver's bit depth). NumPy + SymPy; py-3.13 (both available, torch is not).
"""

import numpy as np

# Standard "colors" of noise, keyed by the PSD exponent alpha in S(f) ~ 1/f^alpha.
NOISE_COLORS = {
    "violet": -2.0,   # S(f) ~ f^2   (differentiated white)
    "blue":   -1.0,   # S(f) ~ f
    "white":   0.0,   # flat, equal power per hertz
    "pink":    1.0,   # 1/f, equal power per octave  <-- flicker noise
    "brown":   2.0,   # 1/f^2, random walk (Brownian)
}


def pink_power_in_band(f1, f2, amplitude=1.0):
    """Power of pink (1/f) noise in the band [f1, f2]: integral of A/f df = A ln(f2/f1).
    Depends only on the ratio f2/f1 -- the hallmark of flicker noise."""
    if f1 <= 0 or f2 <= 0:
        raise ValueError("frequencies must be > 0 (1/f diverges at DC)")
    if f2 <= f1:
        raise ValueError("f2 must exceed f1")
    return amplitude * np.log(f2 / f1)


def power_per_octave(amplitude=1.0):
    """Pink-noise power in any octave [f, 2f]: A ln 2 -- the same for every octave."""
    return amplitude * np.log(2.0)


def power_per_decade(amplitude=1.0):
    """Pink-noise power in any decade [f, 10f]: A ln 10 -- the same for every decade."""
    return amplitude * np.log(10.0)


def symbolic_band_power():
    """Use SymPy to integrate the 1/f PSD symbolically and show the octave power is A*ln 2,
    independent of where the octave sits. Returns (band_power, octave_power) as SymPy exprs."""
    import sympy as sp
    f, f1, f2, A = sp.symbols("f f1 f2 A", positive=True)
    band = sp.integrate(A / f, (f, f1, f2))            # A*log(f2) - A*log(f1)
    band = sp.logcombine(sp.simplify(band), force=True)  # -> A*log(f2/f1)
    octave = sp.simplify(band.subs(f2, 2 * f1))          # -> A*log(2), no f1
    return band, octave


def generate_colored_noise(n, alpha=1.0, seed=None):
    """Synthesize `n` samples of noise with PSD S(f) ~ 1/f^alpha by spectral shaping
    (Timmer-Koenig): scale a white spectrum by f^(-alpha/2), drop DC, inverse-FFT.
    alpha=0 white, 1 pink, 2 brown. Returns a real, zero-mean array."""
    if n < 4:
        raise ValueError("n must be >= 4")
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.arange(spec.size, dtype=float)
    freqs[0] = 1.0                          # avoid div-by-zero at DC
    spec = spec * freqs ** (-alpha / 2.0)
    spec[0] = 0.0                           # remove DC offset
    out = np.fft.irfft(spec, n=n)
    return out - out.mean()


def periodogram(signal):
    """One-sided power spectral density estimate. Returns (freqs, psd) for f > 0
    (bin index as frequency in cycles/sample-block)."""
    n = len(signal)
    spec = np.fft.rfft(signal)
    psd = (np.abs(spec) ** 2) / n
    freqs = np.arange(spec.size, dtype=float)
    return freqs[1:], psd[1:]               # drop DC


def estimate_psd_slope(signal, f_lo_frac=0.01, f_hi_frac=0.5):
    """Fit the log-log slope of the periodogram over the middle band; for S(f)~1/f^alpha
    this returns approximately -alpha. Averaging/binning tames the per-bin chi-square scatter."""
    freqs, psd = periodogram(signal)
    fmax = freqs[-1]
    mask = (freqs >= f_lo_frac * fmax) & (freqs <= f_hi_frac * fmax) & (psd > 0)
    lf, lp = np.log(freqs[mask]), np.log(psd[mask])
    slope, _ = np.polyfit(lf, lp, 1)
    return slope


if __name__ == "__main__":
    band, octave = symbolic_band_power()
    print("=== symbolic: 1/f noise power in a band ===")
    print(f"  integral of A/f over [f1,f2] = {band}")
    print(f"  power in any octave [f,2f]   = {octave}   (independent of f -> equal per octave)")
    print(f"  numeric octave power  A ln2  = {power_per_octave():.5f}")
    print(f"  numeric decade power  A ln10 = {power_per_decade():.5f}")

    print("\n=== synthesized noise: fitted PSD slope should be -alpha ===")
    N = 2 ** 16
    for color, a in NOISE_COLORS.items():
        # average the fitted slope over a few realizations to tame the scatter
        slopes = [estimate_psd_slope(generate_colored_noise(N, alpha=a, seed=s))
                  for s in range(8)]
        print(f"  {color:7s} alpha={a:+.0f}   fitted slope = {np.mean(slopes):+.2f}  "
              f"(expected {-a:+.0f})")
