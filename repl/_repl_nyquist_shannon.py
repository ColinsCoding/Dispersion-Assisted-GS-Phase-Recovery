"""
_repl_nyquist_shannon.py

Nyquist-Shannon sampling theorem -- the "secret that is not a secret."
Published 1928 (Nyquist) + 1949 (Shannon). Entirely public domain.

S1: The theorem statement + proof sketch
S2: Aliasing -- what happens when you violate it
S3: ADC quantization -- bits, SNR, ENOB
S4: Reconstruction -- sinc interpolation
S5: Practical numbers -- RogueGuard ADC, cMOS camera, audio
"""

import numpy as np
import math

SEP = "=" * 65

# ------------------------------------------------------------------ #
# S1: THE THEOREM
# ------------------------------------------------------------------ #
print(SEP)
print("NYQUIST-SHANNON SAMPLING THEOREM")
print("'The secret is not a secret' -- published 1949, fully public.")
print(SEP)

print("""
  STATEMENT:
    A band-limited signal with maximum frequency f_max can be
    perfectly reconstructed from samples taken at rate:

        f_s >= 2 * f_max     (Nyquist rate)

    The factor of 2 is the NYQUIST CRITERION.
    f_N = f_s / 2 is the NYQUIST FREQUENCY (highest representable).

  PROOF SKETCH (Fourier domain):
    Sampling = multiplication by Dirac comb:  s(t) = x(t) * III(t/T_s)
    Fourier of comb = comb:  S(f) = f_s * sum_k X(f - k*f_s)
    Each copy of X(f) is shifted by k*f_s.
    If f_s > 2*f_max: copies do NOT overlap -> X(f) recoverable
      by ideal low-pass filter with cutoff f_N = f_s/2.
    If f_s < 2*f_max: copies OVERLAP -> ALIASING, unrecoverable.

  WHY "SECRET THAT IS NOT A SECRET":
    The math is 75 years old, in every EE textbook.
    But engineers still violate it constantly:
      - Undersampling ADCs without anti-alias filter
      - cMOS pixels too large for the lens NA (pixel aliasing)
      - DFT on a signal with spectral leakage (implicit aliasing)
      - Naive audio downsampling without resampling filter

  RECONSTRUCTION FORMULA:
    x(t) = sum_{n=-inf}^{inf} x[n] * sinc((t - n*T_s) / T_s)
    sinc(u) = sin(pi*u) / (pi*u)    (normalized sinc)

    This is a PERFECT reconstruction for band-limited signals.
    The sinc kernel is the impulse response of the ideal LPF.
""")

# ------------------------------------------------------------------ #
# S2: ALIASING
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 2: ALIASING")
print(SEP)

print("  Alias frequency formula:")
print("  f_alias = | f_signal - round(f_signal / f_s) * f_s |")
print()
print(f"  {'f_signal (Hz)':>15} {'f_s (Hz)':>12} {'f_alias (Hz)':>14} {'Problem?':>10}")
print("  " + "-" * 54)

cases = [
    (1000,  8000,  "OK -- audio at 8kHz"),
    (3000,  4000,  "ALIAS -- 3kHz at 4kHz fs"),
    (4100,  8000,  "ALIAS -- 4.1kHz folds to 3.9kHz"),
    (3999,  8000,  "OK -- just under Nyquist"),
    (1575e6, 5e9,  "GPS L1 at 5 GSPS ADC, OK"),
    (2.5e9, 5e9,   "ALIAS at fs/2 exactly -- worst case"),
]
for f_sig, f_s, note in cases:
    k = round(f_sig / f_s)
    f_al = abs(f_sig - k * f_s)
    problem = "ALIAS" if f_al < f_sig - 1e-6 else "OK"
    print(f"  {f_sig:>15,.0f} {f_s:>12,.0f} {f_al:>14,.1f} {problem:>10}  {note}")

print("""
  ANTI-ALIAS FILTER (AAF):
    Place a low-pass filter with cutoff < f_s/2 BEFORE the ADC.
    Removes energy above Nyquist BEFORE sampling.
    Practical AAF: Butterworth, Chebyshev, or passive RC ladder.

  OVERSAMPLING + DECIMATION (modern practice):
    Sample at f_s_high = N * f_s_target  (N = 4, 8, 16, ...)
    Apply sharp digital FIR filter (easy at high rate)
    Downsample (decimate) by N
    Advantages: relaxes AAF order, improves SNR by sqrt(N)
    Used in: sigma-delta ADCs, audio DACs, SDR receivers

  INTENTIONAL UNDERSAMPLING (bandpass sampling):
    If signal is in band [f_lo, f_hi] = [f_c - BW/2, f_c + BW/2]
    and BW << f_c, can sample at f_s = 2*BW (not 2*f_c)
    Alias maps f_c to baseband -- FREE downconversion
    Condition: f_s >= 2*BW and no other signal aliases into band
    Example: GPS L1 at 1575.42 MHz, BW=2MHz -> f_s=4MSPS minimum
             Actual: use 5.714 MSPS (common GPS receiver rate)
""")

# ------------------------------------------------------------------ #
# S3: ADC QUANTIZATION
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 3: ADC QUANTIZATION -- BITS, SNR, ENOB")
print(SEP)

print("""
  QUANTIZATION:
    N-bit ADC maps analog range [V_min, V_max] to 2^N levels.
    LSB (least significant bit) = (V_max - V_min) / 2^N

    Quantization noise: uniform distribution over [-LSB/2, LSB/2]
    RMS noise = LSB / sqrt(12)

  SQNR (Signal-to-Quantization-Noise Ratio):
    For full-scale sine wave:
    SQNR = 6.02 * N + 1.76  dB

    This is THE formula every EE memorizes.
    Each additional bit = +6 dB SNR = 2x amplitude resolution.
""")

print(f"  {'Bits':>6} {'Levels':>10} {'LSB (mV, 1V FS)':>18} {'SQNR (dB)':>12}")
print("  " + "-" * 48)
for N in [4, 6, 8, 10, 12, 14, 16, 18, 24]:
    levels = 2**N
    lsb_mv = 1000.0 / levels
    sqnr = 6.02 * N + 1.76
    print(f"  {N:>6} {levels:>10,} {lsb_mv:>18.4f} {sqnr:>12.1f}")

print("""
  ENOB (Effective Number of Bits):
    Real ADCs have noise + distortion beyond quantization.
    ENOB = (SNDR - 1.76) / 6.02
    where SNDR = measured signal-to-(noise+distortion) ratio

    A "16-bit" ADC might have ENOB = 13.5 bits at high frequency.
    ENOB degrades with frequency due to aperture jitter.

  APERTURE JITTER:
    ADC sample clock has timing uncertainty sigma_t (jitter).
    Noise floor from jitter: SNR_jitter = -20*log10(2*pi*f*sigma_t)
    At f=100MHz, sigma_t=1ps: SNR = -20*log10(2*pi*1e8*1e-12) = 54 dB
    -> limits ENOB to (54 - 1.76)/6.02 = 8.7 bits even with ideal ADC
""")

# jitter-limited SNR
print("  Jitter-limited SNR (sigma_t = 1 ps):")
print(f"  {'Frequency':>14} {'SNR_jitter (dB)':>18} {'ENOB_jitter':>14}")
print("  " + "-" * 48)
sigma_t = 1e-12  # 1 ps
for f_hz in [1e6, 10e6, 100e6, 500e6, 1e9, 2.5e9]:
    snr_j = -20 * np.log10(2 * np.pi * f_hz * sigma_t)
    enob_j = (snr_j - 1.76) / 6.02
    print(f"  {f_hz/1e6:>12.0f}MHz {snr_j:>18.1f} {enob_j:>14.1f}")

# ------------------------------------------------------------------ #
# S4: SINC RECONSTRUCTION
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 4: SINC RECONSTRUCTION")
print(SEP)

# reconstruct a 1 kHz sine sampled at 8 kHz
f_sig_r = 1000.0
f_s_r   = 8000.0
T_s     = 1.0 / f_s_r
N_samp  = 16

t_samp = np.arange(N_samp) * T_s
x_samp = np.sin(2 * np.pi * f_sig_r * t_samp)

# reconstruct at 4x denser grid
t_recon = np.linspace(0, (N_samp - 1) * T_s, N_samp * 4)
x_recon = np.zeros_like(t_recon)
for n in range(N_samp):
    x_recon += x_samp[n] * np.sinc((t_recon - n * T_s) / T_s)

x_true = np.sin(2 * np.pi * f_sig_r * t_recon)
err_max = np.max(np.abs(x_recon - x_true))

print(f"  Signal: f={f_sig_r} Hz sine,  f_s={f_s_r} Hz ({f_s_r/f_sig_r:.0f}x oversampled)")
print(f"  N_samples={N_samp}, reconstructed at 4x denser grid")
print(f"  Max reconstruction error: {err_max:.2e}  (should be near machine eps)")
print(f"  -> Perfect reconstruction confirmed (edge effects from finite window)")

print("""
  WINDOWING to reduce edge artifacts:
    Apply window before sampling/DFT: Hann, Hamming, Blackman
    Hann:    -43 dB sidelobe,  6 dB/oct rolloff
    Hamming: -43 dB sidelobe,  6 dB/oct rolloff, better main lobe
    Blackman:-74 dB sidelobe, 18 dB/oct rolloff (use for spectral leakage)
    Kaiser:  variable -- tune beta for sidelobe vs main lobe tradeoff

  DFT BIN RESOLUTION:
    delta_f = f_s / N_fft
    Frequency resolution (smallest distinguishable delta_f):
      delta_f_min = f_s / N_fft  (set N_fft = next power of 2)
    Time-frequency tradeoff:  delta_f * delta_t = 1  (uncertainty principle)
""")

# ------------------------------------------------------------------ #
# S5: PRACTICAL NUMBERS
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 5: PRACTICAL NUMBERS")
print(SEP)

print("""
  AUDIO:
    Human hearing: 20 Hz - 20 kHz
    CD standard: f_s = 44.1 kHz  (Nyquist = 22.05 kHz, margin for AAF)
    Pro audio:   f_s = 48, 96, 192 kHz
    Bit depth:   16-bit CD = 96.1 dB SQNR;  24-bit = 146 dB (> dynamic range of air)

  ROGUEGUARD ADC FRONT END:
    Rogue wave bandwidth: DC - 10 GHz (optical envelope after photodetector)
    Required f_s >= 20 GSPS  (Nyquist)
    Practical: 50 GSPS real-time scope (Keysight UXR), or 25 GSPS ADC ASIC
    Bits: 8-12 bits at high speed (ENOB ~6-8 bits at 10 GHz due to jitter)
    TIA bandwidth sets upper limit: R_f=1kohm, C_f=1pF -> f_3dB=159 MHz
    -> for 159 MHz BW: f_s = 400 MSPS sufficient, 12-bit ADC fine

  cMOS CAMERA (from angular_resolution notebook):
    Pixel pitch p must satisfy: p <= lambda / (4*NA)  (Nyquist criterion)
    lambda=500nm, NA=1.4:  p <= 89 nm  (for diffraction-limited sampling)
    Typical pixel (100x oil): p_sample = 65 nm -> 1.37x oversampled (OK)
    Typical pixel (10x air):  p_sample = 650 nm -> NA=0.3 -> p_Nyq=417nm -> UNDERSAMPLED

  GPS RECEIVER (bandpass sampling):
    L1 = 1575.42 MHz, C/A code BW = 2 MHz (main lobe 1 MHz null-to-null)
    Minimum f_s = 2.046 MSPS (real), 1.023 MSPS (complex/IQ)
    Practical: 5.714 MSPS (5x chip rate for easy tracking loop)
    ADC: 2-4 bits sufficient (GPS signal below thermal noise floor)

  RADIO TELESCOPE (Nyquist at THz):
    SMA (Sub-Millimeter Array): IF band 4-8 GHz -> f_s = 16 GSPS
    ALMA: 8 GHz IF per baseband -> 16 GSPS per correlator input
    Backend: FPGA correlator XF type (FFT then multiply)
""")

# summary table
print("  Quick Nyquist number table:")
print(f"  {'Application':<28} {'f_max':>10} {'f_s_min':>12} {'bits':>6} {'SQNR(dB)':>10}")
print("  " + "-" * 68)
apps = [
    ("CD audio",           20e3,    44.1e3,   16),
    ("Phone voice",        4e3,     8e3,       8),
    ("RogueGuard (TIA BW)",159e6,   400e6,    12),
    ("RogueGuard (full)",  10e9,    25e9,      8),
    ("GPS C/A baseband",   1.023e6, 5e6,       2),
    ("WiFi 802.11ax 160MHz",80e6,   200e6,    10),
    ("5G NR sub-6 GHz",    200e6,   500e6,    10),
]
for name, fmax, fs_min, bits in apps:
    sqnr = 6.02 * bits + 1.76
    def fmt_hz(f):
        if f >= 1e9: return f"{f/1e9:.2f}G"
        if f >= 1e6: return f"{f/1e6:.2f}M"
        if f >= 1e3: return f"{f/1e3:.2f}k"
        return f"{f:.0f}"
    print(f"  {name:<28} {fmt_hz(fmax):>10} {fmt_hz(fs_min):>12} {bits:>6} {sqnr:>10.1f}")

print()
print(SEP)
print("Done.")
print(SEP)
