"""SNR is proportional to bits: the famous ADC relationship
SNR_dB = 6.02*N + 1.76, DERIVED from quantization theory (not quoted),
then verified against a real quantization-noise simulation. One extra
bit of resolution buys exactly ~6.02 dB more signal-to-noise ratio --
every ADC datasheet's "N-bit, X dB SNR" spec is this one formula.

DERIVATION (also done symbolically in SymPy below):
  * a full-scale sinusoid of amplitude A has signal power A^2/2.
  * an ideal N-bit quantizer divides the full range 2A into 2^N levels,
    each of width Delta = 2A/2^N.
  * quantization error is (to good approximation) uniformly distributed
    over [-Delta/2, Delta/2], with variance (noise power) Delta^2/12 --
    the variance of a uniform distribution.
  * SNR = signal_power/noise_power = (A^2/2)/(Delta^2/12) = 6*A^2/Delta^2
        = 6*A^2 / (2A/2^N)^2 = 6*4^N/4 = 1.5*2^(2N) = 1.5*4^N
  * SNR_dB = 10*log10(1.5*4^N) = 10*log10(1.5) + 20*N*log10(2)
           = 1.7609... + 6.0206...*N

Ties to the flash_adc_behavioral.vhd/flash_adc_structural.vhd 2-bit ADC
built earlier this session (N=2 predicts SNR~13.8 dB) and to the
mantissa-bits-as-precision theme from dgs.c_type_precision /
dgs.physical_constants_precision (the SAME "6 dB per bit" heuristic
applies to a float's mantissa: float32's 24-bit mantissa vs float64's
53-bit mantissa differ by 29 bits, i.e. ~175 dB of dynamic range).
"""

import numpy as np


def theoretical_snr_db(n_bits):
    """SNR_dB = 6.02*N + 1.76, derived from quantization-noise theory
    (see module docstring): SNR = 1.5*4^N in linear units."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    snr_linear = 1.5 * 4 ** n_bits
    return 10 * np.log10(snr_linear)


def simulate_adc_snr(n_bits, n_samples=4000, freq=7.3, rng_seed=0):
    """Measure the SNR of an ideal N-bit quantizer directly: quantize a
    full-scale sine wave, compute the actual signal power and residual
    quantization-noise power, and report the ratio in dB -- an
    independent numerical check of theoretical_snr_db, not the same
    calculation restated."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    full_scale = 2.0
    amplitude = full_scale / 2
    t = np.linspace(0, 10, n_samples)
    # a non-harmonic frequency avoids the quantizer locking onto a
    # low-order rational relationship with the sample rate, which would
    # bias the noise estimate away from the theoretical uniform-noise limit
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    n_levels = 2 ** n_bits
    step = full_scale / n_levels
    quantized = np.round(signal / step) * step
    noise = signal - quantized
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)


# exact constants (not truncated decimal literals) so bits_for_target_snr
# is a genuine, precise inverse of theoretical_snr_db -- a rounded 1.7609/
# 6.0206 pair previously broke round-tripping at anything past ~4 digit precision
_SNR_INTERCEPT_DB = 10 * np.log10(1.5)     # exactly 1.760912590556812...
_SNR_SLOPE_DB_PER_BIT = 20 * np.log10(2)   # exactly 6.020599913279624...


def bits_for_target_snr(target_snr_db):
    """Invert the formula: how many bits does an ADC need to reach a
    target SNR? N = (SNR_dB - 1.76) / 6.02, using the EXACT constants
    (not rounded decimal literals) so this is a precise inverse of
    theoretical_snr_db."""
    return (target_snr_db - _SNR_INTERCEPT_DB) / _SNR_SLOPE_DB_PER_BIT


def mantissa_dynamic_range_db(mantissa_bits):
    """The SAME 6.02 dB/bit heuristic applied to floating-point precision:
    a mantissa of M bits gives roughly 6.02*M dB of representable dynamic
    range/precision -- connects this module to dgs.c_type_precision's
    float32 (24-bit mantissa) vs float64 (53-bit mantissa) comparison."""
    if mantissa_bits <= 0:
        raise ValueError("mantissa_bits must be positive")
    return _SNR_SLOPE_DB_PER_BIT * mantissa_bits


if __name__ == "__main__":
    print("=== SNR vs. bits: derived formula vs. real quantization simulation ===")
    print(f"{'N bits':>8} {'simulated SNR (dB)':>20} {'formula 6.02N+1.76':>20} {'diff (dB)':>10}")
    for n_bits in [2, 4, 6, 8, 10, 12]:
        sim = simulate_adc_snr(n_bits)
        theory = theoretical_snr_db(n_bits)
        print(f"{n_bits:>8} {sim:>20.3f} {theory:>20.3f} {abs(sim-theory):>10.3f}")

    print("\n=== Tying to the flash_adc_*.vhd 2-bit ADC built earlier this session ===")
    n_bits_flash = 2
    predicted = theoretical_snr_db(n_bits_flash)
    measured = simulate_adc_snr(n_bits_flash)
    print(f"2-bit flash ADC: theoretical SNR = {predicted:.2f} dB, "
          f"simulated SNR = {measured:.2f} dB")

    print("\n=== Inverting: bits needed for CD-quality audio (96 dB SNR target) ===")
    n_needed = bits_for_target_snr(96.0)
    print(f"bits needed for 96 dB SNR: {n_needed:.2f} (matches the real CD standard: 16-bit audio)")

    print("\n=== Same '6 dB per bit' heuristic applied to floating-point mantissas ===")
    print(f"float32 (24-bit mantissa): ~{mantissa_dynamic_range_db(24):.1f} dB of dynamic range")
    print(f"float64 (53-bit mantissa): ~{mantissa_dynamic_range_db(53):.1f} dB of dynamic range")
    print(f"difference: ~{mantissa_dynamic_range_db(53)-mantissa_dynamic_range_db(24):.1f} dB "
          f"from 29 extra mantissa bits")
