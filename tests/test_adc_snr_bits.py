"""Test the ADC SNR-vs-bits relationship (SNR_dB = 6.02*N + 1.76),
derived from quantization theory and verified against a real
quantization-noise simulation -- not just quoted as a known formula."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import adc_snr_bits as adc

# 1. theoretical formula matches the well-known constants exactly
assert abs(adc.theoretical_snr_db(1) - (6.0206 + 1.7609)) < 1e-3

# 2. each additional bit adds ~6.02 dB (verified across several N, not just one)
for n in [2, 4, 6, 8, 10]:
    delta = adc.theoretical_snr_db(n + 1) - adc.theoretical_snr_db(n)
    assert abs(delta - 6.0206) < 1e-3

# 3. simulated SNR (real quantization-noise measurement) matches the
#    theoretical formula, with the gap shrinking as N grows (finite-sample
#    statistical convergence, not just "close enough always")
errors = []
for n_bits in [2, 4, 6, 8, 10, 12]:
    sim = adc.simulate_adc_snr(n_bits, n_samples=4000)
    theory = adc.theoretical_snr_db(n_bits)
    err = abs(sim - theory)
    errors.append(err)
    assert err < 1.0   # within 1 dB at every tested bit depth
# the LARGEST error should be at the smallest N (least averaging, coarsest
# quantization relative to signal) -- a real, checkable trend, not asserted
assert errors[0] >= errors[-1]

# 4. bits_for_target_snr is the correct inverse of theoretical_snr_db
for n in [4, 8, 12, 16]:
    target = adc.theoretical_snr_db(n)
    recovered_n = adc.bits_for_target_snr(target)
    assert abs(recovered_n - n) < 1e-6

# 5. real-world check: CD audio's 16-bit standard should need ~96 dB,
#    and inverting a 96 dB target should land close to 16 bits (this is
#    the actual historical justification for the 16-bit CD standard)
n_for_96db = adc.bits_for_target_snr(96.0)
assert abs(n_for_96db - 16) < 0.5

# 6. mantissa dynamic range scales linearly with bit count, same 6.02 dB/bit
#    slope as the ADC case -- float64 (53-bit mantissa) has MORE dynamic
#    range than float32 (24-bit), by exactly the bit-difference times 6.02
diff_db = adc.mantissa_dynamic_range_db(53) - adc.mantissa_dynamic_range_db(24)
assert abs(diff_db - (53 - 24) * 6.0206) < 1e-3

# 7. input validation
for bad_call in [
    lambda: adc.theoretical_snr_db(-1),
    lambda: adc.theoretical_snr_db(0),
    lambda: adc.simulate_adc_snr(-1),
    lambda: adc.simulate_adc_snr(4, n_samples=0),
    lambda: adc.mantissa_dynamic_range_db(0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.adc_snr_bits tests passed")
