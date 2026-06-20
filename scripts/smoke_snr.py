"""Smoke-test the SNR budget module: shot, quantization, thermal, empirical."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import snr

# 1. basic dB: equal power -> 0 dB; 10x power -> 10 dB
assert abs(snr.snr_db(1.0, 1.0)) < 1e-12
assert abs(snr.snr_db(10.0, 1.0) - 10.0) < 1e-12

# 2. shot noise sqrt(N): doubling photons adds 10*log10(2) ~ 3.01 dB
assert abs(snr.shot_noise_snr_db(200) - snr.shot_noise_snr_db(100) - 3.0103) < 1e-3
assert abs(snr.shot_noise_snr_db(10000) - 40.0) < 1e-9          # 10 log10(1e4)
# matches an actual Poisson draw (amplitude SNR = mean/std ~ sqrt(N))
rng = np.random.default_rng(0)
N = 5000
counts = rng.poisson(N, size=200000).astype(float)
emp_amp_snr = counts.mean() / counts.std()
assert abs(20 * np.log10(emp_amp_snr) - snr.shot_noise_snr_db(N)) < 0.5

# 3. quantization: 6.02 dB per bit, +1.76; ENOB inverts it
assert abs(snr.quantization_snr_db(8) - 49.92) < 0.01
assert abs(snr.quantization_snr_db(16) - snr.quantization_snr_db(8) - 8 * 6.02) < 1e-9
assert abs(snr.effective_bits(snr.quantization_snr_db(12)) - 12.0) < 1e-9

# 4. Johnson noise: sqrt scaling in T, R, B; ~4 uV for 1k/300K/1MHz... check formula
v = snr.johnson_noise_voltage(1e3, 300.0, 1e6)
assert abs(v - np.sqrt(4 * snr.KB * 300 * 1e3 * 1e6)) < 1e-18
assert abs(snr.johnson_noise_voltage(4e3, 300, 1e6) / v - 2.0) < 1e-9   # R x4 -> V x2

# 5. empirical SNR from clean vs noisy
clean = np.sin(np.linspace(0, 8 * np.pi, 4000))
noisy = clean + 0.1 * rng.standard_normal(clean.size)
got = snr.snr_from_signal(clean, noisy)
# signal power 0.5, noise power 0.01 -> ~17 dB
assert 15 < got < 19, got
assert snr.snr_from_signal(clean, clean) == np.inf

# 6. validation
for bad in (lambda: snr.snr_db(1, 0), lambda: snr.shot_noise_snr_db(0),
            lambda: snr.quantization_snr_db(0), lambda: snr.johnson_noise_voltage(-1, 300, 1e6)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad input")

print(f"SMOKE PASS  (shot: 10000 ph -> {snr.shot_noise_snr_db(10000):.0f} dB; "
      f"8-bit ADC -> {snr.quantization_snr_db(8):.1f} dB; empirical {got:.1f} dB)")
