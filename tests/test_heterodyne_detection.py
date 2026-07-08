"""Test dgs.heterodyne_detection: the intermediate/beat frequency, the square-law
detector output (DC + beat at f_IF), frequency recovery to sub-bin accuracy, the
coherent gain that lifts a weak signal via a strong LO, and Doppler/LIDAR velocity
from the beat."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import heterodyne_detection as het

# 1. intermediate (beat) frequency = |f_sig - f_lo|, symmetric
assert het.intermediate_frequency(1_000_000, 1_050_000) == 50_000
assert het.intermediate_frequency(1_050_000, 1_000_000) == 50_000

# 2. the detector output: DC = A_s^2 + A_lo^2, beat amplitude 2 A_s A_lo at f_IF
fs = 2_000_000.0
t, det = het.photodetector_beat(1e6, 1.05e6, fs, duration=1e-3, A_sig=1.0, A_lo=1.0)
assert np.isclose(det.mean(), 1.0 + 1.0, atol=1e-2)               # DC term
assert np.isclose((det.max() - det.min()) / 2, 2 * 1.0 * 1.0, atol=1e-2)  # 2 A_s A_lo
# Nyquist guard: a beat above fs/2 is rejected
try:
    het.photodetector_beat(1e6, 3e6, fs, 1e-3); assert False      # f_IF = 2 MHz > 1 MHz
except ValueError:
    pass

# 3. frequency measurement: recover f_IF, bin-aligned (exact) and off-bin (parabolic)
_, d1 = het.photodetector_beat(1e6, 1.05e6, fs, duration=1e-3)   # f_IF=50 kHz, bin=1 kHz
assert np.isclose(het.estimate_frequency(d1, fs), 50_000, rtol=1e-6)
_, d2 = het.photodetector_beat(1e6, 1e6 + 50_333, fs, duration=0.02)  # off-bin
assert np.isclose(het.estimate_frequency(d2, fs), 50_333, rtol=2e-3)  # sub-bin accuracy

# 4. coherent gain: a weak signal beaten against a bright LO
g = het.coherent_gain(A_sig=1e-3, A_lo=1.0)
assert np.isclose(g["beat_amplitude"], 2 * 1e-3 * 1.0)
assert np.isclose(g["direct_intensity"], (1e-3) ** 2)
assert np.isclose(g["gain_ratio"], 2 * 1.0 / 1e-3)               # = 2000
# heterodyne beats direct detection precisely when the LO dwarfs the signal
assert g["beat_amplitude"] > g["direct_intensity"]
assert het.coherent_gain(1.0, 1.0)["gain_ratio"] == 2.0          # equal amps -> gain 2

# 5. Doppler / LIDAR: velocity <-> beat, round trip
f0 = het.C_LIGHT / 1550e-9
v = 30.0
fb = het.doppler_beat(v, f0)                                     # round-trip default
assert np.isclose(het.velocity_from_beat(fb, f0), v)            # inverse
assert np.isclose(het.doppler_beat(v, f0, round_trip=True),
                  2 * het.doppler_beat(v, f0, round_trip=False))  # round trip = 2x one-way
assert np.isclose(fb, 2 * f0 * v / het.C_LIGHT)

# 6. kwarg bounds
for bad in (lambda: het.intermediate_frequency(-1, 1e6),
            lambda: het.photodetector_beat(1e6, 1.05e6, 0, 1e-3),
            lambda: het.estimate_frequency([1, 2, 3], fs),       # too few samples
            lambda: het.coherent_gain(0, 1),
            lambda: het.doppler_beat(10, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_heterodyne_detection: all checks passed")
