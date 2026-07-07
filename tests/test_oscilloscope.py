"""Test dgs.oscilloscope: the three scope 'bad ideas'. Undersampling aliases a
tone to a false low frequency (formula cross-checked against an FFT of the actual
samples); finite bandwidth inflates rise time in quadrature (with the '5x rule'
falling out); and a probe's capacitance loads the source into a low-pass."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import oscilloscope as scope

# 1. Nyquist and aliasing
assert scope.nyquist_rate(7e3) == 14e3
assert scope.is_aliased(7e3, 10e3) and not scope.is_aliased(3e3, 10e3)
# a 7 kHz tone at 10 kS/s shows up as 3 kHz; 12 kHz -> 2 kHz; below Nyquist unchanged
assert np.isclose(scope.alias_frequency(7e3, 10e3), 3e3)
assert np.isclose(scope.alias_frequency(12e3, 10e3), 2e3)
assert np.isclose(scope.alias_frequency(3e3, 10e3), 3e3)      # no aliasing
# INDEPENDENT check: FFT of the actual samples peaks at the aliased frequency
assert np.isclose(scope.observed_frequency_numeric(7e3, 10e3), 3e3, rtol=0.02)
assert np.isclose(scope.observed_frequency_numeric(12e3, 10e3), 2e3, rtol=0.02)
# and a properly sampled tone is NOT aliased (reads its true frequency)
assert np.isclose(scope.observed_frequency_numeric(3e3, 10e3), 3e3, rtol=0.02)

# 2. bandwidth <-> rise time (0.35/BW), and the quadrature inflation
assert np.isclose(scope.rise_time_from_bandwidth(1e9), 0.35e-9)
assert np.isclose(scope.bandwidth_from_rise_time(0.35e-9), 1e9)
assert np.isclose(scope.rise_time_from_bandwidth(scope.bandwidth_from_rise_time(2e-9)), 2e-9)
# 1 ns edge on a 350 MHz scope (t_scope = 1 ns) reads sqrt(2) ns -> ~41% error
assert np.isclose(scope.measured_rise_time(1e-9, 350e6), np.sqrt(2)*1e-9, rtol=1e-3)
assert np.isclose(scope.rise_time_error(1e-9, 350e6), np.sqrt(2)-1, rtol=1e-3)
# more bandwidth -> less error, and a very fast scope -> ~0
assert scope.rise_time_error(1e-9, 2e9) < scope.rise_time_error(1e-9, 1e9)
assert scope.rise_time_error(1e-9, 100e9) < 1e-3

# the '5x rule' derives itself: the BW for <=2% error is ~5x the signal's own BW
bw = scope.required_bandwidth_for_error(1e-9, 0.02)
signal_bw = scope.bandwidth_from_rise_time(1e-9)
assert np.isclose(bw / signal_bw, 5.0, rtol=0.02)
# and that BW really does give exactly 2% error (round-trip)
assert np.isclose(scope.rise_time_error(1e-9, bw), 0.02, rtol=1e-6)

# 3. probe loading: R_source + C_probe form a low-pass
assert np.isclose(scope.probe_cutoff(10e3, 100e-12), 1/(2*np.pi*10e3*100e-12))
assert np.isclose(scope.probe_cutoff(10e3, 100e-12), 159.15e3, rtol=1e-3)
# at the cutoff, amplitude is -3 dB (0.707); well above it, badly attenuated
fc = scope.probe_cutoff(10e3, 100e-12)
assert np.isclose(scope.probe_amplitude_ratio(fc, 10e3, 100e-12), 1/np.sqrt(2))
assert scope.probe_amplitude_ratio(10e6, 10e3, 100e-12) < 0.02   # 10 MHz -> ~1.6%
# a lower-capacitance (10x-style) probe pushes the cutoff up ~10x
assert np.isclose(scope.probe_cutoff(10e3, 10e-12) / scope.probe_cutoff(10e3, 100e-12), 10.0)

# 4. kwarg bounds
for bad in (lambda: scope.nyquist_rate(0),
            lambda: scope.alias_frequency(1e3, 0),
            lambda: scope.rise_time_from_bandwidth(0),
            lambda: scope.measured_rise_time(-1, 1e9),
            lambda: scope.probe_cutoff(0, 1e-12),
            lambda: scope.required_bandwidth_for_error(1e-9, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_oscilloscope: all checks passed")
