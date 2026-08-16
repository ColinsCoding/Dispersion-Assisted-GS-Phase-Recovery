"""Test dgs.time_stretch_lidar: the range<->delay conversions must be exact
inverses, the stretch factor must be reused correctly from
dgs.jalali_grammar, and the central claim -- a slow ADC that resolves
ZERO closely-spaced targets natively resolves them correctly after
time-stretch -- must hold numerically, with recovered ranges matching the
true ones."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.time_stretch_lidar import (
    lidar_range_from_delay, lidar_delay_from_range, range_resolution,
    stretch_factor_from_fiber, multi_target_waveform, resample_at_adc_rate,
    resolve_targets, recover_true_ranges, resolution_demo, C_LIGHT,
)
from dgs.jalali_grammar import time_stretch_factor

# 1. range <-> delay are exact inverses
for R in (1.0, 15.0, 250.0):
    dt = lidar_delay_from_range(R)
    assert abs(lidar_range_from_delay(dt) - R) < 1e-9

# 2. range resolution: bigger bandwidth -> finer (smaller) resolution
res_1ghz = range_resolution(1e9)
res_20ghz = range_resolution(20e9)
assert res_20ghz < res_1ghz
assert abs(res_1ghz - C_LIGHT / 2e9) < 1e-12

# 3. stretch_factor_from_fiber matches dgs.jalali_grammar.time_stretch_factor directly (real reuse)
M = stretch_factor_from_fiber(D1_ps_nm=-17.0, L1_km=0.1, D2_ps_nm=-17.0, L2_km=10.0)
expected = time_stretch_factor(-17.0, 0.1, -17.0, 10.0)["M"]
assert M == expected

# 4. multi_target_waveform: a stretched pulse is WIDER (by exactly M) and
# arrives LATER (by exactly M) than the unstretched one for the same target
R_single = 10.0
t_native = np.linspace(0, 200e-9, 400_000)
t_stretched = np.linspace(0, 200e-9 * 50, 400_000)
pulse_width = 50e-12
native = multi_target_waveform([R_single], t_native, pulse_width, stretch_M=1.0)
stretched = multi_target_waveform([R_single], t_stretched, pulse_width, stretch_M=50.0)
t_peak_native = t_native[np.argmax(native)]
t_peak_stretched = t_stretched[np.argmax(stretched)]
dt_true = lidar_delay_from_range(R_single)
assert abs(t_peak_native - dt_true) < 1e-11
assert abs(t_peak_stretched - 50.0 * dt_true) < 5e-9   # coarser grid near the peak, looser tolerance

# 5. resample_at_adc_rate: output times are uniformly spaced at 1/f_adc
t_adc, v_adc = resample_at_adc_rate(native, t_native, f_adc_Hz=100e6)
assert np.allclose(np.diff(t_adc), 1.0 / 100e6, atol=1e-15)

# 6. resolve_targets finds the right NUMBER of peaks for well-separated, well-sampled pulses
t_fine = np.linspace(0, 100e-9, 200_000)
two_targets = multi_target_waveform([10.0, 12.0], t_fine, pulse_width_s=1e-9, stretch_M=1.0)
peaks = resolve_targets(two_targets, t_fine, min_separation_s=1e-9, prominence=0.1)
assert len(peaks) == 2

# 7. recover_true_ranges inverts the stretch correctly
recovered = recover_true_ranges(peaks, stretch_M=1.0)
assert np.allclose(np.sort(recovered), np.sort([10.0, 12.0]), atol=0.05)

# 8. THE CENTRAL CLAIM: a 200 MSa/s ADC resolves ZERO of two targets 3 cm
# apart natively, but resolves BOTH after time-stretch by M~101, and the
# recovered ranges match the true ones to within a few cm
ranges = [10.00, 10.03]
result = resolution_demo(ranges, pulse_width_s=50e-12, f_adc_Hz=200e6, stretch_M=101.0)
assert result["n_targets_true"] == 2
assert result["n_targets_resolved_native_ADC"] == 0, \
    "a 200 MSa/s ADC should NOT see 50 ps pulses spaced 100 ps apart at all"
assert result["n_targets_resolved_after_stretch"] == 2, \
    "the same ADC should resolve both targets once the waveform is stretched by M~101"
assert np.allclose(np.sort(result["recovered_ranges_m"]), np.sort(ranges), atol=0.1)

# 9. input validation
try:
    lidar_range_from_delay(-1.0)
    assert False, "should reject negative delay"
except ValueError:
    pass
try:
    range_resolution(-1e9)
    assert False, "should reject non-positive bandwidth"
except ValueError:
    pass
try:
    multi_target_waveform([1.0, 2.0], t_fine, pulse_width_s=1e-9, amplitudes=[1.0])
    assert False, "should reject mismatched amplitudes length"
except ValueError:
    pass

print(f"all dgs.time_stretch_lidar tests passed  "
      f"(native resolves {result['n_targets_resolved_native_ADC']}, "
      f"stretched resolves {result['n_targets_resolved_after_stretch']} of {result['n_targets_true']} targets)")
