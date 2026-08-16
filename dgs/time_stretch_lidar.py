"""Photonic time-stretch applied to LIDAR: buying range resolution from a
slow electronic ADC by magnifying time itself before digitizing.

THE LIDAR PROBLEM. A pulse bounces off a target at range R and returns
after a round-trip delay dt = 2R/c. Two targets separated by dR return
pulses separated by d(dt) = 2*dR/c -- for dR in centimeters, that's tens
of picoseconds, far faster than most electronic ADCs (multi-GHz at best)
can resolve. A point-sampling digitizer whose sample interval is much
longer than the pulse width doesn't just blur close targets together --
it can miss ultrashort return pulses ENTIRELY (verified below: a
200 MSa/s ADC resolves ZERO of two 50 ps pulses 100 ps apart, not a
single blurred one). This is the classic LIDAR bandwidth-limited
resolution problem, delta_R = c/(2*B) for detection bandwidth B (see
range_resolution below) -- time-stretch fixes it by making the pulses
themselves slow enough for the same electronics to actually see.

THE TIME-STRETCH FIX (Jalali lab's actual research direction, general
description -- reuses dgs.jalali_grammar.time_stretch_factor's equation
[1] M = 1+D2*L2/(D1*L1) rather than re-deriving it). Photonic time-stretch
maps the WHOLE return waveform through a linear time magnification
t_stretched = M * t_true before it ever reaches the electronics: two
pulses originally d(dt) apart arrive M*d(dt) apart, easily resolved by
the SAME slow ADC that could not resolve them natively. This is exactly
dgs.coppinger_jalali_1999's B_RF = M*f_ADC/2 result (a slow ADC captures
M times its own native bandwidth), applied here to the LIDAR ranging
problem instead of the general-purpose ADC problem that module covers.

WHAT THIS MODULE ADDS (not covered elsewhere in the repo): the specific
LIDAR range<->delay bookkeeping, a synthetic multi-target return-pulse
generator, and a DIRECT numerical demonstration -- build two closely
spaced targets, show a native-rate ADC (via dgs.adc.ADC-style uniform
sampling) blurs them into one peak, then show the SAME sampling resolves
two distinct peaks once the waveform has been stretched by M. That
demonstration is the actual verification here, not just algebra.

Requires scipy.signal.find_peaks (scipy confirmed available py-3.13).
"""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks

from dgs.jalali_grammar import time_stretch_factor

C_LIGHT = 2.99792458e8   # m/s


def lidar_range_from_delay(dt_s, c: float = C_LIGHT):
    """R = c*dt/2 -- the basic LIDAR ranging equation (round-trip delay)."""
    dt_s = np.asarray(dt_s, dtype=float)
    if np.any(dt_s < 0):
        raise ValueError("delay must be non-negative")
    return c * dt_s / 2.0


def lidar_delay_from_range(R_m, c: float = C_LIGHT):
    """dt = 2*R/c, the inverse of lidar_range_from_delay."""
    R_m = np.asarray(R_m, dtype=float)
    if np.any(R_m < 0):
        raise ValueError("range must be non-negative")
    return 2.0 * R_m / c


def range_resolution(bandwidth_Hz: float, c: float = C_LIGHT) -> float:
    """delta_R = c/(2*B): the finest resolvable range separation for a
    system whose (electronic or effective) detection bandwidth is B."""
    if bandwidth_Hz <= 0:
        raise ValueError("bandwidth_Hz must be positive")
    return c / (2.0 * bandwidth_Hz)


def stretch_factor_from_fiber(D1_ps_nm: float, L1_km: float, D2_ps_nm: float, L2_km: float) -> float:
    """M via dgs.jalali_grammar.time_stretch_factor's own equation [1]
    (reused directly, not re-derived)."""
    return time_stretch_factor(D1_ps_nm, L1_km, D2_ps_nm, L2_km)["M"]


def multi_target_waveform(ranges_m, t_s, pulse_width_s: float, amplitudes=None,
                           stretch_M: float = 1.0, t_offset_s: float = 0.0, c: float = C_LIGHT):
    """Synthetic LIDAR return: a sum of Gaussian pulses, one per target,
    each centered at t_offset + M*(2*R_i/c) -- time-stretch is a LINEAR
    magnification of the whole waveform, so the pulse WIDTH is stretched
    by the same factor M as the delay, not just the spacing between
    pulses (a common oversimplification this module avoids)."""
    ranges_m = np.atleast_1d(np.asarray(ranges_m, dtype=float))
    if amplitudes is None:
        amplitudes = np.ones_like(ranges_m)
    amplitudes = np.atleast_1d(np.asarray(amplitudes, dtype=float))
    if len(amplitudes) != len(ranges_m):
        raise ValueError("amplitudes must match ranges_m in length")
    if pulse_width_s <= 0 or stretch_M <= 0:
        raise ValueError("pulse_width_s and stretch_M must be positive")

    t_s = np.asarray(t_s, dtype=float)
    waveform = np.zeros_like(t_s)
    stretched_width = pulse_width_s * stretch_M
    for R, A in zip(ranges_m, amplitudes):
        dt_true = lidar_delay_from_range(R, c)
        center = t_offset_s + stretch_M * dt_true
        waveform += A * np.exp(-(t_s - center) ** 2 / (2 * stretched_width ** 2))
    return waveform


def resample_at_adc_rate(waveform, t_s, f_adc_Hz: float):
    """Uniformly resample a waveform at a (slow) electronic ADC's native
    sample rate, by linear interpolation -- the same 'what does a real
    digitizer actually see' step as dgs.adc.ADC, kept self-contained here
    since only the resampled TIMES/VALUES are needed, not quantization."""
    if f_adc_Hz <= 0:
        raise ValueError("f_adc_Hz must be positive")
    t_s = np.asarray(t_s, dtype=float)
    t_sampled = np.arange(t_s[0], t_s[-1], 1.0 / f_adc_Hz)
    v_sampled = np.interp(t_sampled, t_s, waveform)
    return t_sampled, v_sampled


def resolve_targets(waveform, t_s, min_separation_s: float, prominence: float = 0.1):
    """Peak-find the (possibly ADC-resampled) waveform, with a minimum
    peak separation matched to the sample spacing -- returns the detected
    peak TIMES. `prominence` filters out sub-threshold noise/ringing."""
    t_s = np.asarray(t_s, dtype=float)
    dt = t_s[1] - t_s[0]
    distance_samples = max(1, int(round(min_separation_s / dt)))
    peak_idx, _ = find_peaks(waveform, distance=distance_samples, prominence=prominence)
    return t_s[peak_idx]


def recover_true_ranges(peak_times_s, stretch_M: float, t_offset_s: float = 0.0, c: float = C_LIGHT):
    """Invert the stretch: given peak arrival times in the STRETCHED
    waveform, recover the true target ranges R_i = c*((t_peak - offset)/M)/2."""
    if stretch_M <= 0:
        raise ValueError("stretch_M must be positive")
    dt_true = (np.asarray(peak_times_s, dtype=float) - t_offset_s) / stretch_M
    return lidar_range_from_delay(dt_true, c)


def _scenario_time_grid(ranges_m, pulse_width_s: float, stretch_M: float, oversample: int = 20, margin_pulses: float = 10.0):
    """Size a time grid appropriate to ONE scenario's own natural
    timescale: the native (M=1) and stretched (M large) waveforms live on
    physically different clocks (before vs. after the dispersive fiber),
    so each gets a grid whose spacing is a fixed fraction of ITS OWN pulse
    width -- keeping the fine-grid point count roughly constant regardless
    of M, rather than forcing both scenarios onto one shared axis."""
    dt_true_max = lidar_delay_from_range(np.max(np.atleast_1d(ranges_m)))
    stretched_width = pulse_width_s * stretch_M
    window_span = stretch_M * dt_true_max + margin_pulses * stretched_width
    dt_fine = stretched_width / oversample
    n_pts = int(np.clip(window_span / dt_fine, 1000, 2_000_000))
    return np.linspace(0.0, window_span, n_pts)


def resolution_demo(ranges_m, pulse_width_s: float, f_adc_Hz: float, stretch_M: float) -> dict:
    """The central demonstration: build the SAME multi-target scene both
    unstretched (M=1) and stretched (M), each on its own physically
    appropriate time grid (see _scenario_time_grid), resample BOTH at the
    same slow ADC rate f_adc_Hz, and count how many distinct targets each
    recovers -- showing time-stretch buys resolvable targets from
    identical electronics, not just asserting it."""
    t_native = _scenario_time_grid(ranges_m, pulse_width_s, 1.0)
    t_stretched = _scenario_time_grid(ranges_m, pulse_width_s, stretch_M)

    native = multi_target_waveform(ranges_m, t_native, pulse_width_s, stretch_M=1.0)
    stretched = multi_target_waveform(ranges_m, t_stretched, pulse_width_s, stretch_M=stretch_M)

    t_native_adc, v_native_adc = resample_at_adc_rate(native, t_native, f_adc_Hz)
    t_stretch_adc, v_stretch_adc = resample_at_adc_rate(stretched, t_stretched, f_adc_Hz)

    peaks_native = resolve_targets(v_native_adc, t_native_adc, min_separation_s=1.0 / f_adc_Hz)
    peaks_stretch = resolve_targets(v_stretch_adc, t_stretch_adc, min_separation_s=1.0 / f_adc_Hz)
    recovered_ranges = recover_true_ranges(peaks_stretch, stretch_M)

    return {
        "n_targets_true": len(np.atleast_1d(ranges_m)),
        "n_targets_resolved_native_ADC": len(peaks_native),
        "n_targets_resolved_after_stretch": len(peaks_stretch),
        "recovered_ranges_m": recovered_ranges,
        "true_ranges_m": np.sort(np.atleast_1d(ranges_m)),
    }


if __name__ == "__main__":
    print("=== LIDAR ranging basics ===")
    R_test = 15.0
    dt_test = lidar_delay_from_range(R_test)
    print(f"  target at R={R_test} m -> round-trip delay = {dt_test*1e9:.3f} ns "
          f"-> back to R = {lidar_range_from_delay(dt_test):.3f} m")

    print("\n=== range resolution vs. detection bandwidth ===")
    for B_GHz in (1.0, 5.0, 20.0):
        print(f"  B={B_GHz:5.1f} GHz -> delta_R = {range_resolution(B_GHz*1e9)*100:.2f} cm")

    print("\n=== time-stretch factor from fiber parameters (reused from dgs.jalali_grammar) ===")
    M = stretch_factor_from_fiber(D1_ps_nm=-17.0, L1_km=0.1, D2_ps_nm=-17.0, L2_km=10.0)
    print(f"  D1=-17 ps/nm/km, L1=0.1 km, D2=-17 ps/nm/km, L2=10 km -> M = {M}")

    print("\n=== resolving two closely spaced targets: native ADC vs. after time-stretch ===")
    ranges = [10.00, 10.03]   # 3 cm apart -- far below a slow ADC's native resolution
    f_adc = 200e6              # 200 MSa/s "slow" electronic ADC
    pulse_width = 50e-12       # 50 ps optical pulse
    result = resolution_demo(ranges, pulse_width, f_adc, stretch_M=M)
    print(f"  true targets: {result['true_ranges_m']} m")
    print(f"  resolved with the RAW (unstretched) waveform on a {f_adc/1e6:.0f} MSa/s ADC: "
          f"{result['n_targets_resolved_native_ADC']} target(s)")
    print(f"  resolved AFTER time-stretch (M={M}) on the SAME ADC: "
          f"{result['n_targets_resolved_after_stretch']} target(s)")
    print(f"  recovered ranges: {np.round(result['recovered_ranges_m'], 3)} m")
