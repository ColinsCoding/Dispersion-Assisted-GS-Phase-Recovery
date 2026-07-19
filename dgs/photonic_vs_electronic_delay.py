"""When does a photonic delay actually become visible on an oscilloscope,
versus a modern CMOS logic gate's delay -- and which one is actually
EASIER to resolve on real bench equipment? Counterintuitive answer:
photonic delays (fiber time-of-flight, and especially chromatic-
dispersion group-delay spread) are nanoseconds-to-tens-of-nanoseconds
and trivially visible on an ordinary scope. Modern CMOS gate delays are
picoseconds and need a genuinely high-end (tens-of-GHz) real-time scope
to resolve -- "instant-looking" logic is actually the HARDER of the two
to catch in the act.

Reuses this repo's own dispersion convention: D in ps/nm (already
length-aggregated, matching dgs.gs_core's H(f)=exp(j*pi*D*f^2) and the
D=-695/-800 ps/nm dispersion-compensating-fiber figure cited there from
the real Coppinger/Jalali 1999 time-stretch ADC paper). A dispersive
fiber's group-delay-vs-wavelength slope IS the mechanism the dispersive
Fourier transform / STEAM relies on -- this module just makes that delay
a number you can compare against an oscilloscope's rise time.
"""

import numpy as np

C = 2.998e8  # m/s, speed of light in vacuum

# real, representative propagation delays per logic gate across CMOS
# technology generations (order-of-magnitude figures from public datasheets
# and published process-node characterization -- not a specific chip's
# exact number, but the right ballpark for each era)
CMOS_GATE_DELAY_S = {
    "TTL_74LS_1970s": 10e-9,
    "CMOS_74HC_1980s": 8e-9,
    "180nm_node_1999": 60e-12,
    "14nm_finfet_2015": 15e-12,
    "5nm_node_2020s": 5e-12,
}


def fiber_time_of_flight_s(length_m, n_group=1.4682):
    """Straight propagation delay through a length of fiber:
    tau = L * n_group / c. n_group=1.4682 is standard SMF-28 fused
    silica fiber's group index near 1550 nm."""
    if length_m <= 0:
        raise ValueError("length_m must be positive")
    if n_group <= 0:
        raise ValueError("n_group must be positive")
    return length_m * n_group / C


def dispersion_induced_delay_spread_s(D_ps_per_nm, delta_lambda_nm):
    """The group-delay spread between two wavelength components separated
    by delta_lambda_nm, given a total (already length-aggregated) fiber
    dispersion D in ps/nm -- exactly this repo's H(f)=exp(j*pi*D*f^2)
    convention (see dgs.gs_core). This IS the physical mechanism that
    turns a wavelength spectrum into a time-domain waveform (dispersive
    Fourier transform / STEAM): different frequencies arrive at
    different times because D != 0."""
    if delta_lambda_nm <= 0:
        raise ValueError("delta_lambda_nm must be positive")
    if abs(D_ps_per_nm) < 1e-9:
        raise ValueError("D_ps_per_nm must be nonzero (zero dispersion means no delay spread)")
    delay_ps = abs(D_ps_per_nm) * delta_lambda_nm
    return delay_ps * 1e-12


def oscilloscope_rise_time_s(bandwidth_hz):
    """Standard scope rule of thumb: rise_time ~= 0.35 / bandwidth. A
    delay needs to be well above this to be cleanly resolved as two
    distinguishable edges rather than blurred together."""
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    return 0.35 / bandwidth_hz


def is_delay_observable(delay_s, scope_bandwidth_hz, margin=3.0):
    """A delay is cleanly observable if it's at least `margin` times the
    scope's own rise time (a common rule of thumb for confidently
    resolving two edges rather than eyeballing a blur). Returns
    (observable: bool, ratio: delay / rise_time)."""
    if delay_s <= 0:
        raise ValueError("delay_s must be positive")
    if scope_bandwidth_hz <= 0:
        raise ValueError("scope_bandwidth_hz must be positive")
    if margin <= 0:
        raise ValueError("margin must be positive")
    rise_time = oscilloscope_rise_time_s(scope_bandwidth_hz)
    ratio = delay_s / rise_time
    return ratio >= margin, ratio


def required_scope_bandwidth_hz(delay_s, margin=3.0):
    """Inverse question: what scope bandwidth do you actually need to
    cleanly resolve a given delay?"""
    if delay_s <= 0:
        raise ValueError("delay_s must be positive")
    if margin <= 0:
        raise ValueError("margin must be positive")
    required_rise_time = delay_s / margin
    return 0.35 / required_rise_time


if __name__ == "__main__":
    print("=== Photonic delay vs. 'instant-looking' CMOS logic delay ===\n")

    fiber_delay = fiber_time_of_flight_s(1.0)
    print(f"1 m of fiber, straight time-of-flight delay: {fiber_delay*1e9:.3f} ns")

    # this repo's own dispersion regime: D=-800 ps/nm (real DCF figure
    # cited in dgs.gs_core from the Coppinger/Jalali 1999 paper), a
    # realistic ~40 nm optical bandwidth pulse
    D_ps_per_nm = -800.0
    delta_lambda_nm = 40.0
    disp_delay = dispersion_induced_delay_spread_s(D_ps_per_nm, delta_lambda_nm)
    print(f"dispersive Fourier transform delay spread (D={D_ps_per_nm} ps/nm, "
          f"{delta_lambda_nm} nm bandwidth): {disp_delay*1e9:.1f} ns")
    print("  (this IS the mechanism dgs.gs_core's H(f)=exp(j*pi*D*f^2) encodes --")
    print("   the whole dispersive-Fourier-transform / STEAM trick.)\n")

    print("=== Bench scope resolvability ===")
    for scope_name, bw_hz in [("100 MHz bench scope", 100e6),
                                ("1 GHz scope", 1e9),
                                ("20 GHz real-time scope", 20e9),
                                ("70 GHz real-time scope", 70e9)]:
        rise = oscilloscope_rise_time_s(bw_hz)
        obs_fiber, ratio_fiber = is_delay_observable(fiber_delay, bw_hz)
        obs_disp, ratio_disp = is_delay_observable(disp_delay, bw_hz)
        print(f"\n{scope_name} (rise time {rise*1e9:.3f} ns):")
        print(f"  1m fiber delay observable: {bool(obs_fiber)} (ratio {ratio_fiber:.1f}x)")
        print(f"  dispersion delay spread observable: {bool(obs_disp)} (ratio {ratio_disp:.1f}x)")

    print("\n=== Now the counterintuitive part: CMOS gate delay ===")
    for gen, gate_delay in CMOS_GATE_DELAY_S.items():
        needed_bw = required_scope_bandwidth_hz(gate_delay)
        print(f"{gen}: {gate_delay*1e12:.1f} ps/gate -> needs a "
              f"{needed_bw/1e9:.1f} GHz scope to cleanly resolve")

    print("\n=== The point ===")
    print("Photonic propagation and dispersion delays (nanoseconds) are EASIER to")
    print("see on an ordinary scope than a single modern CMOS gate's delay")
    print("(picoseconds). Logic only 'looks' instant because we don't usually put")
    print("a 20+ GHz real-time scope on a single gate -- the delay is real, it's")
    print("just smaller than what most bench equipment can resolve. Dispersive")
    print("fiber, by contrast, deliberately STRETCHES delay into the nanosecond")
    print("range precisely so an ordinary scope/ADC can capture a spectrum as a")
    print("time-domain waveform -- which is the entire point of this repo's")
    print("dispersion-assisted GS pipeline.")
