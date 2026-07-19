"""Test photonic (fiber time-of-flight + dispersion-induced) delay vs.
CMOS gate delay and oscilloscope resolvability, confirming the
counterintuitive real result: nanosecond photonic delays are EASIER
to resolve on an ordinary scope than a picosecond modern-CMOS gate
delay."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import photonic_vs_electronic_delay as pved

C = 2.998e8

# 1. fiber time-of-flight matches L*n_g/c directly, and scales linearly with length
delay_1m = pved.fiber_time_of_flight_s(1.0)
delay_2m = pved.fiber_time_of_flight_s(2.0)
assert abs(delay_1m - 1.0 * 1.4682 / C) < 1e-15
assert abs(delay_2m / delay_1m - 2.0) < 1e-9

# 2. dispersion-induced delay spread scales linearly with both D and delta_lambda
d1 = pved.dispersion_induced_delay_spread_s(800.0, 40.0)
d2 = pved.dispersion_induced_delay_spread_s(1600.0, 40.0)
d3 = pved.dispersion_induced_delay_spread_s(800.0, 80.0)
assert abs(d2 / d1 - 2.0) < 1e-9
assert abs(d3 / d1 - 2.0) < 1e-9
# sign of D shouldn't matter -- it's a delay SPREAD (magnitude)
d_neg = pved.dispersion_induced_delay_spread_s(-800.0, 40.0)
assert abs(d_neg - d1) < 1e-15

# 3. oscilloscope rise time: higher bandwidth -> shorter rise time, inverse relationship
rise_100mhz = pved.oscilloscope_rise_time_s(100e6)
rise_1ghz = pved.oscilloscope_rise_time_s(1e9)
assert abs(rise_100mhz / rise_1ghz - 10.0) < 1e-9

# 4. is_delay_observable: a big delay on a fast scope is observable, a tiny
#    delay on a slow scope is not
obs_easy, ratio_easy = pved.is_delay_observable(32e-9, 20e9)   # dispersion delay, 20GHz scope
assert obs_easy
assert ratio_easy > 3.0
obs_hard, ratio_hard = pved.is_delay_observable(5e-12, 1e9)   # 5nm gate delay, slow 1GHz scope
assert not obs_hard
assert ratio_hard < 3.0

# 5. required_scope_bandwidth_hz is the correct inverse of is_delay_observable:
#    at exactly the required bandwidth, the ratio should be right at the margin
delay = 15e-12
needed_bw = pved.required_scope_bandwidth_hz(delay, margin=3.0)
_, ratio_at_needed = pved.is_delay_observable(delay, needed_bw, margin=3.0)
assert abs(ratio_at_needed - 3.0) < 1e-6

# 6. the counterintuitive headline result: fiber/dispersion delays (ns-scale)
#    need a much LOWER scope bandwidth than modern CMOS gate delays (ps-scale)
fiber_bw_needed = pved.required_scope_bandwidth_hz(delay_1m)
gate_bw_needed = pved.required_scope_bandwidth_hz(pved.CMOS_GATE_DELAY_S["5nm_node_2020s"])
assert fiber_bw_needed < gate_bw_needed

# 7. CMOS_GATE_DELAY_S entries are physically ordered: newer processes are faster
assert pved.CMOS_GATE_DELAY_S["TTL_74LS_1970s"] > pved.CMOS_GATE_DELAY_S["CMOS_74HC_1980s"]
assert pved.CMOS_GATE_DELAY_S["CMOS_74HC_1980s"] > pved.CMOS_GATE_DELAY_S["180nm_node_1999"]
assert pved.CMOS_GATE_DELAY_S["180nm_node_1999"] > pved.CMOS_GATE_DELAY_S["14nm_finfet_2015"]
assert pved.CMOS_GATE_DELAY_S["14nm_finfet_2015"] > pved.CMOS_GATE_DELAY_S["5nm_node_2020s"]

# 8. input validation
for bad_call in [
    lambda: pved.fiber_time_of_flight_s(-1.0),
    lambda: pved.fiber_time_of_flight_s(1.0, n_group=-1.0),
    lambda: pved.dispersion_induced_delay_spread_s(800.0, -1.0),
    lambda: pved.dispersion_induced_delay_spread_s(0.0, 40.0),
    lambda: pved.oscilloscope_rise_time_s(-1.0),
    lambda: pved.is_delay_observable(-1.0, 1e9),
    lambda: pved.is_delay_observable(1e-9, -1.0),
    lambda: pved.is_delay_observable(1e-9, 1e9, margin=-1.0),
    lambda: pved.required_scope_bandwidth_hz(-1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.photonic_vs_electronic_delay tests passed")
