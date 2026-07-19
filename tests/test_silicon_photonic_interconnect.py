"""Test the silicon photonics vs. copper interconnect link budget: real
2026 NVIDIA roadmap bandwidths, reusing dgs.transmission_line_tdr's
already-verified skin-effect physics, confirming copper's max reach
shrinks as data rate climbs while optical loss stays essentially flat."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import silicon_photonic_interconnect as spi

# 1. Nyquist bandwidth: PAM4 (2 bits/symbol) needs HALF the bandwidth of
#    NRZ (1 bit/symbol) for the same bit rate
bw_nrz = spi.required_nyquist_bandwidth_hz(1e12, bits_per_symbol=1)
bw_pam4 = spi.required_nyquist_bandwidth_hz(1e12, bits_per_symbol=2)
assert abs(bw_nrz / bw_pam4 - 2.0) < 1e-9

# 2. copper's max reach SHRINKS as data rate increases (skin effect gets
#    worse at higher frequency) -- verified across the real roadmap tiers
reach_1_6, _ = spi.copper_trace_max_reach_m(1.6e12)
reach_6_4, _ = spi.copper_trace_max_reach_m(6.4e12)
reach_12_8, _ = spi.copper_trace_max_reach_m(12.8e12)
assert reach_1_6 > reach_6_4 > reach_12_8

# 3. optical fiber loss barely changes with distance at these short,
#    in-rack scales, and is always tiny compared to copper's 20 dB budget
loss_30cm = spi.optical_fiber_loss_db(0.3)
loss_1m = spi.optical_fiber_loss_db(1.0)
assert loss_30cm < 0.001
assert loss_1m < 0.001
assert loss_1m > loss_30cm   # still monotonic, just a tiny absolute effect

# 4. why_cpo_at_this_bandwidth correctly identifies copper as insufficient
#    at all three real NVIDIA roadmap tiers, for typical in-rack reach
for tbps in spi.NVIDIA_ROADMAP_TBPS.values():
    result = spi.why_cpo_at_this_bandwidth(tbps, typical_reach_m=0.3)
    assert not result["copper_reach_sufficient"]   # numpy bool, not Python bool -- use truthiness not `is`
    assert result["copper_max_reach_m"] < result["typical_needed_reach_m"]
    assert result["optical_loss_db_at_typical_reach"] < 0.01

# 5. at a much shorter required reach (e.g. within-package, a few mm),
#    copper CAN be sufficient even at high data rates -- confirms the
#    function isn't just always returning False regardless of input
result_short = spi.why_cpo_at_this_bandwidth(1.6, typical_reach_m=0.05)
assert result_short["copper_reach_sufficient"]

# 6. the three roadmap entries are the real, correct NVIDIA figures
assert spi.NVIDIA_ROADMAP_TBPS["gen1_OSFP_offpackage"] == 1.6
assert spi.NVIDIA_ROADMAP_TBPS["gen2_CPO_motherboard"] == 6.4
assert spi.NVIDIA_ROADMAP_TBPS["gen3_CPO_inpackage"] == 12.8

# 7. input validation
for bad_call in [
    lambda: spi.required_nyquist_bandwidth_hz(-1.0),
    lambda: spi.required_nyquist_bandwidth_hz(1e12, bits_per_symbol=0),
    lambda: spi.copper_trace_max_reach_m(-1.0),
    lambda: spi.optical_fiber_loss_db(-1.0),
    lambda: spi.why_cpo_at_this_bandwidth(-1.0),
    lambda: spi.why_cpo_at_this_bandwidth(1.6, typical_reach_m=-1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.silicon_photonic_interconnect tests passed")
