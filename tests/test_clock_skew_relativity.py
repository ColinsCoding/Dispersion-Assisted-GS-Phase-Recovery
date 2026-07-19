"""Test clock skew (a classical, finite-signal-speed effect in computer
engineering) against special relativity's relativity of simultaneity,
keeping the scope honest: they share a conceptual root but are NOT the
same formula, verified by confirming the relativistic gap actually
vanishes as v/c -> 0 rather than reducing to the classical skew value."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import clock_skew_relativity as csr
from dgs.transmission_line_tdr import propagation_velocity

L_per_len, C_per_len = 250e-9, 100e-12
v = propagation_velocity(L_per_len, C_per_len)

# 1. equal trace lengths give exactly zero skew
assert csr.clock_skew_between_receivers(0.05, 0.05, v) == 0.0

# 2. longer trace arrives LATER -- skew sign is physically meaningful
skew = csr.clock_skew_between_receivers(0.050, 0.053, v)
assert skew < 0   # trace A (shorter) arrives before trace B (longer): t_a - t_b < 0

# 3. skew scales linearly with the length mismatch (classical, non-relativistic
#    physics: doubling the extra distance doubles the delay)
skew_1mm = csr.clock_skew_between_receivers(0.050, 0.051, v)
skew_2mm = csr.clock_skew_between_receivers(0.050, 0.052, v)
assert abs(skew_2mm / skew_1mm - 2.0) < 1e-9

# 4. faster propagation velocity reduces skew for the same length mismatch
v_faster = v * 1.5
skew_faster = csr.clock_skew_between_receivers(0.050, 0.053, v_faster)
assert abs(skew_faster) < abs(skew)

# 5. the relativistic simultaneity gap genuinely VANISHES as v/c -> 0,
#    confirming it does NOT reduce to the classical skew formula (they
#    answer different physical questions: switching frames vs. finite
#    signal speed within one frame)
gaps = csr.verify_classical_limit(0.050, 0.053)
assert gaps[0] > gaps[1] > gaps[2]   # monotonically shrinking
assert gaps[2] < 1e-15                # essentially zero at v/c=1e-9

# 6. relativistic gap is exactly zero at v_frame=0 (perfectly stationary
#    frames: simultaneity is trivially preserved)
gap_zero = csr.relativistic_simultaneity_gap(0.050, 0.053, 0.0)
assert gap_zero == 0.0

# 7. relativistic gap for EQUAL distances is always zero, at any v_frame
#    (co-located events stay simultaneous regardless of frame -- a real
#    relativity fact, not something that should accidentally hold)
for v_frac in [0.1, 0.5, 0.9]:
    gap_equal = csr.relativistic_simultaneity_gap(0.05, 0.05, v_frac * csr.C_SI)
    assert abs(gap_equal) < 1e-20

# 8. input validation
for bad_call in [
    lambda: csr.clock_arrival_time(0.0, 0.05, -1.0),
    lambda: csr.clock_arrival_time(0.0, -0.05, v),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.clock_skew_relativity tests passed")
