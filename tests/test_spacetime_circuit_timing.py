"""Test dgs/spacetime_circuit_timing.py: real EM propagation physics
(Grace Hopper's nanosecond wire, FR4 propagation delay) extending
dgs/logic_timing.py's ripple-carry-adder critical path with wire delay."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.spacetime_circuit_timing import (
    propagation_velocity, propagation_delay_per_length,
    verify_nanosecond_wire_length, electrically_long_threshold_m,
    ripple_carry_total_delay_with_wire, wire_delay_dominance_sweep, C_LIGHT,
)
from dgs.logic_timing import ripple_carry_delay

# 1. propagation_velocity: eps_r=1 (vacuum) must give exactly c
assert abs(propagation_velocity(1.0) - C_LIGHT) < 1e-3

# 2. propagation_velocity: any real dielectric (eps_r>1) must give v < c
v_fr4 = propagation_velocity(4.3)
assert v_fr4 < C_LIGHT
assert abs(v_fr4 - C_LIGHT / np.sqrt(4.3)) < 1e-3

# 3. propagation_velocity bounds: eps_r < 1 is unphysical
try:
    propagation_velocity(0.5)
    raise AssertionError("expected ValueError for eps_r_eff<1")
except ValueError:
    pass

# 4. Grace Hopper's nanosecond wire: must match the famous 11.8-inch value
ns_check = verify_nanosecond_wire_length()
assert ns_check["matches"] is True
assert abs(ns_check["computed_inches"] - 11.8) < 0.01

# 5. FR4 propagation delay must match the well-known ~170-180 ps/inch range
t_pd_fr4_per_inch_ps = propagation_delay_per_length(eps_r_eff=4.3) * 0.0254 * 1e12
assert 150.0 < t_pd_fr4_per_inch_ps < 190.0, (
    f"expected FR4 delay in the well-known ~150-190 ps/in range, got {t_pd_fr4_per_inch_ps:.1f}")

# 6. electrically_long_threshold_m: longer rise time -> longer allowed
#    trace before it's "electrically long" (slower edges are more forgiving)
L_fast = electrically_long_threshold_m(rise_time_s=50e-12, eps_r_eff=4.3)
L_slow = electrically_long_threshold_m(rise_time_s=500e-12, eps_r_eff=4.3)
assert L_slow > L_fast

# 7. electrically_long_threshold_m bounds
for bad_kwargs in [dict(rise_time_s=0.0, eps_r_eff=4.3),
                    dict(rise_time_s=1e-12, eps_r_eff=4.3, fraction=0.0),
                    dict(rise_time_s=1e-12, eps_r_eff=4.3, fraction=1.5)]:
    try:
        electrically_long_threshold_m(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

# 8. ripple_carry_total_delay_with_wire: at trace_length=0, total delay
#    must exactly equal dgs.logic_timing.ripple_carry_delay's gate-only
#    answer (reused directly, not reimplemented) -- the wire-delay term
#    must vanish cleanly, not just approximately
result_zero_wire = ripple_carry_total_delay_with_wire(
    n_bits=16, gate_delay_ps=20.0, trace_length_per_stage_m=0.0)
expected_gate_only = ripple_carry_delay(16, gate_delay=20.0)
assert abs(result_zero_wire["total_delay_ps"] - expected_gate_only) < 1e-9
assert result_zero_wire["wire_delay_fraction"] == 0.0

# 9. ripple_carry_total_delay_with_wire: nonzero trace length must
#    strictly increase total delay and strictly decrease fmax
result_with_wire = ripple_carry_total_delay_with_wire(
    n_bits=16, gate_delay_ps=20.0, trace_length_per_stage_m=0.002)
assert result_with_wire["total_delay_ps"] > expected_gate_only
assert result_with_wire["fmax_with_wire_GHz"] < result_with_wire["fmax_gate_only_GHz"]
assert 0.0 < result_with_wire["wire_delay_fraction"] < 1.0

# 10. ripple_carry_total_delay_with_wire bounds
for bad_kwargs in [dict(n_bits=0, gate_delay_ps=20.0, trace_length_per_stage_m=0.001),
                    dict(n_bits=8, gate_delay_ps=0.0, trace_length_per_stage_m=0.001),
                    dict(n_bits=8, gate_delay_ps=20.0, trace_length_per_stage_m=-0.001)]:
    try:
        ripple_carry_total_delay_with_wire(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

# 11. wire_delay_dominance_sweep: wire-delay fraction should generally
#     increase with n_bits (more carry stages accumulate more trace
#     length), holding gate_delay_ps and trace_length_per_stage_m fixed
sweep = wire_delay_dominance_sweep(gate_delay_ps=20.0, trace_length_per_stage_m=0.002,
                                    n_bits_range=np.array([4, 32, 128]))
assert sweep["wire_delay_fraction"][0] < sweep["wire_delay_fraction"][1] < sweep["wire_delay_fraction"][2], (
    "wire-delay fraction should increase with adder width, holding per-stage trace length fixed")

print("all dgs.spacetime_circuit_timing tests passed")
