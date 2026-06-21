"""Test sequential logic: D flip-flop, setup/hold, fmax, counter, FSM detector."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import logic_timing as lt

# 1. D flip-flop: Q follows D only on a tick, and HOLDS between ticks (memory)
ff = lt.DFlipFlop(q=0)
assert ff.tick(1) == 1                       # samples D=1
assert ff.q == 1                             # holds it (no tick) -- this is the memory
assert ff.tick(0) == 0
assert ff.tick(1) == 1 and ff.tick(1) == 1   # stays 1

# 2. setup/hold window: data must be stable around the clock edge (edge at t=10)
assert lt.setup_hold_status(2.0, 10.0, t_setup=2.0, t_hold=1.0) == "ok"       # long before
assert lt.setup_hold_status(9.5, 10.0, t_setup=2.0, t_hold=1.0) == "setup_violation"  # too late
assert lt.setup_hold_status(10.5, 10.0, t_setup=2.0, t_hold=1.0) == "hold_violation"  # too soon after
assert lt.setup_hold_status(12.0, 10.0, t_setup=2.0, t_hold=1.0) == "ok"      # well after

# 3. max clock frequency = 1/(t_clk_q + t_comb + t_setup); slower path -> slower clock
f_fast = lt.max_clock_frequency(t_comb=2.0, t_clk_q=1.0, t_setup=1.0)   # T=4 -> 0.25
assert abs(f_fast - 0.25) < 1e-12
f_slow = lt.max_clock_frequency(t_comb=10.0)
assert f_slow < f_fast                                                  # bigger t_comb -> lower fmax
# the ripple adder's critical path IS the t_comb that limits the clock
fmax_add = lt.max_clock_frequency(lt.ripple_carry_delay(16))
fmax_add4 = lt.max_clock_frequency(lt.ripple_carry_delay(4))
assert fmax_add < fmax_add4                                            # wider adder clocks slower

# 4. counter accumulates clock edges (the digital "integrator"), mod 2^n
assert lt.ripple_counter(3, 10) == [1, 2, 3, 4, 5, 6, 7, 0, 1, 2]      # wraps at 8

# 5. FSM '101' detector matches a sliding-window reference (overlapping)
import random
random.seed(0)
bits = [random.randint(0, 1) for _ in range(200)]
got = lt.sequence_detector_101(bits)
ref = [1 if i >= 2 and bits[i-2:i+1] == [1, 0, 1] else 0 for i in range(len(bits))]
assert got == ref
assert lt.sequence_detector_101([1, 0, 1, 0, 1]) == [0, 0, 1, 0, 1]    # overlap: two hits

print(f"TEST PASS  (DFF stores+holds; setup/hold window ok/setup/hold; "
      f"fmax=1/(tcq+tcomb+tsetup), wider adder clocks slower; counter integrates "
      f"clock mod 2^n; '101' FSM == sliding window, overlap-aware)")
