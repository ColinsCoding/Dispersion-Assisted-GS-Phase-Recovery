"""Smoke-test sequential logic in digital_logic: flip-flop, register, shift, counter, FSM."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import digital_logic as dl

# 1. D flip-flop remembers D
assert dl.d_flip_flop(1, 0) == 1 and dl.d_flip_flop(0, 1) == 0

# 2. register: load latches data, no-load holds
assert dl.register_tick([1, 0, 1, 1], load=1, current_bits=[0, 0, 0, 0]) == [1, 0, 1, 1]
assert dl.register_tick([1, 1, 1, 1], load=0, current_bits=[1, 0, 1, 1]) == [1, 0, 1, 1]
try:
    dl.register_tick([1, 0], load=1, current_bits=[0, 0, 0, 0])
except ValueError:
    pass
else:
    raise AssertionError("width mismatch should raise")

# 3. shift register clocks a serial bit through (LSB-first, shifting right)
state = [0, 0, 0, 0]
for bit, expect_out in [(1, 0), (0, 0), (1, 0), (1, 0)]:
    state, serial_out = dl.shift_register_tick(state, bit)
    assert serial_out == expect_out
assert state == [1, 1, 0, 1]                       # bits 1,0,1,1 in, newest at the front

# 4. synchronous counter counts up and wraps mod 2^n (built on the adder)
c = [0, 0, 0]                                       # 3-bit, value 0, LSB-first
seen = []
for _ in range(10):
    seen.append(dl.bits_to_int(c))
    c = dl.counter_tick(c)
assert seen == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]      # wraps at 8
# enable=0 holds
assert dl.counter_tick([1, 0, 1], enable=0) == [1, 0, 1]

# 5. FSM: overlapping "101" sequence detector (Mealy)
T = {("S0", "0"): "S0", ("S0", "1"): "S1",
     ("S1", "0"): "S2", ("S1", "1"): "S1",
     ("S2", "0"): "S0", ("S2", "1"): "S1"}
O = {("S0", "0"): 0, ("S0", "1"): 0,
     ("S1", "0"): 0, ("S1", "1"): 0,
     ("S2", "0"): 0, ("S2", "1"): 1}               # emit 1 exactly when 101 completes
final, out, trace = dl.fsm_run(T, O, "S0", "1101011101", mealy=True)
assert out == [0, 0, 0, 1, 0, 1, 0, 0, 0, 1]       # detect at positions 3, 5, 9 (overlapping)
assert sum(out) == 3
assert len(trace) == len("1101011101") + 1
# unknown transition raises
try:
    dl.fsm_run(T, O, "S0", "1X", mealy=True)
except ValueError:
    pass
else:
    raise AssertionError("undefined transition should raise")

# 6. Moore FSM (output depends on state only): a mod-3 counter
Tm = {("s0", "t"): "s1", ("s1", "t"): "s2", ("s2", "t"): "s0"}
Om = {"s0": 0, "s1": 1, "s2": 2}
_, outm, _ = dl.fsm_run(Tm, Om, "s0", "t" * 7)
assert outm == [1, 2, 0, 1, 2, 0, 1]

print(f"SMOKE PASS  (101-detector fired at {[i for i,v in enumerate(out) if v]}; counter {seen})")
