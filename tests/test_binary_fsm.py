"""Test dgs.binary_fsm: the divisibility-by-N DFA (state = value mod N, accepts
binary multiples of N, matches Python's %) and the serial-adder FSM (state =
carry, reproduces integer +). Integers/booleans/01 as finite state machines."""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import binary_fsm as bf

# 1. DFA structure: N states, delta=(2s+b)%N, start 0, accept {0}
m = bf.divisibility_fsm(3)
assert m["n_states"] == 3 and m["start"] == 0 and m["accept"] == {0}
assert m["delta"][(0, 1)] == 1 and m["delta"][(1, 1)] == 0 and m["delta"][(2, 0)] == 1
assert all(m["delta"][(s, b)] == (2*s + b) % 3 for s in range(3) for b in (0, 1))

# 2. is_divisible_by matches int(s,2) % n over many random strings and several n
rng = random.Random(0)
for n in (2, 3, 5, 7, 13):
    for _ in range(200):
        s = "".join(rng.choice("01") for _ in range(rng.randint(1, 20)))
        assert bf.is_divisible_by(s, n) == (int(s, 2) % n == 0), (s, n)

# 3. the remainder trace equals value-mod-N of every prefix (the DFA invariant)
s, n = "1011010", 7
tr = bf.remainder_trace(s, n)
assert len(tr) == len(s)
for k in range(1, len(s) + 1):
    assert tr[k - 1] == int(s[:k], 2) % n
assert tr[-1] == int(s, 2) % n

# 4. run_fsm generic + accepting behavior on known values
_, acc = bf.run_fsm(bf.divisibility_fsm(3), [1, 1, 0])          # 6
assert acc
_, acc = bf.run_fsm(bf.divisibility_fsm(3), [1, 1, 1])          # 7
assert not acc
# empty input -> value 0 -> divisible by anything
assert bf.is_divisible_by("", 5)
# n = 1: every number is a multiple of 1
assert all(bf.is_divisible_by(bin(k)[2:], 1) for k in range(1, 20))

# 5. serial-adder full-adder truth table
truth = {(0,0,0): (0,0), (0,0,1): (1,0), (0,1,0): (1,0), (0,1,1): (0,1),
         (1,0,0): (1,0), (1,0,1): (0,1), (1,1,0): (0,1), (1,1,1): (1,1)}
for (c, a, b), (s_bit, c_out) in truth.items():
    assert bf.serial_adder_step(c, a, b) == (s_bit, c_out)

# 6. add_via_fsm == a + b over many pairs (including long carry chains)
for _ in range(500):
    a, b = rng.randint(0, 1 << 16), rng.randint(0, 1 << 16)
    assert bf.add_via_fsm(a, b) == a + b
assert bf.add_via_fsm(0, 0) == 0
assert bf.add_via_fsm(255, 1) == 256           # carry ripples through 8 bits
assert bf.add_via_fsm(2**20 - 1, 1) == 2**20   # long carry chain

# 7. kwarg bounds
for bad in (lambda: bf.divisibility_fsm(0),
            lambda: bf.is_divisible_by("102", 3),          # non-binary
            lambda: bf.serial_adder_step(2, 0, 1),         # bad carry
            lambda: bf.add_via_fsm(-1, 3)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_binary_fsm: all checks passed")
