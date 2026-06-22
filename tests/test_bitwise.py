"""Test bitwise vs logical operators, XOR vs inclusive OR, and bit idioms."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import bitwise as bw
from dgs import logic_timing as lt

# 1. the classic trap: bitwise & is NOT logical &&
assert bw.bit_and(2, 1) == 0          # 10 & 01 = 00
assert bw.logical_and(2, 1) == 1      # both nonzero -> true
assert bw.bit_or(2, 1) == 3           # 10 | 01 = 11
assert bw.logical_or(2, 1) == 1       # either nonzero -> true (just 1)

# 2. exclusive vs inclusive OR differ ONLY when both bits are 1
for a in (0, 1):
    for b in (0, 1):
        assert bw.bit_xor(a, b) == (a != b)        # XOR = 1 iff differ
        assert bw.bit_or(a, b) == (a or b)         # OR  = 1 iff either
assert bw.bit_xor(1, 1) == 0 and bw.bit_or(1, 1) == 1   # the one place they differ

# 3. these match the gates in dgs.logic_timing (bit-by-bit)
for a in (0, 1):
    for b in (0, 1):
        assert bw.bit_xor(a, b) == lt.GATE_FUNCS["XOR"](a, b)
        assert bw.bit_or(a, b) == lt.GATE_FUNCS["OR"](a, b)
        assert bw.bit_and(a, b) == lt.GATE_FUNCS["AND"](a, b)
        assert bw.bit_xnor(a, b, width=1) == lt.GATE_FUNCS["XNOR"](a, b)

# 4. XNOR = NOT XOR; XOR is its own inverse (a ^ b ^ b == a)
assert bw.bit_xnor(0b1010, 0b1010, width=4) == 0b1111      # equal -> all ones
assert (0b1100 ^ 0b0110) ^ 0b0110 == 0b1100                # XOR undoes itself

# 5. bit idioms: set / clear / toggle / test round-trip
x = 0b0000
x = bw.set_bit(x, 2); assert x == 0b0100 and bw.test_bit(x, 2) == 1
x = bw.set_bit(x, 0); assert x == 0b0101
x = bw.clear_bit(x, 2); assert x == 0b0001
x = bw.toggle_bit(x, 3); assert x == 0b1001 and bw.test_bit(x, 1) == 0
# toggle twice returns the original (XOR mask twice = identity)
assert bw.toggle_bit(bw.toggle_bit(0b1010, 1), 1) == 0b1010

# 6. popcount and parity (parity = XOR of all bits)
assert bw.popcount(0b1011) == 3 and bw.parity(0b1011) == 1     # odd # of ones
assert bw.popcount(0b1001) == 2 and bw.parity(0b1001) == 0     # even
# parity equals the running XOR of every bit
val = 0b110101
xacc = 0
for i in range(6):
    xacc ^= bw.test_bit(val, i)
assert xacc == bw.parity(val)

print("TEST PASS  (& != &&; XOR vs OR differ only at (1,1); match logic_timing gates; "
      "XNOR=~XOR; set/clear/toggle/test; parity = XOR of all bits)")
