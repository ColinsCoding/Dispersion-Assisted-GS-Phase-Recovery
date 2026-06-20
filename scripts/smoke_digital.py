"""Smoke-test digital_logic against arithmetic truth."""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from dgs import digital_logic as dl

# 1. full adder truth table matches a+b+cin
print("full adder truth table (a b cin -> sum cout):")
ok = True
for a, b, cin in itertools.product((0, 1), repeat=3):
    s, c = dl.full_adder(a, b, cin)
    ref = a + b + cin
    ok &= (s == ref & 1) and (c == ref >> 1)
    print(f"  {a} {b} {cin} -> {s} {c}")
print("full adder correct:", ok)

# 2. ripple-carry adder == integer addition for all 4-bit pairs
n = 4
allok = True
for x, y in itertools.product(range(2**n), repeat=2):
    bits, cout = dl.ripple_carry_add(dl.int_to_bits(x, n), dl.int_to_bits(y, n))
    result = dl.bits_to_int(bits) + (cout << n)
    allok &= (result == x + y)
print(f"\nripple-carry 4-bit == integer add for all {2**n}x{2**n} pairs:", allok)

# 3. carry-lookahead gives the SAME result as ripple-carry
claok = True
for x, y in itertools.product(range(2**n), repeat=2):
    rb, rc = dl.ripple_carry_add(dl.int_to_bits(x, n), dl.int_to_bits(y, n))
    cb, cc = dl.carry_lookahead(dl.int_to_bits(x, n), dl.int_to_bits(y, n))
    claok &= (rb == cb and rc == cc)
print("carry-lookahead == ripple-carry:", claok)

# 4. Boolean form
s_expr, c_expr = dl.adder_boolean()
print("\nsymbolic sum  =", s_expr)
print("symbolic cout =", c_expr, "(majority function)")

# 5. Gray code: consecutive codes differ by exactly one bit; round-trip
print("\nGray code 0..7:", [format(dl.to_gray(i), '03b') for i in range(8)])
single_bit = all(bin(dl.to_gray(i) ^ dl.to_gray(i + 1)).count("1") == 1 for i in range(15))
roundtrip = all(dl.from_gray(dl.to_gray(i)) == i for i in range(256))
print("consecutive differ by 1 bit:", single_bit, "| round-trip exact:", roundtrip)

# 6. CMOS cost
print("\nCMOS transistor counts:", {g: dl.cmos_cost(g) for g in ("NOT", "NAND2", "XOR2")})

# 7. the 7 basic gates match their Boolean truth
gateok = (dl.AND(1,1)==1 and dl.OR(0,0)==0 and dl.NOT(0)==1 and
          dl.NAND(1,1)==0 and dl.NOR(0,0)==1 and dl.XOR(1,0)==1 and dl.XNOR(1,1)==1)
assert len(dl.GATES) == 7, "there are exactly 7 basic gates"
# NAND is universal: NOT, AND, OR rebuilt from NAND alone
nand = dl.NAND
not_n  = lambda a: nand(a, a)
and_n  = lambda a, b: not_n(nand(a, b))
or_n   = lambda a, b: nand(not_n(a), not_n(b))
universal = all(not_n(a)==dl.NOT(a) for a in (0,1)) and \
            all(and_n(a,b)==dl.AND(a,b) and or_n(a,b)==dl.OR(a,b)
                for a in (0,1) for b in (0,1))
print("7 basic gates correct:", gateok, "| NAND is universal:", universal)
assert gateok and universal

# 8. decoder/encoder are inverses; mux selects; demux routes
decok = all(dl.decoder(dl.int_to_bits(i, 3)).index(1) == i for i in range(8))
# priority encoder recovers the index of a one-hot input
encok = all(dl.bits_to_int(dl.priority_encoder(dl.decoder(dl.int_to_bits(i,3)))[0]) == i
            for i in range(8))
data = [10, 20, 30, 40]
muxok = all(dl.mux(data, dl.int_to_bits(i, 2)) == data[i] for i in range(4))
print("decoder one-hot:", decok, "| priority-encoder inverts decoder:", encok, "| mux:", muxok)
assert decok and encok and muxok

# 9. ALU instruction set vs Python arithmetic/bitwise (4-bit, two's complement)
import itertools as _it
aluok = True
for x, y in _it.product(range(16), repeat=2):
    xb, yb = dl.int_to_bits(x, 4), dl.int_to_bits(y, 4)
    add_r, _ = dl.alu("ADD", xb, yb)
    sub_r, fl = dl.alu("SUB", xb, yb)
    and_r, _ = dl.alu("AND", xb, yb)
    aluok &= dl.bits_to_int(add_r) == (x + y) & 0xF
    aluok &= dl.bits_to_int(sub_r) == (x - y) & 0xF        # two's-complement wrap
    aluok &= dl.bits_to_int(and_r) == (x & y)
    aluok &= fl["carry"] == int(x >= y)                    # SUB carry = no-borrow
# zero flag fires when a-a==0
_, zf = dl.alu("SUB", dl.int_to_bits(9, 4), dl.int_to_bits(9, 4))
print("ALU ADD/SUB/AND over all 4-bit pairs:", aluok, "| zero flag on a-a:", zf["zero"])
assert aluok and zf["zero"] == 1

# validation
for bad in [lambda: dl.ripple_carry_add([2], [0]),
            lambda: dl.cmos_cost("FOO"),
            lambda: dl.to_gray(-1)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", str(e)[:50])
print("SMOKE PASS" if (ok and allok and claok and single_bit and roundtrip
                        and gateok and universal and decok and encok and muxok and aluok)
      else "SMOKE FAIL")
