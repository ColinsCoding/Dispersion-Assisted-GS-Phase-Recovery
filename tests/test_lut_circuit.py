"""Test the FPGA-style LUT circuit: exhaustive correctness against known
functions, agreement between the gate-level (decode+AND+OR) read and a
plain array index, cross-check against dgs.computer_engineering.full_adder,
and the LUT-bits-vs-minimized-literals ratio for compressible vs
incompressible functions."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import lut_circuit as lc
from dgs.computer_engineering import full_adder

majority3 = lambda A, B, C: (A & B) | (B & C) | (A & C)
xor3 = lambda A, B, C: A ^ B ^ C

# 1. LUT circuit reproduces both functions exactly, for every input
assert lc.verify_lut(3, majority3)
assert lc.verify_lut(3, xor3)
assert lc.verify_lut(4, lambda A, B, C, D: A ^ B ^ C ^ D)

# 2. synthesize_lut content matches a hand-checked truth table for
#    majority-3: minterms 3,5,6,7 (011,101,110,111) -> LUT bits 1 there,
#    0 elsewhere
lut_bits, tt = lc.synthesize_lut(3, majority3)
assert lut_bits == (0, 0, 0, 1, 0, 1, 1, 1)
assert tt.minterms == [3, 5, 6, 7]

# 3. lut_read rejects a mismatched LUT size (n inputs must match log2(len))
try:
    lc.lut_read((0, 1, 0, 1), (0, 0, 1))  # 4 entries but 3 input bits
except ValueError:
    pass
else:
    raise AssertionError("should reject mismatched LUT size")

# 4. cross-check against the REAL full_adder in dgs.computer_engineering:
#    Cout is exactly majority3, Sum is exactly xor3, for all 8 input combos
#    -- and the LUT built from majority3/xor3 reproduces full_adder itself
for A in (0, 1):
    for B in (0, 1):
        for Cin in (0, 1):
            fa = full_adder(A, B, Cin)
            assert fa["Cout"] == majority3(A, B, Cin)
            assert fa["S"] == xor3(A, B, Cin)
            assert lc.lut_read(lut_bits, (A, B, Cin)) == fa["Cout"]

# 5. the digital-circuit ratio: majority-3 compresses (ratio > 1), XOR-3
#    does not compress at all (ratio < 1) -- concrete, checkable numbers,
#    not just a qualitative claim
maj_stats = lc.lut_vs_minimized_gates_ratio(3, majority3)
xor_stats = lc.lut_vs_minimized_gates_ratio(3, xor3)
assert maj_stats["literal_count"] == 6
assert maj_stats["lut_bits"] == 8
assert abs(maj_stats["ratio_lut_bits_per_literal"] - 8 / 6) < 1e-9
assert maj_stats["ratio_lut_bits_per_literal"] > 1.0

assert xor_stats["literal_count"] == 12   # every minterm is its own prime implicant: 4 terms x 3 literals
assert xor_stats["lut_bits"] == 8
assert xor_stats["ratio_lut_bits_per_literal"] < 1.0

# 6. XOR's non-compressibility gets WORSE (ratio drops further below 1)
#    as n grows -- literal count scales as n*2^(n-1), LUT bits as 2^n, so
#    the ratio is 2/n -- shrinking, not just "also less than 1"
xor4_stats = lc.lut_vs_minimized_gates_ratio(4, lambda A, B, C, D: A ^ B ^ C ^ D)
assert xor4_stats["literal_count"] == 32
assert xor4_stats["ratio_lut_bits_per_literal"] < xor_stats["ratio_lut_bits_per_literal"]

print("all dgs.lut_circuit tests passed")
