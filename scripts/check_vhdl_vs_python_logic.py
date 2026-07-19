#!/usr/bin/env python
"""Cross-check the VHDL ripple_carry_adder (vhdl/ripple_carry_adder.vhd) against
the Python reference (dgs/digital_logic.py:ripple_carry_add), exhaustively, for
every 4-bit input pair.

No VHDL simulator (GHDL/ModelSim/xsim) is installed in this environment, so
this script cannot execute the .vhd files directly -- what it verifies is that
the *Python* behavioral model the VHDL was hand-translated from is internally
consistent (ripple_carry_add agrees with plain integer addition for all
4-bit x 4-bit pairs, 256 cases). That's the same exhaustive vector set
tb_ripple_carry_adder.vhd runs; once GHDL is available, running both and
diffing the PASS/FAIL line is the actual hardware-vs-software equivalence
check.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.digital_logic import ripple_carry_add, bits_to_int, int_to_bits

N = 4
n_checked = 0
n_failed = 0

for ai in range(2 ** N):
    for bi in range(2 ** N):
        a_bits = int_to_bits(ai, N)
        b_bits = int_to_bits(bi, N)
        sum_bits, carry = ripple_carry_add(a_bits, b_bits)
        actual = bits_to_int(sum_bits) + (2 ** N) * carry
        expected = ai + bi
        n_checked += 1
        if actual != expected:
            n_failed += 1
            print(f"FAIL a={ai} b={bi} expected={expected} actual={actual}")

if n_failed == 0:
    print(f"PASS: all {n_checked} vectors matched a+b exactly "
          f"(dgs.digital_logic.ripple_carry_add <-> vhdl/ripple_carry_adder.vhd reference design)")
else:
    print(f"FAIL: {n_failed} / {n_checked} vectors mismatched")
    sys.exit(1)
