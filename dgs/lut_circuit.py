"""An FPGA-style lookup table (LUT): any n-input Boolean function, realized
as a memory read instead of a chain of gates.

A LUT stores a function's ENTIRE truth table (2^n bits) and evaluates it at
runtime with nothing but the address-decode circuit already built in
dgs.memory_address_decoder: the n inputs feed a one-hot minterm decoder
(decoder_outputs), each decoder line is ANDed with its stored bit, and all
of those are ORed together --

    output = OR_j ( decoder_line[j] AND stored_bit[j] )

No gate ever "computes" the function at runtime; the function was baked
into the stored bits ahead of time (dgs.boolean_algebra's TruthTable
enumeration does exactly that baking), and evaluating it afterward is pure
memory access -- the same address-decode structure a DRAM/SRAM cell uses,
just static and combinational instead of dynamic and refreshed.

The tradeoff this buys: LUT cost is a FIXED 2^n bits no matter how complex
or simple the function is, while a minimized gate-level implementation
(dgs.boolean_algebra.minimize_sop) can be much cheaper for "nice", highly
compressible functions -- and, for genuinely incompressible functions
(XOR/parity has NO smaller SOP: every minterm is its own prime implicant,
since no two adjacent minterms ever agree on more than zero variables), can
end up MORE expensive in raw literal count than the LUT's fixed bit budget.
Both regimes are demonstrated here with real functions already in this
repo: dgs.computer_engineering.full_adder's Cout (=majority, compresses
well) and its Sum (a 3-input XOR, does not compress at all).
"""

import re

from dgs.boolean_algebra import TruthTable, minimize_sop
from dgs import memory_address_decoder as mad


def synthesize_lut(n_vars, fn):
    """Bake a Boolean function's truth table into LUT content: a length-2^n
    tuple of stored bits, in minterm-index order (address j corresponds to
    tt.rows[j]), using dgs.boolean_algebra.TruthTable's own enumeration."""
    tt = TruthTable(n_vars, fn)
    return tuple(out for _bits, out in tt.rows), tt


def lut_read(lut_bits, input_bits):
    """Evaluate the LUT circuit for a given input: address-decode
    input_bits with the SAME one-hot minterm decoder used for physical
    memory addressing (dgs.memory_address_decoder.decoder_outputs), AND
    each decoder line with its stored bit, then OR the results -- the
    literal mux/OR-tree structure of a real LUT read."""
    if len(lut_bits) != 2 ** len(input_bits):
        raise ValueError("lut_bits length must be 2**len(input_bits)")
    decoder_lines = mad.decoder_outputs(input_bits)
    return int(any(line & bit for line, bit in zip(decoder_lines, lut_bits)))


def verify_lut(n_vars, fn):
    """Exhaustive check, every possible input: the LUT circuit (decode +
    AND + OR) reproduces fn, and agrees with a plain array index into
    lut_bits -- confirming the gate-level read and the arithmetic address
    are the same function."""
    lut_bits, tt = synthesize_lut(n_vars, fn)
    for bits, expected in tt.rows:
        via_circuit = lut_read(lut_bits, bits)
        j = int("".join(map(str, bits)), 2)
        via_index = lut_bits[j]
        if via_circuit != expected or via_index != expected:
            return False
    return True


def _count_sop_literals(sop_string):
    """Count Boolean literals in a minimize_sop() string like
    'F = BC + AC + AB' -- each uppercase letter (with or without a
    preceding ~) is one literal; the constant terms 'F = 1' / 'F = 0' have
    zero."""
    body = sop_string.split("F = ", 1)[1]
    if body.strip() in ("0", "1"):
        return 0
    return sum(len(re.findall(r"[A-Z]", term)) for term in body.split(" + "))


def lut_vs_minimized_gates_ratio(n_vars, fn):
    """The actual digital-design cost tradeoff: LUT storage is a FIXED
    2**n_vars bits regardless of the function. Compare that to the literal
    count of the function's minimized SOP (dgs.boolean_algebra.minimize_sop):

      ratio > 1  ->  the LUT uses MORE raw bits than minimized gates need
                      (the function compresses well, e.g. majority)
      ratio < 1  ->  the LUT is actually CHEAPER than minimized gates
                      (the function doesn't compress, e.g. XOR/parity --
                      every minterm is its own prime implicant)
    """
    tt = TruthTable(n_vars, fn)
    sop = minimize_sop(tt.minterms, n_vars)
    literals = _count_sop_literals(sop)
    lut_bits = 2 ** n_vars
    return {
        "sop": sop,
        "literal_count": literals,
        "lut_bits": lut_bits,
        "ratio_lut_bits_per_literal": (lut_bits / literals) if literals else float("inf"),
    }


if __name__ == "__main__":
    from dgs.computer_engineering import full_adder

    print("=== LUT circuit reproduces functions exactly (exhaustive) ===\n")
    majority3 = lambda A, B, C: (A & B) | (B & C) | (A & C)
    xor3 = lambda A, B, C: A ^ B ^ C
    for name, fn in [("majority-3 (full_adder Cout)", majority3), ("XOR-3 (full_adder Sum)", xor3)]:
        ok = verify_lut(3, fn)
        print(f"  {name}: LUT circuit matches function for all 8 inputs -> {ok}")

    print("\n=== full_adder's own Cout/Sum really are majority/XOR (cross-check) ===\n")
    for A in (0, 1):
        for B in (0, 1):
            for Cin in (0, 1):
                fa = full_adder(A, B, Cin)
                assert fa["Cout"] == majority3(A, B, Cin)
                assert fa["S"] == xor3(A, B, Cin)
    print("  full_adder(A,B,Cin)['Cout'] == majority3(A,B,Cin) for all 8 inputs: True")
    print("  full_adder(A,B,Cin)['S']    == xor3(A,B,Cin)      for all 8 inputs: True")

    print("\n=== The digital-circuit ratio: LUT bits per minimized literal ===\n")
    for name, fn, n in [
        ("majority-3 (compresses well)", majority3, 3),
        ("XOR-3 (does not compress)", xor3, 3),
        ("XOR-4 (does not compress)", lambda A, B, C, D: A ^ B ^ C ^ D, 4),
    ]:
        stats = lut_vs_minimized_gates_ratio(n, fn)
        print(f"  {name}:")
        print(f"    minimized SOP: {stats['sop']}")
        print(f"    literals={stats['literal_count']}, LUT bits={stats['lut_bits']}, "
              f"ratio={stats['ratio_lut_bits_per_literal']:.3f}")
