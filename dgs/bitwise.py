"""Bitwise vs logical operators -- exclusive (XOR) vs inclusive (OR), and bit tricks.

The trap every C/Verilog/CUDA programmer hits: `&` is not `&&`.
  * BITWISE operators (& | ^ ~) act on every bit in PARALLEL -- 1011 & 0110 = 0010.
  * LOGICAL operators (&& || !) collapse a whole value to one true/false -- any
    nonzero is "true", so 2 && 1 is true (1), while 2 & 1 is 0.
And two kinds of OR:
  * INCLUSIVE OR (|): 1 if EITHER bit is set  (1|1 = 1).
  * EXCLUSIVE OR (^): 1 only if the bits DIFFER (1^1 = 0) -- "one or the other, not
    both". XOR is addition mod 2, the parity gate, and the heart of error detection.

These are the same gates as dgs.logic_timing, applied to integers bit-by-bit. Pure
Python. Education.
"""


# ── bitwise: act on every bit in parallel ───────────────────────────
def bit_and(a, b):
    """Bitwise AND (C/Verilog/CUDA `a & b`): 1 where BOTH bits are 1."""
    return a & b


def bit_or(a, b):
    """Bitwise INCLUSIVE OR (`a | b`): 1 where EITHER bit is 1."""
    return a | b


def bit_xor(a, b):
    """Bitwise EXCLUSIVE OR (`a ^ b`): 1 where the bits DIFFER. = addition mod 2."""
    return a ^ b


def bit_xnor(a, b, width=8):
    """Bitwise XNOR (Verilog `a ~^ b`): 1 where the bits are the SAME. = ~(a ^ b)."""
    return (~(a ^ b)) & ((1 << width) - 1)


def bit_not(a, width=8):
    """Bitwise NOT (`~a`), masked to `width` bits (so it stays unsigned)."""
    return (~a) & ((1 << width) - 1)


# ── logical: collapse the whole value to true/false ─────────────────
def logical_and(a, b):
    """Logical AND (`a && b`): true iff BOTH values are nonzero. NOT the same as &."""
    return int(bool(a) and bool(b))


def logical_or(a, b):
    """Logical OR (`a || b`): true iff EITHER value is nonzero. NOT the same as |."""
    return int(bool(a) or bool(b))


# ── bit-manipulation idioms (the reason bitwise exists) ─────────────
def set_bit(x, n):
    """Turn bit n ON: x | (1 << n)."""
    return x | (1 << n)


def clear_bit(x, n):
    """Turn bit n OFF: x & ~(1 << n)."""
    return x & ~(1 << n)


def toggle_bit(x, n):
    """Flip bit n: x ^ (1 << n)  (XOR with a one-hot mask)."""
    return x ^ (1 << n)


def test_bit(x, n):
    """Read bit n: (x >> n) & 1."""
    return (x >> n) & 1


def popcount(x):
    """Number of 1 bits (C `__builtin_popcount`, CUDA `__popc`, SystemVerilog
    `$countones`)."""
    return bin(x).count("1")


def parity(x):
    """Parity = XOR of ALL bits (Verilog reduction `^x`): 1 if an odd number of bits
    are set. The single-bit error-detection check, built from XOR."""
    return popcount(x) & 1


if __name__ == "__main__":
    a, b = 0b1011, 0b0110
    print(f"a={a:04b} b={b:04b}")
    print(f"  a & b  = {bit_and(a,b):04b}   (bitwise AND)")
    print(f"  a | b  = {bit_or(a,b):04b}   (inclusive OR)")
    print(f"  a ^ b  = {bit_xor(a,b):04b}   (exclusive OR -- bits that differ)")
    print(f"  ~a     = {bit_not(a,4):04b}   (bitwise NOT, 4-bit)")
    print(f"\nthe trap:  2 & 1 = {2 & 1} (bitwise),  but  2 && 1 = {logical_and(2,1)} (logical)")
    print(f"toggle bit 2 of {a:04b} -> {toggle_bit(a,2):04b};  popcount={popcount(a)};  parity={parity(a)}")
