"""Polynomials -> Boolean algebra -> hexadecimal: arithmetic over GF(2).

GF(2) = {0, 1} with  a + b = a XOR b  and  a * b = a AND b.

1. POLYNOMIAL <-> HEX: a GF(2) polynomial's coefficients ARE bits.
       x^8 + x^2 + x + 1  ->  1 0000 0111  ->  0x107
   Addition = XOR of the integers, multiplication = carry-less multiply,
   division = long division with XOR instead of subtraction. Checked against
   SymPy's Poly(..., modulus=2).

2. BOOLEAN ALGEBRA = POLYNOMIALS: every Boolean function has a unique
   polynomial over GF(2) (algebraic normal form / Zhegalkin polynomial):
       NOT a = 1 + a,   a AND b = ab,   a OR b = a + b + ab,   a XOR b = a + b
   Found by the binary Moebius transform of the truth table; checked against
   sympy.logic.boolalg.to_anf.

3. CRC = POLYNOMIAL REMAINDER: CRC(msg) = msg(x) * x^r  mod  g(x), with the
   generator g written in hex (CRC-8 0x07, CRC-16/CCITT 0x1021, CRC-32 0x04C11DB7).
   Checked against zlib.crc32 and binascii.crc_hqx on the standard "123456789".

4. LFSR: a shift register with XOR taps generates sequences whose period is
   2^n - 1 exactly when the feedback polynomial is primitive (e.g. PRBS7, x^7+x^6+1).
"""
import itertools

import sympy as sp

X = sp.Symbol("x")


# ── 1. polynomial <-> integer <-> hex ────────────────────────────────

def poly_to_int(poly):
    """SymPy expression in x with 0/1 coefficients (mod 2) -> integer bit pattern."""
    P = sp.Poly(poly, X, modulus=2)
    v = 0
    for (deg,), c in P.terms():
        if int(c) % 2:
            v |= 1 << deg
    return v


def int_to_poly(v):
    if v < 0:
        raise ValueError("bit pattern must be non-negative")
    return sum((X ** k for k in range(v.bit_length()) if (v >> k) & 1), sp.Integer(0))


def to_hex(poly):
    return hex(poly_to_int(poly))


def gf2_add(a, b):
    return a ^ b


def gf2_mul(a, b):
    """Carry-less multiplication: shift-and-XOR instead of shift-and-add."""
    if a < 0 or b < 0:
        raise ValueError("operands must be non-negative")
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def gf2_divmod(a, b):
    """Long division over GF(2): subtract (= XOR) shifted divisors."""
    if b == 0:
        raise ZeroDivisionError("division by the zero polynomial")
    q, db = 0, b.bit_length()
    while a and a.bit_length() >= db:
        s = a.bit_length() - db
        q |= 1 << s
        a ^= b << s
    return q, a


def sympy_divmod(a, b):
    """Same division via SymPy Poly over GF(2), returned as integers."""
    qa, ra = sp.div(sp.Poly(int_to_poly(a), X, modulus=2), sp.Poly(int_to_poly(b), X, modulus=2))
    return poly_to_int(qa.as_expr()), poly_to_int(ra.as_expr())


# ── 2. Boolean functions as GF(2) polynomials ────────────────────────

def anf_coefficients(truth_table):
    """Binary Moebius transform: truth table (length 2^n, index bits = inputs)
    -> ANF coefficients c[S] with f = XOR over S of c[S] * prod_{i in S} x_i."""
    c = [int(v) & 1 for v in truth_table]
    n = len(c).bit_length() - 1
    if len(c) != 1 << n:
        raise ValueError("truth table length must be a power of two")
    for i in range(n):
        step = 1 << i
        for m in range(len(c)):
            if m & step:
                c[m] ^= c[m ^ step]
    return c


def anf_expression(truth_table, names):
    """ANF as a readable SymPy sum (+ meaning XOR, product meaning AND)."""
    syms = sp.symbols(names, seq=True)          # tuple even for a single name
    terms = []
    for m, cm in enumerate(anf_coefficients(truth_table)):
        if cm:
            term = sp.Integer(1)
            for i, s in enumerate(syms):
                if (m >> i) & 1:
                    term *= s
            terms.append(term)
    return sp.Add(*terms, evaluate=False) if terms else sp.Integer(0)


def truth_table(func, n):
    """Evaluate a Python Boolean function on all 2^n inputs (bit i = input i)."""
    return [int(bool(func(*[(m >> i) & 1 for i in range(n)]))) for m in range(1 << n)]


def eval_anf(coeffs, inputs):
    """Evaluate the ANF polynomial (XOR of AND-monomials) on a tuple of bits."""
    m_in = sum(b << i for i, b in enumerate(inputs))
    return sum(c for m, c in enumerate(coeffs) if c and (m & m_in) == m) & 1


# ── 3. CRC = polynomial remainder ────────────────────────────────────

CRC_SPECS = {
    # name: (width, poly hex (MSB-first, implicit top bit), init, reflect, xorout, check("123456789"))
    "CRC-8/SMBUS":       (8, 0x07, 0x00, False, 0x00, 0xF4),
    "CRC-16/CCITT-FALSE": (16, 0x1021, 0xFFFF, False, 0x0000, 0x29B1),
    "CRC-32":            (32, 0x04C11DB7, 0xFFFFFFFF, True, 0xFFFFFFFF, 0xCBF43926),
}


def _reflect(v, width):
    return int(f"{v:0{width}b}"[::-1], 2)


def crc_bitwise(data: bytes, name: str):
    """Shift-register CRC, MSB-first core; reflected variants reflect in/out."""
    width, poly, init, refl, xorout, _ = CRC_SPECS[name]
    top, mask = 1 << (width - 1), (1 << width) - 1
    reg = init
    for byte in data:
        b = _reflect(byte, 8) if refl else byte
        reg ^= b << (width - 8)
        for _ in range(8):
            reg = ((reg << 1) ^ poly) & mask if reg & top else (reg << 1) & mask
    if refl:
        reg = _reflect(reg, width)
    return reg ^ xorout


def crc_polynomial_remainder(data: bytes, width: int, poly: int):
    """Pure algebra (init 0, no reflection): (M(x) * x^width) mod g(x), g = x^width + poly."""
    m = int.from_bytes(data, "big")
    g = (1 << width) | poly
    return gf2_divmod(m << width, g)[1]


def crc_reference(data: bytes, name: str):
    """Independent library implementations where Python has one."""
    import binascii
    import zlib
    if name == "CRC-32":
        return zlib.crc32(data) & 0xFFFFFFFF
    if name == "CRC-16/CCITT-FALSE":
        return binascii.crc_hqx(data, 0xFFFF)
    return None


def detects_all_single_bit_errors(width, poly, msg_bits=64):
    """g(x) with >= 2 terms never divides x^k, so every 1-bit flip changes the CRC."""
    g = (1 << width) | poly
    return all(gf2_divmod(1 << (k + width), g)[1] != 0 for k in range(msg_bits))


# ── 4. LFSR ──────────────────────────────────────────────────────────

def lfsr_period(taps_poly: int, n: int, seed: int = 1):
    """Fibonacci LFSR for feedback polynomial `taps_poly` (bit k = x^k, degree n).
    Returns the period of the state sequence."""
    if not 0 < seed < (1 << n):
        raise ValueError("seed must be a non-zero n-bit state")
    taps = [k for k in range(1, n + 1) if (taps_poly >> k) & 1]
    state, start = seed, seed
    for period in itertools.count(1):
        fb = 0
        for t in taps:
            fb ^= (state >> (n - t)) & 1
        state = ((state >> 1) | (fb << (n - 1))) & ((1 << n) - 1)
        if state == start:
            return period
        if period > (1 << n):
            return None


def is_irreducible(v):
    return sp.Poly(int_to_poly(v), X, modulus=2).is_irreducible
