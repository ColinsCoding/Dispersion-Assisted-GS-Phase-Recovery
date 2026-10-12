"""Tests for dgs/gf2_polynomials.py."""
import random

import pytest
import sympy as sp
from sympy.logic.boolalg import to_anf

from dgs.gf2_polynomials import (
    X, poly_to_int, int_to_poly, to_hex, gf2_add, gf2_mul, gf2_divmod, sympy_divmod,
    anf_coefficients, anf_expression, truth_table, eval_anf, CRC_SPECS, crc_bitwise,
    crc_polynomial_remainder, crc_reference, detects_all_single_bit_errors,
    lfsr_period, is_irreducible,
)


def test_poly_hex_roundtrip():
    assert to_hex(X ** 8 + X ** 2 + X + 1) == "0x107"
    assert to_hex(X ** 2 + X ** 2 + 1) == "0x1"            # 2x^2 = 0 mod 2
    for v in (0, 1, 0x1021, 0x104C11DB7):
        assert poly_to_int(int_to_poly(v)) == v


def test_mul_and_divmod_match_sympy():
    rng = random.Random(1)
    for _ in range(50):
        a, b = rng.getrandbits(20), rng.getrandbits(10) | 1
        prod = sp.Poly(int_to_poly(a) * int_to_poly(b), X, modulus=2)
        assert gf2_mul(a, b) == poly_to_int(prod.as_expr())
        assert gf2_divmod(a, b) == sympy_divmod(a, b)
        q, r = gf2_divmod(a, b)
        assert gf2_add(gf2_mul(q, b), r) == a                 # a = q b + r
    with pytest.raises(ZeroDivisionError):
        gf2_divmod(5, 0)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_anf_reproduces_every_function(n):
    rng = random.Random(n)
    for _ in range(20):
        tt = [rng.getrandbits(1) for _ in range(1 << n)]
        c = anf_coefficients(tt)
        for m in range(1 << n):
            assert eval_anf(c, [(m >> i) & 1 for i in range(n)]) == tt[m]


def test_anf_matches_sympy_to_anf():
    a, b, c = sp.symbols("a b c")
    cases = [((lambda p, q: p | q), "a b", a | b),
             ((lambda p, q: 1 - (p & q)), "a b", ~(a & b)),
             ((lambda p, q, r: (p & q) | (q & r) | (p & r)), "a b c", (a & b) | (b & c) | (a & c))]
    for f, names, expr in cases:
        n = len(names.split())
        mine = anf_expression(truth_table(f, n), names)
        ref = to_anf(expr)
        syms = sp.symbols(names)
        for m in range(1 << n):
            vals = {s: bool((m >> i) & 1) for i, s in enumerate(syms)}
            assert int(mine.subs({s: int(v) for s, v in vals.items()})) % 2 == int(bool(ref.subs(vals)))


@pytest.mark.parametrize("name", list(CRC_SPECS))
def test_crc_check_values(name):
    d = b"123456789"
    assert crc_bitwise(d, name) == CRC_SPECS[name][5]
    ref = crc_reference(d, name)
    if ref is not None:
        assert crc_bitwise(d, name) == ref


def test_crc8_is_a_polynomial_remainder():
    rng = random.Random(7)
    for _ in range(20):
        d = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 12)))
        assert crc_polynomial_remainder(d, 8, 0x07) == crc_bitwise(d, "CRC-8/SMBUS")


def test_single_bit_errors_detected():
    assert detects_all_single_bit_errors(8, 0x07)
    d = bytearray(b"hello, waveguide")
    good = crc_bitwise(bytes(d), "CRC-32")
    for i in range(len(d) * 8):
        e = bytearray(d); e[i // 8] ^= 1 << (i % 8)
        assert crc_bitwise(bytes(e), "CRC-32") != good


def test_lfsr_primitive_vs_reducible():
    prbs7 = (1 << 7) | (1 << 6) | 1
    assert is_irreducible(prbs7) and lfsr_period(prbs7, 7) == 127
    red = (1 << 4) | (1 << 2) | 1                          # (x^2+x+1)^2
    assert not is_irreducible(red) and lfsr_period(red, 4) < 15
    with pytest.raises(ValueError):
        lfsr_period(prbs7, 7, seed=0)


def test_single_variable_not():
    a = sp.Symbol("a")
    assert anf_expression(truth_table(lambda p: 1 - p, 1), "a") == sp.Add(1, a, evaluate=False)
