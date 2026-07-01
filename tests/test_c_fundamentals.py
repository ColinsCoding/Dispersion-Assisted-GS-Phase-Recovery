import pytest
import sympy as sp
from dgs.c_fundamentals import (
    c_type_table, sizeof_check, bitwise_demo,
    bit_count, is_power_of_2, bit_set, bit_clear, bit_toggle, bit_test,
    rotate_left, no_temp_swap, rot13, rot_n,
    syntax_errors_quiz, pointer_cheatsheet, struct_layout_lesson,
    reverse_bits, c_interview_study_plan, c_types_sympy_5,
)


# ── C type table ──────────────────────────────────────────────────────

def test_type_table_has_all_types():
    table = c_type_table()
    names = [r["type"] for r in table]
    for t in ["char", "int", "long long", "float", "double", "pointer"]:
        assert t in names

def test_char_is_1_byte():
    r = sizeof_check("char")
    assert r["size_bytes"] == 1

def test_int_is_4_bytes():
    r = sizeof_check("int")
    assert r["size_bytes"] == 4

def test_long_long_is_8_bytes():
    r = sizeof_check("long long")
    assert r["size_bytes"] == 8

def test_double_is_8_bytes():
    r = sizeof_check("double")
    assert r["size_bytes"] == 8

def test_pointer_is_8_bytes_on_64bit():
    r = sizeof_check("pointer")
    assert r["size_bytes"] == 8   # x86-64 only

def test_float_is_4_bytes():
    r = sizeof_check("float")
    assert r["size_bytes"] == 4

def test_invalid_type():
    with pytest.raises(ValueError):
        sizeof_check("complex128")


# ── bitwise operators ─────────────────────────────────────────────────

def test_bitwise_and():
    r = bitwise_demo(0b1010, 0b1100)
    assert r["AND"]["value"] == 0b1000

def test_bitwise_or():
    r = bitwise_demo(0b1010, 0b1100)
    assert r["OR"]["value"] == 0b1110

def test_bitwise_xor():
    r = bitwise_demo(0b1010, 0b1100)
    assert r["XOR"]["value"] == 0b0110

def test_bitwise_shl():
    r = bitwise_demo(0b0001, 0b0001, n_bits=8)
    assert r["SHL2"]["value"] == 0b0100

def test_bitwise_shr():
    r = bitwise_demo(0b1000, 0b0001, n_bits=8)
    assert r["SHR1"]["value"] == 0b0100

def test_logical_vs_bitwise_danger():
    # a=2(0b10), b=1(0b01): a&b=0 (no shared bits) but a&&b=True (both nonzero)
    r = bitwise_demo(2, 1)
    assert r["AND"]["value"] == 0
    assert r["logical_AND"] is True   # the DANGER case


# ── bit manipulation patterns ─────────────────────────────────────────

def test_bit_count_zero():
    assert bit_count(0)["popcount"] == 0

def test_bit_count_all_ones():
    # 0b11111111 = 255: 8 bits set
    assert bit_count(255)["popcount"] == 8

def test_bit_count_power_of_2():
    assert bit_count(64)["popcount"] == 1

def test_bit_count_arbitrary():
    assert bit_count(0b10110101)["popcount"] == 5

def test_bit_count_invalid():
    with pytest.raises(ValueError):
        bit_count(-1)

def test_is_power_of_2_true():
    for n in [1, 2, 4, 8, 16, 1024]:
        assert bool(is_power_of_2(n)["is_power_of_2"]) is True

def test_is_power_of_2_false():
    for n in [0, 3, 5, 6, 7, 9, 100]:
        assert bool(is_power_of_2(n)["is_power_of_2"]) is False

def test_bit_set_clears_bit_then_sets():
    r = bit_set(0b1010, 0)   # set bit 0
    assert r["result"] == 0b1011

def test_bit_clear():
    r = bit_clear(0b1011, 0)   # clear bit 0
    assert r["result"] == 0b1010

def test_bit_toggle():
    r = bit_toggle(0b1010, 0)  # toggle bit 0 (was 0 -> becomes 1)
    assert r["result"] == 0b1011

def test_bit_test_set():
    r = bit_test(0b1010, 1)    # bit 1 is set
    assert r["result"] is True

def test_bit_test_clear():
    r = bit_test(0b1010, 0)    # bit 0 is clear
    assert r["result"] is False


# ── rotate left ───────────────────────────────────────────────────────

def test_rotate_left_by_zero():
    r = rotate_left(0b1010, 0, bits=8)
    assert r["result"] == 0b1010

def test_rotate_left_by_1():
    r = rotate_left(0b10000000, 1, bits=8)
    assert r["result"] == 0b00000001   # MSB wraps to LSB

def test_rotate_left_by_32_is_identity():
    r = rotate_left(0xDEADBEEF, 32, bits=32)
    assert r["result"] == 0xDEADBEEF

def test_rotate_invalid_bits():
    with pytest.raises(ValueError):
        rotate_left(5, 1, bits=7)


# ── XOR swap ─────────────────────────────────────────────────────────

def test_xor_swap():
    r = no_temp_swap(42, 17)
    assert r["after_swap"] == (17, 42)

def test_xor_swap_zeros():
    r = no_temp_swap(0, 0)
    assert r["after_swap"] == (0, 0)

def test_xor_swap_negative():
    r = no_temp_swap(-5, 3)
    assert r["after_swap"] == (3, -5)


# ── ROT ciphers ───────────────────────────────────────────────────────

def test_rot13_is_self_inverse():
    r = rot13("Hello Physics!")
    assert bool(r["verify"]) is True

def test_rot13_a_becomes_n():
    r = rot13("a")
    assert r["encoded"] == "n"

def test_rot13_preserves_non_alpha():
    r = rot13("123 !@#")
    assert r["encoded"] == "123 !@#"

def test_rot_n_roundtrip():
    text = "Engineering"
    r1 = rot_n(text, 5)
    r2 = rot_n(r1["encoded"], 21)   # 26 - 5 = 21
    assert r2["encoded"] == text

def test_rot_n_zero_is_identity():
    r = rot_n("Hello", 0)
    assert r["encoded"] == "Hello"

def test_rot_n_13_matches_rot13():
    r_n = rot_n("Physics", 13)
    r_13 = rot13("Physics")
    assert r_n["encoded"] == r_13["encoded"]


# ── syntax errors and pointers ────────────────────────────────────────

def test_syntax_errors_at_least_8():
    errors = syntax_errors_quiz()
    assert len(errors) >= 8

def test_syntax_errors_have_keys():
    for err in syntax_errors_quiz():
        assert "error" in err
        assert "wrong" in err
        assert "correct" in err

def test_pointer_cheatsheet_has_deref():
    cs = pointer_cheatsheet()
    assert "deref" in cs
    assert "*p" in cs["deref"]

def test_struct_lesson():
    r = struct_layout_lesson()
    assert r["sizeof_padded"] > r["sizeof_packed"]   # padding wastes space


# ── reverse bits ──────────────────────────────────────────────────────

def test_reverse_bits_zero():
    r = reverse_bits(0)
    assert r["result"] == 0

def test_reverse_bits_one():
    r = reverse_bits(1, 8)
    assert r["result"] == 0b10000000

def test_reverse_bits_all_ones():
    r = reverse_bits(0xFF, 8)
    assert r["result"] == 0xFF

def test_reverse_bits_invalid():
    with pytest.raises(ValueError):
        reverse_bits(-1)


# ── study plan ────────────────────────────────────────────────────────

def test_study_plan_10_items():
    plan = c_interview_study_plan()
    assert len(plan) == 10

def test_study_plan_first_is_types():
    plan = c_interview_study_plan()
    p1 = [p for p in plan if p["priority"] == 1][0]
    assert "type" in p1["topic"].lower() or "size" in p1["topic"].lower()


# ── SymPy 5 ──────────────────────────────────────────────────────────

def test_c_sympy_5_count():
    eqs = c_types_sympy_5()
    assert len(eqs) == 5

def test_c_sympy_5_types():
    for k, eq in c_types_sympy_5().items():
        assert isinstance(eq, sp.Basic), f"{k} not SymPy"
