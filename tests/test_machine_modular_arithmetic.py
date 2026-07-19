"""Test that machine integer overflow IS arithmetic modulo 2^n, verified
against a real gcc-compiled C program (not simulated). Requires mingw64
on PATH -- run via PowerShell, not the Bash tool (same constraint as
dgs.c_type_precision)."""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import machine_modular_arithmetic as mma

with tempfile.TemporaryDirectory() as tmp:
    machine = mma.compile_and_run_overflow_demo(tmp)

# 1. unsigned overflow at every width matches (value+1) mod 2^n exactly
for name, n_bits in [("u8", 8), ("u16", 16), ("u32", 32)]:
    value, actual = machine[name]
    predicted = mma.predict_unsigned_overflow(value, 1, n_bits)
    assert actual == predicted, f"{name}: {actual} != {predicted}"
    assert value == 2 ** n_bits - 1   # confirms the test actually starts at the max value

# 2. signed overflow (two's complement) matches the mod-2^n-then-reinterpret prediction
for name, n_bits in [("s8", 8), ("s32", 32)]:
    value, actual = machine[name]
    predicted = mma.predict_signed_overflow(value, 1, n_bits)
    assert actual == predicted, f"{name}: {actual} != {predicted}"
    assert value == 2 ** (n_bits - 1) - 1   # confirms starting at INT_MAX
    assert actual == -(2 ** (n_bits - 1))    # wraps to INT_MIN, the classic overflow bug

# 3. multiplication overflow matches too, not just the +1 addition edge case
big, actual_product = machine["mult_overflow"]
predicted_product = mma.predict_unsigned_multiply_overflow(big, 3, 32)
assert actual_product == predicted_product
assert big * 3 > 2 ** 32   # confirms this genuinely overflows (true product exceeds the range)

# 4. general correctness of the prediction functions themselves, independent
#    of the C program: small (non-overflowing) values pass through unchanged
assert mma.predict_unsigned_overflow(10, 5, 8) == 15
assert mma.predict_signed_overflow(10, 5, 8) == 15
assert mma.predict_unsigned_multiply_overflow(3, 4, 8) == 12

# 5. input validation
for bad_call in [
    lambda: mma.predict_unsigned_overflow(0, 0, -1),
    lambda: mma.predict_signed_overflow(0, 0, 0),
    lambda: mma.predict_unsigned_multiply_overflow(0, 0, -8),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.machine_modular_arithmetic tests passed")
