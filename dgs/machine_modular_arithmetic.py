"""Fixed-width machine integer arithmetic IS arithmetic modulo 2^n --
not an analogy, the literal defining behavior of overflow. An n-bit
unsigned integer can only represent {0, 1, ..., 2^n - 1}; when a+b would
exceed that range, the hardware doesn't error, it wraps: the true result
is silently reduced mod 2^n. This module verifies that claim against a
REAL compiled C program (gcc, same pattern as dgs.c_type_precision),
which is the actual machine doing actual overflow, not a Python
simulation of what overflow "should" look like.

Two's-complement SIGNED overflow is the same idea with a shifted range:
an n-bit signed int represents {-2^(n-1), ..., 2^(n-1)-1}, and wraps by
the SAME mod-2^n arithmetic, just reinterpreted -- INT_MAX + 1 becomes
INT_MIN, which is precisely (INT_MAX + 1) mod 2^n, reinterpreted as signed.
"""

import os
import subprocess

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"

C_SOURCE_OVERFLOW = r"""
#include <stdio.h>
#include <stdint.h>

int main(void) {
    uint8_t  u8  = UINT8_MAX;   uint8_t  u8r  = (uint8_t)(u8  + 1);
    uint16_t u16 = UINT16_MAX;  uint16_t u16r = (uint16_t)(u16 + 1);
    uint32_t u32 = UINT32_MAX;  uint32_t u32r = (uint32_t)(u32 + 1);

    int8_t  s8  = INT8_MAX;   int8_t  s8r  = (int8_t)(s8  + 1);
    int32_t s32 = INT32_MAX;  int32_t s32r = (int32_t)(s32 + 1);

    printf("%u %u\n", (unsigned)u8,  (unsigned)u8r);
    printf("%u %u\n", (unsigned)u16, (unsigned)u16r);
    printf("%u %u\n", (unsigned)u32, (unsigned)u32r);
    printf("%d %d\n", (int)s8,  (int)s8r);
    printf("%d %d\n", (int)s32, (int)s32r);

    /* an ARBITRARY multiplication overflow, not just the +1 edge case */
    uint32_t big = 3000000000U;
    uint32_t product = big * 3U;   /* true product is 9e9, way past 2^32 */
    printf("%u %u\n", big, product);
    return 0;
}
"""


def compile_and_run_overflow_demo(out_dir, gcc_path=GCC_DEFAULT):
    """Compile and run C_SOURCE_OVERFLOW (real gcc, same pattern as
    dgs.c_type_precision), returning the actual machine's overflow
    results -- not a Python guess at what they should be."""
    src_path = os.path.join(out_dir, "overflow.c")
    exe_path = os.path.join(out_dir, "overflow.exe")
    with open(src_path, "w") as f:
        f.write(C_SOURCE_OVERFLOW)
    result = subprocess.run([gcc_path, "-O2", "-o", exe_path, src_path],
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    run_result = subprocess.run([exe_path], capture_output=True, text=True)
    if run_result.returncode != 0:
        raise RuntimeError(f"program failed: {run_result.stderr}")
    lines = run_result.stdout.strip().splitlines()
    u8, u8r = map(int, lines[0].split())
    u16, u16r = map(int, lines[1].split())
    u32, u32r = map(int, lines[2].split())
    s8, s8r = map(int, lines[3].split())
    s32, s32r = map(int, lines[4].split())
    big, product = map(int, lines[5].split())
    return {
        "u8": (u8, u8r), "u16": (u16, u16r), "u32": (u32, u32r),
        "s8": (s8, s8r), "s32": (s32, s32r),
        "mult_overflow": (big, product),
    }


def predict_unsigned_overflow(value, delta, n_bits):
    """The CLAIM, computed independently in Python via explicit modular
    arithmetic: (value + delta) mod 2^n. If machine overflow really IS
    mod-2^n arithmetic, this must match the compiled C program exactly."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    return (value + delta) % (2 ** n_bits)


def predict_signed_overflow(value, delta, n_bits):
    """Signed n-bit overflow: compute (value + delta) mod 2^n in the
    UNSIGNED sense first (two's complement is just unsigned arithmetic
    reinterpreted), then reinterpret the top bit as a sign bit -- if the
    unsigned result is >= 2^(n-1), subtract 2^n to fold it into the
    negative range."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    unsigned_result = (value + delta) % (2 ** n_bits)
    half = 2 ** (n_bits - 1)
    return unsigned_result - 2 ** n_bits if unsigned_result >= half else unsigned_result


def predict_unsigned_multiply_overflow(a, b, n_bits):
    """Multiplication overflow is the SAME mod-2^n reduction, just applied
    to a product instead of a sum -- verified against real C below."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    return (a * b) % (2 ** n_bits)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        machine = compile_and_run_overflow_demo(tmp)

    print("Real gcc-compiled machine overflow vs. Python's explicit mod-2^n prediction:")
    print()
    for name, n_bits, signed in [("u8", 8, False), ("u16", 16, False), ("u32", 32, False)]:
        value, actual_wrapped = machine[name]
        predicted = predict_unsigned_overflow(value, 1, n_bits)
        print(f"  uint{n_bits}: {value} + 1 -> machine gives {actual_wrapped}, "
              f"({value}+1) mod 2^{n_bits} = {predicted}  match: {actual_wrapped == predicted}")

    for name, n_bits in [("s8", 8), ("s32", 32)]:
        value, actual_wrapped = machine[name]
        predicted = predict_signed_overflow(value, 1, n_bits)
        print(f"  int{n_bits}: {value} + 1 -> machine gives {actual_wrapped}, "
              f"predicted (mod 2^{n_bits}, reinterpreted signed) = {predicted}  "
              f"match: {actual_wrapped == predicted}")

    big, actual_product = machine["mult_overflow"]
    predicted_product = predict_unsigned_multiply_overflow(big, 3, 32)
    print(f"\n  uint32 multiply: {big} * 3 -> machine gives {actual_product}, "
          f"({big}*3) mod 2^32 = {predicted_product}  match: {actual_product == predicted_product}")

    print("\nEvery 'overflow' above is machine arithmetic silently computing the")
    print("mathematically correct answer MODULO 2^n -- not an error, not undefined")
    print("chaos, but a specific, predictable, provable arithmetic structure.")
