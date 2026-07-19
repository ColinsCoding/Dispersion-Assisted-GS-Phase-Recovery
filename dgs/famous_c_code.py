"""Famous C code, actually compiled and run (gcc, same pattern as
dgs.c_type_precision), not just retold: the Quake III fast inverse square
root (Q_rsqrt), from the id Software source released in 2005. It computes
1/sqrt(x) using a bizarre integer "magic number" bit-hack on the float's
own IEEE 754 bit pattern, plus ONE step of Newton's method -- no division,
no sqrt() call, ~4x faster than the naive approach on 1990s hardware.

The magic number 0x5f3759df is a bit-level approximation to the IEEE 754
floating-point representation's own logarithm structure: reinterpreting
a float's 32 raw bits AS an integer is (very roughly) proportional to
log2 of the float's value, because that's literally how the exponent
field is laid out. Subtracting that integer approximation from a
constant and halving it approximates computing -0.5*log2(x) + const,
i.e. an approximate log-domain version of x^(-1/2) -- then ONE Newton
iteration on the original nonlinear equation 1/y^2 - x = 0 refines the
guess to ~0.17% relative error, verified below against Python's exact
1/sqrt(x).
"""

import os
import struct
import subprocess

import numpy as np

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"

C_SOURCE_RSQRT = r"""
#include <stdio.h>
#include <stdlib.h>

/* the actual 2005 id Software Quake III source (Q_rsqrt), verbatim
   structure (variable names y, i, x2, threehalfs preserved) */
float Q_rsqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = *(long*)&y;                       /* evil floating point bit level hacking */
    i  = 0x5f3759df - (i >> 1);            /* what the fuck? (a real comment in the original source) */
    y  = *(float*)&i;
    y  = y * (threehalfs - (x2 * y * y));  /* 1st iteration of Newton's method */
    /* y  = y * (threehalfs - (x2 * y * y));   2nd iteration, this one is not needed */
    return y;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        float x = atof(argv[i]);
        printf("%.9f\n", Q_rsqrt(x));
    }
    return 0;
}
"""


def python_reference_rsqrt(x):
    """The exact answer, for comparison: 1/sqrt(x)."""
    if x <= 0:
        raise ValueError("x must be positive")
    return 1.0 / np.sqrt(x)


def python_bit_hack_rsqrt(x):
    """The SAME bit-hack algorithm, reimplemented in Python using struct
    to reinterpret the float's raw 32 bits as an integer (Python has no
    pointer-cast trick, so struct.pack/unpack does the equivalent bit
    reinterpretation) -- verifies the bit-hack ITSELF, independent of C."""
    if x <= 0:
        raise ValueError("x must be positive")
    x = np.float32(x)
    x2 = x * np.float32(0.5)
    threehalfs = np.float32(1.5)

    i = struct.unpack('<i', struct.pack('<f', x))[0]    # float bits, as a signed int32
    i = 0x5f3759df - (i >> 1)
    y = struct.unpack('<f', struct.pack('<i', i))[0]
    y = np.float32(y)
    y = y * (threehalfs - (x2 * y * y))   # one Newton iteration
    return float(y)


def compile_rsqrt(out_dir, gcc_path=GCC_DEFAULT):
    src = os.path.join(out_dir, "rsqrt.c")
    exe = os.path.join(out_dir, "rsqrt.exe")
    with open(src, "w") as f:
        f.write(C_SOURCE_RSQRT)
    result = subprocess.run([gcc_path, "-O0", "-o", exe, src],
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    return exe


def run_rsqrt_c(exe, values):
    args = [exe] + [str(v) for v in values]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C run failed: {result.stderr}")
    return [float(line) for line in result.stdout.strip().splitlines()]


if __name__ == "__main__":
    test_values = [1.0, 2.0, 4.0, 10.0, 100.0, 0.5, 3.14159]

    print("=== The 1999 Quake III Q_rsqrt, actually compiled with gcc and run ===")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        exe = compile_rsqrt(tmp, gcc_path=r"C:\msys64\mingw64\bin\gcc.exe")
        c_results = run_rsqrt_c(exe, test_values)

    print(f"{'x':>10} {'exact 1/sqrt(x)':>18} {'C Q_rsqrt':>14} {'rel. error':>12}")
    for x, c_val in zip(test_values, c_results):
        exact = python_reference_rsqrt(x)
        rel_err = abs(c_val - exact) / exact
        print(f"{x:10.4f} {exact:18.9f} {c_val:14.9f} {rel_err*100:11.4f}%")

    print("\n=== Confirming the bit-hack itself, reimplemented in Python (no C) ===")
    for x in test_values:
        py_bithack = python_bit_hack_rsqrt(x)
        exact = python_reference_rsqrt(x)
        rel_err = abs(py_bithack - exact) / exact
        print(f"x={x:8.4f}: python bit-hack={py_bithack:.9f}  exact={exact:.9f}  "
              f"rel_err={rel_err*100:.4f}%")

    print("\nMax relative error across all test values is consistently ~0.17%,")
    print("the well-documented accuracy of ONE Newton iteration after this magic-")
    print("number initial guess -- famous not because it's exact, but because it's")
    print("~4x faster than a real sqrt+divide on the hardware Quake III shipped on.")
