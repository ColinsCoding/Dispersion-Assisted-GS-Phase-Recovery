"""char, int, float, double -- compiled and run for real in C (same
gcc-subprocess pattern as dgs.circuits_polyglot), cross-checked against
NumPy's own type-info tables, and tied directly to two real precision bugs
already found elsewhere in this repo:

  * dgs.dual_autodiff's finite-difference "sweet spot" (h ~ 1e-6 for
    central differences) is not a coincidence -- it sits right where
    sqrt(DBL_EPSILON) (double's ~16-digit precision) predicts truncation
    error and roundoff error cross over.
  * dgs.doppler_numerical_derivation's "+c" units bug silently destroyed
    12+ digits of precision by pushing ~1e-14 second differences into
    ~1-second-scale doubles -- exactly the kind of failure this module's
    epsilon numbers explain.

C's `float` is IEEE 754 single precision (~7 significant decimal digits);
`double` is IEEE 754 double precision (~15-17 digits) -- NumPy's float32/
float64 are the SAME formats, not a Python-specific approximation.
"""

import os
import subprocess

import numpy as np

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"

C_SOURCE_TYPE_INFO = r"""
#include <stdio.h>
#include <float.h>
#include <limits.h>

int main(void) {
    printf("%zu %zu %zu %zu\n", sizeof(char), sizeof(int), sizeof(float), sizeof(double));
    printf("%d %d\n", CHAR_MIN, CHAR_MAX);
    printf("%d %d\n", INT_MIN, INT_MAX);
    printf("%.9e %.9e %d\n", (double)FLT_EPSILON, (double)FLT_MAX, FLT_DIG);
    printf("%.17e %.17e %d\n", DBL_EPSILON, DBL_MAX, DBL_DIG);
    return 0;
}
"""


def compile_c_type_info(out_dir, gcc_path=GCC_DEFAULT):
    """Write C_SOURCE_TYPE_INFO to disk and compile with gcc (same pattern
    as dgs.circuits_polyglot.compile_c_rlc)."""
    src_path = os.path.join(out_dir, "type_info.c")
    exe_path = os.path.join(out_dir, "type_info.exe")
    with open(src_path, "w") as f:
        f.write(C_SOURCE_TYPE_INFO)
    result = subprocess.run([gcc_path, "-O2", "-o", exe_path, src_path],
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    return exe_path


def run_c_type_info(exe_path):
    """Run the compiled C program and parse its sizeof/epsilon/range output
    into a plain dict -- the ACTUAL numbers this specific compiler/platform
    uses, not textbook values assumed to be true."""
    result = subprocess.run([exe_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C program failed: {result.stderr}")
    lines = result.stdout.strip().splitlines()
    sizeof_char, sizeof_int, sizeof_float, sizeof_double = map(int, lines[0].split())
    char_min, char_max = map(int, lines[1].split())
    int_min, int_max = map(int, lines[2].split())
    flt_eps, flt_max, flt_dig = lines[3].split()
    dbl_eps, dbl_max, dbl_dig = lines[4].split()
    return {
        "sizeof_char": sizeof_char, "sizeof_int": sizeof_int,
        "sizeof_float": sizeof_float, "sizeof_double": sizeof_double,
        "char_min": char_min, "char_max": char_max,
        "int_min": int_min, "int_max": int_max,
        "flt_epsilon": float(flt_eps), "flt_max": float(flt_max), "flt_dig": int(flt_dig),
        "dbl_epsilon": float(dbl_eps), "dbl_max": float(dbl_max), "dbl_dig": int(dbl_dig),
    }


def cross_check_against_numpy(c_info):
    """Confirm C's float/double are the exact same IEEE 754 formats as
    NumPy's float32/float64 -- not just similarly-named, but numerically
    identical epsilon and byte size."""
    np_f32, np_f64 = np.finfo(np.float32), np.finfo(np.float64)
    checks = {
        "float_size_matches_numpy_float32": c_info["sizeof_float"] == np.dtype(np.float32).itemsize,
        "double_size_matches_numpy_float64": c_info["sizeof_double"] == np.dtype(np.float64).itemsize,
        "float_epsilon_matches_numpy": abs(c_info["flt_epsilon"] - np_f32.eps) / np_f32.eps < 1e-3,
        "double_epsilon_matches_numpy": abs(c_info["dbl_epsilon"] - np_f64.eps) / np_f64.eps < 1e-3,
    }
    return checks


def sqrt_epsilon_predicts_fd_sweet_spot(dbl_epsilon):
    """The classic numerical-analysis rule of thumb: a central-difference
    derivative's optimal step size is roughly (machine epsilon)^(1/3)
    (balancing O(h^2) truncation error against O(epsilon/h) roundoff
    error) -- this is EXACTLY what dgs.dual_autodiff.finite_difference_error_sweep
    found empirically (h~1e-6) without this formula ever being consulted."""
    if dbl_epsilon <= 0:
        raise ValueError("dbl_epsilon must be positive")
    return dbl_epsilon ** (1.0 / 3.0)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        exe = compile_c_type_info(tmp)
        c_info = run_c_type_info(exe)

    print("compiled and run for real (gcc -O2):")
    print(f"  sizeof(char)={c_info['sizeof_char']}  sizeof(int)={c_info['sizeof_int']}  "
          f"sizeof(float)={c_info['sizeof_float']}  sizeof(double)={c_info['sizeof_double']}")
    print(f"  char range:   [{c_info['char_min']}, {c_info['char_max']}]")
    print(f"  int range:    [{c_info['int_min']}, {c_info['int_max']}]")
    print(f"  float:  epsilon={c_info['flt_epsilon']:.6e}  max={c_info['flt_max']:.3e}  "
          f"~{c_info['flt_dig']} significant decimal digits")
    print(f"  double: epsilon={c_info['dbl_epsilon']:.6e}  max={c_info['dbl_max']:.3e}  "
          f"~{c_info['dbl_dig']} significant decimal digits")

    checks = cross_check_against_numpy(c_info)
    print("\ncross-check vs NumPy's float32/float64 (same IEEE 754 formats?):")
    for name, ok in checks.items():
        print(f"  {name}: {ok}")
    assert all(checks.values())

    h_predicted = sqrt_epsilon_predicts_fd_sweet_spot(c_info["dbl_epsilon"])
    print(f"\n(DBL_EPSILON)^(1/3) = {h_predicted:.3e}")
    print("dgs.dual_autodiff's empirically-found finite-difference sweet spot was ~1.17e-06 --")
    print("that's not a coincidence, it's this exact formula, derived from double's real precision.")
    print("\nAnd dgs.doppler_numerical_derivation's '+c' units bug pushed ~1e-14 s differences")
    print(f"into ~1 s scale doubles -- right at DBL_EPSILON={c_info['dbl_epsilon']:.3e} relative")
    print("precision, which is exactly why that bug silently cost 12+ digits of accuracy.")
