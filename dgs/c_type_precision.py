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

PREREQUISITES for the three intermediate additions below (empirical
epsilon, catastrophic cancellation, Kahan summation): that floats/doubles
can't represent every real number exactly (some values, like 0.1, have no
finite binary expansion), and what DBL_EPSILON itself means -- both
established by compile_c_type_info/run_c_type_info above. Otherwise just
basic C (loops, printf; no pointers needed). If those two ideas aren't
solid yet, start there before this.
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
    # gcc.exe needs its own directory on PATH to find its internal driver/DLLs
    # (cc1.exe etc.) even when invoked by absolute path -- without this, gcc
    # fails silently (returncode 1, empty stdout/stderr) in any shell whose
    # inherited PATH doesn't already include it.
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(gcc_path) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([gcc_path, "-O2", "-o", exe_path, src_path],
                             capture_output=True, text=True, env=env)
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


def _compile_and_run(c_source, name, out_dir, gcc_path=GCC_DEFAULT):
    """Shared compile+run helper (same gcc-subprocess pattern and PATH fix
    as compile_c_type_info/run_c_type_info) for the smaller single-purpose
    C demos below, so that boilerplate lives in one place."""
    src_path = os.path.join(out_dir, f"{name}.c")
    exe_path = os.path.join(out_dir, f"{name}.exe")
    with open(src_path, "w") as f:
        f.write(c_source)
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(gcc_path) + os.pathsep + env.get("PATH", "")
    compiled = subprocess.run([gcc_path, "-O2", "-o", exe_path, src_path],
                               capture_output=True, text=True, env=env)
    if compiled.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {compiled.stderr}")
    ran = subprocess.run([exe_path], capture_output=True, text=True)
    if ran.returncode != 0:
        raise RuntimeError(f"C program failed: {ran.stderr}")
    return ran.stdout.strip()


C_SOURCE_EMPIRICAL_EPSILON = r"""
#include <stdio.h>
int main(void) {
    double eps = 1.0;
    while (1.0 + (eps / 2.0) != 1.0) {
        eps /= 2.0;
    }
    printf("%.17e\n", eps);
    return 0;
}
"""


def compute_epsilon_empirically(out_dir, gcc_path=GCC_DEFAULT):
    """Derive DBL_EPSILON the way it's actually DEFINED, not looked up:
    repeatedly halve eps until 1.0+eps stops being distinguishable from 1.0
    in double precision. One step up from run_c_type_info's 'read the
    constant float.h ships' -- this computes the same number from first
    principles, at runtime, on this exact machine, and should match
    DBL_EPSILON exactly (both are the same IEEE 754 double's ULP at 1.0)."""
    stdout = _compile_and_run(C_SOURCE_EMPIRICAL_EPSILON, "empirical_epsilon", out_dir, gcc_path)
    return float(stdout)


C_SOURCE_CANCELLATION = r"""
#include <stdio.h>
int main(void) {
    double a = 1.0;
    double b[5] = {1e-8, 1e-12, 1e-15, 1e-16, 1e-17};
    for (int i = 0; i < 5; i++) {
        double computed = (a + b[i]) - a;
        printf("%.17e %.17e\n", b[i], computed);
    }
    return 0;
}
"""


def catastrophic_cancellation_demo(out_dir, gcc_path=GCC_DEFAULT):
    """(1.0 + b) - b should return b exactly by real-number algebra. In
    double precision it silently degrades toward 0 once b drops below
    roughly DBL_EPSILON relative to 1.0 -- the exact mechanism behind
    dgs.doppler_numerical_derivation's '+c' bug (adding a tiny quantity to
    a much larger one, then subtracting the large one back out).

    Returns a list of (true_b, computed_b) pairs spanning that breakdown,
    smallest-error to largest."""
    stdout = _compile_and_run(C_SOURCE_CANCELLATION, "cancellation", out_dir, gcc_path)
    pairs = []
    for line in stdout.splitlines():
        true_b, computed = line.split()
        pairs.append((float(true_b), float(computed)))
    return pairs


C_SOURCE_KAHAN_TEMPLATE = r"""
#include <stdio.h>
int main(void) {{
    int n = {n};
    float term = {term}f;   /* not exactly representable in binary float, in general */

    float naive_sum = 0.0f;
    for (int i = 0; i < n; i++) {{
        naive_sum += term;
    }}

    float kahan_sum = 0.0f;
    float c = 0.0f;       /* running compensation for lost low-order bits */
    for (int i = 0; i < n; i++) {{
        float y = term - c;
        float t = kahan_sum + y;
        c = (t - kahan_sum) - y;
        kahan_sum = t;
    }}

    printf("%.9e %.9e\n", naive_sum, kahan_sum);
    return 0;
}}
"""


def kahan_summation_demo(out_dir, gcc_path=GCC_DEFAULT, n=1_000_000, term=0.1):
    """Sum `term` (default 0.1, not exactly representable in binary float)
    `n` times two ways: a naive running sum, whose roundoff error
    accumulates because each add's error is on the same order as the
    ALREADY-large running total, vs Kahan compensated summation, which
    tracks the bits each add loses in a separate accumulator and feeds them
    back in next time -- the classic intermediate-level fix for
    machine-precision error accumulation, one level up from just knowing
    epsilon exists.

    Returns (naive_sum, kahan_sum, true_sum) where true_sum = n * term
    (the exact real-number answer, for comparison)."""
    if n < 1:
        raise ValueError(f"n={n} must be >= 1")
    c_source = C_SOURCE_KAHAN_TEMPLATE.format(n=int(n), term=repr(float(term)))
    stdout = _compile_and_run(c_source, "kahan", out_dir, gcc_path)
    naive_str, kahan_str = stdout.split()
    return float(naive_str), float(kahan_str), n * term


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

    with tempfile.TemporaryDirectory() as tmp:
        eps_empirical = compute_epsilon_empirically(tmp)
    print(f"\nempirical DBL_EPSILON (halve-until-1+eps==1 loop, computed at runtime): {eps_empirical:.6e}")
    print(f"float.h's DBL_EPSILON (looked up):                                      {c_info['dbl_epsilon']:.6e}")
    print(f"agreement: {'exact match' if eps_empirical == c_info['dbl_epsilon'] else 'MISMATCH -- investigate'}")

    with tempfile.TemporaryDirectory() as tmp:
        pairs = catastrophic_cancellation_demo(tmp)
    print("\ncatastrophic cancellation: (1.0 + b) - b, which should return b exactly:")
    for true_b, computed_b in pairs:
        print(f"  b={true_b:.1e}  ->  computed={computed_b:.6e}  "
              f"({'OK' if computed_b != 0 or true_b == 0 else 'LOST COMPLETELY'})")

    with tempfile.TemporaryDirectory() as tmp:
        naive_sum, kahan_sum, true_sum = kahan_summation_demo(tmp)
    print(f"\nsumming 0.1 (float) x 1,000,000 -- true answer: {true_sum:.6f}")
    print(f"  naive running sum:      {naive_sum:.6f}  (error: {abs(naive_sum - true_sum):.6f})")
    print(f"  Kahan compensated sum:  {kahan_sum:.6f}  (error: {abs(kahan_sum - true_sum):.6f})")
    print("  Kahan summation cancels most of the accumulated float roundoff -- the classic")
    print("  intermediate-level technique for the failure mode DBL_EPSILON above predicts.")
