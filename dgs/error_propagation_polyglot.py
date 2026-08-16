"""The SAME uncertainty-propagation computation -- emf = B*h*v (Griffiths
7.13, dgs.error_propagation's own worked example), run for real in Python,
C, and C++ -- to make a genuine SOFTWARE-LANGUAGE-FORMALISM point, not just
a numerics one: C has no operator overloading, so "multiply two
Measurements and combine their uncertainties" MUST be an explicit function
call (measurement_mul(a, b)); C++ has operator overloading, so the exact
same logic can be written as `a * b`, syntactically identical to
dgs.error_propagation.Measurement's Python `__mul__`. All three implement
the identical first-order rule (relative sigmas add in quadrature for a
product) and are cross-checked to agree to near machine precision -- proof
the language difference is purely ergonomic (how you're ALLOWED to write
it), not a difference in what gets computed.

Same subprocess/gcc pattern as dgs.circuits_polyglot and dgs.matmul_benchmark.
"""

import os
import subprocess

import numpy as np

from dgs.error_propagation import Measurement, propagate, product_rule

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"
GPP_DEFAULT = r"C:\msys64\mingw64\bin\g++.exe"

# ── C: no operator overloading -- propagation MUST be an explicit function call ──

C_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct { double value; double sigma; } Measurement;

/* C has no operator overloading: there is no way to write `a * b` for two
   Measurement structs. Every combination rule has to be a named function
   the caller invokes explicitly. */
static Measurement measurement_mul(Measurement a, Measurement b) {
    double v = a.value * b.value;
    double ra = (a.value != 0.0) ? a.sigma / fabs(a.value) : 0.0;
    double rb = (b.value != 0.0) ? b.sigma / fabs(b.value) : 0.0;
    double rel = sqrt(ra*ra + rb*rb);
    Measurement out = { v, fabs(v) * rel };
    return out;
}

int main(int argc, char **argv) {
    Measurement B = { atof(argv[1]), atof(argv[2]) };
    Measurement h = { atof(argv[3]), atof(argv[4]) };
    Measurement v = { atof(argv[5]), atof(argv[6]) };

    /* explicit, procedural: two function calls, no infix syntax available */
    Measurement Bh  = measurement_mul(B, h);
    Measurement emf = measurement_mul(Bh, v);

    printf("%.10e %.10e\n", emf.value, emf.sigma);
    return 0;
}
"""

# ── C++: operator overloading -- propagation reads exactly like Python ──────

CPP_SOURCE = r"""
#include <iostream>
#include <cstdlib>
#include <cmath>

class Measurement {
public:
    double value, sigma;
    Measurement(double v, double s) : value(v), sigma(std::fabs(s)) {}

    /* operator overloading: the SAME logic as C's measurement_mul above,
       but callable as `a * b` -- syntactically identical to Python's
       Measurement.__mul__ in dgs/error_propagation.py. */
    Measurement operator*(const Measurement& o) const {
        double v = value * o.value;
        double ra = (value != 0.0) ? sigma / std::fabs(value) : 0.0;
        double rb = (o.value != 0.0) ? o.sigma / std::fabs(o.value) : 0.0;
        double rel = std::sqrt(ra*ra + rb*rb);
        return Measurement(v, std::fabs(v) * rel);
    }
};

int main(int argc, char **argv) {
    Measurement B(std::atof(argv[1]), std::atof(argv[2]));
    Measurement h(std::atof(argv[3]), std::atof(argv[4]));
    Measurement v(std::atof(argv[5]), std::atof(argv[6]));

    /* infix syntax, exactly like the Python reference */
    Measurement emf = B * h * v;

    printf("%.10e %.10e\n", emf.value, emf.sigma);
    return 0;
}
"""


def _compile(source, filename, out_dir, compiler_path, extra_flags=()):
    """Write `source` to `filename` in out_dir and compile it with
    `compiler_path -O2`, mirroring dgs.circuits_polyglot.compile_c_rlc."""
    stem, ext = os.path.splitext(filename)
    src_path = os.path.join(out_dir, filename)
    exe_path = os.path.join(out_dir, stem + ".exe")
    with open(src_path, "w") as f:
        f.write(source)
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(compiler_path) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([compiler_path, "-O2", *extra_flags, "-o", exe_path, src_path],
                             capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{compiler_path} compile failed: {result.stderr}")
    return exe_path


def compile_c(out_dir, gcc_path=GCC_DEFAULT):
    return _compile(C_SOURCE, "measurement_mul.c", out_dir, gcc_path)


def compile_cpp(out_dir, gpp_path=GPP_DEFAULT):
    return _compile(CPP_SOURCE, "measurement_mul.cpp", out_dir, gpp_path, extra_flags=("-std=c++17",))


def _run_exe(exe_path, B, h, v):
    """Run a compiled binary with (value, sigma) triples for B, h, v and
    parse its "value sigma" stdout line."""
    args = [exe_path, str(B.value), str(B.sigma), str(h.value), str(h.sigma),
            str(v.value), str(v.sigma)]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{exe_path} failed: {result.stderr}")
    value_str, sigma_str = result.stdout.strip().split()
    return float(value_str), float(sigma_str)


def cross_validate_languages(out_dir, B=(0.5, 0.01), h=(2.0, 0.05), v=(3.0, 0.1),
                              gcc_path=GCC_DEFAULT, gpp_path=GPP_DEFAULT,
                              run_c=True, run_cpp=True):
    """Compute emf = B*h*v (with propagated uncertainty) in Python (three
    independent ways: Measurement, propagate(), product_rule()), C
    (explicit function calls), and C++ (operator overloading), and report
    the max absolute disagreement between every pair -- proof they actually
    agree, not an assumption that a faithful translation would."""
    B_m, h_m, v_m = Measurement(*B), Measurement(*h), Measurement(*v)

    emf_py = B_m * h_m * v_m
    f = lambda p: p[0] * p[1] * p[2]
    val_lin, sig_lin = propagate(f, [B[0], h[0], v[0]], [B[1], h[1], v[1]])
    sig_closed = product_rule(val_lin, [(B[0], B[1]), (h[0], h[1]), (v[0], v[1])])

    out = {
        "python_measurement": (emf_py.value, emf_py.sigma),
        "python_propagate_linear": (val_lin, sig_lin),
        "python_product_rule_closed_form": (val_lin, sig_closed),
    }

    if run_c:
        exe_c = compile_c(out_dir, gcc_path=gcc_path)
        out["c_explicit_function_calls"] = _run_exe(exe_c, B_m, h_m, v_m)

    if run_cpp:
        exe_cpp = compile_cpp(out_dir, gpp_path=gpp_path)
        out["cpp_operator_overloading"] = _run_exe(exe_cpp, B_m, h_m, v_m)

    reference_value, reference_sigma = out["python_measurement"]
    max_abs_diff = 0.0
    for name, (val, sig) in out.items():
        max_abs_diff = max(max_abs_diff, abs(val - reference_value), abs(sig - reference_sigma))
    out["max_abs_diff_across_all_implementations"] = float(max_abs_diff)

    return out


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        results = cross_validate_languages(tmp)

    print("emf = B*h*v, propagated uncertainty -- same computation, three languages:\n")
    max_diff = results.pop("max_abs_diff_across_all_implementations")
    for name, (val, sig) in results.items():
        print(f"  {name:38s} emf = {val:.6f} +/- {sig:.6f}")
    print(f"\nmax abs diff across ALL implementations (value or sigma): {max_diff:.2e}")

    print("\nThe language-formalism point: C's measurement_mul(B, h) is an ordinary")
    print("function call (C has no operator overloading), while C++'s `B * h` and")
    print("Python's `B * h` are both operator-overload calls to __mul__ / operator*")
    print("-- SAME logic, syntax availability differs by language, numerics agree exactly.")
