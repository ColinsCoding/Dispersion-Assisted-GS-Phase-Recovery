"""Decomposing a 3D vector into components PARALLEL and PERPENDICULAR to a
given direction -- v_par = (v . a_hat) a_hat, v_perp = v - v_par -- as three
COMPLETE, compiled, runnable programs (Python/numpy, C, C++), continuing
dgs.error_propagation_polyglot's language-formalism theme: which operations
does the language let you write as an OPERATOR on the operands, and which
stay named functions/methods even in a language that supports overloading?

  * C (no operator overloading): vec3_add, vec3_sub, vec3_scale, vec3_dot
    are all explicit function calls -- v_par and v_perp are built by
    composing them by hand.

  * C++ (has operator overloading): operator+, operator-, and
    operator*(scalar) let `a + b`, `a - b`, `a * s` read like ordinary
    arithmetic on Vec3 operands -- but dot() stays a NAMED METHOD even in
    C++, not `a * b`, because `*` between two Vec3s would be genuinely
    ambiguous (dot product? component-wise product? cross product?). Not
    every operation that COULD be overloaded SHOULD be -- operator
    overloading is a tool for the cases where the symbol's meaning is
    unambiguous, not a mandate to overload everything.

Physical checks, not just arithmetic ones: v_par + v_perp must reconstruct
v exactly, v_par . v_perp must be ~0 (orthogonality), and
|v_par| = |v| cos(theta), |v_perp| = |v| sin(theta) against the angle
between v and a.
"""

import os
import subprocess

import numpy as np

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"
GPP_DEFAULT = r"C:\msys64\mingw64\bin\g++.exe"

# ── Python/numpy reference ───────────────────────────────────────────────────

def parallel_perp_numpy(v, a):
    """v_par = (v . a_hat) a_hat ,  v_perp = v - v_par.  Raises ValueError
    if `a` is the zero vector (no direction to project onto)."""
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    a_norm = np.linalg.norm(a)
    if a_norm == 0:
        raise ValueError("direction vector a must be nonzero")
    a_hat = a / a_norm
    v_par = np.dot(v, a_hat) * a_hat
    v_perp = v - v_par
    return v_par, v_perp


# ── C: no operator overloading -- vec3_dot/add/sub/scale are all explicit ──

C_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct { double x, y, z; } Vec3;

static Vec3 vec3_add(Vec3 a, Vec3 b)   { Vec3 r = { a.x+b.x, a.y+b.y, a.z+b.z }; return r; }
static Vec3 vec3_sub(Vec3 a, Vec3 b)   { Vec3 r = { a.x-b.x, a.y-b.y, a.z-b.z }; return r; }
static Vec3 vec3_scale(Vec3 a, double s) { Vec3 r = { a.x*s, a.y*s, a.z*s }; return r; }
static double vec3_dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static double vec3_norm(Vec3 a)        { return sqrt(vec3_dot(a, a)); }

/* every step below is a named function call -- there is no `v * a` in C */
static void parallel_perp(Vec3 v, Vec3 a, Vec3 *v_par, Vec3 *v_perp) {
    double a_norm = vec3_norm(a);
    Vec3 a_hat = vec3_scale(a, 1.0 / a_norm);
    double proj = vec3_dot(v, a_hat);
    *v_par = vec3_scale(a_hat, proj);
    *v_perp = vec3_sub(v, *v_par);
}

int main(int argc, char **argv) {
    Vec3 v = { atof(argv[1]), atof(argv[2]), atof(argv[3]) };
    Vec3 a = { atof(argv[4]), atof(argv[5]), atof(argv[6]) };

    Vec3 v_par, v_perp;
    parallel_perp(v, a, &v_par, &v_perp);

    printf("%.10e %.10e %.10e %.10e %.10e %.10e\n",
           v_par.x, v_par.y, v_par.z, v_perp.x, v_perp.y, v_perp.z);
    return 0;
}
"""

# ── C++: operator overloading for +, -, *(scalar) -- dot() stays a method ──

CPP_SOURCE = r"""
#include <iostream>
#include <cstdlib>
#include <cmath>

class Vec3 {
public:
    double x, y, z;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    Vec3 operator*(double s) const      { return Vec3(x*s, y*s, z*s); }

    /* NOT overloaded as `*` deliberately: a*b between two Vec3 could mean
       dot product, cross product, or component-wise product -- there is
       no single unambiguous meaning the way there is for scalar scaling,
       so this stays a named method even though C++ COULD overload it. */
    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    double norm() const { return std::sqrt(this->dot(*this)); }
};

static void parallel_perp(const Vec3& v, const Vec3& a, Vec3& v_par, Vec3& v_perp) {
    Vec3 a_hat = a * (1.0 / a.norm());
    double proj = v.dot(a_hat);
    v_par = a_hat * proj;      /* operator* : reads like ordinary scaling */
    v_perp = v - v_par;        /* operator- : reads like ordinary subtraction */
}

int main(int argc, char **argv) {
    Vec3 v(std::atof(argv[1]), std::atof(argv[2]), std::atof(argv[3]));
    Vec3 a(std::atof(argv[4]), std::atof(argv[5]), std::atof(argv[6]));

    Vec3 v_par(0,0,0), v_perp(0,0,0);
    parallel_perp(v, a, v_par, v_perp);

    printf("%.10e %.10e %.10e %.10e %.10e %.10e\n",
           v_par.x, v_par.y, v_par.z, v_perp.x, v_perp.y, v_perp.z);
    return 0;
}
"""


def _compile(source, filename, out_dir, compiler_path, extra_flags=()):
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
    return _compile(C_SOURCE, "vec3_parallel_perp.c", out_dir, gcc_path)


def compile_cpp(out_dir, gpp_path=GPP_DEFAULT):
    return _compile(CPP_SOURCE, "vec3_parallel_perp.cpp", out_dir, gpp_path, extra_flags=("-std=c++17",))


def _run_exe(exe_path, v, a):
    args = [exe_path] + [str(x) for x in v] + [str(x) for x in a]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{exe_path} failed: {result.stderr}")
    nums = [float(tok) for tok in result.stdout.split()]
    return np.array(nums[:3]), np.array(nums[3:6])


def verify_decomposition(v, a, v_par, v_perp, atol=1e-9) -> dict:
    """Physical checks -- not just "does it run," but "is this actually a
    valid parallel/perpendicular decomposition": reconstruction, orthogonality,
    and the trig identities |v_par|=|v|cos(theta), |v_perp|=|v|sin(theta)."""
    v = np.asarray(v, float)
    reconstruct_err = float(np.max(np.abs((v_par + v_perp) - v)))
    orthogonality = float(abs(np.dot(v_par, v_perp)))

    v_norm, a_norm = np.linalg.norm(v), np.linalg.norm(a)
    cos_theta = float(np.dot(v, a) / (v_norm * a_norm))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    par_norm_err = abs(np.linalg.norm(v_par) - v_norm * abs(np.cos(theta)))
    perp_norm_err = abs(np.linalg.norm(v_perp) - v_norm * abs(np.sin(theta)))

    return {
        "reconstructs_v": reconstruct_err < atol,
        "orthogonal": orthogonality < atol,
        "matches_cos_theta_identity": par_norm_err < atol,
        "matches_sin_theta_identity": perp_norm_err < atol,
        "reconstruct_err": reconstruct_err, "orthogonality": orthogonality,
        "par_norm_err": par_norm_err, "perp_norm_err": perp_norm_err,
    }


def cross_validate_languages(out_dir, v=(3.0, 4.0, 0.0), a=(1.0, 0.0, 0.0),
                              gcc_path=GCC_DEFAULT, gpp_path=GPP_DEFAULT,
                              run_c=True, run_cpp=True):
    """Compute the parallel/perpendicular decomposition of v with respect
    to a in Python/numpy, C, and C++, and report the max absolute
    disagreement across all languages for BOTH output vectors."""
    v_par_py, v_perp_py = parallel_perp_numpy(v, a)
    out = {"python_numpy": (v_par_py, v_perp_py)}

    if run_c:
        exe_c = compile_c(out_dir, gcc_path=gcc_path)
        out["c_explicit_functions"] = _run_exe(exe_c, v, a)

    if run_cpp:
        exe_cpp = compile_cpp(out_dir, gpp_path=gpp_path)
        out["cpp_operator_overloading"] = _run_exe(exe_cpp, v, a)

    ref_par, ref_perp = out["python_numpy"]
    max_abs_diff = 0.0
    for name, (par, perp) in out.items():
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(par - ref_par))),
                            float(np.max(np.abs(perp - ref_perp))))
    out["max_abs_diff_across_all_implementations"] = max_abs_diff
    return out


if __name__ == "__main__":
    import tempfile
    v, a = (3.0, 4.0, 0.0), (1.0, 0.0, 0.0)

    with tempfile.TemporaryDirectory() as tmp:
        results = cross_validate_languages(tmp, v=v, a=a)
    max_diff = results.pop("max_abs_diff_across_all_implementations")

    print(f"v = {v}, decomposed relative to a = {a}\n")
    for name, (v_par, v_perp) in results.items():
        print(f"  {name:28s} v_par = {tuple(np.round(v_par, 6))}  v_perp = {tuple(np.round(v_perp, 6))}")
    print(f"\nmax abs diff across ALL implementations: {max_diff:.2e}")

    checks = verify_decomposition(v, a, *results["python_numpy"])
    print("\nphysical checks (Python/numpy result):")
    for name in ("reconstructs_v", "orthogonal", "matches_cos_theta_identity", "matches_sin_theta_identity"):
        print(f"  {name}: {checks[name]}")
    assert all(checks[name] for name in
               ("reconstructs_v", "orthogonal", "matches_cos_theta_identity", "matches_sin_theta_identity"))

    print("\ndot() stayed a named method in C++ even though operator+, operator-,")
    print("and operator*(scalar) did not -- operator overloading is for symbols")
    print("with one unambiguous meaning, not everything the language allows.")
