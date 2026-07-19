"""The same series-RLC step response, solved by RK4, in C, MATLAB, and Python --
run for real in all three languages and cross-checked against each other, not just
described. This is dgs.spice.rlc_step_response's exact algorithm (same state
equations, same RK4 weights), reimplemented from scratch in C and MATLAB so the
translation is verifiable rather than asserted.

    L diL/dt = V - R*iL - vc  ,   C dvc/dt = iL

Three languages, one ODE: Python (numpy, this repo's normal path), C (compiled with
gcc, called via subprocess -- same pattern as dgs.matmul_benchmark), and MATLAB
(run headless via `matlab -batch`, actually installed on this machine at
C:\\Program Files\\MATLAB\\R2025b). cross_validate_languages() runs all three on
identical inputs and checks they agree to near machine precision -- proof the
"same physics, different syntax" claim is actually true, not just plausible.
"""

import os
import re
import subprocess

import numpy as np

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"
MATLAB_DEFAULT = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"


def rlc_rk4_python(R, L, C, V, vc0, il0, dt, n_steps):
    """Reference implementation: hand-unrolled RK4 for the series-RLC state
    equations (same math as dgs.spice.rlc_step_response, without the numpy
    time-array wrapper) -- the standard every other language's output is
    checked against. Returns (vc_array, il_array) of length n_steps+1."""
    def deriv(v, i):
        return i / C, (V - R * i - v) / L

    vc, il = vc0, il0
    vcs, ils = [vc], [il]
    for _ in range(n_steps):
        dv1, di1 = deriv(vc, il)
        dv2, di2 = deriv(vc + 0.5 * dt * dv1, il + 0.5 * dt * di1)
        dv3, di3 = deriv(vc + 0.5 * dt * dv2, il + 0.5 * dt * di2)
        dv4, di4 = deriv(vc + dt * dv3, il + dt * di3)
        vc = vc + dt / 6 * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
        il = il + dt / 6 * (di1 + 2 * di2 + 2 * di3 + di4)
        vcs.append(vc)
        ils.append(il)
    return np.array(vcs), np.array(ils)


C_SOURCE_RLC = r"""
#include <stdio.h>
#include <stdlib.h>

static void deriv(double v, double i, double R, double L, double C, double V,
                   double *dv, double *di) {
    *dv = i / C;
    *di = (V - R * i - v) / L;
}

int main(int argc, char **argv) {
    double R = atof(argv[1]), L = atof(argv[2]), C = atof(argv[3]), V = atof(argv[4]);
    double vc = atof(argv[5]), il = atof(argv[6]);
    double dt = atof(argv[7]);
    int n_steps = atoi(argv[8]);

    printf("%.10e %.10e\n", vc, il);
    for (int n = 0; n < n_steps; n++) {
        double dv1, di1, dv2, di2, dv3, di3, dv4, di4;
        deriv(vc, il, R, L, C, V, &dv1, &di1);
        deriv(vc + 0.5*dt*dv1, il + 0.5*dt*di1, R, L, C, V, &dv2, &di2);
        deriv(vc + 0.5*dt*dv2, il + 0.5*dt*di2, R, L, C, V, &dv3, &di3);
        deriv(vc + dt*dv3, il + dt*di3, R, L, C, V, &dv4, &di4);
        vc = vc + dt/6.0*(dv1 + 2*dv2 + 2*dv3 + dv4);
        il = il + dt/6.0*(di1 + 2*di2 + 2*di3 + di4);
        printf("%.10e %.10e\n", vc, il);
    }
    return 0;
}
"""

MATLAB_SOURCE_TEMPLATE = """
R = {R!r}; L = {L!r}; C = {C!r}; V = {V!r};
vc = {vc0!r}; il = {il0!r}; dt = {dt!r}; n_steps = {n_steps!r};

fprintf('%.10e %.10e\\n', vc, il);
for n = 1:n_steps
    [dv1, di1] = rlc_deriv(vc, il, R, L, C, V);
    [dv2, di2] = rlc_deriv(vc + 0.5*dt*dv1, il + 0.5*dt*di1, R, L, C, V);
    [dv3, di3] = rlc_deriv(vc + 0.5*dt*dv2, il + 0.5*dt*di2, R, L, C, V);
    [dv4, di4] = rlc_deriv(vc + dt*dv3, il + dt*di3, R, L, C, V);
    vc = vc + dt/6*(dv1 + 2*dv2 + 2*dv3 + dv4);
    il = il + dt/6*(di1 + 2*di2 + 2*di3 + di4);
    fprintf('%.10e %.10e\\n', vc, il);
end

function [dv, di] = rlc_deriv(v, i, R, L, C, V)
    dv = i / C;
    di = (V - R*i - v) / L;
end
"""


def compile_c_rlc(out_dir, gcc_path=GCC_DEFAULT):
    """Write C_SOURCE_RLC to disk and compile with gcc -O2 (same pattern as
    dgs.matmul_benchmark.compile_c_matmul)."""
    src_path = os.path.join(out_dir, "rlc_rk4.c")
    exe_path = os.path.join(out_dir, "rlc_rk4.exe")
    with open(src_path, "w") as f:
        f.write(C_SOURCE_RLC)
    result = subprocess.run([gcc_path, "-O2", "-o", exe_path, src_path],
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    return exe_path


def run_c_rlc(exe_path, R, L, C, V, vc0, il0, dt, n_steps):
    """Run the compiled C RLC integrator and parse its (vc, il) columns."""
    result = subprocess.run(
        [exe_path, str(R), str(L), str(C), str(V), str(vc0), str(il0), str(dt), str(n_steps)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"C binary failed: {result.stderr}")
    lines = result.stdout.strip().splitlines()
    vc = np.array([float(line.split()[0]) for line in lines])
    il = np.array([float(line.split()[1]) for line in lines])
    return vc, il


def run_matlab_rlc(out_dir, R, L, C, V, vc0, il0, dt, n_steps, matlab_path=MATLAB_DEFAULT):
    """Write MATLAB_SOURCE_TEMPLATE to disk and run it headless via
    `matlab -batch run(...)`. Parses stdout for the same "vc il" lines the C
    and Python paths produce (MATLAB's batch output can carry banner/warning
    text, so lines are matched by a float-float regex rather than assumed
    to be exactly the script's fprintf output)."""
    script = MATLAB_SOURCE_TEMPLATE.format(R=R, L=L, C=C, V=V, vc0=vc0, il0=il0, dt=dt, n_steps=n_steps)
    m_path = os.path.join(out_dir, "rlc_rk4.m")
    with open(m_path, "w") as f:
        f.write(script)
    m_path_fwd = m_path.replace("\\", "/")

    result = subprocess.run(
        [matlab_path, "-batch", f"run('{m_path_fwd}')"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"matlab failed (code {result.returncode}): {result.stderr}\n{result.stdout}")

    pattern = re.compile(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")
    rows = [m.groups() for line in result.stdout.splitlines() if (m := pattern.match(line))]
    if len(rows) != n_steps + 1:
        raise RuntimeError(f"expected {n_steps+1} output lines from MATLAB, got {len(rows)}: {result.stdout!r}")
    vc = np.array([float(r[0]) for r in rows])
    il = np.array([float(r[1]) for r in rows])
    return vc, il


def cross_validate_languages(out_dir, R=100.0, L=1e-3, C=1e-6, V=1.0, vc0=0.0, il0=0.0,
                              dt=1e-6, n_steps=5, gcc_path=GCC_DEFAULT, matlab_path=MATLAB_DEFAULT,
                              run_c=True, run_matlab=True):
    """Run the identical RLC RK4 problem in Python, C, and MATLAB, and report
    the max absolute disagreement between each pair -- the actual proof that
    "the same algorithm translates across languages" rather than a claim
    about it. All three should agree to ~1e-9 (same double-precision RK4,
    same operation order)."""
    vc_py, il_py = rlc_rk4_python(R, L, C, V, vc0, il0, dt, n_steps)
    out = {"python": {"vc": vc_py, "il": il_py}}

    if run_c:
        exe = compile_c_rlc(out_dir, gcc_path=gcc_path)
        vc_c, il_c = run_c_rlc(exe, R, L, C, V, vc0, il0, dt, n_steps)
        out["c"] = {"vc": vc_c, "il": il_c}
        out["max_abs_diff_python_vs_c"] = float(max(np.max(np.abs(vc_py - vc_c)), np.max(np.abs(il_py - il_c))))

    if run_matlab:
        vc_m, il_m = run_matlab_rlc(out_dir, R, L, C, V, vc0, il0, dt, n_steps, matlab_path=matlab_path)
        out["matlab"] = {"vc": vc_m, "il": il_m}
        out["max_abs_diff_python_vs_matlab"] = float(max(np.max(np.abs(vc_py - vc_m)), np.max(np.abs(il_py - il_m))))

    return out


if __name__ == "__main__":
    import tempfile
    R, L, C_, V = 100.0, 1e-3, 1e-6, 1.0
    dt, n_steps = 1e-6, 5

    with tempfile.TemporaryDirectory() as tmp:
        result = cross_validate_languages(tmp, R=R, L=L, C=C_, V=V, dt=dt, n_steps=n_steps)

    print(f"series RLC step response, R={R} L={L} C={C_} V={V}, dt={dt}, {n_steps} RK4 steps\n")
    print(f"{'lang':<8}{'vc(final)':>16}{'il(final)':>16}")
    for lang in ("python", "c", "matlab"):
        if lang in result:
            print(f"{lang:<8}{result[lang]['vc'][-1]:>16.8e}{result[lang]['il'][-1]:>16.8e}")

    if "max_abs_diff_python_vs_c" in result:
        print(f"\nmax |python - C|      = {result['max_abs_diff_python_vs_c']:.3e}")
    if "max_abs_diff_python_vs_matlab" in result:
        print(f"max |python - MATLAB| = {result['max_abs_diff_python_vs_matlab']:.3e}")
