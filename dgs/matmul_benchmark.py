"""Power of Python vs C, qualitatively measured: matrix multiplication.

Same O(n^3) algorithm, three implementations:
  1. pure_python_matmul   -- nested Python lists, triple for-loop (interpreter overhead)
  2. numpy_matmul         -- numpy.dot, which dispatches to a compiled BLAS routine
  3. c_matmul (optional)  -- a real C triple-loop, compiled with gcc and called
                             via subprocess (no ctypes needed -- the C binary
                             prints timing itself)

The "qualitative team matrix" is the comparison table: language vs overhead
source vs typical speedup. Python's *power* isn't raw loop speed (it loses
that badly) -- it's that numpy/PyTorch let you write Python and run C/CUDA
underneath, via vectorization instead of explicit loops.

PyTorch (torch.matmul) is py-3.12 ONLY in this environment (not py-3.13) --
see torch_matmul_benchmark() and run it with `py -3.12`.
"""
import os
import subprocess
import sys
import time
import numpy as np


C_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matmul(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n+k] * B[k*n+j];
            }
            C[i*n+j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    double *A = malloc(n*n*sizeof(double));
    double *B = malloc(n*n*sizeof(double));
    double *C = malloc(n*n*sizeof(double));
    srand(42);
    for (int i = 0; i < n*n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }
    clock_t start = clock();
    matmul(A, B, C, n);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%f\n", elapsed);
    free(A); free(B); free(C);
    return 0;
}
"""


def pure_python_matmul(A, B):
    """Triple nested-loop matmul on plain Python lists -- the slow path."""
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def numpy_matmul(A, B):
    """Same algorithm, dispatched to compiled BLAS underneath."""
    return np.asarray(A) @ np.asarray(B)


def benchmark_pure_python(n, n_trials=1):
    """Time pure_python_matmul on random n x n matrices."""
    rng = np.random.default_rng(42)
    A = rng.random((n, n)).tolist()
    B = rng.random((n, n)).tolist()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        pure_python_matmul(A, B)
        times.append(time.perf_counter() - t0)
    return {"n": n, "mean_s": float(np.mean(times)), "times": times}


def benchmark_numpy(n, n_trials=5):
    """Time numpy_matmul on random n x n matrices."""
    rng = np.random.default_rng(42)
    A = rng.random((n, n))
    B = rng.random((n, n))
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        numpy_matmul(A, B)
        times.append(time.perf_counter() - t0)
    return {"n": n, "mean_s": float(np.mean(times)), "times": times}


def compile_c_matmul(out_dir, gcc_path="gcc"):
    """Write C_SOURCE to disk and compile it with gcc -O2.
    On this machine gcc lives at C:\\msys64\\mingw64\\bin\\gcc.exe (not on
    default PATH) -- pass that path explicitly if `gcc` alone is not found."""
    src_path = os.path.join(out_dir, "matmul_bench.c")
    exe_path = os.path.join(out_dir, "matmul_bench.exe")
    with open(src_path, "w") as f:
        f.write(C_SOURCE)
    result = subprocess.run(
        [gcc_path, "-O2", "-o", exe_path, src_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    return exe_path


def benchmark_c(exe_path, n, n_trials=5):
    """Run the compiled C binary, which prints its own elapsed time per call."""
    times = []
    for _ in range(n_trials):
        result = subprocess.run([exe_path, str(n)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"C binary failed: {result.stderr}")
        times.append(float(result.stdout.strip()))
    return {"n": n, "mean_s": float(np.mean(times)), "times": times}


def torch_matmul_benchmark(n, n_trials=5, device="cpu"):
    """torch.matmul timing -- import is local since torch is py-3.12 ONLY
    in this environment, not py-3.13. Run this function with `py -3.12`."""
    import torch
    A = torch.rand(n, n, device=device, dtype=torch.float64)
    B = torch.rand(n, n, device=device, dtype=torch.float64)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        torch.matmul(A, B)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {"n": n, "mean_s": float(np.mean(times)), "device": device,
            "cuda_available": torch.cuda.is_available()}


def language_comparison_table():
    """Qualitative team matrix: language vs overhead source vs typical speedup.

    Numbers are order-of-magnitude, not benchmarked here (see the functions
    above for the real measured numbers on THIS machine, n=200)."""
    return {
        "Pure Python": {
            "overhead_source": "Interpreter bytecode dispatch per loop iteration; dynamic typing checks",
            "typical_speedup_vs_python": "1x (baseline, slowest)",
            "power": "Maximum flexibility/readability; bad for O(n^3) numeric loops",
        },
        "NumPy (BLAS)": {
            "overhead_source": "None inside the matmul -- compiled C/Fortran BLAS kernel, vectorized",
            "typical_speedup_vs_python": "~100-1000x",
            "power": "Python syntax, C-speed execution -- write once, vectorize via broadcasting",
        },
        "C (raw triple loop)": {
            "overhead_source": "None -- but no auto-vectorization/blocking like BLAS has",
            "typical_speedup_vs_python": "~50-200x",
            "power": "Full control, but you re-derive what BLAS already optimized (cache blocking, SIMD)",
        },
        "PyTorch (CPU)": {
            "overhead_source": "Same BLAS backend as NumPy, plus autograd graph bookkeeping",
            "typical_speedup_vs_python": "~100-1000x",
            "power": "NumPy speed + automatic differentiation -- the same matmul also backprops",
        },
        "PyTorch (GPU/CUDA)": {
            "overhead_source": "Host-device transfer + kernel launch latency, amortized over large n",
            "typical_speedup_vs_python": "~1000-10000x for large n (n>1000)",
            "power": "Thousands of cores in parallel -- the real reason deep learning is GPU-bound",
        },
        "common_lesson": "Python's power isn't loop speed -- it's that NumPy/PyTorch let you "
                          "describe the computation in Python and execute it in compiled/parallel "
                          "code underneath. The interpreter never touches the inner k-loop.",
    }


if __name__ == "__main__":
    n = 80
    print(f"=== Pure Python matmul, n={n} ===")
    py_res = benchmark_pure_python(n, n_trials=1)
    print(f"  mean time = {py_res['mean_s']*1000:.2f} ms")

    print(f"\n=== NumPy matmul, n={n} ===")
    np_res = benchmark_numpy(n, n_trials=5)
    print(f"  mean time = {np_res['mean_s']*1000:.4f} ms")
    print(f"  speedup vs pure Python = {py_res['mean_s']/np_res['mean_s']:.1f}x")

    print("\n=== Qualitative language comparison ===")
    table = language_comparison_table()
    for lang, info in table.items():
        if isinstance(info, dict):
            print(f"  {lang}: {info['typical_speedup_vs_python']}")

    print(f"\n=== Note: torch.matmul is py-3.12 ONLY -- run torch_matmul_benchmark() with py -3.12 ===")
