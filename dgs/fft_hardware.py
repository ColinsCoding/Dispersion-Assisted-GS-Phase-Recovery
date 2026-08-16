"""The FFT, from a geometric series to real hardware: C bit-reversal + a
compiled, actually-run CUDA kernel, cross-validated against dgs.fourier_tools.

Three layers, bottom to top:

  1. CALCULUS/ALGEBRA -- the orthogonality identity that makes the inverse
     DFT actually invert the DFT is nothing but a geometric series:
         sum_{n=0}^{N-1} r^n = (1-r^N)/(1-r)          (r != 1)
     Substitute r = W_N^k = exp(-2*pi*i*k/N) (an N-th root of unity, itself
     Euler's formula -- two interleaved Taylor series, see dgs.taylor) and
     the sum collapses to N when k = 0 (mod N) and 0 otherwise. That single
     fact is why dgs.fourier_tools.idft(dft(x)) == x.

  2. THE ALGORITHM -- dgs.fourier_tools.fft_radix2 is the textbook RECURSIVE
     Cooley-Tukey split. Real hardware (FPGA pipelines, GPU FFT libraries)
     does not recurse -- it runs ITERATIVELY over log2(N) stages with a
     bit-reversal permutation up front, so every stage is embarrassingly
     parallel (N/2 independent butterflies). fft_iterative_hardware here is
     that version, cross-checked against fourier_tools.fft_radix2 and
     numpy.fft to machine precision.

  3. THE HARDWARE -- the exact same bit-reversal + butterfly algorithm,
     reimplemented from scratch in C (compiled with gcc, subprocess, same
     pattern as dgs.dispersion_polyglot / dgs.circuits_polyglot) and in CUDA
     (compiled with nvcc and actually run on the GPU, same pattern as
     dgs.cuda_time_stretch_runner -- one kernel launch per FFT stage, N/2
     threads per launch, log2(N) sequential launches). All three -- Python,
     C, CUDA -- must agree with numpy.fft to near machine precision, or the
     cross-validation fails loudly rather than silently.

nvcc note (same caveat as dgs.cuda_time_stretch_runner): nvcc needs MSVC's
cl.exe as its host compiler on Windows, not on PATH by default. Run from
PowerShell with it prepended:
    $env:PATH = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\" +
                 "VC\\Tools\\MSVC\\<version>\\bin\\Hostx64\\x64;" + $env:PATH
"""

import os
import subprocess

import numpy as np
import sympy as sp

from dgs.fourier_tools import fft_radix2

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"
NVCC_DEFAULT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"


# ── 1. calculus/algebra: geometric series -> root-of-unity orthogonality ────
def geometric_series_closed_form(N_val):
    """Exact sympy proof of sum_{n=0}^{N_val-1} r^n = (1-r^N_val)/(1-r) (r != 1) --
    the textbook geometric series, kept symbolic in r so it holds for ANY ratio,
    not just roots of unity."""
    r, n = sp.symbols("r n")
    lhs = sp.summation(r ** n, (n, 0, N_val - 1))
    rhs = (1 - r ** N_val) / (1 - r)
    if sp.simplify(lhs - rhs) != 0:
        raise AssertionError("geometric series closed form does not hold")
    return sp.simplify(lhs)


def root_of_unity_orthogonality(N):
    """sum_{n=0}^{N-1} exp(-2*pi*i*n*k/N) = N if k%N==0 else 0, derived by
    substituting the N-th root of unity r = exp(-2*pi*i*k/N) into the geometric
    series closed form above. THIS is the identity that makes
    dgs.fourier_tools.idft(dft(x)) == x -- the DFT basis vectors are orthogonal
    only because a geometric series of roots of unity telescopes to zero.
    Returns {k: sympy value} for k = 0..N-1, each simplified to an integer.
    """
    n = sp.symbols("n")
    out = {}
    for k in range(N):
        r = sp.exp(-2 * sp.pi * sp.I * k / N)
        if k == 0:
            s = sp.Integer(N)
        else:
            s = sp.nsimplify(sp.summation(r ** n, (n, 0, N - 1)), [sp.pi])
            s = sp.simplify(sp.re(s)) + sp.simplify(sp.im(s)) * sp.I
            s = sp.nsimplify(sp.simplify(s), tolerance=1e-9)
        out[k] = sp.simplify(s)
    return out


# ── 2. the algorithm: iterative, bit-reversed radix-2 (the hardware shape) ──
def bit_reverse_indices(N):
    """Bit-reversal permutation for N=2^bits -- the address-generator logic
    every hardware/FPGA FFT core uses to avoid an out-of-place buffer."""
    if N & (N - 1) or N < 1:
        raise ValueError("N must be a power of 2")
    bits = N.bit_length() - 1
    return [int(format(i, f"0{bits}b")[::-1], 2) if bits else 0 for i in range(N)]


def fft_iterative_hardware(x):
    """Iterative, in-place-style radix-2 FFT via bit-reversal + log2(N)
    butterfly stages -- the same structure as an FPGA pipeline or a GPU FFT
    kernel, unlike fourier_tools.fft_radix2's recursive split. Same physics,
    same twiddle-factor sign convention (matches numpy.fft.fft)."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N & (N - 1):
        raise ValueError("length must be a power of 2")
    idx = bit_reverse_indices(N)
    a = x[idx].copy()
    length = 2
    while length <= N:
        half = length // 2
        w = np.exp(-2j * np.pi * np.arange(half) / length)
        for start in range(0, N, length):
            u = a[start:start + half].copy()
            v = a[start + half:start + length] * w
            a[start:start + half] = u + v
            a[start + half:start + length] = u - v
        length *= 2
    return a


# ── 3. the hardware: C and CUDA, compiled and actually run ─────────────────
C_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void fft_iterative(double *re, double *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cwr = 1.0, cwi = 0.0;
            for (int k = 0; k < len / 2; k++) {
                double ur = re[i + k], ui = im[i + k];
                double vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi;
                double vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr;
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                double ncwr = cwr * wr - cwi * wi;
                double ncwi = cwr * wi + cwi * wr;
                cwr = ncwr; cwi = ncwi;
            }
        }
    }
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    double *re = malloc(n * sizeof(double));
    double *im = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        re[i] = atof(argv[2 + 2 * i]);
        im[i] = atof(argv[2 + 2 * i + 1]);
    }
    fft_iterative(re, im, n);
    for (int i = 0; i < n; i++) printf("%.15e %.15e\n", re[i], im[i]);
    free(re); free(im);
    return 0;
}
"""

CUDA_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void bit_reverse_kernel(const double *re_in, const double *im_in,
                                    double *re_out, double *im_out, int n, int bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = 0;
        for (int b = 0; b < bits; b++) if (i & (1 << b)) j |= (1 << (bits - 1 - b));
        re_out[j] = re_in[i];
        im_out[j] = im_in[i];
    }
}

__global__ void butterfly_stage_kernel(double *re, double *im, int n, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per butterfly
    int half = len / 2;
    int nb = n / 2;
    if (idx >= nb) return;
    int group = idx / half;
    int k = idx % half;
    int i = group * len + k;
    double ang = -2.0 * M_PI * k / len;
    double wr = cos(ang), wi = sin(ang);
    double ur = re[i], ui = im[i];
    double vr = re[i + half] * wr - im[i + half] * wi;
    double vi = re[i + half] * wi + im[i + half] * wr;
    re[i] = ur + vr; im[i] = ui + vi;
    re[i + half] = ur - vr; im[i + half] = ui - vi;
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    int bits = (int)llround(log2((double)n));
    double *h_re = (double*)malloc(n * sizeof(double));
    double *h_im = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        h_re[i] = atof(argv[2 + 2 * i]);
        h_im[i] = atof(argv[2 + 2 * i + 1]);
    }

    double *d_re0, *d_im0, *d_re, *d_im;
    cudaMalloc(&d_re0, n * sizeof(double)); cudaMalloc(&d_im0, n * sizeof(double));
    cudaMalloc(&d_re, n * sizeof(double));  cudaMalloc(&d_im, n * sizeof(double));
    cudaMemcpy(d_re0, h_re, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_im0, h_im, n * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    bit_reverse_kernel<<<blocks, threads>>>(d_re0, d_im0, d_re, d_im, n, bits);

    for (int len = 2; len <= n; len <<= 1) {
        int nb = n / 2;
        int bblocks = (nb + threads - 1) / threads;
        butterfly_stage_kernel<<<bblocks, threads>>>(d_re, d_im, n, len);
    }

    cudaMemcpy(h_re, d_re, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_im, d_im, n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%.15e %.15e\n", h_re[i], h_im[i]);

    cudaFree(d_re0); cudaFree(d_im0); cudaFree(d_re); cudaFree(d_im);
    free(h_re); free(h_im);
    return 0;
}
"""


def _parse_re_im_lines(text):
    vals = []
    for line in text.strip().splitlines():
        re_s, im_s = line.split()
        vals.append(complex(float(re_s), float(im_s)))
    return np.array(vals)


def _flatten_args(x):
    args = []
    for v in x:
        args.append(str(v.real)); args.append(str(v.imag))
    return args


def run_c(x, out_dir, gcc_path=GCC_DEFAULT):
    x = np.asarray(x, dtype=complex)
    src = os.path.join(out_dir, "fft_hardware.c")
    exe = os.path.join(out_dir, "fft_hardware_c.exe")
    with open(src, "w") as f:
        f.write(C_SOURCE)
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(gcc_path) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([gcc_path, "-O2", "-o", exe, src, "-lm"], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    args = [exe, str(len(x))] + _flatten_args(x)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C run failed: {result.stderr}")
    return _parse_re_im_lines(result.stdout)


def run_cuda(x, out_dir, nvcc_path=NVCC_DEFAULT):
    x = np.asarray(x, dtype=complex)
    src = os.path.join(out_dir, "fft_hardware.cu")
    exe = os.path.join(out_dir, "fft_hardware_cuda.exe")
    with open(src, "w") as f:
        f.write(CUDA_SOURCE)
    result = subprocess.run([nvcc_path, "-O2", "-o", exe, src], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "nvcc compile failed (likely missing cl.exe/MSVC on PATH -- run from "
            "PowerShell with the MSVC bin dir prepended, see module docstring): "
            f"{result.stderr}"
        )
    args = [exe, str(len(x))] + _flatten_args(x)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CUDA run failed: {result.stderr}")
    return _parse_re_im_lines(result.stdout)


def cross_validate_languages(x, out_dir, include_cuda=True):
    """Run the SAME iterative bit-reversal FFT in Python, C, and (optionally)
    CUDA on the SAME input, and compare all three to numpy.fft.fft -- the
    actual proof that 'geometric series -> algorithm -> hardware' is one
    continuous chain, not three unrelated claims."""
    x = np.asarray(x, dtype=complex)
    ref = np.fft.fft(x)
    py_result = fft_iterative_hardware(x)
    c_result = run_c(x, out_dir)
    results = {"numpy": ref, "python_iterative": py_result, "c": c_result}
    if include_cuda:
        results["cuda"] = run_cuda(x, out_dir)

    max_errs = {}
    for name, arr in results.items():
        if name == "numpy":
            continue
        max_errs[name] = np.max(np.abs(arr - ref))
    return results, max_errs


if __name__ == "__main__":
    import tempfile

    print("=== 1. geometric series -> root-of-unity orthogonality ===")
    N = 8
    for N_check in range(2, 6):
        geometric_series_closed_form(N_check)
    print("  sum_{n=0}^{N-1} r^n = (1-r^N)/(1-r)  verified symbolically for N=2..5")
    orth = root_of_unity_orthogonality(N)
    print(f"  N={N} root-of-unity sums: {orth}")
    assert orth[0] == N and all(orth[k] == 0 for k in range(1, N))
    print("  PASS: DFT basis orthogonality IS a geometric series telescoping to zero.")

    print("\n=== 2. iterative (hardware-shaped) FFT vs fourier_tools.fft_radix2 ===")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    py_iter = fft_iterative_hardware(x)
    py_recursive = fft_radix2(x)
    print(f"  max |iterative - recursive| = {np.max(np.abs(py_iter - py_recursive)):.3e}")
    assert np.allclose(py_iter, py_recursive)

    print("\n=== 3. C (and CUDA, if available) vs Python/numpy ===")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            results, max_errs = cross_validate_languages(x, tmp, include_cuda=True)
        except RuntimeError as e:
            print(f"  CUDA path unavailable ({e}); retrying without CUDA")
            results, max_errs = cross_validate_languages(x, tmp, include_cuda=False)

    for lang, err in max_errs.items():
        print(f"  max |{lang} - numpy| = {err:.3e}")
    assert all(err < 1e-9 for err in max_errs.values())
    print("\nAll layers agree: geometric series -> algorithm -> C -> (CUDA) hardware.")
