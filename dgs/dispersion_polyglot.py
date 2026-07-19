"""The single most important computation in this repo -- the dispersion
phase operator H(f) = exp(j*pi*D*f^2) -- computed independently in C,
Java, CUDA, and Python, and cross-validated against each other. Same
pattern as dgs.circuits_polyglot (same physics, different languages,
proven to agree rather than assumed), extended to a 4th language (Java)
and a GPU kernel (CUDA) this repo hasn't covered together before.

Motivation beyond the exercise itself: a "lab on a chip" / OUSD-aligned
technology-transition story benefits from showing the core algorithm
isn't tied to one runtime -- CPU (C), JVM (Java), GPU (CUDA), and a
numpy reference (Python) all compute the IDENTICAL physics, which is a
real (if modest) hardware-portability argument.
"""

import os
import re
import subprocess

import numpy as np

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"
JAVAC_DEFAULT = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.9.10-hotspot\bin\javac.exe"
JAVA_DEFAULT = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.9.10-hotspot\bin\java.exe"
NVCC_DEFAULT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"


def dispersion_phase_python(freqs, D):
    """Reference: H(f) = exp(j*pi*D*f^2), the repo's central operator."""
    freqs = np.asarray(freqs, dtype=float)
    phi = np.pi * D * freqs ** 2
    return np.cos(phi) + 1j * np.sin(phi)


C_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    double D = atof(argv[1]);
    int n = argc - 2;
    for (int i = 0; i < n; i++) {
        double f = atof(argv[i + 2]);
        double phi = M_PI * D * f * f;
        printf("%.15e %.15e\n", cos(phi), sin(phi));
    }
    return 0;
}
"""

JAVA_SOURCE = r"""
public class DispersionPhase {
    public static void main(String[] args) {
        double D = Double.parseDouble(args[0]);
        for (int i = 1; i < args.length; i++) {
            double f = Double.parseDouble(args[i]);
            double phi = Math.PI * D * f * f;
            System.out.printf("%.15e %.15e%n", Math.cos(phi), Math.sin(phi));
        }
    }
}
"""

CUDA_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void dispersion_kernel(double D, double *freqs, double *re, double *im, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double phi = M_PI * D * freqs[i] * freqs[i];
        re[i] = cos(phi);
        im[i] = sin(phi);
    }
}

int main(int argc, char **argv) {
    double D = atof(argv[1]);
    int n = argc - 2;
    double *h_freqs = (double*)malloc(n * sizeof(double));
    double *h_re = (double*)malloc(n * sizeof(double));
    double *h_im = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) h_freqs[i] = atof(argv[i + 2]);

    double *d_freqs, *d_re, *d_im;
    cudaMalloc(&d_freqs, n * sizeof(double));
    cudaMalloc(&d_re, n * sizeof(double));
    cudaMalloc(&d_im, n * sizeof(double));
    cudaMemcpy(d_freqs, h_freqs, n * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    dispersion_kernel<<<blocks, threads>>>(D, d_freqs, d_re, d_im, n);
    cudaMemcpy(h_re, d_re, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_im, d_im, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.15e %.15e\n", h_re[i], h_im[i]);

    cudaFree(d_freqs); cudaFree(d_re); cudaFree(d_im);
    free(h_freqs); free(h_re); free(h_im);
    return 0;
}
"""


def _parse_re_im_lines(text):
    vals = []
    for line in text.strip().splitlines():
        re_s, im_s = line.split()
        vals.append(complex(float(re_s), float(im_s)))
    return np.array(vals)


def run_c(freqs, D, out_dir, gcc_path=GCC_DEFAULT):
    src = os.path.join(out_dir, "dispersion.c")
    exe = os.path.join(out_dir, "dispersion_c.exe")
    with open(src, "w") as f:
        f.write(C_SOURCE)
    result = subprocess.run([gcc_path, "-O2", "-o", exe, src, "-lm"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    args = [exe, str(D)] + [str(f) for f in freqs]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C run failed: {result.stderr}")
    return _parse_re_im_lines(result.stdout)


def run_java(freqs, D, out_dir, javac_path=JAVAC_DEFAULT, java_path=JAVA_DEFAULT):
    src = os.path.join(out_dir, "DispersionPhase.java")
    with open(src, "w") as f:
        f.write(JAVA_SOURCE)
    result = subprocess.run([javac_path, "-d", out_dir, src], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"javac compile failed: {result.stderr}")
    args = [java_path, "-cp", out_dir, "DispersionPhase", str(D)] + [str(f) for f in freqs]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"java run failed: {result.stderr}")
    return _parse_re_im_lines(result.stdout)


def run_cuda(freqs, D, out_dir, nvcc_path=NVCC_DEFAULT):
    src = os.path.join(out_dir, "dispersion.cu")
    exe = os.path.join(out_dir, "dispersion_cuda.exe")
    with open(src, "w") as f:
        f.write(CUDA_SOURCE)
    result = subprocess.run([nvcc_path, "-O2", "-o", exe, src], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc compile failed: {result.stderr}")
    args = [exe, str(D)] + [str(f) for f in freqs]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CUDA run failed: {result.stderr}")
    return _parse_re_im_lines(result.stdout)


def cross_validate_languages(freqs, D, out_dir, include_cuda=True):
    """Run the SAME dispersion-phase computation in Python, C, Java, and
    (optionally) CUDA, and return each result plus their max pairwise
    disagreement -- the actual proof of language/hardware portability,
    not an assumption."""
    py_result = dispersion_phase_python(freqs, D)
    c_result = run_c(freqs, D, out_dir)
    java_result = run_java(freqs, D, out_dir)
    results = {"python": py_result, "c": c_result, "java": java_result}
    if include_cuda:
        results["cuda"] = run_cuda(freqs, D, out_dir)

    max_errs = {}
    for name, arr in results.items():
        if name == "python":
            continue
        max_errs[name] = np.max(np.abs(arr - py_result))
    return results, max_errs


if __name__ == "__main__":
    import tempfile
    freqs = np.linspace(-2.0, 2.0, 9)
    D = -5000.0

    print(f"D = {D}, {len(freqs)} frequency samples")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            results, max_errs = cross_validate_languages(freqs, D, tmp, include_cuda=True)
        except RuntimeError as e:
            print(f"CUDA path failed ({e}); retrying without CUDA")
            results, max_errs = cross_validate_languages(freqs, D, tmp, include_cuda=False)

    for lang, err in max_errs.items():
        print(f"  max |H_{lang} - H_python| = {err:.3e}")

    print("\nSample values (first 3 frequencies):")
    for i in range(3):
        print(f"  f={freqs[i]:+.3f}: python={results['python'][i]:.6f}  "
              f"c={results['c'][i]:.6f}  java={results['java'][i]:.6f}"
              + (f"  cuda={results['cuda'][i]:.6f}" if "cuda" in results else ""))

    assert all(err < 1e-9 for err in max_errs.values())
    print("\nAll languages agree to near machine precision on the exact same physics.")
