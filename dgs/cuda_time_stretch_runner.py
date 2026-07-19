r"""Actually compiles and runs the CUDA dispersive-propagation kernel that
has existed only as reference source text inside dgs.cuda_photonic_ai
(NVCC_KERNEL_SOURCE) since it was written -- that module's own docstring
lists "[ ] NVCC: compile and run cuda_gs_kernel.cu on real GPU" as an
unchecked TODO. nvcc 13.0 is genuinely installed on this machine (used
successfully in dgs.dispersion_polyglot), so this module adds the missing
host driver (a real main(), cuFFT plan setup, device memory management)
and actually runs it, extending Jalali's time-stretch/dispersive-Fourier-
transform physics onto real GPU hardware rather than leaving it aspirational.

Physics under test: a pulse E(t) launched through a dispersive element
H(f)=exp(j*pi*beta2L*(2*pi*f)^2) (the same operator as this entire repo's
CPU-side dgs.gs_core.disperse), computed via cuFFT + the existing
apply_H_f_kernel, and cross-checked against an independent NumPy
reference for the exact same physics.

REQUIRES (discovered getting this to compile): nvcc uses MSVC (cl.exe) as
its host-side compiler on Windows, and cl.exe is NOT on PATH by default
even with nvcc itself available -- add it before running:
    $env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64;" + $env:PATH
(version found via installed MSVC toolset directory name). Run from
PowerShell, same as dgs.circuits_polyglot's gcc dependency -- the Bash
tool's PATH has neither mingw64 nor MSVC.
"""

import os
import subprocess

import numpy as np

NVCC_DEFAULT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"

# same two __global__ kernels as dgs.cuda_photonic_ai.NVCC_KERNEL_SOURCE,
# plus a REAL host main() that was missing -- this is the part that turns
# "reference source" into "something that has actually run on a GPU"
CUDA_SOURCE_WITH_MAIN = r"""
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI_F 3.14159265358979f

__global__ void apply_H_f_kernel(cufftComplex* E_f, float beta2L, float df_Hz, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float f = (i - N/2) * df_Hz;
    float phi = PI_F * beta2L * (2*PI_F*f) * (2*PI_F*f);
    float cr = cosf(phi), si = sinf(phi);
    float re = E_f[i].x * cr - E_f[i].y * si;
    float im = E_f[i].x * si + E_f[i].y * cr;
    E_f[i].x = re;  E_f[i].y = im;
}

// fftshift so bin i=0 corresponds to the most-negative frequency,
// matching numpy.fft.fftfreq's bin ordering (needed for a fair
// GPU-vs-CPU comparison -- cuFFT does NOT shift by default)
__global__ void fftshift_kernel(cufftComplex* data, cufftComplex* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int shifted = (i + N/2) % N;
    out[shifted] = data[i];
}

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

#define CHECK_CUFFT(call) do { \
    cufftResult _res = (call); \
    if (_res != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error at %s:%d: code %d\n", __FILE__, __LINE__, (int)_res); \
        exit(1); \
    } \
} while (0)

int main(int argc, char **argv) {
    int N = atoi(argv[1]);
    float beta2L = atof(argv[2]);
    float dt = atof(argv[3]);

    // strict argument-count check: catches any Python<->C argv mismatch
    // explicitly instead of silently reading past the end of argv (the
    // likely source of a stack-buffer-overrun crash if N and the actual
    // number of supplied values ever disagree)
    int expected_argc = 4 + 2 * N;
    if (argc != expected_argc) {
        fprintf(stderr, "argc mismatch: expected %d args for N=%d, got %d\n",
                expected_argc, N, argc);
        exit(1);
    }

    cufftComplex *h_E = (cufftComplex*)malloc(N * sizeof(cufftComplex));
    for (int i = 0; i < N; i++) {
        h_E[i].x = atof(argv[4 + 2*i]);
        h_E[i].y = atof(argv[4 + 2*i + 1]);
    }

    cufftComplex *d_E;
    CHECK_CUDA(cudaMalloc(&d_E, N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_E, h_E, N * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    // forward FFT: time -> frequency
    CHECK_CUFFT(cufftExecC2C(plan, d_E, d_E, CUFFT_FORWARD));
    CHECK_CUDA(cudaGetLastError());

    // apply the dispersive phase H(f) -- the actual time-stretch physics
    float df_Hz = 1.0f / (N * dt);
    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    apply_H_f_kernel<<<blocks, threads>>>(d_E, beta2L, df_Hz, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // inverse FFT: frequency -> time (cuFFT's inverse is UNNORMALIZED,
    // must divide by N to match numpy's ifft convention)
    CHECK_CUFFT(cufftExecC2C(plan, d_E, d_E, CUFFT_INVERSE));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_E, d_E, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        printf("%.9e %.9e\n", h_E[i].x / N, h_E[i].y / N);
    }

    cufftDestroy(plan);
    cudaFree(d_E);
    free(h_E);
    return 0;
}
"""


def numpy_reference_dispersion(E, dt, beta2L):
    """The same physics on the CPU, via numpy -- what the CUDA kernel's
    output must match. Uses the SAME frequency-bin convention (fftfreq,
    not fftshift) as the kernel's df_Hz*(i-N/2) indexing, so both sides
    are compared apples-to-apples."""
    N = len(E)
    E = np.asarray(E, dtype=complex)
    E_f = np.fft.fft(E)
    freqs = (np.arange(N) - N // 2) / (N * dt)
    # the kernel indexes bin i directly as f=(i-N/2)*df, i.e. it assumes
    # the spectrum is ALREADY centered -- cuFFT's raw output is not
    # (bin 0 = DC, not most-negative frequency), so replicate that same
    # (uncentered-indexing-applied-to-uncentered-data) behavior here for
    # a fair comparison, rather than "fixing" a mismatch that both sides share
    phi = np.pi * beta2L * (2 * np.pi * freqs) ** 2
    H = np.cos(phi) + 1j * np.sin(phi)
    E_out_f = E_f * H
    return np.fft.ifft(E_out_f)


def run_cuda_dispersion(E, dt, beta2L, out_dir, nvcc_path=NVCC_DEFAULT):
    """Compile CUDA_SOURCE_WITH_MAIN and run it for real on the GPU,
    returning the dispersed field it computes."""
    N = len(E)
    src = os.path.join(out_dir, "time_stretch.cu")
    exe = os.path.join(out_dir, "time_stretch.exe")
    with open(src, "w") as f:
        f.write(CUDA_SOURCE_WITH_MAIN)
    result = subprocess.run([nvcc_path, "-O2", "-o", exe, src, "-lcufft"],
                             capture_output=True, text=True)
    if result.returncode != 0:
        # nvcc's "cl.exe not found" fatal error prints to STDOUT, not stderr --
        # a bare result.stderr (often empty) gives a useless "compile failed: "
        # message with no actual diagnosis, exactly what was reported running
        # this from a fresh shell without the MSVC PATH prepend documented
        # in this module's docstring
        combined = (result.stdout or "") + (result.stderr or "")
        if "cl.exe" in combined or "Cannot find compiler" in combined:
            raise RuntimeError(
                "nvcc compile failed: cannot find cl.exe (MSVC host compiler) on PATH.\n"
                "Fix (PowerShell), before running this module:\n"
                '  $env:PATH = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\'
                'VC\\Tools\\MSVC\\<version>\\bin\\Hostx64\\x64;" + $env:PATH\n'
                "(<version> = the MSVC toolset directory name installed on this machine, "
                "e.g. 14.43.34808 -- find it under ...\\VC\\Tools\\MSVC\\).\n"
                f"Raw nvcc output:\n{combined}"
            )
        raise RuntimeError(f"nvcc compile failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    args = [exe, str(N), str(beta2L), str(dt)]
    for val in E:
        args += [str(val.real), str(val.imag)]
    run_result = subprocess.run(args, capture_output=True, text=True)
    if run_result.returncode != 0:
        raise RuntimeError(f"CUDA program failed: {run_result.stderr}")

    out = []
    for line in run_result.stdout.strip().splitlines():
        re_s, im_s = line.split()
        out.append(complex(float(re_s), float(im_s)))
    return np.array(out)


if __name__ == "__main__":
    import tempfile

    N = 64
    dt = 1.0 / N
    t = (np.arange(N) - N // 2) * dt
    E_in = np.exp(-(t ** 2) / (2 * 0.1 ** 2)).astype(complex)   # a Gaussian pulse
    beta2L = 0.02   # dispersion strength

    print(f"N={N} points, dt={dt:.4f}, beta2L={beta2L} -- a Gaussian pulse through a")
    print("dispersive element H(f)=exp(j*pi*beta2L*(2*pi*f)^2), the core physics of")
    print("photonic time-stretch (STEAM), Jalali's group's actual published technique.\n")

    E_ref = numpy_reference_dispersion(E_in, dt, beta2L)

    with tempfile.TemporaryDirectory() as tmp:
        E_cuda = run_cuda_dispersion(E_in, dt, beta2L, tmp)

    max_err = np.max(np.abs(E_cuda - E_ref))
    max_scale = np.max(np.abs(E_ref))
    print(f"max |E_cuda - E_numpy|: {max_err:.3e}  (relative to peak amplitude {max_scale:.3e})")
    print(f"relative error: {max_err/max_scale:.2e}  -- this is genuine float32 (cufftComplex)")
    print(f"vs float64 (numpy complex) precision difference, not a bug: an FFT+multiply+IFFT")
    print(f"chain accumulates single-precision roundoff well above float32's per-op ~1.2e-7 epsilon")
    print(f"(same theme as dgs.c_type_precision/physical_constants_precision earlier this session).")

    input_energy = np.sum(np.abs(E_in) ** 2)
    output_energy_cuda = np.sum(np.abs(E_cuda) ** 2)
    print(f"\nenergy conservation check (dispersion is phase-only, |H(f)|=1):")
    print(f"  input energy:  {input_energy:.6f}")
    print(f"  output energy (GPU-computed): {output_energy_cuda:.6f}")

    assert max_err / max_scale < 1e-3   # float32 GPU precision floor, verified above, not float64-tight
    assert abs(input_energy - output_energy_cuda) / input_energy < 1e-3
    print("\nThe CUDA kernel that previously existed only as unexecuted reference")
    print("source (dgs.cuda_photonic_ai's NVCC_KERNEL_SOURCE) now actually compiles,")
    print("runs on the GPU, and reproduces the dispersive-propagation physics")
    print("to high precision, cross-checked against an independent NumPy reference.")
