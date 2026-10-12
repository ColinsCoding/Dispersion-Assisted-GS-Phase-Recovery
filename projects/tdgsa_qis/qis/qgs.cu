// qgs.cu -- Quantum Gerchberg-Saxton state recovery with shot noise, batched on the GPU
//
// Same physics as qgs.cpp, but every trial runs at once:
//   * shot sampling: one CUDA thread per trial, cuRAND Philox, inverse-CDF lookup
//   * GS: one batched cuFFT plan (batch = trials) of 64-point transforms
//   * fidelity: one thread per trial
//
// Build:  nvcc -O3 -std=c++17 -arch=sm_89 qgs.cu -o qgs_gpu.exe -lcufft -lcurand
// Run:    qgs_gpu.exe <trials> <out.csv>
//
// Colin Casey, 2026. Extends G. Morozowsky and C. Casey, ECE 279AS (UCLA), 2024.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <curand_kernel.h>

constexpr int NQ = 6, DIM = 1 << NQ, GS_ITERS = 300, TPB = 256;
constexpr double PI = 3.14159265358979323846, A1 = 0.0, A2 = 4.0;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(e_) << " line " << __LINE__ << '\n'; std::exit(1); } } while (0)
#define CKF(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
    std::cerr << "cuFFT error " << (int)r_ << " line " << __LINE__ << '\n'; std::exit(1); } } while (0)

__constant__ double c_cdf1[DIM], c_cdf2[DIM];

// counts -> sqrt(empirical probability) for both bases; one thread per trial
__global__ void sample_kernel(double *m1, double *m2, int trials, long shots, unsigned long long seed) {
    int tr = blockIdx.x * blockDim.x + threadIdx.x;
    if (tr >= trials) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, tr, 0, &st);
    int c1[DIM], c2[DIM];
    for (int j = 0; j < DIM; j++) { c1[j] = 0; c2[j] = 0; }
    for (long s = 0; s < shots; s++) {
        double u1 = curand_uniform_double(&st), u2 = curand_uniform_double(&st);
        int lo = 0, hi = DIM - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (c_cdf1[mid] < u1) lo = mid + 1; else hi = mid; }
        c1[lo]++;
        lo = 0; hi = DIM - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (c_cdf2[mid] < u2) lo = mid + 1; else hi = mid; }
        c2[lo]++;
    }
    for (int j = 0; j < DIM; j++) {
        m1[tr * DIM + j] = sqrt((double)c1[j] / shots);
        m2[tr * DIM + j] = sqrt((double)c2[j] / shots);
    }
}

__global__ void init_kernel(cuDoubleComplex *f, const double *m1, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = make_cuDoubleComplex(m1[i], 0.0);
}
// after an unscaled inverse FFT: multiply by phase(k) / DIM (the two 1/sqrt(DIM) of the orthonormal QFTs)
__global__ void phase_kernel(cuDoubleComplex *f, const cuDoubleComplex *ph, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { cuDoubleComplex z = cuCmul(f[i], ph[i % DIM]); f[i] = make_cuDoubleComplex(cuCreal(z) / DIM, cuCimag(z) / DIM); }
}
__global__ void replace_kernel(cuDoubleComplex *f, const double *m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { double p = atan2(cuCimag(f[i]), cuCreal(f[i])); f[i] = make_cuDoubleComplex(m[i] * cos(p), m[i] * sin(p)); }
}
__global__ void fidelity_kernel(const cuDoubleComplex *f, const cuDoubleComplex *psi, double *fid, int trials) {
    int tr = blockIdx.x * blockDim.x + threadIdx.x;
    if (tr >= trials) return;
    double n2 = 0, re = 0, im = 0;
    for (int j = 0; j < DIM; j++) {
        cuDoubleComplex a = f[tr * DIM + j], b = psi[j];
        n2 += cuCreal(a) * cuCreal(a) + cuCimag(a) * cuCimag(a);
        re += cuCreal(a) * cuCreal(b) + cuCimag(a) * cuCimag(b);   // conj(a) * b
        im += cuCreal(a) * cuCimag(b) - cuCimag(a) * cuCreal(b);
    }
    fid[tr] = (re * re + im * im) / n2;
}

// host-side state and exact probabilities (std::complex)
using cd = std::complex<double>;
static std::vector<cd> host_disperse(double a, const std::vector<cd> &v) {
    std::vector<cd> u(DIM), w(DIM);
    for (int k = 0; k < DIM; k++) { cd s = 0; for (int j = 0; j < DIM; j++) s += v[j] * std::polar(1.0, 2 * PI * j * k / DIM); u[k] = s / std::sqrt((double)DIM); }
    for (int k = 0; k < DIM; k++) { double kk = 2 * PI * (k < DIM / 2 ? k : k - DIM) / DIM; u[k] *= std::polar(1.0, -a * kk * kk); }
    for (int j = 0; j < DIM; j++) { cd s = 0; for (int k = 0; k < DIM; k++) s += u[k] * std::polar(1.0, -2 * PI * j * k / DIM); w[j] = s / std::sqrt((double)DIM); }
    return w;
}

int main(int argc, char **argv) {
    if (argc < 3) { std::cerr << "usage: " << argv[0] << " <trials> <out.csv>\n"; return 1; }
    const int trials = std::atoi(argv[1]);
    const int n = trials * DIM, blocks_n = (n + TPB - 1) / TPB, blocks_t = (trials + TPB - 1) / TPB;

    std::vector<cd> psi(DIM); double norm = 0;
    for (int i = 0; i < DIM; i++) {
        double x = i - DIM / 2; cd val = std::exp(cd(-x * x / 16.0, 0.15 * x * x / 4.0));
        psi[(i + DIM / 2) % DIM] = val; norm += std::norm(val);
    }
    for (auto &z : psi) z /= std::sqrt(norm);
    std::vector<cd> d1 = host_disperse(A1, psi), d2 = host_disperse(A2, psi);
    double cdf1[DIM], cdf2[DIM], s1 = 0, s2 = 0;
    for (int j = 0; j < DIM; j++) { s1 += std::norm(d1[j]); s2 += std::norm(d2[j]); cdf1[j] = s1; cdf2[j] = s2; }
    cdf1[DIM - 1] = cdf2[DIM - 1] = 1.0;
    CK(cudaMemcpyToSymbol(c_cdf1, cdf1, sizeof cdf1)); CK(cudaMemcpyToSymbol(c_cdf2, cdf2, sizeof cdf2));

    std::vector<cuDoubleComplex> ph_fwd(DIM), ph_bwd(DIM), psi_d(DIM);
    for (int k = 0; k < DIM; k++) {
        double kk = 2 * PI * (k < DIM / 2 ? k : k - DIM) / DIM;
        ph_fwd[k] = make_cuDoubleComplex(cos(-(A2 - A1) * kk * kk), sin(-(A2 - A1) * kk * kk));
        ph_bwd[k] = make_cuDoubleComplex(cos((A2 - A1) * kk * kk), sin((A2 - A1) * kk * kk));
        psi_d[k] = make_cuDoubleComplex(psi[k].real(), psi[k].imag());
    }
    cuDoubleComplex *f, *dph_f, *dph_b, *dpsi; double *m1, *m2, *dfid;
    CK(cudaMalloc(&f, n * sizeof(cuDoubleComplex))); CK(cudaMalloc(&m1, n * sizeof(double))); CK(cudaMalloc(&m2, n * sizeof(double)));
    CK(cudaMalloc(&dph_f, DIM * sizeof(cuDoubleComplex))); CK(cudaMalloc(&dph_b, DIM * sizeof(cuDoubleComplex)));
    CK(cudaMalloc(&dpsi, DIM * sizeof(cuDoubleComplex))); CK(cudaMalloc(&dfid, trials * sizeof(double)));
    CK(cudaMemcpy(dph_f, ph_fwd.data(), DIM * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dph_b, ph_bwd.data(), DIM * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpsi, psi_d.data(), DIM * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan; int dims[1] = {DIM};
    CKF(cufftPlanMany(&plan, 1, dims, nullptr, 1, DIM, nullptr, 1, DIM, CUFFT_Z2Z, trials));

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    std::ofstream out(argv[2]); out << "backend,shots,trial,fidelity\n" << std::setprecision(10);
    std::cout << "backend=CUDA(" << prop.name << ") trials=" << trials << " qubits=" << NQ << " batched cuFFT\n\n";
    std::cout << std::setw(8) << "shots" << std::setw(14) << "mean F" << std::setw(12) << "time (ms)" << '\n';
    std::vector<double> fid(trials);
    for (long shots : {1L, 10L, 100L, 1000L, 10000L, 100000L}) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
        sample_kernel<<<blocks_t, TPB>>>(m1, m2, trials, shots, 777ULL + (unsigned long long)shots);
        init_kernel<<<blocks_n, TPB>>>(f, m1, n);
        for (int it = 0; it < GS_ITERS; it++) {        // f -> D(a2-a1) f -> |m2| ; -> D(a1-a2) -> |m1|
            CKF(cufftExecZ2Z(plan, f, f, CUFFT_INVERSE)); phase_kernel<<<blocks_n, TPB>>>(f, dph_f, n);
            CKF(cufftExecZ2Z(plan, f, f, CUFFT_FORWARD)); replace_kernel<<<blocks_n, TPB>>>(f, m2, n);
            CKF(cufftExecZ2Z(plan, f, f, CUFFT_INVERSE)); phase_kernel<<<blocks_n, TPB>>>(f, dph_b, n);
            CKF(cufftExecZ2Z(plan, f, f, CUFFT_FORWARD)); replace_kernel<<<blocks_n, TPB>>>(f, m1, n);
        }
        fidelity_kernel<<<blocks_t, TPB>>>(f, dpsi, dfid, trials);
        cudaEventRecord(e1); cudaEventSynchronize(e1); CK(cudaGetLastError());
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        CK(cudaMemcpy(fid.data(), dfid, trials * sizeof(double), cudaMemcpyDeviceToHost));
        double mean = 0;
        for (int tr = 0; tr < trials; tr++) { mean += fid[tr]; out << "gpu," << shots << ',' << tr << ',' << fid[tr] << '\n'; }
        std::cout << std::setw(8) << shots << std::setw(14) << std::setprecision(6) << mean / trials
                  << std::setw(12) << std::setprecision(4) << ms << '\n';
    }
    cufftDestroy(plan);
    cudaFree(f); cudaFree(m1); cudaFree(m2); cudaFree(dph_f); cudaFree(dph_b); cudaFree(dpsi); cudaFree(dfid);
    return 0;
}
