// gs_wavefunction.cu -- the same multi-plane Gerchberg-Saxton wavefunction recovery on the GPU (cuFFT).
//   nvcc -O2 -arch=sm_89 gs_wavefunction.cu -o gs_wavefunction_cuda -lcufft
//   gs_wavefunction_cuda densities.bin psi_out.bin n_iter
// Same file formats and exit codes as gs_wavefunction.c (0 ok, 2 bad arguments, 3 file error,
// 4 out of memory, 5 N not a power of 2, 6 CUDA or cuFFT error). Prints the final misfit and,
// on a second line, the GPU time in ms. Errors are return codes: CUDA C++ here uses no exceptions.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cufft.h>
#ifndef M_PI                       // not standard C++ (MSVC leaves it out), so define it
#define M_PI 3.14159265358979323846
#endif
#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e_)); return 6; } } while (0)
#define CHECKF(call) do { if ((call) != CUFFT_SUCCESS) { fprintf(stderr, "cuFFT error\n"); return 6; } } while (0)

// multiply by exp(-i hm k^2 t / 2), and divide by N to normalise the inverse FFT when asked
__global__ void propagate(cufftDoubleComplex *y, const double *k2, int n, double hm, double t, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double ph = -0.5 * hm * k2[i] * t, c = cos(ph) * scale, s = sin(ph) * scale;
    cufftDoubleComplex v = y[i];
    y[i].x = v.x * c - v.y * s; y[i].y = v.x * s + v.y * c;
}
// keep the phase, impose the measured amplitude
__global__ void project(cufftDoubleComplex *y, const double *amp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double m = hypot(y[i].x, y[i].y);
    if (m > 0) { y[i].x *= amp[i] / m; y[i].y *= amp[i] / m; } else { y[i].x = amp[i]; y[i].y = 0; }
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: gs_wavefunction_cuda densities.bin psi_out.bin n_iter\n"); return 2; }
    int n_iter = atoi(argv[3]); if (n_iter < 1) return 2;
    FILE *f = fopen(argv[1], "rb"); if (!f) return 3;
    int32_t N, K; double dx, hm;
    if (fread(&N, 4, 1, f) != 1 || fread(&K, 4, 1, f) != 1 || fread(&dx, 8, 1, f) != 1 || fread(&hm, 8, 1, f) != 1) { fclose(f); return 3; }
    if (N < 2 || (N & (N - 1)) != 0) { fclose(f); return 5; }
    if (K < 1 || K > 64) { fclose(f); return 2; }
    std::vector<double> t(K), amp((size_t)K * N), k2(N);
    if (fread(t.data(), 8, K, f) != (size_t)K || fread(amp.data(), 8, (size_t)K * N, f) != (size_t)K * N) { fclose(f); return 3; }
    fclose(f);
    for (auto &a : amp) a = std::sqrt(a > 0 ? a : 0);
    for (int i = 0; i < N; i++) { int m = i < N / 2 ? i : i - N; double kk = 2 * M_PI * m / (N * dx); k2[i] = kk * kk; }
    double *d_k2, *d_amp; cufftDoubleComplex *d_psi, *d_y;
    CHECK(cudaMalloc(&d_k2, 8 * N)); CHECK(cudaMalloc(&d_amp, 8 * (size_t)K * N));
    CHECK(cudaMalloc(&d_psi, sizeof(cufftDoubleComplex) * N)); CHECK(cudaMalloc(&d_y, sizeof(cufftDoubleComplex) * N));
    CHECK(cudaMemcpy(d_k2, k2.data(), 8 * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_amp, amp.data(), 8 * (size_t)K * N, cudaMemcpyHostToDevice));
    std::vector<cufftDoubleComplex> h(N);
    for (int i = 0; i < N; i++) { h[i].x = amp[i]; h[i].y = 0; }
    CHECK(cudaMemcpy(d_psi, h.data(), sizeof(cufftDoubleComplex) * N, cudaMemcpyHostToDevice));
    cufftHandle plan; CHECKF(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));
    int B = 256, G = (N + B - 1) / B;
    auto evolve = [&](cufftDoubleComplex *v, double tt) -> int {     // returns 0 or a cuFFT error code
        if (cufftExecZ2Z(plan, v, v, CUFFT_FORWARD) != CUFFT_SUCCESS) return 6;
        propagate<<<G, B>>>(v, d_k2, N, hm, tt, 1.0 / N);
        if (cufftExecZ2Z(plan, v, v, CUFFT_INVERSE) != CUFFT_SUCCESS) return 6;
        return 0;
    };
    if (evolve(d_psi, -t[0])) return 6;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
    for (int it = 0; it < n_iter; it++)
        for (int j = 0; j < K; j++) {
            CHECK(cudaMemcpy(d_y, d_psi, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToDevice));
            if (evolve(d_y, t[j])) return 6;
            project<<<G, B>>>(d_y, d_amp + (size_t)j * N, N);
            if (evolve(d_y, -t[j])) return 6;
            CHECK(cudaMemcpy(d_psi, d_y, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToDevice));
        }
    cudaEventRecord(e1); CHECK(cudaEventSynchronize(e1)); CHECK(cudaGetLastError());
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    double mis = 0;
    for (int j = 0; j < K; j++) {
        CHECK(cudaMemcpy(d_y, d_psi, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToDevice));
        if (evolve(d_y, t[j])) return 6;
        CHECK(cudaMemcpy(h.data(), d_y, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost));
        double s = 0; for (int i = 0; i < N; i++) { double d = std::hypot(h[i].x, h[i].y) - amp[(size_t)j * N + i]; s += d * d; }
        mis += std::sqrt(s);
    }
    CHECK(cudaMemcpy(h.data(), d_psi, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost));
    FILE *o = fopen(argv[2], "wb"); if (!o) return 3;
    for (int i = 0; i < N; i++) { if (fwrite(&h[i].x, 8, 1, o) != 1 || fwrite(&h[i].y, 8, 1, o) != 1) { fclose(o); return 3; } }
    fclose(o);
    printf("%.3e\n%.3f\n", mis, ms);
    cufftDestroy(plan); cudaFree(d_k2); cudaFree(d_amp); cudaFree(d_psi); cudaFree(d_y);
    return 0;
}
