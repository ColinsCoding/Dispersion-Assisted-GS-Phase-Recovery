// taylor.cu -- hydrogen n = 2 radial probabilities on the GPU with a range-reduced Taylor e^x.
//
// One thread per radius r_i on [0, 80] a0: P_2s(r) = r^2 (2-r)^2 e^{-r} / 8, P_2p(r) = r^4 e^{-r} / 24.
// The host recomputes with libm exp() and reports the max relative error, the trapezoid area
// (should be 1), and the kernel time measured with CUDA events on this GPU.
//
// Build: nvcc -O3 -arch=sm_89 taylor.cu -o taylor_gpu.exe   (after vcvars64.bat)
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

__host__ __device__ inline double taylor_exp(double x, int n)
{
    double acc = 1.0;
    for (int k = n - 1; k > 0; --k) acc = 1.0 + x * acc / (double)k;
    return acc;
}

__host__ __device__ inline double exp_reduced(double x)
{
    const double LN2 = 0.693147180559945309417232121458;
    double k = nearbyint(x / LN2);
    return ldexp(taylor_exp(x - k * LN2, 14), (int)k);
}

__global__ void radial_probs(int n, double h, double *p2s, double *p2p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double r = i * h, e = exp_reduced(-r);
    p2s[i] = r * r * (2.0 - r) * (2.0 - r) * e / 8.0;
    p2p[i] = r * r * r * r * e / 24.0;
}

int main()
{
    const int n = 1 << 20;
    const double h = 80.0 / (n - 1);
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    double *d2s, *d2p;
    CHECK(cudaMalloc(&d2s, n * sizeof(double)));
    CHECK(cudaMalloc(&d2p, n * sizeof(double)));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    radial_probs<<<(n + 255) / 256, 256>>>(n, h, d2s, d2p);          // warm-up
    cudaEventRecord(t0);
    radial_probs<<<(n + 255) / 256, 256>>>(n, h, d2s, d2p);
    cudaEventRecord(t1);
    CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    std::vector<double> h2s(n), h2p(n);
    CHECK(cudaMemcpy(h2s.data(), d2s, n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h2p.data(), d2p, n * sizeof(double), cudaMemcpyDeviceToHost));
    double maxrel = 0.0, a2s = 0.0, a2p = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = i * h, e = std::exp(-r);
        double ref = r * r * r * r * e / 24.0;
        if (ref > 1e-300) maxrel = std::fmax(maxrel, std::fabs(h2p[i] / ref - 1.0));
        double w = (i == 0 || i == n - 1) ? 0.5 : 1.0;
        a2s += w * h2s[i];
        a2p += w * h2p[i];
    }
    std::printf("device, %s\n", prop.name);
    std::printf("points, %d\nkernel_ms, %.4f\n", n, ms);
    std::printf("max_rel_err_P2p_vs_libm, %.3e\n", maxrel);
    std::printf("area_P2s, %.12f\narea_P2p, %.12f\n", a2s * h, a2p * h);
    cudaFree(d2s); cudaFree(d2p);
    return 0;
}
