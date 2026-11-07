// vision_gpu.cu -- the Phase Stretch Transform on the GPU: cuFFT forward, one kernel multiplies by the
// low-pass and the warped phase exp(-i S phi), cuFFT inverse, one kernel takes the phase. Same math as
// pst_cpu (the kernel table is computed once on the host by pst_kernel and copied up).
#include "droid.hpp"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

namespace droid {

namespace {
__global__ void apply_kernel(cufftComplex* a, const float* lp, const float* k, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float c = cosf(k[i]), s = -sinf(k[i]);             // exp(-i k)
    cufftComplex z = a[i];
    a[i] = make_cuComplex(lp[i] * (z.x * c - z.y * s), lp[i] * (z.x * s + z.y * c));
}

__global__ void phase_of(const cufftComplex* a, float* out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = atan2f(a[i].y, a[i].x);
}

#define TRY(call, what) do { if ((call) != cudaSuccess) { err = what; goto fail; } } while (0)
}  // namespace

bool pst_gpu(const Image& img, double S, double W, double sigma_lp, std::vector<float>& out, std::string& err) {
    const int n = img.n, count = n * n;
    std::vector<double> lpd;
    std::vector<double> kd = pst_kernel(n, S, W, sigma_lp, lpd);
    std::vector<float> lp(lpd.size()), k(kd.size());
    for (size_t i = 0; i < kd.size(); ++i) { lp[i] = static_cast<float>(lpd[i]); k[i] = static_cast<float>(kd[i]); }
    std::vector<cufftComplex> host(count);
    for (int i = 0; i < count; ++i) host[i] = make_cuComplex(img.px[i], 0.f);
    cufftComplex* d_a = nullptr;
    float *d_lp = nullptr, *d_k = nullptr, *d_out = nullptr;
    cufftHandle plan = 0;
    bool have_plan = false;
    const int threads = 256, blocks = (count + threads - 1) / threads;
    TRY(cudaMalloc(&d_a, sizeof(cufftComplex) * count), "cudaMalloc failed (is there a CUDA GPU?)");
    TRY(cudaMalloc(&d_lp, sizeof(float) * count), "cudaMalloc failed");
    TRY(cudaMalloc(&d_k, sizeof(float) * count), "cudaMalloc failed");
    TRY(cudaMalloc(&d_out, sizeof(float) * count), "cudaMalloc failed");
    TRY(cudaMemcpy(d_a, host.data(), sizeof(cufftComplex) * count, cudaMemcpyHostToDevice), "copy to GPU failed");
    TRY(cudaMemcpy(d_lp, lp.data(), sizeof(float) * count, cudaMemcpyHostToDevice), "copy to GPU failed");
    TRY(cudaMemcpy(d_k, k.data(), sizeof(float) * count, cudaMemcpyHostToDevice), "copy to GPU failed");
    if (cufftPlan2d(&plan, n, n, CUFFT_C2C) != CUFFT_SUCCESS) { err = "cufftPlan2d failed"; goto fail; }
    have_plan = true;
    if (cufftExecC2C(plan, d_a, d_a, CUFFT_FORWARD) != CUFFT_SUCCESS) { err = "cuFFT forward failed"; goto fail; }
    apply_kernel<<<blocks, threads>>>(d_a, d_lp, d_k, count);
    if (cufftExecC2C(plan, d_a, d_a, CUFFT_INVERSE) != CUFFT_SUCCESS) { err = "cuFFT inverse failed"; goto fail; }
    phase_of<<<blocks, threads>>>(d_a, d_out, count);
    out.resize(count);
    TRY(cudaMemcpy(out.data(), d_out, sizeof(float) * count, cudaMemcpyDeviceToHost), "copy from GPU failed");
    cufftDestroy(plan);
    cudaFree(d_a); cudaFree(d_lp); cudaFree(d_k); cudaFree(d_out);
    return true;
fail:
    if (have_plan) cufftDestroy(plan);
    cudaFree(d_a); cudaFree(d_lp); cudaFree(d_k); cudaFree(d_out);
    return false;
}

}  // namespace droid
