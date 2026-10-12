/*
 * nlse_cuda.cu -- the Kerr-fiber kernel of nlse_kernel.c, batched on the GPU (CUDA + cuFFT)
 *
 * Every sample's pulse is propagated at once: one cufftPlanMany plan with batch = number of
 * samples, and element-wise kernels for dispersion, Kerr phase and feature extraction.
 * Same physics, grid and operation order as optical_features() in nlse_kernel.c.
 *
 * Usage:
 *   nlse_cuda.exe check <nlse_features.csv> <out.bin>   features for the CSV's (x, y) points, for comparison
 *   nlse_cuda.exe bench <N>                             time N random samples
 *
 * Build (strict): nvcc -O3 -std=c++17 -arch=sm_89 -Werror all-warnings nlse_cuda.cu -o nlse_cuda.exe -lcufft
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

constexpr int NT = 128, NFEAT = 24, STEPS = 80, TPB = 256;
constexpr double PI = 3.14159265358979323846, TWIN = 40.0, BETA2 = 0.5, LEN = 1.0, GAMMA = 6.0;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)
#define CKF(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { std::fprintf(stderr, "cuFFT %d line %d\n", (int)r_, __LINE__); std::exit(1); } } while (0)

__constant__ double c_w[NT];

__device__ cuDoubleComplex cpolar(double m, double ph) { return make_cuDoubleComplex(m * cos(ph), m * sin(ph)); }

__global__ void encode(cuDoubleComplex *A, const double *xs, const double *ys, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n * NT) return;
    int b = k / NT, i = k % NT;
    double w = c_w[i];
    A[k] = cpolar(exp(-w * w / 2.0), 1.2 * xs[b] * w * w + 0.6 * ys[b] * w * w * w);
}

__global__ void scale(cuDoubleComplex *A, double s, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n * NT) A[k] = make_cuDoubleComplex(s * cuCreal(A[k]), s * cuCimag(A[k]));
}

/* half dispersion step in frequency, then the 1/NT of the following inverse FFT folded in */
__global__ void dispersion_half(cuDoubleComplex *A, double h, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n * NT) return;
    double w = c_w[k % NT];
    A[k] = cuCmul(A[k], cpolar(1.0 / NT, BETA2 / 2.0 * w * w * h / 2.0));
}

__global__ void kerr(cuDoubleComplex *A, double h, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n * NT) return;
    double p = cuCreal(A[k]) * cuCreal(A[k]) + cuCimag(A[k]) * cuCimag(A[k]);
    A[k] = cuCmul(A[k], cpolar(1.0, GAMMA * p * h));
}

__global__ void features(const cuDoubleComplex *A, double *F, int n)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) return;
    double tot = 0.0, v[NFEAT];
    for (int f = 0; f < NFEAT; f++) {
        int idx = f - NFEAT / 2;
        if (idx < 0) idx += NT;
        cuDoubleComplex a = A[b * NT + idx];
        v[f] = cuCreal(a) * cuCreal(a) + cuCimag(a) * cuCimag(a);
        tot += v[f];
    }
    for (int f = 0; f < NFEAT; f++) F[b * NFEAT + f] = v[f] / tot;
}

/* returns GPU milliseconds for encode + propagate + features */
static float gpu_features(const std::vector<double> &xs, const std::vector<double> &ys, std::vector<double> &F)
{
    const int n = (int)xs.size(), blocks = (n * NT + TPB - 1) / TPB;
    const double dt = TWIN / NT, h = LEN / STEPS;
    double w[NT];
    for (int i = 0; i < NT; i++) w[i] = 2.0 * PI * (i < NT / 2 ? i : i - NT) / (NT * dt);
    CK(cudaMemcpyToSymbol(c_w, w, sizeof w));

    /* peak power of the transform-limited reference pulse, as in the CPU code */
    std::vector<double> ref(NT, 0.0);
    double pk = 0.0;
    for (int tI = 0; tI < NT; tI++) {
        double re = 0.0, im = 0.0;
        for (int i = 0; i < NT; i++) { double a = exp(-w[i] * w[i] / 2.0), ph = 2.0 * PI * i * tI / NT; re += a * cos(ph); im += a * sin(ph); }
        re /= NT; im /= NT;
        if (re * re + im * im > pk) pk = re * re + im * im;
    }

    double *dx, *dy, *dF;
    cuDoubleComplex *A;
    CK(cudaMalloc(&dx, n * sizeof(double))); CK(cudaMalloc(&dy, n * sizeof(double)));
    CK(cudaMalloc(&dF, (size_t)n * NFEAT * sizeof(double))); CK(cudaMalloc(&A, (size_t)n * NT * sizeof(cuDoubleComplex)));
    CK(cudaMemcpy(dx, xs.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy, ys.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    cufftHandle plan;
    int dims[1] = {NT};
    CKF(cufftPlanMany(&plan, 1, dims, nullptr, 1, NT, nullptr, 1, NT, CUFFT_Z2Z, n));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
    encode<<<blocks, TPB>>>(A, dx, dy, n);
    CKF(cufftExecZ2Z(plan, A, A, CUFFT_INVERSE));                  /* to time, then 1/NT and peak normalization */
    scale<<<blocks, TPB>>>(A, 1.0 / (NT * std::sqrt(pk)), n);
    for (int s = 0; s < STEPS; s++) {
        CKF(cufftExecZ2Z(plan, A, A, CUFFT_FORWARD)); dispersion_half<<<blocks, TPB>>>(A, h, n);
        CKF(cufftExecZ2Z(plan, A, A, CUFFT_INVERSE)); kerr<<<blocks, TPB>>>(A, h, n);
        CKF(cufftExecZ2Z(plan, A, A, CUFFT_FORWARD)); dispersion_half<<<blocks, TPB>>>(A, h, n);
        CKF(cufftExecZ2Z(plan, A, A, CUFFT_INVERSE));
    }
    CKF(cufftExecZ2Z(plan, A, A, CUFFT_FORWARD));
    features<<<(n + TPB - 1) / TPB, TPB>>>(A, dF, n);
    cudaEventRecord(e1); cudaEventSynchronize(e1); CK(cudaGetLastError());
    float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);

    F.resize((size_t)n * NFEAT);
    CK(cudaMemcpy(F.data(), dF, F.size() * sizeof(double), cudaMemcpyDeviceToHost));
    cufftDestroy(plan); cudaFree(dx); cudaFree(dy); cudaFree(dF); cudaFree(A);
    return ms;
}

int main(int argc, char **argv)
{
    if (argc >= 4 && std::strcmp(argv[1], "check") == 0) {
        FILE *fp = std::fopen(argv[2], "r");
        if (!fp) { std::perror(argv[2]); return 1; }
        char line[8192];
        std::vector<double> xs, ys;
        if (!std::fgets(line, sizeof line, fp)) return 1;          /* header */
        while (std::fgets(line, sizeof line, fp)) {
            double x, y;
            if (std::sscanf(line, "%lf,%lf", &x, &y) == 2) { xs.push_back(x); ys.push_back(y); }
        }
        std::fclose(fp);
        std::vector<double> F;
        float ms = gpu_features(xs, ys, F);
        FILE *fo = std::fopen(argv[3], "wb");
        if (!fo) { std::perror(argv[3]); return 1; }
        std::fwrite(F.data(), sizeof(double), F.size(), fo); std::fclose(fo);
        std::printf("GPU features for %zu samples in %.2f ms -> %s\n", xs.size(), ms, argv[3]);
        return 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "bench") == 0) {
        int n = std::atoi(argv[2]);
        std::vector<double> xs(n), ys(n), F;
        for (int i = 0; i < n; i++) { xs[i] = std::sin(0.37 * i); ys[i] = std::cos(0.61 * i); }
        gpu_features(xs, ys, F);                                    /* warm-up (plan creation, context) */
        float ms = gpu_features(xs, ys, F);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
        std::printf("GPU (%s): %d samples x %d split steps in %.2f ms  (%.2f us/sample)\n", prop.name, n, STEPS, ms, 1000.0 * ms / n);
        return 0;
    }
    std::fprintf(stderr, "usage: %s check <nlse_features.csv> <out.bin> | bench <N>\n", argv[0]);
    return 1;
}
