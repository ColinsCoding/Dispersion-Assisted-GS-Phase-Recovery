/*
 * tdgsa.cu -- Time-domain Gerchberg-Saxton phase recovery on the GPU (CUDA + cuFFT)
 *
 * Same problem and output format as tdgsa.c (gas cell, D1 = 0, D2 = 10 ps/nm, flat start).
 *
 * Build (from a "x64 Native Tools" prompt, or via build.bat):
 *   nvcc -O3 -arch=sm_89 tdgsa.cu -o tdgsa_cuda.exe -lcufft
 * Run:    tdgsa_cuda.exe <iterations> <output_prefix>
 *
 * Colin Casey, 2026. Based on G. Morozowsky and C. Casey, ECE 279AS (UCLA), 2024.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

#define N (1 << 15)
#define PI 3.14159265358979323846
#define THREADS 256
#define BLOCKS ((N + THREADS - 1) / THREADS)

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CUFFT_CHECK(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1); } } while (0)

__global__ void mul(cuDoubleComplex *x, const cuDoubleComplex *h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = cuCmul(x[i], h[i]);
}
__global__ void scale(cuDoubleComplex *x, double s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = make_cuDoubleComplex(s * cuCreal(x[i]), s * cuCimag(x[i]));
}
__global__ void magnitude(const cuDoubleComplex *x, double *a, double s) {      /* a = |s x| */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = s * cuCabs(x[i]);
}
/* x <- A * exp(i arg(s x)); optionally accumulate sum (|s x| - A)^2 into *acc */
__global__ void replace_amp(cuDoubleComplex *x, const double *A, double s, double *acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double re = s * cuCreal(x[i]), im = s * cuCimag(x[i]);
        if (acc) { double d = sqrt(re * re + im * im) - A[i]; atomicAdd(acc, d * d); }
        double ph = atan2(im, re);
        x[i] = make_cuDoubleComplex(A[i] * cos(ph), A[i] * sin(ph));
    }
}

static double beta2L(double D) {
    const double c = 299792458.0, lam = 1550e-9;
    return -D * 1e-3 * lam * lam / (2.0 * PI * c);
}
static cuDoubleComplex cexp_h(double re, double im) {
    double m = exp(re); return make_cuDoubleComplex(m * cos(im), m * sin(im));
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <iterations> <output_prefix>\n", argv[0]); return 1; }
    const long n_iter = atol(argv[1]);
    const double dt = 0.06e-12, T0 = 0.3e-12, D1 = 0.0, D2 = 10.0, invN = 1.0 / N;
    const double lines[5][3] = { {-400e9, 3e9, 1.5}, {-80e9, 1e9, 2.0}, {60e9, 1e9, 2.5},
                                 {300e9, 4e9, 1.2}, {900e9, 6e9, 0.8} };
    const double b1 = beta2L(D1), b2 = beta2L(D2);

    std::vector<cuDoubleComplex> E(N), H(N), Dn1(N), Dn2(N), H12(N), H21(N), Dinv1(N);
    for (int i = 0; i < N; i++) {
        double t = (double)(i - N / 2) * dt;
        long fi = (i < N / 2) ? i : i - N;
        double w = 2.0 * PI * (double)fi / ((double)N * dt);
        E[i] = make_cuDoubleComplex(exp(-t * t / (2.0 * T0 * T0)), 0.0);
        cuDoubleComplex h = make_cuDoubleComplex(1.0, 0.0);
        for (int l = 0; l < 5; l++) {   /* exp(-a / (1 - i x)) = exp(-a (1 + i x) / (1 + x^2)) */
            double x = (w - 2.0 * PI * lines[l][0]) / (2.0 * PI * lines[l][1]);
            double a = lines[l][2];
            h = cuCmul(h, cexp_h(-a / (1 + x * x), -a * x / (1 + x * x)));
        }
        H[i] = h;
        Dn1[i] = cexp_h(0.0, -b1 * w * w / 2.0);
        Dn2[i] = cexp_h(0.0, -b2 * w * w / 2.0);
        H12[i] = cexp_h(0.0, -(b2 - b1) * w * w / 2.0);
        H21[i] = cexp_h(0.0, (b2 - b1) * w * w / 2.0);
        Dinv1[i] = cexp_h(0.0, b1 * w * w / 2.0);
    }

    size_t cb = N * sizeof(cuDoubleComplex);
    cuDoubleComplex *dF, *df, *dg, *dH, *dD1, *dD2, *dH12, *dH21, *dDinv1;
    double *dA1, *dA2, *dAcc;
    CUDA_CHECK(cudaMalloc(&dF, cb)); CUDA_CHECK(cudaMalloc(&df, cb)); CUDA_CHECK(cudaMalloc(&dg, cb));
    CUDA_CHECK(cudaMalloc(&dH, cb)); CUDA_CHECK(cudaMalloc(&dD1, cb)); CUDA_CHECK(cudaMalloc(&dD2, cb));
    CUDA_CHECK(cudaMalloc(&dH12, cb)); CUDA_CHECK(cudaMalloc(&dH21, cb)); CUDA_CHECK(cudaMalloc(&dDinv1, cb));
    CUDA_CHECK(cudaMalloc(&dA1, N * sizeof(double))); CUDA_CHECK(cudaMalloc(&dA2, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dAcc, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dF, E.data(), cb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dH, H.data(), cb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD1, Dn1.data(), cb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD2, Dn2.data(), cb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dH12, H12.data(), cb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dH21, H21.data(), cb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDinv1, Dinv1.data(), cb, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));

    /* spectrum through gas cell, then the two measurements */
    CUFFT_CHECK(cufftExecZ2Z(plan, dF, dF, CUFFT_FORWARD));
    mul<<<BLOCKS, THREADS>>>(dF, dH);
    CUDA_CHECK(cudaMemcpy(df, dF, cb, cudaMemcpyDeviceToDevice));
    mul<<<BLOCKS, THREADS>>>(df, dD1);
    CUFFT_CHECK(cufftExecZ2Z(plan, df, df, CUFFT_INVERSE));
    magnitude<<<BLOCKS, THREADS>>>(df, dA1, invN);
    CUDA_CHECK(cudaMemcpy(df, dF, cb, cudaMemcpyDeviceToDevice));
    mul<<<BLOCKS, THREADS>>>(df, dD2);
    CUFFT_CHECK(cufftExecZ2Z(plan, df, df, CUFFT_INVERSE));
    magnitude<<<BLOCKS, THREADS>>>(df, dA2, invN);

    std::vector<double> A2h(N);
    CUDA_CHECK(cudaMemcpy(A2h.data(), dA2, N * sizeof(double), cudaMemcpyDeviceToHost));
    double den = 0; for (int i = 0; i < N; i++) den += A2h[i] * A2h[i];

    /* flat start: f = A1 */
    std::vector<double> A1h(N);
    CUDA_CHECK(cudaMemcpy(A1h.data(), dA1, N * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<cuDoubleComplex> f0(N);
    for (int i = 0; i < N; i++) f0[i] = make_cuDoubleComplex(A1h[i], 0.0);
    CUDA_CHECK(cudaMemcpy(df, f0.data(), cb, cudaMemcpyHostToDevice));

    std::vector<double> err(n_iter > 0 ? n_iter : 1);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (long it = 0; it < n_iter; it++) {
        CUFFT_CHECK(cufftExecZ2Z(plan, df, dg, CUFFT_FORWARD));
        mul<<<BLOCKS, THREADS>>>(dg, dH12);
        CUFFT_CHECK(cufftExecZ2Z(plan, dg, dg, CUFFT_INVERSE));
        CUDA_CHECK(cudaMemsetAsync(dAcc, 0, sizeof(double)));
        replace_amp<<<BLOCKS, THREADS>>>(dg, dA2, invN, dAcc);
        CUDA_CHECK(cudaMemcpy(&err[it], dAcc, sizeof(double), cudaMemcpyDeviceToHost));
        err[it] = sqrt(err[it] / den);
        CUFFT_CHECK(cufftExecZ2Z(plan, dg, df, CUFFT_FORWARD));
        mul<<<BLOCKS, THREADS>>>(df, dH21);
        CUFFT_CHECK(cufftExecZ2Z(plan, df, df, CUFFT_INVERSE));
        replace_amp<<<BLOCKS, THREADS>>>(df, dA1, invN, nullptr);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);

    /* recovered spectrum F0 = fft(f1) / D1 */
    CUFFT_CHECK(cufftExecZ2Z(plan, df, df, CUFFT_FORWARD));
    mul<<<BLOCKS, THREADS>>>(df, dDinv1);
    std::vector<cuDoubleComplex> Fout(N);
    CUDA_CHECK(cudaMemcpy(Fout.data(), df, cb, cudaMemcpyDeviceToHost));

    char path[512];
    snprintf(path, sizeof path, "%s_F.bin", argv[2]);
    FILE *fp = fopen(path, "wb"); if (!fp) { perror(path); return 1; }
    fwrite(Fout.data(), sizeof(cuDoubleComplex), N, fp); fclose(fp);
    snprintf(path, sizeof path, "%s_err.bin", argv[2]);
    fp = fopen(path, "wb"); if (!fp) { perror(path); return 1; }
    fwrite(err.data(), sizeof(double), n_iter, fp); fclose(fp);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("backend=CUDA(%s) iterations=%ld time_ms=%.3f final_meas_err=%.6e\n",
           prop.name, n_iter, ms, n_iter > 0 ? err[n_iter - 1] : 0.0);

    cufftDestroy(plan);
    cudaFree(dF); cudaFree(df); cudaFree(dg); cudaFree(dH); cudaFree(dD1); cudaFree(dD2);
    cudaFree(dH12); cudaFree(dH21); cudaFree(dDinv1); cudaFree(dA1); cudaFree(dA2); cudaFree(dAcc);
    return 0;
}
