/*
 * pst_cuda.cu -- Phase Stretch Transform on the GPU (CUDA + 2-D cuFFT)
 *
 * Same equations, grid and parameters as pst.c:
 *   low-pass filter -> spectrum x exp(-i S phi(rho)/max) -> inverse FFT -> phase -> normalize to [0, 1]
 *
 * Usage:
 *   pst_cuda.exe bin <in.bin> <width> <height> <out.bin>    float64 image (e.g. cells_input.bin from pst.exe)
 *   pst_cuda.exe pgm <in.pgm> <out.pgm>                     any 8-bit grayscale P5 frame (e.g. a dim camera frame)
 *
 * Build (strict): nvcc -O3 -std=c++17 -arch=sm_89 -Werror all-warnings pst_cuda.cu -o pst_cuda.exe -lcufft
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

constexpr double S = 0.5, W = 15.0, SIGMA_LPF = 0.15;
constexpr int TPB = 256;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)
#define CKF(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { std::fprintf(stderr, "cuFFT %d line %d\n", (int)r_, __LINE__); std::exit(1); } } while (0)

__device__ double rho_at(int k, int nx, int ny)
{
    int i = k % nx, j = k / nx;
    double u = (double)(i < nx / 2 ? i : i - nx) / nx, v = (double)(j < ny / 2 ? j : j - ny) / ny;
    return sqrt(u * u + v * v);
}

__global__ void lowpass(cuDoubleComplex *a, int nx, int ny)        /* includes the 1/(nx ny) of the inverse FFT */
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nx * ny) return;
    double sl = SIGMA_LPF / sqrt(log(2.0)), r = rho_at(k, nx, ny) / sl;
    double g = exp(-0.5 * r * r) / ((double)nx * ny);
    a[k] = make_cuDoubleComplex(g * cuCreal(a[k]), g * cuCimag(a[k]));
}

__global__ void keep_real(cuDoubleComplex *a, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) a[k] = make_cuDoubleComplex(cuCreal(a[k]), 0.0);
}

__global__ void phase_kernel(cuDoubleComplex *a, int nx, int ny, double phimax)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nx * ny) return;
    double wr = W * rho_at(k, nx, ny);
    double ph = -S * (wr * atan(wr) - 0.5 * log(1.0 + wr * wr)) / phimax;
    double s = 1.0 / ((double)nx * ny);
    a[k] = cuCmul(a[k], make_cuDoubleComplex(s * cos(ph), s * sin(ph)));
}

__global__ void angle(const cuDoubleComplex *a, double *out, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) out[k] = atan2(cuCimag(a[k]), cuCreal(a[k]));
}

static float pst_gpu(const std::vector<double> &img, int nx, int ny, std::vector<double> &feat)
{
    const int n = nx * ny, blocks = (n + TPB - 1) / TPB;
    double phimax = 0.0;                                           /* max of the kernel over the grid (host) */
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            double u = (double)(i < nx / 2 ? i : i - nx) / nx, v = (double)(j < ny / 2 ? j : j - ny) / ny;
            double wr = W * std::sqrt(u * u + v * v);
            phimax = std::max(phimax, wr * std::atan(wr) - 0.5 * std::log(1.0 + wr * wr));
        }
    std::vector<cuDoubleComplex> h(n);
    for (int k = 0; k < n; k++) h[k] = make_cuDoubleComplex(img[k], 0.0);
    cuDoubleComplex *a; double *d_out;
    CK(cudaMalloc(&a, n * sizeof(cuDoubleComplex))); CK(cudaMalloc(&d_out, n * sizeof(double)));
    CK(cudaMemcpy(a, h.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    cufftHandle plan;
    CKF(cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z));                      /* row-major: ny rows of nx */

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
    CKF(cufftExecZ2Z(plan, a, a, CUFFT_FORWARD)); lowpass<<<blocks, TPB>>>(a, nx, ny);
    CKF(cufftExecZ2Z(plan, a, a, CUFFT_INVERSE)); keep_real<<<blocks, TPB>>>(a, n);
    CKF(cufftExecZ2Z(plan, a, a, CUFFT_FORWARD)); phase_kernel<<<blocks, TPB>>>(a, nx, ny, phimax);
    CKF(cufftExecZ2Z(plan, a, a, CUFFT_INVERSE)); angle<<<blocks, TPB>>>(a, d_out, n);
    cudaEventRecord(e1); cudaEventSynchronize(e1); CK(cudaGetLastError());
    float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);

    feat.resize(n);
    CK(cudaMemcpy(feat.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
    double lo = *std::min_element(feat.begin(), feat.end()), hi = *std::max_element(feat.begin(), feat.end());
    for (double &f : feat) f = (f - lo) / (hi - lo);
    cufftDestroy(plan); cudaFree(a); cudaFree(d_out);
    return ms;
}

static bool read_pgm(const char *path, std::vector<double> &img, int &nx, int &ny)
{
    FILE *fp = std::fopen(path, "rb");
    if (!fp) return false;
    char magic[3] = {0};
    int maxv = 0;
    if (std::fscanf(fp, "%2s", magic) != 1 || std::strcmp(magic, "P5") != 0) { std::fclose(fp); return false; }
    int vals[3], got = 0;
    while (got < 3) {                                              /* width, height, maxval, skipping # comments */
        int c = std::fgetc(fp);
        if (c == '#') { while (c != '\n' && c != EOF) c = std::fgetc(fp); continue; }
        if (c == EOF) { std::fclose(fp); return false; }
        if (c >= '0' && c <= '9') { std::ungetc(c, fp); if (std::fscanf(fp, "%d", &vals[got]) != 1) { std::fclose(fp); return false; } got++; }
    }
    nx = vals[0]; ny = vals[1]; maxv = vals[2];
    std::fgetc(fp);                                                /* single whitespace before the data */
    if (maxv > 255) { std::fclose(fp); return false; }
    std::vector<unsigned char> raw((size_t)nx * ny);
    size_t r = std::fread(raw.data(), 1, raw.size(), fp);
    std::fclose(fp);
    if (r != raw.size()) return false;
    img.resize(raw.size());
    for (size_t k = 0; k < raw.size(); k++) img[k] = raw[k] / (double)maxv;
    return true;
}

int main(int argc, char **argv)
{
    std::vector<double> img, feat;
    int nx = 0, ny = 0;
    if (argc >= 6 && std::strcmp(argv[1], "bin") == 0) {
        nx = std::atoi(argv[3]); ny = std::atoi(argv[4]);
        img.resize((size_t)nx * ny);
        FILE *fp = std::fopen(argv[2], "rb");
        if (!fp || std::fread(img.data(), sizeof(double), img.size(), fp) != img.size()) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        std::fclose(fp);
        float ms = pst_gpu(img, nx, ny, feat);
        FILE *fo = std::fopen(argv[5], "wb");
        if (!fo) { std::perror(argv[5]); return 1; }
        std::fwrite(feat.data(), sizeof(double), feat.size(), fo); std::fclose(fo);
        std::printf("GPU PST %dx%d in %.3f ms -> %s\n", nx, ny, ms, argv[5]);
        return 0;
    }
    if (argc >= 4 && std::strcmp(argv[1], "pgm") == 0) {
        if (!read_pgm(argv[2], img, nx, ny)) { std::fprintf(stderr, "cannot read P5 PGM %s\n", argv[2]); return 1; }
        float ms = pst_gpu(img, nx, ny, feat);
        FILE *fo = std::fopen(argv[3], "wb");
        if (!fo) { std::perror(argv[3]); return 1; }
        std::fprintf(fo, "P5\n%d %d\n255\n", nx, ny);
        for (double f : feat) std::fputc((int)(255.0 * f + 0.5), fo);
        std::fclose(fo);
        std::printf("GPU PST %dx%d frame in %.3f ms -> %s\n", nx, ny, ms, argv[3]);
        return 0;
    }
    std::fprintf(stderr, "usage: %s bin <in.bin> <w> <h> <out.bin> | pgm <in.pgm> <out.pgm>\n", argv[0]);
    return 1;
}
