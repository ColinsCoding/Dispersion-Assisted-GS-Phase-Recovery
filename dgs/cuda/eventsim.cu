// Many independent M/M/c replications in parallel on the GPU, one replication per thread.
//
//   eventsim_cuda out.bin lambda mu c n warmup reps seed
//
// A priority-queue event list is branchy and serial, a poor fit for GPU threads. For a FIFO
// M/M/c queue the same answer comes from the Kiefer-Wolfowitz recursion: each server stores
// the time it becomes free, and each job starts at max(arrival, earliest free server). Every
// thread runs that loop with its own Philox random stream. The CPU event-list simulator
// (dgs/c/eventsim) and this kernel must agree statistically, and the tests check that they do.
//
// Writes reps doubles of mean wait, then reps doubles of P(wait > 0), to out.bin.
// Prints kernel_ms=... on stdout. Exit 2 = bad arguments, 3 = no usable CUDA device.
// Build: nvcc -O2 -arch=sm_89 eventsim.cu -o eventsim_cuda
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <curand_kernel.h>

#define MAXC 64

__global__ void mmc_replications(double lam, double mu, int c, long n, long warmup,
                                 unsigned long long seed, int reps, double* wq, double* pw) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= reps) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, r, 0, &st);
    double free_at[MAXC];
    for (int k = 0; k < c; ++k) free_at[k] = 0.0;
    double t = 0.0, sum_w = 0.0;
    long n_wait = 0;
    for (long i = 0; i < warmup + n; ++i) {
        t += -log(curand_uniform_double(&st)) / lam;           // u in (0, 1]
        int kmin = 0;
        for (int k = 1; k < c; ++k)
            if (free_at[k] < free_at[kmin]) kmin = k;
        const double s = free_at[kmin] > t ? free_at[kmin] : t;
        free_at[kmin] = s - log(curand_uniform_double(&st)) / mu;
        if (i >= warmup) {
            sum_w += s - t;
            if (s > t) ++n_wait;
        }
    }
    wq[r] = sum_w / n;
    pw[r] = static_cast<double>(n_wait) / n;
}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::fprintf(stderr, "usage: %s out.bin lambda mu c n warmup reps seed\n", argv[0]);
        return 2;
    }
    const double lam = std::atof(argv[2]), mu = std::atof(argv[3]);
    const int c = std::atoi(argv[4]);
    const long n = std::atol(argv[5]), warmup = std::atol(argv[6]);
    const int reps = std::atoi(argv[7]);
    const unsigned long long seed = std::strtoull(argv[8], nullptr, 10);
    if (lam <= 0 || mu <= 0 || c < 1 || c > MAXC || n < 1 || warmup < 0 || reps < 1 || lam >= c * mu) {
        std::fprintf(stderr, "need lambda, mu > 0, 1 <= c <= %d, n >= 1, warmup >= 0, reps >= 1, lambda < c*mu\n", MAXC);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 3;
    }
    double *d_wq = nullptr, *d_pw = nullptr;
    cudaMalloc(&d_wq, reps * sizeof(double));
    cudaMalloc(&d_pw, reps * sizeof(double));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    const int threads = 128, blocks = (reps + threads - 1) / threads;
    cudaEventRecord(a);
    mmc_replications<<<blocks, threads>>>(lam, mu, c, n, warmup, seed, reps, d_wq, d_pw);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
        return 3;
    }
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    std::vector<double> h(2 * reps);
    cudaMemcpy(h.data(), d_wq, reps * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h.data() + reps, d_pw, reps * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_wq);
    cudaFree(d_pw);
    std::FILE* f = std::fopen(argv[1], "wb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    std::fwrite(h.data(), sizeof(double), h.size(), f);
    std::fclose(f);
    std::printf("kernel_ms=%.3f\n", ms);
    return 0;
}
