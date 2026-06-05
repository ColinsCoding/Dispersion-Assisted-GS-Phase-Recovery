/*
 * repl/lut_sincos.cu
 * CUDA LUT sin/cos: __constant__ memory LUT, one thread per frequency bin.
 * Compile: nvcc -O2 -o lut_sincos_cuda lut_sincos.cu
 *
 * Rogue wave context:
 *   GS runs in real-time on incoming fiber intensity traces.
 *   GPU kernel: compute H[k] = exp(i*pi*D*nu[k]^2) for all k in parallel.
 *   __constant__ LUT lives in 64KB constant cache -> ~same latency as register.
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define LUT_BITS  10
#define LUT_N     (1 << LUT_BITS)   /* 1024 */
#define LUT_MASK  (LUT_N - 1)

/* constant memory: shared by all threads, broadcast-cached */
__constant__ float d_lut_sin[LUT_N + 1];
__constant__ float d_lut_cos[LUT_N + 1];

/* ── device inline LUT lookup ─────────────────────────────── */
__device__ __forceinline__ float dev_lut_sin(float theta) {
    float idx_f = theta * ((float)LUT_N / (2.0f * (float)M_PI));
    int   idx_i = (int)idx_f;
    float frac  = idx_f - (float)idx_i;
    idx_i       = ((idx_i % LUT_N) + LUT_N) & LUT_MASK;
    int   idx_n = (idx_i + 1) & LUT_MASK;
    return d_lut_sin[idx_i] + frac * (d_lut_sin[idx_n] - d_lut_sin[idx_i]);
}

__device__ __forceinline__ float dev_lut_cos(float theta) {
    float idx_f = theta * ((float)LUT_N / (2.0f * (float)M_PI));
    int   idx_i = (int)idx_f;
    float frac  = idx_f - (float)idx_i;
    idx_i       = ((idx_i % LUT_N) + LUT_N) & LUT_MASK;
    int   idx_n = (idx_i + 1) & LUT_MASK;
    return d_lut_cos[idx_i] + frac * (d_lut_cos[idx_n] - d_lut_cos[idx_i]);
}

/* ── kernel: H[k] = exp(i*pi*D*nu[k]^2), LUT version ───────── */
__global__ void dispersion_kernel_lut(float2 *H, int N, float D) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    float nu    = (k < N/2) ? (float)k/N : (float)(k - N)/N;
    float phase = (float)M_PI * D * nu * nu;
    H[k].x = dev_lut_cos(phase);
    H[k].y = dev_lut_sin(phase);
}

/* ── kernel: H[k] using CUDA intrinsic __sincosf ────────────── */
__global__ void dispersion_kernel_std(float2 *H, int N, float D) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    float nu    = (k < N/2) ? (float)k/N : (float)(k - N)/N;
    float phase = (float)M_PI * D * nu * nu;
    __sincosf(phase, &H[k].y, &H[k].x);   /* CUDA fast intrinsic */
}

/* ── rogue wave detector kernel ──────────────────────────────── */
/*   Scan intensity trace I[n], flag any sample > threshold*mean */
__global__ void rogue_detect(
    const float *I, int N, float threshold,
    int *flag_count, int *flag_indices, int max_flags)
{
    __shared__ float s_sum;
    __shared__ int   s_count;

    /* step 1: reduce mean in shared memory */
    if (threadIdx.x == 0) { s_sum = 0.0f; s_count = 0; }
    __syncthreads();

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) atomicAdd(&s_sum, I[k]);
    __syncthreads();

    float mean = s_sum / N;

    /* step 2: flag rogues */
    if (k < N && I[k] > threshold * mean) {
        int slot = atomicAdd(&s_count, 1);
        if (slot < max_flags)
            flag_indices[slot] = k;
    }
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(flag_count, s_count);
}

/* ── host helpers ─────────────────────────────────────────── */
static void build_lut_host(float *h_sin, float *h_cos) {
    for (int i = 0; i <= LUT_N; i++) {
        double th = 2.0 * M_PI * i / LUT_N;
        h_sin[i]  = (float)sin(th);
        h_cos[i]  = (float)cos(th);
    }
}

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

int main(void) {
    printf("CUDA LUT sin/cos (N=%d, __constant__ memory)\n", LUT_N);
    printf("----------------------------------------------\n\n");

    /* upload LUT to constant memory */
    float h_sin[LUT_N + 1], h_cos[LUT_N + 1];
    build_lut_host(h_sin, h_cos);
    CUDA_CHECK(cudaMemcpyToSymbol(d_lut_sin, h_sin, (LUT_N+1)*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_lut_cos, h_cos, (LUT_N+1)*sizeof(float)));

    /* allocate device memory */
    int N = 512;
    float2 *d_H_lut, *d_H_std;
    CUDA_CHECK(cudaMalloc(&d_H_lut, N * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_H_std, N * sizeof(float2)));

    dim3 block(128), grid((N + 127)/128);

    /* warm up */
    dispersion_kernel_lut<<<grid, block>>>(d_H_lut, N, 5000.0f);
    dispersion_kernel_std<<<grid, block>>>(d_H_std, N, 5000.0f);
    cudaDeviceSynchronize();

    /* timing */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int reps = 10000;
    float ms_lut, ms_std;

    cudaEventRecord(t0);
    for (int r = 0; r < reps; r++)
        dispersion_kernel_lut<<<grid, block>>>(d_H_lut, N, 5000.0f);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_lut, t0, t1);
    ms_lut /= reps;

    cudaEventRecord(t0);
    for (int r = 0; r < reps; r++)
        dispersion_kernel_std<<<grid, block>>>(d_H_std, N, 5000.0f);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_std, t0, t1);
    ms_std /= reps;

    printf("Dispersion kernel N=%d:\n", N);
    printf("  LUT (__constant__): %.4f ms\n", ms_lut);
    printf("  __sincosf:          %.4f ms\n", ms_std);
    printf("  Note: __sincosf is a hardware intrinsic; LUT is mainly\n");
    printf("        useful on non-GPU targets (FPGA, embedded ARM).\n\n");

    /* verify accuracy */
    float2 *h_lut = (float2*)malloc(N*sizeof(float2));
    float2 *h_std = (float2*)malloc(N*sizeof(float2));
    CUDA_CHECK(cudaMemcpy(h_lut, d_H_lut, N*sizeof(float2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_std, d_H_std, N*sizeof(float2), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int k = 0; k < N; k++) {
        float e = fmaxf(fabsf(h_lut[k].x - h_std[k].x),
                        fabsf(h_lut[k].y - h_std[k].y));
        if (e > max_err) max_err = e;
    }
    printf("Max error LUT vs __sincosf: %.2e\n\n", max_err);

    /* rogue wave detector demo */
    printf("Rogue wave detector kernel:\n");
    float *h_I = (float*)malloc(N*sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) h_I[i] = 1.0f + 0.1f*(float)rand()/RAND_MAX;
    h_I[73]  = 8.5f;   /* inject rogue wave at sample 73  */
    h_I[200] = 9.2f;   /* inject rogue wave at sample 200 */

    float *d_I; int *d_flags, *d_count;
    CUDA_CHECK(cudaMalloc(&d_I, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flags, 32*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_I, h_I, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flags, -1, 32*sizeof(int)));

    rogue_detect<<<1, 512>>>(d_I, N, 5.0f, d_count, d_flags, 32);
    cudaDeviceSynchronize();

    int h_count; int h_flags[32];
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flags, d_flags, 32*sizeof(int), cudaMemcpyDeviceToHost));

    printf("  Threshold: 5x mean\n");
    printf("  Rogues detected: %d\n", h_count);
    for (int i = 0; i < h_count && i < 32; i++)
        printf("  -> sample %d  I=%.2f\n", h_flags[i], h_I[h_flags[i]]);

    /* cleanup */
    free(h_lut); free(h_std); free(h_I);
    cudaFree(d_H_lut); cudaFree(d_H_std);
    cudaFree(d_I); cudaFree(d_flags); cudaFree(d_count);
    return 0;
}
