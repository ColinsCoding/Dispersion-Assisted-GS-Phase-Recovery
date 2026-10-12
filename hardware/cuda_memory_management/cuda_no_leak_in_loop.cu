/* cuda_no_leak_in_loop.cu -- identical to cuda_leak_in_loop.cu except for
 * ONE added line: cudaFree(d_ptr) at the end of each iteration. That one
 * line is the entire difference between GPU memory monotonically
 * shrinking and staying flat across the same number of allocations --
 * this is what "no garbage collector" actually means in practice: not
 * that memory is unmanageable, but that the discipline is 100% manual
 * and there is no safety net if you forget it. */
#include <cstdio>
#include <cuda_runtime.h>

#define BUF_MB 100
#define N_ITERS 15

int main() {
    size_t bytes = (size_t)BUF_MB * 1000 * 1000;

    size_t free0, total0;
    cudaMemGetInfo(&free0, &total0);
    printf("start: FREE_MB=%.1f TOTAL_MB=%.1f\n", free0 / 1e6, total0 / 1e6);

    for (int i = 0; i < N_ITERS; i++) {
        void *d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, bytes);
        if (err != cudaSuccess) {
            printf("iter %2d: cudaMalloc FAILED (%s)\n", i, cudaGetErrorString(err));
            break;
        }

        cudaFree(d_ptr);   /* the ONE line that's different from cuda_leak_in_loop.cu */

        size_t free_b, total_b;
        cudaMemGetInfo(&free_b, &total_b);
        printf("iter %2d: allocated + freed %d MB, FREE_MB=%.1f (drift from start: %.1f MB)\n",
               i, BUF_MB, free_b / 1e6, ((long long)free0 - (long long)free_b) / 1e6);
    }

    size_t free_end, total_end;
    cudaMemGetInfo(&free_end, &total_end);
    printf("\nend: FREE_MB=%.1f  (net drift from start: %.1f MB, across %d allocate+free cycles)\n",
           free_end / 1e6, ((long long)free0 - (long long)free_end) / 1e6, N_ITERS);
    return 0;
}
