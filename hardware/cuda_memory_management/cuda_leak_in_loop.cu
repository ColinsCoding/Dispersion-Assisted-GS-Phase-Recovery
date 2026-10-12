/* cuda_leak_in_loop.cu -- CUDA has no garbage collector: cudaMalloc'd
 * device memory is NEVER reclaimed until either cudaFree is called or
 * the process exits. This allocates a buffer every iteration and NEVER
 * frees any of them, printing free GPU memory (cudaMemGetInfo) after
 * each allocation -- a real, measured, monotonically shrinking number,
 * not an assertion about what "should" happen.
 *
 * Compare directly against cuda_no_leak_in_loop.cu, identical except for
 * one line (a cudaFree call) -- the entire difference between a program
 * that exhausts GPU memory over a long run and one that doesn't. */
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define BUF_MB 100
#define N_ITERS 15

int main() {
    size_t bytes = (size_t)BUF_MB * 1000 * 1000;
    std::vector<void *> leaked;   /* held onto so the compiler can't "helpfully" reuse anything */

    size_t free0, total0;
    cudaMemGetInfo(&free0, &total0);
    printf("start: FREE_MB=%.1f TOTAL_MB=%.1f\n", free0 / 1e6, total0 / 1e6);

    for (int i = 0; i < N_ITERS; i++) {
        void *d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, bytes);
        if (err != cudaSuccess) {
            printf("iter %2d: cudaMalloc FAILED (%s) -- out of memory, as expected eventually\n",
                   i, cudaGetErrorString(err));
            break;
        }
        leaked.push_back(d_ptr);   /* never freed -- this is the leak */

        size_t free_b, total_b;
        cudaMemGetInfo(&free_b, &total_b);
        printf("iter %2d: allocated %d MB (never freed), FREE_MB=%.1f (dropped %.1f MB from start)\n",
               i, BUF_MB, free_b / 1e6, (free0 - free_b) / 1e6);
    }

    size_t free_end, total_end;
    cudaMemGetInfo(&free_end, &total_end);
    printf("\nend: FREE_MB=%.1f  (dropped %.1f MB total across %zu allocations, none freed)\n",
           free_end / 1e6, (free0 - free_end) / 1e6, leaked.size());
    /* deliberately no cudaFree here -- the leak this file demonstrates */
    return 0;
}
