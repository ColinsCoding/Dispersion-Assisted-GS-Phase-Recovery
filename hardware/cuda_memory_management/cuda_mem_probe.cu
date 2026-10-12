/* cuda_mem_probe.cu -- prints current free/total GPU memory and exits.
 * A reusable probe for the other programs in this directory: run it,
 * do something, run it again, and diff the two readings. */
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    size_t free_b, total_b;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("FREE_MB=%.2f TOTAL_MB=%.2f\n", free_b / 1e6, total_b / 1e6);
    return 0;
}
