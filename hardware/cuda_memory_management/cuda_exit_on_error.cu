/* cuda_exit_on_error.cu -- reproduces hardware/phycv_pst_cuda.cu's own
 * CUDA_CHECK macro verbatim (see that file, lines 39-45): on any CUDA
 * error, print and exit(1) immediately. That file allocates FOUR buffers
 * with CUDA_CHECK back-to-back; if a later cudaMalloc ever failed, an
 * earlier buffer's cudaFree at the bottom of the function would never run.
 *
 * NOTE ON A WRONG ASSUMPTION, kept visible rather than fixed silently:
 * the first version of this file tried to force a guaranteed failure by
 * requesting a single allocation of 2x total device memory. On this
 * machine (Windows, WDDM driver), that allocation actually SUCCEEDED --
 * WDDM transparently pages "GPU" memory to system RAM under pressure, so
 * a single oversized cudaMalloc is not a reliable way to trigger
 * cudaErrorMemoryAllocation. Fixed by allocating in a loop of realistic
 * chunks until cudaMalloc genuinely fails (a real resource-exhaustion
 * failure, not a guessed magic number) -- the way this actually happens
 * in a real long-running program, and honest about which assumption was
 * wrong and why.
 *
 * The question this file exists to let a SEPARATE process answer (run
 * cuda_mem_probe.exe before and after this program): does the memory
 * from the successful earlier allocations actually stay leaked once this
 * process is gone, or does the OS/CUDA driver reclaim it on exit? */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

int main() {
    size_t free0, total0;
    cudaMemGetInfo(&free0, &total0);
    printf("start: FREE_MB=%.1f TOTAL_MB=%.1f\n", free0 / 1e6, total0 / 1e6);

    void *d_buf1 = nullptr;
    size_t buf1_bytes = (size_t)500 * 1000 * 1000;   /* 500 MB -- succeeds */
    CUDA_CHECK(cudaMalloc(&d_buf1, buf1_bytes));

    size_t free1, total1;
    cudaMemGetInfo(&free1, &total1);
    printf("after buf1 (500 MB, successful): FREE_MB=%.1f\n", free1 / 1e6);

    /* Keep allocating realistic-sized chunks, never freeing any of them,
     * until cudaMalloc actually fails -- a genuine resource-exhaustion
     * failure, not an assumed size. SAFETY CAP: this machine's WDDM
     * driver pages GPU allocations to system RAM under pressure (see the
     * note above), and this machine has limited free system RAM -- so
     * this loop is capped well below that, and reports honestly if it
     * hits the cap without ever seeing a failure, rather than continuing
     * unboundedly on a shared development machine. */
    const int MAX_CHUNKS = 6;   /* 6 x 1 GB = 6 GB, safely under this machine's free system RAM */
    std::vector<void *> chunks;
    size_t chunk_bytes = (size_t)1000 * 1000 * 1000;   /* 1 GB per chunk */
    int chunk_count = 0;
    bool hit_failure = false;
    for (; chunk_count < MAX_CHUNKS; chunk_count++) {
        void *d_chunk = nullptr;
        cudaError_t chunk_err = cudaMalloc(&d_chunk, chunk_bytes);
        if (chunk_err != cudaSuccess) {
            printf("chunk %d: cudaMalloc(1 GB) FAILED (%s) after %d successful 1 GB chunks --\n",
                   chunk_count, cudaGetErrorString(chunk_err), chunk_count);
            printf("this is the real failure CUDA_CHECK below will turn into exit(1)\n");
            hit_failure = true;
            /* NOTE: deliberately NOT "CUDA_CHECK(chunk_err)" here -- passing an
             * ALREADY-EVALUATED variable into this macro is a real bug this file's
             * first draft actually hit: the macro's own body declares
             * "cudaError_t err = (call)", so CUDA_CHECK(err) with an outer variable
             * also named err shadows it, evaluating "err = (err)" -- an
             * uninitialized self-reference (caught by the compiler's own
             * C4700 warning, not by inspection). CUDA_CHECK must always be
             * called directly on the failing call expression, e.g.
             * CUDA_CHECK(cudaMalloc(&d_chunk, chunk_bytes)) -- never on a
             * variable already holding that call's result. */
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(chunk_err));
            exit(1);   /* triggers here -- buf1 and every chunk's cudaFree below never runs */
        }
        chunks.push_back(d_chunk);
    }
    if (!hit_failure) {
        printf("safety cap of %d x 1 GB chunks reached without a cudaMalloc failure --\n", MAX_CHUNKS);
        printf("this machine's GPU+pageable memory absorbed %d GB without erroring; not forcing further\n",
               MAX_CHUNKS);
    }

    /* unreachable: exactly the same shape of bug as phycv_pst_cuda.cu's
     * "cudaFree(d_data); ..." at its end, if an earlier allocation had failed */
    cudaFree(d_buf1);
    for (void *p : chunks) cudaFree(p);
    return 0;
}
