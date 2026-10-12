// materials.cu -- evaluate n and beta2 for every material on a wavelength grid, one GPU thread per point.
//   nvcc -O2 -arch=sm_89 -I../c/materials materials.cu -o materials_cuda
//   ./materials_cuda out.bin npts lam_lo_um lam_hi_um
// Writes MAT_COUNT * npts * 2 doubles: for each material, n[npts] then beta2[npts] (ps^2/km).
// Points outside a material's valid range are written as NaN. Prints the kernel time in ms.
// Exit codes: 0 ok, 2 bad arguments, 3 CUDA error, 4 cannot write output.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "materials.h"

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e_)); return 3; } } while (0)

__constant__ double dB[MAT_COUNT*3], dC[MAT_COUNT*3], dLo[MAT_COUNT], dHi[MAT_COUNT];

__global__ void eval_grid(double *out, int npts, double lo, double hi) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (k >= npts) return;
    double lam = lo + (hi - lo)*k/(npts - 1);
    double *n_out = out + (size_t)m*2*npts, *b_out = n_out + npts;
    if (lam < dLo[m] || lam > dHi[m]) { n_out[k] = NAN; b_out[k] = NAN; return; }
    n_out[k] = mat_n(dB + 3*m, dC + 3*m, lam);
    b_out[k] = mat_beta2(dB + 3*m, dC + 3*m, lam);
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: materials_cuda out.bin npts lam_lo_um lam_hi_um\n"); return 2; }
    int npts = atoi(argv[2]); double lo = atof(argv[3]), hi = atof(argv[4]);
    if (npts < 2 || !(hi > lo) || lo <= 0) { fprintf(stderr, "need npts >= 2 and 0 < lo < hi\n"); return 2; }
    double hB[MAT_COUNT*3], hC[MAT_COUNT*3], hLo[MAT_COUNT], hHi[MAT_COUNT];
    for (int m = 0; m < MAT_COUNT; m++) {
        for (int i = 0; i < 3; i++) { hB[3*m + i] = MATERIALS[m].B[i]; hC[3*m + i] = MATERIALS[m].C[i]; }
        hLo[m] = MATERIALS[m].lam_min_um; hHi[m] = MATERIALS[m].lam_max_um;
    }
    CHECK(cudaMemcpyToSymbol(dB, hB, sizeof hB)); CHECK(cudaMemcpyToSymbol(dC, hC, sizeof hC));
    CHECK(cudaMemcpyToSymbol(dLo, hLo, sizeof hLo)); CHECK(cudaMemcpyToSymbol(dHi, hHi, sizeof hHi));
    size_t bytes = (size_t)MAT_COUNT*2*npts*sizeof(double);
    double *d_out; CHECK(cudaMalloc(&d_out, bytes));
    dim3 threads(256), blocks((npts + 255)/256, MAT_COUNT);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    eval_grid<<<blocks, threads>>>(d_out, npts, lo, hi);
    cudaEventRecord(t1); CHECK(cudaGetLastError()); CHECK(cudaEventSynchronize(t1));
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    double *h_out = (double *)malloc(bytes); if (!h_out) return 3;
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    FILE *f = fopen(argv[1], "wb"); if (!f) return 4;
    if (fwrite(h_out, 1, bytes, f) != bytes) { fclose(f); return 4; }
    fclose(f); printf("%.4f\n", ms);
    cudaFree(d_out); free(h_out);
    return 0;
}
