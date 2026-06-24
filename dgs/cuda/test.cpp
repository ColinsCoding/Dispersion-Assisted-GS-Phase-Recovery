#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "dgs_forward.h"

int main() {
    int N = 8;

    cuFloatComplex h_x_freq[N];
    float h_beta[N];
    float h_mag[N];

    // visible input
    for (int i = 0; i < N; i++) {
        h_x_freq[i] = make_cuFloatComplex(i, -i);
        h_beta[i] = 0.5f * i;  // phase ramp
    }

    cuFloatComplex *d_x_freq;
    float *d_beta, *d_mag;

    cudaMalloc(&d_x_freq, N * sizeof(cuFloatComplex));
    cudaMalloc(&d_beta, N * sizeof(float));
    cudaMalloc(&d_mag, N * sizeof(float));

    cudaMemcpy(d_x_freq, h_x_freq, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, N * sizeof(float), cudaMemcpyHostToDevice);

    // run real kernels
    launch_dispersion_kernel(d_x_freq, d_beta, N);
    launch_mag_square(d_x_freq, d_mag, N);

    cudaMemcpy(h_mag, d_mag, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "MAG SQUARED OUTPUT:\n";
    for (int i = 0; i < N; i++) {
        std::cout << i << ": " << h_mag[i] << "\n";
    }

    cudaFree(d_x_freq);
    cudaFree(d_beta);
    cudaFree(d_mag);
}
