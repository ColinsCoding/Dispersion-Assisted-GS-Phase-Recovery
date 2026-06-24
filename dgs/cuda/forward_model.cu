// forward_model.cu
// Host-side wrapper for the CUDA kernels + cuFFT forward model.

#include <cuda_runtime.h>
#include <cufft.h>
#include "dispersion.cu"

extern "C" void forward_model(
    cuFloatComplex* d_x_time,
    float* d_beta,
    float* d_y,
    int N,
    cufftHandle plan_fwd
){
    int block = 256;
    int grid  = (N + block - 1) / block;

    // FFT: time → freq
    cufftExecC2C(plan_fwd, d_x_time, d_x_time, CUFFT_FORWARD);

    // Apply dispersion
    apply_dispersion<<<grid, block>>>(d_x_time, d_beta, N);

    // IFFT: freq → time
    cufftExecC2C(plan_fwd, d_x_time, d_x_time, CUFFT_INVERSE);

    // Magnitude square
    mag_square<<<grid, block>>>(d_x_time, d_y, N);
}
extern "C" void launch_forward_model(float* d_out, const float* d_in, int N) {
    // forward_model_kernel<<<...>>>(d_out, d_in, N);
}
