// dispersion.cu
// CUDA kernels for applying dispersion and computing magnitude squared.

#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void apply_dispersion(
    cuFloatComplex* x_freq,
    const float* __restrict__ beta,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float phase = beta[i];
    float c = cosf(phase);
    float s = sinf(phase);

    cuFloatComplex v = x_freq[i];
    float xr = cuCrealf(v);
    float xi = cuCimagf(v);

    x_freq[i] = make_cuFloatComplex(
        xr * c - xi * s,
        xr * s + xi * c
    );
}

__global__ void mag_square(
    const cuFloatComplex* __restrict__ x,
    float* __restrict__ y,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    cuFloatComplex v = x[i];
    float xr = cuCrealf(v);
    float xi = cuCimagf(v);
    y[i] = xr * xr + xi * xi;
}

// host-callable wrappers

extern "C" void launch_dispersion_kernel(
    cuFloatComplex* d_x_freq,
    const float* d_beta,
    int N
){
    int block = 256;
    int grid = (N + block - 1) / block;

    apply_dispersion<<<grid, block>>>(
        d_x_freq,
        d_beta,
        N
    );
}

extern "C" void launch_mag_square(
    const cuFloatComplex* d_x,
    float* d_y,
    int N
){
    int block = 256;
    int grid = (N + block - 1) / block;

    mag_square<<<grid, block>>>(
        d_x,
        d_y,
        N
    );
}
