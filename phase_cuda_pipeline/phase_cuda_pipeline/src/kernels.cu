#include "kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)

__device__ __forceinline__ cufftDoubleComplex cmul(cufftDoubleComplex a, cufftDoubleComplex b) {
    return make_cuDoubleComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__global__ void frequency_hz_kernel(double* f, int n, double df) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f[i] = ((double)i - 0.5 * (double)n) * df;
}

__global__ void frequency_grid_kernel(double* omega, int n, double df) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double f = ((double)i - 0.5 * (double)n) * df;
    omega[i] = 2.0 * M_PI * f;
}

__global__ void init_random_phase_kernel(cufftDoubleComplex* field, const double* amp, int n, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long x = (unsigned long long)i * 2862933555777941757ULL + seed + 3037000493ULL;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    double u = (double)(x & 0xFFFFFF) / (double)0x1000000;
    double ph = 2.0 * M_PI * u - M_PI;
    field[i] = make_cuDoubleComplex(amp[i] * cos(ph), amp[i] * sin(ph));
}

__global__ void apply_gvd_kernel(cufftDoubleComplex* spec, const double* omega, double phi2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double ph = 0.5 * phi2 * omega[i] * omega[i];
    cufftDoubleComplex h = make_cuDoubleComplex(cos(ph), sin(ph));
    spec[i] = cmul(spec[i], h);
}

__global__ void enforce_magnitude_kernel(cufftDoubleComplex* field, const double* amp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double re = field[i].x, im = field[i].y;
    double mag = sqrt(re*re + im*im) + 1e-30;
    double s = amp[i] / mag;
    field[i].x = re * s;
    field[i].y = im * s;
}

__global__ void support_kernel(cufftDoubleComplex* spec, const unsigned char* support, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!support[i]) spec[i] = make_cuDoubleComplex(0.0, 0.0);
}

__global__ void scale_kernel(cufftDoubleComplex* field, double scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    field[i].x *= scale;
    field[i].y *= scale;
}

__global__ void intensity_kernel(const cufftDoubleComplex* field, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double re = field[i].x, im = field[i].y;
    out[i] = re*re + im*im;
}

__global__ void make_support_kernel(unsigned char* support, const double* freq_hz, double lo, double hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    support[i] = (freq_hz[i] >= lo && freq_hz[i] <= hi) ? 1 : 0;
}

static void launch1(int n, void (*dummy)() = nullptr) {}
static inline dim3 blocks_for(int n) { return dim3((n + 255) / 256); }

void launch_frequency_hz(double* freq_hz, int n, double df_hz) { frequency_hz_kernel<<<blocks_for(n),256>>>(freq_hz,n,df_hz); }
void launch_frequency_grid(double* omega, int n, double df_hz) { frequency_grid_kernel<<<blocks_for(n),256>>>(omega,n,df_hz); }
void launch_init_random_phase(cufftDoubleComplex* field, const double* amp, int n, unsigned long long seed) { init_random_phase_kernel<<<blocks_for(n),256>>>(field,amp,n,seed); }
void launch_apply_gvd(cufftDoubleComplex* spec, const double* omega, double phi2, int n) { apply_gvd_kernel<<<blocks_for(n),256>>>(spec,omega,phi2,n); }
void launch_enforce_magnitude(cufftDoubleComplex* field, const double* amp, int n) { enforce_magnitude_kernel<<<blocks_for(n),256>>>(field,amp,n); }
void launch_apply_support(cufftDoubleComplex* spec, const unsigned char* support, int n) { support_kernel<<<blocks_for(n),256>>>(spec,support,n); }
void launch_scale(cufftDoubleComplex* field, double scale, int n) { scale_kernel<<<blocks_for(n),256>>>(field,scale,n); }
void launch_intensity(const cufftDoubleComplex* field, double* out, int n) { intensity_kernel<<<blocks_for(n),256>>>(field,out,n); }
void launch_make_support(unsigned char* support, const double* freq_hz, double lo_hz, double hi_hz, int n) { make_support_kernel<<<blocks_for(n),256>>>(support,freq_hz,lo_hz,hi_hz,n); }
