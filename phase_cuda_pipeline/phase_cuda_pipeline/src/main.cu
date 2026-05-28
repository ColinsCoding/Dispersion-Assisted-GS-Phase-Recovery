#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include "kernels.cuh"

#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ std::cerr<<"CUDA error: "<<cudaGetErrorString(err)<<"\n"; exit(1);} } while(0)
#define CUFFT_CHECK(x) do { cufftResult err=(x); if(err!=CUFFT_SUCCESS){ std::cerr<<"cuFFT error code: "<<err<<"\n"; exit(1);} } while(0)

static constexpr double C_LIGHT = 2.99792458e8;
static constexpr double LAMBDA0 = 1550e-9;
static constexpr double GHZ = 1e9;

static double phi2_from_D(double D_ps_per_nm) {
    return -(D_ps_per_nm * 1e-12 / 1e-9) * LAMBDA0 * LAMBDA0 / (2.0 * M_PI * C_LIGHT);
}

static void usage() {
    std::cout << "phase_recovery_cuda --N 16384 --iters 250 --restarts 4\n";
}

int main(int argc, char** argv) {
    int N = 1 << 14;
    int iters = 250;
    int restarts = 4;
    double df = 0.10 * GHZ;

    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--N") && i+1<argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1<argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--restarts") && i+1<argc) restarts = atoi(argv[++i]);
        else { usage(); return 1; }
    }

    const double phi2_1 = phi2_from_D(300.0);
    const double phi2_2 = phi2_from_D(1200.0);
    std::cout << "N=" << N << " phi2_1=" << phi2_1 << " phi2_2=" << phi2_2 << "\n";

    // Placeholder measured amplitudes. Replace by loading sqrt(I1), sqrt(I2) from data.
    std::vector<double> h_amp1(N), h_amp2(N);
    for (int i=0; i<N; i++) {
        double x = ((double)i - 0.5*N) / (0.5*N);
        h_amp1[i] = std::exp(-x*x*8.0) * (1.0 + 0.05*std::sin(40.0*x));
        h_amp2[i] = std::exp(-x*x*5.0) * (1.0 + 0.10*std::cos(32.0*x));
    }

    cufftDoubleComplex *d_e1, *d_work;
    double *d_amp1, *d_amp2, *d_omega, *d_freq;
    unsigned char* d_support;
    CUDA_CHECK(cudaMalloc(&d_e1, N*sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_work, N*sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_amp1, N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_amp2, N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_omega, N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_freq, N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_support, N*sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_amp1, h_amp1.data(), N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_amp2, h_amp2.data(), N*sizeof(double), cudaMemcpyHostToDevice));

    launch_frequency_grid(d_omega, N, df);
    launch_frequency_hz(d_freq, N, df);
    launch_make_support(d_support, d_freq, -60.0*GHZ, 60.0*GHZ, N);

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));

    double best_score = 1e300;
    for (int r=0; r<restarts; r++) {
        launch_init_random_phase(d_e1, d_amp1, N, 1234ULL + r);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int it=0; it<iters; it++) {
            // e1 -> spectral -> apply delta GVD -> e2
            CUFFT_CHECK(cufftExecZ2Z(plan, d_e1, d_e1, CUFFT_FORWARD));
            launch_apply_gvd(d_e1, d_omega, phi2_2 - phi2_1, N);
            CUFFT_CHECK(cufftExecZ2Z(plan, d_e1, d_e1, CUFFT_INVERSE));
            launch_scale(d_e1, 1.0 / N, N);
            launch_enforce_magnitude(d_e1, d_amp2, N);

            // e2 -> spectral -> apply reverse delta GVD -> e1
            CUFFT_CHECK(cufftExecZ2Z(plan, d_e1, d_e1, CUFFT_FORWARD));
            launch_apply_gvd(d_e1, d_omega, phi2_1 - phi2_2, N);
            CUFFT_CHECK(cufftExecZ2Z(plan, d_e1, d_e1, CUFFT_INVERSE));
            launch_scale(d_e1, 1.0 / N, N);
            launch_enforce_magnitude(d_e1, d_amp1, N);

            // Optional spectral support at reference plane would go here.
            if (it % 50 == 0) std::cout << "restart " << r << " iter " << it << "\n";
        }
        best_score = std::min(best_score, 0.0);
    }

    std::vector<cufftDoubleComplex> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_e1, N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
    std::cout << "done. sample=" << h_out[N/2].x << "+" << h_out[N/2].y << "i\n";

    cufftDestroy(plan);
    cudaFree(d_e1); cudaFree(d_work); cudaFree(d_amp1); cudaFree(d_amp2); cudaFree(d_omega); cudaFree(d_freq); cudaFree(d_support);
    return 0;
}
