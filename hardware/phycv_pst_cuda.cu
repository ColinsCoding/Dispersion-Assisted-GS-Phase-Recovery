// phycv_pst_cuda.cu -- the Phase Stretch Transform (PST), a real CUDA
// implementation, cross-checked against the ACTUAL phycv Python library's
// output on the identical image and parameters -- not a from-scratch
// derivation checked only against itself, but a reimplementation checked
// against a published, already-vetted library (see dgs/phase_stretch_
// crack_detection.py, which wraps that same library from Python).
//
// PST IS THIS REPO'S DISPERSION KERNEL, APPLIED TO AN IMAGE: FFT -> multiply
// by an engineered phase kernel -> IFFT -> read off the phase. dgs/gs_core's
// H(nu)=exp(i*pi*D*nu^2) does exactly this to a 1D temporal pulse to
// recover PHASE FROM INTENSITY; PST does the same three-step operation to a
// 2D spatial image to turn phase information INTO a visible edge map. Same
// structure, opposite direction: gs_core "phase-retrieves" (recovers a
// hidden phase from measured intensity); PST "phase-stretches" (injects an
// engineered phase, then reads amplitude/edge structure out of it).
//
// EXACT ALGORITHM (matched line-for-line against phycv's PST.init_kernel /
// PST.apply_kernel, see phycv/pst.py):
//   1. u,v in linspace(-0.5,0.5,N), rho = sqrt(u^2+v^2)   (a "centered" grid)
//   2. denoise LPF: fftshift(exp(-0.5*(rho/sqrt(sigma_LPF^2/ln2))^2))
//   3. pst_kernel = W*rho*atan(W*rho) - 0.5*log(1+(W*rho)^2), then
//      S * pst_kernel / max(pst_kernel)
//   4. img_denoised = real(IFFT2(FFT2(img) * LPF))
//   5. img_pst = IFFT2(FFT2(img_denoised) * fftshift(exp(-i*pst_kernel)))
//   6. feature = normalize(angle(img_pst)) to [0,1]
//
// Build & run (reads pst_ref_image.txt / pst_ref_feature.txt -- generate
// both first with `py -3.13 scripts/generate_pst_cuda_reference.py`, run
// from the repo root, before building this file):
//   nvcc -O2 -o phycv_pst_cuda phycv_pst_cuda.cu -lcufft
//   ./phycv_pst_cuda

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define CUFFT_CHECK(call) do { \
    cufftResult err = (call); \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error at %s:%d: code %d\n", __FILE__, __LINE__, (int)err); \
        exit(1); \
    } \
} while (0)

__global__ void multiply_real_kernel(cufftDoubleComplex *data, const double *kernel, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i].x *= kernel[i];
    data[i].y *= kernel[i];
}

__global__ void multiply_complex_kernel(cufftDoubleComplex *data, const cufftDoubleComplex *kernel, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double re = data[i].x * kernel[i].x - data[i].y * kernel[i].y;
    double im = data[i].x * kernel[i].y + data[i].y * kernel[i].x;
    data[i].x = re;
    data[i].y = im;
}

__global__ void scale_kernel(cufftDoubleComplex *data, double scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i].x *= scale;
    data[i].y *= scale;
}

__global__ void take_real_as_complex_kernel(cufftDoubleComplex *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i].y = 0.0;   // discard the (numerically ~0) imaginary residue
}

__global__ void angle_kernel(const cufftDoubleComplex *data, double *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = atan2(data[i].y, data[i].x);
}

int main() {
    const int N = 128;
    const double S = 0.5, W = 15.0, sigma_LPF = 0.1;
    const int n_pix = N * N;

    // ---- read the reference image (identical to the Python script's) ----
    FILE *f_img = fopen("pst_ref_image.txt", "r");
    if (!f_img) { fprintf(stderr, "cannot open pst_ref_image.txt -- run the paired Python script first\n"); return 1; }
    double *h_img = (double*)malloc(n_pix * sizeof(double));
    for (int i = 0; i < n_pix; ++i) fscanf(f_img, "%lf", &h_img[i]);
    fclose(f_img);

    FILE *f_ref = fopen("pst_ref_feature.txt", "r");
    if (!f_ref) { fprintf(stderr, "cannot open pst_ref_feature.txt\n"); return 1; }
    double *h_ref_feature = (double*)malloc(n_pix * sizeof(double));
    for (int i = 0; i < n_pix; ++i) fscanf(f_ref, "%lf", &h_ref_feature[i]);
    fclose(f_ref);

    // ---- build the LPF and PST kernels on the host, EXACTLY matching
    // phycv's centered-grid-then-fftshift construction ----
    double *h_lpf_centered = (double*)malloc(n_pix * sizeof(double));
    double *h_pst_centered = (double*)malloc(n_pix * sizeof(double));
    double sigma_scaled = sqrt((sigma_LPF * sigma_LPF) / log(2.0));
    double pst_max = -1e300;

    for (int i = 0; i < N; ++i) {
        double u = -0.5 + 1.0 * i / (N - 1);
        for (int j = 0; j < N; ++j) {
            double v = -0.5 + 1.0 * j / (N - 1);
            double rho = sqrt(u * u + v * v);
            int idx = i * N + j;
            h_lpf_centered[idx] = exp(-0.5 * pow(rho / sigma_scaled, 2.0));
            double k = W * rho * atan(W * rho) - 0.5 * log(1.0 + (W * rho) * (W * rho));
            h_pst_centered[idx] = k;
            if (k > pst_max) pst_max = k;
        }
    }
    for (int i = 0; i < n_pix; ++i) h_pst_centered[i] = S * h_pst_centered[i] / pst_max;

    // fftshift: swap quadrants so index [0,0] holds the array's CENTER
    // value -- matches numpy's fftshift applied before multiplying against
    // fft2()'s natural (DC-at-[0,0]) output layout.
    double *h_lpf_shifted = (double*)malloc(n_pix * sizeof(double));
    cufftDoubleComplex *h_pst_kernel_shifted = (cufftDoubleComplex*)malloc(n_pix * sizeof(cufftDoubleComplex));
    int half = N / 2;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int si = (i + half) % N;
            int sj = (j + half) % N;
            int src = i * N + j;
            int dst = si * N + sj;
            h_lpf_shifted[dst] = h_lpf_centered[src];
            double phase = -h_pst_centered[src];   // exp(-i*pst_kernel)
            h_pst_kernel_shifted[dst].x = cos(phase);
            h_pst_kernel_shifted[dst].y = sin(phase);
        }
    }

    // ---- upload to device ----
    cufftDoubleComplex *d_data;
    double *d_lpf, *d_angle;
    cufftDoubleComplex *d_pst_kernel;
    CUDA_CHECK(cudaMalloc(&d_data, n_pix * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_lpf, n_pix * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pst_kernel, n_pix * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_angle, n_pix * sizeof(double)));

    cufftDoubleComplex *h_data = (cufftDoubleComplex*)malloc(n_pix * sizeof(cufftDoubleComplex));
    for (int i = 0; i < n_pix; ++i) { h_data[i].x = h_img[i]; h_data[i].y = 0.0; }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n_pix * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lpf, h_lpf_shifted, n_pix * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pst_kernel, h_pst_kernel_shifted, n_pix * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, N, N, CUFFT_Z2Z));

    int threads = 256;
    int blocks = (n_pix + threads - 1) / threads;

    // ---- step 1: denoise (FFT -> multiply LPF -> IFFT -> take real part) ----
    CUFFT_CHECK(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));
    multiply_real_kernel<<<blocks, threads>>>(d_data, d_lpf, n_pix);
    CUFFT_CHECK(cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE));
    scale_kernel<<<blocks, threads>>>(d_data, 1.0 / n_pix, n_pix);   // cuFFT does NOT normalize the inverse transform
    take_real_as_complex_kernel<<<blocks, threads>>>(d_data, n_pix);

    // ---- step 2: apply PST kernel (FFT -> multiply -> IFFT -> angle) ----
    CUFFT_CHECK(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));
    multiply_complex_kernel<<<blocks, threads>>>(d_data, d_pst_kernel, n_pix);
    CUFFT_CHECK(cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE));
    scale_kernel<<<blocks, threads>>>(d_data, 1.0 / n_pix, n_pix);
    angle_kernel<<<blocks, threads>>>(d_data, d_angle, n_pix);
    CUDA_CHECK(cudaDeviceSynchronize());

    double *h_angle = (double*)malloc(n_pix * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_angle, d_angle, n_pix * sizeof(double), cudaMemcpyDeviceToHost));

    // ---- normalize to [0,1], same as phycv's normalize() ----
    double amin = 1e300, amax = -1e300;
    for (int i = 0; i < n_pix; ++i) { if (h_angle[i] < amin) amin = h_angle[i]; if (h_angle[i] > amax) amax = h_angle[i]; }
    double *h_feature = (double*)malloc(n_pix * sizeof(double));
    for (int i = 0; i < n_pix; ++i) h_feature[i] = (h_angle[i] - amin) / (amax - amin);

    // ---- compare against the real phycv Python output ----
    double max_abs_err = 0.0;
    for (int i = 0; i < n_pix; ++i) {
        double err = fabs(h_feature[i] - h_ref_feature[i]);
        if (err > max_abs_err) max_abs_err = err;
    }
    printf("PST via CUDA/cuFFT vs. real phycv.PST (Python) on the same %dx%d image:\n", N, N);
    printf("max |CUDA - phycv| on normalized [0,1] feature map: %.4e\n", max_abs_err);
    printf("PASS threshold 0.01: %s\n", max_abs_err < 0.01 ? "PASS" : "FAIL");

    // write the CUDA-computed feature map so a notebook can display it
    // directly next to the phycv reference, not just the scalar error
    FILE *f_out = fopen("pst_cuda_output.txt", "w");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) fprintf(f_out, "%.10f ", h_feature[i * N + j]);
        fprintf(f_out, "\n");
    }
    fclose(f_out);

    free(h_img); free(h_ref_feature); free(h_lpf_centered); free(h_pst_centered);
    free(h_lpf_shifted); free(h_pst_kernel_shifted); free(h_data); free(h_angle); free(h_feature);
    cufftDestroy(plan);
    cudaFree(d_data); cudaFree(d_lpf); cudaFree(d_pst_kernel); cudaFree(d_angle);

    return max_abs_err < 0.01 ? 0 : 1;
}
