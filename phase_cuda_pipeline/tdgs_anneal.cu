/*
 * tdgs_anneal.cu  —  Dispersion-Assisted Phase Retrieval, 6-Stage Annealing
 *
 * Runs for ~40 minutes.  Each stage increases grid resolution and tightens
 * the convergence tolerance.  The solution from stage N seeds stage N+1, so
 * quality strictly improves over time.
 *
 * Build:
 *   nvcc -O3 -arch=sm_86 -lcufft tdgs_anneal.cu -o tdgs_anneal
 * Run:
 *   ./tdgs_anneal
 *
 * Requires: CUDA toolkit ≥ 11, cuFFT
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include <cufft.h>

/* ── compile-time constants ────────────────────────────────────────── */
#define RUN_SECONDS   2400          /* 40 minutes                        */
#define NUM_STAGES    6
#define ALPHA         2.515e-5      /* π·λ₀²/c  [nm·ps/GHz²] at 1550 nm */
#define D1_PS_NM      0.0
#define D2_PS_NM      500.0
#define PI            3.14159265358979323846

/* per-stage config */
static const int    STAGE_N[]   = { 256, 512, 1024, 2048, 4096, 8192 };
static const int    STAGE_ITER[]= { 50,  100, 200,  400,  800,  2000 };
static const double STAGE_TOL[] = { 1e-3,1e-4,1e-5, 1e-6, 1e-7, 1e-9 };

/* ── CUDA error helper ─────────────────────────────────────────────── */
#define CUDA_CHECK(x) do {                                                \
    cudaError_t _e = (x);                                                 \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr,"CUDA error %s:%d: %s\n",                         \
                __FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); }     \
} while(0)
#define CUFFT_CHECK(x) do {                                               \
    cufftResult _r = (x);                                                 \
    if (_r != CUFFT_SUCCESS) {                                            \
        fprintf(stderr,"cuFFT error %s:%d: %d\n",__FILE__,__LINE__,_r);  \
        exit(1); }                                                        \
} while(0)

/* ── kernels ───────────────────────────────────────────────────────── */

/* build dispersion kernel: K[k] = exp(i·α·D·f[k]²) */
__global__ void k_build_kernel(cufftDoubleComplex *K,
                                const double *f_GHz, int N,
                                double alpha, double D_net)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    double phase = alpha * D_net * f_GHz[k] * f_GHz[k];
    K[k].x = cos(phase);
    K[k].y = sin(phase);
}

/* element-wise complex multiply: C = A * B */
__global__ void k_cmul(cufftDoubleComplex *C,
                        const cufftDoubleComplex *A,
                        const cufftDoubleComplex *B, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    double ar = A[k].x, ai = A[k].y;
    double br = B[k].x, bi = B[k].y;
    C[k].x = ar*br - ai*bi;
    C[k].y = ar*bi + ai*br;
}

/* replace magnitude with sqrt(I_meas), keep phase: E ← sqrt(I)·E/|E| */
__global__ void k_replace_mag(cufftDoubleComplex *E,
                               const double *I_meas, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    double re = E[k].x, im = E[k].y;
    double mag = sqrt(re*re + im*im);
    if (mag < 1e-30) mag = 1e-30;
    double scale = sqrt(I_meas[k]) / mag;
    E[k].x = re * scale;
    E[k].y = im * scale;
}

/* scale FFT output by 1/N */
__global__ void k_scale(cufftDoubleComplex *E, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    E[k].x /= N;
    E[k].y /= N;
}

/* compute residual: sum |E|² - I_meas  (writes per-element to scratch) */
__global__ void k_residual(double *res,
                            const cufftDoubleComplex *E,
                            const double *I_meas, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    double re = E[k].x, im = E[k].y;
    double diff = (re*re + im*im) - I_meas[k];
    res[k] = diff * diff;
}

/* fftshift: swap first and second halves */
__global__ void k_fftshift(cufftDoubleComplex *out,
                            const cufftDoubleComplex *in, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    int half = N / 2;
    int src  = (k < half) ? (k + half) : (k - half);
    out[k]   = in[src];
}

/* ── host helpers ──────────────────────────────────────────────────── */

static double wall_seconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* build a synthetic gas-cell spectrum: Gaussian envelope + Lorentzian dips */
static void make_synthetic(double *I1_h, double *I2_h,
                            double *f_h, int N)
{
    double f_min = -150.0, f_max = 150.0;  /* GHz */
    for (int k = 0; k < N; k++) {
        double f = f_min + (f_max - f_min) * k / (N - 1);
        f_h[k] = f;
        /* Gaussian pulse envelope */
        double env = exp(-f*f / (2.0 * 60.0 * 60.0));
        /* three absorption lines */
        double abs1 = 1.0 - 0.8 / (1.0 + (f - 20.0)*(f - 20.0) / 4.0);
        double abs2 = 1.0 - 0.6 / (1.0 + (f + 35.0)*(f + 35.0) / 9.0);
        double abs3 = 1.0 - 0.4 / (1.0 + (f -  5.0)*(f -  5.0) / 1.0);
        double mag  = env * abs1 * abs2 * abs3;
        /* true phase: quadratic + sine wobble */
        double phi  = 0.3*f*f*1e-4 + 0.5*sin(2*PI*f/80.0);
        /* dispersed field at D2 */
        double dphase = ALPHA * D2_PS_NM * f * f;
        I1_h[k] = mag * mag;
        /* simple approximation for I2 — real application uses actual propagation */
        double re2 = mag * cos(phi + dphase);
        double im2 = mag * sin(phi + dphase);
        I2_h[k] = re2*re2 + im2*im2;
    }
}

/* CPU-side thrust-free reduction: sum array of length N */
static double cpu_sum(const double *d_arr, int N)
{
    double *h = (double*)malloc(N * sizeof(double));
    cudaMemcpy(h, d_arr, N * sizeof(double), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (int i = 0; i < N; i++) s += h[i];
    free(h);
    return s;
}

/* ── one full TD-GS stage ──────────────────────────────────────────── */

/*
 * Returns RMS residual.  E1_d is initialized on entry (warm-started
 * from previous stage if N matches) and holds the best field on exit.
 */
static double run_stage(int stage, int N, int n_iter, double tol,
                         double *I1_d, double *I2_d,
                         double *f_d,
                         cufftDoubleComplex *E1_d,   /* in/out */
                         double t_start, double t_deadline)
{
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    /* allocate device working arrays */
    cufftDoubleComplex *E2_d, *Etmp_d, *K12_d, *K21_d, *Esh_d;
    double *res_d;
    CUDA_CHECK(cudaMalloc(&E2_d,  N * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&Etmp_d,N * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&K12_d, N * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&K21_d, N * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&Esh_d, N * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&res_d, N * sizeof(double)));

    /* build dispersion kernels once */
    k_build_kernel<<<blocks,threads>>>(K12_d, f_d, N, ALPHA, D2_PS_NM - D1_PS_NM);
    k_build_kernel<<<blocks,threads>>>(K21_d, f_d, N, ALPHA, D1_PS_NM - D2_PS_NM);

    /* cuFFT plan */
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));

    double best_rms = 1e99, prev_rms = 1e99;
    cufftDoubleComplex *Ebest_d;
    CUDA_CHECK(cudaMalloc(&Ebest_d, N * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(Ebest_d, E1_d,
                          N * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToDevice));

    for (int it = 0; it < n_iter; it++) {

        /* check 40-minute wall clock */
        if (wall_seconds() - t_start >= t_deadline) {
            printf("  [stage %d] wall-clock deadline reached at iter %d\n",
                   stage+1, it);
            break;
        }

        /* ── propagate E1 → E2: E2 = IFFT(FFT(E1) * K12) ── */
        /* fftshift(E1) → Esh */
        k_fftshift<<<blocks,threads>>>(Esh_d, E1_d, N);
        /* FFT */
        CUFFT_CHECK(cufftExecZ2Z(plan, Esh_d, Etmp_d, CUFFT_FORWARD));
        k_scale<<<blocks,threads>>>(Etmp_d, N);
        /* multiply kernel */
        k_cmul<<<blocks,threads>>>(E2_d, Etmp_d, K12_d, N);
        /* IFFT */
        CUFFT_CHECK(cufftExecZ2Z(plan, E2_d, Esh_d, CUFFT_INVERSE));
        k_fftshift<<<blocks,threads>>>(E2_d, Esh_d, N);
        /* replace magnitude with sqrt(I2) */
        k_replace_mag<<<blocks,threads>>>(E2_d, I2_d, N);

        /* ── propagate E2 → E1: E1 = IFFT(FFT(E2) * K21) ── */
        k_fftshift<<<blocks,threads>>>(Esh_d, E2_d, N);
        CUFFT_CHECK(cufftExecZ2Z(plan, Esh_d, Etmp_d, CUFFT_FORWARD));
        k_scale<<<blocks,threads>>>(Etmp_d, N);
        k_cmul<<<blocks,threads>>>(E1_d, Etmp_d, K21_d, N);
        CUFFT_CHECK(cufftExecZ2Z(plan, E1_d, Esh_d, CUFFT_INVERSE));
        k_fftshift<<<blocks,threads>>>(E1_d, Esh_d, N);
        /* replace magnitude with sqrt(I1) */
        k_replace_mag<<<blocks,threads>>>(E1_d, I1_d, N);

        /* ── residual at plane 1 ── */
        k_residual<<<blocks,threads>>>(res_d, E1_d, I1_d, N);
        double sse = cpu_sum(res_d, N);
        double rms = sqrt(sse / N);

        if (rms < best_rms) {
            best_rms = rms;
            CUDA_CHECK(cudaMemcpy(Ebest_d, E1_d,
                                  N * sizeof(cufftDoubleComplex),
                                  cudaMemcpyDeviceToDevice));
        }

        /* print every 10 iters */
        if ((it % 10) == 0) {
            double elapsed = wall_seconds() - t_start;
            printf("  stage %d  iter %4d/%d  rms=%.3e  best=%.3e  "
                   "elapsed=%5.0fs / %ds\n",
                   stage+1, it, n_iter, rms, best_rms,
                   elapsed, RUN_SECONDS);
            fflush(stdout);
        }

        /* early stop */
        if (tol > 0.0 && it > 0 && fabs(prev_rms - rms) < tol) {
            printf("  stage %d converged at iter %d  rms=%.3e\n",
                   stage+1, it, rms);
            break;
        }
        prev_rms = rms;
    }

    /* copy best back into E1_d */
    CUDA_CHECK(cudaMemcpy(E1_d, Ebest_d,
                          N * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToDevice));

    cufftDestroy(plan);
    cudaFree(E2_d); cudaFree(Etmp_d);
    cudaFree(K12_d); cudaFree(K21_d);
    cudaFree(Esh_d); cudaFree(res_d);
    cudaFree(Ebest_d);

    return best_rms;
}

/* ── upsample E to a larger grid (zero-pad in frequency domain) ──── */
static void upsample_field(cufftDoubleComplex *E_out, int N_out,
                            const cufftDoubleComplex *E_in,  int N_in)
{
    /* copy host → CPU, zero-pad centre, copy back */
    size_t sz_in  = N_in  * sizeof(cufftDoubleComplex);
    size_t sz_out = N_out * sizeof(cufftDoubleComplex);
    cufftDoubleComplex *h_in  = (cufftDoubleComplex*)malloc(sz_in);
    cufftDoubleComplex *h_out = (cufftDoubleComplex*)calloc(N_out, sizeof(cufftDoubleComplex));
    cudaMemcpy(h_in, E_in, sz_in, cudaMemcpyDeviceToHost);
    /* copy lower half and upper half with zero-padding in between */
    int half_in  = N_in  / 2;
    int half_out = N_out / 2;
    memcpy(h_out,                     h_in,             half_in * sizeof(cufftDoubleComplex));
    memcpy(h_out + N_out - half_in,   h_in + half_in,   half_in * sizeof(cufftDoubleComplex));
    (void)half_out; /* padding zeros already from calloc */
    cudaMemcpy(E_out, h_out, sz_out, cudaMemcpyHostToDevice);
    free(h_in); free(h_out);
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void)
{
    /* GPU info */
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TD-GS Phase Retrieval — 6-Stage CUDA Annealing\n");
    printf("  GPU: %s  |  SM %d.%d  |  %.0f MB VRAM\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1048576.0);
    printf("  Run time: %d seconds (%.0f minutes)\n",
           RUN_SECONDS, RUN_SECONDS / 60.0);
    printf("═══════════════════════════════════════════════════════\n\n");

    double t_start   = wall_seconds();
    double t_deadline = (double)RUN_SECONDS;

    /* maximum N we'll ever need */
    int N_max = STAGE_N[NUM_STAGES - 1];

    /* allocate max-size host arrays */
    double *f_h  = (double*)malloc(N_max * sizeof(double));
    double *I1_h = (double*)malloc(N_max * sizeof(double));
    double *I2_h = (double*)malloc(N_max * sizeof(double));

    /* device arrays at current stage size — reallocated each stage */
    double             *I1_d = NULL, *I2_d = NULL, *f_d = NULL;
    cufftDoubleComplex *E1_d  = NULL;
    int                 N_prev = 0;

    double stage_rms[NUM_STAGES];

    for (int s = 0; s < NUM_STAGES; s++) {

        /* time budget: distribute remaining time across remaining stages */
        double t_elapsed   = wall_seconds() - t_start;
        double t_remaining = t_deadline - t_elapsed;
        if (t_remaining <= 0.0) {
            printf("Out of time before stage %d — stopping.\n", s+1);
            break;
        }

        int    N       = STAGE_N[s];
        int    n_iter  = STAGE_ITER[s];
        double tol     = STAGE_TOL[s];

        printf("┌─ Stage %d/%d ── N=%d  iters=%d  tol=%.0e ─────────────────\n",
               s+1, NUM_STAGES, N, n_iter, tol);
        printf("│  Elapsed: %.0fs / %ds  (%.1f min remaining)\n",
               t_elapsed, RUN_SECONDS, t_remaining / 60.0);

        /* build synthetic measurements at this resolution */
        make_synthetic(I1_h, I2_h, f_h, N);

        /* allocate/reallocate device arrays */
        if (N != N_prev) {
            if (I1_d) { cudaFree(I1_d); cudaFree(I2_d); cudaFree(f_d); }

            cufftDoubleComplex *E1_new;
            CUDA_CHECK(cudaMalloc(&E1_new, N * sizeof(cufftDoubleComplex)));

            if (E1_d && N_prev > 0) {
                /* warm-start: upsample previous solution */
                printf("│  Upsampling solution from N=%d → N=%d\n", N_prev, N);
                upsample_field(E1_new, N, E1_d, N_prev);
                cudaFree(E1_d);
            } else {
                /* stage 0: random-phase init from I1 */
                cufftDoubleComplex *h_init =
                    (cufftDoubleComplex*)malloc(N * sizeof(cufftDoubleComplex));
                srand(42);
                for (int k = 0; k < N; k++) {
                    double mag   = sqrt(I1_h[k]);
                    double phase = 2.0 * PI * rand() / RAND_MAX;
                    h_init[k].x  = mag * cos(phase);
                    h_init[k].y  = mag * sin(phase);
                }
                CUDA_CHECK(cudaMemcpy(E1_new, h_init,
                                      N * sizeof(cufftDoubleComplex),
                                      cudaMemcpyHostToDevice));
                free(h_init);
            }
            E1_d = E1_new;

            CUDA_CHECK(cudaMalloc(&I1_d, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&I2_d, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&f_d,  N * sizeof(double)));
            N_prev = N;
        }

        CUDA_CHECK(cudaMemcpy(I1_d, I1_h, N*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(I2_d, I2_h, N*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(f_d,  f_h,  N*sizeof(double), cudaMemcpyHostToDevice));

        /* run this stage — it will loop until n_iter or deadline */
        double rms = run_stage(s, N, n_iter, tol,
                                I1_d, I2_d, f_d, E1_d,
                                t_start, t_deadline);
        stage_rms[s] = rms;

        printf("└─ Stage %d done  best_rms=%.4e\n\n", s+1, rms);

        if (wall_seconds() - t_start >= t_deadline) break;

        /* if time remains after early convergence, keep refining */
        double t_left = t_deadline - (wall_seconds() - t_start);
        if (t_left > 60.0 && s == NUM_STAGES - 1) {
            printf("─── Stage 6 extended refinement (%.0fs remaining) ───\n",
                   t_left);
            /* keep running stage 6 until deadline */
            while (wall_seconds() - t_start < t_deadline) {
                double r = run_stage(s, N, 200, 1e-10,
                                     I1_d, I2_d, f_d, E1_d,
                                     t_start, t_deadline);
                printf("  extended pass  rms=%.4e\n", r);
                if (r < stage_rms[s]) stage_rms[s] = r;
            }
        }
    }

    /* ── final report ─────────────────────────────────────────────── */
    double total = wall_seconds() - t_start;
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  Completed in %.1f seconds (%.1f minutes)\n",
           total, total / 60.0);
    printf("  Stage progression (best RMS residual):\n");
    for (int s = 0; s < NUM_STAGES && stage_rms[s] < 1e90; s++)
        printf("    Stage %d  N=%-5d  rms=%.4e\n",
               s+1, STAGE_N[s], stage_rms[s]);
    printf("═══════════════════════════════════════════════════════\n");

    /* cleanup */
    if (E1_d) cudaFree(E1_d);
    if (I1_d) cudaFree(I1_d);
    if (I2_d) cudaFree(I2_d);
    if (f_d)  cudaFree(f_d);
    free(f_h); free(I1_h); free(I2_h);
    return 0;
}
