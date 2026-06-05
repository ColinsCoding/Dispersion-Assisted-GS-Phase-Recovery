/*
 * repl/lut_sincos.cpp
 * Sin/cos LUT with linear interpolation.
 * Compile: g++ -O2 -o lut_sincos lut_sincos.cpp -lm
 * Usage:   ./lut_sincos
 *
 * Optical rogue wave context:
 *   GS inner loop calls exp(i*pi*D*nu^2) = cos(...) + i*sin(...)
 *   On FPGA/embedded: replace std::sin with lut_sin -> ~10x faster
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define LUT_BITS   10               /* table size = 2^LUT_BITS = 1024     */
#define LUT_N      (1 << LUT_BITS)  /* 1024 entries                        */
#define LUT_MASK   (LUT_N - 1)      /* 0x3FF for fast mod                  */

/* ── build the table once at startup ─────────────────────── */
static float lut_cos_f[LUT_N + 1];  /* +1 for interpolation wrap          */
static float lut_sin_f[LUT_N + 1];

void lut_build(void) {
    for (int i = 0; i <= LUT_N; i++) {
        double theta = 2.0 * M_PI * i / LUT_N;
        lut_cos_f[i] = (float)cos(theta);
        lut_sin_f[i] = (float)sin(theta);
    }
}

/*
 * lut_sin / lut_cos  — linear interpolation
 *   theta: radians, any range (wrapped internally)
 *   error: O(1/N^2) ~ 6e-6 for N=1024
 */
static inline float lut_sin(float theta) {
    /* map theta -> [0, LUT_N) */
    float idx_f = theta * (LUT_N / (2.0f * (float)M_PI));
    /* fast floor mod: bring into [0, LUT_N) */
    int   idx_i = (int)idx_f;
    float frac  = idx_f - (float)idx_i;
    idx_i       = ((idx_i % LUT_N) + LUT_N) & LUT_MASK;  /* wrap negative */
    int   idx_n = (idx_i + 1) & LUT_MASK;
    /* linear interpolation between table[i] and table[i+1] */
    return lut_sin_f[idx_i] + frac * (lut_sin_f[idx_n] - lut_sin_f[idx_i]);
}

static inline float lut_cos(float theta) {
    float idx_f = theta * (LUT_N / (2.0f * (float)M_PI));
    int   idx_i = (int)idx_f;
    float frac  = idx_f - (float)idx_i;
    idx_i       = ((idx_i % LUT_N) + LUT_N) & LUT_MASK;
    int   idx_n = (idx_i + 1) & LUT_MASK;
    return lut_cos_f[idx_i] + frac * (lut_cos_f[idx_n] - lut_cos_f[idx_i]);
}

/* ── dispersion filter kernel (GS inner loop) ────────────── */
/*   H[k] = exp(i*pi*D*nu[k]^2) = cos(pi*D*nu^2) + i*sin(pi*D*nu^2) */
typedef struct { float re; float im; } complex_f;

void dispersion_lut(complex_f *H, int N, float D) {
    for (int k = 0; k < N; k++) {
        float nu    = (k < N/2) ? (float)k/N : (float)(k-N)/N;
        float phase = (float)M_PI * D * nu * nu;
        H[k].re = lut_cos(phase);
        H[k].im = lut_sin(phase);
    }
}

void dispersion_std(complex_f *H, int N, float D) {
    for (int k = 0; k < N; k++) {
        float nu    = (k < N/2) ? (float)k/N : (float)(k-N)/N;
        float phase = (float)M_PI * D * nu * nu;
        H[k].re = cosf(phase);
        H[k].im = sinf(phase);
    }
}

/* ── benchmark ───────────────────────────────────────────── */
static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int main(void) {
    lut_build();

    printf("LUT sin/cos  (N=%d entries, %d KB)\n",
           LUT_N, (int)(2 * LUT_N * sizeof(float) / 1024));
    printf("----------------------------------------------\n\n");

    /* accuracy test */
    printf("Accuracy vs std (sample points):\n");
    printf("  %12s  %10s  %10s  %12s\n", "theta(rad)", "std_sin", "lut_sin", "err");
    float test_thetas[] = {0.0f, 0.5f, 1.0f, (float)M_PI/4,
                           (float)M_PI/2, (float)M_PI, 2.5f, 6.0f, -1.3f};
    int n_test = sizeof(test_thetas)/sizeof(test_thetas[0]);
    float max_err = 0.0f;
    for (int i = 0; i < n_test; i++) {
        float th   = test_thetas[i];
        float s_std = sinf(th);
        float s_lut = lut_sin(th);
        float err   = fabsf(s_std - s_lut);
        if (err > max_err) max_err = err;
        printf("  %12.4f  %10.6f  %10.6f  %12.2e\n", th, s_std, s_lut, err);
    }
    printf("  max_err = %.2e  (theory: pi^2/(2*N^2) = %.2e)\n\n",
           max_err, (float)(M_PI*M_PI)/(2.0f*LUT_N*LUT_N));

    /* benchmark: N=512 dispersion kernel, 100k reps */
    int N = 512;
    int reps = 100000;
    complex_f *H_lut = (complex_f*)malloc(N * sizeof(complex_f));
    complex_f *H_std = (complex_f*)malloc(N * sizeof(complex_f));

    double t0, t1;

    t0 = wall_ms();
    for (int r = 0; r < reps; r++) dispersion_lut(H_lut, N, 5000.0f);
    t1 = wall_ms();
    double ms_lut = (t1 - t0) / reps;

    t0 = wall_ms();
    for (int r = 0; r < reps; r++) dispersion_std(H_std, N, 5000.0f);
    t1 = wall_ms();
    double ms_std = (t1 - t0) / reps;

    printf("Benchmark: dispersion kernel N=%d, %d reps\n", N, reps);
    printf("  std cosf/sinf: %.4f ms/call\n", ms_std);
    printf("  LUT interp:    %.4f ms/call\n", ms_lut);
    printf("  Speedup:       %.2fx\n\n", ms_std / ms_lut);

    /* verify dispersion output matches */
    float max_disp_err = 0.0f;
    for (int k = 0; k < N; k++) {
        float er = fabsf(H_lut[k].re - H_std[k].re);
        float ei = fabsf(H_lut[k].im - H_std[k].im);
        float e  = (er > ei) ? er : ei;
        if (e > max_disp_err) max_disp_err = e;
    }
    printf("Max dispersion LUT vs std error: %.2e\n", max_disp_err);

    free(H_lut); free(H_std);
    return 0;
}
