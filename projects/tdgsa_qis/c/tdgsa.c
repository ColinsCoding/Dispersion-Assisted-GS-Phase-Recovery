/*
 * tdgsa.c -- Time-domain Gerchberg-Saxton phase recovery (ANSI C89)
 *
 * Same problem as TDGSA_Dispersion_Phase_Recovery.ipynb, Section 4 (gas cell):
 *   1550 nm Gaussian pulse (T0 = 0.3 ps) -> 5-line causal gas cell ->
 *   intensities after D1 = 0 and D2 = 10 ps/nm -> TDGSA from a flat phase.
 *
 * Build:  gcc -std=c89 -pedantic -Wall -Wextra -O2 tdgsa.c -o tdgsa_c.exe -lm
 * Run:    tdgsa_c.exe <iterations> <output_prefix>
 * Writes: <prefix>_F.bin   recovered spectrum, N complex doubles (re, im interleaved)
 *         <prefix>_err.bin measurement error per iteration, <iterations> doubles
 *
 * Colin Casey, 2026. Based on G. Morozowsky and C. Casey, ECE 279AS (UCLA), 2024.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define LOG2N 15
#define N (1L << LOG2N)
#define PI 3.14159265358979323846

typedef struct { double re, im; } cplx;

static cplx c_mul(cplx a, cplx b) { cplx r; r.re = a.re*b.re - a.im*b.im; r.im = a.re*b.im + a.im*b.re; return r; }
static cplx c_div(cplx a, cplx b) {
    cplx r; double d = b.re*b.re + b.im*b.im;
    r.re = (a.re*b.re + a.im*b.im) / d; r.im = (a.im*b.re - a.re*b.im) / d; return r;
}
static cplx c_exp(cplx a) { cplx r; double m = exp(a.re); r.re = m*cos(a.im); r.im = m*sin(a.im); return r; }
static cplx c_polar(double mag, double ph) { cplx r; r.re = mag*cos(ph); r.im = mag*sin(ph); return r; }
static double c_abs(cplx a) { return sqrt(a.re*a.re + a.im*a.im); }

/* In-place iterative radix-2 FFT. sign = -1 forward (numpy fft), +1 inverse (numpy ifft, scaled 1/N). */
static void fft(cplx *x, int sign)
{
    long i, j, k, len, half;
    cplx tmp, wl, w, u, v;
    for (i = 1, j = 0; i < N; i++) {             /* bit-reversal permutation */
        long bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { tmp = x[i]; x[i] = x[j]; x[j] = tmp; }
    }
    for (len = 2; len <= N; len <<= 1) {
        half = len >> 1;
        wl = c_polar(1.0, sign * 2.0 * PI / (double)len);
        for (i = 0; i < N; i += len) {
            w.re = 1.0; w.im = 0.0;
            for (k = 0; k < half; k++) {
                u = x[i + k];
                v = c_mul(x[i + k + half], w);
                x[i + k].re = u.re + v.re; x[i + k].im = u.im + v.im;
                x[i + k + half].re = u.re - v.re; x[i + k + half].im = u.im - v.im;
                w = c_mul(w, wl);
            }
        }
    }
    if (sign > 0)
        for (i = 0; i < N; i++) { x[i].re /= (double)N; x[i].im /= (double)N; }
}

/* beta2*L (s^2) from total dispersion in ps/nm at 1550 nm */
static double beta2L(double D_ps_per_nm)
{
    const double c = 299792458.0, lam = 1550e-9;
    return -D_ps_per_nm * 1e-3 * lam * lam / (2.0 * PI * c);
}

int main(int argc, char **argv)
{
    const double dt = 0.06e-12, T0 = 0.3e-12, D1 = 0.0, D2 = 10.0;
    const double lines[5][3] = { {-400e9, 3e9, 1.5}, {-80e9, 1e9, 2.0}, {60e9, 1e9, 2.5},
                                 {300e9, 4e9, 1.2}, {900e9, 6e9, 0.8} };
    long i, it, n_iter;
    int l;
    double *w, *A1, *A2, *err, num, den, b1, b2;
    cplx *E, *H12, *H21, *f, *g;
    char path[512];
    FILE *fp;
    clock_t c0, c1;

    if (argc < 3) { fprintf(stderr, "usage: %s <iterations> <output_prefix>\n", argv[0]); return 1; }
    n_iter = atol(argv[1]);

    w = (double *)malloc(N * sizeof(double));
    A1 = (double *)malloc(N * sizeof(double));
    A2 = (double *)malloc(N * sizeof(double));
    err = (double *)malloc((n_iter > 0 ? n_iter : 1) * sizeof(double));
    E = (cplx *)malloc(N * sizeof(cplx));
    H12 = (cplx *)malloc(N * sizeof(cplx));
    H21 = (cplx *)malloc(N * sizeof(cplx));
    f = (cplx *)malloc(N * sizeof(cplx));
    g = (cplx *)malloc(N * sizeof(cplx));
    if (!w || !A1 || !A2 || !err || !E || !H12 || !H21 || !f || !g) { fprintf(stderr, "out of memory\n"); return 1; }

    /* grid, pulse, and gas cell */
    for (i = 0; i < N; i++) {
        double t = (double)(i - N / 2) * dt;
        long fi = (i < N / 2) ? i : i - N;                    /* numpy fftfreq ordering */
        w[i] = 2.0 * PI * (double)fi / ((double)N * dt);
        E[i].re = exp(-t * t / (2.0 * T0 * T0)); E[i].im = 0.0;
    }
    fft(E, -1);
    for (i = 0; i < N; i++) {
        cplx H; H.re = 1.0; H.im = 0.0;
        for (l = 0; l < 5; l++) {
            cplx den_l, num_l;
            den_l.re = 1.0; den_l.im = -(w[i] - 2.0 * PI * lines[l][0]) / (2.0 * PI * lines[l][1]);
            num_l.re = -lines[l][2]; num_l.im = 0.0;
            H = c_mul(H, c_exp(c_div(num_l, den_l)));
        }
        E[i] = c_mul(E[i], H);                                 /* E holds the spectrum F(w) */
    }

    /* the two measurements: |ifft(F * D(w))|^2 */
    b1 = beta2L(D1); b2 = beta2L(D2);
    for (i = 0; i < N; i++) f[i] = c_mul(E[i], c_polar(1.0, -b1 * w[i] * w[i] / 2.0));
    fft(f, +1);
    for (i = 0; i < N; i++) A1[i] = c_abs(f[i]);
    for (i = 0; i < N; i++) f[i] = c_mul(E[i], c_polar(1.0, -b2 * w[i] * w[i] / 2.0));
    fft(f, +1);
    for (i = 0; i < N; i++) A2[i] = c_abs(f[i]);
    for (i = 0; i < N; i++) {
        H12[i] = c_polar(1.0, -(b2 - b1) * w[i] * w[i] / 2.0);
        H21[i] = c_polar(1.0, (b2 - b1) * w[i] * w[i] / 2.0);
    }

    /* TDGSA from a flat phase */
    for (i = 0; i < N; i++) { f[i].re = A1[i]; f[i].im = 0.0; }
    c0 = clock();
    for (it = 0; it < n_iter; it++) {
        for (i = 0; i < N; i++) g[i] = f[i];
        fft(g, -1);
        for (i = 0; i < N; i++) g[i] = c_mul(g[i], H12[i]);
        fft(g, +1);
        num = 0.0; den = 0.0;
        for (i = 0; i < N; i++) {
            double m = c_abs(g[i]);
            num += (m - A2[i]) * (m - A2[i]); den += A2[i] * A2[i];
            g[i] = c_polar(A2[i], atan2(g[i].im, g[i].re));
        }
        err[it] = sqrt(num / den);
        fft(g, -1);
        for (i = 0; i < N; i++) g[i] = c_mul(g[i], H21[i]);
        fft(g, +1);
        for (i = 0; i < N; i++) f[i] = c_polar(A1[i], atan2(g[i].im, g[i].re));
    }
    c1 = clock();

    /* recovered spectrum F0 = fft(f1) / D1(w) */
    fft(f, -1);
    for (i = 0; i < N; i++) f[i] = c_mul(f[i], c_polar(1.0, b1 * w[i] * w[i] / 2.0));

    sprintf(path, "%s_F.bin", argv[2]);
    fp = fopen(path, "wb"); if (!fp) { perror(path); return 1; }
    fwrite(f, sizeof(cplx), (size_t)N, fp); fclose(fp);
    sprintf(path, "%s_err.bin", argv[2]);
    fp = fopen(path, "wb"); if (!fp) { perror(path); return 1; }
    fwrite(err, sizeof(double), (size_t)n_iter, fp); fclose(fp);

    printf("backend=ANSI-C iterations=%ld time_ms=%.3f final_meas_err=%.6e\n",
           n_iter, 1000.0 * (double)(c1 - c0) / CLOCKS_PER_SEC, n_iter > 0 ? err[n_iter - 1] : 0.0);

    free(w); free(A1); free(A2); free(err); free(E); free(H12); free(H21); free(f); free(g);
    return 0;
}
