/*
 * nlse_kernel.c -- "Nature as a Computer": a nonlinear optical fiber as a kernel (ANSI C89)
 *
 * Section 2 of "Physics-AI Symbiosis" describes encoding data onto a laser
 * spectrum, letting a nonlinear optical element transform it (like a kernel),
 * reading the output spectrum, and training only a light linear classifier.
 *
 * Here: two concentric rings of 2-D points (not linearly separable). Each point
 * (x, y) is written onto a pulse's spectral PHASE: x as quadratic phase (chirp),
 * y as cubic phase. The pulse propagates through a Kerr fiber, simulated with the
 * split-step Fourier method for the nonlinear Schrodinger equation (NLSE):
 *     dA/dz = -i (beta2/2) d^2A/dt^2 + i gamma |A|^2 A
 * The output spectral intensity (24 bins) is the feature vector for logistic
 * regression. Three runs:
 *   raw     : linear classifier on (x, y)
 *   linear  : fiber with gamma = 0 (dispersion only, |spectrum| never changes)
 *   kerr    : fiber with gamma > 0 (self-phase modulation mixes everything)
 *
 * Build: gcc -std=c89 -pedantic -Wall -Wextra -O2 nlse_kernel.c -o nlse_kernel.exe -lm
 * Run:   nlse_kernel.exe            (writes nlse_features.csv and nlse_results.csv)
 *        nlse_kernel.exe bench N    (times the feature map alone for N samples)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define NT 128
#define NFEAT 24
#define NSAMP 500
#define NTRAIN 300
#define PI 3.14159265358979323846

typedef struct { double re, im; } cplx;

static unsigned long seed = 7UL;
static double urand(void)
{
    seed = (1103515245UL * seed + 12345UL) & 0x7fffffffUL;
    return ((double)seed + 0.5) / 2147483648.0;
}

static cplx cmul(cplx a, cplx b) { cplx r; r.re = a.re * b.re - a.im * b.im; r.im = a.re * b.im + a.im * b.re; return r; }
static cplx cpolar(double m, double ph) { cplx r; r.re = m * cos(ph); r.im = m * sin(ph); return r; }

/* in-place radix-2 FFT; sign -1 forward, +1 inverse (scaled 1/N) */
static void fft(cplx *x, int n, int sign)
{
    int i, j, k, len, half, bit;
    cplx t, wl, w, u, v;
    for (i = 1, j = 0; i < n; i++) {
        for (bit = n >> 1; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { t = x[i]; x[i] = x[j]; x[j] = t; }
    }
    for (len = 2; len <= n; len <<= 1) {
        half = len >> 1;
        wl = cpolar(1.0, sign * 2.0 * PI / len);
        for (i = 0; i < n; i += len) {
            w.re = 1.0; w.im = 0.0;
            for (k = 0; k < half; k++) {
                u = x[i + k]; v = cmul(x[i + k + half], w);
                x[i + k].re = u.re + v.re; x[i + k].im = u.im + v.im;
                x[i + k + half].re = u.re - v.re; x[i + k + half].im = u.im - v.im;
                w = cmul(w, wl);
            }
        }
    }
    if (sign > 0) for (i = 0; i < n; i++) { x[i].re /= n; x[i].im /= n; }
}

/* encode (x, y) on the spectral phase, propagate, return NFEAT spectral-intensity features */
static void optical_features(double xd, double yd, double gamma, double *feat)
{
    const double Twin = 40.0, beta2 = 0.5, L = 1.0;
    const int steps = 80;
    const double dt = Twin / NT, h = L / steps;
    double w[NT], tot = 0.0;
    cplx A[NT];
    int i, s, b, idx;

    for (i = 0; i < NT; i++)
        w[i] = 2.0 * PI * (i < NT / 2 ? i : i - NT) / (NT * dt);
    for (i = 0; i < NT; i++)           /* spectrum: Gaussian magnitude, data in the phase */
        A[i] = cpolar(exp(-w[i] * w[i] / 2.0), 1.2 * xd * w[i] * w[i] + 0.6 * yd * w[i] * w[i] * w[i]);
    fft(A, NT, +1);                    /* to the time domain */
    {   /* normalize so a transform-limited pulse has peak power 1 */
        double pk = 0.0, ref[NT];
        cplx B[NT];
        for (i = 0; i < NT; i++) B[i] = cpolar(exp(-w[i] * w[i] / 2.0), 0.0);
        fft(B, NT, +1);
        for (i = 0; i < NT; i++) { ref[i] = B[i].re * B[i].re + B[i].im * B[i].im; if (ref[i] > pk) pk = ref[i]; }
        for (i = 0; i < NT; i++) { A[i].re /= sqrt(pk); A[i].im /= sqrt(pk); }
    }
    for (s = 0; s < steps; s++) {      /* symmetric split-step: half dispersion, nonlinearity, half dispersion */
        fft(A, NT, -1);
        for (i = 0; i < NT; i++) A[i] = cmul(A[i], cpolar(1.0, beta2 / 2.0 * w[i] * w[i] * h / 2.0));
        fft(A, NT, +1);
        for (i = 0; i < NT; i++) A[i] = cmul(A[i], cpolar(1.0, gamma * (A[i].re * A[i].re + A[i].im * A[i].im) * h));
        fft(A, NT, -1);
        for (i = 0; i < NT; i++) A[i] = cmul(A[i], cpolar(1.0, beta2 / 2.0 * w[i] * w[i] * h / 2.0));
        fft(A, NT, +1);
    }
    fft(A, NT, -1);                    /* output spectrum */
    for (b = 0; b < NFEAT; b++) {      /* NFEAT bins across the central band, ordered by frequency */
        idx = b - NFEAT / 2;
        if (idx < 0) idx += NT;
        feat[b] = A[idx].re * A[idx].re + A[idx].im * A[idx].im;
        tot += feat[b];
    }
    for (b = 0; b < NFEAT; b++) feat[b] /= tot;
}

/* logistic regression by gradient descent on standardized features; returns test accuracy */
static double train_test(double **X, const int *y, int d, int *pred)
{
    double *mu = (double *)calloc(d, sizeof(double)), *sd = (double *)calloc(d, sizeof(double));
    double *wv = (double *)calloc(d, sizeof(double)), *g = (double *)calloc(d, sizeof(double));
    double b0 = 0.0, gb, z, p, xs;
    int i, j, it, correct = 0;

    for (j = 0; j < d; j++) {
        for (i = 0; i < NTRAIN; i++) mu[j] += X[i][j];
        mu[j] /= NTRAIN;
        for (i = 0; i < NTRAIN; i++) sd[j] += (X[i][j] - mu[j]) * (X[i][j] - mu[j]);
        sd[j] = sqrt(sd[j] / NTRAIN);
        /* a feature that is constant up to rounding carries no information; without this test,
           standardizing would blow 1e-17 rounding noise up to unit size and the classifier
           would "learn" from floating-point noise */
        if (sd[j] < 1e-9 * (fabs(mu[j]) + 1e-30)) sd[j] = 0.0;   /* 0 marks the feature as dropped */
    }
    for (it = 0; it < 3000; it++) {
        for (j = 0; j < d; j++) g[j] = 0.0;
        gb = 0.0;
        for (i = 0; i < NTRAIN; i++) {
            z = b0;
            for (j = 0; j < d; j++) if (sd[j] > 0.0) z += wv[j] * (X[i][j] - mu[j]) / sd[j];
            p = 1.0 / (1.0 + exp(-z));
            for (j = 0; j < d; j++) if (sd[j] > 0.0) g[j] += (p - y[i]) * (X[i][j] - mu[j]) / sd[j];   /* chain rule: (p - y) x */
            gb += p - y[i];
        }
        for (j = 0; j < d; j++) wv[j] -= 0.5 * g[j] / NTRAIN;
        b0 -= 0.5 * gb / NTRAIN;
    }
    for (i = NTRAIN; i < NSAMP; i++) {
        z = b0;
        for (j = 0; j < d; j++) if (sd[j] > 0.0) { xs = (X[i][j] - mu[j]) / sd[j]; z += wv[j] * xs; }
        pred[i] = z > 0.0;
        correct += pred[i] == y[i];
    }
    free(mu); free(sd); free(wv); free(g);
    return (double)correct / (NSAMP - NTRAIN);
}

/* CPU timing of the feature map alone, for comparison with nlse_cuda.exe bench <N> */
static int bench(int n)
{
    double feat[NFEAT];
    clock_t c0 = clock(), c1;
    int i;
    for (i = 0; i < n; i++) optical_features(sin(0.37 * i), cos(0.61 * i), 6.0, feat);
    c1 = clock();
    printf("CPU (ANSI C, one core): %d samples x 80 split steps in %.1f ms  (%.1f us/sample)\n",
           n, 1000.0 * (c1 - c0) / CLOCKS_PER_SEC, 1e6 * (double)(c1 - c0) / CLOCKS_PER_SEC / n);
    return 0;
}

int main(int argc, char **argv)
{
    double *xs = (double *)malloc(NSAMP * sizeof(double)), *ys = (double *)malloc(NSAMP * sizeof(double));
    int *label = (int *)malloc(NSAMP * sizeof(int)), *pred = (int *)malloc(NSAMP * sizeof(int));
    double **raw = (double **)malloc(NSAMP * sizeof(double *));
    double **lin = (double **)malloc(NSAMP * sizeof(double *));
    double **ker = (double **)malloc(NSAMP * sizeof(double *));
    double r, th, acc_raw, acc_lin, acc_ker;
    int i, b;
    FILE *fp;

    if (argc > 2 && strcmp(argv[1], "bench") == 0) return bench(atoi(argv[2]));

    for (i = 0; i < NSAMP; i++) {      /* two concentric rings, scaled into [-1, 1] */
        label[i] = i % 2;
        r = label[i] ? 1.5 + urand() : urand();
        th = 2.0 * PI * urand();
        xs[i] = r * cos(th) / 2.5;
        ys[i] = r * sin(th) / 2.5;
        raw[i] = (double *)malloc(2 * sizeof(double));
        lin[i] = (double *)malloc(NFEAT * sizeof(double));
        ker[i] = (double *)malloc(NFEAT * sizeof(double));
        raw[i][0] = xs[i]; raw[i][1] = ys[i];
        optical_features(xs[i], ys[i], 0.0, lin[i]);
        optical_features(xs[i], ys[i], 6.0, ker[i]);
    }
    acc_raw = train_test(raw, label, 2, pred);
    acc_lin = train_test(lin, label, NFEAT, pred);
    acc_ker = train_test(ker, label, NFEAT, pred);

    fp = fopen("nlse_features.csv", "w");
    if (!fp) { perror("nlse_features.csv"); return 1; }
    fprintf(fp, "x,y,label,pred_kerr");
    for (b = 0; b < NFEAT; b++) fprintf(fp, ",f%d", b);
    fprintf(fp, "\n");
    for (i = 0; i < NSAMP; i++) {
        fprintf(fp, "%.6f,%.6f,%d,%d", xs[i], ys[i], label[i], i >= NTRAIN ? pred[i] : -1);
        for (b = 0; b < NFEAT; b++) fprintf(fp, ",%.8e", ker[i][b]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    fp = fopen("nlse_results.csv", "w");
    if (!fp) { perror("nlse_results.csv"); return 1; }
    fprintf(fp, "model,test_accuracy\nraw_xy,%.4f\nfiber_dispersion_only,%.4f\nfiber_kerr,%.4f\n", acc_raw, acc_lin, acc_ker);
    fclose(fp);
    printf("test accuracy on %d held-out points (chance = 0.5):\n", NSAMP - NTRAIN);
    printf("  linear classifier on raw (x, y)          %.3f\n", acc_raw);
    printf("  ... on fiber output, dispersion only     %.3f\n", acc_lin);
    printf("  ... on fiber output, Kerr nonlinearity   %.3f\n", acc_ker);
    return 0;
}
