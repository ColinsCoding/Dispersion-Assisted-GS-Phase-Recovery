/*
 * pst.c -- Phase Stretch Transform on a faint "live cell" image (ANSI C89)
 *
 * Section 2 of "Physics-AI Symbiosis": the PST emulates near-field propagation
 * through a dispersive medium followed by coherent (phase) detection, and makes
 * nearly transparent cells visible (their Figure 2).
 *   1. low-pass filter in the Fourier domain
 *   2. multiply the spectrum by exp(-i phi(rho)),
 *      phi = S * [W rho atan(W rho) - 0.5 ln(1 + (W rho)^2)] / max
 *   3. inverse FFT and keep the PHASE; edges carry the most phase
 *
 * Build: gcc -std=c89 -pedantic -Wall -Wextra -O2 pst.c -o pst.exe -lm
 * Run:   pst.exe   (writes cells_input.pgm, pst_feature.pgm, pst_edges.pgm,
 *                   cells_input.bin and pst_feature.bin as raw float64 for checking)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 256
#define NY 256
#define PI 3.14159265358979323846

typedef struct { double re, im; } cplx;

static unsigned long seed = 11UL;
static double urand(void)
{
    seed = (1103515245UL * seed + 12345UL) & 0x7fffffffUL;
    return ((double)seed + 0.5) / 2147483648.0;
}
static double gauss(void)
{
    double u1 = urand(), u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}
static cplx cmul(cplx a, cplx b) { cplx r; r.re = a.re * b.re - a.im * b.im; r.im = a.re * b.im + a.im * b.re; return r; }
static cplx cpolar(double m, double ph) { cplx r; r.re = m * cos(ph); r.im = m * sin(ph); return r; }

static void fft1(cplx *x, int n, int stride, int sign)
{
    int i, j, k, len, half, bit;
    cplx t, wl, w, u, v;
    for (i = 1, j = 0; i < n; i++) {
        for (bit = n >> 1; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { t = x[i * stride]; x[i * stride] = x[j * stride]; x[j * stride] = t; }
    }
    for (len = 2; len <= n; len <<= 1) {
        half = len >> 1;
        wl = cpolar(1.0, sign * 2.0 * PI / len);
        for (i = 0; i < n; i += len) {
            w.re = 1.0; w.im = 0.0;
            for (k = 0; k < half; k++) {
                u = x[(i + k) * stride]; v = cmul(x[(i + k + half) * stride], w);
                x[(i + k) * stride].re = u.re + v.re; x[(i + k) * stride].im = u.im + v.im;
                x[(i + k + half) * stride].re = u.re - v.re; x[(i + k + half) * stride].im = u.im - v.im;
                w = cmul(w, wl);
            }
        }
    }
    if (sign > 0) for (i = 0; i < n; i++) { x[i * stride].re /= n; x[i * stride].im /= n; }
}

/* 2-D FFT: rows then columns (row-major, NY rows of NX) */
static void fft2(cplx *a, int sign)
{
    int r, c;
    for (r = 0; r < NY; r++) fft1(a + r * NX, NX, 1, sign);
    for (c = 0; c < NX; c++) fft1(a + c, NY, NX, sign);
}

static int cmp_double(const void *a, const void *b)
{
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

static void write_pgm(const char *name, const double *img)
{
    FILE *fp = fopen(name, "wb");
    double lo = img[0], hi = img[0];
    int i;
    if (!fp) { perror(name); exit(1); }
    for (i = 0; i < NX * NY; i++) { if (img[i] < lo) lo = img[i]; if (img[i] > hi) hi = img[i]; }
    fprintf(fp, "P5\n%d %d\n255\n", NX, NY);
    for (i = 0; i < NX * NY; i++) fputc((int)(255.0 * (img[i] - lo) / (hi - lo + 1e-300) + 0.5), fp);
    fclose(fp);
}

static void write_bin(const char *name, const double *img)
{
    FILE *fp = fopen(name, "wb");
    if (!fp) { perror(name); exit(1); }
    fwrite(img, sizeof(double), (size_t)(NX * NY), fp);
    fclose(fp);
}

int main(void)
{
    const double S = 0.5, W = 15.0, sigma_lpf = 0.15, edge_quantile = 0.92;
    double *img = (double *)malloc(NX * NY * sizeof(double));
    double *feat = (double *)malloc(NX * NY * sizeof(double));
    double *edges = (double *)malloc(NX * NY * sizeof(double));
    double *sorted = (double *)malloc(NX * NY * sizeof(double));
    double *phi = (double *)malloc(NX * NY * sizeof(double));
    cplx *a = (cplx *)malloc(NX * NY * sizeof(cplx));
    double cx[40], cy[40], cr[40], u, v, rho, d, lo, hi, thr, phimax = 0.0, sl;
    int i, j, k, ncell = 40;

    /* a faint field of cells: 3% brighter than the background, smooth rims, sensor noise */
    for (k = 0; k < ncell; k++) { cx[k] = 12 + urand() * (NX - 24); cy[k] = 12 + urand() * (NY - 24); cr[k] = 5 + 5 * urand(); }
    for (j = 0; j < NY; j++)
        for (i = 0; i < NX; i++) {
            double val = 0.5;
            for (k = 0; k < ncell; k++) {
                d = sqrt((i - cx[k]) * (i - cx[k]) + (j - cy[k]) * (j - cy[k]));
                val += 0.03 / (1.0 + exp((d - cr[k]) / 0.8));
            }
            img[j * NX + i] = val + 0.01 * gauss();
        }

    /* spatial-frequency grid (numpy fftfreq order), PST phase kernel and low-pass filter */
    for (j = 0; j < NY; j++)
        for (i = 0; i < NX; i++) {
            u = (double)(i < NX / 2 ? i : i - NX) / NX;
            v = (double)(j < NY / 2 ? j : j - NY) / NY;
            rho = sqrt(u * u + v * v);
            phi[j * NX + i] = W * rho * atan(W * rho) - 0.5 * log(1.0 + (W * rho) * (W * rho));
            if (phi[j * NX + i] > phimax) phimax = phi[j * NX + i];
        }

    /* 1. low-pass filter */
    for (k = 0; k < NX * NY; k++) { a[k].re = img[k]; a[k].im = 0.0; }
    fft2(a, -1);
    sl = sigma_lpf / sqrt(log(2.0));
    for (j = 0; j < NY; j++)
        for (i = 0; i < NX; i++) {
            u = (double)(i < NX / 2 ? i : i - NX) / NX;
            v = (double)(j < NY / 2 ? j : j - NY) / NY;
            rho = sqrt(u * u + v * v);
            d = exp(-0.5 * (rho / sl) * (rho / sl));
            a[j * NX + i].re *= d; a[j * NX + i].im *= d;
        }
    fft2(a, +1);
    for (k = 0; k < NX * NY; k++) a[k].im = 0.0;                         /* keep the real smoothed image */

    /* 2. phase kernel, 3. inverse FFT and take the phase */
    fft2(a, -1);
    for (k = 0; k < NX * NY; k++) a[k] = cmul(a[k], cpolar(1.0, -S * phi[k] / phimax));
    fft2(a, +1);
    for (k = 0; k < NX * NY; k++) feat[k] = atan2(a[k].im, a[k].re);
    lo = hi = feat[0];
    for (k = 0; k < NX * NY; k++) { if (feat[k] < lo) lo = feat[k]; if (feat[k] > hi) hi = feat[k]; }
    for (k = 0; k < NX * NY; k++) feat[k] = (feat[k] - lo) / (hi - lo);

    /* threshold at a quantile of the PST phase */
    for (k = 0; k < NX * NY; k++) sorted[k] = feat[k];
    qsort(sorted, NX * NY, sizeof(double), cmp_double);
    thr = sorted[(int)(edge_quantile * (NX * NY - 1))];
    for (k = 0; k < NX * NY; k++) edges[k] = feat[k] > thr ? 1.0 : 0.0;

    write_pgm("cells_input.pgm", img);
    write_pgm("pst_feature.pgm", feat);
    write_pgm("pst_edges.pgm", edges);
    write_bin("cells_input.bin", img);
    write_bin("pst_feature.bin", feat);
    printf("PST on a %dx%d image with %d faint cells (contrast 3%%, noise 1%%)\n", NX, NY, ncell);
    printf("S = %.2f, W = %.1f, sigma_LPF = %.2f, edge threshold = %.0f%% quantile\n", S, W, sigma_lpf, 100 * edge_quantile);
    printf("wrote cells_input.pgm, pst_feature.pgm, pst_edges.pgm (+ .bin float64 copies)\n");
    free(img); free(feat); free(edges); free(sorted); free(phi); free(a);
    return 0;
}
