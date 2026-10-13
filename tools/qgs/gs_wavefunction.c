/* gs_wavefunction.c -- recover a wavefunction's phase from its probability densities at several times.
 *
 * Free Schrodinger evolution is a phase exp(-i (hbar/m) k^2 t / 2) on each momentum component: the
 * same operator as fiber dispersion. Given |psi(x, t_j)|^2 for j = 1..K, multi-plane Gerchberg-Saxton
 * recovers psi(x, 0) up to a global phase (which has no physical meaning).
 *
 *   gcc -O2 -Wall -Wextra -std=c11 gs_wavefunction.c -o gs_wavefunction -lm
 *   ./gs_wavefunction densities.bin psi_out.bin n_iter
 *
 * densities.bin : int32 N, int32 K, float64 dx (nm), float64 hbar_over_m (nm^2/fs),
 *                 float64 t[K] (fs), float64 rho[K][N]
 * psi_out.bin   : float64 (re, im) pairs, N of them
 * stdout        : final misfit  sum_j || |psi(t_j)| - sqrt(rho_j) ||
 * exit codes    : 0 ok, 2 bad arguments, 3 cannot read/write a file, 4 out of memory, 5 N not a power of 2
 * C has no exceptions: every failure is a return code, checked by the caller (the shell's $?).
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI                       /* not part of standard C (it is POSIX), so define it */
#define M_PI 3.14159265358979323846
#endif

typedef double complex cplx;

static void fft(cplx *a, int n, int inverse) {               /* in-place iterative radix-2 FFT */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { cplx t = a[i]; a[i] = a[j]; a[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inverse ? 1 : -1);
        cplx wl = cos(ang) + I * sin(ang);
        for (int i = 0; i < n; i += len) {
            cplx w = 1.0;
            for (int j = 0; j < len / 2; j++) {
                cplx u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v; a[i + j + len / 2] = u - v;
                w *= wl;
            }
        }
    }
    if (inverse) for (int i = 0; i < n; i++) a[i] /= n;
}

/* psi -> psi evolved by time t (negative t evolves backwards) */
static void evolve(cplx *psi, const double *k2, int n, double hm, double t) {
    fft(psi, n, 0);
    for (int i = 0; i < n; i++) psi[i] *= cexp(-0.5 * I * hm * k2[i] * t);
    fft(psi, n, 1);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: gs_wavefunction densities.bin psi_out.bin n_iter\n"); return 2; }
    int n_iter = atoi(argv[3]);
    if (n_iter < 1) { fprintf(stderr, "n_iter must be >= 1\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 3; }
    int32_t N, K; double dx, hm;
    if (fread(&N, 4, 1, f) != 1 || fread(&K, 4, 1, f) != 1 || fread(&dx, 8, 1, f) != 1 || fread(&hm, 8, 1, f) != 1) { fclose(f); return 3; }
    if (N < 2 || (N & (N - 1)) != 0) { fclose(f); fprintf(stderr, "N = %d is not a power of 2\n", N); return 5; }
    if (K < 1 || K > 64) { fclose(f); return 2; }
    double *t = malloc(sizeof(double) * K), *amp = malloc(sizeof(double) * K * N), *k2 = malloc(sizeof(double) * N);
    cplx *psi = malloc(sizeof(cplx) * N), *y = malloc(sizeof(cplx) * N);
    if (!t || !amp || !k2 || !psi || !y) { fclose(f); return 4; }
    if (fread(t, 8, K, f) != (size_t)K || fread(amp, 8, (size_t)K * N, f) != (size_t)K * N) { fclose(f); return 3; }
    fclose(f);
    for (long i = 0; i < (long)K * N; i++) amp[i] = sqrt(amp[i] > 0 ? amp[i] : 0);
    for (int i = 0; i < N; i++) {                            /* k for FFT bin i, in 1/nm */
        int m = i < N / 2 ? i : i - N;
        double kk = 2.0 * M_PI * m / (N * dx);
        k2[i] = kk * kk;
    }
    for (int i = 0; i < N; i++) psi[i] = amp[i];             /* start: sqrt(rho at the first time), zero phase */
    evolve(psi, k2, N, hm, -t[0]);
    for (int it = 0; it < n_iter; it++)
        for (int j = 0; j < K; j++) {
            for (int i = 0; i < N; i++) y[i] = psi[i];
            evolve(y, k2, N, hm, t[j]);
            for (int i = 0; i < N; i++) { double m = cabs(y[i]); y[i] = m > 0 ? amp[(long)j * N + i] * y[i] / m : amp[(long)j * N + i]; }
            evolve(y, k2, N, hm, -t[j]);
            for (int i = 0; i < N; i++) psi[i] = y[i];
        }
    double mis = 0;
    for (int j = 0; j < K; j++) {
        for (int i = 0; i < N; i++) y[i] = psi[i];
        evolve(y, k2, N, hm, t[j]);
        double s = 0;
        for (int i = 0; i < N; i++) { double d = cabs(y[i]) - amp[(long)j * N + i]; s += d * d; }
        mis += sqrt(s);
    }
    FILE *o = fopen(argv[2], "wb");
    if (!o) return 3;
    for (int i = 0; i < N; i++) { double re = creal(psi[i]), im = cimag(psi[i]); if (fwrite(&re, 8, 1, o) != 1 || fwrite(&im, 8, 1, o) != 1) { fclose(o); return 3; } }
    fclose(o);
    printf("%.3e\n", mis);
    free(t); free(amp); free(k2); free(psi); free(y);
    return 0;
}
