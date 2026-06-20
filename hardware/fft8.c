/* fft8.c -- 8-point radix-2 DIT FFT, the C "golden model" for fft8.v.
 *
 * Same algorithm as the Verilog RTL (hardware/fft8.v): decimation-in-time, the
 * inputs taken in bit-reversed order 0,4,2,6,1,5,3,7, then three stages of
 * butterflies with twiddle factors W = exp(-2*pi*i*k/N). This floating-point C
 * version is the reference a hardware engineer checks the fixed-point Verilog
 * against -- C model + RTL + git, the standard DSP-hardware workflow.
 *
 * Build & run:  gcc -O2 -o fft8 fft8.c -lm  &&  ./fft8
 */
#include <stdio.h>
#include <math.h>
#include <complex.h>

#define N 8

/* in-place radix-2 decimation-in-time FFT of N=8 complex samples */
void fft8(double complex *x)
{
    /* stage 0: reorder into bit-reversed order (matches fft8.v input wiring) */
    static const int br[N] = {0, 4, 2, 6, 1, 5, 3, 7};
    double complex a[N];
    for (int i = 0; i < N; i++) a[i] = x[br[i]];

    /* stages 1..3: butterflies over groups of size m = 2, 4, 8 */
    for (int s = 1; s <= 3; s++) {
        int m = 1 << s;                              /* group size */
        double complex wm = cexp(-2.0 * I * M_PI / m);
        for (int k = 0; k < N; k += m) {
            double complex w = 1.0;
            for (int j = 0; j < m / 2; j++) {
                double complex t = w * a[k + j + m / 2];
                double complex u = a[k + j];
                a[k + j]         = u + t;            /* butterfly + */
                a[k + j + m / 2] = u - t;            /* butterfly - */
                w *= wm;
            }
        }
    }
    for (int i = 0; i < N; i++) x[i] = a[i];
}

static int check(const char *name, double complex *x, const double *expect_mag)
{
    int ok = 1;
    printf("%-10s |X[k]| = ", name);
    for (int k = 0; k < N; k++) {
        double mag = cabs(x[k]);
        printf("%5.2f ", mag);
        if (fabs(mag - expect_mag[k]) > 1e-9) ok = 0;
    }
    printf("  %s\n", ok ? "OK" : "FAIL");
    return ok;
}

int main(void)
{
    int pass = 1;

    /* impulse -> flat spectrum (every bin = 1) */
    double complex imp[N] = {1, 0, 0, 0, 0, 0, 0, 0};
    double exp_imp[N] = {1, 1, 1, 1, 1, 1, 1, 1};
    fft8(imp);
    pass &= check("impulse", imp, exp_imp);

    /* one-cycle cosine -> spikes at k=1 and k=7 (height N/2 = 4) */
    double complex cos1[N];
    for (int k = 0; k < N; k++) cos1[k] = cos(2.0 * M_PI * k / N);
    double exp_cos[N] = {0, 4, 0, 0, 0, 0, 0, 4};
    fft8(cos1);
    pass &= check("cosine", cos1, exp_cos);

    printf("%s\n", pass ? "ALL TESTS PASS" : "TESTS FAILED");
    return pass ? 0 : 1;
}
