/* mie_main.cu -- host driver for single-sphere Mie scattering.
 *
 * Sweeps the size parameter x = 2*pi*r/lambda over a range covering
 * the Rayleigh regime (x << 1), the "Mie ripple" resonance regime
 * (x ~ 1-20, where Qext oscillates as interference between diffracted
 * and transmitted light), and into the geometric-optics limit
 * (x >> 1, where Qext -> 2, the well-known "extinction paradox").
 *
 * Refractive index m=1.33 (real, non-absorbing) is the standard
 * textbook example: a water droplet in air at visible wavelengths.
 *
 * Writes results to mie_output.csv so generate_mie_reference.py can
 * load the SAME x values and cross-check the CUDA output against an
 * independent Python implementation of the same algorithm.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mie.h"

int main(int argc, char **argv) {
    int n_particles = 200;
    double x_min = 0.1;
    double x_max = 20.0;
    double m_re = 1.33;
    double m_im = 0.0;

    double *x = (double *)malloc(n_particles * sizeof(double));
    double *Qext = (double *)malloc(n_particles * sizeof(double));
    double *Qsca = (double *)malloc(n_particles * sizeof(double));
    if (!x || !Qext || !Qsca) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n_particles; i++) {
        x[i] = x_min + (x_max - x_min) * i / (n_particles - 1);
    }

    run_mie_scattering(x, m_re, m_im, Qext, Qsca, n_particles);

    FILE *f = fopen("mie_output.csv", "w");
    if (!f) {
        fprintf(stderr, "could not open mie_output.csv for writing\n");
        return 1;
    }
    fprintf(f, "x,Qext,Qsca\n");
    for (int i = 0; i < n_particles; i++) {
        fprintf(f, "%.10f,%.10f,%.10f\n", x[i], Qext[i], Qsca[i]);
    }
    fclose(f);

    printf("m = %.4f + %.4fi (water droplet, non-absorbing)\n", m_re, m_im);
    printf("swept x from %.2f to %.2f over %d points\n\n", x_min, x_max, n_particles);
    printf("%10s  %10s  %10s\n", "x", "Qext", "Qsca");
    for (int i = 0; i < n_particles; i += 20) {
        printf("%10.4f  %10.6f  %10.6f\n", x[i], Qext[i], Qsca[i]);
    }
    printf("\nfull results written to mie_output.csv\n");
    printf("(large-x limit should approach Qext -> 2.0, the extinction paradox: "
           "last row Qext = %.4f)\n", Qext[n_particles - 1]);

    free(x);
    free(Qext);
    free(Qsca);
    return 0;
}
