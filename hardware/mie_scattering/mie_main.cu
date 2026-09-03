/* mie_main.cu -- host driver for single-sphere Mie scattering.
 *
 * Sweeps the size parameter x = 2*pi*r/lambda over a range covering
 * the Rayleigh regime (x << 1), the "Mie ripple" resonance regime
 * (x ~ 1-20, where Qext oscillates as interference between diffracted
 * and transmitted light), and into the geometric-optics limit
 * (x >> 1, where Qext -> 2, the well-known "extinction paradox").
 *
 * Runs this sweep for SEVERAL materials, each with its own complex
 * relative refractive index m = n + i*k (n=real index, k=absorption
 * coefficient) -- the "ratio of particles to different materials"
 * this module exists to add: a real sphere's scattering behavior
 * depends entirely on what it's made of and what medium it sits in,
 * captured by that single complex ratio. Non-absorbing materials
 * (k=0, water/ice/glass) give Qext == Qsca exactly (no energy lost to
 * absorption); the absorbing materials (soot, gold -- k>0) are the
 * first time this pipeline's Qabs = Qext - Qsca is genuinely nonzero,
 * exercising a code path the water-only version never touched.
 *
 * Reference refractive indices (approximate, visible ~550 nm,
 * illustrative rather than tied to one specific source):
 *   water droplet   m = 1.33  + 0.00i   (non-absorbing)
 *   ice crystal     m = 1.31  + 0.00i   (non-absorbing)
 *   silica/glass    m = 1.46  + 0.00i   (non-absorbing)
 *   soot (black C)  m = 1.85  + 0.71i   (strongly absorbing)
 *   gold nanosphere m = 0.47  + 2.40i   (plasmonic, strongly absorbing)
 *
 * Writes results to mie_output.csv (columns: material,x,Qext,Qsca) so
 * generate_mie_reference.py can load the SAME x values and materials
 * and cross-check the CUDA output against an independent Python
 * implementation of the same algorithm, material by material.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "mie.h"

typedef struct {
    const char *name;
    double m_re;
    double m_im;
} Material;

static const Material MATERIALS[] = {
    {"water",   1.33, 0.00},
    {"ice",     1.31, 0.00},
    {"silica",  1.46, 0.00},
    {"soot",    1.85, 0.71},
    {"gold",    0.47, 2.40},
};
#define N_MATERIALS ((int)(sizeof(MATERIALS) / sizeof(MATERIALS[0])))

int main(int argc, char **argv) {
    int n_particles = 200;
    double x_min = 0.1;
    double x_max = 20.0;

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

    FILE *f = fopen("mie_output.csv", "w");
    if (!f) {
        fprintf(stderr, "could not open mie_output.csv for writing\n");
        return 1;
    }
    fprintf(f, "material,x,Qext,Qsca\n");

    for (int m = 0; m < N_MATERIALS; m++) {
        const Material *mat = &MATERIALS[m];
        run_mie_scattering(x, mat->m_re, mat->m_im, Qext, Qsca, n_particles);

        for (int i = 0; i < n_particles; i++) {
            fprintf(f, "%s,%.10f,%.10f,%.10f\n", mat->name, x[i], Qext[i], Qsca[i]);
        }

        printf("m = %.4f + %.4fi  (%s%s)\n", mat->m_re, mat->m_im, mat->name,
               mat->m_im > 0.0 ? ", absorbing" : ", non-absorbing");
        printf("%10s  %10s  %10s  %10s\n", "x", "Qext", "Qsca", "Qabs");
        for (int i = 0; i < n_particles; i += 40) {
            printf("%10.4f  %10.6f  %10.6f  %10.6f\n", x[i], Qext[i], Qsca[i], Qext[i] - Qsca[i]);
        }
        printf("last-row Qext = %.4f  (large-x -> 2.0 for non-absorbing materials)\n\n",
               Qext[n_particles - 1]);
    }
    fclose(f);

    printf("full results for all %d materials written to mie_output.csv\n", N_MATERIALS);

    free(x);
    free(Qext);
    free(Qsca);
    return 0;
}
