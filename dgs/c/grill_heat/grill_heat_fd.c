/* grill_heat_fd.c -- radial finite-difference solve of the axisymmetric
 * heat equation on a circular stainless steel grill, dT/dt = alpha*(d2T/dr2
 * + (1/r)dT/dr), Dirichlet BC T(R,t)=0.
 *
 * Same scheme (FTCS, L'Hopital-handled r=0 singularity) and same default
 * parameters as dgs.grill_heat_equation.RadialFiniteDifferenceSolver in
 * Python -- built as an independent cross-check, not a port for its own
 * sake: two different languages implementing the same finite-difference
 * stencil should land on the same numbers, and disagreement would mean a
 * real bug in one of them.
 *
 * Prints the final temperature array to stdout, one value per line, after
 * advancing to T_TARGET seconds -- read and compared against the Python
 * solver's own array by dgs.grill_heat_equation_polyglot.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define R 0.15
#define ALPHA 4.0e-6
#define PEAK_TEMP 250.0
#define SEAR_RADIUS 0.05
#define N_POINTS 200
#define STABILITY_FACTOR 0.2
#define T_TARGET 100.0

double f_initial(double r) {
    return PEAK_TEMP * exp(-(r / SEAR_RADIUS) * (r / SEAR_RADIUS));
}

int main(void) {
    double r[N_POINTS], T[N_POINTS], T_new[N_POINTS];
    double dr = R / (N_POINTS - 1);
    double dt = STABILITY_FACTOR * dr * dr / ALPHA;

    for (int i = 0; i < N_POINTS; i++) {
        r[i] = i * dr;
        T[i] = f_initial(r[i]);
    }
    T[N_POINTS - 1] = 0.0;   /* Dirichlet boundary */

    double t = 0.0;
    while (t < T_TARGET) {
        for (int i = 1; i < N_POINTS - 1; i++) {
            double d2T = (T[i + 1] - 2.0 * T[i] + T[i - 1]) / (dr * dr);
            double dTdr = (T[i + 1] - T[i - 1]) / (2.0 * dr);
            T_new[i] = T[i] + dt * ALPHA * (d2T + dTdr / r[i]);
        }
        /* r=0: L'Hopital's rule -> the 1/r*dT/dr term becomes d2T/dr2 there,
         * using the symmetric ghost point T[-1] = T[1] */
        double d2T0 = (2.0 * T[1] - 2.0 * T[0]) / (dr * dr);
        T_new[0] = T[0] + dt * ALPHA * 2.0 * d2T0;
        T_new[N_POINTS - 1] = 0.0;

        for (int i = 0; i < N_POINTS; i++) {
            T[i] = T_new[i];
        }
        t += dt;
    }

    for (int i = 0; i < N_POINTS; i++) {
        printf("%.10f\n", T[i]);
    }
    fprintf(stderr, "final t = %.6f\n", t);
    return 0;
}
