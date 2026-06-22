/* tunneling.c -- quantum tunneling through a rectangular barrier (Griffiths QM Ch 8).
 *
 * The C twin of dgs/tunneling.py: the EXACT transmission
 *     T = 1 / (1 + V0^2 sinh^2(kappa a) / (4 E (V0-E))),  kappa = sqrt(2 m (V0-E))/hbar,
 * and the WKB estimate exp(-2 kappa a). Natural units hbar = m = 1. The two agree on the
 * exponential; their ratio is the O(1) prefactor 16 E (V0-E) / V0^2.
 *
 * Compile:  cc -O2 -o tunneling tunneling.c -lm      Run:  ./tunneling
 */
#include <stdio.h>
#include <math.h>

static double kappa(double E, double V0) { return sqrt(2.0 * (V0 - E)); }

static double exact_T(double E, double V0, double a) {
    double s = sinh(kappa(E, V0) * a);
    return 1.0 / (1.0 + V0 * V0 * s * s / (4.0 * E * (V0 - E)));
}

static double wkb_T(double E, double V0, double a) {
    return exp(-2.0 * kappa(E, V0) * a);
}

int main(void) {
    double E = 1.0, V0 = 2.0;
    printf("kappa = sqrt(2(V0-E)) = %.4f,  penetration depth 1/kappa = %.3f\n",
           kappa(E, V0), 1.0 / kappa(E, V0));
    printf("  width   exact T       WKB exp(-2ka)   ratio\n");
    for (double a = 0.5; a <= 3.01; a += 0.5) {
        double te = exact_T(E, V0, a), tw = wkb_T(E, V0, a);
        printf("  %4.1f   %.4e   %.4e   %.2f\n", a, te, tw, te / tw);
    }
    printf("ratio -> prefactor 16 E (V0-E) / V0^2 = %.1f for thick barriers\n",
           16.0 * E * (V0 - E) / (V0 * V0));
    return 0;
}
