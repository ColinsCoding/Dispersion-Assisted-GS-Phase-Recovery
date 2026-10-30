/*
 * market.c - linear supply and demand as a 2x2 linear system, solved by Cramer's rule. ANSI C (C89).
 *
 *   demand  Q = a - b P        rewrite as   [ 1   b ] [Q]   [a]
 *   supply  Q = c + d P                     [ 1  -d ] [P] = [c]
 *
 * det = -(b + d), so a solution exists whenever b + d != 0 (lean/LinearAlgebraProofs.lean,
 * equilibrium_price). Optional shifts move demand (a) and supply (c) and report the change.
 *
 * Build:  gcc -std=c89 -pedantic -Wall -Wextra -Werror market.c -o market
 * Run:    market a b c d [demand_shift] [supply_shift]
 * Prints: P=... Q=... P2=... Q2=... dP=... dQ=...   (one per line)
 */
#include <stdio.h>
#include <stdlib.h>

/* Solve [[m11, m12], [m21, m22]] x = r by Cramer's rule. Returns 0 on success, 1 if singular. */
static int solve2(double m11, double m12, double m21, double m22, double r1, double r2,
                  double *x1, double *x2)
{
    double det = m11 * m22 - m12 * m21;
    if (det == 0.0) {
        return 1;
    }
    *x1 = (r1 * m22 - m12 * r2) / det;
    *x2 = (m11 * r2 - r1 * m21) / det;
    return 0;
}

static int equilibrium(double a, double b, double c, double d, double *P, double *Q)
{
    return solve2(1.0, b, 1.0, -d, a, c, Q, P);
}

int main(int argc, char **argv)
{
    double a, b, c, d, da = 0.0, dc = 0.0, P, Q, P2, Q2;

    if (argc != 5 && argc != 7) {
        fprintf(stderr, "usage: %s a b c d [demand_shift supply_shift]\n", argv[0]);
        return 2;
    }
    a = atof(argv[1]);
    b = atof(argv[2]);
    c = atof(argv[3]);
    d = atof(argv[4]);
    if (argc == 7) {
        da = atof(argv[5]);
        dc = atof(argv[6]);
    }
    if (b <= 0.0 || d <= 0.0) {
        fprintf(stderr, "slopes b and d must be positive\n");
        return 1;
    }
    if (equilibrium(a, b, c, d, &P, &Q) || equilibrium(a + da, b, c + dc, d, &P2, &Q2)) {
        fprintf(stderr, "singular system: b + d = 0\n");
        return 1;
    }
    if (P < 0.0 || Q < 0.0) {
        fprintf(stderr, "no market: equilibrium P=%g Q=%g is negative\n", P, Q);
        return 1;
    }
    printf("P=%.10g\nQ=%.10g\nP2=%.10g\nQ2=%.10g\ndP=%.10g\ndQ=%.10g\n", P, Q, P2, Q2, P2 - P, Q2 - Q);
    return 0;
}
