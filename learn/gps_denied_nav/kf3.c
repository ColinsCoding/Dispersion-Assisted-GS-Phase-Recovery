/* kf3.c -- 3-state (position, velocity, accel bias) Kalman filter, plain C99.
 *
 * Same model as dgs/gps_denied_navigation.py (KalmanNav, n_states=3):
 *   predict every step with the measured acceleration u,
 *   update with a position fix every FIX_EVERY steps.
 * Written out by hand with shared subexpressions, the way SymPy's CSE counts it,
 * so it runs on a microcontroller with no matrix library.
 *
 * Build and race the optimization levels:
 *   gcc -std=c99 -O0 kf3.c -o kf3_O0   (then -O1, -O2, -O3, -Ofast)
 *   ./kf3_O2 10000000
 * Prints ns per predict step and a checksum (the final state) so the
 * compiler cannot delete the loop and so levels can be compared for equality.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FIX_EVERY 3000

typedef struct { double x, v, b; double p11, p12, p13, p22, p23, p33; } kf3_t;

static void kf3_predict(kf3_t *k, double u, double dt, double q, double qb)
{
    const double h = 0.5 * dt * dt;
    /* state: x'' = u - b */
    k->x += dt * k->v + h * (u - k->b);
    k->v += dt * (u - k->b);
    /* P <- F P F^T + q B B^T + diag(0,0,qb dt), F = [[1,dt,-h],[0,1,-dt],[0,0,1]] */
    const double a11 = k->p11 + dt * k->p12 - h * k->p13;   /* row 1 of F P */
    const double a12 = k->p12 + dt * k->p22 - h * k->p23;
    const double a13 = k->p13 + dt * k->p23 - h * k->p33;
    const double a22 = k->p22 - dt * k->p23;
    const double a23 = k->p23 - dt * k->p33;
    k->p11 = a11 + dt * a12 - h * a13 + q * h * h;
    k->p12 = a12 - dt * a13 + q * h * dt;
    k->p13 = a13;
    k->p22 = a22 - dt * a23 + q * dt * dt;
    k->p23 = a23;
    k->p33 += qb * dt;
}

static void kf3_update(kf3_t *k, double z, double r)
{
    const double s = k->p11 + r;
    const double k1 = k->p11 / s, k2 = k->p12 / s, k3 = k->p13 / s;
    const double innov = z - k->x;
    k->x += k1 * innov;
    k->v += k2 * innov;
    k->b += k3 * innov;
    const double p11 = k->p11, p12 = k->p12, p13 = k->p13;
    k->p11 -= k1 * p11;
    k->p12 -= k1 * p12;
    k->p13 -= k1 * p13;
    k->p22 -= k2 * p12;
    k->p23 -= k2 * p13;
    k->p33 -= k3 * p13;
}

/* deterministic pseudo-noise so every build sees the same input */
static double lcg_unit(unsigned long long *s)
{
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(*s >> 11) / 9007199254740992.0) - 0.5;
}

int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 10000000L;
    if (n <= 0) { fprintf(stderr, "steps must be positive\n"); return 1; }
    const double dt = 0.01, q = 4e-4, qb = 1e-10, r = 4.0, bias = 9.80665e-3;
    kf3_t k = {0, 0, 0, 0, 0, 0, 0, 0, (5 * 9.80665e-3) * (5 * 9.80665e-3)};
    unsigned long long seed = 12345ULL;
    double x_true = 0.0, v_true = 0.0;

    clock_t t0 = clock();
    for (long i = 0; i < n; ++i) {
        const double a_true = 0.0;                       /* hold still: drift is all bias */
        v_true += a_true * dt;
        x_true += v_true * dt;
        kf3_predict(&k, a_true + bias + 0.04 * lcg_unit(&seed), dt, q, qb);
        if (i > 0 && i % FIX_EVERY == 0)
            kf3_update(&k, x_true + 4.0 * lcg_unit(&seed), r);
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("steps=%ld  ns/step=%.2f  bias_est=%.9f  x=%.9f  v=%.9f\n",
           n, 1e9 * secs / (double)n, k.b, k.x, k.v);
    return 0;
}
