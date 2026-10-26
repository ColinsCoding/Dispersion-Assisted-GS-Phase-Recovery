/* taylor.c -- e^x by Taylor series in C (math.h), three ways, checked against libm exp().
 *
 *   naive   : e^{-x} = sum (-x)^k / k!     -> alternating terms cancel, digits lost for large x
 *   stable  : e^{-x} = 1 / sum x^k / k!    -> all terms positive, but needs many terms
 *   reduced : e^{x} = 2^k e^{s}, s = x - k ln2, |s| <= ln2/2 -> 14 terms, full double precision
 *
 * Then uses the reduced version for hydrogen's n = 2 radial probabilities
 *   P_2s(r) = r^2 (2 - r)^2 e^{-r} / 8,   P_2p(r) = r^4 e^{-r} / 24   (atomic units)
 * and checks that each integrates to 1 (the area under the curve in polar coordinates).
 *
 * Build (lint = warnings as errors): gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Wconversion -Werror taylor.c -o taylor_c.exe -lm
 */
#include <math.h>
#include <stdio.h>

static const double LN2 = 0.693147180559945309417232121458;

/* sum_{k < n} x^k / k! by Horner's rule */
static double taylor_exp(double x, int n)
{
    double acc = 1.0;
    for (int k = n - 1; k > 0; --k)
        acc = 1.0 + x * acc / (double)k;
    return acc;
}

static double exp_neg_naive(double x) { return taylor_exp(-x, 60); }
static double exp_neg_stable(double x) { return 1.0 / taylor_exp(x, 60); }

static double exp_reduced(double x)
{
    double k = nearbyint(x / LN2);
    return ldexp(taylor_exp(x - k * LN2, 14), (int)k);
}

static double p2s(double r) { return r * r * (2.0 - r) * (2.0 - r) * exp_reduced(-r) / 8.0; }
static double p2p(double r) { return r * r * r * r * exp_reduced(-r) / 24.0; }

int main(void)
{
    const double xs[] = {1.0, 5.0, 10.0, 20.0, 30.0};
    printf("x, rel_err_naive, rel_err_stable, rel_err_reduced\n");
    for (size_t i = 0; i < sizeof xs / sizeof xs[0]; ++i) {
        double ref = exp(-xs[i]);
        printf("%.0f, %.3e, %.3e, %.3e\n", xs[i], exp_neg_naive(xs[i]) / ref - 1.0,
               exp_neg_stable(xs[i]) / ref - 1.0, exp_reduced(-xs[i]) / ref - 1.0);
    }
    /* trapezoid area of P(r) on [0, 80] */
    const int n = 200000;
    const double h = 80.0 / n;
    double a2s = 0.0, a2p = 0.0;
    for (int i = 0; i <= n; ++i) {
        double w = (i == 0 || i == n) ? 0.5 : 1.0;
        a2s += w * p2s(i * h);
        a2p += w * p2p(i * h);
    }
    printf("area_P2s, %.12f\narea_P2p, %.12f\n", a2s * h, a2p * h);
    printf("P2p_peak_r4, %.12f (exact 4^4 e^-4 / 24 = %.12f)\n", p2p(4.0), 256.0 * exp(-4.0) / 24.0);
    return 0;
}
