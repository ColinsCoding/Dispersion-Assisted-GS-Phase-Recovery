/* uncertainty_demo.c -- propagation of uncertainty for P = V^2 / R.
 *
 * First-order (GUM-style) result for independent V and R:
 *     sigma_P / P = sqrt( (2 sigma_V / V)^2 + (sigma_R / R)^2 )
 * checked against a Monte Carlo with a fixed seed (xorshift64 + Box-Muller),
 * so the output is reproducible on every platform.
 *
 * Compile (from the project root):
 *     gcc -std=c11 -Wall -Wextra c/uncertainty_demo.c -lm -o uncertainty_demo
 * Run:
 *     ./uncertainty_demo            (defaults: V = 5.0 +- 0.05 V, R = 100 +- 1 ohm)
 *     ./uncertainty_demo 12 0.1 50 0.5
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t rng_state = 88172645463325252ULL;

static double uniform01(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return ((rng_state >> 11) + 0.5) * (1.0 / 9007199254740992.0);   /* (0,1) */
}

static double gaussian(void)
{
    const double two_pi = 6.283185307179586;
    double u1 = uniform01(), u2 = uniform01();
    return sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
}

int main(int argc, char **argv)
{
    double V = 5.0, sV = 0.05, R = 100.0, sR = 1.0;
    if (argc == 5) {
        V = atof(argv[1]); sV = atof(argv[2]); R = atof(argv[3]); sR = atof(argv[4]);
    }
    if (R <= 0.0 || V == 0.0 || sV < 0.0 || sR < 0.0) {
        fprintf(stderr, "need R > 0, V != 0, non-negative uncertainties\n");
        return 1;
    }

    double P = V * V / R;
    double sP = P * sqrt(pow(2.0 * sV / V, 2) + pow(sR / R, 2));

    const long n = 1000000;
    double sum = 0.0, sum2 = 0.0;
    for (long i = 0; i < n; i++) {
        double v = V + sV * gaussian();
        double r = R + sR * gaussian();
        double p = v * v / r;
        sum += p;
        sum2 += p * p;
    }
    double mean = sum / n;
    double sd = sqrt((sum2 - n * mean * mean) / (n - 1));

    printf("inputs: V = %.6g +- %.6g V, R = %.6g +- %.6g ohm\n", V, sV, R, sR);
    printf("first-order: P = %.6f W, sigma_P = %.6f W (%.3f %%)\n", P, sP, 100.0 * sP / P);
    printf("Monte Carlo: P = %.6f W, sigma_P = %.6f W (n = %ld)\n", mean, sd, n);
    return 0;
}
