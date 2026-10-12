/*
 * discover.c -- "Discovering physics": sparse regression finds the law of a falling apple (ANSI C89)
 *
 * Section 3 of "Physics-AI Symbiosis" asks whether a machine watching an apple
 * fall could recover the symbolic law. A minimal, interpretable version: given
 * noisy positions y(t), fit y against a LIBRARY of candidate terms
 *     1, t, t^2, t^3, t^4, sin(t), cos(t)
 * with sequentially thresholded least squares (STLSQ, as in SINDy): solve least
 * squares, zero out small coefficients, re-solve on the survivors, repeat.
 * The survivors are the "discovered" law, y = y0 + v0 t - (g/2) t^2, and g follows.
 *
 * Build: gcc -std=c89 -pedantic -Wall -Wextra -O2 discover.c -o discover.exe -lm
 * Run:   discover.exe   (writes discover_data.csv)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NS 61
#define NL 7

static const char *names[NL] = { "1", "t", "t^2", "t^3", "t^4", "sin(t)", "cos(t)" };

static unsigned long seed = 1687UL;            /* Newton's Principia */
static double urand(void)
{
    seed = (1103515245UL * seed + 12345UL) & 0x7fffffffUL;
    return ((double)seed + 0.5) / 2147483648.0;
}
static double gauss(void)
{
    double u1 = urand(), u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static double term(int k, double t)
{
    switch (k) {
    case 0: return 1.0;
    case 1: return t;
    case 2: return t * t;
    case 3: return t * t * t;
    case 4: return t * t * t * t;
    case 5: return sin(t);
    default: return cos(t);
    }
}

/* least squares on the active columns via normal equations + Gaussian elimination with pivoting */
static void lstsq(double th[NS][NL], const double *y, const int *active, double *coef)
{
    double A[NL][NL + 1], f, tmp;
    int idx[NL], m = 0, i, j, k, p, r;

    for (k = 0; k < NL; k++) { coef[k] = 0.0; if (active[k]) idx[m++] = k; }
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            A[i][j] = 0.0;
            for (r = 0; r < NS; r++) A[i][j] += th[r][idx[i]] * th[r][idx[j]];
        }
        A[i][m] = 0.0;
        for (r = 0; r < NS; r++) A[i][m] += th[r][idx[i]] * y[r];
    }
    for (i = 0; i < m; i++) {
        p = i;
        for (k = i + 1; k < m; k++) if (fabs(A[k][i]) > fabs(A[p][i])) p = k;
        for (j = 0; j <= m; j++) { tmp = A[i][j]; A[i][j] = A[p][j]; A[p][j] = tmp; }
        for (k = i + 1; k < m; k++) {
            f = A[k][i] / A[i][i];
            for (j = i; j <= m; j++) A[k][j] -= f * A[i][j];
        }
    }
    for (i = m - 1; i >= 0; i--) {
        f = A[i][m];
        for (j = i + 1; j < m; j++) f -= A[i][j] * coef[idx[j]];
        coef[idx[i]] = f / A[i][i];
    }
}


static double rss_of(double th[NS][NL], const double *y, const int *active, double *coef)
{
    double r, s = 0.0;
    int i, k;
    lstsq(th, y, active, coef);
    for (i = 0; i < NS; i++) {
        r = y[i];
        for (k = 0; k < NL; k++) r -= coef[k] * th[i][k];
        s += r * r;
    }
    return s;
}

static int count_active(const int *active)
{
    int k, m = 0;
    for (k = 0; k < NL; k++) m += active[k];
    return m;
}

/* information criterion: fit quality penalized by the number of terms.
   penalty 2 per term = AIC (Akaike), penalty ln(n) per term = BIC (Bayesian/Schwarz) */
static double crit(double rss, int m, double penalty) { return NS * log(rss / NS) + penalty * m; }

/* backward elimination: drop the term whose removal lowers the criterion most, until none does */
static void eliminate(double th[NS][NL], const double *y, double penalty, const char *label, double y0, double v0, double g)
{
    int active[NL], k, m, best;
    double coef[NL], rss, best_c, trial;

    for (k = 0; k < NL; k++) active[k] = 1;
    rss = rss_of(th, y, active, coef);
    printf("backward elimination by %s (penalty %.2f per term):\n", label, penalty);
    for (;;) {
        m = count_active(active);
        best = -1;
        best_c = crit(rss, m, penalty);
        if (m == 1) break;
        for (k = 0; k < NL; k++) {
            if (!active[k]) continue;
            active[k] = 0;
            trial = crit(rss_of(th, y, active, coef), m - 1, penalty);
            active[k] = 1;
            if (trial < best_c) { best_c = trial; best = k; }
        }
        if (best < 0) break;
        active[best] = 0;
        rss = rss_of(th, y, active, coef);
        printf("  drop %-7s -> %d terms, %s = %.2f\n", names[best], m - 1, label, best_c);
    }
    rss = rss_of(th, y, active, coef);
    printf("  discovered law:  y(t) =");
    for (k = 0; k < NL; k++) if (active[k]) printf(" %+.4f %s", coef[k], names[k]);
    printf("\n  residual RMS %.4f m\n", sqrt(rss / NS));
    if (count_active(active) == 3 && active[0] && active[1] && active[2])
        printf("  -> y0 + v0 t - (g/2) t^2 recovered: y0 = %.3f, v0 = %.3f, g = %.3f m/s^2 (true %.2f, %.1f, %.2f)\n\n",
               coef[0], coef[1], -2.0 * coef[2], y0, v0, g);
    else
        printf("  -> kept a spurious term: on 0 <= t <= 2.4, cos(t) ~ 1 - t^2/2 mimics the polynomial\n\n");
}

int main(void)
{
    const double y0 = 1.5, v0 = 12.0, g = 9.81, noise = 0.05, lambda = 0.15;
    double t[NS], y[NS], th[NS][NL], coef[NL];
    int active[NL], i, k, it, changed;
    FILE *fp = fopen("discover_data.csv", "w");

    if (!fp) { perror("discover_data.csv"); return 1; }
    fprintf(fp, "t,y_measured,y_true\n");
    for (i = 0; i < NS; i++) {
        t[i] = 2.4 * i / (NS - 1);
        y[i] = y0 + v0 * t[i] - 0.5 * g * t[i] * t[i] + noise * gauss();
        fprintf(fp, "%.4f,%.6f,%.6f\n", t[i], y[i], y0 + v0 * t[i] - 0.5 * g * t[i] * t[i]);
        for (k = 0; k < NL; k++) th[i][k] = term(k, t[i]);
    }
    fclose(fp);

    for (k = 0; k < NL; k++) active[k] = 1;
    lstsq(th, y, active, coef);
    printf("full least squares (all %d library terms):\n  ", NL);
    for (k = 0; k < NL; k++) printf("%s: %+.3f  ", names[k], coef[k]);
    printf("\n");
    for (it = 0; it < 10; it++) {           /* STLSQ */
        changed = 0;
        for (k = 0; k < NL; k++)
            if (active[k] && fabs(coef[k]) < lambda) { active[k] = 0; changed = 1; }
        lstsq(th, y, active, coef);
        if (!changed) break;
    }
    printf("STLSQ with lambda = %.2f after %d round(s): %d of %d terms kept", lambda, it + 1, count_active(active), NL);
    printf(" -> collinear library: every coefficient is large, nothing gets thresholded\n\n");

    printf("measurement noise level: %.2f m\n\n", noise);
    eliminate(th, y, 2.0, "AIC", y0, v0, g);
    eliminate(th, y, log((double)NS), "BIC", y0, v0, g);
    return 0;
}
