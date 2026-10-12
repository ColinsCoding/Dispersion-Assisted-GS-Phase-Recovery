/* Command-line test harness for the Scarlett estimator.
 *
 *   scarlett_estimator estimate <vector file>
 *   scarlett_estimator acf      <vector file>
 *
 * "estimate" file (whitespace-separated text, written by
 * scarlett.asic.write_estimate_vectors / scripts/export_c_test_vectors.py):
 *     n_channels n_templates noise_sigma_counts delta_chi2
 *     n_channels measurement values
 *     n_templates * n_channels template values (row-major, Q15)
 * prints: best_index ci_lo_index ci_hi_index n_accepted ambiguous best_corr energy chi2_min
 *
 * "acf" file:
 *     n max_lag
 *     n sample values
 * prints: max_lag + 1 lag sums, one per line
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "scarlett_estimator.h"

static int fail(const char *message)
{
    fprintf(stderr, "ERROR: %s\n", message);
    return 2;
}

static int run_estimate(FILE *f)
{
    int n_channels, n_templates;
    double sigma, delta;
    if (fscanf(f, "%d %d %lf %lf", &n_channels, &n_templates, &sigma, &delta) != 4 ||
        n_channels <= 0 || n_templates <= 0) {
        return fail("bad header (expected: n_channels n_templates sigma delta_chi2)");
    }
    int32_t *meas = malloc((size_t)n_channels * sizeof *meas);
    int16_t *lut = malloc((size_t)n_channels * (size_t)n_templates * sizeof *lut);
    if (meas == NULL || lut == NULL) {
        free(meas); free(lut);
        return fail("out of memory");
    }
    int ok = 1;
    for (int j = 0; ok && j < n_channels; ++j) {
        ok = fscanf(f, "%" SCNd32, &meas[j]) == 1;
    }
    for (long i = 0; ok && i < (long)n_channels * n_templates; ++i) {
        ok = fscanf(f, "%" SCNd16, &lut[i]) == 1;
    }
    int status = 2;
    if (!ok) {
        fail("vector file ended early or contains a non-integer");
    } else {
        scarlett_estimate_t est;
        int rc = scarlett_estimate(meas, n_channels, lut, n_templates, sigma, delta, &est);
        if (rc != SCARLETT_OK) {
            fprintf(stderr, "ERROR: scarlett_estimate returned %d\n", rc);
            status = 3;
        } else {
            printf("%" PRId32 " %" PRId32 " %" PRId32 " %" PRId32 " %" PRId32 " %" PRId64 " %" PRId64 " %.17g\n",
                   est.best_index, est.ci_lo_index, est.ci_hi_index, est.n_accepted,
                   est.ambiguous, est.best_corr, est.energy, est.chi2_min);
            status = 0;
        }
    }
    free(meas); free(lut);
    return status;
}

static int run_acf(FILE *f)
{
    int n, max_lag;
    if (fscanf(f, "%d %d", &n, &max_lag) != 2 || n <= 0 || max_lag < 0 || max_lag >= n) {
        return fail("bad header (expected: n max_lag with 0 <= max_lag < n)");
    }
    int32_t *x = malloc((size_t)n * sizeof *x);
    int64_t *acf = malloc((size_t)(max_lag + 1) * sizeof *acf);
    if (x == NULL || acf == NULL) {
        free(x); free(acf);
        return fail("out of memory");
    }
    int status = 0;
    for (int i = 0; i < n; ++i) {
        if (fscanf(f, "%" SCNd32, &x[i]) != 1) {
            status = fail("vector file ended early or contains a non-integer");
            break;
        }
    }
    if (status == 0) {
        if (scarlett_autocorrelate(x, n, max_lag, acf) != SCARLETT_OK) {
            status = fail("scarlett_autocorrelate rejected its arguments");
        } else {
            for (int lag = 0; lag <= max_lag; ++lag) {
                printf("%" PRId64 "\n", acf[lag]);
            }
        }
    }
    free(x); free(acf);
    return status;
}

int main(int argc, char **argv)
{
    if (argc != 3 || (strcmp(argv[1], "estimate") != 0 && strcmp(argv[1], "acf") != 0)) {
        fprintf(stderr, "usage: %s estimate|acf <vector file>\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[2], "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR: cannot open %s\n", argv[2]);
        return 2;
    }
    int status = strcmp(argv[1], "estimate") == 0 ? run_estimate(f) : run_acf(f);
    fclose(f);
    return status;
}
