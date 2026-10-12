/* Scarlett particle estimator -- see scarlett_estimator.h. */
#include "scarlett_estimator.h"

#include <stddef.h>

/* One correlator: sum over channels of measurement * template (int64). */
static int64_t correlate(const int32_t *meas, const int16_t *row, int32_t n_channels)
{
    int64_t acc = 0;
    for (int32_t j = 0; j < n_channels; ++j) {
        acc += (int64_t)meas[j] * (int64_t)row[j];
    }
    return acc;
}

/* Energy of one stored template: sum of its squared samples (int64). */
static int64_t row_energy(const int16_t *row, int32_t n_channels)
{
    int64_t acc = 0;
    for (int32_t j = 0; j < n_channels; ++j) {
        acc += (int64_t)row[j] * (int64_t)row[j];
    }
    return acc;
}

/* Fit statistic of one template: corr^2 / template_energy [counts^2], or -1
 * when the correlation is not positive. corr^2 exceeds 64 bits, so this step
 * is done in double precision; hardware would multiply by a reciprocal energy
 * held in calibration memory. */
static double fit_statistic(const int32_t *meas, const int16_t *row, int32_t n_channels,
                            int64_t *corr_out)
{
    int64_t corr = correlate(meas, row, n_channels);
    if (corr_out != NULL) {
        *corr_out = corr;
    }
    if (corr <= 0) {
        return -1.0;
    }
    return (double)corr * (double)corr / (double)row_energy(row, n_channels);
}

int scarlett_estimate(const int32_t *meas, int32_t n_channels,
                      const int16_t *templates_q15, int32_t n_templates,
                      double noise_sigma_counts, double delta_chi2,
                      scarlett_estimate_t *out)
{
    if (meas == NULL || templates_q15 == NULL || out == NULL ||
        n_channels <= 0 || n_templates <= 0 ||
        !(noise_sigma_counts > 0.0) || !(delta_chi2 > 0.0)) {
        return SCARLETT_ERR_ARGUMENT;
    }
    for (int32_t k = 0; k < n_templates; ++k) {
        if (row_energy(templates_q15 + (size_t)k * (size_t)n_channels, n_channels) == 0) {
            return SCARLETT_ERR_ARGUMENT;      /* an all-zero template */
        }
    }

    /* Pass 1: correlator bank, keep the largest statistic (first wins a tie). */
    int32_t best_index = 0;
    int64_t best_corr = 0;
    double best_stat = fit_statistic(meas, templates_q15, n_channels, &best_corr);
    for (int32_t k = 1; k < n_templates; ++k) {
        int64_t corr;
        double stat = fit_statistic(meas, templates_q15 + (size_t)k * (size_t)n_channels,
                                    n_channels, &corr);
        if (stat > best_stat) {
            best_stat = stat;
            best_corr = corr;
            best_index = k;
        }
    }
    if (best_corr <= 0) {
        return SCARLETT_ERR_NO_MATCH;
    }

    int64_t energy = 0;
    for (int32_t j = 0; j < n_channels; ++j) {
        energy += (int64_t)meas[j] * (int64_t)meas[j];
    }

    /* Pass 2: confidence set. */
    const double threshold = best_stat - delta_chi2 * noise_sigma_counts * noise_sigma_counts;
    int32_t lo = -1, hi = -1, count = 0;
    for (int32_t k = 0; k < n_templates; ++k) {
        double stat = fit_statistic(meas, templates_q15 + (size_t)k * (size_t)n_channels,
                                    n_channels, NULL);
        if (stat > 0.0 && stat >= threshold) {
            if (lo < 0) {
                lo = k;
            }
            hi = k;
            ++count;
        }
    }

    out->best_index = best_index;
    out->best_corr = best_corr;
    out->energy = energy;
    out->chi2_min = (double)energy - best_stat;
    out->ci_lo_index = lo;
    out->ci_hi_index = hi;
    out->n_accepted = count;
    out->ambiguous = (hi - lo + 1 != count) ? 1 : 0;
    return SCARLETT_OK;
}

int scarlett_autocorrelate(const int32_t *x, int32_t n, int32_t max_lag, int64_t *acf)
{
    if (x == NULL || acf == NULL || n <= 0 || max_lag < 0 || max_lag >= n) {
        return SCARLETT_ERR_ARGUMENT;
    }
    for (int32_t lag = 0; lag <= max_lag; ++lag) {
        int64_t acc = 0;
        for (int32_t i = 0; i + lag < n; ++i) {
            acc += (int64_t)x[i] * (int64_t)x[i + lag];
        }
        acf[lag] = acc;
    }
    return SCARLETT_OK;
}
