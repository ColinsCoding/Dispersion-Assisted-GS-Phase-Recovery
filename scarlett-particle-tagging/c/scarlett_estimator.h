/* Scarlett particle estimator -- integer correlator bank.
 *
 * C version of scarlett.asic.fixed_point_estimate (Python). It mirrors the
 * arithmetic a digital ASIC block would perform: one multiply-accumulate per
 * channel per template with integer accumulators, then one scaling per
 * template by its stored energy.
 *
 * This is a behavioural reference for an ARCHITECTURE PROPOSAL. It is not
 * firmware for an existing chip.
 */
#ifndef SCARLETT_ESTIMATOR_H
#define SCARLETT_ESTIMATOR_H

#include <stdint.h>

#define SCARLETT_Q15 32767          /* full scale of a unit-energy template sample */
#define SCARLETT_CHI2_95 3.841458820694124   /* chi-square quantile, 1 parameter, 95 % */

#define SCARLETT_OK 0
#define SCARLETT_ERR_ARGUMENT (-1)  /* null pointer or non-positive size / sigma */
#define SCARLETT_ERR_NO_MATCH (-2)  /* no template correlates positively */

typedef struct {
    int32_t best_index;     /* row of the template table with the largest correlation */
    int64_t best_corr;      /* <measurement, template_best>, template in Q15 */
    int64_t energy;         /* sum of measurement^2 [counts^2] */
    double  chi2_min;       /* energy - best_corr^2 / template_energy [counts^2] */
    int32_t ci_lo_index;    /* first accepted row */
    int32_t ci_hi_index;    /* last accepted row */
    int32_t n_accepted;     /* number of accepted rows */
    int32_t ambiguous;      /* 1 if the accepted rows are not contiguous */
} scarlett_estimate_t;

/* Find the template that best matches a measurement, and the set of
 * templates statistically indistinguishable from it.
 *
 *   meas               n_channels dark-subtracted ADC code sums
 *   templates_q15      n_templates x n_channels, row-major, each row scaled
 *                      to (nearly) unit energy times SCARLETT_Q15. The exact
 *                      integer energy of each row is used, because 16-bit
 *                      rounding shifts chi-square by a sizeable fraction of
 *                      the confidence threshold.
 *   noise_sigma_counts RMS noise of one measurement element [counts]
 *   delta_chi2         confidence threshold (SCARLETT_CHI2_95 for 95 %)
 *
 * With stat_k = corr_k^2 / energy_k (so that chi2_k = |meas|^2 - stat_k),
 * the best row maximizes stat_k and a row is accepted when
 *   corr_k > 0  and  stat_k >= stat_best - delta_chi2 * sigma^2.
 */
int scarlett_estimate(const int32_t *meas, int32_t n_channels,
                      const int16_t *templates_q15, int32_t n_templates,
                      double noise_sigma_counts, double delta_chi2,
                      scarlett_estimate_t *out);

/* Integer autocorrelation: acf[l] = sum_i x[i] * x[i + l], l = 0 .. max_lag.
 * acf must have room for max_lag + 1 values. */
int scarlett_autocorrelate(const int32_t *x, int32_t n, int32_t max_lag, int64_t *acf);

#endif /* SCARLETT_ESTIMATOR_H */
