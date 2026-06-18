"""Bayesian inference for physics measurement & detection.

Statistics the way it actually shows up in this project: you measure something
noisy (photon counts, a voltage, an alarm) and want the *posterior* over the
hidden quantity. Bayes' theorem,

    P(hypothesis | data)  =  P(data | hypothesis) P(hypothesis) / P(data),

turns a forward noise model into an inverse estimate -- exactly the move the
Gerchberg-Saxton receiver makes when it turns intensities back into phase.

Three workhorses:
  * discrete Bayes update (any hypothesis grid),
  * the Poisson photon-counting posterior (conjugate Gamma) -- the low-light
    detector from dispersion_gs_prototype.photon_shot_noise,
  * the Normal-Normal posterior for a noisy continuous measurement,
plus binary detection (the rare-event / base-rate problem behind RogueGuard).

A recurring theme -- "compressed engineered feature vs raw data": for these
models a *sufficient statistic* (the total count; the sample mean) carries the
entire posterior, so the engineered feature loses nothing. NumPy only.
"""

import numpy as np


# ── 1. discrete Bayes update ─────────────────────────────────────────
def bayes_posterior(prior, likelihood):
    """Posterior over a grid of hypotheses: posterior ∝ prior * likelihood.

    prior, likelihood : 1-D arrays of equal length (P(H_i), P(D|H_i)).
    Returns the normalized posterior P(H_i | D).
    """
    prior = np.asarray(prior, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    if prior.shape != likelihood.shape:
        raise ValueError("prior and likelihood must have the same shape")
    if np.any(prior < 0) or np.any(likelihood < 0):
        raise ValueError("prior and likelihood must be non-negative")
    joint = prior * likelihood
    evidence = joint.sum()
    if evidence == 0:
        raise ValueError("evidence P(D) = 0: data impossible under the prior")
    return joint / evidence


# ── 2. Poisson photon-counting posterior (conjugate Gamma) ───────────
def poisson_rate_posterior(counts, a0=1.0, b0=1e-6):
    """Posterior over the mean rate lambda of Poisson counts (low-light detector).

    Conjugate Gamma(a0, b0) prior  ->  Gamma(a0 + sum k, b0 + n) posterior.
    The *sufficient statistic* is (sum of counts, n): the whole posterior depends
    on the data only through it -- the "compressed engineered feature".

    Returns dict: a, b (posterior params), mean = a/b, mode (MAP) = (a-1)/b,
    var = a/b^2.
    """
    counts = np.asarray(counts, dtype=float)
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    if a0 <= 0 or b0 <= 0:
        raise ValueError("Gamma prior needs a0 > 0, b0 > 0")
    a = a0 + counts.sum()
    b = b0 + counts.size
    return {"a": a, "b": b, "mean": a / b,
            "mode": (a - 1) / b if a > 1 else 0.0, "var": a / b**2}


# ── 3. Normal-Normal posterior for a noisy measurement ───────────────
def gaussian_mean_posterior(data, mu0, sigma0, sigma):
    """Posterior over a mean mu given Gaussian data of known noise sigma.

    Conjugate Normal(mu0, sigma0^2) prior. Precision (1/variance) adds:
        precision_post = 1/sigma0^2 + n/sigma^2
    the posterior mean is the precision-weighted blend of prior and sample mean.
    Returns dict: mean, std, precision.
    """
    data = np.asarray(data, dtype=float)
    if sigma <= 0 or sigma0 <= 0:
        raise ValueError("sigma and sigma0 must be > 0")
    n = data.size
    prec0 = 1.0 / sigma0**2
    prec_data = n / sigma**2
    prec_post = prec0 + prec_data
    mean_post = (prec0 * mu0 + prec_data * data.mean()) / prec_post
    return {"mean": mean_post, "std": prec_post**-0.5, "precision": prec_post}


# ── 4. binary detection: the base-rate / rare-event problem ──────────
def detection_posterior(prior_event, sensitivity, false_alarm, observed="alarm"):
    """P(event | observation) for a binary detector (e.g. rogue-wave alarm).

    prior_event : base rate P(event)         (rare events make this tiny)
    sensitivity : P(alarm | event)           (true-positive / hit rate)
    false_alarm : P(alarm | no event)        (false-positive rate)
    observed    : 'alarm' or 'no_alarm'.

    The lesson: with a rare event even a good detector yields a low posterior
    after a single alarm -- the base rate dominates.
    """
    for name, p in [("prior_event", prior_event), ("sensitivity", sensitivity),
                    ("false_alarm", false_alarm)]:
        if not 0 <= p <= 1:
            raise ValueError(f"{name} must be a probability in [0, 1]")
    pe, pn = prior_event, 1 - prior_event
    if observed == "alarm":
        num = pe * sensitivity
        den = pe * sensitivity + pn * false_alarm
    elif observed == "no_alarm":
        num = pe * (1 - sensitivity)
        den = pe * (1 - sensitivity) + pn * (1 - false_alarm)
    else:
        raise ValueError("observed must be 'alarm' or 'no_alarm'")
    if den == 0:
        raise ValueError("observation has zero probability")
    return num / den


if __name__ == "__main__":
    # low-light: estimate the photon rate from a faint Poisson stream
    rng = np.random.default_rng(0)
    true_rate = 3.0
    counts = rng.poisson(true_rate, size=200)
    post = poisson_rate_posterior(counts)
    print(f"Poisson rate: true={true_rate}, posterior mean={post['mean']:.3f} "
          f"+/- {post['var']**0.5:.3f}")

    # rare-event alarm: 0.1% base rate, 99% sensitivity, 5% false alarm
    p = detection_posterior(0.001, 0.99, 0.05, "alarm")
    print(f"P(rogue | one alarm) = {p:.4f}  (base-rate fallacy: still small)")
