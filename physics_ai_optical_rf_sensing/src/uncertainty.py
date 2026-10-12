"""Measurement uncertainty and propagation of errors.

First-order (GUM-style) propagation for uncorrelated inputs:
    sigma_f^2 = sum_i (df/dx_i)^2 sigma_i^2
For P = V^2 / R this gives the relative form
    (sigma_P / P)^2 = (2 sigma_V / V)^2 + (sigma_R / R)^2.
Checked by Monte Carlo. Also: feature uncertainty by re-measurement (re-drawing
the noise on the same underlying signal) and a bootstrap confidence interval
for classifier accuracy.
"""
import numpy as np
import sympy as sp


def propagate_symbolic(expr, variables, sigmas):
    """Return the first-order sigma_f expression for expr(variables) with sigmas."""
    if len(variables) != len(sigmas):
        raise ValueError("variables and sigmas must have the same length")
    var = sum((sp.diff(expr, v) * s) ** 2 for v, s in zip(variables, sigmas))
    return sp.sqrt(sp.simplify(var))


def power_uncertainty_symbolic():
    V, R, sV, sR = sp.symbols("V R sigma_V sigma_R", positive=True)
    P = V ** 2 / R
    sP = propagate_symbolic(P, [V, R], [sV, sR])
    rel = sp.simplify(sp.sqrt(sp.expand(sP ** 2 / P ** 2)))
    return P, sP, rel, (V, R, sV, sR)


def power_uncertainty(V, sigma_V, R, sigma_R):
    """Analytic first-order (P, sigma_P)."""
    if R <= 0 or V == 0:
        raise ValueError("need R > 0 and V != 0")
    P = V ** 2 / R
    return P, P * np.sqrt((2 * sigma_V / V) ** 2 + (sigma_R / R) ** 2)


def power_uncertainty_mc(V, sigma_V, R, sigma_R, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    v = V + sigma_V * rng.standard_normal(n)
    r = R + sigma_R * rng.standard_normal(n)
    p = v ** 2 / r
    return float(p.mean()), float(p.std(ddof=1))


def feature_repeatability(make_fn, extract_fn, label, n_repeats=50, seed=0, **noise):
    """Re-measure ONE underlying signal n times (fresh noise each time): feature mean and std.

    make_fn(label, rng, **noise) -> (amp, phase); the underlying signal parameters
    are frozen by reusing the same generator state for the clean part."""
    out = []
    for k in range(n_repeats):
        rng_signal = np.random.default_rng(seed)           # same underlying signal each time
        clean_amp, clean_phase = make_fn(label, rng_signal, sigma_amp=0.0, sigma_phase=0.0)
        rng_noise = np.random.default_rng(seed + 1000 + k)
        amp = clean_amp + noise.get("sigma_amp", 0.02) * rng_noise.standard_normal(clean_amp.size)
        phase = clean_phase + noise.get("sigma_phase", 0.05) * rng_noise.standard_normal(clean_phase.size)
        out.append(extract_fn(amp, phase))
    keys = out[0].keys()
    return {k: (float(np.mean([o[k] for o in out])), float(np.std([o[k] for o in out], ddof=1))) for k in keys}


def bootstrap_ci(y_true, y_pred, metric, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval of a metric over test samples."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = [metric(y_true[i], y_pred[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))
