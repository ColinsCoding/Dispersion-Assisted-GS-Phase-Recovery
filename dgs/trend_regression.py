"""Is the trend real? Linear regression with the inference, not just the line.

Fitting a straight line to a scatter is easy; the hard question is whether the SLOPE
is really nonzero or just noise. "Cancer rates have been flat for 30 years" is a claim
about a slope: fit rate vs year, and ask whether the fitted slope b is far enough from
zero, given the scatter, to be believed. That is a hypothesis test:
        H0: true slope = 0   (no trend -- "same rates"),
and you reject it only when the slope is many standard errors from zero.

Ordinary least squares gives the line and, crucially, the UNCERTAINTY on the slope:
        b = Sxy/Sxx,   SE(b) = s / sqrt(Sxx),   s^2 = SSE/(n-2),
        t = b / SE(b),   p = 2 * P(T_{n-2} > |t|).
A small p means the trend is unlikely to be an accident; a large p means the data are
consistent with a flat line -- you CANNOT claim a change. R^2 says how much of the
scatter the line explains, which is a different question from whether the slope is
real (a tiny-but-certain slope can have a small R^2, and vice versa).

This adds the inferential layer missing from dgs.supervised.linear_regression, and is
cross-checked against scipy.stats.linregress. scipy for the exact t-distribution;
NumPy for the algebra. py-3.13.
"""

import numpy as np
from scipy import stats


def fit(x, y):
    """Ordinary least-squares fit of y = intercept + slope*x, WITH inference.
    Returns dict: slope, intercept, r_squared, slope_se, t_stat, p_value (two-
    sided test of slope=0), df, and residual_std. Needs n >= 3 points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 points (df = n-2 >= 1)")
    if len(y) != n:
        raise ValueError("x and y must be the same length")
    xm, ym = x.mean(), y.mean()
    Sxx = np.sum((x - xm) ** 2)
    if Sxx == 0:
        raise ValueError("all x are identical -- slope is undefined")
    Sxy = np.sum((x - xm) * (y - ym))
    Syy = np.sum((y - ym) ** 2)
    slope = Sxy / Sxx
    intercept = ym - slope * xm
    sse = Syy - slope * Sxy                 # residual sum of squares
    df = n - 2
    s2 = sse / df                           # residual variance
    r_squared = 1.0 - sse / Syy if Syy > 0 else 1.0
    if s2 <= 0:                             # a perfect fit: zero-uncertainty slope
        slope_se, t_stat, p_value = 0.0, np.inf, 0.0
    else:
        slope_se = np.sqrt(s2 / Sxx)
        t_stat = slope / slope_se
        p_value = float(2 * stats.t.sf(abs(t_stat), df))
    return {
        "slope": float(slope), "intercept": float(intercept),
        "r_squared": float(r_squared), "slope_se": float(slope_se),
        "t_stat": float(t_stat), "p_value": float(p_value),
        "df": df, "residual_std": float(np.sqrt(max(s2, 0.0))),
    }


def slope_confidence_interval(x, y, alpha=0.05):
    """Two-sided (1-alpha) confidence interval for the slope: b +/- t* SE(b).
    If it straddles zero, you cannot claim a trend at that confidence level."""
    r = fit(x, y)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    tcrit = stats.t.ppf(1 - alpha / 2, r["df"])
    half = tcrit * r["slope_se"]
    return (r["slope"] - half, r["slope"] + half)


def is_trend_significant(x, y, alpha=0.05):
    """Decide the headline question: is the slope significantly nonzero at level
    alpha? Returns (significant, p_value). significant=False means the data are
    consistent with a flat line -- 'the rate has not changed'."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    p = fit(x, y)["p_value"]
    return (p < alpha, p)


def predict(fit_result, x):
    """Evaluate the fitted line at new x from a fit() result."""
    x = np.asarray(x, float)
    return fit_result["intercept"] + fit_result["slope"] * x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    years = np.arange(1994, 2024)             # 30 years

    # Case A: cancer rate genuinely FLAT (per 100k) -> should read "no trend"
    flat = 450 + rng.normal(0, 8, len(years))
    sig, p = is_trend_significant(years, flat)
    r = fit(years, flat)
    print("FLAT rate (no real trend):")
    print(f"  slope = {r['slope']:+.3f}/yr, p = {p:.3f} -> significant trend? {sig}")
    print(f"  95% CI for slope: {tuple(round(v,3) for v in slope_confidence_interval(years, flat))}"
          f"  (straddles 0 -> 'same rates')")

    # Case B: a real decline of ~2 per 100k per year
    decline = 450 - 2.0 * (years - 1994) + rng.normal(0, 8, len(years))
    sig, p = is_trend_significant(years, decline)
    r = fit(years, decline)
    print("\nDECLINING rate (real trend):")
    print(f"  slope = {r['slope']:+.3f}/yr, R^2 = {r['r_squared']:.3f}, "
          f"p = {p:.2e} -> significant trend? {sig}")

    # cross-check against scipy.stats.linregress
    lr = stats.linregress(years, decline)
    print(f"\ncross-check vs scipy.linregress: slope match "
          f"{np.isclose(r['slope'], lr.slope)}, p match {np.isclose(r['p_value'], lr.pvalue)}")
