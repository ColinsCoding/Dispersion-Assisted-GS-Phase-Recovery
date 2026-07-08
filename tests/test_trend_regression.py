"""Test dgs.trend_regression: OLS with inference. A perfect line gives slope/
intercept exactly and p=0; a noisy fit matches scipy.stats.linregress on every
statistic; a flat series reads 'no significant trend' (CI straddles zero) while a
real slope is detected; and the confidence interval matches b +/- t* SE(b)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats
from dgs import trend_regression as tr

# 1. a perfect line y = 2x + 3: exact coefficients, R^2 = 1, zero-uncertainty slope
x = np.arange(10.0)
y = 2 * x + 3
r = tr.fit(x, y)
assert np.isclose(r["slope"], 2.0) and np.isclose(r["intercept"], 3.0)
assert np.isclose(r["r_squared"], 1.0)
assert r["slope_se"] == 0.0 and r["t_stat"] == np.inf and r["p_value"] == 0.0

# 2. a noisy fit matches scipy.stats.linregress on EVERY statistic
rng = np.random.default_rng(1)
xn = np.arange(30.0)
yn = 1.5 * xn - 4 + rng.normal(0, 5, 30)
r = tr.fit(xn, yn)
lr = stats.linregress(xn, yn)
assert np.isclose(r["slope"], lr.slope)
assert np.isclose(r["intercept"], lr.intercept)
assert np.isclose(r["r_squared"], lr.rvalue ** 2)
assert np.isclose(r["slope_se"], lr.stderr)
assert np.isclose(r["p_value"], lr.pvalue)
assert r["df"] == 28

# 3. the headline test: flat data -> NOT significant; real slope -> significant
years = np.arange(1994, 2024)
flat = 450 + rng.normal(0, 8, len(years))
sig_flat, p_flat = tr.is_trend_significant(years, flat)
assert not sig_flat and p_flat > 0.05                  # "same rates"
decline = 450 - 2.0 * (years - 1994) + rng.normal(0, 8, len(years))
sig_dec, p_dec = tr.is_trend_significant(years, decline)
assert sig_dec and p_dec < 0.01                        # a real trend
assert np.isclose(tr.fit(years, decline)["slope"], -2.0, atol=0.5)

# 4. confidence interval: flat straddles 0, real trend does not, and it equals
#    b +/- t_crit * SE(b)
lo, hi = tr.slope_confidence_interval(years, flat)
assert lo < 0 < hi                                     # cannot claim a trend
lo2, hi2 = tr.slope_confidence_interval(years, decline)
assert hi2 < 0                                         # whole interval below zero
rr = tr.fit(years, decline)
tcrit = stats.t.ppf(0.975, rr["df"])
assert np.isclose(lo2, rr["slope"] - tcrit * rr["slope_se"])
assert np.isclose(hi2, rr["slope"] + tcrit * rr["slope_se"])

# 5. predict evaluates the fitted line
assert np.allclose(tr.predict({"intercept": 3.0, "slope": 2.0}, [0, 1, 5]), [3, 5, 13])

# 6. R^2 is in [0,1]; a slope can be significant yet explain little variance
noisy = 100 + 0.3 * years + rng.normal(0, 30, len(years))
rn = tr.fit(years, noisy)
assert 0.0 <= rn["r_squared"] <= 1.0

# 7. kwarg bounds
for bad in (lambda: tr.fit([1, 2], [1, 2]),                 # n < 3
            lambda: tr.fit([1, 2, 3], [1, 2]),              # mismatched
            lambda: tr.fit([5, 5, 5], [1, 2, 3]),           # identical x
            lambda: tr.slope_confidence_interval(x, y, alpha=0),
            lambda: tr.is_trend_significant(x, y, alpha=1.5)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_trend_regression: all checks passed")
